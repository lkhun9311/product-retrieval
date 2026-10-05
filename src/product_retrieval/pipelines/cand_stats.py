"""`pr cand-stats` (contract c5-reranker-v1 section 1): per-candidate image-level statistics.

Reads a rankings file written by ``pr eval`` (``query_id``, ``top_k_product_ids``, ``scores``;
``truth_product_id`` is never read), re-derives the query vectors the same way ``run_eval`` does
(same split selection, same index, cached query embeddings) and writes
``<rankings stem>.cand_stats.jsonl`` next to it. The rankings file itself is never modified.

For a candidate product with gallery image vectors g_1..g_n and a query vector q (all L2-normalised,
similarity = inner product): ``n_images``, ``max_sim``, ``mean_sim``, ``second_sim`` (max_sim when
n = 1), ``std_sim`` (population std, 0 when n = 1) and ``top1_sim`` (inner product of this product's
normalised mean vector with the rank-1 product's normalised mean vector, 1.0 for rank 1 itself).
``max_sim`` must equal the rankings score within ``SCORE_TOL`` or the run fails.

Gallery vectors are grouped by product and only the 20 candidates' images are scored per query, so
the full query x gallery similarity matrix is never materialised.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.ids import sha256_file
from product_retrieval.core.schemas import CropSpec
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.index.flat import GalleryIndex
from product_retrieval.pipelines.build_index import EmbedderName, _embed_shas, _make_embedder
from product_retrieval.pipelines.evaluate import locate_index
from product_retrieval.pipelines.selection import select_split

DEPTH = 20
SCORE_TOL = 1e-5
STAT_KEYS = ("n_images", "max_sim", "mean_sim", "second_sim", "std_sim", "top1_sim")


class CandStatsError(ValueError):
    """Raised for a rankings file that cannot be matched to the selected split/index."""


@dataclass(frozen=True)
class CandStatsResult:
    index_id: str
    rankings_path: Path
    out_path: Path
    n_queries: int
    rankings_sha256: str


def cand_stats_path(rankings_path: str | Path) -> Path:
    """``x.rankings.jsonl`` -> ``x.rankings.cand_stats.jsonl`` in the same directory."""
    p = Path(rankings_path)
    return p.with_name(f"{p.stem}.cand_stats.jsonl")


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def product_stats(q: np.ndarray, g: np.ndarray, mean_unit: np.ndarray, top1_unit: np.ndarray) -> dict:
    """Stats of one candidate. ``q`` (dim,), ``g`` (n, dim) float64; the two unit vectors give top1_sim."""
    sims = g @ q
    n = len(sims)
    ordered = np.sort(sims)[::-1]
    return {
        "n_images": int(n),
        "max_sim": float(ordered[0]),
        "mean_sim": float(sims.mean()),
        "second_sim": float(ordered[1]) if n > 1 else float(ordered[0]),
        "std_sim": float(sims.std()) if n > 1 else 0.0,
        "top1_sim": float(mean_unit @ top1_unit),
    }


def _read_rankings(path: Path) -> list[tuple[str, list[str], list[float]]]:
    """(query_id, top-20 product ids, top-20 scores) per row; no other field is read."""
    out: list[tuple[str, list[str], list[float]]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("query_id", "top_k_product_ids", "scores"):
                if key not in row:
                    raise CandStatsError(f"{path}:{lineno}: missing field {key!r}")
            qid, ids, scores = row["query_id"], row["top_k_product_ids"], row["scores"]
            if qid in seen:
                raise CandStatsError(f"{path}:{lineno}: duplicate query_id {qid!r}")
            if len(scores) != len(ids):
                raise CandStatsError(f"{path}:{lineno}: scores not aligned with top_k_product_ids")
            seen.add(qid)
            out.append((qid, list(ids[:DEPTH]), list(scores[:DEPTH])))
    return out


def stats_for_queries(
    gallery: GalleryIndex,
    rows: list[tuple[str, list[str], list[float]]],
    query_vectors: dict[str, np.ndarray],
    batch_size: int = 1024,
):
    """Yield ``(query_id, stats list)`` in rows order, ``batch_size`` queries at a time."""
    vecs = gallery.vectors().astype(np.float64)
    rows_of: dict[str, list[int]] = {}
    for i, pid in enumerate(gallery.product_ids):
        rows_of.setdefault(pid, []).append(i)
    mean_unit: dict[str, np.ndarray] = {}

    def unit_mean(pid: str) -> np.ndarray:
        if pid not in mean_unit:
            mean_unit[pid] = _unit(vecs[rows_of[pid]].mean(axis=0))
        return mean_unit[pid]

    for start in range(0, len(rows), batch_size):
        for qid, ids, scores in rows[start : start + batch_size]:
            missing = [p for p in ids if p not in rows_of]
            if missing:
                raise CandStatsError(f"query {qid!r}: ranked product {missing[0]!r} is not in the index")
            q = query_vectors[qid].astype(np.float64)
            top1 = unit_mean(ids[0]) if ids else None
            stats = []
            for rank, pid in enumerate(ids):
                s = product_stats(q, vecs[rows_of[pid]], unit_mean(pid), top1)
                if abs(s["max_sim"] - scores[rank]) > SCORE_TOL:
                    raise CandStatsError(
                        f"query {qid!r} product {pid!r}: max_sim {s['max_sim']:.7f} != "
                        f"rankings score {scores[rank]:.7f} (is the rankings file from this index?)"
                    )
                stats.append({"product_id": pid, **s})
            yield qid, stats


def run_cand_stats(
    config: ExperimentConfig,
    split: str,
    rankings_path: str | Path,
    embedder_name: EmbedderName = "siglip",
    artifacts_root: Path = Path("artifacts"),
    limit_products: int | None = None,
    final: bool = False,
    reports_root: Path = Path("reports"),
) -> CandStatsResult:
    """Write ``<rankings stem>.cand_stats.jsonl`` for ``rankings_path``; the rankings file is untouched."""
    if config.crop_kind != "full":
        raise NotImplementedError(f"crop kind {config.crop_kind!r} is not implemented; only 'full' is")
    rankings_path = Path(rankings_path)
    rows = _read_rankings(rankings_path)
    rankings_sha = sha256_file(rankings_path)

    selection = select_split(config, split, limit_products, final, reports_root, command="cand-stats")
    queries_by_id = {q.query_id: q for q in selection.queries}
    unknown = [qid for qid, _, _ in rows if qid not in queries_by_id]
    if unknown:
        raise CandStatsError(
            f"{len(unknown)} rankings query_ids are not in split {split!r} (first: {unknown[0]!r})"
        )

    embedder = _make_embedder(embedder_name, config)
    index_id, _, index_dir = locate_index(
        config, selection.products, embedder, embedder_name, split, limit_products, artifacts_root
    )
    gallery = GalleryIndex.load(index_dir)

    crop_hash_value = compute_crop_hash(CropSpec(kind="full"))
    cache = EmbeddingCache(artifacts_root / "embeddings")
    shas_with_source = [
        (queries_by_id[qid].image_sha, selection.row_source_by_product[queries_by_id[qid].truth_product_id])
        for qid, _, _ in rows
    ]
    by_sha = _embed_shas(shas_with_source, crop_hash_value, embedder, cache, config.data_root)
    query_vectors = {qid: by_sha[queries_by_id[qid].image_sha] for qid, _, _ in rows}

    out_path = cand_stats_path(rankings_path)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            for qid, stats in stats_for_queries(gallery, rows, query_vectors):
                f.write(
                    json.dumps(
                        {
                            "query_id": qid,
                            "rankings_sha256": rankings_sha,
                            "index_id": index_id,
                            "stats": stats,
                        }
                    )
                    + "\n"
                )
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return CandStatsResult(index_id, rankings_path, out_path, len(rows), rankings_sha)
