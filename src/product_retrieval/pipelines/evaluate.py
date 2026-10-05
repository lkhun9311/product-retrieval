"""`pr eval` (D20 sections 4A, 7, 9): score a built index and write an eval report.

Flow: select the same products/queries build-index would for this
``(config, split, limit_products)`` (``pipelines.selection.select_split``) ->
compute the expected ``index_id`` from the gallery actually selected -> load
``artifacts/index/{index_id}/`` (raising a clear error naming the missing index
if ``pr build-index`` hasn't been run for it yet) -> get query vectors from the
``EmbeddingCache``, embedding any missing ones -> search top ``max(ks)`` products
per query -> compute product-macro/micro R@K with bootstrap CIs
(``eval.report.build_metrics_and_ci``) -> write a JSON report plus a JSONL file
of per-query rankings for later error analysis (and the C4 feedback simulator).

Split access control (D20 section 9) lives in ``pipelines.selection``, the same
place build-index enforces it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from product_retrieval.core.config import ExperimentConfig, config_hash
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.ids import gallery_sha as compute_gallery_sha
from product_retrieval.core.ids import index_id as compute_index_id
from product_retrieval.core.schemas import CropSpec
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.eval.report import build_metrics_and_ci
from product_retrieval.eval.retrieval import DEFAULT_KS, QueryResult
from product_retrieval.index.flat import GalleryIndex
from product_retrieval.pipelines.build_index import EmbedderName, _embed_shas, _make_embedder
from product_retrieval.pipelines.selection import TestSplitAccessError, select_split


class IndexNotFoundError(RuntimeError):
    """Raised when ``artifacts/index/{index_id}/`` doesn't exist for the selected gallery."""


@dataclass(frozen=True)
class EvalResult:
    """What ``run_eval`` did, for the CLI to print and tests to assert on."""

    index_id: str
    report_path: Path
    rankings_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def locate_index(
    config: ExperimentConfig,
    products: list[Any],
    embedder: Any,
    embedder_name: str,
    split: str,
    limit_products: int | None,
    artifacts_root: Path,
) -> tuple[str, str, Path]:
    """Return ``(index_id, gallery_sha, index_dir)`` for the selected gallery; raise if not built."""
    gallery_pairs = [(product.product_id, sha) for product in products for sha in product.gallery_shas]
    params = dict(config.index_params)
    gallery_sha_value = compute_gallery_sha(gallery_pairs)
    expected_index_id = compute_index_id(embedder.model_id, gallery_sha_value, params)

    index_dir = artifacts_root / "index" / expected_index_id
    if not (index_dir / "faiss.index").is_file():
        final_flag = " --final" if split == "test" else ""
        limit_flag = f" --limit-products {limit_products}" if limit_products is not None else ""
        raise IndexNotFoundError(
            f"no index found at {index_dir} (expected index_id={expected_index_id!r} for this "
            f"config/split/limit-products). Run `pr build-index --config <config> --split {split}"
            f"{limit_flag}{final_flag} --embedder {embedder_name}` first."
        )
    return expected_index_id, gallery_sha_value, index_dir


def run_eval(
    config: ExperimentConfig,
    split: str,
    final: bool = False,
    limit_products: int | None = None,
    embedder_name: EmbedderName = "siglip",
    ks: tuple[int, ...] = DEFAULT_KS,
    b: int = 1000,
    seed: int = 0,
    artifacts_root: Path = Path("artifacts"),
    reports_root: Path = Path("reports"),
) -> EvalResult:
    """Run the eval pipeline against a previously built index and return where its outputs landed."""
    if config.crop_kind != "full":
        raise NotImplementedError(f"crop kind {config.crop_kind!r} is not implemented; only 'full' is")

    selection = select_split(config, split, limit_products, final, reports_root, command="eval")
    products = selection.products
    queries = selection.queries
    row_source_by_product = selection.row_source_by_product
    manifest_sha_value = selection.manifest_sha

    if not queries:
        raise ValueError(
            f"no queries for split={split!r} (after limit_products={limit_products!r}); nothing to evaluate"
        )

    crop_hash_value = compute_crop_hash(CropSpec(kind="full"))
    embedder = _make_embedder(embedder_name, config)

    expected_index_id, gallery_sha_value, index_dir = locate_index(
        config, products, embedder, embedder_name, split, limit_products, artifacts_root
    )
    gallery = GalleryIndex.load(index_dir)

    stages: dict[str, float] = {}
    cache = EmbeddingCache(artifacts_root / "embeddings")

    query_shas_with_source = [(q.image_sha, row_source_by_product[q.truth_product_id]) for q in queries]
    start = time.perf_counter()
    query_vectors_by_sha = _embed_shas(
        query_shas_with_source, crop_hash_value, embedder, cache, config.data_root
    )
    stages["embed_query_seconds"] = time.perf_counter() - start

    max_k = max(ks)
    start = time.perf_counter()
    query_vecs = np.stack([query_vectors_by_sha[q.image_sha] for q in queries]).astype(np.float32)
    all_hits = gallery.search(query_vecs, max_k)
    stages["search_seconds"] = time.perf_counter() - start

    results: list[QueryResult] = []
    rankings: list[dict[str, Any]] = []
    for query, hits in zip(queries, all_hits, strict=True):
        ranked_product_ids = tuple(hit.product_id for hit in hits)
        results.append(
            QueryResult(
                query_id=query.query_id,
                truth_product_id=query.truth_product_id,
                ranked_product_ids=ranked_product_ids,
            )
        )
        rankings.append(
            {
                "query_id": query.query_id,
                "truth_product_id": query.truth_product_id,
                "top_k_product_ids": list(ranked_product_ids),
                "scores": [hit.score for hit in hits],
            }
        )

    start = time.perf_counter()
    metrics, ci = build_metrics_and_ci(results, ks, b=b, seed=seed)
    stages["metrics_seconds"] = time.perf_counter() - start

    report = {
        "metrics": metrics,
        "ci": ci,
        "counts": {"products": len(products), "queries": len(queries)},
        "index_id": expected_index_id,
        "model_id": embedder.model_id,
        "manifest_sha": manifest_sha_value,
        "gallery_sha": gallery_sha_value,
        "config_hash": config_hash(config),
        "split": split,
        "final": final,
        "b": b,
        "seed": seed,
        "seconds": stages,
    }

    eval_dir = reports_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{expected_index_id}__{config_hash(config)}__{split}"
    report_path = eval_dir / f"{stem}.json"
    rankings_path = eval_dir / f"{stem}.rankings.jsonl"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(rankings_path, "w", encoding="utf-8") as f:
        for row in rankings:
            f.write(json.dumps(row) + "\n")

    return EvalResult(
        index_id=expected_index_id,
        report_path=report_path,
        rankings_path=rankings_path,
        report=report,
    )


def format_report_table(report: dict[str, Any]) -> str:
    """Render a compact macro/micro R@K + 95% CI table for ``report`` (``run_eval``'s output)."""
    metrics = report["metrics"]
    ci = report["ci"]
    ks = sorted(int(k.split("@")[1]) for k in metrics["recall"]["macro"])

    lines = [
        f"index_id={report['index_id'][:12]}  split={report['split']}  "
        f"products={report['counts']['products']}  queries={report['counts']['queries']}  "
        f"b={report['b']}  seed={report['seed']}",
        f"{'K':>5}  {'macro R@K':>18}  {'micro R@K':>18}",
    ]
    for k in ks:
        key = f"R@{k}"
        macro_v = metrics["recall"]["macro"][key]
        micro_v = metrics["recall"]["micro"][key]
        macro_lo, macro_hi = ci["recall"]["macro"][key]
        micro_lo, micro_hi = ci["recall"]["micro"][key]
        lines.append(
            f"{k:>5}  {macro_v:>7.4f} [{macro_lo:.4f},{macro_hi:.4f}]  "
            f"{micro_v:>7.4f} [{micro_lo:.4f},{micro_hi:.4f}]"
        )
    return "\n".join(lines)


__all__ = [
    "EvalResult",
    "IndexNotFoundError",
    "TestSplitAccessError",
    "format_report_table",
    "locate_index",
    "run_eval",
]
