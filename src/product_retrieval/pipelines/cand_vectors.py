"""Query and best-gallery-image vectors for (query_id, product_id) pairs (contract c5-reranker-v2 section 2).

Uses the same split selection, index lookup and cached query embeddings as ``pr eval`` and
``pr cand-stats``. For a pair, ``q`` is the query vector and ``g`` the vector of the product's gallery
image with the largest inner product with ``q`` (the first one on ties); both are L2-normalised
float64. ``truth_product_id`` of the rankings file is never read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.schemas import CropSpec
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.index.flat import GalleryIndex
from product_retrieval.pipelines.build_index import EmbedderName, _embed_shas, _make_embedder
from product_retrieval.pipelines.evaluate import locate_index
from product_retrieval.pipelines.selection import select_split


class VectorSourceError(ValueError):
    """Raised when vectors cannot be sourced consistently (index mismatch, unknown ids)."""


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class CandidateVectors:
    """``pair(query_id, product_id) -> (q, g)`` over an index and a dict of query vectors."""

    def __init__(
        self,
        gallery: GalleryIndex,
        query_vectors: dict[str, np.ndarray],
        index_id: str,
        embed_model_id: str | None = None,
    ):
        self.index_id = index_id
        self.embed_model_id = embed_model_id
        self._vecs = gallery.vectors().astype(np.float64)
        self._rows_of: dict[str, list[int]] = {}
        for i, pid in enumerate(gallery.product_ids):
            self._rows_of.setdefault(pid, []).append(i)
        self._q = {qid: _unit(np.asarray(v, dtype=np.float64)) for qid, v in query_vectors.items()}

    def pair(self, query_id: str, product_id: str) -> tuple[np.ndarray, np.ndarray]:
        if query_id not in self._q:
            raise VectorSourceError(f"no query vector for query_id {query_id!r}")
        rows = self._rows_of.get(product_id)
        if rows is None:
            raise VectorSourceError(f"product {product_id!r} is not in the index")
        q = self._q[query_id]
        g = self._vecs[rows]
        best = int(np.argmax(g @ q))
        return q, _unit(g[best])


def load_candidate_vectors(
    config: ExperimentConfig,
    split: str,
    query_ids: list[str],
    expected_index_id: str | None,
    embedder_name: EmbedderName = "siglip",
    artifacts_root: Path = Path("artifacts"),
    limit_products: int | None = None,
    reports_root: Path = Path("reports"),
) -> CandidateVectors:
    """Vectors for ``query_ids`` of ``split``; fails if the located index is not ``expected_index_id``."""
    if config.crop_kind != "full":
        raise NotImplementedError(f"crop kind {config.crop_kind!r} is not implemented; only 'full' is")
    selection = select_split(config, split, limit_products, False, reports_root, command="rerank-v2")
    queries_by_id = {q.query_id: q for q in selection.queries}
    unknown = [qid for qid in query_ids if qid not in queries_by_id]
    if unknown:
        raise VectorSourceError(
            f"{len(unknown)} query_ids are not in split {split!r} (first: {unknown[0]!r})"
        )
    embedder = _make_embedder(embedder_name, config)
    index_id, _, index_dir = locate_index(
        config, selection.products, embedder, embedder_name, split, limit_products, artifacts_root
    )
    if expected_index_id is not None and index_id != expected_index_id:
        raise VectorSourceError(
            f"index {index_id!r} for this config/split/limit-products differs from the "
            f"cand-stats index_id {expected_index_id!r}"
        )
    gallery = GalleryIndex.load(index_dir)
    cache = EmbeddingCache(artifacts_root / "embeddings")
    crop_hash_value = compute_crop_hash(CropSpec(kind="full"))
    shas_with_source = [
        (queries_by_id[qid].image_sha, selection.row_source_by_product[queries_by_id[qid].truth_product_id])
        for qid in query_ids
    ]
    by_sha = _embed_shas(shas_with_source, crop_hash_value, embedder, cache, config.data_root)
    return CandidateVectors(
        gallery,
        {qid: by_sha[queries_by_id[qid].image_sha] for qid in query_ids},
        index_id,
        embedder.model_id,
    )


def cand_stats_index_id(cand_stats: list[dict]) -> str:
    """The single ``index_id`` the cand-stats rows were written with."""
    ids = sorted({r.get("index_id") for r in cand_stats})
    if len(ids) != 1 or ids[0] is None:
        raise VectorSourceError(f"cand-stats rows must carry exactly one index_id, found {ids}")
    return ids[0]


__all__ = ["CandidateVectors", "VectorSourceError", "cand_stats_index_id", "load_candidate_vectors"]
