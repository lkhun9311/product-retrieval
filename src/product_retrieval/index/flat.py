"""Exact flat gallery index over per-image embeddings (D20 sections 2, 5, 9).

Each row of the underlying ``faiss.IndexFlatIP`` is one gallery *image* embedding.
Per the evaluation contract (D20 section 9), a query is ranked against *products*:
a product's score is the max cosine similarity over that product's gallery images,
and the result list has no duplicate products. ``GalleryIndex.search`` does this by
scoring a query against every gallery image (exact, since ``IndexFlatIP`` never
approximates) and aggregating per product -- correct regardless of how many gallery
images any single product has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import faiss
import numpy as np


class ProductHit(NamedTuple):
    """One ranked product result: its best-matching gallery image and that image's score."""

    product_id: str
    score: float
    best_image_sha: str


class GalleryIndex:
    """A built gallery index: a faiss flat index plus parallel per-row metadata."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        product_ids: list[str],
        image_shas: list[str],
        params: dict[str, Any],
        index_id: str | None = None,
    ) -> None:
        if not (faiss_index.ntotal == len(product_ids) == len(image_shas)):
            raise ValueError("faiss_index, product_ids, and image_shas must have matching lengths")
        self._faiss_index = faiss_index
        self.product_ids = product_ids
        self.image_shas = image_shas
        self.params = dict(params)
        self.index_id = index_id

    @property
    def ntotal(self) -> int:
        return self._faiss_index.ntotal

    def vectors(self) -> np.ndarray:
        """All gallery image vectors, shape (ntotal, dim), row-aligned with ``product_ids``."""
        if self.ntotal == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return self._faiss_index.reconstruct_n(0, self.ntotal)

    def search(self, query_vecs: np.ndarray, k_products: int) -> list[list[ProductHit]]:
        """Return, for each query row, up to ``k_products`` distinct products ranked by score.

        A product's score is the max inner product between the query and any of that
        product's gallery image rows; ``best_image_sha`` is the row that achieved it.
        """
        n_queries = len(query_vecs)
        if self.ntotal == 0:
            return [[] for _ in range(n_queries)]

        query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
        scores, indices = self._faiss_index.search(query_vecs, self.ntotal)

        results: list[list[ProductHit]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            best_for_product: dict[str, tuple[float, str]] = {}
            for score, idx in zip(row_scores, row_indices, strict=True):
                if idx < 0:
                    continue
                product_id = self.product_ids[idx]
                image_sha = self.image_shas[idx]
                current = best_for_product.get(product_id)
                if current is None or score > current[0]:
                    best_for_product[product_id] = (float(score), image_sha)

            ranked = sorted(best_for_product.items(), key=lambda kv: kv[1][0], reverse=True)
            results.append([ProductHit(pid, score, sha) for pid, (score, sha) in ranked[:k_products]])
        return results

    def save(self, directory: Path | str) -> None:
        """Save ``faiss.index``, ``ids.jsonl``, and ``params.json`` under ``directory``."""
        if self.index_id is None:
            raise ValueError("cannot save a GalleryIndex with index_id=None")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(directory / "faiss.index"))

        with open(directory / "ids.jsonl", "w", encoding="utf-8") as f:
            for product_id, image_sha in zip(self.product_ids, self.image_shas, strict=True):
                f.write(json.dumps({"product_id": product_id, "image_sha": image_sha}) + "\n")

        with open(directory / "params.json", "w", encoding="utf-8") as f:
            json.dump({"index_id": self.index_id, "params": self.params}, f, indent=2)

    @classmethod
    def load(cls, directory: Path | str) -> GalleryIndex:
        """Load a ``GalleryIndex`` previously written by ``save``."""
        directory = Path(directory)
        faiss_index = faiss.read_index(str(directory / "faiss.index"))

        product_ids: list[str] = []
        image_shas: list[str] = []
        with open(directory / "ids.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                product_ids.append(row["product_id"])
                image_shas.append(row["image_sha"])

        with open(directory / "params.json", encoding="utf-8") as f:
            meta = json.load(f)

        return cls(
            faiss_index,
            product_ids,
            image_shas,
            meta["params"],
            index_id=meta["index_id"],
        )


def build_index(
    vectors: np.ndarray,
    product_ids: list[str],
    image_shas: list[str],
    params: dict[str, Any] | None = None,
) -> GalleryIndex:
    """Build an exact flat (inner-product) index over per-image gallery ``vectors``.

    ``vectors``, ``product_ids``, and ``image_shas`` must have matching lengths; row
    ``i`` is the embedding for ``(product_ids[i], image_shas[i])`` -- a product with
    several gallery images contributes several rows with the same ``product_id``.

    ``params`` currently only supports ``{"type": "flat"}`` (or ``{}``, which defaults
    to ``"flat"``); any other ``"type"`` raises ``NotImplementedError``. The built
    index has no ``index_id`` set -- callers compute one with
    ``core.ids.index_id(model_id, gallery_manifest_sha, params)`` and assign it
    before calling ``save``.
    """
    params = dict(params) if params else {}
    index_type = params.get("type", "flat")
    if index_type != "flat":
        raise NotImplementedError(f"index type {index_type!r} is not implemented; only 'flat' is supported")

    if not (len(vectors) == len(product_ids) == len(image_shas)):
        raise ValueError("vectors, product_ids, and image_shas must have matching lengths")

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors.shape[1] if vectors.ndim == 2 and len(vectors) else 0
    faiss_index = faiss.IndexFlatIP(dim)
    if len(vectors):
        faiss_index.add(vectors)

    return GalleryIndex(faiss_index, list(product_ids), list(image_shas), params)
