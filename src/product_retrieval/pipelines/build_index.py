"""`pr build-index` (D20 sections 4A, 7, 9): embed the gallery, build a flat index.

Flow: select products/queries for a split (``pipelines.selection.select_split``) ->
embed gallery images (building the index) and query images (warming the cache for
a later `pr eval`) -> build & save a ``GalleryIndex`` -> write a run summary JSON.

Split access control (D20 section 9) lives in ``pipelines.selection``: ``val`` is
open; ``test`` is refused unless ``final=True``, and every time it is opened with
``final=True`` a line is appended to ``reports/test_access.jsonl``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from product_retrieval.core.config import ExperimentConfig, config_hash
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.ids import gallery_sha as compute_gallery_sha
from product_retrieval.core.ids import index_id as compute_index_id
from product_retrieval.core.schemas import CropSpec
from product_retrieval.data.store import ImageStore
from product_retrieval.embed.base import Embedder
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.embed.siglip import SiglipEmbedder
from product_retrieval.index.flat import GalleryIndex, build_index
from product_retrieval.pipelines.selection import TestSplitAccessError, select_split

EmbedderName = Literal["siglip", "fake"]


@dataclass(frozen=True)
class BuildIndexResult:
    """What ``run_build_index`` did, for the CLI to print and tests to assert on."""

    index_id: str
    index_dir: Path
    summary_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _make_embedder(name: EmbedderName, config: ExperimentConfig) -> Embedder:
    if name == "siglip":
        return SiglipEmbedder(model_id=config.embed_model_id)
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"unknown embedder {name!r}; expected 'siglip' or 'fake'")


def _embed_shas(
    shas_with_source: list[tuple[str, str]],
    crop_hash_value: str,
    embedder: Embedder,
    cache: EmbeddingCache,
    data_root: Path,
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Embed the unique shas in ``shas_with_source``, using and filling ``cache``.

    Cache-missing shas are processed ``chunk_size`` at a time: only one chunk of
    decoded images is in memory, and each chunk is written to the cache before the
    next starts, so a crash loses at most one chunk and a rerun skips the rest.

    Returns a ``{sha: vector}`` map. First-seen source wins for a sha seen under
    more than one source (shas are content-addressed and expected to agree).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    source_by_sha: dict[str, str] = {}
    for sha, source in shas_with_source:
        source_by_sha.setdefault(sha, source)
    shas = list(source_by_sha.keys())
    if not shas:
        return {}

    present_vectors, missing = cache.get_many(shas, crop_hash_value, embedder.model_id)
    missing_set = set(missing)
    present_shas = [sha for sha in shas if sha not in missing_set]
    vector_by_sha: dict[str, np.ndarray] = dict(zip(present_shas, present_vectors, strict=True))

    if missing:
        stores: dict[str, ImageStore] = {}
        total = len(missing)
        cached = len(shas) - total
        done = 0
        start = time.perf_counter()
        for offset in range(0, total, chunk_size):
            chunk = missing[offset : offset + chunk_size]
            images = []
            for sha in chunk:
                source = source_by_sha[sha]
                store = stores.setdefault(source, ImageStore(data_root, source))
                images.append(store.open_image(sha))
            new_vectors = embedder.embed(images)
            del images  # drop decoded images before the next chunk is opened
            cache.put_many(chunk, new_vectors, crop_hash_value, embedder.model_id)
            vector_by_sha.update(zip(chunk, new_vectors, strict=True))
            done += len(chunk)
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = _format_eta((total - done) / rate) if rate > 0 else "?"
            print(
                f"embed: {done}/{total} done (cached {cached}) {rate:.1f} img/s eta {eta}",
                file=sys.stderr,
                flush=True,
            )

    return vector_by_sha


def _format_eta(seconds: float) -> str:
    minutes_total = int(round(seconds / 60))
    hours, minutes = divmod(minutes_total, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m" if minutes else f"{int(seconds)}s"


def run_build_index(
    config: ExperimentConfig,
    split: str,
    final: bool = False,
    limit_products: int | None = None,
    embedder_name: EmbedderName = "siglip",
    artifacts_root: Path = Path("artifacts"),
    reports_root: Path = Path("reports"),
) -> BuildIndexResult:
    """Run the build-index pipeline and return where its outputs landed."""
    if config.crop_kind != "full":
        raise NotImplementedError(f"crop kind {config.crop_kind!r} is not implemented; only 'full' is")

    selection = select_split(config, split, limit_products, final, reports_root, command="build-index")
    products = selection.products
    queries = selection.queries
    row_source_by_product = selection.row_source_by_product
    gallery_manifest_sha = selection.manifest_sha

    stages: dict[str, float] = {}

    crop_hash_value = compute_crop_hash(CropSpec(kind="full"))

    embedder = _make_embedder(embedder_name, config)
    cache = EmbeddingCache(artifacts_root / "embeddings")

    gallery_pairs = [(product.product_id, sha) for product in products for sha in product.gallery_shas]
    gallery_shas_with_source = [(sha, row_source_by_product[product_id]) for product_id, sha in gallery_pairs]

    start = time.perf_counter()
    gallery_vectors_by_sha = _embed_shas(
        gallery_shas_with_source, crop_hash_value, embedder, cache, config.data_root
    )
    stages["embed_gallery_seconds"] = time.perf_counter() - start

    query_shas_with_source = [(q.image_sha, row_source_by_product[q.truth_product_id]) for q in queries]
    start = time.perf_counter()
    _embed_shas(query_shas_with_source, crop_hash_value, embedder, cache, config.data_root)
    stages["embed_query_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    vectors = (
        np.stack([gallery_vectors_by_sha[sha] for _, sha in gallery_pairs]).astype(np.float32)
        if gallery_pairs
        else np.zeros((0, embedder.dim), dtype=np.float32)
    )
    product_ids_rows = [pid for pid, _ in gallery_pairs]
    image_shas_rows = [sha for _, sha in gallery_pairs]
    params = dict(config.index_params)
    gallery = build_index(vectors, product_ids_rows, image_shas_rows, params)
    gallery_sha_value = compute_gallery_sha(gallery_pairs)
    computed_index_id = compute_index_id(embedder.model_id, gallery_sha_value, params)
    gallery.index_id = computed_index_id
    stages["build_index_seconds"] = time.perf_counter() - start

    index_dir = artifacts_root / "index" / computed_index_id
    start = time.perf_counter()
    gallery.save(index_dir)
    stages["save_index_seconds"] = time.perf_counter() - start

    total_images = len(gallery_shas_with_source) + len(query_shas_with_source)
    embed_seconds = stages["embed_gallery_seconds"] + stages["embed_query_seconds"]
    images_per_second = total_images / embed_seconds if embed_seconds > 0 else None

    summary = {
        "split": split,
        "final": final,
        "embedder": embedder_name,
        "model_id": embedder.model_id,
        "index_id": computed_index_id,
        "manifest_sha": gallery_manifest_sha,
        "gallery_sha": gallery_sha_value,
        "config_hash": config_hash(config),
        "limit_products": limit_products,
        "counts": {
            "products": len(products),
            "gallery_images": len(gallery_pairs),
            "unique_gallery_shas": len(gallery_vectors_by_sha),
            "queries": len(queries),
        },
        "seconds": stages,
        "images_per_second": images_per_second,
    }

    summary_dir = reports_root / "build_index"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{computed_index_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return BuildIndexResult(
        index_id=computed_index_id,
        index_dir=index_dir,
        summary_path=summary_path,
        summary=summary,
    )


__all__ = ["BuildIndexResult", "GalleryIndex", "TestSplitAccessError", "run_build_index"]
