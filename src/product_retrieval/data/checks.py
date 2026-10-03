"""Manifest completeness and leakage checks (D20 section 3): the gate run before indexing.

``check_manifest`` reports, per split, how many products/queries/gallery images a
manifest declares, which image shas are missing from the ``ImageStore``, which
shas are shared across more than one product, and which products appear in more
than one split (true leakage — this must be zero).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from product_retrieval.data.manifest import ManifestRow
from product_retrieval.data.store import ImageStore


class SplitCounts(BaseModel):
    """Per-split row/image counts declared by a manifest."""

    model_config = ConfigDict(frozen=True)

    products: int = 0
    queries: int = 0
    gallery_images: int = 0


class CheckReport(BaseModel):
    """Result of ``check_manifest``: completeness and leakage findings."""

    model_config = ConfigDict(frozen=True)

    counts_by_split: dict[str, SplitCounts] = Field(default_factory=dict)
    missing_image_shas: list[str] = Field(default_factory=list)
    duplicate_shas: list[str] = Field(default_factory=list)
    cross_split_products: list[str] = Field(default_factory=list)
    ok: bool


def check_manifest(rows: list[ManifestRow], store: ImageStore) -> CheckReport:
    """Check manifest ``rows`` against ``store`` for missing images and leakage.

    ``ok`` is False when any image sha is missing from the store, or when any
    product appears in more than one split.
    """
    counts: dict[str, dict[str, int]] = {}
    product_splits: dict[str, set[str]] = {}
    sha_owners: dict[str, set[str]] = {}
    missing: set[str] = set()

    for row in rows:
        split_counts = counts.setdefault(row.split, {"products": 0, "queries": 0, "gallery_images": 0})
        split_counts["products"] += 1
        split_counts["queries"] += len(row.query)
        split_counts["gallery_images"] += len(row.gallery)

        product_splits.setdefault(row.product_id, set()).add(row.split)

        for sha in (*row.query, *row.gallery):
            sha_owners.setdefault(sha, set()).add(row.product_id)
            if sha not in missing and not store.exists(sha):
                missing.add(sha)

    duplicate_shas = sorted(sha for sha, owners in sha_owners.items() if len(owners) > 1)
    cross_split_products = sorted(pid for pid, splits in product_splits.items() if len(splits) > 1)
    counts_by_split = {split: SplitCounts(**c) for split, c in counts.items()}

    return CheckReport(
        counts_by_split=counts_by_split,
        missing_image_shas=sorted(missing),
        duplicate_shas=duplicate_shas,
        cross_split_products=cross_split_products,
        ok=not missing and not cross_split_products,
    )
