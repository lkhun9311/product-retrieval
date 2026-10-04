"""LRVS-style manifest loading (D20 section 3, section 5).

One line = one product: ``{"source", "product_id", "split", "query": [sha...],
"gallery": [sha...]}``. ``query`` shas are scene/worn photos; ``gallery`` shas
are standalone product photos.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from product_retrieval.core.ids import query_id, sha256_file
from product_retrieval.core.schemas import Product, Query, Source, Split
from product_retrieval.data.store import validate_sha


class ManifestRow(BaseModel):
    """One line of an LRVS-style manifest: a product with its query/gallery image shas."""

    model_config = ConfigDict(frozen=True)

    source: Source
    product_id: str
    split: Split
    query: list[str] = Field(default_factory=list)
    gallery: list[str] = Field(default_factory=list)

    @field_validator("query", "gallery")
    @classmethod
    def _check_shas(cls, v: list[str]) -> list[str]:
        return [validate_sha(sha) for sha in v]


def iter_manifest(path: str | Path) -> Iterator[ManifestRow]:
    """Stream a JSONL manifest, yielding one ``ManifestRow`` per non-blank line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield ManifestRow.model_validate(json.loads(line))


def load_manifest(path: str | Path) -> list[ManifestRow]:
    """Load an entire JSONL manifest into memory."""
    return list(iter_manifest(path))


def manifest_sha(path: str | Path) -> str:
    """sha256 of the manifest file's bytes (D20 section 3: ``dataset_manifest_sha``)."""
    return sha256_file(path)


def to_products(rows: list[ManifestRow], split: str | None = None) -> list[Product]:
    """Convert manifest rows into ``Product`` records, optionally filtered by ``split``."""
    selected = rows if split is None else [r for r in rows if r.split == split]
    return [
        Product(
            product_id=row.product_id,
            source=row.source,
            split=row.split,
            gallery_shas=list(row.gallery),
        )
        for row in selected
    ]


def to_queries(rows: list[ManifestRow], split: str | None = None) -> list[Query]:
    """Convert manifest rows into one ``Query`` per query image, optionally filtered by ``split``.

    ``query_id`` is the opaque ``core.ids.query_id(source, product_id, image_sha)``
    (``f"{source}:{hash[:16]}"``); it does not expose the truth product id.
    """
    selected = rows if split is None else [r for r in rows if r.split == split]
    queries: list[Query] = []
    for row in selected:
        for sha in row.query:
            queries.append(
                Query(
                    query_id=query_id(row.source, row.product_id, sha),
                    image_sha=sha,
                    truth_product_id=row.product_id,
                    source=row.source,
                )
            )
    return queries
