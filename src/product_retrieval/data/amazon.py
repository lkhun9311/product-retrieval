"""Amazon pairs reader (D20 section 3).

The raw JSONL uses ``"source": "amazon2023"`` and carries image *URLs*, not
locally-stored shas (the images have not been downloaded yet). This module
only yields a typed record of each line; it does not map rows to ``Product``/
``Query`` since those require local, content-addressed images. The
``"amazon2023"`` source literal is normalized to ``"amazon"`` on ``AmazonPair``
to match ``core.schemas.Source``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from product_retrieval.core.schemas import Source, Split

_SOURCE_ALIASES = {"amazon2023": "amazon"}


class AmazonPair(BaseModel):
    """One Amazon product record with its (not-yet-downloaded) query/gallery image URLs."""

    model_config = ConfigDict(frozen=True)

    source: Source
    product_id: str
    split: Split
    store: str | None = None
    title: str | None = None
    query_urls: list[str] = Field(default_factory=list)
    gallery_urls: list[str] = Field(default_factory=list)

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _SOURCE_ALIASES.get(v, v)
        return v


def iter_amazon_pairs(path: str | Path, split: str | None = None) -> Iterator[AmazonPair]:
    """Stream the Amazon pairs JSONL, normalizing ``amazon2023`` -> ``amazon``.

    When ``split`` is given, only rows whose split matches are yielded.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = AmazonPair.model_validate(json.loads(line))
            if split is not None and pair.split != split:
                continue
            yield pair
