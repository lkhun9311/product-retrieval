"""Content-addressed identifiers (D20 section 1: every artifact is identified by content).

All functions here are pure and deterministic: the same input always produces the
same output, independent of dict key order or process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from product_retrieval.core.schemas import CropSpec

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex sha256 digest of the file at ``path``, streamed in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a canonical JSON string: sorted keys, compact separators, UTF-8 text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def crop_hash(spec: CropSpec) -> str:
    """Hash of the crop spec's normalized JSON representation (D20 section 3)."""
    data = spec.model_dump(mode="json", exclude_none=True)
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def gallery_sha(pairs: Sequence[tuple[str, str]]) -> str:
    """Identify the gallery actually indexed: hash of its sorted ``(product_id, image_sha)`` rows.

    This -- not the source manifest's sha -- is what makes ``index_id`` sensitive to
    *which* products/images ended up in the gallery. Without it, a val build, a val
    build with ``--limit-products``, and a test build from the same manifest file
    would all compute the same ``index_id`` and silently overwrite each other's
    artifacts (D20 section 1: artifacts are identified by content).
    """
    sorted_pairs = sorted(pairs)
    data = [[product_id, image_sha] for product_id, image_sha in sorted_pairs]
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def index_id(model_id: str, gallery_sha: str, params: dict[str, Any]) -> str:
    """Identify a built index by (embedding model, gallery content, build params).

    ``gallery_sha`` should be ``core.ids.gallery_sha`` of the ``(product_id,
    image_sha)`` rows actually indexed -- not the source manifest's sha, which only
    identifies where the gallery *could* have come from, not which subset of it was
    used (see ``gallery_sha`` above). The manifest sha is still worth recording for
    provenance; callers do so separately in their build summary.
    """
    data = {
        "model_id": model_id,
        "gallery_sha": gallery_sha,
        "params": params,
    }
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def bundle_id(
    embed_model_id: str,
    crop_policy: str,
    index_id: str,
    reranker_version: str | None = None,
    calibrator_version: str | None = None,
) -> str:
    """Identify a deployable bundle by the sha256 of its component ids' canonical JSON."""
    data = {
        "embed_model_id": embed_model_id,
        "crop_policy": crop_policy,
        "index_id": index_id,
        "reranker_version": reranker_version,
        "calibrator_version": calibrator_version,
    }
    return sha256_bytes(canonical_json(data).encode("utf-8"))
