"""Shared product/query selection for `pr build-index` and `pr eval` (D20 sections 4A, 9).

Both pipelines must select the *identical* products and queries from a given
manifest + split + ``limit_products`` -- same sort order, same truncation -- so
that build-index's ``index_id`` and eval's expected ``index_id`` agree bit for
bit. This module is the single place that selection happens, and the single
place the test-split access control (D20 section 9: ``val`` is open, ``test``
requires ``--final`` and is logged) is enforced -- build-index and eval both
call through here instead of each re-implementing the refusal/log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from product_retrieval.core.config import ExperimentConfig, config_hash
from product_retrieval.core.schemas import Product, Query
from product_retrieval.data.manifest import load_manifest, manifest_sha, to_products, to_queries


class TestSplitAccessError(RuntimeError):
    """Raised when ``split="test"`` is requested without ``final=True`` (D20 section 9)."""

    __test__ = False  # not a pytest test class; name refers to the manifest "test" split


@dataclass(frozen=True)
class SplitSelection:
    """The products/queries selected for one ``(config, split, limit_products)``."""

    products: list[Product]
    queries: list[Query]
    row_source_by_product: dict[str, str]
    manifest_sha: str


def select_split(
    config: ExperimentConfig,
    split: str,
    limit_products: int | None,
    final: bool,
    reports_root: Path,
    command: str = "build-index",
) -> SplitSelection:
    """Select the products/queries for ``split``, identically for build and eval.

    Products are sorted by ``product_id`` and truncated to the first
    ``limit_products`` (if given); queries come from exactly those products'
    manifest rows. Raises ``TestSplitAccessError`` if ``split="test"`` and
    ``final`` is not ``True``; an allowed test access is appended to
    ``reports_root/test_access.jsonl`` (``command`` names the caller, e.g.
    ``"build-index"`` or ``"eval"``, for provenance).
    """
    if split == "test":
        if not final:
            raise TestSplitAccessError("split='test' requires --final (D20 section 9)")
        _log_test_access(config, reports_root, command)

    rows = load_manifest(config.manifest_path)
    manifest_sha_value = manifest_sha(config.manifest_path)

    products = sorted(to_products(rows, split=split), key=lambda p: p.product_id)
    if limit_products is not None:
        products = products[:limit_products]
    selected_product_ids = {p.product_id for p in products}

    split_rows = [r for r in rows if r.split == split and r.product_id in selected_product_ids]
    queries = to_queries(split_rows, split=split)
    row_source_by_product = {r.product_id: r.source for r in split_rows}

    return SplitSelection(
        products=products,
        queries=queries,
        row_source_by_product=row_source_by_product,
        manifest_sha=manifest_sha_value,
    )


def _log_test_access(config: ExperimentConfig, reports_root: Path, command: str) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "config_hash": config_hash(config),
        "split": "test",
        "command": command,
    }
    with open(reports_root / "test_access.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


__all__ = ["SplitSelection", "TestSplitAccessError", "select_split"]
