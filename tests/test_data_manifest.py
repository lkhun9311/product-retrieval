import json

import pytest
from pydantic import ValidationError

from product_retrieval.data.manifest import (
    ManifestRow,
    iter_manifest,
    load_manifest,
    manifest_sha,
    to_products,
    to_queries,
)

SHA_Q1 = "1" * 64
SHA_Q2 = "2" * 64
SHA_G1 = "3" * 64
SHA_G2 = "4" * 64


def _write_manifest(tmp_path, rows: list[dict]) -> "object":
    path = tmp_path / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _row(product_id="100354", split="test", query=(SHA_Q1,), gallery=(SHA_G1,)) -> dict:
    return {
        "source": "lrvs",
        "product_id": product_id,
        "split": split,
        "query": list(query),
        "gallery": list(gallery),
    }


def test_manifest_row_parses_lrvs_line():
    row = ManifestRow.model_validate(_row())
    assert row.source == "lrvs"
    assert row.product_id == "100354"
    assert row.query == [SHA_Q1]
    assert row.gallery == [SHA_G1]


def test_manifest_row_rejects_bad_sha():
    bad = _row(query=("not-a-sha",))
    with pytest.raises(ValidationError):
        ManifestRow.model_validate(bad)


def test_iter_manifest_streams_rows(tmp_path):
    path = _write_manifest(tmp_path, [_row(product_id="p1"), _row(product_id="p2")])
    rows = list(iter_manifest(path))
    assert [r.product_id for r in rows] == ["p1", "p2"]


def test_iter_manifest_skips_blank_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(_row(product_id="p1")) + "\n\n\n", encoding="utf-8")
    rows = list(iter_manifest(path))
    assert len(rows) == 1


def test_load_manifest_returns_list(tmp_path):
    path = _write_manifest(tmp_path, [_row(product_id="p1"), _row(product_id="p2")])
    rows = load_manifest(path)
    assert isinstance(rows, list)
    assert len(rows) == 2


def test_manifest_sha_deterministic_and_content_sensitive(tmp_path):
    path_a = tmp_path / "a.jsonl"
    path_a.write_text(json.dumps(_row(product_id="p1")) + "\n", encoding="utf-8")
    path_b = tmp_path / "copy.jsonl"
    path_b.write_bytes(path_a.read_bytes())
    assert manifest_sha(path_a) == manifest_sha(path_b)

    path_c = tmp_path / "c.jsonl"
    path_c.write_text(json.dumps(_row(product_id="p2")) + "\n", encoding="utf-8")
    assert manifest_sha(path_a) != manifest_sha(path_c)


def test_to_products_counts_and_fields(tmp_path):
    rows = [
        _row(product_id="p1", split="train", gallery=(SHA_G1, SHA_G2)),
        _row(product_id="p2", split="test", gallery=(SHA_G1,)),
    ]
    rows = [ManifestRow.model_validate(r) for r in rows]
    products = to_products(rows)
    assert len(products) == 2
    p1 = next(p for p in products if p.product_id == "p1")
    assert p1.source == "lrvs"
    assert p1.split == "train"
    assert p1.gallery_shas == [SHA_G1, SHA_G2]


def test_to_products_filters_by_split(tmp_path):
    rows = [
        ManifestRow.model_validate(_row(product_id="p1", split="train")),
        ManifestRow.model_validate(_row(product_id="p2", split="test")),
    ]
    products = to_products(rows, split="test")
    assert [p.product_id for p in products] == ["p2"]


def test_to_queries_one_per_query_image_with_expected_id_format():
    rows = [ManifestRow.model_validate(_row(product_id="100354", query=(SHA_Q1, SHA_Q2)))]
    queries = to_queries(rows)
    assert len(queries) == 2
    ids = {q.query_id for q in queries}
    assert ids == {
        f"lrvs:100354:{SHA_Q1[:12]}",
        f"lrvs:100354:{SHA_Q2[:12]}",
    }
    for q in queries:
        assert q.truth_product_id == "100354"
        assert q.source == "lrvs"


def test_to_queries_filters_by_split():
    rows = [
        ManifestRow.model_validate(_row(product_id="p1", split="train", query=(SHA_Q1,))),
        ManifestRow.model_validate(_row(product_id="p2", split="test", query=(SHA_Q2,))),
    ]
    queries = to_queries(rows, split="test")
    assert len(queries) == 1
    assert queries[0].truth_product_id == "p2"
