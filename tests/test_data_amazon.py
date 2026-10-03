import json

import pytest
from pydantic import ValidationError

from product_retrieval.data.amazon import AmazonPair, iter_amazon_pairs


def _write_pairs(tmp_path, rows: list[dict]):
    path = tmp_path / "amazon_pairs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _raw(product_id="B097DQPCP2", split="train") -> dict:
    return {
        "source": "amazon2023",
        "product_id": product_id,
        "split": split,
        "store": "Acme",
        "title": "Widget",
        "query_urls": ["https://example.com/q1.jpg"],
        "gallery_urls": ["https://example.com/g1.jpg", "https://example.com/g2.jpg"],
    }


def test_amazon_pair_normalizes_source_literal():
    pair = AmazonPair.model_validate(_raw())
    assert pair.source == "amazon"


def test_amazon_pair_rejects_unknown_source():
    raw = _raw()
    raw["source"] = "ebay"
    with pytest.raises(ValidationError):
        AmazonPair.model_validate(raw)


def test_iter_amazon_pairs_streams_and_normalizes(tmp_path):
    path = _write_pairs(tmp_path, [_raw(product_id="p1"), _raw(product_id="p2")])
    pairs = list(iter_amazon_pairs(path))
    assert [p.product_id for p in pairs] == ["p1", "p2"]
    assert all(p.source == "amazon" for p in pairs)
    assert pairs[0].query_urls == ["https://example.com/q1.jpg"]
    assert len(pairs[0].gallery_urls) == 2


def test_iter_amazon_pairs_filters_by_split(tmp_path):
    path = _write_pairs(
        tmp_path,
        [_raw(product_id="p1", split="train"), _raw(product_id="p2", split="test")],
    )
    pairs = list(iter_amazon_pairs(path, split="test"))
    assert [p.product_id for p in pairs] == ["p2"]
