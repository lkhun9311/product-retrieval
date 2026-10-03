"""Tests for product_retrieval.eval.retrieval (D20 section 9 eval contract v1).

Hand-computed example (also used as the basis for the bootstrap tests):

Gallery: P1, P2, P3 (truth products) plus distractors D1..D5. 6 queries total, with
1 query for P1, 2 for P2, 3 for P3 (uneven query counts per product, which is what
makes product-macro differ from micro).

    query  truth  ranked list                              hit@1  hit@5
    q1     P1     [P1, D1, D2, D3, D4, D5]                   1      1
    q2     P2     [D1, D2, D3, D4, D5, P2]  (P2 at rank 6)    0      0
    q3     P2     [P2, D1, D2, D3, D4, D5]                   1      1
    q4     P3     [D1, P3, D2, D3, D4, D5]  (P3 at rank 2)    0      1
    q5     P3     [D1, D2, D3, D4, D5, P3]  (P3 at rank 6)    0      0
    q6     P3     [P3, D1, D2, D3, D4, D5]                   1      1

Per-product R@1 = mean of that product's query hits:
    P1: hits=[1]       -> 1/1 = 1.0
    P2: hits=[0, 1]     -> 1/2 = 0.5
    P3: hits=[0, 0, 1]  -> 1/3 = 0.333333...
    macro R@1 = (1.0 + 0.5 + 1/3) / 3 = 0.6111111111111112
    micro R@1 = (1+0 + 1+0+0+1) / 6   = 3/6 = 0.5

Per-product R@5:
    P1: hits=[1]       -> 1.0
    P2: hits=[0, 1]     -> 0.5
    P3: hits=[1, 0, 1]  -> 2/3 = 0.666666...
    macro R@5 = (1.0 + 0.5 + 2/3) / 3 = 0.7222222222222222
    micro R@5 = (1+0 + 1+1+0+1) / 6   = 4/6 = 0.6666666666666666

macro != micro at both K: this is the point of product-macro averaging.
"""

from __future__ import annotations

import pytest

from product_retrieval.eval.retrieval import (
    ProductHits,
    QueryResult,
    hits_at_k,
    recall_at_k,
)

DISTRACTORS = ["D1", "D2", "D3", "D4", "D5"]


def _hand_example() -> list[QueryResult]:
    return [
        QueryResult("q1", "P1", tuple(["P1", *DISTRACTORS])),
        QueryResult("q2", "P2", tuple([*DISTRACTORS, "P2"])),
        QueryResult("q3", "P2", tuple(["P2", *DISTRACTORS])),
        QueryResult("q4", "P3", tuple(["D1", "P3", "D2", "D3", "D4", "D5"])),
        QueryResult("q5", "P3", tuple([*DISTRACTORS, "P3"])),
        QueryResult("q6", "P3", tuple(["P3", *DISTRACTORS])),
    ]


# --- hits_at_k -----------------------------------------------------------------


def test_hits_at_k_basic():
    hits = hits_at_k("P1", ["P1", "D1", "D2"], ks=(1, 2, 5))
    assert hits == {1: 1, 2: 1, 5: 1}


def test_hits_at_k_miss_outside_k():
    hits = hits_at_k("P2", ["D1", "D2", "D3", "D4", "D5", "P2"], ks=(1, 5))
    assert hits == {1: 0, 5: 0}


def test_hits_at_k_rejects_duplicate_ranked_products():
    with pytest.raises(ValueError, match="duplicate"):
        hits_at_k("P1", ["P1", "D1", "P1"], ks=(1,))


def test_hits_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        hits_at_k("P1", ["P1"], ks=(0,))
    with pytest.raises(ValueError):
        hits_at_k("P1", ["P1"], ks=(-1,))


def test_hits_at_k_rejects_empty_ks():
    with pytest.raises(ValueError):
        hits_at_k("P1", ["P1"], ks=())


# --- QueryResult construction validation ----------------------------------------


def test_query_result_rejects_duplicate_ranked_products():
    with pytest.raises(ValueError, match="duplicate"):
        QueryResult("q1", "P1", ("P1", "D1", "P1"))


def test_query_result_accepts_deduplicated_list():
    qr = QueryResult("q1", "P1", ("P1", "D1"))
    assert qr.ranked_product_ids == ("P1", "D1")


# --- ProductHits: a product with zero queries can't occur ----------------------


def test_product_hits_rejects_zero_queries():
    with pytest.raises(ValueError, match="zero queries"):
        ProductHits(product_id="P1", hit_sums={1: 0}, n_queries=0)


def test_product_hits_accepts_positive_queries():
    ph = ProductHits(product_id="P1", hit_sums={1: 1}, n_queries=1)
    assert ph.n_queries == 1


# --- recall_at_k: hand-computed macro vs micro ----------------------------------


def test_recall_at_k_hand_computed_macro_and_micro():
    results = _hand_example()
    out = recall_at_k(results, ks=(1, 5))

    assert out["n_products"] == 3
    assert out["n_queries"] == 6

    assert out["macro"][1] == pytest.approx(0.6111111111111112)
    assert out["macro"][5] == pytest.approx(0.7222222222222222)
    assert out["micro"][1] == pytest.approx(0.5)
    assert out["micro"][5] == pytest.approx(0.6666666666666666)

    # The whole point of product-macro vs micro: they must disagree here.
    assert out["macro"][1] != pytest.approx(out["micro"][1])
    assert out["macro"][5] != pytest.approx(out["micro"][5])


def test_recall_at_k_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        recall_at_k([], ks=(1,))


def test_recall_at_k_never_drops_a_query():
    # Every query must contribute to n_queries and to its truth product's group,
    # even products with a single query (P1 above has exactly one).
    results = _hand_example()
    out = recall_at_k(results, ks=(1,))
    assert out["n_queries"] == len(results)


def test_recall_at_k_default_ks_are_contract_defaults():
    results = _hand_example()
    out = recall_at_k(results)
    assert set(out["macro"]) == {1, 5, 10, 100}
    assert set(out["micro"]) == {1, 5, 10, 100}


def test_recall_at_k_no_nan_in_output():
    results = _hand_example()
    out = recall_at_k(results, ks=(1, 5, 10, 100))
    for mapping in (out["macro"], out["micro"]):
        for v in mapping.values():
            assert v == v  # NaN != NaN
