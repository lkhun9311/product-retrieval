"""Tests for product_retrieval.eval.bootstrap (D20 section 9 eval contract v1).

Covers: an independent (loop-based) reference implementation of the per-replicate
macro/micro math checked against the vectorized one on the same indices; paired
bootstrap sanity cases (identical methods -> zero delta, one strictly better ->
positive delta); determinism under a fixed seed; and failure injection for the
validation rules the contract requires (empty input, mismatched query sets).
"""

from __future__ import annotations

import numpy as np
import pytest

from product_retrieval.eval.bootstrap import (
    _bootstrap_indices,
    _replicate_macro_micro,
    bootstrap_recall,
    paired_bootstrap,
)
from product_retrieval.eval.retrieval import QueryResult

DISTRACTORS = ["D1", "D2", "D3", "D4", "D5"]


def _hand_example() -> list[QueryResult]:
    # Same dataset as tests/test_eval_retrieval.py's hand-computed example.
    return [
        QueryResult("q1", "P1", tuple(["P1", *DISTRACTORS])),
        QueryResult("q2", "P2", tuple([*DISTRACTORS, "P2"])),
        QueryResult("q3", "P2", tuple(["P2", *DISTRACTORS])),
        QueryResult("q4", "P3", tuple(["D1", "P3", "D2", "D3", "D4", "D5"])),
        QueryResult("q5", "P3", tuple([*DISTRACTORS, "P3"])),
        QueryResult("q6", "P3", tuple(["P3", *DISTRACTORS])),
    ]


# --- independent loop-based reference for the per-replicate math ---------------


def _reference_replicate_macro_micro(
    hit_sums: np.ndarray, query_counts: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slow, loop-based reimplementation of ``_replicate_macro_micro``.

    Deliberately avoids fancy indexing / broadcasting so it cannot share a bug with
    the vectorized implementation under test.
    """
    b, n_products = idx.shape
    n_ks = hit_sums.shape[1]
    macro = np.zeros((b, n_ks))
    micro = np.zeros((b, n_ks))
    for rep in range(b):
        row = idx[rep]
        for k_i in range(n_ks):
            per_product_vals = []
            total_hits = 0.0
            total_queries = 0.0
            for prod_idx in row:
                total_hits += hit_sums[prod_idx, k_i]
                total_queries += query_counts[prod_idx]
                per_product_vals.append(hit_sums[prod_idx, k_i] / query_counts[prod_idx])
            macro[rep, k_i] = sum(per_product_vals) / len(per_product_vals)
            micro[rep, k_i] = total_hits / total_queries
    return macro, micro


def test_replicate_macro_micro_matches_independent_loop_reference():
    # Hand-picked, not derived from any QueryResult grouping: 4 products, 2 Ks.
    hit_sums = np.array(
        [
            [3.0, 5.0],
            [0.0, 2.0],
            [4.0, 4.0],
            [1.0, 1.0],
        ]
    )
    query_counts = np.array([4.0, 2.0, 4.0, 3.0])
    idx = _bootstrap_indices(n_products=4, b=50, seed=123)

    macro_fast, micro_fast = _replicate_macro_micro(hit_sums, query_counts, idx)
    macro_ref, micro_ref = _reference_replicate_macro_micro(hit_sums, query_counts, idx)

    np.testing.assert_allclose(macro_fast, macro_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(micro_fast, micro_ref, rtol=0, atol=1e-12)


# --- bootstrap_recall ------------------------------------------------------------


def test_bootstrap_recall_point_matches_recall_at_k():
    from product_retrieval.eval.retrieval import recall_at_k

    results = _hand_example()
    point = recall_at_k(results, ks=(1, 5))
    boot = bootstrap_recall(results, ks=(1, 5), b=200, seed=0)
    assert boot["point"]["macro"] == point["macro"]
    assert boot["point"]["micro"] == point["micro"]
    assert boot["n_products"] == 3
    assert boot["n_queries"] == 6


def test_bootstrap_recall_ci_bounds_are_ordered_and_no_nan():
    results = _hand_example()
    boot = bootstrap_recall(results, ks=(1, 5), b=200, seed=0)
    for kind in ("macro", "micro"):
        for _k, (lo, hi) in boot["ci"][kind].items():
            assert lo == lo and hi == hi  # no NaN
            assert lo <= hi
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0


def test_bootstrap_recall_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_recall([], ks=(1,))


def test_bootstrap_recall_determinism_same_seed():
    results = _hand_example()
    boot1 = bootstrap_recall(results, ks=(1, 5), b=300, seed=42)
    boot2 = bootstrap_recall(results, ks=(1, 5), b=300, seed=42)
    assert boot1["ci"] == boot2["ci"]


def test_bootstrap_recall_different_seed_very_likely_differs():
    results = _hand_example()
    boot1 = bootstrap_recall(results, ks=(1, 5), b=300, seed=1)
    boot2 = bootstrap_recall(results, ks=(1, 5), b=300, seed=2)
    assert boot1["ci"] != boot2["ci"]


# --- paired_bootstrap -------------------------------------------------------------


def test_paired_bootstrap_identical_methods_gives_zero_delta():
    results = _hand_example()
    out = paired_bootstrap(results, results, ks=(1, 5), b=200, seed=0)
    for kind in ("macro", "micro"):
        for k in (1, 5):
            entry = out[kind][k]
            assert entry["delta"] == pytest.approx(0.0)
            assert entry["ci"] == pytest.approx((0.0, 0.0))


def test_paired_bootstrap_b_strictly_better_gives_positive_interval():
    # Same queries/truth products; A always misses at K=1, B always hits.
    queries = [("q1", "P1"), ("q2", "P2"), ("q3", "P2"), ("q4", "P3")]
    results_a = [QueryResult(qid, pid, tuple(["D1", pid, "D2"])) for qid, pid in queries]
    results_b = [QueryResult(qid, pid, tuple([pid, "D1", "D2"])) for qid, pid in queries]
    out = paired_bootstrap(results_a, results_b, ks=(1,), b=300, seed=0)
    for kind in ("macro", "micro"):
        entry = out[kind][1]
        assert entry["delta"] > 0
        lo, hi = entry["ci"]
        assert lo > 0
        assert hi > 0
        assert entry["p_le_0"] == 0.0


def test_paired_bootstrap_determinism_same_seed():
    results = _hand_example()
    out1 = paired_bootstrap(results, results, ks=(1, 5), b=200, seed=7)
    out2 = paired_bootstrap(results, results, ks=(1, 5), b=200, seed=7)
    assert out1 == out2


# --- failure injection -------------------------------------------------------------


def test_paired_bootstrap_rejects_empty_results():
    results = _hand_example()
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap([], results, ks=(1,))
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap(results, [], ks=(1,))


def test_paired_bootstrap_rejects_query_missing_from_b():
    results_a = _hand_example()
    results_b = results_a[:-1]  # drop the last query -> missing from B
    with pytest.raises(ValueError, match="missing from B"):
        paired_bootstrap(results_a, results_b, ks=(1,))


def test_paired_bootstrap_rejects_query_missing_from_a():
    results_b = _hand_example()
    results_a = results_b[:-1]  # drop the last query -> missing from A
    with pytest.raises(ValueError, match="missing from A"):
        paired_bootstrap(results_a, results_b, ks=(1,))


def test_paired_bootstrap_rejects_duplicate_query_id():
    results = _hand_example()
    dup = [*results, QueryResult("q1", "P1", ("P1", "D1"))]
    with pytest.raises(ValueError, match="duplicate query_id"):
        paired_bootstrap(dup, dup, ks=(1,))


def test_paired_bootstrap_rejects_mismatched_truth_product():
    results_a = _hand_example()
    results_b = list(results_a)
    # Flip q1's truth product in B only -> same query_id, inconsistent truth.
    results_b[0] = QueryResult("q1", "P2", results_b[0].ranked_product_ids)
    with pytest.raises(ValueError, match="truth_product_id differs"):
        paired_bootstrap(results_a, results_b, ks=(1,))
