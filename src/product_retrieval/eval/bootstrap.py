"""Product-level bootstrap confidence intervals for R@K (D20 section 9, eval contract v1).

- Single-method CI: resample products with replacement (each sampled product brings all
  of its queries along), recompute product-macro R@K per replicate; B=1000, fixed seed,
  95% percentile interval. Micro R@K is reported under the same resampling.
- Paired A/B comparison: the SAME resampled product indices are applied to both methods
  so the comparison is on matched (query, truth product) pairs; Delta = R@K(B) - R@K(A)
  per replicate, with a point estimate, a 95% percentile interval, and the fraction of
  replicates with Delta <= 0.

Vectorized with numpy: per-product hit sums and query counts are computed once, the
(B, n_products) index matrix is drawn once per bootstrap call, and reused across all K.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from product_retrieval.eval.retrieval import (
    DEFAULT_KS,
    QueryResult,
    _group_by_product,
    _validate_ks,
    recall_at_k,
)


def _product_matrices(
    results: Sequence[QueryResult], ks: Sequence[int]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (product_ids sorted, hit_sums[n_products, n_ks], query_counts[n_products])."""
    groups = _group_by_product(results, ks)
    product_ids = sorted(groups)
    hit_sums = np.array([[groups[pid].hit_sums[k] for k in ks] for pid in product_ids], dtype=np.float64)
    query_counts = np.array([groups[pid].n_queries for pid in product_ids], dtype=np.float64)
    return product_ids, hit_sums, query_counts


def _bootstrap_indices(n_products: int, b: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_products, size=(b, n_products))


def _replicate_macro_micro(
    hit_sums: np.ndarray, query_counts: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized macro/micro R@K over all bootstrap replicates at once.

    Per-product R@K is fixed (computed once, outside the replicate loop); each
    replicate only changes *which* products (with repetition) are included, via
    fancy-indexing with the shared ``idx`` matrix of shape (B, n_products).
    """
    per_product_rk = hit_sums / query_counts[:, None]  # (n_products, n_ks)
    sampled_rk = per_product_rk[idx]  # (B, n_products, n_ks)
    macro = sampled_rk.mean(axis=1)  # (B, n_ks)

    sampled_hit_sums = hit_sums[idx]  # (B, n_products, n_ks)
    sampled_counts = query_counts[idx]  # (B, n_products)
    micro = sampled_hit_sums.sum(axis=1) / sampled_counts.sum(axis=1)[:, None]  # (B, n_ks)
    return macro, micro


def bootstrap_recall(
    results: Sequence[QueryResult],
    ks: Sequence[int] = DEFAULT_KS,
    b: int = 1000,
    seed: int = 0,
) -> dict:
    """Product-level bootstrap 95% percentile CI for macro and micro R@K.

    Returns a dict with the point estimate (``recall_at_k`` on the full data), the
    percentile CI per K for macro and micro, and the replicate count/seed used.
    """
    if not results:
        raise ValueError("bootstrap_recall: results must not be empty")
    ks_t = _validate_ks(ks)
    point = recall_at_k(results, ks_t)
    _, hit_sums, query_counts = _product_matrices(results, ks_t)
    n_products = hit_sums.shape[0]
    idx = _bootstrap_indices(n_products, b, seed)
    macro_boot, micro_boot = _replicate_macro_micro(hit_sums, query_counts, idx)

    lo_macro, hi_macro = np.percentile(macro_boot, [2.5, 97.5], axis=0)
    lo_micro, hi_micro = np.percentile(micro_boot, [2.5, 97.5], axis=0)

    return {
        "point": {"macro": point["macro"], "micro": point["micro"]},
        "ci": {
            "macro": {k: (float(lo_macro[i]), float(hi_macro[i])) for i, k in enumerate(ks_t)},
            "micro": {k: (float(lo_micro[i]), float(hi_micro[i])) for i, k in enumerate(ks_t)},
        },
        "b": b,
        "seed": seed,
        "n_products": point["n_products"],
        "n_queries": point["n_queries"],
    }


def _validate_paired(results_a: Sequence[QueryResult], results_b: Sequence[QueryResult]) -> None:
    if not results_a or not results_b:
        raise ValueError("paired_bootstrap: results must not be empty")
    ids_a = [r.query_id for r in results_a]
    ids_b = [r.query_id for r in results_b]
    if len(set(ids_a)) != len(ids_a):
        raise ValueError("paired_bootstrap: results_a has a duplicate query_id")
    if len(set(ids_b)) != len(ids_b):
        raise ValueError("paired_bootstrap: results_b has a duplicate query_id")
    set_a, set_b = set(ids_a), set(ids_b)
    missing_from_b = set_a - set_b
    missing_from_a = set_b - set_a
    if missing_from_b or missing_from_a:
        raise ValueError(
            "paired_bootstrap: query sets differ between A and B "
            f"(missing from B: {sorted(missing_from_b)}, missing from A: {sorted(missing_from_a)})"
        )
    truth_a = {r.query_id: r.truth_product_id for r in results_a}
    truth_b = {r.query_id: r.truth_product_id for r in results_b}
    mismatched = [qid for qid in truth_a if truth_a[qid] != truth_b[qid]]
    if mismatched:
        raise ValueError(
            f"paired_bootstrap: truth_product_id differs between A and B for queries {mismatched}"
        )


def paired_bootstrap(
    results_a: Sequence[QueryResult],
    results_b: Sequence[QueryResult],
    ks: Sequence[int] = DEFAULT_KS,
    b: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired product-level bootstrap for Delta = R@K(B) - R@K(A) on the same queries.

    The same resampled product indices are applied to both A and B in every replicate.
    Raises ``ValueError`` if either side is empty, has duplicate query ids, the query
    sets differ, or a shared query's truth product disagrees between A and B.
    """
    _validate_paired(results_a, results_b)
    ks_t = _validate_ks(ks)

    point_a = recall_at_k(results_a, ks_t)
    point_b = recall_at_k(results_b, ks_t)

    ids_a, hit_sums_a, query_counts_a = _product_matrices(results_a, ks_t)
    ids_b, hit_sums_b, query_counts_b = _product_matrices(results_b, ks_t)
    # Guaranteed by _validate_paired: same queries with matching truth products imply
    # the same set of truth products, and _product_matrices sorts them identically.
    assert ids_a == ids_b

    n_products = hit_sums_a.shape[0]
    idx = _bootstrap_indices(n_products, b, seed)

    macro_a, micro_a = _replicate_macro_micro(hit_sums_a, query_counts_a, idx)
    macro_b, micro_b = _replicate_macro_micro(hit_sums_b, query_counts_b, idx)

    delta_macro = macro_b - macro_a
    delta_micro = micro_b - micro_a

    lo_dm, hi_dm = np.percentile(delta_macro, [2.5, 97.5], axis=0)
    lo_dmi, hi_dmi = np.percentile(delta_micro, [2.5, 97.5], axis=0)
    p_le_0_macro = (delta_macro <= 0).mean(axis=0)
    p_le_0_micro = (delta_micro <= 0).mean(axis=0)

    macro_out: dict[int, dict] = {}
    micro_out: dict[int, dict] = {}
    for i, k in enumerate(ks_t):
        macro_out[k] = {
            "delta": float(point_b["macro"][k] - point_a["macro"][k]),
            "ci": (float(lo_dm[i]), float(hi_dm[i])),
            "p_le_0": float(p_le_0_macro[i]),
        }
        micro_out[k] = {
            "delta": float(point_b["micro"][k] - point_a["micro"][k]),
            "ci": (float(lo_dmi[i]), float(hi_dmi[i])),
            "p_le_0": float(p_le_0_micro[i]),
        }

    return {
        "macro": macro_out,
        "micro": micro_out,
        "b": b,
        "seed": seed,
        "n_products": n_products,
        "n_queries": len(results_a),
    }
