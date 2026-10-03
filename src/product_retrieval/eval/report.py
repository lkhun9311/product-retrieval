"""Build ``core.schemas.EvalReport``-compatible ``metrics``/``ci`` dicts (D20 section 3 and 9).

``EvalReport.metrics`` and ``EvalReport.ci`` are untyped ``dict[str, Any]`` in the fixed
schema, so this module owns the (stable, documented) shape eval writes into them. Gate
pass/fail (D20 section 2 row C5) is out of scope here; callers supply a ``Gate`` if they
want a full ``EvalReport``.
"""

from __future__ import annotations

from collections.abc import Sequence

from product_retrieval.core.schemas import EvalReport, Gate
from product_retrieval.eval.bootstrap import DEFAULT_KS, bootstrap_recall
from product_retrieval.eval.retrieval import QueryResult


def build_metrics_and_ci(
    results: Sequence[QueryResult],
    ks: Sequence[int] = DEFAULT_KS,
    b: int = 1000,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Return ``(metrics, ci)`` for ``EvalReport.metrics`` / ``EvalReport.ci``.

    ``metrics`` = ``{"recall": {"macro": {"R@K": v, ...}, "micro": {...}},
    "n_products": int, "n_queries": int}``.

    ``ci`` = ``{"recall": {"macro": {"R@K": [lo, hi], ...}, "micro": {...}},
    "b": int, "seed": int}``. Intervals are lists (not tuples) so the dict round-trips
    through JSON unchanged.
    """
    boot = bootstrap_recall(results, ks, b=b, seed=seed)
    ks_t = tuple(ks)

    metrics = {
        "recall": {
            "macro": {f"R@{k}": boot["point"]["macro"][k] for k in ks_t},
            "micro": {f"R@{k}": boot["point"]["micro"][k] for k in ks_t},
        },
        "n_products": boot["n_products"],
        "n_queries": boot["n_queries"],
    }
    ci = {
        "recall": {
            "macro": {f"R@{k}": list(boot["ci"]["macro"][k]) for k in ks_t},
            "micro": {f"R@{k}": list(boot["ci"]["micro"][k]) for k in ks_t},
        },
        "b": b,
        "seed": seed,
    }
    return metrics, ci


def build_eval_report(
    bundle_id: str,
    dataset_manifest_sha: str,
    results: Sequence[QueryResult],
    gate: Gate,
    ks: Sequence[int] = DEFAULT_KS,
    b: int = 1000,
    seed: int = 0,
) -> EvalReport:
    """Build a full ``EvalReport`` from retrieval results plus a caller-supplied ``Gate``.

    Gate thresholds (baseline comparison, risk-coverage) are a separate concern (D20
    roadmap C5); this function only fills in ``metrics``/``ci`` and passes ``gate``
    through unchanged.
    """
    metrics, ci = build_metrics_and_ci(results, ks, b=b, seed=seed)
    return EvalReport(
        bundle_id=bundle_id,
        dataset_manifest_sha=dataset_manifest_sha,
        metrics=metrics,
        ci=ci,
        gate=gate,
    )
