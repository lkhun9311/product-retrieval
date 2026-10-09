"""Deployment gate for a retrained reranker, contract c6-deploy-gate-v1 (docs/contracts/c6-deploy-gate.md).

``evaluate_gate`` is a pure function: it takes the rankings of the current and the candidate bundle on
the frozen gate evaluation set and the candidate's ``model.json`` content, and returns the decision.

Three checks are always evaluated; the gate passes only if none fails:

- G1: the 95% lower bound of the paired product-level bootstrap delta of macro R@5 (candidate - current)
  must not be below ``G1_MARGIN``.
- G2: the same for macro R@1 against ``G2_MARGIN``.
- G3: the positive-label share ``n_pos / (n_pos + n_neg)`` of the candidate's training labels must not
  exceed ``G3_MAX_SHARE`` (= 2 * ``P_REF``).

The thresholds, the bootstrap settings and the label policy are constants of this contract version.
No function argument changes them. Anything that makes a decision untrustworthy raises ``GateError``;
an error is neither a pass nor a block.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from product_retrieval.core.ids import sha256_bytes
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.retrieval import QueryResult

CONTRACT_ID = "c6-deploy-gate-v1"
# sha256 of the sorted "query_id<TAB>truth_product_id<LF>" lines of the frozen development validation split
MANIFEST_SHA256 = "92665a22adcfc069901ecb91ed684606d203124e9fc2dcca25b92b71318b11a9"

G1_MARGIN = -0.005  # lower bound of delta macro R@5
G2_MARGIN = -0.03  # lower bound of delta macro R@1
P_REF = 0.05
G3_MAX_SHARE = 0.10  # 2 * P_REF
REQUIRED_LABEL_POLICY = "stratified"
KS = (1, 5)
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 0
MIN_UNIQUE_IDS = 5

PASS = "pass"
BLOCK = "block"


class GateError(ValueError):
    """An input the gate refuses to decide on. ``kind`` names the §4 failure mode."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def manifest_lines(results: Sequence[QueryResult]) -> list[str]:
    """Sorted ``query_id<TAB>truth_product_id<LF>`` lines of a result set."""
    return sorted(f"{r.query_id}\t{r.truth_product_id}\n" for r in results)


def manifest_sha256(results: Sequence[QueryResult]) -> str:
    """sha256 (UTF-8) over the concatenated sorted manifest lines of a result set."""
    return sha256_bytes("".join(manifest_lines(results)).encode("utf-8"))


def _check_results(name: str, results: Sequence[QueryResult], expected_sha: str) -> None:
    if not results:
        raise GateError("manifest_mismatch", f"{name}: no rows")
    seen: set[str] = set()
    for r in results:
        if r.query_id in seen:
            raise GateError("duplicate_query_id", f"{name}: duplicate query_id {r.query_id!r}")
        seen.add(r.query_id)
    got = manifest_sha256(results)
    if got != expected_sha:
        raise GateError(
            "manifest_mismatch",
            f"{name}: manifest sha256 {got} differs from the frozen {expected_sha}",
        )
    for r in results:
        n_unique = len(set(r.ranked_product_ids))
        if n_unique < MIN_UNIQUE_IDS:
            raise GateError(
                "too_few_candidates",
                f"{name}: query {r.query_id!r} has {n_unique} unique product ids, "
                f"need at least {MIN_UNIQUE_IDS}",
            )


def _count(model: Mapping[str, Any], field: str) -> int:
    if field not in model:
        raise GateError("invalid_label_counts", f"candidate model: {field} is missing")
    value = model[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateError("invalid_label_counts", f"candidate model: {field}={value!r} is not an integer")
    if value < 0:
        raise GateError("invalid_label_counts", f"candidate model: {field}={value} is negative")
    return value


def _text(model: Mapping[str, Any], field: str) -> str:
    value = model.get(field)
    if not isinstance(value, str) or not value:
        raise GateError("invalid_model", f"candidate model: {field} is missing or not a non-empty string")
    return value


def _check_model(model: Any) -> dict[str, Any]:
    if not isinstance(model, Mapping):
        raise GateError("invalid_model", "candidate model: not a JSON object")
    n_pos, n_neg = _count(model, "n_pos"), _count(model, "n_neg")
    if n_pos + n_neg == 0:
        raise GateError("invalid_label_counts", "candidate model: n_pos + n_neg is 0")
    label_version = _text(model, "label_version")
    reranker_version = _text(model, "reranker_version")
    policy = model.get("label_policy")
    if policy != REQUIRED_LABEL_POLICY:
        raise GateError(
            "invalid_label_policy",
            f"candidate model: label_policy={policy!r}, this contract needs {REQUIRED_LABEL_POLICY!r}",
        )
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "label_version": label_version,
        "reranker_version": reranker_version,
        "label_policy": policy,
    }


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise GateError("non_finite", "bootstrap produced a non-finite value")


def evaluate_gate(
    current: Sequence[QueryResult],
    candidate: Sequence[QueryResult],
    candidate_model: Mapping[str, Any],
    manifest_sha: str = MANIFEST_SHA256,
) -> dict[str, Any]:
    """Decide pass or block for ``candidate`` against ``current``; raise ``GateError`` on a bad input.

    ``manifest_sha`` defaults to the frozen v1 evaluation set; tests pass the hash of a small fixture.
    It is the only argument that is not data, and it does not touch a threshold.
    Returns ``decision``, ``checks`` (G1..G3), ``reasons`` (failed ids in id order), ``contract`` and the
    file-independent part of ``inputs``.
    """
    meta = _check_model(candidate_model)
    _check_results("current", current, manifest_sha)
    _check_results("candidate", candidate, manifest_sha)

    paired = paired_bootstrap(current, candidate, KS, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    checks: dict[str, dict[str, Any]] = {}
    for check_id, k, margin in (("G1", 5, G1_MARGIN), ("G2", 1, G2_MARGIN)):
        cell = paired["macro"][k]
        delta, (ci_low, ci_high) = float(cell["delta"]), cell["ci"]
        _finite(delta, ci_low, ci_high)
        checks[check_id] = {
            "metric": f"macro R@{k}",
            "value": ci_low,
            "threshold": margin,
            "failed": bool(ci_low < margin),
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    share = meta["n_pos"] / (meta["n_pos"] + meta["n_neg"])
    checks["G3"] = {
        "metric": "n_pos / (n_pos + n_neg)",
        "value": share,
        "threshold": G3_MAX_SHARE,
        "failed": bool(share > G3_MAX_SHARE),
    }

    reasons = [check_id for check_id in ("G1", "G2", "G3") if checks[check_id]["failed"]]
    return {
        "contract": CONTRACT_ID,
        "decision": BLOCK if reasons else PASS,
        "checks": checks,
        "reasons": reasons,
        "inputs": {
            "manifest_sha256": manifest_sha,
            "n_queries": paired["n_queries"],
            "n_products": paired["n_products"],
            **meta,
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
        },
    }
