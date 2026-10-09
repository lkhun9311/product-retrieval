"""H2 comparison of the uncertainty path with the stratified curve, contract c4-uncertainty v1 section 4.

For every seed and measurement point the saved models of both policies are loaded and their validation
rankings are regenerated (the c5 v2 rerank path). Before anything is compared each regenerated macro R@5 must
equal the value recorded in the run's ``runs.jsonl`` within ``R5_TOLERANCE``; the curve-v1 endpoint models
and values are also pinned by the contract (``PINNED_ENDPOINT``). A mismatch aborts (``CompareAbort``).
The shared start is checked too: the uncertainty model at 100 labels must have the same weights as the
stratified one of the same seed.

``h2_comparison`` records: delta = uncertainty - stratified for R@1 and R@5 with the paired product-level
bootstrap (same resampled indices for both methods, B = 1000, seed 0), 95% percentile interval, P(delta <= 0).
Only the endpoint (3,000 labels) records feed the verdict sentence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes, sha256_file
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.gate import MANIFEST_SHA256
from product_retrieval.feedback import simulate as sim
from product_retrieval.pipelines import curve, uncertainty_path
from product_retrieval.pipelines.cand_vectors import cand_stats_index_id
from product_retrieval.pipelines.curve import CurveError
from product_retrieval.pipelines.uncertainty_path import HYPOTHESIS_ID, check_population
from product_retrieval.rerank import v1 as rerank_v1
from product_retrieval.rerank import v2 as rerank_v2

ENDPOINT = 3000
R5_TOLERANCE = 1e-9
KS = (1, 5)
BOOTSTRAP_B = curve.BOOTSTRAP_B
BOOTSTRAP_SEED = curve.BOOTSTRAP_SEED
# contract section 4 [U4]: seed -> (curve-v1 stratified model id at B = 3,000, recorded macro R@5)
PINNED_ENDPOINT: dict[int, tuple[str, float]] = {
    0: ("cadfa7929f8e", 0.8687032876672101),
    1: ("b43a45cb35c5", 0.8683824501123298),
    2: ("6bdb7fec4b7b", 0.8622586376517922),
}
VERDICT_RAISED = "uncertainty selection raised R@5 over stratified at 3,000 labels on these seeds"
VERDICT_LOWERED = "uncertainty selection lowered R@5 below stratified at 3,000 labels on these seeds"
VERDICT_NONE = "no consistent difference between the two policies at 3,000 labels"
VERDICT_INCOMPLETE = "undetermined: incomplete"
# (label, getter on a plan's "inputs") compared between the curve run and the uncertainty run
SHARED_INPUTS: tuple[tuple[str, Callable[[dict], Any]], ...] = (
    ("train rankings sha256", lambda i: i.get("rankings_sha256", {}).get("train")),
    ("val rankings sha256", lambda i: i.get("rankings_sha256", {}).get("val")),
    ("train cand-stats sha256", lambda i: i.get("cand_stats_sha256", {}).get("train")),
    ("val cand-stats sha256", lambda i: i.get("cand_stats_sha256", {}).get("val")),
    ("train index id", lambda i: i.get("index_id", {}).get("train")),
    ("val index id", lambda i: i.get("index_id", {}).get("val")),
    ("embed model id", lambda i: i.get("embed_model_id")),
    ("embedder", lambda i: i.get("embedder")),
    ("config name", lambda i: i.get("config_name")),
)
VectorLoader = Callable[[str, list[str], str], Any]


class CompareAbort(CurveError):
    """A pinned or recorded value was not reproduced, or an input is not the frozen one. No comparison."""


# ---- verdict (pure) -----------------------------------------------------------------------------


def _complete(rec: dict | None) -> bool:
    """A completed comparison with finite R@5 delta, interval and probability."""
    if not rec or rec.get("status") != "completed":
        return False
    cell = rec.get("delta", {}).get("5")
    if not cell:
        return False
    vals = [cell.get("delta"), *cell.get("ci", [None, None]), cell.get("p_le_0")]
    return len(vals) == 4 and all(
        isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in vals
    )


def h2_verdict(comparisons: Sequence[dict], seeds: Sequence[int], budget: int = ENDPOINT) -> dict:
    """The fixed verdict sentence of contract section 4 from the endpoint ``h2_comparison`` records."""
    by_seed = {c["seed"]: c for c in comparisons if c.get("budget") == budget}
    missing = [s for s in seeds if not _complete(by_seed.get(s))]
    if missing:
        text = f"{VERDICT_INCOMPLETE} (missing seeds: {', '.join(str(s) for s in missing)})"
        return {"verdict": "undetermined", "sentence": text, "missing_seeds": missing}
    lows = [by_seed[s]["delta"]["5"]["ci"][0] for s in seeds]
    highs = [by_seed[s]["delta"]["5"]["ci"][1] for s in seeds]
    if all(lo > 0 for lo in lows):
        return {"verdict": "raised", "sentence": VERDICT_RAISED, "missing_seeds": []}
    if all(hi < 0 for hi in highs):
        return {"verdict": "lowered", "sentence": VERDICT_LOWERED, "missing_seeds": []}
    return {"verdict": "none", "sentence": VERDICT_NONE, "missing_seeds": []}


# ---- inputs -------------------------------------------------------------------------------------


def _final_record(runs: list[dict], key_id: str) -> dict | None:
    """The completed attempt of a key, else None."""
    done = [r for r in runs if r["key_id"] == key_id and r["status"] == "completed"]
    return done[0] if done else None


def _read_plan(run_dir: Path) -> dict:
    path = Path(run_dir) / "plan.json"
    if not path.is_file():
        raise CompareAbort(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _rankings_bytes(rows: list[dict]) -> bytes:
    """The bytes ``pr rerank`` writes for these rows."""
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


def _same_weights(a: dict, b: dict) -> bool:
    sa, sb = a["state_dict"], b["state_dict"]
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def _first_reach(records: dict[int, dict | None], points: Sequence[int]) -> dict:
    pts = [
        {"budget": b, "status": "completed", "r5": records[b]["recall"]["5"]}
        if records.get(b)
        else {"budget": b, "status": "not_run"}
        for b in points
    ]
    r = curve.first_observed_point(pts)
    return {k: r[k] for k in ("verdict", "budget", "label")}


def compare_h2(
    uncertainty_dir: Path,
    curve_dir: Path,
    config: ExperimentConfig,
    val_rankings: Path,
    val_cand_stats: Path,
    out_dir: Path | None = None,
    embedder: str = "siglip",
    artifacts_root: Path = Path("artifacts"),
    *,
    vector_loader: VectorLoader | None = None,
    points: Sequence[int] = curve.BUDGETS,
    seeds: Sequence[int] = curve.SEEDS,
    pins: dict[int, tuple[str, float]] | None = None,
    manifest_sha: str = MANIFEST_SHA256,
    bootstrap_b: int = BOOTSTRAP_B,
    endpoint: int = ENDPOINT,
) -> dict:
    """Regenerate rankings, check them against the records, compute ``h2_comparison`` and the verdict.

    Writes ``h2_comparison.json`` and ``h2_comparison.md`` into ``out_dir`` (default: the uncertainty run
    dir) and returns the content. ``pins``, ``manifest_sha`` and ``endpoint`` default to the contract
    values; tests pass the values of their fixtures.
    """
    uncertainty_dir, curve_dir = Path(uncertainty_dir), Path(curve_dir)
    out_dir = uncertainty_dir if out_dir is None else Path(out_dir)
    pins = PINNED_ENDPOINT if pins is None else pins
    seeds, points = tuple(seeds), tuple(points)
    u_plan, c_plan = _read_plan(uncertainty_dir), _read_plan(curve_dir)
    if u_plan.get("hypothesis_id") != HYPOTHESIS_ID:
        raise CompareAbort(f"{uncertainty_dir} is not an {HYPOTHESIS_ID} run")
    if c_plan.get("hypothesis_id") != curve.HYPOTHESIS_ID:
        raise CompareAbort(f"{curve_dir} is not an {curve.HYPOTHESIS_ID} run")

    # both runs must have trained and evaluated on the same frozen inputs (hashes, indexes, embedder, config)
    c_in, u_in = c_plan["inputs"], u_plan["inputs"]
    for label, got in SHARED_INPUTS:
        c_val, u_val = got(c_in), got(u_in)
        if c_val is None or u_val is None or c_val != u_val:
            raise CompareAbort(
                f"the two runs differ in {label}: curve {c_val!r} vs uncertainty {u_val!r}; "
                "they must use the same frozen inputs"
            )

    val_rankings, val_cand_stats = Path(val_rankings), Path(val_cand_stats)
    val_sha, cs_sha = sha256_file(val_rankings), sha256_file(val_cand_stats)
    for name, plan in (("curve", c_plan), ("uncertainty", u_plan)):
        inp = plan["inputs"]
        if inp["rankings_sha256"]["val"] != val_sha or inp["cand_stats_sha256"]["val"] != cs_sha:
            raise CompareAbort(f"the validation rankings / cand-stats files differ from the {name} plan.json")
    if u_plan["inputs"]["index_id"]["val"] != c_plan["inputs"]["index_id"]["val"] or (
        u_plan["inputs"]["embed_model_id"] != c_plan["inputs"]["embed_model_id"]
    ):
        raise CompareAbort("the two runs used different validation indexes or embedding models")
    if u_plan.get("manifest_sha256") != manifest_sha:
        raise CompareAbort("the uncertainty run was not made on the frozen population manifest")

    val_rows = sim.load_rankings(val_rankings)
    val_cs = rerank_v1.load_cand_stats(val_cand_stats)
    rerank_v1.check_alignment(val_rows, val_sha, val_cs)
    truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
    baseline_results = curve.to_results(val_rows, truth)
    check_population("validation rankings", baseline_results, manifest_sha)

    c_runs = curve.read_runs(curve_dir / "runs.jsonl")
    u_runs = curve.read_runs(uncertainty_dir / "runs.jsonl")
    rounds = uncertainty_path.read_rounds(uncertainty_dir / "rounds.jsonl")

    # pinned curve-v1 endpoint: model id and recorded value, before anything is regenerated
    for seed, (pin_id, pin_r5) in pins.items():
        rec = _final_record(c_runs, f"stratified|p0|s{seed}|B{endpoint}")
        if rec is None:
            raise CompareAbort(f"curve record stratified|p0|s{seed}|B{endpoint} is missing or not completed")
        if rec["reranker_version"] != pin_id:
            raise CompareAbort(
                f"seed {seed}: curve model {rec['reranker_version']} differs from the pinned {pin_id}"
            )
        if not abs(rec["recall"]["5"] - pin_r5) <= R5_TOLERANCE:
            raise CompareAbort(
                f"seed {seed}: recorded R@5 {rec['recall']['5']!r} differs from the pinned {pin_r5!r}"
            )

    loader = vector_loader or curve._default_loader(config, embedder, artifacts_root)
    val_vectors = loader("val", [r["query_id"] for r in val_rows], cand_stats_index_id(val_cs))
    val_blind = [curve._blind(r) for r in val_rows]

    def regenerate(run_dir: Path, rec: dict) -> tuple[list, str, float, dict]:
        model = rerank_v2.load_model(run_dir / "models" / rec["reranker_version"] / "model.json")
        if model["reranker_version"] != rec["reranker_version"]:
            raise CompareAbort(f"{run_dir}: model.json of {rec['reranker_version']} carries another version")
        rows = rerank_v2.rerank_rows(
            model, val_blind, val_cs, val_sha, val_vectors.pair, val_vectors.embed_model_id
        )
        results = curve.to_results(rows, truth)
        check_population(f"regenerated rankings of {rec['key_id']}", results, manifest_sha)
        r5 = curve.macro_metrics(results)["5"]
        if not abs(r5 - rec["recall"]["5"]) <= R5_TOLERANCE:
            raise CompareAbort(
                f"{rec['key_id']}: regenerated macro R@5 {r5!r} differs from the recorded "
                f"{rec['recall']['5']!r} by more than {R5_TOLERANCE}"
            )
        return results, sha256_bytes(_rankings_bytes(rows)), r5, model

    comparisons: list[dict] = []
    for seed in seeds:
        for budget in points:
            s_key = f"stratified|p0|s{seed}|B{budget}"
            u_key = f"{uncertainty_path.POLICY}|p0|s{seed}|B{budget}"
            s_rec, u_rec = _final_record(c_runs, s_key), _final_record(u_runs, u_key)
            base = {
                "hypothesis_id": HYPOTHESIS_ID,
                "seed": seed,
                "budget": budget,
                "stratified_key": s_key,
                "uncertainty_key": u_key,
                "population_manifest_sha256": manifest_sha,
                "bootstrap": {"b": bootstrap_b, "seed": BOOTSTRAP_SEED},
            }
            if s_rec is None or u_rec is None:
                missing = [n for n, r in (("stratified", s_rec), ("uncertainty", u_rec)) if r is None]
                comparisons.append({**base, "status": "missing", "reason": f"no completed record: {missing}"})
                continue
            s_res, s_sha, _, s_model = regenerate(curve_dir, s_rec)
            u_res, u_sha, _, u_model = regenerate(uncertainty_dir, u_rec)
            if budget == uncertainty_path.INITIAL_BUDGET and not _same_weights(s_model, u_model):
                raise CompareAbort(
                    f"seed {seed}: the uncertainty model at {budget} labels does not have the weights of the "
                    "stratified one (the paths must share I_s)"
                )
            paired = paired_bootstrap(s_res, u_res, KS, b=bootstrap_b, seed=BOOTSTRAP_SEED)
            delta = {
                str(k): {
                    "delta": paired["macro"][k]["delta"],
                    "ci": list(paired["macro"][k]["ci"]),
                    "p_le_0": paired["macro"][k]["p_le_0"],
                }
                for k in KS
            }
            comparisons.append(
                {
                    **base,
                    "status": "completed",
                    "uncertainty_model_id": u_rec["reranker_version"],
                    "stratified_model_id": s_rec["reranker_version"],
                    "uncertainty_r5": u_rec["recall"]["5"],
                    "stratified_r5": s_rec["recall"]["5"],
                    "uncertainty_r1": u_rec["recall"]["1"],
                    "stratified_r1": s_rec["recall"]["1"],
                    "delta": delta,
                    "n_queries": paired["n_queries"],
                    "n_products": paired["n_products"],
                    "regenerated_rankings_sha256": {"uncertainty": u_sha, "stratified": s_sha},
                }
            )

    verdict = h2_verdict(comparisons, seeds, endpoint)
    secondary: dict[str, Any] = {"first_reach_0.887": {}, "positive_share_and_time": {}}
    for seed in seeds:
        u_by = {b: _final_record(u_runs, f"{uncertainty_path.POLICY}|p0|s{seed}|B{b}") for b in points}
        s_by = {b: _final_record(c_runs, f"stratified|p0|s{seed}|B{b}") for b in points}
        secondary["first_reach_0.887"][str(seed)] = {
            "uncertainty": _first_reach(u_by, points),
            "stratified": _first_reach(s_by, points),
        }
        secondary["positive_share_and_time"][str(seed)] = [
            {
                "budget": b,
                "label_pos_share": (r["n_pos"] / (r["n_pos"] + r["n_neg"])) if r else None,
                "point_seconds": r["seconds"] if r else None,
                **{
                    k: x[k]
                    for x in rounds
                    if x["seed"] == seed and x["from_size"] == b
                    for k in ("batch_pos_share", "selection_seconds")
                },
            }
            for b, r in u_by.items()
        ]
    result = {
        "hypothesis_id": HYPOTHESIS_ID,
        "contract": uncertainty_path.CONTRACT_NAME,
        "verdict": verdict,
        "primary": {"budget": endpoint, "metric": "macro R@5", "delta": "uncertainty - stratified"},
        "population_manifest_sha256": manifest_sha,
        "bootstrap": {"b": bootstrap_b, "seed": BOOTSTRAP_SEED},
        "r5_tolerance": R5_TOLERANCE,
        "pins_checked": {str(s): {"model": p[0], "r5": p[1]} for s, p in pins.items()},
        "uncertainty_run": str(uncertainty_dir),
        "curve_run": str(curve_dir),
        "h2_comparison": comparisons,
        "secondary": secondary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h2_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / "h2_comparison.md").write_text(markdown(result), encoding="utf-8")
    return result


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.4f}"


def markdown(result: dict) -> str:
    lines = [
        f"# {result['hypothesis_id']}: uncertainty vs stratified (noise 0, development validation split)",
        "",
        f"**Verdict:** {result['verdict']['sentence']}",
        "",
        "Delta = uncertainty - stratified, macro recall, paired product-level bootstrap "
        f"(B={result['bootstrap']['b']}, seed {result['bootstrap']['seed']}), 95% interval, "
        "P = share of replicates with delta <= 0. Only the 3,000-label rows feed the verdict.",
        "",
        "| seed | labels | stratified R@5 | uncertainty R@5 | dR@5 [95% CI] | P(d<=0) | dR@1 [95% CI] |",
        "|---:|---:|---:|---:|---|---:|---|",
    ]
    for c in result["h2_comparison"]:
        if c["status"] != "completed":
            lines.append(f"| {c['seed']} | {c['budget']} | - | - | {c['status']}: {c['reason']} | - | - |")
            continue
        d5, d1 = c["delta"]["5"], c["delta"]["1"]
        lines.append(
            f"| {c['seed']} | {c['budget']} | {c['stratified_r5']:.4f} | {c['uncertainty_r5']:.4f} | "
            f"{_fmt(d5['delta'])} [{_fmt(d5['ci'][0])}, {_fmt(d5['ci'][1])}] | {d5['p_le_0']:.3f} | "
            f"{_fmt(d1['delta'])} [{_fmt(d1['ci'][0])}, {_fmt(d1['ci'][1])}] |"
        )
    lines += ["", "First point with R@5 >= 0.887 (secondary, not used for the verdict):", ""]
    lines += ["| seed | uncertainty | stratified |", "|---:|---|---|"]
    for seed, v in result["secondary"]["first_reach_0.887"].items():
        lines.append(f"| {seed} | {v['uncertainty']['label']} | {v['stratified']['label']} |")
    return "\n".join(lines) + "\n"
