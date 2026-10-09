"""Uncertainty label-selection path runner, contract c4-uncertainty v1 (docs/contracts/c4-uncertainty.md).

One path per seed, noise 0. Start from I_s (the first 100 of the stratified order with their answers). At
every measurement point: train reranker v2 from scratch on all labels so far (training seed = s), evaluate
on the frozen validation population, score every pool pair with that model, select the next batch with the
pure selector (``feedback.uncertainty``), ask the simulator for the answers and append them. After the last
batch (3,000 labels) train and evaluate once more.

Written under ``<out_root>/<run_id>/``: ``plan.json``, ``runs.jsonl`` (one line per attempt, curve-v1 like,
with ``delta_vs_baseline``), ``rounds.jsonl`` (selection stats per round),
``events/uncertainty.seed<s>.jsonl``
(FeedbackEvent JSONL in selection order, policy_id ``uncertainty``) and ``models/<version>/``.

Failures follow c4-learning-curve: a failed key keeps its record, at most one retry with the same settings and
only for an OSError. A path whose point failed stops there (the later points are not run); a re-run with the
same run id resumes from the events file and the saved models.

``truth_product_id`` is read by the simulator oracle and by the evaluation step only. The scoring and the
selection receive the training rankings with the truth field removed.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_file
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.gate import MANIFEST_SHA256, manifest_sha256
from product_retrieval.eval.retrieval import QueryResult, recall_at_k
from product_retrieval.feedback import simulate as sim
from product_retrieval.feedback.labels import build_labels, load_events
from product_retrieval.feedback.uncertainty import POOL_DEPTH, Selection, select_uncertain, sigmoid
from product_retrieval.pipelines import curve
from product_retrieval.pipelines.cand_vectors import cand_stats_index_id
from product_retrieval.pipelines.curve import CurveError, Key
from product_retrieval.rerank import v1 as rerank_v1
from product_retrieval.rerank import v2 as rerank_v2

HYPOTHESIS_ID = "H2-uncertainty"
POLICY = "uncertainty"
PREREGISTRATION = "c4-v3"
CONTRACT_NAME = "c4-uncertainty v1"
POINTS = curve.BUDGETS
SEEDS = curve.SEEDS
INITIAL_BUDGET = 100
KS = curve.KS
BOOTSTRAP_B = curve.BOOTSTRAP_B
BOOTSTRAP_SEED = curve.BOOTSTRAP_SEED
SCORE_CHUNK_QUERIES = 2048
CONTRACT_FILES = (
    "c4-uncertainty.md",
    "c4-label-selection.md",
    "c4-learning-curve.md",
    "c4-event-labels.md",
    "c5-reranker-v2.md",
    "c6-deploy-gate.md",
)


class UncertaintyError(CurveError):
    """Invalid input or a corrupt run directory of the uncertainty path."""


def key_for(seed: int, budget: int) -> Key:
    return Key(POLICY, 0.0, seed, budget)


def plan_keys(seeds: Sequence[int] = SEEDS, points: Sequence[int] = POINTS) -> list[Key]:
    return [key_for(s, b) for s in seeds for b in points]


def check_points(points: Sequence[int]) -> tuple[int, ...]:
    pts = tuple(points)
    if len(pts) < 1 or pts[0] != INITIAL_BUDGET or any(b <= a for a, b in zip(pts, pts[1:], strict=False)):
        raise UncertaintyError(
            f"measurement points must be strictly increasing and start at {INITIAL_BUDGET}, got {pts}"
        )
    return pts


def check_population(name: str, results: Sequence[QueryResult], expected_sha: str) -> None:
    """Contract section 1 (U2): no duplicates, and the manifest sha256 equals the frozen one."""
    if not results:
        raise curve.PopulationMismatch(f"{name}: no rows")
    seen: set[str] = set()
    for r in results:
        if r.query_id in seen:
            raise curve.PopulationMismatch(f"{name}: duplicate query_id {r.query_id!r}")
        seen.add(r.query_id)
    got = manifest_sha256(results)
    if got != expected_sha:
        raise curve.PopulationMismatch(
            f"{name}: manifest sha256 {got} differs from the frozen {expected_sha}"
        )


# ---- scoring and selection (no truth, no oracle) --------------------------------------------------


def score_pool(
    model: dict,
    pool: Sequence[dict],
    cand_stats_by_query: dict[str, dict],
    rankings_sha256: str,
    vector_fn: rerank_v2.VectorFn,
    embed_model_id: str | None,
    chunk: int = SCORE_CHUNK_QUERIES,
) -> dict[str, np.ndarray]:
    """``{query_id: p by original position}`` for the top ``POOL_DEPTH`` of every pool row.

    Scores in chunks of ``chunk`` queries with ``rerank_v2.rerank_rows`` (the same function the evaluation
    uses), so memory stays bounded: only the (n_queries, 20) probabilities are kept.
    """
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(pool), chunk):
        rows = list(pool[start : start + chunk])
        stats = [cand_stats_by_query[r["query_id"]] for r in rows]
        reranked = rerank_v2.rerank_rows(model, rows, stats, rankings_sha256, vector_fn, embed_model_id)
        for orig, new in zip(rows, reranked, strict=True):
            ids = list(orig["top_k_product_ids"][:POOL_DEPTH])
            z_of = dict(zip(new["top_k_product_ids"][: len(ids)], new["rerank_scores"], strict=True))
            out[orig["query_id"]] = sigmoid(np.array([z_of[pid] for pid in ids]))
    return out


def select_next_batch(
    model: dict,
    pool: Sequence[dict],
    cand_stats_by_query: dict[str, dict],
    rankings_sha256: str,
    vector_fn: rerank_v2.VectorFn,
    embed_model_id: str | None,
    labelled: dict[tuple[str, str], Any],
    n_take: int,
    chunk: int = SCORE_CHUNK_QUERIES,
) -> list[Selection]:
    """Steps 3-5 of the progression rule. Takes no simulator, no manifest and reads no truth field."""
    probs = score_pool(model, pool, cand_stats_by_query, rankings_sha256, vector_fn, embed_model_id, chunk)
    return select_uncertain(pool, probs, labelled, n_take)


# ---- helpers ------------------------------------------------------------------------------------


def _events_path(out_dir: Path, seed: int) -> Path:
    return out_dir / "events" / f"{POLICY}.seed{seed}.jsonl"


def _write_events_atomic(events: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for ev in events:
            f.write(ev.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _contract_hashes(root: Path = curve._REPO_ROOT) -> dict[str, str]:
    out = {}
    for name in CONTRACT_FILES:
        path = root / "docs" / "contracts" / name
        if not path.is_file():
            raise UncertaintyError(f"contract file {path} is missing")
        out[name] = sha256_file(path)
    return out


class _Ctx:
    def __init__(
        self,
        train_rows,
        train_sha,
        train_cs,
        train_cs_sha,
        train_vectors,
        train_index_id,
        val_rows,
        val_sha,
        val_cs,
        val_vectors,
        manifest_sha,
        bootstrap_b,
        score_chunk,
    ):
        self.train_sha, self.train_cs_sha = train_sha, train_cs_sha
        self.train_vectors, self.train_index_id = train_vectors, train_index_id
        self.embed_model_id = train_vectors.embed_model_id
        # the reranker and the selector never see truth_product_id
        self.pool = [curve._blind(r) for r in sorted(train_rows, key=lambda r: r["query_id"])]
        self.train_blind = {r["query_id"]: r for r in self.pool}
        self.train_cs_by_q = {r["query_id"]: r for r in train_cs}
        self.val_sha, self.val_cs, self.val_vectors = val_sha, val_cs, val_vectors
        self.val_blind = [curve._blind(r) for r in val_rows]
        self.truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
        self.baseline_results = curve.to_results(val_rows, self.truth)
        self.manifest_sha, self.bootstrap_b, self.score_chunk = manifest_sha, bootstrap_b, score_chunk


def _train(ctx: _Ctx, events: list, seed: int) -> tuple[dict, dict]:
    labels, report = build_labels(events)
    qids = sorted({lb.query_id for lb in labels})
    model = rerank_v2.train(
        labels,
        [ctx.train_blind[q] for q in qids],
        ctx.train_sha,
        [ctx.train_cs_by_q[q] for q in qids],
        ctx.train_cs_sha,
        report["label_version"],
        ctx.train_vectors.pair,
        ctx.train_index_id,
        seed,
        ctx.embed_model_id,
    )
    model["label_policy"] = (
        POLICY  # the deployment gate accepts "stratified" only (c6-deploy-gate-v1 section 3)
    )
    return model, report


def _evaluate(ctx: _Ctx, model: dict) -> dict:
    reranked = rerank_v2.rerank_rows(
        model, ctx.val_blind, ctx.val_cs, ctx.val_sha, ctx.val_vectors.pair, ctx.val_vectors.embed_model_id
    )
    ids = [r["query_id"] for r in reranked]
    results = curve.to_results(reranked, ctx.truth)
    check_population("reranked validation", results, ctx.manifest_sha)
    macro = curve.macro_metrics(results)
    paired = paired_bootstrap(ctx.baseline_results, results, KS, b=ctx.bootstrap_b, seed=BOOTSTRAP_SEED)
    delta = {
        str(k): {
            "delta": paired["macro"][k]["delta"],
            "ci": list(paired["macro"][k]["ci"]),
            "p_le_0": paired["macro"][k]["p_le_0"],
        }
        for k in KS
    }
    numbers = [*macro.values()]
    for d in delta.values():
        numbers += [d["delta"], *d["ci"], d["p_le_0"]]
    if not curve._finite(*numbers):
        raise CurveError("non-finite metric")
    return {
        "population_sha256": curve.ids_sha256(set(ids)),
        "manifest_sha256": ctx.manifest_sha,
        "population_match": True,
        "recall": macro,
        "delta_vs_baseline": delta,
    }


def _run_point(
    ctx: _Ctx, key: Key, round_: int, events: list, attempt: int, out_dir: Path, commit: str | None
) -> tuple[dict, dict | None]:
    """One attempt at one point. Returns (record, model); a failure is recorded, not raised."""
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        **key.as_dict(),
        "attempt": attempt,
        "status": "failed",
        "started_at": curve._now(),
        "code_commit": commit,
        "hypothesis_id": HYPOTHESIS_ID,
        "preregistration": PREREGISTRATION,
        "policy": POLICY,
        "round": round_,
        "n_events": len(events),
        "error": None,
        "error_type": None,
        "retryable": False,
    }
    model = None
    try:
        model, report = _train(ctx, events, key.seed)
        rec["label_version"] = report["label_version"]
        rec["n_pos"], rec["n_neg"] = report["pos"], report["neg"]
        rec["reranker_version"] = model["reranker_version"]
        t = model["training"]
        rec["training"] = {
            k: t[k]
            for k in ("mode", "epochs_run", "best_epoch", "early_stopped", "fallback_reason", "n_fit")
            if k in t
        }
        rec["training"]["n_holdout"] = t["n_holdout"]
        rerank_v2.save_model(model, out_dir / "models")
        rec.update(_evaluate(ctx, model))
        rec["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - a failed key is recorded and the path stops there
        rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rec["error_type"] = type(exc).__name__
        rec["retryable"] = isinstance(exc, OSError)
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    rec["finished_at"] = curve._now()
    return rec, model if rec["status"] == "completed" else None


def _plan_dict(run_id, keys, points, seeds, inputs, ctx_info) -> dict:
    return {
        "run_id": run_id,
        "hypothesis_id": HYPOTHESIS_ID,
        "preregistration": PREREGISTRATION,
        "contract": CONTRACT_NAME,
        "started_at": curve._now(),
        "policy": POLICY,
        "noise": 0.0,
        "seeds": list(seeds),
        "points": list(points),
        "keys": [k.as_dict() for k in keys],
        "contract_sha256": _contract_hashes(),
        "git": curve._git_state(),
        "inputs": inputs,
        **ctx_info,
    }


def _check_resume_state(out_dir: Path, seed: int, events: list, points: Sequence[int], initial: list) -> int:
    """Index of the point whose labels the events file holds; refuses a file that is not a path state."""
    sizes = list(points)
    if len(events) not in sizes:
        raise UncertaintyError(
            f"{_events_path(out_dir, seed)}: {len(events)} events is not a measurement point"
        )
    if [e.event_id for e in events[: len(initial)]] != [e.event_id for e in initial]:
        raise UncertaintyError(f"{_events_path(out_dir, seed)}: the first {len(initial)} events are not I_s")
    return sizes.index(len(events))


def run_uncertainty_path(
    config: ExperimentConfig,
    train_rankings: Path,
    val_rankings: Path,
    train_cand_stats: Path,
    val_cand_stats: Path,
    out_root: Path,
    run_id: str,
    seeds: Sequence[int] = SEEDS,
    embedder: str = "siglip",
    artifacts_root: Path = Path("artifacts"),
    *,
    vector_loader: Callable[[str, list[str], str], Any] | None = None,
    points: Sequence[int] = POINTS,
    manifest_sha: str = MANIFEST_SHA256,
    bootstrap_b: int = BOOTSTRAP_B,
    score_chunk: int = SCORE_CHUNK_QUERIES,
) -> dict:
    """Run (or resume) the uncertainty paths of ``seeds``. Returns counts and the run directory.

    ``manifest_sha`` is the frozen validation manifest; tests pass the hash of a fixture, the CLI never does.
    """
    if embedder not in ("siglip", "fake"):
        raise UncertaintyError(f"unknown embedder {embedder!r}; expected 'siglip' or 'fake'")
    if not run_id or "/" in run_id:
        raise UncertaintyError("run_id must be a non-empty directory name")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise UncertaintyError(f"seeds must be non-empty and unique, got {seeds}")
    points = check_points(points)
    train_rankings, val_rankings = Path(train_rankings), Path(val_rankings)

    train_rows = sim.load_rankings(train_rankings)
    val_rows = sim.load_rankings(val_rankings)
    train_sha, val_sha = sha256_file(train_rankings), sha256_file(val_rankings)
    train_cs = rerank_v1.load_cand_stats(train_cand_stats)
    val_cs = rerank_v1.load_cand_stats(val_cand_stats)
    rerank_v1.check_alignment(train_rows, train_sha, train_cs)
    rerank_v1.check_alignment(val_rows, val_sha, val_cs)
    train_index_id, val_index_id = cand_stats_index_id(train_cs), cand_stats_index_id(val_cs)

    truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
    if len(truth) != len(val_rows):
        raise UncertaintyError("duplicate query_id in the validation rankings")
    baseline_results = curve.to_results(val_rows, truth)
    check_population("validation rankings", baseline_results, manifest_sha)  # before any metric
    baseline = recall_at_k(baseline_results, KS)
    if points[-1] > 3 * len(train_rows):
        raise UncertaintyError(f"{points[-1]} labels exceed 3 per query x {len(train_rows)} train queries")

    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path, runs_path, rounds_path = (
        out_dir / "plan.json",
        out_dir / "runs.jsonl",
        out_dir / "rounds.jsonl",
    )
    inputs = {
        "rankings_sha256": {"train": train_sha, "val": val_sha},
        "cand_stats_sha256": {"train": sha256_file(train_cand_stats), "val": sha256_file(val_cand_stats)},
        "index_id": {"train": train_index_id, "val": val_index_id},
        "embedder": embedder,
        "config_name": config.name,
        "embed_model_id": config.embed_model_id,
    }
    plan = _plan_dict(
        run_id,
        plan_keys(seeds, points),
        points,
        seeds,
        inputs,
        {
            "manifest_sha256": manifest_sha,
            "val_query_list_sha256": curve.ids_sha256(list(truth)),
            "val_population": {"n_queries": len(truth), "n_products": len(set(truth.values()))},
            "baseline": {"macro": {str(k): float(baseline["macro"][k]) for k in KS}},
            "bootstrap": {"b": bootstrap_b, "seed": BOOTSTRAP_SEED},
            "exposed_k": curve.EXPOSED_K,
            "per_query_cap": 3,
        },
    )
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        for field in (
            "keys",
            "hypothesis_id",
            "contract_sha256",
            "inputs",
            "manifest_sha256",
            "val_query_list_sha256",
            "bootstrap",
        ):
            if old.get(field) != plan[field]:
                raise UncertaintyError(f"{plan_path} exists with a different {field!r}; use a new --run-id")
        plan = old
    else:
        curve._write_json(plan_path, plan)
    commit = plan["git"]["commit"]

    loader = vector_loader or curve._default_loader(config, embedder, artifacts_root)
    train_vectors = loader("train", [r["query_id"] for r in train_rows], train_index_id)
    val_vectors = loader("val", [r["query_id"] for r in val_rows], val_index_id)
    if train_vectors.embed_model_id != val_vectors.embed_model_id:
        raise UncertaintyError(
            f"train embedding model {train_vectors.embed_model_id!r} differs from the val one "
            f"{val_vectors.embed_model_id!r}"
        )
    ctx = _Ctx(
        train_rows,
        train_sha,
        train_cs,
        inputs["cand_stats_sha256"]["train"],
        train_vectors,
        train_index_id,
        val_rows,
        val_sha,
        val_cs,
        val_vectors,
        manifest_sha,
        bootstrap_b,
        score_chunk,
    )

    existing: dict[str, list[dict]] = defaultdict(list)
    for row in curve.read_runs(runs_path):
        existing[row["key_id"]].append(row)
    result = {"run_dir": str(out_dir), "run_id": run_id, "attempted": 0, "completed": 0, "failed": 0}
    for seed in seeds:
        _run_seed(ctx, train_rows, seed, points, out_dir, commit, existing, runs_path, rounds_path, result)
    return result


def _run_seed(
    ctx, train_rows, seed, points, out_dir, commit, existing, runs_path, rounds_path, result
) -> None:
    oracle = sim.Oracle(train_rows, rankings_sha256=ctx.train_sha, seed=seed, policy_id=POLICY, noise=0.0)
    initial = oracle.answer(sim.stratified_pairs(train_rows, budget=points[0], seed=seed), first_index=0)
    ev_path = _events_path(out_dir, seed)
    if ev_path.exists():
        events = load_events(ev_path)
        j = _check_resume_state(out_dir, seed, events, points, initial)
    else:
        events, j = list(initial), 0
        _write_events_atomic(events, ev_path)

    # every earlier point must be a completed record whose label_version matches the events prefix
    for earlier in range(j):
        done = [a for a in existing[key_for(seed, points[earlier]).id] if a["status"] == "completed"]
        if not done:
            raise UncertaintyError(
                f"seed {seed}: point {points[earlier]} has no completed record but the events go on"
            )
        _, report = build_labels(events[: points[earlier]])
        if done[0]["label_version"] != report["label_version"]:
            raise UncertaintyError(
                f"seed {seed}: point {points[earlier]} record does not match the events file"
            )

    while True:
        key = key_for(seed, points[j])
        attempts = existing[key.id]
        completed = [a for a in attempts if a["status"] == "completed"]
        model = None
        if completed:
            path = out_dir / "models" / completed[0]["reranker_version"] / "model.json"
            model = rerank_v2.load_model(path)
        else:
            attempt = curve._next_attempt(attempts)
            if attempt is None:
                return  # a permanent failure: the path ends here
            rec, model = _run_point(ctx, key, j, events, attempt, out_dir, commit)
            curve._append_line(runs_path, rec)
            attempts.append(rec)
            result["attempted"] += 1
            result["completed" if rec["status"] == "completed" else "failed"] += 1
            if model is None:
                continue  # retried by _next_attempt if it was an OSError, else the loop returns
        if j == len(points) - 1:
            return
        n_take = points[j + 1] - points[j]
        t0 = time.perf_counter()
        batch = select_next_batch(
            model,
            ctx.pool,
            ctx.train_cs_by_q,
            ctx.train_sha,
            ctx.train_vectors.pair,
            ctx.embed_model_id,
            {(e.query_id, e.product_id): e.action for e in events},
            n_take,
            ctx.score_chunk,
        )
        t_select = time.perf_counter() - t0
        new = oracle.answer(list(batch), first_index=len(events))
        events = events + new
        _write_events_atomic(events, ev_path)
        n_pos = sum(e.action == "match" for e in new)
        curve._append_line(
            rounds_path,
            {
                "hypothesis_id": HYPOTHESIS_ID,
                "seed": seed,
                "round": j + 1,
                "from_size": points[j],
                "to_size": points[j + 1],
                "n_selected": len(new),
                "batch_pos": n_pos,
                "batch_neg": len(new) - n_pos,
                "batch_pos_share": n_pos / len(new) if new else None,
                "selection_seconds": round(t_select, 3),
                "model_version": model["reranker_version"],
                "finished_at": curve._now(),
            },
        )
        j += 1


def read_rounds(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


__all__ = [
    "HYPOTHESIS_ID",
    "POINTS",
    "UncertaintyError",
    "check_population",
    "key_for",
    "plan_keys",
    "run_uncertainty_path",
    "score_pool",
    "select_next_batch",
]
