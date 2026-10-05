"""Learning-curve runner, contract c4-learning-curve v1 (docs/contracts/c4-learning-curve.md).

Runs the 108-key grid (policy x noise x seed x budget): for each key it simulates the feedback
events, builds labels (c4-labels-v1), trains reranker v2 from scratch with ``seed = key seed``, reranks
the validation rankings and measures product-macro R@1/5/10/20 with a paired product bootstrap against
the fixed baseline. One JSON line per attempt is appended to ``runs.jsonl`` immediately, so a crash loses at
most the run in flight and a re-run with the same ``run_id`` resumes.

``summarize`` turns ``plan.json`` + ``runs.jsonl`` into ``summary.json``, ``table.md`` and ``curve.png``.
Statistical choices (population, denominators, bootstrap, comparison operators) are fixed by the
contract and by the constants below; nothing here is tuned.

``truth_product_id`` is read only by the simulator (allowed by c4-v3) and by the evaluation step; the
reranker and the trainer receive rankings without that field.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes, sha256_file
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.retrieval import QueryResult, recall_at_k
from product_retrieval.feedback import simulate as sim
from product_retrieval.feedback.labels import build_labels
from product_retrieval.pipelines.cand_vectors import cand_stats_index_id, load_candidate_vectors
from product_retrieval.rerank import v1 as rerank_v1
from product_retrieval.rerank import v2 as rerank_v2

HYPOTHESIS_ID = "H1-rerank-r5"
PREREGISTRATION = "c4-v3"
CONTRACT_NAME = "c4-learning-curve v1"
SCOPE = "모델 선택에 사용한 개발 검증셋의 관측 결과"
TARGET = 0.887  # absolute target macro R@5 (c4-v3 section 7); not rounded from the baseline
PLUS_THREE = 0.03
BUDGETS = (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000)
SEEDS = (0, 1, 2)
# (policy, noise): primary, robustness, robustness, reference (contract section 3, in this order)
CONDITIONS = (("stratified", 0.0), ("stratified", 0.1), ("stratified", 0.2), ("random", 0.0))
PRIMARY = ("stratified", 0.0)
KS = (1, 5, 10, 20)
EXPOSED_K = 20
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 0
REFERENCE_BUDGET = 3000
EXPECTED_POPULATION = (6782, 3243)  # validation queries, products (contract section 4)
CONTRACT_FILES = (
    "c4-learning-curve.md",
    "c4-label-selection.md",
    "c4-event-labels.md",
    "c5-reranker-v2.md",
    "c5-reranker-v1.md",
)
MAX_ATTEMPTS = 2  # first try plus one retry, retry only for non-deterministic (OS) errors
NOT_REACHED_TEXT = "지정 지점 3,000까지 도달 관측 없음"
UNDETERMINED_TEXT = "판정 불가"
NO_RUN_TEXT = "미실행"
FAILED_TEXT = "실패"

_REPO_ROOT = Path(__file__).resolve().parents[3]


class CurveError(ValueError):
    """Invalid curve input, a corrupt run directory or a violated population/nesting check."""


class NestingViolation(CurveError):
    """Stratified events at a smaller budget are not the prefix of the largest-budget events."""


class PopulationMismatch(CurveError):
    """The reranked validation query set differs from the fixed population."""


class Key(NamedTuple):
    policy: str
    noise: float
    seed: int
    budget: int

    @property
    def id(self) -> str:
        return f"{self.policy}|p{self.noise:g}|s{self.seed}|B{self.budget}"

    def as_dict(self) -> dict:
        return {
            "key_id": self.id,
            "policy": self.policy,
            "noise": self.noise,
            "seed": self.seed,
            "budget": self.budget,
        }


def plan_keys() -> list[Key]:
    """The 108 contract keys: condition (primary, 0.1, 0.2, random) > seed > budget ascending."""
    return [
        Key(policy, float(noise), seed, budget)
        for policy, noise in CONDITIONS
        for seed in SEEDS
        for budget in BUDGETS
    ]


# ---- small helpers -----------------------------------------------------------------------------


def reached_target(r5: float) -> bool:
    """The pre-registered target test: macro R@5 >= 0.887 (equality counts as reached)."""
    return r5 >= TARGET


def ci_lower_above_zero(lo: float) -> bool:
    """Conservative: the interval excludes 0 only if the lower bound is strictly above 0."""
    return lo > 0.0


def ci_lower_above_plus_three(lo: float) -> bool:
    """Conservative: the +0.03 claim needs the lower bound strictly above 0.03."""
    return lo > PLUS_THREE


def ids_sha256(query_ids: Sequence[str]) -> str:
    """sha256 over the sorted query ids joined by newlines."""
    return sha256_bytes("\n".join(sorted(query_ids)).encode("utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dumps(obj: Any, *, indent: int | None = None) -> str:
    # allow_nan=False: a NaN can never be written into a result file
    return json.dumps(obj, indent=indent, sort_keys=True, allow_nan=False)


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _append_line(path: Path, obj: dict) -> None:
    line = _dumps(obj) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _finite(*values: float) -> bool:
    return all(isinstance(v, int | float) and math.isfinite(v) for v in values)


def _git_state(root: Path = _REPO_ROOT) -> dict:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "dirty_files": [], "error": head.stderr.strip()[-300:]}
    status = run("status", "--porcelain")
    lines = status.stdout.splitlines() if status.returncode == 0 else []
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(lines) if status.returncode == 0 else None,
        "dirty_files": lines[:50],
        "error": None if status.returncode == 0 else status.stderr.strip()[-300:],
    }


def _contract_hashes(root: Path = _REPO_ROOT) -> dict[str, str]:
    out = {}
    for name in CONTRACT_FILES:
        path = root / "docs" / "contracts" / name
        if not path.is_file():
            raise CurveError(f"contract file {path} is missing")
        out[name] = sha256_file(path)
    return out


# ---- evaluation --------------------------------------------------------------------------------


def to_results(rows: Sequence[dict], truth: dict[str, str]) -> list[QueryResult]:
    """QueryResults of ranking rows; the truth comes from the evaluation-side map, not from the rows."""
    return [QueryResult(r["query_id"], truth[r["query_id"]], tuple(r["top_k_product_ids"])) for r in rows]


def macro_metrics(results: Sequence[QueryResult]) -> dict[str, float]:
    """Product-macro R@K (queries averaged within a product, then products equally) as {"1": ...}."""
    macro = recall_at_k(results, KS)["macro"]
    return {str(k): float(macro[k]) for k in KS}


def seed_mean(values: Sequence[float]) -> float:
    """Equal-weight arithmetic mean over seeds."""
    if not values:
        raise CurveError("seed mean of no values")
    return sum(values) / len(values)


# ---- run directory ----------------------------------------------------------------------------


def read_runs(path: Path) -> list[dict]:
    """All attempt lines of a runs.jsonl; a missing file is an empty run list, a corrupt line an error."""
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise CurveError(f"{path}:{lineno}: unparsable line: {exc}") from exc
            for field in ("key_id", "attempt", "status"):
                if field not in row:
                    raise CurveError(f"{path}:{lineno}: missing field {field!r}")
            rows.append(row)
    return rows


def _next_attempt(attempts: list[dict]) -> int | None:
    """Attempt number to run for a key given its earlier lines, or None if it must not run again."""
    if not attempts:
        return 1
    if any(a["status"] == "completed" for a in attempts):
        return None
    last = max(attempts, key=lambda a: a["attempt"])
    if last["status"] == "failed" and last.get("retryable") and len(attempts) < MAX_ATTEMPTS:
        return last["attempt"] + 1
    return None


class _Context:
    """Everything loaded once and reused by every run."""

    def __init__(
        self,
        train_rows: list[dict],
        train_sha: str,
        train_cs: list[dict],
        train_cs_sha: str,
        train_vectors: Any,
        val_rows: list[dict],
        val_sha: str,
        val_cs: list[dict],
        val_vectors: Any,
        embed_model_id: str | None,
        train_index_id: str,
        bootstrap_b: int,
        reference_budget: int,
    ):
        self.train_rows, self.train_sha = train_rows, train_sha
        self.train_cs, self.train_cs_sha = train_cs, train_cs_sha
        self.train_vectors = train_vectors
        self.train_index_id = train_index_id
        self.embed_model_id = embed_model_id
        self.bootstrap_b, self.reference_budget = bootstrap_b, reference_budget
        # the reranker side never sees truth_product_id
        self.train_blind = {r["query_id"]: _blind(r) for r in train_rows}
        self.train_cs_by_q = {r["query_id"]: r for r in train_cs}
        self.val_sha = val_sha
        self.val_cs = val_cs
        self.val_vectors = val_vectors
        self.val_blind = [_blind(r) for r in val_rows]
        self.truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
        self.population = set(self.truth)
        self.baseline_results = to_results(val_rows, self.truth)
        self.ref_events: dict[tuple[str, float, int], list[tuple]] = {}


def _blind(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "truth_product_id"}


def _signature(events: Sequence) -> list[tuple]:
    return [(e.event_id, e.query_id, e.product_id, e.action, e.position) for e in events]


def _simulate(ctx: _Context, key: Key, budget: int) -> list:
    fn = sim.simulate_stratified if key.policy == "stratified" else sim.simulate_random
    return fn(
        ctx.train_rows,
        budget=budget,
        seed=key.seed,
        rankings_sha256=ctx.train_sha,
        exposed_k=EXPOSED_K,
        noise=key.noise,
    )


def _is_nested(ctx: _Context, key: Key, events: list) -> bool:
    """True if the events are the first ``budget`` events of the reference-budget run."""
    group = (key.policy, key.noise, key.seed)
    if group not in ctx.ref_events:
        ctx.ref_events[group] = _signature(_simulate(ctx, key, ctx.reference_budget))
    return _signature(events) == ctx.ref_events[group][: key.budget]


def _run_one(ctx: _Context, key: Key, attempt: int, out_dir: Path, code_commit: str | None) -> dict:
    """One attempt. Always returns a record; failures are recorded, never raised (except Ctrl-C)."""
    t0 = time.perf_counter()
    rec: dict[str, Any] = {
        **key.as_dict(),
        "attempt": attempt,
        "status": "failed",
        "started_at": _now(),
        "code_commit": code_commit,
        "hypothesis_id": HYPOTHESIS_ID,
        "preregistration": PREREGISTRATION,
        "error": None,
        "error_type": None,
        "retryable": False,
    }
    try:
        events = _simulate(ctx, key, key.budget)
        rec["n_events"] = len(events)
        rec["nested"] = _is_nested(ctx, key, events)
        if key.policy == "stratified" and not rec["nested"]:
            # contract section 3: stratified must nest; random is only recorded
            raise NestingViolation(
                f"{key.id}: the budget-{key.budget} events are not the first {key.budget} of the "
                f"budget-{ctx.reference_budget} events"
            )
        labels, report = build_labels(events)
        rec["label_version"] = report["label_version"]
        rec["n_pos"], rec["n_neg"] = report["pos"], report["neg"]

        qids = {lb.query_id for lb in labels}
        model = rerank_v2.train(
            labels,
            [ctx.train_blind[q] for q in sorted(qids)],
            ctx.train_sha,
            [ctx.train_cs_by_q[q] for q in sorted(qids)],
            ctx.train_cs_sha,
            report["label_version"],
            ctx.train_vectors.pair,
            ctx.train_index_id,
            key.seed,
            ctx.embed_model_id,
        )
        rec["reranker_version"] = model["reranker_version"]
        t = model["training"]
        rec["training"] = {
            k: t[k]
            for k in ("mode", "epochs_run", "best_epoch", "early_stopped", "fallback_reason", "n_fit")
            if k in t
        }
        rec["training"]["n_holdout"] = t["n_holdout"]
        rerank_v2.save_model(model, out_dir / "models")

        reranked = rerank_v2.rerank_rows(
            model,
            ctx.val_blind,
            ctx.val_cs,
            ctx.val_sha,
            ctx.val_vectors.pair,
            ctx.val_vectors.embed_model_id,
        )
        ids = [r["query_id"] for r in reranked]
        rec["population_sha256"] = ids_sha256(set(ids))
        rec["population_match"] = len(ids) == len(set(ids)) and set(ids) == ctx.population
        if not rec["population_match"]:
            raise PopulationMismatch(
                f"{key.id}: reranked query set ({len(set(ids))} unique of {len(ids)}) differs from the "
                f"fixed population ({len(ctx.population)})"
            )
        new_results = to_results(reranked, ctx.truth)
        macro = macro_metrics(new_results)
        paired = paired_bootstrap(
            ctx.baseline_results, new_results, KS, b=ctx.bootstrap_b, seed=BOOTSTRAP_SEED
        )
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
        if not _finite(*numbers):
            raise CurveError(f"{key.id}: non-finite metric")
        rec["recall"], rec["delta"] = macro, delta
        rec["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - a failed key is recorded and the grid continues
        rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rec["error_type"] = type(exc).__name__
        rec["retryable"] = isinstance(exc, OSError)
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    rec["finished_at"] = _now()
    return rec


# ---- run_curve --------------------------------------------------------------------------------

VectorLoader = Callable[[str, list[str], str], Any]


def _default_loader(config: ExperimentConfig, embedder: str, artifacts_root: Path) -> VectorLoader:
    def load(split: str, query_ids: list[str], index_id: str) -> Any:
        if split not in ("train", "val"):
            raise CurveError(f"split {split!r} is not allowed here; only train and val")
        return load_candidate_vectors(
            config,
            split,
            sorted(set(query_ids)),
            index_id,
            embedder,
            artifacts_root,  # type: ignore[arg-type]
        )

    return load


def _check_keys(keys: Sequence[Key], reference_budget: int) -> list[Key]:
    out = [Key(k[0], float(k[1]), int(k[2]), int(k[3])) for k in keys]
    if len({k.id for k in out}) != len(out):
        raise CurveError("duplicate condition keys")
    for k in out:
        if k.policy not in sim.POLICIES:
            raise CurveError(f"unknown policy in key {k.id}")
        if not 0.0 <= k.noise <= 1.0 or k.budget < 1:
            raise CurveError(f"invalid noise or budget in key {k.id}")
        if k.budget > reference_budget:
            raise CurveError(f"budget of {k.id} exceeds the nesting reference budget {reference_budget}")
    return out


def _population_summary(val_rows: list[dict]) -> dict:
    truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
    if len(truth) != len(val_rows):
        raise CurveError("duplicate query_id in the validation rankings")
    return {"n_queries": len(truth), "n_products": len(set(truth.values()))}


def run_curve(
    config: ExperimentConfig,
    train_rankings: Path,
    val_rankings: Path,
    train_cand_stats: Path,
    val_cand_stats: Path,
    out_root: Path = Path("reports/curve"),
    embedder: str = "siglip",
    artifacts_root: Path = Path("artifacts"),
    run_id: str | None = None,
    keys: Sequence[Key | tuple] | None = None,
    *,
    vector_loader: VectorLoader | None = None,
    reference_budget: int = REFERENCE_BUDGET,
    expected_population: tuple[int, int] | None = EXPECTED_POPULATION,
    bootstrap_b: int = BOOTSTRAP_B,
) -> dict:
    """Run (or resume) the grid. ``keys=None`` runs all 108; a subset is for smoke tests and resumes.

    ``plan.json`` always lists the 108 contract keys and, when a subset is run, the requested ones.
    Returns ``{"run_dir", "run_id", "attempted", "completed", "failed", "skipped"}`` for this call.
    """
    if embedder not in ("siglip", "fake"):
        raise CurveError(f"unknown embedder {embedder!r}; expected 'siglip' or 'fake'")
    requested = _check_keys(plan_keys() if keys is None else keys, reference_budget)
    train_rankings, val_rankings = Path(train_rankings), Path(val_rankings)
    train_cand_stats, val_cand_stats = Path(train_cand_stats), Path(val_cand_stats)

    train_rows = sim.load_rankings(train_rankings)
    val_rows = sim.load_rankings(val_rankings)
    train_sha, val_sha = sha256_file(train_rankings), sha256_file(val_rankings)
    train_cs, val_cs = rerank_v1.load_cand_stats(train_cand_stats), rerank_v1.load_cand_stats(val_cand_stats)
    # fail early on mismatched rankings / cand-stats pairs
    rerank_v1.check_alignment(train_rows, train_sha, train_cs)
    rerank_v1.check_alignment(val_rows, val_sha, val_cs)
    train_index_id, val_index_id = cand_stats_index_id(train_cs), cand_stats_index_id(val_cs)

    population = _population_summary(val_rows)
    if expected_population is not None and (population["n_queries"], population["n_products"]) != tuple(
        expected_population
    ):
        raise CurveError(
            f"validation population {population['n_queries']} queries / {population['n_products']} "
            f"products differs from the contract's {expected_population[0]} / {expected_population[1]}"
        )
    truth = {r["query_id"]: r["truth_product_id"] for r in val_rows}
    baseline = recall_at_k(to_results(val_rows, truth), KS)

    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path, runs_path = out_dir / "plan.json", out_dir / "runs.jsonl"

    plan = {
        "run_id": run_id,
        "hypothesis_id": HYPOTHESIS_ID,
        "preregistration": PREREGISTRATION,
        "contract": CONTRACT_NAME,
        "started_at": _now(),
        "keys": [k.as_dict() for k in plan_keys()],
        "requested_key_ids": [k.id for k in requested],
        "contract_sha256": _contract_hashes(),
        "git": _git_state(),
        "inputs": {
            "rankings_sha256": {"train": train_sha, "val": val_sha},
            "cand_stats_sha256": {
                "train": sha256_file(train_cand_stats),
                "val": sha256_file(val_cand_stats),
            },
            "index_id": {"train": train_index_id, "val": val_index_id},
            "embedder": embedder,
            "config_name": config.name,
            "embed_model_id": config.embed_model_id,
        },
        "val_query_list_sha256": ids_sha256(list(truth)),
        "val_population": population,
        "baseline": {
            "macro": {str(k): float(baseline["macro"][k]) for k in KS},
            "micro": {str(k): float(baseline["micro"][k]) for k in KS},
        },
        "target": TARGET,
        "bootstrap": {"b": bootstrap_b, "seed": BOOTSTRAP_SEED},
        "exposed_k": EXPOSED_K,
        "reference_budget": reference_budget,
    }
    current_commit = plan["git"]["commit"]
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        for field in ("keys", "contract_sha256", "inputs", "val_query_list_sha256", "baseline", "bootstrap"):
            if old.get(field) != plan[field]:
                raise CurveError(f"{plan_path} exists with a different {field!r}; use a new --run-id")
        plan = old  # keep the original start time and commit
    else:
        _write_json(plan_path, plan)

    existing: dict[str, list[dict]] = defaultdict(list)
    for row in read_runs(runs_path):
        existing[row["key_id"]].append(row)
    todo = [(k, a) for k in requested if (a := _next_attempt(existing[k.id])) is not None]
    result = {
        "run_dir": str(out_dir),
        "run_id": run_id,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "skipped": len(requested) - len(todo),
    }
    if not todo:
        return result

    loader = vector_loader or _default_loader(config, embedder, artifacts_root)
    train_vectors = loader("train", [r["query_id"] for r in train_rows], train_index_id)
    val_vectors = loader("val", [r["query_id"] for r in val_rows], val_index_id)
    if train_vectors.embed_model_id != val_vectors.embed_model_id:
        raise CurveError(
            f"train embedding model {train_vectors.embed_model_id!r} differs from the val one "
            f"{val_vectors.embed_model_id!r}"
        )
    ctx = _Context(
        train_rows,
        train_sha,
        train_cs,
        plan["inputs"]["cand_stats_sha256"]["train"],
        train_vectors,
        val_rows,
        val_sha,
        val_cs,
        val_vectors,
        train_vectors.embed_model_id,
        train_index_id,
        bootstrap_b,
        reference_budget,
    )
    for key, attempt in todo:
        rec = _run_one(ctx, key, attempt, out_dir, current_commit)
        _append_line(runs_path, rec)
        result["attempted"] += 1
        result["completed" if rec["status"] == "completed" else "failed"] += 1
    return result


# ---- summary ----------------------------------------------------------------------------------


def budget_label(budget: int) -> str:
    return "≤100" if budget == BUDGETS[0] else str(budget)


def first_observed_point(points: Sequence[dict]) -> dict:
    """First pre-specified point whose R@5 reached the target, for one seed.

    ``points``: dicts with ``budget``, ``status`` (completed / failed / not_run) and ``r5`` (completed only),
    ascending by budget. A point that is not completed makes any later reach unconfirmed ("판정 불가"),
    and so does a seed with no reach and a missing point. A drop after the first observed reach is recorded.
    """
    ordered = sorted(points, key=lambda p: p["budget"])
    missing_before = False
    for i, p in enumerate(ordered):
        if p["status"] != "completed":
            missing_before = True
            continue
        if not reached_target(p["r5"]):
            continue
        if missing_before:
            return {
                "verdict": "undetermined",
                "budget": None,
                "label": UNDETERMINED_TEXT,
                "unconfirmed_reach_budget": p["budget"],
                "reason": "an earlier point failed or was not run",
                "persistence": None,
                "dropped_after_reach": [],
            }
        later = ordered[i + 1 :]
        kept = [q["budget"] for q in later if q["status"] == "completed" and reached_target(q["r5"])]
        dropped = [q["budget"] for q in later if q["status"] == "completed" and not reached_target(q["r5"])]
        return {
            "verdict": "reached",
            "budget": p["budget"],
            "label": budget_label(p["budget"]),
            "unconfirmed_reach_budget": None,
            "reason": None,
            "persistence": {
                "n_later": len(later),
                "n_kept": len(kept),
                "n_later_not_observed": sum(q["status"] != "completed" for q in later),
                "ratio": (len(kept) / len(later)) if later else None,
            },
            "dropped_after_reach": dropped,
        }
    if missing_before:
        return {
            "verdict": "undetermined",
            "budget": None,
            "label": UNDETERMINED_TEXT,
            "unconfirmed_reach_budget": None,
            "reason": "no reach observed but a point failed or was not run",
            "persistence": None,
            "dropped_after_reach": [],
        }
    return {
        "verdict": "not_reached",
        "budget": None,
        "label": NOT_REACHED_TEXT,
        "unconfirmed_reach_budget": None,
        "reason": None,
        "persistence": None,
        "dropped_after_reach": [],
    }


def result_sentence(per_seed: Sequence[dict]) -> dict:
    """Contract section 5 result sentence from the per-seed ``first_observed_point`` results."""
    reached = [s["budget"] for s in per_seed if s["verdict"] == "reached"]
    n = len(per_seed)
    seed_text = ", ".join(f"seed{i}: {s['label']}" for i, s in enumerate(per_seed))
    if len(reached) == n:
        med = statistics.median(reached)
        med_text = budget_label(int(med)) if med == int(med) else str(med)
        sentence = f"도달 관측(중앙값 {med_text}, 개발 검증셋)"
    elif reached:
        sentence = f"일부 도달 관측({len(reached)}/{n})"
    elif all(s["verdict"] == "not_reached" for s in per_seed):
        sentence = "미도달"
    else:
        sentence = UNDETERMINED_TEXT
    return {"sentence": sentence, "seed_points": seed_text, "scope": SCOPE}


def _final_records(runs: Sequence[dict], plan_keys_: Sequence[dict]) -> dict[str, dict]:
    """key_id -> the deciding attempt (a completed one, else the latest failed); absent = not run."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_key[r["key_id"]].append(r)
    out = {}
    for k in plan_keys_:
        attempts = by_key.get(k["key_id"], [])
        done = [a for a in attempts if a["status"] == "completed"]
        if done:
            out[k["key_id"]] = done[-1]
        elif attempts:
            out[k["key_id"]] = max(attempts, key=lambda a: a["attempt"])
    return out


def _check_completed(rec: dict) -> None:
    """A completed record with missing or non-finite numbers is a corrupt run directory, not a result."""
    try:
        values = [rec["recall"][str(k)] for k in KS]
        for k in KS:
            d = rec["delta"][str(k)]
            values += [d["delta"], d["ci"][0], d["ci"][1], d["p_le_0"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise CurveError(f"completed record {rec.get('key_id')!r} lacks metrics ({exc!r})") from exc
    if not _finite(*values):
        raise CurveError(f"completed record {rec['key_id']!r} has non-finite metrics")


def condition_name(policy: str, noise: float) -> str:
    return f"{policy}, p={noise:g}"


def summarize(run_dir: Path) -> dict:
    """Write summary.json, table.md and curve.png into ``run_dir`` and return the summary dict."""
    run_dir = Path(run_dir)
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise CurveError(f"{plan_path} is missing; not a curve run directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = read_runs(run_dir / "runs.jsonl")
    final = _final_records(runs, plan["keys"])
    for rec in final.values():
        if rec["status"] == "completed":
            _check_completed(rec)
    baseline_r5 = plan["baseline"]["macro"]["5"]
    if not _finite(baseline_r5):
        raise CurveError("baseline R@5 is not finite")

    def status_of(k: dict) -> str:
        rec = final.get(k["key_id"])
        return rec["status"] if rec else "not_run"

    counts = {"planned": len(plan["keys"]), "completed": 0, "failed": 0, "not_run": 0}
    for k in plan["keys"]:
        counts[status_of(k)] += 1

    # condition -> seed -> budget -> point
    grid: dict[tuple[str, float], dict[int, list[dict]]] = {}
    for k in plan["keys"]:
        rec = final.get(k["key_id"])
        point = {"budget": k["budget"], "status": status_of(k), "key_id": k["key_id"]}
        if rec and rec["status"] == "completed":
            d5 = rec["delta"]["5"]
            point |= {
                "r5": rec["recall"]["5"],
                "recall": rec["recall"],
                "delta_r5": d5["delta"],
                "ci": d5["ci"],
                "p_le_0": d5["p_le_0"],
                "n_pos": rec.get("n_pos"),
                "n_neg": rec.get("n_neg"),
            }
        grid.setdefault((k["policy"], float(k["noise"])), {}).setdefault(k["seed"], []).append(point)

    def mean_points(cond: tuple[str, float]) -> list[dict]:
        out = []
        for b in BUDGETS:
            done = [
                p
                for s in sorted(grid[cond])
                for p in grid[cond][s]
                if p["budget"] == b and p["status"] == "completed"
            ]
            n = len(done)
            entry: dict[str, Any] = {"budget": b, "n_seeds_completed": n, "n_seeds_planned": len(SEEDS)}
            if n:
                entry["mean_r5"] = seed_mean([p["r5"] for p in done])
                entry["mean_delta_r5"] = seed_mean([p["delta_r5"] for p in done])
                entry["mean_label"] = "3-seed 평균" if n == len(SEEDS) else f"성공 {n}/{len(SEEDS)} seed 평균"
                entry["is_full_seed_mean"] = n == len(SEEDS)
            else:
                entry["mean_r5"] = entry["mean_delta_r5"] = None
                entry["mean_label"] = "없음(성공 seed 0)"
                entry["is_full_seed_mean"] = False
            out.append(entry)
        return out

    per_seed = []
    seeds_out = {}
    for s in SEEDS:
        pts = sorted(grid[PRIMARY].get(s, []), key=lambda p: p["budget"])
        first = first_observed_point(pts)
        if first["verdict"] == "reached":
            at = next(p for p in pts if p["budget"] == first["budget"])
            first["delta_ci_lower"] = at["ci"][0]
            first["delta_ci_lower_above_zero"] = ci_lower_above_zero(at["ci"][0])
            first["delta_ci_lower_above_0_03"] = ci_lower_above_plus_three(at["ci"][0])
            first["r5_at_first"] = at["r5"]
            first["p_le_0_at_first"] = at["p_le_0"]
        per_seed.append(first)
        seeds_out[str(s)] = {"first_observed": first, "points": pts}

    sentence = result_sentence(per_seed)
    summary = {
        "hypothesis_id": plan["hypothesis_id"],
        "preregistration": plan["preregistration"],
        "contract": plan["contract"],
        "run_id": plan["run_id"],
        "git_commit": plan["git"]["commit"],
        "git_dirty": plan["git"]["dirty"],
        "contract_sha256": plan["contract_sha256"],
        "rankings_sha256": plan["inputs"]["rankings_sha256"],
        "cand_stats_sha256": plan["inputs"]["cand_stats_sha256"],
        "index_id": plan["inputs"]["index_id"],
        "val_query_list_sha256": plan["val_query_list_sha256"],
        "val_population": plan["val_population"],
        "baseline": {
            "macro": plan["baseline"]["macro"],
            "target_r5": TARGET,
            "target_delta_r5": TARGET - baseline_r5,
        },
        "bootstrap": plan["bootstrap"],
        "counts": counts,
        "primary": {
            "condition": condition_name(*PRIMARY),
            "seeds": seeds_out,
            "result": sentence,
            "points": mean_points(PRIMARY),
        },
        "secondary": {
            condition_name(*c): {
                "points": mean_points(c),
                "seeds": {str(s): grid[c].get(s, []) for s in SEEDS},
            }
            for c in CONDITIONS
            if c != PRIMARY
        },
        "failures": [
            {"key_id": r["key_id"], "attempt": r["attempt"], "error": r.get("error")}
            for r in final.values()
            if r["status"] == "failed"
        ],
        "notes": [
            "random은 공유 초기 라벨(I_s)을 쓰지 않는 참고 정책이다 "
            "(c4-v3 section 3·4의 세 정책에 포함되지 않음).",
            "Δ 구간은 학습이 끝난 한 모델에 조건부인 지점별 구간이며, "
            "모델 선택·지점 선택·학습 seed 변동은 반영하지 않는다. "
            "'Δ ≤ 0 재표집 비율'은 p값이 아니다.",
        ],
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "table.md").write_text(_table_md(plan, final, summary), encoding="utf-8", newline="\n")
    _plot(summary, run_dir / "curve.png")
    return summary


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table_md(plan: dict, final: dict[str, dict], summary: dict) -> str:
    base = plan["baseline"]["macro"]
    lines = [
        f"# 학습 곡선 원자료 ({plan['run_id']})",
        "",
        f"- 가설 {plan['hypothesis_id']}, 사전등록 {plan['preregistration']}, 범위: {SCOPE}",
        "- 기준선 macro R@1/5/10/20 (반올림 없음): "
        + " / ".join(f"{base[str(k)]!r}" for k in KS)
        + f"; 목표 R@5 = {TARGET}, 목표 Δ = {summary['baseline']['target_delta_r5']!r}",
        f"- 완료 {summary['counts']['completed']} · 실패 {summary['counts']['failed']} · "
        f"미실행 {summary['counts']['not_run']} (계획 {summary['counts']['planned']})",
        f"- 결과: {summary['primary']['result']['sentence']} ({summary['primary']['result']['seed_points']})",
        "- random은 공유 초기 라벨(I_s)을 쓰지 않는 참고 정책이다.",
        "",
        "| 조건 | 예산 | seed | 상태 | pos | neg | R@1 | R@5 | R@10 | R@20 | ΔR@5 "
        "| Δ 95% 구간 | Δ≤0 비율 | 중첩 | 시도 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for k in plan["keys"]:
        rec = final.get(k["key_id"])
        cond = condition_name(k["policy"], float(k["noise"]))
        head = f"| {cond} | {k['budget']} | {k['seed']} |"
        if rec is None:
            lines.append(f"{head} {NO_RUN_TEXT} | - | - | - | - | - | - | - | - | - | - |")
        elif rec["status"] == "failed":
            lines.append(f"{head} {FAILED_TEXT} | - | - | - | - | - | - | - | - | - | {rec['attempt']} |")
        else:
            r, d = rec["recall"], rec["delta"]["5"]
            nested = {True: "예", False: "아니오", None: "-"}[rec.get("nested")]
            lines.append(
                f"{head} 완료 | {rec.get('n_pos')} | {rec.get('n_neg')} | "
                + " | ".join(_fmt(r[str(q)]) for q in KS)
                + f" | {_fmt(d['delta'])} | [{_fmt(d['ci'][0])}, {_fmt(d['ci'][1])}] "
                f"| {_fmt(d['p_le_0'], 3)} "
                f"| {nested} | {rec['attempt']} |"
            )
    fails = summary["failures"]
    if fails:
        lines += ["", "## 실패", ""]
        lines += [f"- {f['key_id']} (시도 {f['attempt']}): {f['error']}" for f in fails]
    lines += [
        "",
        "## 주 조건 지점별 평균",
        "",
        "| 예산 | 평균 라벨 | R@5 평균 | ΔR@5 평균 |",
        "|---:|---|---:|---:|",
    ]
    for p in summary["primary"]["points"]:
        lines.append(
            f"| {p['budget']} | {p['mean_label']} | {_fmt(p['mean_r5'])} | {_fmt(p['mean_delta_r5'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _plot(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, block in summary["secondary"].items():
        pts = [p for p in block["points"] if p["is_full_seed_mean"]]
        if pts:
            ax.plot(
                [p["budget"] for p in pts],
                [p["mean_delta_r5"] for p in pts],
                lw=1,
                alpha=0.3,
                label=f"{name} (mean)",
            )
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for s, color in zip(SEEDS, colors, strict=True):
        pts = [p for p in summary["primary"]["seeds"][str(s)]["points"] if p["status"] == "completed"]
        if not pts:
            continue
        x = [p["budget"] for p in pts]
        y = [p["delta_r5"] for p in pts]
        lo = [max(0.0, p["delta_r5"] - p["ci"][0]) for p in pts]  # percentile CI may not bracket the point
        hi = [max(0.0, p["ci"][1] - p["delta_r5"]) for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=4, capsize=2, color=color, alpha=0.8, label=f"seed {s}")
    mean_pts = [p for p in summary["primary"]["points"] if p["is_full_seed_mean"]]
    if mean_pts:
        ax.plot(
            [p["budget"] for p in mean_pts],
            [p["mean_delta_r5"] for p in mean_pts],
            color="black",
            lw=2,
            label="3-seed mean (stratified, p=0)",
        )
    ax.axhline(summary["baseline"]["target_delta_r5"], color="red", ls="--", label="target R@5 0.887")
    ax.axhline(0.0, color="gray", ls=":", label="baseline")
    ax.set_xscale("log")
    ax.set_xlabel("labels (budget, log scale)")
    ax.set_ylabel("delta macro R@5 vs fixed baseline")
    ax.set_title("Learning curve, v2 reranker (development validation set)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
