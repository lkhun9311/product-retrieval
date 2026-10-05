"""Tests for the learning-curve runner (docs/contracts/c4-learning-curve.md).

The run-level tests use small synthetic rankings with an injected vector source (so the grid logic is
exercised without a real index); one test goes through the real vector loader with the fake embedder.
Summary logic is tested on hand-written runs.jsonl files whose expected outputs are worked out in comments.
"""

import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.eval.retrieval import QueryResult
from product_retrieval.feedback import simulate as sim
from product_retrieval.pipelines import curve
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.cand_stats import run_cand_stats
from product_retrieval.pipelines.curve import CurveError, Key
from product_retrieval.pipelines.evaluate import run_eval
from product_retrieval.rerank import v1, v2
from product_retrieval.rerank.v0 import RerankError

DIM = 8
N_TRAIN = 80  # stratified capacity 3 * 80 = 240 >= budgets 100 and 200
N_VAL = 40
N_VAL_PRODUCTS = 15
BASE_R5 = 6 / 7
REPO = Path(__file__).resolve().parents[1]


# ---- synthetic data ----------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _stats(row, sha, index_id):
    out = []
    for i, (pid, score) in enumerate(zip(row["top_k_product_ids"][:20], row["scores"][:20], strict=True)):
        n = 1 + i % 3
        out.append(
            {
                "product_id": pid,
                "n_images": n,
                "max_sim": score,
                "mean_sim": score - 0.05 * (n - 1),
                "second_sim": score - 0.03 * (n - 1),
                "std_sim": 0.01 * (n - 1),
                "top1_sim": 1.0 - 0.02 * i,
            }
        )
    return {"query_id": row["query_id"], "rankings_sha256": sha, "index_id": index_id, "stats": out}


class FakeVectors:
    """pair(query_id, product_id) -> (q, g); the truth candidate's g is close to q, the others random."""

    def __init__(self, truth_pid_of, rows, seed):
        rng = np.random.default_rng(seed)
        self.embed_model_id = "m"
        self.q, self.g = {}, {}
        for r in rows:
            qid = r["query_id"]
            self.q[qid] = _unit(rng.standard_normal(DIM))
            for pid in r["top_k_product_ids"]:
                near = pid == truth_pid_of[qid]
                self.g[(qid, pid)] = _unit(
                    self.q[qid] + 0.1 * rng.standard_normal(DIM) if near else rng.standard_normal(DIM)
                )

    def pair(self, qid, pid):
        return self.q[qid], self.g[(qid, pid)]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def grid(tmp_path):
    def row(qid, truth, pos, n, filler):
        ids = [f"{filler}{qid}-{k}" for k in range(n)]
        ids[pos] = truth
        return {
            "query_id": qid,
            "truth_product_id": truth,
            "top_k_product_ids": ids,
            "scores": [round(0.9 - 0.01 * i, 6) for i in range(n)],
        }

    train_rows = [row(f"t{i}", f"tp{i}", i % 10, 25, "c") for i in range(N_TRAIN)]
    val_rows = [row(f"v{i}", f"vp{i % N_VAL_PRODUCTS}", i % 25, 25, "c") for i in range(N_VAL)]
    paths = {}
    for name, rows in (("train", train_rows), ("val", val_rows)):
        rp = tmp_path / f"{name}.rankings.jsonl"
        _write_jsonl(rp, rows)
        sha = hashlib.sha256(rp.read_bytes()).hexdigest()
        cp = tmp_path / f"{name}.cand_stats.jsonl"
        _write_jsonl(cp, [_stats(r, sha, f"idx-{name}") for r in rows])
        paths[name] = (rp, cp)
    vecs = {
        "train": FakeVectors({r["query_id"]: r["truth_product_id"] for r in train_rows}, train_rows, 1),
        "val": FakeVectors({r["query_id"]: r["truth_product_id"] for r in val_rows}, val_rows, 2),
    }
    calls = []

    def loader(split, query_ids, index_id):
        calls.append((split, len(query_ids), index_id))
        return vecs[split]

    config = ExperimentConfig(
        name="t",
        seed=0,
        embed_model_id="m",
        crop_kind="full",
        index_params={},
        manifest_path=tmp_path / "unused.jsonl",
        data_root=tmp_path,
    )
    return SimpleNamespace(
        train_rows=train_rows,
        val_rows=val_rows,
        paths=paths,
        vecs=vecs,
        loader=loader,
        calls=calls,
        config=config,
        out_root=tmp_path / "curve",
    )


def run(g, keys, run_id="r", **kw):
    kw.setdefault("reference_budget", 200)
    kw.setdefault("expected_population", None)
    return curve.run_curve(
        g.config,
        g.paths["train"][0],
        g.paths["val"][0],
        g.paths["train"][1],
        g.paths["val"][1],
        out_root=g.out_root,
        embedder="fake",
        run_id=run_id,
        keys=keys,
        vector_loader=g.loader,
        **kw,
    )


def lines(g, run_id="r"):
    path = g.out_root / run_id / "runs.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


K_A = Key("stratified", 0.0, 0, 100)
K_B = Key("stratified", 0.0, 0, 200)
K_C = Key("stratified", 0.1, 1, 100)
K_R = Key("random", 0.0, 0, 100)


# ---- plan ---------------------------------------------------------------------------------------


def test_plan_has_108_keys_in_contract_order():
    keys = curve.plan_keys()
    assert len(keys) == 108 == len({k.id for k in keys})
    assert curve.BUDGETS == (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000)
    assert keys[0] == ("stratified", 0.0, 0, 100)
    assert keys[8] == ("stratified", 0.0, 0, 3000)
    assert keys[9] == ("stratified", 0.0, 1, 100)
    assert keys[26] == ("stratified", 0.0, 2, 3000)
    assert keys[27] == ("stratified", 0.1, 0, 100)
    assert keys[54] == ("stratified", 0.2, 0, 100)
    assert keys[81] == ("random", 0.0, 0, 100)
    assert keys[107] == ("random", 0.0, 2, 3000)
    # condition blocks of 27: primary, 0.1, 0.2, random
    assert [(k.policy, k.noise) for k in keys[::27]] == [
        ("stratified", 0.0),
        ("stratified", 0.1),
        ("stratified", 0.2),
        ("random", 0.0),
    ]
    for block in range(4):
        part = keys[27 * block : 27 * block + 27]
        assert [(k.seed, k.budget) for k in part] == [(s, b) for s in (0, 1, 2) for b in curve.BUDGETS]


# ---- run: bookkeeping, independent recomputation -----------------------------------------------


def test_run_records_one_line_per_attempt_and_matches_independent_numbers(grid, monkeypatch):
    seen = {"train_rows": [], "rerank_rows": []}
    real_train, real_rerank = v2.train, v2.rerank_rows

    def spy_train(labels, rankings, *a, **k):
        seen["train_rows"] += rankings
        return real_train(labels, rankings, *a, **k)

    def spy_rerank(model, rows, *a, **k):
        seen["rerank_rows"] += rows
        return real_rerank(model, rows, *a, **k)

    monkeypatch.setattr(curve.rerank_v2, "train", spy_train)
    monkeypatch.setattr(curve.rerank_v2, "rerank_rows", spy_rerank)

    res = run(grid, [K_A, K_B, K_C, K_R])
    assert (res["attempted"], res["completed"], res["failed"], res["skipped"]) == (4, 4, 0, 0)
    recs = lines(grid)
    assert [r["key_id"] for r in recs] == [k.id for k in (K_A, K_B, K_C, K_R)]
    assert all(r["attempt"] == 1 and r["status"] == "completed" for r in recs)
    # vectors were loaded once per split, not once per run
    assert grid.calls == [("train", N_TRAIN, "idx-train"), ("val", N_VAL, "idx-val")]
    # the trainer and the reranker never see the truth field
    assert seen["train_rows"] and seen["rerank_rows"]
    assert all("truth_product_id" not in r for r in seen["train_rows"] + seen["rerank_rows"])

    # independent baseline: hand loop over the val rows
    def macro(rows, k):
        per = {}
        for r in rows:
            per.setdefault(r["truth_product_id"], []).append(
                1.0 if r["truth_product_id"] in r["top_k_product_ids"][:k] else 0.0
            )
        return sum(sum(v) / len(v) for v in per.values()) / len(per)

    plan = json.loads((grid.out_root / "r" / "plan.json").read_text(encoding="utf-8"))
    for k in (1, 5, 10, 20):
        assert plan["baseline"]["macro"][str(k)] == pytest.approx(macro(grid.val_rows, k), abs=1e-12)
    assert plan["val_population"] == {"n_queries": N_VAL, "n_products": N_VAL_PRODUCTS}

    expected_sha = hashlib.sha256(
        "\n".join(sorted(r["query_id"] for r in grid.val_rows)).encode()
    ).hexdigest()
    assert plan["val_query_list_sha256"] == expected_sha
    truth = {r["query_id"]: r["truth_product_id"] for r in grid.train_rows}
    for rec in recs:
        assert rec["population_match"] is True and rec["population_sha256"] == expected_sha
        # independent label counts: a stratified/no-noise match is a pair whose candidate is the truth
        k = Key(rec["policy"], rec["noise"], rec["seed"], rec["budget"])
        events = sim.simulate_stratified if k.policy == "stratified" else sim.simulate_random
        ev = events(
            grid.train_rows,
            budget=k.budget,
            seed=k.seed,
            rankings_sha256=plan["inputs"]["rankings_sha256"]["train"],
            noise=k.noise,
        )
        assert rec["n_events"] == k.budget
        assert rec["n_pos"] + rec["n_neg"] == len({(e.query_id, e.product_id) for e in ev})
        if k.noise == 0.0:
            assert rec["n_pos"] == sum(e.product_id == truth[e.query_id] for e in ev)
        # independent recomputation of the reranked R@K from the saved model
        model = v2.load_model(grid.out_root / "r" / "models" / rec["reranker_version"] / "model.json")
        assert model["seed"] == k.seed
        rows = [{x: y for x, y in r.items() if x != "truth_product_id"} for r in grid.val_rows]
        cs = v1.load_cand_stats(grid.paths["val"][1])
        new = v2.rerank_rows(
            model, rows, cs, plan["inputs"]["rankings_sha256"]["val"], grid.vecs["val"].pair, "m"
        )
        by_q = {r["query_id"]: r for r in new}
        joined = [{**by_q[r["query_id"]], "truth_product_id": r["truth_product_id"]} for r in grid.val_rows]
        for kk in (1, 5, 10, 20):
            assert rec["recall"][str(kk)] == pytest.approx(macro(joined, kk), abs=1e-12)
            d = rec["delta"][str(kk)]
            assert d["delta"] == pytest.approx(
                rec["recall"][str(kk)] - plan["baseline"]["macro"][str(kk)], abs=1e-12
            )
            assert d["ci"][0] <= d["ci"][1] and 0.0 <= d["p_le_0"] <= 1.0
        assert rec["training"]["epochs_run"] >= 1 and "fallback_reason" in rec["training"]
        assert rec["nested"] is not None
    assert [r["nested"] for r in recs[:3]] == [True, True, True]  # stratified nests by contract
    assert recs[1]["n_pos"] > 0


def test_plan_json_is_written_first_with_hashes_commit_and_start(grid, monkeypatch):
    seen = {}

    def boom(*a, **k):
        seen["plan_exists"] = (grid.out_root / "r" / "plan.json").is_file()
        seen["runs_exist"] = (grid.out_root / "r" / "runs.jsonl").exists()
        raise KeyboardInterrupt  # not an Exception: the run is not recorded as a failure

    monkeypatch.setattr(curve.rerank_v2, "train", boom)
    with pytest.raises(KeyboardInterrupt):
        run(grid, [K_A])
    assert seen == {"plan_exists": True, "runs_exist": False}
    plan = json.loads((grid.out_root / "r" / "plan.json").read_text(encoding="utf-8"))
    assert len(plan["keys"]) == 108 and plan["requested_key_ids"] == [K_A.id]
    for name in (
        "c4-learning-curve.md",
        "c4-label-selection.md",
        "c4-event-labels.md",
        "c5-reranker-v2.md",
        "c5-reranker-v1.md",
    ):
        want = hashlib.sha256((REPO / "docs" / "contracts" / name).read_bytes()).hexdigest()
        assert plan["contract_sha256"][name] == want
    assert set(plan["contract_sha256"]) == set(curve.CONTRACT_FILES) and len(curve.CONTRACT_FILES) == 5
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    assert plan["git"]["commit"] == head.stdout.strip()
    assert isinstance(plan["git"]["dirty"], bool)
    assert plan["started_at"].endswith("+00:00") and plan["hypothesis_id"] == "H1-rerank-r5"
    assert (
        plan["inputs"]["rankings_sha256"]["val"]
        == hashlib.sha256(grid.paths["val"][0].read_bytes()).hexdigest()
    )


def test_resume_skips_completed_and_keeps_old_lines(grid, monkeypatch):
    run(grid, [K_A, K_B])
    before = (grid.out_root / "r" / "runs.jsonl").read_text(encoding="utf-8")
    n_calls = len(grid.calls)
    monkeypatch.setattr(curve.rerank_v2, "train", lambda *a, **k: pytest.fail("a completed key was re-run"))
    res = run(grid, [K_A, K_B])
    assert (res["attempted"], res["skipped"]) == (0, 2)
    assert (grid.out_root / "r" / "runs.jsonl").read_text(encoding="utf-8") == before
    assert len(grid.calls) == n_calls  # nothing to run: no vectors reloaded


def test_resume_runs_only_the_missing_key(grid):
    run(grid, [K_A])
    res = run(grid, [K_A, K_B])
    assert (res["attempted"], res["skipped"]) == (1, 1)
    assert [r["key_id"] for r in lines(grid)] == [K_A.id, K_B.id]


def test_plan_mismatch_on_resume_is_refused(grid):
    run(grid, [K_A])
    other = grid.out_root / "r" / "plan.json"
    plan = json.loads(other.read_text(encoding="utf-8"))
    plan["val_query_list_sha256"] = "0" * 64
    other.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(CurveError, match="val_query_list_sha256"):
        run(grid, [K_A])


# ---- failures ----------------------------------------------------------------------------------


def test_os_error_is_retried_once_and_the_old_record_is_kept(grid, monkeypatch):
    real = v2.train
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("disk hiccup")
        return real(*a, **k)

    monkeypatch.setattr(curve.rerank_v2, "train", flaky)
    res = run(grid, [K_A, K_B])
    assert (res["completed"], res["failed"]) == (1, 1)
    first = lines(grid)
    assert first[0]["status"] == "failed" and first[0]["retryable"] is True
    assert first[0]["error"].startswith("OSError: disk hiccup") and "recall" not in first[0]
    res = run(grid, [K_A, K_B])  # retry of K_A only
    assert (res["attempted"], res["completed"], res["skipped"]) == (1, 1, 1)
    recs = lines(grid)
    assert [(r["key_id"], r["attempt"], r["status"]) for r in recs] == [
        (K_A.id, 1, "failed"),
        (K_B.id, 1, "completed"),
        (K_A.id, 2, "completed"),
    ]
    assert recs[0] == first[0]  # the original record is untouched
    assert run(grid, [K_A, K_B])["attempted"] == 0


def test_a_second_os_error_is_not_retried_again(grid, monkeypatch):
    def always(*a, **k):
        raise OSError("still broken")

    monkeypatch.setattr(curve.rerank_v2, "train", always)
    run(grid, [K_A])
    run(grid, [K_A])
    assert run(grid, [K_A])["attempted"] == 0
    assert [(r["attempt"], r["status"]) for r in lines(grid)] == [(1, "failed"), (2, "failed")]


def test_deterministic_failure_is_recorded_and_never_retried(grid, monkeypatch):
    def refuse(*a, **k):
        raise RerankError("need both classes to train (pos=0, neg=9)")

    monkeypatch.setattr(curve.rerank_v2, "train", refuse)
    res = run(grid, [K_A, K_B])
    assert (res["completed"], res["failed"]) == (0, 2)
    rec = lines(grid)[0]
    assert rec["status"] == "failed" and rec["retryable"] is False and rec["error_type"] == "RerankError"
    assert "need both classes" in rec["error"] and "recall" not in rec and "delta" not in rec
    assert run(grid, [K_A, K_B])["attempted"] == 0
    assert len(lines(grid)) == 2


def test_failed_run_is_not_averaged_in_the_summary(grid, monkeypatch):
    real = v2.train

    def only_seed_zero(labels, rankings, rsha, cs, csha, lv, fn, index_id, seed, *a):
        if seed == 1:
            raise RerankError("injected")
        return real(labels, rankings, rsha, cs, csha, lv, fn, index_id, seed, *a)

    monkeypatch.setattr(curve.rerank_v2, "train", only_seed_zero)
    k0, k1 = Key("stratified", 0.0, 0, 100), Key("stratified", 0.0, 1, 100)
    run(grid, [k0, k1])
    s = curve.summarize(grid.out_root / "r")
    p100 = s["primary"]["points"][0]
    assert p100["budget"] == 100 and p100["n_seeds_completed"] == 1
    assert (
        p100["mean_label"] != "3-seed 평균" and "1/3" in p100["mean_label"] and not p100["is_full_seed_mean"]
    )
    assert s["counts"] == {"planned": 108, "completed": 1, "failed": 1, "not_run": 106}
    assert s["failures"][0]["key_id"] == k1.id


def test_stratified_nesting_violation_fails_the_key(grid, monkeypatch):
    real = sim.simulate_stratified

    def broken(rows, *, budget, **kw):
        events = real(rows, budget=budget, **kw)
        return events[::-1] if budget < 200 else events  # the small budget is no longer a prefix

    monkeypatch.setattr(curve.sim, "simulate_stratified", broken)
    run(grid, [K_A, K_B])
    a, b = lines(grid)
    assert a["status"] == "failed" and a["error_type"] == "NestingViolation" and a["nested"] is False
    assert a["retryable"] is False and "recall" not in a
    assert b["status"] == "completed" and b["nested"] is True  # the reference budget itself nests trivially


def test_random_non_nesting_is_recorded_not_failed(grid, monkeypatch):
    real = sim.simulate_random
    monkeypatch.setattr(
        curve.sim,
        "simulate_random",
        lambda rows, *, budget, **kw: (
            real(rows, budget=budget, **kw)[::-1] if budget < 200 else real(rows, budget=budget, **kw)
        ),
    )
    run(grid, [K_R])
    (rec,) = lines(grid)
    assert rec["status"] == "completed" and rec["nested"] is False


def test_population_mismatch_fails_the_run(grid, monkeypatch):
    real = v2.rerank_rows
    monkeypatch.setattr(curve.rerank_v2, "rerank_rows", lambda *a, **k: real(*a, **k)[:-1])
    run(grid, [K_A])
    (rec,) = lines(grid)
    assert rec["status"] == "failed" and rec["error_type"] == "PopulationMismatch"
    assert rec["population_match"] is False and "recall" not in rec
    assert (
        rec["population_sha256"]
        != json.loads((grid.out_root / "r" / "plan.json").read_text())["val_query_list_sha256"]
    )


def test_duplicate_reranked_query_also_fails(grid, monkeypatch):
    real = v2.rerank_rows

    def dup(*a, **k):
        out = real(*a, **k)
        return out[:-1] + [out[0]]  # same size, one query twice, one missing

    monkeypatch.setattr(curve.rerank_v2, "rerank_rows", dup)
    run(grid, [K_A])
    assert lines(grid)[0]["error_type"] == "PopulationMismatch"


def test_nan_metric_fails_the_run(grid, monkeypatch):
    monkeypatch.setattr(curve, "macro_metrics", lambda results: {str(k): float("nan") for k in curve.KS})
    run(grid, [K_A])
    rec = lines(grid)[0]
    assert rec["status"] == "failed" and "non-finite" in rec["error"] and "recall" not in rec


# ---- broken inputs refused up front ------------------------------------------------------------


def test_empty_validation_rankings_are_refused(grid):
    grid.paths["val"][0].write_text("", encoding="utf-8")
    grid.paths["val"][1].write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        run(grid, [K_A])
    assert not (grid.out_root / "r" / "runs.jsonl").exists()


def test_cand_stats_of_another_rankings_file_are_refused(grid):
    rp, cp = grid.paths["val"]
    rows = [json.loads(x) for x in cp.read_text().splitlines()]
    rows[0]["rankings_sha256"] = "f" * 64
    _write_jsonl(cp, rows)
    with pytest.raises(RerankError, match="rankings_sha256"):
        run(grid, [K_A])


def test_default_population_check_uses_the_contract_numbers(grid):
    with pytest.raises(CurveError, match="6782"):
        run(grid, [K_A], expected_population=curve.EXPECTED_POPULATION)


def test_unknown_embedder_duplicate_and_oversized_keys_are_refused(grid):
    with pytest.raises(CurveError, match="embedder"):
        curve.run_curve(
            grid.config, *[grid.paths[s][i] for i in (0, 1) for s in ("train", "val")], embedder="x"
        )
    with pytest.raises(CurveError, match="duplicate"):
        run(grid, [K_A, K_A])
    with pytest.raises(CurveError, match="reference budget"):
        run(grid, [Key("stratified", 0.0, 0, 300)])
    with pytest.raises(CurveError, match="unknown policy"):
        run(grid, [Key("uncertainty", 0.0, 0, 100)])


def test_test_split_is_never_loaded():
    loader = curve._default_loader(None, "fake", Path("artifacts"))  # type: ignore[arg-type]
    with pytest.raises(CurveError, match="only train and val"):
        loader("test", ["q"], "idx")


def test_subset_of_train_rows_gives_the_same_model_as_all_rows(grid):
    """Passing only the labelled queries (speed) must not change the trained model."""
    rp, cp = grid.paths["train"]
    sha = hashlib.sha256(rp.read_bytes()).hexdigest()
    csha = hashlib.sha256(cp.read_bytes()).hexdigest()
    events = sim.simulate_stratified(grid.train_rows, budget=100, seed=0, rankings_sha256=sha)
    from product_retrieval.feedback.labels import build_labels

    labels, report = build_labels(events)
    blind = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in grid.train_rows]
    cs = v1.load_cand_stats(cp)
    full = v2.train(
        labels, blind, sha, cs, csha, report["label_version"], grid.vecs["train"].pair, "idx-train", 0, "m"
    )
    qids = {lb.query_id for lb in labels}
    sub = v2.train(
        labels,
        [r for r in blind if r["query_id"] in qids],
        sha,
        [r for r in cs if r["query_id"] in qids],
        csha,
        report["label_version"],
        grid.vecs["train"].pair,
        "idx-train",
        0,
        "m",
    )
    assert len(qids) < len(blind)
    assert full["reranker_version"] == sub["reranker_version"] and v2._same_state(
        full["state_dict"], sub["state_dict"]
    )


# ---- averaging, helpers ------------------------------------------------------------------------


def test_macro_metrics_hand_checked():
    def qr(qid, truth, rank):  # truth at 1-based rank, or None (absent)
        ids = [f"x{qid}-{i}" for i in range(1, 11)]
        if rank is not None:
            ids[rank - 1] = truth
        return QueryResult(qid, truth, tuple(ids))

    results = [
        qr("a1", "A", 1),  # A: hit at 1
        qr("a2", "A", 7),  # A: hit only from rank 7
        qr("b1", "B", 3),  # B: hit from 3
        qr("c1", "C", None),  # C: never
    ]
    m = curve.macro_metrics(results)
    # per product: A = (R@1 .5, R@5 .5, R@10 1, R@20 1), B = (0, 1, 1, 1), C = (0, 0, 0, 0)
    # mean over the 3 products
    assert m == pytest.approx({"1": 1 / 6, "5": 0.5, "10": 2 / 3, "20": 2 / 3}, abs=1e-12)
    # the micro mean would give R@5 = 2/4 = 0.5 only by coincidence, R@1 = 1/4 != 1/6
    assert m["1"] != pytest.approx(0.25)
    # seed mean: equal weight over seeds
    assert curve.seed_mean([0.5, 0.6, 0.7]) == pytest.approx(0.6, abs=1e-15)
    with pytest.raises(CurveError):
        curve.seed_mean([])


def test_ci_comparisons_are_strict_and_separate():
    lows = [-0.001, 0.0, 0.02, 0.03, 0.031]
    assert [curve.ci_lower_above_zero(x) for x in lows] == [False, False, True, True, True]
    assert [curve.ci_lower_above_plus_three(x) for x in lows] == [False, False, False, False, True]


def test_ids_sha256_is_order_free_and_hand_checked():
    want = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert curve.ids_sha256(["c", "a", "b"]) == want == curve.ids_sha256(["b", "c", "a"])
    assert sha256_bytes(b"a\nb\nc") == want


# ---- summarize on hand-written runs ------------------------------------------------------------


def make_run_dir(tmp_path, records, name="rd"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    plan = {
        "run_id": name,
        "hypothesis_id": "H1-rerank-r5",
        "preregistration": "c4-v3",
        "contract": "c4-learning-curve v1",
        "started_at": "2026-10-05T00:00:00+00:00",
        "keys": [k.as_dict() for k in curve.plan_keys()],
        "requested_key_ids": [k.id for k in curve.plan_keys()],
        "contract_sha256": {n: "0" * 64 for n in curve.CONTRACT_FILES},
        "git": {"commit": "c" * 40, "dirty": False, "dirty_files": [], "error": None},
        "inputs": {
            "rankings_sha256": {"train": "1" * 64, "val": "2" * 64},
            "cand_stats_sha256": {"train": "3" * 64, "val": "4" * 64},
            "index_id": {"train": "idx-t", "val": "idx-v"},
            "embedder": "fake",
        },
        "val_query_list_sha256": "5" * 64,
        "val_population": {"n_queries": 6782, "n_products": 3243},
        "baseline": {"macro": {"1": 0.7, "5": BASE_R5, "10": 0.9, "20": 0.937}, "micro": {}},
        "target": 0.887,
        "bootstrap": {"b": 1000, "seed": 0},
    }
    (run_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with open(run_dir / "runs.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return run_dir


def rec(key, r5, *, lo_margin=0.01, attempt=1, status="completed", error=None):
    base = {"key_id": key.id, **key.as_dict(), "attempt": attempt, "status": status, "n_pos": 5, "n_neg": 95}
    if status != "completed":
        return {**base, "error": error or "boom", "error_type": "RerankError", "retryable": False}
    delta = r5 - BASE_R5
    recall = {"1": r5 - 0.1, "5": r5, "10": r5 + 0.02, "20": 0.93}
    d = {str(k): {"delta": recall[str(k)] - 0.5, "ci": [-0.1, 0.1], "p_le_0": 0.5} for k in curve.KS}
    d["5"] = {"delta": delta, "ci": [delta - lo_margin, delta + lo_margin], "p_le_0": 0.01}
    return {**base, "recall": recall, "delta": d, "nested": True, "population_match": True}


def primary_records(seed_r5, failed=(), lo_margin=0.01):
    """seed_r5: {seed: [R@5 per budget]}; failed: set of (seed, budget) written as failed lines."""
    out = []
    for s, vals in seed_r5.items():
        for b, v in zip(curve.BUDGETS, vals, strict=True):
            k = Key("stratified", 0.0, s, b)
            out.append(rec(k, v, lo_margin=lo_margin, status="failed" if (s, b) in failed else "completed"))
    return out


S0 = [0.860, 0.870, 0.887, 0.880, 0.890, 0.900, 0.900, 0.900, 0.900]
S1 = [0.890, 0.880, 0.880, 0.880, 0.880, 0.880, 0.880, 0.880, 0.880]
S2 = [0.860] * 9


def test_first_observed_point_reached_dropped_and_never(tmp_path):
    # seed 0 hits exactly 0.887 at 300 (>= counts), dips at 500 (stays recorded), 700..3000 reach again
    # => first = 300; later points 500,700,1000,1500,2000,3000 = 6, kept (>= 0.887) = 5
    # -> persistence 5/6, dropped [500]
    # seed 1 reaches at the first point => "≤100"; its 8 later points are all < 0.887 => persistence 0/8
    # seed 2 never reaches => "지정 지점 3,000까지 도달 관측 없음"
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: S0, 1: S1, 2: S2})))
    seeds = s["primary"]["seeds"]
    f0, f1, f2 = (seeds[str(i)]["first_observed"] for i in range(3))
    assert (f0["verdict"], f0["budget"], f0["label"]) == ("reached", 300, "300")
    assert f0["persistence"] == {"n_later": 6, "n_kept": 5, "n_later_not_observed": 0, "ratio": 5 / 6}
    assert f0["dropped_after_reach"] == [500]
    assert (f1["verdict"], f1["budget"], f1["label"]) == ("reached", 100, "≤100")
    assert f1["persistence"]["n_later"] == 8 and f1["persistence"]["ratio"] == 0.0
    assert f1["dropped_after_reach"] == [200, 300, 500, 700, 1000, 1500, 2000, 3000]
    assert (f2["verdict"], f2["label"]) == ("not_reached", "지정 지점 3,000까지 도달 관측 없음")
    assert s["primary"]["result"] == {
        "sentence": "일부 도달 관측(2/3)",
        "seed_points": "seed0: 300, seed1: ≤100, seed2: 지정 지점 3,000까지 도달 관측 없음",
        "scope": "모델 선택에 사용한 개발 검증셋의 관측 결과",
    }
    # R@5 values at 100: (0.86 + 0.89 + 0.86) / 3 = 0.87 -> a complete 3-seed mean
    p100 = s["primary"]["points"][0]
    assert p100["mean_r5"] == pytest.approx(0.87, abs=1e-12) and p100["mean_label"] == "3-seed 평균"
    assert p100["mean_delta_r5"] == pytest.approx(0.87 - BASE_R5, abs=1e-12)
    # target Δ is 0.887 - the unrounded baseline, not 0.03
    assert s["baseline"]["target_delta_r5"] == pytest.approx(0.887 - 6 / 7, abs=1e-15)
    assert s["baseline"]["macro"]["5"] == BASE_R5  # unrounded
    assert s["counts"] == {"planned": 108, "completed": 27, "failed": 0, "not_run": 81}


def test_target_comparison_is_inclusive_gate(tmp_path):
    """A value exactly at 0.887 is a reach. Flipping reached_target to a strict > moves seed 0 to 700."""
    assert curve.reached_target(0.887) is True and curve.reached_target(0.8869999999) is False
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: S0, 1: S1, 2: S2})))
    assert s["primary"]["seeds"]["0"]["first_observed"]["budget"] == 300  # fails if >= is turned into >


def test_all_three_seeds_reached_gives_median_sentence(tmp_path):
    def r(at):  # R@5 reaches the target at budget index `at`
        return [0.80] * at + [0.90] * (9 - at)

    # seed 0 at 300 (index 2), seed 1 at 100 (index 0), seed 2 at 700 (index 4) -> median 300
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: r(2), 1: r(0), 2: r(4)})))
    res = s["primary"]["result"]
    assert res["sentence"] == "도달 관측(중앙값 300, 개발 검증셋)"
    assert res["seed_points"] == "seed0: 300, seed1: ≤100, seed2: 700"
    # median 100 is shown as "≤100"
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: r(0), 1: r(0), 2: r(3)}), "rd2"))
    assert s["primary"]["result"]["sentence"] == "도달 관측(중앙값 ≤100, 개발 검증셋)"


def test_all_three_never_reached_is_misreach_free(tmp_path):
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: S2, 1: S2, 2: S2})))
    assert s["primary"]["result"]["sentence"] == "미도달"


def test_earlier_failure_makes_a_later_reach_undetermined(tmp_path):
    # seed 1: R@5 reaches at 300, but the 200 point failed -> not confirmed as the first reach
    seed1 = [0.86, 0.86, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
    recs = primary_records({0: S0, 1: seed1, 2: S2}, failed={(1, 200)})
    s = curve.summarize(make_run_dir(tmp_path, recs))
    f1 = s["primary"]["seeds"]["1"]["first_observed"]
    assert f1["verdict"] == "undetermined" and f1["label"] == "판정 불가" and f1["budget"] is None
    assert f1["unconfirmed_reach_budget"] == 300
    assert s["primary"]["result"]["sentence"] == "일부 도달 관측(1/3)"  # only seed 0 counts
    assert (
        s["primary"]["result"]["seed_points"]
        == "seed0: 300, seed1: 판정 불가, seed2: 지정 지점 3,000까지 도달 관측 없음"
    )
    # the 200 point: seeds 0 and 2 succeeded -> not a 3-seed mean; mean of the two = (0.870 + 0.860) / 2
    p200 = s["primary"]["points"][1]
    assert p200["n_seeds_completed"] == 2 and not p200["is_full_seed_mean"]
    assert p200["mean_r5"] == pytest.approx(0.865, abs=1e-12) and "2/3" in p200["mean_label"]
    assert "3-seed" not in p200["mean_label"]
    assert s["counts"]["failed"] == 1 and s["failures"][0]["key_id"] == "stratified|p0|s1|B200"
    # a failure after the reach does not change the first observed point
    recs = primary_records({0: S0, 1: seed1, 2: S2}, failed={(1, 700)})
    f1 = curve.summarize(make_run_dir(tmp_path, recs, "rd2"))["primary"]["seeds"]["1"]["first_observed"]
    assert (f1["verdict"], f1["budget"]) == ("reached", 300)
    assert (
        f1["persistence"]["n_later_not_observed"] == 1 and f1["persistence"]["n_kept"] == 5
    )  # 6 later points


def test_no_reach_with_a_missing_point_is_not_a_never_reached_claim(tmp_path):
    recs = primary_records({0: S2, 1: S2, 2: S2}, failed={(0, 3000)})
    s = curve.summarize(make_run_dir(tmp_path, recs))
    assert s["primary"]["seeds"]["0"]["first_observed"]["verdict"] == "undetermined"
    assert s["primary"]["result"]["sentence"] == "판정 불가"


def test_nothing_run_everything_is_undetermined_not_unreached(tmp_path):
    s = curve.summarize(make_run_dir(tmp_path, []))
    assert s["counts"] == {"planned": 108, "completed": 0, "failed": 0, "not_run": 108}
    assert s["primary"]["result"]["sentence"] == "판정 불가"
    assert all(p["mean_r5"] is None and p["n_seeds_completed"] == 0 for p in s["primary"]["points"])
    assert "미실행" in (tmp_path / "rd" / "table.md").read_text(encoding="utf-8")


def test_retry_history_final_status_is_the_completed_attempt(tmp_path):
    k = Key("stratified", 0.0, 0, 100)
    s = curve.summarize(make_run_dir(tmp_path, [rec(k, 0.9, status="failed"), rec(k, 0.9, attempt=2)]))
    assert s["counts"]["completed"] == 1 and s["counts"]["failed"] == 0
    assert s["primary"]["seeds"]["0"]["first_observed"]["label"] == "≤100"
    s = curve.summarize(
        make_run_dir(
            tmp_path,
            [
                rec(k, 0.9, status="failed", error="first"),
                rec(k, 0.9, attempt=2, status="failed", error="second"),
            ],
            "rd2",
        )
    )
    assert (
        s["counts"]["failed"] == 1
        and s["failures"][0]["error"] == "second"
        and s["failures"][0]["attempt"] == 2
    )


def test_ci_lower_bound_comparisons_at_the_first_point(tmp_path):
    # seed 0 first at 300: Δ = 0.887 - 6/7 = 0.0298571...; lower = Δ - margin
    # margin 0.01 -> lower 0.01986: above 0 yes, above 0.03 no
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: S0, 1: S1, 2: S2}, lo_margin=0.01)))
    f0 = s["primary"]["seeds"]["0"]["first_observed"]
    assert f0["delta_ci_lower"] == pytest.approx(0.887 - 6 / 7 - 0.01, abs=1e-12)
    assert (f0["delta_ci_lower_above_zero"], f0["delta_ci_lower_above_0_03"]) == (True, False)
    # margin 0.03 -> lower = -0.00014: not above 0, not above 0.03
    s = curve.summarize(make_run_dir(tmp_path, primary_records({0: S0, 1: S1, 2: S2}, lo_margin=0.03), "rd2"))
    f0 = s["primary"]["seeds"]["0"]["first_observed"]
    assert (f0["delta_ci_lower_above_zero"], f0["delta_ci_lower_above_0_03"]) == (False, False)
    # margin -0.005 (a degenerate interval above the point) -> lower 0.0349 > 0.03
    s = curve.summarize(
        make_run_dir(tmp_path, primary_records({0: S0, 1: S1, 2: S2}, lo_margin=-0.005), "rd3")
    )
    f0 = s["primary"]["seeds"]["0"]["first_observed"]
    assert (f0["delta_ci_lower_above_zero"], f0["delta_ci_lower_above_0_03"]) == (True, True)


def test_corrupt_completed_records_are_refused(tmp_path):
    k = Key("stratified", 0.0, 0, 100)
    good = rec(k, 0.9)
    nan = json.loads(json.dumps(good))
    nan["recall"]["5"] = float("nan")
    run_dir = make_run_dir(tmp_path, [])
    (run_dir / "runs.jsonl").write_text(json.dumps(nan) + "\n", encoding="utf-8")  # json writes NaN
    with pytest.raises(CurveError, match="non-finite"):
        curve.summarize(run_dir)
    missing = {x: y for x, y in good.items() if x != "recall"}
    (run_dir / "runs.jsonl").write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(CurveError, match="lacks metrics"):
        curve.summarize(run_dir)
    (run_dir / "runs.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(CurveError, match="unparsable"):
        curve.summarize(run_dir)
    (run_dir / "plan.json").unlink()
    with pytest.raises(CurveError, match="plan.json"):
        curve.summarize(run_dir)


def test_outputs_files_and_table_content(tmp_path):
    k_fail = Key("stratified", 0.0, 2, 3000)
    records = [r for r in primary_records({0: S0, 1: S1, 2: S2}) if r["key_id"] != k_fail.id]
    records.append(rec(k_fail, 0.0, status="failed", error="injected failure"))
    records.append(rec(Key("random", 0.0, 0, 100), 0.86))
    records.append(rec(Key("stratified", 0.1, 0, 100), 0.87))
    run_dir = make_run_dir(tmp_path, records)
    s = curve.summarize(run_dir)
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == s
    png = (run_dir / "curve.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000
    table = (run_dir / "table.md").read_text(encoding="utf-8")
    assert "실패" in table and "injected failure" in table and "미실행" in table
    assert "공유 초기 라벨" in table and "0.8571428571428571" in table  # unrounded baseline
    assert table.count("\n| stratified, p=0 |") == 27 and table.count("\n| random, p=0 |") == 27
    assert s["hypothesis_id"] == "H1-rerank-r5" and s["preregistration"] == "c4-v3"
    assert s["val_query_list_sha256"] == "5" * 64 and s["git_commit"] == "c" * 40
    # primary 27 lines (26 completed + 1 failed) + 2 completed secondary lines = 28 + 1 of 108 -> 79 not run
    assert s["counts"] == {"planned": 108, "completed": 28, "failed": 1, "not_run": 79}
    assert s["secondary"]["random, p=0"]["points"][0]["n_seeds_completed"] == 1


# ---- real vector loader, CLI --------------------------------------------------------------------

SOURCE = "lrvs"


def _put_image(data_root, color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    data = buf.getvalue()
    sha = sha256_bytes(data)
    path = data_root / SOURCE / "images" / sha[:2] / f"{sha}.img"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha


@pytest.fixture
def env(tmp_path):
    data_root = tmp_path / "data"
    rows = []
    for i in range(52):
        split = "val" if i < 12 else "train"
        gallery = [_put_image(data_root, (4 * i, 30 * j, 17)) for j in range(1 + i % 3)]
        query = [_put_image(data_root, (4 * i + 1, 250, i))]
        rows.append(
            {"source": SOURCE, "product_id": f"p{i}", "split": split, "query": query, "gallery": gallery}
        )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    cfg_dict = {
        "name": "t",
        "seed": 0,
        "embed_model_id": "unused-for-fake",
        "crop_kind": "full",
        "index_params": {},
        "manifest_path": str(manifest),
        "data_root": str(data_root),
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    config = ExperimentConfig(**{**cfg_dict, "manifest_path": manifest, "data_root": data_root})
    art, rep = tmp_path / "artifacts", tmp_path / "reports"
    for split in ("val", "train"):
        run_build_index(config, split=split, embedder_name="fake", artifacts_root=art, reports_root=rep)
    ev_val = run_eval(config, split="val", embedder_name="fake", artifacts_root=art, reports_root=rep)
    cs_val = run_cand_stats(config, "val", ev_val.rankings_path, "fake", art)
    ev_tr = run_eval(config, split="train", embedder_name="fake", artifacts_root=art, reports_root=rep)
    # The fake embedder ranks at random. Make the simulator's oracle say "rank 1 is the truth" for the train
    # split so that positives exist; truth is not read by cand-stats, so the stats stay valid.
    tr_rows = [json.loads(x) for x in ev_tr.rankings_path.read_text().splitlines()]
    for r in tr_rows:
        r["truth_product_id"] = r["top_k_product_ids"][0]
    tr_path = tmp_path / "train_mod.rankings.jsonl"
    _write_jsonl(tr_path, tr_rows)
    cs_tr = run_cand_stats(config, "train", tr_path, "fake", art)
    return SimpleNamespace(
        config=config,
        cfg=cfg,
        art=art,
        val=(ev_val.rankings_path, cs_val.out_path),
        train=(tr_path, cs_tr.out_path),
        tmp=tmp_path,
    )


def test_run_curve_with_the_real_vector_loader(env):
    key = Key("stratified", 0.0, 0, 100)
    res = curve.run_curve(
        env.config,
        env.train[0],
        env.val[0],
        env.train[1],
        env.val[1],
        out_root=env.tmp / "curve",
        embedder="fake",
        artifacts_root=env.art,
        run_id="e",
        keys=[key],
        reference_budget=100,
        expected_population=None,
    )
    assert (res["completed"], res["failed"]) == (1, 0), (env.tmp / "curve" / "e" / "runs.jsonl").read_text()
    (rec_,) = [json.loads(x) for x in (env.tmp / "curve" / "e" / "runs.jsonl").read_text().splitlines()]
    assert rec_["n_pos"] > 0 and rec_["population_match"] is True and rec_["nested"] is True
    plan = json.loads((env.tmp / "curve" / "e" / "plan.json").read_text())
    assert plan["val_population"]["n_queries"] == 12


def test_cli_curve_and_summary(env):
    runner = CliRunner()
    args = [
        "curve",
        "--config",
        str(env.cfg),
        "--train-rankings",
        str(env.train[0]),
        "--val-rankings",
        str(env.val[0]),
    ]
    args += ["--train-cand-stats", str(env.train[1]), "--val-cand-stats", str(env.val[1])]
    args += [
        "--embedder",
        "fake",
        "--artifacts-root",
        str(env.art),
        "--out-root",
        str(env.tmp / "c"),
        "--run-id",
        "x",
    ]
    # the default 3,000-label nesting reference exceeds this tiny train split:
    # recorded as a failed key, exit code 3
    res = runner.invoke(app, [*args, "--only-keys", "1"])
    assert res.exit_code == 3, res.output
    (rec_,) = [json.loads(x) for x in (env.tmp / "c" / "x" / "runs.jsonl").read_text().splitlines()]
    assert rec_["status"] == "failed" and "capacity" in rec_["error"]
    # without --only-keys the contract population (6,782 / 3,243) is enforced: refused, exit code 1
    res = runner.invoke(app, [*args[:-2], "--run-id", "y"])
    assert res.exit_code == 1 and "6782" in res.output
    res = runner.invoke(app, ["curve-summary", "--run-dir", str(env.tmp / "c" / "x")])
    assert res.exit_code == 0, res.output
    assert "failed 1" in res.output and "판정 불가" in res.output
    for name in ("summary.json", "table.md", "curve.png"):
        assert (env.tmp / "c" / "x" / name).is_file()
    assert runner.invoke(app, ["curve", *args[1:], "--only-keys", "0"]).exit_code == 1
    assert runner.invoke(app, ["curve", "--help"]).exit_code == 0
    assert "--split" not in runner.invoke(app, ["curve", "--help"]).output  # no way to name the test split
    bad = runner.invoke(app, ["curve-summary", "--run-dir", str(env.tmp)])
    assert bad.exit_code == 1 and "plan.json" in bad.output
