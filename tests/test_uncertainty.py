"""Tests for the uncertainty label selection (docs/contracts/c4-uncertainty.md).

Selector tests use hand-worked examples and an independent brute-force implementation. The path runner and the
comparator run end to end on the small synthetic fixtures of test_curve (fake vectors, no real embedder).
"""

import copy
import inspect
import json
import random
from collections import Counter

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.gate import MANIFEST_SHA256, manifest_sha256
from product_retrieval.feedback import simulate as sim
from product_retrieval.feedback import uncertainty as unc
from product_retrieval.feedback.labels import build_labels, load_events
from product_retrieval.feedback.uncertainty import SelectionError, select_uncertain
from product_retrieval.pipelines import curve, h2_compare, uncertainty_path
from product_retrieval.pipelines.curve import Key
from product_retrieval.rerank import v1, v2
from test_curve import FakeVectors, env, grid  # noqa: F401 - fixtures

# ---- selector: hand-worked examples ----------------------------------------------------------------


def pool_of(*qids, depth=20):
    return [{"query_id": q, "top_k_product_ids": [f"{q}-{k}" for k in range(depth)]} for q in qids]


def flat(p=0.99, depth=20):
    return [p] * depth


def probs_with(pool, overrides, base=0.99):
    """probabilities with base everywhere and {(query_id, position): p} overrides."""
    out = {r["query_id"]: flat(base) for r in pool}
    for (q, pos), p in overrides.items():
        out[q][pos - 1] = p
    return out


def test_order_is_by_u_then_query_id_then_position_hand_example():
    pool = pool_of("a", "b", "c")
    # u = |p - 0.5|: (c,5)=0.0 (c,2)=0.0625 (a,7)=0.0625 (b,1)=0.25 (b,9)=0.25 (a,3)=0.25 everything else 0.49
    probs = probs_with(
        pool,
        {("c", 5): 0.5, ("c", 2): 0.4375, ("a", 7): 0.5625, ("b", 1): 0.75, ("b", 9): 0.25, ("a", 3): 0.25},
    )
    got = select_uncertain(pool, probs, {}, 6)
    assert [(s.query_id, s.position) for s in got] == [
        ("c", 5),  # u 0
        ("a", 7),  # u 0.0625 ties with (c, 2): query a before c
        ("c", 2),
        ("a", 3),  # u 0.25 ties among a, b: query a first
        ("b", 1),  # then b by position 1 before 9
        ("b", 9),
    ]
    assert got[0].product_id == "c-4" and got[0].position == 5  # product = top_k_product_ids[position - 1]


def test_tie_between_p_below_and_above_one_half_uses_query_then_position():
    pool = pool_of("b", "a")
    probs = probs_with(pool, {("b", 4): 0.25, ("a", 9): 0.75, ("a", 2): 0.75})
    got = select_uncertain(pool, probs, {}, 3)
    assert [(s.query_id, s.position) for s in got] == [("a", 2), ("a", 9), ("b", 4)]


def test_cap_counts_labelled_pairs_and_pairs_taken_this_round():
    pool = pool_of("a", "b", "c")
    # query a: five most uncertain pairs; b has 2 judgements in L; c has 3 in L (already full)
    probs = probs_with(pool, {("a", k): 0.5 + 0.001 * k for k in range(1, 6)})
    probs["b"][0] = 0.6  # u 0.1, the next most uncertain
    probs["c"][0] = 0.5  # c is full: skipped although u = 0 is the minimum
    labelled = {("b", "b-10"): "not_match", ("b", "b-11"): "match"}
    labelled |= {("c", "c-10"): "not_match", ("c", "c-11"): "not_match", ("c", "c-12"): "not_match"}
    # walk: c (u 0) is full, skipped; a1, a2, a3 taken, a4 and a5 skipped by the round cap; b1 takes b's last
    # slot (2 in L + 1); everything left has u 0.49 but belongs to a (3 this round), b (3) or c (3 in L)
    got = select_uncertain(pool, probs, labelled, 4)
    assert [(s.query_id, s.position) for s in got] == [("a", 1), ("a", 2), ("a", 3), ("b", 1)]
    with pytest.raises(SelectionError, match="only 4 pairs"):
        select_uncertain(pool, probs, labelled, 5)


def test_labelled_pairs_are_skipped_even_when_most_uncertain():
    pool = pool_of("a", "b")
    probs = probs_with(pool, {("a", 1): 0.5, ("b", 2): 0.51})
    got = select_uncertain(pool, probs, {("a", "a-0"): "match"}, 1)
    assert [(s.query_id, s.position) for s in got] == [("b", 2)]


def test_cap_does_not_reopen_for_a_later_pair_of_the_same_query():
    pool = pool_of("a", "b")
    probs = probs_with(pool, {("a", 1): 0.5, ("a", 2): 0.5, ("a", 3): 0.5, ("a", 4): 0.5, ("b", 1): 0.6})
    got = select_uncertain(pool, probs, {}, 4)
    assert [(s.query_id, s.position) for s in got] == [("a", 1), ("a", 2), ("a", 3), ("b", 1)]


def test_zero_take_and_exhausted_pool():
    pool = pool_of("a")
    assert select_uncertain(pool, probs_with(pool, {}), {}, 0) == []
    with pytest.raises(SelectionError, match="only 3 pairs"):  # cap 3 x one query
        select_uncertain(pool, probs_with(pool, {}), {}, 4)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda pool, probs, lab: probs["a"].__setitem__(3, float("nan")), "NaN"),
        (lambda pool, probs, lab: probs["a"].__setitem__(3, 1.5), r"\[0, 1\]"),
        (lambda pool, probs, lab: probs.pop("a"), "differ in query ids"),
        (lambda pool, probs, lab: probs.__setitem__("a", [0.5] * 19), "expected 20"),
        (lambda pool, probs, lab: pool.append(dict(pool[0])), "duplicate query_id"),
        (lambda pool, probs, lab: pool[0].__setitem__("top_k_product_ids", ["x"] * 20), "more than once"),
        (lambda pool, probs, lab: pool[0].__setitem__("top_k_product_ids", ["x", "y"]), "needs 20"),
        (lambda pool, probs, lab: lab.__setitem__(("a", "elsewhere"), "match"), "not in the pool"),
        (
            lambda pool, probs, lab: lab.update({("a", f"a-{k}"): "match" for k in range(4)}),
            "more than 3 judgements",
        ),
    ],
)
def test_broken_inputs_are_errors(mutate, message):
    pool = pool_of("a", "b")
    probs = probs_with(pool, {})
    lab: dict = {}
    mutate(pool, probs, lab)
    with pytest.raises(SelectionError, match=message):
        select_uncertain(pool, probs, lab, 1)


def test_bad_arguments():
    pool = pool_of("a")
    for bad in (-1, 1.5, True):
        with pytest.raises(SelectionError):
            select_uncertain(pool, probs_with(pool, {}), {}, bad)
    with pytest.raises(SelectionError):
        select_uncertain(pool, probs_with(pool, {}), {}, 1, cap=0)
    with pytest.raises(SelectionError, match="empty"):
        select_uncertain([], {}, {}, 0)


def test_sigmoid_hand_values_and_stability():
    assert unc.sigmoid(np.array([0.0]))[0] == 0.5
    assert unc.sigmoid(np.array([np.log(3.0)]))[0] == pytest.approx(0.75, abs=1e-15)
    big = unc.sigmoid(np.array([-1000.0, 1000.0]))
    assert big[0] == 0.0 and big[1] == 1.0


# ---- selector: independent brute-force implementation -------------------------------------------------


def brute(pool, probs, labelled, n, cap=3):
    cands = []
    for row in pool:
        for pos, pid in enumerate(row["top_k_product_ids"][:20], start=1):
            if (row["query_id"], pid) not in labelled:
                cands.append((abs(probs[row["query_id"]][pos - 1] - 0.5), row["query_id"], pos, pid))
    cands.sort()
    count = Counter(q for q, _ in labelled)
    out = []
    for _, q, pos, pid in cands:
        if count[q] >= cap:
            continue
        count[q] += 1
        out.append((q, pid, pos))
        if len(out) == n:
            break
    return out


@pytest.mark.parametrize("seed", range(6))
def test_selector_equals_brute_force_with_many_ties(seed):
    rng = random.Random(seed)
    qids = [f"q{rng.randrange(10**6):06d}" for _ in range(30)]
    pool = pool_of(*sorted(set(qids)))
    # coarse probability grid: many exact ties in u
    probs = {
        r["query_id"]: [rng.choice([0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]) for _ in range(20)] for r in pool
    }
    labelled = {}
    for r in pool:
        for k in rng.sample(range(20), rng.choice([0, 0, 1, 2, 3])):
            labelled[(r["query_id"], r["top_k_product_ids"][k])] = "not_match"
    capacity = 3 * len(pool) - len(labelled)
    n = min(rng.choice([1, 7, 40, 100]), capacity)
    got = [tuple(s) for s in select_uncertain(pool, probs, labelled, n)]
    assert got == brute(pool, probs, labelled, n)
    per_query = Counter(q for q, _, _ in got) + Counter(q for q, _ in labelled)
    assert max(per_query.values()) <= 3
    assert not {(q, p) for q, p, _ in got} & set(labelled)


def test_selector_reads_only_query_id_and_candidates():
    pool = pool_of("a", "b")
    probs = probs_with(pool, {("a", 3): 0.5, ("b", 8): 0.52})
    base = select_uncertain(pool, probs, {}, 4)
    for mutate in (
        lambda r: r.__setitem__("truth_product_id", "something-else"),
        lambda r: r.__setitem__("truth_product_id", r["top_k_product_ids"][0]),
        lambda r: r.__setitem__("scores", [9.0] * 20),
        lambda r: r.pop("truth_product_id", None),
    ):
        rows = [{**r, "truth_product_id": r["top_k_product_ids"][5]} for r in pool_of("a", "b")]
        for r in rows:
            mutate(r)
        assert select_uncertain(rows, probs, {}, 4) == base


def test_selector_signature_has_no_simulator_truth_or_manifest():
    for fn in (select_uncertain, uncertainty_path.select_next_batch):
        names = set(inspect.signature(fn).parameters)
        assert not {n for n in names if any(w in n for w in ("oracle", "truth", "manifest", "sim"))}


# ---- path runner: fixtures ----------------------------------------------------------------------------

POINTS = (100, 130, 160)


@pytest.fixture
def frozen(grid):  # noqa: F811
    truth = {r["query_id"]: r["truth_product_id"] for r in grid.val_rows}
    return manifest_sha256(curve.to_results(grid.val_rows, truth))


def u_run(g, manifest, run_id="u", seeds=(0,), points=POINTS, **kw):
    kw.setdefault("bootstrap_b", 50)
    return uncertainty_path.run_uncertainty_path(
        g.config,
        g.paths["train"][0],
        g.paths["val"][0],
        g.paths["train"][1],
        g.paths["val"][1],
        g.out_root,
        run_id,
        seeds,
        "fake",
        vector_loader=g.loader,
        points=points,
        manifest_sha=manifest,
        **kw,
    )


def u_lines(g, run_id="u", name="runs.jsonl"):
    path = g.out_root / run_id / name
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def pairs_of(events):
    return [(e.query_id, e.product_id, e.position) for e in events]


def test_path_end_to_end_matches_independent_selection(grid, frozen):  # noqa: F811
    res = u_run(grid, frozen)
    assert (res["attempted"], res["completed"], res["failed"]) == (3, 3, 0)
    events = load_events(grid.out_root / "u" / "events" / "uncertainty.seed0.jsonl")
    assert len(events) == 160
    assert {e.policy_id for e in events} == {"uncertainty"} and {e.actor for e in events} == {"sim"}
    assert len({(e.query_id, e.product_id) for e in events}) == 160  # no pair judged twice
    assert max(Counter(e.query_id for e in events).values()) <= 3
    assert [e.ts for e in events] == sorted(e.ts for e in events)
    # I_s = the first 100 stratified pairs and answers
    strat = sim.simulate_stratified(
        grid.train_rows, budget=100, seed=0, rankings_sha256=curve.sha256_file(grid.paths["train"][0])
    )
    assert [(e.query_id, e.product_id, e.position, e.action) for e in events[:100]] == [
        (e.query_id, e.product_id, e.position, e.action) for e in strat
    ]
    # answers are the oracle's (noise 0): match iff the product is the truth
    truth = {r["query_id"]: r["truth_product_id"] for r in grid.train_rows}
    assert all((e.action == "match") == (e.product_id == truth[e.query_id]) for e in events)

    # independent re-selection of each batch from the saved model: manual forward pass, brute-force walk
    records = u_lines(grid)
    assert [(r["budget"], r["round"], r["status"]) for r in records] == [
        (100, 0, "completed"),
        (130, 1, "completed"),
        (160, 2, "completed"),
    ]
    vec = grid.vecs["train"]
    cs = {r["query_id"]: r for r in map(json.loads, grid.paths["train"][1].read_text().splitlines())}
    rows = {r["query_id"]: r for r in grid.train_rows}
    for j, (size, nxt) in enumerate([(100, 130), (130, 160)]):
        model = v2.load_model(grid.out_root / "u" / "models" / records[j]["reranker_version"] / "model.json")
        net = v2.build_net(model["input_dim"])
        net.load_state_dict(model["state_dict"])
        net.eval()
        mean, scale = np.asarray(model["scaler_mean"]), np.asarray(model["scaler_scale"])
        probs = {}
        for qid, row in rows.items():
            feats = (v1.features_v1(row, cs[qid]["stats"]) - mean) / scale
            x = np.vstack(
                [v2.input_vector(*vec.pair(qid, row["top_k_product_ids"][i]), feats[i]) for i in range(20)]
            )
            with torch.no_grad():
                z = net(torch.as_tensor(x, dtype=torch.float32)).squeeze(1).double().numpy()
            probs[qid] = list(1.0 / (1.0 + np.exp(-z)))
        blind = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in grid.train_rows]
        labelled = {(e.query_id, e.product_id) for e in events[:size]}
        assert pairs_of(events[size:nxt]) == brute(blind, probs, labelled, nxt - size)

    # records: curve-like fields, delta_vs_baseline, own hypothesis, manifest
    r0 = records[0]
    for field in (
        "recall",
        "delta_vs_baseline",
        "n_pos",
        "n_neg",
        "policy",
        "round",
        "seconds",
        "reranker_version",
    ):
        assert field in r0
    assert "delta" not in r0 and r0["policy"] == "uncertainty" and r0["hypothesis_id"] == "H2-uncertainty"
    assert r0["manifest_sha256"] == frozen and r0["n_pos"] + r0["n_neg"] == 100
    assert [r["n_pos"] + r["n_neg"] for r in records] == [100, 130, 160]
    rounds = u_lines(grid, name="rounds.jsonl")
    assert [(x["from_size"], x["to_size"], x["n_selected"]) for x in rounds] == [
        (100, 130, 30),
        (130, 160, 30),
    ]
    plan = json.loads((grid.out_root / "u" / "plan.json").read_text())
    assert plan["hypothesis_id"] == "H2-uncertainty" and plan["manifest_sha256"] == frozen
    assert len(plan["keys"]) == 3 and "c4-uncertainty.md" in plan["contract_sha256"]
    # every model is saved and carries the label policy
    for r in records:
        saved = json.loads(
            (grid.out_root / "u" / "models" / r["reranker_version"] / "model.json").read_text()
        )
        assert saved["label_policy"] == "uncertainty"
    # the recorded metrics reproduce from the saved model
    model = v2.load_model(grid.out_root / "u" / "models" / records[2]["reranker_version"] / "model.json")
    vblind = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in grid.val_rows]
    vcs = v1.load_cand_stats(grid.paths["val"][1])
    reranked = v2.rerank_rows(
        model, vblind, vcs, curve.sha256_file(grid.paths["val"][0]), grid.vecs["val"].pair, "m"
    )
    vt = {r["query_id"]: r["truth_product_id"] for r in grid.val_rows}
    r5 = curve.macro_metrics(curve.to_results(reranked, vt))["5"]
    assert r5 == records[2]["recall"]["5"]


def test_initial_model_equals_the_stratified_b100_model(grid, frozen):  # noqa: F811
    u_run(grid, frozen, points=(100, 130))
    curve.run_curve(
        grid.config,
        grid.paths["train"][0],
        grid.paths["val"][0],
        grid.paths["train"][1],
        grid.paths["val"][1],
        out_root=grid.out_root,
        embedder="fake",
        run_id="s",
        keys=[Key("stratified", 0.0, 0, 100)],
        vector_loader=grid.loader,
        reference_budget=200,
        expected_population=None,
    )
    (u0, _) = u_lines(grid)
    (s0,) = u_lines(grid, "s")
    mu = v2.load_model(grid.out_root / "u" / "models" / u0["reranker_version"] / "model.json")
    ms = v2.load_model(grid.out_root / "s" / "models" / s0["reranker_version"] / "model.json")
    assert list(mu["state_dict"]) == list(ms["state_dict"])
    assert all(torch.equal(mu["state_dict"][k], ms["state_dict"][k]) for k in mu["state_dict"])
    assert u0["recall"] == s0["recall"] and (u0["n_pos"], u0["n_neg"]) == (s0["n_pos"], s0["n_neg"])
    assert mu["scaler_mean"] == ms["scaler_mean"] and mu["training"] == ms["training"]
    # the event ids differ (policy_id is part of them), so the model id differs, not the weights
    assert u0["reranker_version"] != s0["reranker_version"]


def test_one_round_selection_is_blind_to_truth_fields(grid, frozen):  # noqa: F811
    u_run(grid, frozen, points=(100, 130))
    events = load_events(grid.out_root / "u" / "events" / "uncertainty.seed0.jsonl")
    (r0, _) = u_lines(grid)
    model = v2.load_model(grid.out_root / "u" / "models" / r0["reranker_version"] / "model.json")
    cs = {r["query_id"]: r for r in map(json.loads, grid.paths["train"][1].read_text().splitlines())}
    sha = curve.sha256_file(grid.paths["train"][0])
    labelled = {(e.query_id, e.product_id): e.action for e in events[:100]}

    def select(rows):
        return uncertainty_path.select_next_batch(
            model, rows, cs, sha, grid.vecs["train"].pair, "m", labelled, 30
        )

    sorted_rows = sorted(grid.train_rows, key=lambda r: r["query_id"])
    expected = pairs_of(events[100:130])
    assert [tuple(s) for s in select(copy.deepcopy(sorted_rows))] == expected
    wrong = copy.deepcopy(sorted_rows)
    for r in wrong:
        r["truth_product_id"] = r["top_k_product_ids"][0]
    gone = copy.deepcopy(sorted_rows)
    for r in gone:
        del r["truth_product_id"]
    other = copy.deepcopy(sorted_rows)
    for i, r in enumerate(other):
        r["truth_product_id"] = f"nobody-{i}"
    for rows in (wrong, gone, other):
        assert [tuple(s) for s in select(rows)] == expected


def test_path_resume_does_nothing_and_changes_nothing(grid, frozen):  # noqa: F811
    u_run(grid, frozen)
    before = {p: p.read_bytes() for p in (grid.out_root / "u").rglob("*") if p.is_file()}
    res = u_run(grid, frozen)
    assert res["attempted"] == 0
    after = {p: p.read_bytes() for p in (grid.out_root / "u").rglob("*") if p.is_file()}
    assert before == after


def test_path_resumes_after_a_crash_between_point_and_selection(grid, frozen, monkeypatch):  # noqa: F811
    full = u_run(grid, frozen, run_id="whole")
    assert full["completed"] == 3
    calls = {"n": 0}
    real = uncertainty_path.select_next_batch

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # not an Exception: the run dies like a Ctrl-C
        return real(*a, **k)

    monkeypatch.setattr(uncertainty_path, "select_next_batch", boom)
    with pytest.raises(KeyboardInterrupt):
        u_run(grid, frozen, run_id="part")
    monkeypatch.setattr(uncertainty_path, "select_next_batch", real)
    ev = grid.out_root / "part" / "events" / "uncertainty.seed0.jsonl"
    assert len(ev.read_text().splitlines()) == 130
    res = u_run(grid, frozen, run_id="part")
    assert res["completed"] == 1  # only point 160 is new; 130 is loaded from its saved model
    whole = (grid.out_root / "whole" / "events" / "uncertainty.seed0.jsonl").read_bytes()
    assert ev.read_bytes() == whole
    a = [(r["budget"], r["reranker_version"], r["recall"]) for r in u_lines(grid, "part")]
    b = [(r["budget"], r["reranker_version"], r["recall"]) for r in u_lines(grid, "whole")]
    assert a == b


def test_os_error_is_retried_once_and_the_failed_record_is_kept(grid, frozen, monkeypatch):  # noqa: F811
    real = v2.train
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk hiccup")
        return real(*a, **k)

    monkeypatch.setattr(v2, "train", flaky)
    res = u_run(grid, frozen, points=(100, 130))
    recs = u_lines(grid)
    assert [(r["budget"], r["attempt"], r["status"]) for r in recs] == [
        (100, 1, "failed"),
        (100, 2, "completed"),
        (130, 1, "completed"),
    ]
    assert recs[0]["retryable"] is True and "disk hiccup" in recs[0]["error"]
    assert (res["attempted"], res["failed"]) == (3, 1)


def test_second_os_error_is_not_retried_and_the_path_stops(grid, frozen, monkeypatch):  # noqa: F811
    monkeypatch.setattr(v2, "train", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    res = u_run(grid, frozen)
    recs = u_lines(grid)
    assert [(r["attempt"], r["status"]) for r in recs] == [(1, "failed"), (2, "failed")]
    assert res["completed"] == 0 and len(list(grid.out_root.glob("u/models/*"))) == 0
    assert (
        len(u_lines(grid, name="rounds.jsonl") if (grid.out_root / "u" / "rounds.jsonl").exists() else [])
        == 0
    )


def test_deterministic_failure_is_kept_not_retried_and_later_points_do_not_run(grid, frozen, monkeypatch):  # noqa: F811
    real = v2.train
    calls = {"n": 0}

    def fail_second(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad labels")
        return real(*a, **k)

    monkeypatch.setattr(v2, "train", fail_second)
    res = u_run(grid, frozen)
    recs = u_lines(grid)
    assert [(r["budget"], r["status"]) for r in recs] == [(100, "completed"), (130, "failed")]
    assert recs[1]["retryable"] is False and recs[1]["error_type"] == "ValueError"
    monkeypatch.setattr(v2, "train", real)
    assert u_run(grid, frozen)["attempted"] == 0  # a failed key is not run again
    assert [(r["budget"], r["status"]) for r in u_lines(grid)] == [(100, "completed"), (130, "failed")]
    assert res["failed"] == 1


def test_population_not_matching_the_frozen_manifest_stops_before_any_work(grid, frozen):  # noqa: F811
    with pytest.raises(curve.PopulationMismatch, match="manifest sha256"):
        u_run(grid, "0" * 64)
    assert not (grid.out_root / "u").exists()
    with pytest.raises(curve.PopulationMismatch, match="manifest sha256"):
        u_run(grid, MANIFEST_SHA256)  # the real frozen hash does not match the small fixture


def test_reranked_population_with_a_missing_query_fails_the_point(grid, frozen, monkeypatch):  # noqa: F811
    real = v2.rerank_rows

    def drop(model, rows, *a, **k):
        out = real(model, rows, *a, **k)
        return out[:-1] if len(rows) == len(grid.val_rows) else out

    monkeypatch.setattr(v2, "rerank_rows", drop)
    u_run(grid, frozen, points=(100,))
    (rec,) = u_lines(grid)
    assert rec["status"] == "failed" and rec["error_type"] == "PopulationMismatch"


def test_invalid_points_seeds_and_run_id(grid, frozen):  # noqa: F811
    for bad in ((130, 160), (100, 100), (100, 90), ()):
        with pytest.raises(uncertainty_path.UncertaintyError):
            u_run(grid, frozen, points=bad)
    with pytest.raises(uncertainty_path.UncertaintyError):
        u_run(grid, frozen, seeds=(0, 0))
    with pytest.raises(uncertainty_path.UncertaintyError):
        u_run(grid, frozen, run_id="a/b")
    with pytest.raises(uncertainty_path.UncertaintyError, match="exceed 3 per query"):
        u_run(grid, frozen, points=(100, 500))


def test_resume_refuses_a_changed_plan_and_a_foreign_events_file(grid, frozen):  # noqa: F811
    u_run(grid, frozen, points=(100, 130))
    with pytest.raises(uncertainty_path.UncertaintyError, match="different 'keys'"):
        u_run(grid, frozen, points=(100, 130, 160))
    ev = grid.out_root / "u" / "events" / "uncertainty.seed0.jsonl"
    lines = ev.read_text().splitlines()
    ev.write_text("\n".join(lines[:110]) + "\n")  # 110 events is not a measurement point
    with pytest.raises(uncertainty_path.UncertaintyError, match="not a measurement point"):
        u_run(grid, frozen, points=(100, 130))


# ---- comparator --------------------------------------------------------------------------------------


@pytest.fixture
def both(grid, frozen):  # noqa: F811
    """A curve-style stratified run and an uncertainty run for seed 0 at the same points."""
    keys = [Key("stratified", 0.0, 0, b) for b in POINTS]
    curve.run_curve(
        grid.config,
        grid.paths["train"][0],
        grid.paths["val"][0],
        grid.paths["train"][1],
        grid.paths["val"][1],
        out_root=grid.out_root,
        embedder="fake",
        run_id="s",
        keys=keys,
        vector_loader=grid.loader,
        reference_budget=160,
        expected_population=None,
        bootstrap_b=50,
    )
    u_run(grid, frozen)
    srecs = {r["budget"]: r for r in u_lines(grid, "s")}
    pins = {0: (srecs[160]["reranker_version"], srecs[160]["recall"]["5"])}
    return pins, srecs


def compare(grid, frozen, pins, **kw):  # noqa: F811
    kw.setdefault("bootstrap_b", 50)
    return h2_compare.compare_h2(
        grid.out_root / "u",
        grid.out_root / "s",
        grid.config,
        grid.paths["val"][0],
        grid.paths["val"][1],
        embedder="fake",
        vector_loader=grid.loader,
        points=POINTS,
        seeds=(0,),
        pins=pins,
        manifest_sha=frozen,
        endpoint=160,
        **kw,
    )


def test_compare_records_and_files(grid, frozen, both):  # noqa: F811
    pins, srecs = both
    res = compare(grid, frozen, pins)
    comps = res["h2_comparison"]
    assert [c["budget"] for c in comps] == list(POINTS) and all(c["status"] == "completed" for c in comps)
    urecs = {r["budget"]: r for r in u_lines(grid)}
    for c in comps:
        b = c["budget"]
        assert c["stratified_model_id"] == srecs[b]["reranker_version"]
        assert c["uncertainty_model_id"] == urecs[b]["reranker_version"]
        assert c["population_manifest_sha256"] == frozen and c["bootstrap"] == {"b": 50, "seed": 0}
        for k, field in (("5", "recall"), ("1", "recall")):
            want = urecs[b][field][k] - srecs[b][field][k]
            assert c["delta"][k]["delta"] == pytest.approx(want, abs=1e-12)  # independent: from the records
            lo, hi = c["delta"][k]["ci"]
            assert lo <= hi and 0.0 <= c["delta"][k]["p_le_0"] <= 1.0
        assert len(c["regenerated_rankings_sha256"]["uncertainty"]) == 64
    # the shared start: same weights, same rankings, so the difference is exactly zero
    first = comps[0]
    assert first["delta"]["5"] == {"delta": 0.0, "ci": [0.0, 0.0], "p_le_0": 1.0}
    assert first["delta"]["1"]["delta"] == 0.0
    assert res["verdict"]["sentence"].startswith(("uncertainty selection", "no consistent difference"))
    out = grid.out_root / "u"
    assert json.loads((out / "h2_comparison.json").read_text())["verdict"] == res["verdict"]
    md = (out / "h2_comparison.md").read_text()
    assert res["verdict"]["sentence"] in md and "| seed | labels |" in md


def test_compare_delta_uses_the_paired_bootstrap_with_the_stratified_side_as_a(grid, frozen, both):  # noqa: F811
    pins, _ = both
    res = compare(grid, frozen, pins)
    c = res["h2_comparison"][2]
    # rebuild the two rankings with the pinned code path and call the bootstrap directly
    vblind = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in grid.val_rows]
    vcs = v1.load_cand_stats(grid.paths["val"][1])
    sha = curve.sha256_file(grid.paths["val"][0])
    vt = {r["query_id"]: r["truth_product_id"] for r in grid.val_rows}

    def results(run, version):
        m = v2.load_model(grid.out_root / run / "models" / version / "model.json")
        return curve.to_results(v2.rerank_rows(m, vblind, vcs, sha, grid.vecs["val"].pair, "m"), vt)

    p = paired_bootstrap(
        results("s", c["stratified_model_id"]), results("u", c["uncertainty_model_id"]), (1, 5), b=50, seed=0
    )
    assert c["delta"]["5"] == {
        "delta": p["macro"][5]["delta"],
        "ci": list(p["macro"][5]["ci"]),
        "p_le_0": p["macro"][5]["p_le_0"],
    }


def rewrite(path, fn):
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for rec in lines:
        fn(rec)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))


def test_recorded_r5_mismatch_aborts_on_either_side_and_at_any_point(grid, frozen, both):  # noqa: F811
    pins, _ = both
    compare(grid, frozen, pins)  # positive control
    for run, budget in (("u", 130), ("s", 130), ("u", 160)):
        path = grid.out_root / run / "runs.jsonl"
        original = path.read_text()

        def bump(rec, budget=budget):
            if rec["budget"] == budget:
                rec["recall"]["5"] += 1e-6

        rewrite(path, bump)
        # the pinned value check also needs the same change for the curve side at the endpoint
        with pytest.raises(h2_compare.CompareAbort):
            compare(grid, frozen, pins)
        path.write_text(original)
    # a change below the tolerance is accepted
    path = grid.out_root / "u" / "runs.jsonl"
    original = path.read_text()
    rewrite(path, lambda rec: rec["recall"].__setitem__("5", rec["recall"]["5"] + 1e-11))
    compare(grid, frozen, pins)
    path.write_text(original)


def test_pinned_endpoint_value_and_model_id_are_enforced(grid, frozen, both):  # noqa: F811
    ((seed, (ver, r5)),) = both[0].items()
    with pytest.raises(h2_compare.CompareAbort, match="pinned"):
        compare(grid, frozen, {seed: (ver, r5 + 1e-6)})
    with pytest.raises(h2_compare.CompareAbort, match="pinned"):
        compare(grid, frozen, {seed: ("000000000000", r5)})
    with pytest.raises(h2_compare.CompareAbort, match="missing"):
        compare(grid, frozen, {seed: (ver, r5), 7: ("x", 0.1)})
    # the contract pins are not those of the fixture
    with pytest.raises(h2_compare.CompareAbort):
        compare(grid, frozen, None)


def test_the_real_contract_pins_are_the_ones_in_the_contract():
    assert h2_compare.PINNED_ENDPOINT == {
        0: ("cadfa7929f8e", 0.8687032876672101),
        1: ("b43a45cb35c5", 0.8683824501123298),
        2: ("6bdb7fec4b7b", 0.8622586376517922),
    }
    assert h2_compare.R5_TOLERANCE == 1e-9 and h2_compare.ENDPOINT == 3000


def test_population_mismatch_is_an_error_before_any_comparison(grid, frozen, both):  # noqa: F811
    pins, _ = both
    with pytest.raises(h2_compare.CompareAbort, match="frozen population"):
        compare(grid, "1" * 64, pins)
    with pytest.raises(h2_compare.CompareAbort, match="differ from the curve plan.json"):
        # a validation rankings file that is not the one in the plans (one query removed)
        short = grid.out_root / "short.jsonl"
        short.write_text("".join(json.dumps(r) + "\n" for r in grid.val_rows[:-1]))
        h2_compare.compare_h2(
            grid.out_root / "u", grid.out_root / "s", grid.config, short, grid.paths["val"][1],
            embedder="fake", vector_loader=grid.loader, points=POINTS, seeds=(0,), pins=pins,
            manifest_sha=frozen, endpoint=160,
        )  # fmt: skip


def test_regenerated_rankings_with_a_missing_query_abort(grid, frozen, both, monkeypatch):  # noqa: F811
    pins, _ = both
    real = v2.rerank_rows
    monkeypatch.setattr(v2, "rerank_rows", lambda *a, **k: real(*a, **k)[:-1])
    with pytest.raises(curve.PopulationMismatch):
        compare(grid, frozen, pins)


def test_a_different_initial_model_aborts(grid, frozen, both):  # noqa: F811
    pins, srecs = both
    # swap the stratified model of 100 labels for the one of 130: not the shared start
    rewrite(
        grid.out_root / "s" / "runs.jsonl",
        lambda rec: (
            rec.update(reranker_version=srecs[130]["reranker_version"], recall=srecs[130]["recall"])
            if rec["budget"] == 100
            else None
        ),
    )
    with pytest.raises(h2_compare.CompareAbort, match="weights"):
        compare(grid, frozen, pins)


def test_missing_uncertainty_point_gives_a_missing_comparison_and_undetermined(grid, frozen, both):  # noqa: F811
    pins, _ = both
    path = grid.out_root / "u" / "runs.jsonl"
    rewrite(path, lambda rec: rec.update(status="failed") if rec["budget"] == 160 else None)
    res = compare(grid, frozen, pins)
    last = res["h2_comparison"][-1]
    assert last["status"] == "missing" and last["budget"] == 160
    assert res["verdict"]["sentence"] == "undetermined: incomplete (missing seeds: 0)"
    assert res["verdict"]["verdict"] == "undetermined"


# ---- verdict branches --------------------------------------------------------------------------------


def comp(seed, lo, hi, delta=0.01, status="completed", budget=3000):
    return {
        "seed": seed,
        "budget": budget,
        "status": status,
        "delta": {
            "5": {"delta": delta, "ci": [lo, hi], "p_le_0": 0.1},
            "1": {"delta": 0, "ci": [0, 0], "p_le_0": 1},
        },
    }


SEEDS3 = (0, 1, 2)


def test_verdict_raised_needs_every_lower_bound_above_zero():
    out = h2_compare.h2_verdict([comp(s, 0.001, 0.02) for s in SEEDS3], SEEDS3)
    assert (
        out["sentence"] == "uncertainty selection raised R@5 over stratified at 3,000 labels on these seeds"
    )
    assert out["verdict"] == "raised"
    # a lower bound of exactly 0 is not above 0
    out = h2_compare.h2_verdict([comp(0, 0.001, 0.02), comp(1, 0.0, 0.02), comp(2, 0.001, 0.02)], SEEDS3)
    assert out["sentence"] == "no consistent difference between the two policies at 3,000 labels"


def test_verdict_lowered_needs_every_upper_bound_below_zero():
    out = h2_compare.h2_verdict([comp(s, -0.02, -0.001) for s in SEEDS3], SEEDS3)
    assert (
        out["sentence"] == "uncertainty selection lowered R@5 below stratified at 3,000 labels on these seeds"
    )
    out = h2_compare.h2_verdict([comp(0, -0.02, -0.001), comp(1, -0.02, 0.0), comp(2, -0.02, -0.001)], SEEDS3)
    assert out["verdict"] == "none"


def test_verdict_no_consistent_difference_for_mixed_or_straddling_seeds():
    mixed = [comp(0, 0.001, 0.02), comp(1, -0.02, -0.001), comp(2, 0.001, 0.02)]
    assert h2_compare.h2_verdict(mixed, SEEDS3)["sentence"] == (
        "no consistent difference between the two policies at 3,000 labels"
    )
    straddle = [comp(s, -0.01, 0.01) for s in SEEDS3]
    assert h2_compare.h2_verdict(straddle, SEEDS3)["verdict"] == "none"


def test_verdict_incomplete_names_the_missing_seeds():
    good = [comp(0, 0.001, 0.02), comp(2, 0.001, 0.02)]
    out = h2_compare.h2_verdict(good, SEEDS3)
    assert out["sentence"] == "undetermined: incomplete (missing seeds: 1)" and out["missing_seeds"] == [1]
    out = h2_compare.h2_verdict([*good, comp(1, 0.001, 0.02, status="missing")], SEEDS3)
    assert out["missing_seeds"] == [1]
    out = h2_compare.h2_verdict([comp(0, 0.001, 0.02)], SEEDS3)
    assert out["sentence"] == "undetermined: incomplete (missing seeds: 1, 2)"
    assert h2_compare.h2_verdict([], SEEDS3)["missing_seeds"] == [0, 1, 2]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_verdict_non_finite_endpoint_is_incomplete_not_a_win(bad):
    recs = [comp(0, 0.001, 0.02), comp(1, 0.001, 0.02), comp(2, bad, 0.02)]
    out = h2_compare.h2_verdict(recs, SEEDS3)
    assert out["verdict"] == "undetermined" and out["missing_seeds"] == [2]
    recs[2] = comp(2, 0.001, 0.02, delta=bad)
    assert h2_compare.h2_verdict(recs, SEEDS3)["missing_seeds"] == [2]


def test_verdict_uses_only_the_endpoint_records():
    recs = [comp(s, 0.001, 0.02) for s in SEEDS3] + [comp(0, -0.5, -0.4, budget=1000)]
    assert h2_compare.h2_verdict(recs, SEEDS3)["verdict"] == "raised"
    only_early = [comp(s, 0.001, 0.02, budget=1000) for s in SEEDS3]
    assert h2_compare.h2_verdict(only_early, SEEDS3)["verdict"] == "undetermined"


# ---- CLI ---------------------------------------------------------------------------------------------


def test_cli_uncertainty_path_enforces_the_frozen_population_and_h2_compare_the_pins(env):  # noqa: F811
    runner = CliRunner()
    base = ["--config", str(env.cfg), "--val-rankings", str(env.val[0]), "--val-cand-stats", str(env.val[1])]
    args = [
        "uncertainty-path",
        *base,
        "--train-rankings", str(env.train[0]),
        "--train-cand-stats", str(env.train[1]),
        "--embedder", "fake",
        "--artifacts-root", str(env.art),
        "--out-root", str(env.tmp / "u"),
        "--run-id", "x",
    ]  # fmt: skip
    res = runner.invoke(app, args)
    assert (
        res.exit_code == 1 and "manifest sha256" in res.output
    )  # the 12-query fixture is not the frozen set
    assert not (env.tmp / "u" / "x").exists()
    help_out = runner.invoke(app, ["uncertainty-path", "--help"]).output
    for word in ("--split", "--manifest", "--noise", "--policy"):
        assert word not in help_out
    assert runner.invoke(app, ["uncertainty-path", *base]).exit_code == 2  # --run-id and train inputs missing
    assert runner.invoke(app, ["h2-compare", "--help"]).exit_code == 0


def test_cli_h2_compare_refuses_a_run_that_is_not_h2(env, tmp_path):  # noqa: F811
    runner = CliRunner()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "plan.json").write_text(json.dumps({"hypothesis_id": "H1-rerank-r5"}))
    (tmp_path / "b" / "plan.json").write_text(json.dumps({"hypothesis_id": "H1-rerank-r5"}))
    res = runner.invoke(
        app,
        [
            "h2-compare", "--uncertainty-run", str(tmp_path / "a"), "--curve-run", str(tmp_path / "b"),
            "--config", str(env.cfg), "--val-rankings", str(env.val[0]), "--val-cand-stats", str(env.val[1]),
            "--embedder", "fake",
        ],
    )  # fmt: skip
    assert res.exit_code == 1 and "not an H2-uncertainty run" in res.output


def test_oracle_and_stratified_pairs_match_the_simulator(grid):  # noqa: F811
    sha = curve.sha256_file(grid.paths["train"][0])
    ref = sim.simulate_stratified(grid.train_rows, budget=120, seed=1, rankings_sha256=sha)
    pairs = sim.stratified_pairs(grid.train_rows, budget=120, seed=1)
    oracle = sim.Oracle(grid.train_rows, rankings_sha256=sha, seed=1, policy_id="stratified")
    got = oracle.answer(pairs[:50], 0) + oracle.answer(pairs[50:], 50)  # two rounds = one run
    assert [e.model_dump() for e in got] == [e.model_dump() for e in ref]
    with pytest.raises(sim.SimulationError, match="not in the exposed pool"):
        oracle.answer([("t0", "no-such-product", 1)], 0)
    with pytest.raises(sim.SimulationError, match="not in the exposed pool"):
        oracle.answer([(pairs[0][0], pairs[0][1], pairs[0][2] + 1)], 0)
    labels, _ = build_labels(got)
    assert len(labels) == 120
    assert FakeVectors  # fixture helper imported for the grid


# ---- review fixes: shared inputs, bounded selection failures, recoverable round record ---------------


@pytest.mark.parametrize(
    "path",
    [
        ("rankings_sha256", "train"),
        ("cand_stats_sha256", "train"),
        ("cand_stats_sha256", "val"),
        ("index_id", "train"),
        ("index_id", "val"),
        ("embed_model_id", None),
        ("config_name", None),
        ("embedder", None),
    ],
)
def test_compare_aborts_when_the_runs_used_different_frozen_inputs(grid, frozen, both, path):  # noqa: F811
    pins, _ = both
    compare(grid, frozen, pins)  # positive control: the real pair compares fine
    plan_path = grid.out_root / "u" / "plan.json"
    original = plan_path.read_text()
    plan = json.loads(original)
    key, sub = path
    if sub is None:
        plan["inputs"][key] = "something-else"
    else:
        plan["inputs"][key][sub] = "0" * 64
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(h2_compare.CompareAbort, match="the two runs differ in"):
        compare(grid, frozen, pins)
    plan_path.write_text(original)


def test_a_different_train_rankings_hash_aborts_although_the_initial_weights_are_identical(
    grid,  # noqa: F811
    frozen,
    both,
):
    pins, srecs = both
    urecs = {r["budget"]: r for r in u_lines(grid)}
    ms = v2.load_model(grid.out_root / "s" / "models" / srecs[100]["reranker_version"] / "model.json")
    mu = v2.load_model(grid.out_root / "u" / "models" / urecs[100]["reranker_version"] / "model.json")
    assert all(torch.equal(ms["state_dict"][k], mu["state_dict"][k]) for k in ms["state_dict"])
    plan_path = grid.out_root / "u" / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["inputs"]["rankings_sha256"]["train"] = "f" * 64
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(h2_compare.CompareAbort, match="train rankings sha256"):
        compare(grid, frozen, pins)
    assert not (grid.out_root / "u" / "h2_comparison.json").exists()


def fail_selection(monkeypatch, errors):
    """Make select_next_batch raise ``errors[i]`` on call i (None = work normally)."""
    real = uncertainty_path.select_next_batch
    calls = {"n": 0}

    def fake(*a, **k):
        i = calls["n"]
        calls["n"] += 1
        if i < len(errors) and errors[i] is not None:
            raise errors[i]
        return real(*a, **k)

    monkeypatch.setattr(uncertainty_path, "select_next_batch", fake)
    return real, calls


def sel_lines(g):
    return [r for r in u_lines(g) if r.get("stage") == "selection"]


def test_selection_os_error_is_recorded_and_retried_once(grid, frozen, monkeypatch):  # noqa: F811
    fail_selection(monkeypatch, [OSError("disk hiccup")])
    res = u_run(grid, frozen)
    (bad,) = sel_lines(grid)
    assert (bad["stage"], bad["attempt"], bad["status"], bad["retryable"]) == ("selection", 1, "failed", True)
    assert bad["error_type"] == "OSError" and "disk hiccup" in bad["error"]
    assert bad["key_id"] == "uncertainty|p0|s0|B100|selection" and bad["round"] == 0
    assert (res["completed"], res["failed"]) == (3, 1)
    assert len(load_events(grid.out_root / "u" / "events" / "uncertainty.seed0.jsonl")) == 160
    assert [(x["from_size"], x["to_size"]) for x in u_lines(grid, name="rounds.jsonl")] == [
        (100, 130),
        (130, 160),
    ]


def test_selection_os_error_twice_stops_the_seed_and_resume_does_not_retry_again(grid, frozen, monkeypatch):  # noqa: F811
    _, calls = fail_selection(monkeypatch, [OSError("a"), OSError("b")])
    res = u_run(grid, frozen)
    assert [(r["attempt"], r["retryable"]) for r in sel_lines(grid)] == [(1, True), (2, True)]
    assert calls["n"] == 2 and res["completed"] == 1
    ev = grid.out_root / "u" / "events" / "uncertainty.seed0.jsonl"
    assert len(ev.read_text().splitlines()) == 100
    before = (grid.out_root / "u" / "runs.jsonl").read_bytes()
    monkeypatch.undo()  # a working selector must still not be asked again
    assert u_run(grid, frozen)["attempted"] == 0
    assert (grid.out_root / "u" / "runs.jsonl").read_bytes() == before
    assert len(ev.read_text().splitlines()) == 100


def test_selection_deterministic_error_stops_the_seed_with_the_record_kept(grid, frozen, monkeypatch):  # noqa: F811
    _, calls = fail_selection(monkeypatch, [ValueError("bad probabilities")])
    res = u_run(grid, frozen, seeds=(0, 1))
    (bad,) = [r for r in sel_lines(grid) if r["seed"] == 0]
    assert (bad["attempt"], bad["retryable"], bad["error_type"]) == (1, False, "ValueError")
    # seed 0 stops after one call; seed 1 is independent and runs to the end
    assert calls["n"] == 1 + 2
    assert res["failed"] == 1
    assert [r["status"] for r in u_lines(grid) if r["seed"] == 0 and r.get("stage") != "selection"] == [
        "completed"
    ]
    monkeypatch.undo()
    assert u_run(grid, frozen, seeds=(0, 1))["attempted"] == 0


def test_oracle_error_in_the_selection_transition_is_recorded_too(grid, frozen, monkeypatch):  # noqa: F811
    real = sim.Oracle.answer
    state = {"n": 0}

    def once(self, pairs, first_index=0):
        state["n"] += 1  # the first call answers I_s; the next one is the first selection round
        if state["n"] == 1:
            return real(self, pairs, first_index)
        raise ValueError("oracle down")

    monkeypatch.setattr(sim.Oracle, "answer", once)
    u_run(grid, frozen)
    (bad,) = sel_lines(grid)
    assert "oracle down" in bad["error"] and bad["retryable"] is False


def test_selection_retry_count_survives_a_crash_in_the_retry(grid, frozen, monkeypatch):  # noqa: F811
    fail_selection(monkeypatch, [OSError("first"), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        u_run(grid, frozen)
    assert [r["attempt"] for r in sel_lines(grid)] == [1]
    # resume: exactly one more attempt is allowed; it fails again, then the seed stops for good
    fail_selection(monkeypatch, [OSError("second")])
    u_run(grid, frozen)
    assert [r["attempt"] for r in sel_lines(grid)] == [1, 2]
    monkeypatch.undo()
    assert u_run(grid, frozen)["attempted"] == 0
    assert [r["attempt"] for r in sel_lines(grid)] == [1, 2]


def test_crash_between_the_events_file_and_the_round_record_is_repaired_on_resume(
    grid,  # noqa: F811
    frozen,
    monkeypatch,
):
    u_run(grid, frozen, run_id="whole")
    real_append = curve._append_line
    state = {"armed": True}

    def crashing(path, obj):
        if state["armed"] and path.name == "rounds.jsonl":
            state["armed"] = False
            raise KeyboardInterrupt  # dies after the events file moved on, before the round record
        return real_append(path, obj)

    monkeypatch.setattr(curve, "_append_line", crashing)
    with pytest.raises(KeyboardInterrupt):
        u_run(grid, frozen, run_id="part")
    monkeypatch.undo()
    ev = grid.out_root / "part" / "events" / "uncertainty.seed0.jsonl"
    assert len(ev.read_text().splitlines()) == 130
    assert not (grid.out_root / "part" / "rounds.jsonl").exists()  # the record really was lost
    u_run(grid, frozen, run_id="part")
    got = u_lines(grid, "part", "rounds.jsonl")
    want = u_lines(grid, "whole", "rounds.jsonl")
    assert [x["round"] for x in got] == [1, 2]  # one record per round, none doubled
    skip = {"selection_seconds", "finished_at", "reconstructed"}
    assert [{k: v for k, v in x.items() if k not in skip} for x in got] == [
        {k: v for k, v in x.items() if k not in skip} for x in want
    ]
    assert got[0]["reconstructed"] is True and got[0]["selection_seconds"] is None
    assert got[1]["reconstructed"] is False and isinstance(got[1]["selection_seconds"], float)
    assert ev.read_bytes() == (grid.out_root / "whole" / "events" / "uncertainty.seed0.jsonl").read_bytes()
    # a third resume writes nothing more
    before = (grid.out_root / "part" / "rounds.jsonl").read_bytes()
    assert u_run(grid, frozen, run_id="part")["attempted"] == 0
    assert (grid.out_root / "part" / "rounds.jsonl").read_bytes() == before
