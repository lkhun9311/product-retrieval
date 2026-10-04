import json
import random
from functools import partial

import pytest
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.feedback.simulate import (
    SimulationError,
    run_simulate,
    simulate_random,
    simulate_stratified,
)

SHA = "a" * 64
rnd = partial(simulate_random, rankings_sha256=SHA)
strat = partial(simulate_stratified, rankings_sha256=SHA)


def _rankings(n_queries=5, n_cands=25):
    rows = []
    for q in range(n_queries):
        cands = [f"p{q}_{r}" for r in range(n_cands)]
        rows.append(
            {
                "query_id": f"q{q}",
                "truth_product_id": cands[q % 3],
                "top_k_product_ids": cands,
                "scores": [1.0 - 0.01 * r for r in range(n_cands)],
            }
        )
    return rows


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_n_events_for_budget():
    assert len(rnd(_rankings(), budget=37, seed=1)) == 37


def test_same_seed_identical_different_seed_differs():
    a = rnd(_rankings(), budget=30, seed=1)
    b = rnd(_rankings(), budget=30, seed=1)
    c = rnd(_rankings(), budget=30, seed=2)
    assert a == b
    assert [e.event_id for e in a] != [e.event_id for e in c]


def test_event_id_unique_and_idempotent():
    a = rnd(_rankings(), budget=100, seed=3)
    ids = [e.event_id for e in a]
    assert len(set(ids)) == len(ids)
    assert ids == [e.event_id for e in rnd(_rankings(), budget=100, seed=3)]
    # seed is part of the id
    other = {e.event_id for e in rnd(_rankings(), budget=100, seed=4)}
    assert not other & set(ids)


def test_actor_policy_position_one_based_and_exposed_only():
    evs = rnd(_rankings(), budget=100, seed=0)  # all 5*20 pairs
    assert all(e.actor == "sim" and e.policy_id == "random" for e in evs)
    assert {e.position for e in evs} == set(range(1, 21))
    assert not any(e.product_id.endswith(("_20", "_24")) for e in evs)


def test_match_iff_truth_without_noise():
    rows = _rankings()
    truth = {r["query_id"]: r["truth_product_id"] for r in rows}
    for e in rnd(rows, budget=100, seed=5):
        assert (e.action == "match") == (e.product_id == truth[e.query_id])


def test_noise_one_flips_all_and_noise_is_deterministic():
    rows = _rankings()
    truth = {r["query_id"]: r["truth_product_id"] for r in rows}
    for e in rnd(rows, budget=100, seed=5, noise=1.0):
        assert (e.action == "match") != (e.product_id == truth[e.query_id])
    a = rnd(rows, budget=60, seed=5, noise=0.3)
    assert a == rnd(rows, budget=60, seed=5, noise=0.3)
    clean = rnd(rows, budget=60, seed=5)
    assert [e.action for e in a] != [e.action for e in clean]


def test_budget_over_pairs_errors():
    with pytest.raises(SimulationError, match="exceeds"):
        rnd(_rankings(), budget=101, seed=0)


def test_budget_zero_is_empty(tmp_path):
    src, out = tmp_path / "r.jsonl", tmp_path / "o.jsonl"
    _write(src, _rankings())
    assert run_simulate(src, out, budget=0, seed=0, policy="random") == []
    assert out.read_text() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": -1},
        {"budget": 1, "noise": 1.5},
        {"budget": 1, "noise": -0.1},
        {"budget": 1, "exposed_k": 0},
    ],
)
def test_invalid_args(kwargs):
    with pytest.raises(SimulationError):
        rnd(_rankings(), seed=0, **kwargs)


def test_short_candidate_lists():
    rows = _rankings(n_queries=3, n_cands=4)
    rows[1]["top_k_product_ids"] = []
    assert len(rnd(rows, budget=8, seed=0)) == 8
    with pytest.raises(SimulationError):
        rnd(rows, budget=9, seed=0)


def test_duplicate_query_or_product_rejected():
    rows = _rankings(2, 3)
    with pytest.raises(SimulationError, match="duplicate query_id"):
        rnd(rows + rows[:1], budget=1, seed=0)
    rows[0]["top_k_product_ids"][1] = rows[0]["top_k_product_ids"][0]
    with pytest.raises(SimulationError, match="more than once"):
        rnd(rows, budget=1, seed=0)


def test_missing_field_rejected(tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps({"query_id": "q"}) + "\n")
    with pytest.raises(SimulationError, match="missing field"):
        run_simulate(src, tmp_path / "o.jsonl", budget=0, seed=0, policy="random")


def test_output_has_no_truth_field_and_byte_identical(tmp_path):
    rows = _rankings()
    for r in rows:
        r["truth_product_id"] = "SECRET_TRUTH_" + r["query_id"]
        r["top_k_product_ids"][2] = r["truth_product_id"]
    src = tmp_path / "r.jsonl"
    _write(src, rows)
    o1, o2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_simulate(src, o1, budget=100, seed=9, policy="random")
    run_simulate(src, o2, budget=100, seed=9, policy="random")
    assert o1.read_bytes() == o2.read_bytes()
    text = o1.read_text()
    assert "truth_product_id" not in text
    # truth product id appears as product_id only for match events
    for line in text.splitlines():
        d = json.loads(line)
        assert "truth_product_id" not in d
        assert (d["action"] == "match") == d["product_id"].startswith("SECRET_TRUTH_")


def test_cli_round_trip(tmp_path):
    src, out = tmp_path / "r.jsonl", tmp_path / "sub" / "o.jsonl"
    _write(src, _rankings())
    runner = CliRunner()
    args = ["simulate", "--rankings", str(src), "--budget", "10", "--seed", "3", "--out", str(out)]
    args += ["--policy", "random"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    lines = out.read_text().splitlines()
    assert len(lines) == 10
    assert json.loads(lines[0])["actor"] == "sim"


def test_cli_budget_too_large_exits_1(tmp_path):
    src = tmp_path / "r.jsonl"
    _write(src, _rankings(1, 3))
    res = CliRunner().invoke(
        app,
        [
            "simulate",
            "--rankings",
            str(src),
            "--budget",
            "4",
            "--seed",
            "0",
            "--policy",
            "random",
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert res.exit_code == 1
    assert "exceeds" in res.output
    assert not (tmp_path / "o").exists()


# ---- c4-v3: stratified policy, noise, namespace, summary ----


def _big(n=60, k=20, shift=0):
    rows = []
    for q in range(n):
        cands = [f"p{q}_{r}" for r in range(k)]
        rows.append(
            {
                "query_id": f"q{q:03d}",
                "truth_product_id": cands[(q + shift) % k],
                "top_k_product_ids": cands,
            }
        )
    return rows


def _seq(events):
    return [(e.query_id, e.product_id) for e in events]


def _stratum(pos):
    return 0 if pos <= 5 else 1 if pos <= 10 else 2


def test_stratified_nesting_prefix():
    a = strat(_big(), budget=30, seed=1)
    b = strat(_big(), budget=90, seed=1)
    assert [e.model_dump_json() for e in a] == [e.model_dump_json() for e in b[:30]]


def test_stratified_counts_equal_and_one_judgment_per_query_per_stratum():
    for budget in (0, 1, 2, 3, 31, 100, 179, 180):
        evs = strat(_big(), budget=budget, seed=2)
        assert len(evs) == budget
        counts = [0, 0, 0]
        seen = set()
        for e in evs:
            s = _stratum(e.position)
            counts[s] += 1
            assert (e.query_id, s) not in seen
            seen.add((e.query_id, s))
        assert max(counts) - min(counts) <= 1
    assert counts == [60, 60, 60]


def test_stratified_seed_behaviour():
    assert strat(_big(), budget=60, seed=1) == strat(_big(), budget=60, seed=1)
    assert _seq(strat(_big(), budget=60, seed=1)) != _seq(strat(_big(), budget=60, seed=2))


def test_stratified_input_order_irrelevant():
    rows = _big()
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    assert _seq(strat(rows, budget=90, seed=4)) == _seq(strat(shuffled, budget=90, seed=4))


def test_truth_invariance_of_selection():
    rows = _big()
    perm = [r["truth_product_id"] for r in rows]
    perm = perm[7:] + perm[:7]
    other = [dict(r, truth_product_id=t) for r, t in zip(rows, perm, strict=True)]
    other2 = [dict(r, truth_product_id="nonexistent") for r in rows]
    base = strat(rows, budget=180, seed=3)
    assert _seq(base) == _seq(strat(other, budget=180, seed=3))
    assert _seq(base) == _seq(strat(other2, budget=180, seed=3))
    # answers do depend on truth
    assert [e.action for e in base] != [e.action for e in strat(other, budget=180, seed=3)]


def test_stratified_errors():
    with pytest.raises(SimulationError, match="capacity"):
        strat(_big(10), budget=31, seed=0)
    with pytest.raises(SimulationError, match="exposed_k"):
        strat(_big(), budget=1, seed=0, exposed_k=10)
    with pytest.raises(SimulationError, match="exposed_k"):
        strat(_big(k=25), budget=1, seed=0, exposed_k=25)
    short = _big()
    short[3]["top_k_product_ids"] = short[3]["top_k_product_ids"][:19]
    with pytest.raises(SimulationError, match="19 candidates"):
        strat(short, budget=1, seed=0)
    dup = _big()
    dup[0]["top_k_product_ids"][5] = dup[0]["top_k_product_ids"][0]
    with pytest.raises(SimulationError, match="more than once"):
        strat(dup, budget=1, seed=0)
    with pytest.raises(SimulationError, match="duplicate query_id"):
        strat(_big() + _big(1), budget=1, seed=0)


def test_noise_same_pair_same_answer_across_policies_and_budgets():
    rows = _big()
    p = 0.3
    s_small = {(e.query_id, e.product_id): e.action for e in strat(rows, budget=30, seed=5, noise=p)}
    s_big = {(e.query_id, e.product_id): e.action for e in strat(rows, budget=180, seed=5, noise=p)}
    r_all = {(e.query_id, e.product_id): e.action for e in rnd(rows, budget=1200, seed=5, noise=p)}
    for k, v in s_small.items():
        assert s_big[k] == v == r_all[k]
    for k, v in s_big.items():
        assert r_all[k] == v


def test_noise_changes_event_ids_but_same_u_nests_flips():
    rows = _big()
    a = strat(rows, budget=180, seed=1, noise=0.0)
    b = strat(rows, budget=180, seed=1, noise=0.2)
    assert not {e.event_id for e in a} & {e.event_id for e in b}
    flipped_02 = {(e.query_id, e.product_id) for x, e in zip(a, b, strict=True) if x.action != e.action}
    c = strat(rows, budget=180, seed=1, noise=0.1)
    flipped_01 = {(e.query_id, e.product_id) for x, e in zip(a, c, strict=True) if x.action != e.action}
    assert flipped_01 <= flipped_02  # same u across p


def test_event_id_ignores_budget_and_policy_differs():
    rows = _big()
    a = strat(rows, budget=30, seed=1)
    b = strat(rows, budget=60, seed=1)
    assert [e.event_id for e in a] == [e.event_id for e in b[:30]]
    r = rnd(rows, budget=1200, seed=1)
    assert not {e.event_id for e in r} & {e.event_id for e in b}


def test_flip_rate_close_to_p():
    rows = _big(n=1500)
    for p in (0.1, 0.2):
        clean = strat(rows, budget=4500, seed=0, noise=0.0)
        noisy = strat(rows, budget=4500, seed=0, noise=p)
        rate = sum(x.action != y.action for x, y in zip(clean, noisy, strict=True)) / 4500
        assert abs(rate - p) < 0.02


def test_summary_and_cli_round_trip_both_policies(tmp_path):
    src = tmp_path / "r.jsonl"
    _write(src, _big())
    for policy in ("stratified", "random"):
        out = tmp_path / f"{policy}.jsonl"
        args = ["simulate", "--rankings", str(src), "--budget", "90", "--seed", "3"]
        args += ["--noise", "0.1", "--policy", policy, "--out", str(out)]
        res = CliRunner().invoke(app, args)
        assert res.exit_code == 0, res.output
        events = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(events) == 90
        assert {e["policy_id"] for e in events} == {policy}
        summ = json.loads((tmp_path / f"{policy}.jsonl.summary.json").read_text())
        assert summ["contract"] == "c4-v3"
        assert summ["policy"] == policy
        assert summ["seed"] == 3 and summ["budget"] == 90 and summ["noise"] == 0.1
        assert summ["event_count"] == 90
        assert summ["match_count"] == sum(e["action"] == "match" for e in events)
        assert len(summ["rankings_sha256"]) == 64 and len(summ["namespace"]) == 64
        assert summ["stratum_counts"] == ([30, 30, 30] if policy == "stratified" else None)
        assert "truth" not in out.read_text()


def test_cli_default_is_stratified_and_errors(tmp_path):
    src = tmp_path / "r.jsonl"
    _write(src, _big(10))
    out = tmp_path / "o.jsonl"
    res = CliRunner().invoke(
        app, ["simulate", "--rankings", str(src), "--budget", "30", "--seed", "0", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text().splitlines()[0])["policy_id"] == "stratified"
    bad = CliRunner().invoke(
        app,
        ["simulate", "--rankings", str(src), "--budget", "31", "--seed", "0", "--out", str(tmp_path / "x")],
    )
    assert bad.exit_code == 1 and "capacity" in bad.output
    bad = CliRunner().invoke(
        app,
        [
            "simulate",
            "--rankings",
            str(src),
            "--budget",
            "1",
            "--seed",
            "0",
            "--policy",
            "nope",
            "--out",
            str(tmp_path / "x"),
        ],
    )
    assert bad.exit_code == 1 and "unknown policy" in bad.output


def test_summary_namespace_depends_on_rankings_bytes_and_noise(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, _big())
    _write(b, _big(shift=1))
    run_simulate(a, tmp_path / "o1", budget=3, seed=0)
    run_simulate(b, tmp_path / "o2", budget=3, seed=0)
    run_simulate(a, tmp_path / "o3", budget=3, seed=0, noise=0.1)
    ns = [json.loads((tmp_path / f"o{i}.summary.json").read_text())["namespace"] for i in (1, 2, 3)]
    assert len(set(ns)) == 3
