import json

import pytest
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.feedback.simulate import (
    SimulationError,
    run_simulate,
    simulate_random,
)


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
    assert len(simulate_random(_rankings(), budget=37, seed=1)) == 37


def test_same_seed_identical_different_seed_differs():
    a = simulate_random(_rankings(), budget=30, seed=1)
    b = simulate_random(_rankings(), budget=30, seed=1)
    c = simulate_random(_rankings(), budget=30, seed=2)
    assert a == b
    assert [e.event_id for e in a] != [e.event_id for e in c]


def test_event_id_unique_and_idempotent():
    a = simulate_random(_rankings(), budget=100, seed=3)
    ids = [e.event_id for e in a]
    assert len(set(ids)) == len(ids)
    assert ids == [e.event_id for e in simulate_random(_rankings(), budget=100, seed=3)]
    # seed is part of the id
    other = {e.event_id for e in simulate_random(_rankings(), budget=100, seed=4)}
    assert not other & set(ids)


def test_actor_policy_position_one_based_and_exposed_only():
    evs = simulate_random(_rankings(), budget=100, seed=0)  # all 5*20 pairs
    assert all(e.actor == "sim" and e.policy_id == "random" for e in evs)
    assert {e.position for e in evs} == set(range(1, 21))
    assert not any(e.product_id.endswith(("_20", "_24")) for e in evs)


def test_match_iff_truth_without_noise():
    rows = _rankings()
    truth = {r["query_id"]: r["truth_product_id"] for r in rows}
    for e in simulate_random(rows, budget=100, seed=5):
        assert (e.action == "match") == (e.product_id == truth[e.query_id])


def test_noise_one_flips_all_and_noise_is_deterministic():
    rows = _rankings()
    truth = {r["query_id"]: r["truth_product_id"] for r in rows}
    for e in simulate_random(rows, budget=100, seed=5, noise=1.0):
        assert (e.action == "match") != (e.product_id == truth[e.query_id])
    a = simulate_random(rows, budget=60, seed=5, noise=0.3)
    assert a == simulate_random(rows, budget=60, seed=5, noise=0.3)
    clean = simulate_random(rows, budget=60, seed=5)
    assert [e.action for e in a] != [e.action for e in clean]


def test_budget_over_pairs_errors():
    with pytest.raises(SimulationError, match="exceeds"):
        simulate_random(_rankings(), budget=101, seed=0)


def test_budget_zero_is_empty(tmp_path):
    src, out = tmp_path / "r.jsonl", tmp_path / "o.jsonl"
    _write(src, _rankings())
    assert run_simulate(src, out, budget=0, seed=0) == []
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
        simulate_random(_rankings(), seed=0, **kwargs)


def test_short_candidate_lists():
    rows = _rankings(n_queries=3, n_cands=4)
    rows[1]["top_k_product_ids"] = []
    assert len(simulate_random(rows, budget=8, seed=0)) == 8
    with pytest.raises(SimulationError):
        simulate_random(rows, budget=9, seed=0)


def test_duplicate_query_or_product_rejected():
    rows = _rankings(2, 3)
    with pytest.raises(SimulationError, match="duplicate query_id"):
        simulate_random(rows + rows[:1], budget=1, seed=0)
    rows[0]["top_k_product_ids"][1] = rows[0]["top_k_product_ids"][0]
    with pytest.raises(SimulationError, match="more than once"):
        simulate_random(rows, budget=1, seed=0)


def test_missing_field_rejected(tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps({"query_id": "q"}) + "\n")
    with pytest.raises(SimulationError, match="missing field"):
        run_simulate(src, tmp_path / "o.jsonl", budget=0, seed=0)


def test_output_has_no_truth_field_and_byte_identical(tmp_path):
    rows = _rankings()
    for r in rows:
        r["truth_product_id"] = "SECRET_TRUTH_" + r["query_id"]
        r["top_k_product_ids"][2] = r["truth_product_id"]
    src = tmp_path / "r.jsonl"
    _write(src, rows)
    o1, o2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    run_simulate(src, o1, budget=100, seed=9)
    run_simulate(src, o2, budget=100, seed=9)
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
        ["simulate", "--rankings", str(src), "--budget", "4", "--seed", "0", "--out", str(tmp_path / "o")],
    )
    assert res.exit_code == 1
    assert "exceeds" in res.output
    assert not (tmp_path / "o").exists()
