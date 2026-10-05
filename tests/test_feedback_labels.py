import json
import random
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.feedback.labels import (
    CONTRACT,
    LabelBuildError,
    build_labels,
    load_events,
    run_labels,
)
from product_retrieval.feedback.simulate import run_simulate

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def ev(eid, action, *, q="q1", p="p1", session="s1", t=0):
    return {
        "event_id": eid,
        "ts": (T0 + timedelta(seconds=t)).isoformat(),
        "session_id": session,
        "list_id": "l1",
        "query_id": q,
        "product_id": p,
        "action": action,
        "actor": "human",
    }


def build(rows):
    from product_retrieval.core.schemas import FeedbackEvent

    return build_labels([FeedbackEvent(**r) for r in rows])


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_skip_only_pair_has_no_label_and_never_neg():
    labels, rep = build([ev("e1", "skip"), ev("e2", "match", p="p2")])
    assert [(lb.product_id, lb.kind) for lb in labels] == [("p2", "pos")]
    assert not [lb for lb in labels if lb.kind == "neg"]
    assert rep["unlabeled_pairs"]["skip_only"] == 1
    # an exposed-but-unanswered candidate never appears in the input, hence never in the output
    assert all(lb.product_id != "p_unanswered" for lb in labels)


def test_empty_input_gives_no_labels():
    labels, rep = build([])
    assert labels == [] and rep["pos"] == 0 and rep["neg"] == 0 and rep["contract"] == CONTRACT


def test_match_pos_not_match_neg_weight_one():
    labels, rep = build([ev("e1", "match", p="a"), ev("e2", "not_match", p="b")])
    got = {lb.product_id: lb for lb in labels}
    assert got["a"].kind == "pos" and got["a"].weight == 1.0
    assert got["b"].kind == "neg" and got["b"].weight == 1.0
    assert got["a"].origin_events == ["e1"]
    assert rep["pos"] == 1 and rep["neg"] == 1 and rep["conflict_count"] == 0


def test_in_session_reversal_latest_ts_wins_and_origin_has_both():
    labels, rep = build([ev("e1", "match", t=5), ev("e2", "not_match", t=9)])
    (lb,) = labels
    assert lb.kind == "neg" and lb.weight == 1.0
    assert lb.origin_events == ["e1", "e2"]
    assert rep["conflict_count"] == 1 and rep["conflicts"][0]["resolution"] == "neg"


def test_in_session_same_ts_tie_breaks_on_last_event_id():
    labels, _ = build([ev("b", "match", t=3), ev("a", "not_match", t=3)])
    assert labels[0].kind == "pos"


def test_cross_session_majority_weight_two_thirds():
    # 2 match sessions vs 1 not_match session: winner 2 of 3 judging sessions -> 2/3
    rows = [
        ev("e1", "match", session="s1"),
        ev("e2", "match", session="s2"),
        ev("e3", "not_match", session="s3"),
    ]
    labels, rep = build(rows)
    (lb,) = labels
    assert lb.kind == "pos" and lb.weight == pytest.approx(2 / 3)
    assert lb.origin_events == ["e1", "e2", "e3"]
    assert rep["conflicts"][0]["session_judgments"] == {"s1": "match", "s2": "match", "s3": "not_match"}


def test_cross_session_tie_makes_no_label_and_is_reported():
    labels, rep = build([ev("e1", "match", session="s1"), ev("e2", "not_match", session="s2")])
    assert labels == []
    assert rep["unlabeled_pairs"]["tied"] == 1
    assert rep["conflict_count"] == 1
    assert rep["conflicts"][0]["resolution"] == "none"
    assert rep["conflicts"][0]["event_ids"] == ["e1", "e2"]


def test_reversal_inside_session_then_other_session_disagrees_is_tie():
    # s1 ends as not_match (reversal), s2 says match -> 1 vs 1 -> no label
    rows = [
        ev("e1", "match", session="s1", t=1),
        ev("e2", "not_match", session="s1", t=2),
        ev("e3", "match", session="s2", t=1),
    ]
    labels, rep = build(rows)
    assert labels == [] and rep["unlabeled_pairs"]["tied"] == 1


def test_identical_duplicate_counted_once():
    labels, rep = build([ev("e1", "match"), ev("e1", "match")])
    assert len(labels) == 1 and labels[0].origin_events == ["e1"] and labels[0].weight == 1.0
    assert rep["input_events"] == 2 and rep["unique_events"] == 1 and rep["duplicates_removed"] == 1
    assert rep["action_counts"]["match"] == 1


def test_same_event_id_different_content_raises():
    with pytest.raises(LabelBuildError, match="different content"):
        build([ev("e1", "match"), ev("e1", "not_match")])


def test_prefer_and_point_counted_not_labelled():
    labels, rep = build([ev("e1", "prefer", p="a"), ev("e2", "point", p="b"), ev("e3", "match", p="c")])
    assert [lb.product_id for lb in labels] == ["c"]
    assert rep["action_counts"]["prefer"] == 1 and rep["action_counts"]["point"] == 1
    assert rep["unlabeled_pairs"]["prefer_point_only"] == 2


def test_skip_does_not_override_a_judgment():
    labels, _ = build([ev("e1", "match", t=1), ev("e2", "skip", t=9)])
    assert labels[0].kind == "pos"


def test_shuffled_input_same_labels_and_version():
    rows = []
    for i in range(30):
        rows.append(ev(f"m{i}", "match", q=f"q{i % 4}", p=f"p{i % 5}", session=f"s{i % 3}", t=i))
        rows.append(ev(f"n{i}", "not_match", q=f"q{i % 4}", p=f"p{i % 5}", session=f"s{(i + 1) % 3}", t=i))
    rows.append(ev("dup", "skip"))
    rows.append(ev("dup", "skip"))
    base_labels, base_rep = build(rows)
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        labels, rep = build(shuffled)
        assert labels == base_labels
        assert rep == base_rep
    keys = [(lb.query_id, lb.product_id) for lb in base_labels]
    assert keys == sorted(keys)


def test_label_version_depends_on_event_set():
    a, _ = build([ev("e1", "match")])
    b, _ = build([ev("e1", "match"), ev("e2", "skip", p="x")])
    assert a[0].label_version != b[0].label_version
    assert len(a[0].label_version) == 12


def test_load_events_skips_blank_lines_and_reports_line_number(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(ev("e1", "match")) + "\n\n" + '{"event_id": "x"}\n', encoding="utf-8")
    with pytest.raises(LabelBuildError, match=r":3:"):
        load_events(p)
    p.write_text(json.dumps(ev("e1", "match")) + "\n\n", encoding="utf-8")
    assert len(load_events(p)) == 1


def test_invalid_action_rejected(tmp_path):
    p = tmp_path / "e.jsonl"
    write(p, [ev("e1", "like")])
    with pytest.raises(LabelBuildError):
        load_events(p)


def _rankings(n=10, k=20):
    rows = []
    for q in range(n):
        cands = [f"p{q}_{r}" for r in range(k)]
        rows.append({"query_id": f"q{q:02d}", "truth_product_id": cands[q % k], "top_k_product_ids": cands})
    return rows


def test_simulator_output_has_zero_conflicts(tmp_path):
    rk = tmp_path / "r.jsonl"
    write(rk, _rankings())
    ev_path, out = tmp_path / "e.jsonl", tmp_path / "l.jsonl"
    events = run_simulate(rk, ev_path, budget=30, seed=1, noise=0.2)
    rep = run_labels(ev_path, out)
    assert rep["conflict_count"] == 0 and rep["conflicts"] == []
    assert rep["pos"] + rep["neg"] == 30 == len(events)
    assert rep["unlabeled_pairs"] == {"skip_only": 0, "tied": 0, "prefer_point_only": 0}
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 30 and all(x["weight"] == 1.0 for x in lines)
    assert json.loads((tmp_path / "l.jsonl.report.json").read_text())["label_version"] == rep["label_version"]


def test_cli_happy_path(tmp_path):
    src, out = tmp_path / "e.jsonl", tmp_path / "sub" / "l.jsonl"
    write(src, [ev("e1", "match", p="a"), ev("e2", "not_match", p="b")])
    res = CliRunner().invoke(app, ["labels", "--events", str(src), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "pos=1" in res.output and "neg=1" in res.output and "conflicts=0" in res.output
    assert len(out.read_text().splitlines()) == 2
    assert (tmp_path / "sub" / "l.jsonl.report.json").exists()


def test_cli_conflicting_duplicate_exits_1_and_writes_nothing(tmp_path):
    src, out = tmp_path / "e.jsonl", tmp_path / "l.jsonl"
    write(src, [ev("e1", "match"), ev("e1", "not_match")])
    res = CliRunner().invoke(app, ["labels", "--events", str(src), "--out", str(out)])
    assert res.exit_code == 1
    assert "different content" in res.output
    assert not out.exists()
