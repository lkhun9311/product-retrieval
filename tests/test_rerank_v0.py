import json
import math
import random

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.schemas import Label
from product_retrieval.rerank.v0 import (
    CONTRACT,
    FEATURES,
    HYPERPARAMS,
    RERANK_DEPTH,
    RerankError,
    features_for_row,
    load_model,
    logits,
    rerank_rows,
    run_rerank,
    run_train,
    save_model,
    train,
)

SHA = "a" * 64


def make_row(qid, n=25, truth="p0", step=0.01, top=0.9, cliff_after=None):
    return {
        "query_id": qid,
        "truth_product_id": truth,
        "top_k_product_ids": [f"{qid}-p{i}" for i in range(n)],
        "scores": [
            round(top - step * i - (0.2 if cliff_after is not None and i > cliff_after else 0.0), 6)
            for i in range(n)
        ],
    }


def label(qid, rank0, kind, weight=1.0, version="lv1"):
    return Label(query_id=qid, product_id=f"{qid}-p{rank0}", kind=kind, weight=weight, label_version=version)


def synthetic(n_queries=30):
    """Positive at index 4, the last before a score cliff (large gap_next); negatives at 0, 1, 6."""
    rows = [make_row(f"q{i}", truth=f"q{i}-p4", cliff_after=4) for i in range(n_queries)]
    labels = []
    for i in range(n_queries):
        labels += [
            label(f"q{i}", 4, "pos"),
            label(f"q{i}", 0, "neg"),
            label(f"q{i}", 1, "neg"),
            label(f"q{i}", 6, "neg"),
        ]
    return rows, labels


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_features_hand_computed():
    # scores 0.9, 0.7, 0.5 (3 candidates, so there is no s_21 and the last gap_next is 0)
    # score:    0.9, 0.7, 0.5
    # gap_top1: 0, -0.2, -0.4
    # gap_next: 0.9-0.7=0.2, 0.7-0.5=0.2, 0
    # mean = 0.7; deviations 0.2, 0, -0.2; population var = 0.08/3, std = sqrt(0.08/3) = 0.1632993
    # zscore:   0.2/std = 1.2247449, 0, -1.2247449
    # log_rank: ln1 = 0, ln2 = 0.6931472, ln3 = 1.0986123
    row = {
        "query_id": "q",
        "truth_product_id": "x",
        "top_k_product_ids": ["a", "b", "c"],
        "scores": [0.9, 0.7, 0.5],
    }
    z = math.sqrt(1.5)
    expected = np.array(
        [
            [0.9, 0.0, 0.2, z, 0.0],
            [0.7, -0.2, 0.2, 0.0, math.log(2)],
            [0.5, -0.4, 0.0, -z, math.log(3)],
        ]
    )
    assert FEATURES == ("score", "gap_top1", "gap_next", "zscore", "log_rank")
    np.testing.assert_allclose(features_for_row(row), expected, atol=1e-12)


def test_gap_next_at_rank_20_uses_s21_and_zero_without():
    with_21 = make_row("q", n=21)
    x = features_for_row(with_21)
    assert x.shape == (20, 5)
    assert x[19, 2] == pytest.approx(with_21["scores"][19] - with_21["scores"][20])
    exactly_20 = features_for_row(make_row("q", n=20))
    assert exactly_20[19, 2] == 0.0


def test_zscore_is_zero_for_constant_scores():
    row = make_row("q", n=5, step=0.0)
    assert np.all(features_for_row(row)[:, 3] == 0.0)


def test_features_do_not_read_truth_field():
    row = make_row("q")
    row2 = {**row, "truth_product_id": "other"}
    del row2["truth_product_id"]
    np.testing.assert_array_equal(features_for_row(row), features_for_row(row2))


def test_scores_misaligned_fails():
    row = make_row("q", n=5)
    row["scores"] = row["scores"][:3]
    with pytest.raises(RerankError, match="scores"):
        features_for_row(row)


def test_train_is_deterministic():
    rows, labels = synthetic()
    a = train(labels, rows, SHA, "lv1")
    b = train(list(reversed(labels)), list(reversed(rows)), SHA, "lv1")
    assert a == b
    assert len(a["reranker_version"]) == 12
    assert (a["n_pos"], a["n_neg"], a["n_rows"]) == (30, 90, 120)
    assert a["contract"] == CONTRACT
    assert train(labels, rows, "b" * 64, "lv1")["reranker_version"] != a["reranker_version"]
    assert train(labels, rows, SHA, "lv2")["reranker_version"] != a["reranker_version"]


def test_truth_field_does_not_change_model_or_output():
    rows, labels = synthetic()
    changed = [{**r, "truth_product_id": "something-else"} for r in rows]
    m1, m2 = train(labels, rows, SHA, "lv1"), train(labels, changed, SHA, "lv1")
    assert m1 == m2
    out1 = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in rerank_rows(m1, rows)]
    out2 = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in rerank_rows(m1, changed)]
    assert out1 == out2


def test_label_outside_top_20_fails():
    rows = [make_row("q0")]
    labels = [label("q0", 20, "pos"), label("q0", 0, "neg")]  # index 20 is rank 21
    with pytest.raises(RerankError, match="top 20"):
        train(labels, rows, SHA, "lv1")


def test_label_for_unknown_query_fails():
    with pytest.raises(RerankError, match="not found"):
        train([label("zz", 0, "pos"), label("q0", 0, "neg")], [make_row("q0")], SHA, "lv1")


@pytest.mark.parametrize("kind", ["pos", "neg"])
def test_single_class_fails(kind):
    rows = [make_row("q0")]
    with pytest.raises(RerankError, match="both classes"):
        train([label("q0", 0, kind), label("q0", 1, kind)], rows, SHA, "lv1")


def test_empty_labels_fail():
    with pytest.raises(RerankError):
        train([], [make_row("q0")], SHA, "lv1")


def test_duplicate_label_pair_fails():
    rows = [make_row("q0")]
    with pytest.raises(RerankError, match="duplicate label"):
        train([label("q0", 0, "pos"), label("q0", 0, "neg")], rows, SHA, "lv1")


def test_numpy_inference_equals_sklearn_decision_function():
    rows, labels = synthetic()
    model = train(labels, rows, SHA, "lv1")
    # refit independently with the contract settings and compare on the training rows
    by_q = {r["query_id"]: features_for_row(r) for r in rows}
    ordered = sorted(labels, key=lambda lb: (lb.query_id, lb.product_id))
    x = np.vstack([by_q[lb.query_id][int(lb.product_id.rsplit("-p", 1)[1])] for lb in ordered])
    y = np.array([lb.kind == "pos" for lb in ordered], dtype=int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(**HYPERPARAMS).fit(
        scaler.transform(x), y, sample_weight=[lb.weight for lb in ordered]
    )
    np.testing.assert_allclose(logits(model, x), clf.decision_function(scaler.transform(x)), atol=1e-9)


def test_weights_are_used():
    rows, labels = synthetic()
    light = [lb.model_copy(update={"weight": 0.2}) if lb.kind == "neg" else lb for lb in labels]
    assert train(light, rows, SHA, "lv1")["coef"] != train(labels, rows, SHA, "lv1")["coef"]


def test_rerank_keeps_tail_and_id_set_and_aligns_scores():
    rows, labels = synthetic()
    model = train(labels, rows, SHA, "lv1")
    for src, out in zip(rows, rerank_rows(model, rows), strict=True):
        assert len(out["top_k_product_ids"]) == 25
        assert out["top_k_product_ids"][RERANK_DEPTH:] == src["top_k_product_ids"][RERANK_DEPTH:]
        assert out["scores"][RERANK_DEPTH:] == src["scores"][RERANK_DEPTH:]
        assert set(out["top_k_product_ids"]) == set(src["top_k_product_ids"])
        by_id = dict(zip(src["top_k_product_ids"], src["scores"], strict=True))
        assert out["scores"] == [by_id[p] for p in out["top_k_product_ids"]]
        assert len(out["rerank_scores"]) == RERANK_DEPTH
        assert out["rerank_scores"] == sorted(out["rerank_scores"], reverse=True)
        assert out["reranker_version"] == model["reranker_version"]
        assert out["truth_product_id"] == src["truth_product_id"]
    assert rows[0]["top_k_product_ids"][0] == "q0-p0"  # input not mutated


def test_rerank_handles_short_rows():
    rows, labels = synthetic()
    model = train(labels, rows, SHA, "lv1")
    short = make_row("s", n=3)
    out = rerank_rows(model, [short])[0]
    assert sorted(out["top_k_product_ids"]) == sorted(short["top_k_product_ids"])
    assert len(out["rerank_scores"]) == 3


def test_tie_order_is_stable():
    # zero coefficients make every logit equal, so the original order must survive
    model = {
        "contract": CONTRACT,
        "reranker_version": "t",
        "features": list(FEATURES),
        "coef": [0.0] * 5,
        "intercept": 0.3,
        "scaler_mean": [0.0] * 5,
        "scaler_scale": [1.0] * 5,
    }
    row = make_row("q", n=25)
    out = rerank_rows(model, [row])[0]
    assert out["top_k_product_ids"] == row["top_k_product_ids"]
    assert set(out["rerank_scores"]) == {0.3}


def test_end_to_end_moves_known_positive_up():
    rows, labels = synthetic()
    model = train(labels, rows, SHA, "lv1")
    # a fresh query with the same shape: positive sits at original rank 5 (index 4)
    fresh = make_row("fresh", truth="fresh-p4", cliff_after=4)
    out = rerank_rows(model, [fresh])[0]
    new_index = out["top_k_product_ids"].index("fresh-p4")
    assert new_index < 4
    assert new_index < out["top_k_product_ids"].index("fresh-p0")
    assert new_index < out["top_k_product_ids"].index("fresh-p1")


def test_save_model_idempotent_and_conflict(tmp_path):
    rows, labels = synthetic(5)
    model = train(labels, rows, SHA, "lv1")
    path = save_model(model, tmp_path)
    assert path == tmp_path / model["reranker_version"] / "model.json"
    before = path.read_bytes()
    assert save_model(model, tmp_path) == path
    assert path.read_bytes() == before
    assert load_model(path) == model
    with pytest.raises(RerankError, match="different content"):
        save_model({**model, "intercept": model["intercept"] + 1.0}, tmp_path)
    assert path.read_bytes() == before


def test_load_model_rejects_wrong_contract(tmp_path):
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"contract": "other"}))
    with pytest.raises(RerankError):
        load_model(bad)


def test_run_train_and_run_rerank_files(tmp_path):
    rows, labels = synthetic(10)
    random.Random(0).shuffle(rows)
    rk, lb = tmp_path / "r.jsonl", tmp_path / "l.jsonl"
    write_jsonl(rk, rows)
    lb.write_text("".join(x.model_dump_json() + "\n" for x in labels), encoding="utf-8")
    info = run_train(lb, rk, tmp_path / "art")
    assert info["n_pos"] == 10 and info["n_neg"] == 30
    out = tmp_path / "o" / "r2.jsonl"
    assert run_rerank(info["path"], rk, out) == 10
    assert len(out.read_text().splitlines()) == 10


def test_run_train_rejects_mixed_label_versions(tmp_path):
    rows, labels = synthetic(3)
    labels[0] = labels[0].model_copy(update={"label_version": "other"})
    rk, lb = tmp_path / "r.jsonl", tmp_path / "l.jsonl"
    write_jsonl(rk, rows)
    lb.write_text("".join(x.model_dump_json() + "\n" for x in labels), encoding="utf-8")
    with pytest.raises(RerankError, match="label_version"):
        run_train(lb, rk, tmp_path / "art")


def test_cli_happy_path(tmp_path):
    rows, labels = synthetic(10)
    rk, lb = tmp_path / "r.jsonl", tmp_path / "l.jsonl"
    write_jsonl(rk, rows)
    lb.write_text("".join(x.model_dump_json() + "\n" for x in labels), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(
        app, ["train-rerank", "--labels", str(lb), "--rankings", str(rk), "--out-root", str(tmp_path / "art")]
    )
    assert res.exit_code == 0, res.output
    assert "pos=10" in res.output and "neg=30" in res.output and "reranker_version=" in res.output
    model_path = next((tmp_path / "art").glob("*/model.json"))
    out = tmp_path / "out.jsonl"
    res = runner.invoke(app, ["rerank", "--model", str(model_path), "--rankings", str(rk), "--out", str(out)])
    assert res.exit_code == 0, res.output
    first = json.loads(out.read_text().splitlines()[0])
    assert "rerank_scores" in first and len(first["top_k_product_ids"]) == 25


def test_cli_error_exits_1(tmp_path):
    rows = [make_row("q0")]
    rk, lb = tmp_path / "r.jsonl", tmp_path / "l.jsonl"
    write_jsonl(rk, rows)
    lb.write_text(label("q0", 0, "pos").model_dump_json() + "\n", encoding="utf-8")
    res = CliRunner().invoke(
        app, ["train-rerank", "--labels", str(lb), "--rankings", str(rk), "--out-root", str(tmp_path / "art")]
    )
    assert res.exit_code == 1
    assert "both classes" in res.output
    assert not (tmp_path / "art").exists()
