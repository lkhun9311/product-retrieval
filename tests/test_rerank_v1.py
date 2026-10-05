import json
import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_file
from product_retrieval.core.schemas import Label
from product_retrieval.rerank import v0, v1
from product_retrieval.rerank.v0 import HYPERPARAMS, RerankError, logits

SHA = "a" * 64
CSHA = "c" * 64


def make_row(qid, n=25, top=0.9, step=0.01, cliff_after=None):
    return {
        "query_id": qid,
        "truth_product_id": "hidden",
        "top_k_product_ids": [f"{qid}-p{i}" for i in range(n)],
        "scores": [
            round(top - step * i - (0.2 if cliff_after is not None and i > cliff_after else 0.0), 6)
            for i in range(n)
        ],
    }


def make_stats(row, sha=SHA, n_images=lambda i: 1 + i % 3, top1_sim=lambda i: 1.0 - 0.02 * i):
    stats = []
    for i, (pid, score) in enumerate(zip(row["top_k_product_ids"][:20], row["scores"][:20], strict=True)):
        n = n_images(i)
        stats.append(
            {
                "product_id": pid,
                "n_images": n,
                "max_sim": score,
                "mean_sim": score - 0.05 * (n - 1),
                "second_sim": score - 0.03 * (n - 1),
                "std_sim": 0.01 * (n - 1),
                "top1_sim": top1_sim(i),
            }
        )
    return {"query_id": row["query_id"], "rankings_sha256": sha, "index_id": "idx", "stats": stats}


def label(qid, rank0, kind, weight=1.0):
    return Label(query_id=qid, product_id=f"{qid}-p{rank0}", kind=kind, weight=weight, label_version="lv1")


def synthetic(n_queries=30):
    """Positive at index 4 (before a score cliff); negatives at 0, 1, 6. Stats are query-independent."""
    rows = [make_row(f"q{i}", cliff_after=4) for i in range(n_queries)]
    labels = []
    for i in range(n_queries):
        labels += [label(f"q{i}", 4, "pos"), label(f"q{i}", 0, "neg"), label(f"q{i}", 1, "neg")]
        labels.append(label(f"q{i}", 6, "neg"))
    return rows, [make_stats(r) for r in rows], labels


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_features_hand_computed():
    # rankings scores 0.9, 0.7 (2 candidates). v0 columns are covered in test_rerank_v0.
    # candidate 1: n_images=3 -> log_n = ln 4 = 1.3862944; mean_sim 0.6; max 0.9, second 0.75 ->
    #   gap_second = 0.15; std 0.1; top1_sim 1.0
    # candidate 2: n_images=1 -> log_n = ln 2 = 0.6931472; mean 0.7; max 0.7 = second -> gap_second 0;
    #   std 0; top1_sim 0.8
    row = {"query_id": "q", "top_k_product_ids": ["a", "b"], "scores": [0.9, 0.7]}
    stats = [
        {
            "product_id": "a",
            "n_images": 3,
            "max_sim": 0.9,
            "mean_sim": 0.6,
            "second_sim": 0.75,
            "std_sim": 0.1,
            "top1_sim": 1.0,
        },
        {
            "product_id": "b",
            "n_images": 1,
            "max_sim": 0.7,
            "mean_sim": 0.7,
            "second_sim": 0.7,
            "std_sim": 0.0,
            "top1_sim": 0.8,
        },
    ]
    x = v1.features_v1(row, stats)
    assert v1.FEATURES == v0.FEATURES + ("log_n_images", "mean_sim", "gap_second", "std_sim", "top1_sim")
    assert x.shape == (2, 10)
    np.testing.assert_allclose(x[:, :5], v0.features_for_row(row), atol=1e-12)
    np.testing.assert_allclose(x[0, 5:], [math.log(4), 0.6, 0.15, 0.1, 1.0], atol=1e-12)
    np.testing.assert_allclose(x[1, 5:], [math.log(2), 0.7, 0.0, 0.0, 0.8], atol=1e-12)


def test_features_length_mismatch_fails():
    row = make_row("q", n=5)
    with pytest.raises(RerankError, match="length"):
        v1.features_v1(row, make_stats(row)["stats"][:3])


def test_train_is_deterministic_and_version_covers_cand_stats_sha():
    rows, cs, labels = synthetic()
    a = v1.train(labels, rows, SHA, cs, CSHA, "lv1")
    b = v1.train(list(reversed(labels)), list(reversed(rows)), SHA, list(reversed(cs)), CSHA, "lv1")
    assert a == b
    assert a["contract"] == "c5-rerank-v1" and a["features"] == list(v1.FEATURES)
    assert a["cand_stats_sha256"] == CSHA and len(a["coef"]) == 10
    assert v1.train(labels, rows, SHA, cs, "d" * 64, "lv1")["reranker_version"] != a["reranker_version"]
    assert v1.train(labels, rows, SHA, cs, CSHA, "lv1")["reranker_version"] == a["reranker_version"]
    # same data under v0 gets a different version (different contract / features)
    assert v0.train(labels, rows, SHA, "lv1")["reranker_version"] != a["reranker_version"]


def test_v0_outputs_unchanged_by_refactor():
    rows, _, labels = synthetic(10)
    m = v0.train(labels, rows, SHA, "lv1")
    assert m["contract"] == "c5-rerank-v0" and m["features"] == list(v0.FEATURES)
    assert "cand_stats_sha256" not in m and len(m["coef"]) == 5
    # the v0 version hash is exactly the documented parts, with no extra element
    import sklearn

    expected = v0._hash("c5-rerank-v0", "lv1", SHA, list(v0.FEATURES), v0.HYPERPARAMS, sklearn.__version__)[
        :12
    ]
    assert m["reranker_version"] == expected


def test_numpy_inference_equals_sklearn_decision_function():
    rows, cs, labels = synthetic()
    model = v1.train(labels, rows, SHA, cs, CSHA, "lv1")
    stats = {c["query_id"]: c["stats"] for c in cs}
    by_q = {r["query_id"]: v1.features_v1(r, stats[r["query_id"]]) for r in rows}
    ordered = sorted(labels, key=lambda lb: (lb.query_id, lb.product_id))
    x = np.vstack([by_q[lb.query_id][int(lb.product_id.rsplit("-p", 1)[1])] for lb in ordered])
    y = np.array([lb.kind == "pos" for lb in ordered], dtype=int)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(**HYPERPARAMS).fit(scaler.transform(x), y)
    np.testing.assert_allclose(logits(model, x), clf.decision_function(scaler.transform(x)), atol=1e-9)


def test_stats_features_change_the_model_and_ranking():
    # positives are exactly the candidates with n_images == 3; scores carry no signal
    rows = [make_row(f"q{i}", step=0.0) for i in range(20)]
    cs = [make_stats(r, n_images=lambda i: 3 if i == 7 else 1) for r in rows]
    labels = []
    for r in rows:
        q = r["query_id"]
        labels += [label(q, 7, "pos"), label(q, 0, "neg"), label(q, 1, "neg"), label(q, 2, "neg")]
    model = v1.train(labels, rows, SHA, cs, CSHA, "lv1")
    fresh = make_row("fresh", step=0.0)
    fresh_cs = make_stats(fresh, n_images=lambda i: 3 if i == 9 else 1)
    stats = {"fresh": fresh_cs["stats"]}
    out = v0.rerank_rows(model, [fresh], lambda r: v1.features_v1(r, stats[r["query_id"]]))[0]
    assert out["top_k_product_ids"][0] == "fresh-p9"
    assert out["reranker_version"] == model["reranker_version"]


def test_alignment_failures():
    rows, cs, labels = synthetic(5)
    labels = [lb for lb in labels if lb.query_id in {r["query_id"] for r in rows}]
    with pytest.raises(RerankError, match="rankings_sha256"):
        v1.train(labels, rows, "b" * 64, cs, CSHA, "lv1")
    with pytest.raises(RerankError, match="query_ids differ"):
        v1.train(labels, rows, SHA, cs[:-1], CSHA, "lv1")
    extra = cs + [make_stats(make_row("zz"))]
    with pytest.raises(RerankError, match="query_ids differ"):
        v1.train(labels, rows, SHA, extra, CSHA, "lv1")
    swapped = [dict(c) for c in cs]
    s = list(swapped[0]["stats"])
    s[0], s[1] = s[1], s[0]
    swapped[0]["stats"] = s
    with pytest.raises(RerankError, match="candidates differ"):
        v1.train(labels, rows, SHA, swapped, CSHA, "lv1")


def test_cand_stats_loader_rejects_bad_files(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"query_id": "q", "stats": []}) + "\n", encoding="utf-8")
    with pytest.raises(RerankError, match="rankings_sha256"):
        v1.load_cand_stats(p)
    row = make_stats(make_row("q"))
    write_jsonl(p, [row, row])
    with pytest.raises(RerankError, match="duplicate"):
        v1.load_cand_stats(p)


def test_v1_model_is_not_loadable_as_v0_and_back(tmp_path):
    rows, cs, labels = synthetic(10)
    m1 = v1.train(labels, rows, SHA, cs, CSHA, "lv1")
    m0 = v0.train(labels, rows, SHA, "lv1")
    p1, p0 = v0.save_model(m1, tmp_path / "a"), v0.save_model(m0, tmp_path / "b")
    assert v1.load_model(p1) == m1
    with pytest.raises(RerankError):
        v0.load_model(p1)
    with pytest.raises(RerankError):
        v1.load_model(p0)


def _files(tmp_path, n=10):
    rows, cs, labels = synthetic(n)
    rk, lb, st = tmp_path / "r.rankings.jsonl", tmp_path / "l.jsonl", tmp_path / "r.rankings.cand_stats.jsonl"
    write_jsonl(rk, rows)
    sha = sha256_file(rk)
    write_jsonl(st, [{**c, "rankings_sha256": sha} for c in cs])
    lb.write_text("".join(x.model_dump_json() + "\n" for x in labels), encoding="utf-8")
    return rk, lb, st


def test_run_train_and_rerank_files_are_deterministic(tmp_path):
    rk, lb, st = _files(tmp_path)
    a = v1.run_train(lb, rk, st, tmp_path / "art")
    b = v1.run_train(lb, rk, st, tmp_path / "art")  # identical content: no-op
    assert a["reranker_version"] == b["reranker_version"] and a["cand_stats_sha256"] == sha256_file(st)
    out1, out2 = tmp_path / "o1.jsonl", tmp_path / "o2.jsonl"
    assert v1.run_rerank(a["path"], rk, st, out1) == 10
    v1.run_rerank(a["path"], rk, st, out2)
    assert out1.read_bytes() == out2.read_bytes()
    first = json.loads(out1.read_text().splitlines()[0])
    assert len(first["top_k_product_ids"]) == 25 and "rerank_scores" in first
    # stats from another rankings file are rejected at inference time
    other = tmp_path / "other.jsonl"
    write_jsonl(other, [make_row("q0")])
    with pytest.raises(RerankError):
        v1.run_rerank(a["path"], other, st, tmp_path / "o3.jsonl")
    assert not (tmp_path / "o3.jsonl").exists()


def test_cli_v1_happy_and_error_paths(tmp_path):
    rk, lb, st = _files(tmp_path)
    runner = CliRunner()
    art = tmp_path / "art"
    res = runner.invoke(
        app,
        ["train-rerank", "--labels", str(lb), "--rankings", str(rk), "--out-root", str(art)]
        + ["--version", "v1", "--cand-stats", str(st)],
    )
    assert res.exit_code == 0, res.output
    assert "reranker_version=" in res.output
    model = next(art.glob("*/model.json"))
    out = tmp_path / "out.jsonl"
    res = runner.invoke(
        app,
        ["rerank", "--model", str(model), "--rankings", str(rk), "--out", str(out)]
        + ["--version", "v1", "--cand-stats", str(st)],
    )
    assert res.exit_code == 0, res.output
    assert len(out.read_text().splitlines()) == 10

    base_t = ["train-rerank", "--labels", str(lb), "--rankings", str(rk), "--out-root", str(tmp_path / "x")]
    res = runner.invoke(app, [*base_t, "--version", "v1"])
    assert res.exit_code == 1 and "requires --cand-stats" in res.output
    res = runner.invoke(app, [*base_t, "--version", "v0", "--cand-stats", str(st)])
    assert res.exit_code == 1 and "only valid with --version v1" in res.output
    res = runner.invoke(app, [*base_t, "--version", "v9"])
    assert res.exit_code == 1 and "unknown --version" in res.output
    res = runner.invoke(app, [*base_t, "--version", "v1", "--cand-stats", str(tmp_path / "missing.jsonl")])
    assert res.exit_code == 1 and "does not exist" in res.output
    # v0 model with --version v1 fails; v1 model with default v0 fails
    res = runner.invoke(app, ["rerank", "--model", str(model), "--rankings", str(rk), "--out", str(out)])
    assert res.exit_code == 1
    # mismatched stats (sha of another rankings file)
    bad = tmp_path / "bad.cand_stats.jsonl"
    write_jsonl(
        bad, [{**json.loads(line), "rankings_sha256": "0" * 64} for line in st.read_text().splitlines()]
    )
    res = runner.invoke(
        app,
        ["rerank", "--model", str(model), "--rankings", str(rk), "--out", str(out)]
        + ["--version", "v1", "--cand-stats", str(bad)],
    )
    assert res.exit_code == 1 and "rankings_sha256" in res.output
    assert not (tmp_path / "x").exists()
