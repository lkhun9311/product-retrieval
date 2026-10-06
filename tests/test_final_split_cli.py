"""Test-split access (--final) for `pr cand-stats` and `pr rerank --version v2`, and `pr compare-rankings`."""

import io
import json

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.core.schemas import Label
from product_retrieval.eval.bootstrap import paired_bootstrap
from product_retrieval.eval.retrieval import QueryResult, recall_at_k
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.cand_stats import cand_stats_path, run_cand_stats
from product_retrieval.pipelines.evaluate import run_eval

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


def _log(reports):
    path = reports / "test_access.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def env(tmp_path):
    data_root = tmp_path / "data"
    rows = []
    for i in range(12):
        split = "test" if i < 6 else "train"
        gallery = [_put_image(data_root, (20 * i + 5, 7 * j, 3 * i + j)) for j in range(1 + i % 3)]
        query = [_put_image(data_root, (20 * i + 6, 9 + i, 4 * i))]
        rows.append(
            {"source": SOURCE, "product_id": f"p{i}", "split": split, "query": query, "gallery": gallery}
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
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
    # train side (open): index, rankings, cand-stats, labels, model
    run_build_index(config, split="train", embedder_name="fake", artifacts_root=art, reports_root=rep)
    evt = run_eval(config, split="train", embedder_name="fake", artifacts_root=art, reports_root=rep)
    cst = run_cand_stats(config, "train", evt.rankings_path, "fake", art, reports_root=rep)
    rankings = [json.loads(line) for line in evt.rankings_path.read_text().splitlines()]
    labels = []
    for r in rankings:
        ids = r["top_k_product_ids"]
        labels.append(
            Label(query_id=r["query_id"], product_id=ids[0], kind="pos", weight=1.0, label_version="lv")
        )
        labels.append(
            Label(query_id=r["query_id"], product_id=ids[-1], kind="neg", weight=1.0, label_version="lv")
        )
    lpath = tmp_path / "labels.jsonl"
    lpath.write_text("".join(lb.model_dump_json() + "\n" for lb in labels), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["train-rerank", "--labels", str(lpath), "--rankings", str(evt.rankings_path)]
        + ["--out-root", str(tmp_path / "rr"), "--version", "v2", "--config", str(cfg), "--split", "train"]
        + ["--cand-stats", str(cst.out_path), "--embedder", "fake", "--artifacts-root", str(art)],
    )
    assert res.exit_code == 0, res.output
    model = next((tmp_path / "rr").glob("*/model.json"))
    # test side: index + rankings through the logged --final path
    run_build_index(
        config, split="test", final=True, embedder_name="fake", artifacts_root=art, reports_root=rep
    )
    evx = run_eval(
        config, split="test", final=True, embedder_name="fake", artifacts_root=art, reports_root=rep
    )
    n_logged = len(_log(rep))
    assert n_logged == 2
    return {
        "config": config, "cfg": cfg, "art": art, "rep": rep, "evx": evx, "model": model,
        "tmp": tmp_path, "n_logged": n_logged, "lpath": lpath, "evt": evt, "cst": cst,
    }  # fmt: skip


def _cs_args(e, *extra):
    return [
        "cand-stats", "--config", str(e["cfg"]), "--split", "test", "--rankings", str(e["evx"].rankings_path),
        "--embedder", "fake", "--artifacts-root", str(e["art"]), "--reports-root", str(e["rep"]), *extra,
    ]  # fmt: skip


def test_cand_stats_refuses_test_without_final(env):
    res = CliRunner().invoke(app, _cs_args(env))
    assert res.exit_code == 1 and "--final" in res.output
    assert not cand_stats_path(env["evx"].rankings_path).exists()
    assert len(_log(env["rep"])) == env["n_logged"]


def test_cand_stats_allows_test_with_final_and_logs(env):
    res = CliRunner().invoke(app, _cs_args(env, "--final"))
    assert res.exit_code == 0, res.output
    assert cand_stats_path(env["evx"].rankings_path).is_file()
    log = _log(env["rep"])
    assert len(log) == env["n_logged"] + 1
    assert log[-1]["split"] == "test" and log[-1]["command"] == "cand-stats"


def test_cand_stats_final_does_not_open_unknown_splits(env):
    args = _cs_args(env, "--final")
    args[args.index("--split") + 1] = "holdout"
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 1 and "--split" in res.output


def _rerank_args(e, out, *extra):
    cs = cand_stats_path(e["evx"].rankings_path)
    return [
        "rerank", "--model", str(e["model"]), "--rankings", str(e["evx"].rankings_path), "--out", str(out),
        "--version", "v2", "--config", str(e["cfg"]), "--split", "test", "--cand-stats", str(cs),
        "--embedder", "fake", "--artifacts-root", str(e["art"]), "--reports-root", str(e["rep"]), *extra,
    ]  # fmt: skip


def test_rerank_v2_test_split_needs_final_and_logs(env):
    runner = CliRunner()
    res = runner.invoke(app, ["cand-stats", *_cs_args(env, "--final")[1:]])
    assert res.exit_code == 0, res.output
    after_stats = len(_log(env["rep"]))
    out = env["tmp"] / "reranked.jsonl"
    res = runner.invoke(app, _rerank_args(env, out))
    assert res.exit_code == 1 and "--final" in res.output
    assert not out.exists()
    assert len(_log(env["rep"])) == after_stats
    res = runner.invoke(app, _rerank_args(env, out, "--final"))
    assert res.exit_code == 0, res.output
    assert len(_log(env["rep"])) == after_stats + 1
    assert _log(env["rep"])[-1]["command"] == "rerank-v2"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    src = [json.loads(line) for line in env["evx"].rankings_path.read_text().splitlines()]
    assert [r["query_id"] for r in rows] == [r["query_id"] for r in src]


def test_final_is_rejected_for_v0_v1_and_train_rerank_stays_train_only(env):
    runner = CliRunner()
    out = env["tmp"] / "o.jsonl"
    res = runner.invoke(
        app,
        [
            "rerank",
            "--model",
            str(env["model"]),
            "--rankings",
            str(env["evx"].rankings_path),
            "--out",
            str(out),
        ]
        + ["--version", "v0", "--final"],
    )
    assert res.exit_code == 1 and "only valid with --version v2" in res.output
    res = runner.invoke(
        app,
        ["train-rerank", "--labels", str(env["lpath"]), "--rankings", str(env["evt"].rankings_path)]
        + ["--out-root", str(env["tmp"] / "rr2"), "--version", "v2", "--config", str(env["cfg"])]
        + ["--split", "test", "--cand-stats", str(env["cst"].out_path), "--embedder", "fake"]
        + ["--artifacts-root", str(env["art"])],
    )
    assert res.exit_code == 1 and "--split" in res.output
    assert len(_log(env["rep"])) == env["n_logged"]


# ---- compare-rankings -----------------------------------------------------------------------


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(qid, truth, top):
    return {"query_id": qid, "truth_product_id": truth, "top_k_product_ids": top}


# products A (q1, q2) and B (q3). Baseline hits: q1@1, q2 miss@1/hit@2, q3 miss@1/@2, hit@3.
BASE = [_row("q1", "A", ["A", "X", "Y"]), _row("q2", "A", ["X", "A", "Y"]), _row("q3", "B", ["X", "Y", "B"])]
CAND = [_row("q1", "A", ["A", "X", "Y"]), _row("q2", "A", ["A", "X", "Y"]), _row("q3", "B", ["B", "X", "Y"])]


def _compare(tmp_path, base, cand, *extra):
    b, c = tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    _write(b, base)
    _write(c, cand)
    return CliRunner().invoke(app, ["compare-rankings", "--baseline", str(b), "--candidate", str(c), *extra])


def test_compare_rankings_hand_checked_delta(tmp_path):
    out = tmp_path / "cmp.json"
    res = _compare(tmp_path, BASE, CAND, "--ks", "1,2,3", "--b", "200", "--out", str(out))
    assert res.exit_code == 0, res.output
    m = json.loads(out.read_text())["macro"]
    # macro R@1: baseline A=1/2, B=0 -> 0.25 ; candidate A=1, B=1 -> 1.0 ; delta 0.75
    assert m["1"]["baseline"] == pytest.approx(0.25) and m["1"]["candidate"] == pytest.approx(1.0)
    assert m["1"]["delta"] == pytest.approx(0.75)
    # R@2: baseline A=1, B=0 -> 0.5 ; candidate 1.0 ; delta 0.5.  R@3: both 1.0, delta 0
    assert m["2"]["delta"] == pytest.approx(0.5)
    assert m["3"]["delta"] == pytest.approx(0.0) and m["3"]["p_le_0"] == pytest.approx(1.0)
    assert "+0.7500" in res.output


def test_compare_rankings_matches_paired_bootstrap(tmp_path):
    out = tmp_path / "cmp.json"
    res = _compare(tmp_path, BASE, CAND, "--ks", "1,5", "--b", "300", "--seed", "7", "--out", str(out))
    assert res.exit_code == 0, res.output
    a = [QueryResult(r["query_id"], r["truth_product_id"], tuple(r["top_k_product_ids"])) for r in BASE]
    c = [QueryResult(r["query_id"], r["truth_product_id"], tuple(r["top_k_product_ids"])) for r in CAND]
    direct = paired_bootstrap(a, c, (1, 5), b=300, seed=7)
    point = recall_at_k(a, (1, 5))
    got = json.loads(out.read_text())["macro"]
    for k in (1, 5):
        assert got[str(k)]["delta"] == direct["macro"][k]["delta"]
        assert got[str(k)]["ci_low"] == direct["macro"][k]["ci"][0]
        assert got[str(k)]["ci_high"] == direct["macro"][k]["ci"][1]
        assert got[str(k)]["p_le_0"] == direct["macro"][k]["p_le_0"]
        assert got[str(k)]["baseline"] == point["macro"][k]


def test_compare_rankings_mismatch_fails(tmp_path):
    res = _compare(tmp_path, BASE, CAND[:2])  # candidate misses q3
    assert res.exit_code == 1 and "query sets differ" in res.output
    other_truth = [*CAND[:2], _row("q3", "A", ["B", "X", "Y"])]
    res = _compare(tmp_path, BASE, other_truth)
    assert res.exit_code == 1 and "truth_product_id differs" in res.output
    res = _compare(tmp_path, BASE, [*CAND[:2], {"query_id": "q3", "top_k_product_ids": ["B"]}])
    assert res.exit_code == 1 and "missing field" in res.output
    res = _compare(tmp_path, BASE, [*CAND, CAND[0]])
    assert res.exit_code == 1 and "duplicate query_id" in res.output
    res = _compare(tmp_path, BASE, [*CAND[:2], _row("q3", "B", ["B", "B"])])
    assert res.exit_code == 1 and "duplicates" in res.output
    res = _compare(tmp_path, BASE, CAND, "--ks", "0")
    assert res.exit_code == 1
