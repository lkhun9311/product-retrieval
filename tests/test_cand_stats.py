import io
import json
import math

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes, sha256_file
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.index.flat import build_index
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.cand_stats import (
    CandStatsError,
    cand_stats_path,
    run_cand_stats,
    stats_for_queries,
)
from product_retrieval.pipelines.evaluate import IndexNotFoundError, run_eval
from product_retrieval.pipelines.selection import select_split

SOURCE = "lrvs"
SQ = math.sqrt


def _hand_index():
    # 2-d unit vectors. Product A has three images, B one, C two.
    vecs = np.array(
        [
            [1.0, 0.0],  # A1
            [0.6, 0.8],  # A2
            [0.0, 1.0],  # A3
            [0.0, 1.0],  # B1
            [1.0, 0.0],  # C1
            [1.0, 0.0],  # C2 (duplicate of C1)
        ],
        dtype=np.float32,
    )
    pids = ["A", "A", "A", "B", "C", "C"]
    return build_index(vecs, pids, [f"s{i}" for i in range(6)])


def test_hand_computed_stats():
    gallery = _hand_index()
    q = np.array([1.0, 0.0], dtype=np.float32)
    # q.g: A -> 1.0, 0.6, 0.0 ; B -> 0.0 ; C -> 1.0, 1.0
    # A: n=3 max=1.0 mean=1.6/3=0.5333333 second=0.6
    #    deviations 0.4666667, 0.0666667, -0.5333333 -> squares 0.2177778, 0.0044444, 0.2844444
    #    population var = 0.5066667/3 = 0.1688889, std = 0.4109609
    # B: n=1 max=0 mean=0 second=max=0 std=0
    # C: n=2 max=1 mean=1 second=1 std=0
    # top1 = A (ranked first): mean vector (1.6/3, 1.8/3) = (0.5333333, 0.6), norm = sqrt(0.6444444)
    #    = 0.8027756, unit = (0.6643638, 0.7474093). top1_sim(A) = 1.0
    # B unit mean = (0, 1) -> top1_sim = 0.7474093 ; C unit mean = (1, 0) -> top1_sim = 0.6643638
    rows = [("q", ["A", "B", "C"], [1.0, 0.0, 1.0])]
    ((qid, stats),) = stats_for_queries(gallery, rows, {"q": q})
    assert qid == "q"
    assert [s["product_id"] for s in stats] == ["A", "B", "C"]
    a, b, c = stats
    assert a["n_images"] == 3
    assert a["max_sim"] == pytest.approx(1.0, abs=1e-6)
    assert a["mean_sim"] == pytest.approx(1.6 / 3, abs=1e-6)
    assert a["second_sim"] == pytest.approx(0.6, abs=1e-6)
    assert a["std_sim"] == pytest.approx(0.4109609, abs=1e-6)
    assert a["top1_sim"] == pytest.approx(1.0, abs=1e-6)
    assert b["n_images"] == 1
    assert (b["max_sim"], b["mean_sim"], b["second_sim"], b["std_sim"]) == (0.0, 0.0, 0.0, 0.0)
    assert b["top1_sim"] == pytest.approx(0.7474093, abs=1e-6)
    assert c["n_images"] == 2
    assert (c["max_sim"], c["mean_sim"], c["second_sim"]) == (1.0, 1.0, 1.0)
    assert c["std_sim"] == pytest.approx(0.0, abs=1e-7)
    assert c["top1_sim"] == pytest.approx(0.6643638, abs=1e-6)


def test_single_image_top1_product():
    # n = 1 for the rank-1 product: second_sim = max_sim, std = 0, top1_sim = 1.0
    gallery = _hand_index()
    ((_, stats),) = stats_for_queries(gallery, [("q", ["B"], [1.0])], {"q": np.array([0.0, 1.0])})
    s = stats[0]
    assert (s["n_images"], s["std_sim"], s["top1_sim"]) == (1, 0.0, pytest.approx(1.0, abs=1e-6))
    assert s["second_sim"] == s["max_sim"] == pytest.approx(1.0, abs=1e-6)


def test_max_sim_mismatch_with_rankings_score_fails():
    gallery = _hand_index()
    q = np.array([1.0, 0.0])
    with pytest.raises(CandStatsError, match="max_sim"):
        list(stats_for_queries(gallery, [("q", ["A"], [0.9])], {"q": q}))
    # just inside the tolerance passes
    list(stats_for_queries(gallery, [("q", ["A"], [1.0 - 5e-6])], {"q": q}))


def test_ranked_product_absent_from_index_fails():
    with pytest.raises(CandStatsError, match="not in the index"):
        list(stats_for_queries(_hand_index(), [("q", ["A", "ZZ"], [1.0, 0.0])], {"q": np.array([1.0, 0.0])}))


def test_batches_do_not_change_results():
    gallery = _hand_index()
    rows = [(f"q{i}", ["A", "B"], [1.0, 0.0]) for i in range(5)]
    qv = {f"q{i}": np.array([1.0, 0.0]) for i in range(5)}
    assert list(stats_for_queries(gallery, rows, qv, batch_size=2)) == list(
        stats_for_queries(gallery, rows, qv, batch_size=100)
    )


# ---- end to end with the fake embedder --------------------------------------------------------


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
def setup(tmp_path):
    data_root = tmp_path / "data"
    rows = []
    for i, pid in enumerate(["p1", "p2", "p3", "p4"]):
        n_gallery = 1 + (i % 3)  # 1, 2, 3, 1 gallery images
        gallery = [_put_image(data_root, (20 * i + 5, 7 * j, 3 * i + j)) for j in range(n_gallery)]
        query = [_put_image(data_root, (20 * i + 6, 9 + i, 4 * i))]
        rows.append({"source": SOURCE, "product_id": pid, "split": "val", "query": query, "gallery": gallery})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = ExperimentConfig(
        name="t",
        seed=0,
        embed_model_id="unused-for-fake",
        crop_kind="full",
        index_params={},
        manifest_path=manifest,
        data_root=data_root,
    )
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    run_build_index(config, split="val", embedder_name="fake", artifacts_root=artifacts, reports_root=reports)
    ev = run_eval(config, split="val", embedder_name="fake", artifacts_root=artifacts, reports_root=reports)
    return config, tmp_path, artifacts, ev


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_cand_stats_end_to_end(setup):
    config, tmp_path, artifacts, ev = setup
    before = sha256_file(ev.rankings_path)
    result = run_cand_stats(config, "val", ev.rankings_path, embedder_name="fake", artifacts_root=artifacts)
    assert sha256_file(ev.rankings_path) == before  # rankings file untouched
    assert result.out_path == cand_stats_path(ev.rankings_path)
    assert result.out_path.name == ev.rankings_path.name.replace(".jsonl", ".cand_stats.jsonl")
    rankings, out = _read(ev.rankings_path), _read(result.out_path)
    assert [r["query_id"] for r in out] == [r["query_id"] for r in rankings]
    for rk, row in zip(rankings, out, strict=True):
        assert row["rankings_sha256"] == before and row["index_id"] == ev.index_id
        stats = row["stats"]
        assert [s["product_id"] for s in stats] == rk["top_k_product_ids"][:20]
        assert set(stats[0]) == {
            "product_id",
            "n_images",
            "max_sim",
            "mean_sim",
            "second_sim",
            "std_sim",
            "top1_sim",
        }
        for s, score in zip(stats, rk["scores"], strict=True):
            assert abs(s["max_sim"] - score) <= 1e-5
            assert s["second_sim"] <= s["max_sim"] + 1e-12 and s["std_sim"] >= 0
        assert stats[0]["top1_sim"] == pytest.approx(1.0, abs=1e-6)  # rank 1 vs itself
        assert all(-1 - 1e-6 <= s["top1_sim"] <= 1 + 1e-6 for s in stats)
    n_images = {s["product_id"]: s["n_images"] for row in out for s in row["stats"]}
    assert n_images == {"p1": 1, "p2": 2, "p3": 3, "p4": 1}
    # single-image products: second_sim == max_sim, std == 0
    for row in out:
        for s in row["stats"]:
            if s["n_images"] == 1:
                assert s["second_sim"] == s["max_sim"] and s["std_sim"] == 0.0


def test_cand_stats_match_independent_computation(setup):
    config, tmp_path, artifacts, ev = setup
    out = _read(run_cand_stats(config, "val", ev.rankings_path, "fake", artifacts).out_path)
    # independent path: embed the images straight from disk and brute-force per product
    embedder = FakeEmbedder()

    def vec(sha):
        path = config.data_root / SOURCE / "images" / sha[:2] / f"{sha}.img"
        return embedder.embed([Image.open(path).convert("RGB")])[0].astype(np.float64)

    manifest = [json.loads(line) for line in config.manifest_path.read_text().splitlines()]
    gallery = {r["product_id"]: np.stack([vec(s) for s in r["gallery"]]) for r in manifest}
    qshas = {q.query_id: q.image_sha for q in select_split(config, "val", None, False, tmp_path).queries}
    for row in out:
        q = vec(qshas[row["query_id"]])
        top1 = gallery[row["stats"][0]["product_id"]].mean(axis=0)
        top1 /= np.linalg.norm(top1)
        for s in row["stats"]:
            g = gallery[s["product_id"]]
            sims = np.sort(g @ q)[::-1]
            mean_unit = g.mean(axis=0) / np.linalg.norm(g.mean(axis=0))
            assert s["n_images"] == len(g)
            assert s["max_sim"] == pytest.approx(sims[0], abs=1e-5)
            assert s["mean_sim"] == pytest.approx(sims.mean(), abs=1e-5)
            assert s["second_sim"] == pytest.approx(sims[1] if len(g) > 1 else sims[0], abs=1e-5)
            assert s["std_sim"] == pytest.approx(sims.std() if len(g) > 1 else 0.0, abs=1e-5)
            assert s["top1_sim"] == pytest.approx(mean_unit @ top1, abs=1e-5)


def test_truth_field_is_not_read(setup):
    config, tmp_path, artifacts, ev = setup
    a = _read(run_cand_stats(config, "val", ev.rankings_path, "fake", artifacts).out_path)
    stripped = tmp_path / "strip.rankings.jsonl"
    stripped.write_text(
        "".join(
            json.dumps({k: v for k, v in r.items() if k != "truth_product_id"}) + "\n"
            for r in _read(ev.rankings_path)
        ),
        encoding="utf-8",
    )
    b = _read(run_cand_stats(config, "val", stripped, "fake", artifacts).out_path)
    assert [r["stats"] for r in a] == [r["stats"] for r in b]
    wrong = tmp_path / "wrong.rankings.jsonl"
    wrong.write_text(
        "".join(json.dumps({**r, "truth_product_id": "nope"}) + "\n" for r in _read(ev.rankings_path)),
        encoding="utf-8",
    )
    c = _read(run_cand_stats(config, "val", wrong, "fake", artifacts).out_path)
    assert [r["stats"] for r in a] == [r["stats"] for r in c]


def test_tampered_score_fails_and_leaves_no_output(setup):
    config, tmp_path, artifacts, ev = setup
    rows = _read(ev.rankings_path)
    rows[0]["scores"][0] += 0.01
    bad = tmp_path / "bad.rankings.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(CandStatsError, match="max_sim"):
        run_cand_stats(config, "val", bad, "fake", artifacts)
    assert not cand_stats_path(bad).exists()
    assert not list(tmp_path.glob("bad*.tmp"))


def test_unknown_query_and_missing_index_fail(setup):
    config, tmp_path, artifacts, ev = setup
    rows = _read(ev.rankings_path)
    rows[0]["query_id"] = "not-a-query"
    bad = tmp_path / "unk.rankings.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(CandStatsError, match="not in split"):
        run_cand_stats(config, "val", bad, "fake", artifacts)
    with pytest.raises(IndexNotFoundError):
        run_cand_stats(config, "val", ev.rankings_path, "fake", tmp_path / "empty-artifacts")


def test_rankings_missing_field_and_duplicate_fail(setup):
    config, tmp_path, artifacts, ev = setup
    rows = _read(ev.rankings_path)
    nofield = tmp_path / "nf.rankings.jsonl"
    nofield.write_text(json.dumps({"query_id": "x", "top_k_product_ids": []}) + "\n", encoding="utf-8")
    with pytest.raises(CandStatsError, match="scores"):
        run_cand_stats(config, "val", nofield, "fake", artifacts)
    dup = tmp_path / "dup.rankings.jsonl"
    dup.write_text("".join(json.dumps(r) + "\n" for r in [rows[0], rows[0]]), encoding="utf-8")
    with pytest.raises(CandStatsError, match="duplicate"):
        run_cand_stats(config, "val", dup, "fake", artifacts)


def test_test_split_is_refused_by_cli(setup):
    config, tmp_path, artifacts, ev = setup
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "seed": 0,
                "embed_model_id": "unused-for-fake",
                "crop_kind": "full",
                "index_params": {},
                "manifest_path": str(config.manifest_path),
                "data_root": str(config.data_root),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    base = ["cand-stats", "--config", str(cfg), "--rankings", str(ev.rankings_path), "--embedder", "fake"]
    res = runner.invoke(app, [*base, "--split", "test", "--artifacts-root", str(artifacts)])
    assert res.exit_code == 1 and "--split" in res.output
    res = runner.invoke(app, [*base, "--split", "val", "--artifacts-root", str(artifacts)])
    assert res.exit_code == 0, res.output
    assert cand_stats_path(ev.rankings_path).is_file()
    res = runner.invoke(app, [*base, "--split", "val", "--artifacts-root", str(tmp_path / "none")])
    assert res.exit_code == 1 and "build-index" in res.output
    res = runner.invoke(
        app,
        [
            "cand-stats",
            "--config",
            str(cfg),
            "--split",
            "val",
            "--rankings",
            str(ev.rankings_path),
            "--embedder",
            "x",
        ],
    )
    assert res.exit_code == 2


def test_cli_forwards_limit_products(setup):
    """Rankings from `pr eval --limit-products N` need the same N to find their index."""
    config, tmp_path, _, _ = setup
    artifacts, reports = tmp_path / "artifacts_lim", tmp_path / "reports_lim"
    run_build_index(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts,
        reports_root=reports,
        limit_products=2,
    )
    ev = run_eval(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts,
        reports_root=reports,
        limit_products=2,
    )
    cfg = tmp_path / "config_lim.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "seed": 0,
                "embed_model_id": "unused-for-fake",
                "crop_kind": "full",
                "index_params": {},
                "manifest_path": str(config.manifest_path),
                "data_root": str(config.data_root),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    base = ["cand-stats", "--config", str(cfg), "--split", "val", "--rankings", str(ev.rankings_path)]
    base += ["--embedder", "fake", "--artifacts-root", str(artifacts)]
    res = runner.invoke(app, base)  # no limit: looks for the full-split index, which was never built
    assert res.exit_code == 1 and "build-index" in res.output
    res = runner.invoke(app, [*base, "--limit-products", "2"])
    assert res.exit_code == 0, res.output
    assert len(_read(cand_stats_path(ev.rankings_path))) == len(_read(ev.rankings_path))
