import io
import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.core.schemas import Label
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.index.flat import build_index
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.cand_stats import run_cand_stats
from product_retrieval.pipelines.cand_vectors import (
    CandidateVectors,
    VectorSourceError,
    load_candidate_vectors,
)
from product_retrieval.pipelines.evaluate import run_eval
from product_retrieval.pipelines.selection import select_split
from product_retrieval.rerank import v0, v1, v2
from product_retrieval.rerank.v0 import RerankError

SHA = "a" * 64
CSHA = "c" * 64
DIM = 8


def make_row(qid, n=25, top=0.9, step=0.01):
    return {
        "query_id": qid,
        "truth_product_id": "hidden",
        "top_k_product_ids": [f"{qid}-p{i}" for i in range(n)],
        "scores": [round(top - step * i, 6) for i in range(n)],
    }


def make_stats(row, sha=SHA, index_id="idx"):
    stats = []
    for i, (pid, score) in enumerate(zip(row["top_k_product_ids"][:20], row["scores"][:20], strict=True)):
        n = 1 + i % 3
        stats.append(
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
    return {"query_id": row["query_id"], "rankings_sha256": sha, "index_id": index_id, "stats": stats}


def label(qid, rank0, kind, weight=1.0):
    return Label(query_id=qid, product_id=f"{qid}-p{rank0}", kind=kind, weight=weight, label_version="lv1")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Synth:
    """Queries whose positive candidate has the query's own vector (plus noise); the rank of the
    positive cycles over 0..9, so rank, scores and stats carry no information about it."""

    def __init__(self, prefix, n_queries, sha=SHA, seed=0):
        rng = np.random.default_rng(seed)
        self.rows = [make_row(f"{prefix}{i}") for i in range(n_queries)]
        self.stats = [make_stats(r, sha) for r in self.rows]
        self.q, self.g, self.pos_rank, self.labels = {}, {}, {}, []
        for i, r in enumerate(self.rows):
            qid = r["query_id"]
            self.q[qid] = unit(rng.standard_normal(DIM))
            pos = i % 10
            self.pos_rank[qid] = pos
            for k, pid in enumerate(r["top_k_product_ids"]):
                vec = self.q[qid] + 0.1 * rng.standard_normal(DIM) if k == pos else rng.standard_normal(DIM)
                self.g[(qid, pid)] = unit(vec)
            self.labels.append(label(qid, pos, "pos"))
            for k in [(pos + 1) % 10, (pos + 3) % 10, (pos + 7) % 10]:
                self.labels.append(label(qid, k, "neg"))

    def fn(self, qid, pid):
        return self.q[qid], self.g[(qid, pid)]

    def train(self, seed=0, embed_model_id=None):
        return v2.train(
            self.labels, self.rows, SHA, self.stats, CSHA, "lv1", self.fn, "idx", seed, embed_model_id
        )


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---- input vector, vector source -------------------------------------------------------------


def test_input_vector_hand_computed():
    q, g = [0.6, 0.8], [1.0, 0.0]
    # q*g = (0.6, 0); |q-g| = (0.4, 0.8); then the features as given
    x = v2.input_vector(q, g, [1.5, -0.5])
    np.testing.assert_allclose(x, [0.6, 0.0, 0.4, 0.8, 1.5, -0.5], atol=1e-12)
    assert v2.FEATURES == v1.FEATURES and len(v2.FEATURES) == 10


def test_candidate_vectors_pick_best_image_and_normalise():
    vecs = np.array([[0, 1], [0.6, 0.8], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    gallery = build_index(vecs, ["A", "A", "A", "B", "B"], [f"s{i}" for i in range(5)])
    cv = CandidateVectors(gallery, {"q": np.array([2.0, 0.0]), "r": np.array([0.6, 0.8])}, "idx")
    q, g = cv.pair("q", "A")
    np.testing.assert_allclose(q, [1.0, 0.0], atol=1e-6)  # normalised
    np.testing.assert_allclose(g, [1.0, 0.0], atol=1e-6)  # largest q.g of A is image 3
    q, g = cv.pair("r", "A")
    np.testing.assert_allclose(g, [0.6, 0.8], atol=1e-6)
    q, g = cv.pair("r", "B")  # B: (1,0)->0.6, (0,1)->0.8
    np.testing.assert_allclose(g, [0.0, 1.0], atol=1e-6)
    # tie -> first image
    tie = build_index(np.array([[1, 0], [1, 0]], dtype=np.float32), ["T", "T"], ["s0", "s1"])
    assert CandidateVectors(tie, {"q": np.array([1.0, 0.0])}, "i").pair("q", "T")[1].tolist() == [1.0, 0.0]
    with pytest.raises(VectorSourceError, match="not in the index"):
        cv.pair("q", "ZZ")
    with pytest.raises(VectorSourceError, match="no query vector"):
        cv.pair("nope", "A")


# ---- training --------------------------------------------------------------------------------


def test_truth_field_is_not_read():
    s = Synth("q", 40)
    a = s.train()
    changed = [{**r, "truth_product_id": f"other-{i}"} for i, r in enumerate(s.rows)]
    b = v2.train(s.labels, changed, SHA, s.stats, CSHA, "lv1", s.fn, "idx", 0)
    stripped = [{k: v for k, v in r.items() if k != "truth_product_id"} for r in s.rows]
    c = v2.train(s.labels, stripped, SHA, s.stats, CSHA, "lv1", s.fn, "idx", 0)
    for m in (b, c):
        assert m["reranker_version"] == a["reranker_version"]
        assert v2._same_state(m["state_dict"], a["state_dict"])


def test_determinism_and_version(tmp_path):
    s = Synth("q", 40)
    a, b = s.train(0), s.train(0)
    assert a["reranker_version"] == b["reranker_version"] and len(a["reranker_version"]) == 12
    pa, pb = v2.save_model(a, tmp_path / "a"), v2.save_model(b, tmp_path / "b")
    assert pa.read_bytes() == pb.read_bytes()
    assert (pa.parent / "weights.pt").read_bytes() == (pb.parent / "weights.pt").read_bytes()
    other = s.train(1)
    assert other["reranker_version"] != a["reranker_version"]
    assert not v2._same_state(other["state_dict"], a["state_dict"])
    assert a["contract"] == "c5-rerank-v2" and a["input_dim"] == 2 * DIM + 10
    assert a["torch_version"] == torch.__version__ and a["index_id"] == "idx" and a["seed"] == 0
    # inputs that enter the version
    kw = (s.labels, s.rows, SHA, s.stats, CSHA, "lv1", s.fn, "idx", 0)
    assert v2.train(*kw[:4], "d" * 64, *kw[5:])["reranker_version"] != a["reranker_version"]
    assert v2.train(*kw[:5], "lv2", *kw[6:])["reranker_version"] != a["reranker_version"]
    # the global deterministic-algorithms switch is restored
    assert torch.are_deterministic_algorithms_enabled() is False


def test_holdout_split_is_deterministic_and_by_query():
    ids = [f"query-{i}" for i in range(2000)]
    flags = [v2.is_holdout(q, 0) for q in ids]
    assert flags == [v2.is_holdout(q, 0) for q in ids]
    assert 0.15 < sum(flags) / len(flags) < 0.25
    assert flags != [v2.is_holdout(q, 1) for q in ids]
    # hand check of the definition
    digest = sha256_bytes(json.dumps([3, "x"], separators=(",", ":"), sort_keys=True).encode())
    assert v2.is_holdout("x", 3) == (int(digest[:16], 16) % 100 < 20)
    # all labels of one query land on the same side
    s = Synth("q", 60)
    m = s.train()
    expected = sum(1 for lb in s.labels if v2.is_holdout(lb.query_id, 0))
    assert m["training"]["n_holdout"] == expected > 0
    assert m["training"]["n_fit"] + m["training"]["n_holdout"] == len(s.labels)


def test_early_stopping_log_is_consistent():
    m = Synth("q", 60).train()
    t = m["training"]
    assert t["mode"] == "early_stopping" and t["fallback_reason"] is None
    assert len(t["holdout_losses"]) == t["epochs_run"]
    assert t["best_epoch"] == 1 + int(np.argmin(t["holdout_losses"]))
    if t["early_stopped"]:
        assert t["epochs_run"] == t["best_epoch"] + 10
    else:
        assert t["epochs_run"] == 200


def test_fixed_50_epochs_when_holdout_is_unusable():
    # no holdout query at all
    ids = [f"f{i}" for i in range(200) if not v2.is_holdout(f"f{i}", 0)][:12]
    s = Synth("zz", 12)
    rename = {r["query_id"]: ids[i] for i, r in enumerate(s.rows)}
    s2 = Synth("zz", 12)
    rows = [
        {
            **r,
            "query_id": rename[r["query_id"]],
            "top_k_product_ids": [
                p.replace(r["query_id"], rename[r["query_id"]]) for p in r["top_k_product_ids"]
            ],
        }
        for r in s2.rows
    ]
    stats = [make_stats(r) for r in rows]
    labels = [
        Label(
            query_id=rename[lb.query_id],
            product_id=lb.product_id.replace(lb.query_id, rename[lb.query_id]),
            kind=lb.kind,
            weight=lb.weight,
            label_version="lv1",
        )
        for lb in s2.labels
    ]
    vq = {rename[k]: v for k, v in s2.q.items()}
    vg = {(rename[k[0]], k[1].replace(k[0], rename[k[0]])): v for k, v in s2.g.items()}
    m = v2.train(labels, rows, SHA, stats, CSHA, "lv1", lambda q, p: (vq[q], vg[(q, p)]), "idx", 0)
    t = m["training"]
    assert t["mode"] == "fixed_epochs" and t["epochs_run"] == 50 and t["early_stopped"] is False
    assert t["holdout_losses"] == [] and t["n_holdout"] == 0 and t["fallback_reason"]
    # holdout present but with a single class
    s3 = Synth("q", 60)
    hold_q = next(q for q in (r["query_id"] for r in s3.rows) if v2.is_holdout(q, 0))
    labels3 = [lb for lb in s3.labels if lb.query_id != hold_q or lb.kind == "neg"]
    labels3 = [lb for lb in labels3 if not v2.is_holdout(lb.query_id, 0) or lb.query_id == hold_q]
    m3 = v2.train(labels3, s3.rows, SHA, s3.stats, CSHA, "lv1", s3.fn, "idx", 0)
    assert m3["training"]["mode"] == "fixed_epochs" and "lacks" in m3["training"]["fallback_reason"]
    assert m3["training"]["epochs_run"] == 50


def test_train_failures():
    s = Synth("q", 40)

    def run(labels, stats=s.stats, rows=s.rows, index_id="idx", sha=SHA):
        return v2.train(labels, rows, sha, stats, CSHA, "lv1", s.fn, index_id, 0)

    with pytest.raises(RerankError, match="top 20"):
        run([*s.labels, label("q0", 22, "neg")])
    with pytest.raises(RerankError, match="both classes"):
        run([lb for lb in s.labels if lb.kind == "pos"])
    with pytest.raises(RerankError, match="both classes"):
        run([lb for lb in s.labels if lb.kind == "neg"])
    with pytest.raises(RerankError, match="index_id"):
        run(s.labels, index_id="other-index")
    with pytest.raises(RerankError, match="not found"):
        run([*s.labels, label("ghost", 0, "neg")])
    with pytest.raises(RerankError, match="duplicate label"):
        run([*s.labels, s.labels[0]])
    with pytest.raises(RerankError, match="rankings_sha256"):
        run(s.labels, sha="b" * 64)


def test_save_same_version_is_verified(tmp_path):
    m = Synth("q", 40).train()
    path = v2.save_model(m, tmp_path)
    assert v2.save_model(m, tmp_path) == path  # identical: no-op
    loaded = v2.load_model(path)
    assert v2._same_state(loaded["state_dict"], m["state_dict"])
    tampered = {**m, "state_dict": {k: v + 1 for k, v in m["state_dict"].items()}}
    with pytest.raises(RerankError, match="different content"):
        v2.save_model(tampered, tmp_path)
    with pytest.raises(RerankError, match="different content"):
        v2.save_model({**m, "n_rows": 1}, tmp_path)
    (path.parent / "weights.pt").unlink()
    with pytest.raises(RerankError, match="incomplete"):
        v2.save_model(m, tmp_path)
    with pytest.raises(RerankError, match="weights.pt"):
        v2.load_model(path)
    # a v1 model.json is not a v2 model
    bad = tmp_path / "bad" / "model.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps({"contract": "c5-rerank-v1", "features": list(v1.FEATURES)}))
    with pytest.raises(RerankError, match="not a c5-rerank-v2"):
        v2.load_model(bad)


# ---- rerank ----------------------------------------------------------------------------------


def test_rerank_keeps_tail_and_id_set_and_stable_ties():
    s = Synth("q", 40)
    m = s.train()
    out = v2.rerank_rows(m, s.rows, s.stats, SHA, s.fn)
    for row, new in zip(s.rows, out, strict=True):
        assert new["top_k_product_ids"][20:] == row["top_k_product_ids"][20:]
        assert new["scores"][20:] == row["scores"][20:]
        assert sorted(new["top_k_product_ids"]) == sorted(row["top_k_product_ids"])
        assert new["reranker_version"] == m["reranker_version"]
        assert (
            new["rerank_scores"] == sorted(new["rerank_scores"], reverse=True)
            and len(new["rerank_scores"]) == 20
        )
        assert new["truth_product_id"] == "hidden"
    # a model whose logits are all equal keeps the original order
    flat = {
        **m,
        "state_dict": {
            k: (torch.zeros_like(v) if k == "3.weight" else v) for k, v in m["state_dict"].items()
        },
    }
    same = v2.rerank_rows(flat, s.rows, s.stats, SHA, s.fn)
    assert [r["top_k_product_ids"] for r in same] == [r["top_k_product_ids"] for r in s.rows]
    # short ranking (< 20) and index mismatch
    short = make_row("s", n=5)
    short_stats = make_stats(short)
    n = v2.rerank_rows(m, [short], [short_stats], SHA, lambda q, p: (np.ones(DIM) / 3, np.ones(DIM) / 3))
    assert sorted(n[0]["top_k_product_ids"]) == sorted(short["top_k_product_ids"])
    # Another split's gallery has its own index: reranking with it is valid ...
    v2.rerank_rows(m, s.rows, [make_stats(r, index_id="x") for r in s.rows], SHA, s.fn)
    # ... but cand-stats rows that disagree on the index are not.
    mixed = [make_stats(r, index_id="x" if i else "y") for i, r in enumerate(s.rows)]
    with pytest.raises(RerankError, match="index_id"):
        v2.rerank_rows(m, s.rows, mixed, SHA, s.fn)


def test_rerank_refuses_another_embedding_model():
    """Galleries may differ between splits, the embedding space may not (CLAUDE.md: no version mixing)."""
    s = Synth("q", 40)
    m = {**s.train(), "embed_model_id": "model-a"}
    v2.rerank_rows(m, s.rows, s.stats, SHA, s.fn, "model-a")
    with pytest.raises(RerankError, match="embedding model"):
        v2.rerank_rows(m, s.rows, s.stats, SHA, s.fn, "model-b")
    with pytest.raises(RerankError, match="embedding model"):
        v2.rerank_rows(m, s.rows, s.stats, SHA, s.fn, None)


def test_embed_model_id_is_recorded_and_hashed():
    s = Synth("q", 40)
    a, b = s.train(embed_model_id="model-a"), s.train(embed_model_id="model-b")
    assert a["embed_model_id"] == "model-a"
    assert a["reranker_version"] != b["reranker_version"]


def test_vector_content_alone_lets_v2_move_the_positive_up_but_not_v1():
    train = Synth("t", 120, seed=1)
    m2 = train.train()
    m1 = v1.train(train.labels, train.rows, SHA, train.stats, CSHA, "lv1")
    new = Synth("n", 60, sha="e" * 64, seed=2)  # fresh queries, own rankings sha

    def positive_at_top(rows):
        return sum(
            r["top_k_product_ids"][0] == f"{r['query_id']}-p{new.pos_rank[r['query_id']]}" for r in rows
        )

    after2 = v2.rerank_rows(m2, new.rows, new.stats, "e" * 64, new.fn)
    fn1 = v1._feature_fn(new.rows, "e" * 64, new.stats)
    after1 = v0.rerank_rows(m1, new.rows, fn1)
    before = positive_at_top(new.rows)
    assert before == 6  # only the queries whose positive already was rank 1 (pos rank 0)
    assert positive_at_top(after2) >= 0.8 * len(new.rows)
    assert positive_at_top(after1) <= 0.4 * len(new.rows)  # rank/score/stats carry no label signal


# ---- CLI with the fake embedder --------------------------------------------------------------

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
    for i in range(12):
        split = "val" if i < 6 else "train"
        gallery = [_put_image(data_root, (20 * i + 5, 7 * j, 3 * i + j)) for j in range(1 + i % 3)]
        query = [_put_image(data_root, (20 * i + 6, 9 + i, 4 * i))]
        rows.append(
            {"source": SOURCE, "product_id": f"p{i}", "split": split, "query": query, "gallery": gallery}
        )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
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
    run_build_index(config, split="val", embedder_name="fake", artifacts_root=art, reports_root=rep)
    ev = run_eval(config, split="val", embedder_name="fake", artifacts_root=art, reports_root=rep)
    cs = run_cand_stats(config, "val", ev.rankings_path, "fake", art)
    # The reranker trains on the train split only (contract §3); val is for reranking.
    run_build_index(config, split="train", embedder_name="fake", artifacts_root=art, reports_root=rep)
    evt = run_eval(config, split="train", embedder_name="fake", artifacts_root=art, reports_root=rep)
    cst = run_cand_stats(config, "train", evt.rankings_path, "fake", art)
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
    return config, cfg, art, ev, cs, lpath, tmp_path, evt, cst


def test_load_candidate_vectors_matches_independent_computation(env):
    config, cfg, art, ev, cs, lpath, tmp_path, _, _ = env
    rankings = [json.loads(line) for line in ev.rankings_path.read_text().splitlines()]
    qids = [r["query_id"] for r in rankings]
    cv = load_candidate_vectors(config, "val", qids, ev.index_id, "fake", art)
    assert cv.index_id == ev.index_id
    emb = FakeEmbedder()

    def vec(sha):
        path = config.data_root / SOURCE / "images" / sha[:2] / f"{sha}.img"
        return emb.embed([Image.open(path).convert("RGB")])[0].astype(np.float64)

    manifest = [json.loads(line) for line in config.manifest_path.read_text().splitlines()]
    gallery = {r["product_id"]: np.stack([vec(x) for x in r["gallery"]]) for r in manifest}
    qshas = {q.query_id: q.image_sha for q in select_split(config, "val", None, False, tmp_path).queries}
    for r in rankings:
        q = vec(qshas[r["query_id"]])
        for pid in r["top_k_product_ids"][:20]:
            gq, gg = cv.pair(r["query_id"], pid)
            best = gallery[pid][int(np.argmax(gallery[pid] @ q))]
            np.testing.assert_allclose(gq, q, atol=1e-6)
            np.testing.assert_allclose(gg, best, atol=1e-6)
    with pytest.raises(VectorSourceError, match="differs from the cand-stats index_id"):
        load_candidate_vectors(config, "val", qids, "not-this-index", "fake", art)
    with pytest.raises(VectorSourceError, match="not in split"):
        load_candidate_vectors(config, "val", ["ghost"], ev.index_id, "fake", art)


def test_cli_v2_happy_and_error_paths(env):
    config, cfg, art, ev, cs, lpath, tmp_path, evt, cst = env
    runner = CliRunner()
    out_root = tmp_path / "rr"
    base = ["train-rerank", "--labels", str(lpath), "--rankings", str(evt.rankings_path)]
    base += ["--out-root", str(out_root), "--version", "v2"]
    good = [*base, "--config", str(cfg), "--split", "train", "--cand-stats", str(cst.out_path)]
    good += ["--embedder", "fake", "--artifacts-root", str(art), "--seed", "1"]
    res = runner.invoke(app, good)
    assert res.exit_code == 0, res.output
    assert "reranker_version=" in res.output
    model = next(out_root.glob("*/model.json"))
    assert (model.parent / "weights.pt").is_file()
    assert json.loads(model.read_text())["seed"] == 1
    out = tmp_path / "out.jsonl"
    rr = ["rerank", "--model", str(model), "--rankings", str(ev.rankings_path), "--out", str(out)]
    rr += ["--version", "v2", "--config", str(cfg), "--split", "val", "--cand-stats", str(cs.out_path)]
    rr += ["--embedder", "fake", "--artifacts-root", str(art)]
    res = runner.invoke(app, rr)
    assert res.exit_code == 0, res.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    src = [json.loads(line) for line in ev.rankings_path.read_text().splitlines()]
    assert len(rows) == len(src)
    assert all(
        sorted(a["top_k_product_ids"]) == sorted(b["top_k_product_ids"])
        for a, b in zip(rows, src, strict=True)
    )
    assert all("reranker_version" in r for r in rows)
    assert runner.invoke(app, good).exit_code == 0  # same version again: verified, not overwritten

    def fails(args, text):
        res = runner.invoke(app, args)
        assert res.exit_code == 1 and text in res.output, res.output

    fails([*base, "--split", "train", "--cand-stats", str(cst.out_path)], "requires --config")
    fails([*base, "--config", str(cfg), "--cand-stats", str(cst.out_path)], "requires --config and --split")
    fails([*base, "--config", str(cfg), "--split", "train"], "requires --cand-stats")
    # validation data never enters v2 training (contract §3)
    fails(
        [
            *base,
            "--config",
            str(cfg),
            "--split",
            "val",
            "--cand-stats",
            str(cs.out_path),
            "--embedder",
            "fake",
        ],
        "train split only",
    )
    fails([*base, "--config", str(cfg), "--split", "test", "--cand-stats", str(cs.out_path)], "--split")
    fails(
        [
            *base,
            "--config",
            str(cfg),
            "--split",
            "train",
            "--cand-stats",
            str(cst.out_path),
            "--embedder",
            "x",
        ],
        "--embedder",
    )
    no_art = ["--artifacts-root", str(tmp_path / "none"), "--embedder", "fake"]
    fails(
        [*base, "--config", str(cfg), "--split", "train", "--cand-stats", str(cst.out_path), *no_art],
        "build-index",
    )
    # cand-stats written for another index
    bad = tmp_path / "bad.cand_stats.jsonl"
    write_jsonl(bad, [{**json.loads(x), "index_id": "other"} for x in cst.out_path.read_text().splitlines()])
    fails(
        [
            *base,
            "--config",
            str(cfg),
            "--split",
            "train",
            "--cand-stats",
            str(bad),
            "--embedder",
            "fake",
            "--artifacts-root",
            str(art),
        ],
        "differs from the cand-stats index_id",
    )
    # v0/v1 reject the v2-only options
    plain = [
        "train-rerank",
        "--labels",
        str(lpath),
        "--rankings",
        str(ev.rankings_path),
        "--out-root",
        str(tmp_path / "z"),
    ]
    fails([*plain, "--version", "v0", "--config", str(cfg)], "only valid with --version v2")
    fails([*plain, "--version", "v0", "--seed", "3"], "only valid with --version v2")
    fails(
        [*plain, "--version", "v1", "--cand-stats", str(cs.out_path), "--split", "val"],
        "only valid with --version v2",
    )
    assert not (tmp_path / "z").exists()
    # a v2 model through the v1 path fails
    fails(
        [
            "rerank",
            "--model",
            str(model),
            "--rankings",
            str(ev.rankings_path),
            "--out",
            str(out),
            "--version",
            "v1",
            "--cand-stats",
            str(cs.out_path),
        ],
        "not a",
    )
