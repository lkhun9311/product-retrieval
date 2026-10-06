import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_file
from product_retrieval.crop.fake import FakeDetector
from product_retrieval.eval import merge
from product_retrieval.pipelines import crop_search
from product_retrieval.pipelines.crop_search import CONTRACT_PATH, CropSearchError, run_crop_search
from product_retrieval.pipelines.ikea_check import IkeaCheckError

runner = CliRunner()

RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
P1, P2, P3, P4, P5 = "101.000.01", "102.000.02", "103.000.03", "104.000.04", "105.000.05"


class ColorEmbedder:
    """Mean RGB, unit length: a hand-computable stand-in for SigLIP."""

    model_id = "color-test"
    dim = 3

    def __init__(self):
        self.calls = 0

    def embed(self, images):
        self.calls += len(images)
        out = []
        for im in images:
            m = np.asarray(im, dtype=np.float64).reshape(-1, 3).mean(axis=0)
            out.append(m / np.linalg.norm(m))
        return np.stack(out).astype(np.float32)


def _save(path: Path, size, left, right=None, split=0.6):
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    im = Image.new("RGB", size, color=left)
    if right is not None:
        im.paste(right, (int(w * split), 0, w, h))
    im.save(path, format="JPEG", quality=100, subsampling=0)


def _write_mapping(raw: Path, mapping):
    (raw / "text_data").mkdir(parents=True, exist_ok=True)
    with open(raw / "text_data" / "item_to_room.p", "wb") as f:
        pickle.dump(mapping, f, protocol=2)


def _p(pid):
    return f"images/{pid}.jpg"


def _r(name):
    return f"images/room_scenes/{name}.jpg"


def _split_proposals(image):
    """Left 60% (score 0.9), right 40% (0.8), plus a below-threshold whole-image box."""
    w, h = image.size
    boxes = np.array([[0, 0, 0.6 * w, h], [0.6 * w, 0, w, h], [0, 0, w, h]], dtype=np.float64)
    return boxes, np.array([0.9, 0.8, 0.05])


def _make_raw(tmp_path: Path) -> Path:
    """Tiny IKEA-like folder. Gallery products are solid colours (P5 is listed but has no file).

    80x40 rooms, left 60% / right 40%:
      r1 red | green        lists P1, P2     whole [P4 P1 P2 P3]; crops [P1 P2 P4 P3]
      r2 solid blue         lists P3         both rank P3 first
      r3 green | blue       lists P3, P4     whole [P2 P4 P3 P1]; crops [P2 P3 P4 P1]
      r4 20x20 solid red    lists P1         crops 14 and 9 px wide after expansion (< 16): none -> fallback
      r5 copy of r2         lists P3         same cluster as r2
      r6 solid white        lists P5 only    file exists, no existing product: dropped
    """
    raw = tmp_path / "ikea"
    for pid, folder, col in ((P1, "bed", RED), (P2, "chair", GREEN), (P3, "objects", BLUE), (P4, "", YELLOW)):
        _save(raw / "images" / folder / f"{pid}.jpg", (16, 16), col)
    rooms = raw / "images" / "room_scenes"
    _save(rooms / "r1.jpg", (80, 40), RED, GREEN)
    _save(rooms / "r2.jpg", (80, 40), BLUE)
    _save(rooms / "r3.jpg", (80, 40), GREEN, BLUE)
    _save(rooms / "r4.jpg", (20, 20), RED)
    _save(rooms / "r5.jpg", (80, 40), BLUE)
    _save(rooms / "r6.jpg", (80, 40), (255, 255, 255))
    _write_mapping(
        raw,
        {
            _p(P1): [_r("r1"), _r("r4")],
            _p(P2): [_r("r1")],
            _p(P3): [_r("r2"), _r("r3"), _r("r5")],
            _p(P4): [_r("r3")],
            _p(P5): [_r("r6")],
        },
    )
    return raw


@pytest.fixture
def color(monkeypatch):
    emb = ColorEmbedder()
    monkeypatch.setattr(crop_search, "_make_embedder", lambda name: emb)
    monkeypatch.setattr(crop_search, "_make_detector", lambda name: FakeDetector(_split_proposals))
    return emb


def _run(raw, tmp_path, **kw):
    return run_crop_search(
        raw,
        embedder_name="fake",
        detector="fake",
        artifacts_root=tmp_path / "art",
        reports_root=tmp_path / "rep",
        **kw,
    )


def test_end_to_end_hand_computed_values(tmp_path, color):
    result = _run(_make_raw(tmp_path), tmp_path)
    rep = result.report
    assert result.report_path.parent == tmp_path / "rep" / "crop"
    assert json.loads(result.report_path.read_text()) == json.loads(json.dumps(rep))

    pop = rep["population"]
    assert pop["rooms"] == 5 and pop["gallery_size"] == 4 and pop["matches_29_counts"] is False
    assert pop["rooms_dropped_no_existing_product"] == 1 and pop["arms_identical_population"] is True
    assert rep["bootstrap"]["n_clusters"] == 4  # r2 and r5 are identical photos

    rows = {r["room"]: r for r in rep["per_room"]}
    # r1: left box [0,48] -> expanded [0,53]; right box [48,80] -> [44,80] (80 clip). 10% of 48 is 4.8.
    assert [c["box"] for c in rows["r1"]["crops"]] == [[0, 0, 53, 40], [44, 0, 80, 40]]
    assert rows["r1"]["n_crops"] == 2 and rows["r1"]["fallback_to_whole"] is False
    assert rows["r4"]["n_crops"] == 0 and rows["r4"]["fallback_to_whole"] is True

    # Hand traces (see _make_raw). Answers: r1 {P1,P2}, r2 {P3}, r3 {P3,P4}, r4 {P1}, r5 {P3}.
    # Recall@1 whole: r1 0, r2 1, r3 0, r4 1, r5 1 -> 3/5 ; crops: r1 1/2, r2 1, r3 0, r4 1, r5 1 -> 3.5/5
    # Hit@1    whole: 0,1,0,1,1 -> 3/5            ; crops: 1,1,0,1,1 -> 4/5
    m = rep["metrics"]
    assert m["recall"]["1"]["whole"] == pytest.approx(0.6) and m["recall"]["1"]["crops"] == pytest.approx(0.7)
    assert m["recall"]["1"]["mean_delta"] == pytest.approx(0.1)
    assert m["hit"]["1"]["whole"] == pytest.approx(0.6) and m["hit"]["1"]["crops"] == pytest.approx(0.8)
    assert m["hit"]["1"]["mean_delta"] == pytest.approx(0.2)
    assert (m["recall"]["1"]["rooms_delta_gt_0"], m["recall"]["1"]["rooms_delta_eq_0"]) == (1, 4)
    assert m["recall"]["1"]["rooms_delta_lt_0"] == 0
    # gallery holds 4 products: K >= 5 returns all of them in both arms.
    for k in ("5", "10", "20", "100"):
        assert (
            m["recall"][k]["whole"] == m["recall"][k]["crops"] == 1.0 and m["recall"][k]["mean_delta"] == 0.0
        )
    assert rep["primary"]["metric"] == "recall@10" and rep["primary"]["rooms_delta_eq_0"] == 5
    assert (
        rep["verdict"] == "the interval includes zero"
        and rep["verdict_note"] == "not evidence of equivalence"
    )

    al = rep["alongside"]
    assert al["boxes_per_room"] == {"min": 0, "median": 2.0, "max": 2}
    assert al["rooms_with_fewer_than_m_boxes"] == 5 and al["rooms_fallback_to_whole"] == 1
    assert rep["models"]["detector"] == "fake-detector" and rep["contract_sha256"] == sha256_file(
        CONTRACT_PATH
    )
    assert (
        rep["bootstrap"]["b"] == 10_000
        and rep["bootstrap"]["level"] == 0.95
        and rep["bootstrap"]["seed"] == 0
    )
    for arm in ("whole", "crops"):
        for part in ("detector", "embedding", "total"):
            assert set(al["seconds_per_room"][arm][part]) == {"min", "median", "mean", "max"}


def test_merged_rankings_match_hand_traces_via_recall_at_k(tmp_path, color, monkeypatch):
    # K = 2 is not in the reported ks, so ask for it: r1 whole [P4 P1] -> 1/2 ; crops [P1 P2] -> 1.
    monkeypatch.setattr(crop_search, "IKEA_KS", (1, 2, 10))
    rows = {r["room"]: r for r in _run(_make_raw(tmp_path), tmp_path).report["per_room"]}
    assert rows["r1"]["recall"]["whole"]["2"] == 0.5 and rows["r1"]["recall"]["crops"]["2"] == 1.0
    # r3 answers {P3,P4}: whole [P2 P4] -> 1/2 ; crops [P2 P3] (P4 is dup-skipped into depth 2) -> 1/2
    assert rows["r3"]["recall"]["whole"]["2"] == 0.5 and rows["r3"]["recall"]["crops"]["2"] == 0.5
    assert rows["r4"]["delta_recall_at_10"] == 0.0  # fallback room: the same vector in both arms


def test_independent_bootstrap_recomputation_from_the_report(tmp_path, color):
    rep = _run(_make_raw(tmp_path), tmp_path).report
    rows = rep["per_room"]
    # delta of Recall@1 per room and clusters from the report, resampled by a plain loop
    d = np.array([r["recall"]["crops"]["1"] - r["recall"]["whole"]["1"] for r in rows])
    cl = np.array([r["cluster"] for r in rows])
    assert sorted(set(cl)) == [0, 1, 2, 3] and cl[1] == cl[4]  # r2, r5 share a cluster
    idx = np.random.default_rng(0).integers(0, 4, size=(10_000, 4))
    members = [np.flatnonzero(cl == c) for c in range(4)]
    means = np.array([d[np.concatenate([members[c] for c in row])].mean() for row in idx])
    e = rep["metrics"]["recall"]["1"]
    assert e["ci95"] == pytest.approx([np.quantile(means, 0.025), np.quantile(means, 0.975)])
    assert e["share_resamples_delta_le_0"] == pytest.approx((means <= 0).mean())


def test_second_run_reads_the_cache_and_keeps_both_reports(tmp_path, color):
    raw = _make_raw(tmp_path)
    first = _run(raw, tmp_path)
    calls_after_first = color.calls
    assert first.report["alongside"]["embedding_cache_hits"] == {"whole_rooms": 0, "crops": 0}
    second = _run(raw, tmp_path)
    assert color.calls == calls_after_first  # nothing re-embedded
    hits = second.report["alongside"]["embedding_cache_hits"]
    assert (
        hits == {"whole_rooms": 5, "crops": second.report["alongside"]["crops_embedded"]}
        and hits["crops"] == 8
    )
    assert (
        first.report_path != second.report_path and first.report_path.exists() and second.report_path.exists()
    )
    assert first.report["run_id"] != second.report["run_id"]
    # identical numbers, identical population hashes
    assert first.report["metrics"] == second.report["metrics"]
    assert first.report["population"]["sha256"] == second.report["population"]["sha256"]


def test_population_lists_are_written_and_hashed(tmp_path, color):
    res = _run(_make_raw(tmp_path), tmp_path)
    lists = json.loads(res.population_path.read_text())
    assert lists["products"] == [P1, P2, P3, P4]
    assert lists["rooms"] == ["r1", "r2", "r3", "r4", "r5"]
    assert lists["answers"]["r3"] == [P3, P4]
    h = res.report["population"]["sha256"]
    canon = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    assert h["products"] == hashlib.sha256(canon(lists["products"]).encode()).hexdigest()
    assert h["rooms"] == hashlib.sha256(canon(lists["rooms"]).encode()).hexdigest()
    assert h["all"] == hashlib.sha256(canon(lists).encode()).hexdigest()


def test_all_rooms_falling_back_is_a_valid_zero_difference_run(tmp_path, monkeypatch):
    emb = ColorEmbedder()
    monkeypatch.setattr(crop_search, "_make_embedder", lambda name: emb)
    monkeypatch.setattr(
        crop_search,
        "_make_detector",
        lambda name: FakeDetector(lambda im: (np.zeros((0, 4)), np.zeros(0))),
    )
    rep = _run(_make_raw(tmp_path), tmp_path).report
    assert rep["alongside"]["rooms_fallback_to_whole"] == 5
    assert rep["primary"]["mean_delta"] == 0.0 and rep["primary"]["ci95"] == [0.0, 0.0]
    assert rep["primary"]["rooms_delta_eq_0"] == 5
    assert rep["verdict"] == "the interval includes zero"


def test_room_without_answers_stops_the_run(tmp_path, color, monkeypatch):
    real = crop_search.load_population

    def broken(raw):
        pop = real(raw)
        pop.room_answers["r1"] = set()
        return pop

    monkeypatch.setattr(crop_search, "load_population", broken)
    with pytest.raises(CropSearchError, match="no answer"):
        _run(_make_raw(tmp_path), tmp_path)
    assert not (tmp_path / "rep").exists()  # nothing written for a failed attempt


def test_detector_returning_a_bad_crop_stops_the_run(tmp_path, monkeypatch):
    emb = ColorEmbedder()
    monkeypatch.setattr(crop_search, "_make_embedder", lambda name: emb)

    class Outside:
        model_id, revision = "bad", None

        def detect(self, image):
            from product_retrieval.crop.boxes import Crop

            return [Crop(index=0, score=0.9, box=(0, 0, image.width + 5, image.height))]

    monkeypatch.setattr(crop_search, "_make_detector", lambda name: Outside())
    with pytest.raises(CropSearchError, match="outside"):
        _run(_make_raw(tmp_path), tmp_path)


def test_nan_embeddings_are_rejected(tmp_path, monkeypatch):
    class Nan(ColorEmbedder):
        def embed(self, images):
            out = super().embed(images)
            out[0, 0] = np.nan
            return out

    monkeypatch.setattr(crop_search, "_make_embedder", lambda name: Nan())
    monkeypatch.setattr(crop_search, "_make_detector", lambda name: FakeDetector(_split_proposals))
    with pytest.raises(IkeaCheckError):
        _run(_make_raw(tmp_path), tmp_path)


def test_unknown_detector_and_missing_raw_fail(tmp_path):
    with pytest.raises(ValueError, match="detector"):
        crop_search._make_detector("nope")
    with pytest.raises(IkeaCheckError):
        _run(tmp_path / "nowhere", tmp_path)


def test_real_fake_embedder_and_fake_detector_run_is_deterministic(tmp_path):
    raw = _make_raw(tmp_path)
    a = run_crop_search(raw, "fake", "fake", tmp_path / "art", tmp_path / "rep")
    b = run_crop_search(raw, "fake", "fake", tmp_path / "art", tmp_path / "rep")
    assert a.report["metrics"] == b.report["metrics"] and a.report["primary"] == b.report["primary"]
    assert a.report_path != b.report_path
    assert a.report["models"]["embedder"] == "fake"


def test_cluster_and_gate_constants_in_the_pipeline_are_the_contract_values():
    assert crop_search.BOOTSTRAP_B == merge.BOOTSTRAP_B == 10_000
    assert crop_search.BOOTSTRAP_LEVEL == 0.95 and crop_search.CLUSTER_COSINE == 0.95
    assert crop_search.PRIMARY_K == 10 and crop_search.TOP_M == 10 and crop_search.MIN_CROP_PX == 16


def test_cli_happy_path_and_errors(tmp_path):
    raw = _make_raw(tmp_path)
    base = ["crop-search", "--raw", str(raw), "--artifacts-root", str(tmp_path / "art")]
    ok = runner.invoke(
        app, [*base, "--reports-root", str(tmp_path / "rep"), "--embedder", "fake", "--detector", "fake"]
    )
    assert ok.exit_code == 0, ok.output
    assert "verdict: " in ok.output and "primary: mean delta recall@10" in ok.output
    assert "Hit whole" in ok.output and "report: " in ok.output
    assert len(list((tmp_path / "rep" / "crop").glob("*.json"))) == 2  # report + population list

    bad_det = runner.invoke(app, [*base, "--detector", "nope"])
    assert bad_det.exit_code == 2 and "unknown --detector" in bad_det.output
    bad_emb = runner.invoke(app, [*base, "--embedder", "nope", "--detector", "fake"])
    assert bad_emb.exit_code == 2
    (raw / "text_data" / "item_to_room.p").unlink()
    no_map = runner.invoke(app, [*base, "--embedder", "fake", "--detector", "fake"])
    assert no_map.exit_code == 1 and "crop-search:" in no_map.output
    missing = runner.invoke(app, ["crop-search", "--raw", str(tmp_path / "nowhere")])
    assert missing.exit_code != 0
