"""End-to-end `pr deepfurniture-crop` on a tiny synthetic DeepFurniture folder (no real data, no models).

World (gallery = five val identities, one solid-colour preview each; embedder = mean RGB):
  V1 red, V2 green, V3 blue, V4 yellow, V5 white; T1 is a test identity, R1/R2 are train identities.

  s1  160x120  red 60% | green 40%   #1 V1 box(0,0,96,120) area .5; #2 R1 box(96,0,160,120) area .3
  s2  160x120  blue 60% | green 40%  #1 V3 left, #2 V2 right (no areaRatio); #3 R2 area .9
  s6  400x300  white                 #1 V5 Table area .4; #2 V5 Lamp area .1
  s3  has test identity T1 (excluded) | s4 train identities only (excluded) | s5 val query source (excluded)

Hand-computed rankings (cosine of mean RGB against the gallery colours):
  s1 whole    V4 V1 V5 V2 V3     answers {V1}     recall@1 0   (detector and GT: V1 first, recall@1 1)
  s2 whole    V3 V5 V2 V4 V1     answers {V2,V3}  recall@1 .5  (detector and GT merged: V3 first, .5)
  s6 whole    V5 first           answers {V5}     recall@1 1   (all arms)
"""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from product_retrieval import cli
from product_retrieval.core.ids import canonical_json
from product_retrieval.crop import boxes as crop_boxes
from product_retrieval.crop.fake import FakeDetector
from product_retrieval.crop.owl import OWL_MODEL_ID, OWL_REVISION
from product_retrieval.data.deepfurniture_scenes import SceneBoxError, ScenePopulationError
from product_retrieval.pipelines import crop_search as cs
from product_retrieval.pipelines import deepfurniture_crop as dfc
from product_retrieval.pipelines.deepfurniture_crop import (
    CROP_CONTRACT_SHA256,
    SIGLIP_REVISION,
    DeepFurnitureCropError,
    run_deepfurniture_crop,
)
from product_retrieval.pipelines.ikea_check import IkeaCheckError

runner = CliRunner()

RED, GREEN, BLUE, YELLOW, WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255)
COLORS = {"V1": RED, "V2": GREEN, "V3": BLUE, "V4": YELLOW, "V5": WHITE}
SPLITS = {
    "V1": "val",
    "V2": "val",
    "V3": "val",
    "V4": "val",
    "V5": "val",
    "T1": "test",
    "R1": "train",
    "R2": "train",
}
SMALL = dict(expected_scenes=3, expected_answers=4, expected_gallery=5)


class ColorEmbedder:
    """Mean RGB, unit length."""

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


def _split_proposals(image):
    w, h = image.size
    boxes = np.array([[0, 0, 0.6 * w, h], [0.6 * w, 0, w, h], [0, 0, w, h]], dtype=np.float64)
    return boxes, np.array([0.9, 0.8, 0.05])


def _jpeg(size, left, right=None, split=0.6) -> bytes:
    w, h = size
    im = Image.new("RGB", size, color=left)
    if right is not None:
        im.paste(right, (int(w * split), 0, w, h))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=100, subsampling=0)
    return buf.getvalue()


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _inst(number_id, identity, box, category="Sofa", area=None):
    d = {
        "numberID": number_id,
        "boundingBox": {"xMin": box[0], "yMin": box[1], "xMax": box[2], "yMax": box[3]},
        "identityID": identity,
        "categoryName": category,
        "segmentation": [1, 2, 3],
    }
    if area is not None:
        d["areaRatio"] = area
    return d


def _make_world(tmp_path: Path, mutate=None) -> tuple[Path, Path]:
    """Return ``(raw_root, data_root)``. ``mutate(annotations)`` can edit the annotation dicts."""
    data_root = tmp_path / "data"
    raw = data_root / "deepfurniture"
    (raw / "data" / "scenes").mkdir(parents=True)
    (raw / "data" / "metadata").mkdir(parents=True)

    # the real annotations carry integer identityIDs (parsing is tested in test_data_deepfurniture_scenes)
    annotations = {
        "s1": [
            _inst(1, "V1", (0, 0, 96, 120), area=0.5),
            _inst(2, "R1", (96, 0, 160, 120), "Chair", area=0.3),
        ],
        "s2": [
            _inst(1, "V3", (0, 0, 96, 120)),
            _inst(2, "V2", (96, 0, 160, 120), "Table"),
            _inst(3, "R2", (10, 10, 60, 60), "Lamp", area=0.9),
        ],
        "s3": [_inst(1, "V1", (0, 0, 50, 50)), _inst(2, "T1", (60, 60, 100, 100))],
        "s4": [_inst(1, "R1", (0, 0, 50, 50)), _inst(2, "R2", (60, 60, 100, 100))],
        "s5": [_inst(1, "V4", (0, 0, 50, 50))],
        "s6": [
            _inst(1, "V5", (100, 50, 300, 250), "Table", area=0.4),
            _inst(2, "V5", (20, 20, 80, 80), "Lamp", area=0.1),
        ],
    }
    if mutate:
        mutate(annotations)
    images = {
        "s1": _jpeg((160, 120), RED, GREEN),
        "s2": _jpeg((160, 120), BLUE, GREEN),
        "s3": _jpeg((160, 120), RED),
        "s4": _jpeg((160, 120), BLUE),
        "s5": _jpeg((160, 120), YELLOW),
        "s6": _jpeg((400, 300), WHITE),
    }
    with tarfile.open(raw / "data" / "scenes" / "scenes_0000.tar.gz", "w:gz") as tar:
        for sid in sorted(annotations):
            ann = json.dumps({"scene": {"sceneTaskID": sid}, "instances": annotations[sid]}).encode()
            for name, data in (("image.jpg", images[sid]), ("annotation.json", ann), ("depth.png", b"d")):
                info = tarfile.TarInfo(f"{sid}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    rows = []
    for k, (pid, split) in enumerate(sorted(SPLITS.items())):
        data = _png(COLORS.get(pid, (10 + k, 20, 30)))  # non-val identities: distinct bytes, never ranked
        sha = hashlib.sha256(data).hexdigest()
        path = data_root / "deepfurniture" / "images" / sha[:2] / f"{sha}.img"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(
            {"source": "deepfurniture", "product_id": pid, "split": split, "query": [], "gallery": [sha]}
        )
    (raw / "manifest_deepfurniture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (raw / "data" / "metadata" / "query_index.json").write_text(
        json.dumps({"queries": [{"query_name": "V4_1_s5.jpg"}, {"query_name": "T1_1_s3.jpg"}]})
    )
    return raw, data_root


@pytest.fixture
def stubs(monkeypatch):
    emb = ColorEmbedder()
    seen = {"revisions": []}

    def make_embedder(name, revision):
        seen["revisions"].append(revision)
        return emb

    monkeypatch.setattr(dfc, "_make_embedder", make_embedder)
    monkeypatch.setattr(cs, "_make_detector", lambda name: FakeDetector(_split_proposals))
    return SimpleNamespace(emb=emb, seen=seen)


def _run(tmp_path, world=None, run_id="run1", **kw):
    raw, data_root = world or _make_world(tmp_path)
    args = dict(
        embedder_name="fake",
        detector="fake",
        out_root=tmp_path / "out",
        repo_root=tmp_path,
        require_commit=False,
        **SMALL,
    )
    args.update(kw)
    return run_deepfurniture_crop(raw, data_root, run_id, **args)


# ----------------------------------------------------------------------------------------- results


def test_end_to_end_hand_computed_values(tmp_path, stubs):
    result = _run(tmp_path)
    rep = result.report
    assert result.run_dir == tmp_path / "out" / "run1"
    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "embeddings",
        "population.json",
        "report.json",
        "summary.md",
    ]
    assert json.loads(result.report_path.read_text()) == json.loads(json.dumps(rep))

    pop = rep["population"]
    assert (pop["scenes"], pop["answers"], pop["gallery_size"], pop["scenes_in_archive"]) == (3, 4, 5, 6)
    assert pop["excluded_scenes"] == {"no_val_identity": 1, "test_identity": 1, "val_query_source": 1}
    assert pop["excluded_val_query_source_scenes"] == ["s5"]
    assert pop["answers_per_scene"]["max"] == 2

    rows = {r["scene"]: r for r in rep["per_scene"]}
    assert set(rows) == {"s1", "s2", "s6"}
    assert rows["s1"]["image_size"] == [160, 120] and rows["s6"]["image_size"] == [400, 300]
    assert rep["images"]["sizes"] == {"160x120": 2, "400x300": 1}

    # s1 detector crops: left 96 px box padded to x2=106, right box padded to x1=89
    assert [c["box"] for c in rows["s1"]["arms"]["detector"]["crops"]] == [
        [0, 0, 106, 120],
        [89, 0, 160, 120],
    ]
    assert [c["box"] for c in rows["s1"]["arms"]["gt"]["crops"]] == [[0, 0, 106, 120], [89, 0, 160, 120]]
    # s2 GT order: #3 has areaRatio, #1 and #2 have none and follow by numberID
    assert [c["index"] for c in rows["s2"]["arms"]["gt"]["crops"]] == [3, 1, 2]

    def rec(scene, arm, k):
        return rows[scene]["arms"][arm]["recall"][str(k)]

    assert [rec("s1", a, 1) for a in ("whole", "detector", "gt")] == [0.0, 1.0, 1.0]
    assert [rec("s2", a, 1) for a in ("whole", "detector", "gt")] == [0.5, 0.5, 0.5]
    assert [rec("s6", a, 1) for a in ("whole", "detector", "gt")] == [1.0, 1.0, 1.0]
    assert all(rec(s, a, 5) == 1.0 for s in rows for a in ("whole", "detector", "gt"))
    m = rep["metrics"]
    assert m["recall"]["1"] == pytest.approx({"whole": 0.5, "detector": 5 / 6, "gt": 5 / 6})
    assert m["hit"]["1"] == pytest.approx({"whole": 2 / 3, "detector": 1.0, "gt": 1.0})

    # three distinct scenes: three clusters, below the minimum of 20
    assert rep["bootstrap"]["n_clusters"] == 3 and rep["bootstrap"]["inferential"] is False
    assert rep["verdict"] == "insufficient independent clusters"
    assert rep["primary"]["mean_delta"] == 0.0 and rep["primary"]["ci95"] is None
    assert rep["comparisons"]["detector_minus_whole"]["recall"]["1"]["mean_delta"] == pytest.approx(1 / 3)

    # GT arm report: two instances of s2 lack areaRatio
    gt = rep["gt_box_arm"]
    assert gt["instances_without_area_ratio"] == 2 and gt["scenes_with_instances_without_area_ratio"] == 1
    assert gt["answers_without_box_in_first_10"] == 0
    assert rep["instances_without_area_ratio_by_scene"] == {"s2": [1, 2]}

    # categories: s6's V5 has two categories and counts once for each
    cats = rep["per_category_recall_at_10"]["arms"]["whole"]
    assert {c: v["denominator"] for c, v in cats.items()} == {"Lamp": 1, "Sofa": 2, "Table": 2}
    assert "Chair" not in cats  # R1 is a train identity, never an answer


def test_population_file_and_hashes_by_independent_computation(tmp_path, stubs):
    result = _run(tmp_path)
    pop = json.loads(result.population_path.read_text())
    assert pop["scenes"] == ["s1", "s2", "s6"]
    assert pop["answers"] == {"s1": ["V1"], "s2": ["V2", "V3"], "s6": ["V5"]}
    assert [g["product_id"] for g in pop["gallery"]] == ["V1", "V2", "V3", "V4", "V5"]
    for g in pop["gallery"]:
        assert g["image_sha256"] == hashlib.sha256(_png(COLORS[g["product_id"]])).hexdigest()

    def h(obj):
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    sha = result.report["population"]["sha256"]
    assert sha["scenes"] == h(pop["scenes"]) and sha["answers"] == h(pop["answers"])
    assert sha["gallery"] == h(pop["gallery"]) and sha["all"] == h(pop)
    assert canonical_json(pop) == json.dumps(pop, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def test_report_records_contract_code_revision_and_settings(tmp_path, stubs):
    result = _run(tmp_path)
    rep = result.report
    assert rep["crop_contract"]["sha256"] == CROP_CONTRACT_SHA256 == rep["crop_contract"]["expected_sha256"]
    assert rep["contract_sha256"] == hashlib.sha256(dfc.DF_CONTRACT_PATH.read_bytes()).hexdigest()
    assert set(rep["code"]) == {"commit", "dirty", "status_porcelain", "diff_head_sha256"}
    s = rep["detector_settings"]
    assert (s["score_threshold"], s["nms_iou"], s["top_m"], s["expand_frac"], s["min_crop_px"]) == (
        0.10,
        0.5,
        10,
        0.10,
        16,
    )
    assert s["detector_name"] == "fake"
    assert rep["models"]["embedder_revision_requested"] is None  # fake embedder: nothing pinned
    assert stubs.seen["revisions"] == [SIGLIP_REVISION]  # the default pin is handed to the embedder factory
    text = result.summary_path.read_text()
    assert "insufficient independent clusters" in text and "Pooled Recall@10 per category" in text


def test_code_commit_is_recorded_in_the_real_repo():
    state = dfc.code_state()
    assert state["commit"] is not None and len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool) and len(state["diff_head_sha256"]) == 64


def test_embedding_revision_is_passed_through(tmp_path, stubs):
    _run(tmp_path, siglip_revision="abc123")
    assert stubs.seen["revisions"] == ["abc123"]


def test_nothing_is_read_from_another_runs_cache(tmp_path, stubs):
    world = _make_world(tmp_path)
    _run(tmp_path, world=world, run_id="a")
    first = stubs.emb.calls
    _run(tmp_path, world=world, run_id="b")
    assert stubs.emb.calls == 2 * first  # the second run embedded everything again


# ---------------------------------------------------------------------------- the same code path


def test_detector_arm_uses_the_ikea_functions(tmp_path, stubs, monkeypatch):
    calls = {"detect": [], "rank": 0}
    real_detect, real_rank = cs.detect_crops, cs.rank_vector

    def spy_detect(det, image, label):
        out = real_detect(det, image, label)
        calls["detect"].append((det, label, out[0]))
        return out

    def spy_rank(*a, **k):
        calls["rank"] += 1
        return real_rank(*a, **k)

    monkeypatch.setattr(cs, "detect_crops", spy_detect)
    monkeypatch.setattr(cs, "rank_vector", spy_rank)

    result = _run(tmp_path)
    assert [label for _, label, _ in calls["detect"]] == ["scene 's1'", "scene 's2'", "scene 's6'"]
    assert len({id(d) for d, _, _ in calls["detect"]}) == 1
    rows = {r["scene"]: r for r in result.report["per_scene"]}
    for (_, _label, crops), sid in zip(calls["detect"], ("s1", "s2", "s6"), strict=True):
        assert [c.box for c in crops] == [tuple(c["box"]) for c in rows[sid]["arms"]["detector"]["crops"]]
    # whole: 3 rankings; detector: 2+2+2 crops; GT: 2+3+2 crops
    assert calls["rank"] == 3 + 6 + 7

    # the IKEA run goes through the very same two functions
    n_dfc = (len(calls["detect"]), calls["rank"])
    calls["detect"].clear()
    calls["rank"] = 0
    raw = tmp_path / "ikea"
    (raw / "images").mkdir(parents=True)
    (raw / "images" / "room_scenes").mkdir()
    (raw / "text_data").mkdir()
    (raw / "images" / "101.000.01.jpg").write_bytes(_jpeg((16, 16), RED))
    (raw / "images" / "room_scenes" / "r1.jpg").write_bytes(_jpeg((80, 40), RED, GREEN))
    import pickle

    with open(raw / "text_data" / "item_to_room.p", "wb") as f:
        pickle.dump({"images/101.000.01.jpg": ["images/room_scenes/r1.jpg"]}, f, protocol=2)
    monkeypatch.setattr(cs, "_make_embedder", lambda name: ColorEmbedder())
    cs.run_crop_search(raw, "fake", "fake", tmp_path / "art", tmp_path / "rep")
    assert len(calls["detect"]) == 1 and calls["rank"] == 1 + 2
    assert n_dfc == (3, 16)


def test_ikea_helpers_are_not_copied_into_the_deepfurniture_module():
    src = Path(dfc.__file__).read_text()
    assert "cs.detect_crops(" in src and "cs.rank_vector(" in src
    assert "merge_round_robin(" in src
    assert "def detect_crops" not in src and "def rank_vector" not in src
    assert "argsort" not in src and "det.detect(" not in src


# ------------------------------------------------------------------------------------- guards


def test_contract_hash_assert_fails_on_a_modified_copy(tmp_path):
    good = shutil.copy(cs.CONTRACT_PATH, tmp_path / "good.md")
    assert dfc.assert_crop_contract_hash(Path(good)) == CROP_CONTRACT_SHA256
    bad = tmp_path / "bad.md"
    bad.write_bytes(Path(cs.CONTRACT_PATH).read_bytes() + b" ")
    with pytest.raises(DeepFurnitureCropError, match="expected 10e64eee"):
        dfc.assert_crop_contract_hash(bad)
    with pytest.raises(DeepFurnitureCropError, match="not found"):
        dfc.assert_crop_contract_hash(tmp_path / "missing.md")


def test_run_stops_on_a_modified_contract_before_anything_else(tmp_path, stubs):
    bad = tmp_path / "bad.md"
    bad.write_bytes(Path(cs.CONTRACT_PATH).read_bytes().replace(b"M = 10", b"M = 12"))
    with pytest.raises(DeepFurnitureCropError, match="sha256"):
        _run(tmp_path, crop_contract_path=bad)
    assert not (tmp_path / "out").exists()  # no run dir, nothing read
    assert stubs.seen["revisions"] == [] and stubs.emb.calls == 0


def test_the_pinned_hash_is_the_one_in_the_deepfurniture_contract():
    text = dfc.DF_CONTRACT_PATH.read_text(encoding="utf-8")
    assert CROP_CONTRACT_SHA256 in text and SIGLIP_REVISION in text and OWL_REVISION in text
    assert hashlib.sha256(Path(cs.CONTRACT_PATH).read_bytes()).hexdigest() == CROP_CONTRACT_SHA256


def test_frozen_numbers_match_the_ikea_contract_text():
    text = Path(cs.CONTRACT_PATH).read_text(encoding="utf-8")
    for fragment in ("score ≥ 0.10", "IoU 0.5", "**M = 10**", "10% of its width/height", "16 px"):
        assert fragment in text
    assert dfc.FROZEN_SETTINGS == {
        "score_threshold": 0.10,
        "nms_iou": 0.5,
        "top_m": 10,
        "expand_frac": 0.10,
        "min_crop_px": 16,
    }


def _owl_settings():
    det = SimpleNamespace(model_id=OWL_MODEL_ID, revision=OWL_REVISION)
    return dfc.executable_detector_settings(det, "owl")


def test_owl_settings_pass_against_the_contract_text_and_fail_when_changed():
    text = Path(cs.CONTRACT_PATH).read_text(encoding="utf-8")
    s = _owl_settings()
    assert s["n_prompts"] == 24
    dfc.verify_detector_settings(s, text)
    for key, value, match in (
        ("detector_revision", "0" * 40, "differs from the contract"),
        ("prompts", s["prompts"][:-1] + ["storage bin"], "prompts"),
        ("prompts", s["prompts"][:-1], "prompts"),
        ("top_m", 5, "top_m"),
        ("min_crop_px", 8, "min_crop_px"),
    ):
        with pytest.raises(DeepFurnitureCropError, match=match):
            dfc.verify_detector_settings({**s, key: value}, text)


def test_run_stops_when_an_executable_constant_changed(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(crop_boxes, "SCORE_THRESHOLD", 0.2)
    with pytest.raises(DeepFurnitureCropError, match="SCORE_THRESHOLD"):
        _run(tmp_path)
    assert stubs.emb.calls == 0  # stopped before any embedding


@pytest.mark.parametrize("kw", [dict(expected_scenes=4), dict(expected_answers=5), dict(expected_gallery=6)])
def test_wrong_counts_stop_the_run_before_any_model(tmp_path, stubs, kw):
    with pytest.raises((ScenePopulationError,), match="contract says"):
        _run(tmp_path, **kw)
    assert stubs.seen["revisions"] == [] and stubs.emb.calls == 0


def test_default_counts_are_the_contract_counts(tmp_path, stubs):
    raw, data_root = _make_world(tmp_path)
    with pytest.raises(ScenePopulationError, match="228 / 285"):
        run_deepfurniture_crop(
            raw, data_root, "r", out_root=tmp_path / "o", repo_root=tmp_path, require_commit=False
        )


def test_box_outside_its_image_stops_the_run_before_any_model(tmp_path, stubs):
    def mutate(ann):
        ann["s2"][2]["boundingBox"] = {"xMin": 500, "xMax": 600, "yMin": 10, "yMax": 50}  # the train box

    with pytest.raises(SceneBoxError, match="outside the 160x120 image"):
        _run(tmp_path, world=_make_world(tmp_path, mutate))
    assert stubs.seen["revisions"] == [] and stubs.emb.calls == 0


def test_box_valid_on_a_large_image_is_an_error_on_a_small_one(tmp_path, stubs):
    # the same pixel box (1000..1200) is fine in a 4000x3000 image and entirely outside a 400x300 one
    from product_retrieval.data.deepfurniture_scenes import Instance, SceneAnnotation, clipped_boxes

    s = SceneAnnotation("x", (Instance(1, (1000.0, 1000.0, 1200.0, 1200.0), "V1", "Sofa", None),))
    assert clipped_boxes(s, 4000, 3000) == {1: (1000.0, 1000.0, 1200.0, 1200.0)}
    with pytest.raises(SceneBoxError):
        clipped_boxes(s, 400, 300)


def test_a_non_finite_embedding_stops_the_run(tmp_path, stubs, monkeypatch):
    class NanEmbedder(ColorEmbedder):
        def embed(self, images):
            out = super().embed(images)
            out[0, 0] = np.nan
            return out

    monkeypatch.setattr(dfc, "_make_embedder", lambda name, revision: NanEmbedder())
    with pytest.raises(IkeaCheckError, match="NaN"):
        _run(tmp_path)


def test_all_arms_must_cover_the_same_population():
    ids, ans, gallery = ["a", "b"], [["g1"], ["g2"]], ["g1", "g2", "g3"]
    ok = {arm: [["g1", "g2", "g3"], ["g2", "g1", "g3"]] for arm in ("whole", "detector", "gt")}
    dfc.assert_same_population(ids, ans, gallery, ok)
    for bad in (
        {**ok, "gt": ok["gt"][:1]},  # a scene missing in one arm
        {**ok, "gt": [["g1", "g2"], ["g2", "g1", "g3"]]},  # not min(K, gallery) products
        {**ok, "gt": [["g1", "g2", "zz"], ["g2", "g1", "g3"]]},  # a product outside the gallery
        {k: v for k, v in ok.items() if k != "gt"},  # an arm missing
    ):
        with pytest.raises(DeepFurnitureCropError):
            dfc.assert_same_population(ids, ans, gallery, bad)


def test_a_run_folder_is_never_reused(tmp_path, stubs):
    world = _make_world(tmp_path)
    _run(tmp_path, world=world)
    with pytest.raises(FileExistsError, match="run1"):  # the run folder, not the world folder
        _run(tmp_path, world=world)


@pytest.mark.parametrize("run_id", ["", "../x", "a/b", ".hidden", "a b"])
def test_run_id_must_be_a_plain_name(tmp_path, stubs, run_id):
    with pytest.raises(DeepFurnitureCropError, match="run id"):
        _run(tmp_path, run_id=run_id)


def test_population_only_stops_before_models(tmp_path, stubs):
    result = _run(tmp_path, population_only=True)
    assert result.report["population_only"] is True and result.report["population"]["answers"] == 4
    assert sorted(p.name for p in result.run_dir.iterdir()) == ["population.json", "report.json"]
    assert stubs.seen["revisions"] == [] and stubs.emb.calls == 0
    assert "population only" in dfc.format_console(result.report)


def test_code_commit_is_required_unless_waived(tmp_path, stubs):
    with pytest.raises(DeepFurnitureCropError, match="commit"):
        _run(tmp_path, require_commit=True)


def test_exif_rotated_scene_stops_the_run(tmp_path, stubs, monkeypatch):
    monkeypatch.setattr(dfc, "exif_orientation", lambda f: 6)
    with pytest.raises(DeepFurnitureCropError, match="EXIF"):
        _run(tmp_path)
    assert stubs.emb.calls == 0


# ------------------------------------------------------------------------------------------- CLI


def test_cli_end_to_end(tmp_path, stubs, monkeypatch):
    raw, data_root = _make_world(tmp_path)
    real = cli.run_deepfurniture_crop
    monkeypatch.setattr(
        cli,
        "run_deepfurniture_crop",
        lambda *a, **k: real(*a, repo_root=tmp_path, require_commit=False, **SMALL, **k),
    )
    args = [
        "deepfurniture-crop",
        "--run-id",
        "cli1",
        "--raw",
        str(raw),
        "--data-root",
        str(data_root),
        "--out-root",
        str(tmp_path / "out"),
        "--embedder",
        "fake",
        "--detector",
        "fake",
    ]
    res = runner.invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert "verdict: insufficient independent clusters" in res.output
    assert (tmp_path / "out" / "cli1" / "report.json").is_file()
    again = runner.invoke(cli.app, args)  # the run folder exists now
    assert again.exit_code == 1 and "File exists" in again.output


def test_cli_reports_contract_failures_with_exit_1(tmp_path, stubs):
    raw, data_root = _make_world(tmp_path)
    res = runner.invoke(
        cli.app,
        ["deepfurniture-crop", "--run-id", "x", "--raw", str(raw), "--data-root", str(data_root),
         "--out-root", str(tmp_path / "out"), "--embedder", "fake", "--detector", "fake"],
    )  # fmt: skip
    assert res.exit_code == 1 and "228 / 285" in res.output


@pytest.mark.parametrize("flag,value", [("--embedder", "nope"), ("--detector", "nope")])
def test_cli_rejects_unknown_models(tmp_path, flag, value):
    res = runner.invoke(cli.app, ["deepfurniture-crop", "--run-id", "x", flag, value])
    assert res.exit_code == 2


def test_cli_run_line_defaults_point_at_the_real_layout():
    res = runner.invoke(cli.app, ["deepfurniture-crop", "--help"])
    assert res.exit_code == 0
    for opt in ("--run-id", "--out-root", "--siglip-revision", "--population-only"):
        assert opt in res.output
