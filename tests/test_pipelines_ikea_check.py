import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_file
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.pipelines import ikea_check
from product_retrieval.pipelines.ikea_check import (
    CONTRACT_PATH,
    SIGLIP_MODEL_ID,
    IkeaCheckError,
    _slug,
    run_ikea_check,
)

runner = CliRunner()

RED, GREEN, BLUE, YELLOW, WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255)
P1, P2, P3, P4, P5 = "101.000.01", "102.000.02", "103.000.03", "104.000.04", "105.000.05"


def _jpg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path, format="JPEG")


def _write_mapping(raw: Path, mapping: dict[str, list[str]]) -> None:
    (raw / "text_data").mkdir(parents=True, exist_ok=True)
    with open(raw / "text_data" / "item_to_room.p", "wb") as f:
        pickle.dump(mapping, f, protocol=2)


def _p(pid: str) -> str:
    return f"images/{pid}.jpg"


def _r(name: str) -> str:
    return f"images/room_scenes/{name}.jpg"


def _make_raw(tmp_path: Path) -> Path:
    """Tiny IKEA-like folder. Fake embeddings depend on pixels only, so a room painted
    the same colour as a product has cosine 1.0 with it.

    Products on disk: P1 red, P2 green, P3 blue, P4 yellow (P5 is listed but absent).
    Rooms:
      R1 red     lists P1, P2 (P1 twice in the pickle)  -> answers {P1, P2}
      R2 blue    lists P3, P5                           -> answers {P3} (P5 has no file)
      R3 yellow  lists P2                               -> answers {P2}; nearest product is P4
      R4 white   lists P5 only                          -> file exists, no existing product: dropped
      R5         lists P1, file absent                  -> missing file
    """
    raw = tmp_path / "ikea"
    _jpg(raw / "images" / "bed" / f"{P1}.jpg", RED)
    _jpg(raw / "images" / "chair" / f"{P2}.jpg", GREEN)
    _jpg(raw / "images" / "objects" / f"{P3}.jpg", BLUE)
    _jpg(raw / "images" / f"{P4}.jpg", YELLOW)
    _jpg(raw / "images" / "room_scenes" / "r1.jpg", RED)
    _jpg(raw / "images" / "room_scenes" / "r2.jpg", BLUE)
    _jpg(raw / "images" / "room_scenes" / "r3.jpg", YELLOW)
    _jpg(raw / "images" / "room_scenes" / "r4.jpg", WHITE)
    _write_mapping(
        raw,
        {
            _p(P1): [_r("r1"), _r("r1"), _r("r5")],
            _p(P2): [_r("r1"), _r("r3")],
            _p(P3): [_r("r2")],
            _p(P4): [],
            _p(P5): [_r("r2"), _r("r4")],
        },
    )
    return raw


def _run(raw: Path, tmp_path: Path, **kw):
    return run_ikea_check(
        raw, embedder_name="fake", artifacts_root=tmp_path / "art", reports_root=tmp_path / "rep", **kw
    )


def test_happy_path_counts_and_hand_computed_metrics(tmp_path):
    result = _run(_make_raw(tmp_path), tmp_path)
    rep = result.report
    assert result.report_path == tmp_path / "rep" / "ikea" / "fake.json"
    assert json.loads(result.report_path.read_text()) == json.loads(json.dumps(rep))

    c = rep["counts"]
    assert c["rooms"] == 3
    assert c["rooms_listed_in_mapping"] == 5
    assert c["rooms_dropped_no_existing_product"] == 1  # r4
    assert c["rooms_missing_file"] == 1  # r5
    assert c["gallery_size"] == 4 and c["products_listed_in_mapping"] == 5 and c["products_missing_file"] == 1
    # answers: r1 {P1,P2} = 2, r2 {P3} = 1, r3 {P2} = 1
    assert c["products_per_room"] == {"min": 1, "median": 1.0, "max": 2}

    # Ranks (fake embedding: identical pixels -> cosine 1, nearest product first):
    #   r1 red    -> P1 first. Hit@1 = 1, Recall@1 = 1/2
    #   r2 blue   -> P3 first. Hit@1 = 1, Recall@1 = 1/1
    #   r3 yellow -> P4 first, answer P2 is elsewhere. Hit@1 = 0, Recall@1 = 0
    # Means: Hit@1 = (1+1+0)/3 = 2/3 ; Recall@1 = (1/2+1+0)/3 = 1/2.
    # Gallery has 4 products, so K >= 4 holds everything: Hit@K = Recall@K = 1 for K = 5..100.
    hit, rec = rep["metrics"]["hit"], rep["metrics"]["recall"]
    assert hit["1"]["value"] == pytest.approx(2 / 3)
    assert rec["1"]["value"] == pytest.approx(1 / 2)
    for k in ("5", "10", "20", "100"):
        assert hit[k]["value"] == 1.0 and rec[k]["value"] == 1.0
        assert hit[k]["ci95"] == [1.0, 1.0] and rec[k]["ci95"] == [1.0, 1.0]
    lo, hi = hit["1"]["ci95"]
    assert 0.0 <= lo <= hi <= 1.0

    assert rep["ks"] == [1, 5, 10, 20, 100]
    assert rep["bootstrap"] == {"b": 1000, "seed": 0, "level": 0.95, "unit": "room"}
    assert rep["model_id"] == "fake" and rep["embedder"] == "fake"
    assert rep["contract_sha256"] == sha256_file(CONTRACT_PATH)
    assert rep["raw_commit"] is None  # tmp folder is not a git clone
    assert rep["raw_commit_matches_contract"] is False


def test_report_is_deterministic_across_runs(tmp_path):
    raw = _make_raw(tmp_path)
    a = _run(raw, tmp_path).report
    b = run_ikea_check(
        raw, embedder_name="fake", artifacts_root=tmp_path / "art2", reports_root=tmp_path / "rep2"
    ).report
    assert a == b


class _CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.n_embedded = 0

    def embed(self, images):
        self.n_embedded += len(images)
        return super().embed(images)


def test_embedding_cache_is_keyed_by_content_and_reused(tmp_path, monkeypatch):
    raw = _make_raw(tmp_path)
    emb = _CountingEmbedder()
    monkeypatch.setattr(ikea_check, "_make_embedder", lambda name: emb)
    _run(raw, tmp_path)
    # 4 products + 3 evaluated rooms, but r1/r2/r3 are byte-identical to P1/P3/P4 -> 4 unique files.
    assert emb.n_embedded == 4
    _run(raw, tmp_path)
    assert emb.n_embedded == 4  # second run: all from cache


def test_tie_goes_to_smaller_product_id(tmp_path):
    raw = tmp_path / "ikea"
    _jpg(raw / "images" / "a" / f"{P1}.jpg", RED)
    _jpg(raw / "images" / "b" / f"{P2}.jpg", RED)  # same pixels as P1 -> exact tie
    _jpg(raw / "images" / "room_scenes" / "r1.jpg", RED)
    _write_mapping(raw, {_p(P1): [], _p(P2): [_r("r1")]})
    hit = _run(raw, tmp_path).report["metrics"]["hit"]
    assert hit["1"]["value"] == 0.0  # P1 outranks the tied answer P2
    assert hit["5"]["value"] == 1.0


def test_basename_with_different_content_in_two_folders_is_excluded(tmp_path):
    """Contract v1.1: an ambiguous image is dropped from the gallery and the answers, and recorded."""
    raw = _make_raw(tmp_path)
    before = _run(raw, tmp_path / "a").report["counts"]["gallery_size"]
    _jpg(raw / "images" / "bed" / f"{P2}.jpg", BLUE)  # P2 also in chair/ with other pixels
    rep = _run(raw, tmp_path / "b").report
    assert rep["counts"]["gallery_size"] == before - 1
    assert rep["counts"]["excluded_basenames_conflicting_content"] == [f"{P2}.jpg"]


def test_basename_with_identical_content_in_two_folders_is_one_image_and_recorded(tmp_path):
    raw = _make_raw(tmp_path)
    _jpg(raw / "images" / "bed" / f"{P2}.jpg", GREEN)  # same pixels as chair/P2 -> byte-identical
    rep = _run(raw, tmp_path).report
    assert rep["counts"]["gallery_size"] == 4
    assert rep["counts"]["duplicate_basenames_identical_content"] == [f"{P2}.jpg"]


def test_two_mapping_keys_with_one_article_number_fail(tmp_path):
    raw = _make_raw(tmp_path)
    _jpg(raw / "images" / "bed" / f"{P4}.jpg", YELLOW)
    mapping = {_p(P1): [_r("r1")], "images/bed/" + f"{P4}.jpg": [_r("r1")], _p(P4): [_r("r1")]}
    _write_mapping(raw, mapping)
    with pytest.raises(IkeaCheckError, match="two mapping keys"):
        _run(raw, tmp_path)


def test_missing_mapping_and_missing_images_folder_fail(tmp_path):
    raw = tmp_path / "ikea"
    raw.mkdir()
    with pytest.raises(IkeaCheckError, match="mapping not found"):
        _run(raw, tmp_path)
    _write_mapping(raw, {_p(P1): [_r("r1")]})
    with pytest.raises(IkeaCheckError, match="images folder not found"):
        _run(raw, tmp_path)


def test_malformed_mapping_fails(tmp_path):
    raw = tmp_path / "ikea"
    (raw / "images").mkdir(parents=True)
    _write_mapping(raw, {_p(P1): "not a list"})  # type: ignore[dict-item]
    with pytest.raises(IkeaCheckError, match="bad entry"):
        _run(raw, tmp_path)
    _write_mapping(raw, {})
    with pytest.raises(IkeaCheckError, match="non-empty dict"):
        _run(raw, tmp_path)


def test_no_existing_product_means_empty_gallery_error(tmp_path):
    raw = tmp_path / "ikea"
    _jpg(raw / "images" / "room_scenes" / "r1.jpg", RED)
    _write_mapping(raw, {_p(P1): [_r("r1")]})
    with pytest.raises(IkeaCheckError, match="empty gallery"):
        _run(raw, tmp_path)


def test_all_rooms_dropped_is_an_error_not_a_zero_report(tmp_path):
    raw = tmp_path / "ikea"
    _jpg(raw / "images" / f"{P1}.jpg", RED)
    _jpg(raw / "images" / "room_scenes" / "r1.jpg", RED)
    _write_mapping(raw, {_p(P1): [_r("r9")], _p(P5): [_r("r1")]})  # r9 has no file; r1 lists only absent P5
    with pytest.raises(IkeaCheckError, match="nothing to evaluate"):
        _run(raw, tmp_path)
    assert not (tmp_path / "rep" / "ikea").exists()  # no report written


def test_undecodable_image_fails_loudly(tmp_path):
    raw = _make_raw(tmp_path)
    (raw / "images" / f"{P4}.jpg").write_bytes(b"not an image")
    with pytest.raises(IkeaCheckError, match="cannot decode"):
        _run(raw, tmp_path)


class _NanEmbedder(FakeEmbedder):
    def embed(self, images):
        out = super().embed(images)
        out[0, 0] = np.nan
        return out


class _UnnormalizedEmbedder(FakeEmbedder):
    def embed(self, images):
        return super().embed(images) * 3.0


@pytest.mark.parametrize("bad", [_NanEmbedder, _UnnormalizedEmbedder])
def test_bad_embeddings_are_rejected(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(ikea_check, "_make_embedder", lambda name: bad())
    with pytest.raises(IkeaCheckError, match="embedder output"):
        _run(_make_raw(tmp_path), tmp_path)


def test_unknown_embedder_name_fails(tmp_path):
    with pytest.raises(ValueError, match="unknown embedder"):
        run_ikea_check(_make_raw(tmp_path), embedder_name="nope")  # type: ignore[arg-type]


def test_raw_commit_is_read_from_the_raw_git_repo(tmp_path):
    raw = _make_raw(tmp_path)
    git = ["git", "-C", str(raw), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    expected = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    rep = _run(raw, tmp_path).report
    assert rep["raw_commit"] == expected
    assert rep["raw_commit_matches_contract"] is False  # not the pinned 6f316e3f


def test_slug_and_siglip_model_id_match_the_frozen_baseline_config():
    assert _slug("google/siglip2-base-patch16-224") == "google__siglip2-base-patch16-224"
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml").read_text())
    assert SIGLIP_MODEL_ID == cfg["embed_model_id"]


def test_cli_happy_path(tmp_path):
    raw = _make_raw(tmp_path)
    result = runner.invoke(
        app,
        [
            "ikea-check",
            "--raw", str(raw),
            "--embedder", "fake",
            "--artifacts-root", str(tmp_path / "art"),
            "--reports-root", str(tmp_path / "rep"),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "rooms=3" in result.output and "dropped_rooms=1" in result.output and "gallery=4" in result.output
    assert "Hit@K" in result.output and "Recall@K" in result.output
    assert (tmp_path / "rep" / "ikea" / "fake.json").is_file()


def test_cli_reports_data_errors_with_exit_1_and_bad_embedder_with_exit_2(tmp_path):
    raw = tmp_path / "ikea"
    raw.mkdir()
    base = [
        "ikea-check",
        "--raw",
        str(raw),
        "--artifacts-root",
        str(tmp_path / "a"),
        "--reports-root",
        str(tmp_path / "r"),
    ]
    bad_data = runner.invoke(app, [*base, "--embedder", "fake"])
    assert bad_data.exit_code == 1
    assert "mapping not found" in bad_data.output
    bad_embedder = runner.invoke(app, [*base, "--embedder", "nope"])
    assert bad_embedder.exit_code == 2
    missing_raw = runner.invoke(app, ["ikea-check", "--raw", str(tmp_path / "nowhere")])
    assert missing_raw.exit_code != 0
