import io
import json
import tarfile

import pytest
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.data.deepfurniture import (
    DeepFurnitureError,
    assign_split,
    build_manifest,
    summary_path_for,
)
from product_retrieval.data.manifest import load_manifest
from product_retrieval.data.store import ImageStore

runner = CliRunner()

# Pinned hash buckets of the split rule (sha256 of canonical json, seed 0).
EXPECTED_SPLIT = {
    "1": "test",
    "2": "test",
    "3": "train",
    "4": "test",
    "5": "test",
    "6": "val",
    "7": "train",
    "8": "val",
    "9": "train",
    "10": "test",
}
# query file name -> archive index
QUERIES = {
    "1_1_100.jpg": 0,
    "1_2_101.jpg": 0,
    "3_1_102.jpg": 0,
    "6_1_103.jpg": 1,
    "7_1_104.jpg": 1,
    "7_2_105.jpg": 1,
    "7_3_106.jpg": 0,
}


def _img(name: str) -> bytes:
    return f"image-bytes-{name}".encode()


def _tar(path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=f"./{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _make_raw(root, queries=None, previews=None, shared_preview=None):
    """Ten identities (ids 1..10); category = id % 3 + 1."""
    queries = QUERIES if queries is None else queries
    meta = root / "metadata"
    meta.mkdir(parents=True)
    with open(meta / "furnitures.jsonl", "w") as f:
        for i in range(1, 11):
            f.write(json.dumps({"furniture_id": str(i), "category_id": i % 3 + 1, "style_ids": [1]}) + "\n")
    (meta / "categories.json").write_text(json.dumps({"1": "a", "2": "b", "3": "c"}))
    (meta / "query_index.json").write_text(
        json.dumps(
            {"queries": [{"query_name": n, "chunk_name": f"queries_{a:04d}"} for n, a in queries.items()]}
        )
    )
    ids = [str(i) for i in range(1, 11)] if previews is None else previews
    previews_data = {f"{i}.jpg": _img(f"preview-{i}") for i in ids}
    if shared_preview:
        previews_data[f"{shared_preview[1]}.jpg"] = previews_data[f"{shared_preview[0]}.jpg"]
    names = sorted(previews_data)
    _tar(root / "furnitures" / "furnitures_0000.tar.gz", {n: previews_data[n] for n in names[:5]})
    _tar(root / "furnitures" / "furnitures_0001.tar.gz", {n: previews_data[n] for n in names[5:]})
    for a in (0, 1):
        _tar(
            root / "queries" / f"queries_{a:04d}.tar.gz",
            {n: _img(n) for n, idx in queries.items() if idx == a},
        )


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root)
    return root


def test_assign_split_is_pinned_and_deterministic():
    assert {fid: assign_split(fid) for fid in EXPECTED_SPLIT} == EXPECTED_SPLIT
    assert assign_split("1", seed=0) == assign_split("1", seed=0)
    # a different seed re-draws the buckets
    assert {fid: assign_split(fid, seed=1) for fid in EXPECTED_SPLIT} != EXPECTED_SPLIT


def test_manifest_shape_and_query_mapping(raw, tmp_path):
    out = tmp_path / "out" / "manifest.jsonl"
    build_manifest(raw, tmp_path / "data", out)

    rows = {r.product_id: r for r in load_manifest(out)}
    assert sorted(rows, key=int) == [str(i) for i in range(1, 11)]
    assert all(r.source == "deepfurniture" for r in rows.values())
    assert {pid: r.split for pid, r in rows.items()} == EXPECTED_SPLIT
    assert rows["7"].query == sorted(
        (sha256_bytes(_img(n)) for n in ("7_1_104.jpg", "7_2_105.jpg", "7_3_106.jpg")),
        key=lambda s: ["7_1_104.jpg", "7_2_105.jpg", "7_3_106.jpg"].index(
            next(n for n in QUERIES if sha256_bytes(_img(n)) == s)
        ),
    )
    assert rows["1"].query == [sha256_bytes(_img("1_1_100.jpg")), sha256_bytes(_img("1_2_101.jpg"))]
    assert rows["3"].query == [sha256_bytes(_img("3_1_102.jpg"))]
    for pid, row in rows.items():
        assert row.gallery == [sha256_bytes(_img(f"preview-{pid}"))]
    store = ImageStore(tmp_path / "data", "deepfurniture")
    for row in rows.values():
        for sha in [*row.query, *row.gallery]:
            assert store.read_bytes(sha)


def test_identities_never_in_two_splits(raw, tmp_path):
    out = tmp_path / "manifest.jsonl"
    build_manifest(raw, tmp_path / "data", out)
    rows = load_manifest(out)
    ids = [r.product_id for r in rows]
    assert len(ids) == len(set(ids)) == 10


def test_distractors_stay_gallery_only_in_their_split(raw, tmp_path):
    out = tmp_path / "manifest.jsonl"
    build_manifest(raw, tmp_path / "data", out)
    rows = {r.product_id: r for r in load_manifest(out)}
    # identity 2 has no queries and its hash puts it in test, next to the query identity 1
    assert rows["2"].query == []
    assert rows["2"].split == "test"
    assert len(rows["2"].gallery) == 1
    # every split holds distractors
    for split in ("train", "val", "test"):
        assert any(r.split == split and not r.query for r in rows.values())


def test_summary_counts_hand_checked(raw, tmp_path):
    out = tmp_path / "manifest.jsonl"
    summary = build_manifest(raw, tmp_path / "data", out)

    # train: ids 3, 7, 9 (queries on 3 and 7: 1 + 3); val: 6, 8 (query on 6: 1);
    # test: 1, 2, 4, 5, 10 (queries on 1: 2)
    assert summary["counts"] == {
        "train": {"identities": 3, "identities_with_queries": 2, "queries": 4, "gallery_images": 3},
        "val": {"identities": 2, "identities_with_queries": 1, "queries": 1, "gallery_images": 2},
        "test": {"identities": 5, "identities_with_queries": 1, "queries": 2, "gallery_images": 5},
    }
    # category = id % 3 + 1
    assert summary["category_counts"] == {
        "train": {"1": 2, "2": 1},
        "val": {"1": 1, "3": 1},
        "test": {"2": 3, "3": 2},
    }
    assert summary["seed"] == 0
    assert summary["version"] == "deepfurniture-manifest-v1"
    assert summary["images_written"] == 10 + 7
    assert summary["shared_image_shas"] == {}
    names = [a["name"] for a in summary["archives"]]
    assert names == [
        "furnitures/furnitures_0000.tar.gz",
        "furnitures/furnitures_0001.tar.gz",
        "queries/queries_0000.tar.gz",
        "queries/queries_0001.tar.gz",
    ]
    assert all(len(a["sha256"]) == 64 for a in summary["archives"])
    assert json.loads(summary_path_for(out).read_text()) == summary


def test_rerun_is_idempotent_and_does_not_rewrite_images(raw, tmp_path):
    out = tmp_path / "manifest.jsonl"
    data = tmp_path / "data"
    first = build_manifest(raw, data, out)
    manifest_bytes = out.read_bytes()
    images = sorted((data / "deepfurniture" / "images").rglob("*.img"))
    mtimes = {p: p.stat().st_mtime_ns for p in images}
    assert len(images) == 17

    second = build_manifest(raw, data, out)

    assert out.read_bytes() == manifest_bytes
    assert second["images_written"] == 0 and first["images_written"] == 17
    assert {
        p: p.stat().st_mtime_ns for p in sorted((data / "deepfurniture" / "images").rglob("*.img"))
    } == mtimes
    assert not list((data / "deepfurniture" / "images").rglob("*.tmp"))


def test_query_without_preview_fails(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root, queries={**QUERIES, "99_1_200.jpg": 0})
    with pytest.raises(DeepFurnitureError, match="99_1_200.jpg"):
        build_manifest(root, tmp_path / "data", tmp_path / "m.jsonl")
    assert not (tmp_path / "m.jsonl").exists()


def test_identity_without_preview_fails(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root, previews=[str(i) for i in range(1, 10)])
    with pytest.raises(DeepFurnitureError, match="no preview"):
        build_manifest(root, tmp_path / "data", tmp_path / "m.jsonl")


def test_malformed_query_name_fails(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root, queries={**QUERIES, "1_oops.jpg": 0})
    with pytest.raises(DeepFurnitureError, match="1_oops.jpg"):
        build_manifest(root, tmp_path / "data", tmp_path / "m.jsonl")


def test_query_archive_disagreeing_with_index_fails(raw, tmp_path):
    index = raw / "metadata" / "query_index.json"
    data = json.loads(index.read_text())
    data["queries"].pop()
    index.write_text(json.dumps(data))
    with pytest.raises(DeepFurnitureError, match="query_index"):
        build_manifest(raw, tmp_path / "data", tmp_path / "m.jsonl")


def test_corrupt_archive_fails(raw, tmp_path):
    target = raw / "queries" / "queries_0001.tar.gz"
    target.write_bytes(target.read_bytes()[:-40])
    with pytest.raises(DeepFurnitureError, match="queries_0001"):
        build_manifest(raw, tmp_path / "data", tmp_path / "m.jsonl")


def test_empty_member_fails(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root)
    _tar(root / "furnitures" / "furnitures_0001.tar.gz", {"6.jpg": b""})
    with pytest.raises(DeepFurnitureError, match="empty or truncated"):
        build_manifest(root, tmp_path / "data", tmp_path / "m.jsonl")


def test_shared_image_between_identities_is_recorded_and_fails(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root, shared_preview=("4", "5"))  # identity 5's preview is identity 4's bytes
    out = tmp_path / "m.jsonl"
    with pytest.raises(DeepFurnitureError, match="shared"):
        build_manifest(root, tmp_path / "data", out)
    assert not out.exists()
    summary = json.loads(summary_path_for(out).read_text())
    assert summary["shared_image_shas"] == {sha256_bytes(_img("preview-4")): ["4", "5"]}


def test_cli_prepare_then_data_check_passes(raw, tmp_path):
    out = tmp_path / "manifest.jsonl"
    data = tmp_path / "data"
    result = runner.invoke(
        app,
        [
            "prepare-deepfurniture",
            "--raw",
            str(raw),
            "--data-root",
            str(data),
            "--out",
            str(out),
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["counts"]["test"]["identities"] == 5

    check = runner.invoke(
        app,
        ["data-check", "--manifest", str(out), "--source", "deepfurniture", "--data-root", str(data)],
    )
    assert check.exit_code == 0, check.output
    report = json.loads(check.stdout)
    assert report["ok"] is True
    assert report["missing_image_shas"] == [] and report["cross_split_products"] == []


def test_cli_reports_failure_with_exit_code_1(tmp_path):
    root = tmp_path / "raw"
    _make_raw(root, queries={**QUERIES, "99_1_200.jpg": 0})
    result = runner.invoke(
        app,
        [
            "prepare-deepfurniture",
            "--raw",
            str(root),
            "--data-root",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "m.jsonl"),
        ],
    )
    assert result.exit_code == 1
