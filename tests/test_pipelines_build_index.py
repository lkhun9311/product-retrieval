import io
import json

import pytest
from PIL import Image

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.index.flat import GalleryIndex
from product_retrieval.pipelines.build_index import TestSplitAccessError, run_build_index

SOURCE = "lrvs"


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _put_image(data_root, source: str, color: tuple[int, int, int]) -> str:
    data = _png_bytes(color)
    sha = sha256_bytes(data)
    path = data_root / source / "images" / sha[:2] / f"{sha}.img"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha


def _write_manifest(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def manifest_setup(tmp_path):
    """A manifest with 2 val products (p1: 2 gallery + 1 query, p2: 1 gallery + 1
    query) and 1 train product, backed by real synthetic PNGs under a tmp data root.
    """
    data_root = tmp_path / "data"
    p1_gallery = [_put_image(data_root, SOURCE, (10, 0, 0)), _put_image(data_root, SOURCE, (20, 0, 0))]
    p1_query = [_put_image(data_root, SOURCE, (30, 0, 0))]
    p2_gallery = [_put_image(data_root, SOURCE, (0, 10, 0))]
    p2_query = [_put_image(data_root, SOURCE, (0, 20, 0))]
    train_gallery = [_put_image(data_root, SOURCE, (0, 0, 10))]

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "source": SOURCE,
                "product_id": "p1",
                "split": "val",
                "query": p1_query,
                "gallery": p1_gallery,
            },
            {
                "source": SOURCE,
                "product_id": "p2",
                "split": "val",
                "query": p2_query,
                "gallery": p2_gallery,
            },
            {
                "source": SOURCE,
                "product_id": "p3",
                "split": "train",
                "query": [],
                "gallery": train_gallery,
            },
        ],
    )

    config = ExperimentConfig(
        name="test",
        seed=0,
        embed_model_id="unused-for-fake",
        crop_kind="full",
        index_params={},
        manifest_path=manifest_path,
        data_root=data_root,
    )
    return config, tmp_path


def test_build_index_end_to_end_with_fake_embedder(manifest_setup):
    config, tmp_path = manifest_setup
    artifacts_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    result = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts_root,
        reports_root=reports_root,
    )

    assert result.index_dir == artifacts_root / "index" / result.index_id
    assert (result.index_dir / "faiss.index").is_file()
    assert (result.index_dir / "ids.jsonl").is_file()
    assert (result.index_dir / "params.json").is_file()
    assert result.summary_path.is_file()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary == result.summary
    assert summary["counts"]["products"] == 2  # train product excluded
    assert summary["counts"]["gallery_images"] == 3  # 2 for p1, 1 for p2
    assert summary["counts"]["queries"] == 2
    assert summary["model_id"] == "fake"
    assert summary["index_id"] == result.index_id

    loaded = GalleryIndex.load(result.index_dir)
    assert loaded.ntotal == 3
    assert set(loaded.product_ids) == {"p1", "p2"}


def test_build_index_is_resumable_via_cache(manifest_setup):
    config, tmp_path = manifest_setup
    artifacts_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    first = run_build_index(
        config, split="val", embedder_name="fake", artifacts_root=artifacts_root, reports_root=reports_root
    )
    second = run_build_index(
        config, split="val", embedder_name="fake", artifacts_root=artifacts_root, reports_root=reports_root
    )

    assert first.index_id == second.index_id
    loaded_first = GalleryIndex.load(first.index_dir)
    loaded_second = GalleryIndex.load(second.index_dir)
    assert loaded_first.product_ids == loaded_second.product_ids
    assert loaded_first.image_shas == loaded_second.image_shas


def test_limit_products_selects_first_n_by_sorted_product_id(manifest_setup):
    config, tmp_path = manifest_setup

    result = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        limit_products=1,
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
    )

    loaded = GalleryIndex.load(result.index_dir)
    assert set(loaded.product_ids) == {"p1"}  # "p1" < "p2" sorted
    assert result.summary["counts"]["products"] == 1


def test_crop_kind_other_than_full_raises_not_implemented(manifest_setup):
    config, tmp_path = manifest_setup
    config = config.model_copy(update={"crop_kind": "box"})

    with pytest.raises(NotImplementedError):
        run_build_index(
            config,
            split="val",
            embedder_name="fake",
            artifacts_root=tmp_path / "artifacts",
            reports_root=tmp_path / "reports",
        )


def test_test_split_refused_without_final(manifest_setup):
    config, tmp_path = manifest_setup
    reports_root = tmp_path / "reports"

    with pytest.raises(TestSplitAccessError):
        run_build_index(
            config,
            split="test",
            final=False,
            embedder_name="fake",
            artifacts_root=tmp_path / "artifacts",
            reports_root=reports_root,
        )

    assert not (reports_root / "test_access.jsonl").exists()


def test_test_split_allowed_and_logged_with_final(manifest_setup):
    config, tmp_path = manifest_setup
    reports_root = tmp_path / "reports"

    # No products are in the "test" split in this fixture, so the pipeline runs
    # to completion over an empty gallery -- what matters here is that access is
    # refused/logged correctly, independent of whether "test" has data.
    result = run_build_index(
        config,
        split="test",
        final=True,
        embedder_name="fake",
        artifacts_root=tmp_path / "artifacts",
        reports_root=reports_root,
    )

    assert result.summary["counts"]["products"] == 0

    access_log = reports_root / "test_access.jsonl"
    assert access_log.is_file()
    lines = [json.loads(line) for line in access_log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["split"] == "test"
    assert lines[0]["config_hash"] == result.summary["config_hash"]
    assert "ts" in lines[0]


def test_index_id_differs_between_limited_and_full_val_build(manifest_setup):
    """Regression: a --limit-products build and a full build from the same manifest
    must get different index_ids (so they don't silently overwrite each other's
    artifacts/index/{index_id}/ directory), and re-running the same subset must
    reproduce the same index_id.
    """
    config, tmp_path = manifest_setup

    full = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
    )
    limited = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        limit_products=1,
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
    )
    limited_again = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        limit_products=1,
        artifacts_root=tmp_path / "artifacts",
        reports_root=tmp_path / "reports",
    )

    assert full.index_id != limited.index_id
    assert limited.index_id == limited_again.index_id
    assert full.index_dir != limited.index_dir
    assert GalleryIndex.load(full.index_dir).ntotal == 3  # p1 (2 gallery) + p2 (1 gallery)
    assert GalleryIndex.load(limited.index_dir).ntotal == 2  # p1 only (sorted first)


def test_test_split_access_appends_each_time(manifest_setup):
    config, tmp_path = manifest_setup
    reports_root = tmp_path / "reports"

    for _ in range(2):
        run_build_index(
            config,
            split="test",
            final=True,
            embedder_name="fake",
            artifacts_root=tmp_path / "artifacts",
            reports_root=reports_root,
        )

    access_log = reports_root / "test_access.jsonl"
    lines = access_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
