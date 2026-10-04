import io
import json

import numpy as np
import pytest
from PIL import Image

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.index.flat import GalleryIndex
from product_retrieval.pipelines.build_index import TestSplitAccessError, _embed_shas, run_build_index
from product_retrieval.pipelines.selection import select_split

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


# --- chunked embedding (_embed_shas) -------------------------------------------------


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records batch sizes and can fail on the Nth embed call."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.fail_on_call = fail_on_call

    def embed(self, images):
        self.batch_sizes.append(len(images))
        if self.fail_on_call is not None and len(self.batch_sizes) == self.fail_on_call:
            raise RuntimeError("boom")
        return super().embed(images)


@pytest.fixture
def many_shas(tmp_path):
    data_root = tmp_path / "data"
    shas = [_put_image(data_root, SOURCE, (i, 3, 7)) for i in range(5)]
    return data_root, [(sha, SOURCE) for sha in shas]


def _run_embed(pairs, data_root, cache_root, embedder, chunk_size):
    return _embed_shas(pairs, "crop", embedder, EmbeddingCache(cache_root), data_root, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 100])  # 1, non-multiple, exact multiple, > n
def test_chunked_embedding_matches_unchunked(many_shas, tmp_path, chunk_size):
    data_root, pairs = many_shas
    baseline = _run_embed(pairs, data_root, tmp_path / "c_base", FakeEmbedder(), 10_000)
    embedder = _CountingEmbedder()
    chunked = _run_embed(pairs, data_root, tmp_path / "c_chunk", embedder, chunk_size)

    assert chunked.keys() == baseline.keys()
    for sha in baseline:
        np.testing.assert_array_equal(chunked[sha], baseline[sha])
    expected_calls = -(-len(pairs) // chunk_size)
    assert len(embedder.batch_sizes) == expected_calls
    assert max(embedder.batch_sizes) <= chunk_size
    assert sum(embedder.batch_sizes) == len(pairs)
    _, missing = EmbeddingCache(tmp_path / "c_chunk").get_many([s for s, _ in pairs], "crop", "fake")
    assert missing == []


def test_crash_mid_way_keeps_first_chunk_and_rerun_embeds_only_remainder(many_shas, tmp_path):
    data_root, pairs = many_shas
    cache_root = tmp_path / "cache"
    shas = [s for s, _ in pairs]

    with pytest.raises(RuntimeError, match="boom"):
        _run_embed(pairs, data_root, cache_root, _CountingEmbedder(fail_on_call=2), 2)

    _, missing = EmbeddingCache(cache_root).get_many(shas, "crop", "fake")
    assert missing == shas[2:]  # first chunk (2 shas) survived

    rerun = _CountingEmbedder()
    result = _run_embed(pairs, data_root, cache_root, rerun, 2)
    assert sum(rerun.batch_sizes) == 3  # only the remainder
    baseline = _run_embed(pairs, data_root, tmp_path / "c_base", FakeEmbedder(), 10_000)
    for sha in shas:
        np.testing.assert_array_equal(result[sha], baseline[sha])


def test_fully_cached_rerun_does_not_call_embedder(many_shas, tmp_path):
    data_root, pairs = many_shas
    _run_embed(pairs, data_root, tmp_path / "cache", FakeEmbedder(), 2)
    embedder = _CountingEmbedder()
    _run_embed(pairs, data_root, tmp_path / "cache", embedder, 2)
    assert embedder.batch_sizes == []


def test_progress_line_goes_to_stderr(many_shas, tmp_path, capsys):
    data_root, pairs = many_shas
    _run_embed(pairs, data_root, tmp_path / "cache", FakeEmbedder(), 2)
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("embed: ")]
    assert len(lines) == 3
    assert lines[-1].startswith("embed: 5/5 done (cached 0)")
    assert "img/s" in lines[-1] and "eta" in lines[-1]
    assert captured.out == ""


@pytest.mark.parametrize("bad", [0, -1])
def test_chunk_size_below_one_is_rejected(many_shas, tmp_path, bad):
    data_root, pairs = many_shas
    with pytest.raises(ValueError, match="chunk_size"):
        _run_embed(pairs, data_root, tmp_path / "cache", FakeEmbedder(), bad)


def test_train_split_is_open_without_final(manifest_setup):
    config, tmp_path = manifest_setup
    reports_root = tmp_path / "reports"

    selection = select_split(config, "train", None, False, reports_root)

    assert [p.product_id for p in selection.products] == ["p3"]
    assert not (reports_root / "test_access.jsonl").exists()
