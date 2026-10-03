import io
import json

import pytest
from PIL import Image

from product_retrieval.core.config import ExperimentConfig
from product_retrieval.core.ids import sha256_bytes
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.evaluate import IndexNotFoundError, run_eval
from product_retrieval.pipelines.selection import TestSplitAccessError

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
    """A manifest with 3 val products (each with 1 gallery + 1 query image) and
    1 train product, backed by real synthetic PNGs under a tmp data root.
    """
    data_root = tmp_path / "data"
    rows = []
    for i, product_id in enumerate(["p1", "p2", "p3"]):
        # Query == gallery image bytes per product so FakeEmbedder (pixel-bytes-derived)
        # gives the query its own product's exact vector -- a deterministic R@1 hit,
        # independent of any real visual-similarity semantics.
        color = (10 * (i + 1), 0, 0)
        gallery = [_put_image(data_root, SOURCE, color)]
        query = [_put_image(data_root, SOURCE, color)]
        rows.append(
            {
                "source": SOURCE,
                "product_id": product_id,
                "split": "val",
                "query": query,
                "gallery": gallery,
            }
        )
    train_gallery = [_put_image(data_root, SOURCE, (0, 0, 10))]
    rows.append(
        {
            "source": SOURCE,
            "product_id": "p4",
            "split": "train",
            "query": [],
            "gallery": train_gallery,
        }
    )

    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, rows)

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


def test_eval_end_to_end_after_build_index(manifest_setup):
    config, tmp_path = manifest_setup
    artifacts_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    build_result = run_build_index(
        config, split="val", embedder_name="fake", artifacts_root=artifacts_root, reports_root=reports_root
    )

    eval_result = run_eval(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts_root,
        reports_root=reports_root,
    )

    assert eval_result.index_id == build_result.index_id
    assert eval_result.report_path.is_file()
    assert eval_result.rankings_path.is_file()

    report = json.loads(eval_result.report_path.read_text(encoding="utf-8"))
    assert report == eval_result.report
    assert report["index_id"] == build_result.index_id
    assert report["counts"]["products"] == 3
    assert report["counts"]["queries"] == 3
    assert report["metrics"]["recall"]["macro"]["R@1"] == 1.0  # exact-match fake embeddings
    assert report["metrics"]["recall"]["micro"]["R@1"] == 1.0
    assert "R@1" in report["ci"]["recall"]["macro"]

    rankings = [
        json.loads(line) for line in eval_result.rankings_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rankings) == 3
    for row in rankings:
        assert row["truth_product_id"] in row["top_k_product_ids"]
        assert len(row["top_k_product_ids"]) == len(row["scores"])


def test_eval_refuses_test_without_final(manifest_setup):
    config, tmp_path = manifest_setup

    with pytest.raises(TestSplitAccessError):
        run_eval(
            config,
            split="test",
            final=False,
            embedder_name="fake",
            artifacts_root=tmp_path / "artifacts",
            reports_root=tmp_path / "reports",
        )


def test_eval_errors_clearly_when_index_missing(manifest_setup):
    config, tmp_path = manifest_setup

    with pytest.raises(IndexNotFoundError, match="build-index"):
        run_eval(
            config,
            split="val",
            embedder_name="fake",
            artifacts_root=tmp_path / "artifacts",
            reports_root=tmp_path / "reports",
        )


def test_eval_is_deterministic(manifest_setup):
    config, tmp_path = manifest_setup
    artifacts_root = tmp_path / "artifacts"

    run_build_index(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts_root,
        reports_root=tmp_path / "reports_build",
    )

    first = run_eval(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts_root,
        reports_root=tmp_path / "reports_a",
        seed=0,
    )
    second = run_eval(
        config,
        split="val",
        embedder_name="fake",
        artifacts_root=artifacts_root,
        reports_root=tmp_path / "reports_b",
        seed=0,
    )

    # Every field is deterministic except wall-clock per-stage timings.
    first_report = {k: v for k, v in first.report.items() if k != "seconds"}
    second_report = {k: v for k, v in second.report.items() if k != "seconds"}
    assert first_report == second_report

    first_rankings = first.rankings_path.read_text(encoding="utf-8")
    second_rankings = second.rankings_path.read_text(encoding="utf-8")
    assert first_rankings == second_rankings


def test_eval_matches_limited_build_index_id(manifest_setup):
    """eval with --limit-products must compute the same index_id build-index did."""
    config, tmp_path = manifest_setup
    artifacts_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    build_result = run_build_index(
        config,
        split="val",
        embedder_name="fake",
        limit_products=1,
        artifacts_root=artifacts_root,
        reports_root=reports_root,
    )

    eval_result = run_eval(
        config,
        split="val",
        embedder_name="fake",
        limit_products=1,
        artifacts_root=artifacts_root,
        reports_root=reports_root,
    )

    assert eval_result.index_id == build_result.index_id
    assert eval_result.report["counts"]["products"] == 1
