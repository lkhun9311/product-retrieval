import io
import json

import yaml
from PIL import Image
from typer.testing import CliRunner

from product_retrieval import __version__
from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_bytes

runner = CliRunner()

SHA_Q1 = "1" * 64
SHA_G1 = "3" * 64


def _write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _put_image(tmp_path, source, sha):
    image_dir = tmp_path / source / "images" / sha[:2]
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{sha}.img").write_bytes(b"fake-bytes")


def _put_png_image(data_root, source, color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    data = buf.getvalue()
    sha = sha256_bytes(data)
    image_dir = data_root / source / "images" / sha[:2]
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{sha}.img").write_bytes(data)
    return sha


def test_version_runs_and_prints_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_data_check_exits_0_when_ok(tmp_path):
    data_root = tmp_path / "data"
    _put_image(data_root, "lrvs", SHA_Q1)
    _put_image(data_root, "lrvs", SHA_G1)
    manifest = _write_manifest(
        tmp_path,
        [{"source": "lrvs", "product_id": "p1", "split": "test", "query": [SHA_Q1], "gallery": [SHA_G1]}],
    )

    result = runner.invoke(
        app,
        [
            "data-check",
            "--manifest",
            str(manifest),
            "--source",
            "lrvs",
            "--data-root",
            str(data_root),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["ok"] is True


def test_data_check_exits_1_when_image_missing(tmp_path):
    data_root = tmp_path / "data"  # SHA_Q1/SHA_G1 are never written
    manifest = _write_manifest(
        tmp_path,
        [{"source": "lrvs", "product_id": "p1", "split": "test", "query": [SHA_Q1], "gallery": [SHA_G1]}],
    )

    result = runner.invoke(
        app,
        [
            "data-check",
            "--manifest",
            str(manifest),
            "--source",
            "lrvs",
            "--data-root",
            str(data_root),
        ],
    )

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert SHA_Q1 in report["missing_image_shas"]


def test_build_index_requires_config():
    result = runner.invoke(app, ["build-index"])
    assert result.exit_code != 0


def _write_config(tmp_path, manifest_path, data_root):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli-test",
                "seed": 0,
                "embed_model_id": "unused-for-fake",
                "crop_kind": "full",
                "index_params": {},
                "manifest_path": str(manifest_path),
                "data_root": str(data_root),
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_build_index_runs_end_to_end_with_fake_embedder(tmp_path):
    data_root = tmp_path / "data"
    gallery_sha = _put_png_image(data_root, "lrvs", (10, 0, 0))
    query_sha = _put_png_image(data_root, "lrvs", (20, 0, 0))
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "source": "lrvs",
                "product_id": "p1",
                "split": "val",
                "query": [query_sha],
                "gallery": [gallery_sha],
            }
        ],
    )
    config_path = _write_config(tmp_path, manifest, data_root)

    result = runner.invoke(
        app,
        [
            "build-index",
            "--config",
            str(config_path),
            "--split",
            "val",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--reports-root",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["counts"]["products"] == 1
    assert (tmp_path / "artifacts" / "index" / summary["index_id"] / "faiss.index").is_file()


def test_build_index_refuses_test_split_without_final(tmp_path):
    data_root = tmp_path / "data"
    manifest = _write_manifest(tmp_path, [])
    config_path = _write_config(tmp_path, manifest, data_root)

    result = runner.invoke(
        app,
        [
            "build-index",
            "--config",
            str(config_path),
            "--split",
            "test",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--reports-root",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "test_access.jsonl").exists()


def test_build_index_allows_test_split_with_final_and_logs(tmp_path):
    data_root = tmp_path / "data"
    manifest = _write_manifest(tmp_path, [])
    config_path = _write_config(tmp_path, manifest, data_root)
    reports_root = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "build-index",
            "--config",
            str(config_path),
            "--split",
            "test",
            "--final",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--reports-root",
            str(reports_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (reports_root / "test_access.jsonl").is_file()


def test_eval_requires_config():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0


def test_eval_runs_end_to_end_after_build_index(tmp_path):
    data_root = tmp_path / "data"
    gallery_sha = _put_png_image(data_root, "lrvs", (10, 0, 0))
    query_sha = _put_png_image(data_root, "lrvs", (20, 0, 0))
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "source": "lrvs",
                "product_id": "p1",
                "split": "val",
                "query": [query_sha],
                "gallery": [gallery_sha],
            }
        ],
    )
    config_path = _write_config(tmp_path, manifest, data_root)
    artifacts_root = tmp_path / "artifacts"
    reports_root = tmp_path / "reports"

    build_result = runner.invoke(
        app,
        [
            "build-index",
            "--config",
            str(config_path),
            "--split",
            "val",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(artifacts_root),
            "--reports-root",
            str(reports_root),
        ],
    )
    assert build_result.exit_code == 0, build_result.output

    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(config_path),
            "--split",
            "val",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(artifacts_root),
            "--reports-root",
            str(reports_root),
        ],
    )

    assert eval_result.exit_code == 0, eval_result.output
    assert "macro R@K" in eval_result.output
    report_files = list((reports_root / "eval").glob("*.json"))
    assert len(report_files) == 1


def test_eval_errors_clearly_when_index_missing(tmp_path):
    data_root = tmp_path / "data"
    gallery_sha = _put_png_image(data_root, "lrvs", (10, 0, 0))
    query_sha = _put_png_image(data_root, "lrvs", (20, 0, 0))
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "source": "lrvs",
                "product_id": "p1",
                "split": "val",
                "query": [query_sha],
                "gallery": [gallery_sha],
            }
        ],
    )
    config_path = _write_config(tmp_path, manifest, data_root)

    result = runner.invoke(
        app,
        [
            "eval",
            "--config",
            str(config_path),
            "--split",
            "val",
            "--embedder",
            "fake",
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--reports-root",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 1
    assert "build-index" in result.output


def test_train_rerank_stub_exits_2():
    result = runner.invoke(app, ["train-rerank"])
    assert result.exit_code == 2


def test_serve_stub_exits_2():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 2
