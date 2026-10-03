from pathlib import Path

import pytest
from pydantic import ValidationError

from product_retrieval.core.config import config_hash, load_config

BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


def test_load_baseline_config():
    config = load_config(BASELINE_CONFIG)
    assert config.name == "baseline"
    assert config.embed_model_id == "google/siglip2-base-patch16-224"
    assert config.crop_kind == "full"
    assert str(config.manifest_path).startswith(str(Path.home()))


def test_data_root_defaults_and_expands_home():
    config = load_config(BASELINE_CONFIG)
    assert config.data_root == Path.home() / "data" / "product-retrieval"


def test_config_hash_stable_for_same_content():
    config_a = load_config(BASELINE_CONFIG)
    config_b = load_config(BASELINE_CONFIG)
    assert config_hash(config_a) == config_hash(config_b)


def test_config_hash_changes_with_seed(tmp_path):
    from product_retrieval.core.config import ExperimentConfig

    base_kwargs = dict(
        name="x",
        embed_model_id="m",
        crop_kind="full",
        manifest_path=tmp_path / "manifest.jsonl",
    )
    cfg_a = ExperimentConfig(seed=0, **base_kwargs)
    cfg_b = ExperimentConfig(seed=1, **base_kwargs)
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_config_rejects_bad_crop_kind(tmp_path):
    with pytest.raises(ValidationError):
        from product_retrieval.core.config import ExperimentConfig

        ExperimentConfig(
            name="x",
            seed=0,
            embed_model_id="m",
            crop_kind="square",
            manifest_path=tmp_path / "manifest.jsonl",
        )
