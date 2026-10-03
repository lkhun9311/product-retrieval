"""Load an experiment YAML into a validated, hashable config (D20 section 7).

One experiment = one YAML file (model id, crop policy, index params, seed, data
manifest). The resulting config hash is recorded alongside result reports so a
reported number can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from product_retrieval.core.ids import canonical_json, sha256_bytes
from product_retrieval.core.schemas import CropKind

DEFAULT_DATA_ROOT = Path(os.path.expanduser("~/data/product-retrieval"))


class ExperimentConfig(BaseModel):
    """A single experiment's settings, as loaded from a YAML file."""

    model_config = ConfigDict(frozen=True)

    name: str
    seed: int
    embed_model_id: str
    crop_kind: CropKind
    index_params: dict[str, Any] = Field(default_factory=dict)
    manifest_path: Path
    data_root: Path = Field(default_factory=lambda: DEFAULT_DATA_ROOT)

    @field_validator("manifest_path", "data_root", mode="before")
    @classmethod
    def _expand_path(cls, v: Any) -> Any:
        if isinstance(v, str):
            return Path(os.path.expanduser(v))
        return v


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file and validate it into an ``ExperimentConfig``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def config_hash(config: ExperimentConfig) -> str:
    """Deterministic hash of the config's content, independent of field order."""
    data = config.model_dump(mode="json")
    return sha256_bytes(canonical_json(data).encode("utf-8"))
