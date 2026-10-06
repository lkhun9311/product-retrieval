"""Pydantic v2 models for the D20 section 3 data contracts.

Every model that represents an immutable record (a fact that is written once
and never mutated afterwards) is frozen. Field values are validated at
construction time; invalid data raises ``pydantic.ValidationError``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

Source = Literal["lrvs", "amazon", "deepfurniture", "own", "capture"]
Split = Literal["train", "val", "test"]
CropKind = Literal["full", "box", "mask"]
CropModel = Literal["sam21", "sam3"]
Decision = Literal["same", "similar", "abstain"]
Action = Literal["match", "not_match", "prefer", "point", "skip"]
Actor = Literal["sim", "human"]
LabelKind = Literal["pos", "neg", "pref"]


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"expected a 64-character lowercase hex sha256 digest, got {value!r}")
    return value


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float


class Product(BaseModel):
    """A gallery product. The truth key is ``product_id`` (same model + color)."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    source: Source
    split: Split
    gallery_shas: list[str] = Field(default_factory=list)
    brand: str | None = None
    title: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gallery_shas")
    @classmethod
    def _check_gallery_shas(cls, v: list[str]) -> list[str]:
        return [_validate_sha256(sha) for sha in v]


class Query(BaseModel):
    """A query image to be matched against the gallery."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    image_sha: str
    truth_product_id: str
    source: Source
    scene: dict[str, Any] = Field(default_factory=dict)

    @field_validator("image_sha")
    @classmethod
    def _check_image_sha(cls, v: str) -> str:
        return _validate_sha256(v)


class CropSpec(BaseModel):
    """How a product region is cropped out of a query image.

    ``crop_hash`` (see ``core.ids.crop_hash``) is the hash of this spec's
    normalized JSON representation.
    """

    model_config = ConfigDict(frozen=True)

    kind: CropKind
    box: BBox | None = None
    mask_sha: str | None = None
    model: CropModel | None = None

    @field_validator("mask_sha")
    @classmethod
    def _check_mask_sha(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_sha256(v)

    @model_validator(mode="after")
    def _check_kind_matches_fields(self) -> CropSpec:
        if self.kind == "box":
            if self.box is None:
                raise ValueError("box is required when kind='box'")
        elif self.box is not None:
            raise ValueError("box is only allowed when kind='box'")

        if self.kind == "mask":
            if self.mask_sha is None:
                raise ValueError("mask_sha is required when kind='mask'")
        elif self.mask_sha is not None:
            raise ValueError("mask_sha is only allowed when kind='mask'")

        return self


class Candidate(BaseModel):
    """A single retrieved/reranked candidate product for a query."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    image_sha: str
    score_retrieval: float
    score_rerank: float | None = None
    rank: int

    @field_validator("image_sha")
    @classmethod
    def _check_image_sha(cls, v: str) -> str:
        return _validate_sha256(v)


class LatencyMs(BaseModel):
    """Per-stage latency in milliseconds. Stages that did not run are None."""

    model_config = ConfigDict(frozen=True)

    crop: float | None = None
    embed: float | None = None
    search: float | None = None
    rerank: float | None = None


class SearchResult(BaseModel):
    """The outcome of a single ``/search`` call. ``list_id`` links feedback back."""

    model_config = ConfigDict(frozen=True)

    list_id: str
    bundle_id: str
    decision: Decision
    candidates: list[Candidate] = Field(default_factory=list)
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)


class FeedbackEvent(BaseModel):
    """A single user (or simulator) feedback action. ``event_id`` is an idempotency key."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    ts: datetime
    session_id: str
    list_id: str
    query_id: str
    product_id: str
    action: Action
    position: int | None = None
    policy_id: str | None = None
    actor: Actor


class Label(BaseModel):
    """A training label derived from one or more feedback events."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    product_id: str
    kind: LabelKind
    weight: float
    origin_events: list[str] = Field(default_factory=list)
    label_version: str


class Bundle(BaseModel):
    """A deployable, versioned set of model/index/reranker/calibrator components."""

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    embed_model_id: str
    crop_policy: str
    index_id: str
    reranker_version: str | None = None
    calibrator_version: str | None = None
    created_at: datetime


class Gate(BaseModel):
    """Gate decision attached to an ``EvalReport``."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    passed: bool = Field(alias="pass")
    reasons: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Evaluation metrics for a bundle against a fixed dataset manifest."""

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    dataset_manifest_sha: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    ci: dict[str, Any] = Field(default_factory=dict)
    gate: Gate

    @field_validator("dataset_manifest_sha")
    @classmethod
    def _check_dataset_manifest_sha(cls, v: str) -> str:
        return _validate_sha256(v)
