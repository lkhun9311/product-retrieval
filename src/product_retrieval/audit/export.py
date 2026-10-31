"""Review export for the product-ID audit (contract ladder-1-id-audit v1.1, section 7).

One JSONL line per judgement, append-only. A record names the images it compares by sha256 on
each side, so the product IDs are not needed to read it back: ``judged_groups`` rebuilds which
images were judged to be the same product from the records alone. A split shows up as one
ID's images falling into several groups, a merge as images of two IDs in one group, and an
unresolved replacement as an ``undecidable`` edge that joins nothing.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from product_retrieval.audit.sample import CONTRACT_VERSION

Stage = Literal["target", "C", "A", "B", "D", "titles"]
Unit = Literal["single", "set", "multipack"]
Relation = Literal["same", "consistent", "variant", "different", "undecidable"]
Round = Literal["first", "rereview"]
TargetFlag = Literal["ambiguous", "not_visible"]
_SHA = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-9


class ExportError(Exception):
    """A review record is invalid, or the export file cannot be appended to safely."""


class Box(BaseModel):
    """Target box, normalised to the image: x, y of the top-left corner, width, height."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _inside(self) -> Box:
        if self.x + self.w > 1 + _EPS or self.y + self.h > 1 + _EPS:
            raise ValueError("box extends past the image edge")
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tags: list[Annotated[str, Field(min_length=1)]]
    note: str


class ReviewRecord(BaseModel):
    """Every key is required; ``None`` is allowed only where the stage has nothing to put there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str
    stage: Stage
    display_id: str = Field(min_length=1)
    query_id: str | None
    left_sha: str
    right_sha: str | None
    target: Box | TargetFlag | None
    left_unit: Unit
    right_unit: Unit | None
    relation: Relation | None
    evidence: Evidence
    round: Round
    timestamp: str
    seconds: float

    @field_validator("contract_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if v != CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {CONTRACT_VERSION!r}, got {v!r}")
        return v

    @field_validator("left_sha", "right_sha")
    @classmethod
    def _sha(cls, v: str | None) -> str | None:
        if v is not None and not _SHA.match(v):
            raise ValueError("image sha must be 64 lowercase hex characters")
        return v

    @field_validator("timestamp")
    @classmethod
    def _timestamp(cls, v: str) -> str:
        try:
            parsed = datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"timestamp is not ISO 8601: {v!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp needs a UTC offset")
        return v

    @field_validator("seconds")
    @classmethod
    def _seconds(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("seconds must be finite and >= 0")
        return v

    @model_validator(mode="after")
    def _by_stage(self) -> ReviewRecord:
        if self.stage == "target":
            if self.right_sha is not None or self.right_unit is not None or self.relation is not None:
                raise ValueError("a target record has no right side and no relation")
            if self.target is None:
                raise ValueError("a target record needs a box, ambiguous or not_visible")
            return self
        if self.right_sha is None or self.right_unit is None or self.relation is None:
            raise ValueError(f"stage {self.stage} needs right_sha, right_unit and relation")
        if self.stage == "C" and (self.target is None or self.query_id is None):
            raise ValueError("a C record needs the target it was judged against and a query_id")
        return self


def validate_record(raw: dict) -> ReviewRecord:
    """Validate one record; any problem raises ``ExportError`` naming the field."""
    if not isinstance(raw, dict):
        raise ExportError(f"record is not an object: {type(raw).__name__}")
    try:
        return ReviewRecord.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ExportError(problems) from exc


def append_record(path: str | Path, raw: dict) -> ReviewRecord:
    """Validate and append one line. Existing lines are never touched.

    A file whose last byte is not a newline (a torn write) is refused, not repaired.
    """
    record = validate_record(raw)
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                raise ExportError(f"{path} does not end with a newline; refusing to append")
    line = json.dumps(record.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    return record


def read_export(path: str | Path) -> list[ReviewRecord]:
    """Read and validate every line; a bad line raises with its line number."""
    out: list[ReviewRecord] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                raise ExportError(f"{path}:{n}: blank line")
            try:
                out.append(validate_record(json.loads(line)))
            except (ValueError, ExportError) as exc:
                raise ExportError(f"{path}:{n}: {exc}") from exc
    return out


def judged_groups(records: list[ReviewRecord], include_rounds: Round | None = "first") -> dict:
    """Rebuild, from records alone, what was judged about the images.

    Returns ``groups`` (sets of image shas joined by ``same``; images seen but unjoined are
    singletons), and the pair sets ``consistent``, ``variant``, ``different``, ``undecidable``.
    With several records for one pair in the chosen round, the latest timestamp wins.
    ``include_rounds`` selects ``first`` or ``rereview`` records, or all when None.
    """
    latest: dict[frozenset, ReviewRecord] = {}
    images: set[str] = set()
    for r in records:
        if r.right_sha is None or (include_rounds is not None and r.round != include_rounds):
            continue
        images.update((r.left_sha, r.right_sha))
        key = frozenset((r.left_sha, r.right_sha))
        prev = latest.get(key)
        if prev is None or datetime.fromisoformat(r.timestamp) >= datetime.fromisoformat(prev.timestamp):
            latest[key] = r
    parent = {i: i for i in images}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    kinds = ("consistent", "variant", "different", "undecidable")
    pairs: dict[str, set[frozenset]] = {k: set() for k in kinds}
    for key, r in latest.items():
        if r.relation == "same":
            parent[find(r.left_sha)] = find(r.right_sha)  # type: ignore[index]
        elif r.relation is not None:
            pairs[r.relation].add(key)
    comps: dict[str, set[str]] = {}
    for i in images:
        comps.setdefault(find(i), set()).add(i)
    return {"groups": list(comps.values()), **pairs}
