"""Pure box logic for the crop-before-search check (contract: docs/contracts/crop-before-search.md).

Nothing here loads a model. Input is what the detector predicted (boxes in image pixels, per-prompt
logits); output is the crops to embed:

    max sigmoid over prompts -> clip -> score >= 0.10 -> greedy NMS (IoU 0.5) -> top M=10
    -> expand 10% -> clip -> round outward -> drop crops under 16 px

Choices where the contract is silent (reported, not hidden):
- NMS suppresses a box when IoU is strictly greater than the threshold (IoU == 0.5 survives).
- Clipping happens before NMS, as the contract lists it. A degenerate (zero-area) clipped box still
  takes part in NMS with IoU 0 and can use one of the M slots; it is removed later by the 16 px rule.
- A box with zero union area has IoU 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

SCORE_THRESHOLD = 0.10
NMS_IOU = 0.5
TOP_M = 10
EXPAND_FRAC = 0.10
MIN_CROP_PX = 16


@dataclass(frozen=True)
class Detection:
    """A kept box in float pixels. ``index`` is its position among the predicted boxes."""

    index: int
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Crop:
    """A crop rectangle in integer pixels, ``(x1, y1, x2, y2)`` with x2/y2 exclusive for PIL."""

    index: int
    score: float
    box: tuple[int, int, int, int]


class Detector(Protocol):
    """Image (EXIF-transposed RGB) -> surviving crops in detector score order. May return none."""

    model_id: str
    revision: str | None

    def detect(self, image: Image.Image) -> list[Crop]: ...


def load_exif_rgb(path: str | Path) -> Image.Image:
    """Open an image, apply its EXIF orientation, then convert to RGB (before anything else)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def exif_orientation(path: str | Path) -> int:
    """EXIF orientation tag (1 when absent)."""
    with Image.open(path) as im:
        return int(im.getexif().get(0x0112, 1))


def max_sigmoid(logits: np.ndarray) -> np.ndarray:
    """``(n_boxes, n_prompts)`` logits -> ``(n_boxes,)`` max sigmoid score over prompts."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"logits must be (n_boxes, n_prompts), got shape {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError("logits have no prompts")
    if not np.isfinite(arr).all():
        raise ValueError("logits contain NaN or inf")
    return (1.0 / (1.0 + np.exp(-arr))).max(axis=1)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip ``(n, 4)`` xyxy boxes to ``[0, width] x [0, height]``."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(width))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(height))
    return b


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two xyxy boxes; 0 when the union has no area."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = NMS_IOU) -> list[int]:
    """Greedy NMS. Returns kept indices, highest score first; equal scores go by lower index.

    A candidate is suppressed when its IoU with an already kept box is strictly greater than the
    threshold.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64)
    if len(b) != len(s):
        raise ValueError(f"boxes and scores differ in length: {len(b)} != {len(s)}")
    if not np.isfinite(s).all() or not np.isfinite(b).all():
        raise ValueError("boxes or scores contain NaN or inf")
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    kept: list[int] = []
    for i in order:
        if all(iou(b[i], b[j]) <= iou_threshold for j in kept):
            kept.append(i)
    return kept


def select_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    threshold: float = SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU,
    top_m: int = TOP_M,
) -> list[Detection]:
    """Clip, keep score >= threshold, greedy NMS, first ``top_m``. Order: score descending."""
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(b) != len(s):
        raise ValueError(f"boxes and scores differ in length: {len(b)} != {len(s)}")
    if not np.isfinite(s).all() or not np.isfinite(b).all():
        raise ValueError("boxes or scores contain NaN or inf")
    clipped = clip_boxes(b, width, height)
    idx = np.flatnonzero(s >= threshold)  # original positions, ascending
    if idx.size == 0:
        return []
    kept_local = greedy_nms(clipped[idx], s[idx], iou_threshold)[:top_m]
    return [
        Detection(
            index=int(idx[i]),
            score=float(s[idx[i]]),
            box=tuple(float(v) for v in clipped[idx[i]]),  # type: ignore[arg-type]
        )
        for i in kept_local
    ]


def expand_and_round(
    box: tuple[float, float, float, float], width: int, height: int, frac: float = EXPAND_FRAC
) -> tuple[int, int, int, int]:
    """Grow the box by ``frac`` of its width/height on each side, clip, round outward to integers."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(x1 - frac * bw, 0.0), min(x2 + frac * bw, float(width))
    y1, y2 = max(y1 - frac * bh, 0.0), min(y2 + frac * bh, float(height))
    return (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))


def build_crops(
    detections: list[Detection], width: int, height: int, min_px: int = MIN_CROP_PX
) -> list[Crop]:
    """Expand and round every detection; drop crops narrower or shorter than ``min_px``."""
    crops: list[Crop] = []
    for d in detections:
        x1, y1, x2, y2 = expand_and_round(d.box, width, height)
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        crops.append(Crop(index=d.index, score=d.score, box=(x1, y1, x2, y2)))
    return crops


def crops_from_predictions(boxes: np.ndarray, scores: np.ndarray, width: int, height: int) -> list[Crop]:
    """Full box pipeline from predicted boxes/scores to crops (may be empty: caller falls back)."""
    return build_crops(select_detections(boxes, scores, width, height), width, height)
