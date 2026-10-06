"""A deterministic stand-in for the detector (tests, dry runs). Loads nothing.

It feeds hand-made predictions through the same pure box logic as the real detector.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from product_retrieval.crop.boxes import Crop, crops_from_predictions

Proposals = Callable[[Image.Image], tuple[np.ndarray, np.ndarray]]


def halves(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Left half (score 0.9), right half (score 0.8) and a low-score box (0.05, dropped)."""
    w, h = image.size
    boxes = np.array([[0, 0, w / 2, h], [w / 2, 0, w, h], [0, 0, w, h]], dtype=np.float64)
    return boxes, np.array([0.9, 0.8, 0.05])


class FakeDetector:
    model_id = "fake-detector"
    revision: str | None = None

    def __init__(self, proposals: Proposals = halves) -> None:
        self._proposals = proposals

    def detect(self, image: Image.Image) -> list[Crop]:
        boxes, scores = self._proposals(image)
        return crops_from_predictions(boxes, scores, image.width, image.height)
