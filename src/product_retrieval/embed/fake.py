"""A deterministic, dependency-free stand-in for a real embedder (tests, pipeline dry-runs).

``FakeEmbedder`` never loads a model or touches the network. Each image's vector is
derived purely from its pixel bytes, so the same image content always produces the
same vector, and different content (almost always) produces a different one.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class FakeEmbedder:
    """Deterministic fake embedder satisfying the ``Embedder`` protocol."""

    def __init__(self, dim: int = 16, model_id: str = "fake") -> None:
        self.dim = dim
        self.model_id = model_id

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(image) for image in images]).astype(np.float32)

    def _embed_one(self, image: Image.Image) -> np.ndarray:
        digest = hashlib.sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            norm = 1.0
        return vec / norm
