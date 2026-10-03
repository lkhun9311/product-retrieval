"""The ``Embedder`` contract (D20 section 2): image -> L2-normalized vector.

Every embedder implementation (``SiglipEmbedder``, ``FakeEmbedder``, ...) satisfies
this ``Protocol`` structurally -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class Embedder(Protocol):
    """Image -> embedding vector. Implementations must be deterministic given the same
    model weights and inputs.
    """

    model_id: str
    """Identifies this embedder for cache keys and ``core.ids.index_id`` (D20 section 1)."""

    dim: int
    """Length of each embedding vector."""

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a batch of RGB images.

        Returns a ``(len(images), self.dim)`` float32 array; each row is L2-normalized
        (unit norm), so inner product equals cosine similarity.
        """
        ...
