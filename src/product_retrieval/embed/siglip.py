"""SigLIP2 image embedder (D20 section 2: ``embed`` module).

Wraps ``transformers`` ``AutoModel``/``AutoProcessor`` for ``google/siglip2-*``
checkpoints. The model is loaded once, in eval mode, and every forward pass runs
under ``torch.inference_mode()`` so results are deterministic (no dropout, no
gradient tracking) across repeated calls on the same weights and inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def _extract_image_features(output: object) -> torch.Tensor:
    """Pull the pooled image-feature tensor out of ``get_image_features``'s return value.

    Newer ``transformers`` versions return a ``BaseModelOutputWithPooling`` (or, with
    ``return_dict=False``, a tuple) instead of a bare tensor; older ones return the
    tensor directly. Handle all three so this keeps working across versions.
    """
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    if isinstance(output, tuple):
        return output[-1]
    return output  # type: ignore[return-value]


class SiglipEmbedder:
    """SigLIP2 embedder satisfying the ``Embedder`` protocol."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size

        self._model = AutoModel.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_id)

        # Determine the projected image-feature dimension by running one dummy
        # image through the model, rather than guessing a config attribute name.
        with torch.inference_mode():
            dummy = Image.new("RGB", (224, 224))
            inputs = self._processor(images=[dummy], return_tensors="pt").to(device)
            features = _extract_image_features(self._model.get_image_features(**inputs))
        self.dim = int(features.shape[-1])

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = _extract_image_features(self._model.get_image_features(**inputs))
            batches.append(features.detach().cpu().numpy().astype(np.float32))

        vectors = np.concatenate(batches, axis=0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (vectors / norms).astype(np.float32)
