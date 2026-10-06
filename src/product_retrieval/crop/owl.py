"""OWLv2 detector wrapper (contract: docs/contracts/crop-before-search.md, "Detector, frozen").

The model call lives here; every decision after the raw predictions is in ``crop.boxes`` (pure,
tested without the model). CPU, float32, pinned revision.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from product_retrieval.crop.boxes import Crop, crops_from_predictions

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
OWL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
PROMPTS: tuple[str, ...] = (
    "bed",
    "sofa",
    "armchair",
    "chair",
    "stool",
    "table",
    "desk",
    "shelf",
    "bookcase",
    "cabinet",
    "wardrobe",
    "chest of drawers",
    "lamp",
    "ceiling lamp",
    "mirror",
    "rug",
    "curtain",
    "cushion",
    "pillow",
    "quilt",
    "plant pot",
    "clock",
    "picture frame",
    "storage box",
)


class OwlDetector:
    """``google/owlv2-base-patch16-ensemble`` at a pinned revision, CPU float32."""

    def __init__(self, model_id: str = OWL_MODEL_ID, revision: str = OWL_REVISION) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.model_id = model_id
        self.revision: str | None = revision
        self._torch = torch
        self._processor = Owlv2Processor.from_pretrained(model_id, revision=revision)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id, revision=revision)
        self._model.to("cpu").to(torch.float32).eval()

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boxes (n, 4) xyxy in image pixels, scores (n,))`` for every predicted box.

        ``scores`` is the max sigmoid over the prompts (the processor takes the max logit, which is
        the same number). No thresholding here: ``threshold=0.0`` only drops scores that are exactly
        zero, and the relative order of the remaining boxes is unchanged.
        """
        torch = self._torch
        w, h = image.size
        side = max(h, w)  # OWLv2 pads to a square, so boxes are in max(H, W) coordinates
        inputs = self._processor(text=[list(PROMPTS)], images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=[(side, side)]
        )[0]
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float64)
        scores = result["scores"].detach().cpu().numpy().astype(np.float64)
        return boxes, scores

    def detect(self, image: Image.Image) -> list[Crop]:
        boxes, scores = self.predict(image)
        return crops_from_predictions(boxes, scores, image.width, image.height)
