import numpy as np
import pytest
from PIL import Image

MODEL_ID = "google/siglip2-base-patch16-224"


@pytest.mark.slow
def test_siglip_embedder_output_shape_and_normalization():
    from product_retrieval.embed.siglip import SiglipEmbedder

    embedder = SiglipEmbedder(model_id=MODEL_ID, device="cpu")
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (224, 224), color=(0, 0, 255)),
    ]

    vectors = embedder.embed(images)

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, embedder.dim)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-4)
