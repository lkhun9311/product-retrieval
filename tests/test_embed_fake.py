import numpy as np
from PIL import Image

from product_retrieval.embed.fake import FakeEmbedder


def _image(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_embed_is_deterministic_for_same_image_content():
    embedder = FakeEmbedder(dim=16)
    a = _image((10, 20, 30))
    b = _image((10, 20, 30))

    vec_a = embedder.embed([a])
    vec_b = embedder.embed([b])

    assert np.array_equal(vec_a, vec_b)


def test_embed_differs_for_different_image_content():
    embedder = FakeEmbedder(dim=16)
    a = _image((10, 20, 30))
    b = _image((40, 50, 60))

    vec_a, vec_b = embedder.embed([a, b])

    assert not np.array_equal(vec_a, vec_b)


def test_embed_rows_are_l2_normalized():
    embedder = FakeEmbedder(dim=16)
    vecs = embedder.embed([_image((1, 2, 3)), _image((200, 150, 100))])

    assert vecs.dtype == np.float32
    assert vecs.shape == (2, 16)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_embed_empty_batch_returns_empty_array():
    embedder = FakeEmbedder(dim=16)
    vecs = embedder.embed([])
    assert vecs.shape == (0, 16)


def test_model_id_and_dim_attributes():
    embedder = FakeEmbedder(dim=8, model_id="fake-test")
    assert embedder.model_id == "fake-test"
    assert embedder.dim == 8
