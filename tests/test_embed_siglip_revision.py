"""SiglipEmbedder passes a revision to model and processor only when one is given (no download)."""

import pytest
import torch

from product_retrieval.embed import siglip


class _Inputs(dict):
    def to(self, device):
        return self


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, **inputs):
        return torch.ones(1, 4)


class _FakeProcessor:
    def __call__(self, images, return_tensors):
        return _Inputs(pixel_values=torch.zeros(len(images), 3, 2, 2))


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def loader(cls_name):
        def from_pretrained(model_id, **kwargs):
            seen.append((cls_name, model_id, kwargs))
            return _FakeModel() if cls_name == "model" else _FakeProcessor()

        return from_pretrained

    monkeypatch.setattr(siglip.AutoModel, "from_pretrained", loader("model"))
    monkeypatch.setattr(siglip.AutoProcessor, "from_pretrained", loader("processor"))
    return seen


def test_default_loads_exactly_as_before(calls):
    emb = siglip.SiglipEmbedder(model_id="m")
    assert calls == [("model", "m", {}), ("processor", "m", {})]
    assert emb.revision is None and emb.dim == 4


def test_revision_reaches_model_and_processor(calls):
    emb = siglip.SiglipEmbedder(model_id="m", revision="abc")
    assert calls == [("model", "m", {"revision": "abc"}), ("processor", "m", {"revision": "abc"})]
    assert emb.revision == "abc"
