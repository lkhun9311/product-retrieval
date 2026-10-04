from product_retrieval.core.ids import (
    bundle_id,
    canonical_json,
    crop_hash,
    gallery_sha,
    index_id,
    query_id,
    sha256_bytes,
    sha256_file,
)
from product_retrieval.core.schemas import BBox, CropSpec


def test_query_id_hides_product_id_and_has_source_prefix():
    qid = query_id("lrvs", "48213", "a" * 64)
    assert "48213" not in qid
    assert qid.startswith("lrvs:")
    assert len(qid) == len("lrvs:") + 16


def test_query_id_deterministic():
    assert query_id("lrvs", "48213", "a" * 64) == query_id("lrvs", "48213", "a" * 64)


def test_query_id_distinct_for_same_image_under_different_products():
    assert query_id("lrvs", "1", "a" * 64) != query_id("lrvs", "2", "a" * 64)


def test_query_id_distinct_across_sources_and_images():
    base = query_id("lrvs", "1", "a" * 64)
    assert base != query_id("capture", "1", "a" * 64)
    assert base != query_id("lrvs", "1", "b" * 64)


def test_query_id_no_delimiter_collision():
    # Canonical JSON keeps field boundaries: shifting characters between fields differs.
    assert query_id("lrvs", "1a", "b" * 64) != query_id("lrvs", "1", "ab" + "b" * 62)


def test_sha256_bytes_deterministic():
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")
    assert len(sha256_bytes(b"hello")) == 64


def test_sha256_file_deterministic(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"same content")
    f2.write_bytes(b"same content")
    assert sha256_file(f1) == sha256_file(f2) == sha256_bytes(b"same content")

    f3 = tmp_path / "c.bin"
    f3.write_bytes(b"different content")
    assert sha256_file(f1) != sha256_file(f3)


def test_canonical_json_key_order_irrelevant():
    a = canonical_json({"b": 1, "a": 2})
    b = canonical_json({"a": 2, "b": 1})
    assert a == b
    assert a == '{"a":2,"b":1}'


def test_canonical_json_is_compact_and_utf8():
    out = canonical_json({"name": "제품"})
    assert " " not in out
    assert "제품" in out  # ensure_ascii=False


def test_crop_hash_deterministic():
    spec_a = CropSpec(kind="full")
    spec_b = CropSpec(kind="full")
    assert crop_hash(spec_a) == crop_hash(spec_b)


def test_crop_hash_differs_by_content():
    full = CropSpec(kind="full")
    boxed = CropSpec(kind="box", box=BBox(x1=0, y1=0, x2=1, y2=1))
    assert crop_hash(full) != crop_hash(boxed)


def test_index_id_deterministic_regardless_of_params_key_order():
    id_a = index_id("model-x", "gallery-sha", {"nlist": 100, "metric": "ip"})
    id_b = index_id("model-x", "gallery-sha", {"metric": "ip", "nlist": 100})
    assert id_a == id_b


def test_index_id_differs_by_content():
    id_a = index_id("model-x", "gallery-sha", {"nlist": 100})
    id_b = index_id("model-x", "gallery-sha", {"nlist": 200})
    assert id_a != id_b


def test_gallery_sha_deterministic_regardless_of_pair_order():
    pairs_a = [("p1", "a" * 64), ("p2", "b" * 64)]
    pairs_b = [("p2", "b" * 64), ("p1", "a" * 64)]
    assert gallery_sha(pairs_a) == gallery_sha(pairs_b)


def test_gallery_sha_differs_by_membership():
    full = [("p1", "a" * 64), ("p2", "b" * 64)]
    subset = [("p1", "a" * 64)]
    assert gallery_sha(full) != gallery_sha(subset)


def test_index_id_differs_by_gallery_membership_same_manifest():
    """Regression: a val build with --limit-products, a full val build, and a test
    build from the *same manifest* must not collide on index_id just because
    index_id used to be keyed on the manifest sha rather than on which products
    were actually indexed.
    """
    model_id = "model-x"
    params = {"type": "flat"}

    full_gallery = [("p1", "a" * 64), ("p2", "b" * 64), ("p3", "c" * 64)]
    limited_gallery = [("p1", "a" * 64)]  # same manifest, --limit-products 1

    full_id = index_id(model_id, gallery_sha(full_gallery), params)
    limited_id = index_id(model_id, gallery_sha(limited_gallery), params)
    assert full_id != limited_id

    # The same subset, computed again (e.g. a second --limit-products 1 build),
    # must reproduce the same id.
    limited_id_again = index_id(model_id, gallery_sha(limited_gallery), params)
    assert limited_id == limited_id_again


def test_bundle_id_deterministic():
    kwargs = dict(
        embed_model_id="model-x",
        crop_policy="full",
        index_id="idx-1",
        reranker_version="r1",
        calibrator_version="c1",
    )
    assert bundle_id(**kwargs) == bundle_id(**kwargs)


def test_bundle_id_differs_by_content():
    base = dict(embed_model_id="model-x", crop_policy="full", index_id="idx-1")
    assert bundle_id(**base, reranker_version="r1") != bundle_id(**base, reranker_version="r2")
