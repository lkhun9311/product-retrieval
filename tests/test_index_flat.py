import numpy as np
import pytest

from product_retrieval.index.flat import GalleryIndex, build_index

SHA = lambda n: f"sha-{n}"  # noqa: E731


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_build_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_index(np.zeros((2, 4), dtype=np.float32), ["p1"], ["s1", "s2"], {"type": "flat"})


def test_build_index_rejects_non_flat_type():
    with pytest.raises(NotImplementedError):
        build_index(np.zeros((1, 4), dtype=np.float32), ["p1"], ["s1"], {"type": "ivf"})


def test_aggregation_uses_max_over_product_gallery_images_no_duplicates():
    # p1 has 3 gallery images, one of which is a near-perfect match for the query.
    # p2 has 1 gallery image, a weaker match. The result must rank p1 once (using
    # its best image), then p2 -- never p1 three times.
    query = _unit([1.0, 0.0])
    vectors = np.stack(
        [
            _unit([0.1, 1.0]),  # p1 image A: poor match
            _unit([0.99, 0.1]),  # p1 image B: best match for p1
            _unit([0.2, 0.9]),  # p1 image C: poor match
            _unit([0.5, 0.5]),  # p2 image A: moderate match
        ]
    )
    product_ids = ["p1", "p1", "p1", "p2"]
    image_shas = [SHA(1), SHA(2), SHA(3), SHA(4)]

    gallery = build_index(vectors, product_ids, image_shas, {"type": "flat"})
    results = gallery.search(np.stack([query]), k_products=2)

    assert len(results) == 1
    hits = results[0]
    assert [hit.product_id for hit in hits] == ["p1", "p2"]
    assert hits[0].best_image_sha == SHA(2)
    assert hits[0].score > hits[1].score


def test_search_respects_k_products_limit():
    vectors = np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0]), _unit([0.7, 0.7])])
    gallery = build_index(vectors, ["p1", "p2", "p3"], [SHA(1), SHA(2), SHA(3)], {"type": "flat"})

    results = gallery.search(np.stack([_unit([1.0, 0.0])]), k_products=1)

    assert len(results[0]) == 1
    assert results[0][0].product_id == "p1"


def test_search_handles_multiple_queries_independently():
    vectors = np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0])])
    gallery = build_index(vectors, ["p1", "p2"], [SHA(1), SHA(2)], {"type": "flat"})

    results = gallery.search(np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0])]), k_products=2)

    assert results[0][0].product_id == "p1"
    assert results[1][0].product_id == "p2"


def test_search_on_empty_index_returns_empty_lists():
    gallery = build_index(np.zeros((0, 4), dtype=np.float32), [], [], {"type": "flat"})
    results = gallery.search(np.zeros((2, 4), dtype=np.float32), k_products=5)
    assert results == [[], []]


def test_save_and_load_roundtrip(tmp_path):
    vectors = np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0]), _unit([0.5, 0.5])])
    product_ids = ["p1", "p1", "p2"]
    image_shas = [SHA(1), SHA(2), SHA(3)]
    gallery = build_index(vectors, product_ids, image_shas, {"type": "flat"})
    gallery.index_id = "test-index-id"

    out_dir = tmp_path / "idx"
    gallery.save(out_dir)
    loaded = GalleryIndex.load(out_dir)

    assert loaded.index_id == "test-index-id"
    assert loaded.product_ids == product_ids
    assert loaded.image_shas == image_shas
    assert loaded.params == {"type": "flat"}

    query = np.stack([_unit([1.0, 0.0])])
    original_results = gallery.search(query, k_products=2)
    loaded_results = loaded.search(query, k_products=2)
    assert original_results == loaded_results


def test_save_without_index_id_raises(tmp_path):
    gallery = build_index(np.zeros((0, 4), dtype=np.float32), [], [], {"type": "flat"})
    with pytest.raises(ValueError):
        gallery.save(tmp_path / "idx")
