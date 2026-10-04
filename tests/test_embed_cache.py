import numpy as np
import pytest

from product_retrieval.embed.cache import EmbeddingCache

CROP_HASH = "crop-abc"
MODEL_ID = "google/siglip2-base-patch16-224"


def _vec(seed: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def test_get_many_reports_everything_missing_on_empty_cache(tmp_path):
    cache = EmbeddingCache(tmp_path)
    vectors, missing = cache.get_many(["a", "b"], CROP_HASH, MODEL_ID)
    assert missing == ["a", "b"]
    assert vectors.shape[0] == 0


def test_put_many_then_get_many_round_trips(tmp_path):
    cache = EmbeddingCache(tmp_path)
    shas = ["a", "b", "c"]
    vectors = np.stack([_vec(i) for i in range(3)])

    cache.put_many(shas, vectors, CROP_HASH, MODEL_ID)
    found, missing = cache.get_many(shas, CROP_HASH, MODEL_ID)

    assert missing == []
    np.testing.assert_array_equal(found, vectors)


def test_get_many_partial_hit_preserves_request_order(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put_many(["a", "c"], np.stack([_vec(0), _vec(2)]), CROP_HASH, MODEL_ID)

    found, missing = cache.get_many(["a", "b", "c", "d"], CROP_HASH, MODEL_ID)

    assert missing == ["b", "d"]
    present_shas = [s for s in ["a", "b", "c", "d"] if s not in missing]
    assert present_shas == ["a", "c"]
    np.testing.assert_array_equal(found[0], _vec(0))
    np.testing.assert_array_equal(found[1], _vec(2))


def test_put_many_is_resumable_first_write_wins(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put_many(["a"], np.stack([_vec(0)]), CROP_HASH, MODEL_ID)
    # Simulate resuming: put_many called again with "a" (already cached, should be
    # left untouched) and a new sha "b".
    cache.put_many(["a", "b"], np.stack([_vec(99), _vec(1)]), CROP_HASH, MODEL_ID)

    found, missing = cache.get_many(["a", "b"], CROP_HASH, MODEL_ID)
    assert missing == []
    np.testing.assert_array_equal(found[0], _vec(0))  # original "a" vector, not _vec(99)
    np.testing.assert_array_equal(found[1], _vec(1))


def test_cache_separates_by_crop_hash_and_model_id(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put_many(["a"], np.stack([_vec(0)]), "crop-1", MODEL_ID)

    _, missing_other_crop = cache.get_many(["a"], "crop-2", MODEL_ID)
    _, missing_other_model = cache.get_many(["a"], "crop-1", "other-model")

    assert missing_other_crop == ["a"]
    assert missing_other_model == ["a"]


def test_put_many_rejects_mismatched_lengths(tmp_path):
    cache = EmbeddingCache(tmp_path)
    with pytest.raises(ValueError):
        cache.put_many(["a", "b"], np.stack([_vec(0)]), CROP_HASH, MODEL_ID)


def test_put_many_no_leftover_temp_files(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put_many(["a"], np.stack([_vec(0)]), CROP_HASH, MODEL_ID)

    out_dir = cache._dir(CROP_HASH, MODEL_ID)
    leftovers = [p for p in out_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_put_many_realigns_after_crash_left_extra_vector_rows(tmp_path):
    """Simulate a crash between the vectors.npy and ids.jsonl renames: vectors.npy has
    extra rows that ids.jsonl doesn't list. New puts must stay aligned with their ids.
    """
    cache = EmbeddingCache(tmp_path)
    cache.put_many(["a", "b"], np.stack([_vec(0), _vec(1)]), CROP_HASH, MODEL_ID)
    vectors_path = cache._vectors_path(CROP_HASH, MODEL_ID)
    np.save(vectors_path, np.concatenate([np.load(vectors_path), np.stack([_vec(98)])]))

    cache.put_many(["c"], np.stack([_vec(2)]), CROP_HASH, MODEL_ID)
    found, missing = cache.get_many(["a", "b", "c"], CROP_HASH, MODEL_ID)

    assert missing == []
    np.testing.assert_array_equal(found, np.stack([_vec(0), _vec(1), _vec(2)]))
