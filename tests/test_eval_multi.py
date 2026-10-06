import numpy as np
import pytest

from product_retrieval.eval.multi import (
    IKEA_KS,
    bootstrap_rooms,
    hit_at_k,
    mean_over_rooms,
    recall_at_k,
)

KS = (1, 2, 3, 5)

# Three rooms, hand-computed.
#   R1 ranked a b c d e, answers {b, d}           (2 correct)
#   R2 ranked x y z,     answers {x}              (1 correct)
#   R3 ranked p q r s,   answers {r, s, t}        (3 correct, t is never ranked)
RANKED = [list("abcde"), list("xyz"), list("pqrs")]
ANSWERS = [{"b", "d"}, {"x"}, {"r", "s", "t"}]

# Hit@K per room (columns K = 1, 2, 3, 5):
#   R1: top1 [a] 0 ; top2 [a,b] 1 ; top3 [a,b,c] 1 ; top5 all -> b,d -> 1       => 0 1 1 1
#   R2: x is rank 1 -> 1 at every K (ranking is shorter than 5: top5 = whole)   => 1 1 1 1
#   R3: top1 [p] 0 ; top2 [p,q] 0 ; top3 [p,q,r] 1 ; top5 = [p,q,r,s] -> r,s    => 0 0 1 1
HAND_HIT = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=float)

# Recall@K per room = correct in top K / number of correct:
#   R1: 0/2, 1/2, 1/2, 2/2         R2: 1/1 everywhere
#   R3: 0/3, 0/3, 1/3, 2/3  (t is not retrievable, so room recall tops out at 2/3)
HAND_RECALL = np.array([[0, 1 / 2, 1 / 2, 1], [1, 1, 1, 1], [0, 0, 1 / 3, 2 / 3]])


def test_hit_at_k_hand_computed():
    np.testing.assert_allclose(hit_at_k(RANKED, ANSWERS, KS), HAND_HIT)


def test_recall_at_k_hand_computed():
    np.testing.assert_allclose(recall_at_k(RANKED, ANSWERS, KS), HAND_RECALL)


def test_mean_over_rooms_hand_computed():
    # Hit mean   K=1: (0+1+0)/3 = 1/3 ; K=2: (1+1+0)/3 = 2/3 ; K=3: 3/3 = 1 ; K=5: 1
    np.testing.assert_allclose(mean_over_rooms(hit_at_k(RANKED, ANSWERS, KS)), [1 / 3, 2 / 3, 1, 1])
    # Recall mean K=1: 1/3 ; K=2: (1/2+1+0)/3 = 1/2 ; K=3: (1/2+1+1/3)/3 = 11/18 ; K=5: (1+1+2/3)/3 = 8/9
    np.testing.assert_allclose(
        mean_over_rooms(recall_at_k(RANKED, ANSWERS, KS)), [1 / 3, 1 / 2, 11 / 18, 8 / 9]
    )


def test_ks_column_order_follows_input_order():
    # K = 5 first then 1: columns swap, values do not change.
    np.testing.assert_allclose(hit_at_k(RANKED, ANSWERS, (5, 1)), HAND_HIT[:, [3, 0]])


def test_room_with_several_answers_counts_each_once():
    # answers {a, b, c}; ranked a b x c: K=2 -> 2 of 3 ; K=4 -> 3 of 3.
    ranked, answers = [list("abxc")], [{"a", "b", "c"}]
    np.testing.assert_allclose(recall_at_k(ranked, answers, (1, 2, 3, 4)), [[1 / 3, 2 / 3, 2 / 3, 1.0]])
    np.testing.assert_allclose(hit_at_k(ranked, answers, (1, 2, 3, 4)), [[1, 1, 1, 1]])


def test_answers_given_as_list_with_repeats_are_a_set():
    # ["b", "b", "d"] has two distinct answers, not three.
    np.testing.assert_allclose(recall_at_k([list("abcd")], [["b", "b", "d"]], (2, 4)), [[0.5, 1.0]])


def test_no_answer_in_top_k_is_zero_for_both_metrics():
    # The only answer is at rank 3; at K=1 and K=2 both metrics must be exactly 0.
    ranked, answers = [list("abcde")], [{"c"}]
    assert hit_at_k(ranked, answers, (1, 2, 3)).tolist() == [[0.0, 0.0, 1.0]]
    assert recall_at_k(ranked, answers, (1, 2, 3)).tolist() == [[0.0, 0.0, 1.0]]


def _broken_hit_anywhere(ranked, answers, ks):
    """A deliberately wrong gate: counts a hit if any answer is anywhere in the ranking."""
    return np.array(
        [[float(bool(set(r) & set(a))) for _ in ks] for r, a in zip(ranked, answers, strict=True)]
    )


def test_broken_gate_is_caught():
    # The broken hit rule disagrees with the hand values where no answer is in the top K,
    # so the hand-computed check above is not satisfiable by it.
    broken = _broken_hit_anywhere(RANKED, ANSWERS, KS)
    assert not np.array_equal(broken, HAND_HIT)
    # ...and the real function does not behave like it: R1 at K=1 has no answer in top 1.
    assert hit_at_k(RANKED, ANSWERS, KS)[0, 0] == 0.0


def _reference_per_room(ranked, answers, ks):
    """Independent implementation: locate each answer's rank position, then compare to K."""
    hits, recalls = [], []
    for r, a in zip(ranked, answers, strict=True):
        positions = [r.index(p) + 1 for p in set(a) if p in r]  # 1-based ranks of retrieved answers
        hits.append([float(any(pos <= k for pos in positions)) for k in ks])
        recalls.append([sum(pos <= k for pos in positions) / len(set(a)) for k in ks])
    return np.array(hits), np.array(recalls)


def test_matches_independent_rank_position_implementation_on_random_data():
    rng = np.random.default_rng(123)
    universe = [f"p{i}" for i in range(40)]
    ranked, answers = [], []
    for _ in range(60):
        ranked.append([universe[i] for i in rng.permutation(40)[: int(rng.integers(3, 40))]])
        answers.append({universe[i] for i in rng.choice(40, size=int(rng.integers(1, 8)), replace=False)})
    ref_hit, ref_rec = _reference_per_room(ranked, answers, IKEA_KS)
    np.testing.assert_allclose(hit_at_k(ranked, answers, IKEA_KS), ref_hit)
    np.testing.assert_allclose(recall_at_k(ranked, answers, IKEA_KS), ref_rec)
    # Properties: both metrics are non-decreasing in K, recall <= hit, all in [0, 1].
    h, r = hit_at_k(ranked, answers, IKEA_KS), recall_at_k(ranked, answers, IKEA_KS)
    assert (np.diff(h, axis=1) >= 0).all() and (np.diff(r, axis=1) >= 0).all()
    assert (r <= h + 1e-12).all() and h.min() >= 0 and h.max() <= 1


@pytest.mark.parametrize(
    "ranked, answers, ks",
    [
        ([], [], KS),  # no rooms
        ([list("ab")], [set()], KS),  # room with no answers (denominator 0)
        ([list("ab")], [{"a"}, {"b"}], KS),  # length mismatch
        ([list("aab")], [{"a"}], KS),  # duplicate id in a ranking
        ([list("ab")], [{"a"}], ()),  # no K
        ([list("ab")], [{"a"}], (0,)),  # K = 0
        ([list("ab")], [{"a"}], (-1,)),  # negative K
        ([list("ab")], [{"a"}], (1, 1)),  # duplicate K
        ([list("ab")], [{"a"}], (1.5,)),  # non-integer K
    ],
)
def test_broken_inputs_raise(ranked, answers, ks):
    with pytest.raises(ValueError):
        hit_at_k(ranked, answers, ks)
    with pytest.raises(ValueError):
        recall_at_k(ranked, answers, ks)
    with pytest.raises(ValueError):
        bootstrap_rooms(ranked, answers, ks, b=10)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((0, 3)), np.array([[0.5, np.nan]]), np.array([[0.5, np.inf]]), np.array([1.0, 0.0])],
)
def test_mean_over_rooms_rejects_empty_nonfinite_and_wrong_shape(bad):
    with pytest.raises(ValueError):
        mean_over_rooms(bad)


def test_all_rooms_held_back_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        bootstrap_rooms([], [], KS)


def _random_rooms(n, seed=7):
    rng = np.random.default_rng(seed)
    universe = [f"p{i}" for i in range(30)]
    ranked = [[universe[i] for i in rng.permutation(30)] for _ in range(n)]
    answers = [
        {universe[i] for i in rng.choice(30, size=int(rng.integers(1, 5)), replace=False)} for _ in range(n)
    ]
    return ranked, answers


def test_bootstrap_is_deterministic_and_seed_sensitive():
    ranked, answers = _random_rooms(50)
    a = bootstrap_rooms(ranked, answers, IKEA_KS, b=200, seed=0)
    b = bootstrap_rooms(ranked, answers, IKEA_KS, b=200, seed=0)
    c = bootstrap_rooms(ranked, answers, IKEA_KS, b=200, seed=1)
    assert a == b
    assert a["hit"]["ci"] != c["hit"]["ci"]
    assert a["b"] == 200 and a["seed"] == 0 and a["n_rooms"] == 50 and a["level"] == 0.95


def test_bootstrap_defaults_are_the_contract_values():
    ranked, answers = _random_rooms(10)
    out = bootstrap_rooms(ranked, answers)
    assert out["b"] == 1000 and out["seed"] == 0 and out["level"] == 0.95
    assert sorted(out["hit"]["point"]) == [1, 5, 10, 20, 100]


def test_bootstrap_interval_sanity():
    ranked, answers = _random_rooms(80)
    out = bootstrap_rooms(ranked, answers, IKEA_KS, b=500, seed=0)
    for name, per_room in (
        ("hit", hit_at_k(ranked, answers, IKEA_KS)),
        ("recall", recall_at_k(ranked, answers, IKEA_KS)),
    ):
        for i, k in enumerate(IKEA_KS):
            lo, hi = out[name]["ci"][k]
            point = out[name]["point"][k]
            assert point == pytest.approx(per_room[:, i].mean())
            assert 0.0 <= lo <= hi <= 1.0
            assert lo <= point <= hi  # holds for means of 80 rooms at this B/seed
            if 0.0 < point < 1.0:
                assert hi - lo > 0.0  # a mixed sample has spread
            else:
                assert lo == hi == point  # K covers the whole 30-product gallery: all 0 or all 1


def test_bootstrap_degenerate_samples_have_zero_width():
    # Every room hit at K=1 -> every resample mean is exactly 1.
    ranked = [list("ab"), list("ba"), list("ab")]
    answers = [{"a"}, {"b"}, {"a"}]
    out = bootstrap_rooms(ranked, answers, (1,), b=100)
    assert out["hit"]["ci"][1] == (1.0, 1.0)
    assert out["recall"]["ci"][1] == (1.0, 1.0)


def test_bootstrap_matches_loop_reference_on_the_same_room_indices():
    # Two rooms, hit at K=1: [0, 1]. Each resample mean is (#room2 draws)/2 in {0, .5, 1}.
    ranked = [list("ba"), list("ab")]
    answers = [{"a"}, {"a"}]
    b, seed = 400, 3
    idx = np.random.default_rng(seed).integers(0, 2, size=(b, 2))
    means = sorted(sum(1.0 for j in row if j == 1) / 2.0 for row in idx)
    lo, hi = np.percentile(means, [2.5, 97.5])
    out = bootstrap_rooms(ranked, answers, (1,), b=b, seed=seed)
    assert out["hit"]["ci"][1] == (pytest.approx(lo), pytest.approx(hi))
    assert out["hit"]["point"][1] == 0.5


def test_bootstrap_resamples_rooms_not_products():
    # One room with many answers: resampling must not change per-room values, so with a
    # single room the interval collapses to the point estimate.
    out = bootstrap_rooms([list("abcd")], [{"a", "d"}], (2,), b=50)
    assert out["recall"]["ci"][2] == (0.5, 0.5)
    assert out["recall"]["point"][2] == 0.5


def test_bootstrap_uses_the_95_percent_percentile_interval_via_loop_reference():
    ranked, answers = _random_rooms(30, seed=11)
    b, seed = 300, 5
    per_room = recall_at_k(ranked, answers, (5,))[:, 0]
    idx = np.random.default_rng(seed).integers(0, 30, size=(b, 30))
    means = [float(np.mean([per_room[j] for j in row])) for row in idx]
    expected = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    out = bootstrap_rooms(ranked, answers, (5,), b=b, seed=seed)
    assert out["recall"]["ci"][5] == (pytest.approx(expected[0]), pytest.approx(expected[1]))
    # A 90% interval would be strictly narrower on this sample, so the test can tell them apart.
    assert (float(np.percentile(means, 5)), float(np.percentile(means, 95))) != pytest.approx(expected)
