"""Round-robin merge, cluster construction, paired cluster bootstrap and the verdict gate."""

import numpy as np
import pytest

from product_retrieval.eval import merge
from product_retrieval.eval.merge import (
    VERDICT_LOWERED,
    VERDICT_RAISED,
    VERDICT_ZERO,
    cluster_rooms,
    merge_round_robin,
    paired_cluster_bootstrap,
    rank_products,
    verdict,
)
from product_retrieval.eval.multi import recall_at_k


def _refill_merge(rankings, k):
    """The rule the contract does NOT use: a duplicate does not consume the turn, the same list
    supplies its next unseen product immediately."""
    merged, seen = [], set()
    pos = [0] * len(rankings)
    exhausted = False
    while len(merged) < k and not exhausted:
        exhausted = True
        for ci, r in enumerate(rankings):
            while pos[ci] < len(r) and r[pos[ci]] in seen:
                pos[ci] += 1
            if pos[ci] < len(r):
                exhausted = False
                seen.add(r[pos[ci]])
                merged.append(r[pos[ci]])
                pos[ci] += 1
                if len(merged) == k:
                    return merged
    return merged


# --- rank_products -------------------------------------------------------------------------------


def test_rank_products_ties_go_to_the_smaller_id():
    assert rank_products(np.array([0.5, 0.9, 0.5, 0.9]), ["d", "b", "a", "c"]) == ["b", "c", "a", "d"]
    with pytest.raises(ValueError):
        rank_products(np.array([0.5]), ["a", "b"])


# --- merge ---------------------------------------------------------------------------------------

C1 = ["A", "B", "C", "D"]
C2 = ["A", "C", "B", "D"]


def test_merge_duplicate_consumes_its_turn_hand_traced():
    # K = 3, crops c1 = [A B C D], c2 = [A C B D].
    # depth 1: c1 -> A (new)            merged [A]
    #          c2 -> A (duplicate, its turn is consumed; c2 does NOT advance to C now)
    # depth 2: c1 -> B (new)            merged [A B]
    #          c2 -> C (new)            merged [A B C]  = K, stop
    assert merge_round_robin([C1, C2], 3) == ["A", "B", "C"]
    # K = 2 stops right after c1's B.
    assert merge_round_robin([C1, C2], 2) == ["A", "B"]
    assert merge_round_robin([C1, C2], 1) == ["A"]


def test_consume_rule_differs_from_a_refill_rule_at_k2_recall():
    # Refill rule: c1 -> A ; c2 -> A is a duplicate, so c2 refills with C immediately -> [A C] at K=2.
    assert _refill_merge([C1, C2], 2) == ["A", "C"]
    # Contract rule at K=2: [A B]. Answer {B}: Recall@2 = 1 under the contract, 0 under refill.
    contract = merge_round_robin([C1, C2], 2)
    assert recall_at_k([contract], [{"B"}], [2])[0, 0] == 1.0
    assert recall_at_k([_refill_merge([C1, C2], 2)], [{"B"}], [2])[0, 0] == 0.0
    # Recall@1 cannot tell the rules apart: slot 1 is always crop 1's top product, so the claim
    # "a duplicate changes Recall@1" does not hold for any rule; K = 2 is the first K that can differ.
    assert merge_round_robin([C1, C2], 1) == _refill_merge([C1, C2], 1)


def test_merge_stops_at_exhaustion_with_fewer_than_k():
    # c1 = [A], c2 = [A]: depth 1 gives A, then the duplicate is skipped, lists are exhausted.
    assert merge_round_robin([["A"], ["A"]], 5) == ["A"]
    assert merge_round_robin([], 3) == []
    assert merge_round_robin([[]], 3) == []


def test_merge_handles_lists_of_different_length_and_stops_mid_depth():
    # depth 1: c1 A, c2 B, c3 C ; depth 2: c1 D, c3 E (c2 exhausted)
    assert merge_round_robin([["A", "D"], ["B"], ["C", "E"]], 10) == ["A", "B", "C", "D", "E"]
    assert merge_round_robin([["A", "D"], ["B"], ["C", "E"]], 2) == ["A", "B"]  # stop inside depth 1


def test_merge_ties_between_crops_follow_crop_order():
    # crop order is detector score order: crop 1 first at every depth.
    assert merge_round_robin([["X", "Y"], ["Z", "Y"]], 3) == ["X", "Z", "Y"]


def test_merge_prefix_property_and_uniqueness_on_random_rankings():
    rng = np.random.default_rng(1)
    ids = [f"p{i:02d}" for i in range(30)]
    for _ in range(25):
        rankings = [list(rng.permutation(ids)) for _ in range(int(rng.integers(1, 6)))]
        full = merge_round_robin(rankings, 100)
        assert len(full) == len(set(full)) == 30
        for k in (1, 5, 10, 20):
            assert merge_round_robin(rankings, k) == full[:k]


def test_merge_rejects_non_positive_k():
    with pytest.raises(ValueError):
        merge_round_robin([["A"]], 0)


# --- clusters ------------------------------------------------------------------------------------


def test_cluster_rooms_threshold_is_inclusive_and_links_are_transitive():
    v = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])  # cos(0,1)=0.6, cos(1,2)=0.8, cos(0,2)=0
    assert v[0] @ v[1] == 0.6 and v[1] @ v[2] == 0.8
    assert cluster_rooms(v, 0.6).tolist() == [0, 0, 0]  # 0-1 linked at exactly 0.6, 1-2 at 0.8
    assert cluster_rooms(v, 0.6000001).tolist() == [0, 1, 1]  # 0-1 no longer linked
    assert cluster_rooms(v, 0.9).tolist() == [0, 1, 2]


def test_cluster_rooms_default_is_0_95_and_numbers_by_first_room():
    theta = np.deg2rad([0.0, 90.0, 5.0, 91.0, 180.0])  # cos(5 deg) = 0.9962, cos(1 deg) = 0.99985
    v = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert cluster_rooms(v).tolist() == [0, 1, 0, 1, 2]
    theta = np.deg2rad([0.0, 20.0])  # cos(20 deg) = 0.9397 < 0.95
    assert cluster_rooms(np.stack([np.cos(theta), np.sin(theta)], axis=1)).tolist() == [0, 1]


def test_cluster_rooms_rejects_broken_input():
    with pytest.raises(ValueError):
        cluster_rooms(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        cluster_rooms(np.array([[1.0, np.nan]]))


# --- bootstrap -----------------------------------------------------------------------------------


def test_bootstrap_constant_delta_has_a_degenerate_interval():
    r = paired_cluster_bootstrap(np.full(6, 0.25), np.array([0, 0, 1, 1, 2, 3]), b=200)
    assert r.observed[0] == 0.25 and r.lo[0] == r.hi[0] == pytest.approx(0.25)
    assert r.share_le_zero[0] == 0.0 and r.n_clusters == 4


def test_bootstrap_enumerated_two_cluster_case():
    # cluster a: 1 room with delta +1 ; cluster b: 3 rooms with delta -1.
    # Resample of 2 clusters: (a,a) -> +1 (p .25) ; (a,b),(b,a) -> (1-3)/4 = -0.5 (p .5) ; (b,b) -> -1 (p .25)
    d = np.array([1.0, -1.0, -1.0, -1.0])
    r = paired_cluster_bootstrap(d, np.array([0, 1, 1, 1]), b=10_000)
    assert r.observed[0] == pytest.approx(-0.5)  # mean over the 4 rooms: (1-3)/4
    assert r.lo[0] == -1.0 and r.hi[0] == 1.0  # each extreme has p = .25 >> 2.5%
    assert r.share_le_zero[0] == pytest.approx(0.75, abs=0.02)  # (a,b)+(b,a)+(b,b)


def test_bootstrap_matches_a_loop_implementation_with_the_same_draws():
    d = np.array([[0.5, 0.0], [0.0, -0.25], [0.25, 0.25], [1.0, 0.5], [0.0, 0.0], [-0.5, 0.25]])
    cl = np.array([0, 0, 1, 2, 2, 2])
    b = 400
    r = paired_cluster_bootstrap(d, cl, b=b, seed=7)
    idx = np.random.default_rng(7).integers(0, 3, size=(b, 3))
    members = {c: np.flatnonzero(cl == c) for c in range(3)}
    means = np.array(
        [d[np.concatenate([members[c] for c in row])].mean(axis=0) for row in idx]  # rooms of drawn clusters
    )
    assert r.lo == pytest.approx(np.quantile(means, 0.025, axis=0))
    assert r.hi == pytest.approx(np.quantile(means, 0.975, axis=0))
    assert r.share_le_zero == pytest.approx((means <= 0).mean(axis=0))
    assert r.observed == pytest.approx(d.mean(axis=0))


def test_bootstrap_is_deterministic_and_columns_share_one_draw():
    rng = np.random.default_rng(0)
    d = rng.choice([-0.5, 0.0, 0.25, 0.5], size=(40, 3))
    cl = np.repeat(np.arange(20), 2)
    a = paired_cluster_bootstrap(d, cl, b=500)
    b = paired_cluster_bootstrap(d, cl, b=500)
    assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
    solo = paired_cluster_bootstrap(d[:, 1], cl, b=500)  # same index matrix -> same column result
    assert solo.lo[0] == a.lo[1] and solo.hi[0] == a.hi[1]
    assert not np.array_equal(paired_cluster_bootstrap(d, cl, b=500, seed=1).lo, a.lo)


def test_bootstrap_defaults_are_the_contract_values():
    assert (merge.BOOTSTRAP_B, merge.BOOTSTRAP_SEED, merge.BOOTSTRAP_LEVEL, merge.CLUSTER_COSINE) == (
        10_000,
        0,
        0.95,
        0.95,
    )


@pytest.mark.parametrize(
    "delta,clusters",
    [
        (np.array([]), np.array([])),
        (np.array([np.nan, 0.0]), np.array([0, 1])),
        (np.array([0.5, 0.0]), np.array([0])),
    ],
)
def test_bootstrap_rejects_broken_input(delta, clusters):
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(delta, clusters, b=10)


# --- verdict gate --------------------------------------------------------------------------------


def test_verdict_sentences_are_the_contract_words():
    assert VERDICT_RAISED == "the crop pipeline raised Recall@10 on these rooms"
    assert VERDICT_LOWERED == "the crop pipeline lowered Recall@10 on these rooms"
    assert VERDICT_ZERO == "the interval includes zero"


def test_verdict_bounds_are_strict_at_zero():
    assert verdict(1e-6, 0.5) == VERDICT_RAISED
    assert verdict(-0.5, -1e-6) == VERDICT_LOWERED
    assert verdict(0.0, 0.5) == VERDICT_ZERO  # lower bound == 0 is not "> 0"  (breaks if > becomes >=)
    assert verdict(-0.5, 0.0) == VERDICT_ZERO  # upper bound == 0 is not "< 0"
    assert verdict(-0.1, 0.1) == VERDICT_ZERO
    assert verdict(1e-13, 0.5) == VERDICT_ZERO  # float noise around zero is not an effect
    assert verdict(0.0, 0.0) == VERDICT_ZERO


@pytest.mark.parametrize("lo,hi", [(np.nan, 0.1), (0.1, np.inf)])
def test_verdict_refuses_non_finite_bounds(lo, hi):
    with pytest.raises(ValueError):
        verdict(lo, hi)


def test_gate_on_a_real_bootstrap_edge_95_includes_zero_but_90_would_not():
    # 30 rooms (30 clusters), 3 with delta +0.5, the rest 0. Observed mean 0.05.
    d = np.zeros(30)
    d[:3] = 0.5
    ids = np.arange(30)
    r95 = paired_cluster_bootstrap(d, ids)
    r90 = paired_cluster_bootstrap(d, ids, level=0.90)
    assert r95.lo[0] == 0.0  # P(no positive room drawn) = 0.9**30 ~ 4.2% > 2.5%
    assert verdict(r95.lo[0], r95.hi[0]) == VERDICT_ZERO  # fails if the gate uses >= 0
    assert r90.lo[0] > 0.0  # P(...) ~ 4.2% < 5%: a 90% interval would declare an effect
    assert verdict(r90.lo[0], r90.hi[0]) == VERDICT_RAISED  # what a 90% gate would have said


def test_gate_on_a_real_bootstrap_lowered_and_raised():
    ids = np.arange(40)
    up = np.full(40, 0.1)
    up[::4] = 0.0
    r = paired_cluster_bootstrap(up, ids)
    assert verdict(r.lo[0], r.hi[0]) == VERDICT_RAISED
    r = paired_cluster_bootstrap(-up, ids)
    assert verdict(r.lo[0], r.hi[0]) == VERDICT_LOWERED
