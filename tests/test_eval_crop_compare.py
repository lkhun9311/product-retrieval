"""Arm comparison, verdict branches and per-category recall (contract: crop-before-search-deepfurniture)."""

import numpy as np
import pytest

from product_retrieval.eval import crop_compare as cc
from product_retrieval.eval.multi import IKEA_KS

KS = list(IKEA_KS)


def arms(n, whole=0.0, detector=0.0, gt=0.0):
    """Per-scene arrays with one constant value per arm, for both metrics."""

    def block(v):
        return np.full((n, len(KS)), v, dtype=np.float64)

    table = {"whole": block(whole), "detector": block(detector), "gt": block(gt)}
    return table, {k: v.copy() for k, v in table.items()}


def run(n_scenes, n_clusters, whole, detector, gt=0.0, b=500):
    hit, rec = arms(n_scenes, whole, detector, gt)
    clusters = np.arange(n_scenes) % n_clusters
    return cc.compare_arms(hit, rec, clusters, KS, b=b)


# --------------------------------------------------------------------------------- verdict branches


def test_verdict_raised_when_every_scene_gains():
    out = run(40, 40, whole=0.0, detector=1.0)
    assert out["verdict"] == cc.VERDICT_RAISED
    assert out["verdict"] == "the crop pipeline raised Recall@10 on these DeepFurniture scenes"
    assert out["primary"]["mean_delta"] == 1.0 and out["primary"]["ci95"] == [1.0, 1.0]
    assert out["verdict_note"] is None


def test_verdict_lowered_when_every_scene_loses():
    out = run(40, 40, whole=1.0, detector=0.0)
    assert out["verdict"] == "the crop pipeline lowered Recall@10 on these DeepFurniture scenes"
    assert out["primary"]["ci95"] == [-1.0, -1.0]


def test_verdict_includes_zero_when_effects_cancel():
    hit, rec = arms(40)
    rec["detector"][::2] = 0.5  # half the scenes +0.5, half 0.0 in whole; flip the sign below
    rec["whole"][1::2] = 0.5
    out = cc.compare_arms(hit, rec, np.arange(40), KS, b=500)
    lo, hi = out["primary"]["ci95"]
    assert lo < 0 < hi
    assert out["verdict"] == "the interval includes zero"
    assert out["verdict_note"] == "not evidence of equivalence"


def test_no_change_at_all_includes_zero():
    out = run(40, 40, whole=0.3, detector=0.3)
    assert out["verdict"] == cc.VERDICT_ZERO
    assert out["primary"]["scenes_delta_eq_0"] == 40


def test_minimum_cluster_rule_boundary():
    # 40 scenes in 19 clusters: below the minimum, even though every delta is +1
    few = run(40, 19, whole=0.0, detector=1.0)
    assert few["n_clusters"] == 19 and few["inferential"] is False
    assert few["verdict"] == "insufficient independent clusters"
    assert few["primary"]["mean_delta"] == 1.0  # the observed mean is still reported
    assert few["primary"]["ci95"] is None and few["primary"]["share_resamples_delta_le_0"] is None
    for comp in few["comparisons"].values():
        for metric in ("hit", "recall"):
            assert all(e["ci95"] is None for e in comp[metric].values())
    # exactly 20 clusters is enough
    enough = run(40, 20, whole=0.0, detector=1.0)
    assert enough["n_clusters"] == 20 and enough["inferential"] is True
    assert enough["verdict"] == cc.VERDICT_RAISED


def test_verdict_function_branches_and_tolerance():
    assert cc.verdict(0.01, 0.2, 25) == cc.VERDICT_RAISED
    assert cc.verdict(-0.2, -0.01, 25) == cc.VERDICT_LOWERED
    assert cc.verdict(-0.1, 0.1, 25) == cc.VERDICT_ZERO
    assert cc.verdict(1e-13, 0.1, 25) == cc.VERDICT_ZERO  # float noise is not an effect
    assert cc.verdict(0.5, 0.9, 19) == cc.VERDICT_INSUFFICIENT
    assert cc.verdict(None, None, 5) == cc.VERDICT_INSUFFICIENT
    with pytest.raises(ValueError, match="required"):
        cc.verdict(None, None, 25)
    with pytest.raises(ValueError, match="finite"):
        cc.verdict(float("nan"), 0.1, 25)


# ----------------------------------------------------------------------------- independent numbers


def test_observed_means_and_signs_by_hand():
    hit, rec = arms(4)
    rec["whole"][:, 2] = [0.0, 0.5, 1.0, 0.5]  # K=10 column
    rec["detector"][:, 2] = [1.0, 0.5, 0.5, 0.5]
    rec["gt"][:, 2] = [1.0, 1.0, 1.0, 1.0]
    out = cc.compare_arms(hit, rec, np.arange(4), KS, b=50, min_clusters=1)
    det = out["comparisons"]["detector_minus_whole"]["recall"]["10"]
    assert det["mean_delta"] == pytest.approx((1.0 + 0.0 - 0.5 + 0.0) / 4)
    assert (det["scenes_delta_gt_0"], det["scenes_delta_eq_0"], det["scenes_delta_lt_0"]) == (1, 2, 1)
    assert det["whole"] == pytest.approx(0.5) and det["detector"] == pytest.approx(0.625)
    gw = out["comparisons"]["gt_minus_whole"]["recall"]["10"]
    assert gw["mean_delta"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.5) / 4)
    gd = out["comparisons"]["gt_minus_detector"]["recall"]["10"]
    assert gd["mean_delta"] == pytest.approx((0.0 + 0.5 + 0.5 + 0.5) / 4)
    # Hit columns were all equal: no change
    assert out["comparisons"]["detector_minus_whole"]["hit"]["10"]["mean_delta"] == 0.0


def test_bootstrap_matches_an_independent_loop():
    rng = np.random.default_rng(5)
    clusters = np.array([0, 0, 1, 1, 1, *range(2, 23), 3])  # 23 clusters of unequal size
    n = len(clusters)
    hit, rec = arms(n)
    rec["whole"][:, 2] = rng.integers(0, 4, n) / 3
    rec["detector"][:, 2] = rng.integers(0, 4, n) / 3
    delta = rec["detector"][:, 2] - rec["whole"][:, 2]

    b = 300
    labels = sorted(set(clusters))
    draw = np.random.default_rng(0).integers(0, len(labels), size=(b, len(labels)))
    means = []
    for row in draw:
        members = [i for c in row for i in range(n) if clusters[i] == labels[c]]
        means.append(sum(delta[i] for i in members) / len(members))
    lo, hi = np.quantile(means, [0.025, 0.975])

    out = cc.compare_arms(hit, rec, clusters, KS, b=b)
    assert out["n_clusters"] == len(labels) == 23
    assert out["primary"]["ci95"] == pytest.approx([lo, hi], abs=1e-12)
    assert out["primary"]["mean_delta"] == pytest.approx(delta.mean())
    share = float(np.mean(np.array(means) <= 1e-12))
    assert out["primary"]["share_resamples_delta_le_0"] == pytest.approx(share)


def test_same_resampling_for_every_column():
    # identical per-scene deltas in two comparisons must give identical intervals (one shared draw)
    hit, rec = arms(30)
    rec["detector"][:, 2] = np.linspace(0, 1, 30)
    rec["gt"][:, 2] = np.linspace(0, 1, 30)
    out = cc.compare_arms(hit, rec, np.arange(30), KS, b=200)
    a = out["comparisons"]["detector_minus_whole"]["recall"]["10"]["ci95"]
    b = out["comparisons"]["gt_minus_whole"]["recall"]["10"]["ci95"]
    assert a == b


@pytest.mark.parametrize("bad", ["nan", "shape", "missing_arm"])
def test_broken_inputs_are_errors(bad):
    hit, rec = arms(30)
    clusters = np.arange(30)
    if bad == "nan":
        rec["detector"][3, 2] = np.nan
    elif bad == "shape":
        rec["gt"] = rec["gt"][:, :3]
    else:
        del hit["gt"]
    with pytest.raises(ValueError):
        cc.compare_arms(hit, rec, clusters, KS, b=20)


# ----------------------------------------------------------------------------------- per category


def test_pooled_category_recall_by_hand_with_a_two_category_identity():
    ranked = [["A", "B", "C"], ["D", "E", "F"]]
    triples = [
        (0, "A", "Sofa"),  # rank 1: hit
        (0, "C", "Sofa"),  # rank 3, k=2: miss
        (0, "B", "Table"),  # rank 2: hit
        (1, "D", "Table"),  # rank 1: hit
        (1, "D", "Sofa"),  # the same identity in a second category counts again: hit
        (1, "F", "Sofa"),  # rank 3: miss
    ]
    out = cc.pooled_category_recall(ranked, triples, k=2)
    assert out == {
        "Sofa": {"denominator": 4, "hits": 2, "recall": 0.5},
        "Table": {"denominator": 2, "hits": 2, "recall": 1.0},
    }
    # K=10 pools to a different number: nothing else changes
    assert cc.pooled_category_recall(ranked, triples, k=10)["Sofa"]["recall"] == 1.0


def test_pooled_category_recall_rejects_broken_input():
    ranked = [["A"]]
    assert cc.pooled_category_recall(ranked, []) == {}
    with pytest.raises(ValueError, match="distinct"):
        cc.pooled_category_recall(ranked, [(0, "A", "x"), (0, "A", "x")])
    with pytest.raises(ValueError, match="scene index"):
        cc.pooled_category_recall(ranked, [(1, "A", "x")])
    with pytest.raises(ValueError, match="k must"):
        cc.pooled_category_recall(ranked, [(0, "A", "x")], k=0)
