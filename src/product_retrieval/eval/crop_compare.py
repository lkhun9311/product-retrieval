"""Arm comparison and per-category recall for the DeepFurniture crop run.

Contract: docs/contracts/crop-before-search-deepfurniture.md (Metrics and comparison).

Pure functions; no I/O. The bootstrap, the clustering and the interval-to-sentence rule are those of
``eval.merge``; only the sentences differ (they name DeepFurniture scenes) and an inferential verdict
needs at least ``MIN_CLUSTERS`` clusters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from product_retrieval.eval import merge

MIN_CLUSTERS = 20
PRIMARY_K = 10

VERDICT_RAISED = "the crop pipeline raised Recall@10 on these DeepFurniture scenes"
VERDICT_LOWERED = "the crop pipeline lowered Recall@10 on these DeepFurniture scenes"
VERDICT_ZERO = "the interval includes zero"
VERDICT_INSUFFICIENT = "insufficient independent clusters"

_SENTENCE = {
    merge.VERDICT_RAISED: VERDICT_RAISED,
    merge.VERDICT_LOWERED: VERDICT_LOWERED,
    merge.VERDICT_ZERO: VERDICT_ZERO,
}

# (comparison name, arm A, arm B): the delta is A - B.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("detector_minus_whole", "detector", "whole"),
    ("gt_minus_whole", "gt", "whole"),
    ("gt_minus_detector", "gt", "detector"),
)


def verdict(lo: float | None, hi: float | None, n_clusters: int, min_clusters: int = MIN_CLUSTERS) -> str:
    """The fixed sentence for the primary interval; fewer than ``min_clusters`` clusters is no verdict."""
    if n_clusters < min_clusters:
        return VERDICT_INSUFFICIENT
    if lo is None or hi is None:
        raise ValueError("interval bounds are required when there are enough clusters")
    return _SENTENCE[merge.verdict(lo, hi)]


def compare_arms(
    hit: Mapping[str, np.ndarray],
    recall: Mapping[str, np.ndarray],
    clusters: np.ndarray,
    ks: Sequence[int],
    b: int = merge.BOOTSTRAP_B,
    seed: int = merge.BOOTSTRAP_SEED,
    level: float = merge.BOOTSTRAP_LEVEL,
    min_clusters: int = MIN_CLUSTERS,
) -> dict[str, Any]:
    """Paired cluster bootstrap of every comparison in ``COMPARISONS`` at every K, Hit and Recall.

    ``hit``/``recall`` map arm name -> ``(n_scenes, n_ks)`` per-scene arrays. One resampling draw is
    shared by every column (``merge.paired_cluster_bootstrap``). With fewer than ``min_clusters``
    clusters the intervals and resample shares are withheld (``None``), only observed means remain.
    """
    ks = list(ks)
    n_k = len(ks)
    arms = {a for _, x, y in COMPARISONS for a in (x, y)}
    for name, table in (("hit", hit), ("recall", recall)):
        for arm in arms:
            if arm not in table or np.asarray(table[arm]).shape != (len(clusters), n_k):
                raise ValueError(f"{name}[{arm!r}] must be a ({len(clusters)}, {n_k}) array")
    cols = [
        (np.asarray(table[x], dtype=np.float64) - np.asarray(table[y], dtype=np.float64))
        for _, x, y in COMPARISONS
        for table in (hit, recall)
    ]
    boot = merge.paired_cluster_bootstrap(np.concatenate(cols, axis=1), clusters, b, seed, level)
    inferential = boot.n_clusters >= min_clusters

    out: dict[str, Any] = {
        "n_clusters": boot.n_clusters,
        "min_clusters": min_clusters,
        "inferential": inferential,
        "b": boot.b,
        "seed": boot.seed,
        "level": boot.level,
        "comparisons": {},
    }
    for ci, (name, x, y) in enumerate(COMPARISONS):
        entry: dict[str, Any] = {"a": x, "b": y, "delta": f"{x} - {y}"}
        for mi, (metric, table) in enumerate((("hit", hit), ("recall", recall))):
            block: dict[str, Any] = {}
            for j, k in enumerate(ks):
                col = (ci * 2 + mi) * n_k + j
                d = cols[ci * 2 + mi][:, j]
                block[str(k)] = {
                    x: float(np.asarray(table[x])[:, j].mean()),
                    y: float(np.asarray(table[y])[:, j].mean()),
                    "mean_delta": float(boot.observed[col]),
                    "ci95": [float(boot.lo[col]), float(boot.hi[col])] if inferential else None,
                    "share_resamples_delta_le_0": float(boot.share_le_zero[col]) if inferential else None,
                    "scenes_delta_gt_0": int((d > 0).sum()),
                    "scenes_delta_eq_0": int((d == 0).sum()),
                    "scenes_delta_lt_0": int((d < 0).sum()),
                }
            entry[metric] = block
        out["comparisons"][name] = entry

    primary = out["comparisons"]["detector_minus_whole"]["recall"][str(PRIMARY_K)]
    lo, hi = primary["ci95"] if primary["ci95"] is not None else (None, None)
    out["primary"] = {
        "comparison": "detector_minus_whole",
        "metric": f"recall@{PRIMARY_K}",
        **{k: primary[k] for k in ("mean_delta", "ci95", "share_resamples_delta_le_0")},
        "scenes_delta_gt_0": primary["scenes_delta_gt_0"],
        "scenes_delta_eq_0": primary["scenes_delta_eq_0"],
        "scenes_delta_lt_0": primary["scenes_delta_lt_0"],
    }
    out["verdict"] = verdict(lo, hi, boot.n_clusters, min_clusters)
    out["verdict_note"] = "not evidence of equivalence" if out["verdict"] == VERDICT_ZERO else None
    return out


def pooled_category_recall(
    ranked: Sequence[Sequence[str]],
    triples: Sequence[tuple[int, str, str]],
    k: int = PRIMARY_K,
) -> dict[str, dict[str, float | int]]:
    """Pooled Recall@K per category over distinct ``(scene index, identity, category)`` triples.

    A triple is a hit when its identity is among the first ``k`` products of that scene's ranking.
    An identity with instances of two categories in one scene appears once per category. Returns
    ``category -> {"denominator", "hits", "recall"}``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(set(triples)) != len(triples):
        raise ValueError("triples must be distinct")
    hits: dict[str, int] = {}
    total: dict[str, int] = {}
    for scene, identity, category in triples:
        if not 0 <= scene < len(ranked):
            raise ValueError(f"triple refers to scene index {scene}, only {len(ranked)} ranked lists")
        total[category] = total.get(category, 0) + 1
        hits[category] = hits.get(category, 0) + (identity in set(ranked[scene][:k]))
    return {
        c: {"denominator": total[c], "hits": hits[c], "recall": hits[c] / total[c]} for c in sorted(total)
    }
