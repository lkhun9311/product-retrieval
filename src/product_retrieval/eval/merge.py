"""Round-robin merge of per-crop rankings and the paired cluster bootstrap.

Contract: docs/contracts/crop-before-search.md. Pure functions; no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_LEVEL = 0.95
CLUSTER_COSINE = 0.95
# Bounds closer to zero than this are float noise from summing fractions, not an effect.
VERDICT_ZERO_TOL = 1e-12

VERDICT_RAISED = "the crop pipeline raised Recall@10 on these rooms"
VERDICT_LOWERED = "the crop pipeline lowered Recall@10 on these rooms"
VERDICT_ZERO = "the interval includes zero"


def rank_products(sims: np.ndarray, product_ids: Sequence[str]) -> list[str]:
    """Rank one query row of cosines against ``product_ids``; ties go to the smaller product id."""
    ids = list(product_ids)
    if len(ids) != len(sims):
        raise ValueError(f"sims and product_ids differ in length: {len(sims)} != {len(ids)}")
    order = sorted(range(len(ids)), key=lambda j: (-float(sims[j]), ids[j]))
    return [ids[j] for j in order]


def merge_round_robin(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Merge per-crop rankings: traversal order (depth, crop order).

    Depth 1 of crop 1 .. crop m, then depth 2 of crop 1 .. crop m, and so on. A product already in
    the merged list is skipped and its turn is consumed (no refill from the same list). Stops at
    ``k`` unique products or when every list is exhausted.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    merged: list[str] = []
    seen: set[str] = set()
    depth_max = max((len(r) for r in rankings), default=0)
    for depth in range(depth_max):
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            pid = ranking[depth]
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(pid)
            if len(merged) == k:
                return merged
    return merged


def cluster_rooms(vectors: np.ndarray, threshold: float = CLUSTER_COSINE) -> np.ndarray:
    """Connected components of rooms linked when cosine >= ``threshold``.

    ``vectors`` are unit-norm whole-photo embeddings. Cosines are computed in float64. Returns
    cluster ids ``0..C-1`` numbered by the first room (lowest index) of each component.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError(f"vectors must be a non-empty (n, dim) array, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("vectors contain NaN or inf")
    n = v.shape[0]
    adj = (v @ v.T) >= threshold
    labels = np.full(n, -1, dtype=np.int64)
    c = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = c
        while stack:
            u = stack.pop()
            for w in np.flatnonzero(adj[u] & (labels == -1)):
                labels[w] = c
                stack.append(int(w))
        c += 1
    return labels


@dataclass(frozen=True)
class ClusterBootstrap:
    observed: np.ndarray  # (n_cols,) mean over rooms
    lo: np.ndarray
    hi: np.ndarray
    share_le_zero: np.ndarray  # share of resamples with mean <= 0 (not a p-value)
    n_clusters: int
    b: int
    seed: int
    level: float


def paired_cluster_bootstrap(
    delta: np.ndarray,
    cluster_ids: np.ndarray,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    level: float = BOOTSTRAP_LEVEL,
) -> ClusterBootstrap:
    """Bootstrap the mean per-room difference by resampling whole clusters.

    ``delta`` is ``(n_rooms, n_cols)`` of Crops - Whole (already paired by room). Each replicate draws
    ``C`` clusters with replacement (``default_rng(seed).integers(0, C, (b, C))``, one draw shared by
    every column) and averages delta over the rooms of the drawn clusters (a drawn cluster counts
    with all its rooms, as many times as drawn). Interval: ``numpy.quantile`` default method.
    """
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2 or d.shape[0] == 0:
        raise ValueError(f"delta must be a non-empty (n_rooms, n_cols) array, got shape {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("delta contains NaN or inf")
    ids = np.asarray(cluster_ids)
    if ids.shape != (d.shape[0],):
        raise ValueError(f"cluster_ids must have one entry per room, got {ids.shape} for {d.shape[0]} rooms")
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    uniq, inv = np.unique(ids, return_inverse=True)
    n_c = len(uniq)
    sums = np.zeros((n_c, d.shape[1]))
    np.add.at(sums, inv, d)
    sizes = np.bincount(inv, minlength=n_c).astype(np.float64)

    idx = np.random.default_rng(seed).integers(0, n_c, size=(b, n_c))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)[:, None]  # (b, n_cols)
    lo, hi = np.quantile(boot, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return ClusterBootstrap(
        observed=d.mean(axis=0),
        lo=lo,
        hi=hi,
        share_le_zero=(boot <= VERDICT_ZERO_TOL).mean(axis=0),
        n_clusters=n_c,
        b=b,
        seed=seed,
        level=level,
    )


def verdict(lo: float, hi: float) -> str:
    """The pre-registered sentence for the primary interval of mean Delta Recall@10."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"interval bounds must be finite, got ({lo}, {hi})")
    if lo > VERDICT_ZERO_TOL:
        return VERDICT_RAISED
    if hi < -VERDICT_ZERO_TOL:
        return VERDICT_LOWERED
    return VERDICT_ZERO
