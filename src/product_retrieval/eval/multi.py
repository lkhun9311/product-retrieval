"""Multi-answer retrieval metrics and a room-level bootstrap.

Contract: docs/contracts/ikea-real-photo-check.md.

A query (a room photo) has a *set* of correct products. Per room, at each K:

- Hit@K    = 1 if at least one correct product is in the top K, else 0.
- Recall@K = (number of correct products in the top K) / (number of correct products).

Both are computed per room first, then averaged over rooms (room-macro). The
bootstrap resamples whole rooms with replacement (B=1000, seed 0, 95% percentile
interval by default), using one (B, n_rooms) index matrix shared by every K and both
metrics, in the same vectorised style as ``eval.bootstrap``.

Pure functions; no I/O. Inputs that would make a metric meaningless (a room with no
answers, duplicated ids in a ranking, no rooms, non-positive K) raise ``ValueError``
instead of producing a number.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

IKEA_KS: tuple[int, ...] = (1, 5, 10, 20, 100)


def _validate_ks(ks: Sequence[int]) -> tuple[int, ...]:
    ks_t = tuple(ks)
    if not ks_t:
        raise ValueError("ks must not be empty")
    for k in ks_t:
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
            raise ValueError(f"every K must be a positive integer, got {k!r}")
    if len(set(ks_t)) != len(ks_t):
        raise ValueError(f"ks must not contain duplicates, got {ks_t}")
    return tuple(int(k) for k in ks_t)


def _validate_rooms(ranked: Sequence[Sequence[str]], answers: Sequence[Collection[str]]) -> None:
    if len(ranked) == 0:
        raise ValueError("no rooms: ranked is empty")
    if len(ranked) != len(answers):
        raise ValueError(f"ranked and answers differ in length: {len(ranked)} != {len(answers)}")
    for i, (r, a) in enumerate(zip(ranked, answers, strict=True)):
        if len(a) == 0:
            raise ValueError(f"room {i}: no correct products; drop such rooms before scoring")
        if len(set(r)) != len(r):
            raise ValueError(f"room {i}: ranked list contains duplicate product ids")


def _correct_in_top_k(
    ranked: Sequence[Sequence[str]], answers: Sequence[Collection[str]], ks: tuple[int, ...]
) -> np.ndarray:
    """Return ``(n_rooms, n_ks)`` counts of distinct correct products inside each top K.

    A ranking shorter than K simply contributes its whole list to that K.
    """
    out = np.zeros((len(ranked), len(ks)), dtype=np.float64)
    for i, (r, a) in enumerate(zip(ranked, answers, strict=True)):
        correct = set(a)
        for j, k in enumerate(ks):
            out[i, j] = sum(1 for pid in r[:k] if pid in correct)
    return out


def hit_at_k(
    ranked: Sequence[Sequence[str]], answers: Sequence[Collection[str]], ks: Sequence[int] = IKEA_KS
) -> np.ndarray:
    """Per-room Hit@K, shape ``(n_rooms, n_ks)`` with columns in the order of ``ks``."""
    ks_t = _validate_ks(ks)
    _validate_rooms(ranked, answers)
    return (_correct_in_top_k(ranked, answers, ks_t) >= 1).astype(np.float64)


def recall_at_k(
    ranked: Sequence[Sequence[str]], answers: Sequence[Collection[str]], ks: Sequence[int] = IKEA_KS
) -> np.ndarray:
    """Per-room Recall@K, shape ``(n_rooms, n_ks)``: correct in top K / number of correct."""
    ks_t = _validate_ks(ks)
    _validate_rooms(ranked, answers)
    n_correct = np.array([len(set(a)) for a in answers], dtype=np.float64)
    return _correct_in_top_k(ranked, answers, ks_t) / n_correct[:, None]


def mean_over_rooms(per_room: np.ndarray) -> np.ndarray:
    """Average a ``(n_rooms, n_ks)`` per-room matrix over rooms."""
    if per_room.ndim != 2 or per_room.shape[0] == 0:
        raise ValueError(f"per_room must be a non-empty (n_rooms, n_ks) array, got shape {per_room.shape}")
    if not np.isfinite(per_room).all():
        raise ValueError("per_room contains NaN or inf")
    return per_room.mean(axis=0)


def bootstrap_rooms(
    ranked: Sequence[Sequence[str]],
    answers: Sequence[Collection[str]],
    ks: Sequence[int] = IKEA_KS,
    b: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Room-level bootstrap percentile intervals for mean Hit@K and mean Recall@K.

    Rooms are resampled with replacement, ``b`` replicates, one index matrix
    ``rng.integers(0, n_rooms, (b, n_rooms))`` shared by every K and both metrics.
    Returns point estimates (full data) and ``(lo, hi)`` per K, plus ``b``, ``seed``,
    ``level`` and ``n_rooms``.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    ks_t = _validate_ks(ks)
    hit = hit_at_k(ranked, answers, ks_t)
    rec = recall_at_k(ranked, answers, ks_t)
    n_rooms = hit.shape[0]

    idx = np.random.default_rng(seed).integers(0, n_rooms, size=(b, n_rooms))
    q = [(1.0 - level) / 2.0 * 100.0, (1.0 + level) / 2.0 * 100.0]

    out: dict = {"b": b, "seed": seed, "level": level, "n_rooms": n_rooms}
    for name, per_room in (("hit", hit), ("recall", rec)):
        boot = per_room[idx].mean(axis=1)  # (b, n_ks)
        lo, hi = np.percentile(boot, q, axis=0)
        point = mean_over_rooms(per_room)
        out[name] = {
            "point": {k: float(point[i]) for i, k in enumerate(ks_t)},
            "ci": {k: (float(lo[i]), float(hi[i])) for i, k in enumerate(ks_t)},
        }
    return out
