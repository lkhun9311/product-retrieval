"""Uncertainty label selection, contract c4-uncertainty v1 (docs/contracts/c4-uncertainty.md section 2).

``select_uncertain`` is a pure function. It receives only public inputs: the pool rankings (it reads
``query_id`` and ``top_k_product_ids`` of each row and nothing else, so a ``truth_product_id`` in the rows
is ignored), the model probability of every pool pair, and the pairs already labelled (L). It has no
access to the simulator, to a truth field or to a manifest.

Rule (steps 3-5):
- uncertainty ``u = |p - 0.5|`` for every pool pair; order by (u ascending, query_id ascending, position
  ascending);
- walk that order, skip pairs already in L, skip a pair whose query already has ``cap`` (3) judgements in
  L plus the pairs taken in this call, stop after ``n_take`` pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

POOL_DEPTH = 20
PER_QUERY_CAP = 3


class SelectionError(ValueError):
    """Invalid selector input, or fewer selectable pairs than asked for."""


class Selection(NamedTuple):
    query_id: str
    product_id: str
    position: int  # 1-based rank in the exposed list


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function in float64."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def select_uncertain(
    pool: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, Sequence[float]],
    labelled: Mapping[tuple[str, str], Any] | Sequence[tuple[str, str]],
    n_take: int,
    *,
    cap: int = PER_QUERY_CAP,
    depth: int = POOL_DEPTH,
) -> list[Selection]:
    """The next ``n_take`` pairs to label, in selection order.

    ``pool``: rows with ``query_id`` and ``top_k_product_ids`` (the first ``depth`` are the pool pairs, the
    1-based index is the position). ``probabilities``: ``{query_id: p by position}`` with ``depth`` values in
    [0, 1] each. ``labelled``: the pairs (query_id, product_id) already in L (a mapping to answers or any
    collection of pairs; only the pairs are used). Raises ``SelectionError`` on malformed input, on an L pair
    outside the pool or a query over the cap in L, and when fewer than ``n_take`` pairs can be taken.
    """
    if isinstance(n_take, bool) or not isinstance(n_take, int) or n_take < 0:
        raise SelectionError(f"n_take must be an integer >= 0, got {n_take!r}")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise SelectionError(f"cap must be an integer >= 1, got {cap!r}")

    ids_by_query: dict[str, list[str]] = {}
    for row in pool:
        qid = row["query_id"]
        if qid in ids_by_query:
            raise SelectionError(f"duplicate query_id in the pool: {qid!r}")
        ids = list(row["top_k_product_ids"][:depth])
        if len(ids) < depth:
            raise SelectionError(f"query {qid!r} has {len(ids)} candidates, the pool needs {depth}")
        if len(set(ids)) != depth:
            raise SelectionError(f"query {qid!r} lists a product more than once in its top {depth}")
        ids_by_query[qid] = ids
    if not ids_by_query:
        raise SelectionError("the pool is empty")
    if set(probabilities) != set(ids_by_query):
        missing = sorted(set(ids_by_query) - set(probabilities))[:3]
        extra = sorted(set(probabilities) - set(ids_by_query))[:3]
        raise SelectionError(f"probabilities and pool differ in query ids (missing {missing}, extra {extra})")

    qids = sorted(ids_by_query)  # row index follows query_id ascending
    row_of = {qid: i for i, qid in enumerate(qids)}
    p = np.empty((len(qids), depth), dtype=np.float64)
    for qid, i in row_of.items():
        vals = probabilities[qid]
        if len(vals) != depth:
            raise SelectionError(f"query {qid!r}: {len(vals)} probabilities, expected {depth}")
        p[i] = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(p).all():
        raise SelectionError("probabilities contain NaN or infinity")
    if ((p < 0.0) | (p > 1.0)).any():
        raise SelectionError("probabilities must lie in [0, 1]")

    in_l = np.zeros((len(qids), depth), dtype=bool)
    taken_per_query = np.zeros(len(qids), dtype=np.int64)
    for qid, pid in list(labelled):
        i = row_of.get(qid)
        if i is None or pid not in ids_by_query[qid]:
            raise SelectionError(f"labelled pair ({qid!r}, {pid!r}) is not in the pool")
        j = ids_by_query[qid].index(pid)
        if in_l[i, j]:
            raise SelectionError(f"labelled pair ({qid!r}, {pid!r}) appears twice")
        in_l[i, j] = True
        taken_per_query[i] += 1
    if (taken_per_query > cap).any():
        bad = qids[int(np.argmax(taken_per_query > cap))]
        raise SelectionError(f"query {bad!r} already has more than {cap} judgements in L")

    u = np.abs(p - 0.5).ravel()
    q_idx = np.repeat(np.arange(len(qids)), depth)
    pos0 = np.tile(np.arange(depth), len(qids))
    # lexsort sorts by the last key first: u, then query_id (row index), then position
    order = np.lexsort((pos0, q_idx, u))

    out: list[Selection] = []
    blocked = in_l.ravel()
    for flat in order.tolist():
        if len(out) == n_take:
            break
        if blocked[flat]:
            continue
        i = flat // depth
        if taken_per_query[i] >= cap:
            continue
        j = flat % depth
        taken_per_query[i] += 1
        qid = qids[i]
        out.append(Selection(qid, ids_by_query[qid][j], j + 1))
    if len(out) < n_take:
        raise SelectionError(f"only {len(out)} pairs can be selected, {n_take} were requested")
    return out
