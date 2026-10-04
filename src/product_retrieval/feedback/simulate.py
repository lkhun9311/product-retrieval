"""C4 simulated feedback generator, random policy (D20).

Reads the per-query rankings JSONL written by ``pr eval`` and emits
``FeedbackEvent`` JSONL. The ground truth (``truth_product_id``) is used only
inside this module to decide match/not_match; it is never written to the output.

Conventions:
- ``position`` is the **1-based** rank of the candidate in the exposed list.
- ``event_id`` is a deterministic hash of (policy_id, seed, query_id, product_id),
  so regenerating with the same arguments yields the same ids (idempotent).
- ``ts`` is a fixed epoch plus the event index in seconds, so the same arguments
  give a byte-identical file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from product_retrieval.core.ids import canonical_json, sha256_bytes
from product_retrieval.core.schemas import FeedbackEvent

POLICY_ID = "random"
DEFAULT_EXPOSED_K = 20
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class SimulationError(ValueError):
    """Raised for invalid simulator input (budget too large, bad rankings, bad noise)."""


@dataclass(frozen=True)
class _Pair:
    query_id: str
    product_id: str
    position: int  # 1-based
    is_truth: bool


def _hash(*parts: object) -> str:
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))


def event_id(policy_id: str, seed: int, query_id: str, product_id: str) -> str:
    """Deterministic idempotency key for one (policy, seed, query, product) judgment."""
    return _hash(policy_id, seed, query_id, product_id)


def load_rankings(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("query_id", "truth_product_id", "top_k_product_ids"):
                if key not in row:
                    raise SimulationError(f"{path}:{lineno}: missing field {key!r}")
            rows.append(row)
    return rows


def _build_pairs(rankings: list[dict], exposed_k: int) -> list[_Pair]:
    pairs: list[_Pair] = []
    seen_queries: set[str] = set()
    for row in rankings:
        qid = row["query_id"]
        if qid in seen_queries:
            raise SimulationError(f"duplicate query_id in rankings: {qid!r}")
        seen_queries.add(qid)
        seen_products: set[str] = set()
        for rank0, pid in enumerate(row["top_k_product_ids"][:exposed_k]):
            if pid in seen_products:
                raise SimulationError(f"query {qid!r} lists product {pid!r} more than once")
            seen_products.add(pid)
            pairs.append(_Pair(qid, pid, rank0 + 1, pid == row["truth_product_id"]))
    return pairs


def simulate_random(
    rankings: list[dict],
    *,
    budget: int,
    seed: int,
    exposed_k: int = DEFAULT_EXPOSED_K,
    noise: float = 0.0,
) -> list[FeedbackEvent]:
    """Sample ``budget`` (query, candidate) pairs uniformly without replacement and judge them.

    The answer is ``match`` iff the candidate is the query's truth product; with
    probability ``noise`` the answer is flipped. Raises ``SimulationError`` if
    ``budget`` exceeds the number of exposed pairs (no silent truncation).
    """
    if budget < 0:
        raise SimulationError(f"budget must be >= 0, got {budget}")
    if exposed_k < 1:
        raise SimulationError(f"exposed_k must be >= 1, got {exposed_k}")
    if not 0.0 <= noise <= 1.0:
        raise SimulationError(f"noise must be in [0, 1], got {noise}")

    pairs = _build_pairs(rankings, exposed_k)
    if budget > len(pairs):
        raise SimulationError(
            f"budget {budget} exceeds the {len(pairs)} available (query, candidate) pairs "
            f"(exposed_k={exposed_k})"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pairs), size=budget, replace=False)
    flips = rng.random(budget) < noise

    events: list[FeedbackEvent] = []
    for i, (idx, flip) in enumerate(zip(chosen.tolist(), flips.tolist(), strict=True)):
        pair = pairs[idx]
        is_match = pair.is_truth != bool(flip)
        events.append(
            FeedbackEvent(
                event_id=event_id(POLICY_ID, seed, pair.query_id, pair.product_id),
                ts=EPOCH + timedelta(seconds=i),
                session_id=f"sim-{POLICY_ID}-{seed}",
                list_id=_hash("list", POLICY_ID, seed, pair.query_id),
                query_id=pair.query_id,
                product_id=pair.product_id,
                action="match" if is_match else "not_match",
                position=pair.position,
                policy_id=POLICY_ID,
                actor="sim",
            )
        )
    return events


def write_events(events: list[FeedbackEvent], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for ev in events:
            f.write(ev.model_dump_json() + "\n")


def run_simulate(
    rankings_path: str | Path,
    out_path: str | Path,
    *,
    budget: int,
    seed: int,
    exposed_k: int = DEFAULT_EXPOSED_K,
    noise: float = 0.0,
) -> list[FeedbackEvent]:
    events = simulate_random(
        load_rankings(rankings_path), budget=budget, seed=seed, exposed_k=exposed_k, noise=noise
    )
    write_events(events, out_path)
    return events
