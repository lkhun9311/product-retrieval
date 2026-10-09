"""C4 simulated feedback generator, contract c4-v3 (D20, docs/contracts/c4-label-selection.md).

Reads the per-query rankings JSONL written by ``pr eval`` and emits
``FeedbackEvent`` JSONL. The ground truth (``truth_product_id``) is used only
inside this module to decide match/not_match; it is never written to the output.

Conventions:
- ``position`` is the **1-based** rank of the candidate in the exposed list.
- ``event_id`` = sha256(namespace, policy_id, seed, query_id, product_id); the namespace
  covers (contract version, rankings file sha256, noise p). Budget is not part of it.
- The shared initial label set I_s of the contract is simply the first 100 lines of a
  ``stratified`` run with seed s (selection is nested in the budget), so no separate output exists.
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

CONTRACT = "c4-v3"
POLICIES = ("random", "stratified")
DEFAULT_EXPOSED_K = 20
# 0-based half-open rank ranges of S1 (1-5), S2 (6-10), S3 (11-20)
STRATA = ((0, 5), (5, 10), (10, 20))
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


def namespace_base(rankings_sha256: str) -> str:
    """Noise-independent namespace: the same per-pair uniform u is reused across p."""
    return _hash(CONTRACT, rankings_sha256)


def make_namespace(rankings_sha256: str, noise: float) -> str:
    return _hash(CONTRACT, rankings_sha256, noise)


def event_id(namespace: str, policy_id: str, seed: int, query_id: str, product_id: str) -> str:
    """Deterministic idempotency key for one judgment. Budget is deliberately absent."""
    return _hash(namespace, policy_id, seed, query_id, product_id)


def noise_u(base: str, seed: int, query_id: str, product_id: str) -> float:
    """Per-pair uniform in [0, 1), independent of policy, budget and order."""
    return int(_hash(base, seed, query_id, product_id)[:16], 16) / 2**64


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


def _validate(rankings: list[dict], exposed_k: int, *, require_full: bool) -> None:
    seen_queries: set[str] = set()
    for row in rankings:
        qid = row["query_id"]
        if qid in seen_queries:
            raise SimulationError(f"duplicate query_id in rankings: {qid!r}")
        seen_queries.add(qid)
        # Contract section 2: query ids must be opaque. The legacy `source:product_id:sha12`
        # form would carry the truth into every event, so refuse it outright.
        if row["truth_product_id"] in qid.split(":"):
            raise SimulationError(
                f"query_id {qid!r} reveals its truth product; regenerate rankings with opaque query ids"
            )
        exposed = row["top_k_product_ids"][:exposed_k]
        if len(set(exposed)) != len(exposed):
            raise SimulationError(f"query {qid!r} lists a product more than once")
        if require_full and len(exposed) < exposed_k:
            raise SimulationError(
                f"query {qid!r} has {len(exposed)} candidates, stratified needs {exposed_k}"
            )


def _build_pairs(rankings: list[dict], exposed_k: int) -> list[_Pair]:
    pairs: list[_Pair] = []
    for row in rankings:
        for rank0, pid in enumerate(row["top_k_product_ids"][:exposed_k]):
            pairs.append(_Pair(row["query_id"], pid, rank0 + 1, pid == row["truth_product_id"]))
    return pairs


def _stratified_order(rankings: list[dict], budget: int, seed: int) -> list[_Pair]:
    """First ``budget`` pairs of the round-robin S1,S2,S3 order. Never reads the truth."""
    rows = sorted(rankings, key=lambda r: r["query_id"])
    n = len(rows)
    if budget > 3 * n:
        raise SimulationError(f"budget {budget} exceeds the stratified capacity 3 x {n} queries = {3 * n}")
    per_stratum: list[list[tuple[str, str, int]]] = []
    for (lo, hi), child in zip(STRATA, np.random.SeedSequence(seed).spawn(3), strict=True):
        rng = np.random.Generator(np.random.PCG64(child))
        perm = rng.permutation(n)
        ranks = rng.integers(lo, hi, size=n)  # ranks[i] belongs to sorted query i
        stratum = []
        for qi in perm.tolist():
            r0 = int(ranks[qi])
            stratum.append((rows[qi]["query_id"], rows[qi]["top_k_product_ids"][r0], r0 + 1))
        per_stratum.append(stratum)
    order: list[_Pair] = []
    for i in range(budget):
        qid, pid, pos = per_stratum[i % 3][i // 3]
        order.append(_Pair(qid, pid, pos, False))
    return order


def _check_args(budget: int, exposed_k: int, noise: float) -> None:
    if budget < 0:
        raise SimulationError(f"budget must be >= 0, got {budget}")
    if exposed_k < 1:
        raise SimulationError(f"exposed_k must be >= 1, got {exposed_k}")
    if not 0.0 <= noise <= 1.0:
        raise SimulationError(f"noise must be in [0, 1], got {noise}")


def _judge(
    selected: list[_Pair],
    truth: dict[str, str],
    *,
    policy_id: str,
    seed: int,
    noise: float,
    rankings_sha256: str,
    first_index: int = 0,
) -> list[FeedbackEvent]:
    ns = make_namespace(rankings_sha256, noise)
    base = namespace_base(rankings_sha256)
    events: list[FeedbackEvent] = []
    for i, pair in enumerate(selected):
        flip = noise_u(base, seed, pair.query_id, pair.product_id) < noise
        is_match = (pair.product_id == truth[pair.query_id]) != flip
        events.append(
            FeedbackEvent(
                event_id=event_id(ns, policy_id, seed, pair.query_id, pair.product_id),
                ts=EPOCH + timedelta(seconds=first_index + i),
                session_id=f"sim-{policy_id}-{seed}",
                list_id=_hash("list", ns, policy_id, seed, pair.query_id),
                query_id=pair.query_id,
                product_id=pair.product_id,
                action="match" if is_match else "not_match",
                position=pair.position,
                policy_id=policy_id,
                actor="sim",
            )
        )
    return events


def simulate_random(
    rankings: list[dict],
    *,
    budget: int,
    seed: int,
    rankings_sha256: str,
    exposed_k: int = DEFAULT_EXPOSED_K,
    noise: float = 0.0,
) -> list[FeedbackEvent]:
    """Reference policy: sample ``budget`` pairs uniformly without replacement, then judge them.

    Raises ``SimulationError`` if ``budget`` exceeds the number of exposed pairs.
    """
    _check_args(budget, exposed_k, noise)
    _validate(rankings, exposed_k, require_full=False)
    pairs = _build_pairs(rankings, exposed_k)
    if budget > len(pairs):
        raise SimulationError(
            f"budget {budget} exceeds the {len(pairs)} available (query, candidate) pairs "
            f"(exposed_k={exposed_k})"
        )
    chosen = np.random.default_rng(seed).choice(len(pairs), size=budget, replace=False)
    truth = {r["query_id"]: r["truth_product_id"] for r in rankings}
    return _judge(
        [pairs[i] for i in chosen.tolist()],
        truth,
        policy_id="random",
        seed=seed,
        noise=noise,
        rankings_sha256=rankings_sha256,
    )


def simulate_stratified(
    rankings: list[dict],
    *,
    budget: int,
    seed: int,
    rankings_sha256: str,
    exposed_k: int = DEFAULT_EXPOSED_K,
    noise: float = 0.0,
) -> list[FeedbackEvent]:
    """Pre-registered policy: rank strata S1/S2/S3, one judgment per query per stratum."""
    _check_args(budget, exposed_k, noise)
    if exposed_k != 20:
        raise SimulationError(f"stratified requires exposed_k == 20, got {exposed_k}")
    _validate(rankings, exposed_k, require_full=True)
    selected = _stratified_order(rankings, budget, seed)
    truth = {r["query_id"]: r["truth_product_id"] for r in rankings}
    return _judge(
        selected,
        truth,
        policy_id="stratified",
        seed=seed,
        noise=noise,
        rankings_sha256=rankings_sha256,
    )


def stratified_pairs(
    rankings: list[dict], *, budget: int, seed: int, exposed_k: int = DEFAULT_EXPOSED_K
) -> list[tuple[str, str, int]]:
    """The first ``budget`` (query_id, product_id, position) of the stratified order; truth is not read."""
    if exposed_k != 20:
        raise SimulationError(f"stratified requires exposed_k == 20, got {exposed_k}")
    _validate(rankings, exposed_k, require_full=True)
    return [(p.query_id, p.product_id, p.position) for p in _stratified_order(rankings, budget, seed)]


class Oracle:
    """Answers chosen pairs from the truth. The one place the uncertainty path reads ``truth_product_id``.

    Answers and noise are the c4-v3 rules of ``_judge``; ``first_index`` continues the event clock so a
    path built round by round has the timestamps of a single run.
    """

    def __init__(
        self,
        rankings: list[dict],
        *,
        rankings_sha256: str,
        seed: int,
        policy_id: str,
        noise: float = 0.0,
        exposed_k: int = DEFAULT_EXPOSED_K,
    ):
        _check_args(0, exposed_k, noise)
        _validate(rankings, exposed_k, require_full=True)
        self._truth = {r["query_id"]: r["truth_product_id"] for r in rankings}
        self._exposed = {r["query_id"]: list(r["top_k_product_ids"][:exposed_k]) for r in rankings}
        self._kw = {"policy_id": policy_id, "seed": seed, "noise": noise, "rankings_sha256": rankings_sha256}

    def answer(self, pairs: list[tuple[str, str, int]], first_index: int = 0) -> list[FeedbackEvent]:
        selected = []
        for qid, pid, pos in pairs:
            exposed = self._exposed.get(qid)
            if exposed is None or pos < 1 or pos > len(exposed) or exposed[pos - 1] != pid:
                raise SimulationError(f"pair ({qid!r}, {pid!r}, position {pos}) is not in the exposed pool")
            selected.append(_Pair(qid, pid, pos, False))
        return _judge(selected, self._truth, first_index=first_index, **self._kw)


def stratum_of(position: int) -> int:
    """0-based stratum index of a 1-based rank."""
    for i, (lo, hi) in enumerate(STRATA):
        if lo < position <= hi:
            return i
    raise SimulationError(f"position {position} is outside the strata")


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
    policy: str = "stratified",
    exposed_k: int = DEFAULT_EXPOSED_K,
    noise: float = 0.0,
) -> list[FeedbackEvent]:
    """Run a policy, write events to ``out_path`` and ``<out_path>.summary.json``."""
    if policy not in POLICIES:
        raise SimulationError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    sha = sha256_bytes(Path(rankings_path).read_bytes())
    fn = simulate_stratified if policy == "stratified" else simulate_random
    events = fn(
        load_rankings(rankings_path),
        budget=budget,
        seed=seed,
        rankings_sha256=sha,
        exposed_k=exposed_k,
        noise=noise,
    )
    write_events(events, out_path)
    strata = None
    if policy == "stratified":
        strata = [0, 0, 0]
        for ev in events:
            strata[stratum_of(ev.position)] += 1
    summary = {
        "contract": CONTRACT,
        "namespace": make_namespace(sha, noise),
        "rankings_sha256": sha,
        "policy": policy,
        "seed": seed,
        "budget": budget,
        "noise": noise,
        "stratum_counts": strata,
        "match_count": sum(ev.action == "match" for ev in events),
        "event_count": len(events),
    }
    out = Path(out_path)
    out.with_name(out.name + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return events
