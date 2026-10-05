"""C4 feedback events to labels, contract c4-labels-v1 (D20, docs/contracts/c4-event-labels.md).

Reads ``FeedbackEvent`` JSONL and emits ``Label`` JSONL plus ``<out>.report.json``.
The builder never sees the exposed candidate list or the ground truth, so an unanswered
candidate cannot become a negative: only ``match`` / ``not_match`` events make labels.

Rules, per (query_id, product_id) pair:
- identical ``event_id`` repeats are collapsed; the same id with different content is an error.
- within a session the judgment with the latest ``ts`` wins (tie: last ``event_id``).
- across sessions the majority wins; a tie yields no label.
- ``weight`` = winning sessions / judging sessions.
The output depends only on the set of events, not on their order.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from product_retrieval.core.ids import canonical_json, sha256_bytes
from product_retrieval.core.schemas import FeedbackEvent, Label

CONTRACT = "c4-labels-v1"
_JUDGMENTS = ("match", "not_match")


class LabelBuildError(ValueError):
    """Raised for invalid event input (unparseable line, conflicting duplicate event_id)."""


def _hash(*parts: object) -> str:
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))


def load_events(path: str | Path) -> list[FeedbackEvent]:
    events: list[FeedbackEvent] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                events.append(FeedbackEvent.model_validate_json(line))
            except ValueError as exc:
                raise LabelBuildError(f"{path}:{lineno}: invalid event: {exc}") from exc
    return events


def _utc(ts: datetime) -> datetime:
    # naive timestamps are read as UTC so naive and aware values stay comparable
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _dedupe(events: list[FeedbackEvent]) -> list[FeedbackEvent]:
    by_id: dict[str, FeedbackEvent] = {}
    for ev in events:
        prev = by_id.get(ev.event_id)
        if prev is None:
            by_id[ev.event_id] = ev
        elif prev != ev:
            raise LabelBuildError(f"event_id {ev.event_id!r} appears twice with different content")
    return [by_id[k] for k in sorted(by_id)]


def build_labels(events: list[FeedbackEvent]) -> tuple[list[Label], dict]:
    """Return labels sorted by (query_id, product_id) and the report dict (contract section 4)."""
    unique = _dedupe(events)
    version = _hash(CONTRACT, [ev.event_id for ev in unique])[:12]
    action_counts = Counter(ev.action for ev in unique)

    judged: dict[tuple[str, str], list[FeedbackEvent]] = defaultdict(list)
    skip_pairs: set[tuple[str, str]] = set()
    other_pairs: set[tuple[str, str]] = set()
    for ev in unique:
        pair = (ev.query_id, ev.product_id)
        if ev.action in _JUDGMENTS:
            judged[pair].append(ev)
        elif ev.action == "skip":
            skip_pairs.add(pair)
        else:
            other_pairs.add(pair)

    labels: list[Label] = []
    conflicts: list[dict] = []
    tied = 0
    for pair in sorted(judged):
        evs = judged[pair]
        final: dict[str, FeedbackEvent] = {}
        for ev in evs:
            cur = final.get(ev.session_id)
            if cur is None or (_utc(ev.ts), ev.event_id) > (_utc(cur.ts), cur.event_id):
                final[ev.session_id] = ev
        votes = Counter(ev.action for ev in final.values())
        n_match, n_not = votes["match"], votes["not_match"]
        origin = sorted(ev.event_id for ev in evs)
        if n_match == n_not:
            kind = None
            tied += 1
        else:
            kind = "pos" if n_match > n_not else "neg"
            labels.append(
                Label(
                    query_id=pair[0],
                    product_id=pair[1],
                    kind=kind,
                    weight=max(n_match, n_not) / (n_match + n_not),
                    origin_events=origin,
                    label_version=version,
                )
            )
        if {ev.action for ev in evs} == set(_JUDGMENTS):
            conflicts.append(
                {
                    "query_id": pair[0],
                    "product_id": pair[1],
                    "session_judgments": {s: final[s].action for s in sorted(final)},
                    "resolution": kind or "none",
                    "event_ids": origin,
                }
            )

    skip_only = len(skip_pairs - judged.keys())
    other_only = len(other_pairs - judged.keys() - skip_pairs)
    report = {
        "contract": CONTRACT,
        "label_version": version,
        "input_events": len(events),
        "unique_events": len(unique),
        "duplicates_removed": len(events) - len(unique),
        "action_counts": {
            a: action_counts.get(a, 0) for a in ("match", "not_match", "prefer", "point", "skip")
        },
        "pos": sum(lb.kind == "pos" for lb in labels),
        "neg": sum(lb.kind == "neg" for lb in labels),
        "unlabeled_pairs": {"skip_only": skip_only, "tied": tied, "prefer_point_only": other_only},
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    return labels, report


def run_labels(events_path: str | Path, out_path: str | Path) -> dict:
    """Convert events to labels; write Label JSONL to ``out_path`` and ``<out_path>.report.json``."""
    labels, report = build_labels(load_events(events_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for lb in labels:
            f.write(lb.model_dump_json() + "\n")
    out.with_name(out.name + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
