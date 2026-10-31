"""Sampler for the product-ID audit (contract ladder-1-id-audit v1.1, sections 2, 5, 6 branch 8, 8).

Draws the 600 audit products (stratified over the validation split), the 200 pilot, the
re-review sets, the fresh validation sample, and the opaque display order for each stage.
Everything is a pure function of the manifest bytes and the fixed seeds.

TODO (not in this piece): item counts for stage B (across validation IDs) and stage D
(train-validation) need frozen SigLIP2 product scores and pHash hits; the decoy pairs of
stage C also need product scores (only the number of decoy slots is counted here).

Choices the contract leaves open (reported, not silently decided):
- ``rereview_pilot`` and ``rereview_full`` allocate over strata proportionally with largest
  remainder (ties go to the lower stratum); the contract only says "proportionally" for the
  pilot 40, and says nothing about strata for the 80.
- Within a stratum the re-review draw takes the first k of a permutation of the candidate IDs
  (sorted as integers); one generator per draw is used across strata in order h = 1..4.
- Display order is drawn over the items of the 600, not per pilot/rest batch.
- Products with no query or no gallery image fit no stratum and are an error.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from product_retrieval.core.ids import canonical_json, sha256_bytes
from product_retrieval.data.manifest import ManifestRow, load_manifest, manifest_sha

CONTRACT_VERSION = "ladder-1-id-audit-v1.1"
SAMPLE_SEED = 20261031
DISPLAY_SEED = 20261032
REREVIEW_SEED = 20261033
FRESH_SEED = 20261035
STRATA = (1, 2, 3, 4)
STAGES = {1: "target", 2: "C", 3: "A"}


class AuditSampleError(Exception):
    """The audit sample cannot be drawn (changed manifest, bad strata, impossible sizes)."""


@dataclass(frozen=True)
class AuditDesign:
    """Sample sizes. ``DEFAULT_DESIGN`` is the contract's; tests use small ones."""

    sample: dict[int, int]
    pilot: dict[int, int]
    rereview_pilot: int = 40
    rereview_full: int = 80
    fresh: int = 600
    expected_population: dict[int, int] | None = field(default=None)


DEFAULT_DESIGN = AuditDesign(
    sample={1: 285, 2: 94, 3: 190, 4: 31},
    pilot={1: 95, 2: 31, 3: 63, 4: 11},
    expected_population={1: 1538, 2: 508, 3: 1025, 4: 172},
)


def stratum_of(row: ManifestRow) -> int:
    """Stratum h of a product (contract section 2)."""
    nq, ng = len(row.query), len(row.gallery)
    if nq == 0 or ng == 0:
        raise AuditSampleError(f"product {row.product_id}: {nq} queries, {ng} gallery images; no stratum")
    if ng >= 2:
        return 4
    return 1 if nq == 1 else 2 if nq == 2 else 3


def largest_remainder(total: int, weights: Sequence[int]) -> list[int]:
    """Split ``total`` proportionally to ``weights`` (integers summing to > 0); the sum is exact.

    Floors first, then the leftover units go to the largest remainders; ties go to the lower index.
    """
    if total < 0 or any(w < 0 for w in weights):
        raise AuditSampleError("largest_remainder: negative total or weight")
    w_sum = sum(weights)
    if w_sum == 0:
        raise AuditSampleError("largest_remainder: weights sum to zero")
    if total > w_sum:
        raise AuditSampleError(f"largest_remainder: total {total} exceeds the weights' sum {w_sum}")
    base = [total * w // w_sum for w in weights]
    rem = [total * w % w_sum for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-rem[i], i))
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def _by_stratum(rows: Sequence[ManifestRow]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {h: [] for h in STRATA}
    for row in rows:
        out[stratum_of(row)].append(row.product_id)
    for ids in out.values():
        ids.sort(key=int)
    return out


def _check_ints(rows: Sequence[ManifestRow]) -> None:
    seen: set[str] = set()
    for row in rows:
        if not row.product_id.isascii() or not row.product_id.isdigit():
            raise AuditSampleError(f"product_id {row.product_id!r} is not an integer string")
        if row.product_id in seen:
            raise AuditSampleError(f"product_id {row.product_id} appears twice in split val")
        seen.add(row.product_id)


def draw_sample(
    rows: Sequence[ManifestRow], design: AuditDesign = DEFAULT_DESIGN
) -> tuple[dict[int, list[str]], dict[int, list[str]], dict[int, int]]:
    """Return ``(sample, pilot, N_h)``, each ID list per stratum in draw order."""
    _check_ints(rows)
    strata = _by_stratum(rows)
    n_h = {h: len(ids) for h, ids in strata.items()}
    if design.expected_population is not None and n_h != design.expected_population:
        raise AuditSampleError(f"stratum sizes {n_h} differ from expected {design.expected_population}")
    sample: dict[int, list[str]] = {}
    pilot: dict[int, list[str]] = {}
    for h in STRATA:
        n, p = design.sample[h], design.pilot[h]
        if not 0 <= p <= n <= n_h[h]:
            raise AuditSampleError(f"stratum {h}: need pilot {p} <= sample {n} <= population {n_h[h]}")
        ids = list(strata[h])
        order = np.random.default_rng([SAMPLE_SEED, h]).permutation(len(ids))
        shuffled = [ids[i] for i in order]
        sample[h] = shuffled[:n]
        pilot[h] = shuffled[:p]
    return sample, pilot, n_h


def _flat(per_stratum: dict[int, list[str]]) -> list[str]:
    return [i for h in STRATA for i in per_stratum[h]]


def _proportional_draw(
    candidates: dict[int, list[str]], total: int, rng: np.random.Generator
) -> dict[int, list[str]]:
    quotas = largest_remainder(total, [len(candidates[h]) for h in STRATA])
    out: dict[int, list[str]] = {}
    for h, k in zip(STRATA, quotas, strict=True):
        ids = sorted(candidates[h], key=int)
        out[h] = [ids[i] for i in rng.permutation(len(ids))[:k]]
    return out


def draw_rereview(
    sample: dict[int, list[str]], pilot: dict[int, list[str]], design: AuditDesign = DEFAULT_DESIGN
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Return ``(rereview_pilot, rereview_extra)``: 40 of the pilot, then 80 of the other 400."""
    rng1 = np.random.default_rng([REREVIEW_SEED, 1])
    first = _proportional_draw(pilot, design.rereview_pilot, rng1)
    pilot_ids = set(_flat(pilot))
    rest = {h: [i for i in sample[h] if i not in pilot_ids] for h in STRATA}
    rng2 = np.random.default_rng([REREVIEW_SEED, 2])
    extra = _proportional_draw(rest, design.rereview_full, rng2)
    return first, extra


def draw_fresh(
    rows: Sequence[ManifestRow],
    sampled: Sequence[str],
    design: AuditDesign = DEFAULT_DESIGN,
    risk_group: Sequence[str] = (),
) -> dict[int, list[str]]:
    """Branch 8: ``design.fresh`` products from those not sampled (and not in the risk group).

    ``n_h`` is proportional to the remaining ``N_h`` (largest remainder, sum exact); stratum h
    is shuffled with ``default_rng([20261035, h])`` over IDs sorted as integers.
    """
    _check_ints(rows)
    taken = set(sampled) | set(risk_group)
    pool = _by_stratum([r for r in rows if r.product_id not in taken])
    quotas = largest_remainder(design.fresh, [len(pool[h]) for h in STRATA])
    out: dict[int, list[str]] = {}
    for h, k in zip(STRATA, quotas, strict=True):
        ids = pool[h]
        order = np.random.default_rng([FRESH_SEED, h]).permutation(len(ids))
        out[h] = [ids[i] for i in order[:k]]
    return out


def stage_counts(rows: Sequence[ManifestRow], product_ids: Sequence[str]) -> dict[str, int]:
    """Item counts per stage for the given products (contract section 8; B and D not included)."""
    by_id = {r.product_id: r for r in rows}
    targets = c_pairs = a_pairs = 0
    for pid in product_ids:
        r = by_id[pid]
        nq, ng = len(r.query), len(r.gallery)
        targets += nq
        c_pairs += nq * ng
        a_pairs += nq * (nq - 1) // 2 + ng * (ng - 1) // 2
    return {"target": targets, "C_pairs": c_pairs, "C_decoys": c_pairs, "A_pairs": a_pairs}


def stage_items(rows: Sequence[ManifestRow], product_ids: Sequence[str]) -> dict[int, list[tuple]]:
    """Items per stage as key tuples, for the display permutation. Decoy slots are ``("decoy", n)``."""
    by_id = {r.product_id: r for r in rows}
    target: list[tuple] = []
    c_items: list[tuple] = []
    a_items: list[tuple] = []
    for pid in product_ids:
        r = by_id[pid]
        target += [("target", q) for q in r.query]
        c_items += [("C", q, g) for q in r.query for g in r.gallery]
        for imgs in (r.query, r.gallery):
            a_items += [("A", imgs[i], imgs[j]) for i in range(len(imgs)) for j in range(i + 1, len(imgs))]
    c_items += [("decoy", n) for n in range(len(c_items))]
    return {1: target, 2: c_items, 3: a_items}


def display_plan(items: Sequence[tuple], stage: int) -> list[tuple[str, tuple]]:
    """Opaque display order for one stage: ``[(display_id, item)]`` in display order.

    The permutation and the tokens both come from ``default_rng([20261032, stage])``; tokens are
    random 64-bit values, unrelated to product IDs, item order or the sort order of the IDs.
    """
    rng = np.random.default_rng([DISPLAY_SEED, stage])
    order = rng.permutation(len(items))
    tokens: list[str] = []
    used: set[str] = set()
    while len(tokens) < len(items):
        token = rng.bytes(8).hex()
        if token not in used:
            used.add(token)
            tokens.append(token)
    return [(f"{STAGES[stage][0].lower()}{t}", items[i]) for t, i in zip(tokens, order, strict=True)]


def ids_sha256(ids: Sequence[str]) -> str:
    """sha256 of the canonical JSON of an ID list, in draw order."""
    return sha256_bytes(canonical_json(list(ids)).encode("utf-8"))


def run_audit_sample(
    manifest_path: str | Path,
    expect_sha: str,
    out: str | Path,
    design: AuditDesign = DEFAULT_DESIGN,
) -> dict[str, Any]:
    """Draw everything and write the JSON to ``out`` (refuses to overwrite)."""
    actual = manifest_sha(manifest_path)
    if actual != expect_sha:
        raise AuditSampleError(f"manifest sha256 {actual} differs from expected {expect_sha}")
    out = Path(out)
    if out.exists():
        raise AuditSampleError(f"{out} exists; the committed sample is never rewritten")
    rows = [r for r in load_manifest(manifest_path) if r.split == "val"]
    if not rows:
        raise AuditSampleError("manifest has no val products")
    sample, pilot, n_h = draw_sample(rows, design)
    rr_pilot, rr_extra = draw_rereview(sample, pilot, design)
    sample_ids, pilot_ids = _flat(sample), _flat(pilot)
    fresh = draw_fresh(rows, sample_ids, design)
    sets = {
        "sample": sample_ids,
        "pilot": pilot_ids,
        "rereview_pilot": _flat(rr_pilot),
        "rereview_extra": _flat(rr_extra),
        "fresh": _flat(fresh),
    }
    display: dict[str, list[dict[str, Any]]] = {}
    for stage, items in stage_items(rows, sample_ids).items():
        display[STAGES[stage]] = [
            {"display_id": d, "item": list(item)} for d, item in display_plan(items, stage)
        ]
    result = {
        "contract": CONTRACT_VERSION,
        "manifest_sha256": actual,
        "population": len(rows),
        "N_h": {str(h): n for h, n in n_h.items()},
        "n_h": {str(h): len(sample[h]) for h in STRATA},
        "pilot_h": {str(h): len(pilot[h]) for h in STRATA},
        "seeds": {
            "sample": [SAMPLE_SEED, "h"],
            "display": [DISPLAY_SEED, "stage"],
            "rereview_pilot": [REREVIEW_SEED, 1],
            "rereview_extra": [REREVIEW_SEED, 2],
            "fresh": [FRESH_SEED, "h"],
        },
        "sets": sets,
        "sets_sha256": {k: ids_sha256(v) for k, v in sets.items()},
        "stage_counts": {
            "sample": stage_counts(rows, sample_ids),
            "pilot": stage_counts(rows, pilot_ids),
        },
        "display": display,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result
