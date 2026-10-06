"""`pr crop-search`: whole-photo search vs detector -> crop -> merge on IKEA real room photos.

Contract: docs/contracts/crop-before-search.md (builds on ikea-real-photo-check.md v1.1).

Flow: population (gallery, rooms, answers) exactly as ``ikea_check.load_population`` -> embed the
gallery and every whole room photo (``EmbeddingCache``, full-image crop hash) -> per room detect
crops on the EXIF-transposed RGB photo, embed each crop (cache key: image sha, crop-spec hash,
model id) -> arm Whole: ranking of the whole-photo vector; arm Crops: per-crop rankings merged
round-robin -> per-room Hit@K / Recall@K for both arms -> paired cluster bootstrap of the mean
difference -> ``reports/crop/<run-id>.json`` (never overwritten; every attempt is kept).

Evaluation only: nothing here trains or selects a model.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from product_retrieval.core.ids import canonical_json, sha256_bytes, sha256_file
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.schemas import BBox, CropSpec
from product_retrieval.crop.boxes import (
    EXPAND_FRAC,
    MIN_CROP_PX,
    NMS_IOU,
    SCORE_THRESHOLD,
    TOP_M,
    Crop,
    Detector,
    exif_orientation,
    load_exif_rgb,
)
from product_retrieval.crop.fake import FakeDetector
from product_retrieval.embed.base import Embedder
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.eval.merge import (
    BOOTSTRAP_B,
    BOOTSTRAP_LEVEL,
    BOOTSTRAP_SEED,
    CLUSTER_COSINE,
    cluster_rooms,
    merge_round_robin,
    paired_cluster_bootstrap,
    verdict,
)
from product_retrieval.eval.multi import IKEA_KS, hit_at_k, recall_at_k
from product_retrieval.pipelines.ikea_check import (
    _check_vectors,
    _embed_files,
    _load_image,
    _make_embedder,
    load_population,
)

CropDetectorName = Literal["owl", "fake"]

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "contracts" / "crop-before-search.md"
CONTRACT_NAME = "crop-before-search v1"
PRIMARY_K = 10
EXPECTED_ROOMS = 205  # #29 population; recorded as a flag, not asserted (tests use tiny folders)
EXPECTED_GALLERY = 2173
_EMBED_CHUNK = 64


class CropSearchError(RuntimeError):
    """Raised when the population, a detector output or an embedding cannot be trusted."""


@dataclass(frozen=True)
class CropSearchResult:
    report_path: Path
    population_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def _make_detector(name: str) -> Detector:
    if name == "owl":
        from product_retrieval.crop.owl import OwlDetector  # heavy import only when used

        return OwlDetector()
    if name == "fake":
        return FakeDetector()
    raise ValueError(f"unknown detector {name!r}; expected 'owl' or 'fake'")


def _crop_spec_hash(box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    spec = CropSpec(kind="box", box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return compute_crop_hash(spec)


@dataclass(frozen=True)
class _Item:
    sha: str
    crop_hash: str
    load: Callable[[], Image.Image]


def _embed_items(
    items: list[_Item], embedder: Embedder, cache: EmbeddingCache
) -> tuple[list[np.ndarray], list[float], int]:
    """Embed items through the cache. Returns vectors, seconds attributed per item, cache hits.

    Seconds are the embedder wall time of the chunk an item was computed in, divided by the chunk
    size; an item read from the cache costs 0.
    """
    vectors: list[np.ndarray | None] = [None] * len(items)
    seconds = [0.0] * len(items)
    hits = 0
    todo: list[int] = []
    for i, it in enumerate(items):
        got, missing = cache.get_many([it.sha], it.crop_hash, embedder.model_id)
        if missing:
            todo.append(i)
        else:
            _check_vectors(got, 1, embedder.dim, "cached vectors")
            vectors[i] = got[0]
            hits += 1
    for start in range(0, len(todo), _EMBED_CHUNK):
        chunk = todo[start : start + _EMBED_CHUNK]
        images = [items[i].load() for i in chunk]
        t0 = time.perf_counter()
        out = np.asarray(embedder.embed(images))
        elapsed = time.perf_counter() - t0
        _check_vectors(out, len(chunk), embedder.dim, "embedder output")
        for i, vec in zip(chunk, out, strict=True):
            cache.put_many([items[i].sha], vec[None, :], items[i].crop_hash, embedder.model_id)
            vectors[i] = vec
            seconds[i] = elapsed / len(chunk)
    return [v for v in vectors if v is not None], seconds, hits


def _stats(values: list[float] | list[int]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def _list_sha(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def _commit_hash_of(embedder: object) -> str | None:
    model = getattr(embedder, "_model", None)
    cfg = getattr(model, "config", None)
    value = getattr(cfg, "_commit_hash", None)
    return value if isinstance(value, str) else None


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("transformers", "torch", "numpy", "PIL"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # version is informational
            out[mod] = "unknown"
    return out


def _new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"


def run_crop_search(
    raw_root: Path,
    embedder_name: str = "siglip",
    detector: CropDetectorName = "owl",
    artifacts_root: Path = Path("artifacts"),
    reports_root: Path = Path("reports"),
) -> CropSearchResult:
    """Run both arms on the IKEA folder and write ``reports/crop/<run-id>.json`` (new file each run)."""
    t_start = time.perf_counter()
    raw_root = Path(raw_root)
    if not CONTRACT_PATH.is_file():
        raise CropSearchError(f"contract file not found: {CONTRACT_PATH}")

    pop = load_population(raw_root)
    room_ids = sorted(pop.room_path)
    product_ids = sorted(pop.product_path)
    answers = [pop.room_answers[r] for r in room_ids]
    for rid, ans in zip(room_ids, answers, strict=True):
        if len(ans) < 1:
            raise CropSearchError(f"room {rid!r} has no answer product")

    # Population lists and hashes, written before any ranking is computed.
    population = {
        "products": product_ids,
        "rooms": room_ids,
        "answers": {r: sorted(pop.room_answers[r]) for r in room_ids},
    }
    pop_hashes = {name: _list_sha(population[name]) for name in population}
    population_sha = _list_sha(population)

    embedder = _make_embedder(embedder_name)  # type: ignore[arg-type]
    det = _make_detector(detector)
    cache = EmbeddingCache(Path(artifacts_root) / "embeddings")
    full_hash = compute_crop_hash(CropSpec(kind="full"))

    sha_of = pop.sha_of
    for path in [*pop.product_path.values(), *pop.room_path.values()]:
        if path not in sha_of:
            sha_of[path] = sha256_file(path)

    # Gallery (one image per product) and whole-photo room vectors: the same cache keys as ikea-check.
    gallery_by_sha = {sha_of[p]: p for p in pop.product_path.values()}
    vec_by_sha = _embed_files(gallery_by_sha, embedder, cache, full_hash)
    gallery = np.stack([vec_by_sha[sha_of[pop.product_path[p]]] for p in product_ids]).astype(np.float32)

    room_items = [
        _Item(sha_of[pop.room_path[r]], full_hash, lambda p=pop.room_path[r]: _load_image(p))
        for r in room_ids
    ]
    whole_list, whole_secs, whole_hits = _embed_items(room_items, embedder, cache)
    whole = np.stack(whole_list).astype(np.float32)

    # Clusters from whole-photo embeddings, fixed before any crop ranking.
    clusters = cluster_rooms(whole)

    # Detection per room on the EXIF-transposed RGB photo.
    crops_by_room: list[list[Crop]] = []
    det_secs: list[float] = []
    exif_rotated: list[str] = []
    for rid in room_ids:
        path = pop.room_path[rid]
        if exif_orientation(path) != 1:
            exif_rotated.append(rid)
        image = load_exif_rgb(path)
        t0 = time.perf_counter()
        crops = det.detect(image)
        det_secs.append(time.perf_counter() - t0)
        for c in crops:
            x1, y1, x2, y2 = c.box
            if not (0 <= x1 and 0 <= y1 and x2 <= image.width and y2 <= image.height):
                raise CropSearchError(f"room {rid!r}: crop {c.box} outside the {image.size} image")
            if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
                raise CropSearchError(f"room {rid!r}: crop {c.box} is under {MIN_CROP_PX} px")
        crops_by_room.append(crops)
    fallback = [len(c) == 0 for c in crops_by_room]

    # Embed every crop of every non-fallback room (cache key: image sha, crop-spec hash, model id).
    crop_items: list[_Item] = []
    crop_owner: list[tuple[int, int]] = []  # (room index, crop order)
    for ri, rid in enumerate(room_ids):
        path = pop.room_path[rid]
        for ci, c in enumerate(crops_by_room[ri]):
            crop_items.append(
                _Item(
                    sha_of[path],
                    _crop_spec_hash(c.box),
                    lambda p=path, box=c.box: load_exif_rgb(p).crop(box),
                )
            )
            crop_owner.append((ri, ci))
    crop_vecs, crop_secs, crop_hits = _embed_items(crop_items, embedder, cache)

    room_crop_vecs: list[list[np.ndarray]] = [[] for _ in room_ids]
    emb_secs_crops = [0.0] * len(room_ids)
    for (ri, _ci), vec, sec in zip(crop_owner, crop_vecs, crop_secs, strict=True):
        room_crop_vecs[ri].append(vec)
        emb_secs_crops[ri] += sec
    for ri in range(len(room_ids)):
        if fallback[ri]:  # whole photo is the only crop
            room_crop_vecs[ri] = [whole[ri]]
            emb_secs_crops[ri] = whole_secs[ri]

    # Rankings. Gallery is in ascending product id and the sort is stable, so ties go to the smaller id.
    ks = IKEA_KS
    k_max = max(ks)

    def rank(vec: np.ndarray) -> list[str]:
        sims = gallery @ vec.astype(np.float32)
        if not np.isfinite(sims).all():
            raise CropSearchError("similarity vector contains NaN or inf")
        return [product_ids[j] for j in np.argsort(-sims, kind="stable")]

    ranked_whole: list[list[str]] = []
    ranked_crops: list[list[str]] = []
    merge_secs: list[float] = []
    for ri in range(len(room_ids)):
        ranked_whole.append(rank(whole[ri])[:k_max])
        t0 = time.perf_counter()
        per_crop = [rank(v) for v in room_crop_vecs[ri]]
        ranked_crops.append(merge_round_robin(per_crop, k_max))
        merge_secs.append(time.perf_counter() - t0)

    # Same population in both arms, by construction and by check.
    if len(ranked_whole) != len(ranked_crops) or len(ranked_whole) != len(answers):
        raise CropSearchError("arms do not cover the same rooms")
    for rid, rw, rc in zip(room_ids, ranked_whole, ranked_crops, strict=True):
        if len(rw) != min(k_max, len(product_ids)) or len(rc) != min(k_max, len(product_ids)):
            raise CropSearchError(f"room {rid!r}: an arm did not return min(K, gallery) products")
        if not set(rw) <= set(product_ids) or not set(rc) <= set(product_ids):
            raise CropSearchError(f"room {rid!r}: an arm returned a product outside the gallery")

    hit_w, hit_c = hit_at_k(ranked_whole, answers, ks), hit_at_k(ranked_crops, answers, ks)
    rec_w, rec_c = recall_at_k(ranked_whole, answers, ks), recall_at_k(ranked_crops, answers, ks)
    d_hit, d_rec = hit_c - hit_w, rec_c - rec_w
    n_k = len(ks)
    boot = paired_cluster_bootstrap(
        np.concatenate([d_hit, d_rec], axis=1), clusters, BOOTSTRAP_B, BOOTSTRAP_SEED, BOOTSTRAP_LEVEL
    )

    def delta_entry(col: int, per_room: np.ndarray) -> dict[str, Any]:
        j = col % n_k
        v = per_room[:, j]
        return {
            "mean_delta": float(boot.observed[col]),
            "ci95": [float(boot.lo[col]), float(boot.hi[col])],
            "share_resamples_delta_le_0": float(boot.share_le_zero[col]),
            "rooms_delta_gt_0": int((v > 0).sum()),
            "rooms_delta_eq_0": int((v == 0).sum()),
            "rooms_delta_lt_0": int((v < 0).sum()),
        }

    metrics: dict[str, Any] = {}
    for name, w, c, dm, off in (("hit", hit_w, hit_c, d_hit, 0), ("recall", rec_w, rec_c, d_rec, n_k)):
        metrics[name] = {
            str(k): {
                "whole": float(w[:, j].mean()),
                "crops": float(c[:, j].mean()),
                **delta_entry(off + j, dm),
            }
            for j, k in enumerate(ks)
        }
    primary_col = n_k + ks.index(PRIMARY_K)
    p_lo, p_hi = float(boot.lo[primary_col]), float(boot.hi[primary_col])
    verdict_sentence = verdict(p_lo, p_hi)

    n_boxes = [len(c) for c in crops_by_room]
    per_room_rows = []
    for ri, rid in enumerate(room_ids):
        per_room_rows.append(
            {
                "room": rid,
                "cluster": int(clusters[ri]),
                "n_answers": len(answers[ri]),
                "n_crops": n_boxes[ri],
                "fallback_to_whole": fallback[ri],
                "crops": [
                    {"box": list(c.box), "score": c.score, "index": c.index} for c in crops_by_room[ri]
                ],
                "hit": {
                    "whole": {str(k): float(hit_w[ri, j]) for j, k in enumerate(ks)},
                    "crops": {str(k): float(hit_c[ri, j]) for j, k in enumerate(ks)},
                },
                "recall": {
                    "whole": {str(k): float(rec_w[ri, j]) for j, k in enumerate(ks)},
                    "crops": {str(k): float(rec_c[ri, j]) for j, k in enumerate(ks)},
                },
                "delta_recall_at_10": float(d_rec[ri, ks.index(PRIMARY_K)]),
                "seconds": {
                    "whole": {
                        "detector": 0.0,
                        "embedding": whole_secs[ri],
                        "total": whole_secs[ri],
                    },
                    "crops": {
                        "detector": det_secs[ri],
                        "embedding": emb_secs_crops[ri],
                        "total": det_secs[ri] + emb_secs_crops[ri],
                    },
                },
            }
        )

    def arm_seconds(arm: str, part: str) -> dict[str, float]:
        return _stats([row["seconds"][arm][part] for row in per_room_rows])

    run_id = _new_run_id()
    out_dir = Path(reports_root) / "crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    population_path = out_dir / f"{run_id}.population.json"
    report_path = out_dir / f"{run_id}.json"

    model_ids = {"embedder": embedder.model_id, "detector": det.model_id}
    report: dict[str, Any] = {
        "run_id": run_id,
        "contract": CONTRACT_NAME,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "created_utc": datetime.now(UTC).isoformat(),
        "exploratory": True,
        "models": {
            **model_ids,
            "embedder_revision": _commit_hash_of(embedder),  # null = not pinned / unknown
            "detector_revision": det.revision,
            "versions": _versions(),
        },
        "embedder_name": embedder_name,
        "detector_name": detector,
        "population": {
            "rooms": len(room_ids),
            "gallery_size": len(product_ids),
            "matches_29_counts": len(room_ids) == EXPECTED_ROOMS and len(product_ids) == EXPECTED_GALLERY,
            "sha256": {**pop_hashes, "all": population_sha},
            "list_file": str(population_path),
            "rooms_missing_file": pop.rooms_missing_file,
            "rooms_dropped_no_existing_product": pop.rooms_dropped,
            "duplicate_basenames_identical_content": sorted(set(pop.identical_dups)),
            "excluded_basenames_conflicting_content": sorted(set(pop.conflicts)),
            "arms_identical_population": True,  # asserted above; a failure raises
            "answers_per_room": _stats([len(a) for a in answers]),
            "rooms_with_exif_rotation": exif_rotated,
        },
        "detector_settings": {
            "score_threshold": SCORE_THRESHOLD,
            "nms_iou": NMS_IOU,
            "top_m": TOP_M,
            "expand_frac": EXPAND_FRAC,
            "min_crop_px": MIN_CROP_PX,
        },
        "ks": list(ks),
        "primary": {
            "metric": f"recall@{PRIMARY_K}",
            "mean_delta": float(boot.observed[primary_col]),
            "ci95": [p_lo, p_hi],
            "share_resamples_delta_le_0": float(boot.share_le_zero[primary_col]),
            "rooms_delta_gt_0": metrics["recall"][str(PRIMARY_K)]["rooms_delta_gt_0"],
            "rooms_delta_eq_0": metrics["recall"][str(PRIMARY_K)]["rooms_delta_eq_0"],
            "rooms_delta_lt_0": metrics["recall"][str(PRIMARY_K)]["rooms_delta_lt_0"],
        },
        "verdict": verdict_sentence,
        "verdict_note": (
            "not evidence of equivalence" if verdict_sentence == "the interval includes zero" else None
        ),
        "bootstrap": {
            "unit": "cluster",
            "paired": True,
            "b": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "level": BOOTSTRAP_LEVEL,
            "quantile": "numpy.quantile default (linear)",
            "n_clusters": boot.n_clusters,
            "cluster_rule": f"connected components of rooms with whole-photo cosine >= {CLUSTER_COSINE}",
            "share_resamples_delta_le_0_note": "not a p-value",
        },
        "metrics": metrics,
        "alongside": {
            "boxes_per_room": {
                "min": int(min(n_boxes)),
                "median": float(np.median(n_boxes)),
                "max": int(max(n_boxes)),
            },
            "rooms_with_fewer_than_m_boxes": int(sum(1 for n in n_boxes if n < TOP_M)),
            "rooms_fallback_to_whole": int(sum(fallback)),
            "seconds_per_room": {
                arm: {part: arm_seconds(arm, part) for part in ("detector", "embedding", "total")}
                for arm in ("whole", "crops")
            },
            "seconds_note": (
                "embedding seconds count only vectors computed in this run (chunk wall time divided by chunk "
                "size); cache hits cost 0. total = detector + embedding; ranking and merge are excluded."
            ),
            "embedding_cache_hits": {"whole_rooms": whole_hits, "crops": crop_hits},
            "crops_embedded": len(crop_items),
            "merge_seconds_total": float(sum(merge_secs)),
            "run_seconds_total": time.perf_counter() - t_start,
        },
        "per_room": per_room_rows,
    }
    for entry in (*(m for name in ("hit", "recall") for m in metrics[name].values()),):
        for key in ("whole", "crops", "mean_delta", "share_resamples_delta_le_0"):
            if not np.isfinite(entry[key]):
                raise CropSearchError(f"non-finite {key} in report")

    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with open(population_path, "x", encoding="utf-8") as f:  # exclusive: never overwrite an attempt
        f.write(json.dumps(population, indent=1, ensure_ascii=False) + "\n")
    with open(report_path, "x", encoding="utf-8") as f:
        f.write(text)
    return CropSearchResult(report_path=report_path, population_path=population_path, report=report)


def format_crop_table(report: dict[str, Any]) -> str:
    """Whole vs crops table with paired mean differences, then the primary line and the verdict."""
    c = report["population"]
    a = report["alongside"]
    lines = [
        f"run={report['run_id']}  embedder={report['models']['embedder']}  "
        f"detector={report['models']['detector']}  rooms={c['rooms']}  gallery={c['gallery_size']}  "
        f"clusters={report['bootstrap']['n_clusters']}",
        f"boxes/room min/median/max={a['boxes_per_room']['min']}/{a['boxes_per_room']['median']:g}/"
        f"{a['boxes_per_room']['max']}  rooms<M boxes={a['rooms_with_fewer_than_m_boxes']}  "
        f"fallback rooms={a['rooms_fallback_to_whole']}",
        f"{'K':>5}  {'Hit whole':>9} {'crops':>7} {'delta':>8}   "
        f"{'Recall whole':>12} {'crops':>7} {'delta':>8}",
    ]
    for k in report["ks"]:
        h = report["metrics"]["hit"][str(k)]
        r = report["metrics"]["recall"][str(k)]
        lines.append(
            f"{k:>5}  {h['whole']:>9.4f} {h['crops']:>7.4f} {h['mean_delta']:>+8.4f}   "
            f"{r['whole']:>12.4f} {r['crops']:>7.4f} {r['mean_delta']:>+8.4f}"
        )
    p = report["primary"]
    lines.append(
        f"primary: mean delta {p['metric']} (crops - whole) = {p['mean_delta']:+.4f}  "
        f"95% cluster-bootstrap interval [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  "
        f"rooms delta>0/=0/<0 = {p['rooms_delta_gt_0']}/{p['rooms_delta_eq_0']}/{p['rooms_delta_lt_0']}  "
        f"resample share delta<=0 = {p['share_resamples_delta_le_0']:.4f} (not a p-value)"
    )
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)
