"""`pr deepfurniture-crop`: Whole vs Detector vs GT-box search on DeepFurniture scenes.

Contract: docs/contracts/crop-before-search-deepfurniture.md. It copies every detector, crop and merge
setting from docs/contracts/crop-before-search.md, so the detector arm here goes through the same
functions as ``pipelines.crop_search`` (``detect_crops``, ``rank_vector``, ``merge_round_robin``).

Flow, in this order:

1. assert the sha256 of ``crop-before-search.md``; record the code commit and the executable detector
   settings (checked against that contract);
2. population: scene annotations from the scene archive, eligibility rules, asserted counts; val gallery;
   write ``population.json`` (lists and their sha256);
3. before any model: scene image bytes (kept in memory), image sizes, every annotated box validated,
   GT boxes built;
4. pinned SigLIP2 revision: gallery and whole-scene embeddings (computed in this run, caches live in
   the run dir), clusters from whole-scene vectors;
5. Detector arm and GT-box arm: crops -> embeddings -> per-crop rankings -> round-robin merge;
6. metrics, paired cluster bootstrap, per-category recall, timings -> ``report.json`` + ``summary.md``.

Everything is written under ``<out_root>/<run_id>/``; an existing run dir is never reused.
Evaluation only: nothing here trains or selects a model.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from product_retrieval.core.ids import canonical_json, sha256_bytes, sha256_file
from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.schemas import CropSpec
from product_retrieval.crop import boxes as crop_boxes
from product_retrieval.crop.boxes import Crop, exif_orientation, load_exif_rgb
from product_retrieval.crop.owl import PROMPTS
from product_retrieval.data import deepfurniture_scenes as scenes_mod
from product_retrieval.data.deepfurniture import MANIFEST_VERSION
from product_retrieval.data.deepfurniture_scenes import (
    EXPECTED_ANSWERS,
    EXPECTED_GALLERY,
    EXPECTED_SCENES,
)
from product_retrieval.data.manifest import load_manifest
from product_retrieval.data.store import ImageStore
from product_retrieval.embed.base import Embedder
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.embed.siglip import SiglipEmbedder
from product_retrieval.eval import crop_compare
from product_retrieval.eval.merge import CLUSTER_COSINE, cluster_rooms, merge_round_robin
from product_retrieval.eval.multi import IKEA_KS, hit_at_k, recall_at_k
from product_retrieval.pipelines import crop_search as cs
from product_retrieval.pipelines.ikea_check import SIGLIP_MODEL_ID, _embed_files

CONTRACT_NAME = "crop-before-search-deepfurniture v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DF_CONTRACT_PATH = REPO_ROOT / "docs" / "contracts" / "crop-before-search-deepfurniture.md"
CROP_CONTRACT_SHA256 = "10e64eee0f82d4d5024b5f1cafbf097fd0d1c7bad60210c5abed485dfadce8cc"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SCENE_ARCHIVE = Path("data") / "scenes" / "scenes_0000.tar.gz"
QUERY_INDEX = Path("data") / "metadata" / "query_index.json"
MANIFEST_NAME = "manifest_deepfurniture.jsonl"
SOURCE = "deepfurniture"
FROZEN_SETTINGS: dict[str, float | int] = {
    "score_threshold": 0.10,
    "nms_iou": 0.5,
    "top_m": 10,
    "expand_frac": 0.10,
    "min_crop_px": 16,
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

EmbedderName = Literal["siglip", "fake"]


class DeepFurnitureCropError(RuntimeError):
    """Raised when a contract assertion, the population, a box or an embedding cannot be trusted."""


@dataclass(frozen=True)
class DeepFurnitureCropResult:
    run_dir: Path
    report_path: Path
    summary_path: Path
    population_path: Path
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- contract, code and settings guards


def assert_crop_contract_hash(path: Path = cs.CONTRACT_PATH, expected: str = CROP_CONTRACT_SHA256) -> str:
    """The IKEA crop contract must be byte-identical to the one the IKEA run recorded."""
    if not Path(path).is_file():
        raise DeepFurnitureCropError(f"crop contract not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DeepFurnitureCropError(f"crop contract {path} has sha256 {actual}, expected {expected}")
    return actual


def code_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit of the code being run, whether the tree differs from it, and a hash of that difference."""

    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout if proc.returncode == 0 else None

    commit = (git("rev-parse", "HEAD") or "").strip()
    status = git("status", "--porcelain")
    diff = git("diff", "HEAD")
    return {
        "commit": commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None,
        "dirty": None if status is None else bool(status.strip()),
        "status_porcelain": None if status is None else status.splitlines(),
        "diff_head_sha256": None if diff is None else sha256_bytes(diff.encode("utf-8")),
    }


def contract_detector_facts(text: str) -> dict[str, Any]:
    """Model id, revision and prompts as written in ``crop-before-search.md``."""
    model = re.search(r"`(google/owlv2-[^`]+)` at revision `([0-9a-f]{40})`", text)
    prompts = re.search(r"Prompts, fixed: (.+?)\. ", text)
    if not model or not prompts:
        raise DeepFurnitureCropError("crop contract: cannot find the detector model/revision or prompt list")
    return {
        "model_id": model.group(1),
        "revision": model.group(2),
        "prompts": tuple(p.strip() for p in prompts.group(1).split(",")),
    }


def executable_detector_settings(det: Any, detector_name: str) -> dict[str, Any]:
    """The settings the code will actually use (function defaults and constants), plus the detector's own."""

    def default(fn: Any, name: str) -> Any:
        return inspect.signature(fn).parameters[name].default

    settings: dict[str, Any] = {
        "score_threshold": default(crop_boxes.select_detections, "threshold"),
        "nms_iou": default(crop_boxes.select_detections, "iou_threshold"),
        "top_m": default(crop_boxes.select_detections, "top_m"),
        "expand_frac": default(crop_boxes.expand_and_round, "frac"),
        "min_crop_px": default(crop_boxes.build_crops, "min_px"),
        "constants": {
            "SCORE_THRESHOLD": crop_boxes.SCORE_THRESHOLD,
            "NMS_IOU": crop_boxes.NMS_IOU,
            "TOP_M": crop_boxes.TOP_M,
            "EXPAND_FRAC": crop_boxes.EXPAND_FRAC,
            "MIN_CROP_PX": crop_boxes.MIN_CROP_PX,
        },
        "detector_name": detector_name,
        "detector_model_id": det.model_id,
        "detector_revision": det.revision,
        "merge": "eval.merge.merge_round_robin (rank depth, crop order; duplicates consume their turn)",
        "m_for_gt_arm": crop_boxes.TOP_M,
    }
    if detector_name == "owl":
        settings["prompts"] = list(PROMPTS)
        settings["n_prompts"] = len(PROMPTS)
        settings["prompts_sha256"] = sha256_bytes(canonical_json(list(PROMPTS)).encode("utf-8"))
    return settings


def verify_detector_settings(settings: dict[str, Any], crop_contract_text: str) -> None:
    """Stop when the executable settings differ from the frozen numbers or the IKEA contract's text."""
    constant_of = {
        "score_threshold": "SCORE_THRESHOLD",
        "nms_iou": "NMS_IOU",
        "top_m": "TOP_M",
        "expand_frac": "EXPAND_FRAC",
        "min_crop_px": "MIN_CROP_PX",
    }
    for key, want in FROZEN_SETTINGS.items():
        if settings[key] != want:
            raise DeepFurnitureCropError(
                f"detector setting {key}={settings[key]!r}, frozen value is {want!r}"
            )
        if settings["constants"][constant_of[key]] != want:
            raise DeepFurnitureCropError(
                f"detector constant {constant_of[key]}={settings['constants'][constant_of[key]]!r}, "
                f"frozen value is {want!r}"
            )
    if settings["detector_name"] == "owl":
        facts = contract_detector_facts(crop_contract_text)
        if (settings["detector_model_id"], settings["detector_revision"]) != (
            facts["model_id"],
            facts["revision"],
        ):
            raise DeepFurnitureCropError(
                f"detector {settings['detector_model_id']}@{settings['detector_revision']} "
                f"differs from the contract {facts['model_id']}@{facts['revision']}"
            )
        if tuple(settings["prompts"]) != facts["prompts"] or settings["n_prompts"] != 24:
            raise DeepFurnitureCropError("detector prompts differ from the 24 prompts of the crop contract")


# ---------------------------------------------------------------------------------- small helpers


def _make_embedder(name: str, revision: str) -> Embedder:
    if name == "siglip":
        return SiglipEmbedder(model_id=SIGLIP_MODEL_ID, revision=revision)
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"unknown embedder {name!r}; expected 'siglip' or 'fake'")


def _load(data: bytes) -> Image.Image:
    return load_exif_rgb(BytesIO(data))


def _spec_hash(box: tuple[int, int, int, int]) -> str:
    return cs._crop_spec_hash(box)


def _write_exclusive(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def _dump(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False) + "\n"


def _stats_or_none(values: list[float] | list[int]) -> dict[str, float] | None:
    return cs._stats(values) if values else None


# ---------------------------------------------------------------------------------------- the run


def run_deepfurniture_crop(
    raw_root: Path,
    data_root: Path,
    run_id: str,
    out_root: Path = Path("reports/crop"),
    manifest_path: Path | None = None,
    embedder_name: EmbedderName = "siglip",
    detector: cs.CropDetectorName = "owl",
    siglip_revision: str = SIGLIP_REVISION,
    population_only: bool = False,
    expected_scenes: int = EXPECTED_SCENES,
    expected_answers: int = EXPECTED_ANSWERS,
    expected_gallery: int = EXPECTED_GALLERY,
    crop_contract_path: Path = cs.CONTRACT_PATH,
    crop_contract_sha256: str = CROP_CONTRACT_SHA256,
    require_commit: bool = True,
    repo_root: Path = REPO_ROOT,
) -> DeepFurnitureCropResult:
    """Run the three arms and write ``<out_root>/<run_id>/{population.json,report.json,summary.md}``."""
    t_start = time.perf_counter()
    # 1. Contract, code and settings, before anything else.
    crop_contract_actual = assert_crop_contract_hash(crop_contract_path, crop_contract_sha256)
    if not _RUN_ID_RE.fullmatch(run_id):
        raise DeepFurnitureCropError(f"run id {run_id!r} must match {_RUN_ID_RE.pattern}")
    if not DF_CONTRACT_PATH.is_file():
        raise DeepFurnitureCropError(f"contract file not found: {DF_CONTRACT_PATH}")
    code = code_state(repo_root)
    if require_commit and code["commit"] is None:
        raise DeepFurnitureCropError(f"cannot read the code commit from git in {repo_root}")

    raw_root, data_root = Path(raw_root), Path(data_root)
    manifest_path = Path(manifest_path) if manifest_path else raw_root / MANIFEST_NAME
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # an attempt is never overwritten or resumed

    # 2. Population from annotations only.
    archive = raw_root / SCENE_ARCHIVE
    rows = load_manifest(manifest_path)
    split_of = {r.product_id: r.split for r in rows}
    if len(split_of) != len(rows):
        raise DeepFurnitureCropError("manifest has duplicate product ids")
    query_index_path = raw_root / QUERY_INDEX
    query_index = json.loads(query_index_path.read_text(encoding="utf-8"))
    annotations = scenes_mod.read_annotations(archive)
    query_sources = scenes_mod.val_query_source_scenes(query_index, split_of)
    pop = scenes_mod.select_population(
        annotations, split_of, query_sources, expected_scenes, expected_answers
    )
    gallery_rows = scenes_mod.select_gallery(rows, expected_gallery)
    product_ids = [pid for pid, _ in gallery_rows]  # ascending: ties in a ranking go to the smaller id
    if not set(a for ans in pop.answers.values() for a in ans) <= set(product_ids):
        raise DeepFurnitureCropError("an answer identity is not in the val gallery")
    scene_ids = pop.scenes
    answers = [pop.answers[s] for s in scene_ids]
    population = {
        "scenes": scene_ids,
        "answers": {s: pop.answers[s] for s in scene_ids},
        "gallery": [{"product_id": pid, "image_sha256": sha} for pid, sha in gallery_rows],
    }
    pop_hashes = {k: cs._list_sha(v) for k, v in population.items()}
    pop_hashes["all"] = cs._list_sha(population)
    population_path = run_dir / "population.json"
    _write_exclusive(population_path, _dump(population, indent=1))
    population_report = {
        "manifest_version": MANIFEST_VERSION,
        "scenes": len(scene_ids),
        "answers": pop.n_answers,
        "gallery_size": len(product_ids),
        "expected": {"scenes": expected_scenes, "answers": expected_answers, "gallery": expected_gallery},
        "counts_asserted": True,  # a mismatch raises before this point
        "scenes_in_archive": len(annotations),
        "excluded_scenes": pop.excluded,
        "excluded_val_query_source_scenes": pop.excluded_query_source_scenes,
        "answers_per_scene": {
            "max": max(len(a) for a in answers),
            "distribution": dict(sorted(Counter(len(a) for a in answers).items())),
        },
        "sha256": pop_hashes,
        "list_file": str(population_path),
        "inputs_sha256": {
            "scene_archive": sha256_file(archive),
            "manifest": sha256_file(manifest_path),
            "query_index": sha256_file(query_index_path),
        },
    }
    base_report: dict[str, Any] = {
        "run_id": run_id,
        "contract": CONTRACT_NAME,
        "contract_sha256": sha256_file(DF_CONTRACT_PATH),
        "crop_contract": {
            "path": str(crop_contract_path),
            "sha256": crop_contract_actual,
            "expected_sha256": crop_contract_sha256,
            "asserted_equal": True,
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "code": code,
        "population": population_report,
    }
    if population_only:
        base_report["population_only"] = True
        report_path = run_dir / "report.json"
        _write_exclusive(report_path, _dump(base_report))
        return DeepFurnitureCropResult(run_dir, report_path, report_path, population_path, base_report)

    # 3. Before any model: images, sizes, boxes (all of them), GT boxes.
    images = scenes_mod.read_scene_images(archive, scene_ids)
    image_sha = {s: sha256_bytes(images[s]) for s in scene_ids}
    sizes: dict[str, tuple[int, int]] = {}
    rotated: list[str] = []
    for s in scene_ids:
        with Image.open(BytesIO(images[s])) as im:
            sizes[s] = im.size
        if exif_orientation(BytesIO(images[s])) != 1:
            rotated.append(s)
    if rotated:
        raise DeepFurnitureCropError(
            f"{len(rotated)} scene images carry an EXIF rotation (box frame ambiguous), e.g. {rotated[:3]}"
        )
    gt_by_scene = {
        s: scenes_mod.gt_crops(annotations[s], *sizes[s]) for s in scene_ids
    }  # validates all boxes
    store = ImageStore(data_root, SOURCE)
    gallery_path_by_sha: dict[str, Path] = {}
    for pid, sha in gallery_rows:
        p = store.path(sha)
        if not p.is_file():
            raise DeepFurnitureCropError(f"gallery image of {pid!r} is missing: {p}")
        if sha256_file(p) != sha:
            raise DeepFurnitureCropError(f"gallery image of {pid!r} does not hash to {sha}")
        gallery_path_by_sha[sha] = p

    no_area = {
        s: sorted(i.number_id for i in annotations[s].instances if i.area_ratio is None) for s in scene_ids
    }
    no_area = {s: v for s, v in no_area.items() if v}
    gt_missing_first_m: list[dict[str, Any]] = []
    gt_missing_after_min: list[dict[str, Any]] = []
    for s in scene_ids:
        first = [i.number_id for i in gt_by_scene[s].first_m]
        kept = [c.index for c in gt_by_scene[s].crops]
        for ident in scenes_mod.answers_without_box(annotations[s], pop.answers[s], first):
            gt_missing_first_m.append({"scene": s, "identity": ident})
        for ident in scenes_mod.answers_without_box(annotations[s], pop.answers[s], kept):
            gt_missing_after_min.append({"scene": s, "identity": ident})
    size_counts = Counter(f"{w}x{h}" for w, h in sizes.values())
    images_report = {
        "native_size_used": True,
        "sizes": dict(sorted(size_counts.items())),
        "max_width": max(w for w, _ in sizes.values()),
        "max_height": max(h for _, h in sizes.values()),
        "exif_rotated_scenes": 0,
        "all_annotated_boxes_valid": True,  # a bad box raises SceneBoxError above
        "instances_total": sum(len(annotations[s].instances) for s in scene_ids),
    }
    gt_report = {
        "instances_without_area_ratio": sum(len(v) for v in no_area.values()),
        "scenes_with_instances_without_area_ratio": len(no_area),
        "answers_without_box_in_first_10": len(gt_missing_first_m),
        "answers_without_box_in_first_10_list": gt_missing_first_m,
        "answers_without_box_after_min_size": len(gt_missing_after_min),
        "order": "areaRatio desc; missing areaRatio last by numberID asc; ties by numberID asc",
    }

    # 4. Models and the executable settings actually used.
    embedder = _make_embedder(embedder_name, siglip_revision)
    model_commit = cs._commit_hash_of(embedder)
    if embedder_name == "siglip" and model_commit is not None and model_commit != siglip_revision:
        raise DeepFurnitureCropError(f"SigLIP2 resolved to {model_commit}, requested {siglip_revision}")
    det = cs._make_detector(detector)
    det_settings = executable_detector_settings(det, detector)
    verify_detector_settings(det_settings, Path(crop_contract_path).read_text(encoding="utf-8"))

    emb_root = run_dir / "embeddings"  # per run: no vector from another run is read
    full_hash = compute_crop_hash(CropSpec(kind="full"))
    gallery_cache = EmbeddingCache(emb_root / "common")
    t0 = time.perf_counter()
    vec_by_sha = _embed_files(gallery_path_by_sha, embedder, gallery_cache, full_hash)
    gallery_seconds = time.perf_counter() - t0
    gallery = np.stack([vec_by_sha[sha] for _, sha in gallery_rows]).astype(np.float32)

    whole_items = [cs._Item(image_sha[s], full_hash, lambda b=images[s]: _load(b)) for s in scene_ids]
    whole_list, whole_secs, whole_hits = cs._embed_items(whole_items, embedder, gallery_cache)
    whole = np.stack(whole_list).astype(np.float32)
    clusters = cluster_rooms(whole)  # fixed before any crop ranking

    # 5a. Detector arm: the same functions as the IKEA path.
    det_crops: list[list[Crop]] = []
    det_secs: list[float] = []
    for s in scene_ids:
        crops, seconds = cs.detect_crops(det, _load(images[s]), f"scene {s!r}")
        det_crops.append(crops)
        det_secs.append(seconds)
    gt_crops = [gt_by_scene[s].crops for s in scene_ids]

    def arm_vectors(
        arm: str, crops_by_scene: list[list[Crop]]
    ) -> tuple[list[list[np.ndarray]], list[float], list[bool], int]:
        cache = EmbeddingCache(emb_root / arm)
        items: list[cs._Item] = []
        owner: list[int] = []
        for si, s in enumerate(scene_ids):
            for c in crops_by_scene[si]:
                items.append(
                    cs._Item(
                        image_sha[s],
                        _spec_hash(c.box),
                        lambda b=images[s], box=c.box: _load(b).crop(box),
                    )
                )
                owner.append(si)
        vecs, secs, _hits = cs._embed_items(items, embedder, cache)
        per_scene: list[list[np.ndarray]] = [[] for _ in scene_ids]
        emb_secs = [0.0] * len(scene_ids)
        for si, v, sec in zip(owner, vecs, secs, strict=True):
            per_scene[si].append(v)
            emb_secs[si] += sec
        fallback = [len(c) == 0 for c in crops_by_scene]
        for si in range(len(scene_ids)):
            if fallback[si]:  # the whole image is the only crop
                per_scene[si] = [whole[si]]
                emb_secs[si] = whole_secs[si]
        return per_scene, emb_secs, fallback, len(items)

    det_vecs, det_emb_secs, det_fallback, det_n_embedded = arm_vectors("detector", det_crops)
    gt_vecs, gt_emb_secs, gt_fallback, gt_n_embedded = arm_vectors("gt", gt_crops)

    # 5b. Rankings. Every arm is cut at the same K.
    ks = IKEA_KS
    k_max = max(ks)
    ranked: dict[str, list[list[str]]] = {"whole": [], "detector": [], "gt": []}
    merge_secs = {"detector": 0.0, "gt": 0.0}
    for si in range(len(scene_ids)):
        ranked["whole"].append(cs.rank_vector(gallery, product_ids, whole[si])[:k_max])
        for arm, vecs in (("detector", det_vecs), ("gt", gt_vecs)):
            t0 = time.perf_counter()
            per_crop = [cs.rank_vector(gallery, product_ids, v) for v in vecs[si]]
            ranked[arm].append(merge_round_robin(per_crop, k_max))
            merge_secs[arm] += time.perf_counter() - t0
    assert_same_population(scene_ids, answers, product_ids, ranked)

    # 6. Metrics.
    hit = {arm: hit_at_k(r, answers, ks) for arm, r in ranked.items()}
    rec = {arm: recall_at_k(r, answers, ks) for arm, r in ranked.items()}
    comparison = crop_compare.compare_arms(hit, rec, clusters, ks)
    metrics = {
        name: {str(k): {arm: float(table[arm][:, j].mean()) for arm in ranked} for j, k in enumerate(ks)}
        for name, table in (("hit", hit), ("recall", rec))
    }

    triples = [
        (si, ident, cat)
        for si, s in enumerate(scene_ids)
        for ident, cat in scenes_mod.answer_categories(annotations[s], pop.answers[s])
    ]
    categories = {arm: crop_compare.pooled_category_recall(ranked[arm], triples) for arm in ranked}

    n_boxes = {"detector": [len(c) for c in det_crops], "gt": [len(c) for c in gt_crops]}
    fallbacks = {"detector": det_fallback, "gt": gt_fallback}
    secs = {
        "whole": {"detector": [0.0] * len(scene_ids), "embedding": whole_secs},
        "detector": {"detector": det_secs, "embedding": det_emb_secs},
        "gt": {"detector": [0.0] * len(scene_ids), "embedding": gt_emb_secs},
    }
    per_scene_rows = []
    for si, s in enumerate(scene_ids):
        per_scene_rows.append(
            {
                "scene": s,
                "cluster": int(clusters[si]),
                "image_size": list(sizes[s]),
                "n_answers": len(answers[si]),
                "answers": answers[si],
                "n_instances": len(annotations[s].instances),
                "arms": {
                    arm: {
                        **(
                            {
                                "n_crops": n_boxes[arm][si],
                                "fallback_to_whole": fallbacks[arm][si],
                                "crops": [
                                    {"box": list(c.box), "score": c.score, "index": c.index}
                                    for c in (det_crops if arm == "detector" else gt_crops)[si]
                                ],
                            }
                            if arm != "whole"
                            else {}
                        ),
                        "hit": {str(k): float(hit[arm][si, j]) for j, k in enumerate(ks)},
                        "recall": {str(k): float(rec[arm][si, j]) for j, k in enumerate(ks)},
                        "seconds": {
                            "detector": secs[arm]["detector"][si],
                            "embedding": secs[arm]["embedding"][si],
                            "total": secs[arm]["detector"][si] + secs[arm]["embedding"][si],
                        },
                    }
                    for arm in ("whole", "detector", "gt")
                },
            }
        )

    report: dict[str, Any] = {
        **base_report,
        "models": {
            "embedder": embedder.model_id,
            "embedder_name": embedder_name,
            "embedder_revision_requested": siglip_revision if embedder_name == "siglip" else None,
            "embedder_model_commit_resolved": model_commit,
            "processor_revision_requested": siglip_revision if embedder_name == "siglip" else None,
            "revision_note": (
                "model and processor are both loaded with revision=<requested>; the resolved commit is read "
                "from the model config (null = not exposed, then only the request is known)"
            ),
            "detector": det.model_id,
            "detector_revision": det.revision,
            "versions": cs._versions(),
        },
        "detector_settings": det_settings,
        "images": images_report,
        "gt_box_arm": gt_report,
        "instances_without_area_ratio_by_scene": no_area,
        "ks": list(ks),
        "primary": comparison["primary"],
        "verdict": comparison["verdict"],
        "verdict_note": comparison["verdict_note"],
        "bootstrap": {
            "unit": "cluster",
            "paired": True,
            "b": comparison["b"],
            "seed": comparison["seed"],
            "level": comparison["level"],
            "quantile": "numpy.quantile default (linear)",
            "n_clusters": comparison["n_clusters"],
            "min_clusters": comparison["min_clusters"],
            "inferential": comparison["inferential"],
            "cluster_rule": f"connected components of scenes with whole-scene cosine >= {CLUSTER_COSINE}",
            "share_resamples_delta_le_0_note": "not a p-value",
        },
        "metrics": metrics,
        "comparisons": comparison["comparisons"],
        "per_category_recall_at_10": {
            "unit": (
                "distinct (scene, val identity, category); an identity with two categories in a scene "
                "counts once per category"
            ),
            "descriptive_only": True,
            "arms": categories,
        },
        "alongside": {
            "boxes_per_scene": {arm: _stats_or_none(v) for arm, v in n_boxes.items()},
            "scenes_fallback_to_whole": {arm: int(sum(f)) for arm, f in fallbacks.items()},
            "crops_embedded": {"detector": det_n_embedded, "gt": gt_n_embedded},
            "embedding_cache_hits_whole_scenes": whole_hits,
            "seconds_per_scene": {
                arm: {
                    part: cs._stats([row["arms"][arm]["seconds"][part] for row in per_scene_rows])
                    for part in ("detector", "embedding", "total")
                }
                for arm in ("whole", "detector", "gt")
            },
            "seconds_note": (
                "embedding seconds are chunk wall time divided by chunk size; ranking and merge are excluded"
            ),
            "gallery_embedding_seconds": gallery_seconds,
            "merge_seconds_total": merge_secs,
            "run_seconds_total": time.perf_counter() - t_start,
        },
        "per_scene": per_scene_rows,
    }
    report_path = run_dir / "report.json"
    summary_path = run_dir / "summary.md"
    _write_exclusive(report_path, _dump(report))  # allow_nan=False: a non-finite number raises here
    _write_exclusive(summary_path, format_summary(report))
    return DeepFurnitureCropResult(run_dir, report_path, summary_path, population_path, report)


def assert_same_population(
    scene_ids: list[str], answers: list[list[str]], product_ids: list[str], ranked: dict[str, list[list[str]]]
) -> None:
    """All arms cover the same scenes, return min(K, gallery) products each, all inside the gallery."""
    if set(ranked) != {"whole", "detector", "gt"}:
        raise DeepFurnitureCropError(f"expected arms whole/detector/gt, got {sorted(ranked)}")
    if len(scene_ids) != len(answers):
        raise DeepFurnitureCropError("scenes and answers differ in length")
    gallery = set(product_ids)
    want = min(max(IKEA_KS), len(product_ids))
    for arm, lists in ranked.items():
        if len(lists) != len(scene_ids):
            raise DeepFurnitureCropError(f"arm {arm!r} does not cover the same scenes")
        for s, r in zip(scene_ids, lists, strict=True):
            if len(r) != want:
                raise DeepFurnitureCropError(
                    f"scene {s!r}, arm {arm!r}: did not return min(K, gallery) products"
                )
            if not set(r) <= gallery:
                raise DeepFurnitureCropError(
                    f"scene {s!r}, arm {arm!r}: returned a product outside the gallery"
                )


# ------------------------------------------------------------------------------------- summary


def format_summary(report: dict[str, Any]) -> str:
    """Markdown summary: population, primary line, verdict, secondary deltas, categories, timings."""
    p, pop, b = report["primary"], report["population"], report["bootstrap"]
    ci = (
        "withheld (fewer than the minimum clusters)"
        if p["ci95"] is None
        else f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    )
    lines = [
        f"# DeepFurniture crop run {report['run_id']}",
        "",
        f"Contract: {report['contract']} (sha256 {report['contract_sha256']}); crop contract sha256 "
        f"{report['crop_contract']['sha256']} (asserted).",
        f"Code commit: {report['code']['commit']} (dirty: {report['code']['dirty']}). "
        "Offline experiment on rendered scenes; not a real-photo or real-user result.",
        "",
        f"Scenes {pop['scenes']}, answers {pop['answers']}, gallery {pop['gallery_size']}, "
        f"clusters {b['n_clusters']} (minimum {b['min_clusters']}).",
        f"Embedder {report['models']['embedder']} "
        f"revision {report['models']['embedder_revision_requested']}; "
        f"detector {report['models']['detector']} revision {report['models']['detector_revision']}.",
        "",
        "## Primary",
        "",
        f"Delta {p['metric']} (Detector - Whole) = {p['mean_delta']:+.4f}, "
        f"95% cluster-bootstrap interval {ci}; "
        f"scenes delta>0/=0/<0 = {p['scenes_delta_gt_0']}/{p['scenes_delta_eq_0']}/{p['scenes_delta_lt_0']}.",
        "",
        f"Verdict: {report['verdict']}" + (f" ({report['verdict_note']})" if report["verdict_note"] else ""),
        "",
        "## Mean metrics by arm",
        "",
        "| K | Hit whole | Hit detector | Hit GT | Recall whole | Recall detector | Recall GT |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in report["ks"]:
        h, r = report["metrics"]["hit"][str(k)], report["metrics"]["recall"][str(k)]
        lines.append(
            f"| {k} | {h['whole']:.4f} | {h['detector']:.4f} | {h['gt']:.4f} | "
            f"{r['whole']:.4f} | {r['detector']:.4f} | {r['gt']:.4f} |"
        )
    lines += [
        "",
        "## Secondary deltas (not used for the verdict)",
        "",
        "| Comparison | Metric | K | Mean delta | 95% interval |",
        "|---|---|---:|---:|---|",
    ]
    for comp in report["comparisons"].values():
        for metric in ("recall", "hit"):
            for k in report["ks"]:
                e = comp[metric][str(k)]
                ci_k = "withheld" if e["ci95"] is None else f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
                lines.append(f"| {comp['delta']} | {metric} | {k} | {e['mean_delta']:+.4f} | {ci_k} |")
    lines += [
        "",
        "Delta (GT box - Detector) is not a measure of detector error alone: the GT arm differs in box "
        "geometry, selection and order. The GT arm is a reference, never a deployable result.",
        "",
        "## Pooled Recall@10 per category (descriptive)",
        "",
        "| Category | Denominator | Whole | Detector | GT box |",
        "|---|---:|---:|---:|---:|",
    ]
    arms = report["per_category_recall_at_10"]["arms"]
    for cat, cell in arms["whole"].items():
        lines.append(
            f"| {cat} | {cell['denominator']} | {cell['recall']:.4f} | "
            f"{arms['detector'][cat]['recall']:.4f} | {arms['gt'][cat]['recall']:.4f} |"
        )
    a, g, im = report["alongside"], report["gt_box_arm"], report["images"]
    lines += [
        "",
        "## Alongside",
        "",
        f"- Image sizes: {im['sizes']}.",
        f"- Boxes per scene: detector {a['boxes_per_scene']['detector']}, GT {a['boxes_per_scene']['gt']}.",
        f"- Scenes falling back to the whole image: {a['scenes_fallback_to_whole']}.",
        f"- Instances without areaRatio: {g['instances_without_area_ratio']} in "
        f"{g['scenes_with_instances_without_area_ratio']} scenes; answers with no box in the first 10: "
        f"{g['answers_without_box_in_first_10']} (after the minimum size rule: "
        f"{g['answers_without_box_after_min_size']}).",
        "- Mean seconds per scene (detector / embedding): "
        + "; ".join(
            f"{arm} {v['detector']['mean']:.2f} / {v['embedding']['mean']:.2f}"
            for arm, v in a["seconds_per_scene"].items()
        )
        + ".",
        f"- Total run seconds: {a['run_seconds_total']:.0f}.",
    ]
    return "\n".join(lines) + "\n"


def format_console(report: dict[str, Any]) -> str:
    """Short console block: population line, primary line, verdict."""
    pop = report["population"]
    if report.get("population_only"):
        return (
            f"run={report['run_id']}  population only: scenes={pop['scenes']}  answers={pop['answers']}  "
            f"gallery={pop['gallery_size']}  excluded={pop['excluded_scenes']}"
        )
    p = report["primary"]
    ci = "withheld" if p["ci95"] is None else f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    return "\n".join(
        [
            f"run={report['run_id']}  scenes={pop['scenes']}  answers={pop['answers']}  "
            f"gallery={pop['gallery_size']}  clusters={report['bootstrap']['n_clusters']}",
            f"primary: mean delta {p['metric']} (detector - whole) = {p['mean_delta']:+.4f}  "
            f"95% interval {ci}  scenes delta>0/=0/<0 = "
            f"{p['scenes_delta_gt_0']}/{p['scenes_delta_eq_0']}/{p['scenes_delta_lt_0']}",
            f"verdict: {report['verdict']}",
        ]
    )
