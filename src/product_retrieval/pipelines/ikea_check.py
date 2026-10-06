"""`pr ikea-check`: real-room-photo check on IKEA Interior.

Contract: docs/contracts/ikea-real-photo-check.md.

Flow: load ``text_data/item_to_room.p`` -> match its image paths to files on disk by
basename -> gallery = existing product images (product id = article number from the
file name), queries = existing room photos, each with the set of existing products the
mapping lists for it -> drop rooms with no existing product (counted) -> embed every
image through the existing embedder + ``EmbeddingCache`` (key: sha256 of the file bytes,
full-image crop hash, model id) -> rank all products per room by cosine -> mean
Hit@K / Recall@K over rooms with a room-level bootstrap -> write
``reports/ikea/<model-slug>.json``.

Evaluation only: nothing here trains or selects a model.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from product_retrieval.core.ids import crop_hash as compute_crop_hash
from product_retrieval.core.ids import sha256_file
from product_retrieval.core.schemas import CropSpec
from product_retrieval.embed.base import Embedder
from product_retrieval.embed.cache import EmbeddingCache
from product_retrieval.embed.fake import FakeEmbedder
from product_retrieval.embed.siglip import SiglipEmbedder
from product_retrieval.eval.multi import IKEA_KS, bootstrap_rooms

IkeaEmbedderName = Literal["siglip", "fake"]

# Same frozen model as configs/baseline.yaml (a test keeps the two in step).
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
CONTRACT_PATH = Path(__file__).resolve().parents[3] / "docs" / "contracts" / "ikea-real-photo-check.md"
CONTRACT_RAW_COMMIT = "6f316e3f"
MAPPING_RELPATH = Path("text_data") / "item_to_room.p"
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 0
_UNIT_NORM_TOL = 1e-3
_EMBED_CHUNK = 64


class IkeaCheckError(RuntimeError):
    """Raised when the raw folder, the mapping, or an embedding cannot be trusted."""


@dataclass(frozen=True)
class IkeaCheckResult:
    report_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def _make_embedder(name: str) -> Embedder:
    if name == "siglip":
        return SiglipEmbedder(model_id=SIGLIP_MODEL_ID)
    if name == "fake":
        return FakeEmbedder()
    raise ValueError(f"unknown embedder {name!r}; expected 'siglip' or 'fake'")


def _load_mapping(raw_root: Path) -> dict[str, list[str]]:
    path = raw_root / MAPPING_RELPATH
    if not path.is_file():
        raise IkeaCheckError(f"mapping not found: {path}")
    # The pickle is a Python 2 artifact of a local, pinned clone; latin1 decodes its str keys.
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")  # noqa: S301
    if not isinstance(raw, dict) or not raw:
        raise IkeaCheckError(f"{path}: expected a non-empty dict item -> rooms")
    mapping: dict[str, list[str]] = {}
    for item, rooms in raw.items():
        if not isinstance(item, str) or not isinstance(rooms, list | tuple):
            raise IkeaCheckError(f"{path}: bad entry {item!r}: expected str -> list of str")
        if not all(isinstance(r, str) for r in rooms):
            raise IkeaCheckError(f"{path}: bad room list for {item!r}")
        mapping[item] = list(rooms)
    return mapping


def _index_files(raw_root: Path) -> dict[str, list[Path]]:
    images_dir = raw_root / "images"
    if not images_dir.is_dir():
        raise IkeaCheckError(f"images folder not found: {images_dir}")
    by_name: dict[str, list[Path]] = {}
    for dirpath, _dirs, files in os.walk(images_dir):
        for name in files:
            by_name.setdefault(name, []).append(Path(dirpath) / name)
    return by_name


def _resolve(
    basename: str,
    by_name: dict[str, list[Path]],
    sha_of: dict[Path, str],
    identical_dups: list[str],
    conflicts: list[str],
) -> Path | None:
    """Return the one file for ``basename``, ``None`` if absent or ambiguous.

    The same basename in two folders is one image when the files are byte-identical (the clone
    files some products under two category folders). When the bytes differ there is no way to
    tell which is the product, so the basename is treated as missing and recorded (contract v1.1).
    """
    paths = sorted(by_name.get(basename, []))
    if not paths:
        return None
    if len(paths) > 1:
        for p in paths:
            if p not in sha_of:
                sha_of[p] = sha256_file(p)
        if len({sha_of[p] for p in paths}) > 1:
            conflicts.append(basename)
            return None
        identical_dups.append(basename)
    return paths[0]


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:  # undecodable file: stop, do not skip silently
        raise IkeaCheckError(f"cannot decode image {path}: {exc}") from exc


def _check_vectors(vectors: np.ndarray, n: int, dim: int, where: str) -> None:
    if vectors.shape != (n, dim):
        raise IkeaCheckError(f"{where}: expected shape {(n, dim)}, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise IkeaCheckError(f"{where}: vectors contain NaN or inf")
    norms = np.linalg.norm(vectors, axis=1)
    if np.abs(norms - 1.0).max(initial=0.0) > _UNIT_NORM_TOL:
        raise IkeaCheckError(f"{where}: vectors are not unit norm")


def _embed_files(
    path_by_sha: dict[str, Path], embedder: Embedder, cache: EmbeddingCache, crop_hash_value: str
) -> dict[str, np.ndarray]:
    shas = list(path_by_sha)
    present, missing = cache.get_many(shas, crop_hash_value, embedder.model_id)
    missing_set = set(missing)
    present_shas = [s for s in shas if s not in missing_set]
    if len(present_shas):
        _check_vectors(present, len(present_shas), embedder.dim, "cached vectors")
    out: dict[str, np.ndarray] = dict(zip(present_shas, present, strict=True))
    for start in range(0, len(missing), _EMBED_CHUNK):
        chunk = missing[start : start + _EMBED_CHUNK]
        vectors = np.asarray(embedder.embed([_load_image(path_by_sha[s]) for s in chunk]))
        _check_vectors(vectors, len(chunk), embedder.dim, "embedder output")
        cache.put_many(chunk, vectors, crop_hash_value, embedder.model_id)
        out.update(zip(chunk, vectors, strict=True))
    return out


def _raw_commit(raw_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(raw_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _stem(basename: str) -> str:
    return os.path.splitext(basename)[0]


@dataclass
class IkeaPopulation:
    """Gallery, rooms and answer sets of the IKEA check (contract v1.1), shared by later checks."""

    mapping: dict[str, list[str]]
    product_path: dict[str, Path]  # product id -> file
    room_path: dict[str, Path]  # room id -> file
    room_answers: dict[str, set[str]]  # room id -> existing products listed for it
    room_listed: dict[str, set[str]]  # every room key in the mapping -> existing products listed
    rooms_missing_file: int
    rooms_dropped: int
    identical_dups: list[str]
    conflicts: list[str]
    sha_of: dict[Path, str]


def load_population(raw_root: Path) -> IkeaPopulation:
    """Resolve the gallery, rooms and answers from the raw folder (contract v1.1)."""
    raw_root = Path(raw_root)
    mapping = _load_mapping(raw_root)
    by_name = _index_files(raw_root)
    sha_of: dict[Path, str] = {}
    identical_dups: list[str] = []
    conflicts: list[str] = []

    # Gallery: one image per product; product id = article number from the file name.
    product_path: dict[str, Path] = {}  # product id -> file
    product_key_ids: dict[str, str] = {}  # mapping key -> product id (existing only)
    id_owner: dict[str, str] = {}
    for key in sorted(mapping):
        basename = os.path.basename(key)
        pid = _stem(basename)
        if pid in id_owner and id_owner[pid] != key:
            raise IkeaCheckError(
                f"product id {pid!r} is produced by two mapping keys: {id_owner[pid]!r}, {key!r}"
            )
        id_owner[pid] = key
        path = _resolve(basename, by_name, sha_of, identical_dups, conflicts)
        if path is not None:
            product_path[pid] = path
            product_key_ids[key] = pid
    if not product_path:
        raise IkeaCheckError("no product image from the mapping exists on disk; empty gallery")

    # Room -> existing products listed for it (duplicates in the mapping collapse).
    room_listed: dict[str, set[str]] = {}
    for key, rooms in mapping.items():
        for room_key in rooms:
            room_listed.setdefault(room_key, set())
            if key in product_key_ids:
                room_listed[room_key].add(product_key_ids[key])

    room_path: dict[str, Path] = {}
    room_answers: dict[str, set[str]] = {}
    rooms_missing_file = 0
    rooms_dropped = 0
    for room_key in sorted(room_listed):
        basename = os.path.basename(room_key)
        path = _resolve(basename, by_name, sha_of, identical_dups, conflicts)
        if path is None:
            rooms_missing_file += 1
            continue
        if not room_listed[room_key]:
            rooms_dropped += 1
            continue
        rid = _stem(basename)
        if rid in room_path:
            raise IkeaCheckError(f"room id {rid!r} is produced by two mapping keys")
        room_path[rid] = path
        room_answers[rid] = room_listed[room_key]
    if not room_path:
        raise IkeaCheckError("no room with an existing photo and an existing product; nothing to evaluate")
    return IkeaPopulation(
        mapping=mapping,
        product_path=product_path,
        room_path=room_path,
        room_answers=room_answers,
        room_listed=room_listed,
        rooms_missing_file=rooms_missing_file,
        rooms_dropped=rooms_dropped,
        identical_dups=identical_dups,
        conflicts=conflicts,
        sha_of=sha_of,
    )


def run_ikea_check(
    raw_root: Path,
    embedder_name: IkeaEmbedderName = "siglip",
    artifacts_root: Path = Path("artifacts"),
    reports_root: Path = Path("reports"),
) -> IkeaCheckResult:
    """Run the check on the IKEA folder at ``raw_root`` and write ``reports/ikea/<model-slug>.json``."""
    raw_root = Path(raw_root)
    if not CONTRACT_PATH.is_file():
        raise IkeaCheckError(f"contract file not found: {CONTRACT_PATH}")
    pop = load_population(raw_root)
    mapping, product_path, room_path, room_answers = (
        pop.mapping,
        pop.product_path,
        pop.room_path,
        pop.room_answers,
    )
    room_listed, sha_of = pop.room_listed, pop.sha_of
    rooms_missing_file, rooms_dropped = pop.rooms_missing_file, pop.rooms_dropped
    identical_dups, conflicts = pop.identical_dups, pop.conflicts

    room_ids = sorted(room_path)
    product_ids = sorted(product_path)

    # Embed: cache key is (sha256 of file bytes, full-image crop hash, model id).
    embedder = _make_embedder(embedder_name)
    cache = EmbeddingCache(Path(artifacts_root) / "embeddings")
    crop_hash_value = compute_crop_hash(CropSpec(kind="full"))
    path_by_sha: dict[str, Path] = {}
    sha_by_path: dict[Path, str] = {}
    for path in [*product_path.values(), *room_path.values()]:
        sha = sha_of.get(path) or sha256_file(path)
        sha_of[path] = sha
        sha_by_path[path] = sha
        path_by_sha.setdefault(sha, path)
    vector_by_sha = _embed_files(path_by_sha, embedder, cache, crop_hash_value)

    gallery = np.stack([vector_by_sha[sha_by_path[product_path[p]]] for p in product_ids]).astype(np.float32)
    queries = np.stack([vector_by_sha[sha_by_path[room_path[r]]] for r in room_ids]).astype(np.float32)
    sims = queries @ gallery.T  # unit vectors -> cosine
    if not np.isfinite(sims).all():
        raise IkeaCheckError("similarity matrix contains NaN or inf")
    # Stable sort on -cosine over product ids in ascending order: ties go to the smaller id.
    order = np.argsort(-sims, axis=1, kind="stable")
    ranked = [[product_ids[j] for j in row] for row in order]
    answers = [room_answers[r] for r in room_ids]

    ks = IKEA_KS
    boot = bootstrap_rooms(ranked, answers, ks, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    metrics = {
        name: {str(k): {"value": boot[name]["point"][k], "ci95": list(boot[name]["ci"][k])} for k in ks}
        for name in ("hit", "recall")
    }

    per_room = np.array([len(a) for a in answers])
    raw_commit = _raw_commit(raw_root)
    report: dict[str, Any] = {
        "contract": "ikea-real-photo-check v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "model_id": embedder.model_id,
        "embedder": embedder_name,
        "raw_commit": raw_commit,
        "raw_commit_matches_contract": bool(raw_commit and raw_commit.startswith(CONTRACT_RAW_COMMIT)),
        "ks": list(ks),
        "bootstrap": {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "level": 0.95, "unit": "room"},
        "counts": {
            "rooms": len(room_ids),
            "rooms_listed_in_mapping": len(room_listed),
            "rooms_missing_file": rooms_missing_file,
            "rooms_dropped_no_existing_product": rooms_dropped,
            "gallery_size": len(product_ids),
            "products_listed_in_mapping": len(mapping),
            "products_missing_file": len(mapping) - len(product_ids),
            "products_per_room": {
                "min": int(per_room.min()),
                "median": float(np.median(per_room)),
                "max": int(per_room.max()),
            },
            "duplicate_basenames_identical_content": sorted(set(identical_dups)),
            "excluded_basenames_conflicting_content": sorted(set(conflicts)),
        },
        "metrics": metrics,
    }
    for name in ("hit", "recall"):
        for k in ks:
            entry = metrics[name][str(k)]
            if not all(math.isfinite(x) for x in (entry["value"], *entry["ci95"])):
                raise IkeaCheckError(f"non-finite {name}@{k} in report")

    out_dir = Path(reports_root) / "ikea"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{_slug(embedder.model_id)}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return IkeaCheckResult(report_path=report_path, report=report)


def format_ikea_table(report: dict[str, Any]) -> str:
    """Render a compact Hit@K / Recall@K table with 95% intervals."""
    c = report["counts"]
    ppr = c["products_per_room"]
    dropped = c["rooms_dropped_no_existing_product"]
    lines = [
        f"model={report['model_id']}  rooms={c['rooms']}  dropped_rooms={dropped}  "
        f"gallery={c['gallery_size']}",
        f"products/room min/median/max={ppr['min']}/{ppr['median']:g}/{ppr['max']}",
        f"{'K':>5}  {'Hit@K':>18}  {'Recall@K':>18}",
    ]
    for k in report["ks"]:
        h = report["metrics"]["hit"][str(k)]
        r = report["metrics"]["recall"][str(k)]
        lines.append(
            f"{k:>5}  {h['value']:>7.4f} [{h['ci95'][0]:.4f},{h['ci95'][1]:.4f}]  "
            f"{r['value']:>7.4f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]"
        )
    return "\n".join(lines)
