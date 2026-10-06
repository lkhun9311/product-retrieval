"""DeepFurniture manifest builder (M2 stage 2).

Reads the raw ``furnitures_XXXX.tar.gz`` (one preview per identity) and
``queries_XXXX.tar.gz`` (query crops named ``<furniture_id>_<n>_<scene>.jpg``)
archives as streams, stores every image content-addressed in the shared
``ImageStore`` layout, and writes one manifest line per identity.

The identity of a query is the first field of its file name; that was checked
against the scene annotations (``numberID`` -> ``identityID``) on one scene
archive, it is not documented by the dataset. Splits are by identity with a
deterministic hash, so no identity is ever in two splits. Identities without
queries stay as gallery-only distractors in the split their hash picks.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from product_retrieval.core.ids import canonical_json, sha256_bytes, sha256_file
from product_retrieval.data.store import ImageStore

SOURCE = "deepfurniture"
SPLIT_VERSION = "deepfurniture-split-v1"
MANIFEST_VERSION = "deepfurniture-manifest-v1"
TRAIN_BELOW = 80
VAL_BELOW = 90
SPLITS = ("train", "val", "test")


class DeepFurnitureError(ValueError):
    """Raised when the raw DeepFurniture files are inconsistent or unreadable."""


def assign_split(furniture_id: str, seed: int = 0) -> str:
    """Deterministic identity split: hash bucket < 80 train, < 90 val, else test."""
    digest = sha256_bytes(canonical_json([SPLIT_VERSION, seed, furniture_id]).encode("utf-8"))
    bucket = int(digest[:16], 16) % 100
    if bucket < TRAIN_BELOW:
        return "train"
    if bucket < VAL_BELOW:
        return "val"
    return "test"


def _iter_archive(path: Path) -> Iterator[tuple[str, bytes]]:
    """Stream ``(member_name, bytes)`` of regular files in a tar.gz, in archive order."""
    try:
        with gzip.open(path, "rb") as gz, tarfile.open(fileobj=gz, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    raise DeepFurnitureError(f"{path}: unreadable member {member.name!r}")
                data = f.read()
                if len(data) != member.size or not data:
                    raise DeepFurnitureError(f"{path}: member {member.name!r} is empty or truncated")
                yield Path(member.name).name, data
            # tarfile treats a truncated stream as a clean end; draining the gzip layer
            # makes a cut-off archive raise EOFError instead of passing silently.
            while gz.read(1024 * 1024):
                pass
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise DeepFurnitureError(f"{path}: cannot read archive: {exc}") from exc


def _store_image(store: ImageStore, data: bytes) -> tuple[str, bool]:
    """Write ``data`` content-addressed unless present. Returns ``(sha, written)``."""
    sha = sha256_bytes(data)
    path = store.path(sha)
    if path.is_file():
        return sha, False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return sha, True


def _archives(raw_root: Path, folder: str) -> list[Path]:
    paths = sorted((raw_root / folder).glob("*.tar.gz"))
    if not paths:
        raise DeepFurnitureError(f"no .tar.gz archives under {raw_root / folder}")
    return paths


def _parse_query_name(name: str) -> str:
    stem = name[: -len(".jpg")] if name.endswith(".jpg") else None
    parts = stem.split("_") if stem else []
    if len(parts) != 3 or not all(parts):
        raise DeepFurnitureError(f"query file name {name!r} is not <furniture_id>_<n>_<scene>.jpg")
    return parts[0]


def build_manifest(
    raw_root: str | Path,
    data_root: str | Path,
    out_path: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the DeepFurniture manifest and images; return (and write) the summary.

    Raises ``DeepFurnitureError`` when a query's identity has no preview, a
    preview has no metadata row, the query archives disagree with
    ``query_index.json``, an archive member is unreadable, or two identities
    share an image sha. On a shared sha the summary (with the offending shas) is
    still written next to the manifest, but the manifest is not.
    """
    raw_root, out_path = Path(raw_root), Path(out_path)
    store = ImageStore(data_root, SOURCE)
    meta_dir = raw_root / "metadata"

    furnitures: dict[str, int] = {}
    with open(meta_dir / "furnitures.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                fid = str(row["furniture_id"])
                if fid in furnitures:
                    raise DeepFurnitureError(f"duplicate furniture_id {fid!r} in furnitures.jsonl")
                furnitures[fid] = int(row["category_id"])
    index = json.loads((meta_dir / "query_index.json").read_text(encoding="utf-8"))
    indexed_queries = {q["query_name"] for q in index["queries"]}

    written = 0
    previews: dict[str, str] = {}
    for archive in _archives(raw_root, "furnitures"):
        for name, data in _iter_archive(archive):
            fid = Path(name).stem
            if fid not in furnitures:
                raise DeepFurnitureError(f"{archive}: preview {name!r} has no row in furnitures.jsonl")
            if fid in previews:
                raise DeepFurnitureError(f"{archive}: duplicate preview for furniture_id {fid!r}")
            previews[fid], new = _store_image(store, data)
            written += new
    no_preview = sorted(set(furnitures) - set(previews))
    if no_preview:
        raise DeepFurnitureError(f"{len(no_preview)} identities have no preview, e.g. {no_preview[:5]}")

    queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_queries: set[str] = set()
    for archive in _archives(raw_root, "queries"):
        for name, data in _iter_archive(archive):
            fid = _parse_query_name(name)
            if fid not in previews:
                raise DeepFurnitureError(f"{archive}: query {name!r} has no preview for furniture_id {fid!r}")
            if name in seen_queries:
                raise DeepFurnitureError(f"{archive}: duplicate query file {name!r}")
            seen_queries.add(name)
            sha, new = _store_image(store, data)
            written += new
            queries[fid].append((name, sha))
    if seen_queries != indexed_queries:
        raise DeepFurnitureError(
            f"query archives disagree with query_index.json: {len(seen_queries - indexed_queries)} extra, "
            f"{len(indexed_queries - seen_queries)} missing"
        )

    owners: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for fid in sorted(furnitures):
        query_shas: list[str] = []
        for _, sha in sorted(queries.get(fid, [])):
            if sha not in query_shas:
                query_shas.append(sha)
        for sha in [*query_shas, previews[fid]]:
            owners[sha].add(fid)
        rows.append(
            {
                "source": SOURCE,
                "product_id": fid,
                "split": assign_split(fid, seed),
                "query": query_shas,
                "gallery": [previews[fid]],
            }
        )
    shared = {sha: sorted(ids) for sha, ids in sorted(owners.items()) if len(ids) > 1}

    counts: dict[str, dict[str, int]] = {
        s: {"identities": 0, "identities_with_queries": 0, "queries": 0, "gallery_images": 0} for s in SPLITS
    }
    categories: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    for row in rows:
        c = counts[row["split"]]
        c["identities"] += 1
        c["identities_with_queries"] += bool(row["query"])
        c["queries"] += len(row["query"])
        c["gallery_images"] += len(row["gallery"])
        cat = str(furnitures[row["product_id"]])
        categories[row["split"]][cat] = categories[row["split"]].get(cat, 0) + 1

    archives = [
        {"name": f"{p.parent.name}/{p.name}", "sha256": sha256_file(p)}
        for folder in ("furnitures", "queries")
        for p in _archives(raw_root, folder)
    ]
    summary: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "counts": counts,
        "category_counts": {
            s: dict(sorted(categories[s].items(), key=lambda kv: int(kv[0]))) for s in SPLITS
        },
        "images_written": written,
        "shared_image_shas": shared,
        "archives": archives,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = summary_path_for(out_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if shared:
        raise DeepFurnitureError(
            f"{len(shared)} image shas are shared by several identities (see {summary_path}); "
            "manifest not written"
        )
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(out_path)
    return summary


def summary_path_for(out_path: str | Path) -> Path:
    """Summary JSON path next to the manifest: ``<stem>.summary.json``."""
    out_path = Path(out_path)
    return out_path.with_name(out_path.stem + ".summary.json")
