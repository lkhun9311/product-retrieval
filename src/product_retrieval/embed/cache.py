"""Disk cache for embedding vectors, keyed by (image sha, crop hash, model id) (D20 section 5).

Layout: ``<root>/<model_id_sanitized>/<crop_hash>/``

* ``vectors.npy`` -- float32 array, shape ``(n, dim)``; row ``i`` is the embedding for
  the sha on line ``i`` of ``ids.jsonl``.
* ``ids.jsonl`` -- one ``{"sha": ...}`` object per line, in the same row order as
  ``vectors.npy``.

Writes are resumable: ``put_many`` reads the current (vectors, ids) pair, merges in
only the shas not already present, and writes the merged pair to temp files in the
same directory before swapping them into place with ``os.replace`` (atomic on a
single filesystem). ``vectors.npy`` is swapped in before ``ids.jsonl``, so a crash
between the two renames can at worst leave a few unindexed extra rows in
``vectors.npy`` -- it can never leave ``ids.jsonl`` pointing at rows that don't
exist, and a sha is never reported present until both files agree on it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize(model_id: str) -> str:
    """Make ``model_id`` (e.g. ``"google/siglip2-base-patch16-224"``) filesystem-safe."""
    return _UNSAFE_CHARS_RE.sub("__", model_id)


class EmbeddingCache:
    """Content-addressed, resumable disk cache for embedding vectors."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, crop_hash: str, model_id: str) -> Path:
        return self.root / _sanitize(model_id) / crop_hash

    def _vectors_path(self, crop_hash: str, model_id: str) -> Path:
        return self._dir(crop_hash, model_id) / "vectors.npy"

    def _ids_path(self, crop_hash: str, model_id: str) -> Path:
        return self._dir(crop_hash, model_id) / "ids.jsonl"

    def _load(self, crop_hash: str, model_id: str) -> tuple[np.ndarray, list[str]]:
        ids_path = self._ids_path(crop_hash, model_id)
        vectors_path = self._vectors_path(crop_hash, model_id)
        if not ids_path.is_file() or not vectors_path.is_file():
            return np.zeros((0, 0), dtype=np.float32), []
        with open(ids_path, encoding="utf-8") as f:
            ids = [json.loads(line)["sha"] for line in f if line.strip()]
        vectors = np.load(vectors_path)
        return vectors, ids

    def get_many(self, shas: list[str], crop_hash: str, model_id: str) -> tuple[np.ndarray, list[str]]:
        """Return ``(vectors, missing_shas)`` for ``shas`` under ``(crop_hash, model_id)``.

        ``vectors`` rows correspond, in order, to the subset of ``shas`` that is
        cached -- i.e. ``[s for s in shas if s not in missing_shas]`` reproduces that
        same order. ``missing_shas`` preserves the order its members appear in ``shas``.
        """
        vectors, ids = self._load(crop_hash, model_id)
        index = {sha: i for i, sha in enumerate(ids)}
        present_rows: list[int] = []
        missing: list[str] = []
        for sha in shas:
            row = index.get(sha)
            if row is None:
                missing.append(sha)
            else:
                present_rows.append(row)

        dim = vectors.shape[1] if vectors.ndim == 2 else 0
        found = vectors[present_rows] if present_rows else np.zeros((0, dim), dtype=np.float32)
        return found.astype(np.float32), missing

    def put_many(self, shas: list[str], vectors: np.ndarray, crop_hash: str, model_id: str) -> None:
        """Cache ``vectors`` for ``shas`` under ``(crop_hash, model_id)``.

        Shas already cached are left untouched (first write wins); this makes the
        call safe to repeat after an interrupted run.
        """
        if len(shas) != len(vectors):
            raise ValueError(f"shas and vectors length mismatch: {len(shas)} != {len(vectors)}")
        if not shas:
            return

        existing_vectors, existing_ids = self._load(crop_hash, model_id)
        existing_set = set(existing_ids)

        new_shas: list[str] = []
        new_rows: list[np.ndarray] = []
        for sha, vec in zip(shas, vectors, strict=True):
            if sha in existing_set:
                continue
            existing_set.add(sha)
            new_shas.append(sha)
            new_rows.append(np.asarray(vec, dtype=np.float32))
        if not new_shas:
            return

        new_vectors = np.stack(new_rows).astype(np.float32)
        merged_vectors = (
            np.concatenate([existing_vectors, new_vectors], axis=0) if existing_vectors.size else new_vectors
        )
        merged_ids = existing_ids + new_shas

        out_dir = self._dir(crop_hash, model_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_write_npy(out_dir, self._vectors_path(crop_hash, model_id), merged_vectors)
        self._atomic_write_jsonl(out_dir, self._ids_path(crop_hash, model_id), merged_ids)

    @staticmethod
    def _atomic_write_npy(out_dir: Path, final_path: Path, array: np.ndarray) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, array)
            os.replace(tmp_name, final_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_jsonl(out_dir: Path, final_path: Path, shas: list[str]) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".jsonl.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for sha in shas:
                    f.write(json.dumps({"sha": sha}) + "\n")
            os.replace(tmp_name, final_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
