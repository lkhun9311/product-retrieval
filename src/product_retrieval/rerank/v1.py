"""C5 reranker v1, contract c5-rerank-v1 (docs/contracts/c5-reranker-v1.md).

v0's five rankings-file features plus five image-level candidate features read from the
``<rankings stem>.cand_stats.jsonl`` written by ``pr cand-stats``: ``log_n_images`` = ln(1 + n_images),
``mean_sim``, ``gap_second`` = max_sim - second_sim, ``std_sim`` and ``top1_sim``. Training, inference
and the model JSON are v0's (shared code in ``rerank.v0``); the version hash additionally covers the
cand-stats file sha256. Rankings and cand-stats must agree on ``rankings_sha256``, the query id set
and the candidate order, otherwise ``RerankError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from product_retrieval.core.ids import sha256_file
from product_retrieval.core.schemas import Label
from product_retrieval.rerank import v0
from product_retrieval.rerank.v0 import (
    RERANK_DEPTH,
    RerankError,
    features_for_row,
    fit_model,
    load_labels,
    load_rankings,
    rerank_rows,
    save_model,
)

CONTRACT = "c5-rerank-v1"
FEATURES = v0.FEATURES + ("log_n_images", "mean_sim", "gap_second", "std_sim", "top1_sim")


def load_cand_stats(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("query_id", "rankings_sha256", "stats"):
                if key not in row:
                    raise RerankError(f"{path}:{lineno}: missing field {key!r}")
            if row["query_id"] in seen:
                raise RerankError(f"{path}:{lineno}: duplicate query_id {row['query_id']!r}")
            seen.add(row["query_id"])
            rows.append(row)
    return rows


def check_alignment(
    rankings: list[dict], rankings_sha256: str, cand_stats: list[dict]
) -> dict[str, list[dict]]:
    """Validate the pairing and return ``{query_id: stats list}``."""
    wrong = sorted({r["rankings_sha256"] for r in cand_stats if r["rankings_sha256"] != rankings_sha256})
    if wrong:
        raise RerankError(
            f"cand-stats rankings_sha256 {wrong[0][:12]} does not match "
            f"the rankings file {rankings_sha256[:12]}"
        )
    stats_by_q = {r["query_id"]: r["stats"] for r in cand_stats}
    if set(stats_by_q) != {r["query_id"] for r in rankings}:
        raise RerankError("cand-stats query_ids differ from rankings query_ids")
    for row in rankings:
        ids = row["top_k_product_ids"][:RERANK_DEPTH]
        if [s["product_id"] for s in stats_by_q[row["query_id"]]] != ids:
            raise RerankError(
                f"query {row['query_id']!r}: cand-stats candidates differ from the rankings order"
            )
    return stats_by_q


def features_v1(row: dict, stats: list[dict]) -> np.ndarray:
    """(n, 10) matrix: v0 features, then the five cand-stats features, in rank order."""
    base = features_for_row(row)
    if len(stats) != len(base):
        raise RerankError(f"query {row.get('query_id')!r}: cand-stats length differs from the candidates")
    extra = np.array(
        [
            [
                np.log1p(s["n_images"]),
                s["mean_sim"],
                s["max_sim"] - s["second_sim"],
                s["std_sim"],
                s["top1_sim"],
            ]
            for s in stats
        ],
        dtype=np.float64,
    ).reshape(len(stats), 5)
    return np.hstack([base, extra])


def _feature_fn(rankings: list[dict], rankings_sha256: str, cand_stats: list[dict]):
    stats_by_q = check_alignment(rankings, rankings_sha256, cand_stats)
    return lambda row: features_v1(row, stats_by_q[row["query_id"]])


def train(
    labels: list[Label],
    rankings: list[dict],
    rankings_sha256: str,
    cand_stats: list[dict],
    cand_stats_sha256: str,
    label_version: str,
) -> dict:
    fn = _feature_fn(rankings, rankings_sha256, cand_stats)
    model = fit_model(
        labels, rankings, rankings_sha256, label_version, fn, CONTRACT, FEATURES, (cand_stats_sha256,)
    )
    model["cand_stats_sha256"] = cand_stats_sha256
    return model


def load_model(path: str | Path) -> dict:
    return v0.load_model(path, CONTRACT, FEATURES)


def run_train(
    labels_path: str | Path, rankings_path: str | Path, cand_stats_path: str | Path, out_root: str | Path
) -> dict:
    labels = load_labels(labels_path)
    versions = sorted({lb.label_version for lb in labels})
    if len(versions) != 1:
        raise RerankError(f"labels must carry exactly one label_version, found {versions}")
    model = train(
        labels,
        load_rankings(rankings_path),
        sha256_file(rankings_path),
        load_cand_stats(cand_stats_path),
        sha256_file(cand_stats_path),
        versions[0],
    )
    path = save_model(model, out_root)
    return {**model, "path": str(path)}


def run_rerank(
    model_path: str | Path, rankings_path: str | Path, cand_stats_path: str | Path, out_path: str | Path
) -> int:
    rankings = load_rankings(rankings_path)
    fn = _feature_fn(rankings, sha256_file(rankings_path), load_cand_stats(cand_stats_path))
    rows = rerank_rows(load_model(model_path), rankings, fn)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)
