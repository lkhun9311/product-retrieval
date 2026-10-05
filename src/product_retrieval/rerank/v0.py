"""C5 reranker v0, contract c5-rerank-v0 (D20, docs/contracts/c5-reranker-v0.md).

A logistic regression over five features read only from a rankings file (``top_k_product_ids``
and ``scores``; ``truth_product_id`` is never read). Only the top ``RERANK_DEPTH`` (20) candidates
of each query are re-ordered, ranks 21+ stay as they are.

Features for candidate i (1-based rank r, score s_i) with s_1..s_20 the query's top-20 scores:
``score`` = s_i, ``gap_top1`` = s_i - s_1, ``gap_next`` = s_i - s_{i+1} (rank 20 uses s_21, 0 if
absent), ``zscore`` = (s_i - mean) / population std (0 if std is 0), ``log_rank`` = ln(r).

Training standardizes the features over the labelled rows, then fits an L2 logistic regression
(C=1.0, lbfgs, max_iter=1000) weighted by ``Label.weight``. The model is stored as JSON (no pickle)
and inference is plain numpy, so it does not need scikit-learn. ``reranker_version`` is the first
12 hex digits of sha256 over (contract, label_version, rankings sha256, features, hyperparameters,
scikit-learn version).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from product_retrieval.core.ids import canonical_json, sha256_bytes, sha256_file
from product_retrieval.core.schemas import Label

CONTRACT = "c5-rerank-v0"
FEATURES = ("score", "gap_top1", "gap_next", "zscore", "log_rank")
RERANK_DEPTH = 20
HYPERPARAMS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}


class RerankError(ValueError):
    """Raised for invalid rankings/labels input or a conflicting model artifact."""


def _hash(*parts: object) -> str:
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))


def features_for_row(row: dict) -> np.ndarray:
    """Feature matrix, shape (min(20, n_candidates), 5), in rank order. Truth fields are not read."""
    ids = row["top_k_product_ids"]
    scores = row.get("scores")
    if scores is None or len(scores) != len(ids):
        raise RerankError(
            f"query {row.get('query_id')!r}: scores missing or not aligned with top_k_product_ids"
        )
    s = np.asarray(scores, dtype=np.float64)
    n = min(RERANK_DEPTH, len(s))
    if n == 0:
        return np.zeros((0, len(FEATURES)))
    top = s[:n]
    nxt = np.zeros(n)
    nxt[: n - 1] = top[:-1] - top[1:]
    if len(s) > n:
        nxt[n - 1] = top[n - 1] - s[n]
    std = top.std()
    z = (top - top.mean()) / std if std > 0 else np.zeros(n)
    log_rank = np.log(np.arange(1, n + 1, dtype=np.float64))
    return np.column_stack([top, top - top[0], nxt, z, log_rank])


def load_rankings(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("query_id", "top_k_product_ids", "scores"):
                if key not in row:
                    raise RerankError(f"{path}:{lineno}: missing field {key!r}")
            if row["query_id"] in seen:
                raise RerankError(f"{path}:{lineno}: duplicate query_id {row['query_id']!r}")
            seen.add(row["query_id"])
            rows.append(row)
    return rows


def load_labels(path: str | Path) -> list[Label]:
    labels: list[Label] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                labels.append(Label.model_validate_json(line))
            except ValueError as exc:
                raise RerankError(f"{path}:{lineno}: invalid label: {exc}") from exc
    return labels


def train(labels: list[Label], rankings: list[dict], rankings_sha256: str, label_version: str) -> dict:
    """Fit the reranker on labelled (query, product) rows and return the model dict."""
    by_query = {row["query_id"]: row for row in rankings}
    if len(by_query) != len(rankings):
        raise RerankError("duplicate query_id in rankings")
    seen_pairs: set[tuple[str, str]] = set()
    x_rows: list[np.ndarray] = []
    y: list[int] = []
    w: list[float] = []
    cache: dict[str, np.ndarray] = {}
    for lb in sorted(labels, key=lambda lb: (lb.query_id, lb.product_id)):
        pair = (lb.query_id, lb.product_id)
        if pair in seen_pairs:
            raise RerankError(f"duplicate label for {pair!r}")
        seen_pairs.add(pair)
        row = by_query.get(lb.query_id)
        if row is None:
            raise RerankError(f"label query_id {lb.query_id!r} not found in rankings")
        top_ids = row["top_k_product_ids"][:RERANK_DEPTH]
        if lb.product_id not in top_ids:
            raise RerankError(f"label {pair!r} is not in the top {RERANK_DEPTH} of its ranking")
        if lb.query_id not in cache:
            cache[lb.query_id] = features_for_row(row)
        x_rows.append(cache[lb.query_id][top_ids.index(lb.product_id)])
        y.append(1 if lb.kind == "pos" else 0)
        w.append(lb.weight)
    n_pos, n_neg = sum(y), len(y) - sum(y)
    if n_pos == 0 or n_neg == 0:
        raise RerankError(f"need both classes to train (pos={n_pos}, neg={n_neg})")

    x = np.vstack(x_rows)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(**HYPERPARAMS)
    clf.fit(scaler.transform(x), np.asarray(y), sample_weight=np.asarray(w))
    version = _hash(
        CONTRACT, label_version, rankings_sha256, list(FEATURES), HYPERPARAMS, sklearn.__version__
    )[:12]
    return {
        "contract": CONTRACT,
        "reranker_version": version,
        "features": list(FEATURES),
        "hyperparams": dict(HYPERPARAMS),
        "sklearn_version": sklearn.__version__,
        "coef": [float(c) for c in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": [float(m) for m in scaler.mean_],
        "scaler_scale": [float(s) for s in scaler.scale_],
        "label_version": label_version,
        "rankings_sha256": rankings_sha256,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": len(y),
    }


def _dump(model: dict) -> str:
    return json.dumps(model, indent=2, sort_keys=True) + "\n"


def save_model(model: dict, root: str | Path) -> Path:
    """Write ``root/<version>/model.json``; identical content is a no-op, different content an error."""
    path = Path(root) / model["reranker_version"] / "model.json"
    text = _dump(model)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RerankError(f"{path} already exists with different content")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_model(path: str | Path) -> dict:
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RerankError(f"{path}: invalid model json: {exc}") from exc
    if model.get("contract") != CONTRACT or model.get("features") != list(FEATURES):
        raise RerankError(f"{path}: not a {CONTRACT} model with features {list(FEATURES)}")
    return model


def logits(model: dict, x: np.ndarray) -> np.ndarray:
    """Plain-numpy decision function: ((x - mean) / scale) @ coef + intercept."""
    mean = np.asarray(model["scaler_mean"])
    scale = np.asarray(model["scaler_scale"])
    return ((x - mean) / scale) @ np.asarray(model["coef"]) + model["intercept"]


def rerank_rows(model: dict, rows: list[dict]) -> list[dict]:
    """Return new rows with the top 20 re-ordered by logit (ties keep the original order)."""
    out: list[dict] = []
    for row in rows:
        x = features_for_row(row)
        n = len(x)
        z = logits(model, x) if n else np.zeros(0)
        order = np.argsort(-z, kind="stable")
        ids, scores = list(row["top_k_product_ids"]), list(row["scores"])
        new = dict(row)
        new["top_k_product_ids"] = [ids[i] for i in order] + ids[n:]
        new["scores"] = [scores[i] for i in order] + scores[n:]
        new["rerank_scores"] = [float(z[i]) for i in order]
        new["reranker_version"] = model["reranker_version"]
        out.append(new)
    return out


def run_train(labels_path: str | Path, rankings_path: str | Path, out_root: str | Path) -> dict:
    """Train from files and save the model; returns the model dict plus ``path``."""
    labels = load_labels(labels_path)
    versions = sorted({lb.label_version for lb in labels})
    if len(versions) != 1:
        raise RerankError(f"labels must carry exactly one label_version, found {versions}")
    model = train(labels, load_rankings(rankings_path), sha256_file(rankings_path), versions[0])
    path = save_model(model, out_root)
    return {**model, "path": str(path)}


def run_rerank(model_path: str | Path, rankings_path: str | Path, out_path: str | Path) -> int:
    """Re-order rankings with a saved model; writes JSONL, returns the row count."""
    rows = rerank_rows(load_model(model_path), load_rankings(rankings_path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)
