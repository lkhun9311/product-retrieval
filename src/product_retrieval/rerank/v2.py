"""C5 reranker v2, contract c5-rerank-v2 (docs/contracts/c5-reranker-v2.md).

A small MLP over the query/candidate image interaction. For candidate product p of query q:
``x = [q * g, |q - g|, standardized v1 features]`` where ``g`` is the vector of p's gallery image with
the largest inner product with ``q`` (both L2-normalised, from the same index and embedding cache as
``pr eval``; see ``pipelines.cand_vectors``) and the v1 features are the ten scalars of
``rerank.v1``. Model: Linear(d, 256) - ReLU - Dropout(0.2) - Linear(256, 1), the output is the logit.

Training is deterministic on CPU. Labelled queries are split 80/20 by ``sha256(seed, query_id)``; the
20% holdout drives early stopping (patience 10, at most 200 epochs, best-holdout-loss weights kept).
If the fit or holdout part lacks a positive or a negative, all labelled rows are trained for a fixed
50 epochs without early stopping and the model JSON records that. The validation split never enters:
``train`` takes only the training rankings, cand-stats and a vector function. The model is saved as
``model.json`` (config, scaler, training log) plus ``weights.pt`` (state_dict, loaded with
``weights_only=True``). Only the top 20 of each ranking are re-ordered, ranks 21+ are unchanged.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from product_retrieval.core.ids import canonical_json, sha256_bytes, sha256_file
from product_retrieval.core.schemas import Label
from product_retrieval.rerank import v1
from product_retrieval.rerank.v0 import RERANK_DEPTH, RerankError, load_labels, load_rankings

CONTRACT = "c5-rerank-v2"
FEATURES = v1.FEATURES
HYPERPARAMS = {
    "hidden": 256,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 10,
    "fixed_epochs": 50,
    "holdout_percent": 20,
}

VectorFn = Callable[[str, str], tuple[np.ndarray, np.ndarray]]


def _hash(*parts: object) -> str:
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))


def input_vector(q: np.ndarray, g: np.ndarray, features: np.ndarray) -> np.ndarray:
    """``[q * g, |q - g|, features]`` (features already standardized); float64."""
    q, g = np.asarray(q, dtype=np.float64), np.asarray(g, dtype=np.float64)
    return np.concatenate([q * g, np.abs(q - g), np.asarray(features, dtype=np.float64)])


def is_holdout(query_id: str, seed: int) -> bool:
    """Deterministic 20% holdout membership from sha256 over (seed, query_id)."""
    digest = _hash(seed, query_id)
    return int(digest[:16], 16) % 100 < HYPERPARAMS["holdout_percent"]


def build_net(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, HYPERPARAMS["hidden"]),
        nn.ReLU(),
        nn.Dropout(HYPERPARAMS["dropout"]),
        nn.Linear(HYPERPARAMS["hidden"], 1),
    )


def _weighted_loss(logit: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    per = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (per * w).sum() / w.sum()


def _both(y: np.ndarray) -> bool:
    return len(y) > 0 and 0 < int(y.sum()) < len(y)


def fit_network(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    holdout: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Train the MLP; ``holdout=None`` means a fixed ``fixed_epochs`` run. Returns (state_dict, log)."""
    hp = HYPERPARAMS
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(seed)
        net = build_net(x_fit.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        gen = torch.Generator().manual_seed(seed)
        xt = torch.as_tensor(x_fit, dtype=torch.float32)
        yt = torch.as_tensor(y_fit, dtype=torch.float32)
        wt = torch.as_tensor(w_fit, dtype=torch.float32)
        if holdout is not None:
            hx, hy, hw = (torch.as_tensor(a, dtype=torch.float32) for a in holdout)
        max_epochs = hp["max_epochs"] if holdout is not None else hp["fixed_epochs"]
        best_loss, best_epoch, best_state = float("inf"), 0, None
        holdout_losses: list[float] = []
        epochs_run, early_stopped = 0, False
        for epoch in range(1, max_epochs + 1):
            net.train()
            perm = torch.randperm(len(xt), generator=gen)
            for start in range(0, len(perm), hp["batch_size"]):
                idx = perm[start : start + hp["batch_size"]]
                opt.zero_grad()
                loss = _weighted_loss(net(xt[idx]).squeeze(1), yt[idx], wt[idx])
                loss.backward()
                opt.step()
            epochs_run = epoch
            if holdout is None:
                continue
            net.eval()
            with torch.no_grad():
                h = float(_weighted_loss(net(hx).squeeze(1), hy, hw))
            holdout_losses.append(h)
            if h < best_loss:
                best_loss, best_epoch, best_state = h, epoch, copy.deepcopy(net.state_dict())
            elif epoch - best_epoch >= hp["patience"]:
                early_stopped = True
                break
        state = best_state if holdout is not None else copy.deepcopy(net.state_dict())
    finally:
        torch.use_deterministic_algorithms(previous)
    log = {
        "mode": "early_stopping" if holdout is not None else "fixed_epochs",
        "epochs_run": epochs_run,
        "best_epoch": best_epoch if holdout is not None else epochs_run,
        "holdout_losses": holdout_losses,
        "early_stopped": early_stopped,
    }
    return {k: v.detach().clone().contiguous() for k, v in state.items()}, log


def _check_index(cand_stats: list[dict], index_id: str) -> None:
    ids = sorted({str(r.get("index_id")) for r in cand_stats})
    if ids != [index_id]:
        raise RerankError(f"cand-stats index_id {ids} differs from the index {index_id!r}")


def _collect(
    labels: list[Label],
    rankings: list[dict],
    rankings_sha256: str,
    cand_stats: list[dict],
    vector_fn: VectorFn,
):
    """Per-label (query_id, qg block, v1 feature row, y, w), sorted by (query_id, product_id)."""
    by_query = {row["query_id"]: row for row in rankings}
    if len(by_query) != len(rankings):
        raise RerankError("duplicate query_id in rankings")
    stats_by_q = v1.check_alignment(rankings, rankings_sha256, cand_stats)
    seen: set[tuple[str, str]] = set()
    feats_cache: dict[str, np.ndarray] = {}
    out = []
    for lb in sorted(labels, key=lambda lb: (lb.query_id, lb.product_id)):
        pair = (lb.query_id, lb.product_id)
        if pair in seen:
            raise RerankError(f"duplicate label for {pair!r}")
        seen.add(pair)
        row = by_query.get(lb.query_id)
        if row is None:
            raise RerankError(f"label query_id {lb.query_id!r} not found in rankings")
        top_ids = row["top_k_product_ids"][:RERANK_DEPTH]
        if lb.product_id not in top_ids:
            raise RerankError(f"label {pair!r} is not in the top {RERANK_DEPTH} of its ranking")
        if lb.query_id not in feats_cache:
            feats_cache[lb.query_id] = v1.features_v1(row, stats_by_q[lb.query_id])
        feats = feats_cache[lb.query_id][top_ids.index(lb.product_id)]
        q, g = vector_fn(lb.query_id, lb.product_id)
        qg = np.concatenate([np.asarray(q) * np.asarray(g), np.abs(np.asarray(q) - np.asarray(g))])
        out.append((lb.query_id, qg, feats, 1 if lb.kind == "pos" else 0, lb.weight))
    return out


def _scaler(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = feats.mean(axis=0)
    scale = feats.std(axis=0)
    return mean, np.where(scale > 0, scale, 1.0)


def train(
    labels: list[Label],
    rankings: list[dict],
    rankings_sha256: str,
    cand_stats: list[dict],
    cand_stats_sha256: str,
    label_version: str,
    vector_fn: VectorFn,
    index_id: str,
    seed: int = 0,
) -> dict:
    """Fit v2 on labelled (query, product) rows of the training split; returns the model dict.

    The dict carries the in-memory ``state_dict``; ``save_model`` writes it to ``weights.pt``.
    """
    _check_index(cand_stats, index_id)
    rows = _collect(labels, rankings, rankings_sha256, cand_stats, vector_fn)
    y = np.array([r[3] for r in rows])
    w = np.array([r[4] for r in rows], dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        raise RerankError(f"need both classes to train (pos={n_pos}, neg={n_neg})")
    qg = np.vstack([r[1] for r in rows])
    feats = np.vstack([r[2] for r in rows])
    hold = np.array([is_holdout(r[0], seed) for r in rows])

    fallback = None
    if not (hold.any() and (~hold).any()):
        fallback = "holdout or fit part is empty"
    elif not _both(y[hold]) or not _both(y[~hold]):
        fallback = "holdout or fit part lacks positives or negatives"
    fit_mask = np.ones(len(y), dtype=bool) if fallback else ~hold
    mean, scale = _scaler(feats[fit_mask])

    def build(mask: np.ndarray) -> np.ndarray:
        return np.hstack([qg[mask], (feats[mask] - mean) / scale])

    holdout = None if fallback else (build(hold), y[hold], w[hold])
    state, log = fit_network(build(fit_mask), y[fit_mask], w[fit_mask], holdout, seed)
    log["fallback_reason"] = fallback
    log["n_fit"] = int(fit_mask.sum())
    log["n_holdout"] = 0 if fallback else int(hold.sum())
    version = _hash(
        CONTRACT,
        label_version,
        rankings_sha256,
        cand_stats_sha256,
        index_id,
        HYPERPARAMS,
        seed,
        torch.__version__,
    )[:12]
    return {
        "contract": CONTRACT,
        "reranker_version": version,
        "features": list(FEATURES),
        "hyperparams": dict(HYPERPARAMS),
        "seed": seed,
        "torch_version": torch.__version__,
        "input_dim": int(qg.shape[1] + feats.shape[1]),
        "scaler_mean": [float(m) for m in mean],
        "scaler_scale": [float(s) for s in scale],
        "label_version": label_version,
        "rankings_sha256": rankings_sha256,
        "cand_stats_sha256": cand_stats_sha256,
        "index_id": index_id,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": len(y),
        "training": log,
        "state_dict": state,
    }


def _dump(model: dict) -> str:
    plain = {k: v for k, v in model.items() if k != "state_dict"}
    return json.dumps(plain, indent=2, sort_keys=True) + "\n"


def _same_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def save_model(model: dict, root: str | Path) -> Path:
    """Write ``root/<version>/{model.json,weights.pt}``; an identical existing pair is a no-op."""
    folder = Path(root) / model["reranker_version"]
    path, wpath = folder / "model.json", folder / "weights.pt"
    text = _dump(model)
    if path.exists() or wpath.exists():
        if not (path.exists() and wpath.exists()):
            raise RerankError(f"{folder} exists but is incomplete (model.json / weights.pt)")
        old = torch.load(wpath, weights_only=True, map_location="cpu")
        if path.read_text(encoding="utf-8") != text or not _same_state(old, model["state_dict"]):
            raise RerankError(f"{folder} already exists with different content")
        return path
    folder.mkdir(parents=True, exist_ok=True)
    tmp = wpath.with_name("weights.pt.tmp")
    try:
        torch.save(model["state_dict"], tmp)
        os.replace(tmp, wpath)
    finally:
        if tmp.exists():
            tmp.unlink()
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def load_model(path: str | Path) -> dict:
    """Read ``model.json`` and the sibling ``weights.pt`` (``weights_only=True``)."""
    path = Path(path)
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RerankError(f"{path}: invalid model json: {exc}") from exc
    if model.get("contract") != CONTRACT or model.get("features") != list(FEATURES):
        raise RerankError(f"{path}: not a {CONTRACT} model with features {list(FEATURES)}")
    wpath = path.with_name("weights.pt")
    if not wpath.is_file():
        raise RerankError(f"{wpath} is missing")
    model["state_dict"] = torch.load(wpath, weights_only=True, map_location="cpu")
    return model


def _feature_fn(rankings: list[dict], rankings_sha256: str, cand_stats: list[dict]):
    stats_by_q = v1.check_alignment(rankings, rankings_sha256, cand_stats)
    return lambda row: v1.features_v1(row, stats_by_q[row["query_id"]])


def rerank_rows(
    model: dict,
    rows: list[dict],
    cand_stats: list[dict],
    rankings_sha256: str,
    vector_fn: VectorFn,
) -> list[dict]:
    """New rows with the top 20 re-ordered by MLP logit (stable ties), ranks 21+ unchanged.

    The model's ``index_id`` is the training gallery; reranking another split uses that split's own
    index, so only the cand-stats rows' single index_id is checked here (the vector source checks it
    against the located index).
    """
    ids = sorted({str(r.get("index_id")) for r in cand_stats})
    if len(ids) != 1:
        raise RerankError(f"cand-stats rows must carry exactly one index_id, found {ids}")
    feature_fn = _feature_fn(rows, rankings_sha256, cand_stats)
    mean, scale = np.asarray(model["scaler_mean"]), np.asarray(model["scaler_scale"])
    net = build_net(model["input_dim"])
    net.load_state_dict(model["state_dict"])
    net.eval()
    out: list[dict] = []
    for row in rows:
        feats = (feature_fn(row) - mean) / scale
        n = len(feats)
        ids, scores = list(row["top_k_product_ids"]), list(row["scores"])
        if n:
            x = np.vstack([input_vector(*vector_fn(row["query_id"], ids[i]), feats[i]) for i in range(n)])
            with torch.no_grad():
                z = net(torch.as_tensor(x, dtype=torch.float32)).squeeze(1).double().numpy()
        else:
            z = np.zeros(0)
        order = np.argsort(-z, kind="stable")
        new = dict(row)
        new["top_k_product_ids"] = [ids[i] for i in order] + ids[n:]
        new["scores"] = [scores[i] for i in order] + scores[n:]
        new["rerank_scores"] = [float(z[i]) for i in order]
        new["reranker_version"] = model["reranker_version"]
        out.append(new)
    return out


def run_train(
    labels_path: str | Path,
    rankings_path: str | Path,
    cand_stats_path: str | Path,
    out_root: str | Path,
    vector_fn: VectorFn,
    index_id: str,
    seed: int = 0,
) -> dict:
    labels = load_labels(labels_path)
    versions = sorted({lb.label_version for lb in labels})
    if len(versions) != 1:
        raise RerankError(f"labels must carry exactly one label_version, found {versions}")
    model = train(
        labels,
        load_rankings(rankings_path),
        sha256_file(rankings_path),
        v1.load_cand_stats(cand_stats_path),
        sha256_file(cand_stats_path),
        versions[0],
        vector_fn,
        index_id,
        seed,
    )
    path = save_model(model, out_root)
    return {**{k: v for k, v in model.items() if k != "state_dict"}, "path": str(path)}


def run_rerank(
    model_path: str | Path,
    rankings_path: str | Path,
    cand_stats_path: str | Path,
    out_path: str | Path,
    vector_fn: VectorFn,
) -> int:
    rankings = load_rankings(rankings_path)
    rows = rerank_rows(
        load_model(model_path),
        rankings,
        v1.load_cand_stats(cand_stats_path),
        sha256_file(rankings_path),
        vector_fn,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)
