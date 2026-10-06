"""D20 section 7 CLI.

`pr build-index | eval | cand-stats | simulate | labels | train-rerank | rerank | curve | gate | serve`
(plus `curve-summary`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from product_retrieval import __version__
from product_retrieval.core.config import DEFAULT_DATA_ROOT, load_config
from product_retrieval.data.checks import check_manifest
from product_retrieval.data.deepfurniture import DeepFurnitureError, build_manifest
from product_retrieval.data.manifest import load_manifest
from product_retrieval.data.store import ImageStore
from product_retrieval.eval.retrieval import DEFAULT_KS
from product_retrieval.feedback.labels import LabelBuildError, run_labels
from product_retrieval.feedback.simulate import SimulationError, run_simulate
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.cand_stats import CandStatsError, run_cand_stats
from product_retrieval.pipelines.cand_vectors import (
    VectorSourceError,
    cand_stats_index_id,
    load_candidate_vectors,
)
from product_retrieval.pipelines.crop_search import CropSearchError, format_crop_table, run_crop_search
from product_retrieval.pipelines.curve import (
    EXPECTED_POPULATION,
    CurveError,
    plan_keys,
    run_curve,
    summarize,
)
from product_retrieval.pipelines.evaluate import IndexNotFoundError, format_report_table, run_eval
from product_retrieval.pipelines.ikea_check import IkeaCheckError, format_ikea_table, run_ikea_check
from product_retrieval.pipelines.selection import TestSplitAccessError
from product_retrieval.rerank import v0 as rerank_v0
from product_retrieval.rerank import v1 as rerank_v1
from product_retrieval.rerank import v2 as rerank_v2
from product_retrieval.rerank.v0 import RerankError

app = typer.Typer(add_completion=False, help="product-retrieval command line interface")


def _not_implemented(name: str) -> None:
    typer.echo(f"{name}: not implemented yet", err=True)
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the installed product-retrieval version."""
    typer.echo(__version__)


@app.command("data-check")
def data_check(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, help="LRVS-style manifest JSONL path"),
    ],
    source: Annotated[str, typer.Option("--source", help="source name under data-root, e.g. lrvs")],
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="root containing <source>/images/<sha[:2]>/<sha>.img"),
    ] = DEFAULT_DATA_ROOT,
) -> None:
    """Check a manifest for missing images and cross-split product leakage (C1)."""
    rows = load_manifest(manifest)
    store = ImageStore(data_root, source)
    report = check_manifest(rows, store)
    typer.echo(report.model_dump_json(indent=2))
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("prepare-deepfurniture")
def prepare_deepfurniture(
    raw: Annotated[
        Path,
        typer.Option(
            "--raw",
            exists=True,
            file_okay=False,
            help="DeepFurniture data dir (metadata/, furnitures/, queries/)",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="manifest JSONL to write")],
    data_root: Annotated[
        Path,
        typer.Option("--data-root", help="root receiving deepfurniture/images/<sha[:2]>/<sha>.img"),
    ] = DEFAULT_DATA_ROOT,
    seed: Annotated[int, typer.Option("--seed", help="identity split seed")] = 0,
) -> None:
    """Stream the raw archives into the image store and write the manifest and summary."""
    try:
        summary = build_manifest(raw, data_root, out, seed=seed)
    except DeepFurnitureError as exc:
        typer.echo(f"prepare-deepfurniture: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "archives"}, indent=2))


@app.command("build-index")
def build_index(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (core.config)"),
    ],
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="manifest split to index: train, val (default) or test "
            "(train and val are open; test needs --final)",
        ),
    ] = "val",
    limit_products: Annotated[
        int | None,
        typer.Option("--limit-products", help="index only the first N products (sorted by product_id)"),
    ] = None,
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder to use: siglip (default) or fake")
    ] = "siglip",
    final: Annotated[
        bool, typer.Option("--final", help="allow split=test; logs the access (D20 section 9)")
    ] = False,
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts")
    ] = Path("artifacts"),
    reports_root: Annotated[Path, typer.Option("--reports-root", help="root for run-summary reports")] = Path(
        "reports"
    ),
) -> None:
    """Embed the gallery, build a flat FAISS index, and write a run summary (C2)."""
    if embedder not in ("siglip", "fake"):
        typer.echo(f"build-index: unknown --embedder {embedder!r}; expected 'siglip' or 'fake'", err=True)
        raise typer.Exit(code=2)

    experiment_config = load_config(config)
    try:
        result = run_build_index(
            experiment_config,
            split=split,
            final=final,
            limit_products=limit_products,
            embedder_name=embedder,  # type: ignore[arg-type]
            artifacts_root=artifacts_root,
            reports_root=reports_root,
        )
    except TestSplitAccessError as exc:
        typer.echo(f"build-index: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except NotImplementedError as exc:
        typer.echo(f"build-index: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(json.dumps(result.summary, indent=2))


@app.command("ikea-check")
def ikea_check_cmd(
    raw: Annotated[
        Path,
        typer.Option(
            "--raw", exists=True, file_okay=False, help="IKEA Interior clone (has text_data/, images/)"
        ),
    ],
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder to use: siglip (default) or fake")
    ] = "siglip",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for the embedding cache")
    ] = Path("artifacts"),
    reports_root: Annotated[
        Path, typer.Option("--reports-root", help="root for reports/ikea/<model-slug>.json")
    ] = Path("reports"),
) -> None:
    """Real-room-photo check on IKEA Interior: room-level Hit@K / Recall@K (see docs/contracts)."""
    if embedder not in ("siglip", "fake"):
        typer.echo(f"ikea-check: unknown --embedder {embedder!r}; expected 'siglip' or 'fake'", err=True)
        raise typer.Exit(code=2)
    try:
        result = run_ikea_check(
            raw,
            embedder_name=embedder,  # type: ignore[arg-type]
            artifacts_root=artifacts_root,
            reports_root=reports_root,
        )
    except IkeaCheckError as exc:
        typer.echo(f"ikea-check: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_ikea_table(result.report))
    typer.echo(f"report: {result.report_path}")


@app.command("crop-search")
def crop_search_cmd(
    raw: Annotated[
        Path,
        typer.Option(
            "--raw", exists=True, file_okay=False, help="IKEA Interior clone (has text_data/, images/)"
        ),
    ],
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder to use: siglip (default) or fake")
    ] = "siglip",
    detector: Annotated[
        str, typer.Option("--detector", help="detector to use: owl (default) or fake")
    ] = "owl",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for the embedding cache")
    ] = Path("artifacts"),
    reports_root: Annotated[
        Path, typer.Option("--reports-root", help="root for reports/crop/<run-id>.json")
    ] = Path("reports"),
) -> None:
    """Whole room photo vs detector -> crop -> merge on IKEA rooms (see docs/contracts)."""
    if embedder not in ("siglip", "fake"):
        typer.echo(f"crop-search: unknown --embedder {embedder!r}; expected 'siglip' or 'fake'", err=True)
        raise typer.Exit(code=2)
    if detector not in ("owl", "fake"):
        typer.echo(f"crop-search: unknown --detector {detector!r}; expected 'owl' or 'fake'", err=True)
        raise typer.Exit(code=2)
    try:
        result = run_crop_search(
            raw,
            embedder_name=embedder,
            detector=detector,  # type: ignore[arg-type]
            artifacts_root=artifacts_root,
            reports_root=reports_root,
        )
    except (IkeaCheckError, CropSearchError) as exc:
        typer.echo(f"crop-search: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_crop_table(result.report))
    typer.echo(f"report: {result.report_path}")


@app.command("eval")
def eval_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (core.config)"),
    ],
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="manifest split to evaluate: train, val (default) or test "
            "(train and val are open; test needs --final)",
        ),
    ] = "val",
    limit_products: Annotated[
        int | None,
        typer.Option("--limit-products", help="evaluate only the first N products (sorted by product_id)"),
    ] = None,
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder to use: siglip (default) or fake")
    ] = "siglip",
    final: Annotated[
        bool, typer.Option("--final", help="allow split=test; logs the access (D20 section 9)")
    ] = False,
    k: Annotated[
        list[int] | None,
        typer.Option("--k", help="R@K values to report (repeatable); default 1 5 10 100"),
    ] = None,
    bootstrap: Annotated[int, typer.Option("--bootstrap", help="number of bootstrap replicates")] = 1000,
    seed: Annotated[int, typer.Option("--seed", help="bootstrap RNG seed")] = 0,
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts")
    ] = Path("artifacts"),
    reports_root: Annotated[
        Path, typer.Option("--reports-root", help="root for eval report/ranking output")
    ] = Path("reports"),
) -> None:
    """Score a previously built index on a split and write an eval report (C2)."""
    if embedder not in ("siglip", "fake"):
        typer.echo(f"eval: unknown --embedder {embedder!r}; expected 'siglip' or 'fake'", err=True)
        raise typer.Exit(code=2)

    ks = tuple(k) if k else DEFAULT_KS

    experiment_config = load_config(config)
    try:
        result = run_eval(
            experiment_config,
            split=split,
            final=final,
            limit_products=limit_products,
            embedder_name=embedder,  # type: ignore[arg-type]
            ks=ks,
            b=bootstrap,
            seed=seed,
            artifacts_root=artifacts_root,
            reports_root=reports_root,
        )
    except TestSplitAccessError as exc:
        typer.echo(f"eval: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except IndexNotFoundError as exc:
        typer.echo(f"eval: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except NotImplementedError as exc:
        typer.echo(f"eval: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(format_report_table(result.report))


@app.command("cand-stats")
def cand_stats_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (core.config)"),
    ],
    split: Annotated[str, typer.Option("--split", help="split the rankings were made on: train or val")],
    rankings: Annotated[
        Path,
        typer.Option("--rankings", exists=True, dir_okay=False, help="rankings JSONL written by `pr eval`"),
    ],
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder to use: siglip (default) or fake")
    ] = "siglip",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts")
    ] = Path("artifacts"),
    limit_products: Annotated[
        int | None,
        typer.Option("--limit-products", help="the N given to `pr eval --limit-products` for these rankings"),
    ] = None,
) -> None:
    """Write <rankings stem>.cand_stats.jsonl, image-level candidate stats (C5, c5-rerank-v1)."""
    if split not in ("train", "val"):
        typer.echo(f"cand-stats: --split must be 'train' or 'val', got {split!r}", err=True)
        raise typer.Exit(code=1)
    if embedder not in ("siglip", "fake"):
        typer.echo(f"cand-stats: unknown --embedder {embedder!r}; expected 'siglip' or 'fake'", err=True)
        raise typer.Exit(code=2)
    experiment_config = load_config(config)
    try:
        result = run_cand_stats(
            experiment_config,
            split=split,
            rankings_path=rankings,
            embedder_name=embedder,  # type: ignore[arg-type]
            artifacts_root=artifacts_root,
            limit_products=limit_products,
        )
    except (CandStatsError, IndexNotFoundError, TestSplitAccessError) as exc:
        typer.echo(f"cand-stats: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {result.n_queries} rows to {result.out_path}")


@app.command("simulate")
def simulate(
    rankings: Annotated[
        Path,
        typer.Option("--rankings", exists=True, dir_okay=False, help="rankings JSONL written by `pr eval`"),
    ],
    budget: Annotated[int, typer.Option("--budget", help="number of judgments (= events)")],
    seed: Annotated[int, typer.Option("--seed", help="sampling RNG seed")],
    out: Annotated[Path, typer.Option("--out", help="output FeedbackEvent JSONL path")],
    policy: Annotated[str, typer.Option("--policy", help="random | stratified")] = "stratified",
    exposed_k: Annotated[int, typer.Option("--exposed-k", help="candidates exposed per query")] = 20,
    noise: Annotated[float, typer.Option("--noise", help="probability of flipping an answer")] = 0.0,
) -> None:
    """Run a simulated feedback policy (C4, contract c4-v3); writes <out>.summary.json too."""
    try:
        events = run_simulate(
            rankings, out, budget=budget, seed=seed, policy=policy, exposed_k=exposed_k, noise=noise
        )
    except SimulationError as exc:
        typer.echo(f"simulate: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {len(events)} events to {out}")


@app.command("labels")
def labels(
    events: Annotated[
        Path,
        typer.Option(
            "--events", exists=True, dir_okay=False, help="FeedbackEvent JSONL (e.g. `pr simulate`)"
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="output Label JSONL path")],
) -> None:
    """Convert feedback events to labels (C4, contract c4-labels-v1); writes <out>.report.json too."""
    try:
        report = run_labels(events, out)
    except LabelBuildError as exc:
        typer.echo(f"labels: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"wrote {report['pos'] + report['neg']} labels to {out} "
        f"(pos={report['pos']}, neg={report['neg']}, conflicts={report['conflict_count']})"
    )


V2_ERRORS = (RerankError, VectorSourceError, CandStatsError, IndexNotFoundError, TestSplitAccessError)


def _check_version(
    command: str,
    version: str,
    cand_stats: Path | None,
    v2_options: dict[str, object] | None = None,
) -> None:
    if version not in ("v0", "v1", "v2"):
        raise RerankError(f"unknown --version {version!r}; expected 'v0', 'v1' or 'v2'")
    if version in ("v1", "v2") and cand_stats is None:
        raise RerankError(f"--version {version} requires --cand-stats")
    if version == "v0" and cand_stats is not None:
        raise RerankError("--cand-stats is only valid with --version v1 or v2")
    if cand_stats is not None and not cand_stats.is_file():
        raise RerankError(f"--cand-stats {cand_stats} does not exist")
    opts = v2_options or {}
    if version == "v2":
        if opts.get("config") is None or opts.get("split") is None:
            raise RerankError("--version v2 requires --config and --split")
        if opts["split"] not in ("train", "val"):
            raise RerankError(f"--split must be 'train' or 'val', got {opts['split']!r}")
        if opts.get("embedder") not in ("siglip", "fake"):
            raise RerankError(f"unknown --embedder {opts.get('embedder')!r}; expected 'siglip' or 'fake'")
    else:
        given = sorted(k for k, v in opts.items() if v is not None)
        if given:
            raise RerankError(f"--{given[0].replace('_', '-')} is only valid with --version v2")


def _v2_opts(config, split, embedder, artifacts_root, limit_products, version, seed) -> dict[str, object]:
    """v2-only options; a value that is still the default under v0/v1 counts as not given."""
    if version == "v2":
        return {
            "config": config,
            "split": split,
            "embedder": embedder,
            "artifacts_root": artifacts_root,
            "limit_products": limit_products,
            "seed": seed,
        }
    return {
        "config": config,
        "split": split,
        "limit_products": limit_products,
        "seed": seed,
        "embedder": None if embedder == "siglip" else embedder,
        "artifacts_root": None if artifacts_root == Path("artifacts") else artifacts_root,
    }


def _v2_vector_fn(opts: dict[str, object], cand_stats: Path, query_ids: list[str]):
    """(vector_fn, index_id, embed_model_id) from the cand-stats index; fails if the located index differs."""
    index_id = cand_stats_index_id(rerank_v1.load_cand_stats(cand_stats))
    vectors = load_candidate_vectors(
        load_config(opts["config"]),  # type: ignore[arg-type]
        opts["split"],  # type: ignore[arg-type]
        sorted(set(query_ids)),
        expected_index_id=index_id,
        embedder_name=opts["embedder"],  # type: ignore[arg-type]
        artifacts_root=opts["artifacts_root"],  # type: ignore[arg-type]
        limit_products=opts["limit_products"],  # type: ignore[arg-type]
    )
    return vectors.pair, index_id, vectors.embed_model_id


@app.command("train-rerank")
def train_rerank(
    labels: Annotated[
        Path, typer.Option("--labels", exists=True, dir_okay=False, help="Label JSONL from `pr labels`")
    ],
    rankings: Annotated[
        Path,
        typer.Option(
            "--rankings", exists=True, dir_okay=False, help="training rankings JSONL from `pr eval`"
        ),
    ],
    out_root: Annotated[Path, typer.Option("--out-root", help="root for reranker artifacts")] = Path(
        "artifacts/rerank"
    ),
    version: Annotated[
        str, typer.Option("--version", help="reranker version: v0 (default), v1 or v2")
    ] = "v0",
    cand_stats: Annotated[
        Path | None,
        typer.Option("--cand-stats", help="cand-stats JSONL from `pr cand-stats` (v1 only, required)"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (v2 only, required)"),
    ] = None,
    split: Annotated[
        str | None, typer.Option("--split", help="split the rankings were made on: train or val (v2 only)")
    ] = None,
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder: siglip (default) or fake (v2 only)")
    ] = "siglip",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts (v2 only)")
    ] = Path("artifacts"),
    limit_products: Annotated[
        int | None, typer.Option("--limit-products", help="N given to `pr eval --limit-products` (v2 only)")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="training seed (v2 only, default 0)")] = None,
) -> None:
    """Train the reranker from labels (C5, contract c5-rerank-v0 / v1 / v2)."""
    try:
        opts = _v2_opts(config, split, embedder, artifacts_root, limit_products, version, seed)
        _check_version("train-rerank", version, cand_stats, opts)
        if version == "v2":
            if opts["split"] != "train":
                # c5-rerank-v2 §3: validation data never enters fitting or early stopping.
                raise RerankError(
                    f"train-rerank --version v2 trains on the train split only, got {opts['split']!r}"
                )
            lbls = rerank_v0.load_labels(labels)
            fn, index_id, embed_id = _v2_vector_fn(opts, cand_stats, [lb.query_id for lb in lbls])
            model = rerank_v2.run_train(
                labels, rankings, cand_stats, out_root, fn, index_id, 0 if seed is None else seed, embed_id
            )
        elif version == "v1":
            model = rerank_v1.run_train(labels, rankings, cand_stats, out_root)
        else:
            model = rerank_v0.run_train(labels, rankings, out_root)
    except V2_ERRORS as exc:
        typer.echo(f"train-rerank: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"reranker_version={model['reranker_version']} "
        f"(pos={model['n_pos']}, neg={model['n_neg']}) saved to {model['path']}"
    )


@app.command("rerank")
def rerank(
    model: Annotated[Path, typer.Option("--model", exists=True, dir_okay=False, help="model.json")],
    rankings: Annotated[
        Path, typer.Option("--rankings", exists=True, dir_okay=False, help="rankings JSONL from `pr eval`")
    ],
    out: Annotated[Path, typer.Option("--out", help="output rankings JSONL path")],
    version: Annotated[
        str, typer.Option("--version", help="reranker version: v0 (default), v1 or v2")
    ] = "v0",
    cand_stats: Annotated[
        Path | None,
        typer.Option("--cand-stats", help="cand-stats JSONL from `pr cand-stats` (v1 only, required)"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (v2 only, required)"),
    ] = None,
    split: Annotated[
        str | None, typer.Option("--split", help="split the rankings were made on: train or val (v2 only)")
    ] = None,
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder: siglip (default) or fake (v2 only)")
    ] = "siglip",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts (v2 only)")
    ] = Path("artifacts"),
    limit_products: Annotated[
        int | None, typer.Option("--limit-products", help="N given to `pr eval --limit-products` (v2 only)")
    ] = None,
) -> None:
    """Re-order the top 20 of each ranking with a trained reranker (C5, contract c5-rerank-v0 / v1 / v2)."""
    try:
        opts = _v2_opts(config, split, embedder, artifacts_root, limit_products, version, None)
        _check_version("rerank", version, cand_stats, opts)
        if version == "v2":
            fn, _, embed_id = _v2_vector_fn(
                opts, cand_stats, [r["query_id"] for r in rerank_v0.load_rankings(rankings)]
            )
            n = rerank_v2.run_rerank(model, rankings, cand_stats, out, fn, embed_id)
        elif version == "v1":
            n = rerank_v1.run_rerank(model, rankings, cand_stats, out)
        else:
            n = rerank_v0.run_rerank(model, rankings, out)
    except V2_ERRORS as exc:
        typer.echo(f"rerank: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {n} reranked rows to {out}")


@app.command("curve")
def curve(
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, help="experiment YAML (core.config)"),
    ],
    train_rankings: Annotated[
        Path, typer.Option("--train-rankings", exists=True, dir_okay=False, help="train rankings JSONL")
    ],
    val_rankings: Annotated[
        Path, typer.Option("--val-rankings", exists=True, dir_okay=False, help="val rankings JSONL")
    ],
    train_cand_stats: Annotated[
        Path, typer.Option("--train-cand-stats", exists=True, dir_okay=False, help="train cand-stats JSONL")
    ],
    val_cand_stats: Annotated[
        Path, typer.Option("--val-cand-stats", exists=True, dir_okay=False, help="val cand-stats JSONL")
    ],
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="run directory name; reuse it to resume")
    ] = None,
    out_root: Annotated[Path, typer.Option("--out-root", help="root for curve runs")] = Path("reports/curve"),
    embedder: Annotated[
        str, typer.Option("--embedder", help="embedder: siglip (default) or fake")
    ] = "siglip",
    artifacts_root: Annotated[
        Path, typer.Option("--artifacts-root", help="root for embeddings/index artifacts")
    ] = Path("artifacts"),
    only_keys: Annotated[
        int | None,
        typer.Option(
            "--only-keys",
            help="smoke test: run only the first N of the 108 keys (skips the population count check)",
        ),
    ] = None,
) -> None:
    """Run the learning-curve grid on train/val (C4, contract c4-learning-curve); never opens test."""
    try:
        keys = None if only_keys is None else plan_keys()[: max(only_keys, 0)]
        if keys is not None and not keys:
            raise CurveError("--only-keys must be at least 1")
        result = run_curve(
            load_config(config),
            train_rankings,
            val_rankings,
            train_cand_stats,
            val_cand_stats,
            out_root=out_root,
            embedder=embedder,
            artifacts_root=artifacts_root,
            run_id=run_id,
            keys=keys,
            expected_population=None if only_keys is not None else EXPECTED_POPULATION,
        )
    except (CurveError, SimulationError, LabelBuildError, *V2_ERRORS) as exc:
        typer.echo(f"curve: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"run {result['run_id']}: attempted {result['attempted']} "
        f"(completed {result['completed']}, failed {result['failed']}), skipped {result['skipped']} -> "
        f"{result['run_dir']}"
    )
    if result["failed"]:
        raise typer.Exit(code=3)


@app.command("curve-summary")
def curve_summary(
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False, help="curve run dir")],
) -> None:
    """Write summary.json, table.md and curve.png for a curve run directory."""
    try:
        summary = summarize(run_dir)
    except CurveError as exc:
        typer.echo(f"curve-summary: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    c = summary["counts"]
    typer.echo(
        f"completed {c['completed']} failed {c['failed']} not run {c['not_run']} of {c['planned']}; "
        f"{summary['primary']['result']['sentence']} ({summary['primary']['result']['seed_points']})"
    )


@app.command("gate")
def gate() -> None:
    """Gate a candidate bundle against the current one (C5)."""
    _not_implemented("gate")


@app.command("serve")
def serve() -> None:
    """Run the search/feedback API (C6)."""
    _not_implemented("serve")


if __name__ == "__main__":
    app()
