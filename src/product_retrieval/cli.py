"""D20 section 7 CLI: `pr build-index | eval | simulate | labels | train-rerank | rerank | gate | serve`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from product_retrieval import __version__
from product_retrieval.core.config import DEFAULT_DATA_ROOT, load_config
from product_retrieval.data.checks import check_manifest
from product_retrieval.data.manifest import load_manifest
from product_retrieval.data.store import ImageStore
from product_retrieval.eval.retrieval import DEFAULT_KS
from product_retrieval.feedback.labels import LabelBuildError, run_labels
from product_retrieval.feedback.simulate import SimulationError, run_simulate
from product_retrieval.pipelines.build_index import run_build_index
from product_retrieval.pipelines.evaluate import IndexNotFoundError, format_report_table, run_eval
from product_retrieval.pipelines.selection import TestSplitAccessError
from product_retrieval.rerank.v0 import RerankError, run_rerank, run_train

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
) -> None:
    """Train the logistic-regression reranker from labels (C5, contract c5-rerank-v0)."""
    try:
        model = run_train(labels, rankings, out_root)
    except RerankError as exc:
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
) -> None:
    """Re-order the top 20 of each ranking with a trained reranker (C5, contract c5-rerank-v0)."""
    try:
        n = run_rerank(model, rankings, out)
    except RerankError as exc:
        typer.echo(f"rerank: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {n} reranked rows to {out}")


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
