"""Command-line interface. This is the only official way to run the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from forecastgate.config import settings_for
from forecastgate.errors import ForecastgateError
from forecastgate.pipeline import backtest as run_backtest
from forecastgate.pipeline import reproduce as run_reproduce

app = typer.Typer(add_completion=False, no_args_is_help=True)

FixtureOption = Annotated[
    bool, typer.Option("--fixture", help="Run offline on the committed fixture slice.")
]
ArtifactsOption = Annotated[
    Path | None, typer.Option(help="Where artifacts go. Default: artifacts/<source>.")
]


@app.callback()
def _root() -> None:
    """Weekly sales forecasting with a promotion gate and versioned batch predictions."""


@app.command()
def reproduce(fixture: FixtureOption = False, artifacts_dir: ArtifactsOption = None) -> None:
    """Ingest, fit, predict and publish, from a clean checkout."""
    root = Path.cwd()
    settings = settings_for(fixture=fixture, root=root, artifacts_dir=artifacts_dir)
    try:
        summary = run_reproduce(settings, root=root)
    except ForecastgateError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc

    typer.echo(f"source            {settings.source.name} ({settings.source.sha256[:12]})")
    typer.echo(f"sales rows        {summary.sales_rows}")
    for rule, excluded in summary.exclusions.items():
        typer.echo(f"  excluded        {rule}: {excluded.rows}")
    typer.echo(f"complete weeks    {summary.complete_weeks}")
    typer.echo(f"zero-tx weeks     {', '.join(summary.zero_transaction_weeks) or 'none'}")
    typer.echo(f"cutoff week       {summary.cutoff_week}")
    typer.echo(f"universe          {summary.universe_size} products")
    typer.echo(f"recipe            {summary.recipe}")
    typer.echo(f"data audit        {summary.audit}")
    result = summary.published
    typer.echo(f"publication       {result.status} {result.identity} at {result.path}")


@app.command()
def backtest(
    fixture: FixtureOption = False,
    artifacts_dir: ArtifactsOption = None,
    inject_leak: Annotated[
        Literal["future_window", "origin_filter"] | None,
        typer.Option(help="Plant a deliberate leak in lgbm, for the failure demonstrations."),
    ] = None,
) -> None:
    """Rolling-origin backtest of every recipe; writes the backtest report."""
    root = Path.cwd()
    settings = settings_for(fixture=fixture, root=root, artifacts_dir=artifacts_dir)
    try:
        summary = run_backtest(settings, root=root, fixture=fixture, leak=inject_leak)
    except ForecastgateError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc
    for fold in summary.result.folds:
        typer.echo(
            f"fold {fold.index}            {fold.cutoff:%Y-%m-%d}, {fold.universe_size} products"
        )
    typer.echo(f"report            {summary.report}")


def main() -> None:
    app()
