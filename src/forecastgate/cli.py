"""Command-line interface. This is the only official way to run the pipeline.

Exit codes: 0 success; 10 the gate rejected the candidate (`gate` only); 2 to 9 an
explained failure (see errors.py, gate.py and registry.py); 1 an unexpected error.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import typer

from forecastgate.config import Settings, settings_for
from forecastgate.errors import ForecastgateError
from forecastgate.gate import GATE_REJECTED_EXIT
from forecastgate.pipeline import GateOutcome, PredictOutcome, gate_command, predict_command
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


def _settings(fixture: bool, artifacts_dir: Path | None) -> tuple[Path, Settings]:
    root = Path.cwd()
    return root, settings_for(fixture=fixture, root=root, artifacts_dir=artifacts_dir)


def _run[T](action: Callable[[], T]) -> T:
    try:
        return action()
    except ForecastgateError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(exc.exit_code) from exc


def _line(label: str, value: object) -> None:
    typer.echo(f"{label:<18}{value}")


def _echo_gate(outcome: GateOutcome) -> None:
    decision = outcome.decision
    verdict = (
        "promoted"
        if decision.promoted
        else f"rejected (criteria {', '.join(map(str, decision.failed))} failed)"
    )
    _line("gate", f"{decision.candidate} {verdict}; best baseline {decision.base}")
    for version in outcome.registered:
        _line("registered", f"{version.label} ({version.reason})")
    _line("champion", outcome.champion.label)
    _line("backtest report", outcome.report)


def _echo_predict(outcome: PredictOutcome) -> None:
    _line("cutoff week", outcome.cutoff_week)
    _line("universe", f"{outcome.universe_size} products")
    _line("model", outcome.champion.label)
    result = outcome.published
    _line("publication", f"{result.status} {result.identity} at {result.path}")


@app.command()
def reproduce(fixture: FixtureOption = False, artifacts_dir: ArtifactsOption = None) -> None:
    """Audit, gate and publish, from a clean checkout. A gate rejection is not a failure."""
    root, settings = _settings(fixture, artifacts_dir)
    outcome = _run(lambda: run_reproduce(settings, root=root, fixture=fixture))
    _line("source", f"{settings.source.name} ({settings.source.sha256[:12]})")
    _line("sales rows", len(outcome.data.sales))
    for rule, excluded in outcome.data.exclusions.items():
        _line("  excluded", f"{rule}: {excluded.rows}")
    _line("complete weeks", outcome.complete_weeks)
    _line("zero-tx weeks", ", ".join(outcome.zero_transaction_weeks) or "none")
    _line("data audit", outcome.audit)
    _echo_gate(outcome.gate)
    _echo_predict(outcome.predict)


@app.command()
def gate(fixture: FixtureOption = False, artifacts_dir: ArtifactsOption = None) -> None:
    """Training: backtest, apply the frozen gate, register. Exits 10 on rejection."""
    root, settings = _settings(fixture, artifacts_dir)
    outcome = _run(lambda: gate_command(settings, root=root, fixture=fixture))
    _echo_gate(outcome)
    if not outcome.decision.promoted:
        raise typer.Exit(GATE_REJECTED_EXIT)


@app.command()
def predict(fixture: FixtureOption = False, artifacts_dir: ArtifactsOption = None) -> None:
    """Batch inference: publish the registered champion's forecasts."""
    root, settings = _settings(fixture, artifacts_dir)
    _echo_predict(_run(lambda: predict_command(settings, root=root)))


@app.command()
def backtest(
    fixture: FixtureOption = False,
    artifacts_dir: ArtifactsOption = None,
    inject_leak: Annotated[
        Literal["future_window", "origin_filter"] | None,
        typer.Option(help="Plant a deliberate leak in lgbm, for the failure demonstrations."),
    ] = None,
) -> None:
    """Rolling-origin backtest of every recipe; writes the report, registers nothing."""
    root, settings = _settings(fixture, artifacts_dir)
    summary = _run(lambda: run_backtest(settings, root=root, fixture=fixture, leak=inject_leak))
    for fold in summary.result.folds:
        _line(f"fold {fold.index}", f"{fold.cutoff:%Y-%m-%d}, {fold.universe_size} products")
    _line("report", summary.report)


def main() -> None:
    app()
