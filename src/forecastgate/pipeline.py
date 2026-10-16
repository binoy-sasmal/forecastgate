"""Orchestrators for the CLI commands.

- `backtest`: evaluate every recipe on the rolling-origin folds and write the report.
- `gate` (training): backtest, apply the frozen gate, register the initial champion on
  an empty registry, refit and register a promoted candidate. Exit 10 on rejection.
- `predict` (batch inference): load the registered champion and publish its forecasts.
- `reproduce`: data audit, then gate (a rejection is an expected outcome), then predict.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from forecastgate.audit import AUDIT_FILE, render_data_audit, write_report
from forecastgate.backtest import BacktestResult, render_report, run_backtest, summary
from forecastgate.config import Settings
from forecastgate.features import Leak
from forecastgate.gate import (
    GATE_FILE,
    Freeze,
    GateConfig,
    GateConfigError,
    GateDecision,
    GateNotFrozenError,
    check_configuration,
    evaluate,
    load_gate,
    verify_freeze,
)
from forecastgate.ingest import IngestResult, ingest
from forecastgate.panel import Panel, build_panel
from forecastgate.publish import ArtifactIdentity, PublishResult, publish
from forecastgate.recipes import LightGBM, MovingAverage, Naive, Recipe
from forecastgate.registry import ModelVersion, Registry, RegistryError
from forecastgate.universe import select_universe

BACKTEST_FILE = "backtest.md"


@dataclass(frozen=True)
class BacktestSummary:
    result: BacktestResult
    report: Path


@dataclass(frozen=True)
class GateOutcome:
    decision: GateDecision
    champion: ModelVersion
    registered: list[ModelVersion]
    report: Path


@dataclass(frozen=True)
class PredictOutcome:
    champion: ModelVersion
    cutoff_week: str
    universe_size: int
    published: PublishResult


@dataclass(frozen=True)
class ReproduceOutcome:
    data: IngestResult
    complete_weeks: int
    zero_transaction_weeks: list[str]
    audit: Path
    gate: GateOutcome
    predict: PredictOutcome


def git_sha(root: Path) -> str:
    """The checkout's commit, or FORECASTGATE_GIT_SHA where there is no .git (Docker)."""
    if sha := os.environ.get("FORECASTGATE_GIT_SHA"):
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def git_dirty(root: Path) -> bool | None:
    """Whether tracked files differ from the commit; None where there is no checkout."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


@dataclass(frozen=True)
class RunInfo:
    """Where the code that runs came from, captured once when a command starts.

    Captured before any report is written: the reports are tracked files, so checking
    the working tree after writing one would report the run's own output as a change.
    """

    git_sha: str
    git_dirty: bool | None


def capture_run(root: Path) -> RunInfo:
    return RunInfo(git_sha=git_sha(root), git_dirty=git_dirty(root))


def input_hash(panel_fingerprint: str, universe: list[str], horizons: tuple[int, ...]) -> str:
    parts = [panel_fingerprint, ",".join(universe), ",".join(map(str, horizons))]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def backtest_recipes(leak: Leak | None = None) -> dict[str, Recipe]:
    return {"naive": Naive(), "ma4": MovingAverage(), "lgbm": LightGBM(leak=leak)}


def frozen_gate(
    settings: Settings, *, root: Path, fixture: bool
) -> tuple[GateConfig, Freeze | None]:
    """The gate a run may use. Full-data runs verify the freeze before any download."""
    freeze = None if fixture else verify_freeze(root)
    if not (root / GATE_FILE).exists():
        raise GateNotFrozenError(f"{GATE_FILE} is missing")
    gate = load_gate(root / GATE_FILE)
    check_configuration(gate, settings, backtest_recipes())
    return gate, freeze


def _load(settings: Settings) -> tuple[IngestResult, Panel]:
    data = ingest(settings.source, cache_dir=settings.cache_dir)
    return data, build_panel(data)


def backtest(
    settings: Settings, *, root: Path, fixture: bool, leak: Leak | None = None
) -> BacktestSummary:
    """Backtest every recipe and report the gate decision, without registering anything.

    A full-data run first verifies that gate.yaml is committed and unchanged, before any
    download. Planted leaks are only allowed on the fixture.
    """
    run = capture_run(root)
    if leak is not None and not fixture:
        raise GateConfigError("planted leaks are for fixture demonstrations only")
    gate: GateConfig | None
    if fixture and not (root / GATE_FILE).exists():
        gate, freeze = None, None
    else:
        gate, freeze = frozen_gate(settings, root=root, fixture=fixture)
    data, panel = _load(settings)
    recipes = backtest_recipes(leak)
    result = run_backtest(panel, settings, recipes)
    name = BACKTEST_FILE if leak is None else f"backtest_leak_{leak}.md"
    report = _write_backtest_report(
        settings, run, name, result, recipes, data=data, gate=gate, freeze=freeze
    )
    return BacktestSummary(result=result, report=report)


def _write_backtest_report(
    settings: Settings,
    run: RunInfo,
    name: str,
    result: BacktestResult,
    recipes: dict[str, Recipe],
    *,
    data: IngestResult,
    gate: GateConfig | None,
    freeze: Freeze | None,
) -> Path:
    gate_provenance: dict[str, object] = {
        "gate_sha256": gate.sha256 if gate else None,
        "gate_freeze_commit": freeze.commit if freeze else None,
        "git_dirty": run.git_dirty,
    }
    return write_report(
        settings.reports_dir,
        name,
        render_report(
            result, settings, recipes, data_hash=data.data_hash, gate=gate, freeze=freeze
        ),
        settings,
        run.git_sha,
        extra=gate_provenance,
    )


def run_gate(
    settings: Settings,
    *,
    root: Path,
    data: IngestResult,
    panel: Panel,
    gate: GateConfig,
    freeze: Freeze | None,
    recipes: dict[str, Recipe] | None = None,
    run: RunInfo | None = None,
) -> GateOutcome:
    """Backtest, decide, and register. The decision never depends on the registry's
    bookkeeping beyond which recipe is champion."""
    run = run or capture_run(root)
    recipes = recipes or backtest_recipes()
    registry = Registry(settings.registry_dir)
    result = run_backtest(panel, settings, recipes)
    current = registry.champion()
    decision = evaluate(summary(result), gate, champion=current.recipe if current else None)
    report = _write_backtest_report(
        settings, run, BACKTEST_FILE, result, recipes, data=data, gate=gate, freeze=freeze
    )

    cutoff = panel.last_complete_week
    visible = panel.truncated(cutoff)
    universe = select_universe(visible, cutoff, settings)
    model_input = input_hash(panel.fingerprint(), universe, settings.horizons)
    sha = run.git_sha

    def refit_and_register(name: str, reason: str) -> ModelVersion:
        recipe = recipes[name]
        fitted = recipe.fit(visible, universe, settings.horizons)
        return registry.register(
            fitted,
            recipe=name,
            params=recipe.params(),
            recipe_config_hash=recipe.config_hash(),
            data_hash=data.data_hash,
            input_hash=model_input,
            reason=reason,
            gate_sha256=gate.sha256,
            gate_freeze_commit=freeze.commit if freeze else None,
            git_sha=sha,
            make_champion=True,
        )

    registered = []
    if decision.promoted:
        registered.append(refit_and_register(decision.candidate, "promoted by gate"))
    elif current is None:
        registered.append(
            refit_and_register(decision.base, "initial champion: best single baseline")
        )
    elif current.input_hash != model_input:
        # The champion keeps its role but is refitted on the current input, so predict
        # never serves a model fitted on different data.
        registered.append(refit_and_register(current.recipe, "champion refitted on new input"))

    champion = registry.champion()
    if champion is None:  # unreachable: every branch above leaves a champion
        raise RegistryError("no champion after the gate run")
    registry.record_decision(
        {
            "candidate": decision.candidate,
            "base": decision.base,
            "champion_before": current.label if current else None,
            "champion_after": champion.label,
            "promoted": decision.promoted,
            "failed_criteria": decision.failed,
            "registered": [v.label for v in registered],
            "gate_sha256": gate.sha256,
            "gate_freeze_commit": freeze.commit if freeze else None,
            "data_hash": data.data_hash,
            "git_sha": sha,
        }
    )
    return GateOutcome(decision=decision, champion=champion, registered=registered, report=report)


def gate_command(settings: Settings, *, root: Path, fixture: bool) -> GateOutcome:
    run = capture_run(root)
    gate, freeze = frozen_gate(settings, root=root, fixture=fixture)
    data, panel = _load(settings)
    return run_gate(settings, root=root, data=data, panel=panel, gate=gate, freeze=freeze, run=run)


def run_predict(
    settings: Settings,
    *,
    root: Path,
    data: IngestResult,
    panel: Panel,
    run: RunInfo | None = None,
) -> PredictOutcome:
    """Publish the champion's forecasts from the last complete week."""
    run = run or capture_run(root)
    registry = Registry(settings.registry_dir)
    champion = registry.champion()
    if champion is None:
        raise RegistryError("no champion registered; run `forecastgate gate` first")
    cutoff = panel.last_complete_week
    visible = panel.truncated(cutoff)
    universe = select_universe(visible, cutoff, settings)
    model_input = input_hash(panel.fingerprint(), universe, settings.horizons)
    if champion.data_hash != data.data_hash or champion.input_hash != model_input:
        raise RegistryError(
            f"{champion.label} was fitted on different input (cutoff {champion.cutoff_week}); "
            "run `forecastgate gate` to refit before publishing"
        )
    fitted = registry.load(champion)
    predictions = fitted.predict(visible, universe, settings.horizons)
    identity = ArtifactIdentity(
        cutoff_week=f"{cutoff:%Y-%m-%d}",
        model_version=champion.label,
        recipe_config_hash=champion.recipe_config_hash,
        data_hash=data.data_hash,
        input_hash=model_input,
    )
    published = publish(
        settings.predictions_dir,
        identity,
        predictions,
        universe=universe,
        horizons=settings.horizons,
        git_sha=run.git_sha,
    )
    return PredictOutcome(
        champion=champion,
        cutoff_week=identity.cutoff_week,
        universe_size=len(universe),
        published=published,
    )


def predict_command(settings: Settings, *, root: Path) -> PredictOutcome:
    run = capture_run(root)
    data, panel = _load(settings)
    return run_predict(settings, root=root, data=data, panel=panel, run=run)


def reproduce(settings: Settings, *, root: Path, fixture: bool) -> ReproduceOutcome:
    """Clean checkout to published forecasts. A gate rejection is recorded, not fatal."""
    run = capture_run(root)
    gate, freeze = frozen_gate(settings, root=root, fixture=fixture)
    data, panel = _load(settings)
    cutoff = panel.last_complete_week
    universe = select_universe(panel.truncated(cutoff), cutoff, settings)
    audit = write_report(
        settings.reports_dir,
        AUDIT_FILE,
        render_data_audit(data, panel, cutoff=cutoff, universe=universe, settings=settings),
        settings,
        run.git_sha,
        extra={"git_dirty": run.git_dirty},
    )
    outcome = run_gate(
        settings, root=root, data=data, panel=panel, gate=gate, freeze=freeze, run=run
    )
    predicted = run_predict(settings, root=root, data=data, panel=panel, run=run)
    return ReproduceOutcome(
        data=data,
        complete_weeks=len(panel.weeks),
        zero_transaction_weeks=[f"{w:%Y-%m-%d}" for w in panel.zero_transaction_weeks],
        audit=audit,
        gate=outcome,
        predict=predicted,
    )
