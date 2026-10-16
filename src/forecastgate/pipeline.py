"""Orchestrators for the CLI commands.

`reproduce` currently publishes a fixed baseline fitted at the last complete week; the gate
and registry choose the champion in later steps (docs/design.md section 9). `backtest`
evaluates all recipes on the rolling-origin folds and writes the backtest report.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from forecastgate.audit import AUDIT_FILE, render_data_audit, write_report
from forecastgate.backtest import BacktestResult, render_report, run_backtest
from forecastgate.config import Settings
from forecastgate.errors import ForecastgateError
from forecastgate.features import Leak
from forecastgate.ingest import Exclusion, ingest, sha256_file
from forecastgate.panel import build_panel
from forecastgate.publish import ArtifactIdentity, PublishResult, publish
from forecastgate.recipes import LightGBM, MovingAverage, Naive, Recipe
from forecastgate.universe import select_universe

GATE_FILE = "gate.yaml"
BACKTEST_FILE = "backtest.md"


class GateNotFrozenError(ForecastgateError):
    """A full-data evaluation was requested before the gate config was committed."""

    exit_code = 7


@dataclass(frozen=True)
class RunSummary:
    exclusions: dict[str, Exclusion]
    sales_rows: int
    complete_weeks: int
    zero_transaction_weeks: list[str]
    cutoff_week: str
    universe_size: int
    recipe: str
    published: PublishResult
    audit: Path


@dataclass(frozen=True)
class BacktestSummary:
    result: BacktestResult
    report: Path


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


def input_hash(panel_fingerprint: str, universe: list[str], horizons: tuple[int, ...]) -> str:
    parts = [panel_fingerprint, ",".join(universe), ",".join(map(str, horizons))]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def reproduce(settings: Settings, *, root: Path, recipe: Recipe | None = None) -> RunSummary:
    recipe = recipe or MovingAverage()
    data = ingest(settings.source, cache_dir=settings.cache_dir)
    panel = build_panel(data)
    cutoff = panel.last_complete_week
    universe = select_universe(panel, cutoff, settings)
    sha = git_sha(root)
    audit = write_report(
        settings.reports_dir,
        AUDIT_FILE,
        render_data_audit(data, panel, cutoff=cutoff, universe=universe, settings=settings),
        settings,
        sha,
    )

    visible = panel.truncated(cutoff)
    predictions = recipe.fit(visible, universe, settings.horizons).predict(
        visible, universe, settings.horizons
    )

    config_hash = recipe.config_hash()
    identity = ArtifactIdentity(
        cutoff_week=f"{cutoff:%Y-%m-%d}",
        model_version=f"{recipe.name}-{config_hash[:12]}",
        recipe_config_hash=config_hash,
        data_hash=data.data_hash,
        input_hash=input_hash(panel.fingerprint(), universe, settings.horizons),
    )
    published = publish(
        settings.predictions_dir,
        identity,
        predictions,
        universe=universe,
        horizons=settings.horizons,
        git_sha=sha,
    )
    return RunSummary(
        exclusions=data.exclusions,
        sales_rows=len(data.sales),
        complete_weeks=len(panel.weeks),
        zero_transaction_weeks=[f"{w:%Y-%m-%d}" for w in panel.zero_transaction_weeks],
        cutoff_week=identity.cutoff_week,
        universe_size=len(universe),
        recipe=identity.model_version,
        published=published,
        audit=audit,
    )


def backtest_recipes(leak: Leak | None = None) -> dict[str, Recipe]:
    return {"naive": Naive(), "ma4": MovingAverage(), "lgbm": LightGBM(leak=leak)}


def backtest(
    settings: Settings, *, root: Path, fixture: bool, leak: Leak | None = None
) -> BacktestSummary:
    """Backtest every recipe. A full-data run requires the frozen gate config."""
    gate_path = root / GATE_FILE
    if fixture:
        gate_hash = None
    elif gate_path.exists():
        gate_hash = sha256_file(gate_path)
    else:
        raise GateNotFrozenError(
            f"{GATE_FILE} is not committed yet; freeze the gate before the first full-data "
            "backtest (docs/design.md section 5)"
        )
    data = ingest(settings.source, cache_dir=settings.cache_dir)
    panel = build_panel(data)
    recipes = backtest_recipes(leak)
    result = run_backtest(panel, settings, recipes)
    name = BACKTEST_FILE if leak is None else f"backtest_leak_{leak}.md"
    report = write_report(
        settings.reports_dir,
        name,
        render_report(result, settings, recipes, data_hash=data.data_hash, gate_hash=gate_hash),
        settings,
        git_sha(root),
    )
    return BacktestSummary(result=result, report=report)
