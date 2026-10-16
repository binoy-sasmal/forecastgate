"""The reproduce orchestrator.

Current scope: ingest, data audit, panel, universe, a baseline recipe fitted at the last
complete week, prediction and publication. Backtesting, the gate and the registry replace the fixed
baseline in later steps (docs/design.md section 9).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from forecastgate.audit import render_data_audit, write_reports
from forecastgate.config import Settings
from forecastgate.ingest import Exclusion, ingest
from forecastgate.panel import build_panel
from forecastgate.publish import ArtifactIdentity, PublishResult, publish
from forecastgate.recipes import MovingAverage, Recipe
from forecastgate.universe import select_universe


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
    reports: tuple[Path, Path]


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
    reports = write_reports(
        settings.reports_dir,
        render_data_audit(data, panel, cutoff=cutoff, universe=universe, settings=settings),
        settings,
        sha,
    )

    fitted = recipe.fit(panel.history(cutoff), cutoff)
    predictions = fitted.predict(universe, settings.horizons)

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
        reports=reports,
    )
