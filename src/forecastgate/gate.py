"""The promotion gate: frozen configuration, freeze verification and criteria.

`gate.yaml` holds the thresholds and the full evaluation configuration (universe rule,
folds, horizons, recipe parameters). A full-data evaluation requires that the file is
committed, unchanged since that commit, and equal to the configuration the code runs
(docs/design.md section 5). The criteria are inequalities, so a zero error never needs a
division, and any undefined metric a criterion uses fails that criterion.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from forecastgate.config import Settings
from forecastgate.errors import ForecastgateError
from forecastgate.ingest import sha256_file
from forecastgate.recipes import Recipe

GATE_FILE = "gate.yaml"
TOP_LEVEL_KEYS = {
    "version",
    "candidate",
    "baselines",
    "universe",
    "folds",
    "horizons",
    "recipes",
    "thresholds",
}


class GateNotFrozenError(ForecastgateError):
    """A full-data evaluation was requested before the gate config was committed."""

    exit_code = 7


class GateConfigError(ForecastgateError):
    """gate.yaml is malformed, or differs from the configuration the code runs."""

    exit_code = 8


@dataclass(frozen=True)
class Thresholds:
    pooled_wape_ratio_max: float
    min_fold_wins: int
    fold_wape_ratio_max: float
    horizon_wape_ratio_max: float
    abs_bias_max: float
    champion_wape_ratio_max: float


@dataclass(frozen=True)
class GateConfig:
    sha256: str
    candidate: str
    baselines: tuple[str, ...]
    universe: dict[str, int]
    folds: dict[str, int]
    horizons: tuple[int, ...]
    recipes: dict[str, dict[str, Any]]
    thresholds: Thresholds


@dataclass(frozen=True)
class Freeze:
    commit: str  # the last commit that changed gate.yaml
    sha256: str


def load_gate(path: Path) -> GateConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GateConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL_KEYS:
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise GateConfigError(f"{path} must have exactly {sorted(TOP_LEVEL_KEYS)}, has {found}")
    if raw["version"] != 1:
        raise GateConfigError(f"unsupported gate version {raw['version']!r}")
    try:
        thresholds = Thresholds(**raw["thresholds"])
    except TypeError as exc:
        raise GateConfigError(f"thresholds: {exc}") from exc
    return GateConfig(
        sha256=sha256_file(path),
        candidate=str(raw["candidate"]),
        baselines=tuple(raw["baselines"]),
        universe=dict(raw["universe"]),
        folds=dict(raw["folds"]),
        horizons=tuple(raw["horizons"]),
        recipes={name: dict(params or {}) for name, params in raw["recipes"].items()},
        thresholds=thresholds,
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def verify_freeze(root: Path) -> Freeze:
    """gate.yaml must be tracked and identical to its last committed version."""
    try:
        tracked = _git(root, "ls-files", "--error-unmatch", GATE_FILE)
    except OSError as exc:
        raise GateNotFrozenError("verifying the gate freeze needs git and a checkout") from exc
    if tracked.returncode != 0:
        raise GateNotFrozenError(
            f"{GATE_FILE} is not committed; freeze the gate before the first full-data "
            "backtest (docs/design.md section 5)"
        )
    status = _git(root, "status", "--porcelain", "--", GATE_FILE)
    if status.returncode != 0 or status.stdout.strip():
        raise GateNotFrozenError(
            f"{GATE_FILE} differs from its committed version; a full-data evaluation "
            "only runs against the committed gate"
        )
    commit = _git(root, "log", "-1", "--format=%H", "--", GATE_FILE).stdout.strip()
    if not commit:
        raise GateNotFrozenError(f"no commit found for {GATE_FILE}")
    return Freeze(commit=commit, sha256=sha256_file(root / GATE_FILE))


def check_configuration(gate: GateConfig, settings: Settings, recipes: dict[str, Recipe]) -> None:
    """The configuration the code runs must equal the frozen one, field by field."""
    effective_universe = {
        "size": settings.universe_size,
        "min_size": settings.min_universe_size,
        "min_history_weeks": settings.min_history_weeks,
        "ranking_window_weeks": settings.ranking_window_weeks,
    }
    effective_folds = {"count": settings.n_folds, "spacing_weeks": settings.fold_spacing_weeks}
    mismatches = []
    if gate.universe != effective_universe:
        mismatches.append(f"universe {effective_universe} != {gate.universe}")
    if gate.folds != effective_folds:
        mismatches.append(f"folds {effective_folds} != {gate.folds}")
    if gate.horizons != settings.horizons:
        mismatches.append(f"horizons {settings.horizons} != {gate.horizons}")
    if set(gate.recipes) != set(recipes):
        mismatches.append(f"recipes {sorted(recipes)} != {sorted(gate.recipes)}")
    for name, recipe in recipes.items():
        if name in gate.recipes and recipe.params() != gate.recipes[name]:
            mismatches.append(f"{name} params {recipe.params()} != {gate.recipes[name]}")
    if gate.candidate not in recipes or not set(gate.baselines) <= set(recipes):
        mismatches.append("candidate or baselines are not among the recipes")
    if mismatches:
        raise GateConfigError(
            "effective configuration differs from gate.yaml: " + "; ".join(mismatches)
        )


@dataclass(frozen=True)
class Criterion:
    number: int
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    candidate: str
    base: str
    champion: str
    criteria: list[Criterion]

    @property
    def promoted(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def failed(self) -> list[int]:
        return [c.number for c in self.criteria if not c.passed]


def _metric(table: pd.DataFrame, recipe: str, scope: str, column: str) -> float | None:
    rows = table[(table["recipe"] == recipe) & (table["scope"] == scope)]
    if rows.empty:
        return None
    value = rows.iloc[0][column]
    return None if value is None or pd.isna(value) else float(value)


def _scopes(table: pd.DataFrame, kind: str) -> list[str]:
    scopes = {s for s in table["scope"] if s.startswith(f"{kind}=")}
    return sorted(scopes, key=lambda s: int(s.split("=")[1]))


def _f(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _within(cand: float | None, base: float | None, ratio: float) -> bool:
    return cand is not None and base is not None and cand <= ratio * base


def evaluate(table: pd.DataFrame, gate: GateConfig, *, champion: str | None = None) -> GateDecision:
    """Apply the six criteria to a backtest summary (see `backtest.summary`).

    The best single baseline is the one with the lower pooled WAPE; ties go to the one
    listed first. Without a registered champion, the champion is that baseline.
    """
    t = gate.thresholds
    cand = gate.candidate
    pooled = {b: _metric(table, b, "pooled", "wape") for b in gate.baselines}
    defined = [b for b in gate.baselines if pooled[b] is not None]
    base = (
        min(defined, key=lambda b: (pooled[b], gate.baselines.index(b)))
        if defined
        else gate.baselines[0]
    )
    champion = champion or base

    criteria = []
    c_pooled, b_pooled = _metric(table, cand, "pooled", "wape"), pooled.get(base)
    criteria.append(
        Criterion(
            1,
            "pooled WAPE vs best baseline",
            _within(c_pooled, b_pooled, t.pooled_wape_ratio_max),
            f"{_f(c_pooled)} <= {t.pooled_wape_ratio_max} x {_f(b_pooled)}",
        )
    )

    folds = _scopes(table, "fold")
    pairs = [(_metric(table, cand, s, "wape"), _metric(table, base, s, "wape")) for s in folds]
    all_defined = all(c is not None and b is not None for c, b in pairs)
    wins = sum(1 for c, b in pairs if c is not None and b is not None and c < b)
    criteria.append(
        Criterion(
            2,
            "folds won against best baseline",
            all_defined and wins >= t.min_fold_wins,
            f"{wins} of {len(folds)} folds, need {t.min_fold_wins}"
            + ("" if all_defined else "; a fold metric is undefined"),
        )
    )
    worst_fold = [
        s
        for s, (c, b) in zip(folds, pairs, strict=True)
        if not _within(c, b, t.fold_wape_ratio_max)
    ]
    criteria.append(
        Criterion(
            3,
            "no fold far worse than best baseline",
            not worst_fold,
            f"every fold within {t.fold_wape_ratio_max} x baseline"
            + (f"; fails at {', '.join(worst_fold)}" if worst_fold else ""),
        )
    )

    horizons = _scopes(table, "horizon")
    bad_horizons = [
        s
        for s in horizons
        if not _within(
            _metric(table, cand, s, "wape"),
            _metric(table, base, s, "wape"),
            t.horizon_wape_ratio_max,
        )
    ]
    criteria.append(
        Criterion(
            4,
            "no horizon worse than best baseline",
            not bad_horizons,
            f"every horizon within {t.horizon_wape_ratio_max} x baseline"
            + (f"; fails at {', '.join(bad_horizons)}" if bad_horizons else ""),
        )
    )

    c_bias = _metric(table, cand, "pooled", "bias")
    criteria.append(
        Criterion(
            5,
            "pooled bias",
            c_bias is not None and abs(c_bias) <= t.abs_bias_max,
            f"abs({_f(c_bias)}) <= {t.abs_bias_max}",
        )
    )

    champ_pooled = _metric(table, champion, "pooled", "wape")
    criteria.append(
        Criterion(
            6,
            "improvement over champion",
            _within(c_pooled, champ_pooled, t.champion_wape_ratio_max),
            f"{_f(c_pooled)} <= {t.champion_wape_ratio_max} x {_f(champ_pooled)} (`{champion}`"
            + (", the initial champion: best baseline)" if champion == base else ")"),
        )
    )
    return GateDecision(candidate=cand, base=base, champion=champion, criteria=criteria)
