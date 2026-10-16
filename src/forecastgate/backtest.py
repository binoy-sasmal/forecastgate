"""Rolling-origin backtest and its deterministic report (docs/design.md section 4).

Each fold refits every recipe on the panel truncated at the fold cutoff, for the same
universe, and scores its forecasts against the recorded sales that followed. All recipes
see the same folds, so their scores are directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from forecastgate.config import Settings
from forecastgate.errors import DataContractError
from forecastgate.folds import backtest_cutoffs
from forecastgate.metrics import bias, mase, mase_scales, wape
from forecastgate.panel import WEEK, Panel
from forecastgate.recipes import Recipe
from forecastgate.universe import select_universe

BASELINES = ("naive", "ma4")
ORACLE = "oracle"
SCORED_COLUMNS = [
    "fold",
    "cutoff_week",
    "recipe",
    "stock_code",
    "horizon",
    "target_week",
    "actual",
    "prediction",
]


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: pd.Timestamp
    universe_size: int


@dataclass(frozen=True)
class BacktestResult:
    folds: list[Fold]
    scored: pd.DataFrame  # SCORED_COLUMNS, one row per recipe x fold x product x horizon
    scales: pd.Series  # MASE scale indexed by (fold, stock_code)


def run_backtest(panel: Panel, settings: Settings, recipes: dict[str, Recipe]) -> BacktestResult:
    cutoffs = backtest_cutoffs(panel.weeks, settings)
    targets_needed = {c + h * WEEK for c in cutoffs for h in settings.horizons}
    blocked = sorted(targets_needed & set(panel.zero_transaction_weeks))
    if blocked:
        # Scored rows are never dropped silently (docs/design.md section 1).
        raise DataContractError(
            f"backtest target weeks include zero-transaction weeks: {[f'{w:%F}' for w in blocked]}"
        )

    folds: list[Fold] = []
    parts: list[pd.DataFrame] = []
    scales: list[pd.Series] = []
    for index, cutoff in enumerate(cutoffs):
        visible = panel.truncated(cutoff)
        universe = select_universe(visible, cutoff, settings)
        folds.append(Fold(index, cutoff, len(universe)))

        target_weeks = [cutoff + h * WEEK for h in settings.horizons]
        actuals = panel.frame[
            panel.frame["stock_code"].isin(universe) & panel.frame["week"].isin(target_weeks)
        ][["stock_code", "week", "quantity"]].rename(
            columns={"week": "target_week", "quantity": "actual"}
        )
        expected_rows = len(universe) * len(settings.horizons)
        if len(actuals) != expected_rows or actuals["actual"].isna().any():
            raise DataContractError(f"fold {index}: actuals incomplete for the universe")

        for name, recipe in recipes.items():
            fitted = recipe.fit(visible, universe, settings.horizons)
            forecast = fitted.predict(visible, universe, settings.horizons)
            scored = forecast.merge(actuals, on=["stock_code", "target_week"], how="inner")
            if len(scored) != expected_rows:
                raise ValueError(f"fold {index}: {name} scored {len(scored)} of {expected_rows}")
            parts.append(scored.assign(fold=index, recipe=name))

        scale = mase_scales(visible).reindex(universe)
        scale.index = pd.MultiIndex.from_product(
            [[index], scale.index], names=["fold", "stock_code"]
        )
        scales.append(scale)

    scored_all = pd.concat(parts, ignore_index=True)[SCORED_COLUMNS]
    scored_all = scored_all.sort_values(
        ["recipe", "fold", "stock_code", "horizon"], kind="mergesort"
    ).reset_index(drop=True)
    return BacktestResult(folds=folds, scored=scored_all, scales=pd.concat(scales))


def with_oracle(scored: pd.DataFrame) -> pd.DataFrame:
    """Add the per-fold oracle: in each fold, the baseline with the lower fold WAPE.

    A demanding reference, not a deployable model: it chooses after seeing each fold.
    Ties go to the baseline listed first in BASELINES.
    """
    picks = []
    for _, rows in scored[scored["recipe"].isin(BASELINES)].groupby("fold"):
        scores = {name: wape(r["actual"], r["prediction"]) for name, r in rows.groupby("recipe")}
        defined = [b for b in BASELINES if scores.get(b) is not None]
        if not defined:
            continue
        best = min(defined, key=lambda b: (scores[b], BASELINES.index(b)))
        picks.append(rows[rows["recipe"] == best].assign(recipe=ORACLE))
    return pd.concat([scored, *picks], ignore_index=True)


def summary(result: BacktestResult) -> pd.DataFrame:
    """One row per (recipe, scope): pooled, per fold and per horizon."""
    scored = with_oracle(result.scored)
    records = []
    for recipe, rows in scored.groupby("recipe", sort=True):
        value, excluded = mase(rows, result.scales)
        records.append(
            {
                "recipe": recipe,
                "scope": "pooled",
                "wape": wape(rows["actual"], rows["prediction"]),
                "bias": bias(rows["actual"], rows["prediction"]),
                "mase": value,
                "mase_excluded": excluded,
            }
        )
        for kind in ("fold", "horizon"):
            for key, part in rows.groupby(kind, sort=True):
                records.append(
                    {
                        "recipe": recipe,
                        "scope": f"{kind}={key}",
                        "wape": wape(part["actual"], part["prediction"]),
                        "bias": bias(part["actual"], part["prediction"]),
                        "mase": None,
                        "mase_excluded": None,
                    }
                )
    return pd.DataFrame.from_records(records)


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(
    result: BacktestResult,
    settings: Settings,
    recipes: dict[str, Recipe],
    *,
    data_hash: str,
    gate_hash: str | None,
) -> str:
    table = summary(result)
    names = [*recipes, ORACLE]
    lines = [
        "# Backtest report",
        "",
        "Generated by `uv run forecastgate backtest`. Deterministic content only; run",
        "provenance is in `provenance.json`.",
        "",
    ]
    leaky = [n for n, r in recipes.items() if r.params().get("leak")]
    if leaky:
        lines += [
            f"> **Planted leak in {', '.join(leaky)}.** This report demonstrates a failure;",
            "> its numbers are not evidence of model quality.",
            "",
        ]
    lines += [
        "| Item | Value |",
        "|---|---|",
        f"| Source | `{settings.source.name}`, archive SHA-256 `{data_hash}` |",
        "| Gate config SHA-256 | "
        + (f"`{gate_hash}`" if gate_hash else "not frozen (fixture run)")
        + " |",
        f"| Horizons | {', '.join(map(str, settings.horizons))} weeks |",
        f"| Universe rule | first sale at least {settings.min_history_weeks} weeks before the "
        f"cutoff; top {settings.universe_size} by quantity over "
        f"{settings.ranking_window_weeks} weeks |",
        "",
        "## Folds",
        "",
        "Each recipe is refitted per fold on data up to the cutoff only.",
        "",
        "| Fold | Cutoff week | Products | Scored rows per recipe |",
        "|---:|---|---:|---:|",
    ]
    for fold in result.folds:
        rows = fold.universe_size * len(settings.horizons)
        lines.append(f"| {fold.index} | {fold.cutoff:%Y-%m-%d} | {fold.universe_size} | {rows} |")
    lines += [
        "",
        "## Recipes",
        "",
        "| Recipe | Config SHA-256 | Parameters |",
        "|---|---|---|",
    ]
    for name, recipe in recipes.items():
        params = json.dumps(recipe.params(), sort_keys=True)
        lines.append(f"| `{name}` | `{recipe.config_hash()[:12]}` | `{params}` |")
    lines += [
        f"| `{ORACLE}` | n/a | per fold, the better of {' and '.join(BASELINES)}; "
        "a reference, not deployable |",
        "",
        "## Pooled",
        "",
        "Lower WAPE is better. Bias is signed: positive means over-forecasting.",
        "",
        "| Recipe | WAPE | Bias | MASE | Series excluded from MASE |",
        "|---|---:|---:|---:|---:|",
    ]
    pooled = table[table["scope"] == "pooled"].set_index("recipe")
    excluded = pooled["mase_excluded"].astype("int64").to_dict()
    for name in names:
        row = pooled.loc[name]
        lines.append(
            f"| `{name}` | {_fmt(row['wape'])} | {_fmt(row['bias'])} | {_fmt(row['mase'])} "
            f"| {excluded[name]} |"
        )
    for kind, label in (("fold", "Fold"), ("horizon", "Horizon")):
        for metric in ("wape", "bias"):
            lines += [
                "",
                f"## {metric.upper() if metric == 'wape' else 'Bias'} by {kind}",
                "",
                f"| {label} | " + " | ".join(f"`{n}`" for n in names) + " |",
                "|---:|" + "---:|" * len(names),
            ]
            scoped = table[table["scope"].str.startswith(f"{kind}=")]
            keys = sorted({s.split("=")[1] for s in scoped["scope"]}, key=int)
            for key in keys:
                cells = []
                for name in names:
                    match = scoped[
                        (scoped["recipe"] == name) & (scoped["scope"] == f"{kind}={key}")
                    ]
                    cells.append(_fmt(match.iloc[0][metric]) if len(match) else "n/a")
                lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
