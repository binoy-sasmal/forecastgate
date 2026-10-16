"""Product universe at a cutoff, computed from data up to that cutoff only."""

from __future__ import annotations

import pandas as pd

from forecastgate.config import Settings
from forecastgate.errors import DataContractError
from forecastgate.panel import WEEK, Panel


def eligible_products(panel: Panel, cutoff: pd.Timestamp, settings: Settings) -> pd.Index:
    """Products whose first recorded sale is at least `min_history_weeks` before `cutoff`."""
    first = panel.first_sale_week
    return first[first <= cutoff - settings.min_history_weeks * WEEK].index


def select_universe(panel: Panel, cutoff: pd.Timestamp, settings: Settings) -> list[str]:
    """Top products by recorded quantity in the ranking window ending at `cutoff`.

    Eligible products had their first recorded sale at least `min_history_weeks` before
    the cutoff. Ties are broken by stock code so the result is deterministic.
    """
    if cutoff not in panel.weeks:
        raise DataContractError(f"cutoff {cutoff.date()} is not a complete week")
    eligible = eligible_products(panel, cutoff, settings)
    history = panel.history(cutoff)
    window = history[
        (history["week"] > cutoff - settings.ranking_window_weeks * WEEK)
        & history["stock_code"].isin(eligible)
    ]
    totals = window.groupby("stock_code")["quantity"].sum().reset_index()
    ranked = totals.sort_values(
        ["quantity", "stock_code"], ascending=[False, True], kind="mergesort"
    )
    universe = [str(code) for code in ranked["stock_code"].head(settings.universe_size)]
    if len(universe) < settings.min_universe_size:
        raise DataContractError(
            f"only {len(universe)} products qualify at {cutoff.date()}, "
            f"minimum is {settings.min_universe_size}"
        )
    return sorted(universe)
