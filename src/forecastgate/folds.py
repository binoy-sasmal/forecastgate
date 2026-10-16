"""Backtest cutoffs: rolling origins spaced evenly, with every horizon observable."""

from __future__ import annotations

import pandas as pd

from forecastgate.config import Settings
from forecastgate.errors import DataContractError
from forecastgate.panel import WEEK


def backtest_cutoffs(weeks: pd.DatetimeIndex, settings: Settings) -> list[pd.Timestamp]:
    """The last cutoff leaves room for the longest horizon inside the complete weeks."""
    last = weeks[-1] - max(settings.horizons) * WEEK
    cutoffs = [
        last - k * settings.fold_spacing_weeks * WEEK for k in reversed(range(settings.n_folds))
    ]
    missing = [c for c in cutoffs if c not in weeks]
    if missing:
        raise DataContractError(
            f"{settings.n_folds} folds need cutoffs from {cutoffs[0].date()}, "
            f"but complete weeks start {weeks[0].date()}"
        )
    return cutoffs
