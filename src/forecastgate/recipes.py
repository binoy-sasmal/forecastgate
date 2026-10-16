"""Forecasting recipes: a method plus its config, behind one fit/predict interface.

Baselines are first-class recipes, so a baseline can be the champion and its artifacts
carry the same metadata as a learned model's.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass

import pandas as pd

from forecastgate.panel import WEEK

PREDICTION_COLUMNS = ["stock_code", "cutoff_week", "horizon", "target_week", "prediction"]


class FittedRecipe(ABC):
    def __init__(self, cutoff: pd.Timestamp) -> None:
        self.cutoff = cutoff

    @abstractmethod
    def predict(self, universe: list[str], horizons: tuple[int, ...]) -> pd.DataFrame: ...


class Recipe(ABC):
    name: str

    def params(self) -> dict[str, object]:
        if is_dataclass(self) and not isinstance(self, type):
            return asdict(self)
        return {}

    def config_hash(self) -> str:
        canonical = json.dumps({"name": self.name, "params": self.params()}, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def fit(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> FittedRecipe:
        latest = history["week"].max()
        if latest > cutoff:
            raise ValueError(f"history reaches {latest.date()}, past cutoff {cutoff.date()}")
        return self._fit(history, cutoff)

    @abstractmethod
    def _fit(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> FittedRecipe: ...


def postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shared by backtest and inference: forecasts of recorded sales are never negative."""
    return predictions.assign(prediction=predictions["prediction"].clip(lower=0.0))


class _LevelForecast(FittedRecipe):
    """A flat forecast: one level per product, repeated for every horizon."""

    def __init__(self, cutoff: pd.Timestamp, levels: pd.Series) -> None:
        super().__init__(cutoff)
        self.levels = levels

    def predict(self, universe: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
        missing = sorted(set(universe) - set(self.levels.dropna().index))
        if missing:
            raise ValueError(f"no observed history for {len(missing)} products: {missing[:5]}")
        rows = [
            (code, self.cutoff, h, self.cutoff + h * WEEK, float(self.levels[code]))
            for code in universe
            for h in horizons
        ]
        return postprocess(pd.DataFrame(rows, columns=PREDICTION_COLUMNS))


def _last_observed(history: pd.DataFrame) -> pd.Series:
    observed = history.dropna(subset=["quantity"])
    return observed.sort_values("week", kind="mergesort").groupby("stock_code")["quantity"].last()


@dataclass(frozen=True)
class Naive(Recipe):
    """Most recent observed week; zero-transaction weeks are skipped, not read as 0."""

    name = "naive"

    def _fit(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> FittedRecipe:
        return _LevelForecast(cutoff, _last_observed(history))


@dataclass(frozen=True)
class MovingAverage(Recipe):
    """Mean of the observed weeks among the last `window` calendar weeks.

    Falls back to `naive` for a product whose whole window is missing.
    """

    window: int = 4
    name = "ma4"

    def _fit(self, history: pd.DataFrame, cutoff: pd.Timestamp) -> FittedRecipe:
        recent = history[history["week"] > cutoff - self.window * WEEK]
        means = recent.groupby("stock_code")["quantity"].mean()
        levels = means.fillna(_last_observed(history)).combine_first(_last_observed(history))
        return _LevelForecast(cutoff, levels)
