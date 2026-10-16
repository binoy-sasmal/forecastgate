"""Forecasting recipes: a method plus its config, behind one fit/predict interface.

A recipe is fitted on a panel truncated at its cutoff (`Panel.truncated`), so the last
week it can see is the cutoff. Baselines are first-class recipes, so a baseline can be the
champion and its artifacts carry the same metadata as a learned model's.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass

import lightgbm as lgb
import pandas as pd

from forecastgate.features import FEATURE_COLUMNS, Leak, inference_rows, training_rows
from forecastgate.panel import WEEK, Panel

PREDICTION_COLUMNS = ["stock_code", "cutoff_week", "horizon", "target_week", "prediction"]


def postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shared by backtest and inference: forecasts of recorded sales are never negative."""
    return predictions.assign(prediction=predictions["prediction"].clip(lower=0.0))


class FittedRecipe(ABC):
    def __init__(self, cutoff: pd.Timestamp) -> None:
        self.cutoff = cutoff

    def predict(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
        if panel.last_complete_week != self.cutoff:
            raise ValueError(
                f"fitted at {self.cutoff.date()}, asked to predict from "
                f"{panel.last_complete_week.date()}"
            )
        return postprocess(self._predict(panel, universe, horizons))

    @abstractmethod
    def _predict(
        self, panel: Panel, universe: list[str], horizons: tuple[int, ...]
    ) -> pd.DataFrame: ...


class Recipe(ABC):
    name: str

    def params(self) -> dict[str, object]:
        if is_dataclass(self) and not isinstance(self, type):
            return asdict(self)
        return {}

    def config_hash(self) -> str:
        canonical = json.dumps({"name": self.name, "params": self.params()}, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def fit(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> FittedRecipe:
        """Fit at the panel's last complete week, which is the cutoff."""
        return self._fit(panel, universe, horizons)

    @abstractmethod
    def _fit(
        self, panel: Panel, universe: list[str], horizons: tuple[int, ...]
    ) -> FittedRecipe: ...


class _LevelForecast(FittedRecipe):
    """A flat forecast: one level per product, repeated for every horizon."""

    def __init__(self, cutoff: pd.Timestamp, levels: pd.Series) -> None:
        super().__init__(cutoff)
        self.levels = levels

    def _predict(
        self, panel: Panel, universe: list[str], horizons: tuple[int, ...]
    ) -> pd.DataFrame:
        missing = sorted(set(universe) - set(self.levels.dropna().index))
        if missing:
            raise ValueError(f"no observed history for {len(missing)} products: {missing[:5]}")
        rows = [
            (code, self.cutoff, h, self.cutoff + h * WEEK, float(self.levels[code]))
            for code in universe
            for h in horizons
        ]
        return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _last_observed(frame: pd.DataFrame) -> pd.Series:
    observed = frame.dropna(subset=["quantity"])
    return observed.sort_values("week", kind="mergesort").groupby("stock_code")["quantity"].last()


@dataclass(frozen=True)
class Naive(Recipe):
    """Most recent observed week; zero-transaction weeks are skipped, not read as 0."""

    name = "naive"

    def _fit(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> FittedRecipe:
        return _LevelForecast(panel.last_complete_week, _last_observed(panel.frame))


@dataclass(frozen=True)
class MovingAverage(Recipe):
    """Mean of the observed weeks among the last `window` calendar weeks.

    Falls back to `naive` for a product whose whole window is missing.
    """

    window: int = 4
    name = "ma4"

    def _fit(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> FittedRecipe:
        cutoff = panel.last_complete_week
        recent = panel.frame[panel.frame["week"] > cutoff - self.window * WEEK]
        means = recent.groupby("stock_code")["quantity"].mean()
        last = _last_observed(panel.frame)
        return _LevelForecast(cutoff, means.fillna(last).combine_first(last))


@dataclass(frozen=True)
class LightGBM(Recipe):
    """One global gradient-boosted model, direct multi-horizon (horizon is a feature).

    Parameters are fixed and not tuned (docs/design.md section 4). A Tweedie objective
    suits non-negative, often-zero weekly sales and predicts non-negative values.
    Training is single-threaded and deterministic so reruns give identical predictions.
    `leak` plants a deliberate leak for the failure demonstrations; it is never set in a
    real run.
    """

    objective: str = "tweedie"
    tweedie_variance_power: float = 1.5
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    seed: int = 0
    leak: Leak | None = None
    name = "lgbm"

    def _fit(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> FittedRecipe:
        cutoff = panel.last_complete_week
        rows = training_rows(panel, cutoff, universe, horizons, leak=self.leak)
        if rows.empty:
            raise ValueError(f"no training rows at {cutoff.date()}")
        params = {
            "objective": self.objective,
            "tweedie_variance_power": self.tweedie_variance_power,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbosity": -1,
        }
        dataset = lgb.Dataset(rows[FEATURE_COLUMNS], label=rows["target"], free_raw_data=True)
        model = lgb.train(params, dataset, num_boost_round=self.n_estimators)
        return _LightGBMForecast(cutoff, model, training_rows=len(rows))


class _LightGBMForecast(FittedRecipe):
    def __init__(self, cutoff: pd.Timestamp, model: lgb.Booster, training_rows: int) -> None:
        super().__init__(cutoff)
        self.model = model
        self.training_rows = training_rows

    def _predict(
        self, panel: Panel, universe: list[str], horizons: tuple[int, ...]
    ) -> pd.DataFrame:
        rows = inference_rows(panel, self.cutoff, universe, horizons)
        return pd.DataFrame(
            {
                "stock_code": rows["stock_code"],
                "cutoff_week": rows["origin_week"],
                "horizon": rows["horizon"],
                "target_week": rows["target_week"],
                "prediction": pd.Series(self.model.predict(rows[FEATURE_COLUMNS])).astype(
                    "float64"
                ),
            }
        )[PREDICTION_COLUMNS]


RECIPES: dict[str, type[Recipe]] = {"naive": Naive, "ma4": MovingAverage, "lgbm": LightGBM}
