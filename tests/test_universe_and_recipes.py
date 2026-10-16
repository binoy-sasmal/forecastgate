from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from forecastgate.config import Settings, settings_for
from forecastgate.errors import DataContractError
from forecastgate.panel import WEEK, Panel
from forecastgate.recipes import MovingAverage, Naive, postprocess
from forecastgate.universe import select_universe

CUTOFF = pd.Timestamp("2010-07-05")


def _panel(series: dict[str, list[float]], first_week: str = "2010-01-04") -> Panel:
    weeks = pd.date_range(first_week, periods=max(map(len, series.values())), freq="7D")
    rows = []
    first_sale = {}
    for code, values in series.items():
        offset = len(weeks) - len(values)
        first_sale[code] = weeks[offset]
        rows += [(code, weeks[offset + i], v) for i, v in enumerate(values)]
    frame = pd.DataFrame(rows, columns=["stock_code", "week", "quantity"])
    nan_weeks = frame.loc[frame["quantity"].isna(), "week"].unique()
    return Panel(
        frame=frame,
        first_sale_week=pd.Series(first_sale),
        weeks=pd.DatetimeIndex(weeks),
        zero_transaction_weeks=pd.DatetimeIndex(nan_weeks),
    )


def _settings(*, universe_size: int = 200, min_universe_size: int = 50) -> Settings:
    base = settings_for(fixture=True, root=Path("."))
    return replace(base, universe_size=universe_size, min_universe_size=min_universe_size)


def test_universe_requires_history_and_ranks_with_ties_by_code() -> None:
    panel = _panel(
        {
            "B": [1.0] * 27,  # first sale 26 weeks before the cutoff: eligible
            "A": [1.0] * 27,  # same total as B: tie broken by code
            "C": [100.0] * 26,  # too young despite the volume
            "D": [5.0] * 27,
        }
    )
    cutoff = panel.weeks[-1]
    universe = select_universe(panel, cutoff, _settings(universe_size=2, min_universe_size=1))
    assert universe == ["A", "D"]


def test_universe_ignores_data_after_the_cutoff() -> None:
    history = {"A": [1.0] * 30, "B": [2.0] * 30}
    future_spike = {"A": [1.0] * 27 + [1000.0] * 3, "B": [2.0] * 30}
    cutoff = pd.date_range("2010-01-04", periods=30, freq="7D")[26]
    settings = _settings(universe_size=1, min_universe_size=1)
    assert select_universe(_panel(history), cutoff, settings) == ["B"]
    assert select_universe(_panel(future_spike), cutoff, settings) == ["B"]


def test_too_small_universe_fails_the_contract() -> None:
    panel = _panel({"A": [1.0] * 27})
    with pytest.raises(DataContractError, match="minimum"):
        select_universe(panel, panel.weeks[-1], _settings(min_universe_size=2))


def test_naive_skips_missing_weeks() -> None:
    panel = _panel({"A": [3.0, 4.0, float("nan")]})
    cutoff = panel.weeks[-1]
    forecast = Naive().fit(panel.history(cutoff), cutoff).predict(["A"], (1, 2))
    assert forecast["prediction"].tolist() == [4.0, 4.0]
    assert forecast["target_week"].tolist() == [cutoff + WEEK, cutoff + 2 * WEEK]


def test_moving_average_uses_observed_weeks_and_falls_back_to_naive() -> None:
    nan = float("nan")
    panel = _panel({"A": [9.0, 2.0, 4.0, nan, 6.0], "B": [5.0, nan, nan, nan, nan]})
    cutoff = panel.weeks[-1]
    forecast = MovingAverage().fit(panel.history(cutoff), cutoff).predict(["A", "B"], (1,))
    assert forecast.set_index("stock_code")["prediction"].to_dict() == {"A": 4.0, "B": 5.0}


def test_fit_refuses_history_past_the_cutoff() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="past cutoff"):
        Naive().fit(panel.frame, panel.weeks[-2])


def test_postprocess_clips_negative_forecasts() -> None:
    frame = pd.DataFrame({"prediction": [-1.5, 0.0, 2.0]})
    assert postprocess(frame)["prediction"].tolist() == [0.0, 0.0, 2.0]


def test_config_hash_tracks_parameters() -> None:
    assert MovingAverage().config_hash() == MovingAverage().config_hash()
    assert MovingAverage(window=8).config_hash() != MovingAverage().config_hash()
    assert Naive().config_hash() != MovingAverage().config_hash()
