"""Hand-computed checks of feature alignment, label eligibility and fold cutoffs."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from forecastgate.config import settings_for
from forecastgate.errors import DataContractError
from forecastgate.features import (
    FEATURE_COLUMNS,
    inference_rows,
    label_violations,
    origin_features,
    training_rows,
)
from forecastgate.folds import backtest_cutoffs
from forecastgate.panel import WEEK, Panel

NAN = float("nan")
WEEKS = pd.date_range("2010-01-04", periods=10, freq="7D", unit="us")  # w0 .. w9, Mondays
# Product A: w4 is a zero-transaction week (missing), w6 recorded no sales (0, no price).
A_QUANTITY = [1.0, 2.0, 3.0, 4.0, NAN, 6.0, 0.0, 8.0, 9.0, 10.0]
A_PRICE = [1.0, 1.0, 2.0, 2.0, NAN, 3.0, NAN, 3.0, 3.0, 3.0]
# Product B: first sale in w2, so it has no rows before.
B_QUANTITY = [5.0, 5.0, NAN, 5.0, 5.0, 5.0, 5.0, 5.0]


def _panel() -> Panel:
    rows = [("A", w, q, p) for w, q, p in zip(WEEKS, A_QUANTITY, A_PRICE, strict=True)]
    rows += [
        ("B", w, q, NAN if math.isnan(q) else 2.5)
        for w, q in zip(WEEKS[2:], B_QUANTITY, strict=True)
    ]
    frame = pd.DataFrame(rows, columns=["stock_code", "week", "quantity", "price"])
    return Panel(
        frame=frame,
        first_sale_week=pd.Series({"A": WEEKS[0], "B": WEEKS[2]}),
        weeks=pd.DatetimeIndex(WEEKS),
        zero_transaction_weeks=pd.DatetimeIndex([WEEKS[4]]),
    )


def _at(features: pd.DataFrame, code: str, week: pd.Timestamp) -> pd.Series:
    row = features[(features["stock_code"] == code) & (features["week"] == week)]
    assert len(row) == 1
    return row.iloc[0]


def test_lags_count_calendar_weeks_back_from_the_origin() -> None:
    panel = _panel()
    row = _at(origin_features(panel.frame, panel.first_sale_week), "A", WEEKS[5])
    assert row["lag_0"] == 6.0  # the origin week itself is known
    assert math.isnan(row["lag_1"])  # w4 is missing, not 0
    assert [row[f"lag_{k}"] for k in (2, 3, 4, 5)] == [4.0, 3.0, 2.0, 1.0]
    assert math.isnan(row["lag_6"]) and math.isnan(row["lag_7"])  # before the first sale


def test_windows_end_at_the_origin_and_skip_missing_weeks() -> None:
    panel = _panel()
    row = _at(origin_features(panel.frame, panel.first_sale_week), "A", WEEKS[5])
    # mean_4 covers w2..w5 = 3, 4, missing, 6.
    assert row["mean_4"] == pytest.approx(13 / 3)
    # Population std of 3, 4, 6.
    assert row["std_4"] == pytest.approx(
        math.sqrt(((3 - 13 / 3) ** 2 + (4 - 13 / 3) ** 2 + (6 - 13 / 3) ** 2) / 3)
    )
    # mean_8 covers w0..w5 (only six weeks exist) = 1, 2, 3, 4, missing, 6.
    assert row["mean_8"] == pytest.approx(16 / 5)
    assert row["weeks_since_first_sale"] == 5.0


def test_zero_share_and_price_use_observed_weeks_only() -> None:
    panel = _panel()
    row = _at(origin_features(panel.frame, panel.first_sale_week), "A", WEEKS[6])
    # Observed weeks w0..w6: 1, 2, 3, 4, 6, 0 (w4 missing): one zero in six.
    assert row["zero_share_8"] == pytest.approx(1 / 6)
    # Prices w0..w6: 1, 1, 2, 2, 3 (w4 and w6 have no sales): median 2.
    assert row["price_median_8"] == 2.0


def test_std_needs_two_observed_weeks() -> None:
    panel = _panel()
    features = origin_features(panel.frame, panel.first_sale_week)
    assert math.isnan(_at(features, "A", WEEKS[0])["std_4"])
    assert _at(features, "A", WEEKS[1])["std_4"] == pytest.approx(0.5)


def test_planted_future_window_shifts_mean_4_by_one_week() -> None:
    panel = _panel()
    leaky = _at(
        origin_features(panel.frame, panel.first_sale_week, leak="future_window"), "A", WEEKS[5]
    )
    # The leaky window is w3..w6 = 4, missing, 6, 0: it reads y[t + 1].
    assert leaky["mean_4"] == pytest.approx(10 / 3)


def test_non_contiguous_series_is_rejected() -> None:
    panel = _panel()
    gappy = panel.frame.drop(index=3)
    with pytest.raises(ValueError, match="contiguous"):
        origin_features(gappy, panel.first_sale_week)


def test_training_rows_only_use_labels_observable_at_the_cutoff() -> None:
    panel = _panel()
    cutoff = WEEKS[7]
    rows = training_rows(panel, cutoff, ["A"], (1, 2))
    assert label_violations(rows, cutoff).empty
    assert (rows["origin_week"] + rows["horizon"] * WEEK <= cutoff).all()
    # h=1: origins w0..w6 target w1..w7; h=2: origins w0..w5 target w2..w7.
    # The zero-transaction target week w4 is skipped in both.
    assert len(rows[rows["horizon"] == 1]) == 6
    assert len(rows[rows["horizon"] == 2]) == 5
    assert WEEKS[4] not in set(rows["target_week"])
    one = rows[(rows["origin_week"] == WEEKS[5]) & (rows["horizon"] == 2)].iloc[0]
    assert one["target"] == 8.0  # y[w7]
    assert one["lag_0"] == 6.0  # features are those of origin w5


def test_planted_origin_filter_produces_label_violations() -> None:
    panel = _panel()
    cutoff = WEEKS[7]
    leaky = training_rows(panel, cutoff, ["A"], (1, 2), leak="origin_filter")
    late = label_violations(leaky, cutoff)
    # Origins w6 (h=2) and w7 (h=1, h=2) reach past the cutoff.
    assert set(zip(late["origin_week"], late["horizon"], strict=True)) == {
        (WEEKS[6], 2),
        (WEEKS[7], 1),
        (WEEKS[7], 2),
    }


def test_inference_rows_use_the_cutoff_as_origin() -> None:
    panel = _panel()
    rows = inference_rows(panel, WEEKS[9], ["A", "B"], (1, 2, 3, 4))
    assert len(rows) == 8
    assert (rows["origin_week"] == WEEKS[9]).all()
    assert list(rows.columns[4:]) == [c for c in FEATURE_COLUMNS if c != "horizon"]
    a1 = rows[(rows["stock_code"] == "A") & (rows["horizon"] == 1)].iloc[0]
    assert a1["lag_0"] == 10.0
    assert a1["target_week"] == WEEKS[9] + WEEK
    assert a1["target_week_of_year"] == float(a1["target_week"].isocalendar().week)


def test_backtest_cutoffs_leave_room_for_the_longest_horizon() -> None:
    settings = settings_for(fixture=True, root=Path("."))
    weeks = pd.date_range("2010-06-07", periods=42, freq="7D", unit="us")
    cutoffs = backtest_cutoffs(pd.DatetimeIndex(weeks), settings)
    assert cutoffs[-1] == weeks[-5]
    assert cutoffs[0] == weeks[-25]
    assert all(b - a == 4 * WEEK for a, b in pairwise(cutoffs))


def test_backtest_cutoffs_fail_when_history_is_too_short() -> None:
    settings = replace(settings_for(fixture=True, root=Path(".")), n_folds=20)
    weeks = pd.date_range("2010-06-07", periods=42, freq="7D", unit="us")
    with pytest.raises(DataContractError, match="folds"):
        backtest_cutoffs(pd.DatetimeIndex(weeks), settings)
