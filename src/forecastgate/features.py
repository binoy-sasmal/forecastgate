"""Features at a forecast origin, and the training and inference rows built from them.

Notation: origin t is a complete week, horizon h is in weeks, target week is t + h, and
y[w] is recorded sales in week w. At origin t, everything up to and including week t is
known; week t + 1 onwards is not.

Every feature for origin t is a function of panel rows with week <= t only. Training rows
at cutoff T need `t + h <= T`, so their targets also come from rows with week <= T.
`leak` deliberately breaks one of those rules so the leakage tests can prove they catch it.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from forecastgate.panel import WEEK, Panel

Leak = Literal["future_window", "origin_filter"]

LAGS = range(8)  # lag_k = y[t - k]; lag_0 is the origin week itself
FEATURE_COLUMNS = [
    *(f"lag_{k}" for k in LAGS),
    "mean_4",
    "std_4",
    "mean_8",
    "std_8",
    "zero_share_8",
    "price_median_8",
    "weeks_since_first_sale",
    "horizon",
    "target_week_of_year",
]
KEY_COLUMNS = ["stock_code", "origin_week", "horizon", "target_week"]


def _check_contiguous(frame: pd.DataFrame) -> None:
    """Positional windows equal calendar windows only if each product has every week."""
    gaps = frame.groupby("stock_code")["week"].diff().dropna()
    if (gaps != WEEK).any():
        raise ValueError("panel rows are not contiguous weekly series per product")


def _window(values: pd.Series, codes: pd.Series, window: int, stat: str) -> pd.Series:
    """Per-product trailing window over `window` calendar weeks ending at each row."""

    def apply(series: pd.Series) -> pd.Series:
        rolling = series.rolling(window, min_periods=2 if stat == "std" else 1)
        if stat == "mean":
            return rolling.mean()
        if stat == "std":
            return rolling.std(ddof=0)
        return rolling.median()

    result: pd.Series = values.groupby(codes, sort=False).transform(apply)
    return result


def origin_features(
    frame: pd.DataFrame, first_sale_week: pd.Series, *, leak: Leak | None = None
) -> pd.DataFrame:
    """Horizon-independent features for every (stock_code, week) row of `frame`.

    Windows count calendar weeks ending at the origin and skip missing (zero-transaction)
    weeks: a mean or median needs one observed week, a standard deviation (population,
    ddof=0) needs two.
    """
    frame = frame.sort_values(["stock_code", "week"], kind="mergesort").reset_index(drop=True)
    _check_contiguous(frame)
    codes = frame["stock_code"]
    quantity = frame["quantity"]

    features = frame[["stock_code", "week"]].copy()
    for k in LAGS:
        features[f"lag_{k}"] = quantity.groupby(codes, sort=False).shift(k)
    features["mean_4"] = _window(quantity, codes, 4, "mean")
    if leak == "future_window":
        # Planted leak: the window covers t - 2 to t + 1 instead of t - 3 to t.
        features["mean_4"] = features["mean_4"].groupby(codes, sort=False).shift(-1)
    features["std_4"] = _window(quantity, codes, 4, "std")
    features["mean_8"] = _window(quantity, codes, 8, "mean")
    features["std_8"] = _window(quantity, codes, 8, "std")
    is_zero = (quantity == 0).astype("float64").where(quantity.notna())
    features["zero_share_8"] = _window(is_zero, codes, 8, "mean")
    features["price_median_8"] = _window(frame["price"], codes, 8, "median")
    first = codes.map(first_sale_week)
    features["weeks_since_first_sale"] = ((frame["week"] - first).dt.days / 7).astype("float64")
    return features


def _with_horizon(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_week = features["week"] + horizon * WEEK
    return features.assign(
        horizon=horizon,
        target_week=target_week,
        target_week_of_year=target_week.dt.isocalendar().week.astype("float64"),
    ).rename(columns={"week": "origin_week"})


def training_rows(
    panel: Panel,
    cutoff: pd.Timestamp,
    universe: list[str],
    horizons: tuple[int, ...],
    *,
    leak: Leak | None = None,
) -> pd.DataFrame:
    """Labelled rows for fitting at `cutoff`: every origin t and horizon h with t + h <= T.

    Rows whose target week is a zero-transaction week are skipped (decided from data
    up to T, because their target week is <= T).
    """
    history = panel.history(cutoff)
    in_universe = history[history["stock_code"].isin(universe)]
    features = origin_features(in_universe, panel.first_sale_week, leak=leak)
    # Planted leak: labels taken from the full panel, filtered on the origin only.
    label_source = panel.frame if leak == "origin_filter" else history
    labels = label_source[["stock_code", "week", "quantity"]].rename(
        columns={"week": "target_week", "quantity": "target"}
    )

    parts = []
    for h in horizons:
        rows = _with_horizon(features, h).merge(labels, on=["stock_code", "target_week"])
        if leak != "origin_filter":
            rows = rows[rows["target_week"] <= cutoff]
        parts.append(rows.dropna(subset=["target"]))
    out = pd.concat(parts, ignore_index=True)
    return _ordered(out, with_target=True)


def inference_rows(
    panel: Panel, cutoff: pd.Timestamp, universe: list[str], horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Unlabelled rows at origin `cutoff` for every universe product and horizon."""
    history = panel.history(cutoff)
    features = origin_features(history[history["stock_code"].isin(universe)], panel.first_sale_week)
    at_origin = features[features["week"] == cutoff]
    missing = sorted(set(universe) - set(at_origin["stock_code"]))
    if missing:
        raise ValueError(f"no panel row at {cutoff.date()} for {missing[:5]}")
    out = pd.concat([_with_horizon(at_origin, h) for h in horizons], ignore_index=True)
    return _ordered(out, with_target=False)


def _ordered(rows: pd.DataFrame, *, with_target: bool) -> pd.DataFrame:
    columns = KEY_COLUMNS + [c for c in FEATURE_COLUMNS if c != "horizon"]
    if with_target:
        columns.append("target")
    return rows[columns].sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def label_violations(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Training rows whose label was not observable at `cutoff` (should always be empty)."""
    late = rows["origin_week"] + rows["horizon"] * WEEK > cutoff
    return rows[late | (rows["target_week"] > cutoff)]
