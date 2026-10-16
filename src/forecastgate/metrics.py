"""Forecast accuracy metrics (docs/design.md section 4).

With e = |y - yhat| over a set S of scored rows:

- WAPE(S) = sum(e) / sum(y), undefined (None) when sum(y) = 0.
- Bias(S) = sum(yhat - y) / sum(y), undefined (None) when sum(y) = 0.
- MASE per series = mean(e) / scale, where scale is the series' mean absolute one-step
  change over its history up to the fold cutoff, using only consecutive calendar weeks
  that are both observed. Series with a zero or undefined scale are excluded and counted.

Undefined is returned as None, never as 0 or infinity, so a caller cannot mistake it for
a perfect or a terrible score.
"""

from __future__ import annotations

import pandas as pd

from forecastgate.panel import Panel


def wape(actual: pd.Series, prediction: pd.Series) -> float | None:
    total = float(actual.sum())
    if total == 0:
        return None
    return float((actual - prediction).abs().sum()) / total


def bias(actual: pd.Series, prediction: pd.Series) -> float | None:
    total = float(actual.sum())
    if total == 0:
        return None
    return float((prediction - actual).sum()) / total


def mase_scales(panel: Panel) -> pd.Series:
    """Per-product in-sample scale from a panel truncated at the fold cutoff."""
    frame = panel.frame.sort_values(["stock_code", "week"], kind="mergesort")
    step = frame.groupby("stock_code", sort=False)["quantity"].diff().abs()
    scales: pd.Series = step.groupby(frame["stock_code"], sort=False).mean()
    return scales


def mase(scored: pd.DataFrame, scales: pd.Series) -> tuple[float | None, int]:
    """Mean of per-series MASE over (fold, product) series; returns (value, excluded).

    `scored` needs columns fold, stock_code, actual, prediction; `scales` is indexed by
    (fold, stock_code).
    """
    errors = (
        scored.assign(error=(scored["actual"] - scored["prediction"]).abs())
        .groupby(["fold", "stock_code"])["error"]
        .mean()
    )
    scale = scales.reindex(errors.index)
    usable = scale > 0  # NaN compares False, so undefined scales are excluded too
    excluded = int((~usable).sum())
    if not usable.any():
        return None, excluded
    return float((errors[usable] / scale[usable]).mean()), excluded
