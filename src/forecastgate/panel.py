"""Weekly sales panel.

Weeks are identified by their Monday. A week is complete when the data's first timestamp
is no later than that Monday 00:00 and its last timestamp is no earlier than the following
Monday 00:00. A complete week with no transactions store-wide is treated as missing
(assumed closure); every other in-range week without sales is 0 observed sales.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from forecastgate.errors import DataContractError
from forecastgate.ingest import IngestResult

WEEK = pd.Timedelta(days=7)

PANEL_SCHEMA = pa.DataFrameSchema(
    {
        "stock_code": pa.Column(str, nullable=False),
        "week": pa.Column("datetime64[us]", pa.Check(lambda s: s.dt.dayofweek == 0, name="monday")),
        "quantity": pa.Column("float64", pa.Check.ge(0), nullable=True),
        "price": pa.Column("float64", pa.Check.gt(0), nullable=True),
    },
    unique=["stock_code", "week"],
    strict=True,
    ordered=True,
)


@dataclass(frozen=True)
class Panel:
    # stock_code, week, quantity (NaN on zero-transaction weeks), price (median unit price
    # of the week's sales lines; NaN in weeks without sales)
    frame: pd.DataFrame
    first_sale_week: pd.Series  # stock_code -> week of first recorded sale
    weeks: pd.DatetimeIndex  # complete weeks, ascending
    zero_transaction_weeks: pd.DatetimeIndex

    @property
    def last_complete_week(self) -> pd.Timestamp:
        return self.weeks[-1]

    def fingerprint(self) -> str:
        """SHA-256 of the panel in a canonical byte form, for artifact identity."""
        payload = self.frame.to_csv(
            index=False, lineterminator="\n", float_format="%.6f", date_format="%Y-%m-%d"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def history(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Panel rows observable at the end of week `cutoff`."""
        return self.frame[self.frame["week"] <= cutoff]


def week_of(timestamps: pd.Series) -> pd.Series:
    day = timestamps.dt.normalize()
    monday: pd.Series = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")).astype("datetime64[us]")
    return monday


def complete_weeks(data_start: pd.Timestamp, data_end: pd.Timestamp) -> pd.DatetimeIndex:
    first_monday = data_start.normalize() - pd.Timedelta(days=data_start.dayofweek)
    if first_monday < data_start:
        first_monday += WEEK
    weeks = pd.date_range(first_monday, data_end - WEEK, freq="7D", unit="us")
    if len(weeks) == 0:
        raise DataContractError(f"no complete week between {data_start} and {data_end}")
    return weeks


def build_panel(data: IngestResult) -> Panel:
    weeks = complete_weeks(data.data_start, data.data_end)
    active = pd.DatetimeIndex(week_of(data.transactions["InvoiceDate"]).unique())
    zero_weeks = weeks.difference(active)

    sales = data.sales.assign(week=week_of(data.sales["InvoiceDate"]))
    first_sale = sales.groupby("StockCode")["week"].min().rename("first_sale_week")
    first_sale.index = first_sale.index.astype(str).rename("stock_code")
    weekly = (
        sales.groupby(["StockCode", "week"])
        .agg(quantity=("Quantity", "sum"), price=("Price", "median"))
        .reset_index()
        .rename(columns={"StockCode": "stock_code"})
    )

    grid = pd.MultiIndex.from_product(
        [first_sale.index, weeks], names=["stock_code", "week"]
    ).to_frame(index=False)
    grid = grid[grid["week"] >= grid["stock_code"].map(first_sale)]
    frame = grid.merge(weekly, on=["stock_code", "week"], how="left")
    frame["quantity"] = frame["quantity"].fillna(0).astype("float64")
    frame["price"] = frame["price"].astype("float64")
    frame.loc[frame["week"].isin(zero_weeks), "quantity"] = float("nan")
    frame = frame.sort_values(["stock_code", "week"], kind="mergesort").reset_index(drop=True)
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["week"] = frame["week"].astype("datetime64[us]")

    panel = Panel(
        frame=frame,
        first_sale_week=first_sale,
        weeks=weeks,
        zero_transaction_weeks=zero_weeks,
    )
    validate_panel(panel)
    return panel


def validate_panel(panel: Panel) -> None:
    try:
        PANEL_SCHEMA.validate(panel.frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].drop_duplicates()
        summary = "; ".join(f"{row.column}: {row.check}" for row in cases.itertuples())
        raise DataContractError(f"panel contract failed: {summary}") from exc
    missing = panel.frame["quantity"].isna()
    if not panel.frame.loc[missing, "week"].isin(panel.zero_transaction_weeks).all():
        raise DataContractError("panel has missing quantities outside zero-transaction weeks")
