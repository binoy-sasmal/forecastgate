from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from forecastgate.ingest import (
    COLUMNS,
    IngestResult,
    SheetStats,
    apply_exclusions,
    normalise_codes,
)

Row = tuple[str, str, int, str, float]  # invoice, stock code, quantity, timestamp, price


def raw_frame(rows: Sequence[Row]) -> pd.DataFrame:
    """Transactions in the raw file's column layout and dtypes."""
    frame = pd.DataFrame(
        {
            "Invoice": [r[0] for r in rows],
            "StockCode": [r[1] for r in rows],
            "Description": ["ITEM"] * len(rows),
            "Quantity": pd.Series([r[2] for r in rows], dtype="int64"),
            "InvoiceDate": pd.to_datetime([r[3] for r in rows]).astype("datetime64[us]"),
            "Price": pd.Series([r[4] for r in rows], dtype="float64"),
            "Customer ID": pd.Series([12345.0] * len(rows), dtype="float64"),
            "Country": ["United Kingdom"] * len(rows),
        }
    )
    frame["Invoice"] = frame["Invoice"].astype(str)
    frame["StockCode"] = frame["StockCode"].astype(str)
    frame["Description"] = frame["Description"].astype(str)
    frame["Country"] = frame["Country"].astype(str)
    assert tuple(frame.columns) == COLUMNS
    return frame


def ingest_result(rows: Sequence[Row], *, start: str, end: str) -> IngestResult:
    """An IngestResult whose coverage interval is [start, end], independent of the rows."""
    transactions, codes = normalise_codes(raw_frame(rows))
    sales, exclusions, non_product_codes = apply_exclusions(transactions)
    sheet = SheetStats(
        name="synthetic",
        rows=len(transactions),
        first=transactions["InvoiceDate"].min(),
        last=transactions["InvoiceDate"].max(),
    )
    return IngestResult(
        transactions=transactions,
        sales=sales,
        exclusions=exclusions,
        non_product_codes=non_product_codes,
        codes=codes,
        sheets=(sheet,),
        overlap_boundary=pd.Timestamp(end),
        overlap_rows=0,
        data_start=pd.Timestamp(start),
        data_end=pd.Timestamp(end),
        data_hash="0" * 64,
    )
