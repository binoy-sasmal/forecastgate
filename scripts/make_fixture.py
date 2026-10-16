"""Cut the committed test fixture from the real Online Retail II archive.

Usage: uv run python scripts/make_fixture.py <path to online+retail+ii.zip>

The fixture keeps the real file's structure (two sheets, identical overlap rows, mixed
StockCode types) so the offline run exercises the same ingest code as the full run.
Selection is deterministic; the output .xlsx is not byte-reproducible (openpyxl writes
timestamps into the workbook metadata), so the committed file's SHA-256 is the pin.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

SHEETS = ("Year 2009-2010", "Year 2010-2011")
# From October 2010 to the end of the file: keeps the sheet overlap, the zero-transaction
# week of 2010-12-27 and the partial final week, and leaves the six backtest folds of the
# full run (late May to November 2011) clear of the closure, under the full-run settings.
WINDOW_START = pd.Timestamp("2010-10-01")
WINDOW_END = pd.Timestamp("2011-12-10")  # exclusive; the file ends 2011-12-09 12:50
N_PRODUCTS = 60
# Non-product rows kept so the exclusion rules have real inputs. The sub-window spans the
# sheet overlap (2010-12-01 to 2010-12-09) so overlap handling sees them too.
NON_PRODUCT_CODES = ("POST", "M", "DOT", "C2", "D", "BANK CHARGES", "AMAZONFEE")
NON_PRODUCT_START = pd.Timestamp("2010-11-15")
NON_PRODUCT_END = pd.Timestamp("2010-12-20")
PRODUCT_CODE = r"\d{5}[A-Za-z]*"
OUT = Path("data/fixture/online_retail_II_fixture.zip")


def _read(path: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        (member,) = zf.namelist()
        payload = zf.read(member)
    sheets = pd.read_excel(
        io.BytesIO(payload),
        sheet_name=list(SHEETS),
        dtype={"Invoice": str, "StockCode": str, "Description": str},
    )
    return {name: sheets[name] for name in SHEETS}


def _choose_products(sheets: dict[str, pd.DataFrame]) -> list[str]:
    first, second = sheets[SHEETS[0]], sheets[SHEETS[1]]
    merged = pd.concat([first[first["InvoiceDate"] < second["InvoiceDate"].min()], second])
    window = merged[(merged["InvoiceDate"] >= WINDOW_START) & (merged["InvoiceDate"] < WINDOW_END)]
    sales = window[
        ~window["Invoice"].str.startswith("C")
        & (window["Quantity"] > 0)
        & window["StockCode"].str.fullmatch(PRODUCT_CODE)
    ]
    weeks = sales["InvoiceDate"].dt.to_period("W-SUN")
    stats = (
        sales.assign(week=weeks)
        .groupby("StockCode")
        .agg(weeks=("week", "nunique"), rows=("Invoice", "size"))
        .reset_index()
    )
    # Products sold in nearly every week, lightest first, to keep the fixture small.
    steady = stats[stats["weeks"] >= stats["weeks"].max() - 2]
    steady = steady.sort_values(["rows", "StockCode"], kind="mergesort")
    return [str(code) for code in steady["StockCode"].head(N_PRODUCTS)]


def _slice(frame: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    in_window = (frame["InvoiceDate"] >= WINDOW_START) & (frame["InvoiceDate"] < WINDOW_END)
    is_product = frame["StockCode"].isin(products)
    is_non_product = (
        frame["StockCode"].isin(NON_PRODUCT_CODES)
        & (frame["InvoiceDate"] >= NON_PRODUCT_START)
        & (frame["InvoiceDate"] < NON_PRODUCT_END)
    )
    out = frame[in_window & (is_product | is_non_product)].copy()
    # Restore the real file's mixed typing: purely numeric codes are stored as integers.
    numeric = out["StockCode"].str.fullmatch(r"\d+")
    out["StockCode"] = out["StockCode"].astype(object)
    out.loc[numeric, "StockCode"] = out.loc[numeric, "StockCode"].astype(int)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    sheets = _read(Path(argv[1]))
    products = _choose_products(sheets)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name in SHEETS:
            _slice(sheets[name], products).to_excel(writer, sheet_name=name, index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("online_retail_II.xlsx", buffer.getvalue())
    print(f"wrote {OUT} with {len(products)} products")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
