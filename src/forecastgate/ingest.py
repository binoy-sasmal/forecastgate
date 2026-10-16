"""Fetch, verify and clean the Online Retail II transactions.

Order of operations: hash check, read both sheets, raw contract per sheet, overlap rule,
then exclusions in a fixed order so each exclusion count is attributable to one rule.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from forecastgate.config import DataSource
from forecastgate.errors import DataContractError

SHEETS = ("Year 2009-2010", "Year 2010-2011")
COLUMNS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)
# Product codes are five digits with an optional letter suffix. Everything else (POST, M,
# DOT, BANK CHARGES, gift vouchers, ...) is a fee, adjustment or service line.
PRODUCT_CODE = r"\d{5}[A-Za-z]*"

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Invoice": pa.Column(str, nullable=False),
        "StockCode": pa.Column(str, nullable=False),
        "Description": pa.Column(str, nullable=True),
        "Quantity": pa.Column("int64", nullable=False),
        "InvoiceDate": pa.Column("datetime64[us]", nullable=False),
        "Price": pa.Column("float64", nullable=False),
        "Customer ID": pa.Column("float64", nullable=True),
        "Country": pa.Column(str, nullable=True),
    },
    strict=True,
    ordered=True,
)


# Bump when parsing or coercion changes, so stale caches are never read.
CACHE_VERSION = 1


@dataclass(frozen=True)
class Exclusion:
    rows: int
    quantity: int  # sum of Quantity over the removed rows


@dataclass(frozen=True)
class CodeNormalisation:
    rows_changed: int
    raw_codes: int
    normalised_codes: int


@dataclass(frozen=True)
class SheetStats:
    name: str
    rows: int
    first: pd.Timestamp
    last: pd.Timestamp


@dataclass(frozen=True)
class IngestResult:
    transactions: pd.DataFrame  # every row after the overlap rule and code normalisation
    sales: pd.DataFrame  # rows that count towards the target
    exclusions: dict[str, Exclusion]  # rule name -> removed rows, in application order
    non_product_codes: dict[str, int]  # excluded code -> rows, among rows reaching that rule
    codes: CodeNormalisation
    sheets: tuple[SheetStats, ...]
    overlap_boundary: pd.Timestamp
    overlap_rows: int  # rows present in both sheets, counted once in transactions
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    data_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source: DataSource) -> Path:
    """Return the verified local archive, downloading it first when allowed."""
    if not source.path.exists():
        if source.url is None:
            raise DataContractError(f"{source.name} data missing at {source.path}")
        source.path.parent.mkdir(parents=True, exist_ok=True)
        partial = source.path.with_name(source.path.name + ".part")
        with urllib.request.urlopen(source.url, timeout=120) as response:
            partial.write_bytes(response.read())
        _verify_hash(partial, source)
        os.replace(partial, source.path)
    _verify_hash(source.path, source)
    return source.path


def _verify_hash(path: Path, source: DataSource) -> None:
    actual = sha256_file(path)
    if actual != source.sha256:
        raise DataContractError(
            f"{path} has SHA-256 {actual}, expected {source.sha256} for {source.name} data"
        )


def read_sheets(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise DataContractError(f"expected one workbook in {path}, found {members}")
        payload = archive.read(members[0])
    workbook = pd.ExcelFile(io.BytesIO(payload), engine="openpyxl")
    if tuple(workbook.sheet_names) != SHEETS:
        raise DataContractError(f"expected sheets {SHEETS}, found {workbook.sheet_names}")
    first, second = (_read_sheet(workbook, name) for name in SHEETS)
    return first, second


def read_sheets_cached(
    path: Path, data_hash: str, cache_dir: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsing the workbook dominates run time, so parsed sheets are cached as Parquet.

    The cache is derived data keyed by the verified archive hash and CACHE_VERSION. Cached
    sheets still pass the raw contract on every read, and deleting the cache is always safe.
    """
    if cache_dir is None:
        return read_sheets(path)
    files = [cache_dir / f"{data_hash}-v{CACHE_VERSION}-sheet{i}.parquet" for i in (1, 2)]
    if all(f.exists() for f in files):
        first, second = (
            validate_raw(pd.read_parquet(f), f"cached {name}")
            for f, name in zip(files, SHEETS, strict=True)
        )
        return first, second
    first, second = read_sheets(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for frame, target in zip((first, second), files, strict=True):
        partial = target.with_name(target.name + ".part")
        frame.to_parquet(partial, index=False)
        os.replace(partial, target)
    return first, second


def _read_sheet(workbook: pd.ExcelFile, name: str) -> pd.DataFrame:
    frame = workbook.parse(
        name,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
    )
    if tuple(frame.columns) != COLUMNS:
        raise DataContractError(f"sheet {name!r} has columns {list(frame.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(frame["InvoiceDate"]):
        raise DataContractError(f"sheet {name!r}: InvoiceDate is not a datetime column")
    frame["InvoiceDate"] = frame["InvoiceDate"].astype("datetime64[us]")
    if pd.api.types.is_integer_dtype(frame["Price"]):
        frame["Price"] = frame["Price"].astype("float64")
    if pd.api.types.is_integer_dtype(frame["Customer ID"]):
        frame["Customer ID"] = frame["Customer ID"].astype("float64")
    return validate_raw(frame, name)


def validate_raw(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        validated: pd.DataFrame = RAW_SCHEMA.validate(frame, lazy=True)
        return validated
    except SchemaErrors as exc:
        cases = exc.failure_cases[["column", "check"]].drop_duplicates()
        summary = "; ".join(f"{row.column}: {row.check}" for row in cases.itertuples())
        raise DataContractError(f"raw contract failed for {label}: {summary}") from exc


def merge_sheets(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the overlap rule: sheet 1 before sheet 2's first timestamp, sheet 2 after.

    The rows both sheets cover must be the same multiset; otherwise precedence would
    silently discard data, so the file is rejected. Returns the merged rows and the number
    of rows the sheets share.
    """
    boundary = second["InvoiceDate"].min()
    first_tail = first[first["InvoiceDate"] >= boundary]
    second_head = second[second["InvoiceDate"] <= first["InvoiceDate"].max()]
    if not _canonical(first_tail).equals(_canonical(second_head)):
        raise DataContractError(
            f"sheet overlap from {boundary} is not identical: "
            f"{len(first_tail)} rows in sheet 1, {len(second_head)} in sheet 2"
        )
    merged: pd.DataFrame = pd.concat(
        [first[first["InvoiceDate"] < boundary], second], ignore_index=True
    )
    return merged, len(second_head)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(COLUMNS), kind="mergesort").reset_index(drop=True)


def normalise_codes(transactions: pd.DataFrame) -> tuple[pd.DataFrame, CodeNormalisation]:
    """Strip whitespace and upper-case stock codes.

    The audit of the full file found lower-case codes (e.g. `15056p`) selling alongside
    their upper-case twins, usually with the same description, and one code with a trailing space.
    Left alone, each such product's history is split across two series.
    """
    raw = transactions["StockCode"]
    normalised = raw.str.strip().str.upper()
    stats = CodeNormalisation(
        rows_changed=int((raw != normalised).sum()),
        raw_codes=int(raw.nunique()),
        normalised_codes=int(normalised.nunique()),
    )
    return transactions.assign(StockCode=normalised), stats


def apply_exclusions(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Exclusion], dict[str, int]]:
    remaining = transactions
    counts: dict[str, Exclusion] = {}
    non_product: dict[str, int] = {}

    def drop(name: str, mask: pd.Series) -> None:
        nonlocal remaining, non_product
        removed = remaining[mask]
        counts[name] = Exclusion(rows=len(removed), quantity=int(removed["Quantity"].sum()))
        if name == "non_product_code":
            codes = removed["StockCode"].value_counts()
            non_product = {str(code): int(n) for code, n in sorted(codes.items())}
        remaining = remaining[~mask]

    drop("cancellation_invoice", remaining["Invoice"].str.startswith("C"))
    drop("quantity_not_positive", remaining["Quantity"] <= 0)
    drop("price_not_positive", remaining["Price"] <= 0)
    drop("non_product_code", ~remaining["StockCode"].str.fullmatch(PRODUCT_CODE))
    # Assumption, not a proven defect: identical rows may be repeated invoice lines.
    drop("identical_row", remaining.duplicated(keep="first"))
    return remaining.reset_index(drop=True), counts, non_product


def ingest(source: DataSource, *, cache_dir: Path | None = None) -> IngestResult:
    path = fetch(source)
    first, second = read_sheets_cached(path, source.sha256, cache_dir)
    merged, overlap_rows = merge_sheets(first, second)
    transactions, codes = normalise_codes(merged)
    sales, exclusions, non_product_codes = apply_exclusions(transactions)
    sheets = tuple(
        SheetStats(
            name=name,
            rows=len(frame),
            first=frame["InvoiceDate"].min(),
            last=frame["InvoiceDate"].max(),
        )
        for name, frame in zip(SHEETS, (first, second), strict=True)
    )
    return IngestResult(
        transactions=transactions,
        sales=sales,
        exclusions=exclusions,
        non_product_codes=non_product_codes,
        codes=codes,
        sheets=sheets,
        overlap_boundary=second["InvoiceDate"].min(),
        overlap_rows=overlap_rows,
        data_start=transactions["InvoiceDate"].min(),
        data_end=transactions["InvoiceDate"].max(),
        data_hash=source.sha256,
    )
