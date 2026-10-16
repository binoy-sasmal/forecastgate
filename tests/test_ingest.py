from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from forecastgate.config import DataSource
from forecastgate.errors import DataContractError
from forecastgate.ingest import (
    SHEETS,
    apply_exclusions,
    fetch,
    merge_sheets,
    normalise_codes,
    read_sheets,
    validate_raw,
)
from tests.conftest import raw_frame


def _source(path: Path, sha256: str) -> DataSource:
    return DataSource(name="test", path=path, sha256=sha256, url=None)


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"not the pinned archive")
    with pytest.raises(DataContractError, match="SHA-256"):
        fetch(_source(archive, "0" * 64))


def test_missing_offline_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="missing"):
        fetch(_source(tmp_path / "absent.zip", "0" * 64))


def test_raw_contract_rejects_non_integer_quantity() -> None:
    frame = raw_frame([("1", "10001", 1, "2010-01-04 10:00", 1.0)])
    frame["Quantity"] = frame["Quantity"].astype("float64") + 0.5
    with pytest.raises(DataContractError, match="Quantity"):
        validate_raw(frame, "test")


def test_raw_contract_rejects_missing_stock_code() -> None:
    frame = raw_frame([("1", "10001", 1, "2010-01-04 10:00", 1.0)])
    frame.loc[0, "StockCode"] = None
    with pytest.raises(DataContractError, match="StockCode"):
        validate_raw(frame, "test")


def test_identical_overlap_takes_each_period_from_one_sheet() -> None:
    first = raw_frame(
        [
            ("1", "10001", 1, "2010-11-30 10:00", 1.0),
            ("2", "10001", 2, "2010-12-02 10:00", 1.0),
        ]
    )
    second = raw_frame(
        [
            ("2", "10001", 2, "2010-12-02 10:00", 1.0),
            ("3", "10001", 3, "2010-12-20 10:00", 1.0),
        ]
    )
    merged, shared = merge_sheets(first, second)
    assert list(merged["Invoice"]) == ["1", "2", "3"]
    assert shared == 1


def test_differing_overlap_is_rejected() -> None:
    first = raw_frame([("2", "10001", 2, "2010-12-02 10:00", 1.0)])
    second = raw_frame([("2", "10001", 5, "2010-12-02 10:00", 1.0)])
    with pytest.raises(DataContractError, match="overlap"):
        merge_sheets(first, second)


def test_exclusions_are_counted_in_order() -> None:
    frame = raw_frame(
        [
            ("C1", "10001", -1, "2010-01-04 10:00", 1.0),  # cancellation
            ("2", "10001", 0, "2010-01-04 10:00", 1.0),  # quantity not positive
            ("3", "10001", 1, "2010-01-04 10:00", 0.0),  # price not positive
            ("4", "POST", 1, "2010-01-04 10:00", 18.0),  # non-product
            ("5", "10001", 2, "2010-01-04 10:00", 1.0),
            ("5", "10001", 2, "2010-01-04 10:00", 1.0),  # identical row
            ("6", "85123A", 3, "2010-01-04 10:00", 2.5),
        ]
    )
    sales, counts, non_product = apply_exclusions(frame)
    assert {rule: excluded.rows for rule, excluded in counts.items()} == {
        "cancellation_invoice": 1,
        "quantity_not_positive": 1,
        "price_not_positive": 1,
        "non_product_code": 1,
        "identical_row": 1,
    }
    assert counts["identical_row"].quantity == 2
    assert counts["cancellation_invoice"].quantity == -1
    assert non_product == {"POST": 1}
    assert list(sales["Invoice"]) == ["5", "6"]


def test_reader_coerces_mixed_stock_code_types(tmp_path: Path) -> None:
    sheet = raw_frame([("1", "85123A", 1, "2010-01-04 10:00", 1.0)])
    sheet["StockCode"] = sheet["StockCode"].astype(object)
    numeric = sheet.copy()
    numeric["StockCode"] = [85048]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name=SHEETS[0], index=False)
        numeric.assign(InvoiceDate=pd.Timestamp("2011-01-04 10:00")).to_excel(
            writer, sheet_name=SHEETS[1], index=False
        )
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("online_retail_II.xlsx", buffer.getvalue())

    first, second = read_sheets(archive)
    assert first["StockCode"].tolist() == ["85123A"]
    assert second["StockCode"].tolist() == ["85048"]


def test_codes_are_stripped_and_upper_cased_into_one_series() -> None:
    frame = raw_frame(
        [
            ("1", "15056P", 1, "2010-01-04 10:00", 1.0),
            ("2", "15056p", 2, "2010-01-05 10:00", 1.0),
            ("3", "47503J ", 3, "2010-01-06 10:00", 1.0),
        ]
    )
    normalised, stats = normalise_codes(frame)
    assert normalised["StockCode"].tolist() == ["15056P", "15056P", "47503J"]
    assert (stats.rows_changed, stats.raw_codes, stats.normalised_codes) == (2, 3, 2)
