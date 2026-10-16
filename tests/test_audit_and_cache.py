"""Data audit determinism, the parse cache, and contract failures on real archive files."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from forecastgate import ingest as ingest_module
from forecastgate.cli import app
from forecastgate.config import DataSource
from forecastgate.errors import DataContractError
from forecastgate.ingest import SHEETS, ingest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "fixture" / "online_retail_II_fixture.zip"
runner = CliRunner()


def _fixture_source() -> DataSource:
    from forecastgate.config import FIXTURE_SHA256

    return DataSource(name="fixture", path=FIXTURE, sha256=FIXTURE_SHA256, url=None)


@pytest.fixture(scope="module")
def raw_sheets() -> dict[str, pd.DataFrame]:
    """The fixture's sheets as stored, without the reader's type coercion."""
    with zipfile.ZipFile(FIXTURE) as zf:
        payload = zf.read(zf.namelist()[0])
    sheets = pd.read_excel(io.BytesIO(payload), sheet_name=list(SHEETS), dtype=object)
    return {name: sheets[name] for name in SHEETS}


def _archive(
    tmp_path: Path, sheets: dict[str, pd.DataFrame], name: str = "modified.zip"
) -> DataSource:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("online_retail_II.xlsx", buffer.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # A correctly pinned hash, so the failure comes from the contract, not the hash check.
    return DataSource(name="modified", path=path, sha256=digest, url=None)


def _alter_overlap_row(sheets: dict[str, pd.DataFrame]) -> None:
    second = sheets[SHEETS[1]]
    first_end = pd.to_datetime(sheets[SHEETS[0]]["InvoiceDate"]).max()
    row = second.index[pd.to_datetime(second["InvoiceDate"]) <= first_end][0]
    second.loc[row, "Quantity"] = int(second.loc[row, "Quantity"]) + 1


def _text_quantity(sheets: dict[str, pd.DataFrame]) -> None:
    sheets[SHEETS[0]].loc[0, "Quantity"] = "twelve"


def _drop_column(sheets: dict[str, pd.DataFrame]) -> None:
    sheets[SHEETS[1]] = sheets[SHEETS[1]].drop(columns=["Price"])


def _missing_invoice_date(sheets: dict[str, pd.DataFrame]) -> None:
    sheets[SHEETS[0]].loc[0, "InvoiceDate"] = None


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_alter_overlap_row, "overlap"),
        (_text_quantity, "Quantity"),
        (_drop_column, "columns"),
        (_missing_invoice_date, "InvoiceDate"),
    ],
    ids=["overlap_mismatch", "text_quantity", "missing_column", "missing_date"],
)
def test_contract_rejects_modified_archive(
    tmp_path: Path,
    raw_sheets: dict[str, pd.DataFrame],
    corrupt: Callable[[dict[str, pd.DataFrame]], None],
    message: str,
) -> None:
    sheets = {name: frame.copy() for name, frame in raw_sheets.items()}
    corrupt(sheets)
    with pytest.raises(DataContractError, match=message):
        ingest(_archive(tmp_path, sheets))


def test_cache_is_written_then_read_instead_of_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source()
    parsed = ingest(source, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 2

    def no_parsing(path: Path) -> None:
        raise AssertionError("workbook parsed despite a valid cache")

    monkeypatch.setattr(ingest_module, "read_sheets", no_parsing)
    cached = ingest(source, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(parsed.transactions, cached.transactions)
    pd.testing.assert_frame_equal(parsed.sales, cached.sales)
    assert parsed.exclusions == cached.exclusions


def test_cache_from_another_version_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source()
    ingest(source, cache_dir=tmp_path)
    monkeypatch.setattr(ingest_module, "CACHE_VERSION", ingest_module.CACHE_VERSION + 1)
    ingest(source, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.parquet"))) == 4


def test_tampered_cache_still_fails_the_raw_contract(tmp_path: Path) -> None:
    source = _fixture_source()
    ingest(source, cache_dir=tmp_path)
    target = sorted(tmp_path.glob("*sheet1.parquet"))[0]
    frame = pd.read_parquet(target)
    frame["Quantity"] = frame["Quantity"].astype("float64") / 2
    frame.to_parquet(target, index=False)
    with pytest.raises(DataContractError, match="cached"):
        ingest(source, cache_dir=tmp_path)


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(REPO / "data" / "fixture", tmp_path / "data" / "fixture")
    shutil.copy(REPO / "gate.yaml", tmp_path / "gate.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_audit_is_deterministic_and_provenance_is_separate(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = checkout / "artifacts" / "fixture" / "reports"
    monkeypatch.setenv("FORECASTGATE_GIT_SHA", "first-commit")
    assert runner.invoke(app, ["reproduce", "--fixture"]).exit_code == 0
    first = (reports / "data_audit.md").read_bytes()

    # A second run on a later commit, served from the cache, must not change the audit.
    shutil.rmtree(checkout / "artifacts" / "fixture" / "predictions")
    monkeypatch.setenv("FORECASTGATE_GIT_SHA", "later-commit")
    assert runner.invoke(app, ["reproduce", "--fixture"]).exit_code == 0
    assert (reports / "data_audit.md").read_bytes() == first

    assert b"\r" not in first
    assert b"commit" not in first.lower().replace(b"(commit,", b"")
    provenance = json.loads((reports / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["data_audit.md"]["git_sha"] == "later-commit"
    assert set(provenance["data_audit.md"]["packages"]) >= {"pandas", "pandera"}


def test_fixture_audit_records_the_ingest_facts(checkout: Path) -> None:
    assert runner.invoke(app, ["reproduce", "--fixture"]).exit_code == 0
    audit = (checkout / "artifacts/fixture/reports/data_audit.md").read_text(encoding="utf-8")
    assert "Sheet 2 starts at 2010-12-01 08:45" in audit
    assert "| `POST` |" in audit
    assert "Zero-transaction weeks (assumed closures, treated as missing) | 2010-12-27 |" in audit
    assert "| Selected products | 60 |" in audit
