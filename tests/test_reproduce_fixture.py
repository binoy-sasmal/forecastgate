"""End-to-end runs of the official command on the committed fixture, offline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from forecastgate import ingest as ingest_module
from forecastgate.cli import app
from forecastgate.errors import DataContractError
from forecastgate.ingest import Exclusion

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal copy of the repository layout, so runs never touch the real artifacts."""
    fixture = REPO / "data" / "fixture"
    shutil.copytree(fixture, tmp_path / "data" / "fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORECASTGATE_GIT_SHA", "test")
    return tmp_path


def test_fixture_run_publishes_then_rerun_is_unchanged(checkout: Path) -> None:
    first = runner.invoke(app, ["reproduce", "--fixture"])
    assert first.exit_code == 0, first.output
    assert "publication       published" in first.output
    assert "universe          60 products" in first.output

    second = runner.invoke(app, ["reproduce", "--fixture"])
    assert second.exit_code == 0, second.output
    assert "publication       unchanged" in second.output

    current = (checkout / "artifacts/fixture/predictions/CURRENT").read_text().strip()
    published = checkout / "artifacts/fixture/predictions" / current / "predictions.csv"
    assert len(published.read_text().splitlines()) == 1 + 60 * 4


def test_changed_preparation_yields_a_new_artifact_not_a_refusal(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["reproduce", "--fixture"]).exit_code == 0
    # Simulate a data-preparation change: one more exclusion rule alters the panel.
    original = ingest_module.apply_exclusions

    def stricter(
        transactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Exclusion], dict[str, int]]:
        return original(transactions[transactions["Quantity"] < 100])

    monkeypatch.setattr(ingest_module, "apply_exclusions", stricter)
    result = runner.invoke(app, ["reproduce", "--fixture"])
    assert result.exit_code == 0, result.output
    assert "publication       published" in result.output
    artifacts = [p for p in (checkout / "artifacts/fixture/predictions").iterdir() if p.is_dir()]
    assert len([p for p in artifacts if not p.name.startswith(".")]) == 2


def test_corrupted_fixture_is_rejected_and_nothing_is_written(checkout: Path) -> None:
    archive = checkout / "data/fixture/online_retail_II_fixture.zip"
    payload = bytearray(archive.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    archive.write_bytes(bytes(payload))

    result = runner.invoke(app, ["reproduce", "--fixture"])
    assert result.exit_code == DataContractError.exit_code
    assert "SHA-256" in result.output
    assert not (checkout / "artifacts").exists()
