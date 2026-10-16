"""Serving refuses stale, future-dated, tampered and inconsistent artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from forecastgate.api import create_app
from forecastgate.cli import app as cli
from forecastgate.errors import PredictionContractError
from forecastgate.panel import WEEK
from forecastgate.publish import CURRENT_FILE, ArtifactIdentity, PublishResult, publish
from forecastgate.recipes import PREDICTION_COLUMNS
from forecastgate.serving import ServingRefusedError, load_current

CUTOFF = pd.Timestamp("2011-03-21")  # a Monday; its data runs through Sunday 2011-03-27
DATA_THROUGH = date(2011, 3, 27)
UNIVERSE = ["10001", "85123A"]
HORIZONS = (1, 2)
MAX_AGE = 7


def _identity(version: str = "ma4@v1") -> ArtifactIdentity:
    return ArtifactIdentity(
        cutoff_week=f"{CUTOFF:%Y-%m-%d}",
        model_version=version,
        recipe_config_hash="c" * 64,
        data_hash="d" * 64,
        input_hash="e" * 64,
    )


def _predictions(value: float = 2.5) -> pd.DataFrame:
    rows = [
        (code, CUTOFF, h, CUTOFF + h * WEEK, value + i)
        for i, code in enumerate(UNIVERSE)
        for h in HORIZONS
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _publish(root: Path, predictions: pd.DataFrame, version: str = "ma4@v1") -> PublishResult:
    return publish(
        root, _identity(version), predictions, universe=UNIVERSE, horizons=HORIZONS, git_sha="t"
    )


@pytest.fixture
def published(tmp_path: Path) -> Path:
    _publish(tmp_path, _predictions())
    return tmp_path


def _client(root: Path, as_of: date) -> TestClient:
    return TestClient(create_app(root, as_of=as_of, max_age_days=MAX_AGE))


# --- serving a valid artifact --------------------------------------------------------------


def test_health_and_predictions_for_a_fresh_artifact(published: Path) -> None:
    client = _client(published, DATA_THROUGH + timedelta(days=1))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == "ma4@v1"
    assert health.json()["age_days"] == 1

    response = client.get("/predictions/85123A")
    assert response.status_code == 200
    body = response.json()
    assert [f["horizon"] for f in body["forecasts"]] == [1, 2]
    assert body["forecasts"][0] == {"horizon": 1, "target_week": "2011-03-28", "prediction": 3.5}

    assert client.get("/predictions/99999").status_code == 404


# --- freshness ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days_after", "served"),
    [(0, True), (MAX_AGE, True), (MAX_AGE + 1, False), (-1, False)],
    ids=["same_day", "oldest_allowed", "stale", "future_dated"],
)
def test_freshness_bounds(published: Path, days_after: int, served: bool) -> None:
    as_of = DATA_THROUGH + timedelta(days=days_after)
    if served:
        assert load_current(published, as_of=as_of, max_age_days=MAX_AGE).age_days == days_after
    else:
        with pytest.raises(ServingRefusedError, match="stale" if days_after > 0 else "later than"):
            load_current(published, as_of=as_of, max_age_days=MAX_AGE)


def test_stale_artifact_is_a_503_with_the_reason(published: Path) -> None:
    client = _client(published, DATA_THROUGH + timedelta(days=30))
    health = client.get("/health")
    assert health.status_code == 503
    assert "stale" in health.json()["reason"]
    response = client.get("/predictions/10001")
    assert response.status_code == 503
    assert "stale" in response.json()["detail"]


# --- integrity ---------------------------------------------------------------------------------


def _current_dir(root: Path) -> Path:
    return root / (root / CURRENT_FILE).read_text().strip()


def test_tampering_after_start_up_is_refused_on_the_next_request(published: Path) -> None:
    client = _client(published, DATA_THROUGH)
    assert client.get("/predictions/10001").status_code == 200
    target = _current_dir(published) / "predictions.csv"
    target.write_bytes(target.read_bytes().replace(b"2.500000", b"9.500000"))
    response = client.get("/predictions/10001")
    assert response.status_code == 503
    assert "checksum" in response.json()["detail"]


def test_invalid_content_is_refused_even_with_a_matching_checksum(published: Path) -> None:
    # Defence in depth: someone rewrites the file and its manifest checksum consistently.
    artifact = _current_dir(published)
    target = artifact / "predictions.csv"
    target.write_bytes(target.read_bytes().replace(b"2.500000", b"-1.000000"))
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"]["predictions.csv"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ServingRefusedError, match="negative"):
        load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE)


def test_unreadable_manifest_is_refused(published: Path) -> None:
    (_current_dir(published) / "manifest.json").write_text("{not json")
    with pytest.raises(ServingRefusedError, match="manifest is unreadable"):
        load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE)


def test_manifest_from_another_artifact_is_refused(published: Path) -> None:
    copy = published / "0000000000000000"
    shutil.copytree(_current_dir(published), copy)
    (published / CURRENT_FILE).write_text("0000000000000000\n")
    with pytest.raises(ServingRefusedError, match="another artifact"):
        load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE)


@pytest.mark.parametrize("pointer", ["", ".staging", "missing0000000000"])
def test_current_must_name_a_committed_artifact(published: Path, pointer: str) -> None:
    (published / ".staging").mkdir(exist_ok=True)
    (published / CURRENT_FILE).write_text(pointer + "\n")
    with pytest.raises(ServingRefusedError, match="not a published artifact"):
        load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE)


def test_nothing_published_yet(tmp_path: Path) -> None:
    response = _client(tmp_path, DATA_THROUGH).get("/health")
    assert response.status_code == 503
    assert response.json()["reason"] == "no artifact has been published"


# --- publication failures, as seen by serving --------------------------------------------------


def test_failed_publication_keeps_serving_the_previous_artifact(published: Path) -> None:
    before = load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE).identity
    with pytest.raises(PredictionContractError):
        _publish(published, _predictions().iloc[1:], version="ma4@v2")  # incomplete grid
    after = load_current(published, as_of=DATA_THROUGH, max_age_days=MAX_AGE)
    assert after.identity == before
    assert after.manifest["identity"]["model_version"] == "ma4@v1"


def test_serve_check_exits_11_when_serving_would_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    predictions_dir = tmp_path / "artifacts" / "predictions"
    _publish(predictions_dir, _predictions())
    runner = CliRunner()
    ok = runner.invoke(
        cli, ["serve-check", "--fixture", "--artifacts-dir", "artifacts", "--as-of", "2011-03-28"]
    )
    assert ok.exit_code == 0, ok.output
    assert "ma4@v1" in ok.output
    stale = runner.invoke(
        cli, ["serve-check", "--fixture", "--artifacts-dir", "artifacts", "--as-of", "2011-05-01"]
    )
    assert stale.exit_code == ServingRefusedError.exit_code
    assert "stale" in stale.output
