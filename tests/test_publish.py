from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from forecastgate import publish as publish_module
from forecastgate.errors import LockHeldError, PredictionContractError, PublishRefusedError
from forecastgate.panel import WEEK
from forecastgate.publish import (
    CURRENT_FILE,
    ArtifactIdentity,
    PublishResult,
    publish,
    render_csv,
)
from forecastgate.recipes import PREDICTION_COLUMNS

CUTOFF = pd.Timestamp("2011-03-21")
UNIVERSE = ["10001", "85123A"]
HORIZONS = (1, 2)
IDENTITY = ArtifactIdentity(
    cutoff_week="2011-03-21",
    model_version="ma4-test",
    recipe_config_hash="c" * 64,
    data_hash="d" * 64,
    input_hash="e" * 64,
)


def _predictions(value: float = 1.5) -> pd.DataFrame:
    rows = [
        (code, CUTOFF, h, CUTOFF + h * WEEK, value + i)
        for i, code in enumerate(UNIVERSE)
        for h in HORIZONS
    ]
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _publish(
    root: Path, predictions: pd.DataFrame, identity: ArtifactIdentity = IDENTITY
) -> PublishResult:
    return publish(root, identity, predictions, universe=UNIVERSE, horizons=HORIZONS, git_sha="abc")


def _current(root: Path) -> str:
    return (root / CURRENT_FILE).read_text(encoding="utf-8").strip()


def test_csv_byte_format_is_fixed() -> None:
    payload = render_csv(_predictions().iloc[::-1])
    assert payload.startswith(b"stock_code,cutoff_week,horizon,target_week,prediction\n")
    assert b"\r" not in payload
    assert payload.splitlines()[1] == b"10001,2011-03-21,1,2011-03-28,1.500000"


def test_publish_then_rerun_is_a_noop(tmp_path: Path) -> None:
    first = _publish(tmp_path, _predictions())
    second = _publish(tmp_path, _predictions())
    assert (first.status, second.status) == ("published", "unchanged")
    assert _current(tmp_path) == first.identity
    manifest = json.loads((first.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["predictions.csv"]["rows"] == 4


def test_same_identity_with_different_content_is_refused(tmp_path: Path) -> None:
    first = _publish(tmp_path, _predictions())
    with pytest.raises(PublishRefusedError, match="refusing to overwrite"):
        _publish(tmp_path, _predictions(value=9.0))
    assert _current(tmp_path) == first.identity


def test_new_model_version_is_a_new_artifact(tmp_path: Path) -> None:
    first = _publish(tmp_path, _predictions())
    newer = ArtifactIdentity(**{**IDENTITY.__dict__, "model_version": "ma4-other"})
    second = _publish(tmp_path, _predictions(value=9.0), newer)
    assert second.status == "published"
    assert second.identity != first.identity
    assert _current(tmp_path) == second.identity


def test_tampered_artifact_is_refused_on_rerun(tmp_path: Path) -> None:
    first = _publish(tmp_path, _predictions())
    target = first.path / "predictions.csv"
    target.write_bytes(target.read_bytes().replace(b"1.500000", b"7.000000"))
    with pytest.raises(PublishRefusedError, match="checksum"):
        _publish(tmp_path, _predictions())


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: p.iloc[1:],  # incomplete grid
        lambda p: pd.concat([p, p.iloc[:1]]),  # duplicate key
        lambda p: p.assign(prediction=p["prediction"] * -1),  # negative
        lambda p: p.assign(prediction=float("nan")),  # non-finite
        lambda p: p.assign(target_week=p["cutoff_week"]),  # wrong target date
    ],
    ids=["grid", "duplicate", "negative", "nan", "target_week"],
)
def test_contract_failure_publishes_nothing(
    tmp_path: Path, corrupt: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    previous = _publish(tmp_path, _predictions())
    newer = ArtifactIdentity(**{**IDENTITY.__dict__, "model_version": "broken"})
    with pytest.raises(PredictionContractError):
        _publish(tmp_path, corrupt(_predictions()), newer)
    assert _current(tmp_path) == previous.identity
    assert not (tmp_path / newer.digest()).exists()


def test_held_lock_refuses_a_second_publisher(tmp_path: Path) -> None:
    tmp_path.joinpath(".lock").write_text("pid 1 on elsewhere", encoding="utf-8")
    with pytest.raises(LockHeldError, match="pid 1 on elsewhere"):
        _publish(tmp_path, _predictions())
    assert not (tmp_path / CURRENT_FILE).exists()


def test_lock_is_released_after_a_failure(tmp_path: Path) -> None:
    with pytest.raises(PredictionContractError):
        _publish(tmp_path, _predictions().iloc[1:])
    assert not (tmp_path / ".lock").exists()
    assert _publish(tmp_path, _predictions()).status == "published"


def test_leftover_staging_is_removed(tmp_path: Path) -> None:
    leftover = tmp_path / ".staging" / "deadbeef-00000000"
    leftover.mkdir(parents=True)
    _publish(tmp_path, _predictions())
    assert not leftover.exists()


def test_crash_before_pointer_swap_is_repaired_by_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(root: Path, digest: str) -> None:
        raise RuntimeError("simulated crash after rename")

    original = publish_module._point_current
    monkeypatch.setattr(publish_module, "_point_current", crash)
    with pytest.raises(RuntimeError):
        _publish(tmp_path, _predictions())
    assert not (tmp_path / CURRENT_FILE).exists()
    assert not (tmp_path / ".lock").exists()

    monkeypatch.setattr(publish_module, "_point_current", original)
    result = _publish(tmp_path, _predictions())
    assert result.status == "unchanged"
    assert _current(tmp_path) == result.identity
