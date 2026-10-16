"""Read-only serving of the committed prediction artifact (docs/design.md section 6).

Serving never writes and never falls back silently. On every request it resolves
`CURRENT`, verifies the manifest and the predictions file against their recorded
checksums, re-checks the predictions, and checks freshness against a simulated clock.
Anything that fails is refused with its reason; the HTTP layer turns that into a 503.

Freshness: an artifact made at cutoff week T (identified by its Monday) is based on data
up to the end of that week, T + 6 days. With `as_of` the simulated current date, it is
served only if `0 <= as_of - (T + 6 days) <= max_age_days`, so stale artifacts and
artifacts dated in the future are both refused. The data is historical, so `as_of` is
always supplied explicitly rather than read from the system clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from forecastgate.errors import ForecastgateError, PredictionContractError
from forecastgate.panel import WEEK
from forecastgate.publish import CURRENT_FILE, MANIFEST_FILE, PREDICTIONS_FILE, read_predictions
from forecastgate.recipes import PREDICTION_COLUMNS


class ServingRefusedError(ForecastgateError):
    """The current artifact cannot be served; the message says why."""

    exit_code = 11


@dataclass(frozen=True)
class ServedArtifact:
    identity: str
    manifest: dict[str, Any]
    predictions: pd.DataFrame
    cutoff_week: date
    data_through: date  # last day covered by the data: end of the cutoff week
    age_days: int


def load_current(predictions_dir: Path, *, as_of: date, max_age_days: int) -> ServedArtifact:
    pointer = predictions_dir / CURRENT_FILE
    if not pointer.exists():
        raise ServingRefusedError("no artifact has been published")
    identity = pointer.read_text(encoding="utf-8").strip()
    artifact = predictions_dir / identity
    if not identity or identity.startswith(".") or not artifact.is_dir():
        raise ServingRefusedError(f"CURRENT names {identity!r}, which is not a published artifact")

    try:
        manifest: dict[str, Any] = json.loads(
            (artifact / MANIFEST_FILE).read_text(encoding="utf-8")
        )
        recorded = manifest["files"][PREDICTIONS_FILE]
        cutoff = date.fromisoformat(manifest["identity"]["cutoff_week"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ServingRefusedError(f"{identity}: manifest is unreadable ({exc})") from exc
    if manifest.get("identity_digest") != identity:
        raise ServingRefusedError(f"{identity}: manifest belongs to another artifact")

    path = artifact / PREDICTIONS_FILE
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ServingRefusedError(f"{identity}: {PREDICTIONS_FILE} is unreadable") from exc
    if hashlib.sha256(content).hexdigest() != recorded["sha256"]:
        raise ServingRefusedError(f"{identity}: {PREDICTIONS_FILE} does not match its checksum")

    try:
        predictions = read_predictions(path)
    except PredictionContractError as exc:
        raise ServingRefusedError(f"{identity}: {exc}") from exc
    problems = _content_problems(predictions, cutoff, int(recorded["rows"]))
    if problems:
        raise ServingRefusedError(f"{identity}: " + "; ".join(problems))

    data_through = cutoff + timedelta(days=6)
    age = (as_of - data_through).days
    if age < 0:
        raise ServingRefusedError(
            f"{identity}: data through {data_through} is later than as-of {as_of}"
        )
    if age > max_age_days:
        raise ServingRefusedError(
            f"{identity}: stale, data through {data_through} is {age} days old at {as_of} "
            f"(maximum {max_age_days})"
        )
    return ServedArtifact(
        identity=identity,
        manifest=manifest,
        predictions=predictions,
        cutoff_week=cutoff,
        data_through=data_through,
        age_days=age,
    )


def _content_problems(predictions: pd.DataFrame, cutoff: date, rows: int) -> list[str]:
    problems = []
    if list(predictions.columns) != PREDICTION_COLUMNS:
        return [f"columns are {list(predictions.columns)}"]
    if len(predictions) != rows:
        problems.append(f"{len(predictions)} rows, manifest records {rows}")
    if predictions[["stock_code", "horizon"]].duplicated().any():
        problems.append("duplicate keys")
    if (predictions["cutoff_week"] != pd.Timestamp(cutoff)).any():
        problems.append("rows from another cutoff")
    if (
        predictions["target_week"] != predictions["cutoff_week"] + predictions["horizon"] * WEEK
    ).any():
        problems.append("target weeks inconsistent with horizons")
    values = predictions["prediction"]
    if not values.map(lambda v: pd.notna(v) and abs(v) != float("inf")).all() or (values < 0).any():
        problems.append("non-finite or negative predictions")
    return problems
