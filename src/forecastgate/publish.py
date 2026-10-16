"""Versioned, atomic publication of batch predictions (docs/design.md section 6).

Layout under the predictions directory:

    .lock                  held by the single active publisher
    .staging/<id>-<nonce>/ assembled and validated here, never read by serving
    <id>/predictions.csv   committed artifact
    <id>/manifest.json
    CURRENT                identity of the artifact to serve, swapped atomically
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from forecastgate.errors import LockHeldError, PredictionContractError, PublishRefusedError
from forecastgate.panel import WEEK
from forecastgate.recipes import PREDICTION_COLUMNS

PREDICTIONS_FILE = "predictions.csv"
MANIFEST_FILE = "manifest.json"
CURRENT_FILE = "CURRENT"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactIdentity:
    cutoff_week: str  # ISO date of the cutoff week's Monday
    model_version: str
    recipe_config_hash: str
    data_hash: str  # the pinned source archive
    # The exact model input: prepared panel, universe and horizons. A change in data
    # preparation changes this, so it yields a new artifact instead of a refused one.
    input_hash: str

    def digest(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PublishResult:
    status: Literal["published", "unchanged"]
    identity: str
    path: Path


def render_csv(predictions: pd.DataFrame) -> bytes:
    """Fixed byte format: UTF-8, LF, fixed columns, ISO dates, %.6f, sorted keys."""
    ordered = predictions.sort_values(["stock_code", "horizon"], kind="mergesort")
    buffer = io.StringIO()
    buffer.write(",".join(PREDICTION_COLUMNS) + "\n")
    for row in ordered.itertuples(index=False):
        buffer.write(
            f"{row.stock_code},{row.cutoff_week:%Y-%m-%d},{row.horizon},"
            f"{row.target_week:%Y-%m-%d},{row.prediction:.6f}\n"
        )
    return buffer.getvalue().encode("utf-8")


def read_predictions(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            dtype={"stock_code": "str", "horizon": "int64", "prediction": "float64"},
            parse_dates=["cutoff_week", "target_week"],
            keep_default_na=False,
        )
    except ValueError as exc:
        raise PredictionContractError(f"{path.name} is not parseable: {exc}") from exc


def check_prediction_contract(
    predictions: pd.DataFrame,
    *,
    universe: list[str],
    cutoff: pd.Timestamp,
    horizons: tuple[int, ...],
) -> None:
    problems: list[str] = []
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise PredictionContractError(f"columns are {list(predictions.columns)}")
    keys = predictions[["stock_code", "horizon"]]
    if keys.duplicated().any():
        problems.append(f"{int(keys.duplicated().sum())} duplicate keys")
    expected = {(code, h) for code in universe for h in horizons}
    actual = set(zip(predictions["stock_code"], predictions["horizon"], strict=True))
    if actual != expected:
        problems.append(
            f"grid mismatch: {len(expected - actual)} missing, {len(actual - expected)} extra"
        )
    if (predictions["cutoff_week"] != cutoff).any():
        problems.append("cutoff_week differs from the artifact cutoff")
    target = predictions["cutoff_week"] + predictions["horizon"] * WEEK
    if (predictions["target_week"] != target).any():
        problems.append("target_week is not cutoff_week + horizon weeks")
    values = pd.to_numeric(predictions["prediction"], errors="coerce")
    if not values.map(lambda v: pd.notna(v) and abs(v) != float("inf")).all():
        problems.append("non-finite predictions")
    elif (values < 0).any():
        problems.append("negative predictions")
    if problems:
        raise PredictionContractError("; ".join(problems))


@contextlib.contextmanager
def publication_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock.read_text(encoding="utf-8").strip() if lock.exists() else "unknown"
        raise LockHeldError(
            f"publication lock {lock} is held by {holder}; if no publisher is running, "
            "delete the file by hand"
        ) from None
    try:
        holder = f"pid {os.getpid()} on {socket.gethostname()} since {datetime.now(UTC):%FT%TZ}"
        os.write(fd, holder.encode("utf-8"))
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish(
    root: Path,
    identity: ArtifactIdentity,
    predictions: pd.DataFrame,
    *,
    universe: list[str],
    horizons: tuple[int, ...],
    git_sha: str,
) -> PublishResult:
    cutoff = pd.Timestamp(identity.cutoff_week)
    payload = render_csv(predictions)
    content_hash = hashlib.sha256(payload).hexdigest()
    digest = identity.digest()
    final = root / digest

    with publication_lock(root):
        staging_root = root / ".staging"
        shutil.rmtree(staging_root, ignore_errors=True)

        if final.exists():
            existing = _verified_content_hash(final)
            if existing != content_hash:
                raise PublishRefusedError(
                    f"artifact {digest} exists with content {existing[:12]}, "
                    f"new content is {content_hash[:12]}; refusing to overwrite"
                )
            _point_current(root, digest)
            return PublishResult("unchanged", digest, final)

        staging = staging_root / f"{digest}-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True)
        (staging / PREDICTIONS_FILE).write_bytes(payload)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "identity": asdict(identity),
            "identity_digest": digest,
            "git_sha": git_sha,
            "files": {PREDICTIONS_FILE: {"sha256": content_hash, "rows": len(predictions)}},
        }
        (staging / MANIFEST_FILE).write_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        # Validate what is on disk, not the in-memory frame.
        check_prediction_contract(
            read_predictions(staging / PREDICTIONS_FILE),
            universe=universe,
            cutoff=cutoff,
            horizons=horizons,
        )
        _verified_content_hash(staging)
        os.rename(staging, final)
        _point_current(root, digest)
        return PublishResult("published", digest, final)


def _verified_content_hash(artifact: Path) -> str:
    manifest = json.loads((artifact / MANIFEST_FILE).read_text(encoding="utf-8"))
    recorded = manifest["files"][PREDICTIONS_FILE]["sha256"]
    actual = hashlib.sha256((artifact / PREDICTIONS_FILE).read_bytes()).hexdigest()
    if actual != recorded:
        raise PublishRefusedError(
            f"{artifact.name}: {PREDICTIONS_FILE} does not match its manifest checksum"
        )
    return str(actual)


def _point_current(root: Path, digest: str) -> None:
    tmp = root / f"{CURRENT_FILE}.tmp"
    tmp.write_bytes(f"{digest}\n".encode())
    os.replace(tmp, root / CURRENT_FILE)
