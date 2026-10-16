"""A small, file-based model registry with a `champion` alias.

Layout under the registry directory:

    .lock                       held by the single active writer
    registry.json               versions and aliases, replaced atomically
    decisions.jsonl             one line per gate run, appended
    models/v<N>/fitted.json     kind and cutoff of the fitted model
    models/v<N>/<payload files> e.g. levels.csv or model.txt

Every version records the recipe, its parameters, the data and input it was fitted on,
and the gate decision that registered it. Loading a model verifies its files against the
checksums recorded at registration, so a modified model is refused rather than served.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forecastgate.errors import ForecastgateError
from forecastgate.publish import publication_lock
from forecastgate.recipes import FittedRecipe, load_fitted

CHAMPION = "champion"
SCHEMA_VERSION = 1


class RegistryError(ForecastgateError):
    """The registry is missing a champion, or a registered model fails its checksums."""

    exit_code = 9


@dataclass(frozen=True)
class ModelVersion:
    version: int
    recipe: str
    params: dict[str, Any]
    recipe_config_hash: str
    kind: str
    cutoff_week: str
    data_hash: str
    input_hash: str
    files: dict[str, str]  # file name -> SHA-256
    reason: str  # "initial champion" or "promoted by gate"
    gate_sha256: str | None
    gate_freeze_commit: str | None
    git_sha: str

    @property
    def label(self) -> str:
        return f"{self.recipe}@v{self.version}"


class Registry:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def _index(self) -> Path:
        return self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if not self._index.exists():
            return {"schema_version": SCHEMA_VERSION, "versions": [], "aliases": {}}
        data: dict[str, Any] = json.loads(self._index.read_text(encoding="utf-8"))
        return data

    def _write(self, data: dict[str, Any]) -> None:
        partial = self._index.with_name("registry.json.part")
        partial.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.replace(partial, self._index)

    def versions(self) -> list[ModelVersion]:
        return [ModelVersion(**v) for v in self._read()["versions"]]

    def champion(self) -> ModelVersion | None:
        data = self._read()
        number = data["aliases"].get(CHAMPION)
        if number is None:
            return None
        return next(ModelVersion(**v) for v in data["versions"] if v["version"] == number)

    def register(
        self,
        fitted: FittedRecipe,
        *,
        recipe: str,
        params: dict[str, Any],
        recipe_config_hash: str,
        data_hash: str,
        input_hash: str,
        reason: str,
        gate_sha256: str | None,
        gate_freeze_commit: str | None,
        git_sha: str,
        make_champion: bool,
    ) -> ModelVersion:
        """Store a fitted model as the next version; optionally point `champion` at it."""
        self.root.mkdir(parents=True, exist_ok=True)
        with publication_lock(self.root):
            data = self._read()
            number = 1 + max((v["version"] for v in data["versions"]), default=0)
            shutil.rmtree(self.root / ".staging", ignore_errors=True)  # interrupted runs
            payload = fitted.payload()
            payload["fitted.json"] = (
                json.dumps({"kind": fitted.kind, "cutoff_week": f"{fitted.cutoff:%Y-%m-%d}"}) + "\n"
            ).encode("utf-8")
            staging = self.root / ".staging" / f"v{number}-{uuid.uuid4().hex[:8]}"
            staging.mkdir(parents=True)
            for name, content in payload.items():
                (staging / name).write_bytes(content)
            target = self.root / "models" / f"v{number}"
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, target)
            shutil.rmtree(self.root / ".staging", ignore_errors=True)

            version = ModelVersion(
                version=number,
                recipe=recipe,
                params=params,
                recipe_config_hash=recipe_config_hash,
                kind=fitted.kind,
                cutoff_week=f"{fitted.cutoff:%Y-%m-%d}",
                data_hash=data_hash,
                input_hash=input_hash,
                files={name: hashlib.sha256(c).hexdigest() for name, c in sorted(payload.items())},
                reason=reason,
                gate_sha256=gate_sha256,
                gate_freeze_commit=gate_freeze_commit,
                git_sha=git_sha,
            )
            data["versions"].append(asdict(version))
            if make_champion:
                data["aliases"][CHAMPION] = number
            self._write(data)
            return version

    def load(self, version: ModelVersion) -> FittedRecipe:
        directory = self.root / "models" / f"v{version.version}"
        files: dict[str, bytes] = {}
        for name, expected in version.files.items():
            path = directory / name
            if not path.exists():
                raise RegistryError(f"{version.label}: {name} is missing")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RegistryError(
                    f"{version.label}: {name} does not match its registered checksum"
                )
            files[name] = content
        meta = json.loads(files.pop("fitted.json"))
        return load_fitted(meta["kind"], pd.Timestamp(meta["cutoff_week"]), files)

    def record_decision(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        stamped = {"recorded_at": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", **record}
        with (self.root / "decisions.jsonl").open("a", encoding="utf-8", newline="\n") as log:
            log.write(json.dumps(stamped, sort_keys=True) + "\n")
