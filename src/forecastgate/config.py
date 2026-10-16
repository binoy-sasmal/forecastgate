"""Data sources and pipeline settings.

The universe rule (size, history, ranking window) is frozen together with the gate before
the first full-data evaluation; see docs/design.md section 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    name: str
    path: Path
    sha256: str
    url: str | None  # None means offline only: the file must already be present


FULL_SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
FULL_SOURCE_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
FIXTURE_SHA256 = "dd91b620dbc68f35f3d67bc4580a5e87d993558bcc078975b9278b54131270cb"


@dataclass(frozen=True)
class Settings:
    source: DataSource
    artifacts_dir: Path
    universe_size: int = 200
    min_universe_size: int = 50
    min_history_weeks: int = 26
    ranking_window_weeks: int = 26
    horizons: tuple[int, ...] = (1, 2, 3, 4)
    reports_dir: Path = Path("reports")

    @property
    def predictions_dir(self) -> Path:
        return self.artifacts_dir / "predictions"

    @property
    def cache_dir(self) -> Path:
        return self.artifacts_dir / "cache"


def settings_for(*, fixture: bool, root: Path, artifacts_dir: Path | None = None) -> Settings:
    if fixture:
        source = DataSource(
            name="fixture",
            path=root / "data" / "fixture" / "online_retail_II_fixture.zip",
            sha256=FIXTURE_SHA256,
            url=None,
        )
    else:
        source = DataSource(
            name="full",
            path=root / "data" / "raw" / "online_retail_ii.zip",
            sha256=FULL_SOURCE_SHA256,
            url=FULL_SOURCE_URL,
        )
    artifacts = artifacts_dir or root / "artifacts" / source.name
    # Only the full run writes the committed reports; fixture reports stay with its artifacts.
    reports = artifacts / "reports" if fixture else root / "reports"
    return Settings(source=source, artifacts_dir=artifacts, reports_dir=reports)
