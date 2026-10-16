"""The frozen gate: loading, freeze verification, configuration match and criteria."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from forecastgate.audit import write_report
from forecastgate.config import FIXTURE_SHA256, DataSource, settings_for
from forecastgate.gate import (
    GATE_FILE,
    Freeze,
    GateConfig,
    GateConfigError,
    GateNotFrozenError,
    check_configuration,
    evaluate,
    load_gate,
    verify_freeze,
)
from forecastgate.pipeline import backtest_recipes
from forecastgate.recipes import LightGBM

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / GATE_FILE


@pytest.fixture(scope="module")
def gate() -> GateConfig:
    return load_gate(GATE)


# --- configuration ------------------------------------------------------------------


def test_committed_gate_matches_the_code_for_full_and_fixture_runs(gate: GateConfig) -> None:
    for fixture in (False, True):
        check_configuration(gate, settings_for(fixture=fixture, root=REPO), backtest_recipes())


def test_changed_universe_rule_is_a_mismatch(gate: GateConfig) -> None:
    settings = replace(settings_for(fixture=False, root=REPO), universe_size=100)
    with pytest.raises(GateConfigError, match="universe"):
        check_configuration(gate, settings, backtest_recipes())


def test_changed_recipe_parameter_is_a_mismatch(gate: GateConfig) -> None:
    recipes = {**backtest_recipes(), "lgbm": LightGBM(num_leaves=15)}
    with pytest.raises(GateConfigError, match="lgbm params"):
        check_configuration(gate, settings_for(fixture=False, root=REPO), recipes)


def test_planted_leak_is_a_mismatch(gate: GateConfig) -> None:
    with pytest.raises(GateConfigError, match="lgbm params"):
        check_configuration(
            gate, settings_for(fixture=False, root=REPO), backtest_recipes("future_window")
        )


def test_malformed_gate_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / GATE_FILE
    broken.write_text(GATE.read_text(encoding="utf-8").replace("thresholds:", "limits:"))
    with pytest.raises(GateConfigError, match="exactly"):
        load_gate(broken)


# --- freeze -------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    shutil.copy(GATE, tmp_path / GATE_FILE)
    return tmp_path


def test_untracked_gate_is_not_frozen(repo: Path) -> None:
    with pytest.raises(GateNotFrozenError, match="not committed"):
        verify_freeze(repo)


def test_staged_but_uncommitted_gate_is_not_frozen(repo: Path) -> None:
    _git(repo, "add", GATE_FILE)
    with pytest.raises(GateNotFrozenError, match="differs from its committed version"):
        verify_freeze(repo)


def test_committed_gate_is_frozen_at_its_own_commit(repo: Path) -> None:
    _git(repo, "add", GATE_FILE)
    _git(repo, "commit", "-q", "-m", "freeze")
    frozen_at = _git(repo, "rev-parse", "HEAD")
    (repo / "other.txt").write_text("later work\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", "later")

    freeze = verify_freeze(repo)
    assert freeze.commit == frozen_at  # the last commit touching gate.yaml, not HEAD
    assert freeze.sha256 == load_gate(repo / GATE_FILE).sha256


def test_gate_edited_after_the_freeze_is_refused(repo: Path) -> None:
    _git(repo, "add", GATE_FILE)
    _git(repo, "commit", "-q", "-m", "freeze")
    text = (repo / GATE_FILE).read_text(encoding="utf-8")
    (repo / GATE_FILE).write_text(text.replace("abs_bias_max: 0.05", "abs_bias_max: 0.15"))
    with pytest.raises(GateNotFrozenError, match="differs"):
        verify_freeze(repo)


def test_directory_outside_git_is_not_frozen(tmp_path: Path) -> None:
    shutil.copy(GATE, tmp_path / GATE_FILE)
    with pytest.raises(GateNotFrozenError):
        verify_freeze(tmp_path)


# --- criteria -----------------------------------------------------------------------

BASE = 1.0  # ma4's WAPE in every scope; naive is worse everywhere


def _table(
    *,
    pooled: float | None = 0.9,
    bias: float = 0.0,
    folds: list[float] | None = None,
    horizons: list[float] | None = None,
    extra: dict[str, float] | None = None,
) -> pd.DataFrame:
    folds = folds if folds is not None else [0.9] * 6
    horizons = horizons if horizons is not None else [0.9] * 4
    records = []
    for recipe, scale in (("naive", 1.2), ("ma4", 1.0)):
        records.append({"recipe": recipe, "scope": "pooled", "wape": BASE * scale, "bias": 0.0})
        records += [
            {"recipe": recipe, "scope": f"fold={i}", "wape": BASE * scale} for i in range(6)
        ]
        records += [
            {"recipe": recipe, "scope": f"horizon={h}", "wape": BASE * scale} for h in range(1, 5)
        ]
    records.append({"recipe": "lgbm", "scope": "pooled", "wape": pooled, "bias": bias})
    records += [{"recipe": "lgbm", "scope": f"fold={i}", "wape": w} for i, w in enumerate(folds)]
    records += [
        {"recipe": "lgbm", "scope": f"horizon={h}", "wape": w} for h, w in enumerate(horizons, 1)
    ]
    for recipe, value in (extra or {}).items():
        records.append({"recipe": recipe, "scope": "pooled", "wape": value, "bias": 0.0})
    return pd.DataFrame.from_records(records)


def test_a_clearly_better_candidate_is_promoted(gate: GateConfig) -> None:
    decision = evaluate(_table(), gate)
    assert decision.base == "ma4"
    assert decision.champion == "ma4"
    assert decision.promoted, [c.detail for c in decision.criteria]


@pytest.mark.parametrize(
    ("table", "failed"),
    [
        ({"pooled": 0.96}, [1]),  # 0.96 > 0.95 x 1.0, but within 0.99 x the champion's 1.0
        ({"pooled": 0.995}, [1, 6]),  # also beyond 0.99 x the champion
        ({"folds": [0.9, 0.9, 0.9, 1.05, 1.05, 1.05]}, [2]),  # 3 wins, all within 1.10
        ({"folds": [0.9, 0.9, 0.9, 0.9, 0.9, 1.11]}, [3]),  # one fold beyond 1.10 x
        ({"horizons": [0.9, 0.9, 0.9, 1.01]}, [4]),
        ({"bias": -0.051}, [5]),
        ({"pooled": None}, [1, 6]),  # undefined fails, never passes
        ({"folds": [0.9, 0.9, 0.9, 0.9, 0.9, None]}, [2, 3]),
    ],
    ids=[
        "pooled",
        "pooled_and_champion",
        "fold_wins",
        "worst_fold",
        "horizon",
        "bias",
        "undefined",
        "undefined_fold",
    ],
)
def test_each_criterion_fails_on_its_own(
    gate: GateConfig, table: dict[str, object], failed: list[int]
) -> None:
    decision = evaluate(_table(**table), gate)  # type: ignore[arg-type]
    assert decision.failed == failed
    assert not decision.promoted


def test_a_registered_champion_needs_a_one_percent_improvement(gate: GateConfig) -> None:
    decision = evaluate(_table(pooled=0.9, extra={"lgbm_v1": 0.905}), gate, champion="lgbm_v1")
    assert decision.failed == [6]  # 0.9 > 0.99 x 0.905 = 0.896


def test_baseline_tie_goes_to_the_first_listed(gate: GateConfig) -> None:
    table = _table()
    table.loc[(table["recipe"] == "naive") & (table["scope"] == "pooled"), "wape"] = BASE
    assert evaluate(table, gate).base == "naive"


# --- recording the freeze -------------------------------------------------------------


def test_report_and_provenance_record_the_freeze(tmp_path: Path) -> None:
    from forecastgate.backtest import render_report, run_backtest
    from forecastgate.ingest import ingest
    from forecastgate.panel import build_panel
    from forecastgate.recipes import MovingAverage, Naive

    settings = settings_for(fixture=True, root=REPO, artifacts_dir=tmp_path)
    source = DataSource(
        "fixture", REPO / "data/fixture/online_retail_II_fixture.zip", FIXTURE_SHA256, None
    )
    panel = build_panel(ingest(source))
    recipes = {"naive": Naive(), "ma4": MovingAverage()}
    result = run_backtest(panel, settings, recipes)
    gate = replace(load_gate(GATE), candidate="ma4", baselines=("naive",))
    freeze = Freeze(commit="f" * 40, sha256=gate.sha256)
    report = render_report(
        result, settings, recipes, data_hash=FIXTURE_SHA256, gate=gate, freeze=freeze
    )
    assert f"| Gate freeze commit | `{'f' * 40}` |" in report
    assert f"`{gate.sha256}`" in report
    assert "## Gate" in report

    write_report(
        tmp_path,
        "backtest.md",
        report,
        settings,
        "abc",
        extra={"gate_sha256": gate.sha256, "gate_freeze_commit": freeze.commit},
    )
    recorded = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert recorded["backtest.md"]["gate_freeze_commit"] == "f" * 40
    assert recorded["backtest.md"]["gate_sha256"] == gate.sha256


def test_exactly_the_minimum_fold_wins_passes(gate: GateConfig) -> None:
    decision = evaluate(_table(folds=[0.9, 0.9, 0.9, 0.9, 1.05, 1.05]), gate)
    assert decision.promoted, [c.detail for c in decision.criteria]
