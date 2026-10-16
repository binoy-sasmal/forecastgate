"""Registry, gate command, batch prediction and the reproduce orchestration."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from forecastgate.cli import app
from forecastgate.config import FIXTURE_SHA256, DataSource, Settings, settings_for
from forecastgate.gate import GATE_REJECTED_EXIT, GateConfig, load_gate
from forecastgate.ingest import IngestResult, ingest
from forecastgate.panel import Panel, build_panel
from forecastgate.pipeline import backtest_recipes, run_gate, run_predict
from forecastgate.recipes import LightGBM, MovingAverage, Naive, load_fitted
from forecastgate.registry import Registry, RegistryError

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture(scope="module")
def data() -> IngestResult:
    source = DataSource(
        "fixture", REPO / "data/fixture/online_retail_II_fixture.zip", FIXTURE_SHA256, None
    )
    return ingest(source)


@pytest.fixture(scope="module")
def panel(data: IngestResult) -> Panel:
    return build_panel(data)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return settings_for(fixture=True, root=REPO, artifacts_dir=tmp_path / "artifacts")


@pytest.fixture(scope="module")
def gate() -> GateConfig:
    return load_gate(REPO / "gate.yaml")


def _universe(panel: Panel) -> list[str]:
    from forecastgate.universe import select_universe

    cutoff = panel.last_complete_week
    return select_universe(panel.truncated(cutoff), cutoff, settings_for(fixture=True, root=REPO))


# --- fitted models round-trip exactly ---------------------------------------------------


@pytest.mark.parametrize(
    "recipe", [MovingAverage(), Naive(), LightGBM()], ids=["ma4", "naive", "lgbm"]
)
def test_saved_model_predicts_identically_after_reload(
    panel: Panel, recipe: MovingAverage | Naive | LightGBM
) -> None:
    visible = panel.truncated(panel.last_complete_week)
    universe = _universe(panel)
    fitted = recipe.fit(visible, universe, (1, 2, 3, 4))
    reloaded = load_fitted(fitted.kind, fitted.cutoff, fitted.payload())
    pd.testing.assert_frame_equal(
        fitted.predict(visible, universe, (1, 2, 3, 4)),
        reloaded.predict(visible, universe, (1, 2, 3, 4)),
        check_exact=True,
    )


# --- registry -----------------------------------------------------------------------------


def _register(registry: Registry, panel: Panel, *, champion: bool = True) -> object:
    visible = panel.truncated(panel.last_complete_week)
    fitted = MovingAverage().fit(visible, _universe(panel), (1, 2, 3, 4))
    return registry.register(
        fitted,
        recipe="ma4",
        params=MovingAverage().params(),
        recipe_config_hash=MovingAverage().config_hash(),
        data_hash="d" * 64,
        input_hash="i" * 64,
        reason="test",
        gate_sha256=None,
        gate_freeze_commit=None,
        git_sha="abc",
        make_champion=champion,
    )


def test_versions_increment_and_the_alias_moves_only_when_asked(
    tmp_path: Path, panel: Panel
) -> None:
    registry = Registry(tmp_path)
    assert registry.champion() is None
    _register(registry, panel)
    _register(registry, panel, champion=False)
    assert [v.version for v in registry.versions()] == [1, 2]
    champion = registry.champion()
    assert champion is not None and champion.label == "ma4@v1"


def test_tampered_model_file_is_refused(tmp_path: Path, panel: Panel) -> None:
    registry = Registry(tmp_path)
    _register(registry, panel)
    levels = tmp_path / "models" / "v1" / "levels.csv"
    levels.write_bytes(levels.read_bytes().replace(b",", b",9", 1))
    champion = registry.champion()
    assert champion is not None
    with pytest.raises(RegistryError, match="checksum"):
        registry.load(champion)


def test_missing_model_file_is_refused(tmp_path: Path, panel: Panel) -> None:
    registry = Registry(tmp_path)
    _register(registry, panel)
    (tmp_path / "models" / "v1" / "levels.csv").unlink()
    champion = registry.champion()
    assert champion is not None
    with pytest.raises(RegistryError, match="missing"):
        registry.load(champion)


def test_interrupted_registration_leaves_no_partial_version(tmp_path: Path, panel: Panel) -> None:
    leftover = tmp_path / ".staging" / "v1-deadbeef"
    leftover.mkdir(parents=True)
    registry = Registry(tmp_path)
    _register(registry, panel)
    assert not (tmp_path / ".staging").exists()
    assert [v.version for v in registry.versions()] == [1]


# --- gate decisions and registration --------------------------------------------------------


def test_rejection_registers_the_best_baseline_as_initial_champion(
    settings: Settings, data: IngestResult, panel: Panel, gate: GateConfig
) -> None:
    outcome = run_gate(settings, root=REPO, data=data, panel=panel, gate=gate, freeze=None)
    assert not outcome.decision.promoted
    assert [v.label for v in outcome.registered] == ["ma4@v1"]
    assert outcome.champion.label == "ma4@v1"
    assert outcome.champion.reason == "initial champion: best single baseline"

    again = run_gate(settings, root=REPO, data=data, panel=panel, gate=gate, freeze=None)
    assert again.registered == []  # nothing new on an identical rerun
    assert again.champion.label == "ma4@v1"
    decisions = (settings.registry_dir / "decisions.jsonl").read_text().splitlines()
    assert [json.loads(line)["promoted"] for line in decisions] == [False, False]


def test_a_passing_candidate_is_refitted_registered_and_made_champion(
    settings: Settings, data: IngestResult, panel: Panel, gate: GateConfig
) -> None:
    # On the fixture, ma4 clears every criterion against naive, so it serves as a
    # candidate that genuinely passes the frozen thresholds.
    as_candidate = replace(gate, candidate="ma4", baselines=("naive",))
    recipes = {"naive": Naive(), "ma4": MovingAverage()}
    outcome = run_gate(
        settings, root=REPO, data=data, panel=panel, gate=as_candidate, freeze=None, recipes=recipes
    )
    assert outcome.decision.promoted, [c.detail for c in outcome.decision.criteria]
    assert outcome.champion.label == "ma4@v1"
    assert outcome.champion.reason == "promoted by gate"
    assert outcome.champion.cutoff_week == f"{panel.last_complete_week:%Y-%m-%d}"


def test_weak_candidate_is_blocked_and_the_champion_still_publishes(
    settings: Settings, data: IngestResult, panel: Panel, gate: GateConfig
) -> None:
    first = run_gate(settings, root=REPO, data=data, panel=panel, gate=gate, freeze=None)
    registry_before = (settings.registry_dir / "registry.json").read_bytes()

    crippled = {**backtest_recipes(), "lgbm": LightGBM(n_estimators=1, num_leaves=2)}
    blocked = run_gate(
        settings, root=REPO, data=data, panel=panel, gate=gate, freeze=None, recipes=crippled
    )
    assert not blocked.decision.promoted
    assert 1 in blocked.decision.failed  # far from the pooled WAPE bar
    assert blocked.champion == first.champion
    assert (settings.registry_dir / "registry.json").read_bytes() == registry_before

    published = run_predict(settings, root=REPO, data=data, panel=panel)
    assert published.champion.label == "ma4@v1"
    assert published.published.status == "published"


def test_stale_champion_is_refused_then_refitted_by_the_gate(
    settings: Settings, data: IngestResult, panel: Panel, gate: GateConfig
) -> None:
    earlier = panel.truncated(panel.weeks[-5])  # the data as it was four weeks earlier
    run_gate(settings, root=REPO, data=data, panel=earlier, gate=gate, freeze=None)

    with pytest.raises(RegistryError, match="different input"):
        run_predict(settings, root=REPO, data=data, panel=panel)

    refit = run_gate(settings, root=REPO, data=data, panel=panel, gate=gate, freeze=None)
    assert [v.reason for v in refit.registered] == ["champion refitted on new input"]
    assert refit.champion.label == "ma4@v2"
    assert run_predict(settings, root=REPO, data=data, panel=panel).champion.label == "ma4@v2"


# --- the commands -----------------------------------------------------------------------------


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(REPO / "data" / "fixture", tmp_path / "data" / "fixture")
    shutil.copy(REPO / "gate.yaml", tmp_path / "gate.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORECASTGATE_GIT_SHA", "test")
    return tmp_path


def test_gate_command_exits_10_and_predict_publishes_the_champion(checkout: Path) -> None:
    before = runner.invoke(app, ["predict", "--fixture"])
    assert before.exit_code == RegistryError.exit_code
    assert "no champion" in before.output

    gated = runner.invoke(app, ["gate", "--fixture"])
    assert gated.exit_code == GATE_REJECTED_EXIT, gated.output
    assert "registered        ma4@v1 (initial champion: best single baseline)" in gated.output

    predicted = runner.invoke(app, ["predict", "--fixture"])
    assert predicted.exit_code == 0, predicted.output
    assert "model             ma4@v1" in predicted.output
    assert "publication       published" in predicted.output


def test_reproduce_treats_the_rejection_as_an_outcome_and_publishes(checkout: Path) -> None:
    result = runner.invoke(app, ["reproduce", "--fixture"])
    assert result.exit_code == 0, result.output
    assert "lgbm rejected" in result.output
    assert "champion          ma4@v1" in result.output
    assert "publication       published" in result.output
    manifest = next((checkout / "artifacts/fixture/predictions").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["identity"]["model_version"] == "ma4@v1"


# --- provenance captures the tree state at run start ----------------------------------------


def _git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("dirty_before", [False, True], ids=["clean", "dirty"])
def test_git_dirty_reflects_the_tree_before_the_run_wrote_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty_before: bool
) -> None:
    from forecastgate.pipeline import reproduce

    monkeypatch.delenv("FORECASTGATE_GIT_SHA", raising=False)
    shutil.copytree(REPO / "data" / "fixture", tmp_path / "data" / "fixture")
    shutil.copy(REPO / "gate.yaml", tmp_path / "gate.yaml")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "provenance.json").write_text("{}\n")  # tracked, like the real one
    (tmp_path / "notes.txt").write_text("tracked\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "start")
    if dirty_before:
        (tmp_path / "notes.txt").write_text("edited before the run\n")

    settings = replace(settings_for(fixture=True, root=tmp_path), reports_dir=tmp_path / "reports")
    reproduce(settings, root=tmp_path, fixture=True)
    recorded = json.loads((tmp_path / "reports" / "provenance.json").read_text())
    assert {name: entry["git_dirty"] for name, entry in recorded.items()} == {
        "data_audit.md": dirty_before,
        "backtest.md": dirty_before,
    }
