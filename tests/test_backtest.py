"""Metrics by hand, the backtest harness on the real fixture, and the backtest command."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from forecastgate.backtest import ORACLE, run_backtest, summary, with_oracle
from forecastgate.cli import app
from forecastgate.config import FIXTURE_SHA256, DataSource, settings_for
from forecastgate.errors import DataContractError
from forecastgate.gate import GateNotFrozenError
from forecastgate.ingest import ingest
from forecastgate.metrics import bias, mase, mase_scales, wape
from forecastgate.panel import WEEK, Panel, build_panel
from forecastgate.pipeline import backtest_recipes
from forecastgate.recipes import FittedRecipe, MovingAverage, Recipe

REPO = Path(__file__).resolve().parents[1]
SETTINGS = settings_for(fixture=True, root=REPO)
NAN = float("nan")
runner = CliRunner()


# --- metrics, by hand ---------------------------------------------------------------


def test_wape_and_bias_by_hand() -> None:
    actual = pd.Series([10.0, 0.0, 5.0])
    prediction = pd.Series([8.0, 1.0, 5.0])
    assert wape(actual, prediction) == pytest.approx(3 / 15)  # |-2| + |1| + 0 over 15
    assert bias(actual, prediction) == pytest.approx(-1 / 15)  # (14 - 15) / 15


def test_metrics_are_undefined_when_actuals_sum_to_zero() -> None:
    zeros = pd.Series([0.0, 0.0])
    assert wape(zeros, pd.Series([1.0, 0.0])) is None
    assert bias(zeros, pd.Series([1.0, 0.0])) is None


def _scale_panel() -> Panel:
    weeks = pd.date_range("2011-01-03", periods=5, freq="7D", unit="us")
    rows = [("A", w, q) for w, q in zip(weeks, [1.0, 3.0, NAN, 6.0, 6.0], strict=True)]
    rows += [("B", w, 5.0) for w in weeks[2:]]
    frame = pd.DataFrame(rows, columns=["stock_code", "week", "quantity"]).assign(price=1.0)
    return Panel(
        frame=frame,
        first_sale_week=pd.Series({"A": weeks[0], "B": weeks[2]}),
        weeks=pd.DatetimeIndex(weeks),
        zero_transaction_weeks=pd.DatetimeIndex([weeks[2]]),
    )


def test_mase_scale_uses_consecutive_observed_weeks_only() -> None:
    scales = mase_scales(_scale_panel())
    # A: steps 1->3 (2) and 6->6 (0); steps touching the missing week are skipped.
    assert scales["A"] == pytest.approx(1.0)
    assert scales["B"] == 0.0


def test_mase_excludes_series_with_zero_scale() -> None:
    scales = mase_scales(_scale_panel())
    scales.index = pd.MultiIndex.from_product([[0], scales.index], names=["fold", "stock_code"])
    scored = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "stock_code": ["A", "A", "B"],
            "actual": [4.0, 6.0, 5.0],
            "prediction": [5.0, 3.0, 4.0],
        }
    )
    value, excluded = mase(scored, scales)
    assert value == pytest.approx(2.0)  # A: mean error (1 + 3) / 2 = 2, scale 1
    assert excluded == 1  # B's scale is 0


def test_mase_is_undefined_when_every_series_is_excluded() -> None:
    scales = pd.Series([0.0], index=pd.MultiIndex.from_tuples([(0, "B")]))
    scored = pd.DataFrame({"fold": [0], "stock_code": ["B"], "actual": [1.0], "prediction": [1.0]})
    assert mase(scored, scales) == (None, 1)


def test_oracle_picks_the_better_baseline_in_each_fold() -> None:
    rows = []
    # Fold 0: naive is perfect. Fold 1: ma4 is perfect.
    for fold, naive_pred, ma4_pred in ((0, 10.0, 0.0), (1, 0.0, 10.0)):
        rows.append((fold, "naive", "A", 1, 10.0, naive_pred))
        rows.append((fold, "ma4", "A", 1, 10.0, ma4_pred))
    scored = pd.DataFrame(
        rows, columns=["fold", "recipe", "stock_code", "horizon", "actual", "prediction"]
    )
    oracle = with_oracle(scored)
    oracle = oracle[oracle["recipe"] == ORACLE].sort_values("fold")
    assert oracle["prediction"].tolist() == [10.0, 10.0]


# --- the harness on the real fixture ------------------------------------------------


@pytest.fixture(scope="module")
def panel() -> Panel:
    source = DataSource(
        name="fixture",
        path=REPO / "data/fixture/online_retail_II_fixture.zip",
        sha256=FIXTURE_SHA256,
        url=None,
    )
    return build_panel(ingest(source))


@dataclass(frozen=True)
class Spy(Recipe):
    """Records what each fit is shown; forecasts with ma4."""

    seen: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    name = "spy"

    def _fit(self, panel: Panel, universe: list[str], horizons: tuple[int, ...]) -> FittedRecipe:
        self.seen.append(
            (panel.last_complete_week, panel.frame["week"].max(), len(panel.first_sale_week))
        )
        return MovingAverage().fit(panel, universe, horizons)


def test_recipes_only_see_data_up_to_each_fold_cutoff(panel: Panel) -> None:
    spy = Spy()
    result = run_backtest(panel, SETTINGS, {"spy": spy})
    cutoffs = [fold.cutoff for fold in result.folds]
    assert [seen[0] for seen in spy.seen] == cutoffs
    assert [seen[1] for seen in spy.seen] == cutoffs
    assert all(panel.last_complete_week > c for c in cutoffs)


def test_ma4_fold_wape_matches_an_independent_hand_computation(panel: Panel) -> None:
    result = run_backtest(panel, SETTINGS, {"ma4": MovingAverage()})
    fold = result.folds[0]
    universe = sorted(result.scored.loc[result.scored["fold"] == 0, "stock_code"].unique())
    quantity = panel.frame.set_index(["stock_code", "week"])["quantity"]
    errors, total = 0.0, 0.0
    for code in universe:
        window = [quantity.get((code, fold.cutoff - k * WEEK), NAN) for k in range(4)]
        observed = [q for q in window if not math.isnan(q)]
        level = sum(observed) / len(observed)
        for h in SETTINGS.horizons:
            actual = quantity[(code, fold.cutoff + h * WEEK)]
            errors += abs(actual - level)
            total += actual
    table = summary(result)
    reported = table[(table["recipe"] == "ma4") & (table["scope"] == "fold=0")]["wape"].iloc[0]
    assert reported == pytest.approx(errors / total)


def test_backtest_is_deterministic_across_runs(panel: Panel) -> None:
    first = run_backtest(panel, SETTINGS, backtest_recipes())
    second = run_backtest(panel, SETTINGS, backtest_recipes())
    pd.testing.assert_frame_equal(first.scored, second.scored, check_exact=True)
    pd.testing.assert_frame_equal(summary(first), summary(second), check_exact=True)
    assert set(first.scored["recipe"]) == {"naive", "ma4", "lgbm"}
    assert (first.scored["prediction"] >= 0).all()


def test_zero_transaction_target_week_stops_the_backtest(panel: Panel) -> None:
    from forecastgate.folds import backtest_cutoffs

    target = backtest_cutoffs(panel.weeks, SETTINGS)[2] + 2 * WEEK
    blocked = replace(
        panel,
        zero_transaction_weeks=pd.DatetimeIndex([*panel.zero_transaction_weeks, target]),
    )
    with pytest.raises(DataContractError, match="zero-transaction"):
        run_backtest(blocked, SETTINGS, {"ma4": MovingAverage()})


# --- the command ----------------------------------------------------------------------


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(REPO / "data" / "fixture", tmp_path / "data" / "fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORECASTGATE_GIT_SHA", "test")
    return tmp_path


def test_full_data_backtest_is_refused_before_the_gate_is_frozen(checkout: Path) -> None:
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code == GateNotFrozenError.exit_code
    assert "gate.yaml" in result.output
    assert not (checkout / "data" / "raw").exists()  # refused before any download


def test_planted_leak_writes_a_separate_labelled_report(checkout: Path) -> None:
    result = runner.invoke(app, ["backtest", "--fixture", "--inject-leak", "future_window"])
    assert result.exit_code == 0, result.output
    reports = checkout / "artifacts" / "fixture" / "reports"
    leaky = (reports / "backtest_leak_future_window.md").read_text(encoding="utf-8")
    assert "Planted leak in lgbm" in leaky
    assert not (reports / "backtest.md").exists()
