"""Leakage tests on the real fixture (docs/design.md section 4, leakage tests 1 to 3).

For a point in time t, the pipeline is run twice:

- on raw data truncated at the end of week t, the only data available at t;
- on the full raw data, after altering every later record: quantities and prices
  change, descriptions change, one product's future rows vanish, a new product appears.

Anything computed for origins up to t must be identical. A planted leak must be caught.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from forecastgate.config import FIXTURE_SHA256, DataSource, settings_for
from forecastgate.features import Leak, label_violations, origin_features, training_rows
from forecastgate.folds import backtest_cutoffs
from forecastgate.ingest import IngestResult, apply_exclusions, ingest
from forecastgate.panel import WEEK, Panel, build_panel
from forecastgate.universe import select_universe

REPO = Path(__file__).resolve().parents[1]
SETTINGS = settings_for(fixture=True, root=REPO)


@pytest.fixture(scope="module")
def data() -> IngestResult:
    source = DataSource(
        name="fixture",
        path=REPO / "data/fixture/online_retail_II_fixture.zip",
        sha256=FIXTURE_SHA256,
        url=None,
    )
    return ingest(source)


@pytest.fixture(scope="module")
def full_panel(data: IngestResult) -> Panel:
    return build_panel(data)


def _rebuild(data: IngestResult, transactions: pd.DataFrame, end: pd.Timestamp) -> IngestResult:
    sales, exclusions, non_product = apply_exclusions(transactions.reset_index(drop=True))
    return IngestResult(
        transactions=transactions.reset_index(drop=True),
        sales=sales,
        exclusions=exclusions,
        non_product_codes=non_product,
        codes=data.codes,
        sheets=data.sheets,
        overlap_boundary=data.overlap_boundary,
        overlap_rows=data.overlap_rows,
        data_start=data.data_start,
        data_end=end,
        data_hash=data.data_hash,
    )


def truncated(data: IngestResult, t: pd.Timestamp) -> IngestResult:
    """What was known at the end of week t. Coverage ends at the next Monday 00:00."""
    boundary = t + WEEK
    tx = data.transactions
    return _rebuild(data, tx[tx["InvoiceDate"] < boundary], boundary)


def altered(data: IngestResult, t: pd.Timestamp) -> IngestResult:
    """The full data with every record after week t changed."""
    boundary = t + WEEK
    tx = data.transactions.copy()
    future = tx["InvoiceDate"] >= boundary
    # Product-specific factors reorder future volumes, so a ranking that peeked would change.
    codes = sorted(tx["StockCode"].unique())
    factor = tx["StockCode"].map({code: 1 + 10 * i for i, code in enumerate(codes)})
    tx.loc[future, "Quantity"] = tx.loc[future, "Quantity"] * factor[future]
    tx.loc[future, "Price"] = tx.loc[future, "Price"] * 2
    tx.loc[future, "Description"] = "ALTERED"
    vanished = sorted(tx.loc[future, "StockCode"].unique())[0]
    tx = tx[~(future & (tx["StockCode"] == vanished))]
    newcomer = tx[tx["InvoiceDate"] >= boundary].head(50).assign(StockCode="99999")
    return _rebuild(data, pd.concat([tx, newcomer]), data.data_end)


def leak_findings(data: IngestResult, t: pd.Timestamp, leak: Leak | None = None) -> list[str]:
    """Which artefacts for origins up to t differ between truncated and altered-full data."""
    past = build_panel(truncated(data, t))
    full = build_panel(altered(data, t))
    findings = []

    history = full.history(t).reset_index(drop=True)
    if not past.frame.equals(history):
        findings.append("panel")

    past_features = origin_features(past.frame, past.first_sale_week, leak=leak)
    full_features = origin_features(full.frame, full.first_sale_week, leak=leak)
    full_features = full_features[full_features["week"] <= t].reset_index(drop=True)
    if not past_features.equals(full_features):
        findings.append("features")

    products = sorted(past.first_sale_week.index)
    past_rows = training_rows(past, t, products, SETTINGS.horizons, leak=leak)
    full_rows = training_rows(full, t, products, SETTINGS.horizons, leak=leak)
    if not past_rows.equals(full_rows):
        findings.append("training_rows")
    return findings


def fold_cutoffs(panel: Panel) -> list[pd.Timestamp]:
    return backtest_cutoffs(panel.weeks, SETTINGS)


def training_origins(panel: Panel) -> list[pd.Timestamp]:
    """Origins inside the first fold's training window, plus the publication cutoff."""
    first_fold = fold_cutoffs(panel)[0]
    early = [w for w in panel.weeks if w < first_fold]
    return [early[2], early[len(early) // 2], early[-1], panel.last_complete_week]


def test_truncation_equivalence_at_fold_cutoffs(data: IngestResult, full_panel: Panel) -> None:
    for t in fold_cutoffs(full_panel):
        assert leak_findings(data, t) == [], t


def test_truncation_equivalence_at_training_origins(data: IngestResult, full_panel: Panel) -> None:
    for t in training_origins(full_panel):
        assert leak_findings(data, t) == [], t


def test_alteration_really_changes_the_future(data: IngestResult, full_panel: Panel) -> None:
    t = fold_cutoffs(full_panel)[0]
    original = full_panel.frame[full_panel.frame["week"] > t].reset_index(drop=True)
    changed = build_panel(altered(data, t))
    future = changed.frame[changed.frame["week"] > t].reset_index(drop=True)
    assert not original.equals(future)
    assert "99999" in set(future["stock_code"])


def test_universe_ignores_altered_future(data: IngestResult, full_panel: Panel) -> None:
    # A top-20 cut out of 60 products, so the ranking decides membership.
    settings = replace(SETTINGS, universe_size=20, min_universe_size=10)
    for t in [*fold_cutoffs(full_panel), full_panel.last_complete_week]:
        past = build_panel(truncated(data, t))
        full = build_panel(altered(data, t))
        expected = select_universe(past, t, settings)
        assert len(expected) == 20
        assert select_universe(full, t, settings) == expected, t


def test_training_labels_are_observable_at_every_fold_cutoff(full_panel: Panel) -> None:
    products = sorted(full_panel.first_sale_week.index)
    for t in fold_cutoffs(full_panel):
        rows = training_rows(full_panel, t, products, SETTINGS.horizons)
        assert not rows.empty
        assert label_violations(rows, t).empty, t


def test_planted_future_window_is_caught_by_truncation(
    data: IngestResult, full_panel: Panel
) -> None:
    for t in [*fold_cutoffs(full_panel), *training_origins(full_panel)[:-1]]:
        assert "features" in leak_findings(data, t, leak="future_window"), t


def test_planted_origin_filter_is_caught_by_label_eligibility(full_panel: Panel) -> None:
    products = sorted(full_panel.first_sale_week.index)
    for t in fold_cutoffs(full_panel):
        leaky = training_rows(full_panel, t, products, SETTINGS.horizons, leak="origin_filter")
        assert not label_violations(leaky, t).empty, t
