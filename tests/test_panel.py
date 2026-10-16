from __future__ import annotations

import pandas as pd

from forecastgate.panel import build_panel, complete_weeks
from tests.conftest import ingest_result


def test_partial_first_and_last_weeks_are_dropped() -> None:
    # Tuesday 2009-12-01 07:45 to Friday 2011-12-09 12:50, as in the real file.
    weeks = complete_weeks(pd.Timestamp("2009-12-01 07:45"), pd.Timestamp("2011-12-09 12:50"))
    assert weeks[0] == pd.Timestamp("2009-12-07")
    assert weeks[-1] == pd.Timestamp("2011-11-28")


def test_week_starting_exactly_at_data_start_is_complete() -> None:
    weeks = complete_weeks(pd.Timestamp("2010-01-04 00:00"), pd.Timestamp("2010-01-11 00:00"))
    assert list(weeks) == [pd.Timestamp("2010-01-04")]


def test_panel_rows_start_at_first_sale_and_fill_gaps() -> None:
    data = ingest_result(
        [
            ("1", "10001", 5, "2010-01-05 10:00", 1.0),
            ("2", "10001", 7, "2010-01-19 10:00", 1.0),
            ("3", "10002", 3, "2010-01-12 10:00", 1.0),
        ],
        start="2010-01-04",
        end="2010-01-25 12:00",
    )
    panel = build_panel(data)
    frame = panel.frame.set_index(["stock_code", "week"])["quantity"]
    assert frame[("10001", pd.Timestamp("2010-01-04"))] == 5
    assert frame[("10001", pd.Timestamp("2010-01-11"))] == 0  # no sale: 0 observed
    assert frame[("10001", pd.Timestamp("2010-01-18"))] == 7
    assert ("10002", pd.Timestamp("2010-01-04")) not in frame.index  # before first sale
    assert panel.first_sale_week["10002"] == pd.Timestamp("2010-01-11")


def test_zero_transaction_week_is_missing_not_zero() -> None:
    data = ingest_result(
        [
            ("1", "10001", 5, "2010-01-05 10:00", 1.0),
            # nothing at all in the week of 2010-01-11
            ("2", "10001", 7, "2010-01-19 10:00", 1.0),
        ],
        start="2010-01-04",
        end="2010-01-25 12:00",
    )
    panel = build_panel(data)
    assert list(panel.zero_transaction_weeks) == [pd.Timestamp("2010-01-11")]
    value = panel.frame.set_index("week").loc[pd.Timestamp("2010-01-11"), "quantity"]
    assert pd.isna(value)


def test_week_with_only_excluded_rows_is_zero_not_missing() -> None:
    data = ingest_result(
        [
            ("1", "10001", 5, "2010-01-05 10:00", 1.0),
            ("C2", "10001", -5, "2010-01-12 10:00", 1.0),  # a transaction, but no sale
            ("3", "10001", 7, "2010-01-19 10:00", 1.0),
        ],
        start="2010-01-04",
        end="2010-01-25 12:00",
    )
    panel = build_panel(data)
    assert len(panel.zero_transaction_weeks) == 0
    value = panel.frame.set_index("week").loc[pd.Timestamp("2010-01-11"), "quantity"]
    assert value == 0
