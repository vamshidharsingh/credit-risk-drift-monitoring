"""Tests for the label delay function."""

import pandas as pd
import pytest

from drift.label_delay import known_periods, outcomes_known


class TestOutcomesKnown:
    """Label delay: outcomes available only after the full window."""

    # --- Freddie Mac (24-month window) ---

    def test_freddie_year_too_recent(self):
        # Origination 2020, latest first payment Dec 2020, +24m = Dec 2022
        # At replay year 2022, outcomes NOT yet known (need to be past end of 2022)
        assert outcomes_known(2020, 2022, "freddie") is False

    def test_freddie_year_just_known(self):
        # At replay year 2023, 2020 outcomes are known
        assert outcomes_known(2020, 2023, "freddie") is True

    def test_freddie_timestamp_before_window(self):
        assert outcomes_known(2020, pd.Timestamp("2022-11-30"), "freddie") is False

    def test_freddie_timestamp_after_window(self):
        assert outcomes_known(2020, pd.Timestamp("2023-01-01"), "freddie") is True

    # --- SBA (5-year window) ---

    def test_sba_year_too_recent(self):
        # FY2015 ends Sep 30 2015, +5 years = Sep 30 2020
        # At replay year 2020, NOT yet known (need to be past 2020)
        assert outcomes_known(2015, 2020, "sba") is False

    def test_sba_year_just_known(self):
        assert outcomes_known(2015, 2021, "sba") is True

    def test_sba_timestamp_before_window(self):
        assert outcomes_known(2015, pd.Timestamp("2020-09-29"), "sba") is False

    def test_sba_timestamp_at_window(self):
        assert outcomes_known(2015, pd.Timestamp("2020-09-30"), "sba") is True

    # --- UCI ---

    def test_uci_always_known(self):
        assert outcomes_known(2005, 2005, "uci") is True

    # --- Edge cases ---

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            outcomes_known(2020, 2025, "mystery")


class TestKnownPeriods:
    """Filter periods to those with known outcomes."""

    def test_sba_filtering(self):
        periods = [2005, 2010, 2015, 2020, 2025]
        result = known_periods(periods, 2021, "sba")
        assert result == [2005, 2010, 2015]

    def test_freddie_filtering(self):
        periods = [2005, 2010, 2015, 2020]
        result = known_periods(periods, 2018, "freddie")
        assert result == [2005, 2010, 2015]
