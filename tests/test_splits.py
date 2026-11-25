"""Tests for time-ordered splits."""

import pytest

from drift.splits import compute_split, compute_uci_split


class TestComputeSplit:
    """Split logic: 5 reference + monitoring + 3 final eval."""

    def test_minimum_9_years(self):
        years = list(range(2000, 2009))  # 9 years
        split = compute_split("test", years)
        assert split.reference_years == tuple(range(2000, 2005))
        assert split.monitoring_years == (2005,)
        assert split.final_eval_years == (2006, 2007, 2008)

    def test_larger_set(self):
        years = list(range(2000, 2020))  # 20 years
        split = compute_split("test", years)
        assert len(split.reference_years) == 5
        assert len(split.final_eval_years) == 3
        assert len(split.monitoring_years) == 12
        assert split.reference_years == tuple(range(2000, 2005))
        assert split.final_eval_years == (2017, 2018, 2019)

    def test_no_overlap(self):
        years = list(range(2000, 2015))
        split = compute_split("test", years)
        all_years = set(split.reference_years) | set(split.monitoring_years) | set(split.final_eval_years)
        assert len(all_years) == len(years)
        assert all_years == set(years)

    def test_too_few_years_raises(self):
        with pytest.raises(ValueError, match="at least 9 usable years"):
            compute_split("test", list(range(2000, 2008)))

    def test_sorted_input_not_required(self):
        years = [2010, 2005, 2000, 2015, 2020, 2001, 2002, 2003, 2004]
        split = compute_split("test", years)
        assert split.reference_years == (2000, 2001, 2002, 2003, 2004)

    def test_describe_format(self):
        split = compute_split("sba", list(range(2000, 2010)))
        desc = split.describe()
        assert "sba" in desc
        assert "Reference" in desc
        assert "Monitoring" in desc
        assert "Final eval" in desc


class TestUCISplit:
    def test_uci_split_structure(self):
        result = compute_uci_split()
        assert result["reference_fraction"] == 0.40
        assert result["n_simulated_periods"] == 20
