"""Tests for drift detection — PSI, KS, chi-square, alerts."""

import numpy as np
import pandas as pd
import pytest

from drift.drift import (
    FeatureDrift,
    PeriodDriftResult,
    compute_chi2_categorical,
    compute_feature_drift,
    compute_ks_numeric,
    compute_psi_categorical,
    compute_psi_numeric,
    compute_psi_scores,
)


class TestPSINumeric:
    def test_identical_distribution_is_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 10_000)
        assert compute_psi_numeric(x, x) == pytest.approx(0, abs=0.01)

    def test_shifted_distribution_is_positive(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 10_000)
        cur = rng.normal(1, 1, 10_000)
        assert compute_psi_numeric(ref, cur) > 0.1

    def test_binary_feature_shift_detected(self):
        ref = np.array([1.0] * 1000)
        cur = np.array([1.0] * 700 + [0.0] * 300)
        assert compute_psi_numeric(ref, cur) > 0.1

    def test_empty_reference_returns_zero(self):
        assert compute_psi_numeric(np.array([]), np.array([1, 2, 3])) == 0.0

    def test_empty_current_returns_zero(self):
        assert compute_psi_numeric(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_nan_values_excluded(self):
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        psi = compute_psi_numeric(ref, ref)
        assert psi == pytest.approx(0, abs=0.01)

    def test_all_same_value_returns_zero(self):
        ref = np.array([5.0] * 100)
        cur = np.array([5.0] * 100)
        assert compute_psi_numeric(ref, cur) == 0.0


class TestPSICategorical:
    def test_identical_is_zero(self):
        s = pd.Series(["A"] * 500 + ["B"] * 500)
        assert compute_psi_categorical(s, s) == pytest.approx(0, abs=0.01)

    def test_shifted_is_positive(self):
        ref = pd.Series(["A"] * 500 + ["B"] * 500)
        cur = pd.Series(["A"] * 800 + ["B"] * 200)
        assert compute_psi_categorical(ref, cur) > 0.05

    def test_new_category_detected(self):
        ref = pd.Series(["A"] * 500 + ["B"] * 500)
        cur = pd.Series(["A"] * 400 + ["B"] * 400 + ["C"] * 200)
        assert compute_psi_categorical(ref, cur) > 0.05

    def test_single_category_returns_zero(self):
        s = pd.Series(["A"] * 100)
        assert compute_psi_categorical(s, s) == 0.0


class TestPSIScores:
    def test_identical_is_zero(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 10_000)
        assert compute_psi_scores(scores, scores) == pytest.approx(0, abs=0.01)

    def test_shifted_is_positive(self):
        rng = np.random.default_rng(42)
        ref = rng.beta(2, 8, 10_000)
        cur = rng.beta(4, 6, 10_000)
        assert compute_psi_scores(ref, cur) > 0.1


class TestKSNumeric:
    def test_identical_is_zero(self):
        x = np.arange(100, dtype=float)
        assert compute_ks_numeric(x, x) == pytest.approx(0, abs=0.001)

    def test_shifted_is_large(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(2, 1, 1000)
        assert compute_ks_numeric(ref, cur) > 0.5

    def test_empty_returns_zero(self):
        assert compute_ks_numeric(np.array([]), np.array([1.0])) == 0.0


class TestChi2Categorical:
    def test_identical_not_significant(self):
        s = pd.Series(["A"] * 500 + ["B"] * 500)
        _, pval = compute_chi2_categorical(s, s)
        assert pval > 0.05

    def test_different_is_significant(self):
        ref = pd.Series(["A"] * 500 + ["B"] * 500)
        cur = pd.Series(["A"] * 800 + ["B"] * 200)
        _, pval = compute_chi2_categorical(ref, cur)
        assert pval < 0.05

    def test_single_category_returns_neutral(self):
        s = pd.Series(["A"] * 100)
        stat, pval = compute_chi2_categorical(s, s)
        assert stat == 0.0
        assert pval == 1.0


class TestFeatureDrift:
    def test_green_alert(self):
        d = FeatureDrift("x", psi=0.05, ks=0.03, chi2_stat=None, chi2_pval=None,
                         is_categorical=False)
        assert d.alert == "green"

    def test_amber_alert(self):
        d = FeatureDrift("x", psi=0.15, ks=0.1, chi2_stat=None, chi2_pval=None,
                         is_categorical=False)
        assert d.alert == "amber"

    def test_red_alert(self):
        d = FeatureDrift("x", psi=0.30, ks=0.2, chi2_stat=None, chi2_pval=None,
                         is_categorical=False)
        assert d.alert == "red"


class TestComputeFeatureDrift:
    def test_returns_sorted_by_psi(self):
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "a": rng.normal(0, 1, 1000),
            "b": rng.normal(0, 1, 1000),
        })
        cur = pd.DataFrame({
            "a": rng.normal(0, 1, 1000),
            "b": rng.normal(5, 1, 1000),  # big shift
        })
        drifts = compute_feature_drift(ref, cur, ["a", "b"], [])
        assert drifts[0].name == "b"
        assert drifts[0].psi > drifts[1].psi


class TestPeriodDriftResult:
    def _make(self, score_psi=0.05, feature_drifts=None, auc_drop=0.01):
        return PeriodDriftResult(
            period=2000, n_samples=1000, n_defaults=50, default_rate=0.05,
            score_psi=score_psi, feature_drifts=feature_drifts or [],
            auc=0.90, auc_drop=auc_drop, reference_auc=0.91,
        )

    def test_green_overall(self):
        assert self._make().overall_alert == "green"

    def test_amber_score_drives_overall(self):
        r = self._make(score_psi=0.15)
        assert r.score_alert == "amber"
        assert r.overall_alert == "amber"

    def test_red_score_drives_overall(self):
        r = self._make(score_psi=0.30)
        assert r.score_alert == "red"
        assert r.overall_alert == "red"

    def test_feature_count_amber(self):
        drifts = [
            FeatureDrift(f"f{i}", 0.15, 0.1, None, None, False) for i in range(3)
        ]
        r = self._make(feature_drifts=drifts)
        assert r.feature_alert == "amber"
        assert r.overall_alert == "amber"

    def test_feature_count_red(self):
        drifts = [
            FeatureDrift(f"f{i}", 0.30, 0.2, None, None, False) for i in range(3)
        ]
        r = self._make(feature_drifts=drifts)
        assert r.feature_alert == "red"
        assert r.overall_alert == "red"

    def test_auc_drop_amber(self):
        r = self._make(auc_drop=0.05)
        assert r.auc_alert == "amber"
        assert r.overall_alert == "amber"

    def test_summary_dict_keys(self):
        d = self._make().summary_dict()
        assert "period" in d
        assert "score_psi" in d
        assert "overall_alert" in d
        assert "top_drifting" in d
