"""Drift detection — PSI, KS, chi-square, and alert classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from drift.config import (
    ALERT_AMBER_FEATURE_PSI,
    ALERT_AMBER_SCORE_PSI,
    ALERT_GREEN_FEATURE_PSI,
    ALERT_GREEN_SCORE_PSI,
    ALERT_RED_FEATURE_COUNT,
    AUC_DROP_THRESHOLD,
    PSI_BINS,
    PSI_FLOOR,
)

# ── PSI ─────────────────────────────────────────────────────────────────────


def compute_psi_numeric(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = PSI_BINS,
    floor: float = PSI_FLOOR,
) -> float:
    """PSI for a numeric feature.

    Uses quantile-based bins from the reference distribution. For features
    with few unique values (binary indicators, small integers), falls back
    to value-based binning so that shifts from all-1 to mixed 0/1 are detected.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    n_unique_ref = len(np.unique(ref))

    if n_unique_ref <= n_bins:
        all_vals = np.unique(np.concatenate([ref, cur]))
        if len(all_vals) < 2:
            return 0.0
        ref_pcts = np.array([max(float(np.mean(ref == v)), floor) for v in all_vals])
        cur_pcts = np.array([max(float(np.mean(cur == v)), floor) for v in all_vals])
        return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref, quantiles))

    if len(bin_edges) < 2:
        return 0.0

    ref_counts = np.histogram(ref, bins=bin_edges)[0]
    cur_counts = np.histogram(cur, bins=bin_edges)[0]

    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), floor)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), floor)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    floor: float = PSI_FLOOR,
) -> float:
    """PSI for a categorical feature using frequency proportions."""
    all_cats = sorted(set(reference.unique()) | set(current.unique()))
    if len(all_cats) < 2:
        return 0.0

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    ref_total = max(len(reference), 1)
    cur_total = max(len(current), 1)

    psi = 0.0
    for cat in all_cats:
        ref_pct = max(ref_counts.get(cat, 0) / ref_total, floor)
        cur_pct = max(cur_counts.get(cat, 0) / cur_total, floor)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(psi)


def compute_psi_scores(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = PSI_BINS,
    floor: float = PSI_FLOOR,
) -> float:
    """PSI for model scores using equal-width bins on [0, 1]."""
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ref_counts = np.histogram(reference_scores, bins=bin_edges)[0]
    cur_counts = np.histogram(current_scores, bins=bin_edges)[0]

    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), floor)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), floor)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ── KS ──────────────────────────────────────────────────────────────────────


def compute_ks_numeric(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic for a numeric feature."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    return float(sp_stats.ks_2samp(ref, cur).statistic)


# ── Chi-square ──────────────────────────────────────────────────────────────


def compute_chi2_categorical(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Chi-square test of independence between reference and current periods."""
    all_cats = sorted(set(reference.unique()) | set(current.unique()))

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    ref_obs = np.array([ref_counts.get(c, 0) for c in all_cats])
    cur_obs = np.array([cur_counts.get(c, 0) for c in all_cats])

    contingency = np.array([ref_obs, cur_obs])
    nonzero = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero]

    if contingency.shape[1] < 2:
        return 0.0, 1.0

    stat, p_val, _, _ = sp_stats.chi2_contingency(contingency)
    return float(stat), float(p_val)


# ── Feature drift ───────────────────────────────────────────────────────────


@dataclass
class FeatureDrift:
    """Drift result for one feature."""

    name: str
    psi: float
    ks: float | None
    chi2_stat: float | None
    chi2_pval: float | None
    is_categorical: bool

    @property
    def alert(self) -> str:
        if self.psi >= ALERT_AMBER_FEATURE_PSI:
            return "red"
        if self.psi >= ALERT_GREEN_FEATURE_PSI:
            return "amber"
        return "green"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "psi": round(self.psi, 4),
            "ks": round(self.ks, 4) if self.ks is not None else None,
            "chi2_stat": round(self.chi2_stat, 2) if self.chi2_stat is not None else None,
            "chi2_pval": round(self.chi2_pval, 4) if self.chi2_pval is not None else None,
            "alert": self.alert,
            "is_categorical": self.is_categorical,
        }


def compute_feature_drift(
    ref_X: pd.DataFrame,
    cur_X: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[FeatureDrift]:
    """Compute drift for all features, returned sorted by PSI descending."""
    results: list[FeatureDrift] = []

    for col in numeric_cols:
        if col not in ref_X.columns or col not in cur_X.columns:
            continue
        ref_vals = ref_X[col].to_numpy(dtype=float)
        cur_vals = cur_X[col].to_numpy(dtype=float)
        psi = compute_psi_numeric(ref_vals, cur_vals)
        ks = compute_ks_numeric(ref_vals, cur_vals)
        results.append(FeatureDrift(
            name=col, psi=psi, ks=ks,
            chi2_stat=None, chi2_pval=None, is_categorical=False,
        ))

    for col in categorical_cols:
        if col not in ref_X.columns or col not in cur_X.columns:
            continue
        psi = compute_psi_categorical(ref_X[col], cur_X[col])
        chi2_stat, chi2_pval = compute_chi2_categorical(ref_X[col], cur_X[col])
        results.append(FeatureDrift(
            name=col, psi=psi, ks=None,
            chi2_stat=chi2_stat, chi2_pval=chi2_pval, is_categorical=True,
        ))

    return sorted(results, key=lambda d: d.psi, reverse=True)


# ── Period result and alert classification ──────────────────────────────────


@dataclass
class PeriodDriftResult:
    """Full drift result for one monitoring period."""

    period: int
    n_samples: int
    n_defaults: int
    default_rate: float
    score_psi: float
    feature_drifts: list[FeatureDrift]
    auc: float | None
    auc_drop: float | None
    reference_auc: float

    @property
    def score_alert(self) -> str:
        if self.score_psi >= ALERT_AMBER_SCORE_PSI:
            return "red"
        if self.score_psi >= ALERT_GREEN_SCORE_PSI:
            return "amber"
        return "green"

    @property
    def n_features_amber(self) -> int:
        return sum(1 for d in self.feature_drifts if d.alert in ("amber", "red"))

    @property
    def n_features_red(self) -> int:
        return sum(1 for d in self.feature_drifts if d.alert == "red")

    @property
    def feature_alert(self) -> str:
        if self.n_features_red >= ALERT_RED_FEATURE_COUNT:
            return "red"
        if self.n_features_amber >= ALERT_RED_FEATURE_COUNT:
            return "amber"
        return "green"

    @property
    def auc_alert(self) -> str:
        if self.auc_drop is not None and self.auc_drop >= AUC_DROP_THRESHOLD:
            return "amber"
        return "green"

    @property
    def overall_alert(self) -> str:
        alerts = [self.score_alert, self.feature_alert, self.auc_alert]
        if "red" in alerts:
            return "red"
        if "amber" in alerts:
            return "amber"
        return "green"

    def summary_dict(self) -> dict:
        return {
            "period": self.period,
            "n_samples": self.n_samples,
            "default_rate_pct": round(self.default_rate * 100, 2),
            "score_psi": round(self.score_psi, 4),
            "score_alert": self.score_alert,
            "n_features_amber": self.n_features_amber,
            "n_features_red": self.n_features_red,
            "feature_alert": self.feature_alert,
            "auc_alert": self.auc_alert,
            "overall_alert": self.overall_alert,
            "auc": round(self.auc, 4) if self.auc is not None else None,
            "auc_drop": round(self.auc_drop, 4) if self.auc_drop is not None else None,
            "top_drifting": [d.to_dict() for d in self.feature_drifts[:5]],
        }
