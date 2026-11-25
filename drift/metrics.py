"""Evaluation metrics — AUC, Gini, KS, PR-AUC, Brier, calibration, cost threshold."""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from drift.config import CALIBRATION_BINS, COST_RATIOS


@dataclass
class ModelMetrics:
    """Full evaluation metrics for one model on one dataset."""

    auc: float
    gini: float
    ks: float
    pr_auc: float
    brier: float
    calibration_bins: list[dict]
    overall_predicted: float
    overall_actual: float
    threshold_primary: float
    threshold_low: float
    threshold_high: float
    n_samples: int
    n_defaults: int

    def summary_dict(self) -> dict:
        return {
            "AUC": round(self.auc, 4),
            "Gini": round(self.gini, 4),
            "KS": round(self.ks, 4),
            "PR-AUC": round(self.pr_auc, 4),
            "Brier": round(self.brier, 4),
            "Pred default %": round(self.overall_predicted * 100, 2),
            "Actual default %": round(self.overall_actual * 100, 2),
            "Threshold (5:1)": round(self.threshold_primary, 4),
            "Threshold (3:1)": round(self.threshold_low, 4),
            "Threshold (10:1)": round(self.threshold_high, 4),
            "Samples": self.n_samples,
            "Defaults": self.n_defaults,
        }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
    """Compute all evaluation metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # KS statistic
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    ks_stat = stats.ks_2samp(pos, neg).statistic

    # Calibration
    cal_bins = _calibration_bins(y_true, y_prob, CALIBRATION_BINS)
    overall_predicted = float(y_prob.mean())
    overall_actual = float(y_true.mean())

    # Cost-sensitive thresholds
    t_primary = _cost_threshold(y_true, y_prob, COST_RATIOS["primary"])
    t_low = _cost_threshold(y_true, y_prob, COST_RATIOS["sensitivity_low"])
    t_high = _cost_threshold(y_true, y_prob, COST_RATIOS["sensitivity_high"])

    return ModelMetrics(
        auc=auc,
        gini=gini,
        ks=ks_stat,
        pr_auc=pr_auc,
        brier=brier,
        calibration_bins=cal_bins,
        overall_predicted=overall_predicted,
        overall_actual=overall_actual,
        threshold_primary=t_primary,
        threshold_low=t_low,
        threshold_high=t_high,
        n_samples=len(y_true),
        n_defaults=int(y_true.sum()),
    )


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> list[dict]:
    """Compute calibration: predicted vs actual in equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi)
        if hi == 1.0:
            mask = mask | (y_prob == 1.0)
        n = int(mask.sum())
        if n == 0:
            continue
        result.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "count": n,
            "mean_predicted": round(float(y_prob[mask].mean()), 4),
            "mean_actual": round(float(y_true[mask].mean()), 4),
        })
    return result


def _cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratio: float,
) -> float:
    """Find the threshold that minimises expected cost.

    cost_ratio = cost of missed default / cost of wrongly declining.
    At threshold t:
      - FN (missed defaults) cost: cost_ratio per FN
      - FP (wrongly declined) cost: 1 per FP
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = np.inf
    best_t = 0.5

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        cost = fn * cost_ratio + fp
        if cost < best_cost:
            best_cost = cost
            best_t = t

    return float(best_t)
