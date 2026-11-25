"""Retraining strategies — never, scheduled, drift-triggered.

Compares three approaches:
1. Never retrain: original baseline for all periods.
2. Scheduled: retrain every year using the most recent data with known labels.
3. Drift-triggered: retrain only when drift alert is red.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from drift.config import (
    N_BOOTSTRAP_FAST,
    PROMOTION_AUC_THRESHOLD,
    REPORTS_DIR,
    ROLLING_WINDOW_YEARS,
    SEED,
)
from drift.drift import PeriodDriftResult
from drift.features import group_rare_categories
from drift.label_delay import known_periods
from drift.pipeline import TrainedModel, _build_gbm_pipeline, score_period

# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class RetrainEvent:
    """Record of a retraining attempt."""

    period: int
    training_years: list[int]
    comparison_year: int
    champion_auc_comp: float
    challenger_auc_comp: float
    auc_diff: float
    ci_lo: float
    ci_hi: float
    promoted: bool

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "training_years": self.training_years,
            "comparison_year": self.comparison_year,
            "champion_auc": round(self.champion_auc_comp, 4),
            "challenger_auc": round(self.challenger_auc_comp, 4),
            "auc_diff": round(self.auc_diff, 4),
            "ci_lo": round(self.ci_lo, 4),
            "ci_hi": round(self.ci_hi, 4),
            "promoted": self.promoted,
        }


@dataclass
class PeriodResult:
    """Per-period result for one strategy."""

    period: int
    n_samples: int
    n_defaults: int
    default_rate: float
    auc: float | None
    champion_name: str
    champion_training_years: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "n_samples": self.n_samples,
            "default_rate_pct": round(self.default_rate * 100, 2),
            "auc": round(self.auc, 4) if self.auc is not None else None,
            "champion": self.champion_name,
            "training_years": self.champion_training_years,
        }


@dataclass
class StrategyResult:
    """Full result for one retraining strategy."""

    name: str
    periods: list[PeriodResult]
    retrain_events: list[RetrainEvent]
    final_eval: list[PeriodResult]

    @property
    def n_retrains(self) -> int:
        return sum(1 for e in self.retrain_events if e.promoted)

    @property
    def n_attempts(self) -> int:
        return len(self.retrain_events)

    @property
    def mean_monitoring_auc(self) -> float:
        aucs = [p.auc for p in self.periods if p.auc is not None]
        return float(np.mean(aucs)) if aucs else 0.0

    @property
    def final_mean_auc(self) -> float:
        aucs = [p.auc for p in self.final_eval if p.auc is not None]
        return float(np.mean(aucs)) if aucs else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_retrains": self.n_retrains,
            "n_attempts": self.n_attempts,
            "mean_monitoring_auc": round(self.mean_monitoring_auc, 4),
            "final_mean_auc": round(self.final_mean_auc, 4),
            "periods": [p.to_dict() for p in self.periods],
            "retrain_events": [e.to_dict() for e in self.retrain_events],
            "final_eval": [p.to_dict() for p in self.final_eval],
        }


# ── Bootstrap comparison ───────────────────────────────────────────────────


def bootstrap_auc_comparison(
    y_true: np.ndarray,
    y_prob_champion: np.ndarray,
    y_prob_challenger: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_FAST,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Bootstrap the AUC difference (challenger - champion).

    Returns (mean_diff, ci_2.5, ci_97.5).
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_champ = roc_auc_score(y_b, y_prob_champion[idx])
        auc_chall = roc_auc_score(y_b, y_prob_challenger[idx])
        diffs.append(auc_chall - auc_champ)

    if not diffs:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


# ── Retrain a GBM ──────────────────────────────────────────────────────────


def retrain_gbm(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    year_col: str,
    training_years: list[int],
    categorical_cols: list[str],
    numeric_cols: list[str],
    best_params: dict,
) -> TrainedModel:
    """Retrain a GBM on specified years with given hyperparameters."""
    mask = df[year_col].isin(training_years) & y.notna()
    tr_X = X.loc[mask].reset_index(drop=True)
    tr_y = y.loc[mask].astype(int).reset_index(drop=True)

    tr_X, _, rare_map = group_rare_categories(tr_X, None, categorical_cols)

    pipe = _build_gbm_pipeline(numeric_cols, categorical_cols)
    pipe_params = {f"classifier__{k}": v for k, v in best_params.items()}
    pipe.set_params(**pipe_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(tr_X, tr_y)

    year_str = f"{min(training_years)}-{max(training_years)}"
    return TrainedModel(
        name=f"gbm_FY{year_str}",
        pipeline=pipe,
        rare_map=rare_map,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        training_years=sorted(training_years),
        best_params=best_params,
    )


# ── Attempt retrain ────────────────────────────────────────────────────────


def _attempt_retrain(
    champion: TrainedModel,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    year_col: str,
    known_yrs: list[int],
    categorical_cols: list[str],
    numeric_cols: list[str],
    best_params: dict,
    period: int,
) -> tuple[TrainedModel, RetrainEvent | None]:
    """Train a challenger, compare with champion, decide promotion."""
    min_years = ROLLING_WINDOW_YEARS + 1
    if len(known_yrs) < min_years:
        return champion, None

    sorted_known = sorted(known_yrs)
    comparison_year = sorted_known[-1]
    training_years = sorted_known[-(ROLLING_WINDOW_YEARS + 1):-1]

    challenger = retrain_gbm(
        df, X, y, year_col, training_years,
        categorical_cols, numeric_cols, best_params,
    )

    # Score comparison year with both models
    comp_mask = (df[year_col] == comparison_year) & y.notna()
    comp_X = X.loc[comp_mask]
    comp_y = y.loc[comp_mask]

    if len(comp_y) == 0 or len(np.unique(comp_y.values)) < 2:
        return champion, None

    champ_probs, champ_metrics = score_period(champion, comp_X, comp_y)
    chall_probs, chall_metrics = score_period(challenger, comp_X, comp_y)

    auc_diff, ci_lo, ci_hi = bootstrap_auc_comparison(
        comp_y.values, champ_probs, chall_probs,
    )

    promoted = chall_metrics.auc > champ_metrics.auc + PROMOTION_AUC_THRESHOLD

    event = RetrainEvent(
        period=period,
        training_years=training_years,
        comparison_year=comparison_year,
        champion_auc_comp=champ_metrics.auc,
        challenger_auc_comp=chall_metrics.auc,
        auc_diff=auc_diff,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        promoted=promoted,
    )

    return (challenger if promoted else champion), event


# ── Strategy runner ─────────────────────────────────────────────────────────


def run_strategy(
    name: str,
    df: pd.DataFrame,
    baseline_model: TrainedModel,
    X: pd.DataFrame,
    y: pd.Series,
    year_col: str,
    all_usable_years: list[int],
    monitoring_years: list[int],
    final_eval_years: list[int],
    categorical_cols: list[str],
    numeric_cols: list[str],
    best_params: dict,
    drift_results: list[PeriodDriftResult] | None = None,
    dataset: str = "sba",
) -> StrategyResult:
    """Run one retraining strategy through monitoring and final evaluation."""
    champion = baseline_model
    period_results: list[PeriodResult] = []
    retrain_events: list[RetrainEvent] = []

    drift_map: dict[int, PeriodDriftResult] = {}
    if drift_results:
        drift_map = {r.period: r for r in drift_results}

    for year in sorted(monitoring_years):
        # Score this year with current champion
        mask = (df[year_col] == year) & y.notna()
        yr_X = X.loc[mask]
        yr_y = y.loc[mask]

        auc = None
        if len(yr_y) > 0 and len(np.unique(yr_y.values)) >= 2:
            _, metrics = score_period(champion, yr_X, yr_y)
            auc = metrics.auc

        n_def = int(yr_y.sum()) if len(yr_y) > 0 else 0
        period_results.append(PeriodResult(
            period=year,
            n_samples=len(yr_y),
            n_defaults=n_def,
            default_rate=n_def / max(len(yr_y), 1),
            auc=auc,
            champion_name=champion.name,
            champion_training_years=list(champion.training_years),
        ))

        # Decide whether to attempt retrain
        should_retrain = False
        if name == "scheduled":
            should_retrain = True
        elif name == "drift_triggered":
            dr = drift_map.get(year)
            should_retrain = dr is not None and dr.overall_alert == "red"

        if should_retrain:
            known_yrs = known_periods(all_usable_years, year, dataset)
            champion, event = _attempt_retrain(
                champion, df, X, y, year_col, known_yrs,
                categorical_cols, numeric_cols, best_params, year,
            )
            if event:
                retrain_events.append(event)

    # Final evaluation — score once with final champion
    final_results: list[PeriodResult] = []
    for year in sorted(final_eval_years):
        mask = (df[year_col] == year) & y.notna()
        yr_X = X.loc[mask]
        yr_y = y.loc[mask]

        auc = None
        if len(yr_y) > 0 and len(np.unique(yr_y.values)) >= 2:
            _, metrics = score_period(champion, yr_X, yr_y)
            auc = metrics.auc

        n_def = int(yr_y.sum()) if len(yr_y) > 0 else 0
        final_results.append(PeriodResult(
            period=year,
            n_samples=len(yr_y),
            n_defaults=n_def,
            default_rate=n_def / max(len(yr_y), 1),
            auc=auc,
            champion_name=champion.name,
            champion_training_years=list(champion.training_years),
        ))

    return StrategyResult(
        name=name,
        periods=period_results,
        retrain_events=retrain_events,
        final_eval=final_results,
    )


# ── Save ────────────────────────────────────────────────────────────────────


def save_strategy_results(
    strategies: list[StrategyResult],
    filename: str = "sba_strategy_comparison.json",
) -> str:
    """Save strategy comparison results to reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename

    data = {
        "dataset": "sba",
        "strategies": [s.to_dict() for s in strategies],
        "comparison": {
            s.name: {
                "mean_monitoring_auc": round(s.mean_monitoring_auc, 4),
                "final_mean_auc": round(s.final_mean_auc, 4),
                "n_retrains": s.n_retrains,
                "n_attempts": s.n_attempts,
            }
            for s in strategies
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return str(path)
