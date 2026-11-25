"""Replay — score each monitoring period and measure drift against reference."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from drift.config import REPORTS_DIR
from drift.drift import PeriodDriftResult, compute_feature_drift, compute_psi_scores
from drift.metrics import compute_metrics
from drift.pipeline import TrainedModel


def replay_monitoring(
    df: pd.DataFrame,
    model: TrainedModel,
    year_col: str,
    reference_years: list[int],
    monitoring_years: list[int],
    reference_auc: float,
    prepare_fn: Callable,
) -> list[PeriodDriftResult]:
    """Score each monitoring year, measure drift against reference.

    Parameters
    ----------
    df : DataFrame
        Full labelled dataset with a year column and ``default`` column.
    model : TrainedModel
        Baseline model (from Stage 2).
    year_col : str
        Column that holds the period (e.g. ``ApprovalFY``).
    reference_years : list[int]
        Years the model was trained on.
    monitoring_years : list[int]
        Years to replay through.
    reference_auc : float
        Out-of-sample AUC from baseline evaluation.
    prepare_fn : callable
        Feature preparation function (e.g. ``prepare_sba_features``).

    Returns
    -------
    list[PeriodDriftResult]
        One result per monitoring year, in chronological order.
    """
    # Prepare reference features (once)
    ref_mask = df[year_col].isin(reference_years) & df["default"].notna()
    ref_df = df.loc[ref_mask]
    ref_X, ref_y, _, cat_cols, num_cols = prepare_fn(ref_df)

    # Score reference data to get reference score distribution
    ref_probs = _score_raw(model, ref_X)

    results: list[PeriodDriftResult] = []

    for year in sorted(monitoring_years):
        yr_mask = (df[year_col] == year) & df["default"].notna()
        yr_df = df.loc[yr_mask]

        if len(yr_df) == 0:
            continue

        yr_X, yr_y, _, _, _ = prepare_fn(yr_df)

        # Score with baseline model
        yr_probs = _score_raw(model, yr_X)

        # Model performance (retrospective — labels available in our data)
        yr_metrics = None
        if len(np.unique(yr_y.dropna().values)) >= 2:
            yr_metrics = compute_metrics(yr_y.values, yr_probs)

        # Feature drift vs reference
        feature_drifts = compute_feature_drift(ref_X, yr_X, num_cols, cat_cols)

        # Score drift vs reference
        score_psi = compute_psi_scores(ref_probs, yr_probs)

        # AUC drop from reference baseline
        auc = yr_metrics.auc if yr_metrics else None
        auc_drop = (reference_auc - auc) if auc is not None else None

        n_defaults = int(yr_y.sum())

        results.append(PeriodDriftResult(
            period=year,
            n_samples=len(yr_y),
            n_defaults=n_defaults,
            default_rate=n_defaults / max(len(yr_y), 1),
            score_psi=score_psi,
            feature_drifts=feature_drifts,
            auc=auc,
            auc_drop=auc_drop,
            reference_auc=reference_auc,
        ))

    return results


def _score_raw(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Score a DataFrame with the model, applying rare-category grouping."""
    X_scored = X.copy()
    for col in model.categorical_cols:
        if col in X_scored.columns:
            keep = model.rare_map.get(col, set())
            if keep:
                X_scored[col] = X_scored[col].where(X_scored[col].isin(keep), "Other")
    return model.pipeline.predict_proba(X_scored)[:, 1]


def save_replay_results(
    results: list[PeriodDriftResult],
    filename: str = "sba_drift_replay.json",
) -> str:
    """Save replay results to reports/ as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename

    data = {
        "dataset": "sba",
        "n_periods": len(results),
        "periods": [r.summary_dict() for r in results],
        "alert_summary": {
            "green": sum(1 for r in results if r.overall_alert == "green"),
            "amber": sum(1 for r in results if r.overall_alert == "amber"),
            "red": sum(1 for r in results if r.overall_alert == "red"),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return str(path)
