"""UCI synthetic drift benchmark.

No real time dimension — we split the 30k rows into a reference set (40%)
and create 20 simulated periods with progressively increasing drift:
  - Periods 1-5: mild drift (small mean-shift on numerics)
  - Periods 6-10: moderate drift (larger shifts + class imbalance change)
  - Periods 11-15: severe drift (distribution replacement on key features)
  - Periods 16-20: recovery (drift decreases back toward reference)

This exercises the drift detector on known ground truth so we can verify
it correctly ranks green → amber → red → green.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift.config import REPORTS_DIR, SEED
from drift.drift import (
    PeriodDriftResult,
    compute_feature_drift,
    compute_psi_scores,
)
from drift.features import prepare_uci_features
from drift.uci.loader import load_uci


@dataclass
class UCIPeriod:
    """One simulated monitoring period."""

    period_id: int
    phase: str
    drift_intensity: float
    df: pd.DataFrame


def _apply_drift(
    df: pd.DataFrame,
    numeric_cols: list[str],
    intensity: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply synthetic drift to a sample."""
    out = df.copy()
    for col in numeric_cols:
        std = out[col].std()
        if std > 0:
            shift = intensity * std * rng.choice([-1, 1])
            noise = rng.normal(0, abs(intensity) * std * 0.3, size=len(out))
            out[col] = out[col] + shift + noise
    return out


def _adjust_default_rate(
    y: pd.Series,
    target_rate: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Resample labels to hit a target default rate."""
    y = y.copy()
    current_rate = y.mean()
    if abs(current_rate - target_rate) < 0.01:
        return y

    n = len(y)
    n_target_defaults = int(target_rate * n)
    n_current_defaults = int(y.sum())

    if n_target_defaults > n_current_defaults:
        non_default_idx = y[y == 0].index
        n_flip = min(n_target_defaults - n_current_defaults, len(non_default_idx))
        flip_idx = rng.choice(non_default_idx, size=n_flip, replace=False)
        y.loc[flip_idx] = 1
    elif n_target_defaults < n_current_defaults:
        default_idx = y[y == 1].index
        n_flip = min(n_current_defaults - n_target_defaults, len(default_idx))
        flip_idx = rng.choice(default_idx, size=n_flip, replace=False)
        y.loc[flip_idx] = 0

    return y


def generate_periods(
    df_pool: pd.DataFrame,
    y_pool: pd.Series,
    numeric_cols: list[str],
    n_periods: int = 20,
    samples_per_period: int = 600,
    seed: int = SEED,
) -> list[UCIPeriod]:
    """Generate n_periods of progressively drifted data."""
    rng = np.random.default_rng(seed)
    base_default_rate = y_pool.mean()
    periods = []

    drift_schedule = []
    for i in range(1, n_periods + 1):
        if i <= 5:
            intensity = 0.1 * i
            phase = "mild"
            dr_shift = 0.0
        elif i <= 10:
            intensity = 0.5 + 0.2 * (i - 5)
            phase = "moderate"
            dr_shift = 0.05 * (i - 5)
        elif i <= 15:
            intensity = 1.5 + 0.3 * (i - 10)
            phase = "severe"
            dr_shift = 0.10 + 0.03 * (i - 10)
        else:
            intensity = max(0.1, 3.0 - 0.5 * (i - 15))
            phase = "recovery"
            dr_shift = max(0.0, 0.25 - 0.05 * (i - 15))
        drift_schedule.append((i, phase, intensity, dr_shift))

    for period_id, phase, intensity, dr_shift in drift_schedule:
        idx = rng.choice(len(df_pool), size=samples_per_period, replace=True)
        sample = df_pool.iloc[idx].reset_index(drop=True)
        y_sample = y_pool.iloc[idx].reset_index(drop=True)

        sample = _apply_drift(sample, numeric_cols, intensity, rng)

        target_rate = min(base_default_rate + dr_shift, 0.60)
        y_sample = _adjust_default_rate(y_sample, target_rate, rng)
        sample["DEFAULT"] = y_sample.values

        periods.append(UCIPeriod(
            period_id=period_id,
            phase=phase,
            drift_intensity=intensity,
            df=sample,
        ))

    return periods


def run_uci_benchmark(seed: int = SEED) -> dict:
    """Run the full UCI synthetic drift benchmark.

    Returns a results dict saved to reports/uci_drift_benchmark.json.
    """
    rng = np.random.default_rng(seed)
    df = load_uci()

    # Shuffle and split: 40% reference, 60% pool for simulated periods
    idx = rng.permutation(len(df))
    n_ref = int(0.4 * len(df))
    ref_idx = idx[:n_ref]
    pool_idx = idx[n_ref:]

    df_ref = df.iloc[ref_idx].reset_index(drop=True)
    df_pool = df.iloc[pool_idx].reset_index(drop=True)

    # Prepare features
    X_ref, y_ref, feature_names, categorical_cols, numeric_cols = prepare_uci_features(df_ref)
    X_pool, y_pool, _, _, _ = prepare_uci_features(df_pool)

    # Train baseline GBM on reference (UCI has no time dimension — use stratified k-fold)
    print("Training baseline GBM on UCI reference split...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=20, random_state=seed,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(pipe, X_ref, y_ref, cv=cv, scoring="roc_auc")
    print(f"  Baseline AUC (5-fold CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_ref, y_ref)

    # Score reference to get baseline score distribution
    ref_scores = pipe.predict_proba(X_ref)[:, 1]
    ref_auc = roc_auc_score(y_ref, ref_scores)
    print(f"  Baseline AUC (reference): {ref_auc:.4f}")

    # Generate simulated periods
    print("Generating 20 simulated drift periods...")
    periods = generate_periods(
        df_pool, y_pool, numeric_cols,
        n_periods=20, samples_per_period=600, seed=seed,
    )

    # Run drift detection on each period
    results = {
        "dataset": "uci",
        "n_reference": len(df_ref),
        "n_pool": len(df_pool),
        "baseline_auc": round(ref_auc, 4),
        "baseline_cv_auc": round(float(cv_scores.mean()), 4),
        "seed": seed,
        "periods": [],
    }

    for p in periods:
        X_p, y_p, _, _, _ = prepare_uci_features(p.df)

        # Score with baseline model
        scores_p = pipe.predict_proba(X_p)[:, 1]
        try:
            auc_p = roc_auc_score(y_p, scores_p)
        except ValueError:
            auc_p = None

        # Score PSI
        score_psi = compute_psi_scores(ref_scores, scores_p)

        # Feature drift
        feature_drifts = compute_feature_drift(
            X_ref, X_p, numeric_cols, categorical_cols,
        )

        auc_drop = (ref_auc - auc_p) if auc_p is not None else None

        period_result = PeriodDriftResult(
            period=p.period_id,
            n_samples=len(p.df),
            n_defaults=int(y_p.sum()),
            default_rate=float(y_p.mean()),
            score_psi=score_psi,
            feature_drifts=feature_drifts,
            auc=auc_p,
            auc_drop=auc_drop,
            reference_auc=ref_auc,
        )

        period_dict = period_result.summary_dict()
        period_dict["phase"] = p.phase
        period_dict["drift_intensity"] = round(p.drift_intensity, 2)
        results["periods"].append(period_dict)

        alert = period_result.overall_alert
        auc_str = f"{auc_p:.4f}" if auc_p is not None else "N/A"
        print(
            f"  Period {p.period_id:2d} ({p.phase:8s}) | "
            f"PSI={score_psi:.4f} | AUC={auc_str} | Alert={alert}"
        )

    # Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "uci_drift_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results
