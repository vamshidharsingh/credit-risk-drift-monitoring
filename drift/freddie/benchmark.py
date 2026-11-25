"""Freddie Mac full pipeline — load, label, split, train, replay, retrain.

Runs the complete drift monitoring pipeline on Freddie Mac SFLLD sample data:
  1. Load origination + performance for all available years
  2. Derive default labels (90+ DPD or credit-related ZB within 24 months)
  3. Determine usable years (full outcome window covered)
  4. Compute time-ordered split (5 ref / N monitoring / 3 final eval)
  5. Train baseline LR + GBM with time-ordered CV on reference years
  6. Replay through monitoring years — drift detection at each step
  7. Run three retraining strategies (never, scheduled, drift-triggered)
  8. Save results to reports/
"""

from __future__ import annotations

import json
import time

import pandas as pd

from drift.config import REPORTS_DIR, SEED
from drift.drift import PeriodDriftResult
from drift.features import prepare_freddie_features
from drift.freddie.labels import (
    derive_freddie_labels,
    freddie_label_summary,
    freddie_usable_years,
)
from drift.freddie.loader import (
    available_years,
    load_all_originations,
    load_origination_year,
    load_performance_year,
)
from drift.pipeline import train_baseline
from drift.replay import replay_monitoring, save_replay_results
from drift.retrain import run_strategy, save_strategy_results
from drift.splits import DatasetSplit, compute_split


def _load_and_label_all(force: bool = False) -> pd.DataFrame:
    """Load all years, derive labels, return single labelled DataFrame."""
    years = available_years()
    frames = []
    for year in years:
        print(f"  Loading and labelling {year}...", end=" ", flush=True)
        t0 = time.time()
        orig = load_origination_year(year, force=force)
        perf = load_performance_year(year, force=force)
        labelled = derive_freddie_labels(orig, perf)
        summary = freddie_label_summary(labelled)
        elapsed = time.time() - t0
        print(
            f"{len(labelled):,} loans, "
            f"{summary['default_rate']:.1f}% default, "
            f"{elapsed:.1f}s"
        )
        frames.append(labelled)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(df):,} loans across {len(years)} years")
    return df


def _find_perf_cutoff() -> pd.Period:
    """Find the latest period in performance data (vectorized, checks last year only)."""
    years = available_years()
    perf = load_performance_year(years[-1])
    return perf["period"].max()


def run_freddie_benchmark(
    seed: int = SEED,
    fast: bool = True,
    force_reload: bool = False,
) -> dict:
    """Run the full Freddie Mac drift monitoring pipeline."""
    print("=" * 70)
    print("FREDDIE MAC DRIFT MONITORING PIPELINE")
    print("=" * 70)

    # Step 1: Load and label
    print("\n[1/6] Loading and labelling all years...")
    df = _load_and_label_all(force=force_reload)

    # Step 2: Determine usable years
    print("\n[2/6] Computing usable years and split...")
    perf_cutoff = _find_perf_cutoff()
    all_years = available_years()
    usable = freddie_usable_years(all_years, perf_cutoff)
    print(f"  Performance cutoff: {perf_cutoff}")
    print(f"  Available years: {all_years[0]}-{all_years[-1]} ({len(all_years)} years)")
    print(f"  Usable years: {usable[0]}-{usable[-1]} ({len(usable)} years)")

    split = compute_split("freddie", usable)
    print(f"  {split.describe()}")

    # Filter to usable years only
    df = df[df["origination_year"].isin(usable)].reset_index(drop=True)
    print(f"  Usable loans: {len(df):,}")

    # Step 3: Prepare features
    print("\n[3/6] Preparing features...")
    X, y, feature_names, categorical_cols, numeric_cols = prepare_freddie_features(df)
    print(f"  Features: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical = {len(feature_names)}")
    print(f"  Default rate: {y.mean() * 100:.2f}%")

    # Step 4: Train baseline
    print("\n[4/6] Training baseline models on reference years...")
    t0 = time.time()
    lr_model, gbm_model = train_baseline(
        df, X, y,
        year_col="origination_year",
        reference_years=list(split.reference_years),
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        fast=fast,
    )
    elapsed = time.time() - t0
    print(f"  LR best params: {lr_model.best_params}")
    print(f"  GBM best params: {gbm_model.best_params}")

    if lr_model.cv_results:
        lr_cv_aucs = [r["mean_auc"] for r in lr_model.cv_results]
        print(f"  LR best CV AUC: {max(lr_cv_aucs):.4f}")
    if gbm_model.cv_results:
        gbm_cv_aucs = [r["mean_auc"] for r in gbm_model.cv_results]
        print(f"  GBM best CV AUC: {max(gbm_cv_aucs):.4f}")
    print(f"  Training took {elapsed:.1f}s")

    # Score first monitoring year for reference AUC
    from drift.pipeline import score_period
    first_mon = split.monitoring_years[0]
    first_mask = (df["origination_year"] == first_mon) & y.notna()
    first_X = X.loc[first_mask]
    first_y = y.loc[first_mask]
    _, first_metrics = score_period(gbm_model, first_X, first_y)
    reference_auc = first_metrics.auc
    print(f"  GBM AUC on first monitoring year ({first_mon}): {reference_auc:.4f}")

    # Step 5: Drift replay
    print(f"\n[5/6] Running drift replay through {len(split.monitoring_years)} monitoring years...")
    t0 = time.time()
    drift_results = replay_monitoring(
        df=df,
        model=gbm_model,
        year_col="origination_year",
        reference_years=list(split.reference_years),
        monitoring_years=list(split.monitoring_years),
        reference_auc=reference_auc,
        prepare_fn=prepare_freddie_features,
    )
    elapsed = time.time() - t0
    print(f"  Replay took {elapsed:.1f}s")

    for r in drift_results:
        auc_str = f"{r.auc:.4f}" if r.auc is not None else "N/A"
        print(
            f"  Year {r.period} | PSI={r.score_psi:.4f} | AUC={auc_str} | "
            f"Alert={r.overall_alert}"
        )

    # Save drift replay
    replay_path = save_replay_results(
        drift_results, filename="freddie_drift_replay.json"
    )
    # Patch the saved file to say freddie
    with open(replay_path) as f:
        replay_data = json.load(f)
    replay_data["dataset"] = "freddie"
    with open(replay_path, "w") as f:
        json.dump(replay_data, f, indent=2)
    print(f"  Saved to {replay_path}")

    # Step 6: Retraining strategies
    print(f"\n[6/6] Running retraining strategies...")
    t0 = time.time()
    strategies = []
    for strat_name in ["never", "scheduled", "drift_triggered"]:
        print(f"  Running '{strat_name}'...")
        result = run_strategy(
            name=strat_name,
            df=df,
            baseline_model=gbm_model,
            X=X,
            y=y,
            year_col="origination_year",
            all_usable_years=list(split.usable_years),
            monitoring_years=list(split.monitoring_years),
            final_eval_years=list(split.final_eval_years),
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            best_params=gbm_model.best_params,
            drift_results=drift_results,
            dataset="freddie",
        )
        strategies.append(result)
        print(
            f"    {strat_name}: {result.n_retrains} retrains, "
            f"monitoring AUC={result.mean_monitoring_auc:.4f}, "
            f"final AUC={result.final_mean_auc:.4f}"
        )
    elapsed = time.time() - t0
    print(f"  Strategies took {elapsed:.1f}s")

    strat_path = save_strategy_results(
        strategies, filename="freddie_strategy_comparison.json"
    )
    with open(strat_path) as f:
        strat_data = json.load(f)
    strat_data["dataset"] = "freddie"
    with open(strat_path, "w") as f:
        json.dump(strat_data, f, indent=2)
    print(f"  Saved to {strat_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset: Freddie Mac SFLLD Sample ({len(df):,} loans)")
    print(f"Split: ref={split.reference_years[0]}-{split.reference_years[-1]}, "
          f"mon={split.monitoring_years[0]}-{split.monitoring_years[-1]}, "
          f"eval={split.final_eval_years[0]}-{split.final_eval_years[-1]}")

    alert_counts = {"green": 0, "amber": 0, "red": 0}
    for r in drift_results:
        alert_counts[r.overall_alert] += 1
    print(f"Alerts: {alert_counts}")

    print("\nStrategy comparison:")
    print(f"  {'Strategy':<18} {'Retrains':>8} {'Mon AUC':>10} {'Final AUC':>10}")
    print(f"  {'-' * 48}")
    for s in strategies:
        print(
            f"  {s.name:<18} {s.n_retrains:>8} "
            f"{s.mean_monitoring_auc:>10.4f} {s.final_mean_auc:>10.4f}"
        )

    return {
        "split": {
            "reference": list(split.reference_years),
            "monitoring": list(split.monitoring_years),
            "final_eval": list(split.final_eval_years),
        },
        "baseline_reference_auc": reference_auc,
        "drift_results": [r.summary_dict() for r in drift_results],
        "strategies": {s.name: s.to_dict() for s in strategies},
    }
