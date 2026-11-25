"""Model pipelines — LR and GBM with time-ordered CV and baseline training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from drift.config import (
    SEED,
    TUNING_GRID_GBM,
    TUNING_GRID_GBM_FAST,
    TUNING_GRID_LR,
    TUNING_GRID_LR_FAST,
)
from drift.features import group_rare_categories
from drift.metrics import ModelMetrics, compute_metrics


@dataclass
class TrainedModel:
    """A trained model with its metadata."""

    name: str
    pipeline: Pipeline
    rare_map: dict[str, set]
    categorical_cols: list[str]
    numeric_cols: list[str]
    training_years: list[int]
    cv_results: list[dict] = field(default_factory=list)
    best_params: dict = field(default_factory=dict)
    calibrated: bool = False


def _build_lr_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a Logistic Regression pipeline with imputation, scaling and one-hot encoding."""
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=1),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=SEED,
        )),
    ])


def _build_gbm_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build a HistGradientBoosting pipeline with ordinal encoding for categoricals."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_cols,
        ))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(
            random_state=SEED,
            categorical_features="from_dtype" if not categorical_cols else None,
        )),
    ])


def _time_ordered_cv_folds(
    df: pd.DataFrame,
    year_col: str,
    years: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window time-ordered CV folds.

    Fold k: train on years[:k+1], validate on years[k+1].
    """
    sorted_years = sorted(years)
    folds = []
    for i in range(1, len(sorted_years)):
        train_years = sorted_years[:i]
        val_year = sorted_years[i]
        train_idx = df[df[year_col].isin(train_years)].index.to_numpy()
        val_idx = df[df[year_col] == val_year].index.to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds


def train_baseline(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    year_col: str,
    reference_years: list[int],
    categorical_cols: list[str],
    numeric_cols: list[str],
    fast: bool = False,
) -> tuple[TrainedModel, TrainedModel]:
    """Train LR and GBM baselines with time-ordered CV on reference years.

    Returns (lr_model, gbm_model).
    """
    # Filter to reference years with valid labels — reset index so iloc works
    mask = df[year_col].isin(reference_years) & y.notna()
    ref_df = df.loc[mask].reset_index(drop=True)
    ref_X = X.loc[mask].reset_index(drop=True)
    ref_y = y.loc[mask].astype(int).reset_index(drop=True)

    # Group rare categories on reference data
    ref_X, _, rare_map = group_rare_categories(ref_X, None, categorical_cols)

    # Time-ordered CV folds
    folds = _time_ordered_cv_folds(ref_df, year_col, sorted(reference_years))

    # Tune and train LR
    lr_grid = TUNING_GRID_LR_FAST if fast else TUNING_GRID_LR
    lr_model = _tune_and_train(
        "logistic_regression", ref_X, ref_y, folds, _build_lr_pipeline,
        lr_grid, "classifier__C", numeric_cols, categorical_cols, rare_map,
        sorted(reference_years),
    )

    # Tune and train GBM
    gbm_grid = TUNING_GRID_GBM_FAST if fast else TUNING_GRID_GBM
    gbm_model = _tune_and_train_gbm(
        "gradient_boosting", ref_X, ref_y, folds, _build_gbm_pipeline,
        gbm_grid, numeric_cols, categorical_cols, rare_map,
        sorted(reference_years),
    )

    return lr_model, gbm_model


def _tune_and_train(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    build_fn,
    param_grid: dict,
    param_prefix: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    rare_map: dict,
    training_years: list[int],
) -> TrainedModel:
    """Tune LR by grid search with time-ordered CV, then refit on all data."""
    best_auc = -1.0
    best_params = {}
    cv_results = []

    for params in ParameterGrid(param_grid):
        fold_aucs = []
        for train_idx, val_idx in folds:
            pipe = build_fn(numeric_cols, categorical_cols)
            for k, v in params.items():
                pipe.set_params(**{f"{param_prefix.rsplit('__', 1)[0]}__{k}": v})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

            y_prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            auc = compute_metrics(y.iloc[val_idx].values, y_prob).auc
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        cv_results.append({"params": params, "mean_auc": mean_auc, "fold_aucs": fold_aucs})
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # Refit on all reference data with best params
    final_pipe = build_fn(numeric_cols, categorical_cols)
    for k, v in best_params.items():
        final_pipe.set_params(**{f"{param_prefix.rsplit('__', 1)[0]}__{k}": v})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(X, y)

    return TrainedModel(
        name=name,
        pipeline=final_pipe,
        rare_map=rare_map,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        training_years=training_years,
        cv_results=cv_results,
        best_params=best_params,
    )


def _tune_and_train_gbm(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    build_fn,
    param_grid: dict,
    numeric_cols: list[str],
    categorical_cols: list[str],
    rare_map: dict,
    training_years: list[int],
) -> TrainedModel:
    """Tune GBM by grid search over its own multi-param grid."""
    best_auc = -1.0
    best_params = {}
    cv_results = []

    for params in ParameterGrid(param_grid):
        fold_aucs = []
        for train_idx, val_idx in folds:
            pipe = build_fn(numeric_cols, categorical_cols)
            pipe_params = {f"classifier__{k}": v for k, v in params.items()}
            pipe.set_params(**pipe_params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

            y_prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            auc = compute_metrics(y.iloc[val_idx].values, y_prob).auc
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        cv_results.append({"params": params, "mean_auc": mean_auc, "fold_aucs": fold_aucs})
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # Refit on all reference data
    final_pipe = build_fn(numeric_cols, categorical_cols)
    pipe_params = {f"classifier__{k}": v for k, v in best_params.items()}
    final_pipe.set_params(**pipe_params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(X, y)

    return TrainedModel(
        name=name,
        pipeline=final_pipe,
        rare_map=rare_map,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        training_years=training_years,
        cv_results=cv_results,
        best_params=best_params,
    )


def score_period(
    model: TrainedModel,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, ModelMetrics]:
    """Score a dataset with a trained model, returning probabilities and metrics."""
    X_scored = X.copy()
    for col in model.categorical_cols:
        if col in X_scored.columns:
            keep = model.rare_map.get(col, set())
            if keep:
                X_scored[col] = X_scored[col].where(X_scored[col].isin(keep), "Other")

    y_prob = model.pipeline.predict_proba(X_scored)[:, 1]
    metrics = compute_metrics(y.values, y_prob)
    return y_prob, metrics


def check_calibration(
    model: TrainedModel,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> tuple[TrainedModel, bool]:
    """Check calibration on the latest training year. If poor, recalibrate.

    Returns (possibly recalibrated model, was_recalibrated).
    """
    _, metrics = score_period(model, X_cal, y_cal)
    pred_rate = metrics.overall_predicted
    actual_rate = metrics.overall_actual

    # Calibration is "poor" if predicted default rate differs from actual by >2pp
    if abs(pred_rate - actual_rate) > 0.02:
        X_cal_grouped = X_cal.copy()
        for col in model.categorical_cols:
            if col in X_cal_grouped.columns:
                keep = model.rare_map.get(col, set())
                if keep:
                    X_cal_grouped[col] = X_cal_grouped[col].where(
                        X_cal_grouped[col].isin(keep), "Other"
                    )

        calibrated = CalibratedClassifierCV(
            model.pipeline, method="isotonic", cv="prefit",
        )
        calibrated.fit(X_cal_grouped, y_cal)

        new_model = TrainedModel(
            name=model.name,
            pipeline=model.pipeline,
            rare_map=model.rare_map,
            categorical_cols=model.categorical_cols,
            numeric_cols=model.numeric_cols,
            training_years=model.training_years,
            cv_results=model.cv_results,
            best_params=model.best_params,
            calibrated=True,
        )
        new_model.pipeline = calibrated
        return new_model, True

    return model, False
