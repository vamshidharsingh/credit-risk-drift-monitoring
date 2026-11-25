"""Feature engineering — prepare raw data for modelling.

Each dataset function returns (X, y, feature_names, categorical_columns, numeric_columns).
Features are known-at-origination only. Encoders/scalers are NOT applied here —
they are fitted inside the pipeline on the training window.
"""

import numpy as np
import pandas as pd

from drift.config import (
    RARE_CATEGORY_THRESHOLD,
    UCI_EXCLUDED_FEATURES,
    UCI_MONITORED_ONLY,
)

# ── SBA ──────────────────────────────────────────────────────────────────────

SBA_NUMERIC = [
    "GrossApproval",
    "SBAGuaranteedApproval",
    "TermInMonths",
    "InitialInterestRate",
    "JobsSupported",
]

SBA_CATEGORICAL = [
    "BusinessType",
    "ProjectState",
    "RevolverStatus",
    "BusinessAge",
    "ProcessingMethod",
    "CollateralInd",
    "SoldSecMrktInd",
    "FixedorVariableInterestInd",
]

SBA_DERIVED = ["sba_guaranteed_pct", "naics_sector"]


def prepare_sba_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list, list]:
    """Prepare SBA features from labelled data.

    Returns (X, y, feature_names, categorical_cols, numeric_cols).
    """
    out = df.copy()

    # Derived: SBA guarantee share
    out["sba_guaranteed_pct"] = (
        out["SBAGuaranteedApproval"] / out["GrossApproval"].replace(0, np.nan)
    ).clip(0, 1)

    # Derived: NAICS sector (first 2 digits)
    out["naics_sector"] = out["NaicsCode"].astype(str).str[:2]
    out.loc[out["naics_sector"].isin(["na", "Na", "NA", ""]), "naics_sector"] = pd.NA

    # Missing indicators for high-missing features
    out["interest_rate_missing"] = out["InitialInterestRate"].isna().astype("Int8")
    out["sold_sec_mrkt_missing"] = out["SoldSecMrktInd"].isna().astype("Int8")

    # Fill categorical NaN with "Unknown" for consistent handling
    for col in SBA_CATEGORICAL + ["naics_sector"]:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    numeric_cols = SBA_NUMERIC + ["sba_guaranteed_pct", "interest_rate_missing", "sold_sec_mrkt_missing"]
    categorical_cols = SBA_CATEGORICAL + ["naics_sector"]
    feature_cols = numeric_cols + categorical_cols

    X = out[feature_cols].copy()

    # Fill remaining NaN in numeric columns with 0 — missing indicators carry the info
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    y = out["default"].copy()

    return X, y, feature_cols, categorical_cols, numeric_cols


# ── Freddie Mac ──────────────────────────────────────────────────────────────

FREDDIE_NUMERIC = [
    "credit_score",
    "original_cltv",
    "original_dti",
    "original_ltv",
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "mi_pct",
    "num_units",
    "num_borrowers",
]

FREDDIE_CATEGORICAL = [
    "first_time_homebuyer",
    "occupancy_status",
    "channel",
    "property_state",
    "property_type",
    "loan_purpose",
]


def prepare_freddie_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list, list]:
    """Prepare Freddie Mac features from labelled data."""
    out = df.copy()

    # Missing-value indicators (these are missing for a reason)
    out["credit_score_missing"] = out["credit_score"].isna().astype("Int8")
    out["dti_missing"] = out["original_dti"].isna().astype("Int8")
    out["cltv_missing"] = out["original_cltv"].isna().astype("Int8")
    out["mi_pct_missing"] = out["mi_pct"].isna().astype("Int8")
    out["num_borrowers_missing"] = out["num_borrowers"].isna().astype("Int8")

    missing_indicators = [
        "credit_score_missing", "dti_missing", "cltv_missing",
        "mi_pct_missing", "num_borrowers_missing",
    ]

    for col in FREDDIE_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    numeric_cols = FREDDIE_NUMERIC + missing_indicators
    categorical_cols = FREDDIE_CATEGORICAL
    feature_cols = numeric_cols + categorical_cols

    X = out[feature_cols].copy()
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    y = out["default"].astype("float64")

    return X, y, feature_cols, categorical_cols, numeric_cols


# ── UCI ──────────────────────────────────────────────────────────────────────

def prepare_uci_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list, list, list]:
    """Prepare UCI features — drop SEX/MARRIAGE, keep AGE monitor-only."""
    out = df.copy()

    # Drop excluded features
    drop_cols = [c for c in UCI_EXCLUDED_FEATURES if c in out.columns]
    out = out.drop(columns=drop_cols)

    # AGE is monitor-only — keep in data but exclude from model features
    monitor_cols = [c for c in UCI_MONITORED_ONLY if c in out.columns]

    y = out["DEFAULT"].copy()
    feature_cols = [c for c in out.columns if c != "DEFAULT" and c not in monitor_cols]

    # UCI features are all numeric (integers)
    numeric_cols = feature_cols
    categorical_cols = []

    X = out[feature_cols].copy()

    return X, y, feature_cols, categorical_cols, numeric_cols


# ── Rare-category grouper ───────────────────────────────────────────────────

def group_rare_categories(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame | None,
    categorical_cols: list[str],
    threshold: float = RARE_CATEGORY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, set]]:
    """Group categories below threshold into 'Other'. Fit on X_train, apply to both.

    Returns (X_train_grouped, X_score_grouped, rare_map).
    rare_map[col] is the set of categories kept (not grouped).
    """
    rare_map = {}
    X_train = X_train.copy()
    if X_score is not None:
        X_score = X_score.copy()

    for col in categorical_cols:
        if col not in X_train.columns:
            continue
        freqs = X_train[col].value_counts(normalize=True)
        keep = set(freqs[freqs >= threshold].index)
        rare_map[col] = keep

        X_train[col] = X_train[col].where(X_train[col].isin(keep), "Other")
        if X_score is not None and col in X_score.columns:
            X_score[col] = X_score[col].where(X_score[col].isin(keep), "Other")

    return X_train, X_score, rare_map
