"""SBA 7(a) label derivation — default = charged off within 5 years of approval."""

import pandas as pd

from drift.config import (
    SBA_DEFAULT_STATUS,
    SBA_EXCLUDE_STATUSES,
    SBA_EXEMPT_STATUS,
    SBA_OUTCOME_WINDOW_YEARS,
    SBA_PAID_STATUS,
)


def derive_sba_labels(df: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Derive default labels for SBA loans.

    Returns a copy with columns: default (0/1/NaN), exclusion_reason, label_status.
    """
    out = df.copy()
    out["default"] = pd.array([pd.NA] * len(out), dtype="Int8")
    out["exclusion_reason"] = ""
    out["label_status"] = ""

    # Exclude cancelled and undisbursed
    excluded = out["LoanStatus"].isin(SBA_EXCLUDE_STATUSES)
    out.loc[excluded, "exclusion_reason"] = out.loc[excluded, "LoanStatus"]
    out.loc[excluded, "label_status"] = "excluded"

    # Charged off = default
    is_chgoff = out["LoanStatus"] == SBA_DEFAULT_STATUS
    out.loc[is_chgoff, "label_status"] = "resolved"

    if "ChargeOffDate" in out.columns and "ApprovalDate" in out.columns:
        has_dates = is_chgoff & out["ChargeOffDate"].notna() & out["ApprovalDate"].notna()
        years_to_chgoff = (
            (out["ChargeOffDate"] - out["ApprovalDate"]).dt.days / 365.25
        )
        within_window = has_dates & (years_to_chgoff <= SBA_OUTCOME_WINDOW_YEARS)
        out.loc[within_window, "default"] = 1
        # Charged off but after window — treat as non-default for the window period
        after_window = has_dates & (years_to_chgoff > SBA_OUTCOME_WINDOW_YEARS)
        out.loc[after_window, "default"] = 0
        out.loc[after_window, "label_status"] = "resolved_late_chgoff"
        # Charged off but missing dates — still mark as default (conservative)
        no_dates = is_chgoff & ~has_dates
        out.loc[no_dates, "default"] = 1
        out.loc[no_dates, "label_status"] = "resolved_no_dates"
    else:
        out.loc[is_chgoff, "default"] = 1

    # Paid in full = non-default
    is_pif = out["LoanStatus"] == SBA_PAID_STATUS
    out.loc[is_pif, "default"] = 0
    out.loc[is_pif, "label_status"] = "resolved"

    # EXEMPT (active/undisclosed) — check if 5 years have passed
    is_exempt = out["LoanStatus"] == SBA_EXEMPT_STATUS
    if as_of_date is not None and "ApprovalDate" in out.columns:
        years_since_approval = (as_of_date - out["ApprovalDate"]).dt.days / 365.25
        matured = is_exempt & (years_since_approval >= SBA_OUTCOME_WINDOW_YEARS)
        out.loc[matured, "default"] = 0
        out.loc[matured, "label_status"] = "resolved_survived"
        censored = is_exempt & (years_since_approval < SBA_OUTCOME_WINDOW_YEARS)
        out.loc[censored, "label_status"] = "censored"
    else:
        out.loc[is_exempt, "label_status"] = "censored"

    return out


def sba_usable_years(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> list[int]:
    """Return fiscal years whose full outcome window has elapsed by as_of_date.

    A fiscal year FY is usable if 5 full years have passed since the end of that FY.
    Federal fiscal years end on Sep 30, so FY2010 ends 2010-09-30.
    """
    if "ApprovalFY" not in df.columns:
        return []
    all_fys = sorted(df["ApprovalFY"].dropna().unique())
    usable = []
    for fy in all_fys:
        fy_end = pd.Timestamp(f"{int(fy)}-09-30")
        window_end = fy_end + pd.DateOffset(years=SBA_OUTCOME_WINDOW_YEARS)
        if window_end <= as_of_date:
            usable.append(int(fy))
    return usable


def sba_label_summary(labelled: pd.DataFrame) -> dict:
    """Summarise label derivation results."""
    total = len(labelled)
    resolved = labelled["label_status"].str.startswith("resolved")
    return {
        "total": total,
        "excluded": int((labelled["label_status"] == "excluded").sum()),
        "excluded_by_status": labelled.loc[
            labelled["label_status"] == "excluded", "exclusion_reason"
        ].value_counts().to_dict(),
        "resolved": int(resolved.sum()),
        "defaults": int((labelled["default"] == 1).sum()),
        "non_defaults": int((labelled["default"] == 0).sum()),
        "censored": int((labelled["label_status"] == "censored").sum()),
        "default_rate_resolved": round(
            labelled.loc[resolved, "default"].mean() * 100, 2
        ) if resolved.any() else None,
    }
