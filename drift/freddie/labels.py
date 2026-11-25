"""Freddie Mac label derivation — default within 24-month outcome window."""

import pandas as pd

from drift.config import (
    FREDDIE_CREDIT_ZERO_BALANCE_CODES,
    FREDDIE_DEFAULT_DLQ_DAYS,
    FREDDIE_OUTCOME_WINDOW_MONTHS,
)

DLQ_THRESHOLD = FREDDIE_DEFAULT_DLQ_DAYS // 30  # 90 days = 3 months


def derive_freddie_labels(
    orig: pd.DataFrame,
    perf: pd.DataFrame,
) -> pd.DataFrame:
    """Derive default labels for Freddie Mac loans from origination + performance data.

    Default: within 24 months of first_payment_date, the loan reaches 90+ DPD
    OR exits with a credit-related zero balance code (02, 03, 09, 15).

    Returns orig with added columns: default (0/1), default_reason, months_to_event,
    pandemic_forbearance (bool).
    """
    out = orig.copy()

    perf_work = perf.copy()
    perf_work["dlq_months"] = pd.to_numeric(
        perf_work["delinquency_status"], errors="coerce"
    )

    # Filter to outcome window upfront
    window = perf_work[perf_work["loan_age"] <= FREDDIE_OUTCOME_WINDOW_MONTHS].copy()

    # --- Severe delinquency (90+ DPD) ---
    severe = window[window["dlq_months"] >= DLQ_THRESHOLD]
    if len(severe) > 0:
        first_severe = severe.sort_values("loan_age").groupby("loan_id").first()
        first_severe = first_severe[["dlq_months", "loan_age"]].rename(
            columns={"loan_age": "dlq_month_at"}
        )
        first_severe["dlq_reason"] = "DPD_" + first_severe["dlq_months"].astype(int).astype(str) + "m"
    else:
        first_severe = pd.DataFrame(
            columns=["dlq_months", "dlq_month_at", "dlq_reason"]
        )
        first_severe.index.name = "loan_id"

    # --- Credit-related zero balance ---
    zb = window[window["zero_balance_code"].isin(FREDDIE_CREDIT_ZERO_BALANCE_CODES)]
    if len(zb) > 0:
        first_zb = zb.sort_values("loan_age").groupby("loan_id").first()
        first_zb = first_zb[["zero_balance_code", "loan_age"]].rename(
            columns={"loan_age": "zb_month_at"}
        )
        first_zb["zb_reason"] = "ZB_" + first_zb["zero_balance_code"]
    else:
        first_zb = pd.DataFrame(columns=["zero_balance_code", "zb_month_at", "zb_reason"])
        first_zb.index.name = "loan_id"

    # --- Pandemic forbearance ---
    pandemic = pd.Series(False, index=window["loan_id"].unique(), name="pandemic_forbearance")
    pandemic.index.name = "loan_id"
    if "delinquency_due_to_disaster" in window.columns:
        disaster_loans = window.loc[
            window["delinquency_due_to_disaster"].fillna("").str.strip() == "Y", "loan_id"
        ].unique()
        pandemic.loc[pandemic.index.isin(disaster_loans)] = True
    if "borrower_assistance_plan" in window.columns:
        bap = window["borrower_assistance_plan"].fillna("").str.strip()
        bap_loans = window.loc[(bap != "") & (bap != "9"), "loan_id"].unique()
        pandemic.loc[pandemic.index.isin(bap_loans)] = True

    # --- Combine: pick the earliest event ---
    all_loans = pd.DataFrame(index=orig["loan_id"].unique())
    all_loans.index.name = "loan_id"

    merged = all_loans.join(first_severe[["dlq_month_at", "dlq_reason"]], how="left")
    merged = merged.join(first_zb[["zb_month_at", "zb_reason"]], how="left")
    merged = merged.join(pandemic, how="left")

    # Pick the earliest event
    dlq_at = merged["dlq_month_at"].fillna(9999)
    zb_at = merged["zb_month_at"].fillna(9999)
    has_dlq = dlq_at < 9999
    has_zb = zb_at < 9999

    merged["default"] = (has_dlq | has_zb).astype("Int8")
    merged["months_to_event"] = pd.array([pd.NA] * len(merged), dtype="Int32")
    merged["default_reason"] = ""

    # DLQ is earlier or only event
    dlq_first = has_dlq & (~has_zb | (dlq_at <= zb_at))
    merged.loc[dlq_first, "default_reason"] = merged.loc[dlq_first, "dlq_reason"]
    merged.loc[dlq_first, "months_to_event"] = merged.loc[dlq_first, "dlq_month_at"].astype("Int32")

    # ZB is earlier
    zb_first = has_zb & (~has_dlq | (zb_at < dlq_at))
    merged.loc[zb_first, "default_reason"] = merged.loc[zb_first, "zb_reason"]
    merged.loc[zb_first, "months_to_event"] = merged.loc[zb_first, "zb_month_at"].astype("Int32")

    merged["pandemic_forbearance"] = merged["pandemic_forbearance"].fillna(False)

    labels = merged[["default", "default_reason", "months_to_event", "pandemic_forbearance"]]
    out = out.merge(labels, on="loan_id", how="left")

    out["default"] = out["default"].fillna(0).astype("Int8")
    out["default_reason"] = out["default_reason"].fillna("")
    out["pandemic_forbearance"] = out["pandemic_forbearance"].fillna(False)

    return out


def freddie_usable_years(
    available_years: list[int],
    perf_cutoff: pd.Period,
) -> list[int]:
    """Return origination years whose full 24-month outcome window is covered.

    An origination year Y is usable if the performance data extends at least
    24 months past the latest possible first_payment_date in that year.
    The latest first_payment in year Y is typically Dec of Y (period Y-12),
    so we need performance data through Y+2 December.
    """
    usable = []
    for year in available_years:
        latest_first_payment = pd.Period(f"{year}-12", freq="M")
        window_end = latest_first_payment + FREDDIE_OUTCOME_WINDOW_MONTHS
        if window_end <= perf_cutoff:
            usable.append(year)
    return usable


def freddie_label_summary(labelled: pd.DataFrame) -> dict:
    """Summarise Freddie Mac label derivation results."""
    total = len(labelled)
    defaults = int((labelled["default"] == 1).sum())
    non_defaults = int((labelled["default"] == 0).sum())
    return {
        "total": total,
        "defaults": defaults,
        "non_defaults": non_defaults,
        "default_rate": round(defaults / total * 100, 2) if total > 0 else 0,
        "pandemic_forbearance_count": int(labelled["pandemic_forbearance"].sum()),
        "default_reasons": labelled.loc[
            labelled["default"] == 1, "default_reason"
        ].value_counts().to_dict() if defaults > 0 else {},
        "no_perf_records": int(labelled["default_reason"].eq("").sum() & labelled["default"].eq(0).sum()),
    }
