"""Load and cache SBA 7(a) FOIA loan data."""

from pathlib import Path

import pandas as pd

from drift.config import (
    CACHE_DIR,
    SBA_EXCLUDE_STATUSES,
    SBA_FEATURE_COLUMNS,
    SBA_FILE_URLS,
    SBA_LABEL_COLUMNS,
    SBA_RAW,
)

SBA_DATE_COLUMNS = ["ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"]

SBA_COLUMN_DTYPES = {
    "AsOfDate": str,
    "Program": str,
    "BorrName": str,
    "BorrState": str,
    "BorrZip": str,
    "BankName": str,
    "BankState": str,
    "GrossApproval": "Float64",
    "SBAGuaranteedApproval": "Float64",
    "ApprovalDate": str,
    "ApprovalFY": "Int32",
    "FirstDisbursementDate": str,
    "ProcessingMethod": str,
    "InitialInterestRate": "Float64",
    "FixedorVariableInterestInd": str,
    "TermInMonths": "Int32",
    "NaicsCode": str,
    "NaicsDescription": str,
    "FranchiseCode": str,
    "FranchiseName": str,
    "ProjectCounty": str,
    "ProjectState": str,
    "SBADistrictOffice": str,
    "CongressionalDistrict": str,
    "BusinessType": str,
    "BusinessAge": str,
    "LoanStatus": str,
    "PaidInFullDate": str,
    "ChargeOffDate": str,
    "GrossChargeOffAmount": "Float64",
    "RevolverStatus": str,
    "JobsSupported": "Int32",
    "CollateralInd": str,
    "SoldSecMrktInd": str,
}

KEEP_COLUMNS = list(
    set(SBA_FEATURE_COLUMNS) | set(SBA_LABEL_COLUMNS) | {"BorrState", "NaicsDescription"}
)


def _normalise_status(s: pd.Series) -> pd.Series:
    """Normalise LoanStatus: strip, upper, collapse whitespace."""
    cleaned = s.astype(str).str.strip().str.upper()
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    # Also strip all internal spaces for comparison
    no_spaces = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.where(no_spaces != "PIF", "PIF")
    cleaned = cleaned.where(no_spaces != "CHGOFF", "CHGOFF")
    cleaned = cleaned.where(no_spaces != "CANCLD", "CANCLD")
    cleaned = cleaned.where(no_spaces != "EXEMPT", "EXEMPT")
    cleaned = cleaned.where(no_spaces != "COMMIT", "COMMIT")

    mapping = {
        "PAID IN FULL": "PIF",
        "PAIDINFULL": "PIF",
        "CHARGED OFF": "CHGOFF",
        "CHARGE OFF": "CHGOFF",
        "CHARGED-OFF": "CHGOFF",
        "CHARGEDOFF": "CHGOFF",
        "CANCELLED": "CANCLD",
        "CANCELED": "CANCLD",
        "COMMITTED": "COMMIT",
        "UNDISBURSED": "COMMIT",
        "NOT FUNDED": "COMMIT",
        "NOTFUNDED": "COMMIT",
        "EXEMPT": "EXEMPT",
    }
    return cleaned.replace(mapping)


def _parse_date_flexible(s: pd.Series) -> pd.Series:
    """Parse dates that may come in several formats."""
    return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")


def _find_csv_files() -> list[Path]:
    """Find SBA 7(a) CSV files in the raw directory."""
    patterns = ["FOIA*7a*.csv", "FOIA*7(a)*.csv", "foia*7a*.csv", "*7a*.csv"]
    files = []
    for pat in patterns:
        files.extend(SBA_RAW.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No SBA 7(a) CSV files found in {SBA_RAW}.\n"
            f"Download from: https://data.sba.gov/dataset/7a-504-foia\n"
            f"Expected files matching FOIA*7a*.csv\n"
            f"Download URLs:\n" + "\n".join(f"  {v}" for v in SBA_FILE_URLS.values())
        )
    return files


def load_sba_raw(force: bool = False) -> pd.DataFrame:
    """Load all SBA 7(a) CSV files, filter to 7(a) only, and cache."""
    cache_path = CACHE_DIR / "sba_raw.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    csv_files = _find_csv_files()
    frames = []
    for csv_path in csv_files:
        chunks = pd.read_csv(
            csv_path,
            dtype=str,
            na_values=["", " ", "NA", "N/A"],
            keep_default_na=True,
            chunksize=100_000,
            encoding="utf-8",
            encoding_errors="replace",
        )
        for chunk in chunks:
            chunk.columns = chunk.columns.str.strip()
            if "Program" in chunk.columns:
                chunk = chunk[chunk["Program"].str.strip().str.upper() == "7A"]
            frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)

    # Type conversion
    for col in ["GrossApproval", "SBAGuaranteedApproval", "GrossChargeOffAmount",
                 "InitialInterestRate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("$", "").str.strip(),
                errors="coerce",
            )

    for col in ["TermInMonths", "JobsSupported", "ApprovalFY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    df["LoanStatus"] = _normalise_status(df["LoanStatus"])

    for col in SBA_DATE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_date_flexible(df[col])

    if "RevolverStatus" in df.columns:
        df["RevolverStatus"] = df["RevolverStatus"].astype(str).str.strip()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def validate_sba(df: pd.DataFrame) -> dict:
    """Validate SBA data and return a report."""
    report = {"rows": len(df), "issues": []}

    status_counts = df["LoanStatus"].value_counts()
    report["status_counts"] = status_counts.to_dict()

    if "ApprovalFY" in df.columns:
        report["fy_min"] = int(df["ApprovalFY"].min())
        report["fy_max"] = int(df["ApprovalFY"].max())
        report["loans_per_fy"] = df["ApprovalFY"].value_counts().sort_index().to_dict()

    for col in SBA_FEATURE_COLUMNS + SBA_LABEL_COLUMNS:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > 0:
                report[f"{col}_missing_pct"] = round(pct, 1)

    excluded = df["LoanStatus"].isin(SBA_EXCLUDE_STATUSES).sum()
    report["excluded_count"] = int(excluded)
    report["excluded_statuses"] = {
        s: int((df["LoanStatus"] == s).sum()) for s in SBA_EXCLUDE_STATUSES
    }

    return report
