"""Load and cache Freddie Mac Single-Family Loan-Level sample data."""

import zipfile
from pathlib import Path

import pandas as pd

from drift.config import (
    CACHE_DIR,
    FREDDIE_MISSING_BORROWERS,
    FREDDIE_MISSING_CREDIT_SCORE,
    FREDDIE_MISSING_DTI,
    FREDDIE_ORIG_COLUMNS,
    FREDDIE_ORIG_KEEP,
    FREDDIE_PERF_COLUMNS,
    FREDDIE_PERF_KEEP,
    FREDDIE_RAW,
)

ORIG_DTYPES = {
    "credit_score": "Int32",
    "first_payment_date": str,
    "first_time_homebuyer": str,
    "maturity_date": str,
    "msa": str,
    "mi_pct": "Int16",
    "num_units": "Int8",
    "occupancy_status": str,
    "original_cltv": "Int16",
    "original_dti": "Int16",
    "original_upb": "Int64",
    "original_ltv": "Int16",
    "original_interest_rate": "Float64",
    "channel": str,
    "prepayment_penalty": str,
    "amortization_type": str,
    "property_state": str,
    "property_type": str,
    "postal_code": str,
    "loan_id": str,
    "loan_purpose": str,
    "original_loan_term": "Int16",
    "num_borrowers": "Int8",
    "seller_name": str,
    "super_conforming_flag": str,
    "pre_harp_loan_id": str,
    "special_eligibility": str,
    "harp_indicator": str,
    "property_valuation_method": str,
    "io_indicator": str,
    "vantage_score": "Int32",
}

PERF_DTYPES = {
    "loan_id": str,
    "period": str,
    "current_upb": "Float64",
    "delinquency_status": str,
    "loan_age": "Int16",
    "remaining_months": "Int16",
    "defect_settlement_date": str,
    "modification_flag": str,
    "zero_balance_code": str,
    "zero_balance_date": str,
    "current_interest_rate": "Float64",
    "current_non_interest_bearing_upb": "Float64",
    "ddlpi": str,
    "mi_recoveries": "Float64",
    "net_sales_proceeds": str,
    "non_mi_recoveries": "Float64",
    "total_expenses": "Float64",
    "legal_costs": "Float64",
    "maintenance_costs": "Float64",
    "taxes_and_insurance": "Float64",
    "misc_expenses": "Float64",
    "actual_loss": "Float64",
    "cumulative_mod_costs": "Float64",
    "interest_rate_step": str,
    "payment_deferral_flag": str,
    "eltv": "Float64",
    "zero_balance_removal_upb": "Float64",
    "delinquent_accrued_interest": "Float64",
    "delinquency_due_to_disaster": str,
    "borrower_assistance_plan": str,
    "current_period_mod_costs": "Float64",
    "current_interest_bearing_upb": "Float64",
    "mi_cancellation_indicator": str,
    "servicer_name": str,
    "bankruptcy_cramdown_costs": "Float64",
}


def _find_zip_files() -> list[Path]:
    """Find all sample_YYYY.zip files sorted by year."""
    zips = sorted(FREDDIE_RAW.glob("sample_*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No sample_YYYY.zip files found in {FREDDIE_RAW}.\n"
            "Download them from https://freddiemac.com/research/datasets/sf-loanlevel-dataset\n"
            "(requires free registration)."
        )
    return zips


def _year_from_zip(zip_path: Path) -> int:
    """Extract the vintage year from a zip filename like sample_2005.zip."""
    stem = zip_path.stem  # sample_2005
    return int(stem.split("_")[1])


def _find_file_in_zip(zf: zipfile.ZipFile, prefix: str) -> str | None:
    """Find a file inside a zip whose name starts with the given prefix."""
    for name in zf.namelist():
        basename = name.split("/")[-1]
        if basename.lower().startswith(prefix.lower()) and basename.endswith(".txt"):
            return name
    return None


def _clean_missing_orig(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Freddie Mac missing-value codes with proper NaN."""
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].where(
            df["credit_score"] != FREDDIE_MISSING_CREDIT_SCORE
        )
    if "original_dti" in df.columns:
        df["original_dti"] = df["original_dti"].where(
            df["original_dti"] != FREDDIE_MISSING_DTI
        )
    if "num_borrowers" in df.columns:
        df["num_borrowers"] = df["num_borrowers"].where(
            df["num_borrowers"] != FREDDIE_MISSING_BORROWERS
        )
    return df


def _parse_freddie_date(s: pd.Series) -> pd.Series:
    """Parse YYYYMM date strings to Period (monthly)."""
    return pd.to_datetime(s, format="%Y%m", errors="coerce").dt.to_period("M")


def load_origination_year(year: int, force: bool = False) -> pd.DataFrame:
    """Load origination data for one vintage year, with caching."""
    cache_path = CACHE_DIR / f"freddie_orig_{year}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    zip_path = FREDDIE_RAW / f"sample_{year}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        orig_file = _find_file_in_zip(zf, f"sample_orig_{year}")
        if orig_file is None:
            raise FileNotFoundError(
                f"No origination file (sample_orig_{year}*.txt) in {zip_path}. "
                f"Files found: {zf.namelist()}"
            )

        with zf.open(orig_file) as f:
            df = pd.read_csv(
                f,
                sep="|",
                header=None,
                names=FREDDIE_ORIG_COLUMNS,
                dtype=ORIG_DTYPES,
                na_values=["", " "],
                keep_default_na=False,
            )

    df = _clean_missing_orig(df)
    df["first_payment_date"] = _parse_freddie_date(df["first_payment_date"])
    df["origination_year"] = year

    keep = [c for c in FREDDIE_ORIG_KEEP if c in df.columns] + ["origination_year"]
    df = df[keep]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def load_performance_year(year: int, force: bool = False) -> pd.DataFrame:
    """Load performance data for one vintage year, keeping only label-relevant columns."""
    cache_path = CACHE_DIR / f"freddie_perf_{year}.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    zip_path = FREDDIE_RAW / f"sample_{year}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}")

    perf_keep_set = set(FREDDIE_PERF_KEEP)
    keep_indices = [i for i, c in enumerate(FREDDIE_PERF_COLUMNS) if c in perf_keep_set]
    chunks = []
    with zipfile.ZipFile(zip_path) as zf:
        perf_file = _find_file_in_zip(zf, f"sample_svcg_{year}")
        if perf_file is None:
            perf_file = _find_file_in_zip(zf, f"sample_perf_{year}")
        if perf_file is None:
            raise FileNotFoundError(
                f"No performance file (sample_svcg_{year} or sample_perf_{year}) in {zip_path}. "
                f"Files found: {zf.namelist()}"
            )

        with zf.open(perf_file) as f:
            reader = pd.read_csv(
                f,
                sep="|",
                header=None,
                names=FREDDIE_PERF_COLUMNS,
                dtype=PERF_DTYPES,
                usecols=keep_indices,
                na_values=["", " "],
                keep_default_na=False,
                chunksize=500_000,
            )
            for chunk in reader:
                chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df["period"] = _parse_freddie_date(df["period"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def available_years() -> list[int]:
    """Return sorted list of vintage years with zip files present."""
    return sorted(_year_from_zip(z) for z in _find_zip_files())


def load_all_originations(force: bool = False) -> pd.DataFrame:
    """Load all available origination years."""
    years = available_years()
    frames = [load_origination_year(y, force=force) for y in years]
    return pd.concat(frames, ignore_index=True)


def performance_cutoff_date() -> pd.Period:
    """Find the latest period in the performance data across all years."""
    years = available_years()
    latest = pd.Period("1900-01", freq="M")
    for year in years:
        perf = load_performance_year(year)
        year_max = perf["period"].max()
        if year_max > latest:
            latest = year_max
    return latest


def validate_origination(df: pd.DataFrame, year: int) -> dict:
    """Validate origination data, returning a report dict."""
    report = {"year": year, "rows": len(df), "issues": []}

    expected = set(FREDDIE_ORIG_KEEP) | {"origination_year"}
    actual = set(df.columns)
    missing_cols = expected - actual
    if missing_cols:
        report["issues"].append(f"Missing columns: {missing_cols}")

    dup_ids = df["loan_id"].duplicated().sum()
    if dup_ids > 0:
        report["issues"].append(f"{dup_ids} duplicate loan IDs")

    for col in ["credit_score", "original_dti", "original_ltv", "original_cltv"]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            report[f"{col}_missing_pct"] = round(pct, 1)

    return report


def validate_performance(df: pd.DataFrame, year: int) -> dict:
    """Validate performance data, returning a report dict."""
    report = {"year": year, "rows": len(df), "issues": []}

    expected = set(FREDDIE_PERF_KEEP)
    actual = set(df.columns)
    missing_cols = expected - actual
    if missing_cols:
        report["issues"].append(f"Missing columns: {missing_cols}")

    unique_loans = df["loan_id"].nunique()
    report["unique_loans"] = unique_loans

    if "period" in df.columns:
        report["period_min"] = str(df["period"].min())
        report["period_max"] = str(df["period"].max())

    return report
