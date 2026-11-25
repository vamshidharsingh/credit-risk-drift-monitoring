"""Load and cache UCI Default of Credit Card Clients dataset."""

import pandas as pd

from drift.config import CACHE_DIR, UCI_DATASET_ID, UCI_RAW


def load_uci(force: bool = False) -> pd.DataFrame:
    """Load UCI dataset, trying ucimlrepo first, then local file, with caching."""
    cache_path = CACHE_DIR / "uci_credit.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    df = None

    # Try ucimlrepo package first
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        features = dataset.data.features
        targets = dataset.data.targets
        df = pd.concat([features, targets], axis=1)
    except Exception:
        pass

    # Fallback: local file
    if df is None:
        local_files = list(UCI_RAW.glob("*.csv")) + list(UCI_RAW.glob("*.xls*"))
        if not local_files:
            raise FileNotFoundError(
                f"UCI dataset not available via network and no local file in {UCI_RAW}.\n"
                "Download from: https://archive.ics.uci.edu/dataset/350"
            )
        fpath = local_files[0]
        if fpath.suffix in (".xls", ".xlsx"):
            df = pd.read_excel(fpath, header=1)
        else:
            df = pd.read_csv(fpath)

    # Standardise column names
    df.columns = df.columns.str.strip().str.upper()

    # The target column may be named differently
    target_candidates = [
        "DEFAULT.PAYMENT.NEXT.MONTH",
        "DEFAULT PAYMENT NEXT MONTH",
        "Y",
        "DEFAULT",
    ]
    target_col = None
    for cand in target_candidates:
        cand_upper = cand.upper().replace(".", " ").replace("_", " ")
        for col in df.columns:
            col_clean = col.upper().replace(".", " ").replace("_", " ")
            if col_clean == cand_upper:
                target_col = col
                break
        if target_col:
            break

    if target_col and target_col != "DEFAULT":
        df = df.rename(columns={target_col: "DEFAULT"})

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def validate_uci(df: pd.DataFrame) -> dict:
    """Validate UCI data and return a report."""
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "issues": [],
    }

    if "DEFAULT" not in df.columns:
        report["issues"].append("No target column 'DEFAULT' found")

    if "DEFAULT" in df.columns:
        report["default_rate"] = round(df["DEFAULT"].mean() * 100, 2)
        report["default_counts"] = df["DEFAULT"].value_counts().to_dict()

    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            report[f"{col}_missing_pct"] = round(pct, 1)

    expected_rows = 30_000
    if len(df) != expected_rows:
        report["issues"].append(f"Expected {expected_rows} rows, got {len(df)}")

    return report
