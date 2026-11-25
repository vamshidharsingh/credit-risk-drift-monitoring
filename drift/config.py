"""Central configuration — every setting, threshold, seed and path in one place."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = DOCS_DIR / "images"

FREDDIE_RAW = DATA_DIR / "freddie"
SBA_RAW = DATA_DIR / "sba"
UCI_RAW = DATA_DIR / "uci"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 1_000
N_BOOTSTRAP_FAST = 200

# ---------------------------------------------------------------------------
# Freddie Mac
# ---------------------------------------------------------------------------
FREDDIE_OUTCOME_WINDOW_MONTHS = 24
FREDDIE_DEFAULT_DLQ_DAYS = 90  # 90+ days delinquent
FREDDIE_CREDIT_ZERO_BALANCE_CODES = {"02", "03", "09", "15"}
FREDDIE_VOLUNTARY_ZERO_BALANCE_CODES = {"01"}
FREDDIE_REPURCHASE_ZERO_BALANCE_CODES = {"06", "96"}

FREDDIE_MISSING_CREDIT_SCORE = 9999
FREDDIE_MISSING_DTI = 999
FREDDIE_MISSING_BORROWERS = 99

FREDDIE_ORIG_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_homebuyer",
    "maturity_date", "msa", "mi_pct", "num_units", "occupancy_status",
    "original_cltv", "original_dti", "original_upb", "original_ltv",
    "original_interest_rate", "channel", "prepayment_penalty",
    "amortization_type", "property_state", "property_type", "postal_code",
    "loan_id", "loan_purpose", "original_loan_term", "num_borrowers",
    "seller_name", "super_conforming_flag", "pre_harp_loan_id",
    "special_eligibility", "harp_indicator", "property_valuation_method",
    "io_indicator", "vantage_score",
]

FREDDIE_PERF_COLUMNS = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age",
    "remaining_months", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_date", "current_interest_rate",
    "current_non_interest_bearing_upb", "ddlpi", "mi_recoveries",
    "net_sales_proceeds", "non_mi_recoveries", "total_expenses",
    "legal_costs", "maintenance_costs", "taxes_and_insurance",
    "misc_expenses", "actual_loss", "cumulative_mod_costs",
    "interest_rate_step", "payment_deferral_flag", "eltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_due_to_disaster", "borrower_assistance_plan",
    "current_period_mod_costs", "current_interest_bearing_upb",
    "mi_cancellation_indicator", "servicer_name", "bankruptcy_cramdown_costs",
]

# Columns we keep from origination (for features and labels)
FREDDIE_ORIG_KEEP = [
    "loan_id", "credit_score", "first_payment_date", "first_time_homebuyer",
    "mi_pct", "num_units", "occupancy_status", "original_cltv", "original_dti",
    "original_upb", "original_ltv", "original_interest_rate", "channel",
    "property_state", "property_type", "loan_purpose", "original_loan_term",
    "num_borrowers",
]

# Columns we keep from performance (for label derivation)
FREDDIE_PERF_KEEP = [
    "loan_id", "period", "delinquency_status", "zero_balance_code",
    "zero_balance_date", "loan_age", "delinquency_due_to_disaster",
    "borrower_assistance_plan",
]

# ---------------------------------------------------------------------------
# SBA 7(a)
# ---------------------------------------------------------------------------
SBA_OUTCOME_WINDOW_YEARS = 5
SBA_DEFAULT_STATUS = "CHGOFF"
SBA_PAID_STATUS = "PIF"
SBA_CANCELLED_STATUS = "CANCLD"
SBA_EXEMPT_STATUS = "EXEMPT"
SBA_COMMIT_STATUS = "COMMIT"
SBA_EXCLUDE_STATUSES = {SBA_CANCELLED_STATUS, SBA_COMMIT_STATUS}

SBA_FILE_URLS = {
    "FY1991_FY1999": "https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY1991_FY1999_asof_260930.csv",
    "FY2000_FY2009": "https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2000_FY2009_asof_260930.csv",
    "FY2010_FY2019": "https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2010_FY2019_asof_260930.csv",
    "FY2020_Present": "https://data.sba.gov/sites/default/files/uploaded_resources/FOIA_7a_FY2020_Present_asof_260930.csv",
}

SBA_FEATURE_COLUMNS = [
    "GrossApproval", "SBAGuaranteedApproval", "TermInMonths",
    "NaicsCode", "BusinessType", "ProjectState", "JobsSupported",
    "RevolverStatus", "InitialInterestRate", "BusinessAge",
    "ProcessingMethod", "CollateralInd", "SoldSecMrktInd",
    "FixedorVariableInterestInd",
]

SBA_LABEL_COLUMNS = [
    "LoanStatus", "ApprovalDate", "ApprovalFY", "ChargeOffDate",
    "PaidInFullDate", "FirstDisbursementDate",
]

# ---------------------------------------------------------------------------
# UCI
# ---------------------------------------------------------------------------
UCI_DATASET_ID = 350
UCI_EXCLUDED_FEATURES = ["SEX", "MARRIAGE"]
UCI_MONITORED_ONLY = ["AGE"]  # kept for monitoring, excluded from model input

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
RARE_CATEGORY_THRESHOLD = 0.005
COST_RATIOS = {"primary": 5, "sensitivity_low": 3, "sensitivity_high": 10}
CALIBRATION_BINS = 10

TUNING_GRID_LR = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

TUNING_GRID_GBM = {
    "max_iter": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "min_samples_leaf": [20, 50],
    "max_leaf_nodes": [31, 63],
}

TUNING_GRID_GBM_FAST = {
    "max_iter": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.1],
    "min_samples_leaf": [20],
    "max_leaf_nodes": [31],
}

TUNING_GRID_LR_FAST = {
    "C": [0.01, 1.0],
}

# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------
PSI_BINS = 10
PSI_FLOOR = 0.0001
TOP_FEATURES_COUNT = 10

ALERT_GREEN_SCORE_PSI = 0.10
ALERT_AMBER_SCORE_PSI = 0.25
ALERT_GREEN_FEATURE_PSI = 0.10
ALERT_AMBER_FEATURE_PSI = 0.25
ALERT_RED_FEATURE_COUNT = 3
AUC_DROP_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
PROMOTION_AUC_THRESHOLD = 0.005
RETRAIN_SCHEDULE_YEARS = 1
ROLLING_WINDOW_YEARS = 3

# ---------------------------------------------------------------------------
# Charts and presentation
# ---------------------------------------------------------------------------
CHART_PALETTE = {
    "champion": "#1e293b",
    "challenger": "#2563eb",
    "never_retrain": "#64748b",
    "scheduled": "#0ea5e9",
    "drift_triggered": "#16a34a",
    "green": "#16a34a",
    "amber": "#f59e0b",
    "red": "#dc2626",
    "reference": "#94a3b8",
    "highlight": "#7c3aed",
}

# Validated colorblind-safe categorical palette (dataviz skill, light mode)
COLORBLIND_SAFE_PALETTE = [
    "#2a78d6",  # blue (slot 1)
    "#eb6834",  # orange (slot 2)
    "#1baf7a",  # aqua (slot 3)
    "#eda100",  # yellow (slot 4)
    "#e87ba4",  # magenta (slot 5)
    "#008300",  # green (slot 6)
    "#6250d6",  # violet (slot 7)
    "#e34948",  # red (slot 8)
]

EVENT_ANNOTATIONS = {
    2007: "Housing crisis begins",
    2008: "Financial crisis peak",
    2009: "Recovery begins",
    2020: "COVID-19 pandemic",
    2022: "Interest rate rises",
}

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e5e7eb"
GREEN = "#16a34a"
RED = "#dc2626"
BLUE = "#2563eb"
BRAND = "#1e293b"

FEATURE_READABLE_NAMES = {
    # Freddie Mac
    "credit_score": "Credit Score (FICO)",
    "original_cltv": "Combined LTV (%)",
    "original_dti": "Debt-to-Income (%)",
    "original_ltv": "Loan-to-Value (%)",
    "original_interest_rate": "Interest Rate (%)",
    "original_upb": "Loan Amount ($)",
    "original_loan_term": "Loan Term (months)",
    "mi_pct": "Mortgage Insurance (%)",
    "num_units": "Number of Units",
    "num_borrowers": "Number of Borrowers",
    "first_time_homebuyer": "First-Time Buyer",
    "occupancy_status": "Occupancy Status",
    "channel": "Origination Channel",
    "property_state": "Property State",
    "property_type": "Property Type",
    "loan_purpose": "Loan Purpose",
    "credit_score_missing": "Credit Score Missing",
    "dti_missing": "DTI Missing",
    "cltv_missing": "CLTV Missing",
    "mi_pct_missing": "MI % Missing",
    "num_borrowers_missing": "Borrowers Missing",
    # SBA
    "GrossApproval": "Gross Approval ($)",
    "SBAGuaranteedApproval": "SBA Guaranteed ($)",
    "TermInMonths": "Term (months)",
    "NaicsCode": "NAICS Sector",
    "BusinessType": "Business Type",
    "ProjectState": "Project State",
    "JobsSupported": "Jobs Supported",
    "RevolverStatus": "Revolver",
    "InitialInterestRate": "Interest Rate (%)",
    "BusinessAge": "Business Age",
    "ProcessingMethod": "Processing Method",
    "CollateralInd": "Collateral",
    "SoldSecMrktInd": "Sold Secondary Market",
    "FixedorVariableInterestInd": "Fixed/Variable Rate",
    "sba_guaranteed_pct": "SBA Guarantee Share (%)",
    "naics_sector": "NAICS Sector (2-digit)",
    # UCI
    "LIMIT_BAL": "Credit Limit",
    "AGE": "Age",
    "PAY_0": "Repayment Status (Sep)",
    "PAY_2": "Repayment Status (Aug)",
    "PAY_3": "Repayment Status (Jul)",
    "PAY_4": "Repayment Status (Jun)",
    "PAY_5": "Repayment Status (May)",
    "PAY_6": "Repayment Status (Apr)",
    "BILL_AMT1": "Bill Amount (Sep)",
    "BILL_AMT2": "Bill Amount (Aug)",
    "BILL_AMT3": "Bill Amount (Jul)",
    "BILL_AMT4": "Bill Amount (Jun)",
    "BILL_AMT5": "Bill Amount (May)",
    "BILL_AMT6": "Bill Amount (Apr)",
    "PAY_AMT1": "Payment Amount (Sep)",
    "PAY_AMT2": "Payment Amount (Aug)",
    "PAY_AMT3": "Payment Amount (Jul)",
    "PAY_AMT4": "Payment Amount (Jun)",
    "PAY_AMT5": "Payment Amount (May)",
    "PAY_AMT6": "Payment Amount (Apr)",
    "EDUCATION": "Education Level",
}

CREDIT_LINE = (
    "Freddie Mac Single-Family Loan-Level Dataset used under licence; "
    "SBA 7(a) FOIA data: U.S. Government Works; "
    "UCI Default of Credit Card Clients: Yeh & Lien (2009)."
)
