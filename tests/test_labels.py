"""Tests for label derivation — hand-made examples including edge cases."""

import pandas as pd

from drift.sba.labels import derive_sba_labels, sba_usable_years


class TestSBALabels:
    """SBA label derivation on hand-made data."""

    def _make_sba_df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for col in ["ApprovalDate", "ChargeOffDate", "PaidInFullDate", "FirstDisbursementDate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        if "ApprovalFY" not in df.columns:
            df["ApprovalFY"] = pd.array([pd.NA] * len(df), dtype="Int32")
        return df

    def test_pif_is_non_default(self):
        df = self._make_sba_df([{
            "LoanStatus": "PIF",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": pd.NaT,
            "PaidInFullDate": "2013-06-01",
        }])
        result = derive_sba_labels(df)
        assert result["default"].iloc[0] == 0

    def test_chgoff_within_5_years_is_default(self):
        df = self._make_sba_df([{
            "LoanStatus": "CHGOFF",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": "2014-06-01",  # 4.4 years
        }])
        result = derive_sba_labels(df)
        assert result["default"].iloc[0] == 1

    def test_chgoff_exactly_5_years_is_default(self):
        df = self._make_sba_df([{
            "LoanStatus": "CHGOFF",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": "2015-01-15",  # exactly 5 years
        }])
        result = derive_sba_labels(df)
        assert result["default"].iloc[0] == 1

    def test_chgoff_just_outside_5_years_is_non_default(self):
        df = self._make_sba_df([{
            "LoanStatus": "CHGOFF",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": "2015-06-01",  # 5.4 years
        }])
        result = derive_sba_labels(df)
        assert result["default"].iloc[0] == 0

    def test_cancelled_excluded(self):
        df = self._make_sba_df([{
            "LoanStatus": "CANCLD",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": pd.NaT,
        }])
        result = derive_sba_labels(df)
        assert result["label_status"].iloc[0] == "excluded"
        assert pd.isna(result["default"].iloc[0])

    def test_commit_excluded(self):
        df = self._make_sba_df([{
            "LoanStatus": "COMMIT",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": pd.NaT,
        }])
        result = derive_sba_labels(df)
        assert result["label_status"].iloc[0] == "excluded"

    def test_exempt_matured_is_non_default(self):
        df = self._make_sba_df([{
            "LoanStatus": "EXEMPT",
            "ApprovalDate": "2010-01-15",
            "ChargeOffDate": pd.NaT,
        }])
        as_of = pd.Timestamp("2020-01-01")  # 10 years later
        result = derive_sba_labels(df, as_of_date=as_of)
        assert result["default"].iloc[0] == 0
        assert result["label_status"].iloc[0] == "resolved_survived"

    def test_exempt_recent_is_censored(self):
        df = self._make_sba_df([{
            "LoanStatus": "EXEMPT",
            "ApprovalDate": "2022-01-15",
            "ChargeOffDate": pd.NaT,
        }])
        as_of = pd.Timestamp("2025-01-01")  # only 3 years
        result = derive_sba_labels(df, as_of_date=as_of)
        assert pd.isna(result["default"].iloc[0])
        assert result["label_status"].iloc[0] == "censored"


class TestSBAUsableYears:
    """SBA usable year computation."""

    def test_basic_usable(self):
        df = pd.DataFrame({"ApprovalFY": pd.array([2005, 2010, 2015, 2020], dtype="Int32")})
        as_of = pd.Timestamp("2026-09-30")
        # FY2005 ends 2005-09-30, +5 = 2010-09-30 <= 2026-09-30 ✓
        # FY2020 ends 2020-09-30, +5 = 2025-09-30 <= 2026-09-30 ✓
        # FY2021 ends 2021-09-30, +5 = 2026-09-30 <= 2026-09-30 ✓ (would be usable)
        result = sba_usable_years(df, as_of)
        assert 2005 in result
        assert 2010 in result
        assert 2015 in result
        assert 2020 in result

    def test_recent_fy_not_usable(self):
        df = pd.DataFrame({"ApprovalFY": pd.array([2022, 2023], dtype="Int32")})
        as_of = pd.Timestamp("2026-09-30")
        # FY2022 ends 2022-09-30, +5 = 2027-09-30 > 2026-09-30 ✗
        result = sba_usable_years(df, as_of)
        assert 2022 not in result
        assert 2023 not in result
