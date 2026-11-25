"""Label delay — one function that decides when a period's outcomes are known.

This module enforces the no-look-ahead rule: at any replay date, a period's
outcomes count as known only after its full outcome window has passed.
"""

import pandas as pd

from drift.config import FREDDIE_OUTCOME_WINDOW_MONTHS, SBA_OUTCOME_WINDOW_YEARS


def outcomes_known(
    period: int,
    replay_date: pd.Timestamp | int,
    dataset: str,
) -> bool:
    """Return True if outcomes for the given period are known at replay_date.

    Parameters
    ----------
    period : int
        The origination year (Freddie) or approval fiscal year (SBA).
    replay_date : Timestamp or int
        The date (or year) at which we are asking. For annual replay, an int year.
    dataset : str
        "freddie" or "sba".
    """
    if dataset == "freddie":
        # Origination year Y: latest first payment is Dec Y.
        # Outcomes known 24 months later = end of year Y+2.
        outcome_known_year = period + (FREDDIE_OUTCOME_WINDOW_MONTHS // 12)
        if isinstance(replay_date, int):
            return replay_date > outcome_known_year
        return replay_date >= pd.Timestamp(f"{outcome_known_year + 1}-01-01")

    if dataset == "sba":
        # Fiscal year FY: ends Sep 30 of FY.
        # Outcomes known 5 years after FY end = Sep 30 of FY+5.
        outcome_known_year = period + SBA_OUTCOME_WINDOW_YEARS
        if isinstance(replay_date, int):
            return replay_date > outcome_known_year
        fy_end = pd.Timestamp(f"{period}-09-30")
        window_end = fy_end + pd.DateOffset(years=SBA_OUTCOME_WINDOW_YEARS)
        return replay_date >= window_end

    if dataset == "uci":
        # UCI has no label delay (all outcomes observed at statement time)
        return True

    raise ValueError(f"Unknown dataset: {dataset}")


def known_periods(
    all_periods: list[int],
    replay_date: pd.Timestamp | int,
    dataset: str,
) -> list[int]:
    """Return the subset of periods whose outcomes are known at replay_date."""
    return [p for p in all_periods if outcomes_known(p, replay_date, dataset)]
