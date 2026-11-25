"""Time-ordered data splits — fixed once, never changed after seeing results.

Split structure (from spec):
- Reference: first 5 usable years → baseline model trained here.
- Monitoring: following years up to last 3 usable → replay runs through these.
- Final evaluation: last 3 usable years → scored once at the end.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSplit:
    """Immutable record of a dataset's time-ordered split."""

    dataset: str
    usable_years: tuple[int, ...]
    reference_years: tuple[int, ...]
    monitoring_years: tuple[int, ...]
    final_eval_years: tuple[int, ...]

    def describe(self) -> str:
        ref = f"{self.reference_years[0]}–{self.reference_years[-1]}" if self.reference_years else "none"
        mon = f"{self.monitoring_years[0]}–{self.monitoring_years[-1]}" if self.monitoring_years else "none"
        final = f"{self.final_eval_years[0]}–{self.final_eval_years[-1]}" if self.final_eval_years else "none"
        return (
            f"{self.dataset}: {len(self.usable_years)} usable years\n"
            f"  Reference ({len(self.reference_years)}): {ref}\n"
            f"  Monitoring ({len(self.monitoring_years)}): {mon}\n"
            f"  Final eval ({len(self.final_eval_years)}): {final}"
        )


def compute_split(dataset: str, usable_years: list[int]) -> DatasetSplit:
    """Compute the fixed split for a dataset.

    Rules:
    - Reference: first 5 usable years
    - Final evaluation: last 3 usable years
    - Monitoring: everything in between

    Requires at least 9 usable years (5 ref + 1 mon + 3 eval).
    """
    years = sorted(usable_years)
    n = len(years)

    if n < 9:
        raise ValueError(
            f"{dataset}: need at least 9 usable years for the split "
            f"(5 reference + 1 monitoring + 3 evaluation), got {n}: {years}"
        )

    reference = tuple(years[:5])
    final_eval = tuple(years[-3:])
    monitoring = tuple(years[5:-3])

    return DatasetSplit(
        dataset=dataset,
        usable_years=tuple(years),
        reference_years=reference,
        monitoring_years=monitoring,
        final_eval_years=final_eval,
    )


def compute_uci_split() -> dict:
    """UCI has no time dimension — return a train/test description.

    UCI is used only for the synthetic drift benchmark (Stage 3), not for
    the main replay. The spec says: 40% reference + 20 simulated periods.
    """
    return {
        "dataset": "uci",
        "reference_fraction": 0.40,
        "n_simulated_periods": 20,
        "note": "No real time dimension; used for synthetic drift benchmark only.",
    }
