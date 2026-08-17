"""
Part B: independent pandas-based constraint validator.

This module does NOT trust the heuristic's internal bookkeeping -- it
recomputes every hard constraint from the final roster DataFrame plus the
raw reference data (guards, leave, locations, requirements), using pandas
groupby / pivot / window-style operations. This is the "monitoring" /
"validation report" the project spec asks for: a pass/fail + a concrete
violation list, not a claimed "100% valid".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CONSECUTIVE_DAYS = 6


def _max_consecutive_run(row: np.ndarray) -> int:
    """Longest run of worked (1) days in a single guard's day sequence."""
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def validate_roster(
    roster: pd.DataFrame,
    requirements: pd.DataFrame,
    guards: pd.DataFrame,
    leave: pd.DataFrame,
    locations: pd.DataFrame,
    n_days: int,
    max_consecutive: int = MAX_CONSECUTIVE_DAYS,
) -> dict:
    violations = []

    # --- coverage: required (location, shift, day) vs what the roster delivered
    assigned_counts = (
        roster.groupby(["location_id", "shift", "day"]).size().rename("assigned_count")
    )
    coverage = requirements.merge(
        assigned_counts, on=["location_id", "shift", "day"], how="left"
    )
    coverage["assigned_count"] = coverage["assigned_count"].fillna(0)
    coverage["covered"] = coverage["assigned_count"] >= coverage["required_count"]
    coverage_pct = 100.0 * coverage["covered"].mean()

    # --- hard rule 1: no guard double-booked same day
    per_day_counts = roster.groupby(["guard_id", "day"]).size()
    double_booked = per_day_counts[per_day_counts > 1]
    for (guard_id, day), n in double_booked.items():
        violations.append(
            {"type": "double_booking", "guard_id": guard_id, "day": day, "detail": f"{n} shifts"}
        )

    # --- hard rule 2: zone eligibility (guard's home zone == location's zone)
    merged = roster.merge(guards[["guard_id", "zone"]], on="guard_id").merge(
        locations[["location_id", "zone"]], on="location_id", suffixes=("_guard", "_loc")
    )
    ineligible = merged[merged["zone_guard"] != merged["zone_loc"]]
    for _, r in ineligible.iterrows():
        violations.append(
            {
                "type": "eligibility",
                "guard_id": r["guard_id"],
                "day": r["day"],
                "detail": f"guard zone {r['zone_guard']} != location zone {r['zone_loc']}",
            }
        )

    # --- hard rule 3: no guard scheduled while on approved leave
    leave_conflicts = roster.merge(leave, on=["guard_id", "day"], how="inner")
    for _, r in leave_conflicts.iterrows():
        violations.append(
            {"type": "leave_conflict", "guard_id": r["guard_id"], "day": r["day"], "detail": "assigned while on leave"}
        )

    # --- hard rule 4: min rest -- no night(day d) -> morning(day d+1) for same guard
    shift_pivot = roster.pivot_table(index="guard_id", columns="day", values="shift", aggfunc="first")
    for d in range(1, n_days):
        if d not in shift_pivot.columns or (d + 1) not in shift_pivot.columns:
            continue
        bad = shift_pivot[(shift_pivot[d] == "night") & (shift_pivot[d + 1] == "morning")]
        for guard_id in bad.index:
            violations.append(
                {"type": "rest_violation", "guard_id": guard_id, "day": d + 1, "detail": f"night(day {d}) -> morning(day {d + 1})"}
            )

    # --- hard rule 5: max consecutive working days
    worked = shift_pivot.reindex(columns=range(1, n_days + 1)).notna().astype(int)
    max_runs = worked.apply(lambda row: _max_consecutive_run(row.values), axis=1)
    over_run = max_runs[max_runs > max_consecutive]
    for guard_id, run_len in over_run.items():
        violations.append(
            {"type": "max_consecutive_days", "guard_id": guard_id, "day": None, "detail": f"worked {run_len} days straight"}
        )

    # --- workload distribution (all guards, including zero-shift ones)
    hours_by_guard = roster.groupby("guard_id")["hours"].sum().rename("hours")
    all_hours = guards[["guard_id"]].merge(hours_by_guard, on="guard_id", how="left").fillna({"hours": 0})
    workload_stats = {
        "min_hours": float(all_hours["hours"].min()),
        "max_hours": float(all_hours["hours"].max()),
        "mean_hours": float(all_hours["hours"].mean()),
        "std_hours": float(all_hours["hours"].std()),
        "guards_with_zero_hours": int((all_hours["hours"] == 0).sum()),
    }

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_violations": len(violations),
        "coverage_pct": coverage_pct,
        "coverage_df": coverage,
        "workload_stats": workload_stats,
    }
