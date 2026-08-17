"""
Part B instance data: guard pool + campus locations + weekly coverage
requirements. Parametrized so the SAME generator builds both the full
286-guard roster and the small sub-instance used for the heuristic-vs-exact
benchmark (section 9a of the spec) -- one source of truth for instance shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHIFTS = ["morning", "evening", "night"]
SHIFT_HOURS = 8
N_DAYS_DEFAULT = 7
LEAVE_PROB = 0.06  # chance a guard is on pre-approved leave on any given day

# Coverage tiers: not every location needs guarding around the clock. This
# mirrors real campuses (a data centre needs 24/7 cover, a parking lot might
# only need a morning presence) and is what keeps 286 guards / 110+ locations
# a meaningfully tight -- not trivially over-supplied -- rostering problem.
TIER_PROBS = {"high": 0.30, "medium": 0.40, "low": 0.30}
TIER_SHIFTS = {
    "high": ["morning", "evening", "night"],
    "medium": ["morning", "evening"],
    "low": ["morning"],
}


def generate_guards(n_guards: int, n_zones: int, seed: int) -> pd.DataFrame:
    """Guards are hired into a home zone and are only eligible for locations
    in that zone (real campus-security staffing is zoned, not campus-wide)."""
    rng = np.random.default_rng(seed)
    zones = rng.integers(0, n_zones, size=n_guards)
    return pd.DataFrame({"guard_id": range(1, n_guards + 1), "zone": zones})


def generate_leave(guards: pd.DataFrame, n_days: int, seed: int) -> pd.DataFrame:
    """Long-form (guard_id, day) rows: pre-approved leave, independent per day."""
    rng = np.random.default_rng(seed + 1)
    rows = [
        {"guard_id": g, "day": d}
        for g in guards["guard_id"]
        for d in range(1, n_days + 1)
        if rng.random() < LEAVE_PROB
    ]
    return pd.DataFrame(rows, columns=["guard_id", "day"])


def generate_locations(n_locations: int, n_zones: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    zones = rng.integers(0, n_zones, size=n_locations)
    tiers = rng.choice(
        list(TIER_PROBS.keys()), size=n_locations, p=list(TIER_PROBS.values())
    )
    return pd.DataFrame(
        {"location_id": range(1, n_locations + 1), "zone": zones, "tier": tiers}
    )


def build_requirements(locations: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """One row per (location, shift, day) that actually needs a guard, per
    the location's tier. required_count is always 1 (single-guard posts)."""
    rows = []
    for _, loc in locations.iterrows():
        for shift in TIER_SHIFTS[loc["tier"]]:
            for day in range(1, n_days + 1):
                rows.append(
                    {
                        "location_id": loc["location_id"],
                        "zone": loc["zone"],
                        "tier": loc["tier"],
                        "shift": shift,
                        "day": day,
                        "required_count": 1,
                    }
                )
    return pd.DataFrame(rows)


def build_instance(
    n_guards: int = 286,
    n_locations: int = 112,
    n_zones: int = 10,
    n_days: int = N_DAYS_DEFAULT,
    seed: int = 42,
):
    guards = generate_guards(n_guards, n_zones, seed)
    leave = generate_leave(guards, n_days, seed)
    locations = generate_locations(n_locations, n_zones, seed)
    requirements = build_requirements(locations, n_days)
    return guards, leave, locations, requirements
