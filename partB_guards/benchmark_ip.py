"""
Part B honesty check (spec 9a): on a SMALL guard sub-instance, solve the
same hard-constrained rostering problem two ways -- the Part-B heuristic,
and an EXACT ILP (Pyomo + HiGHS) that maximizes coverage subject to the
identical hard rules (eligibility, availability, one-shift/day, min rest,
max consecutive days). The gap between the two coverage percentages is the
evidence that the heuristic is "good enough", not an assumption.

Small-instance ILP formulation:
    y[g,l,s,d] in {0,1}  -- only created for (guard, location, shift, day)
                             combos that are eligibility+availability
                             feasible AND correspond to a required slot
    maximize   sum y[g,l,s,d]                                  (coverage)
    s.t.       sum_g y[g,l,s,d] <= 1              for each required slot   (no overstaffing)
               sum_{l,s} y[g,l,s,d] <= 1           for each (g,d)          (one shift/day)
               y[g,*,night,d] + y[g,*,morning,d+1] <= 1  for each (g,d)    (min rest)
               sum_{l,s,d} y[g,l,s,d] <= max_consecutive  for each g       (max consecutive days;
                     valid as a single sum because n_days == max_consecutive + 1 here)
"""
from __future__ import annotations

import pyomo.environ as pyo
import pandas as pd

from partB_guards.instance import build_instance
from partB_guards.heuristic import run_heuristic

SMALL_N_GUARDS = 16
SMALL_N_LOCATIONS = 6
SMALL_N_ZONES = 3


def build_small_instance(seed: int = 7, n_days: int = 7):
    return build_instance(
        n_guards=SMALL_N_GUARDS,
        n_locations=SMALL_N_LOCATIONS,
        n_zones=SMALL_N_ZONES,
        n_days=n_days,
        seed=seed,
    )


def solve_exact_coverage_ip(
    guards: pd.DataFrame,
    leave: pd.DataFrame,
    locations: pd.DataFrame,
    requirements: pd.DataFrame,
    n_days: int,
    max_consecutive: int = 6,
) -> dict:
    zone_of = dict(zip(guards["guard_id"], guards["zone"]))
    leave_set = set(zip(leave["guard_id"], leave["day"]))

    var_keys = [
        (g, row["location_id"], row["shift"], row["day"])
        for _, row in requirements.iterrows()
        for g in guards["guard_id"]
        if zone_of[g] == row["zone"] and (g, row["day"]) not in leave_set
    ]

    slot_to_vars: dict = {}
    guard_day_to_vars: dict = {}
    guard_night_vars: dict = {}
    guard_morning_vars: dict = {}
    for v in var_keys:
        g, l, s, d = v
        slot_to_vars.setdefault((l, s, d), []).append(v)
        guard_day_to_vars.setdefault((g, d), []).append(v)
        if s == "night":
            guard_night_vars.setdefault((g, d), []).append(v)
        if s == "morning":
            guard_morning_vars.setdefault((g, d), []).append(v)

    model = pyo.ConcreteModel()
    model.V = pyo.Set(initialize=var_keys, dimen=4)
    model.y = pyo.Var(model.V, domain=pyo.Binary)
    model.obj = pyo.Objective(expr=sum(model.y[v] for v in model.V), sense=pyo.maximize)

    model.slot_keys = pyo.Set(initialize=list(slot_to_vars.keys()), dimen=3)
    model.slot_cap = pyo.Constraint(
        model.slot_keys, rule=lambda m, l, s, d: sum(m.y[v] for v in slot_to_vars[(l, s, d)]) <= 1
    )

    model.gd_keys = pyo.Set(initialize=list(guard_day_to_vars.keys()), dimen=2)
    model.one_per_day = pyo.Constraint(
        model.gd_keys, rule=lambda m, g, d: sum(m.y[v] for v in guard_day_to_vars[(g, d)]) <= 1
    )

    rest_pairs = [
        (g, d)
        for g in guards["guard_id"]
        for d in range(1, n_days)
        if (g, d) in guard_night_vars and (g, d + 1) in guard_morning_vars
    ]
    model.rest_keys = pyo.Set(initialize=rest_pairs, dimen=2)
    model.rest = pyo.Constraint(
        model.rest_keys,
        rule=lambda m, g, d: sum(m.y[v] for v in guard_night_vars[(g, d)])
        + sum(m.y[v] for v in guard_morning_vars[(g, d + 1)])
        <= 1,
    )

    model.G = pyo.Set(initialize=list(guards["guard_id"]))

    def maxc_rule(m, g):
        vs = [v for v in var_keys if v[0] == g]
        if not vs:
            return pyo.Constraint.Skip
        return sum(m.y[v] for v in vs) <= max_consecutive

    model.maxc = pyo.Constraint(model.G, rule=maxc_rule)

    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model, tee=False)
    status = str(result.solver.termination_condition)

    covered = int(round(pyo.value(model.obj)))
    total_required = len(requirements)
    return {
        "status": status,
        "covered_slots": covered,
        "total_required_slots": total_required,
        "coverage_pct": 100.0 * covered / total_required,
    }


def run_benchmark(seed: int = 7, n_days: int = 7) -> dict:
    guards, leave, locations, requirements = build_small_instance(seed=seed, n_days=n_days)

    heuristic_roster, uncovered, n_swaps = run_heuristic(guards, leave, locations, requirements, n_days=n_days)
    heuristic_covered = len(heuristic_roster)
    total_required = len(requirements)
    heuristic_coverage_pct = 100.0 * heuristic_covered / total_required

    ip_result = solve_exact_coverage_ip(guards, leave, locations, requirements, n_days=n_days)

    gap_pp = ip_result["coverage_pct"] - heuristic_coverage_pct
    gap_relative_pct = (
        100.0 * gap_pp / ip_result["coverage_pct"] if ip_result["coverage_pct"] > 0 else 0.0
    )

    return {
        "n_guards": SMALL_N_GUARDS,
        "n_locations": SMALL_N_LOCATIONS,
        "total_required_slots": total_required,
        "heuristic_covered_slots": heuristic_covered,
        "heuristic_coverage_pct": heuristic_coverage_pct,
        "ip_status": ip_result["status"],
        "ip_covered_slots": ip_result["covered_slots"],
        "ip_coverage_pct": ip_result["coverage_pct"],
        "gap_percentage_points": gap_pp,
        "gap_relative_pct": gap_relative_pct,
    }
