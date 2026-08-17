"""
Part A: crew-pairing generation + set-partitioning-style ILP.

Step 1 (pairing generation): enumerate every feasible PAIRING -- a chain of
trips, all at the same depot, one crew could legally work in a single day.
Feasibility is enforced by construction here (duty time, spread, min break),
so the ILP itself never has to reason about time -- only about which
pairings to pick.

Step 2 (ILP): a set-covering selection over the generated pairings --
        minimize   sum_k cost[k] * x[k]
        s.t.       sum_{k covers trip t} x[k] >= 1   for every trip t   (coverage)
                    sum_k x[k] <= n_crew                                (crew availability)
                    x[k] in {0,1}
cost[k] is primarily "1 crew used" with a tiny secondary weight on duty
hours, so the solver first minimizes headcount and, among equally-good
headcounts, prefers shorter total duty time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pyomo.environ as pyo

MAX_TRIPS_PER_PAIRING = 6  # duty cap (480 min) / (min trip+break, ~60 min) ~= 6
MAX_CONNECTION_GAP_MIN = 90  # a crew connects to the next trip within 90 min
                              # of finishing the last one -- no multi-hour idling
MAX_BRANCH = 3  # at each step only the 3 nearest-in-time feasible connections
                 # are explored (bounded neighbour search) -- this is what
                 # keeps enumeration tractable; real pairing generators use
                 # the same trick since a crew would never rationally choose
                 # a far-future connection over a near one anyway


@dataclass
class Pairing:
    pairing_id: int
    trip_ids: tuple
    depot: str
    duty_min: int     # sum of trip durations (paid driving time)
    spread_min: int    # sign-on to sign-off span (includes breaks)


def generate_pairings(trips: pd.DataFrame, crew_rules: pd.Series) -> list[Pairing]:
    """DFS-enumerate every feasible trip chain (pairing) per depot.

    A chain can be extended by any not-yet-used trip at the same depot that
    starts at least `min_break_min` after the chain's current end, as long as
    the extended chain still respects `max_duty_min` (sum of trip durations)
    and `max_spread_min` (last trip's end - first trip's start).
    """
    max_duty = int(crew_rules["max_duty_min"])
    max_spread = int(crew_rules["max_spread_min"])
    min_break = int(crew_rules["min_break_min"])

    pairings: list[Pairing] = []
    next_id = 0

    for depot, depot_trips in trips.groupby("depot"):
        depot_trips = depot_trips.sort_values("start_min").to_dict("records")

        def dfs(chain: list[dict]):
            nonlocal next_id
            duty = sum(t["duration_min"] for t in chain)
            spread = chain[-1]["end_min"] - chain[0]["start_min"]
            pairings.append(
                Pairing(
                    pairing_id=next_id,
                    trip_ids=tuple(t["trip_id"] for t in chain),
                    depot=depot,
                    duty_min=duty,
                    spread_min=spread,
                )
            )
            next_id += 1
            if len(chain) >= MAX_TRIPS_PER_PAIRING:
                return
            last_end = chain[-1]["end_min"]
            used = {t["trip_id"] for t in chain}
            reachable = [
                cand
                for cand in depot_trips
                if last_end + min_break
                <= cand["start_min"]
                <= last_end + min_break + MAX_CONNECTION_GAP_MIN
                and cand["trip_id"] not in used
            ]
            reachable.sort(key=lambda c: c["start_min"])
            for cand in reachable[:MAX_BRANCH]:
                new_duty = duty + cand["duration_min"]
                new_spread = cand["end_min"] - chain[0]["start_min"]
                if new_duty > max_duty or new_spread > max_spread:
                    continue
                dfs(chain + [cand])

        for start_trip in depot_trips:
            dfs([start_trip])

    return pairings


def pairing_cost(p: Pairing, duty_weight: float = 0.001) -> float:
    """1 unit per crew used, plus a small tie-break on duty hours."""
    return 1.0 + duty_weight * (p.duty_min / 60.0)


def solve_crew_pairing(
    trips: pd.DataFrame, pairings: list[Pairing], n_crew_available: int
):
    """Build and solve the set-covering ILP with Pyomo + HiGHS."""
    trip_ids = list(trips["trip_id"])
    pairing_ids = [p.pairing_id for p in pairings]
    pairings_by_id = {p.pairing_id: p for p in pairings}
    covers = {p.pairing_id: set(p.trip_ids) for p in pairings}
    cost = {p.pairing_id: pairing_cost(p) for p in pairings}

    model = pyo.ConcreteModel()
    model.K = pyo.Set(initialize=pairing_ids)
    model.T = pyo.Set(initialize=trip_ids)
    model.x = pyo.Var(model.K, domain=pyo.Binary)

    model.obj = pyo.Objective(
        expr=sum(cost[k] * model.x[k] for k in model.K), sense=pyo.minimize
    )

    # coverage: every trip covered by at least one selected pairing
    def coverage_rule(m, t):
        covering = [k for k in pairing_ids if t in covers[k]]
        return sum(m.x[k] for k in covering) >= 1

    model.coverage = pyo.Constraint(model.T, rule=coverage_rule)

    # crew availability: can't select more pairings than crew on hand
    model.crew_cap = pyo.Constraint(
        expr=sum(model.x[k] for k in model.K) <= n_crew_available
    )

    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model, tee=False)

    status = str(result.solver.termination_condition)
    selected = [k for k in pairing_ids if pyo.value(model.x[k]) > 0.5]
    selected_pairings = [pairings_by_id[k] for k in selected]

    return {
        "status": status,
        "selected_pairings": selected_pairings,
        "total_cost": pyo.value(model.obj),
        "n_crews_used": len(selected_pairings),
    }
