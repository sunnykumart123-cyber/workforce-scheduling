"""
Part B: rule-based greedy guard rostering + local-improvement rebalancing.

THE ALGORITHM (exact rule order, this IS the heuristic):
  1. Process days in chronological order 1..7. Rest/consecutive-day rules are
     temporal, so slot order across days can't be arbitrary -- day order must
     be respected.
  2. Within a day, sort that day's required (location, shift) slots by
     SCARCITY = number of currently eligible+available candidate guards,
     ascending (most-constrained-first / "fail-fast" scheduling: a slot with
     one possible guard must be filled before that guard gets grabbed by an
     easier slot).
  3. For each slot, among eligible candidates, pick the guard with:
       a. fewest hours worked so far this week   (load balancing)
       b. then shortest current consecutive-day streak (rest-compliance
          preference: prefer the more-rested guard)
       c. then lowest guard_id                    (deterministic tie-break)
  4. Hard rules checked before ANY assignment (guard is simply not a
     candidate if any fail):
       - zone eligibility (guard's home zone == location's zone)
       - availability (not on pre-approved leave that day)
       - one shift per guard per day
       - min rest: a guard who worked NIGHT on day d-1 cannot work MORNING
         on day d
       - max consecutive working days (6): a guard who has already worked 6
         days in a row is forced to rest before working again
  5. Rebalancing sweep: after the greedy fill, repeatedly try to move a
     shift from an over-loaded guard to an eligible, rest/streak-compliant,
     under-loaded guard in the same zone, to shrink the hours std-dev.

A slot with zero eligible candidates is left uncovered -- the heuristic does
not fabricate coverage. Uncovered slots surface honestly in the validator.
"""
from __future__ import annotations

import pandas as pd

MAX_CONSECUTIVE_DAYS = 6
SHIFT_HOURS = 8
REBALANCE_PASSES = 30  # per zone; cheap since a zone has only a few dozen guards
REBALANCE_THRESHOLD_HOURS = 8  # stop once max-min spread is <= 1 shift (8h)


def _streak_ending(assignments: dict, g: int, day: int) -> int:
    """Consecutive days g worked, ending at `day` (walking backward). Always
    derived from the `assignments` dict directly -- never cached -- so it
    stays correct through arbitrary rebalancing swaps."""
    if day is None or day < 1:
        return 0
    s = 0
    d = day
    while d in assignments[g]:
        s += 1
        d -= 1
    return s


def _is_eligible(assignments: dict, leave_set: set, zone_of: dict, g: int,
                  loc_zone: int, day: int, shift: str) -> bool:
    if zone_of[g] != loc_zone:
        return False
    if (g, day) in leave_set:
        return False
    if day in assignments[g]:
        return False
    prev = assignments[g].get(day - 1)
    if shift == "morning" and prev is not None and prev[1] == "night":
        return False
    if _streak_ending(assignments, g, day - 1) >= MAX_CONSECUTIVE_DAYS:
        return False
    return True


def _best_candidate(assignments: dict, leave_set: set, zone_of: dict,
                     zone_guards: dict, loc_zone: int, day: int, shift: str):
    pool = [
        g for g in zone_guards.get(loc_zone, [])
        if _is_eligible(assignments, leave_set, zone_of, g, loc_zone, day, shift)
    ]
    if not pool:
        return None
    pool.sort(
        key=lambda g: (
            SHIFT_HOURS * len(assignments[g]),          # (a) fewest hours so far
            _streak_ending(assignments, g, day - 1),      # (b) most rested
            g,                                             # (c) deterministic
        )
    )
    return pool[0]


def run_heuristic(guards: pd.DataFrame, leave: pd.DataFrame,
                   locations: pd.DataFrame, requirements: pd.DataFrame,
                   n_days: int, rebalance: bool = True) -> tuple[pd.DataFrame, list[dict], int]:
    zone_of = dict(zip(guards["guard_id"], guards["zone"]))
    loc_zone_of = dict(zip(locations["location_id"], locations["zone"]))
    zone_guards: dict[int, list] = {}
    for g, z in zone_of.items():
        zone_guards.setdefault(z, []).append(g)
    for z in zone_guards:
        zone_guards[z].sort()
    leave_set = set(zip(leave["guard_id"], leave["day"]))

    assignments: dict[int, dict] = {g: {} for g in guards["guard_id"]}
    roster_map: dict[tuple, int] = {}  # (location_id, shift, day) -> guard_id
    uncovered = []

    for day in range(1, n_days + 1):
        day_slots = requirements[requirements["day"] == day]

        # scarcity computed once at day-start (before today's assignments)
        scarcity = []
        for _, row in day_slots.iterrows():
            loc_zone = row["zone"]
            pool_size = sum(
                1
                for g in zone_guards.get(loc_zone, [])
                if _is_eligible(assignments, leave_set, zone_of, g, loc_zone, day, row["shift"])
            )
            scarcity.append((pool_size, row["location_id"], row["shift"], loc_zone))
        scarcity.sort(key=lambda x: (x[0], x[1], x[2]))  # most-constrained-first

        for pool_size, location_id, shift, loc_zone in scarcity:
            best = _best_candidate(assignments, leave_set, zone_of, zone_guards, loc_zone, day, shift)
            if best is None:
                uncovered.append({"location_id": location_id, "shift": shift, "day": day})
                continue
            assignments[best][day] = (location_id, shift)
            roster_map[(location_id, shift, day)] = best

    n_swaps = 0
    if rebalance:
        n_swaps = _rebalance(roster_map, assignments, zone_of, loc_zone_of, leave_set, zone_guards, guards)

    roster_rows = [
        {"location_id": loc, "shift": shift, "day": day, "guard_id": g, "hours": SHIFT_HOURS}
        for (loc, shift, day), g in roster_map.items()
    ]
    roster_df = pd.DataFrame(roster_rows).sort_values(["day", "location_id", "shift"]).reset_index(drop=True)
    return roster_df, uncovered, n_swaps


def _rebalance(roster_map: dict, assignments: dict, zone_of: dict, loc_zone_of: dict,
                leave_set: set, zone_guards: dict, guards: pd.DataFrame) -> int:
    """Local-improvement swap sweep, per zone: repeatedly move one shift from
    the zone's most-loaded guard to its least-loaded eligible guard, until
    the zone's max-min hour spread can't be reduced further (<= one shift --
    the best possible with integer 8h shifts) or no feasible swap exists.

    Swaps are necessarily zone-local: eligibility ties a guard to their home
    zone, so a guard-rich zone can never lend hours to a guard-scarce zone.
    That means the achievable fairness ceiling here is "balanced within each
    zone", not "balanced across the whole roster" -- see the honesty report.
    """
    n_swaps = 0
    for zone, members in zone_guards.items():
        for _ in range(REBALANCE_PASSES):
            hours = {g: SHIFT_HOURS * len(assignments[g]) for g in members}
            g_max = max(members, key=lambda g: hours[g])
            g_min = min(members, key=lambda g: hours[g])
            if hours[g_max] - hours[g_min] <= REBALANCE_THRESHOLD_HOURS:
                break  # spread already at the best achievable granularity
            swapped = False
            for day, (loc, shift) in list(assignments[g_max].items()):
                if loc_zone_of[loc] != zone:
                    continue
                if not _is_eligible(assignments, leave_set, zone_of, g_min, zone, day, shift):
                    continue
                del assignments[g_max][day]
                assignments[g_min][day] = (loc, shift)
                roster_map[(loc, shift, day)] = g_min
                n_swaps += 1
                swapped = True
                break
            if not swapped:
                break  # g_max holds no shift g_min can legally take -- stuck
    return n_swaps
