"""
Part A instance data: Route A-53 daily trip timetable + crew pool.

A "trip" is one out-and-back run of a single bus. Buses are stationed at one
of two depots; a crew can only chain trips that start/end at the same depot
(no mid-day depot transfer modelled) -- this is what makes "pairing
generation" (instance.py + pairing_ip.py) a real combinatorial step instead
of a formality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_START_MIN = 360   # 06:00, minutes since midnight
SERVICE_END_MIN = 840     # 14:00 -- single 8h service window (keeps pairing
                           # enumeration in Part A small enough to stay EXACT)
TRIP_DURATION_RANGE = (45, 65)   # minutes, one out-and-back run
LAYOVER_RANGE = (15, 25)         # minutes, scheduled recovery time at depot


def generate_trips(n_buses: int = 14, seed: int = 42) -> pd.DataFrame:
    """Synthesize a daily trip timetable for `n_buses` on Route A-53.

    Each bus runs back-to-back out-and-back trips from its depot, starting at
    a slightly staggered time, until the service window closes. Buses
    alternate between Depot-A and Depot-B (odd/even bus_id).
    """
    rng = np.random.default_rng(seed)
    rows = []
    trip_id = 0
    for bus_id in range(1, n_buses + 1):
        depot = "Depot-A" if bus_id % 2 == 0 else "Depot-B"
        t = SERVICE_START_MIN + int(rng.integers(0, 15))  # staggered pull-out
        while True:
            dur = int(rng.integers(*TRIP_DURATION_RANGE))
            end = t + dur
            if end > SERVICE_END_MIN:
                break
            rows.append(
                {
                    "trip_id": trip_id,
                    "bus_id": bus_id,
                    "depot": depot,
                    "start_min": t,
                    "end_min": end,
                    "duration_min": dur,
                }
            )
            trip_id += 1
            layover = int(rng.integers(*LAYOVER_RANGE))
            t = end + layover
    return pd.DataFrame(rows)


def generate_crew(n_crew: int = 31) -> pd.DataFrame:
    """31 crew members, all under the same duty agreement (single-day scope).

    max_duty_min   -- max total trip time a crew may work in a day (8h)
    max_spread_min -- max sign-on-to-sign-off span, i.e. duty + breaks (10h)
    min_break_min  -- minimum recovery time required between two trips
                       assigned to the same crew
    Weekly hour caps aren't modelled: Part A is a single-day pairing problem,
    so they can't bind here (they would matter in a multi-day extension).
    """
    return pd.DataFrame(
        {
            "crew_id": range(1, n_crew + 1),
            "max_duty_min": 480,
            "max_spread_min": 600,
            "min_break_min": 15,
        }
    )


def build_instance(n_buses: int = 14, n_crew: int = 31, seed: int = 42):
    trips = generate_trips(n_buses=n_buses, seed=seed)
    crew = generate_crew(n_crew=n_crew)
    return trips, crew
