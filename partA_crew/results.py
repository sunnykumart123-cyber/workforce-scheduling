"""
Part A: turn the ILP solution into a crew assignment table, a coverage Gantt
chart, and an INDEPENDENT re-validation of the solution (never trust the
solver blindly -- recompute duty/spread/coverage straight from the raw trip
data).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_assignment_table(trips: pd.DataFrame, selected_pairings) -> pd.DataFrame:
    """One row per (crew, trip): crew_id is just the pairing's rank, 1..K."""
    trips_by_id = trips.set_index("trip_id")
    rows = []
    for crew_id, pairing in enumerate(selected_pairings, start=1):
        for seq, trip_id in enumerate(pairing.trip_ids, start=1):
            t = trips_by_id.loc[trip_id]
            rows.append(
                {
                    "crew_id": crew_id,
                    "pairing_id": pairing.pairing_id,
                    "sequence": seq,
                    "trip_id": trip_id,
                    "bus_id": t["bus_id"],
                    "depot": t["depot"],
                    "start_min": t["start_min"],
                    "end_min": t["end_min"],
                    "duration_min": t["duration_min"],
                }
            )
    df = pd.DataFrame(rows).sort_values(["crew_id", "sequence"]).reset_index(drop=True)
    return df


def _fmt_hhmm(m: int) -> str:
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def validate_solution(
    trips: pd.DataFrame, crew_rules: pd.Series, selected_pairings
) -> dict:
    """Recompute every hard constraint from raw trip data -- independent of
    whatever the ILP / pairing generator internally claimed."""
    violations = []

    # 1. coverage: every trip covered at least once
    covered_counts = pd.Series(0, index=trips["trip_id"])
    for p in selected_pairings:
        for t in p.trip_ids:
            covered_counts[t] += 1
    uncovered = covered_counts[covered_counts == 0]
    if len(uncovered) > 0:
        violations.append(f"{len(uncovered)} trips uncovered: {list(uncovered.index)}")

    # 2. per-pairing duty/spread/break limits, recomputed from trips table
    trips_by_id = trips.set_index("trip_id")
    max_duty = int(crew_rules["max_duty_min"])
    max_spread = int(crew_rules["max_spread_min"])
    min_break = int(crew_rules["min_break_min"])
    for p in selected_pairings:
        chain = [trips_by_id.loc[t] for t in p.trip_ids]
        duty = sum(t["duration_min"] for t in chain)
        spread = chain[-1]["end_min"] - chain[0]["start_min"]
        if duty > max_duty:
            violations.append(f"pairing {p.pairing_id}: duty {duty} > {max_duty} min")
        if spread > max_spread:
            violations.append(
                f"pairing {p.pairing_id}: spread {spread} > {max_spread} min"
            )
        for a, b in zip(chain, chain[1:]):
            gap = b["start_min"] - a["end_min"]
            if gap < min_break:
                violations.append(
                    f"pairing {p.pairing_id}: break {gap} < {min_break} min "
                    f"between trip {a.name} and {b.name}"
                )
            if a["depot"] != b["depot"]:
                violations.append(
                    f"pairing {p.pairing_id}: depot mismatch between trips"
                )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "n_trips": len(trips),
        "n_trips_covered": int((covered_counts > 0).sum()),
    }


def plot_coverage_gantt(trips: pd.DataFrame, selected_pairings, output_path: str):
    """One horizontal row per bus; each trip segment coloured by assigned crew."""
    trip_to_crew = {}
    for crew_id, p in enumerate(selected_pairings, start=1):
        for t in p.trip_ids:
            trip_to_crew.setdefault(t, crew_id)  # first covering crew, for colour

    buses = sorted(trips["bus_id"].unique())
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(14, 0.35 * len(buses) + 2))
    for row, bus_id in enumerate(buses):
        bus_trips = trips[trips["bus_id"] == bus_id]
        for _, t in bus_trips.iterrows():
            crew_id = trip_to_crew.get(t["trip_id"])
            color = cmap((crew_id - 1) % 20) if crew_id else "lightgray"
            ax.broken_barh(
                [(t["start_min"], t["duration_min"])],
                (row - 0.4, 0.8),
                facecolors=color,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.text(
                t["start_min"] + t["duration_min"] / 2,
                row,
                f"C{crew_id}",
                ha="center",
                va="center",
                fontsize=6,
            )

    ax.set_yticks(range(len(buses)))
    ax.set_yticklabels([f"Bus {b}" for b in buses])
    xticks = range(360, 841, 60)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([_fmt_hhmm(x) for x in xticks])
    ax.set_xlabel("Time of day")
    ax.set_title("Part A -- Route A-53 trip coverage by crew (14 buses)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
