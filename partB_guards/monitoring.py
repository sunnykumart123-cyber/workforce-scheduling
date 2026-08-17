"""
Part B "roster monitoring": a weekly summary (coverage %, workload
distribution, violations) built on top of the validator's output, plus two
plots. This is the ongoing-monitoring view an ops manager would actually
look at, as opposed to the validator's raw pass/fail/violation-list detail.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def weekly_summary(validation: dict, roster: pd.DataFrame, guards: pd.DataFrame) -> dict:
    ws = validation["workload_stats"]
    return {
        "coverage_pct": validation["coverage_pct"],
        "n_violations": validation["n_violations"],
        "passed": validation["passed"],
        "guards_used": roster["guard_id"].nunique(),
        "guards_total": len(guards),
        "min_hours": ws["min_hours"],
        "max_hours": ws["max_hours"],
        "mean_hours": ws["mean_hours"],
        "std_hours": ws["std_hours"],
        "guards_with_zero_hours": ws["guards_with_zero_hours"],
    }


def print_weekly_summary(summary: dict, violations: list[dict]) -> None:
    print("\n--- Part B weekly roster monitoring ---")
    print(f"Coverage:            {summary['coverage_pct']:.1f}% of required (location, shift, day) slots filled")
    print(f"Guards used:         {summary['guards_used']} / {summary['guards_total']}")
    print(
        f"Workload (hours/wk): min={summary['min_hours']:.0f}  mean={summary['mean_hours']:.1f}  "
        f"max={summary['max_hours']:.0f}  std={summary['std_hours']:.2f}"
    )
    print(f"Guards with 0 hours: {summary['guards_with_zero_hours']}")
    print(f"Validator:           {'PASS' if summary['passed'] else 'FAIL'} ({summary['n_violations']} hard-rule violations)")
    if violations:
        by_type = pd.Series([v["type"] for v in violations]).value_counts()
        for vtype, n in by_type.items():
            print(f"  - {vtype}: {n}")


def plot_workload_histogram(roster: pd.DataFrame, guards: pd.DataFrame, output_path: str):
    hours = guards[["guard_id"]].merge(
        roster.groupby("guard_id")["hours"].sum().rename("hours"), on="guard_id", how="left"
    ).fillna({"hours": 0})["hours"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hours, bins=range(0, int(hours.max()) + 16, 8), edgecolor="black", color="#4C72B0")
    ax.axvline(hours.mean(), color="red", linestyle="--", label=f"mean = {hours.mean():.1f}h")
    ax.set_xlabel("Hours worked in the week")
    ax.set_ylabel("Number of guards")
    ax.set_title("Part B -- guard workload distribution (286 guards)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_coverage_by_day(coverage_df: pd.DataFrame, output_path: str):
    by_day = coverage_df.groupby("day")["covered"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_day.index, by_day.values, color="#55A868", edgecolor="black")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Coverage %")
    ax.set_ylim(0, 100)
    ax.set_title("Part B -- required-slot coverage % by day")
    for d, v in by_day.items():
        ax.text(d, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
