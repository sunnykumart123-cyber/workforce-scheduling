"""
Run both parts of the workforce-scheduling project end to end:
  Part A -- exact ILP crew pairing for a 14-bus route
  Part B -- rule-based heuristic guard rostering for 286 guards / 110+ sites,
            validated with pandas, and honesty-checked against an exact ILP
            on a small sub-instance.

    python run.py
    python run.py --n-guards 400 --n-locations 150 --seed 7
"""
from __future__ import annotations

import argparse
import os

from partA_crew.instance import build_instance as build_instance_a
from partA_crew.pairing_ip import generate_pairings, solve_crew_pairing
from partA_crew.results import build_assignment_table, validate_solution, plot_coverage_gantt

from partB_guards.instance import build_instance as build_instance_b
from partB_guards.heuristic import run_heuristic
from partB_guards.validator import validate_roster
from partB_guards.monitoring import (
    weekly_summary,
    print_weekly_summary,
    plot_workload_histogram,
    plot_coverage_by_day,
)
from partB_guards.benchmark_ip import run_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="Crew pairing (exact) + guard rostering (heuristic)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-buses", type=int, default=14)
    p.add_argument("--n-crew", type=int, default=31)
    p.add_argument("--n-guards", type=int, default=286)
    p.add_argument("--n-locations", type=int, default=112)
    p.add_argument("--n-zones", type=int, default=10)
    p.add_argument("--n-days", type=int, default=7)
    p.add_argument("--bench-seed", type=int, default=7)
    p.add_argument("--outdir", type=str, default="outputs")
    return p.parse_args()


def run_part_a(args, outdir):
    print("=" * 70)
    print("PART A -- Crew Pairing (Integer Programming, exact)")
    print("=" * 70)

    trips, crew = build_instance_a(n_buses=args.n_buses, n_crew=args.n_crew, seed=args.seed)
    print(f"Instance: {len(trips)} trips across {args.n_buses} buses, {args.n_crew} crew available")

    pairings = generate_pairings(trips, crew.iloc[0])
    print(f"Pairing generation: {len(pairings)} feasible candidate pairings enumerated")

    result = solve_crew_pairing(trips, pairings, n_crew_available=args.n_crew)
    print(f"ILP status: {result['status']}")
    print(f"Crews used: {result['n_crews_used']} / {args.n_crew} available")

    table = build_assignment_table(trips, result["selected_pairings"])
    validation = validate_solution(trips, crew.iloc[0], result["selected_pairings"])
    print(f"Independent validation: {'PASS' if validation['passed'] else 'FAIL'} "
          f"({validation['n_trips_covered']}/{validation['n_trips']} trips covered, "
          f"{len(validation['violations'])} violations)")
    for v in validation["violations"]:
        print(f"  - {v}")

    print("\nCrew -> trip assignment (first 15 rows):")
    print(table.head(15).to_string(index=False))

    table.to_csv(os.path.join(outdir, "partA_assignments.csv"), index=False)
    plot_coverage_gantt(trips, result["selected_pairings"], os.path.join(outdir, "partA_gantt.png"))
    print(f"\nSaved: {outdir}/partA_assignments.csv, {outdir}/partA_gantt.png")

    return {
        "n_trips": len(trips),
        "n_pairings_generated": len(pairings),
        "n_crews_used": result["n_crews_used"],
        "n_crews_available": args.n_crew,
        "validation_passed": validation["passed"],
        "table": table,
    }


def run_part_b(args, outdir):
    print("\n" + "=" * 70)
    print("PART B -- Guard Rostering (rule-based heuristic, large-scale)")
    print("=" * 70)

    guards, leave, locations, requirements = build_instance_b(
        n_guards=args.n_guards,
        n_locations=args.n_locations,
        n_zones=args.n_zones,
        n_days=args.n_days,
        seed=args.seed,
    )
    print(
        f"Instance: {len(guards)} guards, {len(locations)} locations, "
        f"{args.n_days} days, {len(requirements)} required (location, shift, day) slots"
    )

    roster, uncovered, n_swaps = run_heuristic(guards, leave, locations, requirements, n_days=args.n_days)
    print(f"Heuristic: {len(roster)} slots filled, {len(uncovered)} left uncovered, "
          f"{n_swaps} rebalancing swaps applied")

    validation = validate_roster(roster, requirements, guards, leave, locations, n_days=args.n_days)
    summary = weekly_summary(validation, roster, guards)
    print_weekly_summary(summary, validation["violations"])

    roster.to_csv(os.path.join(outdir, "partB_roster.csv"), index=False)
    with open(os.path.join(outdir, "partB_validation_report.txt"), "w") as f:
        f.write(f"PASSED: {validation['passed']}\n")
        f.write(f"Coverage: {validation['coverage_pct']:.2f}%\n")
        f.write(f"Violations: {validation['n_violations']}\n")
        for v in validation["violations"]:
            f.write(f"  {v}\n")
        if uncovered:
            f.write(f"\nUncovered slots ({len(uncovered)}):\n")
            for u in uncovered:
                f.write(f"  {u}\n")

    plot_workload_histogram(roster, guards, os.path.join(outdir, "partB_workload_histogram.png"))
    plot_coverage_by_day(validation["coverage_df"], os.path.join(outdir, "partB_coverage_by_day.png"))
    print(f"\nSaved: {outdir}/partB_roster.csv, {outdir}/partB_validation_report.txt, "
          f"{outdir}/partB_workload_histogram.png, {outdir}/partB_coverage_by_day.png")

    return {"summary": summary, "validation": validation, "n_uncovered": len(uncovered)}


def run_part_b_benchmark(args):
    print("\n" + "=" * 70)
    print("PART B honesty check -- heuristic vs exact ILP on a small sub-instance")
    print("=" * 70)

    bench = run_benchmark(seed=args.bench_seed, n_days=args.n_days)
    print(f"Small instance: {bench['n_guards']} guards, {bench['n_locations']} locations, "
          f"{bench['total_required_slots']} required slots")
    print(f"Heuristic coverage: {bench['heuristic_covered_slots']}/{bench['total_required_slots']} "
          f"= {bench['heuristic_coverage_pct']:.2f}%")
    print(f"Exact ILP coverage: {bench['ip_covered_slots']}/{bench['total_required_slots']} "
          f"= {bench['ip_coverage_pct']:.2f}%  (status: {bench['ip_status']})")
    print(f"Heuristic-vs-optimal gap: {bench['gap_percentage_points']:.2f} percentage points "
          f"({bench['gap_relative_pct']:.2f}% relative)")
    return bench


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    part_a = run_part_a(args, args.outdir)
    part_b = run_part_b(args, args.outdir)
    bench = run_part_b_benchmark(args)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Part A -- optimal crews used: {part_a['n_crews_used']} / {part_a['n_crews_available']} "
          f"(all {part_a['n_trips']} trips covered, validation "
          f"{'PASSED' if part_a['validation_passed'] else 'FAILED'})")
    print(f"Part B -- full-roster coverage: {part_b['summary']['coverage_pct']:.1f}% "
          f"({part_b['n_uncovered']} slots uncovered), validator "
          f"{'PASSED' if part_b['summary']['passed'] else 'FAILED'} "
          f"({part_b['summary']['n_violations']} violations)")
    print(f"Part B -- workload spread: {part_b['summary']['min_hours']:.0f}-"
          f"{part_b['summary']['max_hours']:.0f}h (std {part_b['summary']['std_hours']:.2f}h)")
    print(f"Part B -- heuristic-vs-optimal gap on small benchmark: "
          f"{bench['gap_percentage_points']:.2f} pp "
          f"(heuristic {bench['heuristic_coverage_pct']:.1f}% vs optimal {bench['ip_coverage_pct']:.1f}%)")


if __name__ == "__main__":
    main()
