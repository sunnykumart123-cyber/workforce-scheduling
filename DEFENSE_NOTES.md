# Defense Notes -- Q&A

### 1. Why is crew pairing a set-partitioning / set-covering problem, not something simpler?

Each trip must be covered, and each pairing is a pre-built *bundle* of
trips (a feasible day's work for one crew) rather than a single item. The
decision isn't "which trips does each crew do" directly -- it's "which
*bundles* do we buy" so that every trip ends up in at least one bought
bundle. That's exactly the set-covering structure: binary `x[k]` per
candidate bundle, one coverage constraint per trip. Classic crew scheduling
is usually posed as set-*partitioning* (`= 1` coverage, no trip covered
twice); I relaxed it to set-*covering* (`>= 1`) per the assignment's stated
formulation, which is also more forgiving of the fact that pairing
generation here doesn't consider deadheading between depots. In this
instance the optimal solution happens to cover every trip exactly once
anyway, so the relaxation doesn't change the answer.

### 2. Why do you need a separate "pairing generation" step before the ILP? Why not just let the ILP decide which trips each crew takes directly?

Because "which trips does each crew do" is not a simple binary choice --
it's a *sequencing and feasibility* problem (does this chain of trips fit
in 8h, do the breaks between them respect the minimum, does the spread stay
under 10h). Baking all of that directly into ILP constraints would need a
much heavier formulation (time-indexed or network-flow variables per
crew). It's far simpler and more standard in real crew-scheduling practice
to generate a rich *pool* of already-feasible candidate duties up front
(pairing generation, a search problem) and hand the ILP only a clean binary
choice: which duties to buy. The ILP becomes trivial to state correctly;
all the time-window complexity is isolated in one well-tested function
(`generate_pairings`).

### 3. Why can't Part B just be solved exactly like Part A?

Scale. Part A's ILP has one binary variable per candidate pairing (6,535 of
them) and one constraint per trip (84). Part B has 1,575 required slots
across 286 guards -- a comparably-structured exact IP would need on the
order of hundreds of thousands of `guard x location x shift x day`
variables (even after eligibility pruning), plus the rest and
max-consecutive-day constraints, which are considerably harder to encode
than Part A's "feasibility baked into the pairing" trick. That's the kind
of model where a MIP solver's runtime becomes unpredictable, not something
you'd want to run as an operational nightly rostering job. Pairing
generation for 84 trips already needed branching limits to stay under a
few thousand candidates; scaling that same exhaustive-enumeration idea to
1,575 slots isn't tractable. A greedy heuristic that runs in well under a
second and is validated against every hard constraint is the right
engineering trade-off -- which is the entire point of Part B existing next
to Part A.

### 4. What's the exact greedy rule order, and why that order and not some other?

1. Process **days in chronological order** -- non-negotiable, because rest
   and max-consecutive-day rules are temporal; you cannot know if a guard
   can work day 5 without knowing what they did on day 4.
2. Within a day, sort slots by **scarcity** (fewest eligible+available
   guards first) -- this is "fail-fast" / most-constrained-variable-first,
   a standard CSP heuristic: if a slot only has one possible guard, it must
   be filled before an easier slot with five possible guards accidentally
   takes that guard.
3. Among candidates for a slot, prefer **fewest hours worked so far** --
   this is what produces load balancing *as a side effect of the fill
   order*, rather than needing a separate optimization pass.
4. Tie-break on **shortest current consecutive-day streak** (prefer the
   more-rested guard), then **guard_id** for determinism (same seed always
   produces the same roster).

### 5. How does the pandas validator actually *prove* feasibility, rather than just trusting the heuristic?

It never touches the heuristic's internal Python state (its `assignments`
dict, its running hour counters). It takes only the *output* -- the final
roster DataFrame -- plus the original reference tables (requirements,
guards, leave, locations) and recomputes each hard constraint from scratch
using pandas: `groupby(guard,day).size()>1` for double-booking, a merge
against `leave` for leave conflicts, a merge of guard-zone against
location-zone for eligibility, a `pivot_table` of guard x day shifts to
check for night->morning adjacency, and a run-length scan of each guard's
worked/rest sequence for the 6-consecutive-day cap. If the heuristic had a
bug that silently violated a rule, the validator -- built independently
from the same specification, not from the heuristic's code -- would catch
it. On this instance it reports 0 violations across 286 guards, which is
the actual evidence of feasibility, not a claim.

### 6. What does the measured heuristic-vs-optimal gap actually tell you?

On the 16-guard/6-location benchmark, the heuristic matched the exact ILP's
coverage exactly (0.00 pp gap) at the default seed, and across 5 tested
seeds the worst gap was 1.43 percentage points. That's evidence -- on a
small instance where "optimal" is a knowable, provable number -- that the
greedy's scarcity-first + load-balanced rule order isn't leaving easy
coverage on the table. It is *not* proof that the full 286-guard roster is
within 1.4 points of its own (unknown, unsolvable-in-reasonable-time)
optimum -- a small instance's gap is suggestive evidence, not a guarantee
that generalizes to a 27x larger, structurally similar instance. The
honest claim is "the heuristic's design has been checked against ground
truth wherever ground truth is computable," not "the full roster is
provably near-optimal."

### 7. Why does the rebalancing pass barely change the workload numbers, and is that a bug?

Not a bug -- checked directly (see `README.md`'s per-zone table): within
every zone, hour spread is already tiny (std 1.6-4.4h) straight out of the
greedy fill, because "fewest hours so far" as a tie-break rule already
does most of the load-balancing work by construction. The large *overall*
std (9.84h) is inter-zone: guards are only eligible within their home zone,
and zones differ sharply in guards-per-location ratio (zone 4: 4.6
guards/location; zone 5: 1.7). A same-zone swap sweep, by definition,
cannot move hours across that boundary -- so the rebalancing pass correctly
finds almost nothing left to fix. The real lever for fixing that imbalance
is re-drawing zones or staffing levels, not a better scheduling algorithm;
reporting that honestly is more useful than quietly tuning the rebalancer's
threshold until the aggregate std looks smaller without actually changing
who's overworked.

### 8. Why did you switch solvers from CBC/GLPK to HiGHS?

Neither CBC nor GLPK is installed in this environment (`SolverFactory`
reports both unavailable); Pyomo's `appsi_highs` interface to HiGHS is
installed and available, so both Part A's and Part B-benchmark's ILPs use
it. HiGHS is a modern, actively-maintained open-source MIP/LP solver
(used as SciPy's default LP backend); functionally it plays the same role
CBC/GLPK would here -- an exact branch-and-bound solver that proves
optimality, which both parts confirm via `termination_condition ==
optimal`.
