#!/usr/bin/env python
"""How many annotations each gate and each human-centred metric actually needs.

WHY THIS EXISTS, AND WHY IT IS NOT A RULE OF THUMB. `scripts/power_analysis.py` exists because
a sensitivity figure computed from a different test than the one you run is not a sensitivity
figure. The same argument applies to a campaign size: "annotate 100 items" is a number from a
textbook, and the quantity that decides whether G-M2 can be concluded is Krippendorff's alpha
computed by `pi_eval.judges.paired.krippendorff_alpha_nominal` over units built by
`pi_eval.annotate`, with the matcher entered as one more rater. So the simulation here runs
through THOSE functions rather than through a closed-form approximation of something similar.

WHAT IT REPORTS, AND WHY IT IS A TABLE RATHER THAN A NUMBER. The n a gate needs depends on the
answer, which is what the campaign is for. A matcher that agrees with humans at kappa 0.90
needs far fewer adjudicated pairs to clear a 0.70 floor than one sitting at 0.75, and no
sample size can be quoted without assuming which. So every row is reported ACROSS A RANGE of
true values, exactly as `power_analysis.py` reports the continuous endpoints across a range of
paired sds. Read the row you believe; if you do not know which to believe, the pessimistic row
is the one to fund.

THE CRITERION. A gate is CONCLUDED when the 95% bootstrap interval lies entirely above its
preregistered floor -- not when the point estimate clears it. A point estimate of 0.82 against
a 0.80 floor concludes nothing, and "it passed" from an interval straddling the threshold is
the shape this repository refuses everywhere else. n is the smallest sample at which that
happens in at least 80% of simulated campaigns, the same power convention as the endpoints.

ADR IS SIZED DIFFERENTLY ON PURPOSE. It carries no threshold and no test -- `pi_eval.report`
declares it exploratory, point estimate and CI, never a p-value -- so there is nothing to
power. What matters is how wide its interval is, and that is driven by the number of TASKS
rather than the number of nodes: nodes within one task share an annotator, a question and a
reading, so the effective sample is the cluster count. The ICC column is what that costs.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Sequence

from pi_eval.judges.paired import krippendorff_alpha_nominal

# The preregistered floors, transcribed from `pi_eval.mining.pipeline.gate_report`. Kept here
# as literals with their source named rather than imported, because `gate_report` builds them
# into Gate objects at call time and reaching in for them would couple a planning script to
# the shape of a report.
G_M1_NODE_RECALL = 0.75
G_M1_EDGE_PRECISION = 0.80
G_M2_MATCHER_KAPPA = 0.70

POWER = 0.80
N_SIM = 300
N_BOOT = 250


def _boot_lo(sample: Sequence, stat: Callable[[Sequence], float], rng: random.Random) -> float:
    """Percentile lower bound at 95%. Resamples the UNIT the gate is computed over -- an item,
    an edge, an (ask, node) pair -- because that is what a second campaign would redraw."""
    n = len(sample)
    if n == 0:
        return float("nan")
    vals = []
    for _ in range(N_BOOT):
        draw = [sample[rng.randrange(n)] for _ in range(n)]
        v = stat(draw)
        if v == v:  # drop NaN replicates rather than letting them poison the quantile
            vals.append(v)
    if not vals:
        return float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))]


# --------------------------------------------------------------------------- proportions


def wilson_lo(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound.

    Closed form rather than a bootstrap, and not for speed: on a proportion the bootstrap is
    ESTIMATING what Wilson computes exactly, and it degenerates at the top of the range --
    resampling a sample of all-ones returns all-ones every time, so the interval collapses to a
    point and reports certainty from a handful of observations. These gates sit at 0.75-0.80
    against true values that may be near 1.0, which is exactly where that bites.
    """
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - half) / d


def n_for_proportion(true_p: float, floor: float, rng: random.Random) -> int | None:
    """Smallest n whose 95% lower bound clears `floor` in POWER of campaigns."""
    for n in _ladder():
        wins = sum(
            1
            for _ in range(N_SIM)
            if wilson_lo(sum(1 for _ in range(n) if rng.random() < true_p), n) > floor
        )
        if wins / N_SIM >= POWER:
            return n
    return None


# --------------------------------------------------------------------------- matcher kappa


def _kappa_of(draw: Sequence[tuple[str, str, str]]) -> float:
    """Through the REAL estimator. Each unit is (human_a, human_b, matcher) and the matcher is
    one more rater, which is what G-M2 asks about."""
    ratings = {str(i): {"h1": u[0], "h2": u[1], "matcher": u[2]} for i, u in enumerate(draw)}
    return krippendorff_alpha_nominal(ratings)


def _kappa_units(n: int, agree: float, rng: random.Random) -> list[tuple[str, str, str]]:
    """`agree` is the probability the matcher matches the humans on a unit. The two humans are
    given a small independent disagreement rate of their own, because a campaign in which the
    annotators never disagree is not one this simulation should flatter."""
    out = []
    for _ in range(n):
        truth = "addresses" if rng.random() < 0.5 else "does_not_address"
        other = "does_not_address" if truth == "addresses" else "addresses"
        h1 = truth
        h2 = truth if rng.random() < 0.92 else other
        m = truth if rng.random() < agree else other
        out.append((h1, h2, m))
    return out


def n_for_kappa(agree: float, floor: float, rng: random.Random) -> int | None:
    for n in _ladder():
        wins = 0
        for _ in range(N_SIM):
            units = _kappa_units(n, agree, rng)
            if _boot_lo(units, _kappa_of, rng) > floor:
                wins += 1
        if wins / N_SIM >= POWER:
            return n
    return None


# --------------------------------------------------------------------------- ADR width


def adr_halfwidth(n_tasks: int, per_task: int, icc: float, rng: random.Random) -> float:
    """Half-width of a 95% task-clustered bootstrap interval on ADR.

    Nodes are drawn with a shared per-task offset so that `icc` is the within-task
    correlation. The bootstrap resamples TASKS, matching how `pi_eval.report` clusters every
    interval it publishes -- resampling nodes instead would treat four judgments by one person
    on one question as four independent observations and report an interval roughly sqrt(deff)
    too narrow.
    """
    tasks = []
    for _ in range(n_tasks):
        shift = rng.gauss(0, icc**0.5)
        p = min(max(0.5 + shift * 0.5, 0.02), 0.98)
        tasks.append([1 if rng.random() < p else 0 for _ in range(per_task)])

    def stat(draw):
        flat = [v for t in draw for v in t]
        return sum(flat) / len(flat) if flat else float("nan")

    vals = []
    for _ in range(N_BOOT):
        d = [tasks[rng.randrange(n_tasks)] for _ in range(n_tasks)]
        v = stat(d)
        if v == v:
            vals.append(v)
    vals.sort()
    lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]
    return (hi - lo) / 2


def _ladder() -> list[int]:
    return [20, 30, 40, 60, 80, 100, 140, 180, 240, 320, 420, 560, 750, 1000]


# --------------------------------------------------------------------------- report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    print("ANNOTATION SIZING -- units needed, simulated through the real estimators")
    print(f"criterion: 95% bootstrap lower bound above the floor in >={POWER:.0%} of campaigns")
    print(f"({N_SIM} simulated campaigns x {N_BOOT} bootstrap replicates per n)\n")

    print(f"G-M1 edge precision      floor {G_M1_EDGE_PRECISION}")
    print("   if the miner's true edge precision is ...   double-annotated edges needed")
    for p in (0.85, 0.90, 0.95, 0.98):
        n = n_for_proportion(p, G_M1_EDGE_PRECISION, rng)
        print(f"      {p:.2f}                                    {n if n else '> 1000'}")

    print(f"\nG-M1 node recall         floor {G_M1_NODE_RECALL}")
    print("   if the miner's true node recall is ...      confirmed + missing needs")
    for p in (0.80, 0.85, 0.90, 0.95):
        n = n_for_proportion(p, G_M1_NODE_RECALL, rng)
        print(f"      {p:.2f}                                    {n if n else '> 1000'}")

    print(f"\nG-M2 matcher kappa       floor {G_M2_MATCHER_KAPPA}")
    print("   if the matcher agrees with humans at ...     (ask, node) pairs, 2 humans each")
    for agree in (0.85, 0.90, 0.95, 0.98):
        n = n_for_kappa(agree, G_M2_MATCHER_KAPPA, rng)
        print(f"      {agree:.2f}                                    {n if n else '> 1000'}")

    print("\nADR -- no floor and no test, so this is interval WIDTH, not power")
    print("   half-width of the 95% task-clustered interval, 4 annotated needs per task")
    print("      tasks     ICC 0.0    ICC 0.2    ICC 0.4")
    for nt in (20, 30, 50, 80, 120):
        row = [f"{adr_halfwidth(nt, 4, icc, rng):.3f}" for icc in (0.0, 0.2, 0.4)]
        print(f"      {nt:<9} {row[0]:<10} {row[1]:<10} {row[2]}")
    print("\n   Nodes per task buy little once ICC is real; TASKS are what narrow this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
