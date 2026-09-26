"""S4 — ablation-verified necessity. The ONLY stage that produces a causal claim.

Drop the evidence for a candidate need, regenerate the answer with the FROZEN answerer, and
measure the drop. This is what upgrades "appeared in 7 of 8 successful traces" (correlational,
and satisfied by any wasted work the traces happened to contain) into "required".

The four verdicts are deliberately not a threshold on a point estimate: a need whose CI
straddles zero is UNTESTABLE, not INERT, and the difference matters because UNTESTABLE nodes
are excluded from denominators while INERT nodes are dropped from the graph entirely.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from pinq.types import Evidence, TaskView

Verdict = Literal["NECESSARY", "CONTRIBUTORY", "INERT", "UNTESTABLE"]

# Q over an evidence subset, at a given seed.
ScoreFn = Callable[[TaskView, Evidence, int], float]


@dataclass(frozen=True, slots=True)
class AblationResult:
    node_id: str
    verdict: Verdict
    delta: float
    ci_lo: float
    ci_hi: float
    n_seeds: int
    base: float
    ablated_mean: float


def _mean_ci(xs: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, m, m
    se = statistics.stdev(xs) / (len(xs) ** 0.5)
    return m, m - z * se, m + z * se


def ablation_verdict(
    *,
    node_id: str,
    view: TaskView,
    reference: Evidence,
    ev_uids: frozenset[str],
    score: ScoreFn,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    necessary_delta: float = 0.8,
    inert_band: float = 0.1,
) -> AblationResult:
    if not ev_uids:
        return AblationResult(node_id, "UNTESTABLE", 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    ablated_ev = reference.without(ev_uids)
    bases = [score(view, reference, s) for s in seeds]
    ablated = [score(view, ablated_ev, s) for s in seeds]
    deltas = [b - a for b, a in zip(bases, ablated)]
    delta, lo, hi = _mean_ci(deltas)

    if delta >= necessary_delta and lo > 0:
        verdict: Verdict = "NECESSARY"
    elif lo > 0:
        verdict = "CONTRIBUTORY"
    elif lo <= 0 <= hi and abs(delta) < inert_band:
        verdict = "INERT"
    else:
        verdict = "UNTESTABLE"

    return AblationResult(
        node_id=node_id,
        verdict=verdict,
        delta=delta,
        ci_lo=lo,
        ci_hi=hi,
        n_seeds=len(seeds),
        base=statistics.fmean(bases),
        ablated_mean=statistics.fmean(ablated),
    )


def substitution_rate(
    *,
    view: TaskView,
    reference: Evidence,
    group_uids: Sequence[frozenset[str]],
    score: ScoreFn,
    seed: int = 0,
) -> float:
    """How much of a group's value survives dropping any ONE member.

    Substitutes are why a leave-one-out estimate can read ~0 for every member of a set that
    is jointly decisive. Reporting the rate is what stops that artefact being read as
    "none of these questions mattered".
    """
    if len(group_uids) < 2:
        return 0.0
    base = score(view, reference, seed)
    singles = [base - score(view, reference.without(u), seed) for u in group_uids]
    joint = base - score(view, reference.without(frozenset().union(*group_uids)), seed)
    if joint <= 0:
        return 0.0
    return max(0.0, 1.0 - sum(singles) / joint)
