"""The quality-vs-budget frontier.

Three corrections over the naive version, each forced by a specific failure:

* THE X-AXIS IS CUMULATIVE SPEND, NOT TURNS. "Turns" is not a budget: a policy asking 4
  questions that each fire 10 retrievals beats one asking 16 that fire 1, at a nominal B=4.
  Worse, `drafter_only` asks zero questions, so it has no turn index at all and a
  "paired Delta-AUC" against it would be paired against nothing. Indexing by spend gives
  every arm, including the zero-question ones, a well-defined x.
* AUC IS COMPUTED PER BOOTSTRAP REPLICATE. Resample tasks once, recompute the WHOLE curve,
  then integrate. Taking a CI of a scalar computed from the mean curve throws away exactly
  the covariance that makes the paired comparison powerful.
* THE SCALAR NEVER APPEARS WITHOUT THE CURVE, and the curve carries a SUP-T SIMULTANEOUS
  band alongside the pointwise one. Only the simultaneous band licenses the sentence
  "the dip at B=4 is noise".
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Curve:
    grid: tuple[float, ...]
    mean: tuple[float, ...]
    lo_pointwise: tuple[float, ...]
    hi_pointwise: tuple[float, ...]
    lo_simultaneous: tuple[float, ...]
    hi_simultaneous: tuple[float, ...]
    auc: float
    auc_lo: float
    auc_hi: float
    n_tasks: int


def step_at_spend(spend: Sequence[float], quality: Sequence[float], x: float) -> float:
    """Quality achieved by the last checkpoint at or below spend x.

    A step function, not an interpolation: at spend x the system has genuinely only produced
    the answer from its most recent completed step, and smoothing between checkpoints would
    invent quality that never existed.
    """
    best = float("nan")
    for s, q in zip(spend, quality):
        if s <= x:
            best = q
        else:
            break
    return best


def log2_grid(lo: float, hi: float, n: int = 5) -> tuple[float, ...]:
    """A log2 budget grid, because marginal gain per DOUBLING is the quantity of interest."""
    if lo <= 0:
        lo = 1.0
    a, b = math.log2(lo), math.log2(hi)
    return tuple(2 ** (a + (b - a) * i / (n - 1)) for i in range(n)) if n > 1 else (lo,)


def auc_log2(grid: Sequence[float], values: Sequence[float]) -> float:
    """Trapezoid over the log2-spend axis, normalized by the axis length so the number is
    comparable across suites with different absolute budgets."""
    pts = [(math.log2(g), v) for g, v in zip(grid, values) if not math.isnan(v)]
    if len(pts) < 2:
        return float("nan")
    total = sum(
        (pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1)
    )
    return total / (pts[-1][0] - pts[0][0])


def _nan_last(x: float) -> tuple[int, float]:
    """Sort key for one value. NaN compares false against everything, so a plain `sorted` of a
    list holding one falls back on exactly the input order this module is removing."""
    return (1, 0.0) if math.isnan(x) else (0, x)


def _curve_key(vals: Sequence[float]) -> tuple[tuple[int, float], ...]:
    """One task's ordering key: its whole curve over the grid, NaN-safe.

    A resampling unit here is VECTOR-VALUED -- a bag of tasks, each a vector over the grid --
    so a plain `sorted` on the units is not available and the key has to be spelled out. It is
    built from the VALUES the estimator reads, never from the task id or the cluster key: a
    label sorts deterministically without being canonical, which is how this survived once.
    """
    return tuple(_nan_last(v) for v in vals)


def frontier(
    per_task: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    grid: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    clusters: Mapping[str, str] | None = None,
) -> Curve:
    """per_task: task_id -> (cumulative_spend, quality_at_that_spend).

    `clusters` maps task_id -> template_id for suites whose tasks are not independent, and the
    BOOTSTRAP RESAMPLES CLUSTERS when it is given. It resampled tasks unconditionally and took
    no cluster map at all, while the AUC point estimate reached through `report.frontier_auc`
    IS clustered -- so the interval in the F1 legend and the interval in the T1 row for the same
    quantity were computed over different units.

    THE ANSWER DOES NOT DEPEND ON THE ORDER `per_task` OR `clusters` ARRIVE IN. The resampling
    draws unit INDICES, so both the units and the tasks inside each unit are canonicalised BY
    VALUE first -- see the comment at the sort. Relabelling the cluster keys used to move every
    band endpoint while the point curve stood still.

    Measured on tau2's real template sizes [27,12,10,8,7,6,4,4,4,3,2,2,2,2,1,1,1,1], nominal 95%
    coverage of the AUC interval, 250 datasets:

        ICC = 0.0    94.0%     (correct: with no clustering the two agree)
        ICC = 0.5    52.0%     (a "95%" band that misses the truth half the time)

    This is the band that licenses "that dip is noise", printed under a figure.
    """
    tasks = sorted(per_task)
    if not tasks:
        nan = tuple(float("nan") for _ in grid)
        return Curve(
            tuple(grid), nan, nan, nan, nan, nan, float("nan"), float("nan"), float("nan"), 0
        )

    curves = {t: [step_at_spend(*per_task[t], x) for x in grid] for t in tasks}

    def mean_at(sample: Sequence[str], i: int) -> float:
        vals = [curves[t][i] for t in sample if not math.isnan(curves[t][i])]
        return statistics.fmean(vals) if vals else float("nan")

    point = [mean_at(tasks, i) for i in range(len(grid))]
    rng = random.Random(seed)
    # THE RESAMPLING UNIT. Clusters when a map is given, tasks otherwise -- and with singleton
    # clusters the two are the same draw, so no suite but tau2 moves.
    groups: dict[str, list[str]] = {}
    for t in tasks:
        groups.setdefault((clusters or {}).get(t, t), []).append(t)

    # THE ORDER THE UNITS ARRIVE IN IS NOT AN INPUT TO THE ANSWER.
    #
    # The line below used to be `[groups[k] for k in sorted(groups)]`: ordered by cluster KEY,
    # which is a label (`f"{suite_id}/{cluster or task}"` from `report.frontier_curves`, or the
    # task id when no map is given). The resampling draws unit INDICES, so `units[randrange(n)]`
    # reads a DIFFERENT unit for the same RNG draw once the list is ordered differently -- and
    # relabelling the keys reorders it while the multiset of units is untouched. Sorting keys is
    # not canonicalising values.
    #
    # MEASURED at 20 clusters / 60 tasks, n_boot=2000, seed=0, relabelling keys only: the AUC
    # POINT moved by exactly 0.0 -- `point` is computed from `tasks` and a mean over clusters is
    # order-invariant -- while auc_lo moved +0.000846576, auc_hi +0.000285956, and all 20
    # pointwise and simultaneous band endpoints moved, by up to 0.003937826. A stable centre
    # beside a wandering interval is this defect's signature, and why review passed it. These are
    # the bands printed under the headline figure, which carries 352 interval endpoints.
    #
    # Same defect and same fix as `pi_eval.stats.inference.cluster_bootstrap` and
    # `pinq_train.gate.bca_ci`. This function does not go through either -- it carries its own
    # resampler because it recomputes the WHOLE curve per replicate -- which is how it was missed.
    #
    # BOTH LEVELS are canonicalised, because a unit is a MULTISET of vector-valued tasks and
    # `cluster_bootstrap`'s docstring warns that the outer list alone "would leave the same defect
    # reachable one level down". Here the inner level is reordered first, so the outer key is a
    # function of the unit's contents and not of the order they arrived in. Two units with
    # identical value multisets tie; `sorted(groups)` supplies a deterministic tie-break, and a
    # tie is harmless because the resampler reads a task only through `curves[t]`, so swapping two
    # units with equal contents cannot change a single replicate.
    units = [sorted(groups[k], key=lambda t: _curve_key(curves[t])) for k in sorted(groups)]
    units.sort(key=lambda u: (len(u), tuple(_curve_key(curves[t]) for t in u)))

    reps: list[list[float]] = []
    aucs: list[float] = []
    for _ in range(n_boot):
        sample = [t for _ in units for t in units[rng.randrange(len(units))]]
        c = [mean_at(sample, i) for i in range(len(grid))]
        reps.append(c)
        aucs.append(auc_log2(grid, c))

    lo_pt, hi_pt = [], []
    for i in range(len(grid)):
        col = sorted(r[i] for r in reps if not math.isnan(r[i]))
        lo_pt.append(_pct(col, alpha / 2))
        hi_pt.append(_pct(col, 1 - alpha / 2))

    # sup-t: one multiplier c such that 95% of bootstrap CURVES lie wholly inside
    sds = [
        statistics.pstdev([r[i] for r in reps if not math.isnan(r[i])]) or 1e-12
        for i in range(len(grid))
    ]
    tstats = [
        max(abs(r[i] - point[i]) / sds[i] for i in range(len(grid)) if not math.isnan(r[i]))
        for r in reps
        if any(not math.isnan(r[i]) for i in range(len(grid)))
    ]
    c = _pct(sorted(tstats), 1 - alpha) if tstats else float("nan")
    lo_sim = [point[i] - c * sds[i] for i in range(len(grid))]
    hi_sim = [point[i] + c * sds[i] for i in range(len(grid))]

    aucs_ok = sorted(a for a in aucs if not math.isnan(a))
    return Curve(
        grid=tuple(grid),
        mean=tuple(point),
        lo_pointwise=tuple(lo_pt),
        hi_pointwise=tuple(hi_pt),
        lo_simultaneous=tuple(lo_sim),
        hi_simultaneous=tuple(hi_sim),
        auc=auc_log2(grid, point),
        auc_lo=_pct(aucs_ok, alpha / 2),
        auc_hi=_pct(aucs_ok, 1 - alpha / 2),
        n_tasks=len(tasks),
    )


def marginal_gain_per_doubling(curve: Curve) -> tuple[float, ...]:
    return tuple(curve.mean[i + 1] - curve.mean[i] for i in range(len(curve.mean) - 1))


def _pct(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)
