"""Matched-calls comparison: pure logic, no I/O.

THE RULE (item 2 of Lane L1.4). A comparator's cap-indexed curve is 5 points, `(cap,
mean_realized_calls, mean_coverage)`, sorted by calls. To compare the trained arm's cap-C
point against the comparator AT THE SAME SPEND:

  * if the trained point's calls value falls BETWEEN two adjacent comparator points, the
    comparator's value AT that calls value is the linear interpolation between them -- never
    a step function here, because the question is "what would the comparator have scored at
    this spend", and a straight line between two measured points is the least assumption that
    still uses both of them.
  * if the trained point's calls value falls OUTSIDE every comparator point (to the left of
    the comparator's cheapest cap or to the right of its most expensive one), there is no
    comparator measurement to interpolate BETWEEN. Extrapolating a line beyond the two
    outermost measured points would manufacture a comparator value nobody measured -- exactly
    the "never state a measurement you did not take" rule (CONTRIBUTING.md rule 3) -- so this rule
    refuses to do that. Instead the nearest comparator POINT (the comparator's own cheapest or
    most expensive cap cell) is reported, labelled as unmatched, with the spend gap stated
    alongside the delta.

`locate` returns which of the three cases applies and, for the interior case, the
interpolation fraction; `interp_value` applies it to any parallel list of y-values (coverage,
but also usable for n_asks or anything else indexed the same way). Both are pure and total:
no exceptions, no NaN in, silent NaN out only if the input curve is empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from pi_eval.stats.inference import cluster_bootstrap, sign_flip_p

Mode = Literal["interior", "left_of_domain", "right_of_domain"]


@dataclass(frozen=True, slots=True)
class Bracket:
    mode: Mode
    i: int  # index of the lower (or, off-domain, the only) bracketing point
    j: int  # index of the upper bracketing point; equals i off-domain
    t: float  # interpolation fraction in [0, 1]; 0.0 off-domain (weight 1.0 on i)


def locate(xs: Sequence[float], x: float) -> Bracket:
    """Where `x` sits against the sorted-by-construction curve `xs`.

    `xs` MUST already be sorted ascending by calls (the caller sorts once when it builds the
    curve; this function does not re-sort, so a caller passing an unsorted curve gets a
    meaningless bracket rather than a silently "corrected" one -- see `curve_from_cells`).
    """
    n = len(xs)
    if n == 0:
        raise ValueError("locate: empty curve")
    if n == 1 or x <= xs[0]:
        return Bracket("left_of_domain", 0, 0, 0.0)
    if x >= xs[-1]:
        return Bracket("right_of_domain", n - 1, n - 1, 0.0)
    for k in range(n - 1):
        if xs[k] <= x <= xs[k + 1]:
            span = xs[k + 1] - xs[k]
            t = 0.0 if span == 0 else (x - xs[k]) / span
            return Bracket("interior", k, k + 1, t)
    raise AssertionError("locate: xs not sorted ascending, or x is NaN")


def interp_value(ys: Sequence[float], b: Bracket) -> float:
    """`ys[i]` off-domain (t=0 and i==j collapse the formula to that point anyway; the
    explicit branch below just avoids relying on that coincidence)."""
    if b.i == b.j:
        return ys[b.i]
    return ys[b.i] * (1.0 - b.t) + ys[b.j] * b.t


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


NEAR_ZERO_TOL = 0.01
NEAR_ZERO_CHECK_N_BOOT = 50_000
NEAR_ZERO_CHECK_SEEDS: tuple[int, ...] = (101, 202, 303)


@dataclass(frozen=True, slots=True)
class RobustBound:
    """Whether a bound within `NEAR_ZERO_TOL` of zero keeps its sign under heavier resampling.

    Added to this brief after a peer measured, on gated verdicts elsewhere in this repo, that
    at 1,000 resamples 3 of 7 cells whose bound sat within 0.01 of zero flipped that bound's
    sign across bootstrap seeds, and one printed PASS died when re-checked at 10,000. Every
    interval this module reports is already computed at `n_boot=10_000` (the caller's default);
    this re-checks ONLY a bound landing within `NEAR_ZERO_TOL` of zero, at
    `NEAR_ZERO_CHECK_N_BOOT` and three more seeds -- a bound nowhere near zero cannot flip sign
    from resampling noise, so re-running it would only spend time reconfirming the obvious.
    """

    checked: bool
    lo_stable: bool | None
    hi_stable: bool | None
    check_seeds: tuple[int, ...]
    check_n_boot: int
    check_los: tuple[float, ...]
    check_his: tuple[float, ...]


def check_near_zero_bounds(
    units: Sequence[Sequence[float]],
    lo: float,
    hi: float,
    *,
    tol: float = NEAR_ZERO_TOL,
    check_n_boot: int = NEAR_ZERO_CHECK_N_BOOT,
    check_seeds: tuple[int, ...] = NEAR_ZERO_CHECK_SEEDS,
) -> RobustBound:
    """`units` must be EXACTLY the units the primary `(lo, hi)` was computed from -- this
    reruns `cluster_bootstrap` on them at heavier settings, it does not re-derive them."""
    lo_near = abs(lo) <= tol
    hi_near = abs(hi) <= tol
    if not (lo_near or hi_near):
        return RobustBound(False, None, None, (), check_n_boot, (), ())
    reruns = [cluster_bootstrap(units, n_boot=check_n_boot, seed=s) for s in check_seeds]
    los = tuple(r[1] for r in reruns)
    his = tuple(r[2] for r in reruns)
    lo_stable = (len({_sign(x) for x in los}) == 1) if lo_near else None
    hi_stable = (len({_sign(x) for x in his}) == 1) if hi_near else None
    return RobustBound(True, lo_stable, hi_stable, check_seeds, check_n_boot, los, his)


def verdict_for(lo: float, hi: float, robust: RobustBound) -> str:
    """'excludes_zero_above' | 'excludes_zero_below' | 'covers_zero' | 'undecided'.

    A checked bound whose sign is not stable across `NEAR_ZERO_CHECK_SEEDS` makes the whole
    verdict `undecided`, even if the OTHER bound (or the primary point) looks decisive: a
    verdict is a claim about which side of zero the interval sits on, and one unstable edge is
    enough to withdraw that claim.
    """
    if robust.checked:
        if robust.lo_stable is False or robust.hi_stable is False:
            return "undecided"
    if lo > 0:
        return "excludes_zero_above"
    if hi < 0:
        return "excludes_zero_below"
    return "covers_zero"


def curve_from_cells(
    cells: Sequence[tuple[float, float, float]],
) -> tuple[list[float], list[float], list[int]]:
    """`cells`: unordered `(cap, mean_calls, mean_value)` triples for one arm's 5 cap cells.

    Returns `(xs, ys, caps)`, all three sorted ascending BY CALLS (not by cap -- the two
    coincide in every population measured here, but the sort key is what the estimator is
    entitled to assume, not what happens to be true today). Canonicalising here, once, is what
    lets `locate` skip re-sorting on every call.
    """
    ordered = sorted(cells, key=lambda c: c[1])
    caps = [int(c[0]) for c in ordered]
    xs = [c[1] for c in ordered]
    ys = [c[2] for c in ordered]
    return xs, ys, caps


@dataclass(frozen=True, slots=True)
class MatchedDelta:
    mode: Mode
    point: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n: int
    n_clusters: int
    comp_caps_used: tuple[int, ...]  # one cap (off-domain) or two (interior)
    weight_hi: float  # 0.0 off-domain; interpolation fraction on the upper cap otherwise
    note: str
    verdict: str = "covers_zero"
    robust: RobustBound = RobustBound(False, None, None, (), NEAR_ZERO_CHECK_N_BOOT, (), ())


def matched_calls_difference(
    trained: Mapping[str, float],
    comp_lo: Mapping[str, float],
    comp_hi: Mapping[str, float],
    t: float,
    *,
    mode: Mode,
    comp_caps_used: tuple[int, ...],
    clusters: Mapping[str, str] | None = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> MatchedDelta:
    """`trained` minus the comparator's value AT THE TRAINED POINT'S CALLS, all three
    `Mapping`s keyed by the same per-unit id (this repo's convention: `f"{task_id}|{seed}"`).

    `comp_lo`/`comp_hi`/`t` are exactly what `locate` + a comparator curve produce: off-domain,
    `comp_lo is comp_hi` (the same single cell passed twice) and `t` is ignored by construction
    (the interpolation formula collapses to that one point regardless of `t` when both sides are
    identical), so this one function is correct for all three modes in `Mode` without a branch.

    THE INTERPOLATION WEIGHT IS FIXED AT THE POINT ESTIMATE, not re-derived inside the
    bootstrap. Resampling clusters changes each cell's mean COVERAGE; it does not change how
    far along the x-axis the trained arm's realized spend sits, which is observed, not
    estimated, for the purpose of this comparison. Re-deriving `t` per replicate would add a
    second, unrelated source of resampling noise (in the comparator's realized-calls axis) to
    an interval that is supposed to answer one question: how uncertain is the coverage gap at
    this fixed spend. Documented here because it is a modelling choice, not the only one.

    Reduces to a single `cluster_bootstrap` call over the per-unit difference
    `trained[k] - lerp(comp_lo[k], comp_hi[k], t)`, canonicalised exactly as
    `pi_eval.stats.inference.cluster_bootstrap` canonicalises everything it resamples --
    THIS FUNCTION CARRIES NO RESAMPLER OF ITS OWN, so it cannot reintroduce the ordering
    defect `cluster_bootstrap` and `paired_difference` were fixed for (commit `4b7b24b`): any
    order sensitivity in the endpoints below would have to come from `cluster_bootstrap` itself
    regressing, which `tests/test_stats.py` already guards.
    """
    keys = sorted(set(trained) & set(comp_lo) & set(comp_hi))
    diffs = {
        k: trained[k] - (comp_lo[k] * (1.0 - t) + comp_hi[k] * t)
        for k in keys
        if not (math.isnan(trained[k]) or math.isnan(comp_lo[k]) or math.isnan(comp_hi[k]))
    }
    if not diffs:
        return MatchedDelta(
            mode,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
            comp_caps_used,
            t,
            "no overlapping units",
            verdict="undecided",
        )

    groups: dict[str, list[float]] = {}
    for k, v in diffs.items():
        groups.setdefault(clusters.get(k, k) if clusters else k, []).append(v)
    units = list(groups.values())

    point, lo, hi = cluster_bootstrap(units, n_boot=n_boot, seed=seed)
    cluster_means = [sum(v) / len(v) for v in units]
    p = sign_flip_p(cluster_means, n_perm=n_perm, seed=seed)
    robust = check_near_zero_bounds(units, lo, hi)
    verdict = verdict_for(lo, hi, robust)
    note = (
        f"{len(units)} clusters over {len(diffs)} units; weight_hi={t:.4f}; "
        "estimand = unweighted mean over clusters of (trained - matched-calls comparator)"
    )
    return MatchedDelta(
        mode, point, lo, hi, p, len(diffs), len(units), comp_caps_used, t, note, verdict, robust
    )


def dominance_at_matched_calls(
    trained_calls_point: float,
    trained_values: Mapping[str, float],
    comp_cells: Sequence[tuple[int, float, Mapping[str, float]]],
    *,
    clusters: Mapping[str, str] | None = None,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> MatchedDelta:
    """The orchestrator: locate the comparator bracket for `trained_calls_point`, then hand
    the (possibly degenerate) bracket to `matched_calls_difference`.

    `comp_cells`: the comparator's five `(cap, mean_calls_point_estimate, unit_values)`
    triples, any order (canonicalised here by calls, once, via `curve_from_cells`).
    """
    xs, _ys, caps = curve_from_cells([(cap, calls, 0.0) for cap, calls, _ in comp_cells])
    by_cap = {cap: vals for cap, _calls, vals in comp_cells}
    b = locate(xs, trained_calls_point)
    lo_cap, hi_cap = caps[b.i], caps[b.j]
    return matched_calls_difference(
        trained_values,
        by_cap[lo_cap],
        by_cap[hi_cap],
        b.t,
        mode=b.mode,
        comp_caps_used=(lo_cap,) if b.i == b.j else (lo_cap, hi_cap),
        clusters=clusters,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
