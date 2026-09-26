"""Bootstrap sign-stability check for a CI bound near zero.

MEASURED (peer lane, 2026-09-18): at 1,000 resamples, 3 of 7 gated cells whose bound sat
within 0.01 of zero flipped that bound's SIGN across bootstrap seeds, and one printed PASS
died at 10,000 resamples. A bound that close to zero is not resolved by one more digit of
resample count at a fixed seed -- it is resolved (or shown unresolved) by reseeding.

This module is the reseed-and-compare step. Every interval this lane prints is read at 10k
resamples; any bound within `tol` of zero is read again at 50k resamples under three more
seeds, and `verdict_label` reports "undecided" -- not "pass" or "fail" -- when the flagged
bound's sign does not agree across all three reseeds.
"""

from __future__ import annotations

from typing import Callable, Sequence

DEFAULT_SEEDS: tuple[int, ...] = (101, 202, 303)
DEFAULT_RESEED_RESAMPLES = 50_000
DEFAULT_TOL = 0.01


def is_near_zero(x: float | None, tol: float = DEFAULT_TOL) -> bool:
    """True for a finite bound within `tol` of zero. `None` and NaN are "not near zero" --
    there is no sign to be unstable, and a missing bound is a different failure (`_nan` in
    `pinq_train.gate` already says "not measured" for those, which this must not paper over)."""
    if x is None:
        return False
    if x != x:  # NaN, without importing math for one comparison
        return False
    return abs(x) <= tol


def sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def bound_stability(
    reseed: Callable[[int, int], tuple[float, float]],
    *,
    primary_lo: float | None,
    primary_hi: float | None,
    tol: float = DEFAULT_TOL,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_resamples: int = DEFAULT_RESEED_RESAMPLES,
) -> dict | None:
    """`None` when neither bound is near zero -- the common case, and no reseeding happens.
    Otherwise a record of every reseed's `(ci_lo, ci_hi)` and whether the FLAGGED bound's sign
    agreed across all of them. `reseed(seed, n_resamples)` must return a fresh `(ci_lo, ci_hi)`
    computed at that seed and resample count -- the caller owns what "recompute" means (a
    suite's matched-cost delta, a gate criterion, ...); this function only owns the tolerance
    check and the sign comparison.
    """
    flagged = [b for b, v in (("lo", primary_lo), ("hi", primary_hi)) if is_near_zero(v, tol)]
    if not flagged:
        return None
    reseeds = [reseed(s, n_resamples) for s in seeds]
    stable = {}
    for b in flagged:
        idx = 0 if b == "lo" else 1
        signs = {sign(r[idx]) for r in reseeds}
        stable[b] = len(signs) <= 1
    return {
        "flagged_bounds": flagged,
        "tol": tol,
        "seeds": list(seeds),
        "n_resamples": n_resamples,
        "reseed_bounds": [
            {"seed": s, "ci_lo": r[0], "ci_hi": r[1]} for s, r in zip(seeds, reseeds)
        ],
        "stable": stable,
        "all_stable": all(stable.values()),
    }


def verdict_label(passed: bool, stability: dict | None) -> str:
    """ "pass"/"fail" unless a flagged bound's sign disagreed across reseeds, in which case
    "undecided" overrides whatever the primary seed's `passed` said."""
    if stability is not None and not stability["all_stable"]:
        return "undecided"
    return "pass" if passed else "fail"
