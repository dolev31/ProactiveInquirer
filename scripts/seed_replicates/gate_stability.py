"""Reseed-and-compare stability for `pinq_train.gate.run_gate` criteria' CI bounds.

Lane L2.2's coordinator rule (2026-09-18, from a peer's measurement): a gated criterion's
`ci_lo`/`ci_hi` within 0.01 of zero is read again at 50k resamples under three more bootstrap
seeds, and reported "undecided" rather than "pass"/"fail" when the flagged bound's sign
disagrees across them. `contrast.py` gives `_matched_cost`'s by-suite/pooled deltas this
treatment; this module gives the SAME treatment to a `run_gate` verdict's criteria.

CHECK EVERY CI-BEARING CRITERION, NOT JUST ONE. An earlier version of this module took a single
`criterion` (defaulting to `length_equivalence`, the only one of the four cells this lane prints
side by side -- `distinct3`, `malformed`, `length`, `stop` -- that carries a bootstrap interval
at all). MEASURED on this lane's own s0/musique dev verdict: `evidence_coverage`'s `ci_lo` came
back `-0.00063`, inside the 0.01 tolerance, and would have been silently skipped by that
default -- the exact near-miss the coordinator's rule exists to catch, on the criterion this
lane's headline claim is actually about. `gate_with_stability` now checks every criterion named
in `criteria` (default: every CI-bearing one `run_gate` can emit) in ONE reseed pass: `run_gate`
computes all criteria together, so reseeding once per extra seed re-checks all of them, not one
seed-quadrupling per criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # a checkout without an editable install
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `scripts/` is not a package; sibling import by path
    sys.path.insert(0, str(HERE))

from stability import (  # noqa: E402
    DEFAULT_RESEED_RESAMPLES,
    DEFAULT_SEEDS,
    bound_stability,
    is_near_zero,
    verdict_label,
)

from pinq_train.gate import run_gate  # noqa: E402

# Every criterion `pinq_train.gate.run_gate` can stamp with `ci_lo`/`ci_hi` (measured against a
# real verdict's `criteria` keys); `distinct3`, `malformed` and `stop_2x2` are never in this set
# because they carry no bootstrap interval to begin with.
CI_BEARING_CRITERIA: tuple[str, ...] = (
    "evidence_coverage",
    "cad_ge2",
    "facet_breadth",
    "length_equivalence",
    "newly_reachable_share",
)


def gate_with_stability(
    *, criteria: Sequence[str] = CI_BEARING_CRITERIA, **gate_kwargs: Any
) -> dict[str, Any]:
    """Run `run_gate(**gate_kwargs)` once at its own `bootstrap_seed`/`n_resamples` (the
    PRIMARY read, expected to be 10k). For every name in `criteria` whose `ci_lo`/`ci_hi` is
    within tolerance of zero, reseed-check it at 50k resamples under three more seeds -- ONE
    shared reseed pass covers every flagged criterion, since `run_gate` computes all of them
    together.

    Returns the primary `verdict`; `stability`, a `{criterion: record_or_None}` map (`None` for
    a criterion whose bound was never near zero, or absent from this verdict); and `labels`, a
    `{criterion: "pass"|"fail"|"undecided"}` map so a caller never has to apply the override
    itself.
    """
    verdict = run_gate(**gate_kwargs)
    primary_bounds = {
        name: (verdict["criteria"][name].get("ci_lo"), verdict["criteria"][name].get("ci_hi"))
        for name in criteria
        if name in verdict["criteria"]
    }
    any_flagged = any(is_near_zero(lo) or is_near_zero(hi) for lo, hi in primary_bounds.values())
    reseed_cache: dict[int, dict[str, Any]] = {}

    def reseed_full(seed: int, n: int) -> dict[str, Any]:
        if seed not in reseed_cache:
            kw = dict(gate_kwargs)
            kw["bootstrap_seed"] = seed
            kw["n_resamples"] = n
            reseed_cache[seed] = run_gate(**kw)
        return reseed_cache[seed]

    stability: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for name, (lo, hi) in primary_bounds.items():

        def reseed_one(seed: int, n: int, name=name) -> tuple[float, float]:
            r = reseed_full(seed, n)["criteria"][name]
            return (r["ci_lo"], r["ci_hi"])

        stability[name] = (
            bound_stability(
                reseed_one,
                primary_lo=lo,
                primary_hi=hi,
                seeds=DEFAULT_SEEDS,
                n_resamples=DEFAULT_RESEED_RESAMPLES,
            )
            if any_flagged
            else None
        )
        labels[name] = verdict_label(bool(verdict["criteria"][name].get("passed")), stability[name])

    return {"verdict": verdict, "stability": stability, "labels": labels}
