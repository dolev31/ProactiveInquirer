"""Lane L1.4: the matched-calls comparison logic in `scripts/frontier_calls/interp.py`.

Pure-logic tests only -- no parquet, no artifacts/, nothing that only exists in the main
checkout. `scripts/frontier_calls/compute.py` is the I/O layer that reads the real frontier
snapshots; it is exercised by hand from the repository root (see its own docstring) and is
deliberately not imported here, mirroring how `tests/test_frontier_human.py` tests `frontier()`
against synthetic ladders rather than a real scores.parquet.
"""

from __future__ import annotations

import math
import random

import pytest
from scripts.frontier_calls.interp import (
    RobustBound,
    check_near_zero_bounds,
    curve_from_cells,
    dominance_at_matched_calls,
    interp_value,
    locate,
    matched_calls_difference,
    verdict_for,
)

from pi_eval.stats.inference import paired_difference

# --------------------------------------------------------------------------------- locate


def test_locate_interior_interpolates_linearly():
    xs = [1.0, 2.0, 4.0, 8.0]
    b = locate(xs, 3.0)
    assert b.mode == "interior"
    assert (b.i, b.j) == (1, 2)  # bracketed by xs[1]=2.0 and xs[2]=4.0
    assert b.t == pytest.approx(0.5)
    assert interp_value([0.0, 10.0, 20.0, 30.0], b) == pytest.approx(15.0)


def test_locate_at_an_exact_grid_point_does_not_interpolate():
    xs = [1.0, 2.0, 4.0, 8.0]
    b = locate(xs, 4.0)
    assert b.t in (0.0, 1.0)  # landed exactly on a grid point either way
    assert interp_value([0.0, 10.0, 20.0, 30.0], b) == pytest.approx(20.0)


def test_locate_left_of_domain_uses_nearest_point_not_extrapolation():
    """THE RULE: a trained point cheaper than the comparator's own cheapest cap gets the
    comparator's cheapest-cap value, never a line extended past it."""
    xs = [3.6, 6.0, 7.8, 9.4, 11.6]
    ys = [0.80, 0.82, 0.82, 0.82, 0.83]
    b = locate(xs, 1.5)  # far below xs[0]
    assert b.mode == "left_of_domain"
    assert (b.i, b.j) == (0, 0)
    assert interp_value(ys, b) == pytest.approx(ys[0])


def test_locate_right_of_domain_uses_nearest_point_not_extrapolation():
    xs = [3.6, 6.0, 7.8, 9.4, 11.6]
    ys = [0.80, 0.82, 0.82, 0.82, 0.83]
    b = locate(xs, 50.0)  # far above xs[-1]
    assert b.mode == "right_of_domain"
    assert (b.i, b.j) == (4, 4)
    assert interp_value(ys, b) == pytest.approx(ys[-1])


def test_extrapolation_is_never_attempted_beyond_measured_points():
    """For ANY x outside the measured range, the returned value must equal one of the
    measured y's exactly -- never something beyond them, which is what a line extended past
    the outermost points would produce. Swept over a grid of x's on both sides."""
    xs = [2.0, 5.0, 9.0]
    ys = [0.10, 0.50, 0.40]  # deliberately non-monotone, so a bad extrapolation would show
    for x in (-100.0, -1.0, 0.0, 1.999):
        assert interp_value(ys, locate(xs, x)) == pytest.approx(ys[0])
    for x in (9.001, 10.0, 1000.0):
        assert interp_value(ys, locate(xs, x)) == pytest.approx(ys[-1])


def test_locate_single_point_curve_is_always_off_domain():
    b = locate([5.0], 5.0)
    assert b.mode == "left_of_domain"
    assert interp_value([0.42], b) == pytest.approx(0.42)


def test_locate_rejects_empty_curve():
    with pytest.raises(ValueError):
        locate([], 1.0)


def test_curve_from_cells_sorts_by_calls_not_input_order():
    """Cells arrive in whatever order a dict or a query happens to produce. The curve must be
    ordered by CALLS (the x-axis), which need not equal cap order in general even though it
    does on every population this lane measured."""
    cells = [(24, 11.6, 0.83), (4, 3.6, 0.80), (16, 9.4, 0.82), (8, 6.0, 0.82), (12, 7.8, 0.82)]
    xs, ys, caps = curve_from_cells(cells)
    assert xs == sorted(xs)
    assert caps == [4, 8, 12, 16, 24]
    assert ys[0] == pytest.approx(0.80)


# ------------------------------------------------------------- matched_calls_difference


def _cluster_map(n_clusters=10, per_cluster=4):
    """`n_clusters` templates of `per_cluster` (task, seed) units each -> cluster map."""
    clusters = {}
    keys = []
    for c in range(n_clusters):
        for j in range(per_cluster):
            k = f"t{c:02d}_{j}|{j % 2}"
            clusters[k] = f"template{c:02d}"
            keys.append(k)
    return keys, clusters


def test_matched_calls_difference_reduces_to_paired_difference_off_domain():
    """THE SUBSUMPTION CLAIM: when the bracket collapses to one point (comp_lo is comp_hi),
    `matched_calls_difference` must be identical, to the bit, to calling the codebase's own
    tested `paired_difference` on trained vs that one comparator cell. This is what licenses
    using ONE function for all three `Mode`s instead of branching into separate statistics."""
    keys, clusters = _cluster_map()
    rng = random.Random(7)
    trained = {k: rng.gauss(0.84, 0.05) for k in keys}
    comp = {k: rng.gauss(0.80, 0.05) for k in keys}

    got = matched_calls_difference(
        trained, comp, comp, 0.0, mode="left_of_domain", comp_caps_used=(4,), clusters=clusters
    )
    want = paired_difference(trained, comp, clusters=clusters)

    assert got.point == pytest.approx(want.point)
    assert got.ci_lo == pytest.approx(want.ci_lo)
    assert got.ci_hi == pytest.approx(want.ci_hi)
    assert got.p_value == pytest.approx(want.p_value)
    assert got.n == want.n


def test_matched_calls_difference_off_domain_is_insensitive_to_t():
    """`t` is documented as ignored when `comp_lo is comp_hi`; prove it, since a caller
    passing a stray nonzero `t` off-domain must not silently change the answer."""
    keys, clusters = _cluster_map()
    rng = random.Random(11)
    trained = {k: rng.random() for k in keys}
    comp = {k: rng.random() for k in keys}
    a = matched_calls_difference(
        trained, comp, comp, 0.0, mode="left_of_domain", comp_caps_used=(4,), clusters=clusters
    )
    b = matched_calls_difference(
        trained, comp, comp, 0.73, mode="left_of_domain", comp_caps_used=(4,), clusters=clusters
    )
    assert a.point == pytest.approx(b.point)
    assert a.ci_lo == pytest.approx(b.ci_lo)
    assert a.ci_hi == pytest.approx(b.ci_hi)


def test_matched_calls_difference_interior_matches_hand_computed_interpolation():
    keys, clusters = _cluster_map(n_clusters=8, per_cluster=2)
    rng = random.Random(3)
    trained = {k: rng.gauss(0.5, 0.1) for k in keys}
    lo = {k: rng.gauss(0.4, 0.1) for k in keys}
    hi = {k: rng.gauss(0.6, 0.1) for k in keys}
    t = 0.3

    got = matched_calls_difference(
        trained,
        lo,
        hi,
        t,
        mode="interior",
        comp_caps_used=(8, 12),
        clusters=clusters,
        n_boot=500,
    )

    # hand-computed point estimate: unweighted mean over clusters of the per-unit diff
    hand_diffs = {k: trained[k] - (lo[k] * (1 - t) + hi[k] * t) for k in keys}
    by_cluster: dict[str, list[float]] = {}
    for k, v in hand_diffs.items():
        by_cluster.setdefault(clusters[k], []).append(v)
    hand_point = sum(sum(v) / len(v) for v in by_cluster.values()) / len(by_cluster)

    assert got.point == pytest.approx(hand_point)
    assert got.mode == "interior"
    assert got.ci_lo <= got.point <= got.ci_hi


def test_matched_calls_difference_no_overlap_is_nan_not_a_crash():
    got = matched_calls_difference(
        {"a": 1.0}, {"b": 1.0}, {"b": 1.0}, 0.0, mode="left_of_domain", comp_caps_used=(4,)
    )
    assert math.isnan(got.point)
    assert got.n == 0


# --------------------------------------------------------------- dominance_at_matched_calls


def _make_cells(caps, calls, values_by_cap):
    return [(c, x, values_by_cap[c]) for c, x in zip(caps, calls)]


def test_dominance_at_matched_calls_interpolates_when_trained_sits_inside_the_comp_range():
    keys, clusters = _cluster_map(n_clusters=6, per_cluster=2)
    rng = random.Random(5)
    trained_values = {k: rng.gauss(0.85, 0.02) for k in keys}
    caps = [4, 8, 12, 16, 24]
    calls = [3.0, 5.0, 7.0, 9.0, 11.0]
    values_by_cap = {
        c: {k: rng.gauss(0.80 + 0.01 * i, 0.02) for k in keys} for i, c in enumerate(caps)
    }
    comp_cells = _make_cells(caps, calls, values_by_cap)

    got = dominance_at_matched_calls(6.0, trained_values, comp_cells, clusters=clusters, n_boot=300)
    assert got.mode == "interior"
    assert got.comp_caps_used == (8, 12)  # 6.0 sits between cap8's 5.0 and cap12's 7.0
    assert got.weight_hi == pytest.approx(0.5)


def test_dominance_at_matched_calls_flags_left_of_domain_like_the_real_suites():
    """Shape of the real finding on all three suites: the trained arm's realized calls sit
    below the comparator's own cheapest cap. Fixture values are the published cap-4/cap-24
    means from artifacts/frontier/FRONTIER.md (musique, base8b vs trained), used as a
    regression fixture -- not re-read from the file, so this test has no I/O."""
    keys, clusters = _cluster_map(n_clusters=20, per_cluster=2)
    rng = random.Random(1)
    # trained's mean calls: 2.4950 (cap4) .. 2.9950 (cap24) -- all below base8b's cheapest.
    trained_values = {k: rng.gauss(0.8379, 0.03) for k in keys}
    caps = [4, 8, 12, 16, 24]
    calls = [3.6225, 5.9950, 7.8400, 9.4000, 11.5775]  # base8b, from FRONTIER.md
    means = [0.7987, 0.8173, 0.8227, 0.8236, 0.8263]
    values_by_cap = {c: {k: rng.gauss(m, 0.03) for k in keys} for c, m in zip(caps, means)}
    comp_cells = _make_cells(caps, calls, values_by_cap)

    got = dominance_at_matched_calls(
        2.4950, trained_values, comp_cells, clusters=clusters, n_boot=300
    )
    assert got.mode == "left_of_domain"
    assert got.comp_caps_used == (4,)
    assert got.weight_hi == pytest.approx(0.0)


def test_dominance_at_matched_calls_does_not_depend_on_comp_cells_input_order():
    """THE BOOTSTRAP NOTE: `frontier.py` was made order-canonical in 62a22a4 specifically
    because a caller's list order used to leak into bootstrap endpoints. This function is new
    code built on top of the fixed primitives, so its own new surface -- the order `comp_cells`
    arrives in -- gets the same positive-control discipline `test_frontier_human.py` uses:
    assert equality under every permutation, not just under one."""
    keys, clusters = _cluster_map(n_clusters=15, per_cluster=3)
    rng = random.Random(42)
    trained_values = {k: rng.gauss(0.85, 0.02) for k in keys}
    caps = [4, 8, 12, 16, 24]
    calls = [3.0, 5.0, 7.0, 9.0, 11.0]
    values_by_cap = {
        c: {k: rng.gauss(0.80 + 0.01 * i, 0.02) for k in keys} for i, c in enumerate(caps)
    }
    comp_cells = _make_cells(caps, calls, values_by_cap)

    base = dominance_at_matched_calls(
        6.0, trained_values, comp_cells, clusters=clusters, n_boot=400
    )
    for perm_seed in (1, 2, 3):
        order = list(comp_cells)
        random.Random(perm_seed).shuffle(order)
        got = dominance_at_matched_calls(6.0, trained_values, order, clusters=clusters, n_boot=400)
        assert got == base, f"permutation {perm_seed} moved the result"


def test_the_order_invariance_control_has_teeth():
    """POSITIVE CONTROL for the test above: replacing one comparator cell's values (not just
    reordering) MUST move the point estimate, or the order-invariance test above would be
    vacuous (it could pass by both sides being broken identically)."""
    keys, clusters = _cluster_map(n_clusters=15, per_cluster=3)
    rng = random.Random(42)
    trained_values = {k: rng.gauss(0.85, 0.02) for k in keys}
    caps = [4, 8, 12, 16, 24]
    calls = [3.0, 5.0, 7.0, 9.0, 11.0]
    values_by_cap = {
        c: {k: rng.gauss(0.80 + 0.01 * i, 0.02) for k in keys} for i, c in enumerate(caps)
    }
    comp_cells = _make_cells(caps, calls, values_by_cap)
    base = dominance_at_matched_calls(
        6.0, trained_values, comp_cells, clusters=clusters, n_boot=400
    )

    # x=6.0 brackets between cap8 (calls=5.0) and cap12 (calls=7.0) -- index 1 in `comp_cells`,
    # which is what must be mutated for the control to have anything to detect. Asserted first,
    # so a future change to the fixture's shape fails loudly here instead of passing vacuously.
    assert base.comp_caps_used == (8, 12)
    mutated = list(comp_cells)
    cap, x, _vals = mutated[1]
    mutated[1] = (cap, x, {k: v + 1.0 for k, v in trained_values.items()})  # genuinely different
    moved = dominance_at_matched_calls(6.0, trained_values, mutated, clusters=clusters, n_boot=400)
    assert moved.point != pytest.approx(base.point), (
        "control: replacing a cell did not move the point"
    )


# ----------------------------------------------------------- near-zero bound robustness check
#
# Added after a peer measured, on gated cells elsewhere in this repo, that at 1,000 resamples
# 3 of 7 cells whose bound sat within 0.01 of zero flipped that bound's sign across bootstrap
# seeds, and one printed PASS died when re-checked at 10,000. A verdict whose near-zero bound
# is not stable under heavier resampling must read "undecided", never a pass or a fail.


def test_check_near_zero_bounds_does_not_trigger_when_far_from_zero():
    units = [[0.80, 0.82], [0.81, 0.83], [0.79, 0.80]]
    got = check_near_zero_bounds(units, lo=0.40, hi=0.60)
    assert got.checked is False
    assert got.lo_stable is None and got.hi_stable is None
    assert got.check_los == () and got.check_his == ()


def test_check_near_zero_bounds_triggers_exactly_at_the_tolerance_boundary():
    units = [[0.1]] * 5
    assert check_near_zero_bounds(units, lo=0.01, hi=0.50).checked is True
    assert check_near_zero_bounds(units, lo=0.0100001, hi=0.50).checked is False
    assert check_near_zero_bounds(units, lo=-0.50, hi=-0.01).checked is True


def test_check_near_zero_bounds_only_checks_the_bound_that_is_actually_near():
    """If only the upper bound is near zero, the lower bound's stability is not even asked
    about (`None`, not `True`) -- it was never at risk."""
    rng = random.Random(9)
    units = [[rng.gauss(0.3, 0.05)] for _ in range(60)]  # clearly positive, lo far from 0
    # synthesize hi near zero by also passing a hi close to 0 even though it is not this
    # units list's own upper bound -- the function trusts the caller's (lo, hi), it does not
    # recompute them, which is documented and tested by the reduction test below.
    got = check_near_zero_bounds(units, lo=0.20, hi=0.005)
    assert got.checked is True
    assert got.lo_stable is None, "lo was not near zero; its stability must not be reported"
    assert got.hi_stable in (True, False)
    assert len(got.check_los) == 3 and len(got.check_his) == 3


def test_check_near_zero_bounds_reruns_are_cluster_bootstrap_on_the_same_units():
    """Not a black box: the reruns must be exactly what calling `cluster_bootstrap` on these
    same `units` at the stated seeds and `check_n_boot` produces -- otherwise "50k x3 seeds"
    in the table would be a caption, not a description of what ran."""
    from pi_eval.stats.inference import cluster_bootstrap

    rng = random.Random(4)
    units = [[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(30)]
    point, lo, hi = cluster_bootstrap(units, n_boot=2000, seed=0)
    got = check_near_zero_bounds(units, lo, hi, tol=1.0, check_n_boot=777, check_seeds=(11, 22, 33))
    assert got.checked is True  # tol=1.0 forces the trigger regardless of lo/hi
    for s, want_lo, want_hi in zip((11, 22, 33), got.check_los, got.check_his):
        _p, exp_lo, exp_hi = cluster_bootstrap(units, n_boot=777, seed=s)
        assert want_lo == pytest.approx(exp_lo)
        assert want_hi == pytest.approx(exp_hi)


def test_check_near_zero_bounds_does_not_depend_on_units_order():
    """Inherits order-invariance from `cluster_bootstrap` (commit `4b7b24b`); checked directly
    here anyway, since this is new code sitting on top of that primitive."""
    rng = random.Random(6)
    units = [[rng.gauss(0.0, 0.5) for _ in range(2)] for _ in range(25)]
    base = check_near_zero_bounds(units, lo=0.001, hi=0.5, check_n_boot=300)
    shuffled = list(units)
    random.Random(1).shuffle(shuffled)
    got = check_near_zero_bounds(shuffled, lo=0.001, hi=0.5, check_n_boot=300)
    assert got.check_los == base.check_los
    assert got.check_his == base.check_his


# ------------------------------------------------------------------------------- verdict_for


def test_verdict_excludes_zero_above_when_not_checked():
    assert verdict_for(0.02, 0.08, RobustBound(False, None, None, (), 50_000, (), ())) == (
        "excludes_zero_above"
    )


def test_verdict_excludes_zero_below_when_not_checked():
    assert verdict_for(-0.08, -0.02, RobustBound(False, None, None, (), 50_000, (), ())) == (
        "excludes_zero_below"
    )


def test_verdict_covers_zero_when_not_checked():
    assert verdict_for(-0.02, 0.08, RobustBound(False, None, None, (), 50_000, (), ())) == (
        "covers_zero"
    )


def test_verdict_is_undecided_when_a_checked_bound_is_unstable():
    """THE RULE'S CENTRE: even though lo=+0.002 says 'excludes zero above' at face value, an
    unstable near-zero bound withdraws that claim rather than letting the primary sign stand."""
    unstable = RobustBound(True, False, None, (101, 202, 303), 50_000, (0.001, -0.0004, 0.0002), ())
    assert verdict_for(0.002, 0.09, unstable) == "undecided"


def test_verdict_stays_decisive_when_the_checked_bound_is_stable():
    stable = RobustBound(True, True, None, (101, 202, 303), 50_000, (0.003, 0.004, 0.002), ())
    assert verdict_for(0.002, 0.09, stable) == "excludes_zero_above"


def test_verdict_undecided_if_either_bound_is_unstable_even_when_the_other_is_not_checked():
    only_hi_checked_and_unstable = RobustBound(
        True, None, False, (101, 202, 303), 50_000, (), (0.001, -0.0003, 0.0001)
    )
    assert verdict_for(-0.30, 0.004, only_hi_checked_and_unstable) == "undecided"


def test_matched_calls_difference_flags_undecided_when_a_bound_is_unstable_in_practice():
    """End-to-end at the module boundary this repo actually calls: construct a units list
    engineered so its BCa lower bound sits within 0.01 of zero, and confirm the returned
    `MatchedDelta.verdict` is either a clean call or 'undecided' -- never silently wrong --
    by cross-checking `robust.lo_stable` against an independent direct rerun."""
    keys, clusters = _cluster_map(n_clusters=40, per_cluster=1)
    rng = random.Random(23)
    # centred barely above zero with real spread, so the primary lower bound has a good chance
    # of landing within 0.01 of zero for at least one of the two arms below.
    trained = {k: 0.015 + rng.gauss(0.0, 0.06) for k in keys}
    comp = {k: rng.gauss(0.0, 0.06) for k in keys}
    got = matched_calls_difference(
        trained, comp, comp, 0.0, mode="left_of_domain", comp_caps_used=(4,), clusters=clusters
    )
    assert got.verdict in (
        "undecided",
        "excludes_zero_above",
        "excludes_zero_below",
        "covers_zero",
    )
    if got.robust.checked:
        # the verdict must be internally consistent with the robustness record it carries
        if got.robust.lo_stable is False or got.robust.hi_stable is False:
            assert got.verdict == "undecided"
