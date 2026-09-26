"""The two axes, separately falsifiable on ONE task.

No real gold suite in this programme supports this: none carries a facet denominator above one
and depth->=2 nodes in the same tasks (artifacts/axis_complementarity_20260919). The synthetic
suite does by construction, so it is the only place a horizontal claim and a vertical claim can
be shown to measure DIFFERENT things rather than asserted to.

The instrument is a 2x2 of degenerate policies. If shallow-wide and deep-narrow score the same
on both axes, the axes do not separate and no two-axis claim may be made on this suite -- which
is the falsifier this file exists to run.
"""

import pytest

from pi_eval.build.synth_build import build, load_graphs_for_test
from pi_eval.matcher.base import MechanicalMatcher
from pi_eval.metrics.structure import coverage_at_depth, facet_breadth, max_depth_reached
from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq_adapters.synth.suite import SynthSuite
from pinq_expt.fakes import (
    BreadthOnlyInquirer,
    ChainInquirer,
    DeepNarrowInquirer,
    EchoDrafter,
    FrozenAnswerer,
    ShallowWideInquirer,
)

FACETS, DEPTH = 3, 3


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("twoaxis")
    corpus, gold, _ = build(n_tasks=4, n_facets=FACETS, depth=DEPTH, seed=7, root=root)
    graphs = {g.gold_task_key: g for g in load_graphs_for_test(gold)}
    return SynthSuite(corpus.parent), graphs


def _match(suite, tid, pol, graphs):
    t = run_loop(
        view=suite.view(tid),
        inquirer=pol,
        retriever=suite.retriever(tid),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=64),
        max_turns=16,
        k=5,
        seed=0,
    )
    return MechanicalMatcher().match(graph=graphs[tid], trajectory=t, run_id="r")


def test_shallow_wide_gets_breadth_without_depth(env):
    """Every facet touched, but no node below depth 1. The horizontal corner."""
    suite, graphs = env
    recs = _match(suite, "s0", ShallowWideInquirer(), graphs)
    assert facet_breadth(recs, graphs["s0"]) == (FACETS, FACETS)
    assert max_depth_reached(recs, graphs["s0"]) == 1
    cad = coverage_at_depth(recs, graphs["s0"])
    assert cad[1][0] == 1.0
    assert all(cad[d][0] == 0.0 for d in range(2, DEPTH))


def test_deep_narrow_gets_depth_without_breadth(env):
    """One facet, all the way down. The vertical corner."""
    suite, graphs = env
    recs = _match(suite, "s0", DeepNarrowInquirer(), graphs)
    assert facet_breadth(recs, graphs["s0"]) == (1, FACETS)
    assert max_depth_reached(recs, graphs["s0"]) == DEPTH - 1
    cad = coverage_at_depth(recs, graphs["s0"])
    assert all(cad[d][0] == pytest.approx(1 / FACETS) for d in range(1, DEPTH))


def test_the_two_axes_are_not_the_same_measurement(env):
    """THE FALSIFIER. Each corner must beat the other on its own axis and lose on the other's.

    If this fails, facet_breadth and coverage_at_depth are reading one underlying quantity on
    this suite and the paper's two-axis framing has no instrument behind it here.
    """
    suite, graphs = env
    for tid in graphs:
        wide = _match(suite, tid, ShallowWideInquirer(), graphs)
        deep = _match(suite, tid, DeepNarrowInquirer(), graphs)
        g = graphs[tid]
        assert facet_breadth(wide, g)[0] > facet_breadth(deep, g)[0], f"{tid}: breadth"
        assert max_depth_reached(deep, g) > max_depth_reached(wide, g), f"{tid}: depth"


def test_both_corners_are_strictly_below_the_ceiling(env):
    """Neither degenerate policy may match the chain policy, or it is not degenerate."""
    suite, graphs = env
    g = graphs["s0"]
    chain = _match(suite, "s0", ChainInquirer(), g and graphs)
    assert facet_breadth(chain, g) == (FACETS, FACETS)
    assert max_depth_reached(chain, g) == DEPTH - 1
    for pol in (ShallowWideInquirer(), DeepNarrowInquirer(), BreadthOnlyInquirer()):
        recs = _match(suite, "s0", pol, graphs)
        assert (facet_breadth(recs, g)[0], max_depth_reached(recs, g)) != (FACETS, DEPTH - 1)
