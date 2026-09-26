"""The two-axis 2x2 on the synthetic suite, with COST, over a grid of (facets, depth).

Zero LLM calls, zero dollars, deterministic. The question is not whether the two axes separate
-- tests/test_two_axis_instrument.py settles that -- but whether they separate at MATCHED COST.
Shallow-wide pays 2*facets asks and deep-narrow pays depth, so at most shapes one corner is
simply cheaper, and a separation confounded with spend is not a separation.

Run: PYTHONPATH=$PWD/src .venv/bin/python scripts/two_axis_instrument/probe.py
"""

import tempfile
from pathlib import Path

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
    NeverAsk,
    ShallowWideInquirer,
)

POLICIES = [
    ("chain", ChainInquirer),
    ("shallow_wide", ShallowWideInquirer),
    ("deep_narrow", DeepNarrowInquirer),
    ("depth1_only", BreadthOnlyInquirer),
    ("never_ask", NeverAsk),
]


def run(suite, tid, pol, graphs):
    t = run_loop(
        view=suite.view(tid),
        inquirer=pol,
        retriever=suite.retriever(tid),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=256),
        max_turns=64,
        k=5,
        seed=0,
    )
    recs = MechanicalMatcher().match(graph=graphs[tid], trajectory=t, run_id="r")
    g = graphs[tid]
    cad = coverage_at_depth(recs, g)
    asks = (
        sum(1 for st in t.steps if getattr(st, "ask", None) is not None)
        if hasattr(t, "steps")
        else len([x for x in getattr(t, "turns", [])])
    )
    return dict(
        asks=asks,
        breadth=facet_breadth(recs, g)[0],
        nfacets=facet_breadth(recs, g)[1],
        maxdepth=max_depth_reached(recs, g),
        cad={d: round(v[0], 4) for d, v in cad.items()},
    )


for F, D in [(3, 3), (2, 4), (4, 8), (3, 6)]:
    with tempfile.TemporaryDirectory() as td:
        corpus, gold, _ = build(n_tasks=4, n_facets=F, depth=D, seed=7, root=Path(td))
        graphs = {g.gold_task_key: g for g in load_graphs_for_test(gold)}
        suite = SynthSuite(corpus.parent)
        print(
            f"\n{'=' * 78}\nfacets={F} depth={D}   (shallow-wide pays 2*{F}={2 * F}, deep-narrow pays {D})"
        )
        print(f"{'policy':<14}{'asks':>6}{'breadth':>10}{'max depth':>11}  per-depth coverage")
        rows = {}
        for name, cls in POLICIES:
            r = run(suite, "s0", cls(), graphs)
            rows[name] = r
            cad = " ".join(f"d{d}:{v:.2f}" for d, v in sorted(r["cad"].items()))
            print(
                f"{name:<14}{r['asks']:>6}{r['breadth']:>4}/{r['nfacets']:<5}{r['maxdepth']:>11}  {cad}"
            )
        w, dn = rows["shallow_wide"], rows["deep_narrow"]
        matched = w["asks"] == dn["asks"]
        print(
            f"  cost match: shallow_wide {w['asks']} asks vs deep_narrow {dn['asks']} asks -> "
            f"{'MATCHED' if matched else 'NOT matched'}"
        )
        if matched:
            print(
                f"  AT EQUAL COST: breadth {w['breadth']} vs {dn['breadth']}, "
                f"depth {w['maxdepth']} vs {dn['maxdepth']}  <-- the two axes trade off at fixed spend"
            )
