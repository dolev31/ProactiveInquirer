"""scripts/plan_metrics_symmetric/sweep.py, on synthetic fixtures -- no gold, no parquet.

WHY A SECOND TEST FILE. `tests/test_plan_metrics_symmetric.py` pins lane L2.3's `ladder`
module, which this one imports and must not change. What is tested HERE is only what `sweep`
adds on top: the five metric functions `ladder.METRIC_FNS` does not define, the synthetic
`MatchRecord` encoding that lets a uid-level metric (`evidence_coverage`) and a turn-count
metric (`stop_overshoot`) ride the SAME `matched_turn_idx < k` truncation as the record-level
ones, the runtime registration that adds them without editing `ladder`, and the
DISCRIMINATING probe.

THE PROBE IS THE POINT. A reproduction lock on a cell whose value could not have come out any
other way certifies nothing: a published delta of exactly 0.0 reproduces under any aggregation
rule at all. So every lock in this sweep is paired with a one-rung ladder shift, and a cell
whose delta does NOT move under that shift is reported as non-discriminating. The tests below
pin both directions of that probe -- it must fire on a metric that can vary, and it must
report "cannot fail" on one that cannot.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_metrics_symmetric import ladder  # noqa: E402
from plan_metrics_symmetric.sweep import (  # noqa: E402
    DIAGNOSTIC_METRIC_FNS,
    EXTRA_METRIC_FNS,
    LockRow,
    coverage_ladder_records,
    lock_cell,
    probe_verdict,
    published_matched_cells,
    registered_metrics,
    relative_to_root,
    render_table,
    seed_matched_contrast,
)

from pi_eval.gold import GoldEdge, GoldFacet, GoldGraph, GoldNode  # noqa: E402
from pi_eval.matcher.base import MatchRecord  # noqa: E402

# ---------------------------------------------------------------------------- fixtures


def _node(nid, *, depth=0, ev=(), facet=None, partition="required"):
    return GoldNode(
        gold_suite="s",
        gold_task_key="t",
        gold_node_id=nid,
        gold_text=nid,
        gold_partition=partition,
        gold_depth=depth,
        gold_ev_uids=tuple(ev),
        gold_facet_id=facet,
    )


def _edge(src, dst, kind="prerequisite"):
    return GoldEdge(
        gold_suite="s",
        gold_task_key="t",
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind=kind,
    )


def _rec(node_id, turn, kind="resolve"):
    return MatchRecord(
        run_id="r",
        suite_id="s",
        task_id="t",
        node_id=node_id,
        match_kind=kind,
        matched_turn_idx=turn,
        matcher_id="m",
        matcher_family="rule",
        matcher_score=1.0,
        threshold=1.0,
        graph_version="v1",
    )


# A two-node prerequisite chain, two gold spans, one facet.
CHAIN = GoldGraph(
    gold_suite="s",
    gold_task_key="t",
    gold_nodes=(_node("u", depth=0, ev=("e1",), facet="f1"), _node("v", depth=1, ev=("e2",))),
    gold_edges=(_edge("u", "v"),),
    gold_facets=(GoldFacet(gold_suite="s", gold_task_key="t", gold_facet_id="f1"),),
)

FLAT = GoldGraph(
    gold_suite="s",
    gold_task_key="t",
    gold_nodes=(_node("a", depth=0, ev=("e1",)), _node("b", depth=0, ev=("e2",))),
    gold_edges=(),
    gold_facets=(),
)


def _ladder(run_id, records, n_asks, task="t"):
    return ladder.RunLadder(
        run_id=run_id, task_id=task, suite_id="s", n_asks=n_asks, records=tuple(records)
    )


# ---------------------------------------------------------------- the synthetic encoding


def test_coverage_records_reproduce_the_frontier_q_ladder_by_hand():
    """`evidence_coverage` is a uid-level metric, not a MatchRecord one, so it cannot ride
    `matches.parquet`. It is encoded here as one synthetic record per GOLD uid, stamped with
    the turn that uid was first retrieved -- so `RunLadder.at(k)` truncates it by exactly the
    same `matched_turn_idx < k` rule as every record-level metric.

    Turn 0 retrieves e1 (gold) and x9 (not gold); turn 1 retrieves e2. Gold is {e1, e2}, so
    the hand-computed ladder is Q(0)=0.0, Q(1)=0.5, Q(2)=1.0 -- exactly `frontier_q#k`.
    """
    recs = coverage_ladder_records(
        run_id="r", suite_id="s", task_id="t", turn_uids=[("x9", "e1"), ("e2",)], graph=CHAIN
    )
    lad = _ladder("r", recs, n_asks=2)
    fn = EXTRA_METRIC_FNS["evidence_coverage"]
    assert fn(lad.at(0), CHAIN) == pytest.approx(0.0)
    assert fn(lad.at(1), CHAIN) == pytest.approx(0.5)
    assert fn(lad.at(2), CHAIN) == pytest.approx(1.0)
    # A uid retrieved twice is counted once, and a NON-gold uid never enters the numerator.
    assert fn(lad.at(99), CHAIN) == pytest.approx(1.0)


def test_coverage_is_nan_not_zero_when_the_task_has_no_required_gold_evidence():
    """`pi_eval.metrics.discovery.evidence_coverage` returns NaN on an empty gold set and the
    scorer emits NO ROW. A 0.0 here would read as "the policy retrieved none of it" on a task
    that never had any, which is the absent-is-not-zero rule this repository states twice."""
    bare = GoldGraph(gold_suite="s", gold_task_key="t", gold_nodes=(_node("u", ev=()),))
    recs = coverage_ladder_records(
        run_id="r", suite_id="s", task_id="t", turn_uids=[("z",)], graph=bare
    )
    assert math.isnan(EXTRA_METRIC_FNS["evidence_coverage"](list(recs), bare))


def test_stop_overshoot_reads_k_hat_off_the_truncated_prefix_not_the_whole_run():
    """`stop_overshoot = max(0, k_hat - k*)` with `k_hat = len(turns)`. Under truncation at k
    the run HAS only k turns, so `k_hat` must become `min(k, n_turns)` -- otherwise the matched
    reading charges the comparator for turns the truncation removed.

    Hand case: 4 turns, gold {e1}, retrieved on turn 0. Untruncated Q = [0, 1, 1, 1, 1], so
    k* = 1 and k_hat = 4 -> overshoot 3. Truncated at k=2: Q = [0, 1, 1], k* = 1, k_hat = 2 ->
    overshoot 1. Truncated at k=1: k* = 1, k_hat = 1 -> 0.
    """
    one_span = GoldGraph(gold_suite="s", gold_task_key="t", gold_nodes=(_node("u", ev=("e1",)),))
    recs = coverage_ladder_records(
        run_id="r",
        suite_id="s",
        task_id="t",
        turn_uids=[("e1",), (), (), ()],
        graph=one_span,
    )
    lad = _ladder("r", recs, n_asks=4)
    fn = EXTRA_METRIC_FNS["stop_overshoot"]
    assert fn(lad.at(4), one_span) == pytest.approx(3.0)
    assert fn(lad.at(2), one_span) == pytest.approx(1.0)
    assert fn(lad.at(1), one_span) == pytest.approx(0.0)


def test_facet_total_is_a_property_of_the_graph_and_cannot_move_under_truncation():
    """The published cell is +0.0 on all three suites. This test states WHY that lock is
    uninformative rather than asserting the number: the value does not read `records` at all,
    so both arms return the same float at every k and the delta is 0.0 by construction."""
    fn = EXTRA_METRIC_FNS["facet_total"]
    recs = (_rec("u", 0), _rec("v", 1))
    assert fn(list(recs), CHAIN) == fn([], CHAIN) == 1.0
    assert fn([], FLAT) == 0.0


def test_latent_metrics_are_nan_on_a_graph_with_no_latent_needs():
    """A flat graph cannot exhibit vertical proactivity in either direction, so the rate is
    NaN and the scorer writes no row -- the paired population is selected on the GRAPH, which
    is why these metrics' published n is far below the suite's task count."""
    for name in ("latent_discovery_rate", "newly_reachable_share"):
        assert math.isnan(EXTRA_METRIC_FNS[name]([_rec("a", 0)], FLAT))


def test_latent_discovery_rate_counts_only_latent_needs_that_became_reachable():
    """u (depth 0) resolved at turn 0 makes v (depth 1) available; v resolved at turn 1. One
    latent need available, one resolved -> 1.0. Truncated to k=1 only u survives, so v is
    available but unresolved -> 0.0."""
    recs = (_rec("u", 0), _rec("v", 1))
    lad = _ladder("r", recs, n_asks=2)
    fn = EXTRA_METRIC_FNS["latent_discovery_rate"]
    assert fn(lad.at(2), CHAIN) == pytest.approx(1.0)
    assert fn(lad.at(1), CHAIN) == pytest.approx(0.0)


# ------------------------------------------------------------- registration without edits


def test_registered_metrics_adds_and_then_restores_ladders_table():
    """`ladder.METRIC_FNS` belongs to another lane. Registering into it at RUNTIME is how this
    sweep runs the SAME pairing and truncation code without editing that file; leaking an entry
    past the context would mutate a module this lane does not own."""
    before = dict(ladder.METRIC_FNS)
    with registered_metrics(EXTRA_METRIC_FNS):
        assert "evidence_coverage" in ladder.METRIC_FNS
        assert "precedence_violation_rate" in ladder.METRIC_FNS  # ladder's own, untouched
    assert ladder.METRIC_FNS == before
    assert "evidence_coverage" not in ladder.METRIC_FNS


def test_registered_metrics_refuses_to_shadow_a_name_ladder_already_defines():
    """Silently overriding `ladder`'s own `dwr` would make this sweep report a lock on code
    that is not the code under lock."""
    with pytest.raises(ValueError, match="already defines"):
        with registered_metrics({"dwr": lambda records, graph: 0.0}):
            pass


# ------------------------------------------------------------------ the published parser


_FAKE_PUBLISHED = [
    {
        "metric": "dwr",
        "suite": "musique",
        "basis": "matched",
        "delta_distinct": 39,
        "task": {"delta": 0.5, "lo": 0.1, "hi": 0.9, "n": 176},
    },
    {
        "metric": "dwr",
        "suite": "musique",
        "basis": "unmatched",
        "delta_distinct": 35,
        "task": {"delta": 0.25, "lo": 0.0, "hi": 0.5, "n": 176},
    },
]


def test_published_matched_cells_reads_the_matched_basis_and_the_task_clustering_only():
    """Two traps in one file: the `unmatched` row carries a different number under the same
    metric name, and each row ALSO carries a `template`-clustered delta. Reading either by
    accident reports a lock against the wrong published value."""
    cells = published_matched_cells(_FAKE_PUBLISHED)
    assert set(cells) == {("dwr", "musique")}
    assert cells[("dwr", "musique")].published_delta == 0.5
    assert cells[("dwr", "musique")].published_n == 176


# ------------------------------------------------------------------- verdict + the probe


def _cells(n_asks_ck=2, n_asks_ba=2):
    ckpt = {"c": _ladder("c", (_rec("u", 0), _rec("v", 1)), n_asks=n_asks_ck)}
    base = {"b": _ladder("b", (_rec("v", 0), _rec("u", 1)), n_asks=n_asks_ba)}
    return ckpt, base


def test_lock_cell_reports_locked_only_when_the_published_value_is_reproduced():
    ckpt, base = _cells()
    ok = lock_cell(
        "precedence_violation_rate",
        ckpt,
        base,
        {"t": CHAIN},
        published_delta=-1.0,
        published_n=1,
        n_boot=10,
    )
    assert isinstance(ok, LockRow)
    assert ok.verdict == "LOCKED"
    assert ok.abs_diff == pytest.approx(0.0)

    bad = lock_cell(
        "precedence_violation_rate",
        ckpt,
        base,
        {"t": CHAIN},
        published_delta=0.25,
        published_n=1,
        n_boot=10,
    )
    assert bad.verdict == "FAILED"
    assert bad.abs_diff == pytest.approx(1.25)


# Two prerequisite edges, so a one-rung shift can drop ONE of them and leave the metric
# defined -- which is what lets the probe move NUMERICALLY rather than into NaN.
TWO_EDGE = GoldGraph(
    gold_suite="s",
    gold_task_key="t",
    gold_nodes=tuple(_node(n, depth=d) for n, d in (("u", 0), ("v", 1), ("w", 0), ("x", 1))),
    gold_edges=(_edge("u", "v"), _edge("w", "x")),
)


def test_the_discriminating_probe_fires_when_a_one_rung_shift_moves_the_delta():
    """NON-VACUITY, the other half of every lock here. The checkpoint takes u@0,v@1 in order
    and x@2,w@3 OUT of order: at its own k=4 that is 1 violation of 2 edges = 0.5, and the
    baseline (v@0,u@1 violating, w@0,x@1 clean) is also 0.5, so the published delta is 0.0.
    One rung short the checkpoint loses w entirely, the w->x edge leaves the denominator, and
    its rate falls to 0.0 -- delta -0.5. The lock could therefore have failed."""
    ckpt = {"c": _ladder("c", (_rec("u", 0), _rec("v", 1), _rec("x", 2), _rec("w", 3)), n_asks=4)}
    base = {"b": _ladder("b", (_rec("v", 0), _rec("u", 1), _rec("w", 0), _rec("x", 1)), n_asks=9)}
    row = lock_cell(
        "precedence_violation_rate",
        ckpt,
        base,
        {"t": TWO_EDGE},
        published_delta=0.0,
        published_n=1,
        n_boot=10,
    )
    assert row.verdict == "LOCKED"
    assert row.discriminating is True
    assert row.probe_delta == pytest.approx(-0.5)
    assert "different delta" in row.discriminating_reason


def test_a_probe_that_leaves_the_cell_undefined_also_counts_as_moved_and_says_so():
    """The checkpoint resolves v on its LAST turn, so one rung short the only prerequisite
    edge leaves the denominator and the cell is undefined. That IS a move -- the published
    value would not have reproduced -- but it is the population collapsing, not the value
    differing, and the reason string has to say which."""
    ckpt, base = _cells()
    row = lock_cell(
        "precedence_violation_rate",
        ckpt,
        base,
        {"t": CHAIN},
        published_delta=-1.0,
        published_n=1,
        n_boot=10,
    )
    assert row.discriminating is True
    assert row.probe_delta is None
    assert "undefined" in row.discriminating_reason


def test_the_probe_reports_non_discriminating_for_a_structurally_zero_metric():
    """`facet_total` reads no record, so a one-rung shift returns the identical delta. A LOCKED
    verdict on this cell certifies nothing and must be reported as such, not tallied."""
    ckpt, base = _cells()
    with registered_metrics(EXTRA_METRIC_FNS):
        row = lock_cell(
            "facet_total",
            ckpt,
            base,
            {"t": CHAIN},
            published_delta=0.0,
            published_n=1,
            n_boot=10,
        )
    assert row.verdict == "LOCKED"
    assert row.discriminating is False
    assert "shift" in row.discriminating_reason


def test_unavailable_is_a_verdict_and_carries_its_reason():
    row = lock_cell(
        "precedence_violation_rate",
        {},
        {},
        {},
        published_delta=None,
        published_n=None,
        n_boot=10,
    )
    assert row.verdict == "UNAVAILABLE"
    assert row.reason


# ------------------------------------------------------- the DOCUMENTED pairing, as a probe
#
# `artifacts/testsplit_plan_metrics_20260918/RESULT.md:176` states the pairing as "pair at
# (suite, task, seed), average seeds into the task". `ladder.asymmetric_contrast` instead
# folds seeds into one (suite, task) key and averages every checkpoint run against EVERY
# baseline run of that task. The two agree only when the truncation is a no-op. These tests
# pin the difference on a hand-computed fixture, so that a cell that fails the lock under one
# rule and passes under the other is diagnosed rather than reported as an unreproducible
# number.


def _seeded(run_id, records, n_asks):
    return _ladder(run_id, records, n_asks)


def test_seed_matched_contrast_pairs_seed_to_seed_not_across_seeds():
    """Checkpoint: seed 0 stops after 1 question (u only, max_depth 0); seed 1 runs to 3 (u
    and v, max_depth 1). Baseline: seed 0 has u@0 and v@2, seed 1 has only u.

    Seed-matched, as documented: seed 0 gives 0 - (baseline 0 truncated to k=1 -> 0) = 0;
    seed 1 gives 1 - (baseline 1 at k=3 -> 0) = 1. Task delta = 0.5.
    Cross-product, as `ladder` does it: checkpoint seed 0 is also charged against baseline
    seed 1 and vice versa, so the baseline's task value becomes 0.25 and the delta 0.25.
    """
    ckpt = {
        "c0": _seeded("c0", (_rec("u", 0),), 1),
        "c1": _seeded("c1", (_rec("u", 0), _rec("v", 2)), 3),
    }
    base = {
        "b0": _seeded("b0", (_rec("u", 0), _rec("v", 2)), 3),
        "b1": _seeded("b1", (_rec("u", 0),), 3),
    }
    seeds = {"c0": 0, "c1": 1, "b0": 0, "b1": 1}

    got = seed_matched_contrast("max_depth_reached", ckpt, base, {"t": CHAIN}, seeds, n_boot=10)
    assert got.delta == pytest.approx(0.5)
    assert got.n_pairs == got.n_pairs_shared == 2
    assert got.pairs_dropped == 0

    cross = ladder.asymmetric_contrast("max_depth_reached", ckpt, base, {"t": CHAIN}, n_boot=10)
    assert cross.delta == pytest.approx(0.25)


def test_seed_matched_contrast_drops_a_pair_where_either_arm_is_undefined_and_counts_it():
    """`pairs_dropped` is published per cell, so it is a second lock dimension: a pairing that
    reproduces the delta but not the drop count is reproducing by luck."""
    ckpt = {"c0": _seeded("c0", (_rec("u", 0), _rec("v", 1)), 2), "c1": _seeded("c1", (), 2)}
    base = {"b0": _seeded("b0", (_rec("v", 0), _rec("u", 1)), 5), "b1": _seeded("b1", (), 5)}
    seeds = {"c0": 0, "c1": 1, "b0": 0, "b1": 1}
    got = seed_matched_contrast(
        "precedence_violation_rate", ckpt, base, {"t": CHAIN}, seeds, n_boot=10
    )
    # `n_pairs` is pairs KEPT, matching what `contrasts.json` publishes under that name.
    assert got.n_pairs_shared == 2
    assert got.n_pairs == 1
    assert got.pairs_dropped == 1  # seed 1: neither arm resolved a prerequisite pair
    assert got.delta == pytest.approx(-1.0)


def test_seed_matched_contrast_refuses_two_runs_under_one_task_seed_key():
    """Two runs sharing (suite, task, seed) means the run-id lists are not what this pairing
    assumes, and silently keeping one of them would report a number over half the data."""
    ckpt = {"c0": _seeded("c0", (_rec("u", 0),), 1), "c0b": _seeded("c0b", (_rec("u", 0),), 1)}
    base = {"b0": _seeded("b0", (_rec("u", 0),), 1)}
    seeds = {"c0": 0, "c0b": 0, "b0": 0}
    with pytest.raises(ValueError, match="more than one run"):
        seed_matched_contrast("max_depth_reached", ckpt, base, {"t": CHAIN}, seeds, n_boot=10)


# ------------------------------------------------------- what `facet_breadth` actually is


def test_facet_breadth_scorer_is_a_count_and_is_zero_not_nan_where_there_are_no_facets():
    """`pi_eval.score` emits `float(touched)` -- a COUNT -- and emits it even when the graph
    has no facets at all, where `0/0` reads as "no breadth". `ladder.METRIC_FNS`'s own
    `facet_breadth` is `touched/total`, NaN on a facet-free graph: a different function under
    the same name. This pins the scorer's version, which is the one the published cell used.
    """
    fn = DIAGNOSTIC_METRIC_FNS["facet_breadth_scorer"]
    assert fn([_rec("u", 0)], CHAIN) == pytest.approx(1.0)  # u carries facet f1
    assert fn([], CHAIN) == pytest.approx(0.0)
    # FLAT has no facets at all: the scorer writes 0.0, not an absence.
    assert fn([_rec("a", 0)], FLAT) == pytest.approx(0.0)
    assert not math.isnan(fn([_rec("a", 0)], FLAT))


def test_probe_verdict_reports_the_three_outcomes_separately():
    """The helper both probes use. NaN is a move, a different number is a move, the same
    number is not -- and the caller has to be able to tell the first two apart."""
    moved, why = probe_verdict(float("nan"), 0.5, 1e-6)
    assert moved and "undefined" in why
    moved, why = probe_verdict(0.25, 0.5, 1e-6)
    assert moved and "different delta" in why
    moved, why = probe_verdict(0.5, 0.5, 1e-6)
    assert not moved and "SAME delta" in why


def test_lock_cell_probes_the_documented_pairing_under_the_documented_pairing():
    """A probe qualifies a verdict, so it has to be computed under the rule that produced that
    verdict. Reporting the cross-product probe beside a documented-pairing LOCKED verdict
    would attach a non-vacuity claim to a different computation."""
    ckpt = {
        "c0": _seeded("c0", (_rec("u", 0),), 1),
        "c1": _seeded("c1", (_rec("u", 0), _rec("v", 2)), 3),
    }
    base = {
        "b0": _seeded("b0", (_rec("u", 0), _rec("v", 2)), 3),
        "b1": _seeded("b1", (_rec("u", 0),), 3),
    }
    seeds = {"c0": 0, "c1": 1, "b0": 0, "b1": 1}
    row = lock_cell(
        "max_depth_reached",
        ckpt,
        base,
        {"t": CHAIN},
        published_delta=0.5,
        published_n=1,
        n_boot=10,
        seeds=seeds,
    )
    assert row.seedmatched_verdict == "LOCKED"
    assert row.seedmatched_discriminating is True
    # One rung short, checkpoint seed 1 loses v: its depth falls to 0 and the task delta to 0.
    assert row.seedmatched_probe_delta == pytest.approx(0.0)
    # The two rules are genuinely different computations on this fixture -- the UNSHIFTED
    # columns show it (0.5 documented against 0.25 cross-product), so the documented-pairing
    # probe is not a relabelling of the cross-product one. The two SHIFTED values coincide at
    # 0.0 here by arithmetic, which is why this asserts the rules differ rather than that the
    # probes do.
    assert row.seedmatched_delta == pytest.approx(0.5)
    assert row.repro_delta == pytest.approx(0.25)
    assert row.verdict == "FAILED"


def test_relative_to_root_keeps_an_outside_path_absolute_rather_than_mangling_it(tmp_path):
    """Recording `artifacts/x` for a file that is NOT under the repo would be a plausible-
    looking wrong provenance line, which is worse than an honest absolute path."""
    root = tmp_path / "repo"
    (root / "artifacts").mkdir(parents=True)
    inside = root / "artifacts" / "store"
    inside.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert relative_to_root(inside, root) == "artifacts/store"
    assert relative_to_root(outside, root) == str(outside.resolve())


def test_render_table_prints_the_json_values_and_never_retypes_them():
    """The published table and the machine-readable record must not be two sources of truth.
    The full float has to survive into the markdown, not a rounded copy of it."""
    payload = {
        "cells": {
            "dwr::musique::matched": {
                "metric": "dwr",
                "suite": "musique",
                "published_delta": 0.13764204545454548,
                "seedmatched_delta": 0.13764204545454548,
                "published_n": 176,
                "seedmatched_n": 176,
                "seedmatched_abs_diff": 0.0,
                "seedmatched_verdict": "LOCKED",
                "seedmatched_discriminating": True,
                "seedmatched_discriminating_reason": "a one-rung-short ladder gives a "
                "different delta, so it could have failed",
            }
        }
    }
    out = render_table(payload)
    assert "0.13764204545454548" in out
    assert "**LOCKED**" in out
    assert out.count("\n") == 2  # header, separator, one row
