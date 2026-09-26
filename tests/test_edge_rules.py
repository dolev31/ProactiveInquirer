"""scripts/plan_metrics_symmetric/edge_rules.py -- synthetic fixtures only, no gold/parquet.

Two things this file must prove before the real-data findings in
artifacts/symmetric_matched_cost_20260919/RESULT.md can be trusted: (1) the refactor from
ladder.asymmetric_contrast's fixed reduction to a swappable `aggregator` changed nothing for the
rule already in production (`pooled_per_seed_mean` must reproduce `asymmetric_contrast` exactly);
(2) the three named rules are not accidentally identical -- if they always agreed, "try every
plausible rule" would be a vacuous test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_metrics_symmetric.edge_rules import (  # noqa: E402
    AGGREGATORS,
    any_violation_indicator,
    asymmetric_contrast_rule,
    edge_counts,
    edge_histogram,
    per_task_values_rule,
    pooled_across_seeds,
    pooled_per_seed_mean,
)
from plan_metrics_symmetric.ladder import RunLadder, asymmetric_contrast  # noqa: E402

from pi_eval.matcher.base import MatchRecord  # noqa: E402


def _rec(node_id, turn):
    return MatchRecord(
        run_id="r",
        suite_id="s",
        task_id="t",
        node_id=node_id,
        match_kind="resolve",
        matched_turn_idx=turn,
        matcher_id="m",
        matcher_family="rule",
        matcher_score=1.0,
        threshold=1.0,
        graph_version="v1",
    )


class _Graph:
    def __init__(self, edges):
        self.gold_edges = edges


class _Edge:
    def __init__(self, kind, src, dst):
        self.gold_edge_kind = kind
        self.gold_src_node_id = src
        self.gold_dst_node_id = dst


def test_edge_counts_agrees_with_the_source_metric_on_every_non_empty_case():
    """Pins the duplicated numerator/denominator to precedence_violation_rate's own ratio, so
    the two cannot silently drift apart."""
    from pi_eval.metrics.structure import precedence_violation_rate

    graph = _Graph((_Edge("prerequisite", "u", "v"), _Edge("prerequisite", "x", "y")))
    records = [_rec("u", 0), _rec("v", 1), _rec("x", 3), _rec("y", 2)]  # u->v in order, x->y NOT
    viol, total = edge_counts(records, graph)
    assert (viol, total) == (1, 2)
    assert viol / total == pytest.approx(precedence_violation_rate(records, graph))


def test_edge_histogram_buckets_by_prerequisite_edge_count_not_by_depth():
    """A task can have several independent depth-1 edges; the histogram counts EDGES, not
    chain length -- the exact distinction a peer's correction turned on."""
    graphs = {
        "none": _Graph(()),
        "one": _Graph((_Edge("prerequisite", "a", "b"),)),
        "two_independent": _Graph(
            (_Edge("prerequisite", "a", "b"), _Edge("prerequisite", "c", "d"))
        ),
        "non_prerequisite_ignored": _Graph((_Edge("supports", "a", "b"),)),
    }
    assert edge_histogram(graphs) == {"0": 2, "1": 1, "2+": 1}


# Two checkpoint seeds sharing a task, with DIFFERENT numbers of qualifying edges: seed 1 sees
# 1 edge (0 violations), seed 2 sees 3 edges (all 3 violated). A baseline of one seed, no
# violations, sees all 3 edges every time it is asked (never truncated below its own reach here).
GRAPH = {
    "t": _Graph(
        (
            _Edge("prerequisite", "p", "q"),
            _Edge("prerequisite", "r", "s"),
            _Edge("prerequisite", "m", "n"),
        )
    )
}


def _ladder(run_id, records, n_asks):
    return RunLadder(
        run_id=run_id, task_id="t", suite_id="s", n_asks=n_asks, records=tuple(records)
    )


def test_the_three_rules_are_not_accidentally_identical():
    """If pooled_per_seed_mean, pooled_across_seeds and any_violation_indicator always agreed,
    trying all three on MuSiQue would prove nothing. The aggregator only acts on the BASELINE
    side of `_per_task_values` (the checkpoint side always contributes a single (viol, total)
    pair per seed, so any aggregator reduces it to that seed's own ratio) -- this fixture gives
    the baseline two seeds with different edge-counts so the three rules combine them
    differently: one checkpoint seed with a trivial 1-edge, 0-violation reading, against two
    baseline seeds sharing its task, one with 0/1 (no violation) and one with 3/3 (all violated).
    """
    ckpt = {"c1": _ladder("c1", [_rec("p", 0), _rec("q", 1)], n_asks=2)}  # 1 edge, in order
    base = {
        "b1": _ladder("b1", [_rec("p", 0), _rec("q", 1)], n_asks=2),  # 1 edge, in order: 0/1
        "b2": _ladder(  # all 3 edges present and violated (dst before src): 3/3
            "b2",
            [_rec("q", 0), _rec("p", 1), _rec("s", 0), _rec("r", 1), _rec("n", 0), _rec("m", 1)],
            n_asks=2,
        ),
    }

    _, b_seed_mean = per_task_values_rule(pooled_per_seed_mean, ckpt, base, GRAPH)
    _, b_pooled = per_task_values_rule(pooled_across_seeds, ckpt, base, GRAPH)
    _, b_indicator = per_task_values_rule(any_violation_indicator, ckpt, base, GRAPH)

    # seed-mean: mean(0/1, 3/3) = mean(0.0, 1.0) = 0.5
    assert b_seed_mean["t"] == pytest.approx(0.5)
    # pooled-across-seeds: (0+3)/(1+3) = 0.75 -- the busier seed (more qualifying edges)
    # dominates the ratio, unlike seed-mean which weights both seeds equally
    assert b_pooled["t"] == pytest.approx(0.75)
    # any-violation: seed b1 had zero violations (0.0), b2 had at least one (1.0) -> mean 0.5
    assert b_indicator["t"] == pytest.approx(0.5)
    # pooled-across-seeds must differ from the other two here, or the fixture is not doing its job
    assert b_pooled["t"] != pytest.approx(b_seed_mean["t"])


def test_rule_a_reproduces_the_production_ladder_exactly():
    """`pooled_per_seed_mean` through `asymmetric_contrast_rule` must equal
    `ladder.asymmetric_contrast` bit for bit -- the control that proves this module's refactor
    changed only which reduction runs, not the pairing or truncation."""
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 1)], n_asks=2)}
    base = {"b1": _ladder("b1", [_rec("v", 0), _rec("u", 1)], n_asks=5)}
    graph = {"t": _Graph((_Edge("prerequisite", "u", "v"),))}

    original = asymmetric_contrast("precedence_violation_rate", ckpt, base, graph, n_boot=10)
    viaRule = asymmetric_contrast_rule(
        AGGREGATORS["pooled_per_seed_mean"], ckpt, base, graph, n_boot=10
    )
    assert viaRule.point == pytest.approx(original.delta)
    assert viaRule.n == original.n
