"""scripts/plan_metrics_symmetric/ladder.py, on a synthetic fixture -- no gold, no parquet.

The failure this file exists to catch: a truncation or pairing bug that produces a plausible
number for the wrong reason, exactly the class of bug CONTRIBUTING.md's rules exist for. Every case
here is hand-computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_metrics_symmetric.ladder import (  # noqa: E402
    ReproductionFailed,
    RunLadder,
    asymmetric_contrast,
    reproduce_and_recompute,
    symmetric_contrast,
)

from pi_eval.matcher.base import MatchRecord  # noqa: E402


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


def test_run_ladder_at_truncates_by_matched_turn_idx_strictly_less_than_k():
    lad = RunLadder(
        run_id="r",
        task_id="t",
        suite_id="s",
        n_asks=3,
        records=(_rec("a", 0), _rec("b", 1), _rec("c", 2), _rec("never", None, kind="none")),
    )
    assert {r.node_id for r in lad.at(0)} == set()
    assert {r.node_id for r in lad.at(1)} == {"a"}
    assert {r.node_id for r in lad.at(2)} == {"a", "b"}
    assert {r.node_id for r in lad.at(3)} == {"a", "b", "c"}
    # A record that was never matched (match_kind="none", matched_turn_idx=None) never survives.
    assert "never" not in {r.node_id for r in lad.at(100)}


class _Graph:
    """Just enough of GoldGraph for precedence_violation_rate: gold_edges with
    gold_edge_kind/gold_src_node_id/gold_dst_node_id."""

    def __init__(self, edges):
        self.gold_edges = edges


class _Edge:
    def __init__(self, kind, src, dst):
        self.gold_edge_kind = kind
        self.gold_src_node_id = src
        self.gold_dst_node_id = dst


# One task, two seeds per arm. Checkpoint resolves both nodes in-order (u then v) at k=2 on
# both seeds -- zero violations, matched at its own full k=2. Baseline resolves v BEFORE u
# (out of order) but only reaches k=1 on one seed -- so at min(2, 1)=1, only u survives via
# neither node ... construct concretely below with n_asks controlling truncation directly.
GRAPH = {"t": _Graph((_Edge("prerequisite", "u", "v"),))}


def _ladder(run_id, records, n_asks):
    return RunLadder(
        run_id=run_id, task_id="t", suite_id="s", n_asks=n_asks, records=tuple(records)
    )


def test_asymmetric_contrast_matches_the_documented_rule_by_hand():
    """Checkpoint asks 2 (u@0, v@1: in order, 0 violations at its own k=2). Baseline asks 5
    (v@0, u@1: OUT of order -- but truncated to min(2, 5)=2 for a normal, safe pairing) still
    sees both nodes, 1 violation. delta = 0 - 1 = -1 (checkpoint strictly better ordering)."""
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 1)], n_asks=2)}
    base = {"b1": _ladder("b1", [_rec("v", 0), _rec("u", 1)], n_asks=5)}
    res = asymmetric_contrast("precedence_violation_rate", ckpt, base, GRAPH, n_boot=10)
    assert res.delta == pytest.approx(-1.0)
    assert res.n_unsafe == 0  # baseline (5) was NOT shorter than checkpoint's k (2)


def test_asymmetric_contrast_flags_unsafe_when_baseline_is_the_shorter_arm():
    """Checkpoint asks 5 (u@0, v@4: in order over a long run). Baseline asks only 1 (u@0 only,
    v never resolved) -- SHORTER than the checkpoint's k, so it is charged its own single
    question rather than truncated down. min(5, 1) = 1: baseline sees only u, no violation
    possible (den=0 -> nan), so this task contributes nothing to the baseline mean unless other
    seeds do; here it is the only baseline run, so the pair is dropped (nan), and n_unsafe still
    counts the fact that baseline < k."""
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 4)], n_asks=5)}
    base = {"b1": _ladder("b1", [_rec("u", 0)], n_asks=1)}
    res = asymmetric_contrast("precedence_violation_rate", ckpt, base, GRAPH, n_boot=10)
    assert res.n_unsafe == 1
    assert res.n == 0  # both sides had only NaN-contributing data for the one shared task


def test_symmetric_contrast_reads_both_arms_at_the_lower_count_and_is_antisymmetric():
    """Same fixture as the first asymmetric test, but the baseline's true out-of-order move
    only shows up once it is ALSO evaluated at a shared budget with the checkpoint. Here both
    have already resolved both nodes by k=2, so the symmetric reading equals the asymmetric one
    -- the safe-direction invariant this whole module exists to preserve for cells that were
    already matched correctly (mirrors src/pinq_train/gate.py's byte-identical guarantee)."""
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 1)], n_asks=2)}
    base = {"b1": _ladder("b1", [_rec("v", 0), _rec("u", 1)], n_asks=5)}
    sym = symmetric_contrast("precedence_violation_rate", ckpt, base, GRAPH, n_boot=10)
    assert sym.delta == pytest.approx(-1.0)

    swapped = symmetric_contrast("precedence_violation_rate", base, ckpt, GRAPH, n_boot=10)
    assert swapped.delta == pytest.approx(1.0)


def test_reproduce_and_recompute_raises_when_the_published_value_is_not_reproducible():
    """FAILING-FIRST in spirit: an unreproducible published number must surface as a named
    exception, not a silently-close approximation. Give a published_delta this fixture cannot
    possibly match."""
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 1)], n_asks=2)}
    base = {"b1": _ladder("b1", [_rec("v", 0), _rec("u", 1)], n_asks=5)}
    with pytest.raises(ReproductionFailed, match="no reproducible source"):
        reproduce_and_recompute(
            "precedence_violation_rate",
            ckpt,
            base,
            GRAPH,
            published_delta=999.0,
            n_boot=10,
        )


def test_reproduce_and_recompute_returns_both_readings_when_the_lock_passes():
    ckpt = {"c1": _ladder("c1", [_rec("u", 0), _rec("v", 1)], n_asks=2)}
    base = {"b1": _ladder("b1", [_rec("v", 0), _rec("u", 1)], n_asks=5)}
    asym, sym = reproduce_and_recompute(
        "precedence_violation_rate",
        ckpt,
        base,
        GRAPH,
        published_delta=-1.0,
        n_boot=10,
    )
    assert asym.delta == pytest.approx(-1.0)
    assert sym.delta == pytest.approx(-1.0)
