"""Horizontal proactivity on a gold graph that has no facets, which is every tau2 graph.

`facet_breadth` is the repo's only horizontal metric and it is defined over `gold_facets`.
MEASURED: tau2_airline carries 0 facets on 43 of 43 graphs and tau2_retail on 112 of 112, so the
whole tau2 family scores `0 / 0` -- an undefined number that reads as "no breadth" rather than as
"not measured here". Vertical is fine on the same graphs (airline depths {0:204, 1:129, 2:29},
retail {0:592, 1:278, 2:122, 3:23}), so without this the two axes are not comparable and a gain
cannot be attributed to either.

WHAT REPLACES A FACET. A facet is a hand-declared grouping of needs. The prerequisite DAG already
carries the same information structurally: two needs joined by a chain are one line of inquiry,
and two needs in different weakly-connected components are independent things the agent had to
think of separately. Breadth is therefore the number of COMPONENTS touched, and it needs no
annotation that tau2 does not have.

WHY NOT SIMPLY COUNT RESOLVED ROOTS. A policy that resolves six needs down one chain and a policy
that resolves six needs across six unrelated chains have identical root counts if the chains
happen to share roots, and identical node counts always. Components separate them, which is the
distinction the horizontal axis exists to make.

THE DENOMINATOR IS WHAT WAS AVAILABLE, not what was touched. A rate over touched components is
1.0 for a policy that touched one component out of nine, which is exactly backwards.
"""

from __future__ import annotations

import math

from pi_eval.metrics.structure import breadth_components


class _Node:
    def __init__(self, nid, depth=0, kind="kb"):
        self.gold_node_id = nid
        self.gold_depth = depth
        self.gold_discoverability = kind
        self.gold_facet_id = None


class _Edge:
    def __init__(self, src, dst, kind="prerequisite"):
        self.gold_src_node_id = src
        self.gold_dst_node_id = dst
        self.gold_edge_kind = kind


class _Graph:
    def __init__(self, nodes, edges):
        self.gold_nodes = nodes
        self.gold_edges = edges
        self.gold_facets = ()


class _Rec:
    def __init__(self, nid, rank=2, turn=0):
        self.node_id = nid
        self.rank = rank
        self.matched_turn_idx = turn


def _chain_graph():
    """Three independent lines of inquiry: a->b, c->d, and a lone e."""
    nodes = [_Node(x, depth=d) for x, d in (("a", 0), ("b", 1), ("c", 0), ("d", 1), ("e", 0))]
    return _Graph(nodes, [_Edge("a", "b"), _Edge("c", "d")])


def test_one_deep_chain_is_one_component() -> None:
    """Six needs down one line is depth, not breadth. This is the whole distinction."""
    touched, total, _s = breadth_components([_Rec("a"), _Rec("b")], _chain_graph())
    assert (touched, total) == (1, 3)


def test_two_unrelated_lines_are_two_components() -> None:
    touched, total, _s = breadth_components([_Rec("a"), _Rec("c")], _chain_graph())
    assert (touched, total) == (2, 3)


def test_an_isolated_node_is_its_own_component() -> None:
    touched, total, _s = breadth_components([_Rec("e")], _chain_graph())
    assert (touched, total) == (1, 3)


def test_the_denominator_is_what_existed_not_what_was_touched() -> None:
    """A rate over touched components scores 1.0 for a policy that found one of three."""
    touched, total, _s = breadth_components([_Rec("a")], _chain_graph())
    assert total == 3, "the denominator must not shrink to what the policy happened to reach"


def test_an_unresolved_node_does_not_count() -> None:
    """rank below the level asked for is a need the policy raised but did not resolve."""
    touched, _t, _s = breadth_components([_Rec("a", rank=0)], _chain_graph(), level="resolve")
    assert touched == 0


def test_a_graph_with_no_nodes_is_not_measured() -> None:
    """NaN, never 0: a suite with nothing to find must not report perfect narrowness."""
    touched, total, _s = breadth_components([], _Graph([], []))
    assert total == 0
    assert math.isnan(_rate(touched, total))


def _rate(touched: int, total: int) -> float:
    return touched / total if total else float("nan")


def test_non_prerequisite_edges_do_not_merge_components() -> None:
    """Only a prerequisite edge means "you had to know u to name v". Merging on any edge kind
    would silently make an annotation choice into a breadth claim."""
    g = _chain_graph()
    g.gold_edges = [*g.gold_edges, _Edge("a", "c", kind="supports")]
    touched, total, _s = breadth_components([_Rec("a"), _Rec("c")], g)
    assert (touched, total) == (2, 3)


def test_it_reports_how_fragmented_its_own_denominator_is() -> None:
    """A breadth number over mostly-singleton components describes the gold builder, not the
    policy. On tau2 that is 90% of airline components, so the share has to be visible."""
    _t, total, singles = breadth_components([_Rec("a")], _chain_graph())
    assert (total, singles) == (3, 1), "a->b and c->d are pairs; e is the lone one"


# ------------------------------------------------------- what the scorer must actually emit
#
# The three tests below are NOT ported from the branch that first wrote `breadth_components`.
# They exist because the branch's tests pin the FUNCTION and nothing pinned the ROW: a metric
# that is correct and never emitted is a column of NaN in every table.


def _two_chain_graph():
    """Two lines, two roots: a->b and c->d. Nothing else."""
    nodes = [_Node(x, depth=d) for x, d in (("a", 0), ("b", 1), ("c", 0), ("d", 1))]
    return _Graph(nodes, [_Edge("a", "b"), _Edge("c", "d")])


def test_resolving_both_roots_of_a_two_chain_graph_scores_two_components() -> None:
    """The reading the axis exists to license: two roots taken is two independent lines of
    inquiry, and the recall is 1.0 because there are exactly two to take."""
    touched, total, _s = breadth_components([_Rec("a"), _Rec("c")], _two_chain_graph())
    assert (touched, total) == (2, 2)
    assert _rate(touched, total) == 1.0


def test_six_down_one_chain_and_six_across_six_chains_differ() -> None:
    """THE WHOLE POINT, stated as the comparison that separates the axes. Both policies
    resolve six nodes; the node count cannot tell them apart and the component count can."""
    deep_nodes = [_Node(f"v{i}", depth=i) for i in range(6)]
    deep = _Graph(deep_nodes, [_Edge(f"v{i}", f"v{i + 1}") for i in range(5)])
    wide = _Graph([_Node(f"w{i}", depth=0) for i in range(6)], [])

    deep_touched, deep_total, _ = breadth_components([_Rec(f"v{i}") for i in range(6)], deep)
    wide_touched, wide_total, _ = breadth_components([_Rec(f"w{i}") for i in range(6)], wide)

    assert (deep_touched, deep_total) == (1, 1), "one chain is one line of inquiry"
    assert (wide_touched, wide_total) == (6, 6), "six unrelated needs are six"
    assert deep_touched != wide_touched, "six resolved nodes either way; only breadth separates"


def test_a_suite_with_facets_still_emits_facet_breadth_unchanged() -> None:
    """ADDITIVE, NOT A REPLACEMENT. `facet_breadth` is the horizontal metric on the suites
    that have facets; adding a facet-free one must not move it by a single row."""
    import math as _math

    from pi_eval.metrics.structure import facet_breadth

    nodes = [_Node(x, depth=d) for x, d in (("a", 0), ("b", 1), ("c", 0), ("d", 1))]
    for n, fid in zip(nodes, ("f1", "f1", "f2", "f2")):
        n.gold_facet_id = fid
    g = _Graph(nodes, [_Edge("a", "b"), _Edge("c", "d")])
    g.gold_facets = ("f1", "f2")

    touched, total = facet_breadth([_Rec("a"), _Rec("c")], g)
    assert (touched, total) == (2, 2)
    # and the two families are computed from different things, so they can disagree
    b_touched, b_total, _ = breadth_components([_Rec("a")], g)
    assert (b_touched, b_total) == (1, 2)
    assert not _math.isnan(_rate(b_touched, b_total))


# ------------------------------------------------------------- the row, over a REAL GoldGraph
#
# The helpers above are duck-typed stand-ins, which is right for pinning the arithmetic and
# wrong for pinning the emission: `score_run` reads `GoldGraph`, `GoldNode` and `MatchRecord`,
# and a metric that is correct on a stub and absent on the real type is not measured.


def _gold_graph(nodes, edges, facets=()):
    from pi_eval.gold import GoldEdge, GoldGraph, GoldNode

    gn = tuple(
        GoldNode(
            gold_suite="s",
            gold_task_key="t",
            gold_node_id=nid,
            gold_text=nid,
            gold_partition="required",
            gold_depth=depth,
            gold_facet_id=facet,
            gold_ev_uids=(f"s:{nid}:0-1",),
        )
        for nid, depth, facet in nodes
    )
    ge = tuple(
        GoldEdge(
            gold_suite="s",
            gold_task_key="t",
            gold_src_node_id=u,
            gold_dst_node_id=v,
            gold_edge_kind=kind,
        )
        for u, v, kind in edges
    )
    return GoldGraph(
        gold_suite="s", gold_task_key="t", gold_nodes=gn, gold_edges=ge, gold_facets=tuple(facets)
    )


def _gold_rec(nid):
    from pi_eval.matcher.base import MatchRecord

    return MatchRecord("r", "s", "t", nid, "resolve", 0, "m", "rule", 1.0, 1.0, "v1")


def _scored(graph, records):
    from pi_eval.score import score_run

    run = {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "a"}
    rows = score_run(
        run,
        graph=graph,
        turns=(),
        evidence=(),
        env_calls=(),
        ledger=(),
        records=records,
        answer=None,
    )
    return {r["metric_name"]: r for r in rows}


def test_the_scorer_emits_breadth_per_run() -> None:
    """A metric nothing emits is a column of NaN in every table. Two chains, one root taken."""
    g = _gold_graph(
        [("a", 0, None), ("b", 1, None), ("c", 0, None), ("d", 1, None)],
        [("a", "b", "prerequisite"), ("c", "d", "prerequisite")],
    )
    by = _scored(g, [_gold_rec("a")])
    assert by["breadth_components"]["value"] == 1.0
    assert by["breadth_components"]["n"] == 2
    assert by["breadth_recall"]["value"] == 0.5


def test_the_scorer_writes_no_breadth_row_for_a_graph_with_no_nodes() -> None:
    """NaN, never 0.0: an empty graph is not a policy that found nothing."""
    by = _scored(_gold_graph([], []), [])
    assert "breadth_components" not in by
    assert "breadth_recall" not in by


def test_facet_breadth_is_unchanged_by_the_addition() -> None:
    """The facet family keeps its rows, its value and its denominator on a faceted suite."""
    g = _gold_graph(
        [("a", 0, "f1"), ("b", 1, "f1"), ("c", 0, "f2"), ("d", 1, "f2")],
        [("a", "b", "prerequisite"), ("c", "d", "prerequisite")],
        facets=("f1", "f2"),
    )
    by = _scored(g, [_gold_rec("a"), _gold_rec("c")])
    assert by["facet_breadth"]["value"] == 2.0
    assert by["facet_breadth"]["n"] == 2
    assert by["facet_total"]["value"] == 2.0
