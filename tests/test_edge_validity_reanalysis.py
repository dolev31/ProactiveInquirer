"""scripts/edge_validity/{reanalysis,order_split}.py on hand-built fixtures: no gold root, no store.

The failures these tests exist to catch are the silent ones: a pruning that drops a node or
moves a facet, a depth recomputed by shortest path instead of longest, a class file that does
not cover the population and is read anyway, a null draw that ignores the child-depth strata,
a fast point estimate that drifts from the published reader's, and an out-of-order event
credited as TARGETED because `#N` was resolved to the wrong node. Every expected value here is
computed by hand in the comment beside it.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.edge_validity import order_split as osp  # noqa: E402
from scripts.edge_validity import reanalysis as ra  # noqa: E402
from scripts.precedence_mechanism.events import NodeMatch, RunArm  # noqa: E402

from pi_eval.gold import GoldEdge, GoldFacet, GoldGraph, GoldNode  # noqa: E402
from pi_eval.matcher.base import MatchRecord  # noqa: E402
from pi_eval.metrics.structure import precedence_violation_rate  # noqa: E402
from pi_eval.score import DWR_WEIGHTS  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402

# --------------------------------------------------------------------------- fixtures


def _node(nid, depth, *, text="", aliases=(), facet=None, task="t1", suite="musique"):
    return GoldNode(
        gold_suite=suite,
        gold_task_key=task,
        gold_node_id=nid,
        gold_text=text or f"need {nid}",
        gold_aliases=tuple(aliases),
        gold_depth=depth,
        gold_facet_id=facet,
        gold_ev_uids=(f"uid-{task}-{nid}",),
        gold_partition="required",
    )


def _edge(u, v, task="t1", suite="musique"):
    return GoldEdge(gold_suite=suite, gold_task_key=task, gold_src_node_id=u, gold_dst_node_id=v)


def _graph(nodes, edges, *, task="t1", suite="musique", facets=()):
    has_parent = {v for _, v in edges}
    return GoldGraph(
        gold_suite=suite,
        gold_task_key=task,
        gold_nodes=tuple(nodes),
        gold_edges=tuple(_edge(u, v, task, suite) for u, v in edges),
        gold_facets=tuple(facets),
        gold_seed_node_ids=tuple(n.gold_node_id for n in nodes if n.gold_node_id not in has_parent),
        gold_graph_version="v1",
    )


def _chain():
    """a -> b -> c, depths 0/1/2, one facet holding b and c."""
    facet = GoldFacet(
        gold_suite="musique", gold_task_key="t1", gold_facet_id="f0", gold_node_ids=("b", "c")
    )
    return _graph(
        [_node("a", 0), _node("b", 1, facet="f0"), _node("c", 2, facet="f0")],
        [("a", "b"), ("b", "c")],
        facets=(facet,),
    )


def _rec(node_id, turn, kind="resolve", task="t1", run="r"):
    return MatchRecord(
        run_id=run,
        suite_id="musique",
        task_id=task,
        node_id=node_id,
        match_kind=kind,
        matched_turn_idx=turn,
        matcher_id="mechanical_v3",
        matcher_family="rule",
        matcher_score=1.0,
        threshold=1.0,
        graph_version="v1",
    )


def _depths(g):
    return {n.gold_node_id: n.gold_depth for n in g.gold_nodes}


# --------------------------------------------------------------------------- pruning


def test_chain_with_the_middle_edge_removed_puts_the_child_at_depth_zero():
    g = ra.prune_graph(_chain(), {("b", "c")})
    # c lost its only parent: it is a root now, so depth 0 -- not None, not 2.
    assert _depths(g) == {"a": 0, "b": 1, "c": 0}
    assert set(g.gold_seed_node_ids) == {"a", "c"}
    assert [(e.gold_src_node_id, e.gold_dst_node_id) for e in g.gold_edges] == [("a", "b")]


def test_pruning_drops_no_node_and_holds_facets_fixed():
    before = _chain()
    after = ra.prune_graph(before, {("a", "b"), ("b", "c")})
    assert [n.gold_node_id for n in after.gold_nodes] == ["a", "b", "c"]
    assert after.gold_facets == before.gold_facets
    assert [n.gold_facet_id for n in after.gold_nodes] == [None, "f0", "f0"]
    assert _depths(after) == {"a": 0, "b": 0, "c": 0}
    # every field but depth is untouched on every node
    for x, y in zip(before.gold_nodes, after.gold_nodes, strict=True):
        assert (x.gold_text, x.gold_ev_uids, x.gold_partition) == (
            y.gold_text,
            y.gold_ev_uids,
            y.gold_partition,
        )


def test_a_diamond_keeps_the_longest_path():
    # a->b->c->d and a shortcut a->d: d is depth 3 (longest), never 1 (shortest).
    g = _graph(
        [_node("a", 0), _node("b", 1), _node("c", 2), _node("d", 3)],
        [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
    )
    assert _depths(ra.prune_graph(g, set())) == {"a": 0, "b": 1, "c": 2, "d": 3}
    # dropping the shortcut changes nothing; dropping c->d leaves only the shortcut: d = 1.
    assert _depths(ra.prune_graph(g, {("a", "d")}))["d"] == 3
    assert _depths(ra.prune_graph(g, {("c", "d")}))["d"] == 1


def test_pruning_an_edge_the_graph_does_not_have_refuses():
    with pytest.raises(ValueError, match="not in"):
        ra.prune_graph(_chain(), {("a", "c")})


def test_identity_pruning_reproduces_the_stored_depths_and_returns_equal_nodes():
    g = _chain()
    assert ra.prune_graph(g, set()).gold_nodes == g.gold_nodes


def test_pruned_dwr_and_max_depth_are_the_hand_computed_values():
    g = _chain()
    recs = [_rec("c", 0)]  # only c resolved
    # unpruned: C@0=0 (a), C@1=0 (b), C@2=1 (c) -> dwr = 1.5*1 / (0.5+1.0+1.5) = 0.5
    assert ra.metric_fn("dwr")(recs, g) == pytest.approx(0.5)
    assert ra.metric_fn("max_depth_reached")(recs, g) == 2.0
    p = ra.prune_graph(g, {("b", "c")})
    # pruned: depth 0 = {a, c} -> C@0 = 0.5; depth 1 = {b} -> 0; dwr = 0.5*0.5 / (0.5+1.0)
    assert ra.metric_fn("dwr")(recs, p) == pytest.approx(0.25 / 1.5)
    assert ra.metric_fn("max_depth_reached")(recs, p) == 0.0
    assert DWR_WEIGHTS[0] == 0.5 and DWR_WEIGHTS[1] == 1.0  # the arithmetic above assumes these


# --------------------------------------------------------------------------- class file


def _row(task, u, v, depth, cls, suite="musique"):
    return {"suite": suite, "task_id": task, "u": u, "v": v, "child_depth": depth, "class": cls}


def test_class_file_must_cover_the_population_exactly():
    graphs = {"t1": _chain()}
    good = [_row("t1", "a", "b", 1, "REFUTED"), _row("t1", "b", "c", 2, "VALIDATED")]
    idx = ra.index_classes(good, graphs, ["t1"], "musique")
    assert idx == {("t1", "a", "b"): "REFUTED", ("t1", "b", "c"): "VALIDATED"}
    with pytest.raises(ra.Refusal, match="missing"):
        ra.index_classes(good[:1], graphs, ["t1"], "musique")
    with pytest.raises(ra.Refusal, match="not a prerequisite edge"):
        ra.index_classes([*good, _row("t1", "a", "c", 2, "REFUTED")], graphs, ["t1"], "musique")
    with pytest.raises(ra.Refusal, match="duplicate"):
        ra.index_classes([*good, good[0]], graphs, ["t1"], "musique")
    with pytest.raises(ra.Refusal, match="child_depth"):
        ra.index_classes([_row("t1", "a", "b", 0, "REFUTED"), good[1]], graphs, ["t1"], "musique")
    with pytest.raises(ra.Refusal, match="class"):
        ra.index_classes([_row("t1", "a", "b", 1, "MAYBE"), good[1]], graphs, ["t1"], "musique")
    # a row of another suite is ignored, not counted as extra
    other = _row("t1", "a", "b", 1, "REFUTED", suite="wiki2")
    assert ra.index_classes([*good, other], graphs, ["t1"], "musique") == idx


def test_variants_remove_refuted_or_everything_not_validated():
    cls = {("t", "a", "b"): "REFUTED", ("t", "b", "c"): "VALIDATED", ("t", "a", "c"): "UNTESTABLE"}
    assert ra.removed_for_variant(cls, "primary") == {("t", "a", "b")}
    assert ra.removed_for_variant(cls, "strict") == {("t", "a", "b"), ("t", "a", "c")}
    with pytest.raises(ValueError):
        ra.removed_for_variant(cls, "loose")


# --------------------------------------------------------------------------- null


def test_null_draws_match_the_counts_within_each_child_depth_and_are_seeded():
    pool = {1: [("t", "a", f"b{i}") for i in range(10)], 2: [("t", "x", f"c{i}") for i in range(5)]}
    counts = {1: 3, 2: 2}
    d1 = ra.draw_matched_removal(pool, counts, random.Random(7))
    d2 = ra.draw_matched_removal(pool, counts, random.Random(7))
    assert d1 == d2
    assert sum(1 for e in d1 if e in pool[1]) == 3
    assert sum(1 for e in d1 if e in pool[2]) == 2
    rng = random.Random(7)
    draws = {frozenset(ra.draw_matched_removal(pool, counts, rng)) for _ in range(20)}
    assert len(draws) > 1  # successive draws from one stream differ
    with pytest.raises(ValueError):
        ra.draw_matched_removal(pool, {2: 6}, random.Random(0))


def test_null_seed_is_stable_per_variant_and_suite():
    assert ra.null_seed("primary", "musique") == ra.null_seed("primary", "musique")
    assert ra.null_seed("primary", "musique") != ra.null_seed("strict", "musique")
    assert ra.null_seed("primary", "musique") != ra.null_seed("primary", "wiki2")


def test_edges_by_child_depth_reads_the_unpruned_depth():
    graphs = {"t1": _chain()}
    by_d = ra.edges_by_child_depth([("t1", "a", "b"), ("t1", "b", "c")], graphs)
    assert by_d == {1: [("t1", "a", "b")], 2: [("t1", "b", "c")]}


# --------------------------------------------------------------------------- pooled point


class _Lad:
    """The two attributes of ladder.RunLadder that the pairing reads, plus `at`."""

    def __init__(self, task, n_asks, recs, suite="musique"):
        self.task_id, self.suite_id, self.n_asks, self.records = task, suite, n_asks, tuple(recs)

    def at(self, k):
        return [
            r for r in self.records if r.matched_turn_idx is not None and r.matched_turn_idx < k
        ]


def test_pairs_truncate_both_arms_at_the_smaller_count_and_pool_training_seeds_in_the_task():
    graphs = {"t1": _chain(), "t2": _chain()}
    seeds = {"a1": 0, "a2": 0, "b1": 0, "c1": 0, "a3": 0}
    s1 = {"a1": _Lad("t1", 3, [_rec("a", 0), _rec("b", 1), _rec("c", 2)])}
    s2 = {"a2": _Lad("t1", 1, [_rec("a", 0), _rec("b", 1)])}
    base = {"b1": _Lad("t1", 2, [_rec("a", 1)]), "c1": _Lad("t2", 2, [])}
    pairs = ra.build_pairs([s1, s2], base, graphs, seeds)
    # s1 x base at k=2: a, b survive; base's a at turn 1 survives. s2 x base at k=1: a only; base
    # has nothing before turn 1. t2 has no trained run and so no pair.
    assert [
        (p.task_id, sorted(r.node_id for r in p.a), sorted(r.node_id for r in p.b)) for p in pairs
    ] == [("t1", ["a", "b"], ["a"]), ("t1", ["a"], [])]
    per_a, per_b, n_pairs = ra.per_task(ra.metric_fn("max_depth_reached"), pairs, graphs)
    assert n_pairs == 2
    # max depth: s1 pair 1 vs 0; s2 pair 0 vs -1 -> task means 0.5 and -0.5
    assert per_a == {"t1": 0.5} and per_b == {"t1": -0.5}
    assert ra.point(per_a, per_b) == 1.0
    with pytest.raises(ValueError, match="two runs"):
        ra.build_pairs([{**s1, "a3": _Lad("t1", 2, [])}], base, graphs, seeds)


def test_fast_point_equals_paired_difference_point_bit_for_bit():
    rng = random.Random(3)
    per_a = {f"t{i:03d}": rng.random() for i in range(157)}
    per_b = {k: rng.random() for k in per_a}
    per_b["t005"] = math.nan  # dropped by both
    est = paired_difference(per_a, per_b, n_boot=200, seed=0)
    assert ra.point(per_a, per_b) == est.point


def test_depth_coverage_prints_the_cardinality_beside_the_recall():
    graphs = {"t1": _chain()}
    pairs = [ra.Pair("t1", (_rec("a", 0), _rec("c", 0)), (_rec("a", 0),))]
    cov = ra.depth_coverage(pairs, graphs)
    assert cov[0] == {"n_nodes": 1, "n_tasks": 1, "recipe": 1.0, "base": 1.0, "diff": 0.0}
    assert cov[2] == {"n_nodes": 1, "n_tasks": 1, "recipe": 1.0, "base": 0.0, "diff": 1.0}
    pruned = {"t1": ra.prune_graph(graphs["t1"], {("b", "c")})}
    cov_p = ra.depth_coverage(pairs, pruned)
    assert sorted(cov_p) == [0, 1]
    assert cov_p[0]["n_nodes"] == 2 and cov_p[0]["recipe"] == 1.0 and cov_p[0]["base"] == 0.5


def test_resolve_share_counts_only_resolved_children_of_the_class():
    fn = ra.resolve_share_fn({"t1": frozenset({"b", "c"})})
    g = _chain()
    assert fn([_rec("b", 0), _rec("c", 0, kind="ask")], g) == 0.5  # an ASK is not a resolve
    assert fn([_rec("b", 0, kind="use"), _rec("c", 1)], g) == 1.0
    assert math.isnan(fn([], _graph([_node("z", 0)], [], task="t9")))


# --------------------------------------------------------------------------- order split


def _musique_misordered_graph():
    """Node LIST order s2, s1, s3; s3 = 'When was #2 founded?' depends on s2 only.

    `matcher.base._resolve_refs` maps `#N` to the Nth node in its own topological order, which
    here is [s2, s1, s3], so it would print s1's answer ('Alphaville') for `#2`. The published
    rule, `resolve_placeholders`, maps `#2` to node s2's answer ('Beta College').
    """
    nodes = [
        _node("s2", 0, text="Gamma Weekly >> owned by", aliases=("Beta College",)),
        _node("s1", 0, text="Delta River >> mouth", aliases=("Alphaville",)),
        _node("s3", 1, text="When was #2 founded?", aliases=("1960",)),
    ]
    return _graph(nodes, [("s2", "s3")])


def test_targeted_resolves_placeholders_by_node_id_not_topological_position():
    g = _musique_misordered_graph()
    v = g.gold_nodes[2]
    assert osp.names_node(g, v, "musique", "When was Beta College founded?")
    assert not osp.names_node(g, v, "musique", "When was Alphaville founded?")
    # a question about the parent alone never names the child (the core term is required)
    assert not osp.names_node(g, v, "musique", "Who owns the Gamma Weekly paper?")


def test_arm_rates_split_the_metric_into_targeted_plus_incidental():
    g = _musique_misordered_graph()
    # child s3 retrieved at turn 0, parent s2 at turn 1: one violation over one qualifying edge.
    full = [NodeMatch("s3", "resolve", 0), NodeMatch("s2", "resolve", 1)]
    run = RunArm("r1", 2)
    q_target = {0: "When was Beta College founded?", 1: "Who owns Gamma Weekly?"}
    q_incid = {0: "What river flows past Alphaville?", 1: "Who owns Gamma Weekly?"}
    kw = dict(suite="musique", task_id="t1", seed=0, arm="x", run=run, full=full, graph=g)
    t = osp.arm_rates(**kw, basis_k=2, questions=q_target)
    i = osp.arm_rates(**kw, basis_k=2, questions=q_incid)
    assert (t.total, t.n_viol, t.n_targeted, t.n_incidental) == (1, 1, 1, 0)
    assert (i.total, i.n_viol, i.n_targeted, i.n_incidental) == (1, 1, 0, 1)
    # the unsplit rate is exactly the published metric on the same truncated records
    recs = [_rec(m.node_id, m.turn, m.match_kind) for m in full]
    assert t.rate("viol") == precedence_violation_rate(recs, g) == 1.0
    assert t.rate("targeted") + t.rate("incidental") == t.rate("viol")
    # at k=1 the parent is outside the window: no qualifying edge, every rate undefined
    w = osp.arm_rates(**kw, basis_k=1, questions=q_target)
    assert w.total == 0 and math.isnan(w.rate("viol")) and math.isnan(w.rate("targeted"))


def test_run_lock_row_matches_the_metric_on_the_full_run():
    g = _musique_misordered_graph()
    full = [NodeMatch("s3", "resolve", 0), NodeMatch("s2", "resolve", 1)]
    rebuilt, official = osp.run_lock_row(full=full, n_asks=2, graph=g, suite="musique")
    assert rebuilt == official == 1.0
    rebuilt, official = osp.run_lock_row(full=full[:1], n_asks=2, graph=g, suite="musique")
    assert math.isnan(rebuilt) and math.isnan(official)


def test_primary_instrument_needs_every_term_of_a_placeholder_free_node_cov50_does_not():
    # MEASURED property of `_matches`: with no `#N`, core == terms and ALL of them are required,
    # although its docstring says v2's >=50% rule is unchanged. The cov50 sensitivity applies the
    # documented rule (`asked_about`: >= 2 hits and >= 50% of the terms) to the resolved text.
    v = _node(
        "s2", 0, text="What term is used in Belgium and the Netherlands for a Fachhochschule?"
    )
    g = _graph([v], [])
    q = "What term is used in Belgium for a Fachhochschule?"  # 4 of 5 terms, NETHERLANDS absent
    assert not osp.names_node(g, v, "musique", q)
    assert osp.names_node_cov50(g, v, "musique", q)
    assert not osp.names_node_cov50(g, v, "musique", "What is a Fachhochschule?")  # 1 hit < 2


def test_cov50_resolves_placeholders_by_node_id_too():
    g = _musique_misordered_graph()
    v = g.gold_nodes[2]  # resolved: {BETA, COLLEGE, FOUNDED}
    assert osp.names_node_cov50(g, v, "musique", "Tell me about Beta College")  # 2 of 3
    assert not osp.names_node_cov50(g, v, "musique", "Tell me about Alphaville College")


def test_arm_rates_count_the_sensitivities_and_questions_naming_the_childs_answer():
    g = _musique_misordered_graph()
    full = [NodeMatch("s3", "resolve", 0), NodeMatch("s2", "resolve", 1)]
    kw = dict(suite="musique", task_id="t1", seed=0, arm="x", run=RunArm("r1", 2), full=full)
    # asks past the child: names the child's ANSWER (1960), not the child
    past = osp.arm_rates(**kw, graph=g, basis_k=2, questions={0: "What happened in 1960 there?"})
    assert (past.n_targeted, past.n_incidental, past.n_incidental_names_child_answer) == (0, 1, 1)
    loose = osp.arm_rates(**kw, graph=g, basis_k=2, questions={0: "Beta College history"})
    assert (loose.n_targeted, loose.n_targeted_cov50) == (0, 1)
    assert loose.rate("targeted_cov50") + loose.rate("incidental_cov50") == loose.rate("viol")
