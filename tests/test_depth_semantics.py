"""Depth semantics: AND (all prerequisites) vs OR (any route).

The divergence case is the whole point. A shortest-path BFS understates depth whenever a
node has several prerequisites at different depths, and understated depth drags genuinely
deep needs into shallow buckets — thinning exactly the strata the vertical-proactivity claim
is demonstrated on, and blurring the depth-1-restricted ablation.
"""

import pytest

from pi_eval.gold import GoldEdge, compute_depths, orphan_rate


def _e(src, dst, kind="prerequisite"):
    return GoldEdge(
        gold_suite="s",
        gold_task_key="t",
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind=kind,
        gold_verified="mechanical",
    )


def test_the_divergence_case():
    """v4 needs BOTH v2 (a seed) and v3 (depth 1).

    OR says 1 — it can be reached from v2 in one hop. AND says 2 — it genuinely cannot be
    resolved until v3 has been. AND is correct and is the default.
    """
    nodes = ["v1", "v2", "v3", "v4"]
    edges = [_e("v1", "v3"), _e("v2", "v4"), _e("v3", "v4")]
    seeds = ["v1", "v2"]

    and_d = compute_depths(nodes, edges, seeds)  # default
    or_d = compute_depths(nodes, edges, seeds, semantics="any")

    assert and_d == {"v1": 0, "v2": 0, "v3": 1, "v4": 2}
    assert or_d == {"v1": 0, "v2": 0, "v3": 1, "v4": 1}
    assert and_d["v4"] > or_d["v4"], "BFS understates depth here"


def test_and_is_the_default():
    nodes, edges, seeds = ["a", "b", "c"], [_e("a", "c"), _e("b", "c")], ["a"]
    # b is unreachable, so c cannot be resolved at all under AND semantics
    assert compute_depths(nodes, edges, seeds)["c"] is None
    # ...whereas OR happily reaches c through a
    assert compute_depths(nodes, edges, seeds, semantics="any")["c"] == 1


def test_a_simple_chain_agrees_under_both_semantics():
    """Where every node has one prerequisite, the two definitions must coincide — otherwise
    one of them is wrong on the easy case."""
    nodes = [f"v{i}" for i in range(5)]
    edges = [_e(f"v{i}", f"v{i + 1}") for i in range(4)]
    a = compute_depths(nodes, edges, ["v0"])
    o = compute_depths(nodes, edges, ["v0"], semantics="any")
    assert a == o == {f"v{i}": i for i in range(5)}


def test_diamond_takes_the_longer_arm():
    """a -> b -> d and a -> c1 -> c2 -> d. d is depth 3, not 2."""
    nodes = ["a", "b", "c1", "c2", "d"]
    edges = [_e("a", "b"), _e("b", "d"), _e("a", "c1"), _e("c1", "c2"), _e("c2", "d")]
    assert compute_depths(nodes, edges, ["a"])["d"] == 3
    assert compute_depths(nodes, edges, ["a"], semantics="any")["d"] == 2


def test_cycles_have_no_depth_rather_than_a_wrong_one():
    nodes = ["a", "b", "c"]
    edges = [_e("a", "b"), _e("b", "c"), _e("c", "b")]
    d = compute_depths(nodes, edges, ["a"])
    assert d["a"] == 0
    assert d["b"] is None and d["c"] is None
    assert orphan_rate(d) == pytest.approx(2 / 3)


def test_unreachable_nodes_are_none_not_zero():
    d = compute_depths(["a", "island"], [], ["a"])
    assert d["a"] == 0 and d["island"] is None
    assert orphan_rate(d) == 0.5


def test_a_seed_stays_at_zero_even_with_parents():
    """x is stated in the question AND happens to be derivable; being given beats deriving."""
    nodes, edges = ["a", "x"], [_e("a", "x")]
    assert compute_depths(nodes, edges, ["a", "x"])["x"] == 0


def test_relevance_edges_are_excluded_by_default():
    """prereq_only is the default basis: 'worth asking next' is not 'unanswerable until'."""
    nodes = ["a", "b"]
    edges = [_e("a", "b", kind="relevance")]
    assert compute_depths(nodes, edges, ["a"])["b"] is None
    assert compute_depths(nodes, edges, ["a"], basis="prereq_plus_relevance")["b"] == 1


def test_depth_is_invariant_to_node_and_edge_order():
    nodes = ["v1", "v2", "v3", "v4"]
    edges = [_e("v1", "v3"), _e("v2", "v4"), _e("v3", "v4")]
    a = compute_depths(nodes, edges, ["v1", "v2"])
    b = compute_depths(list(reversed(nodes)), list(reversed(edges)), ["v2", "v1"])
    assert a == b


def test_edges_to_unknown_nodes_are_ignored():
    """A dangling edge must not create a phantom node or crash the topological pass."""
    d = compute_depths(["a", "b"], [_e("a", "b"), _e("a", "ghost"), _e("ghost", "b")], ["a"])
    assert d == {"a": 0, "b": 1}
