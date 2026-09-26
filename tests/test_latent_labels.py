"""Which needs were LATENT, and did the policy find them.

The paper's claim is vertical proactivity: asking about a need that becomes visible only
after earlier evidence is read. Nothing measured it. The label is free and already on disk --
MEASURED over all 800 musique gold graphs, 1,464 of 2,660 required nodes (55%) are gated
behind a prerequisite edge, depths {0:1196, 1:800, 2:530, 3:134}. Those edges are
human-authored (`#N` in `question_decomposition`), `mechanical`, confidence 1.0.

Two readings, both emitted, because they answer different questions:

  is_latent        the need sits at gold_depth >= 1, i.e. behind a prerequisite. A property
                   of the GRAPH.
  newly_reachable  the need's last prerequisite was resolved on the PREVIOUS turn, so it
                   became askable exactly now. A property of the TRAJECTORY, and the
                   operational reading of "the policy asked about a need that had just
                   become visible".

`frontier_size` is the denominator that keeps the second honest: resolving a latent need when
one was available is a different event from resolving one when six were.

Judge-free and deterministic: depth comes from `gold_depth` (AND semantics, 1 + max(parents)),
reachability from prerequisite edges, and resolution from the matcher. Nothing here is behind
the sigma_J noise floor.
"""

from __future__ import annotations

from dataclasses import replace

from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_eval.matcher.base import MatchRecord
from pi_eval.metrics.latent import latent_labels


def _n(nid, depth, partition="required"):
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t0",
        gold_node_id=nid,
        gold_text=f"need {nid}",
        gold_partition=partition,
        gold_depth=depth,
    )


def _e(src, dst):
    return GoldEdge(
        gold_suite="musique",
        gold_task_key="t0",
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind="prerequisite",
    )


def _rec(nid, kind, turn):
    return MatchRecord(
        run_id="r0",
        suite_id="musique",
        task_id="t0",
        node_id=nid,
        match_kind=kind,
        matched_turn_idx=turn,
        matcher_id="mechanical_v3",
        matcher_family="rule",
        matcher_score=1.0,
        threshold=1.0,
        graph_version="v1",
    )


# a -> b -> c : a is askable at once, b only after a, c only after b.
CHAIN = GoldGraph(
    gold_suite="musique",
    gold_task_key="t0",
    gold_nodes=(_n("a", 0), _n("b", 1), _n("c", 2)),
    gold_edges=(_e("a", "b"), _e("b", "c")),
)


def test_a_depth_zero_need_is_not_latent() -> None:
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0)])
    assert lab.per_turn[0].latent_depth == 0
    assert lab.per_turn[0].is_latent is False


def test_a_gated_need_is_latent() -> None:
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0), _rec("b", "resolve", 1)])
    assert lab.per_turn[1].latent_depth == 1
    assert lab.per_turn[1].is_latent is True


def test_a_turn_that_resolved_nothing_has_no_depth() -> None:
    """-1, not 0: zero is a real depth, so the sentinel must sit outside the value range."""
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0)])
    assert lab.per_turn[3].latent_depth == -1
    assert lab.per_turn[3].is_latent is False


def test_newly_reachable_is_true_only_when_the_need_just_became_askable() -> None:
    """b becomes reachable the moment a resolves, and is taken on the very next turn."""
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0), _rec("b", "resolve", 1)])
    assert lab.per_turn[1].newly_reachable is True


def test_a_need_taken_later_is_still_latent_but_not_newly_reachable() -> None:
    """The distinction the two labels exist to draw.

    b was askable from turn 1 and the policy took it at turn 4. It is still a latent need --
    it sits behind a prerequisite -- but the policy did not act the moment it appeared.
    """
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0), _rec("b", "resolve", 4)])
    assert lab.per_turn[4].is_latent is True
    assert lab.per_turn[4].newly_reachable is False


def test_a_root_is_never_newly_reachable() -> None:
    """Depth-0 needs are askable from the start; nothing makes them 'appear'."""
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0)])
    assert lab.per_turn[0].newly_reachable is False


def test_the_frontier_is_what_was_askable_at_that_turn() -> None:
    """The denominator. At turn 0 only `a` is askable; after it resolves, only `b`."""
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0), _rec("b", "resolve", 1)])
    assert lab.per_turn[0].frontier_size == 1
    assert lab.per_turn[1].frontier_size == 1


def test_an_unresolved_prerequisite_keeps_its_child_off_the_frontier() -> None:
    """c is never askable if b was never resolved, however many turns pass."""
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0)])
    assert lab.per_turn[2].frontier_size == 1, "only b is askable; c is still gated"


def test_optional_nodes_are_not_in_the_frontier() -> None:
    """The denominator is the REQUIRED partition, as every other coverage number here is."""
    g = GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=(_n("a", 0), _n("opt", 0, partition="optional")),
    )
    assert latent_labels(g, []).per_turn[0].frontier_size == 1


# --------------------------------------------------------------- the run summary


def test_the_discovery_rate_is_resolved_over_available() -> None:
    """Of the latent needs that ever became askable, how many did the policy take?

    Resolving b UNLOCKS c, so c became askable too and the policy simply did not take it.
    That is a genuine miss and the rate is 0.5, not 1.0. This test first asserted 1.0 on the
    belief that "c never became reachable"; that belief was mine and it was wrong -- and the
    version that catches it is the better test, because a denominator that only ever contains
    needs the policy already took cannot produce a rate below 1.
    """
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0), _rec("b", "resolve", 1)])
    assert lab.n_latent_available == 2, "b became askable when a resolved, c when b did"
    assert lab.n_latent_resolved == 1
    assert lab.latent_discovery_rate == 0.5


def test_a_latent_need_that_never_became_askable_is_not_in_the_denominator() -> None:
    """Charging a policy for a need the task never made reachable is charging it for the task.

    c sits behind b. If b was never resolved, c was never askable, and counting it as a miss
    would make the rate a function of how far the policy got rather than of what it did with
    what it had.
    """
    lab = latent_labels(CHAIN, [_rec("a", "resolve", 0)])
    assert lab.n_latent_available == 1 and lab.n_latent_resolved == 0
    assert lab.latent_discovery_rate == 0.0


def test_a_task_with_no_latent_needs_yields_nan_not_zero() -> None:
    """Absent is not zero. A flat graph cannot exhibit vertical proactivity either way, and a
    0.0 would read as 'the policy found no latent needs' on a task that has none."""
    import math

    flat = GoldGraph(gold_suite="drgym", gold_task_key="t0", gold_nodes=(_n("a", 0), _n("b", 0)))
    assert math.isnan(latent_labels(flat, []).latent_discovery_rate)


# ------------------------------------------------------------ the optional link


# a -> opt -> b : the only route to required b runs THROUGH an optional node. Optional nodes
# carry no evidence -- wiki2_build and strategyqa_build stamp `required` iff `gold_ev_uids`
# is non-empty -- so `opt` can never be RESOLVED by the mechanical matcher, and a frontier
# that waits for it walls b off for the whole run.
THROUGH_OPTIONAL = GoldGraph(
    gold_suite="musique",
    gold_task_key="t0",
    gold_nodes=(_n("a", 0), _n("opt", 1, partition="optional"), _n("b", 2)),
    gold_edges=(_e("a", "opt"), _e("opt", "b")),
)


def test_a_need_behind_an_optional_link_is_askable_once_its_required_ancestor_resolves() -> None:
    """The policy took b the turn after a -- exactly the behaviour the thesis is about.

    Before the fix `parents["b"] == {"opt"}`, and `done` can only ever hold REQUIRED ids
    (`resolved_at` filters on the required map), so `parents["b"] <= done` was unsatisfiable:
    b never entered the frontier, was left out of `n_latent_available`, and could not be
    `newly_reachable`. The policy did what the thesis asks for and scored nothing for it.
    """
    lab = latent_labels(THROUGH_OPTIONAL, [_rec("a", "resolve", 0), _rec("b", "resolve", 1)])
    assert lab.per_turn[1].frontier_size == 1, "b, and only b, is askable once a is in"
    assert lab.n_latent_available == 1, "b became askable, so it is in the denominator"
    assert lab.per_turn[1].newly_reachable is True, "b appeared when a resolved; taken at once"


def test_a_need_behind_an_optional_link_is_not_askable_before_its_required_ancestor() -> None:
    """The link is transparent, not absent.

    Dropping the `opt -> b` edge outright would make b a root: askable at turn 0 before a
    is in, in the denominator of a run that never resolved a, and never `newly_reachable`
    because it was always there. The graph puts b two steps behind the task statement; the
    frontier has to agree with it.
    """
    lab = latent_labels(THROUGH_OPTIONAL, [])
    assert lab.per_turn[0].frontier_size == 1, "only a is nameable from the task statement"
    assert lab.n_latent_available == 0, "a never resolved, so b never became askable"


def test_an_optional_node_is_still_never_in_the_frontier() -> None:
    """Routing THROUGH optional nodes must not put them ON the frontier.

    opt is reachable once a resolves, and sits at depth 1, but it is not a required need:
    not askable, not in the denominator. `test_optional_nodes_are_not_in_the_frontier` pins
    this for an optional ROOT; this pins it for an optional node whose parent resolved.
    """
    g = GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=(_n("a", 0), _n("opt", 1, partition="optional")),
        gold_edges=(_e("a", "opt"),),
    )
    lab = latent_labels(g, [_rec("a", "resolve", 0)])
    assert lab.per_turn[1].frontier_size == 0, "a is in and opt is not a required need"
    assert lab.n_latent_available == 0


def test_no_gold_node_resolved_is_its_own_state_not_not_latent():
    """MEASURED DEFECT. `latent_depth = -1` means NO required node was resolved at this turn,
    and `is_latent = deepest > 0` folded that into False -- indistinguishable from a real
    depth-0 root. "No identified need" is not "the need was stated in the task", and 7,322 of
    13,539 SFT rows (54.1%) carry -1.

    Found by three A4 raters: on the depth--1 half they split 46/83, barely above chance,
    because they were reading a real question while the label had no opinion to give.
    """
    g = GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=[_n("root", 0), _n("deep", 1)],
        gold_edges=[_e("root", "deep")],
    )
    lab = latent_labels(g, [_rec("root", "resolve", 0)], turn_idxs=[0, 1])

    resolved = lab.per_turn[0]
    assert resolved.latent_depth == 0 and resolved.has_gold_node is True

    empty = lab.per_turn[1]
    assert empty.latent_depth == -1
    assert empty.is_latent is False
    assert empty.has_gold_node is False, (
        "a turn that resolved nothing must be distinguishable from a depth-0 root"
    )


def test_a_need_the_task_names_is_flagged_nameable_whatever_its_depth():
    """MEASURED. All 21 items where three A4 raters UNANIMOUSLY said `stated_in_task` while the
    label said latent sit at depth 3, and none at any other depth. musique's 4-hop questions
    refer to their intermediate entities descriptively -- "the city where The Killers formed" --
    so a deep node is nameable from x even though it sits late in the decomposition chain.

    `gold_depth >= 1` therefore does not mean "could not have been asked from the task", which
    is what the module docstring claimed. `nameable_from_task` measures the claim directly.
    """
    root, city = _n("root", 0), _n("city", 3)
    # The real shape: gold_text carries the entity's surface form, which musique's composed
    # question also carries because it refers to the intermediate descriptively.
    city = replace(city, gold_text="the city where The Killers formed")
    g = GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=[root, city],
        gold_edges=[_e("root", "city")],
    )
    task = "How many undergraduates attend the university in the city where The Killers formed?"
    lab = latent_labels(g, [_rec("city", "resolve", 0)], turn_idxs=[0], task_text=task)
    t0 = lab.per_turn[0]
    assert t0.latent_depth == 3 and t0.is_latent is True
    assert t0.nameable_from_task is True, "the task names this need; depth 3 does not change that"

    lab2 = latent_labels(
        g, [_rec("city", "resolve", 0)], turn_idxs=[0], task_text="Unrelated question."
    )
    assert lab2.per_turn[0].nameable_from_task is False


def test_nameability_is_absent_not_false_when_no_task_text_is_given():
    """A caller that passes no task cannot distinguish "not nameable" from "not measured", and
    defaulting to False would silently assert the stronger claim on every existing caller."""
    g = GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=[_n("root", 0)],
        gold_edges=[],
    )
    lab = latent_labels(g, [_rec("root", "resolve", 0)], turn_idxs=[0])
    assert lab.per_turn[0].nameable_from_task is None
