"""WHICH STATES ARE WORTH FORKING, and why "newly reachable" was the wrong answer.

The first branch pass targeted turns where `newly_reachable` was true -- a need whose last
prerequisite had just landed. That criterion STRUCTURALLY EXCLUDES TURN 0: a root need is on
the frontier from t=0, so nothing ever makes it "newly" appear, and `frontier(-1)` is empty by
definition. Measured on the built musique corpus, turn 0 is where the choice is widest -- 44%
of tasks offer more than one frontier need at t=0, against 27% at t=1 and 20% at t>=2 -- so
the pass targeted the second-best states and skipped the best ones.

A fork is only informative where a CHOICE EXISTS: with one need on the frontier, every
candidate is a paraphrase of the same question and the preference pair carries no signal about
WHICH need to pursue. So the criterion is `frontier_size >= 2`, and turn 0 is included.
"""

from __future__ import annotations

from pi_run.cmd_train import branchable_states


class _Node:
    def __init__(self, nid, depth, partition="required"):
        self.gold_node_id = nid
        self.gold_depth = depth
        self.gold_partition = partition


class _Edge:
    def __init__(self, src, dst):
        self.gold_src_node_id = src
        self.gold_dst_node_id = dst
        self.gold_edge_kind = "prerequisite"


class _Graph:
    """Two roots (a, b) and one child (c) gated behind a. Frontier at t=0 is {a, b}."""

    gold_nodes = [_Node("a", 0), _Node("b", 0), _Node("c", 1)]
    gold_edges = [_Edge("a", "c")]


class _Rec:
    def __init__(self, nid, turn, rank=2):
        self.node_id = nid
        self.matched_turn_idx = turn
        self.rank = rank


def test_turn_zero_is_offered_when_two_roots_compete() -> None:
    """The case the previous criterion could not see."""
    states = branchable_states(_Graph(), [_Rec("a", 0), _Rec("b", 1)], n_turns=3)
    assert 0 in {s.turn_idx for s in states}
    at0 = next(s for s in states if s.turn_idx == 0)
    assert at0.frontier_size == 2


def test_a_single_option_state_is_not_offered() -> None:
    """One need on the frontier means every candidate asks the same thing."""

    class _One:
        gold_nodes = [_Node("a", 0), _Node("c", 1)]
        gold_edges = [_Edge("a", "c")]

    states = branchable_states(_One(), [_Rec("a", 0)], n_turns=3)
    assert all(s.frontier_size >= 2 for s in states)
    assert 0 not in {s.turn_idx for s in states}


def test_states_are_ordered_earliest_first() -> None:
    """'Prioritising early': a fixed budget should buy the widest choices first, and an early
    fork also dominates a later one -- every later state is downstream of it."""
    states = branchable_states(_Graph(), [_Rec("a", 0), _Rec("b", 1)], n_turns=4)
    assert [s.turn_idx for s in states] == sorted(s.turn_idx for s in states)


def test_frontier_shrinks_as_needs_resolve() -> None:
    """After a resolves, the frontier is {b, c} -- still a choice; after both, only c."""
    states = branchable_states(_Graph(), [_Rec("a", 0), _Rec("b", 1), _Rec("c", 2)], n_turns=3)
    by_turn = {s.turn_idx: s.frontier_size for s in states}
    assert by_turn.get(0) == 2  # {a, b}
    assert by_turn.get(1) == 2  # {b, c}: a resolved, so c is now reachable
    assert 2 not in by_turn  # only {c} left -- no choice


def test_no_gold_depth_does_not_crash_the_selector() -> None:
    """drgym ships 0 edges and null depths; the selector must return nothing, not raise."""

    class _Flat:
        gold_nodes = [_Node("a", None), _Node("b", None)]
        gold_edges = []

    states = branchable_states(_Flat(), [], n_turns=2)
    assert isinstance(states, list)
