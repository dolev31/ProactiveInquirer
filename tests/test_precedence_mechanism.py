"""Lane L1.3: does the event-level reconstruction reproduce `precedence_violation_rate`, and
does the stratified interaction respond to which stratum is which.

Everything here is synthetic and hermetic (no store, no proxy, no PI_GOLD_ROOT) except
`test_lock_check_against_the_real_store`, which needs `PI_TESTSPLIT_STORE` and
`PI_TESTSPLIT_GOLD_ROOT` pointed at the isolated store this lane reads
(`artifacts/testsplit_qa/scores_parquet`, `data/gold`, both in the MAIN checkout: this
worktree does not carry them, and `conftest.py` scrubs `PI_GOLD_ROOT` from every test's
environment by design, so the path is passed through a differently-named var and applied
with `monkeypatch` -- "the only way it should ever arrive"). It skips, not fails, everywhere
else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.precedence_mechanism.events import (
    ARM_PROMPTED,
    ARM_TRAINED,
    NodeMatch,
    RunArm,
    events_for_arm,
    qualifying_edges_for_arm,
)
from scripts.precedence_mechanism.probe import (
    CanaryLeak,
    assert_canary_clean,
    closed_book_question,
    score_answer,
    to_question,
)
from scripts.precedence_mechanism.stratify import (
    interaction_bca,
    logistic_violation_model,
    stratified_contrast,
)

from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_eval.matcher.base import MatchRecord
from pi_eval.metrics.structure import precedence_violation_rate

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- fixtures


def _node(
    nid: str, text: str, *, aliases: tuple[str, ...] = (), depth: int | None = None
) -> GoldNode:
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t1",
        gold_node_id=nid,
        gold_text=text,
        gold_aliases=aliases,
        gold_partition="required",
        gold_depth=depth,
    )


def _edge(src: str, dst: str) -> GoldEdge:
    return GoldEdge(
        gold_suite="musique", gold_task_key="t1", gold_src_node_id=src, gold_dst_node_id=dst
    )


def _chain_graph() -> GoldGraph:
    """s1 -> s2 -> s3, a straight 3-node prerequisite chain."""
    nodes = (
        _node("s1", "The Collegian >> owned by", aliases=("Houston Baptist University",), depth=0),
        _node("s2", "When was #1 founded?", aliases=("1963",), depth=1),
        _node("s3", "Who is the president of #2?", aliases=("Some Person",), depth=2),
    )
    edges = (_edge("s1", "s2"), _edge("s2", "s3"))
    return GoldGraph(gold_suite="musique", gold_task_key="t1", gold_nodes=nodes, gold_edges=edges)


def _records(
    pairs: list[tuple[str, int]], *, run_id: str, graph_version: str = "v1"
) -> list[MatchRecord]:
    return [
        MatchRecord(
            run_id=run_id,
            suite_id="musique",
            task_id="t1",
            node_id=nid,
            match_kind="resolve",
            matched_turn_idx=t,
            matcher_id="mechanical_v3",
            matcher_family="mechanical",
            matcher_score=1.0,
            threshold=0.0,
            graph_version=graph_version,
        )
        for nid, t in pairs
    ]


# --------------------------------------------------------------------------- events.py


def test_events_and_qualifying_edges_reproduce_the_official_metric():
    """s2 (child) resolves at turn 0, its parent s1 at turn 2: a genuine violation. s3 (child
    of s2) resolves at turn 1, after s2: correctly ordered. `precedence_violation_rate` on
    this exact record set must read 1/2 = 0.5, and this module's edge-level reconstruction
    must sum to the same total and the same violating count."""
    graph = _chain_graph()
    full = [
        NodeMatch("s1", "resolve", 2),
        NodeMatch("s2", "resolve", 0),
        NodeMatch("s3", "resolve", 1),
    ]
    basis_k = 3  # the trained arm's own natural stop: never truncates its own full run

    official = precedence_violation_rate(
        _records([("s1", 2), ("s2", 0), ("s3", 1)], run_id="r1"), graph
    )
    assert official == pytest.approx(0.5)

    run = RunArm(run_id="r1", n_asks=3)
    edges = qualifying_edges_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=basis_k,
        full=full,
        graph=graph,
    )
    assert len(edges) == 2  # both prerequisite edges have both endpoints resolved
    viol = [e for e in edges if e.is_violation]
    assert [(e.parent_node_id, e.child_node_id) for e in viol] == [("s1", "s2")]
    assert sum(1 for e in edges if e.is_violation) / len(edges) == pytest.approx(official)

    events = events_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=basis_k,
        full=full,
        graph=graph,
        scores={"answer_correct": 1.0, "evidence_coverage": 0.5},
    )
    assert len(events) == 1
    e = events[0]
    assert (e.parent_node_id, e.child_node_id, e.child_turn) == ("s1", "s2", 0)
    assert e.parent_turn_full == 2
    assert e.parent_resolved_later is True
    assert e.parent_never_resolved is False
    assert e.counts_in_matched_metric is True  # parent IS inside the (untruncated) window
    assert e.answer_correct == 1.0
    assert e.evidence_coverage == 0.5


def test_parent_never_resolved_is_invisible_to_the_official_metric_but_not_to_this_table():
    """The child resolves; its prerequisite parent never appears in the run at all. The
    official metric's `total` never increments (one endpoint missing), so this edge
    contributes nothing to the published rate -- but it is exactly the "asks the leaf
    directly and never goes back" behaviour the mechanism hypothesis is about, so the
    broader event table must still surface it, tagged as never-resolved and NOT counted."""
    graph = _chain_graph()
    full = [NodeMatch("s2", "resolve", 0)]  # s1 (parent) absent entirely
    run = RunArm(run_id="r2", n_asks=1)

    official = precedence_violation_rate(_records([("s2", 0)], run_id="r2"), graph)
    assert official != official  # NaN: s1/s2 pair has a missing endpoint, s2/s3 pair too

    edges = qualifying_edges_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=1,
        full=full,
        graph=graph,
    )
    assert edges == []  # matches structure.py: total does not increment on a missing endpoint

    events = events_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=1,
        full=full,
        graph=graph,
        scores={},
    )
    assert len(events) == 1
    e = events[0]
    assert e.parent_never_resolved is True
    assert e.parent_resolved_later is False
    assert e.counts_in_matched_metric is False


def test_baseline_truncation_can_hide_a_recovery_beyond_the_matched_window():
    """The prompted (baseline) arm resolves the parent at turn 5 in its own, longer, natural
    trajectory, but the matched-cost basis truncates it to k=3 (the paired trained run's own
    n_asks). Within the window the parent looks unresolved -- same as "never" to the official
    metric -- but the full run shows it WAS eventually resolved, just past the point the
    matched-cost accounting charges for. `counts_in_matched_metric` must be False (truncated
    out) while `parent_resolved_later` must still be True (it is true of the real run)."""
    graph = _chain_graph()
    full = [NodeMatch("s1", "resolve", 5), NodeMatch("s2", "resolve", 0)]
    run = RunArm(run_id="r3", n_asks=10)
    basis_k = 3  # min(trained_k=3, this run's own n_asks=10)

    edges = qualifying_edges_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_PROMPTED,
        run=run,
        basis_k=basis_k,
        full=full,
        graph=graph,
    )
    assert edges == []  # s1 not in the truncated window: no "total" credit either way

    events = events_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_PROMPTED,
        run=run,
        basis_k=basis_k,
        full=full,
        graph=graph,
        scores={},
    )
    assert len(events) == 1
    e = events[0]
    assert e.counts_in_matched_metric is False
    assert e.parent_resolved_later is True
    assert e.parent_turn_full == 5


def test_a_correctly_ordered_edge_is_not_an_event():
    graph = _chain_graph()
    full = [NodeMatch("s1", "resolve", 0), NodeMatch("s2", "resolve", 1)]
    run = RunArm(run_id="r4", n_asks=2)
    events = events_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=2,
        full=full,
        graph=graph,
        scores={},
    )
    assert events == []


def test_a_tie_is_not_a_violation_matching_the_strict_less_than_in_structure_py():
    graph = _chain_graph()
    full = [NodeMatch("s1", "resolve", 1), NodeMatch("s2", "resolve", 1)]
    run = RunArm(run_id="r5", n_asks=2)
    edges = qualifying_edges_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=2,
        full=full,
        graph=graph,
    )
    assert len(edges) == 1
    assert edges[0].is_violation is False
    events = events_for_arm(
        suite="musique",
        task_id="t1",
        seed=0,
        arm=ARM_TRAINED,
        run=run,
        basis_k=2,
        full=full,
        graph=graph,
        scores={},
    )
    assert events == []


# --------------------------------------------------------------------------- probe.py


def test_closed_book_question_resolves_the_musique_placeholder_from_the_graph_itself():
    graph = _chain_graph()
    node = next(n for n in graph.gold_nodes if n.gold_node_id == "s2")
    assert closed_book_question(graph, node) == "When was Houston Baptist University founded?"


def test_closed_book_question_turns_a_relation_triple_into_a_question():
    graph = _chain_graph()
    node = next(n for n in graph.gold_nodes if n.gold_node_id == "s1")
    assert closed_book_question(graph, node) == "What is The Collegian's owned by?"


def test_closed_book_question_is_none_on_an_unresolved_placeholder():
    nodes = (_node("s1", "When was #9 founded?", aliases=("1963",)),)  # #9 names nothing
    graph = GoldGraph(gold_suite="musique", gold_task_key="t1", gold_nodes=nodes, gold_edges=())
    assert closed_book_question(graph, nodes[0]) is None


def test_to_question_passes_through_a_real_question_unchanged():
    assert to_question("When was #1 founded?") == "When was #1 founded?"


def test_score_answer_matches_the_gold_span_token_sequence():
    node = _node("s1", "x", aliases=("Houston Baptist University",))
    assert score_answer("It was Houston Baptist University, founded ...", node) == 1.0
    assert score_answer("I don't know", node) == 0.0


def test_canary_scan_refuses_a_leaked_nonce_and_passes_a_clean_string():
    canaries = frozenset({"PINQCANARY_TESTONLY0000000000"})
    assert_canary_clean("a perfectly ordinary question", canaries, where="test")
    with pytest.raises(CanaryLeak):
        assert_canary_clean("oops PINQCANARY_TESTONLY0000000000 leaked", canaries, where="test")


# --------------------------------------------------------------------------- stratify.py


def _synthetic_edges(hot_parent: str, cold_parent: str):
    """8 tasks per parent. `hot_parent`: trained violates 6/8, prompted violates 0/8 (a big,
    forced delta). `cold_parent`: both arms violate 2/8 (no delta). Two children per parent so
    a stratum is not carried by one single edge."""
    from scripts.precedence_mechanism.events import QualifyingEdge

    edges = []
    for i in range(8):
        task = f"hot{i}"
        for arm, viol in ((ARM_TRAINED, i < 6), (ARM_PROMPTED, False)):
            edges.append(
                QualifyingEdge("musique", task, 0, arm, f"{arm}-{task}", hot_parent, "child", viol)
            )
        for arm, viol in ((ARM_TRAINED, i < 6), (ARM_PROMPTED, False)):
            edges.append(
                QualifyingEdge("musique", task, 0, arm, f"{arm}-{task}", hot_parent, "child2", viol)
            )
    for i in range(8):
        task = f"cold{i}"
        for arm in (ARM_TRAINED, ARM_PROMPTED):
            edges.append(
                QualifyingEdge(
                    "musique", task, 0, arm, f"{arm}-{task}", cold_parent, "child", i < 2
                )
            )
    return edges


def _keys(parent: str, tasks: list[str]) -> frozenset[tuple[str, str]]:
    """`answerable` is keyed `(task_id, parent_node_id)`, never a bare node id -- MuSiQue ids
    like "s1" are local to a task and reused across all 200 (see `stratify.py`'s module
    docstring on the 3-bare-ids-vs-26-real-pairs measurement)."""
    return frozenset((t, parent) for t in tasks)


def test_interaction_responds_to_which_stratum_is_answerable_and_flips_on_relabel():
    """Non-vacuity per the repo rule: a statistic that cannot be shown to move on a forced,
    known change is not trustworthy on a real one. Label the hot parent answerable: the
    interaction (answerable delta - not-answerable delta) must be large and positive and
    exclude zero. Relabel the SAME data with the cold parent as answerable instead: the sign
    must flip. A hardcoded or order-blind implementation would not respond to the relabel."""
    edges = _synthetic_edges(hot_parent="pHot", cold_parent="pCold")
    hot_tasks = [f"hot{i}" for i in range(8)]
    cold_tasks = [f"cold{i}" for i in range(8)]

    hot_answerable = interaction_bca(edges, _keys("pHot", hot_tasks), n_boot=2000, seed=0)
    assert hot_answerable.ci_lo > 0, hot_answerable
    assert hot_answerable.point > 0.3

    cold_answerable = interaction_bca(edges, _keys("pCold", cold_tasks), n_boot=2000, seed=0)
    assert cold_answerable.ci_hi < 0, cold_answerable
    assert cold_answerable.point < -0.3


def test_stratified_contrast_matches_paired_difference_on_the_hot_stratum():
    edges = _synthetic_edges(hot_parent="pHot", cold_parent="pCold")
    hot_tasks = [f"hot{i}" for i in range(8)]
    strata = stratified_contrast(edges, _keys("pHot", hot_tasks))
    assert strata["answerable"].estimate.point == pytest.approx(0.75, abs=1e-9)  # 6/8 - 0/8
    assert strata["answerable"].n_edges_trained == 16  # 2 children x 8 tasks
    assert strata["not_answerable"].estimate.point == pytest.approx(0.0, abs=1e-9)


def test_stability_check_flags_a_near_zero_bound_and_passes_a_clear_one():
    from scripts.precedence_mechanism.stratify import check_stability

    # A bound far from zero: never triggers the expensive re-check.
    far = check_stability(lambda n, s: (0.5, 0.9), 0.5, 0.9)
    assert far.checked is False
    assert far.stable is None

    # A lower bound within 0.01 of zero, and every reseed agrees it stays non-negative: stable.
    agree = check_stability(lambda n, s: (0.008, 0.9), 0.005, 0.9)
    assert agree.checked is True
    assert agree.stable is True

    # A lower bound near zero whose reseeds disagree on the sign: unstable, i.e. undecided.
    calls = iter([(-0.02, 0.9), (0.01, 0.9), (-0.01, 0.9)])
    disagree = check_stability(lambda n, s: next(calls), 0.005, 0.9)
    assert disagree.checked is True
    assert disagree.stable is False


def test_logistic_model_cell_counts_are_exact_regardless_of_convergence():
    """Cell counts (n_edges, n_tasks per arm x answerable) are a plain tabulation and must be
    exactly right even on data where the fit itself cannot converge (a real risk with small,
    separated cells -- see the next test)."""
    edges = _synthetic_edges(hot_parent="pHot", cold_parent="pCold")
    hot_tasks = [f"hot{i}" for i in range(8)]
    result = logistic_violation_model(edges, _keys("pHot", hot_tasks))
    assert result.n_edges == 48
    assert result.n_tasks == 16
    assert result.cells["arm=trained/answerable=True"] == {"n_edges": 16, "n_tasks": 8}
    assert result.cells["arm=prompted/answerable=True"] == {"n_edges": 16, "n_tasks": 8}
    assert result.cells["arm=trained/answerable=False"] == {"n_edges": 8, "n_tasks": 8}
    assert result.cells["arm=prompted/answerable=False"] == {"n_edges": 8, "n_tasks": 8}


def test_logistic_model_reports_nonconvergence_rather_than_raising_or_lying():
    """The synthetic hot/cold fixture has a stratum where one arm never violates at all --
    quasi-complete separation, a real logistic-regression failure mode. The function must
    surface that honestly (`converged=False`, a note) rather than raising or printing a
    coefficient as if it meant something."""
    edges = _synthetic_edges(hot_parent="pHot", cold_parent="pCold")
    hot_tasks = [f"hot{i}" for i in range(8)]
    result = logistic_violation_model(edges, _keys("pHot", hot_tasks))
    assert result.converged is False
    assert result.note
    assert "arm:answerable" in result.coef  # still fit and reported, just flagged unreliable


def test_logistic_model_on_a_degenerate_outcome_does_not_raise():
    from scripts.precedence_mechanism.events import QualifyingEdge

    edges = [
        QualifyingEdge("musique", "t1", 0, ARM_TRAINED, "r1", "p1", "c1", False),
        QualifyingEdge("musique", "t2", 0, ARM_PROMPTED, "r2", "p1", "c1", False),
    ]
    result = logistic_violation_model(edges, frozenset())
    assert result.converged is False
    assert "degenerate" in result.note
    assert result.coef == {}
    assert result.n_edges == 2


# --------------------------------------------------------------------------- real-store lock


@pytest.mark.integration
def test_lock_check_against_the_real_store(monkeypatch):
    store = os.environ.get("PI_TESTSPLIT_STORE", "")
    gold_root = os.environ.get("PI_TESTSPLIT_GOLD_ROOT", "")
    if not store or not gold_root or not (Path(store) / "runs.parquet").exists():
        pytest.skip(
            "PI_TESTSPLIT_STORE / PI_TESTSPLIT_GOLD_ROOT not pointed at the isolated "
            "testsplit_qa store (only present in the main checkout, not every worktree)"
        )
    monkeypatch.setenv(
        "PI_GOLD_ROOT", gold_root
    )  # conftest scrubs it; this is the sanctioned way back
    import duckdb
    from scripts.precedence_mechanism.events import lock_check

    from pi_eval.gold import load_graphs

    graphs = load_graphs("musique", "v1")
    con = duckdb.connect()
    n_checked, n_mismatch, n_matched_events = lock_check(con, store, graphs, "musique")
    assert n_checked > 0
    assert n_mismatch == 0, f"{n_mismatch}/{n_checked} runs disagree with precedence_violation_rate"
