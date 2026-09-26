"""The loop's record of what an ask surfaced.

`Turn.retrieved_uids` recorded only the loop's own `retriever.search()` while state evidence
grew from the union with whatever `drafter.resolve()` retrieved.
`QueryExpansionDrafter._sub_retrieve` does exactly that, charges the ledger for it, and
`query_expansion` is a preregistered stage-1 arm -- so the two records of "what this run
retrieved" disagreed, and four consumers split across the gap:

  * `pi_eval.score` bases coverage and phi_LOO on `retrieved | evidence` (both paths) but
    accumulates its budget-frontier ladder from `retrieved_uids` (one path). Measured before
    the fix, on one run with one sub-retrieved gold unit: evidence_coverage 1.0 beside a
    terminal frontier_q of 0.5 -- the same run, the same definition of coverage, two numbers,
    and the understatement landing only on the arm whose mechanism is extra retrieval.
  * `Trajectory.prefix()` rebuilds evidence with `only(keep_uids)` from `retrieved_uids`, so
    `prefix(len(turns))` was not the identity and `phi_prefix_marginal` understated the true
    marginal by whatever resolve() had found.
  * `evidence_rows()` stamps `first_turn_idx` from `retrieved_uids`, so a sub-retrieved unit
    was written attributable to no turn at all on the spend axis.
  * `pinq_train.reward` and `pi_eval.metrics.qvalue` read it as ev(q_t) -- and the sub-queries
    were fired BECAUSE of that ask.
"""

import pytest

from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.types import Ask, Draft, EvidenceUnit, Stop
from pinq_expt.fakes import FrozenAnswerer


def _unit(uid, doc="d"):
    return EvidenceUnit(uid=uid, corpus_id="c", doc_id=doc, span="0-1", title=uid, text="x")


class _OneAsk:
    """Ask once, then stop. The narrowest policy that produces exactly one Turn."""

    policy_id = "one-ask"

    def reset(self, view, seed):
        self._done = False

    def act(self, s):
        if self._done:
            return Stop()
        self._done = True
        return Ask(text="q")


class _FixedRetriever:
    def __init__(self, units):
        self._units = tuple(units)

    def search(self, query, k):
        return self._units[:k]


class _SubRetrievingDrafter:
    """The shape of QueryExpansionDrafter: resolve() adds evidence the loop never searched for."""

    prompt_hash = "sub-retrieving-v1"

    def __init__(self, extra):
        self._extra = tuple(extra)

    def resolve(self, view, ask, ev, *, seed, ledger):
        return ("", ev.with_units(self._extra) if self._extra else ev)

    def draft(self, view, ev, *, seed, ledger):
        return Draft(text=f"d{len(ev.uids)}")


def _run(extra, loop_units=("loop1",), view=None, drafter=None):
    from pinq.types import TaskView

    v = view or TaskView(
        suite_id="s",
        task_id="t",
        question="q",
        instructions="",
        corpus_id="c",
        corpus_hash="ch",
        word_cap=50,
    )
    return run_loop(
        view=v,
        inquirer=_OneAsk(),
        retriever=_FixedRetriever([_unit(u) for u in loop_units]),
        drafter=drafter or _SubRetrievingDrafter(extra),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=32),
        max_turns=4,
        k=5,
        seed=0,
    )


def test_a_turn_records_evidence_its_drafter_retrieved():
    """The Turn is the objective record of what an ask surfaced. A sub-query fired because of
    that ask surfaced evidence for it, whichever component issued the search."""
    traj = _run([_unit("sub1", doc="d2")])
    t = traj.turns[0]
    assert "loop1" in t.retrieved_uids, "the loop's own retrieval must still be recorded"
    assert "sub1" in t.retrieved_uids, "the Drafter's sub-retrieval is evidence for this ask"
    assert set(t.retrieved_uids) == set(traj.evidence.uids), (
        "the Turn and the evidence set are two records of ONE fact and must not disagree"
    )
    assert "sub1" in t.new_uids
    assert len(t.retrieved_uids) == len(set(t.retrieved_uids)), "no duplicates"


def test_prefix_of_the_full_trajectory_is_the_identity():
    """phi_prefix_marginal and the stop test are undefined if prefix(len) drops evidence.
    tests/test_synth_closed_form.py asserts this invariant and passed throughout the defect,
    because EchoDrafter.resolve is a no-op and had nothing to drop."""
    traj = _run([_unit("sub1", doc="d2")])
    assert set(traj.prefix(len(traj.turns)).evidence.uids) == set(traj.evidence.uids)


def test_the_loop_ranking_comes_first():
    """drgym reads the sequence as the retriever's ranking; sub-retrievals extend it, never
    displace it."""
    traj = _run([_unit("sub1", doc="d2")], loop_units=("loop1", "loop2"))
    got = traj.turns[0].retrieved_uids
    assert got[:2] == ("loop1", "loop2") and got[-1] == "sub1"


def test_a_drafter_that_does_not_sub_retrieve_is_byte_identical():
    """Every Drafter but QueryExpansion has an empty _queries(), so no recorded run moves."""
    traj = _run([])
    assert traj.turns[0].retrieved_uids == ("loop1",)
    assert traj.turns[0].new_uids == ("loop1",)


def test_a_unit_both_paths_found_is_recorded_once():
    """Deduplicated, or `retrieval_volume` would double-count a unit the expansion re-found."""
    traj = _run([_unit("loop1")])
    assert traj.turns[0].retrieved_uids == ("loop1",)


def test_the_scorer_reports_ONE_coverage_for_a_run():
    """The published symptom: evidence_coverage and the terminal frontier point are the same
    quantity computed from the two records, and disagreed by 2x."""
    from pi_eval.gold import GoldGraph, GoldNode
    from pi_eval.score import score_run

    traj = _run([_unit("sub1", doc="d2")])
    nodes = tuple(
        GoldNode(
            gold_suite="s",
            gold_task_key="t",
            gold_node_id=n,
            gold_text=n,
            gold_partition="required",
            gold_discoverability="kb",
            gold_ev_uids=(u,),
        )
        for n, u in (("n1", "loop1"), ("n2", "sub1"))
    )
    graph = GoldGraph(gold_suite="s", gold_task_key="t", gold_nodes=nodes)
    turns = [
        {"turn_idx": t.turn_idx, "retrieved_uids": list(t.retrieved_uids), "action": "ask"}
        for t in traj.turns
    ]
    rows = score_run(
        {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "query_expansion"},
        graph=graph,
        turns=turns,
        evidence=[{"uid": u} for u in traj.evidence.uids],
        env_calls=[],
        ledger=[{"turn_idx": 0, "usd": 0.01, "kind": "llm"}],
        records=[],
        answer=None,
    )
    by = {r["metric_name"]: r["value"] for r in rows}
    qs = [v for k, v in by.items() if k.startswith("frontier_q#")]
    assert by["evidence_coverage"] == pytest.approx(1.0)
    assert max(qs) == pytest.approx(by["evidence_coverage"]), (
        "the frontier's terminal point IS the coverage this run achieved"
    )


class _EvidenceReportingDrafter:
    """`resolve` answers from exactly the evidence it is handed, and says what that was.

    The real Drafter renders `render_evidence(ev)` into its resolve prompt and asks a model.
    This one skips the model and reports the uids, which is the only part the loop controls.
    """

    prompt_hash = "evidence-reporting-v1"

    def resolve(self, view, ask, ev, *, seed, ledger):
        return (",".join(sorted(ev.uids)) or "(nothing)", ev)

    def draft(self, view, ev, *, seed, ledger):
        return Draft(text=f"d{len(ev.uids)}")


def test_an_ask_is_answered_from_the_evidence_that_ask_retrieved():
    """THE ANSWER TO A QUESTION MUST SEE THE DOCUMENTS FETCHED TO ANSWER IT.

    MEASURED, on 680 asks from 79 retail runs at 828a720: the loop retrieved for the ask and
    then called `resolve(view, action, state.evidence)` -- the evidence set from BEFORE those
    units merged. So every answer was computed one turn stale, and 55 asks that retrieved a
    unit were answered "No evidence has been retrieved ... so this cannot be answered". On
    turn 0, where nothing is held yet, 48% of asks were refused that way.

    That refusal is not confined to the run. `render_history` puts it in the Inquirer's next
    prompt as `A1: No evidence has been retrieved`, and `state_text` carries it into every SFT
    row built from the run -- so the training data teaches that a good question comes back
    empty, and the live policy re-asks for what it already holds.

    `components.py::_queries` states the intended contract in its own docstring -- "run_loop
    already retrieved for the ask itself, and re-issuing it would spend a second call on the
    same units" -- so the Drafter deliberately does not re-search. The loop then never handed
    those units over. The two halves have disagreed since d739909, the initial implementation,
    and no test named the seam.
    """
    traj = _run([], loop_units=("loop1", "loop2"), drafter=_EvidenceReportingDrafter())
    assert traj.turns[0].response_text == "loop1,loop2", (
        "resolve must see the units this ask retrieved, not the evidence held before it"
    )


def test_the_ask_that_retrieves_nothing_still_sees_what_was_already_held():
    """The fix must ADD the new units, never replace the history. A second ask that retrieves
    nothing new must still be answered from everything the run holds."""

    class _TwoAsks:
        policy_id = "two-asks"

        def reset(self, view, seed):
            self._n = 0

        def act(self, s):
            self._n += 1
            return Ask(text=f"q{self._n}") if self._n <= 2 else Stop()

    from pinq.types import TaskView

    traj = run_loop(
        view=TaskView(
            suite_id="s",
            task_id="t",
            question="q",
            instructions="",
            corpus_id="c",
            corpus_hash="ch",
            word_cap=50,
        ),
        inquirer=_TwoAsks(),
        retriever=_FixedRetriever([_unit("loop1")]),
        drafter=_EvidenceReportingDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=32),
        max_turns=4,
        k=5,
        seed=0,
    )
    assert traj.turns[1].response_text == "loop1", (
        "the second ask must still be answered from the evidence the first one surfaced"
    )
