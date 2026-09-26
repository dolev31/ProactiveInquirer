"""`reconcile`'s doc check counts two different things, and the run it kills is already paid for.

MEASURED, on this lane's own probe units. 15 of 42 died on `ReconcileError` and each left a run
directory containing `manifest.json` ALONE -- no turns, no evidence, no ledger, no status. The
dialogue had completed and been billed; the raise discards it whole. Over the airline arms that
is 5 of 20 units (25%), and over the stopped campaign's shard logs 106 of 625 attempts (17%).

THE MECHANISM. `BudgetLedger._docs` is one set, fed from two places that put different KINDS of
string into it:

    pinq.loop.run_loop     ledger.note_docs(u.doc_id for u in units)   -> DOC IDS
    meter_env_calls        ledger.note_docs(seen)                      -> UIDS

`reconcile` then asserts `ledger.unique_docs == len(traj.evidence.doc_ids)` -- the size of that
mixed set against a count of DOC IDS. A document reached BOTH ways therefore enters the ledger
twice, under two unequal strings, while the evidence carries one doc_id for it. The ledger
over-counts by exactly the size of the overlap and the check fires.

WHY IT LOOKED NON-DETERMINISTIC. Whether the overlap exists depends on whether the Drafter's
tool call happens to touch a record the Inquirer already retrieved, which is a property of the
dialogue rather than of the fork point -- airline task 30 k=13 failed under the base pin and
succeeded under both trained pins.

WHY THE ERROR MESSAGE IS THE WRONG DIAGNOSIS. It says "Unmetered nested retrieval", which is
`evidence > ledger`. This failure is the OPPOSITE direction, `ledger > evidence`, and the two
have opposite causes: one is a Drafter smuggling evidence past the meter, the other is the meter
counting one document twice. `tests/test_runtime.py` covers the first direction only, which is
why no test caught this one.

WHAT THIS FILE DOES NOT CLAIM. That every one of the 15 failures was this. The failing units
left no trajectory on disk, so the direct evidence was destroyed by the raise itself. What is
measured is that the mechanism exists and fires (below), and that all 26 probe units which
PASSED have a loop/env document overlap of exactly zero -- consistent with, and predicted by,
overlap being the discriminator.
"""

from __future__ import annotations

import pytest

from pi_run.stages.tau2_runner import meter_env_calls
from pi_run.worker import reconcile
from pinq.budget import BudgetLedger
from pinq.types import Evidence, EvidenceUnit, Outcome, Trajectory, Usage
from pinq.view import make_view

DOC = "orders:W001"
UID = "retail/orders/W001#0"


class _Index:
    """The two attributes `uids_for_call` and `_attach_env_evidence` read."""

    uids = {DOC: UID}
    titles: dict[str, str] = {}
    item_to_product: dict[str, str] = {}

    def uid(self, table: str, rid: str) -> str:  # pragma: no cover - not reached here
        return f"{table}/{rid}"


class _Suite:
    index = _Index()
    corpus_id = "tau2_retail_v1"

    def uids_for_calls(self, *a, **k):  # presence is what the guard checks
        return ()


def _loop_retrieved_unit() -> EvidenceUnit:
    """What the RETRIEVER yields: a real span and text, and a uid that is not the doc_id."""
    return EvidenceUnit.make(
        corpus_id="tau2_retail_v1", doc_id=DOC, span="0:12", title="order W001", text="..."
    )


def _traj(units) -> Trajectory:
    """A finished trajectory carrying exactly these evidence units. `reconcile` reads only
    `turns`, `terminal_usage` and `evidence`, and the first two are empty here so that the
    doc check is the only thing under test."""
    return Trajectory(
        view=make_view(
            task_id="t",
            suite_id="tau2_retail",
            question="q",
            instructions="",
            corpus_id="tau2_retail_v1",
            corpus_hash="ab" * 32,
            word_cap=180,
        ),
        turns=(),
        evidence=Evidence(units=tuple(units)),
        outcome=Outcome(),
        usage=Usage(),
        stop_reason="policy_stop",
    )


def test_a_uid_in_the_document_set_is_what_killed_those_runs():
    """THE HISTORICAL MECHANISM, reproduced by doing to the ledger exactly what the old line did.

    `meter_env_calls` used to end `ledger.note_docs(seen)` with `seen` a list of UIDS. This
    feeds one uid into the same set, which is that line's whole effect, and shows the
    consequence: one document, two strings, and a check that compares the set's size against a
    count of DOC IDS.

    Kept after the fix rather than deleted, for two reasons. It documents what the 15 lost units
    died of, and it proves `docs_ok` can still DETECT the condition -- a fix that made the
    detector blind would pass every test that only asserts the happy path.
    """
    ledger = BudgetLedger(cap=16)
    ledger.note_docs([DOC])  # the Inquirer's retrieval, metered by run_loop as a doc_id
    ledger.note_docs([UID])  # the old env-meter line, metered as a uid
    assert ledger.unique_docs == 2, "one document, two strings, because the namespaces differ"

    traj = _traj(
        [
            _loop_retrieved_unit(),
            EvidenceUnit(
                uid=UID, corpus_id="tau2_retail_v1", doc_id=DOC, span="", title="", text=""
            ),
        ]
    )
    assert len(traj.evidence.doc_ids) == 1, "both units name the same document"

    rec = reconcile(traj, ledger)
    assert rec["docs_ok"] is False, "this is the raise that deleted a paid-for run"
    assert rec["ledger_unique_docs"] == 2
    assert rec["evidence_unique_docs"] == 1
    # AND THE DIRECTION IS THE OPPOSITE OF WHAT THE MESSAGE SAYS.
    assert rec["ledger_unique_docs"] > rec["evidence_unique_docs"], (
        "the driver's message calls this 'Unmetered nested retrieval', which is the "
        "evidence > ledger direction. This was ledger > evidence and had the opposite cause."
    )


def test_the_live_path_no_longer_produces_that_state():
    """The same overlap, through the REAL `meter_env_calls`, must now reconcile."""
    ledger = BudgetLedger(cap=16)
    ledger.note_docs([DOC])
    uids, overrun = meter_env_calls(
        _Index(), ledger, [{"tool_name": "get_order_details", "kwargs_json": {"order_id": "W001"}}]
    )
    assert uids == (UID,), "the call must resolve to the record's uid, or this tests nothing"
    assert not overrun
    assert ledger.unique_docs == 1

    traj = _traj(
        [
            _loop_retrieved_unit(),
            EvidenceUnit(
                uid=UID, corpus_id="tau2_retail_v1", doc_id=DOC, span="", title="", text=""
            ),
        ]
    )
    rec = reconcile(traj, ledger)
    assert rec["docs_ok"] is True


def test_without_the_overlap_the_same_two_paths_agree():
    """NON-VACUITY. If the counts disagreed whenever both paths ran, the test above would be
    showing that the check is broken rather than that the OVERLAP breaks it -- and the fix
    would be aimed at the wrong thing. Two DIFFERENT documents must reconcile cleanly."""
    other_doc, other_uid = "orders:W002", "retail/orders/W002#0"

    class _Idx(_Index):
        uids = {other_doc: other_uid}

    ledger = BudgetLedger(cap=16)
    ledger.note_docs([DOC])
    uids, _ = meter_env_calls(
        _Idx(), ledger, [{"tool_name": "get_order_details", "kwargs_json": {"order_id": "W002"}}]
    )
    assert uids == (other_uid,)
    assert ledger.unique_docs == 2

    traj = _traj(
        [
            _loop_retrieved_unit(),
            EvidenceUnit(
                uid=other_uid,
                corpus_id="tau2_retail_v1",
                doc_id=other_doc,
                span="",
                title="",
                text="",
            ),
        ]
    )
    rec = reconcile(traj, ledger)
    assert rec["docs_ok"] is True, "disjoint documents must still reconcile"
    assert rec["ledger_unique_docs"] == rec["evidence_unique_docs"] == 2


def test_the_env_meter_counts_documents_and_not_uids():
    """THE FIX. `unique_docs` is a DOCUMENT counter; `meter_env_calls` must put a doc_id in it.

    Not merely "reconcile stops raising": with the raise removed but the count left wrong, the
    run survives and is then dropped anyway, because `pi_eval.report.ELIGIBLE` gates on
    reconciliation. Only counting the right thing restores the run to the reported population.
    """
    ledger = BudgetLedger(cap=16)
    ledger.note_docs([DOC])
    meter_env_calls(
        _Index(), ledger, [{"tool_name": "get_order_details", "kwargs_json": {"order_id": "W001"}}]
    )
    assert ledger.unique_docs == 1, (
        "the same document read twice, once by the retriever and once by a tool call, is ONE "
        "unique document. Two is the uid being counted in a set of doc ids."
    )

    traj = _traj(
        [
            _loop_retrieved_unit(),
            EvidenceUnit(
                uid=UID, corpus_id="tau2_retail_v1", doc_id=DOC, span="", title="", text=""
            ),
        ]
    )
    rec = reconcile(traj, ledger)
    assert rec["docs_ok"] is True, "with the document counted once, the run reconciles"


def test_the_env_meter_still_charges_the_call_that_re_read_a_known_document():
    """NOT A LICENCE TO RE-READ FOR FREE. Counting the document once must not stop the CALL
    being charged -- `meter_env_calls`'s own contract is "the call every time, the document
    only once", and a fix that quietly stopped debiting tool calls would make an agent that
    re-reads exhaustively look cheap."""
    ledger = BudgetLedger(cap=16)
    ledger.note_docs([DOC])
    before = dict(ledger.spent)
    meter_env_calls(
        _Index(), ledger, [{"tool_name": "get_order_details", "kwargs_json": {"order_id": "W001"}}]
    )
    after = dict(ledger.spent)
    assert after.get("retrieval_calls", 0) == before.get("retrieval_calls", 0) + 1


@pytest.mark.parametrize("n_shared", [1, 2, 3])
def test_no_overlap_of_any_size_produces_a_gap_any_more(n_shared):
    """The discrepancy used to be exactly the number of documents reached twice -- 1, 2 and 3
    shared records gave gaps of 1, 2 and 3. Parametrised over the size because a fix that
    happened to work for a single shared document and not for several would otherwise pass."""
    docs = [f"orders:W{i:03d}" for i in range(5)]
    uids = {d: f"retail/orders/{d.split(':')[1]}#0" for d in docs}

    class _Idx(_Index):
        pass

    _Idx.uids = uids
    ledger = BudgetLedger(cap=16)
    shared = docs[:n_shared]
    ledger.note_docs(shared)  # the loop retrieved these
    calls = [
        {"tool_name": "get_order_details", "kwargs_json": {"order_id": d.split(":")[1]}}
        for d in shared
    ]
    got, _ = meter_env_calls(_Idx(), ledger, calls)
    assert len(got) == n_shared

    units = [
        EvidenceUnit.make(corpus_id="tau2_retail_v1", doc_id=d, span="0:1", title="", text="x")
        for d in shared
    ] + [
        EvidenceUnit(uid=uids[d], corpus_id="tau2_retail_v1", doc_id=d, span="", title="", text="")
        for d in shared
    ]
    rec = reconcile(_traj(units), ledger)
    assert rec["ledger_unique_docs"] - rec["evidence_unique_docs"] == 0, (
        f"{n_shared} record(s) reached by both the retriever and a tool call must still count "
        "as that many unique documents, not twice that many"
    )
    assert rec["docs_ok"] is True
    assert rec["ledger_unique_docs"] == n_shared
