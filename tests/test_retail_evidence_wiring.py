"""Retail tool calls debit the budget and deliver evidence. Both, or neither is coherent.

THE DECISION, stated so it is not re-litigated by accident: a tool call debits `budget_cap`,
the same counter a retrieval debits on musique. A proactive agent retrieves AND acts, and
looking something up has to cost something or anticipation is worthless -- an agent that can
call tools for free calls them exhaustively, which is brute force wearing proactivity's name.

WHAT FOLLOWS FROM IT, and why the two halves cannot be separated:

  * `reconcile` asserts `ledger.unique_docs == len(traj.evidence.doc_ids)`. It is the detector
    for unmetered nested retrieval -- a Drafter that retrieves privately and folds the result
    into its Evidence grows the second set and not the first, which is invisible in tokens and
    fatal to a budget-parity claim. So attaching evidence WITHOUT metering it breaks the
    invariant, and metering WITHOUT attaching evidence leaves gold unsatisfiable.

  * MUTATIONS DEBIT TOO but carry no evidence: 176 of retail's 550 required calls are writes
    (`cancel_pending_order`, the `modify_*` family). A write costs the agent a turn and
    changes the world; it does not teach it a fact. So the call is charged and no uid is
    added, and `unique_docs` still equals the evidence set.

  * The uids come from the call's ARGUMENTS, never its result -- a retail tool result is a
    customer record and the runner stores only a digest.
"""

from __future__ import annotations

import pytest

from pinq.budget import BudgetLedger


class _Idx:
    """Stands in for RetailIndex: two readable records, addressed by key."""

    uids = {"orders:#W1": "u-order-1", "users:u77": "u-user-77"}
    item_to_product: dict[str, str] = {}
    titles = {"orders:#W1": "orders/#W1", "users:u77": "users/u77"}
    text_sha = {"orders:#W1": "a", "users:u77": "b"}

    def uid(self, table, rid):
        return self.uids[f"{table}:{rid}"]


def _calls(*specs):
    return [{"tool_name": n, "kwargs_json": a, "ok": True} for n, a in specs]


def test_a_read_debits_the_budget_and_adds_its_record() -> None:
    from pi_run.stages.tau2_runner import meter_env_calls

    led = BudgetLedger(cap=8)
    uids, _ = meter_env_calls(_Idx(), led, _calls(("get_order_details", '{"order_id":"#W1"}')))
    assert uids == ("u-order-1",)
    assert led.spent.get("retrieval_calls") == 1
    assert led.unique_docs == 1


def test_a_write_debits_but_adds_no_evidence() -> None:
    """It cost a turn and changed the world; it did not teach the agent a fact."""
    from pi_run.stages.tau2_runner import meter_env_calls

    led = BudgetLedger(cap=8)
    uids, _ = meter_env_calls(_Idx(), led, _calls(("cancel_pending_order", '{"reason":"x"}')))
    assert uids == ()
    assert led.spent.get("retrieval_calls") == 1
    assert led.unique_docs == 0


def test_the_reconcile_invariant_holds() -> None:
    """`ledger.unique_docs == len(evidence.doc_ids)` is what `reconcile` checks. Reads and
    writes mixed, repeats included."""
    from pi_run.stages.tau2_runner import meter_env_calls

    led = BudgetLedger(cap=16)
    uids, _ = meter_env_calls(
        _Idx(),
        led,
        _calls(
            ("get_order_details", '{"order_id":"#W1"}'),
            ("cancel_pending_order", '{"order_id":"#W1","reason":"x"}'),
            ("get_user_details", '{"user_id":"u77"}'),
        ),
    )
    assert led.unique_docs == len(set(uids))


def test_a_repeated_read_still_costs_a_CALL_but_not_a_second_document() -> None:
    """`note_docs` meters unique documents so a policy cannot free-ride on the cache; the call
    itself is charged every time, because the agent still spent a turn on it."""
    from pi_run.stages.tau2_runner import meter_env_calls

    led = BudgetLedger(cap=8)
    meter_env_calls(
        _Idx(),
        led,
        _calls(
            ("get_order_details", '{"order_id":"#W1"}'),
            ("get_order_details", '{"order_id":"#W1"}'),
        ),
    )
    assert led.spent.get("retrieval_calls") == 2
    assert led.unique_docs == 1


def test_a_failed_call_is_not_charged() -> None:
    """An error returns nothing; charging for it would make a flaky environment look like an
    expensive policy."""
    from pi_run.stages.tau2_runner import meter_env_calls

    led = BudgetLedger(cap=8)
    calls = _calls(("get_order_details", '{"order_id":"#W1"}'))
    calls[0]["ok"] = False
    assert meter_env_calls(_Idx(), led, calls)[0] == ()
    assert led.spent.get("retrieval_calls", 0) == 0


def test_exceeding_the_cap_is_recorded_and_reported_rather_than_raised() -> None:
    """WAS `test_exceeding_the_cap_raises_rather_than_silently_overspending`. The belief in that
    name was wrong about THIS call site, and the code was right to be changed. Kept, not deleted,
    because the property it was defending is real.

    THE PROPERTY THAT SURVIVES: an overspend must never be SILENT. It is not. The spend lands in
    `spent["retrieval_calls"]` and the overrun comes back as a flag `_attach_env_evidence`
    records on the run as `post_hoc_budget_overrun`. Both are asserted below.

    THE BELIEF THAT DID NOT: that raising here prevents an overspend. `meter_env_calls` runs from
    `_attach_env_evidence`, AFTER `_simulate` has returned -- the Orchestrator has already
    executed those tool calls and the money is already gone. The raise was caught by
    `run_tau2_unit`'s own `except Exception`, which wrote `status: error` over a unit that had
    finished and carried a real reward. So it prevented no spend and deleted a measurement, and
    it deleted one selectively: replayed over the 204 recorded tau2_retail/test fork runs behind
    the published transfer result it refuses 70, 57 on `self_ask` against 13 on
    `inquirer_prompted`, filtering the endpoint by its own value
    (`tests/test_post_hoc_harvest_meter.py`).

    THE CAP THIS TEST STILL PROTECTS IS THE OTHER ONE. In-loop, `pinq.loop` charges through
    `charge_retrieval` while the policy is still choosing, and exceeding the cap truncates the
    rollout with `stop_reason="budget"`. That is the hard cap working and it is unchanged; the
    two paragraphs below assert it here so this file cannot be read as having dropped it, and
    `test_synth_closed_form.py` reaches it through a real `run_loop`.
    """
    from pi_run.stages.tau2_runner import meter_env_calls
    from pinq.budget import BudgetExceeded

    led = BudgetLedger(cap=1)
    uids, overrun = meter_env_calls(
        _Idx(),
        led,
        _calls(
            ("get_order_details", '{"order_id":"#W1"}'),
            ("get_user_details", '{"user_id":"u77"}'),
        ),
    )
    assert overrun is True, "the overspend is reported"
    assert led.spent["retrieval_calls"] == 2.0, "and recorded, not swallowed"
    assert len(uids) == 2, "and the evidence those calls really delivered is still attached"

    # THE IN-LOOP CAP, WHICH DID NOT MOVE. Same ledger class, same single hard currency: a
    # pre-dispatch charge over the cap still refuses, and the refused charge is not recorded.
    inloop = BudgetLedger(cap=1)
    inloop.charge_retrieval(1.0)
    with pytest.raises(BudgetExceeded):
        inloop.charge_retrieval(1.0)
    assert inloop.spent["retrieval_calls"] == 1.0
