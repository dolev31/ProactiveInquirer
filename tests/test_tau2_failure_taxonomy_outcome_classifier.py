"""scripts/tau2_failure_taxonomy/outcome_classifier.py.

`gold_actions.gold_action_names` is monkeypatched throughout rather than exercised for real:
this suite must stay collectible on a machine with no `tau2` extra installed (the repository's
own convention -- `pinq_adapters/tau2/_probe.py` -- is that a tau2 import never happens at
module scope), and the classifier's OWN branching logic is what needs to fail before a fix
here, not tau2's registry.
"""

from __future__ import annotations

from scripts.tau2_failure_taxonomy import gold_actions
from scripts.tau2_failure_taxonomy import outcome_classifier as oc


def _manifest(**over):
    base = {"suite_id": "tau2_retail", "task_id": "task_1", "budget_cap": 16}
    base.update(over)
    return base


def _status(**over):
    base = {
        "status": "ok",
        "termination_reason": "TerminationReason.USER_STOP",
        "native": {"tau_reward": 0.0},
        "spent": {"retrieval_calls": 5},
    }
    base.update(over)
    return base


def _outcome(env_calls=()):
    return {"env_calls": list(env_calls)}


# ------------------------------------------------------------------------------- not scored


def test_absent_tau_reward_is_not_scored_and_not_a_fail():
    status = _status(native={})
    result = oc.classify_outcome(_manifest(), status, _outcome())
    assert result["scored"] is False
    assert result["passed"] is None
    assert result["fail_class"] is None


def test_tau_reward_one_is_a_pass():
    status = _status(native={"tau_reward": 1.0})
    result = oc.classify_outcome(_manifest(), status, _outcome())
    assert result["scored"] is True
    assert result["passed"] is True
    assert result["fail_class"] is None


# ----------------------------------------------------------- test that fails without the fix:
# a run whose reconciliation never completed must not be silently scored a normal DB failure.


def test_top_level_error_status_is_simulator_ended_early_not_a_generic_fail():
    status = _status(status="error")
    result = oc.classify_outcome(_manifest(), status, _outcome())
    assert result["passed"] is False
    assert result["fail_class"] == "simulator_ended_early"
    assert "error" in result["note"]


def test_user_error_termination_is_simulator_ended_early():
    status = _status(termination_reason="TerminationReason.USER_ERROR")
    result = oc.classify_outcome(_manifest(), status, _outcome())
    assert result["fail_class"] == "simulator_ended_early"


# ------------------------------------------------------------------------------- over cap


def test_over_cap_is_detected_by_shared_rule():
    manifest = _manifest(budget_cap=8)
    status = _status(spent={"retrieval_calls": 9})
    assert oc.is_over_cap(manifest, status) is True
    result = oc.classify_outcome(manifest, status, _outcome())
    assert result["fail_class"] == "over_cap_runaway"
    assert "9" in result["note"] and "8" in result["note"]


def test_exactly_at_cap_is_not_over_cap():
    manifest = _manifest(budget_cap=8)
    status = _status(spent={"retrieval_calls": 8})
    assert oc.is_over_cap(manifest, status) is False


# ------------------------------------------------------------------------------ policy violation


def test_mutating_call_outside_gold_actions_is_policy_violation(monkeypatch):
    monkeypatch.setattr(
        gold_actions, "gold_action_names", lambda suite, task: ("get_order_details",)
    )
    outcome = _outcome(
        env_calls=[
            {"tool_name": "get_order_details", "ok": True, "mutating": False},
            {"tool_name": "cancel_pending_order", "ok": True, "mutating": True},
        ]
    )
    result = oc.classify_outcome(_manifest(), _status(), outcome)
    assert result["fail_class"] == "policy_violation"
    assert "cancel_pending_order" in result["note"]


def test_a_failed_mutating_call_is_not_a_violation(monkeypatch):
    # `ok: false` means the environment REFUSED it -- nothing was actually mutated.
    monkeypatch.setattr(gold_actions, "gold_action_names", lambda suite, task: ())
    outcome = _outcome(
        env_calls=[{"tool_name": "cancel_pending_order", "ok": False, "mutating": True}]
    )
    result = oc.classify_outcome(_manifest(), _status(), outcome, turns=[{"n_retrieved": 1}])
    assert result["fail_class"] != "policy_violation"


def test_untrusted_mutating_flag_never_reports_a_violation(monkeypatch):
    # the banking shape: EVERY call, including obvious reads, recorded `mutating: true`
    # (`pi_run/stages/tau2_runner.py::env_calls_from`'s own "conservatively" default). With
    # `trust_mutating=False` this must fall through rather than call it a policy violation.
    monkeypatch.setattr(
        gold_actions, "gold_action_names", lambda suite, task: ("log_verification",)
    )
    outcome = _outcome(
        env_calls=[
            {"tool_name": "log_verification", "ok": True, "mutating": True},
            {"tool_name": "get_current_time", "ok": True, "mutating": True},
        ]
    )
    result = oc.classify_outcome(
        _manifest(), _status(), outcome, turns=[{"n_retrieved": 1}], trust_mutating=False
    )
    assert result["fail_class"] != "policy_violation"


# ------------------------------------------------------- mutating_flag_is_informative itself


def test_mutating_flag_is_informative_when_both_values_are_observed():
    calls = [{"ok": True, "mutating": True}, {"ok": True, "mutating": False}]
    assert oc.mutating_flag_is_informative(calls) is True


def test_mutating_flag_is_not_informative_when_every_ok_call_is_true():
    # the measured banking shape: 100% True across every distinct tool name.
    calls = [{"ok": True, "mutating": True}] * 5
    assert oc.mutating_flag_is_informative(calls) is False


def test_mutating_flag_informativeness_ignores_failed_calls():
    # a `False` observed only on a call the environment REFUSED (ok=False) does not count --
    # that call never executed, so it says nothing about whether the field discriminates
    # among calls that did.
    calls = [{"ok": True, "mutating": True}, {"ok": False, "mutating": False}]
    assert oc.mutating_flag_is_informative(calls) is False


# -------------------------------------------------------------------------- info never obtained


def test_all_asks_empty_is_info_never_obtained(monkeypatch):
    monkeypatch.setattr(gold_actions, "gold_action_names", lambda suite, task: ())
    turns = [{"n_retrieved": 0}, {"n_retrieved": 0}, {"n_retrieved": 0}]
    result = oc.classify_outcome(_manifest(), _status(), _outcome(), turns=turns)
    assert result["fail_class"] == "info_never_obtained"
    assert "0 total n_retrieved" in result["note"]


def test_info_never_obtained_takes_priority_over_missing_tool_action(monkeypatch):
    # Both signals are present (nothing retrieved AND a gold action never called); the more
    # specific, more upstream explanation wins by the priority order the module docstring states.
    monkeypatch.setattr(
        gold_actions, "gold_action_names", lambda suite, task: ("get_order_details",)
    )
    turns = [{"n_retrieved": 0}]
    result = oc.classify_outcome(_manifest(), _status(), _outcome(), turns=turns)
    assert result["fail_class"] == "info_never_obtained"


# --------------------------------------------------------------------- wrong or missing action


def test_gold_action_never_called_is_wrong_or_missing_tool_action(monkeypatch):
    monkeypatch.setattr(
        gold_actions, "gold_action_names", lambda suite, task: ("cancel_pending_order",)
    )
    turns = [{"n_retrieved": 2}]
    result = oc.classify_outcome(_manifest(), _status(), _outcome(env_calls=[]), turns=turns)
    assert result["fail_class"] == "wrong_or_missing_tool_action"
    assert "cancel_pending_order" in result["note"]


def test_gold_action_called_successfully_does_not_reach_wrong_action(monkeypatch):
    monkeypatch.setattr(
        gold_actions, "gold_action_names", lambda suite, task: ("cancel_pending_order",)
    )
    turns = [{"n_retrieved": 2}]
    outcome = _outcome(
        env_calls=[{"tool_name": "cancel_pending_order", "ok": True, "mutating": True}]
    )
    result = oc.classify_outcome(_manifest(), _status(), outcome, turns=turns)
    # still a DB fail (native tau_reward 0.0 in _status()), but not because a gold action
    # was missed -- falls through to "other" with the db_match/reward_error note.
    assert result["fail_class"] == "other"


# ---------------------------------------------------------------------------------------- other


def test_unexplained_fail_is_other_with_a_note(monkeypatch):
    monkeypatch.setattr(gold_actions, "gold_action_names", lambda suite, task: ())
    result = oc.classify_outcome(_manifest(), _status(), _outcome(), turns=[{"n_retrieved": 3}])
    assert result["fail_class"] == "other"
    assert "db_match" in result["note"]


def test_gold_actions_lookup_failure_degrades_to_other_rather_than_raising(monkeypatch):
    def _raise(suite, task):
        raise RuntimeError("tau2 not installed")

    monkeypatch.setattr(gold_actions, "gold_action_names", _raise)
    result = oc.classify_outcome(_manifest(), _status(), _outcome(), turns=[{"n_retrieved": 1}])
    assert result["fail_class"] == "other"
