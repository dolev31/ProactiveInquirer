"""scripts/tau2_failure_taxonomy/gold_actions.py's pure logic, with the tau2-registry-backed
functions monkeypatched -- this suite must stay collectible without the `tau2` extra installed.

The live, un-mocked check (all 97 banking_knowledge tasks, and a random 20-task subsample,
against the real registered tool set) is run separately and pasted with its output into
`artifacts/tau2_failure_taxonomy_20260919/RESULT.md` ss4, per CLAUDE.md rule 3 -- a pytest run
is not where a citable measurement belongs, but the SET-DIFFERENCE LOGIC that measurement
depends on is exactly what belongs in a test, pinned here so it cannot silently invert.
"""

from __future__ import annotations

from scripts.tau2_failure_taxonomy import gold_actions as ga


def test_domain_for_maps_the_three_suites_this_lane_reads():
    assert ga.domain_for("tau2_retail") == "retail"
    assert ga.domain_for("tau2_airline") == "airline"
    assert ga.domain_for("tau2") == "banking_knowledge"
    assert ga.domain_for("musique") is None


def test_unresolvable_gold_actions_empty_when_every_name_is_registered(monkeypatch):
    monkeypatch.setattr(
        ga, "gold_action_names", lambda suite, task: ("get_order_details", "cancel_pending_order")
    )
    monkeypatch.setattr(
        ga,
        "registered_tool_names",
        lambda suite: frozenset({"get_order_details", "cancel_pending_order", "get_user_details"}),
    )
    assert ga.unresolvable_gold_actions("tau2_retail", "task_1") == ()


def test_unresolvable_gold_actions_names_exactly_the_gap(monkeypatch):
    monkeypatch.setattr(ga, "gold_action_names", lambda suite, task: ("shell", "get_current_time"))
    monkeypatch.setattr(ga, "registered_tool_names", lambda suite: frozenset({"get_current_time"}))
    assert ga.unresolvable_gold_actions("tau2", "task_x") == ("shell",)


def test_an_empty_gold_action_list_is_not_a_missing_tool(monkeypatch):
    # retail/airline ship `actions: []` on tasks whose correct outcome is a database that does
    # NOT change (pi_run/suites.py's own note) -- an empty gold list must read as "nothing to
    # resolve", never as "everything is missing".
    monkeypatch.setattr(ga, "gold_action_names", lambda suite, task: ())
    monkeypatch.setattr(ga, "registered_tool_names", lambda suite: frozenset())
    assert ga.unresolvable_gold_actions("tau2_retail", "task_no_op") == ()


def test_unresolvable_gold_actions_returns_none_distinctly_from_empty(monkeypatch):
    # a task tau2's own registry does not know about is a wiring bug in the caller, not a
    # "nothing missing" result -- `gold_action_names` returns None for it, and this function
    # must not paper over that by treating None the same as `()`.
    monkeypatch.setattr(ga, "gold_action_names", lambda suite, task: None)
    monkeypatch.setattr(ga, "registered_tool_names", lambda suite: frozenset())
    assert ga.unresolvable_gold_actions("tau2_retail", "no_such_task") == ()
