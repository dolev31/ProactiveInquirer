"""Where a foreign trace may be cut, and why the rule is "before the first mutating call".

THE FORK. Take a public trajectory that SUCCEEDED, replay its first k messages into a fresh
environment, and hand control to our policy with the same user simulator continuing. That is a
first-class upstream path: `Orchestrator.initialize()` consumes
`task.initial_state.message_history`, replays it through `Environment.set_state`, and seeds both
the agent's and the user's state from it. It is how we start from a state a frontier model
reached that our agent alone might not.

THE CONSTRAINT, AND IT IS NOT A STYLE CHOICE. `EnvironmentEvaluator.calculate_reward` seeds the
GOLD environment with the same prefix and then applies EVERY gold action on top. So a mutating
call inside the prefix is applied twice in gold -- `book_reservation` makes a second
reservation, `refuel_data` refuels twice -- and a perfectly correct continuation is graded
against a world that never existed. The run would score 0 for a reason that has nothing to do
with the policy, and it would do so silently.

Reads are safe: `set_state` skips non-mutating tools on replay, so a read-only prefix cannot
even fail strict replay. That is why the cut is placed before the first mutation rather than
anywhere else, and why a failure of strict replay on such a prefix is a real data defect rather
than something to soften.

The cut must also land where the AGENT is next to speak -- upstream's `initialize` only routes
cleanly when the last message is a plain user turn.
"""

from __future__ import annotations

from pi_run.stages.tau2_fork import ForkPoint, fork_points

MUTATING = {"book_reservation", "cancel_reservation", "make_payment", "refuel_data"}


def mutates(name: str) -> bool:
    return name in MUTATING


def user(text: str) -> dict:
    return {"role": "user", "content": text, "tool_calls": None}


def asst(text: str = "", calls: list[str] | None = None) -> dict:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {"id": f"c{i}", "name": n, "arguments": {}} for i, n in enumerate(calls or [])
        ]
        or None,
    }


def tool(name: str = "t") -> dict:
    return {"role": "tool", "content": "result", "id": "c0", "requestor": "assistant"}


def test_a_cut_lands_after_a_plain_user_turn() -> None:
    msgs = [user("hi"), asst("hello"), user("cancel my flight")]
    assert [p.k for p in fork_points(msgs, mutates)] == [1, 3]


def test_no_cut_after_an_assistant_turn() -> None:
    """Upstream's initialize routes on the LAST message; an assistant turn leaves the user
    next to speak, which is not the handover this fork is for."""
    msgs = [user("hi"), asst("hello")]
    assert [p.k for p in fork_points(msgs, mutates)] == [1]


def test_no_cut_after_a_user_turn_that_is_a_tool_call() -> None:
    """`validate_message_history` requires a participant message to be text XOR tool calls,
    and a user tool call has a ToolMessage owed to it."""
    msgs = [user("hi"), {"role": "user", "content": None, "tool_calls": [{"id": "c", "name": "x"}]}]
    assert [p.k for p in fork_points(msgs, mutates)] == [1]


def test_reads_before_the_cut_are_fine() -> None:
    msgs = [user("hi"), asst(calls=["get_reservation_details"]), tool(), user("and then?")]
    assert [p.k for p in fork_points(msgs, mutates)] == [1, 4]


def test_nothing_is_offered_at_or_after_the_first_mutation() -> None:
    """The whole rule: gold would apply that booking a second time."""
    msgs = [
        user("book it"),
        asst(calls=["book_reservation"]),
        tool(),
        user("thanks"),
        asst("done"),
        user("anything else?"),
    ]
    assert [p.k for p in fork_points(msgs, mutates)] == [1]


def test_a_user_side_mutation_closes_the_prefix_too() -> None:
    """393 of telecom's 516 gold actions are the CUSTOMER acting. A mutation is a mutation
    whoever requested it -- gold replays user actions as well."""
    msgs = [
        user("I paid"),
        {"role": "user", "content": None, "tool_calls": [{"id": "c", "name": "make_payment"}]},
        {"role": "tool", "content": "paid", "id": "c", "requestor": "user"},
        user("now what?"),
    ]
    assert [p.k for p in fork_points(msgs, mutates)] == [1]


def test_an_unknown_tool_is_treated_as_mutating() -> None:
    """Erring toward 'mutating' loses a fork point. Erring the other way silently double-applies
    an action and grades a correct continuation as a failure."""
    msgs = [user("hi"), asst(calls=["some_tool_we_do_not_know"]), tool(), user("next")]
    assert [p.k for p in fork_points(msgs, lambda n: n not in {"get_x"})] == [1]


def test_every_point_carries_what_identifies_it() -> None:
    pts = fork_points([user("hi")], mutates, trace_sha="abc123", task_id="7")
    assert pts == [ForkPoint(trace_sha="abc123", task_id="7", k=1)]


def test_an_empty_trace_offers_nothing() -> None:
    assert fork_points([], mutates) == []
