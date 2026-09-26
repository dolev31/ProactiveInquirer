"""Forking a tau2 rollout from a foreign trace's prefix.

WHY. Public tau2 trajectories exist from frontier models that SOLVE tasks our policy alone may
not reach. Replaying such a trace's first k messages into a fresh environment and handing
control to our policy puts it in a state it could not have produced, with the same user
simulator continuing -- which is the only way to measure "given this much progress, does asking
help from here?" on states that actually arise in successful dialogues.

It is a first-class upstream path, not a hack. `Orchestrator.initialize()` consumes
`task.initial_state.message_history`: it replays the prefix into the environment through
`Environment.set_state`, seeds the agent's state with the history minus the last message and
the user's with all of it, and renumbers `turn_idx` across prefix and continuation. The user
simulator holds no hidden state -- its whole condition is the scenario plus a message list --
so a foreign prefix reads to it as its own past.

WHERE THE CUT MAY GO, AND WHY THAT IS THE WHOLE DESIGN.
`EnvironmentEvaluator.calculate_reward` seeds the GOLD environment with the same prefix and then
applies every gold action on top. A mutating call inside the prefix is therefore applied TWICE
in gold -- a second booking, a second payment -- and a correct continuation is compared against a
world that never existed. The run scores 0 for a reason that has nothing to do with the policy,
and nothing in the output says so. So a prefix is cut strictly BEFORE the first mutating call,
by either participant, and an unknown tool counts as mutating: erring that way costs a fork
point, erring the other way silently manufactures failures.

Reads are safe by construction -- `set_state` skips non-mutating tools when replaying -- so a
read-only prefix cannot fail strict replay, which is what makes a strict-replay failure on one
a real data defect rather than something to soften.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ForkPoint:
    """One legal cut: replay `messages[:k]`, then hand over."""

    trace_sha: str
    task_id: str
    k: int


def _tool_calls(msg: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    calls = msg.get("tool_calls")
    return (
        tuple(calls) if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)) else ()
    )


def is_handover(msg: Mapping[str, Any]) -> bool:
    """A plain user turn: text, no tool calls.

    Upstream's `initialize` routes on the LAST message of the history. A user turn with tool
    calls owes ToolMessages and is refused by `validate_message_history`; an assistant turn
    leaves the USER next to speak, which is not the handover a fork is for.
    """
    return str(msg.get("role")) == "user" and not _tool_calls(msg) and bool(msg.get("content"))


def work_remains(messages: Sequence[Mapping[str, Any]], k: int) -> bool:
    """Did the SOURCE trace still have agent work to do after this cut?

    `fork_points` decides whether a cut is LEGAL -- a mutating call inside the prefix is applied
    twice by the gold evaluator, so the continuation cannot be scored. This decides whether a
    legal cut is INFORMATIVE, which is a different question and was missing.

    MEASURED on the first 28 fork runs: 14 produced an empty continuation, our agent taking zero
    turns, and all 14 scored tau_reward 1.0 -- inherited from a prefix that had already finished
    the task. Read naively that is a 57% fork success rate; on the 14 continuations that actually
    ran it is 14%.

    JUDGED ON THE SOURCE TRACE, NEVER ON OUR RUN. `turns == 0` is a property of the arm being
    measured, so filtering on it after the fact makes the surviving sample a function of the
    result. This is fixed before any policy runs. Validated against those 28 runs: it drops
    exactly the 14 empty ones and keeps exactly the 14 real ones.
    """
    return any(_tool_calls(m) for m in messages[k:])


def fork_points(
    messages: Sequence[Mapping[str, Any]],
    mutates: Callable[[str], bool],
    *,
    trace_sha: str = "",
    task_id: str = "",
) -> list[ForkPoint]:
    """Every k at which this trace may be cut, in order.

    `mutates` is asked of the environment (`Environment._is_mutating_tool`), never of a literal
    list: a list goes stale the moment upstream adds a tool, and the failure is a silently
    double-applied action.
    """
    out: list[ForkPoint] = []
    for i, msg in enumerate(messages):
        for call in _tool_calls(msg):
            if mutates(str(call.get("name") or "")):
                return out  # everything from here on would be double-applied in gold
        if is_handover(msg):
            out.append(ForkPoint(trace_sha=trace_sha, task_id=task_id, k=i + 1))
    return out
