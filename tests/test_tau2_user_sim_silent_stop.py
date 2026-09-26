"""The user simulator's own empty completion must end the episode, not crash the unit.

Two banking-suite drafter_only runs crashed identically:

    runs/f7843949e387af86db2bf60b5e2624ec/status.json  (task_007)
    runs/3369bcad62e69f8400944769cfad232b/status.json  (task_010)

    "error": "ValueError: UserMessage must have either content or tool_calls. Got
    UserMessage\\nis_final_chunk: True"

Both tasks script the customer to resolve its own need with its OWN tool call
(`apply_for_credit_card`, `submit_referral`) and then say, verbatim, "there is no need to
respond to the agent." `UserSimulator._generate_next_message` (tau2, vendored,
`tau2/user/user_simulator.py`) hands back whatever the underlying completion contained,
unchecked -- and the completion right after that tool call has neither text nor a further
tool call to give. `Orchestrator.step()`'s AGENT/ENV -> USER branch calls
`user_msg.validate()` UNCONDITIONALLY, before it ever asks `is_stop` (`orchestrator.py:841
-844` in the installed tau2), so the malformed message raises before the loop gets a chance
to end the episode on its own.

An agent-side fix (an `is_stop` override on our own `PinqDriverAgent`) cannot reach this:
the crash follows the CUSTOMER's own tool call, and `Orchestrator.step()`'s tool-execution
branch hands control straight back to `Role.USER` (`self.to_role = self.from_role`,
`orchestrator.py:891-892`) without ever invoking our agent in between. The correction has
to sit on the way out of `user.generate_next_message` itself.

The fix (`guard_silent_user_turn` in `pi_run.stages.tau2_runner`) wraps that call so that
ONLY a result already failing tau2's own `validate()` predicate -- `has_content() or
is_tool_call()`, the exact predicate `validate()` uses, not a looser guess -- is replaced,
and only with tau2's own end-of-conversation sentinel (`STOP = "###STOP###"`), not a
fabricated reply. A well-formed message, including a tool-call-only message with no text,
is returned completely unchanged.
"""

from __future__ import annotations

import pytest


def test_the_two_crashed_runs_recorded_this_exact_error() -> None:
    """Ground the premise in the actual persisted failures, not a guess."""
    import json
    from pathlib import Path

    expected_error = (
        "ValueError: UserMessage must have either content or tool_calls. "
        "Got UserMessage\nis_final_chunk: True"
    )
    for run_id, task_id in [
        ("f7843949e387af86db2bf60b5e2624ec", "task_007"),
        ("3369bcad62e69f8400944769cfad232b", "task_010"),
    ]:
        status = json.loads(Path(f"runs/{run_id}/status.json").read_text())
        assert status["task_id"] == task_id
        assert status["arm_id"] == "drafter_only"
        assert status["status"] == "error"
        assert status["error"] == expected_error


def test_an_empty_completion_fails_validate_today() -> None:
    """The upstream shape this fix exists for. If tau2 ever stops doing this, revisit."""
    pytest.importorskip("tau2")
    from tau2.data_model.message import UserMessage

    empty = UserMessage(role="user", content=None)
    assert not empty.has_content()
    assert not empty.is_tool_call()
    with pytest.raises(ValueError, match="must have either content or tool_calls"):
        empty.validate()


def test_guard_replaces_a_silent_completion_with_the_stop_sentinel() -> None:
    """RED until `guard_silent_user_turn` exists and actually intercepts the empty reply."""
    pytest.importorskip("tau2")
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    from pi_run.stages.tau2_runner import guard_silent_user_turn

    class _StubUser:
        def generate_next_message(self, message, state):
            # The exact malformed shape `_generate_next_message` produces when the
            # underlying completion has neither text nor a tool call.
            return UserMessage(role="user", content=None), state

    guarded = guard_silent_user_turn(_StubUser())
    msg, new_state = guarded.generate_next_message(None, "state-in")

    msg.validate()  # must not raise -- this is the actual crash site in Orchestrator.step()
    assert UserSimulator.is_stop(msg), "the substitute must be tau2's own recognized stop signal"
    assert new_state == "state-in", "state must pass through untouched"


def test_guard_leaves_a_well_formed_reply_completely_unchanged() -> None:
    """No over-firing: real content, and a tool-call-only message, pass through as-is."""
    pytest.importorskip("tau2")
    from tau2.data_model.message import ToolCall, UserMessage

    from pi_run.stages.tau2_runner import guard_silent_user_turn

    real_reply = UserMessage(role="user", content="I would like to apply for a card.")

    class _StubUserText:
        def generate_next_message(self, message, state):
            return real_reply, state

    guarded = guard_silent_user_turn(_StubUserText())
    msg, _state = guarded.generate_next_message(None, "s")
    assert msg is real_reply, "a well-formed message must be returned unchanged, not rebuilt"

    tool_only = UserMessage(
        role="user",
        content=None,
        tool_calls=[ToolCall(id="c1", name="submit_referral", arguments={}, requestor="user")],
    )

    class _StubUserTool:
        def generate_next_message(self, message, state):
            return tool_only, state

    guarded_tool = guard_silent_user_turn(_StubUserTool())
    msg2, _state2 = guarded_tool.generate_next_message(None, "s")
    assert msg2 is tool_only, "a tool-call-only message must pass through unchanged too"


def test_guard_passes_through_an_object_with_no_generate_next_message() -> None:
    """`_simulate` must stay usable with a bare test double for `user`.

    `tests/test_tau2_fork_wiring.py` fakes the `Orchestrator` itself and hands `_simulate`
    a plain `object()` for `user`, precisely so it can test task/prefix wiring "before
    `orch.run()` is reached" without needing a real, callable user. Before this guard
    existed, `_simulate` never touched `user` ahead of constructing that (faked)
    `Orchestrator`, so `object()` was always sufficient. `guard_silent_user_turn` must not
    newly demand an attribute those doubles were never built to have.
    """
    from pi_run.stages.tau2_runner import guard_silent_user_turn

    bare = object()
    assert not hasattr(bare, "generate_next_message")
    guarded = guard_silent_user_turn(bare)
    assert guarded is bare, "an object with nothing to wrap must be handed back unchanged"


def test_guard_is_wired_into_simulate() -> None:
    """Source-level: the guard must actually run inside `_simulate`, not just exist unused."""
    from pathlib import Path

    src = Path("src/pi_run/stages/tau2_runner.py").read_text()
    sim_start = src.index("def _simulate(")
    next_def = src.index("\ndef ", sim_start + 1)
    sim_body = src[sim_start:next_def]
    assert "guard_silent_user_turn(user)" in sim_body, (
        "guard_silent_user_turn exists but _simulate never calls it -- the crash is still unguarded"
    )
