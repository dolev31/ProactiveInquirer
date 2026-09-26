"""Loading a foreign trace's prefix, and handing it to the run as a state it must continue.

`tau2_fork.fork_points` already decides WHERE a public trace may legally be cut. These are the
two functions that EXECUTE such a cut: `load_prefix` turns (trace sha, k) into messages, and
`forked_task` puts those messages where upstream's Orchestrator and its evaluator both read
them. Both were on convlog-work and neither was on main, which is why `scripts/run_tau2_forks.py`
could not be ported with T10 -- a launcher without them writes UNFORKED rollouts carrying fork
identity straight into the fork table.

THE TWO REFUSALS ARE THE POINT. A named trace that cannot be found raises, and an empty or
badly-cut prefix raises, because the alternative in both cases is a run that looks exactly like
a fork, is counted as one, and is a fresh rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from pi_run.stages.tau2_runner import count_user_turns, forked_task, load_prefix

tau2 = pytest.importorskip("tau2.data_model.tasks", reason="tau2 is not installed")


@dataclass
class Spec:
    """Only the two attributes `load_prefix` reads. `getattr` is used there so an ordinary
    UnitSpec from before the fields existed still means 'not a fork'."""

    foreign_trace_sha: str | None = None
    foreign_prefix_k: int | None = None


def _msg(role: str, content: str, **kw):
    return {"role": role, "content": content, **kw}


TRACE_SHA = "a" * 64
MESSAGES = [
    _msg("assistant", "Hi! How can I help you today?"),
    _msg("user", "I need to cancel order W123."),
    _msg("assistant", None, tool_calls=[{"name": "get_order_details", "arguments": {}}]),
    _msg("tool", '{"order_id": "W123"}', id="c1", requestor="assistant"),
    _msg("assistant", "I found it. Shall I cancel?"),
    _msg("user", "Yes please."),
]


@pytest.fixture
def trace_root(tmp_path):
    """One imported trace, in the layout `pi traces import` writes: <root>/<source>/<domain>.jsonl."""
    d = tmp_path / "as_fixture"
    d.mkdir()
    (d / "retail.jsonl").write_text(
        json.dumps({"trace_sha": TRACE_SHA, "task_id": "42", "messages": MESSAGES})
        + "\n"
        # A second row, to prove the scan selects by sha rather than taking the first line.
        + json.dumps({"trace_sha": "b" * 64, "task_id": "7", "messages": MESSAGES[:2]})
        + "\n"
    )
    return str(tmp_path)


# ------------------------------------------------------------------------------ load_prefix


def test_an_ordinary_rollout_loads_no_prefix(trace_root):
    """Every non-fork unit goes through this call, so the empty answer is the common path."""
    assert load_prefix(Spec(), root=trace_root) == []
    assert load_prefix(Spec(foreign_trace_sha=TRACE_SHA), root=trace_root) == []
    assert load_prefix(Spec(foreign_prefix_k=3), root=trace_root) == []


def test_a_spec_without_the_fields_at_all_is_not_a_fork(trace_root):
    """`getattr(spec, ..., None)`: a caller holding an older spec object must not crash."""

    class Bare:
        pass

    assert load_prefix(Bare(), root=trace_root) == []


def test_the_first_k_messages_come_back_in_order(trace_root):
    out = load_prefix(Spec(TRACE_SHA, 2), root=trace_root)
    assert [m.role for m in out] == ["assistant", "user"]
    assert out[1].content == "I need to cancel order W123."


def test_the_sha_selects_the_row_not_the_file_order(trace_root):
    """The wanted row is the FIRST in the file here, so a passing test proves nothing unless
    the other row is also reachable. It is, and it has different messages."""
    assert len(load_prefix(Spec("b" * 64, 2), root=trace_root)) == 2
    assert len(load_prefix(Spec(TRACE_SHA, 6), root=trace_root)) == 6


def test_a_tool_message_keeps_its_id_and_requestor(trace_root):
    """A ToolMessage with no id fails upstream's own history validation, so the round trip
    through dicts has to carry them."""
    out = load_prefix(Spec(TRACE_SHA, 4), root=trace_root)
    tool = out[3]
    assert tool.role == "tool"
    assert tool.id == "c1"
    assert tool.requestor == "assistant"


def test_an_assistant_tool_call_survives(trace_root):
    out = load_prefix(Spec(TRACE_SHA, 3), root=trace_root)
    assert out[2].role == "assistant"
    assert out[2].tool_calls


def test_a_missing_trace_raises_rather_than_running_unforked(trace_root):
    """THE REFUSAL THAT MATTERS. A fork that quietly becomes a fresh rollout still carries
    fork identity, still lands in the fork table, and is still paired against real forks."""
    with pytest.raises(FileNotFoundError, match="a fork cannot run unprefixed"):
        load_prefix(Spec("c" * 64, 3), root=trace_root)


def test_a_missing_trace_root_raises_too(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prefix(Spec(TRACE_SHA, 3), root=str(tmp_path / "nothing-here"))


# ----------------------------------------------------------------------------- forked_task


def _task(**kw):
    from tau2.data_model.tasks import Task, UserScenario

    return Task(id="42", user_scenario=UserScenario(instructions="be a customer"), **kw)


def test_the_prefix_lands_where_the_orchestrator_reads_it(trace_root):
    """`Orchestrator.initialize()` consumes `task.initial_state.message_history` -- it replays
    it into the environment and seeds BOTH participants from it. Anywhere else is invisible."""
    prefix = load_prefix(Spec(TRACE_SHA, 6), root=trace_root)
    out = forked_task(_task(), prefix)
    assert [m.role for m in out.initial_state.message_history] == [m["role"] for m in MESSAGES]


def test_the_original_task_is_not_touched(trace_root):
    """`suite.tau2_task_object` may hand back the SAME object for the next seed, so mutating
    in place leaks one fork's prefix into every later rollout of that task in the process."""
    prefix = load_prefix(Spec(TRACE_SHA, 6), root=trace_root)
    task = _task()
    out = forked_task(task, prefix)
    assert task.initial_state is None
    assert out is not task


def test_initialisation_actions_survive_a_fork(trace_root):
    """`initial_state` is EDITED, not replaced. All 114 telecom base tasks carry
    `initialization_actions` that set the handset state the scenario depends on; replacing the
    object wholesale starts every telecom fork with a phone that was never misconfigured."""
    from tau2.data_model.tasks import InitialState

    prefix = load_prefix(Spec(TRACE_SHA, 6), root=trace_root)
    task = _task(initial_state=InitialState(initialization_data=None, initialization_actions=[]))
    before = task.initial_state.initialization_actions
    out = forked_task(task, prefix)
    assert out.initial_state.initialization_actions == before
    assert len(out.initial_state.message_history) == 6


def test_an_empty_prefix_is_refused(trace_root):
    """A fresh rollout wearing a fork's run identity is the exact thing the fork table must
    not contain."""
    with pytest.raises(ValueError, match="an empty one is a fresh rollout"):
        forked_task(_task(), [])


def test_a_prefix_that_ends_on_the_agent_is_refused(trace_root):
    """Upstream routes on the LAST message. An assistant turn leaves the USER next to speak,
    which is not the handover a fork is for -- the policy would never get the microphone."""
    prefix = load_prefix(Spec(TRACE_SHA, 5), root=trace_root)
    assert prefix[-1].role == "assistant"
    with pytest.raises(ValueError, match="so the AGENT speaks next"):
        forked_task(_task(), prefix)


def test_a_user_turn_carrying_tool_calls_is_refused(trace_root):
    """`validate_message_history` refuses one: it owes ToolMessages that the cut discarded."""
    from tau2.data_model.message import UserMessage

    bad = [UserMessage(role="user", content="hi", tool_calls=[{"name": "x", "arguments": {}}])]
    with pytest.raises(ValueError, match="with tool calls"):
        forked_task(_task(), bad)


# ----------------------------------------------------------------- n_prefix_user_turns


def test_the_prefix_user_turns_are_counted_with_the_same_function_as_the_run(trace_root):
    """`fork_report.follow_ups` is `n_user_turns - n_prefix_user_turns`, so the two counts must
    come from ONE definition. Counting the prefix differently from the transcript would make
    the difference meaningless while both numbers stayed plausible."""
    prefix = load_prefix(Spec(TRACE_SHA, 6), root=trace_root)
    assert count_user_turns(prefix) == 2


def test_an_unforked_run_counts_zero_prefix_turns():
    """0, and it must be a real 0 rather than a missing key: `compact` writes NULL when absent
    and `environment.prefix_user_turns` defaults a missing value to 0.0, which is how a forked
    run came to be charged for its prefix's user turns with nothing on the row to show it."""
    assert count_user_turns([]) == 0
