"""A fork point is only useful if the source trace still had work to do there.

MEASURED, on the 28 fork runs that existed when this was written: 14 of them produced an
EMPTY continuation -- our agent took zero turns -- and every one of those 14 scored
tau_reward 1.0. The prefix had already earned the reward and the fork inherited it. Read
naively the fork success rate was 16/28 = 57%; on the 14 continuations that actually ran it
was 2/14 = 14%.

WHY THE FILTER MUST BE OFFLINE. Dropping empty continuations after the fact is selecting on
the outcome: `turns == 0` is a property of OUR run, and conditioning a comparison on it makes
the surviving sample a function of the arm being measured. `work_remains` is a property of the
SOURCE trace at the cut point, fixed before any policy runs, so it partitions fork points
rather than run results.

VALIDATED AGAINST THE OBSERVED RUNS, not against an intuition: it drops exactly the 14 runs
with `turns == 0` and keeps exactly the 14 with a real continuation -- 14/14 both ways.

This is not the same requirement as `fork_points`' mutation rule. That one asks whether a cut
is LEGAL (a mutating call inside the prefix is applied twice by the gold evaluator, so the
continuation cannot be scored). This one asks whether a legal cut is INFORMATIVE.
"""

from __future__ import annotations

from pi_run.stages.tau2_fork import work_remains


def _assistant_call(name: str = "get_reservation_details"):
    return {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": {}}]}


def _text(role: str = "user", content: str = "hello"):
    return {"role": role, "content": content}


def test_a_cut_with_a_later_tool_call_still_has_work() -> None:
    ms = [_text(), _assistant_call(), _text(), _assistant_call()]
    assert work_remains(ms, 2) is True


def test_a_cut_past_the_last_tool_call_has_none() -> None:
    """The shape of the 14 empty continuations: everything the task needed already happened."""
    ms = [_text(), _assistant_call(), _text("assistant", "All set!"), _text()]
    assert work_remains(ms, 2) is False


def test_a_cut_at_the_very_end_has_none() -> None:
    ms = [_text(), _assistant_call()]
    assert work_remains(ms, len(ms)) is False


def test_a_cut_at_zero_has_work_when_the_trace_does() -> None:
    ms = [_text(), _assistant_call()]
    assert work_remains(ms, 0) is True


def test_trailing_text_alone_is_not_work() -> None:
    """Chat after the last action is not work the fork can be measured on."""
    ms = [_assistant_call()] + [_text("assistant", "anything else?") for _ in range(5)]
    assert work_remains(ms, 1) is False
