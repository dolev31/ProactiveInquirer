"""THE ARM THAT OPENS THE USER CHANNEL COULD NOT OPEN IT ON THE ONLY SUITE WITH A USER.

`inquirer_may_ask_user` exists to "quantify the user-private ceiling": the policy is told it
may address the customer, and what it asks measures how much of the task was never in the KB
at all. Two independent gates gate that, and they were not agreeing.

  1. the POLICY gate -- `PromptedInquirer(may_ask_user=True)` -- which the arm table sets;
  2. the LOOP gate -- `run_loop(allow_user_target=...)` -- which `tau2_runner` hard-coded to
     False for every arm.

So on tau2 the prompt said "you may ask the user", the policy emitted `target="user"`, and the
loop charged it as a wasted turn and rewrote it to "kb". The arm had never been run on tau2:
0 runs on disk. The ceiling it exists to measure was never measured.

The fix derives the loop gate FROM the policy rather than adding a second switch. A new global
flag would be free to disagree with the arm table; reading the policy's own declaration cannot
be. The confirmatory guarantee -- "the user channel is closed by construction in every
confirmatory arm" -- is then a property of the arm table, which is where it is already stated.
"""

from __future__ import annotations

import pytest

from pi_run.stages.tau2_runner import allows_user_target
from pinq_expt.arms import ARMS


def _build(arm):
    """Arm inquirer factories have DIFFERENT signatures -- `NeverAsk()` takes none, the
    prompted policies take an llm. Try both rather than assuming one."""
    factory = getattr(arm, "inquirer", None)
    if factory is None:
        return None
    for args in ((), (None,)):
        try:
            return factory(*args)
        except TypeError:
            continue
    return None


class _Plain:
    """A policy that never declared anything, i.e. every arm but one."""


class _Opens:
    may_ask_user = True


class _Closes:
    may_ask_user = False


def test_a_policy_that_declares_nothing_keeps_the_channel_closed() -> None:
    """The default must be CLOSED. A policy with no opinion cannot open a channel that the
    confirmatory design closes by construction."""
    assert allows_user_target(_Plain()) is False


def test_the_declaration_is_honoured_in_both_directions() -> None:
    assert allows_user_target(_Opens()) is True
    assert allows_user_target(_Closes()) is False


def test_none_is_closed_not_crashing() -> None:
    assert allows_user_target(None) is False


def test_exactly_one_shipped_arm_opens_the_channel() -> None:
    """If a second arm ever opens it, that is a design decision and must be made deliberately;
    this test is what forces it to be noticed."""
    opens = {aid for aid, arm in ARMS.items() if allows_user_target(_build(arm))}
    assert opens == {"inquirer_may_ask_user"}, f"arms opening the user channel: {sorted(opens)}"


@pytest.mark.parametrize("arm_id", ["inquirer_prompted", "drafter_only", "self_inquire"])
def test_the_confirmatory_arms_stay_closed(arm_id: str) -> None:
    """The property the prereg cites. Asserted against the live arm table, not restated."""
    assert allows_user_target(_build(ARMS[arm_id])) is False
