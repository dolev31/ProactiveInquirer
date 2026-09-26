"""ConvState obeys the same two walls State and TaskView obey.

Wall 1, the budget rule (AGENTS.md): a policy that can see how much is left confounds STOP
with the cap it was told about. Wall 2, the type wall: no field name is shared with GoldNode,
so a gold record cannot be smuggled across by name.
"""

import dataclasses

from pi_eval.gold import GoldNode
from pinq_adapters.convlog.types import ConvState, ToolStep

FORBIDDEN = {
    "budget",
    "cap",
    "remaining",
    "turns_left",
    "max_turns",
    "step_limit",
    "arm",
    "arm_id",
    "n_turns",
    "turn_budget",
    "tokens_left",
    "spend",
}


def test_conv_state_has_no_budget_field():
    names = {f.name for f in dataclasses.fields(ConvState)}
    assert not (names & FORBIDDEN)


def test_conv_state_shares_no_field_name_with_gold_node():
    conv = {f.name for f in dataclasses.fields(ConvState)}
    gold = {f.name for f in dataclasses.fields(GoldNode)}
    assert not (conv & gold)


def test_conv_state_is_frozen_and_hashable():
    s = ConvState(
        session_id="S",
        dp_index=0,
        cwd_norm="~/x",
        git_branch_norm="main",
        after_compaction=False,
        task_statement="t",
        prior_user_turns=(),
        prior_assistant_text=(),
        tool_trace=(ToolStep("Bash", "ls", True, 2, "ok"),),
    )
    assert hash(s)
    try:
        s.dp_index = 1
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ConvState must be frozen")
