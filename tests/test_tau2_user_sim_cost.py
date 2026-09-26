"""The tau2 user simulator's spend must reach `usd_billed`, or `--spend-cap` is blind to it.

The simulator is a SECOND model conversation, driven by tau2's own litellm rather than our
`MeteredClient` -- so it is uncached, unmetered by `BudgetLedger`, and real money. The driver
already intends to fold it in: `tau2_runner` reads `sim.user_cost` and adds it to
`usd_billed`, with the comment "they are added to usd_billed, so the campaign cap sees the
actual invoice rather than two thirds of it".

THAT COMMENT WAS FALSE IN PRACTICE. `sim.user_cost` comes from tau2's `get_cost(messages)`,
which returns None if ANY non-tool message lacks a `.cost` -- and our driver hand-builds
`AssistantMessage`s, whose `cost` defaults to None. Measured directly against the installed
tau2:

    AssistantMessage(...).cost              -> None
    get_cost([user_with_cost])              -> (0, 0.001)
    get_cost([user_with_cost, asst_no_cost]) -> None        <-- poisons the whole sum
    get_cost([user_with_cost, asst_cost_0])  -> (0.0, 0.001) <-- sums correctly

So on every tau2 unit `user_sim_usd` was 0.0 and `usd_billed` under-reported the invoice by
the entire user-simulator share. On a 2,037-unit tier2 grid that is the difference between a
spend cap that holds and one that silently does not.

0.0 is the CORRECT value for our assistant messages, not a placeholder: those turns are
produced by our own metered client and already counted in `BudgetLedger`. Their incremental
cost *in tau2's ledger* is genuinely zero, and stating it lets tau2 sum the user's turns.
"""

from __future__ import annotations

import pytest


def test_tau2_get_cost_is_poisoned_by_a_costless_message() -> None:
    """The upstream behaviour this fix exists for. If tau2 ever stops doing this, revisit."""
    pytest.importorskip("tau2")
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.utils.llm_utils import get_cost

    user = UserMessage(role="user", content="hi")
    user.cost = 0.001

    bare = AssistantMessage(role="assistant", content="hello")
    assert bare.cost is None, "tau2 changed its default; the fix below may be unnecessary"
    assert get_cost([user, bare]) is None, "a costless message no longer poisons the sum"

    priced = AssistantMessage(role="assistant", content="hello")
    priced.cost = 0.0
    assert get_cost([user, priced]) is not None, "cost=0.0 no longer lets the sum through"


def test_our_driver_prices_every_assistant_message_it_builds() -> None:
    """Both construction sites -- the tool-call branch and the final-answer branch."""
    import inspect

    from pi_run.stages import tau2_runner

    src = inspect.getsource(tau2_runner)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    built = code.count("AssistantMessage(")
    assert built >= 2, f"expected both construction sites, found {built}"
    # every construction must carry an explicit cost
    for chunk in code.split("AssistantMessage(")[1:]:
        head = chunk[: chunk.index(")")] if ")" in chunk else chunk
        assert "cost=" in head, (
            f"an AssistantMessage is built with no cost: AssistantMessage({head})"
        )


def test_a_priced_transcript_sums_to_the_user_share() -> None:
    """End-to-end on the shape our driver produces: user turns priced, ours at zero."""
    pytest.importorskip("tau2")
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.utils.llm_utils import get_cost

    msgs = []
    for i in range(3):
        u = UserMessage(role="user", content=f"u{i}")
        u.cost = 0.002
        msgs.append(u)
        a = AssistantMessage(role="assistant", content=f"a{i}")
        a.cost = 0.0
        msgs.append(a)

    got = get_cost(msgs)
    assert got is not None
    _agent, user = got
    assert user == pytest.approx(0.006), f"user share should be 3 x 0.002, got {user}"
