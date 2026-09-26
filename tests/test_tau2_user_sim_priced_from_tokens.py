"""The user simulator's invoice, priced from TOKENS rather than read from the provider.

`sim.user_cost` is litellm's dollar figure, and `tau2.utils.llm_utils.get_response_cost`
catches every exception from `completion_cost` and returns 0.0 (llm_utils.py:127-131).
Our user-sim model is a proxy name (`openai/aws/gpt-oss-120b`) that litellm has no price
row for, so it raises, is swallowed, and every user message carries `cost == 0.0`.

MEASURED on the first live tau2 rollout (4 units, seed 0, budget-cap 8, k 5):

    arm                n_user_turns   user_sim_usd
    inquirer_prompted       4            0.0
    inquirer_prompted       3            0.0
    drafter_only           19            0.0
    drafter_only           17            0.0

43 user turns of real traffic priced at exactly zero. `usd_billed` then under-reports the
invoice and `--spend-cap` cannot see it -- on a 2,037-unit grid that is the difference
between a cap that binds and a cap that is decorative.

The repo already had the rule this violates. `scripts/price_tables/2026-08.json` states it
in its own `_why`: "USD is computed as tokens x these rates and NEVER read from a provider
response field." The fix is to obey it here too -- tau2 records `usage` on each message
(`UserMessage.usage`, and `get_response_usage` populates it), so we can price the user
simulator with the same `PriceTable` that prices every other call we make.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pi_run.stages.tau2_runner import user_sim_usd
from pinq_adapters.llm.pricing import PriceTable


@dataclass
class _Msg:
    role: str
    usage: dict[str, int] | None = None
    cost: float | None = 0.0


TABLE = PriceTable(
    version="test",
    models={"m": {"input": 1_000_000.0, "output": 2_000_000.0}},
    fallback={"input": 0.0, "output": 0.0, "cached_input": 0.0},
    strict=True,
)


def test_prices_user_messages_from_tokens_when_provider_says_zero() -> None:
    """The real shape: usage is present, cost is a swallowed 0.0."""
    msgs = [
        _Msg("user", {"prompt_tokens": 10, "completion_tokens": 5}, cost=0.0),
        _Msg("assistant", {"prompt_tokens": 999, "completion_tokens": 999}, cost=0.0),
        _Msg("user", {"prompt_tokens": 20, "completion_tokens": 1}, cost=0.0),
    ]
    # 30 prompt @ $1/tok + 6 completion @ $2/tok  (rates are per 1M, set to 1M above)
    usd, unpriced = user_sim_usd(msgs, "m", table=TABLE)
    assert usd == pytest.approx(30 * 1.0 + 6 * 2.0)
    assert unpriced == 0


def test_counts_only_user_role() -> None:
    """Assistant and tool tokens are the POLICY's spend and are already in the ledger.

    Counting them here would double-bill them into usd_billed and destroy the cross-arm
    token parity the ledger exists to protect.
    """
    big = {"prompt_tokens": 1_000, "completion_tokens": 1_000}
    assert user_sim_usd([_Msg("assistant", big)], "m", table=TABLE) == (0.0, 0)
    assert user_sim_usd([_Msg("tool", big)], "m", table=TABLE) == (0.0, 0)


def test_missing_usage_is_counted_not_raised_and_not_silently_zero() -> None:
    """An unpriceable message is COUNTED. Visible, but it does not cost the unit.

    This test asserted `pytest.raises` when first written, and that belief was wrong -- mine,
    not the requirement's. `user_sim_usd` is called at the END of a finished rollout, so a
    raise discards a unit whose money is already spent and escapes to kill the sweep by the
    same path `ReconcileError` does. The requirement was only ever that a $0.00 meaning
    "free" be distinguishable from a $0.00 meaning "unmeasured", and a returned count does
    that without destroying paid work.
    """
    assert user_sim_usd([_Msg("user", None)], "m", table=TABLE) == (0.0, 1)


def test_unpriced_model_is_counted_rather_than_billed_as_zero() -> None:
    """A model with no price row is the SAME kind of gap, reported the same way.

    `PriceTable.rates` raises in strict mode, which is right for a live call about to be
    made and wrong for an invoice being totted up afterwards. Caught and counted here.
    """
    msgs = [_Msg("user", {"prompt_tokens": 1, "completion_tokens": 1})]
    assert user_sim_usd(msgs, "not-in-table", table=TABLE) == (0.0, 1)


def test_no_user_messages_is_genuinely_zero() -> None:
    """The one case where 0.0 is a fact, not an absence."""
    assert user_sim_usd([], "m", table=TABLE) == (0.0, 0)


def test_runner_does_not_read_the_provider_dollar_field() -> None:
    """Source-level: `sim.user_cost` must not be what lands in `user_sim_usd`.

    A behavioural test cannot catch a regression that re-reads the provider field and only
    falls back to pricing, because the provider field returns 0.0 and the sum would still
    look plausible. This asserts the rule directly.
    """
    from pathlib import Path

    src = Path("src/pi_run/stages/tau2_runner.py").read_text()
    assert 'getattr(sim, "user_cost"' not in src, (
        "tau2_runner reads litellm's dollar field again; it is 0.0 for proxy model names "
        "and the price table's own _why forbids reading USD from a provider response."
    )
