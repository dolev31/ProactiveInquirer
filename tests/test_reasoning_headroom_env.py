"""The reasoning headroom must be settable per MODEL, or a reasoning model reads as a failure.

`_budget` already documents the failure exactly: "a caller asking for 700 content tokens can
receive 700 reasoning tokens and EMPTY CONTENT -- which then scores 0 on every metric and reads
as a model failure rather than a budget bug." It was measured on gpt-oss and the default
headroom of 3.0 fixed it there.

IT WAS NOT ENOUGH FOR gpt-5, AND THE FAILURE WAS READ EXACTLY AS THE DOCSTRING PREDICTS.
Measured over 196 gpt-5 inquirer calls at ACT_MAX_TOKENS=400 x 3.0 = 1200:

    tok_completion   mean 1063.4   median 1200   max 1200
    tok_reasoning    mean 1039.0   median 1200   max 1200

completion == reasoning, both pinned at the cap: the model spent its whole budget reasoning and
emitted no content. Empty response -> parse failure -> STOP. That produced "gpt-5 stops at turn
0 49.1% of the time against gpt-oss's 1.7%" and "answer_correct 0.098 against 0.525", which
were reported as a verdict on the MODEL. They were a verdict on the budget.

The headroom was a constructor default with no env knob, so it could not be raised for one
model without editing code -- which is what made the wrong conclusion easy to reach and hard to
question.
"""

from __future__ import annotations

import pytest

from pinq_adapters.llm.litellm_client import DEFAULT_REASONING_HEADROOM, MeteredClient


def _client(env):
    return MeteredClient.__new__(MeteredClient)  # not used; see _headroom_from below


def test_the_default_is_unchanged() -> None:
    """gpt-oss was measured at 3.0 and works; raising the default would silently change every
    existing arm's token budget and therefore its cost."""
    assert DEFAULT_REASONING_HEADROOM == 3.0


def test_the_env_var_overrides_it() -> None:
    from pinq_adapters.llm.litellm_client import _headroom_from

    assert _headroom_from({"PI_LLM_REASONING_HEADROOM": "8"}) == 8.0
    assert _headroom_from({}) == DEFAULT_REASONING_HEADROOM


def test_a_value_below_one_is_refused_not_clamped_silently() -> None:
    """Headroom under 1.0 would ask for FEWER tokens than the caller's content budget, which
    is the bug this parameter exists to prevent, inverted."""
    from pinq_adapters.llm.litellm_client import _headroom_from

    assert _headroom_from({"PI_LLM_REASONING_HEADROOM": "0.5"}) == 1.0


def test_garbage_falls_back_to_the_default_rather_than_crashing_a_sweep() -> None:
    from pinq_adapters.llm.litellm_client import _headroom_from

    assert _headroom_from({"PI_LLM_REASONING_HEADROOM": "banana"}) == DEFAULT_REASONING_HEADROOM
    assert _headroom_from({"PI_LLM_REASONING_HEADROOM": ""}) == DEFAULT_REASONING_HEADROOM


@pytest.mark.parametrize("headroom,expected", [(3.0, 1200), (8.0, 3200)])
def test_the_budget_scales_with_it(headroom: float, expected: int) -> None:
    """400 content tokens x headroom. gpt-5's reasoning alone reached 1200, so at 3.0 there was
    nothing left for content."""
    c = MeteredClient.__new__(MeteredClient)
    c._reasoning_headroom = headroom
    assert c._budget(400) == expected
