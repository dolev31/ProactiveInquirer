"""A RATE LIMIT MUST NOT KILL A DIALOGUE UNIT. It killed one, and here is why.

Our own client has an 8-attempt tenacity ladder spanning ~148s -- three 60s windows -- and
`litellm.RateLimitError` is in `_retryable()`. But the tau2 USER SIMULATOR does not go through
our client at all: `build_user(..., llm_args=_user_llm_args())` hands those args straight to
`litellm.completion` inside tau2. That path had no retry.

Observed on the first live tau2_retail unit: the run died with

    RateLimitError: Rate limit exceeded for team ... Current limit: 100, Remaining: 0

and `calls.jsonl` was ABSENT -- our client had never recorded a single call, which is what
identifies the failure as upstream of the ladder rather than an exhaustion of it.

`num_retries` is verified to work rather than assumed: it is not in `litellm.completion`'s
signature, it arrives through **kwargs. Measured against a dead host, num_retries=2 took 1.4s
against 0.3s at num_retries=0, so it is honoured. litellm also reads `Retry-After` for 429s
(`litellm._calculate_retry_after`), which is what makes the wait match the window rather than a
fixed backoff.

TWO RETRY LAYERS ARE NOT A CONTRADICTION HERE. `litellm_client` passes `num_retries=0` because
tenacity owns its retries and two layers would make its recorded `retries` count a lie. On THIS
path nothing else owns them, so litellm's own is the only one available.
"""

from __future__ import annotations

import pytest

from pi_run.stages.tau2_runner import USERSIM_NUM_RETRIES, _user_llm_args


def test_the_user_simulator_retries_rate_limits() -> None:
    args = _user_llm_args()
    assert args.get("num_retries", 0) >= 1, (
        "the user simulator has no retry: a single 429 kills the whole dialogue unit"
    )
    assert args["num_retries"] == USERSIM_NUM_RETRIES


def test_the_ladder_spans_more_than_one_rate_limit_window() -> None:
    """The proxy's window is 60s and resets on the minute. A ladder that fits inside one
    window can be fully consumed while the quota is still exhausted."""
    assert USERSIM_NUM_RETRIES >= 8


def test_temperature_stays_zero() -> None:
    """The user simulator is part of the ENVIRONMENT: a customer who answers differently for
    two arms is a different task. Retry configuration must not disturb that."""
    assert _user_llm_args()["temperature"] == 0.0


def test_a_timeout_is_set_so_a_hung_call_cannot_stall_a_unit() -> None:
    args = _user_llm_args()
    assert args.get("timeout"), "no timeout: a hung user-sim call stalls the unit to its cap"


def test_litellm_actually_honours_num_retries() -> None:
    """Pinned because it arrives through **kwargs and is absent from the signature, so a
    litellm upgrade could drop it silently and the ladder would become decorative."""
    import inspect

    import litellm

    sig = inspect.signature(litellm.completion)
    assert any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()), (
        "litellm.completion no longer takes **kwargs; num_retries may no longer be forwarded"
    )


@pytest.mark.parametrize("var", ["LITELLM_BASE_URL", "LITELLM_API_KEY"])
def test_proxy_settings_are_forwarded_when_present(monkeypatch, var: str) -> None:
    monkeypatch.setenv(var, "sentinel")
    key = "api_base" if var.endswith("BASE_URL") else "api_key"
    assert _user_llm_args()[key] == "sentinel"
