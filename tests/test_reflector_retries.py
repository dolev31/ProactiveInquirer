"""One transient 429 ended a multi-hour search. The reflector was the only LLM caller with no
retry at all.

Every other caller has one. `litellm_client` runs an 8-attempt tenacity ladder spanning ~148s
(three 60s rate-limit windows) and treats RateLimitError as retryable. The tau2 user simulator
was given `num_retries=8` after a 429 killed a dialogue unit. `OpenAIReflector` posted once and
raised `SearchRefused`, which aborts `run_search` entirely.

OBSERVED: rung 0 completed two full generations of real work --

    gen 1  78ef9faf9641 <- 85ae43a5b538 mean=0.7556
    gen 2  d65a10d40818 <- 9d2ba1892159 mean=0.8763

-- and then died on `HTTP Error 429: Too Many Requests` from the reflector, discarding all of
it. Over a 9.5-hour search sharing a proxy with other sweeps, a 429 is not an exceptional
event; it is an expected one.

The ladder matches `litellm_client`'s so the two behave the same under the same outage. Only
transient statuses are retried: a 401 or a 404 is a configuration error that will not fix
itself, and retrying it for two minutes just delays a message the operator needs now.
"""

from __future__ import annotations

import urllib.error

import pytest

from pinq_train.rung0_gepa.search import REFLECT_MAX_ATTEMPTS, is_transient_http


def _err(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "msg", {}, None)  # type: ignore[arg-type]


def test_rate_limits_are_transient() -> None:
    assert is_transient_http(_err(429))


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_server_errors_are_transient(code: int) -> None:
    assert is_transient_http(_err(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(code: int) -> None:
    """A misconfigured key or URL will not fix itself; retrying only delays the message. The
    403 from the wrong base_url is exactly this case."""
    assert not is_transient_http(_err(code))


def test_a_connection_error_is_transient() -> None:
    assert is_transient_http(urllib.error.URLError("connection reset"))
    assert is_transient_http(TimeoutError("timed out"))


def test_the_ladder_spans_more_than_one_rate_limit_window() -> None:
    """The proxy's window is 60s. A ladder that fits inside one can be fully consumed while
    the quota is still exhausted -- matched to `litellm_client`'s 8."""
    assert REFLECT_MAX_ATTEMPTS >= 8


def test_a_transient_failure_then_success_returns_the_text() -> None:
    from pinq_train.rung0_gepa.search import call_with_retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _err(429)
        return "the prompt"

    assert call_with_retry(flaky, sleep=lambda _: None) == "the prompt"
    assert calls["n"] == 3


def test_a_permanent_failure_raises_immediately() -> None:
    from pinq_train.rung0_gepa.search import call_with_retry

    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _err(403)

    with pytest.raises(urllib.error.HTTPError):
        call_with_retry(bad, sleep=lambda _: None)
    assert calls["n"] == 1, "a 403 was retried"


def test_exhausting_the_ladder_reraises() -> None:
    from pinq_train.rung0_gepa.search import call_with_retry

    calls = {"n": 0}

    def always429():
        calls["n"] += 1
        raise _err(429)

    with pytest.raises(urllib.error.HTTPError):
        call_with_retry(always429, sleep=lambda _: None)
    assert calls["n"] == REFLECT_MAX_ATTEMPTS
