"""An upstream infrastructure failure that arrives as AuthenticationError must be retried.

WHY. `_retryable()` deliberately excludes `litellm.AuthenticationError`, and for a real credential
failure that is right: a wrong key does not become right in 148 seconds, and burning the ladder on it
hides the bug behind a delay. But the gateway in front of this project returns 401
AuthenticationError for failures of its OWN infrastructure. Measured verbatim on 2026-09-22:

    401 - {'error': {'message': 'Authentication Error, Error in connector: Error querying the
      database: FATAL: ...'}}
    Authentication Error, All connection attempts failed. Received Model Group=aws/claude-sonnet-5

Both are transport-shaped: the gateway could not reach its own key database, or could not connect at
all. Neither says anything about our credentials -- probed direct in the same minute, both keys
returned 200 on both frozen roles. Three outages in one campaign produced 164 AuthenticationError
units that got ZERO retries, because the exception CLASS cannot distinguish "your key is wrong" from
"our database is down" and only the MESSAGE can.

This is the same hazard already recorded for the 403 permission wall, which arrives in the same
exception class as a transient. The rule that follows: classify on the body when the class is
ambiguous, and keep failing fast on a genuine credential error.
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")

from pinq_adapters.llm.litellm_client import _retryable, is_infrastructure_auth_error  # noqa: E402

INFRA_BODIES = [
    "Authentication Error, Error in connector: Error querying the database: FATAL: sorry",
    "Authentication Error, All connection attempts failed. Received Model Group=aws/claude-sonnet-5",
    "AuthenticationError: OpenAIException - Authentication Error, All connection attempts failed.",
]

CREDENTIAL_BODIES = [
    "Incorrect API key provided: sk-...h3tw. You can find your API key at ...",
    "Invalid API key",
    "401 Unauthorized",
    "team not allowed to access model. This team can only access models=['azure/Kimi']",
]


@pytest.mark.parametrize("body", INFRA_BODIES)
def test_infrastructure_auth_errors_are_classified_retryable(body: str) -> None:
    """The gateway's own connector/database failures, verbatim from the outage."""
    assert is_infrastructure_auth_error(litellm.AuthenticationError(body, "openai", "m")) is True


@pytest.mark.parametrize("body", CREDENTIAL_BODIES)
def test_genuine_credential_errors_still_fail_fast(body: str) -> None:
    """A wrong key, and a permission wall, must NOT burn the retry ladder."""
    assert is_infrastructure_auth_error(litellm.AuthenticationError(body, "openai", "m")) is False


def test_a_non_auth_exception_is_not_claimed_by_this_predicate() -> None:
    """The predicate must speak only about AuthenticationError, or it would widen silently."""
    assert is_infrastructure_auth_error(ValueError("Error querying the database")) is False


def test_the_retryable_tuple_still_excludes_bare_authentication_error() -> None:
    """The fix must NOT put AuthenticationError in the retryable tuple wholesale.

    Doing that would retry a wrong key eight times, which is the behaviour `_retryable`'s own
    docstring rejects. The class stays out; the predicate decides per-exception.
    """
    assert litellm.AuthenticationError not in _retryable()
