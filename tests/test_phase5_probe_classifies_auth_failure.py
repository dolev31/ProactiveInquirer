"""A 401 from the gateway must be refused, not retried as transient.

`scripts/tau2_campaign/phase5_launch.sh`'s model-probe loop classifies three ways:

    404 / "does not exist" / NotFoundError   -> ABSENT, exit 9, no retry
    403 / "not allowed to access"            -> PERMISSION WALL, exit 12, no retry
    anything else                            -> "transient: no status/5xx/connection", retry 4x

MEASURED 2026-09-20: both frozen roles the campaign requires returned HTTP 401 from the upstream
gateway -- `litellm.AuthenticationError: Incorrect API key provided` for `aws/claude-sonnet-5` and
for `aws/gpt-oss-120b` -- while a local vLLM pin on the same proxy returned 200. So the fault was
upstream authentication, not absence, not a permission wall, and not transient.

A 401 fell through to the transient branch. Two costs, and the second is the real one. It burns
four retries and twenty seconds per role on a condition that cannot clear by waiting, on every arm.
And it then exits with the ABSENT code and the message "did not answer after 4 attempts ... NOT
evidence the checkpoint is gone" -- which names the wrong cause. An invalid key is exactly as
permanent as a permission wall, which this script already refuses immediately for that reason.

The branch is asserted here against the real body text the gateway returns, so a future reader can
see which string was classified rather than trusting a paraphrase.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tau2_campaign" / "phase5_launch.sh"

#: Verbatim from the 2026-09-20 probe of `aws/gpt-oss-120b` through the proxy at :4000.
REAL_401_BODY = (
    '{"error":{"message":"litellm.AuthenticationError: AuthenticationError: '
    "OpenAIException - Incorrect API key provided: sk-baUHA*************m1Sg. "
    'Received Model Group=aws/gpt-oss-120b"}}'
)


def _source() -> str:
    assert SCRIPT.is_file(), f"missing launcher: {SCRIPT}"
    return SCRIPT.read_text()


def test_the_launcher_has_an_authentication_branch() -> None:
    src = _source()
    assert re.search(r'"\$code"\s*=\s*"401"', src), (
        "phase5_launch.sh has no 401 branch, so an invalid gateway key is classified as "
        "transient and retried four times before exiting with the ABSENT code."
    )


def test_the_authentication_branch_refuses_rather_than_retries() -> None:
    """It must exit, and with neither the ABSENT nor the PERMISSION code."""
    src = _source()
    m = re.search(r'if \[ "\$code" = "401" \].*?\bfi\b', src, re.S)
    assert m, "no 401 block to inspect"
    block = m.group(0)
    assert "exit" in block, "the 401 branch does not exit; it would fall through to retries"
    codes = set(re.findall(r"exit (\d+)", block))
    assert codes, "the 401 branch exits with no status"
    assert codes.isdisjoint({"9", "12"}), (
        f"the 401 branch reuses exit {codes & {'9', '12'}}: an auth fault must be "
        "distinguishable from ABSENT (9) and PERMISSION WALL (12) by exit code alone."
    )


def test_the_branch_matches_the_body_the_gateway_actually_returns() -> None:
    """Not only the status code: a proxy can return 200-with-error-body or a bare 500."""
    src = _source()
    m = re.search(r'if \[ "\$code" = "401" \].*?\bfi\b', src, re.S)
    assert m, "no 401 block to inspect"
    pats = re.findall(r'grep -qi "([^"]+)"', m.group(0))
    assert pats, "the 401 branch matches no body text, only the status code"
    alts = [a.lower() for p in pats for a in p.split(r"\|")]
    hit = [a for a in alts if a and a in REAL_401_BODY.lower()]
    assert hit, (
        f"none of the 401 branch's body patterns {alts} match the real gateway body:\n"
        f"{REAL_401_BODY}"
    )


def test_absent_and_permission_branches_are_untouched() -> None:
    """Non-vacuity: the two existing refusals must still be there, with their own codes."""
    src = _source()
    assert re.search(r'"\$code"\s*=\s*"404"', src) and "exit 9" in src
    assert re.search(r'"\$code"\s*=\s*"403"', src) and "exit 12" in src
