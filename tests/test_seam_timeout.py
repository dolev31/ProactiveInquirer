"""The seam's 120s timeout was shorter than the episodes it waits for.

MEASURED over 5,641 recorded musique rollouts: median 64s, p90 223s, p99 379s, max 717s. The
120s default sat between the median and the p90, so roughly a quarter of rollouts exceeded it
-- and a rung 0 search issues thousands.

It surfaced as `TimeoutError: timed out` out of `client.rollout`, AFTER the seed generation had
scored (`gen 0  seed 85ae43a5b538 mean=0.9300`). The rollout it abandoned kept running
server-side and completed; only the client stopped waiting, so the work was paid for and
discarded.

900s covers the measured p99 with headroom and still bounds a genuinely hung request. It is a
CLIENT patience setting, not a budget: `unit_timeout_s` is what actually stops a runaway
episode server-side.
"""

from __future__ import annotations

from pinq_train.client import DEFAULT_SEAM_TIMEOUT_S, SeamClient


def test_the_default_covers_the_measured_p99() -> None:
    """p99 is 379s and the max seen is 717s."""
    assert DEFAULT_SEAM_TIMEOUT_S >= 720


def test_both_endpoints_get_it() -> None:
    c = SeamClient("http://s:1", rollout_url="http://r:2")
    assert c.ep.timeout == DEFAULT_SEAM_TIMEOUT_S
    assert c.rollout_ep.timeout == DEFAULT_SEAM_TIMEOUT_S


def test_it_is_still_overridable() -> None:
    c = SeamClient("http://s:1", timeout=30.0)
    assert c.ep.timeout == 30.0
    assert c.rollout_ep.timeout == 30.0


def test_the_env_var_overrides_the_default(monkeypatch) -> None:
    """A slow proxy is an operational condition, not a code change."""
    monkeypatch.setenv("PI_SEAM_TIMEOUT_S", "1234")
    assert SeamClient("http://s:1").ep.timeout == 1234.0


def test_garbage_in_the_env_falls_back_rather_than_crashing(monkeypatch) -> None:
    monkeypatch.setenv("PI_SEAM_TIMEOUT_S", "soon")
    assert SeamClient("http://s:1").ep.timeout == DEFAULT_SEAM_TIMEOUT_S


def test_an_explicit_argument_beats_the_env(monkeypatch) -> None:
    monkeypatch.setenv("PI_SEAM_TIMEOUT_S", "1234")
    assert SeamClient("http://s:1", timeout=7.0).ep.timeout == 7.0
