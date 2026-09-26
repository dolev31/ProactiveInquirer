"""Rung 0 needs TWO servers, and `SeamClient` could only address one.

THE FIREWALL MAKES ONE PROCESS IMPOSSIBLE, and says so:

  * `pi_run/serve/rollout.py` -- "If someone hosts both endpoints in one process, this endpoint
    must still refuse to run, because a rollout worker that can read gold makes every number it
    produces unusable. `assert_firewall()` is called before anything else and is not catchable
    into a warning."  So a gold-set server REFUSES /rollout.
  * `/score` without gold returns 503 by design: "this process cannot score at all:
    PI_GOLD_ROOT is unset."

So a scoring server cannot roll out and a rollout server cannot score -- and `SeamClient` took
a single `base_url` and posted all four endpoints to it. Whichever way it was pointed, half its
calls failed. That is consistent with rung 0 never having run: every one of 5,291 recorded runs
carries `prompt_variant_id = "v1"`, the shipped template.

`rollout_url` defaults to `base_url`, so every existing caller and test is unchanged; only a
caller that needs the split has to know about it.
"""

from __future__ import annotations

import pytest

from pinq_train.client import SeamClient


def test_one_url_still_serves_everything_by_default() -> None:
    """Unchanged for every existing caller: the split is opt-in."""
    c = SeamClient("http://s:1")
    assert c.ep.url("/score") == "http://s:1/score"
    assert c.rollout_ep.url("/rollout") == "http://s:1/rollout"


def test_the_rollout_url_can_be_pointed_at_a_second_server() -> None:
    c = SeamClient("http://score:1", rollout_url="http://roll:2")
    assert c.ep.url("/score") == "http://score:1/score"
    assert c.rollout_ep.url("/rollout") == "http://roll:2/rollout"


def test_healthz_follows_the_score_server() -> None:
    """`healthz` is how a caller checks `gold_root_set`, which is a property of the SCORING
    server -- reporting the rollout server's health there would say the opposite of what the
    caller is asking."""
    c = SeamClient("http://score:1", rollout_url="http://roll:2")
    assert c.ep.url("/healthz") == "http://score:1/healthz"


def test_retrieve_follows_the_ROLLOUT_server() -> None:
    """Retrieval is the rollout side of the seam: it touches the corpus, not gold. Sending it
    to the scoring server would ask a gold-bearing process to serve corpus text."""
    c = SeamClient("http://score:1", rollout_url="http://roll:2")
    assert c.rollout_ep.url("/retrieve") == "http://roll:2/retrieve"


def test_the_trainer_firewall_still_fires(monkeypatch) -> None:
    """Splitting the URLs must not create a way around the check that the TRAINER itself
    cannot read gold."""
    from pinq_train.client import TrainerFirewallError

    monkeypatch.setenv("PI_GOLD_ROOT", "/somewhere")
    with pytest.raises(TrainerFirewallError):
        SeamClient("http://score:1", rollout_url="http://roll:2")


def test_a_trailing_slash_does_not_double_up() -> None:
    c = SeamClient("http://score:1/", rollout_url="http://roll:2/")
    assert c.ep.url("/score") == "http://score:1/score"
    assert c.rollout_ep.url("/rollout") == "http://roll:2/rollout"
