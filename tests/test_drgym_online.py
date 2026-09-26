"""The DRGym key is useless unless the runner can be told to go online.

`load_suite` hardcoded `DrGymSuite(..., offline=True)`, and `offline` has no env override,
so with a valid DRGYM_API_KEY installed and verified the runner still replayed a cache it
does not have: a 2-task probe on 2026-08-28 failed with

    SearchCacheMiss: "offline replay: no cached search for corpus=fineweb k=3 query=..."

on every unit whose policy actually searched.

OFFLINE STAYS THE DEFAULT. A confirmatory sweep must replay a recorded cache rather than
re-query a hosted index whose contents can move under it -- that is a reproducibility
property, not a convenience. Going online is therefore an EXPLICIT opt-in, and the miss
message names it, so the next person does not have to read the adapter to find the switch.
"""

from __future__ import annotations


def test_offline_is_still_the_default() -> None:
    from pi_run.worker import drgym_offline

    assert drgym_offline({}) is True
    assert drgym_offline({"DRGYM_API_KEY": "x" * 40}) is True, (
        "holding a key must not silently change what a sweep replays"
    )


def test_the_opt_in_turns_it_online() -> None:
    from pi_run.worker import drgym_offline

    assert drgym_offline({"PI_DRGYM_ONLINE": "1", "DRGYM_API_KEY": "x" * 40}) is False


def test_going_online_without_a_key_is_refused() -> None:
    """Better to refuse than to run every unit into a 401 one search at a time."""
    import pytest

    from pi_run.worker import DrGymOnlineWithoutKey, drgym_offline

    with pytest.raises(DrGymOnlineWithoutKey):
        drgym_offline({"PI_DRGYM_ONLINE": "1"})


def test_falsey_spellings_stay_offline() -> None:
    from pi_run.worker import drgym_offline

    for v in ("0", "", "false", "no"):
        assert drgym_offline({"PI_DRGYM_ONLINE": v, "DRGYM_API_KEY": "k" * 40}) is True, v


def test_load_suite_consults_it() -> None:
    """A toggle load_suite ignores is decoration."""
    import inspect

    from pi_run.worker import load_suite

    src = inspect.getsource(load_suite)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "drgym_offline(" in code, "load_suite still hardcodes offline"
    assert "offline=True)" not in code, "the hardcoded default is still there"


def test_the_cache_miss_names_the_remedy() -> None:
    """The probe's error told the operator nothing about how to fix it."""
    import inspect

    from pinq_adapters.drgym import suite as m

    src = inspect.getsource(m)
    assert "PI_DRGYM_ONLINE" in src, "SearchCacheMiss does not name the opt-in"
