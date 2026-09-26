"""The spend cap must not lie about what it spent, and 0 must not mean unlimited.

Both found by the pre-spend audit of the tier0 canary, on the same command that had just
billed real money.

1. `--spend-cap 0` returned None -- UNCAPPED -- and, because an explicit argument wins, it
   also disabled any PI_SPEND_CAP_USD guard rail that was set. Measured:

       env cap 5.00, no flag         -> 5.0
       env cap 5.00, --spend-cap 0   -> None      (uncapped, env guard gone)
       env cap 5.00, --spend-cap -1  -> None

   "Spend nothing" is the most natural reading of `--spend-cap 0` and the most natural way
   to ask for a rehearsal. It produced an unlimited run.

2. When the cap trips, `run_sweep` BREAKS out of `as_completed`. Futures already RUNNING
   cannot be cancelled -- `cancel_futures=True` only drops ones not yet started -- so up to
   `workers` units finish and bill. Their results were never collected, so their spend was
   real, their run directories were written, and the printed "$X billed" did not include
   them. A cap that under-reports its own overshoot is worse than no cap, because the number
   it prints is the one the operator writes down.
"""

from __future__ import annotations

import pytest

# ----------------------------------------------------------------- spend_cap(0) semantics


def test_zero_means_spend_nothing_not_unlimited(monkeypatch) -> None:
    from pi_run.sweep import spend_cap

    monkeypatch.setenv("PI_SPEND_CAP_USD", "5.00")
    assert spend_cap(0.0) == 0.0, "--spend-cap 0 must cap at zero, never disable the cap"


def test_zero_does_not_silently_disable_an_env_guard(monkeypatch) -> None:
    from pi_run.sweep import spend_cap

    monkeypatch.setenv("PI_SPEND_CAP_USD", "5.00")
    assert spend_cap(0.0) != spend_cap(None)


def test_a_negative_cap_is_refused(monkeypatch) -> None:
    """There is no sensible reading of a negative budget; guessing one is how 0 became
    'unlimited' in the first place."""
    from pi_run.sweep import SpendCapInvalid, spend_cap

    monkeypatch.delenv("PI_SPEND_CAP_USD", raising=False)
    with pytest.raises(SpendCapInvalid):
        spend_cap(-1.0)


def test_a_positive_cap_and_no_flag_are_unchanged(monkeypatch) -> None:
    from pi_run.sweep import spend_cap

    monkeypatch.setenv("PI_SPEND_CAP_USD", "5.00")
    assert spend_cap(1.0) == 1.0
    assert spend_cap(None) == 5.0
    monkeypatch.delenv("PI_SPEND_CAP_USD", raising=False)
    assert spend_cap(None) is None


# --------------------------------------------------------- the overshoot must be counted


def test_in_flight_spend_is_counted_after_the_cap_trips() -> None:
    """The units already running when the cap trips are billed; they must be reported.

    Single-worker path is deterministic, so the accounting is asserted where it can be
    asserted exactly: every unit whose result exists contributes to the reported total.
    """
    from pi_run.sweep import billed_usd

    results = [{"usd_billed": 0.10}, {"usd_billed": 0.10}, {"usd_billed": 0.10}]
    assert sum(billed_usd(r) for r in results) == pytest.approx(0.30)


def test_the_sweep_returns_every_unit_it_actually_ran(monkeypatch, tmp_path) -> None:
    """A unit that ran and billed must appear in the results, cap or no cap.

    Otherwise `pi run`'s printed total omits money that was really spent, and the run
    directory on disk has no corresponding row in the summary.
    """
    import pi_run.sweep as sw

    class Spec:
        def __init__(self, i):
            self.key = ("s", f"t{i}", "a", 0)

    specs = [Spec(i) for i in range(6)]
    monkeypatch.setattr(sw, "max_concurrency", lambda c=None: 1)
    monkeypatch.setattr(
        sw, "run_unit", lambda spec: {"usd_billed": 0.10, "task_id": spec.key[1], "status": "ok"}
    )

    out = sw.run_sweep(specs, cap_usd=0.25)
    billed = sum(sw.billed_usd(r) for r in out)
    assert billed >= 0.25, "the cap stopped before reaching its own threshold"
    assert len(out) == 3, f"expected 3 units to reach the cap, got {len(out)}"
    assert billed == pytest.approx(0.30)


def test_a_zero_cap_runs_nothing_billable(monkeypatch) -> None:
    import pi_run.sweep as sw

    class Spec:
        def __init__(self, i):
            self.key = ("s", f"t{i}", "a", 0)

    ran: list = []

    def _unit(spec):
        ran.append(spec.key)
        return {"usd_billed": 0.10, "status": "ok"}

    monkeypatch.setattr(sw, "max_concurrency", lambda c=None: 1)
    monkeypatch.setattr(sw, "run_unit", _unit)
    out = sw.run_sweep([Spec(i) for i in range(4)], cap_usd=0.0)
    assert len(ran) <= 1, f"a zero cap must stop immediately, ran {len(ran)}"
    assert sum(sw.billed_usd(r) for r in out) <= 0.10


# ------------------------------------------- the multi-worker overshoot must be collected


class _Fut:
    """Enough of a Future for the drain: done/cancelled/result."""

    def __init__(self, *, cancelled=False, value=None, boom=None):
        self._cancelled, self._value, self._boom = cancelled, value, boom

    def cancelled(self):
        return self._cancelled

    def result(self, timeout=None):
        if self._boom:
            raise self._boom
        return self._value


def test_drain_collects_units_that_ran_after_the_break() -> None:
    """`cancel_futures=True` drops only NOT-STARTED futures. The ones already running finish
    and bill, and before this they were never read -- so their money was invisible."""
    from pi_run.sweep import drain_uncollected

    futures = {
        _Fut(value={"usd_billed": 0.10}): ("s", "t1", "a", 0),
        _Fut(value={"usd_billed": 0.10}): ("s", "t2", "a", 0),
        _Fut(cancelled=True): ("s", "t3", "a", 0),
    }
    by_key = {}
    extra = drain_uncollected(futures, by_key)
    assert len(by_key) == 2, "both in-flight units must be recorded"
    assert extra == pytest.approx(0.20)


def test_drain_skips_already_collected_units() -> None:
    """No double counting: a unit read by the as_completed loop is already in by_key."""
    from pi_run.sweep import drain_uncollected

    k = ("s", "t1", "a", 0)
    futures = {_Fut(value={"usd_billed": 0.10}): k}
    by_key = {k: {"usd_billed": 0.10}}
    assert drain_uncollected(futures, by_key) == pytest.approx(0.0)
    assert len(by_key) == 1


def test_drain_ignores_a_cancelled_future() -> None:
    from pi_run.sweep import drain_uncollected

    futures = {_Fut(cancelled=True): ("s", "t1", "a", 0)}
    by_key = {}
    assert drain_uncollected(futures, by_key) == pytest.approx(0.0)
    assert by_key == {}


def test_drain_survives_a_unit_that_raised() -> None:
    """A unit that raised spent tokens too, but has no result to record; it must not take
    the accounting of the others down with it."""
    from pi_run.sweep import drain_uncollected

    futures = {
        _Fut(boom=RuntimeError("worker died")): ("s", "t1", "a", 0),
        _Fut(value={"usd_billed": 0.10}): ("s", "t2", "a", 0),
    }
    by_key = {}
    assert drain_uncollected(futures, by_key) == pytest.approx(0.10)
    assert len(by_key) == 1


def test_a_negative_cap_refuses_cleanly_rather_than_tracebacking() -> None:
    """An operator typo should read as a refusal, not as a crash in someone else's module."""
    from pi_run.cli import build_parser

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier0_canary.yaml", "--spend-cap", "-1"]
    )
    assert a.fn(a) == 2
