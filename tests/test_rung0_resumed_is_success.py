"""A RESUMED ROLLOUT IS A SUCCESS. Scoring it -inf makes rung 0 optimise nothing.

`SeamEvaluator.__call__` read `if roll.status != "ok": return TaskResult(reward=-inf)`. But
`run_unit` returns "resumed" whenever a SUCCESSFUL run for that exact run_id already exists --
and only then: a prior whose status was not "ok" is discarded and re-run, precisely so "resume
must NOT bake in a failure" (worker.py). So "resumed" means the artifacts exist and are
scoreable, and treating it as a failure is a bug.

IT IS FATAL, NOT COSMETIC, AND IT SCALES WITH HOW MUCH WORK YOU HAVE ALREADY DONE:

  * rung 0 re-evaluates candidates across generations -- the seed sits in the Pareto front and
    is re-scored every time -- so the first evaluation is "ok" and every later one is
    "resumed" -> -inf.
  * on a repository that already holds runs for those tasks (this one has thousands of musique
    runs), even the FIRST evaluation of the seed is "resumed" -> -inf.

Observed exactly that: `gen 0  seed 85ae43a5b538 mean=-inf`, with the underlying runs all
`status: ok` on disk and both servers returning HTTP 200. The search would have run its full
budget selecting between candidates that all scored -inf.
"""

from __future__ import annotations

from pinq.wire import RolloutResponse
from pinq_train.rung0_gepa.search import rollout_succeeded


def _resp(status: str) -> RolloutResponse:
    return RolloutResponse(
        run_id="r1",
        status=status,
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_prompted",
        seed=0,
    )


def test_ok_is_a_success() -> None:
    assert rollout_succeeded(_resp("ok"))


def test_resumed_is_a_success() -> None:
    """The whole point: a successful prior run is exactly what resume returns."""
    assert rollout_succeeded(_resp("resumed"))


def test_a_real_failure_is_not() -> None:
    for bad in ("error", "timeout", "refused", "", "skipped"):
        assert not rollout_succeeded(_resp(bad)), bad


def test_the_check_is_not_a_substring_match() -> None:
    """'not ok' must not pass because it contains 'ok'."""
    assert not rollout_succeeded(_resp("not ok"))
    assert not rollout_succeeded(_resp("broken"))
