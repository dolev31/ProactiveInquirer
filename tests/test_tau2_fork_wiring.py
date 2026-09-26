"""The driver must actually USE the prefix: load it, fork the task, grade against the fork,
and record how much of the dialogue was inherited.

Each of these was a separate silent failure on the branch before it was fixed there, and each
produces a run that looks completely normal:

  * prefix never loaded          -> a fresh rollout carrying fork identity, in the fork table
  * task forked, gold not        -> every continuation graded against a world it was never in
  * `n_prefix_user_turns` unset  -> `follow_ups` charges the fork for the prefix's user turns
  * `foreign_*` off the manifest -> two cut points of one task share a run_id and a directory

The Orchestrator is faked here. A real one needs an environment, a user simulator and an LLM,
and none of those are available to a test -- but the four wirings above are all decided before
`orch.run()` is reached, so faking it tests the thing at issue rather than around it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pi_run.manifest import build_manifest, manifest_to_dict
from pi_run.stages import tau2_runner

pytest.importorskip("tau2.data_model.tasks", reason="tau2 is not installed")


# ------------------------------------------------------------------- the manifest carries it


def _manifest(**over):
    kw: dict[str, Any] = dict(
        suite_id="tau2_retail",
        task_id="55",
        arm_id="inquirer_prompted",
        policy_id="inquirer_prompted",
        seed=0,
        corpus_hash="c",
        budget_cap=16,
        max_turns=16,
        word_cap=180,
        code_version="abc",
        dirty=False,
    )
    kw.update(over)
    return build_manifest(**kw)


def test_the_fork_reaches_the_manifest_and_the_file_it_writes():
    """`build_manifest` is the only way a runner can put these on disk, and `manifest_to_dict`
    is the only thing `pi compact` and `report_forks.py` ever read."""
    m = _manifest(foreign_trace_sha="d" * 64, foreign_prefix_k=34)
    assert m.foreign_trace_sha == "d" * 64
    assert m.foreign_prefix_k == 34
    d = manifest_to_dict(m)
    assert d["foreign_trace_sha"] == "d" * 64
    assert d["foreign_prefix_k"] == 34


def test_an_unforked_manifest_writes_null_not_empty_string():
    """`""` is not popped by `ids.semantic_hash` and would rename every non-fork run on disk.
    See tests/test_run_id_stability.py::test_an_empty_string_is_not_the_same_as_absent."""
    d = manifest_to_dict(_manifest())
    assert d["foreign_trace_sha"] is None
    assert d["foreign_prefix_k"] is None


def test_two_cut_points_of_one_task_are_two_run_directories():
    """The whole reason the fields are in SEMANTIC_FIELDS, exercised through the real builder."""
    a = _manifest(foreign_trace_sha="d" * 64, foreign_prefix_k=12)
    b = _manifest(foreign_trace_sha="d" * 64, foreign_prefix_k=21)
    plain = _manifest()
    assert a.run_id != b.run_id
    assert a.run_id != plain.run_id
    assert plain.run_id == _manifest().run_id


# ------------------------------------------------------------------------- _simulate forks


class _FakeEnv:
    """Only what `_simulate` asks of an environment before `orch.run()`."""

    def get_tools(self):
        return []

    def get_policy(self):
        return ""


class _FakeSuite:
    domain = "retail"
    suite_id = "tau2_retail"

    def __init__(self, task):
        self._task = task

    def environment(self, tid):
        return _FakeEnv()

    def tau2_task_object(self, tid):
        return self._task


class _FakeSpec:
    suite_id = "tau2_retail"
    task_id = "55"
    max_turns = 16
    k = 4
    seed = 0
    branch_turn_idx = None
    branch_seed = None
    foreign_trace_sha = None
    foreign_prefix_k = None


@pytest.fixture
def fake_orchestrator(monkeypatch):
    """Capture the task the Orchestrator was handed, and stop before anything runs."""
    import tau2.orchestrator.orchestrator as orch_mod

    seen: dict[str, Any] = {}

    class _Orch:
        def __init__(self, **kw):
            seen.update(kw)
            self.agent_state = object()

        def run(self):
            return object()

    monkeypatch.setattr(orch_mod, "Orchestrator", _Orch)
    monkeypatch.setattr(tau2_runner, "make_driver_agent_class", lambda: lambda **kw: object())
    return seen


def _task_with(prefix_len: int = 0):
    from tau2.data_model.tasks import Task, UserScenario

    return Task(id="55", user_scenario=UserScenario(instructions="be a customer"))


def _user_prefix(n: int = 2):
    from tau2.data_model.message import AssistantMessage, UserMessage

    return [
        AssistantMessage(role="assistant", content="Hi!"),
        UserMessage(role="user", content="I need help."),
        AssistantMessage(role="assistant", content="Sure."),
        UserMessage(role="user", content="Cancel W123 please."),
    ][: 2 * n]


def test_the_orchestrator_is_handed_the_forked_task(fake_orchestrator):
    task = _task_with()
    prefix = _user_prefix()
    tau2_runner._simulate(
        _FakeSuite(task), _FakeSpec(), lambda *a, **k: {}, user=object(), prefix=prefix
    )
    handed = fake_orchestrator["task"]
    assert handed is not task, "the original task object must not be mutated"
    assert len(handed.initial_state.message_history) == len(prefix)


def test_an_unforked_simulation_hands_over_the_task_unchanged(fake_orchestrator):
    task = _task_with()
    tau2_runner._simulate(
        _FakeSuite(task), _FakeSpec(), lambda *a, **k: {}, user=object(), prefix=None
    )
    assert fake_orchestrator["task"] is task
    assert task.initial_state is None


def test_the_simulation_returns_the_task_it_actually_ran(fake_orchestrator):
    """`_reward_of` must grade against THIS object. Re-fetching the original there seeds the
    gold environment from a world the policy was never in, and every fork scores 0 for a
    reason that has nothing to do with the policy."""
    task = _task_with()
    prefix = _user_prefix()
    out = tau2_runner._simulate(
        _FakeSuite(task), _FakeSpec(), lambda *a, **k: {}, user=object(), prefix=prefix
    )
    assert len(out) == 3, "_simulate returns (sim, agent_state, task)"
    assert out[2] is fake_orchestrator["task"]


# -------------------------------------------------------------- gold is seeded from the fork


def test_the_grader_is_seeded_from_the_forked_task(monkeypatch):
    """`EnvironmentEvaluator.calculate_reward` replays the transcript into a GOLD environment
    seeded from `task.initial_state` and then applies every gold action on top. Handing it the
    original while the Orchestrator ran the fork is the two-worlds bug."""
    import tau2.evaluator.evaluator_env as ev

    seen: dict[str, Any] = {}

    class _Ev:
        @staticmethod
        def calculate_reward(constructor, task, messages, **kw):
            seen["task"] = task
            raise RuntimeError("stop here; the task is all this test needs")

    monkeypatch.setattr(ev, "EnvironmentEvaluator", _Ev)

    from tau2.data_model.tasks import Task, UserScenario

    forked = Task(id="55", user_scenario=UserScenario(instructions="x"))
    original = Task(id="55", user_scenario=UserScenario(instructions="x"))

    class _Suite:
        domain = "retail"

        def tau2_task_object(self, tid):
            return original

        def task_record(self, tid):
            return None

        def env_kwargs(self, tid):
            return {}

        def environment(self, tid):
            return object()

    class _Crit:
        reward_basis = ()

    forked.evaluation_criteria = _Crit()
    original.evaluation_criteria = _Crit()

    class _Sim:
        messages = ()

    native, err = tau2_runner._reward_of(_Sim(), _Suite(), "55", task=forked)
    assert seen["task"] is forked, "the grader must see the task the dialogue actually ran"
    assert native == {} and "stop here" in err


def test_without_a_fork_the_grader_still_uses_the_suites_own_task(monkeypatch):
    """The default path is unchanged: `task=None` means 'ask the suite', exactly as before."""
    import tau2.evaluator.evaluator_env as ev

    seen: dict[str, Any] = {}

    class _Ev:
        @staticmethod
        def calculate_reward(constructor, task, messages, **kw):
            seen["task"] = task
            raise RuntimeError("stop")

    monkeypatch.setattr(ev, "EnvironmentEvaluator", _Ev)
    from tau2.data_model.tasks import Task, UserScenario

    original = Task(id="55", user_scenario=UserScenario(instructions="x"))

    class _Crit:
        reward_basis = ()

    original.evaluation_criteria = _Crit()

    class _Suite:
        domain = "retail"

        def tau2_task_object(self, tid):
            return original

        def task_record(self, tid):
            return None

        def env_kwargs(self, tid):
            return {}

        def environment(self, tid):
            return object()

    class _Sim:
        messages = ()

    tau2_runner._reward_of(_Sim(), _Suite(), "55")
    assert seen["task"] is original


# ------------------------------------------------------- the status file records the prefix


def test_the_unit_loads_the_prefix_and_records_its_user_turns():
    """Wiring inside `run_tau2_unit`, which cannot be run here without an environment and an
    LLM. Asserted on the source in the style `test_tau2_fork_reaches_the_loop` established for
    the same reason -- the alternative is not testing it at all.

    The three lines are one mechanism: load it, hand it to the simulation, record how much of
    the dialogue it accounted for so `follow_ups` can subtract it.
    """
    import inspect

    src = inspect.getsource(tau2_runner.run_tau2_unit)
    assert "load_prefix(spec)" in src
    assert "prefix=prefix" in src
    assert '"n_prefix_user_turns": count_user_turns(prefix)' in src
    assert "foreign_trace_sha=getattr(spec" in src
    assert "foreign_prefix_k=getattr(spec" in src


def test_the_status_file_is_json_serialisable_with_the_new_key():
    """`n_prefix_user_turns` is written with `json.dumps`, so an int is required -- a numpy
    integer or a bare message list would take the whole status write down with it."""
    assert json.dumps({"n_prefix_user_turns": tau2_runner.count_user_turns(_user_prefix())})
