"""tau2 RETAIL, GRADING SIDE: the domain it is graded in, and what `EnvCall.ok` means.

WHY A SEPARATE FILE FROM `test_adapters_tau2.py`
    That file is banking's adapter: its fixtures are the 698 knowledge documents and its
    subject is the doc->tool edge. Everything here is about the SECOND domain being graded as
    itself, which is a property of the runner and the actuator rather than of either corpus.

THE TWO DEFECTS PINNED HERE, BOTH SILENT
    * `tau2_runner` routed both suites through one path and then hardcoded banking's domain
      constant, so every retail rollout was graded against the BANKING database. With no
      OPENAI_API_KEY the banking build raises and `tau_reward` is merely ABSENT; with one set
      it would have been present and WRONG, which is worse. Measured before the fix:
      `_reward_of(<empty transcript>, Tau2RetailSuite(), "0")` returned `({}, "OpenAIError:
      Missing credentials ...")` -- that error is banking's document-embedding index being
      built, and retail's own environment constructs offline in the same process.
    * `Tau2Actuator._call` canonicalized the tool result INSIDE the try that guards the
      environment call, so a result `canon` refuses -- every retail tool returns a pydantic
      model -- landed in the `except` and was recorded `ok=False` AFTER the environment had
      already executed and mutated. Measured on retail task 0: 4 of the 5 gold actions,
      including the mutating `exchange_delivered_order_items`, recorded as failures.

WHAT `ok` MEANS, AND WHAT IT DOES NOT
    `ok` is "the environment executed this call". Whether the value it returned can be
    written down reproducibly is a DIFFERENT property, and conflating them is what let a
    successful mutation be counted in `n_failed_calls`, deleted from the synthesized
    trajectory, and replaced -- in `_results` -- by the text of a serialization error.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from pinq_adapters.tau2._probe import DOMAIN, available
from pinq_adapters.tau2.actuator import Tau2Actuator
from pinq_adapters.tau2.retail_suite import RETAIL_DOMAIN, Tau2RetailSuite
from pinq_adapters.tau2.suite import Tau2Suite

tau2_only = pytest.mark.skipif(not available()[0], reason=available()[1])


# --------------------------------------------------------------- defect 1: the domain


def test_each_suite_declares_the_domain_it_is_graded_in() -> None:
    """ONE lookup, on the suite, rather than an if-chain at every grading site.

    A ClassVar rather than a module constant because there are two suites and only one
    `tau2_runner`: the runner has to be able to ask which domain it is driving, and the only
    thing that knows is the suite it was handed.
    """
    assert Tau2Suite.domain == DOMAIN == "banking_knowledge"
    assert Tau2RetailSuite.domain == RETAIL_DOMAIN == "retail"
    assert Tau2Suite.domain != Tau2RetailSuite.domain


@tau2_only
def test_the_grader_builds_the_suites_own_domain() -> None:
    """THE DEFECT, as a recorded argument rather than as a downstream symptom.

    `_reward_of` replays the transcript into a FRESH environment built by a constructor it
    hands to upstream. Which domain that constructor names is the whole question, and it is
    observable without a key: the recorder captures the name and then delegates.
    """
    import tau2.runner as tau2_runner_mod

    from pi_run.stages import tau2_runner

    seen: list[str] = []
    real = tau2_runner_mod.build_environment

    def recording(domain, *a, **kw):
        seen.append(str(domain))
        return real(domain, *a, **kw)

    suite = Tau2RetailSuite()
    sim = type("Sim", (), {"messages": []})()
    with contextlib.redirect_stdout(io.StringIO()):
        tau2_runner_mod.build_environment = recording
        try:
            tau2_runner._reward_of(sim, suite, "0")
        finally:
            tau2_runner_mod.build_environment = real

    assert seen, "the grader never built an environment at all"
    assert set(seen) == {RETAIL_DOMAIN}, (
        f"a retail rollout was graded in {sorted(set(seen))}; every tau_reward it produced "
        "is a measurement of the wrong database"
    )


@tau2_only
def test_retail_grades_offline_and_reports_a_db_reward() -> None:
    """The consequence, measured end to end and with no network.

    Retail's environment needs no embedding index; banking's default one does, so grading
    retail in banking's domain raised `OpenAIError: Missing credentials` and `native` came
    back EMPTY with the failure recorded in `reward_error`. An absent endpoint is the
    friendlier half of this bug -- with a key set the same code path returns a number graded
    against the wrong DB -- but it is the half that is testable for free.

    An empty transcript did not reach the gold state, so 0.0 here is a MEASUREMENT and not a
    default; what is being asserted is that a measurement happened at all.
    """
    from pi_run.stages.tau2_runner import _reward_of

    suite = Tau2RetailSuite()
    sim = type("Sim", (), {"messages": []})()
    with contextlib.redirect_stdout(io.StringIO()):
        native, err = _reward_of(sim, suite, "0")

    assert err == "", f"retail could not be graded: {err}"
    assert "db_reward" in native, f"no DB check was produced: {dict(native)}"
    assert native["db_reward"] == 0.0, "an empty transcript cannot have reached the gold state"


@tau2_only
def test_the_retail_suite_hands_out_an_actuator_bound_to_retail() -> None:
    """`worker.run_unit` calls `suite.actuator(tid)` unguarded. Retail had no such method at
    all -- it was saved only by `view()` raising earlier on the flat path, which is a
    coincidence of ordering and not a contract."""
    suite = Tau2RetailSuite()
    with contextlib.redirect_stdout(io.StringIO()):
        act = suite.actuator("0")
    assert isinstance(act, Tau2Actuator)
    assert act._domain == RETAIL_DOMAIN


@tau2_only
def test_the_retail_suite_satisfies_the_task_suite_protocol() -> None:
    from pinq.protocols import TaskSuite

    assert isinstance(Tau2RetailSuite(), TaskSuite)


# ------------------------------------------------- defect 2: ok means "the env executed it"


class _Payload:
    """A tool result `canon` refuses, with no pydantic dependency. Retail's real results are
    pydantic models; what matters to the actuator is only that they are not `str`."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _Model(_Payload):
    """The pydantic shape, duck-typed: `model_dump` is the only method the adapter uses."""

    def model_dump(self, mode: str = "python") -> dict:
        return {"tag": self.tag}


class _ObjEnv:
    """Returns objects rather than strings, which is what retail's environment does."""

    def __init__(self, payload=None, fail: frozenset[str] = frozenset()) -> None:
        self.calls: list[str] = []
        self._fail = fail
        self._payload = payload or (lambda name: _Payload(name))
        self.db = "db0"

    def make_tool_call(self, name, *, requestor="assistant", **kwargs):
        if name in self._fail:
            raise RuntimeError(f"{name} refused")
        self.calls.append(name)
        if name == "mutate":
            self.db = "db1"
        return self._payload(name)

    def _is_mutating_tool(self, name):
        return name == "mutate"

    def get_db_hash(self):
        return self.db

    def get_user_db_hash(self):
        return "udb0"


def test_a_call_the_environment_executed_is_ok_even_when_its_result_is_not_a_string() -> None:
    """The invariant. `ok` is a fact about the ENVIRONMENT, not about our serializer.

    The mutating call is the one that shows why: by the time canonicalization is attempted
    the DB has already changed, so recording `ok=False` describes a world that does not
    exist -- and `_compute_reward` then synthesizes a trajectory in which the mutation never
    happened.
    """
    env = _ObjEnv()
    act = Tau2Actuator(env)
    calls = act.execute([{"name": "read"}, {"name": "mutate"}], turn_idx=0)

    assert [c.ok for c in calls] == [True, True]
    assert env.calls == ["read", "mutate"], "the environment executed both"
    assert env.db == "db1", "and the mutation landed"
    assert sum(1 for c in calls if not c.ok) == 0


def test_the_raw_result_survives_a_serializer_that_could_not_take_it() -> None:
    """`_results` feeds `_trajectory()`, which renders each result with the ENVIRONMENT's own
    `to_json_str` -- the same bytes `Environment.get_response` produced, which is what
    `set_state` compares against. Overwriting the object with the text of our own
    canonicalization error replaced every retail tool result with `Error: <exc>` in the
    trajectory upstream regrades."""
    act = Tau2Actuator(_ObjEnv())
    act.execute([{"name": "read"}], turn_idx=0)

    (_name, _kw, _req, raw, ok, _snap) = act._results[0]
    assert ok is True
    assert isinstance(raw, _Payload), f"the result was replaced by {raw!r}"
    assert raw.tag == "read"


def test_a_call_that_did_not_execute_is_still_recorded_as_a_failure() -> None:
    """The other direction, so the fix cannot be "ok is always True". A call the environment
    refused never happened, and its recorded result is the error, exactly as before."""
    act = Tau2Actuator(_ObjEnv(fail=frozenset({"read"})))
    (call,) = act.execute([{"name": "read"}], turn_idx=0)

    assert call.ok is False
    (_n, _k, _r, raw, ok, _snap) = act._results[0]
    assert ok is False and raw == "read refused"


def test_two_different_results_do_not_share_one_digest() -> None:
    """`result_digest` is the only trace of what a call returned that the log is allowed to
    keep -- the payload is a customer record. Every retail result hashing to the digest of
    one shared serialization-error string made two different orders indistinguishable in
    `env_calls.parquet`, which is a wrong value in the shape of a right one."""
    act = Tau2Actuator(_ObjEnv(payload=lambda name: _Model(name)))
    a, b = act.execute([{"name": "alpha"}, {"name": "beta"}], turn_idx=0)
    assert a.result_digest != b.result_digest


def test_a_result_no_canonicalization_can_reach_is_marked_rather_than_faked() -> None:
    """The third state, and the reason `ok` did not have to grow a second meaning.

    A value neither `canon` nor `model_dump` can reduce still EXECUTED, so `ok` stays True;
    its digest is minted in a different `h()` domain, so it can never be mistaken for -- or
    collide with -- a digest of an actual result.
    """
    from pinq.ids import h

    act = Tau2Actuator(_ObjEnv(payload=lambda name: _Payload(name)))
    (call,) = act.execute([{"name": "read"}], turn_idx=0)

    assert call.ok is True
    assert call.result_digest == h("res-uncanonical", "_Payload")[:16]
    assert call.result_digest != h("res", "_Payload")[:16]


@tau2_only
def test_replaying_retail_gold_records_no_failed_calls() -> None:
    """MEASURED, on the real environment: retail task 0's answer key is
    find_user_id_by_name_zip, get_order_details, get_product_details x2 and the mutating
    exchange_delivered_order_items. Before the fix the first was `ok=True` (it returns a
    plain string) and the other four were recorded as failures while succeeding."""
    suite = Tau2RetailSuite()
    task = suite.tau2_task_object("0")
    actions = list(task.evaluation_criteria.actions or [])
    assert len(actions) == 5, "upstream changed retail task 0; re-read this test"

    with contextlib.redirect_stdout(io.StringIO()):
        act = Tau2Actuator(
            suite.environment("0"),
            task=suite.task_record("0"),
            domain=Tau2RetailSuite.domain,
            env_kwargs=suite.env_kwargs("0"),
        )
        calls = act.execute(
            [
                {"name": a.name, "args": dict(a.arguments), "requestor": a.requestor}
                for a in actions
            ],
            turn_idx=0,
        )

    failed = [c.tool_name for c in calls if not c.ok]
    assert failed == [], (
        f"the environment executed these and they were logged as failures: {failed}"
    )


# ----------------------------------- defect 3: a recorded result must not keep mutating


class _LiveOrder:
    """The retail environment hands back a LIVE reference into its own DB, not a copy.

    Measured on the real thing: replaying task 71's two gold actions,
    `act._results[0][3] is act._results[1][3]` -- one object, returned twice, mutated in
    between by `modify_pending_order_items`.
    """

    def __init__(self) -> None:
        self.status = "pending"

    def model_dump(self, mode: str = "python") -> dict:
        return {"status": self.status}


class _MutatingEnv:
    """`read` returns the order; `mutate` changes THAT SAME OBJECT, as retail's tools do."""

    def __init__(self) -> None:
        self.order = _LiveOrder()

    def make_tool_call(self, name, *, requestor="assistant", **kwargs):
        if name == "mutate":
            self.order.status = "pending (item modified)"
        return self.order

    def _is_mutating_tool(self, name):
        return name == "mutate"

    def get_db_hash(self):
        return self.order.status

    def get_user_db_hash(self):
        return "udb0"

    @classmethod
    def to_json_str(cls, resp):
        import json as _json

        return _json.dumps(resp.model_dump(), sort_keys=True)


def test_a_recorded_tool_result_is_the_state_at_the_time_of_the_call() -> None:
    """`_trajectory()` serialized `raw` at GRADING time, long after later calls had mutated
    the very object it was holding -- so every earlier read was replayed carrying the
    episode's FINAL state.

    `Environment.get_response` stringifies a result when the call is made and puts THOSE
    bytes in the transcript; `set_state` then re-executes and compares against them. A
    deferred serialization is not a smaller version of that, it is a different transcript,
    and upstream rejects it with `ValueError: Tool call: ... Returned: ... Expected: ...`.
    """
    from pinq.types import EnvCall  # noqa: F401  (documents what execute returns)

    act = Tau2Actuator(_MutatingEnv())
    act.execute([{"name": "read"}, {"name": "mutate"}], turn_idx=0)
    traj = act._trajectory()

    contents = [m.content for m in traj if getattr(m, "role", "") == "tool"]
    assert contents[0] == '{"status": "pending"}', (
        f"the first read was replayed carrying a later call's mutation: {contents[0]}"
    )
    assert contents[1] == '{"status": "pending (item modified)"}'


@tau2_only
def test_retail_gold_replay_reproduces_a_task_whose_reads_precede_its_mutations() -> None:
    """Task 71 is the smallest case: `modify_pending_order_address` then
    `modify_pending_order_items` on ONE order object. 15 of retail's 114 tasks have this
    shape and every one of them failed to grade at all."""
    from pi_run.stages.tau2_runner import replay_gold

    suite = Tau2RetailSuite()
    (row,) = replay_gold(suite, ["71"])
    assert row.error == "", row.error[:400]
    assert row.db_reward == 1.0, "replaying the answer key must reproduce the gold DB state"


# ------------------------------------ the instrument: an ungradeable task is not a skipped one


def test_verify_tau2_does_not_report_a_grading_error_as_a_skip(monkeypatch, capsys) -> None:
    """`pi verify tau2` classified every `db_reward is None` row as SKIPPED and exited 0.

    `GoldReplay` returns None for two unlike events: a task whose criteria carry no actions
    (nothing to grade) and a task whose grade RAISED (something to grade, and the harness
    could not). The command's own docstring says absent and failing must not look alike, and
    on the retail side this hid 15 tasks whose trajectory replay died in upstream's
    `set_state` -- reported as a clean `reproduced: 97, ok: true` with the errors folded into
    a skip list nobody reads twice. The whole point of this command is to distinguish a
    broken harness from a bad policy, so it may not launder one into a skip.
    """
    import argparse

    from pi_run import cli
    from pi_run.stages.tau2_runner import GoldReplay

    rows = [
        GoldReplay("a", 1.0, ("DB",), 3, 0, ""),
        GoldReplay("b", None, ("DB",), 0, 0, "no evaluation actions"),
        GoldReplay("c", None, ("DB",), 4, 0, "ValueError: Tool call: ... Expected: ..."),
    ]
    monkeypatch.setattr(cli, "_emit", lambda payload: print(__import__("json").dumps(payload)))
    monkeypatch.setattr("pi_run.stages.tau2_runner.replay_gold", lambda suite, ids: rows)
    monkeypatch.setattr(
        "pi_run.worker.load_suite",
        lambda sid, cd: type(
            "S", (), {"suite_id": sid, "domain": "d", "task_ids": lambda self: ("a", "b", "c")}
        )(),
    )

    code = cli.cmd_verify_tau2(argparse.Namespace(n=None, suite="tau2_retail", replay_gold=True))
    out = __import__("json").loads(capsys.readouterr().out)

    assert out["skipped"] == ["b"], "only the task with nothing to grade is a skip"
    assert [e["task_id"] for e in out["errors"]] == ["c"]
    assert out["graded"] == 1, "an errored task was not graded and must not be counted as such"
    assert out["ok"] is False and code == 1, "a task the harness could not grade fails the check"


def test_a_serializer_that_explodes_still_leaves_the_call_recorded_as_executed() -> None:
    """The regression guard on the guard. `_result_digest` runs on the SUCCESS path, so an
    exception escaping it lands in `_call`'s handler and re-creates the original bug in a new
    place -- a call the environment executed, filed as a failure because we could not write
    down what it returned."""

    class _Exploding:
        def model_dump(self, mode: str = "python"):
            raise RuntimeError("this serializer is broken")

    act = Tau2Actuator(_ObjEnv(payload=lambda name: _Exploding()))
    (call,) = act.execute([{"name": "read"}], turn_idx=0)

    assert call.ok is True
    assert call.result_digest, "a digest must still be recorded"


@tau2_only
def test_grading_retail_in_bankings_domain_fabricates_a_success() -> None:
    """WHY DEFECT 1 IS CRITICAL AND NOT MERELY AN ABSENT NUMBER.

    Offline, banking's build raises for want of an embedding index, so the bug presented as a
    missing `tau_reward`. That is an accident of this machine: banking builds fine with the
    bm25 variant it uses for itself, which is the same position a machine WITH
    OPENAI_API_KEY is in. Grade a retail task there and a number comes back.

    Measured, retail task 0, empty transcript (a policy that did nothing):

        graded in banking : db_reward = 1.0, db_match = True
        graded in retail  : db_reward = 0.0, db_match = False

    Neither database moved, so the banking comparison matches and reports success -- on the
    paper's tau2 PRIMARY endpoint, for a policy that took no action at all. The direction is
    what makes it fatal: the wrong domain cannot fail, so it flatters every arm equally and
    the contrast between them collapses toward zero with nothing anywhere reporting an error.
    """
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.runner import build_environment

    suite = Tau2RetailSuite()
    task = suite.tau2_task_object("0")

    def grade(domain: str, env_kwargs: dict) -> float | None:
        def ctor(solo_mode: bool = False, **kw):
            return build_environment(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)

        info = EnvironmentEvaluator.calculate_reward(ctor, task, [], env_kwargs={})
        return getattr(getattr(info, "db_check", None), "db_reward", None)

    with contextlib.redirect_stdout(io.StringIO()):
        wrong = grade(DOMAIN, {"retrieval_variant": "bm25"})
        right = grade(RETAIL_DOMAIN, {})

    assert right == 0.0, "a policy that did nothing did not reach retail's gold state"
    assert wrong != right, (
        "grading in the wrong domain returned retail's own answer by coincidence; this test "
        "can no longer detect the defect and needs a different task"
    )
    assert wrong == 1.0, "the wrong domain reports SUCCESS, which is the whole hazard"
