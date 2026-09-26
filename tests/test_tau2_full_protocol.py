"""tau2 under the BENCHMARK's own protocol, rather than under this repository's.

Three things separate a leaderboard-comparable tau2 number from the one this repository has
been recording, and each of them is a place where the two quietly disagree.

1. THE REWARD. `tau2_runner._reward_of` calls `EnvironmentEvaluator.calculate_reward` and
   `Tau2Actuator.attach_reward` then stores `tau_reward = db_reward` whenever `DB` is in the
   task's `reward_basis`. On retail 112 of 114 tasks declare `(DB, NL_ASSERTION)` and on
   airline all 50 declare `(DB, COMMUNICATE)`, so on BOTH domains the recorded `tau_reward`
   drops a factor upstream multiplies in. Upstream's reward is the PRODUCT of every component
   named in the basis, so the recorded number is an upper bound on the benchmark's own -- it
   can only ever be too high, never too low, and by an amount no field on the run reveals.

2. THE STEP AND ERROR CAPS. `DEFAULT_MAX_STEPS = 100` and `DEFAULT_MAX_ERRORS = 30` here;
   `tau2.config` says 200 and 10. Both are in `upstream_pins`, which is in
   `pinq.ids.SEMANTIC_FIELDS`, so they are already part of run identity -- which is exactly
   why a run made under one pair may not be reported as a run made under the other.

3. THE TERMINATION GATE. `evaluate_simulation` scores a prematurely terminated dialogue 0.0;
   `_reward_of` deliberately bypasses that gate (see its docstring). Under the benchmark's own
   protocol the gate IS the protocol, so both numbers are recorded and neither is inferred.

The protocol is named on the spec and stamped into `upstream_pins` rather than read from the
environment, because an environment variable is not in run identity: two runs under two
protocols would share a run id, a run directory and a `--resume` sentinel, and the second
would silently inherit the first.
"""

from __future__ import annotations

import pytest

from pi_run.stages import tau2_runner as T
from pi_run.worker import UnitSpec

# --------------------------------------------------------------------------- routing


def test_all_three_tau2_domains_route_to_the_dialogue_driver() -> None:
    """airline is a dialogue benchmark too, and it was missing from the set.

    `worker.run_unit` dispatches on `DIALOGUE_SUITES`; a suite absent from it goes down the
    flat path, where `Tau2Suite.view()` raises `Tau2NeedsOrchestrator` and every unit fails.
    airline ran only because `scripts/run_tau2_forks.py` calls `run_tau2_unit` directly, so
    the gap was invisible to every fork campaign and fatal to any `pi run` on the suite.
    """
    assert "tau2" in T.DIALOGUE_SUITES
    assert "tau2_retail" in T.DIALOGUE_SUITES
    assert "tau2_airline" in T.DIALOGUE_SUITES


# --------------------------------------------------------------------------- the caps


def test_the_default_protocol_keeps_this_repositorys_caps() -> None:
    """An unnamed protocol must reproduce every tau2 run already on disk, byte for byte."""
    assert T.protocol_caps("") == (T.DEFAULT_MAX_STEPS, T.DEFAULT_MAX_ERRORS)
    assert T.protocol_caps("") == (100, 30)


def test_the_upstream_protocol_reads_its_caps_from_upstream() -> None:
    """From `tau2.config`, not from a literal: a literal goes stale the moment upstream moves
    its own defaults, and the failure is a number labelled leaderboard-comparable that is not.
    """
    from tau2 import config as tau2_config

    assert T.protocol_caps(T.UPSTREAM_PROTOCOL) == (
        tau2_config.DEFAULT_MAX_STEPS,
        tau2_config.DEFAULT_MAX_ERRORS,
    )
    # And the two protocols genuinely differ, or this axis is not being varied at all.
    assert T.protocol_caps(T.UPSTREAM_PROTOCOL) != T.protocol_caps("")


def test_an_unknown_protocol_is_refused() -> None:
    """Never a silent fallback: a typo that resolved to the default would produce runs stamped
    with a protocol they did not run under."""
    with pytest.raises(T.Tau2DriverError):
        T.protocol_caps("tau2_upstrem")


# --------------------------------------------------------------------------- identity


def _spec(**kw) -> UnitSpec:
    base = dict(
        suite_id="tau2_retail",
        corpus_dir="",
        task_id="1",
        arm_id="drafter_only",
        seed=0,
        runs_root="/tmp/runs",
        cache_root="/tmp/cache",
    )
    base.update(kw)
    return UnitSpec(**base)


def test_the_protocol_is_part_of_the_reassembly_key() -> None:
    """`run_sweep` returns `[by_key[spec.key] for spec in specs]`. Two units of one task that
    differ only in protocol sharing a key would hand back two aliases of one status."""
    assert _spec().key != _spec(protocol=T.UPSTREAM_PROTOCOL).key
    assert _spec().key == _spec().key


def test_the_protocol_reaches_upstream_pins_and_therefore_run_identity() -> None:
    """`upstream_pins` is in `pinq.ids.SEMANTIC_FIELDS`, so a protocol that lands there moves
    the run id. Asserted through `semantic_hash` itself rather than by reading the source."""
    from pinq.ids import semantic_hash

    fields = {
        "suite_id": "tau2_retail",
        "arm_id": "drafter_only",
        "seed": 0,
        "task_id": "1",
        "upstream_pins": T.upstream_pins_for("", user_sim="m", suite_version="v"),
    }
    other = dict(fields)
    other["upstream_pins"] = T.upstream_pins_for(
        T.UPSTREAM_PROTOCOL, user_sim="m", suite_version="v"
    )
    assert semantic_hash(fields) != semantic_hash(other)


def test_upstream_pins_names_the_protocol_and_the_caps_it_implies() -> None:
    pins = T.upstream_pins_for(T.UPSTREAM_PROTOCOL, user_sim="gpt-4.1", suite_version="v1")
    assert pins["protocol"] == T.UPSTREAM_PROTOCOL
    assert pins["user_sim"] == "gpt-4.1"
    assert (int(pins["max_steps"]), int(pins["max_errors"])) == T.protocol_caps(T.UPSTREAM_PROTOCOL)
    # The default protocol OMITS the key, so every tau2 run already on disk keeps its run id.
    old = T.upstream_pins_for("", user_sim="gpt-4.1", suite_version="v1")
    assert "protocol" not in old
    assert (old["max_steps"], old["max_errors"]) == ("100", "30")


def test_the_default_protocol_leaves_every_recorded_run_id_untouched() -> None:
    """The specific hazard: `upstream_pins` is in `SEMANTIC_FIELDS` and `canon` hashes the
    dict, so a key merely present with the empty value renames every tau2 run directory."""
    from pinq.ids import semantic_hash

    before = {
        "suite_version": "v1",
        "user_sim": "openai/aws/gpt-oss-120b",
        "max_steps": "100",
        "max_errors": "30",
    }
    assert T.upstream_pins_for("", user_sim="openai/aws/gpt-oss-120b", suite_version="v1") == before
    assert semantic_hash({"upstream_pins": before}) == semantic_hash(
        {
            "upstream_pins": T.upstream_pins_for(
                "", user_sim="openai/aws/gpt-oss-120b", suite_version="v1"
            )
        }
    )


# --------------------------------------------------------------------------- the stock arm


def test_the_stock_arm_exists_and_is_a_control() -> None:
    from pinq_expt import arms as arm_table

    arm = arm_table.get(T.STOCK_ARM)
    assert arm.kind == "control"
    assert arm.expects_asks is False
    # NOT llm_free, although this repository's client makes no call for it. `llm_free` is
    # exactly the three `fake_*` loop-debug arms (tests/test_runtime.py), and the hazard that
    # invariant exists to stop is a paper arm relabelled LLM-free to get past the worker's
    # token assertion. Its components are therefore the real ones, constructed and never
    # invoked, which is also what puts `PI_MODEL_DRAFTER` into `model_pin_hash`.
    assert arm.llm_free is False
    assert arm.home_suites == ("tau2", "tau2_retail", "tau2_airline")


def test_the_stock_arms_model_is_in_run_identity_twice_over() -> None:
    """Through `upstream_pins.agent_model` AND through the drafter pin in `model_pin_hash`.
    Without the first, a stock run on sonnet and a stock run on gpt-oss-120b would share a run
    id; the second exists because the arm's components carry `llm_role` and `collect_pins`
    reads them, which is what makes the two agree instead of merely coexisting."""
    from pinq_expt import arms as arm_table

    arm = arm_table.get(T.STOCK_ARM)
    built = arm_table.build(arm, llm=None)
    assert getattr(built.drafter, "llm_role", None) == "drafter"
    assert getattr(built.inquirer, "llm_role", None) is None, "no questioner, so no pin for one"
    pins = T.upstream_pins_for(
        T.UPSTREAM_PROTOCOL, user_sim="Azure/gpt-4.1", suite_version="v1", agent_model="A"
    )
    assert pins["agent_model"] == "A"
    assert "agent_model" not in T.upstream_pins_for(
        T.UPSTREAM_PROTOCOL, user_sim="Azure/gpt-4.1", suite_version="v1"
    )


def test_the_stock_arm_builds_upstreams_agent_and_not_ours() -> None:
    """The comparison reviewers know is 'stock tau2 agent versus ours', so the stock arm must
    be upstream's own loop rather than a configuration of ours that resembles it."""
    from tau2.agent.llm_agent import LLMAgent

    cls = T.stock_agent_class()
    assert cls is LLMAgent
    assert cls is not T.make_driver_agent_class()


class _FakeEnv:
    def get_tools(self):
        return []

    def get_policy(self):
        return "policy text"


class _FakeSuite:
    domain = "retail"
    suite_id = "tau2_retail"

    def __init__(self, task) -> None:
        self._task = task

    def environment(self, tid):
        return _FakeEnv()

    def tau2_task_object(self, tid):
        return self._task


class _FakeSpec:
    suite_id = "tau2_retail"
    task_id = "55"
    arm_id = T.STOCK_ARM
    max_turns = 16
    k = 4
    seed = 0
    branch_turn_idx = None
    branch_seed = None
    foreign_trace_sha = None
    foreign_prefix_k = None


def _fake_task():
    from tau2.data_model.tasks import Task, UserScenario

    return Task(id="55", user_scenario=UserScenario(instructions="be a customer"))


def test_the_stock_arm_builds_no_component_of_ours(monkeypatch) -> None:
    """The property that makes it a control, asserted by DENYING the machinery rather than by
    grepping for a word: `pinq_expt.arms.build` is replaced with a raiser, and
    `build_stock_agent` still returns an agent. An arm that constructed a policy and then
    never used it would still have rendered its prompt and would still be a different
    experiment, so 'zero asks afterwards' is the weaker claim and not the one made here."""
    import pinq_expt.arms as arm_table

    def refuse(*a, **k):
        raise AssertionError("the stock arm built a component of ours")

    monkeypatch.setattr(arm_table, "build", refuse)
    monkeypatch.setattr(T, "make_driver_agent_class", refuse)
    agent = T.build_stock_agent(_FakeEnv(), model="m", llm_args={"temperature": 0.0})
    from tau2.agent.llm_agent import LLMAgent

    assert isinstance(agent, LLMAgent)
    assert agent.llm == "m"
    assert agent.domain_policy == "policy text"


def test_simulate_hands_the_orchestrator_upstreams_agent_for_the_stock_arm(monkeypatch) -> None:
    """The branch has to be on the ARM and `_simulate` has to take it: a stock arm that exists
    in the table and is never selected produces our own runs wearing its name. Checked by
    making our driver class unbuildable, so a run that reaches it fails loudly."""
    import tau2.orchestrator.orchestrator as orch_mod
    from tau2.agent.llm_agent import LLMAgent

    seen: dict = {}

    class _Orch:
        def __init__(self, **kw):
            seen.update(kw)
            self.agent_state = object()

        def run(self):
            class _Sim:
                messages = ()

            return _Sim()

    monkeypatch.setattr(orch_mod, "Orchestrator", _Orch)
    monkeypatch.setattr(
        T,
        "make_driver_agent_class",
        lambda: (_ for _ in ()).throw(AssertionError("our driver was built for the stock arm")),
    )
    monkeypatch.setenv("PI_MODEL_DRAFTER", "openai/aws/claude-sonnet-5")
    _sim, state, _task = T._simulate(
        _FakeSuite(_fake_task()),
        _FakeSpec(),
        lambda *a, **k: {},
        user=object(),
        prefix=None,
        protocol=T.UPSTREAM_PROTOCOL,
    )
    assert isinstance(seen["agent"], LLMAgent)
    assert seen["agent"].llm == "openai/aws/claude-sonnet-5"
    # And the caps the Orchestrator was given are upstream's, not ours.
    assert (seen["max_steps"], seen["max_errors"]) == T.protocol_caps(T.UPSTREAM_PROTOCOL)
    # The harvest reads `state.trajectories` and `state.rejected`; upstream's state has
    # neither, so the stock arm is handed an empty DriverState rather than crashing.
    assert isinstance(state, T.DriverState)
    assert state.trajectories == []
    assert state.rejected == 0


def test_the_augmented_arms_still_get_our_driver(monkeypatch) -> None:
    """Non-vacuity of the branch above: with the arm changed and nothing else, the SAME call
    must reach our driver. Without this, an always-stock branch would pass every test here."""
    import tau2.orchestrator.orchestrator as orch_mod

    seen: dict = {}

    class _Orch:
        def __init__(self, **kw):
            seen.update(kw)
            self.agent_state = object()

        def run(self):
            class _Sim:
                messages = ()

            return _Sim()

    monkeypatch.setattr(orch_mod, "Orchestrator", _Orch)
    monkeypatch.setattr(T, "make_driver_agent_class", lambda: lambda **kw: "OUR-DRIVER")

    class _Spec(_FakeSpec):
        arm_id = "inquirer_prompted"

    T._simulate(
        _FakeSuite(_fake_task()),
        _Spec(),
        lambda *a, **k: {},
        user=object(),
        prefix=None,
        protocol=T.UPSTREAM_PROTOCOL,
    )
    assert seen["agent"] == "OUR-DRIVER"


# --------------------------------------------------------------------------- the reward


class _Basis:
    def __init__(self, value: str) -> None:
        self.value = value


class _Task:
    class _Crit:
        reward_basis = (_Basis("DB"), _Basis("COMMUNICATE"))

    evaluation_criteria = _Crit()


class _Sim:
    messages = ()
    termination_reason = "TerminationReason.MAX_STEPS"


def test_upstream_grade_is_zero_on_a_premature_termination(monkeypatch) -> None:
    """Upstream's own rule, and the one `_reward_of` deliberately bypasses. Under the
    benchmark's protocol the gate IS the protocol, so this path must keep it."""
    seen: dict[str, object] = {}

    def fake_evaluate(**kw):
        seen.update(kw)

        class R:
            reward = 0.0
            reward_breakdown = None
            db_check = None

        return R()

    monkeypatch.setattr(T, "_evaluate_simulation", fake_evaluate)
    out = T.grade_upstream(_Sim(), _Task(), domain="airline", env_kwargs={})
    assert out["upstream_reward"] == 0.0
    assert out["upstream_success"] == 0.0
    # And it went through upstream's ALL evaluation type, which is what applies the basis.
    from tau2.evaluator.evaluator import EvaluationType

    assert seen["evaluation_type"] is EvaluationType.ALL
    assert seen["solo_mode"] is False


def test_upstream_grade_records_the_components_it_multiplied(monkeypatch) -> None:
    class R:
        reward = 1.0
        reward_breakdown = {_Basis("DB"): 1.0, _Basis("COMMUNICATE"): 1.0}
        db_check = None

    monkeypatch.setattr(T, "_evaluate_simulation", lambda **kw: R())
    out = T.grade_upstream(_Sim(), _Task(), domain="airline", env_kwargs={})
    assert out["upstream_reward"] == 1.0
    assert out["upstream_success"] == 1.0
    assert out["upstream.DB"] == 1.0
    assert out["upstream.COMMUNICATE"] == 1.0


def test_upstream_success_uses_upstreams_own_tolerance(monkeypatch) -> None:
    """`tau2.metrics.agent_metrics.is_successful` is `1 - 1e-6 <= r <= 1 + 1e-6`, and pass^k
    counts successes rather than averaging rewards. A partial reward is not a success."""
    from tau2.metrics.agent_metrics import is_successful

    class R:
        reward = 0.5
        reward_breakdown = None
        db_check = None

    monkeypatch.setattr(T, "_evaluate_simulation", lambda **kw: R())
    out = T.grade_upstream(_Sim(), _Task(), domain="airline", env_kwargs={})
    assert out["upstream_reward"] == 0.5
    assert out["upstream_success"] == 0.0
    assert is_successful(0.5) is False
    assert is_successful(1.0) is True


def test_a_failed_upstream_grade_is_data_and_not_a_sweep_abort(monkeypatch) -> None:
    """Same rule `_reward_of` follows: one unit's ungradable transcript may not take the
    campaign down, and the reason must be recorded rather than swallowed into a 0."""

    def boom(**kw):
        raise RuntimeError("no judge key")

    monkeypatch.setattr(T, "_evaluate_simulation", boom)
    out = T.grade_upstream(_Sim(), _Task(), domain="airline", env_kwargs={})
    assert "upstream_reward" not in out
    assert out["upstream_error"].startswith("RuntimeError:")
