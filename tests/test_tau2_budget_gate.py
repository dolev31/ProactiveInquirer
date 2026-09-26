"""The tau2 retrieval budget binds WHILE THE DIALOGUE RUNS: refuse and tell.

THE DEFECT. `budget_cap` is 16 retrieval calls for a whole tau2 unit, one ledger per unit. The
agent's tool calls were executed the moment they were emitted and charged only after the dialogue
ended (`meter_env_calls`, "RECORDED AND CHECKED, NEVER CHARGED"), so nothing stopped a policy from
calling past the cap: 122 of 408 recorded fork runs did, and every tau2 claim was withdrawn.

WHERE A DIALOGUE'S TOOL CALL ACTUALLY EXECUTES. Not in `Tau2Actuator`: the tau2 driver hands
`run_loop` `actuator=None`, and the actuator appears on this path only as `replay_gold`'s answer-key
executor and `_reward_of`'s attach-only holder. The call goes `Orchestrator.step` ->
`Orchestrator._execute_tool_calls` -> `Environment.get_response` -> `make_tool_call`. So every test
here drives the REAL upstream Orchestrator over a REAL tau2 `Environment` -- the mock domain's
tools over an in-memory DB, so no data checkout is needed -- through `run_tau2_unit`, the
production entry point. Only the policy, the customer and the suite's bookkeeping are scripted.

THE RULE BEING PINNED (RULES amendment 7: SPLIT BUDGETS). Asks and tool calls each get
`budget_cap` (16). The ask budget is the unit's ledger, charged by `run_loop` exactly as on QA.
The tool budget is the gate's own counter: once `budget_cap` tool calls have executed ok, a further
one is not executed (`make_tool_call` never runs, the world does not move), not counted, and
answered with a tool result reading `retrieval budget exhausted`. The ledger never receives a tool
charge, so asking can never starve the agent and acting can never stop the questioner. A call the
environment rejects is not counted. A refused call is never graded as executed. A fork's inherited
prefix is its starting state, not its spend. A GENERIC tool (conf/tau2/tool_types.json) is not
retrieval: never gated, never counted, so the hand-off survives exhaustion; a tool the map does
not type stays gated. (ba48ab0..e5e8206 shared ONE cap between the two; tests whose belief was
that shared cap say where amendment 7 changed them.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Skipped without tau2 -- unless PINQ_REQUIRE_TAU2_DATA=1, when a missing tau2 must FAIL
# (the imports below then raise at collection). See tests/_tau2_data.py.
if os.environ.get("PINQ_REQUIRE_TAU2_DATA") != "1":
    pytest.importorskip("tau2")

from pinq.budget import BudgetLedger  # noqa: E402
from pinq.types import Answer, Ask, Draft, Stop  # noqa: E402

PHRASE = "retrieval budget exhausted"
TID = "rig_0"
TID_REAL = "0"  # retail's first task id, for the one test on the real adapter
USER = "u1"

# The gold answer key for the reward tests: exactly one task created, titled "a".
TASK = {
    "id": TID,
    "user_scenario": {"instructions": "You want a task created."},
    "evaluation_criteria": {
        "actions": [
            {
                "action_id": "gold_1",
                "requestor": "assistant",
                "name": "create_task",
                "arguments": {"user_id": USER, "title": "a"},
            }
        ],
        "reward_basis": ["DB"],
    },
}


# ------------------------------------------------------------------------------------ the world


def _db():
    from tau2.domains.mock.data_model import MockDB, Task, User

    return MockDB(
        tasks={"t_1": Task(task_id="t_1", title="seed", status="pending")},
        users={USER: User(user_id=USER, name="Ada", tasks=["t_1"])},
    )


def _env(counter: list | None = None):
    """A real tau2 Environment. `counter` records every `make_tool_call` that REACHES it."""
    from tau2.domains.mock.tools import MockTools
    from tau2.environment.environment import Environment

    env = Environment(domain_name="mock", policy="Help the customer.", tools=MockTools(_db()))
    if counter is not None:
        original = env.make_tool_call

        def counting(tool_name: str, requestor: str = "assistant", **kwargs: Any) -> Any:
            counter.append((tool_name, dict(kwargs)))
            return original(tool_name, requestor=requestor, **kwargs)

        env.make_tool_call = counting
    return env


def _create(title: str) -> dict:
    return {"name": "create_task", "args": {"user_id": USER, "title": title}}


def _update_missing() -> dict:
    """A WRITE the environment rejects: the task does not exist, so the tool raises."""
    return {"name": "update_task_status", "args": {"task_id": "nope", "status": "completed"}}


def _read() -> dict:
    return {"name": "get_users", "args": {}}


def _transfer() -> dict:
    """The hand-off. GENERIC in upstream's declaration: reads no record, writes no DB row."""
    return {"name": "transfer_to_human_agents", "args": {"summary": "needs a human"}}


def _hash_after(plan: list[dict]) -> str:
    env = _env()
    for step in plan:
        env.make_tool_call(step["name"], requestor="assistant", **step["args"])
    return env.get_db_hash()


# ------------------------------------------------------------------------------------ the policy


def _last_line(view) -> str:
    return str(view.question).splitlines()[-1]


class _Inquirer:
    """Asks `asks[last user line]` kb questions, then stops. Pure in the view it was reset on."""

    policy_id = "rig"
    may_ask_user = False

    def __init__(self, asks: dict[str, int]) -> None:
        self._asks = asks
        self._left = 0

    def reset(self, view, seed: int) -> None:
        self._left = int(self._asks.get(_last_line(view), 0))

    def act(self, s):
        if self._left > 0:
            self._left -= 1
            return Ask(text=f"q{self._left}")
        return Stop()


class _Drafter:
    """Plans `plans[last user line]`. Pure in (view, evidence, seed), as the protocol requires."""

    def __init__(self, plans: dict[str, list[dict]]) -> None:
        self._plans = plans

    def resolve(self, view, ask, ev, *, seed, ledger):
        return "", ev

    def draft(self, view, ev, *, seed, ledger) -> Draft:
        plan = tuple(dict(p) for p in self._plans.get(_last_line(view), ()))
        return Draft(text="working on it", tool_plan=plan)


class _Answerer:
    def answer(self, view, ev, draft, *, seed, ledger) -> Answer:
        return Answer(text="Done.", evidence_hash=ev.subset_hash)


class _Retriever:
    def search(self, query: str, k: int):
        return ()


class _Index:
    """What `meter_env_calls` / `uids_for_call` read. Empty: these tests count charges, not uids."""

    uids: dict[str, str] = {}
    titles: dict[str, str] = {}
    item_to_product: dict[str, str] = {}

    def uid(self, table: str, rid: str) -> str:  # pragma: no cover - not reached
        return f"{table}/{rid}"


class _UnchargedSuite:
    """A tau2 dialogue suite with no tool-call index, like banking: its tool calls were never
    charged, because `_attach_env_evidence` only meters a suite that advertises `index` and
    `uids_for_calls`."""

    suite_id = "tau2_retail"
    domain = "mock"
    instructions = "Help the customer."
    corpus_id = "rig"
    corpus_hash = "rig-corpus"
    suite_version = "rig"

    def __init__(self, counter: list) -> None:
        self._env = _env(counter)

    def environment(self, tid=None):
        return self._env

    def tau2_task_object(self, tid):
        from tau2.data_model.tasks import Task

        return Task.model_validate(TASK)

    def task_record(self, tid):
        return json.loads(json.dumps(TASK))

    def env_kwargs(self, tid=None):
        return {}

    def tool_schemas(self, tid):
        return tuple({"name": t.name} for t in self._env.get_tools())

    def retriever(self, tid, **kw):
        return _Retriever()


class _Suite(_UnchargedSuite):
    """Retail- and airline-shaped: tool calls are CHARGED, because the suite has an index."""

    def __init__(self, counter: list) -> None:
        super().__init__(counter)
        self.index = _Index()

    def uids_for_calls(self, calls):
        return ()


def _scripted_user(lines: list[str]):
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator_base import HalfDuplexUser

    class ScriptedUser(HalfDuplexUser):
        def __init__(self) -> None:
            super().__init__(instructions=None, tools=None)

        def get_init_state(self, message_history=None):
            return {"i": 0}

        def set_seed(self, seed):
            return None

        def generate_next_message(self, message, state):
            i = state["i"]
            state["i"] = i + 1
            text = lines[i] if i < len(lines) else "Thanks. ###STOP###"
            return UserMessage(role="user", content=text), state

    return ScriptedUser()


class _Rig:
    """One tau2 unit through `run_tau2_unit`, with the world real and everything else scripted."""

    def __init__(
        self,
        *,
        cap: int,
        plans: dict[str, list[dict]],
        lines: list[str],
        asks: dict[str, int] | None = None,
        charged: bool = True,
        prefix: list | None = None,
        arm_id: str = "fake_chain",
        suite: Any = None,
        tool_types: dict | None = None,
    ) -> None:
        self.cap = cap
        # A tool-type map for the MOCK domain, which the committed conf/tau2/tool_types.json does
        # not carry (so every rig tool is unmapped -- gated and charged -- unless a test says so).
        self.tool_types = tool_types
        self.plans = plans
        self.lines = lines
        self.asks = dict(asks or {})
        self.prefix = prefix
        self.arm_id = arm_id
        self.executed: list[tuple[str, dict]] = []
        # `suite` is a REAL adapter (retail), whose own environment is instrumented in place;
        # otherwise the mock-domain rig suite, and the grader is pointed at the mock domain too.
        self.real_suite = suite is not None
        if suite is None:
            suite = (_Suite if charged else _UnchargedSuite)(self.executed)
        else:
            env = suite.environment(TID_REAL)
            original = env.make_tool_call

            def counting(tool_name: str, requestor: str = "assistant", **kwargs: Any) -> Any:
                self.executed.append((tool_name, dict(kwargs)))
                return original(tool_name, requestor=requestor, **kwargs)

            env.make_tool_call = counting
        self.suite = suite

    def build(self, arm, **kw):
        return SimpleNamespace(
            inquirer=_Inquirer(self.asks), drafter=_Drafter(self.plans), answerer=_Answerer()
        )

    def run(self, tmp_path: Path, monkeypatch) -> tuple[dict, dict]:
        import tau2.runner

        from pi_run import worker
        from pi_run.stages import tau2_runner as T
        from pi_run.worker import UnitSpec
        from pinq_expt import arms as arm_table

        monkeypatch.setattr(worker, "load_suite", lambda sid, cd: self.suite)
        monkeypatch.setattr(arm_table, "build", self.build)
        if not self.real_suite:
            # The GRADER's environments: fresh, uncounted, and built from the same initial DB.
            monkeypatch.setattr(
                tau2.runner,
                "build_environment",
                lambda domain, solo_mode=False, env_kwargs=None: _env(),
            )
        captured: dict[str, Any] = {}
        real = T._simulate

        def spy(*a, **kw):
            out = real(*a, **kw)
            captured["sim"], captured["state"], captured["task"] = out
            return out

        monkeypatch.setattr(T, "_simulate", spy)
        if self.prefix is not None:
            monkeypatch.setattr(T, "load_prefix", lambda spec, *a, **k: list(self.prefix))
        if self.tool_types is not None:
            path = tmp_path / "tool_types.json"
            path.write_text(json.dumps(self.tool_types))
            # raising=False: the attribute does not exist before the exemption, and the test
            # must then fail on BEHAVIOUR (the call was refused), not on a missing name.
            monkeypatch.setattr(T, "TOOL_TYPES_PATH", path, raising=False)
        spec = UnitSpec(
            suite_id="tau2_retail",
            corpus_dir=str(tmp_path),
            task_id=TID_REAL if self.real_suite else TID,
            arm_id=self.arm_id,
            seed=0,
            runs_root=str(tmp_path / "runs"),
            cache_root=str(tmp_path / "cache"),
            code_version="testsha",
            dirty=False,
            max_turns=20,
            k=3,
            budget_cap=self.cap,
            foreign_trace_sha="rigtrace" if self.prefix is not None else None,
            foreign_prefix_k=len(self.prefix) if self.prefix is not None else None,
        )
        status = T.run_tau2_unit(spec, user=_scripted_user(self.lines))
        captured["dir"] = Path(spec.runs_root) / str(status.get("run_id") or "")
        return status, captured


def _tool_messages(sim) -> list:
    return [m for m in sim.messages if getattr(m, "role", "") == "tool"]


def _ledger_rows(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------------------------ T1


def test_t1_a_spent_budget_stops_the_call_before_the_environment_sees_it(tmp_path, monkeypatch):
    """Cap 2, one rollout, a plan of four WRITES. Exactly two reach `make_tool_call`; the other
    two come back to the agent as a refusal and the world is exactly the two-write world."""
    rig = _Rig(
        cap=2,
        plans={"L1": [_create("a"), _create("b"), _create("c"), _create("d")]},
        lines=["L1"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert [a["title"] for _n, a in rig.executed] == ["a", "b"], (
        f"make_tool_call ran {len(rig.executed)} times under a cap of 2"
    )

    tool = _tool_messages(got["sim"])
    assert len(tool) == 4, "every emitted call is answered, executed or not"
    assert [PHRASE in str(m.content) for m in tool] == [False, False, True, True]
    assert not any(m.error for m in tool[2:]), "a refusal is not an environment error"

    assert rig.suite.environment().get_db_hash() == _hash_after([_create("a"), _create("b")]), (
        "a refused write moved the world"
    )
    # AMENDMENT 7 (split budgets): the two writes count on the gate's own counter; the ask
    # ledger holds asks only, and there were none. (Was `spent.retrieval_calls == 2.0`.)
    assert status["n_gate_charged"] == 2
    assert status["spent"].get("retrieval_calls", 0.0) == 0.0


# ------------------------------------------------------------------------------------ T2


def test_t2_a_call_the_environment_rejects_is_not_charged(tmp_path, monkeypatch):
    """`meter_env_calls` has always said A FAILED CALL IS NOT CHARGED. The gate agrees: the
    rejected write reaches the environment, costs nothing, and the next two writes fit the cap."""
    rig = _Rig(
        cap=2,
        plans={"L1": [_update_missing(), _create("a"), _create("b"), _create("c")]},
        lines=["L1"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert [n for n, _a in rig.executed] == ["update_task_status", "create_task", "create_task"]
    tool = _tool_messages(got["sim"])
    assert [bool(m.error) for m in tool] == [True, False, False, False]
    assert [PHRASE in str(m.content) for m in tool] == [False, False, False, True]
    assert status["n_gate_charged"] == 2  # amendment 7: the gate's counter, not the ledger

    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    assert [c["ok"] for c in outcome["env_calls"]] == [False, True, True]
    assert status["n_errors_agent"] == 1, "the rejected write is an error; the refusal is not"


# ------------------------------------------------------------------------------------ T3


#
# RULES AMENDMENT 7 replaced the two tests that stood here. Their belief was ONE cap shared by
# asks and tool calls (`test_t3_the_asks_and_the_tool_calls_share_one_cap`: 15 asks of 16 leave
# one tool call; `test_t3_a_tool_call_spends_budget_the_next_rollouts_asks_can_see`: rollout 1's
# tool calls stop rollout 2's Ask on `budget`). The user split them -- asks and tool calls get
# 16 EACH, so asking can never starve the agent -- and the tests below state the new rule.


def test_t3_a_unit_whose_asks_spend_their_budget_still_executes_tool_calls(tmp_path, monkeypatch):
    """16 asks exhaust the ASK budget: the 17th stops the rollout on `budget`, as on QA. The
    Drafter's tool plan then executes in full, because tool calls are on their own counter."""
    rig = _Rig(
        cap=16,
        asks={"L1": 17},
        plans={"L1": [_create("a"), _create("b"), _create("c")]},
        lines=["L1"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert got["state"].trajectories[0].stop_reason == "budget"
    assert status["n_asks"] == 16
    assert status["spent"]["retrieval_calls"] == 16.0
    assert [a["title"] for _n, a in rig.executed] == ["a", "b", "c"]
    assert status["n_refused_budget"] == 0
    assert status["n_gate_charged"] == 3


def test_t3_seventeen_tool_calls_execute_sixteen_and_leave_the_asks_alone(tmp_path, monkeypatch):
    """The tool budget binds at 16 on its own: 17 writes give 16 executed and 1 refused, and the
    3 asks of the same rollout are charged to the ask ledger exactly as they would be alone."""
    rig = _Rig(
        cap=16,
        asks={"L1": 3},
        plans={"L1": [_create(f"t{i}") for i in range(17)]},
        lines=["L1"],
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert [a["title"] for _n, a in rig.executed] == [f"t{i}" for i in range(16)]
    assert [json.loads(r["kwargs_json"])["title"] for r in status["refused_budget_calls"]] == [
        "t16"
    ]
    assert status["n_gate_charged"] == 16
    assert status["n_asks"] == 3
    assert status["spent"]["retrieval_calls"] == 3.0


def test_t3_the_ledger_never_receives_a_tool_charge(tmp_path, monkeypatch):
    """Every `retrieval_calls` row in the ask ledger is an Ask's: as many rows as asks, in the
    turns the asks were charged in, with a tool plan that executed four writes around them."""
    rig = _Rig(
        cap=16,
        asks={"L1": 2, "L2": 1},
        plans={"L1": [_create("a"), _create("b")], "L2": [_create("c"), _create("d")]},
        lines=["L1", "L2"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert len(rig.executed) == 4 and status["n_gate_charged"] == 4
    rc = [r for r in _ledger_rows(got["dir"]) if r["currency"] == "retrieval_calls"]
    assert [r["cumulative"] for r in rc] == [1.0, 2.0, 3.0], "a tool call reached the ledger"
    assert status["spent"]["retrieval_calls"] == status["n_asks"] == 3


# ------------------------------------------------------------------------------------ T4


def test_t4_nothing_is_charged_twice_and_the_post_hoc_meter_agrees(tmp_path, monkeypatch):
    """ONE OWNER PER BUDGET. Asks: the ledger (`run_loop`). Tool calls: the gate's own counter,
    and the post-hoc harvest charges neither. THE IDENTITIES (amendment 7 -- this test's belief
    was `spent == asks + ok non-GENERIC calls` under the shared cap, and changed with it):
    spent == asks, and n_gate_charged == ok executed non-GENERIC calls.

    Checked against an instrument that does not share the gate's code: the ORIGINAL charging
    meter (`charge=True`, the pre-gate rule, which charges GENERIC calls too), run over this run's
    recorded non-GENERIC calls on a FRESH ledger with the tool cap, must reach n_gate_charged with
    no overrun. Over ALL recorded calls it counts the GENERIC ones too -- the documented old rule
    -- and the run's own record must still carry no overrun, because a gated unit never meters
    with `charge=True`."""
    from pi_run.stages.tau2_runner import meter_env_calls

    types = {
        "mock": {
            "create_task": "WRITE",
            "update_task_status": "WRITE",
            "get_users": "READ",
            "transfer_to_human_agents": "GENERIC",
        }
    }
    rig = _Rig(
        cap=3,
        asks={"L1": 1},
        plans={
            "L1": [
                _create("a"),
                _update_missing(),
                _transfer(),
                _create("b"),
                _create("c"),
                _transfer(),
                _create("d"),
            ]
        },
        lines=["L1"],
        tool_types=types,
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    generic = {n for n, t in types["mock"].items() if t == "GENERIC"}
    charged = [c for c in outcome["env_calls"] if c["ok"] and c["tool_name"] not in generic]
    assert [json.loads(c["kwargs_json"])["title"] for c in charged] == ["a", "b", "c"]
    assert status["n_gate_charged"] == len(charged) == 3
    assert status["spent"]["retrieval_calls"] == status["n_asks"] == 1
    assert "post_hoc_budget_overrun" not in status["spent"]
    assert [r["tool_name"] for r in status["refused_budget_calls"]] == ["create_task"]

    rc = [r for r in _ledger_rows(got["dir"]) if r["currency"] == "retrieval_calls"]
    assert len(rc) == 1, "the ledger holds the one Ask and no tool call"
    assert all(r["hard"] and r["cap"] == 3 for r in rc), "the Ask went through the hard cap"

    def replay_meter(calls):
        led = BudgetLedger(cap=status["tool_call_cap"])
        _uids, overrun = meter_env_calls(_Index(), led, calls)
        return led.spent["retrieval_calls"], overrun

    own_rule = [c for c in outcome["env_calls"] if c["tool_name"] not in generic]
    assert replay_meter(own_rule) == (float(status["n_gate_charged"]), False)
    assert replay_meter(outcome["env_calls"]) == (5.0, True), "the old rule charges GENERIC"


def _prefix_with_one_write() -> list:
    """A foreign prefix in which the ORIGINAL agent created a task, ending on a user turn."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    tc = ToolCall(
        id="pre_0",
        name="create_task",
        arguments={"user_id": USER, "title": "p"},
        requestor="assistant",
    )
    result = _env().get_response(tc)  # the exact bytes the fork's set_state will compare against
    return [
        AssistantMessage(role="assistant", content="Hi, how can I help?", cost=0.0),
        UserMessage(role="user", content="P1"),
        AssistantMessage(role="assistant", content=None, tool_calls=[tc], cost=0.0),
        result,
        AssistantMessage(role="assistant", content="Created it.", cost=0.0),
        UserMessage(role="user", content="L1"),
    ]


def test_t4_a_forks_inherited_prefix_is_not_its_spend(tmp_path, monkeypatch):
    """The prefix is replayed into the live world by upstream's own `set_state` and is the task's
    STARTING STATE -- the evaluator seeds the gold environment from it too. It is not charged:
    under a cap of 2 both of the fork's own writes execute."""
    rig = _Rig(
        cap=2,
        plans={"L1": [_create("a"), _create("b")]},
        lines=[],
        prefix=_prefix_with_one_write(),
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    # The prefix write is re-executed once, by upstream's replay; then the fork's own two.
    assert [a["title"] for _n, a in rig.executed] == ["p", "a", "b"]
    # Amendment 7: the tool counter, not the ledger (was `spent.retrieval_calls == 2.0`).
    assert status["n_gate_charged"] == 2, "the inherited write was charged"
    assert status["n_refused_budget"] == 0

    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    own = [c for c in outcome["env_calls"] if c["turn_idx"] >= status["n_prefix_user_turns"]]
    assert len(outcome["env_calls"]) == 3 and len(own) == 2


# ------------------------------------------------------------------------------------ T5


def test_t5_the_refusal_is_delivered_and_the_pinq_policy_meets_the_budget_in_its_loop(
    tmp_path, monkeypatch
):
    """WHAT REACHES THE PINQ POLICY, traced rather than assumed. The refusal ToolMessage is handed
    to `PinqDriverAgent.generate_next_message` and lands in `DriverState.messages`. Nothing the
    policy reads is built from that list except `dialogue_view`, which keeps user turns only --
    so no tool result, refused or executed, reaches the Inquirer or the Drafter.

    AMENDMENT 7 changed the second half. Under the shared cap the tool budget reached the policy
    through `run_loop`: rollout 2's first Ask stopped on `budget`. With split budgets the tool
    budget reaches our policy in NO way at all -- rollout 2's Ask is charged to the ask ledger
    and runs -- which is the point of the split: asking and acting cannot starve each other."""
    from pi_run.stages.tau2_runner import dialogue_view

    rig = _Rig(
        cap=1,
        asks={"L2": 1},
        plans={"L1": [_create("a"), _create("b")], "L2": [_create("c")]},
        lines=["L1", "L2"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    state = got["state"]

    delivered = [
        m for m in state.messages if getattr(m, "role", "") == "tool" and PHRASE in str(m.content)
    ]
    assert len(delivered) == 2, "b (rollout 1) and c (rollout 2) were refused to the agent"
    view = dialogue_view(rig.suite, TID, state.messages)
    assert PHRASE not in view.question, "dialogue_view renders user turns only"
    assert state.trajectories[1].stop_reason == "policy_stop"
    assert status["n_asks"] == 1 and status["spent"]["retrieval_calls"] == 1.0
    assert [a["title"] for _n, a in rig.executed] == ["a"]


def test_t5_the_stock_agent_reads_the_refusal_on_its_next_step(tmp_path, monkeypatch):
    """Upstream's `LLMAgent` appends every tool message to its history and sends that history
    to the model, so for the stock arm the refusal text IS the budget signal. Driven through
    `run_tau2_unit` with arm `tau2_stock`: the gate is the same for every arm."""
    import tau2.agent.llm_agent as llm_agent
    from tau2.data_model.message import AssistantMessage, ToolCall

    seen: list[list] = []

    def fake_generate(model, tools, messages, call_name=None, **kw):
        seen.append(list(messages))
        n = len(seen)
        if n <= 2:
            tc = ToolCall(
                id=f"stock_{n}",
                name="create_task",
                arguments={"user_id": USER, "title": "ab"[n - 1]},
                requestor="assistant",
            )
            return AssistantMessage(role="assistant", content=None, tool_calls=[tc], cost=0.0)
        return AssistantMessage(role="assistant", content="All done.", cost=0.0)

    monkeypatch.setattr(llm_agent, "generate", fake_generate)
    monkeypatch.setenv("PI_MODEL_DRAFTER", "rig-model")
    rig = _Rig(cap=1, plans={}, lines=["L1"], arm_id="tau2_stock")
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert [a["title"] for _n, a in rig.executed] == ["a"]
    assert len(seen) == 3
    last = seen[2][-1]
    assert getattr(last, "role", "") == "tool" and last.id == "stock_2"
    assert PHRASE in str(last.content), "the model's next request carries the refusal"
    assert status["budget_enforcement"] == "refuse_and_tell_split"  # amendment 7
    assert status["n_refused_budget"] == 1


# ------------------------------------------------------------------------------------ T6


def test_t6_refused_writes_do_not_move_the_reward(tmp_path, monkeypatch):
    """The reward of a run whose refused calls are WRITES equals the reward of the same run with
    those calls absent. Gold is "create a"; the gated run also emitted "create b" and "create c".
    Graded as executed, those two would put three tasks in the predicted DB and score 0."""
    gated = _Rig(cap=1, plans={"L1": [_create("a"), _create("b"), _create("c")]}, lines=["L1"])
    s_gated, _ = gated.run(tmp_path / "gated", monkeypatch)
    absent = _Rig(cap=16, plans={"L1": [_create("a")]}, lines=["L1"])
    s_absent, _ = absent.run(tmp_path / "absent", monkeypatch)

    assert s_gated["status"] == s_absent["status"] == "ok"
    assert s_gated["reward_error"] == s_absent["reward_error"] == ""
    assert s_absent["native"]["tau_reward"] == 1.0
    assert s_gated["native"]["tau_reward"] == s_absent["native"]["tau_reward"]


def test_t6_upstreams_grader_would_execute_a_refused_write_left_in_the_transcript(
    tmp_path, monkeypatch
):
    """Why the refusals must be taken out before grading, measured on upstream's own evaluator.
    `Environment.set_state` replays every MUTATING tool call of the trajectory whatever its
    ToolMessage says, then compares contents strictly. A refused write left in would therefore
    be executed on the predicted world and then fail the comparison against the refusal text."""
    from pi_run.stages.tau2_runner import _reward_of, split_refused

    rig = _Rig(cap=1, plans={"L1": [_create("a"), _create("b")]}, lines=["L1"])
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    sim, task = got["sim"], got["task"]

    native, err = _reward_of(sim, rig.suite, TID, task=task)
    assert native == {} and err.startswith("ValueError"), (native, err)

    refused_ids = {r["tool_call_id"]: r["mutating"] for r in status["refused_budget_calls"]}
    executed, refused = split_refused(sim.messages, refused_ids)
    assert [r["tool_name"] for r in refused] == ["create_task"]
    native, err = _reward_of(
        sim.model_copy(update={"messages": executed}), rig.suite, TID, task=task
    )
    assert err == "" and native["tau_reward"] == 1.0


# ------------------------------------------------------------------------------------ T7


def test_t7_refusals_are_recorded_apart_from_failures(tmp_path, monkeypatch):
    """The stored log keeps `env_calls` to what the environment EXECUTED -- a refused call in it
    would be replayed by anyone recomputing the reward offline -- and records each refusal on
    the status with its own fields, its mutating flag asked of the environment itself."""
    rig = _Rig(
        cap=2,
        plans={"L1": [_create("a"), _update_missing(), _create("b"), _read(), _create("c")]},
        lines=["L1"],
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert status["budget_enforcement"] == "refuse_and_tell_split"  # amendment 7
    assert status["budget_cap"] == 2
    assert status["tool_call_cap"] == 2
    assert status["n_refused_budget"] == 2
    refused = status["refused_budget_calls"]
    assert [(r["tool_name"], r["mutating"]) for r in refused] == [
        ("get_users", False),
        ("create_task", True),
    ]
    assert [r["transcript_idx"] for r in refused] == [4, 5]
    assert all(r["turn_idx"] == 1 and r["requestor"] == "assistant" for r in refused)

    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    assert [c["tool_name"] for c in outcome["env_calls"]] == [
        "create_task",
        "update_task_status",
        "create_task",
    ]
    assert status["n_env_calls"] == 3
    assert status["n_errors_agent"] == 1, "the one failure is the rejected write"

    on_disk = json.loads((got["dir"] / "status.json").read_text())
    assert on_disk["refused_budget_calls"] == refused


@pytest.mark.integration
def test_the_gate_on_the_real_retail_adapter_keeps_evidence_and_reconcile_true(
    tmp_path, monkeypatch
):
    """The rig above has an empty index, so it never exercises the half of the harvest that
    survives on a gated unit: minting evidence from the calls that EXECUTED and noting their
    documents, which `reconcile` then checks against the trajectory. On the real retail adapter,
    cap 2, a plan of four reads: two execute and are evidence, two are refused and are not."""
    from tests._tau2_data import require_tau2_data

    require_tau2_data("retail")
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    suite = Tau2RetailSuite()
    assert suite.task_ids()[0] == TID_REAL
    users = sorted(suite.db["users"])[:3]
    order = sorted(suite.db["orders"])[0]
    plan = [{"name": "get_user_details", "args": {"user_id": u}} for u in users]
    plan.append({"name": "get_order_details", "args": {"order_id": order}})

    rig = _Rig(cap=2, plans={"L1": plan}, lines=["L1"], suite=suite)
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert rig.executed == [("get_user_details", {"user_id": u}) for u in users[:2]]
    assert status["n_gate_charged"] == 2  # amendment 7: was spent.retrieval_calls
    assert [(r["tool_name"], r["mutating"]) for r in status["refused_budget_calls"]] == [
        ("get_user_details", False),
        ("get_order_details", False),
    ]
    assert status["n_evidence"] == 2, "evidence from the two reads that ran, none from refusals"
    assert status["reconcile"]["docs_ok"] and status["reconciled"]


def test_t7_a_suite_that_never_charged_tool_calls_is_not_gated(tmp_path, monkeypatch):
    """Banking's evidence arrives through a retriever and its tool calls were never charged
    (`_attach_env_evidence` is a no-op there). Gating them would invent a charge; the regime is
    recorded instead, so the two kinds of unit cannot be pooled by accident."""
    rig = _Rig(
        cap=2,
        plans={"L1": [_create("a"), _create("b"), _create("c"), _create("d")]},
        lines=["L1"],
        charged=False,
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert len(rig.executed) == 4
    assert status["budget_enforcement"] == "tools_uncharged"
    assert status["n_refused_budget"] == 0


# ------------------------------------------------------------------ GENERIC tools are not retrieval
#
# The user chose refuse-and-tell "so the agent can still answer or hand off". A GENERIC tool
# (`calculate`, `transfer_to_human_agents`, per conf/tau2/tool_types.json) reads no record and
# writes no row, so it is not retrieval: it is neither gated nor charged. The type is read from
# the committed map, never from a list of names here, and a name the map does not carry is NOT
# exempt -- it stays gated and charged, which is the safe direction.

COMMITTED_TYPES = Path(__file__).resolve().parents[1] / "conf" / "tau2" / "tool_types.json"


def _retail_rig(cap: int) -> _Rig:
    from tests._tau2_data import require_tau2_data

    require_tau2_data("retail")
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    return _Rig(cap=cap, plans={"L1": []}, lines=["L1"], suite=Tau2RetailSuite())


def _user_read(suite, i: int) -> dict:
    return {"name": "get_user_details", "args": {"user_id": sorted(suite.db["users"])[i]}}


@pytest.mark.integration
def test_a_hand_off_executes_after_exhaustion(tmp_path, monkeypatch):
    """(a) On the real retail adapter and the committed map: cap 1, one read spends it, a second
    read is refused, and `transfer_to_human_agents` still runs -- reaching `make_tool_call`,
    costing nothing, refusing nothing, and landing in `env_calls`."""
    rig = _retail_rig(1)
    handoff = {"name": "transfer_to_human_agents", "args": {"summary": "customer needs a human"}}
    rig.plans["L1"] = [_user_read(rig.suite, 0), _user_read(rig.suite, 1), handoff]
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert [n for n, _a in rig.executed] == ["get_user_details", "transfer_to_human_agents"]
    # amendment 7: the tool counter, not the ledger (was spent.retrieval_calls == 1.0)
    assert status["n_gate_charged"] == 1, "the hand-off was charged"
    assert status["n_transfer_to_human"] == 1
    assert [r["tool_name"] for r in status["refused_budget_calls"]] == ["get_user_details"]
    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    assert [c["tool_name"] for c in outcome["env_calls"]] == [
        "get_user_details",
        "transfer_to_human_agents",
    ]
    assert outcome["env_calls"][-1]["ok"] is True
    assert status["budget_exempt_tools"] == ["calculate", "transfer_to_human_agents"]


@pytest.mark.integration
def test_b_calculate_is_never_charged(tmp_path, monkeypatch):
    """(b) `calculate` before the budget is spent, between, and after: it runs every time and the
    ledger holds exactly the one read. Charged, it would have taken the cap of 1 itself and the
    read would have been refused."""
    rig = _retail_rig(1)
    calc = {"name": "calculate", "args": {"expression": "2 + 2"}}
    rig.plans["L1"] = [calc, _user_read(rig.suite, 0), calc, _user_read(rig.suite, 1), calc]
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")

    assert [n for n, _a in rig.executed] == [
        "calculate",
        "get_user_details",
        "calculate",
        "calculate",
    ]
    assert status["n_gate_charged"] == 1  # amendment 7: was spent.retrieval_calls
    assert [r["tool_name"] for r in status["refused_budget_calls"]] == ["get_user_details"]


def test_c_a_tool_the_map_does_not_type_is_still_gated(tmp_path, monkeypatch):
    """(c) A map that types the hand-off GENERIC and says nothing about `create_task`: after the
    one write spends the cap, the second write is REFUSED -- absence from the map is not an
    exemption -- while the hand-off runs. Before the exemption this failed on the hand-off (it
    was refused too); the unmapped-name half alone could not have failed there, because
    everything was gated, which is why the two halves are asserted together."""
    types = {"mock": {"transfer_to_human_agents": "GENERIC", "get_users": "READ"}}
    rig = _Rig(
        cap=1,
        plans={"L1": [_create("a"), _create("b"), _transfer()]},
        lines=["L1"],
        tool_types=types,
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert [n for n, _a in rig.executed] == ["create_task", "transfer_to_human_agents"]
    assert [(r["tool_name"], r["mutating"]) for r in status["refused_budget_calls"]] == [
        ("create_task", True)
    ]
    assert status["n_gate_charged"] == 1  # amendment 7: was spent.retrieval_calls


def test_c_the_exemption_is_read_from_the_map_and_only_as_generic():
    """The loader itself: GENERIC entries of the named domain and nothing else. An unmapped
    domain exempts nothing; READ, WRITE and unknown types are never exempt."""
    from pi_run.stages.tau2_runner import budget_exempt_tools

    committed = json.loads(COMMITTED_TYPES.read_text())
    for domain in ("retail", "airline"):
        want = {n for n, t in committed[domain].items() if t == "GENERIC"}
        assert want, f"the committed map types no {domain} tool GENERIC"
        assert budget_exempt_tools(domain) == want
    assert budget_exempt_tools("mock") == frozenset()
    assert budget_exempt_tools("banking_knowledge") == frozenset()


def test_a_map_that_types_a_write_generic_refuses_the_unit(tmp_path, monkeypatch):
    """The committed map is a copy of upstream's decorators at one tau2-bench commit. If it ever
    typed a tool GENERIC that the INSTALLED environment declares a write, the exemption would let
    a write go free of the budget. The unit refuses instead, before any call executes."""
    rig = _Rig(
        cap=1,
        plans={"L1": [_create("a"), _create("b")]},
        lines=["L1"],
        tool_types={"mock": {"create_task": "GENERIC"}},
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "error"
    assert "create_task" in status["error"] and "GENERIC" in status["error"]
    assert rig.executed == []


# ------------------------------------------------------------------ the refusal text is true

_IDENTIFIER = r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"


def _gated_tools_named(text: str) -> dict[str, list[str]]:
    """Every tool-like identifier in `text` that is NOT exempt in some gated domain -> those
    domains. Anything shaped like a tool name counts, mapped or not: an unmapped name is gated."""
    import re

    from pi_run.stages.tau2_runner import budget_exempt_tools

    domains = [d for d in json.loads(COMMITTED_TYPES.read_text()) if not d.startswith("_")]
    out: dict[str, list[str]] = {}
    for name in sorted(set(re.findall(_IDENTIFIER, text))):
        bad = [d for d in domains if name not in budget_exempt_tools(d)]
        if bad:
            out[name] = bad
    return out


def test_the_refusal_names_only_tools_that_still_run():
    """BUDGET_REFUSAL tells the agent what it may still do. Every tool it names must be exempt in
    every gated domain the committed map covers; a text naming a gated tool would be a lie the
    stock agent acts on. Non-vacuous twice over: the text must name at least one tool, and the
    same check must reject a text that names a gated one."""
    import re

    from pi_run.stages.tau2_runner import BUDGET_REFUSAL

    assert BUDGET_REFUSAL.startswith(PHRASE)
    assert re.findall(_IDENTIFIER, BUDGET_REFUSAL), "the text names no tool the agent may use"
    assert _gated_tools_named(BUDGET_REFUSAL) == {}
    assert _gated_tools_named(BUDGET_REFUSAL + " Or call get_user_details.") == {
        "get_user_details": ["retail", "airline"]
    }
    assert "No further tool calls" not in BUDGET_REFUSAL, "false once GENERIC tools still run"


# ------------------------------------------------------------------ an instrument that can disagree
#
# With `charge=False` the post-hoc meter never checks, so its overrun could not fire on a gated
# unit, and `BudgetGate.n_charged` cannot catch the gate that keeps it. The status therefore
# carries a census of the TRANSCRIPT (`env_call_census`): executed calls inside the inherited
# prefix, and live calls after it, split GENERIC / non-GENERIC. Under RULES amendment 7 the
# tool budget is on the fork's LIVE ok non-GENERIC calls after the fork point, and the ask
# budget on the ledger; the two never share a counter.

MOCK_TYPES = {
    "mock": {
        "create_task": "WRITE",
        "update_task_status": "WRITE",
        "get_users": "READ",
        "transfer_to_human_agents": "GENERIC",
    }
}


def _gated_fork(**kw) -> _Rig:
    """cap 2, a one-write prefix, one Ask, then [a, hand-off, rejected write, b, c]: the prefix
    write is free, a and b take the 2 tool calls, the hand-off is free, c is refused, and the
    Ask is on the ledger alone. (Cap 3 under the shared cap, where the Ask took one.)"""
    return _Rig(
        cap=2,
        asks={"L1": 1},
        plans={"L1": [_create("a"), _transfer(), _update_missing(), _create("b"), _create("c")]},
        lines=[],
        prefix=_prefix_with_one_write(),
        tool_types=MOCK_TYPES,
        **kw,
    )


def _identity(status: dict, outcome: dict) -> tuple[bool, bool]:
    """(spent == asks, n_gate_charged == chargeable live calls from env_calls).

    Amendment 7 changed the first half; under the shared cap it was
    `spent == asks + n_gate_charged`, and the ledger now never sees a tool call.

    The second half is recomputed HERE from `env_calls[n_prefix_env_calls:]` and the map, not
    read off the census, so the census itself is under test too."""
    generic = {n for n, t in MOCK_TYPES["mock"].items() if t == "GENERIC"}
    live = outcome["env_calls"][status["n_prefix_env_calls"] :]
    chargeable = sum(1 for c in live if c["ok"] and c["tool_name"] not in generic)
    return (
        status["spent"].get("retrieval_calls", 0.0) == status["n_asks"],
        status["n_gate_charged"] == chargeable,
    )


def test_a_gated_forks_live_spend_decomposes_from_its_transcript(tmp_path, monkeypatch):
    rig = _gated_fork()
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    outcome = json.loads((got["dir"] / "outcome.json").read_text())

    # The boundary is where upstream put it: the prefix's own messages, first, in order.
    prefix = _prefix_with_one_write()
    head = got["sim"].messages[: len(prefix)]
    assert [(m.role, m.content) for m in head] == [(m.role, m.content) for m in prefix]

    assert status["n_prefix_env_calls"] == 1
    assert status["n_live_env_calls"] == 4, "a, hand-off, rejected write, b; c was refused"
    assert status["n_live_env_calls_generic"] == 1
    assert status["n_live_env_calls_nongeneric"] == 3
    assert status["n_live_env_calls_nongeneric_ok"] == 2
    assert status["n_gate_charged"] == 2
    assert status["n_env_calls"] == 5, "the whole transcript, prefix included"
    assert status["spent"]["retrieval_calls"] == 1.0, "the one Ask, and no tool call"
    assert status["tool_call_cap"] == 2
    assert _identity(status, outcome) == (True, True)
    assert status["gate_census_agrees"] is True
    assert "post_hoc_budget_overrun" not in status["spent"]


def test_a_gate_that_double_counts_fails_the_identity(tmp_path, monkeypatch):
    """NON-VACUITY. The same unit under a gate whose counter moves twice per charge. It refuses
    `b` early (its counter reads 2 after `a`), and only an instrument outside the gate -- the
    transcript recount -- sees that 2 charges stand for 1 executed call."""
    from pi_run.stages import tau2_runner as T

    class DoubleCount(T.BudgetGate):
        def install(self, orch):
            super().install(orch)
            inner = orch._execute_tool_calls

            def twice(tool_calls):
                before = self.n_charged
                out = inner(tool_calls)
                self.n_charged += self.n_charged - before
                return out

            orch._execute_tool_calls = twice

    monkeypatch.setattr(T, "BudgetGate", DoubleCount)
    status, got = _gated_fork().run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    outcome = json.loads((got["dir"] / "outcome.json").read_text())
    assert status["n_gate_charged"] == 2
    assert status["n_live_env_calls_nongeneric_ok"] == 1
    assert _identity(status, outcome) == (True, False)
    assert status["gate_census_agrees"] is False


def test_a_gate_that_leaks_is_reported_as_an_overrun(tmp_path, monkeypatch):
    """NON-VACUITY for the overrun row. A gate that lets a write through once its own counter is
    full: `c` executes past the tool cap uncounted. Neither the ledger nor the gate's counter
    can see it -- only the transcript can show that a forbidden call ran, and the revived
    `post_hoc_budget_overrun` (live ok non-GENERIC calls > tool_call_cap) fires on it."""
    from pi_run.stages import tau2_runner as T

    class Leaky(T.BudgetGate):
        def install(self, orch):
            super().install(orch)
            inner = orch._execute_tool_calls
            execute = type(orch)._execute_tool_calls.__get__(orch)

            def leak(tool_calls):
                out = []
                for tc in tool_calls:
                    spent = self.n_charged >= self.cap
                    out.extend(execute([tc]) if spent and tc.name == "create_task" else inner([tc]))
                return out

            orch._execute_tool_calls = leak

    monkeypatch.setattr(T, "BudgetGate", Leaky)
    rig = _gated_fork()
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert [a["title"] for n, a in rig.executed if n == "create_task"] == ["p", "a", "b", "c"]
    assert status["spent"]["retrieval_calls"] == 1.0, "the ledger holds the Ask only"
    assert status["n_gate_charged"] == 2, "the gate cannot see its own leak"
    assert status["n_live_env_calls_nongeneric_ok"] == 3 > status["tool_call_cap"]
    assert status["gate_census_agrees"] is False
    assert status["spent"].get("post_hoc_budget_overrun") == 1.0


def test_a_history_ending_on_a_pending_call_counts_that_call_as_live():
    """THE EDGE. A call inside the inherited history whose ANSWER falls after the boundary was
    executed live -- upstream would run it through the gate -- so the census counts it live, and
    `env_calls[n_prefix_env_calls:]` still starts at it, because every answered prefix call
    precedes it."""
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

    from pi_run.stages.tau2_runner import env_call_census, env_calls_from

    def call(i: str) -> ToolCall:
        return ToolCall(id=i, name="create_task", arguments={"title": i}, requestor="assistant")

    def answer(i: str) -> ToolMessage:
        return ToolMessage(id=i, role="tool", content="{}", requestor="assistant", error=False)

    history = [
        AssistantMessage(role="assistant", content="Hi", cost=0.0),
        UserMessage(role="user", content="P1"),
        AssistantMessage(role="assistant", content=None, tool_calls=[call("pre")], cost=0.0),
        answer("pre"),
        AssistantMessage(role="assistant", content=None, tool_calls=[call("pend")], cost=0.0),
    ]
    live = [
        answer("pend"),
        AssistantMessage(role="assistant", content=None, tool_calls=[call("own")], cost=0.0),
        answer("own"),
    ]
    census = env_call_census(history + live, boundary=len(history), refused={}, exempt=frozenset())
    assert census["n_prefix_env_calls"] == 1
    assert census["n_live_env_calls"] == census["n_live_env_calls_nongeneric_ok"] == 2
    tail = env_calls_from(history + live)[census["n_prefix_env_calls"] :]
    assert [json.loads(c.kwargs_json)["title"] for c in tail] == ["pend", "own"]


def test_the_pending_call_shape_cannot_start_a_dialogue_at_this_tau2():
    """Why the edge above is tested on the census and not through a unit: at the installed tau2,
    `Orchestrator.initialize` replays the history into the live world with `set_state`, which
    raises on a trailing unanswered call before `self.message = last_message` is reached; and
    `forked_task` refuses the shape before that. If an upgrade makes it startable, this fails and
    the unit-level test becomes writable."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.data_model.tasks import Task
    from tau2.orchestrator.orchestrator import Orchestrator

    from pi_run.stages.tau2_runner import forked_task

    tc = ToolCall(id="pend", name="create_task", arguments={"user_id": USER, "title": "p"})
    history = [
        AssistantMessage(role="assistant", content="Hi", cost=0.0),
        UserMessage(role="user", content="P1"),
        AssistantMessage(role="assistant", content=None, tool_calls=[tc], cost=0.0),
    ]
    with pytest.raises(ValueError, match="plain user turn"):
        forked_task(Task.model_validate(TASK), history)

    task = Task.model_validate(TASK).model_copy(deep=True)
    from tau2.data_model.tasks import InitialState

    task.initial_state = InitialState(message_history=history)

    class _Party:
        def get_init_state(self, message_history=None):
            return {}

        def set_seed(self, seed):
            return None

        def is_stop(self, message):
            return False

        def generate_next_message(self, message, state):  # pragma: no cover - not reached
            raise AssertionError("initialize must fail before any turn")

    orch = Orchestrator(domain="mock", agent=_Party(), user=_Party(), environment=_env(), task=task)
    with pytest.raises(ValueError, match="Tool message expected"):
        orch.initialize()


def _history_with_an_unanswered_call_in_the_middle() -> list:
    """A prefix that ends correctly on a user turn, with an unanswered call BEFORE the end."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    tc = ToolCall(id="mid", name="create_task", arguments={"user_id": USER, "title": "m"})
    return [
        AssistantMessage(role="assistant", content="Hi", cost=0.0),
        UserMessage(role="user", content="P1"),
        AssistantMessage(role="assistant", content=None, tool_calls=[tc], cost=0.0),
        UserMessage(role="user", content="L1"),
    ]


def test_an_unanswered_call_in_the_middle_of_a_prefix_cannot_start_a_unit(tmp_path, monkeypatch):
    """The reader indexes `env_calls` from `n_prefix_env_calls`, which counts ANSWERED prefix calls
    only, so an unanswered call inside a prefix would shift that index. It cannot occur: upstream's
    `Environment.set_state` (tau2/environment/environment.py:319-350, reached from
    `Orchestrator.initialize` via `_initialize_environment` at orchestrator.py:519, BEFORE
    `validate_message_history` at :532) pops, for every tool call, the very next message and
    raises unless it is the matching ToolMessage -- "Tool message expected. Got <type>" at :343.
    That check is not the `strict` one (which only compares contents), so it holds in both modes.

    A PIN, NOT A FIX: nothing in this commit changes it, and it passed before this commit too.
    Checked both at upstream's own function and end to end through `run_tau2_unit`, where
    `forked_task` accepts the prefix (it ends on a plain user turn) and the unit then fails before
    any tool call executes -- so no such unit can ever write an `env_calls` log at all."""
    history = _history_with_an_unanswered_call_in_the_middle()
    for strict in (True, False):
        with pytest.raises(ValueError, match="Tool message expected. Got <class"):
            _env().set_state(
                initialization_data=None,
                initialization_actions=None,
                message_history=list(history),
                strict=strict,
            )

    rig = _Rig(cap=4, plans={"L1": [_create("a")]}, lines=[], prefix=history)
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "error"
    assert "Tool message expected" in status["error"]
    assert rig.executed == [], "nothing executed: the unit died at initialisation"


# ------------------------------------------------------------------ RULES amendment 4 secondaries


def test_the_first_rollout_and_the_spend_before_the_first_live_call(tmp_path, monkeypatch):
    """`rollout1_*` is the FIRST run_loop rollout, not the last one `stop_reason` reports; and
    `spent_before_first_live_call` is the ledger at the moment the first live non-GENERIC call
    executes, before its own charge -- here the 2 + 3 asks of two rollouts. The hand-off that
    runs before it is GENERIC and does not set it."""
    rig = _Rig(
        cap=16,
        asks={"L1": 2, "L2": 3},
        plans={"L1": [], "L2": [_transfer(), _create("a"), _create("b")]},
        lines=["L1", "L2"],
        tool_types=MOCK_TYPES,
    )
    status, got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert len(got["state"].trajectories) == 2
    assert status["rollout1_stop_reason"] == "policy_stop"
    assert status["rollout1_n_asks"] == 2
    assert status["n_asks"] == 5
    assert status["first_live_call_executed"] is True
    assert status["spent_before_first_live_call"] == 5.0, "the asks charged before the call"
    assert [n for n, _a in rig.executed] == [
        "transfer_to_human_agents",
        "create_task",
        "create_task",
    ]


def test_asks_that_exhaust_their_budget_still_reach_the_first_live_call(tmp_path, monkeypatch):
    """Rollout 1 asks 2 and stops; rollout 2's asks fill the ask cap of 6 and it stops on `budget`,
    so `rollout1_stop_reason` and `stop_reason` differ. Its plan's hand-off runs (GENERIC) and its
    write EXECUTES on the tool budget, with the ask ledger full at 6.

    AMENDMENT 7 changed this test. Its belief under the shared cap was that asks filling the cap
    refused the write, leaving `spent_before_first_live_call` null; with split budgets that no
    longer happens, and a null now needs a unit that never asks for a non-GENERIC call (below)."""
    rig = _Rig(
        cap=6,
        asks={"L1": 2, "L2": 5},
        plans={"L1": [], "L2": [_transfer(), _create("a")]},
        lines=["L1", "L2"],
        tool_types=MOCK_TYPES,
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert status["rollout1_stop_reason"] == "policy_stop"
    assert status["rollout1_n_asks"] == 2
    assert status["stop_reason"] == "budget"
    assert status["spent"]["retrieval_calls"] == 6.0
    assert status["first_live_call_executed"] is True
    assert status["spent_before_first_live_call"] == 6.0
    assert status["n_refused_budget"] == 0
    assert [n for n, _a in rig.executed] == ["transfer_to_human_agents", "create_task"]
    assert status["n_transfer_to_human"] == 1


def test_a_unit_with_no_live_non_generic_call_has_no_first_live_call(tmp_path, monkeypatch):
    """The null case: only the hand-off runs, which is GENERIC and does not count."""
    rig = _Rig(
        cap=16,
        asks={"L1": 3},
        plans={"L1": [_transfer()]},
        lines=["L1"],
        tool_types=MOCK_TYPES,
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "ok", status.get("error")
    assert status["first_live_call_executed"] is False
    assert status["spent_before_first_live_call"] is None
    assert [n for n, _a in rig.executed] == ["transfer_to_human_agents"]


def test_the_secondaries_survive_onto_an_error_status(tmp_path, monkeypatch):
    """A unit killed AFTER its dialogue (here the outcome step raises) still records them: the
    agent state and the gate both exist by then. A unit killed before any rollout records the
    rollout fields as null and the gate's as not-executed."""
    from pi_run.stages import tau2_runner as T

    def boom(*a, **k):
        raise RuntimeError("injected after the dialogue")

    monkeypatch.setattr(T, "_with_outcome", boom)
    rig = _Rig(
        cap=16,
        asks={"L1": 1},
        plans={"L1": [_create("a")]},
        lines=["L1"],
        tool_types=MOCK_TYPES,
    )
    status, _got = rig.run(tmp_path, monkeypatch)
    assert status["status"] == "error" and "injected" in status["error"]
    assert status["rollout1_stop_reason"] == "policy_stop"
    assert status["rollout1_n_asks"] == 1
    assert status["first_live_call_executed"] is True
    assert status["spent_before_first_live_call"] == 1.0

    monkeypatch.undo()
    early = _Rig(
        cap=1,
        plans={"L1": [_create("a")]},
        lines=["L1"],
        tool_types={"mock": {"create_task": "GENERIC"}},  # refused at install: no rollout ran
    )
    (tmp_path / "early").mkdir()
    status, _got = early.run(tmp_path / "early", monkeypatch)
    assert status["status"] == "error"
    assert status["rollout1_stop_reason"] is None and status["rollout1_n_asks"] is None
    assert status["first_live_call_executed"] is False
    assert status["spent_before_first_live_call"] is None
