"""The tau2 AIRLINE suite, VIEW SIDE. A sibling of `Tau2RetailSuite`, not a parameterisation.

WHY IT IS A SEPARATE SUITE ID AND WHY IT IS THE TRAINING ONE. Airline is the only tau2 domain
upstream ships a task split for -- `split_tasks.json`, 30 train and 20 test -- and the public
trajectories worth forking from were produced against that split. Adopting upstream's split
rather than hashing ids ourselves is what makes a prefix taken from a public "train" trace
safe; see `pinq.splitting.FIXED_SPLITS`. Banking stays eval-only and telecom is held whole, so
this is the tool domain the SFT shard is mined from.

WHAT IT SHARES WITH RETAIL AND BANKING. `view()` refuses. The only task text outside the
Orchestrator is `user_scenario.instructions` -- the customer's roleplay script -- and handing
it to the agent inverts the role (observed live on banking: the agent answered AS the customer)
and gives away the user-private partition the discoverability ceiling is defined over.

WHAT IT DOES NOT SHARE. Banking searches 698 documents through `KB_search`; airline has 13
typed tools and no free-text search at all. Its retrieval channel is `ToolBackedRetriever`,
which answers a question by selecting one READ-only tool -- without it an `Ask` on this suite
resolves to nothing, which is the state retail was in.

MUTATION IS WHY THE ENVIRONMENT IS MEMOISED PER TASK. `book_reservation`,
`update_reservation_*` and `cancel_reservation` move the DB. Sharing one environment across
tasks would leak a booking from one task into another's starting state; rebuilding it mid
episode would reset state the episode depends on. Both are true of retail and true here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pinq.types import TaskId, TaskView
from pinq.view import make_view

from ._probe import available, domain_data_dir, policy_enabled
from .airline_units import CORPUS_ID, AirlineIndex

AIRLINE_DOMAIN = "airline"

# What the agent is told the job is. Says nothing about the customer's situation: that arrives
# through the user simulator, which is the whole point of the Orchestrator path.


# The policy says to obtain confirmation before a write; it never says what to do once you have
# one. MEASURED on 60 matched retail fork points, the policy ALONE cut invalid writes (failed
# writes 0.95 -> 0.30, p=0.0003) and also cut action itself (writes 2.12 -> 1.17, p=0.0005),
# costing reward where the failure was already under-acting (tau 0.533 -> 0.417). On airline,
# where failures over-act, the same change gained (+0.196, p=0.0074). This states the half the
# policy leaves implicit, and it is a restatement of the rules rather than a fact about any task.
_COMPLETION_CLAUSE = (
    "\n\nOnce you have the confirmation this policy requires, carry it out in the same reply: "
    "make the tool call. Describing an action you have not made leaves the customer's request "
    "unfulfilled."
)


def _domain_policy(domain: str) -> str:
    """tau2's own `policy.md` for this domain -- the AGENT's rulebook.

    NOT the document `view()` refuses. That one is `user_scenario.instructions`, the CUSTOMER's
    roleplay script: handing it over inverts the role and gives away the user-private partition
    the discoverability ceiling is defined over. This is domain-level, identical for every task,
    and names no user, reservation or order.

    Upstream this file IS the agent's system prompt, which is how the published 92-96% agents
    know the rules. Ours did not have it: MEASURED, airline failures are dominated by
    `cancel_reservation` (39 of 48 writes across 34 failures against 7 across 26 successes) and
    every one of those cancels had already READ the reservation. The agent was not cancelling
    blind, it did not know the rules -- which live only here, along with "The current time is
    2024-05-15 15:00:00 EST" that the 24-hour window cannot be evaluated without.

    Absent or unreadable, the generic instructions stand alone rather than the suite failing to
    construct: a missing upstream file is a degraded prompt, not a broken adapter.
    """
    if not policy_enabled(domain):
        return ""
    try:
        p = domain_data_dir(domain) / "policy.md"
        text = p.read_text().strip() if p.is_file() else ""
    except Exception:  # noqa: BLE001 - a missing rulebook must not break suite construction
        text = ""
    return f"\n\nOPERATING POLICY\n{text}{_COMPLETION_CLAUSE}" if text else ""


INSTRUCTIONS = (
    "You are an airline customer-service agent. Resolve the customer's request by calling the "
    "tools available to you against the reservation, flight and user records. Several facts "
    "you need are held only by the customer; others must be looked up. Ask before assuming."
)
INSTRUCTIONS = INSTRUCTIONS + _domain_policy("airline")


class Tau2AirlineUnavailable(RuntimeError):
    """tau2 is not importable, or its data directory is missing."""


@dataclass
class Tau2AirlineSuite:
    """Satisfies the parts of `pinq.protocols.TaskSuite` the Orchestrator path uses."""

    suite_id: ClassVar[str] = "tau2_airline"
    suite_version: ClassVar[str] = "v1.0.1"
    corpus_id: ClassVar[str] = CORPUS_ID
    instructions: ClassVar[str] = INSTRUCTIONS
    domain: ClassVar[str] = AIRLINE_DOMAIN
    word_cap: ClassVar[int] = 180
    # The tau2 databases are synthetic fixtures, so a stored transcript carries no customer.
    # `tau2_runner` writes `simulation.json` only for suites that say so; the digest-only
    # default stays in force for anything holding real people.
    transcript_is_synthetic: ClassVar[bool] = True

    allow_user_script: bool = False

    _tasks: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _envs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    db: dict = field(default_factory=dict, init=False, repr=False)
    index: AirlineIndex | None = field(default=None, init=False, repr=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        d = domain_data_dir(AIRLINE_DOMAIN)
        self.db = json.loads((d / "db.json").read_text())
        rows = json.loads((d / "tasks.json").read_text())
        self._tasks = {str(r["id"]): r for r in rows}
        self.index = AirlineIndex.from_db(self.db)
        # MUST equal the gold builder's corpus hash. Gold is written against this DB, and a
        # disagreement is invisible downstream: an unmatched uid reads as "the policy
        # retrieved nothing relevant" rather than as an error.
        self.corpus_hash = self.index.corpus_hash()

    # ------------------------------------------------------------------ tasks

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._tasks)

    def task_record(self, tid: TaskId) -> dict:
        """The raw task JSON. GOLD-BEARING (`evaluation_criteria`); never handed to a policy."""
        return self._tasks[str(tid)]

    @staticmethod
    def template_id_for(tid: TaskId) -> str | None:
        """None, as retail does. An airline id is a bare integer with no composition to
        recover, and inventing a grouping would cluster unrelated tasks into one CI cluster.
        The split does not need it either: `FIXED_SPLITS` assigns every id by name."""
        return None

    def template_id(self, tid: TaskId) -> str | None:
        return self.template_id_for(tid)

    # ------------------------------------------------------------------ the view

    def view(self, tid: TaskId) -> TaskView:
        """REFUSES. `pi suites validate` asserts the refusal rather than reporting it."""
        from .suite import Tau2NeedsOrchestrator

        rec = self._tasks[str(tid)]
        ins = (rec.get("user_scenario") or {}).get("instructions") or {}
        script = ins if isinstance(ins, str) else json.dumps(ins, sort_keys=True)
        if not self.allow_user_script:
            raise Tau2NeedsOrchestrator(
                f"tau2_airline/{tid}: the only task text available is the customer's script, "
                "which inverts the agent's role and hands over the user-private partition. "
                "Drive airline through the Orchestrator, or pass "
                "Tau2AirlineSuite(allow_user_script=True) for debugging -- runs made that way "
                "are excluded from reported tables."
            )
        return make_view(
            task_id=str(tid),
            suite_id=self.suite_id,
            question=script,
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=self.word_cap,
        )

    # ------------------------------------------------------------------ environment

    def tau2_task_object(self, tid: TaskId) -> Any:
        from tau2.domains.airline.environment import get_tasks

        want = str(tid)
        for t in get_tasks():
            if str(getattr(t, "id", "")) == want:
                return t
        raise KeyError(f"no airline task {want!r}")

    def env_kwargs(self, tid: TaskId | None = None) -> dict[str, Any]:
        """Airline's environment takes no retrieval variant and no read-log allowlist: it has
        neither a document index nor discoverable tools, so passing either would be inventing
        configuration the domain does not have."""
        return {}

    def environment(self, tid: TaskId | None = None) -> Any:
        """The live airline Environment, MEMOISED PER TASK. See the module docstring."""
        ok, why = available()
        if not ok:
            raise Tau2AirlineUnavailable(why)
        key = str(tid) if tid is not None else ""
        if key in self._envs:
            return self._envs[key]
        from tau2.domains.airline.environment import get_environment

        # `solo_mode` is not passed: airline's get_environment RAISES on it, and every run
        # here is a two-participant dialogue.
        env = get_environment()
        self._envs[key] = env
        return env

    def actuator(self, tid: TaskId) -> Any:
        """THE SAME Environment the schemas and retriever were built from -- a rollout is one
        world, and a second Environment would let the policy read from world A while the
        graded mutations landed in world B."""
        from .actuator import Tau2Actuator

        return Tau2Actuator(
            self.environment(tid),
            task=self.task_record(tid),
            domain=self.domain,
            env_kwargs=self.env_kwargs(tid),
        )

    def tool_schemas(self, tid: TaskId) -> tuple[dict[str, Any], ...]:
        """Flat `name`/`description`/`parameters`, sorted by name -- the shape `tau2_runner`
        reads `t["name"]` from. A nested OpenAI form killed live dialogues on `KeyError`."""
        from .suite import _description_of, _params_of

        env = self.environment(tid)
        return tuple(
            {"name": t.name, "description": _description_of(t), "parameters": _params_of(t)}
            for t in sorted(env.get_tools(), key=lambda t: t.name)
        )

    def retriever(
        self,
        tid: TaskId,
        *,
        llm: Any = None,
        llm_free: bool = False,
        seed: int = 0,
        recorder: Any = None,
    ) -> Any:
        """A READ-only tool selector. Carries the SUITE's corpus_hash, not a fresh one: that
        hash binds a run to the DB its gold was built against, and `pi score` refuses a
        mismatched pairing."""
        from .airline_units import uids_for_call
        from .tool_retriever import ToolBackedRetriever

        return ToolBackedRetriever(
            self.environment(tid),
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            index=self.index,
            uids_for_call=uids_for_call,
            tool_schemas=self.tool_schemas(tid),
            llm=llm,
            llm_free=llm_free,
            seed=seed,
            recorder=recorder,
        )

    # ------------------------------------------------------------------ evidence

    def units(self, tid: TaskId | None = None) -> tuple[dict[str, Any], ...]:
        """The corpus at the grain a tool call returns. `cmd_train.render_state` needs a unit
        pool to rebuild a prompt; without one no dialogue run is exportable."""
        from .airline_units import units_from_db
        from .db_units import evidence_units

        return evidence_units(units_from_db(self.db))

    def uids_for_calls(self, calls: Any) -> tuple[str, ...]:
        """Records a turn's recorded EnvCalls read, from ARGUMENTS only -- never a result."""
        from .airline_units import uids_from_env_calls

        assert self.index is not None
        return uids_from_env_calls(self.index, calls)
