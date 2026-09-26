"""The tau2 TELECOM suite, VIEW SIDE. EVAL-ONLY: a second zero-shot transfer target.

WHY EVAL-ONLY (D20). Banking already carries the transfer claim, and one transfer target is a
single point. Telecom is a second action-consequential environment with a different mechanic --
the agent instructs and the CUSTOMER acts on the handset -- so a policy that transfers to both
has been shown to transfer, not to have fitted one domain's shape. Airline is the tool domain
we mine; holding telecom whole is what keeps that claim available. `EVAL_ONLY_SUITES` enforces
it in `split_of` before any hash is taken, and `pi suites audit` refuses a suite that is both.

THREE THINGS TELECOM DOES THAT NEITHER BANKING NOR RETAIL DOES.

  THE GOLD IS MOSTLY THE USER'S ACTIONS. 393 of the 516 gold actions in the base split are
  `requestor: user` (MEASURED) -- `toggle_airplane_mode`, `set_network_mode_preference`,
  `grant_app_permission`. The agent's job is largely to work out what the customer must do and
  say so. That is a proactivity mechanic banking does not have: the wrong instruction is not
  a wrong tool call, it is a wasted user turn.

  THERE ARE TWO DATABASES AND THE SECOND IS THE USER'S. `db.toml` is the agent's; `user_db.toml`
  is the handset's own state, coupled by `TelecomEnvironment.sync_tools`. Only the first is a
  corpus -- see `telecom_units` for why indexing the second would silently raise the
  discoverability ceiling.

  THE REWARD BASIS IS NOT `DB`. Telecom's 114 base tasks are `ENV_ASSERTION` (94) or
  `ACTION+ENV_ASSERTION` (20), where banking, airline and retail are all DB-based. `Tau2Actuator`
  writes `tau_reward` only from a DB check, so without the basis rule in `attach_reward` this
  suite would produce no reward at all; the 20 ACTION-basis tasks stay ungraded and NAMED,
  because the environment evaluator does not cover that component and a free 1.0 there would be
  a fabricated success.

`view()` refuses for the reason it refuses everywhere in tau2: the only task text outside the
Orchestrator is the customer's roleplay script.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pinq.types import TaskId, TaskView
from pinq.view import make_view

from ._probe import available, domain_data_dir, policy_enabled
from .telecom_units import CORPUS_ID, TelecomIndex

TELECOM_DOMAIN = "telecom"

# Upstream's default policy. `workflow` is a different domain name upstream
# (`telecom-workflow`) with a different policy document, and the public leaderboard
# trajectories were produced under `manual`; running a different one would make our number
# and theirs incomparable without saying so.
POLICY_TYPE = "manual"

# The 114-task `base` split. `full` is 2,285 tasks generated combinatorially, which would make
# a sweep cost 20x for no additional claim.
TASK_SPLIT = "base"


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
    "You are a telecom customer-service agent. Diagnose the customer's problem using the "
    "account, line, device and billing records available to you, and tell the customer what "
    "to do on their handset when the fix is theirs to make. Some facts are visible only on "
    "their device; others must be looked up. Ask before assuming."
)
INSTRUCTIONS = INSTRUCTIONS + _domain_policy("telecom")


class Tau2TelecomUnavailable(RuntimeError):
    """tau2 is not importable, or its data directory is missing."""


@dataclass
class Tau2TelecomSuite:
    """Satisfies the parts of `pinq.protocols.TaskSuite` the Orchestrator path uses."""

    suite_id: ClassVar[str] = "tau2_telecom"
    suite_version: ClassVar[str] = "v1.0.1"
    corpus_id: ClassVar[str] = CORPUS_ID
    instructions: ClassVar[str] = INSTRUCTIONS
    domain: ClassVar[str] = TELECOM_DOMAIN
    word_cap: ClassVar[int] = 180
    transcript_is_synthetic: ClassVar[bool] = True

    allow_user_script: bool = False

    _tasks: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _envs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    db: dict = field(default_factory=dict, init=False, repr=False)
    index: TelecomIndex | None = field(default=None, init=False, repr=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        d = domain_data_dir(TELECOM_DOMAIN)
        with (d / "db.toml").open("rb") as fh:
            self.db = tomllib.load(fh)
        rows = json.loads((d / "tasks.json").read_text())
        by_id = {str(r["id"]): r for r in rows}
        # `tasks.json` holds all 2,285; `split_tasks.json["base"]` names the 114 this suite
        # reports on. Nothing else in the repo reads that file, so the restriction is applied
        # here rather than assumed.
        base = json.loads((d / "split_tasks.json").read_text())[TASK_SPLIT]
        self._tasks = {str(tid): by_id[str(tid)] for tid in base if str(tid) in by_id}
        self.index = TelecomIndex.from_db(self.db)
        self.corpus_hash = self.index.corpus_hash()

    # ------------------------------------------------------------------ tasks

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._tasks)

    def task_record(self, tid: TaskId) -> dict:
        """The raw task JSON. GOLD-BEARING (`evaluation_criteria`); never handed to a policy."""
        return self._tasks[str(tid)]

    @staticmethod
    def template_id_for(tid: TaskId) -> str | None:
        """None. Telecom ids do encode a fault family, but this suite is eval-only, so the
        split never consults it -- and turning the families into CI clusters would be a
        clustering decision made by a side effect rather than argued for."""
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
                f"tau2_telecom/{tid}: the only task text available is the customer's script, "
                "which inverts the agent's role and hands over the user-private partition. "
                "Drive telecom through the Orchestrator, or pass "
                "Tau2TelecomSuite(allow_user_script=True) for debugging -- runs made that way "
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
        from tau2.domains.telecom.environment import get_tasks

        want = str(tid)
        for t in get_tasks(task_split_name=TASK_SPLIT):
            if str(getattr(t, "id", "")) == want:
                return t
        raise KeyError(f"no telecom task {want!r} in the {TASK_SPLIT!r} split")

    def env_kwargs(self, tid: TaskId | None = None) -> dict[str, Any]:
        """`policy_type` is the one piece of configuration telecom genuinely has, and it
        selects a different POLICY DOCUMENT. Passing it explicitly keeps the run comparable
        with the public trajectories, which were produced under `manual`."""
        return {"policy_type": POLICY_TYPE}

    def environment(self, tid: TaskId | None = None) -> Any:
        """The live telecom Environment, MEMOISED PER TASK.

        Per task for the reason retail is: 20 of the base tasks ship
        `initialization_actions` and the dialogue mutates both databases, so one shared
        environment would carry a suspended line or a toggled radio into the next task's
        starting state.
        """
        ok, why = available()
        if not ok:
            raise Tau2TelecomUnavailable(why)
        key = str(tid) if tid is not None else ""
        if key in self._envs:
            return self._envs[key]
        from tau2.domains.telecom.environment import get_environment

        env = get_environment(**self.env_kwargs(tid))
        self._envs[key] = env
        return env

    def actuator(self, tid: TaskId) -> Any:
        """THE SAME Environment the schemas and retriever were built from."""
        from .actuator import Tau2Actuator

        return Tau2Actuator(
            self.environment(tid),
            task=self.task_record(tid),
            domain=self.domain,
            env_kwargs=self.env_kwargs(tid),
        )

    def tool_schemas(self, tid: TaskId) -> tuple[dict[str, Any], ...]:
        """The AGENT's tools only. `env.get_tools()` returns those; the customer's handset
        tools live on `user_tools` and are the simulator's to call. Advertising them would let
        a Drafter plan an action only the customer can take."""
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
        """A READ-only tool selector, carrying the SUITE's corpus_hash."""
        from .telecom_units import uids_for_call
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
        """The agent DB at record grain. `user_db.toml` is deliberately absent."""
        from .db_units import evidence_units
        from .telecom_units import units_from_db

        return evidence_units(units_from_db(self.db))

    def uids_for_calls(self, calls: Any) -> tuple[str, ...]:
        """Records a turn's recorded EnvCalls read, from ARGUMENTS only -- never a result."""
        from .telecom_units import uids_from_env_calls

        assert self.index is not None
        return uids_from_env_calls(self.index, calls)
