"""The tau2 RETAIL suite, VIEW SIDE. A sibling of `Tau2Suite`, not a parameterisation of it.

WHAT IT SHARES WITH BANKING
    `view()` refuses, for the same two reasons. The only task text available is the customer's
    roleplay script (`user_scenario.instructions`), and handing it to the agent inverts the
    role — observed live on banking, the agent answered AS the customer and never attempted the
    task — and hands over every user-private fact for free, collapsing the partition that
    bounds what any autonomous inquirer could reach. Retail is driven through the Orchestrator,
    where the task arrives turn by turn through the user simulator.

WHAT IT CANNOT SHARE
    The retriever. Banking searches 698 knowledge documents through `search_documents`; retail
    has 15 typed, key-addressed tools over a relational DB and NO free-text search. There is
    no retrieval channel to wrap, so evidence is reconstructed from what tool calls asked for
    (`retail_units.uids_from_env_calls`) rather than from anything returned.

    That is also why retail is the suite where the user channel matters. With no search
    surface, a fact the customer holds and the DB does not carry is reachable only by ASKING —
    which is what `inquirer_may_ask_user` exists to measure and, until the loop gate was
    derived from the arm rather than pinned False, could not.

MUTATION IS WHY THE ENVIRONMENT IS MEMOISED PER TASK AND NOT SHARED ACROSS THEM
    176 of retail's 550 required calls mutate the DB (`cancel_pending_order`, the `modify_*`
    family, `return_*`, `exchange_*`). Rebuilding the environment mid-episode would reset state
    the episode depends on; sharing one across tasks would leak a mutation from one task into
    another's starting state. Both are tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pinq.types import TaskId, TaskView
from pinq.view import make_view

from ._probe import RETRIEVAL_VARIANT, available, domain_data_dir, policy_enabled
from .retail_units import CORPUS_ID, RetailIndex

RETAIL_DOMAIN = "retail"

# What the agent is told the job is. Deliberately says nothing about the customer's situation:
# that arrives through the user simulator, which is the whole point of the Orchestrator path.


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
    "You are a retail customer-service agent. Resolve the customer's request by calling the "
    "tools available to you against the order, user and product records. Several facts you "
    "need are held only by the customer; others must be looked up. Ask before assuming."
)
INSTRUCTIONS = INSTRUCTIONS + _domain_policy("retail")


class Tau2RetailUnavailable(RuntimeError):
    """tau2 is not importable, or its data directory is missing."""


@dataclass
class Tau2RetailSuite:
    """Satisfies the parts of `pinq.protocols.TaskSuite` the Orchestrator path uses."""

    suite_id: ClassVar[str] = "tau2_retail"
    suite_version: ClassVar[str] = "v1.0.1"
    corpus_id: ClassVar[str] = CORPUS_ID
    instructions: ClassVar[str] = INSTRUCTIONS
    # Spelled the same way banking spells it, and read the same way by `tau2_runner`. See
    # `Tau2Suite.domain`: the driver serves both suites and may not name either domain.
    domain: ClassVar[str] = RETAIL_DOMAIN
    # Retail answers are conversational rather than a short span; matched to banking's cap so
    # answer length can never be the thing a judge is rewarding across the two.
    word_cap: ClassVar[int] = 180

    allow_user_script: bool = False
    retrieval_variant: str = RETRIEVAL_VARIANT

    _tasks: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _envs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    db: dict = field(default_factory=dict, init=False, repr=False)
    index: RetailIndex | None = field(default=None, init=False, repr=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        d = domain_data_dir(RETAIL_DOMAIN)
        self.db = json.loads((d / "db.json").read_text())
        rows = json.loads((d / "tasks.json").read_text())
        self._tasks = {str(r["id"]): r for r in rows}
        self.index = RetailIndex.from_db(self.db)
        # MUST equal `retail_build.corpus_hash_of`. Gold is written against this DB; if the two
        # disagree the mismatch is invisible downstream, because an unmatched uid reads as
        # "the policy retrieved nothing relevant" rather than as an error.
        self.corpus_hash = self.index.corpus_hash()

    # ------------------------------------------------------------------ tasks

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._tasks)

    def task_record(self, tid: TaskId) -> dict:
        """The raw task JSON. GOLD-BEARING (`evaluation_criteria`) — read by the builder, and
        never handed to a policy."""
        return self._tasks[str(tid)]

    @staticmethod
    def template_id_for(tid: TaskId) -> str | None:
        """None on purpose. A MuSiQue id encodes the hops it was composed from, so a
        permutation of one question is recoverable and must be bound to one side of the split
        wall. A retail id is a bare integer with no composition to recover, so there is no
        near-duplicate structure — and inventing a grouping would silently cluster unrelated
        tasks into one CI cluster.
        """
        return None

    def template_id(self, tid: TaskId) -> str | None:
        return self.template_id_for(tid)

    # ------------------------------------------------------------------ the view

    def view(self, tid: TaskId) -> TaskView:
        """REFUSES, exactly as banking does, and for the same two reasons.

        `user_scenario.instructions` is the customer's script — `known_info`, `unknown_info`
        and `reason_for_call`. Handing it to the agent inverts the role and hands over every
        user-private fact for free, so the 219 user_private gold nodes become free and the
        ceiling they define becomes meaningless.

        `allow_user_script=True` restores the old behaviour for debugging and is deliberately
        ugly to type.
        """
        from .suite import Tau2NeedsOrchestrator

        rec = self._tasks[str(tid)]
        ins = (rec.get("user_scenario") or {}).get("instructions") or {}
        script = ins if isinstance(ins, str) else json.dumps(ins, sort_keys=True)
        if not self.allow_user_script:
            raise Tau2NeedsOrchestrator(
                f"tau2_retail/{tid}: the only task text available is the customer's script, "
                "which inverts the agent's role and hands over the user-private partition. "
                "Drive retail through the Orchestrator, or pass "
                "Tau2RetailSuite(allow_user_script=True) for debugging — runs made that way "
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
        from tau2.domains.retail.environment import get_tasks

        want = str(tid)
        for t in get_tasks():
            if str(getattr(t, "id", "")) == want:
                return t
        raise KeyError(f"no retail task {want!r}")

    def env_kwargs(self, tid: TaskId | None = None) -> dict[str, Any]:
        """Retail's environment takes no retrieval variant and no read-log allowlist.

        Banking needs both: `retrieval_variant` selects its document search, and
        `read_log_allowlist` keeps discoverable-tool reads out of the DB reward hash. Retail
        has neither a document index nor discoverable tools, so passing either would be
        inventing configuration the domain does not have.
        """
        return {}

    def environment(self, tid: TaskId | None = None) -> Any:
        """The live retail Environment, MEMOISED PER TASK.

        Per task, not per suite: 176 of 550 required calls mutate the DB, so one shared
        environment would leak a cancelled order or a changed address from one task into the
        next task's starting state. Memoised rather than rebuilt, because rebuilding mid
        episode would reset state the episode depends on.
        """
        ok, why = available()
        if not ok:
            raise Tau2RetailUnavailable(why)
        key = str(tid) if tid is not None else ""
        if key in self._envs:
            return self._envs[key]
        from tau2.domains.retail.environment import get_environment

        env = get_environment()
        self._envs[key] = env
        return env

    def actuator(self, tid: TaskId) -> Any:
        """The Actuator half of `TaskSuite`, which this suite did not have.

        `pi_run.worker.run_unit` calls `suite.actuator(spec.task_id)` UNGUARDED, so its
        absence was an AttributeError waiting on the flat path; retail was saved from it only
        because `view()` raises a few lines earlier, which is an accident of ordering and not
        a contract. `isinstance(suite, TaskSuite)` was False for the same reason, so nothing
        typed or checked could see the gap either.

        THE SAME Environment `tool_schemas` and the retriever were built from — memoised per
        task — because a rollout is one world: a second Environment here would let the policy
        read from world A while the graded mutations landed in world B.
        """
        from .actuator import Tau2Actuator

        return Tau2Actuator(
            self.environment(tid),
            task=self.task_record(tid),
            domain=self.domain,
            env_kwargs=self.env_kwargs(tid),
        )

    def tool_schemas(self, tid: TaskId) -> tuple[dict[str, Any], ...]:
        """The tools a Drafter may plan against: exactly what the environment advertises.

        FLAT `name`/`description`/`parameters`, matching `Tau2Suite.tool_schemas`. The first
        version returned each tool's nested OpenAI form (`{"function": {"name": ...}}`), and
        `tau2_runner` reads `t["name"]` at the top level to build the allowed-tool set -- so a
        live dialogue got as far as the customer's opening message and then died on
        `KeyError: 'name'`. Two shapes for one contract is the failure; there is now one.

        Sorted by name so the advertised list is byte-identical between two runs of one task.
        """
        from .suite import _description_of, _params_of

        env = self.environment(tid)
        return tuple(
            {
                "name": t.name,
                "description": _description_of(t),
                "parameters": _params_of(t),
            }
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
        """A READ-only tool selector, the same channel airline and telecom use.

        WAS `RetailNullRetriever`, which returns () for every query. That was an honest
        description of the domain -- retail has 15 typed key-addressed tools and no free-text
        search -- but its consequence was that an ASK on this suite could not gain evidence at
        all, and the consequence of THAT was measured: 33 exported rows from three runs, every
        one with `value = -0.05`, `coverage_before = 0.0` and `frontier_size = 6`, and
        `n_no_target_dropped = 33`. Gold nodes existed, the policy asked sixteen questions, and
        none of them could reach a record. Retail was a declared training source that could not
        produce a training row for a structural reason.

        `RetailNullRetriever` predates `ToolBackedRetriever`; giving retail the same channel is
        what makes its arms comparable with airline's and telecom's rather than a special case.
        The user channel is still the point of the suite -- a fact the customer holds and the DB
        does not carry is reachable only by asking them -- and it is unaffected: this changes
        what a `target="kb"` ask resolves to, not whether `target="user"` is allowed.

        Carries the SUITE's corpus_hash, not a fresh one: that hash binds a run to the DB its
        gold was built against, and `pi score` refuses a mismatched pairing.
        """
        from .retail_units import uids_for_call
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
        """Every DB record as an evidence unit -- the corpus at the grain a tool call returns.

        `cmd_train._task_units` needs the whole pool to recover the text a uid named, and falls
        back to `retriever._units` when a suite has no accessor. `RetailNullRetriever` holds no
        pool because retail has no search surface, so retail raised `PoolUnavailable` and every
        one of its runs exported ZERO rows -- measured, `not exported: pool_unavailable=3` on a
        three-run sample -- while being declared a training source in the registry.
        """
        from .db_units import evidence_units
        from .retail_units import units_from_db

        return evidence_units(units_from_db(self.db))

    def uids_for_calls(self, calls: Any) -> tuple[str, ...]:
        """Records a turn's recorded EnvCalls read, from their ARGUMENTS only.

        The runner writes `retrieved_uids` from this. Nothing here touches a tool result: a
        retail tool result is a customer record, and `tau2_runner` deliberately stores only a
        `result_digest`.
        """
        from .retail_units import uids_from_env_calls

        assert self.index is not None
        return uids_from_env_calls(self.index, calls)
