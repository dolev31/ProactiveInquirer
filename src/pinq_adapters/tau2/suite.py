"""tau2 / tau-Knowledge, VIEW SIDE.

WHAT THIS FILE MUST NOT DO
    `Task.required_documents` is the bench author's answer key: the ids of the documents
    needed to solve the task. It is GOLD. It is read by pi_eval.build.tau2_build and it is
    never passed to make_view() — make_view would raise anyway, which is the point of
    routing every view through one allowlist.

    Same for `evaluation_criteria` (the reference action sequence), `initial_state` and
    `annotations`. The agent sees the user's own words and nothing else, exactly as a real
    customer-service agent would.

WHY THE VIEW'S `question` IS THE USER PERSONA TEXT
    tau2 is a dialogue benchmark: there is no standalone question string. The user
    simulator is driven by `user_scenario.instructions`, and the agent legitimately sees
    whatever the user chooses to say. Handing the Inquirer the scenario text is the
    faithful analogue of "what the user opened with" — and it is public by construction,
    since the user simulator will say it out loud.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import TaskId, TaskView
from pinq.view import make_view

from ._probe import (
    DOMAIN,
    GOLDEN_RETRIEVAL_VARIANT,
    N_DOCUMENTS,
    N_TASKS,
    RETRIEVAL_VARIANT,
    available,
    load_documents,
    load_task_records,
)
from .actuator import Tau2Actuator
from .retail_retriever import RetailNullRetriever
from .retriever import Tau2Retriever, edges_for_env, pick_kb_tool

CORPUS_ID = "tau2_banking_knowledge"

INSTRUCTIONS = (
    "You are a bank customer-service agent. You may search the internal knowledge base "
    "before acting. Some tools are not listed and must be discovered: a knowledge-base "
    "document will name the exact tool, after which it can be unlocked and called."
)

# tau2 freezes the environment clock so a date-sensitive policy is reproducible.
KNOWLEDGE_FIXED_DATE = "2025-11-14"


class Tau2NeedsOrchestrator(RuntimeError):
    """Raised when tau2 is asked for a flat view it cannot honestly construct."""


def _has_kb_tool(env: Any) -> bool:
    """Does this environment expose a KB search tool at all?

    Asked of `pick_kb_tool`'s own preference list rather than a second literal, so a variant
    that gains a tool cannot be classified differently by the two call sites.
    """
    try:
        pick_kb_tool([t.name for t in env.get_tools()])
    except RuntimeError:
        return False
    return True


class Tau2SuiteUnavailable(RuntimeError):
    """Raised when a live-environment operation is attempted without tau2 installed."""


@dataclass
class Tau2Suite:
    """Satisfies pinq.protocols.TaskSuite over banking_knowledge.

    Documents and tasks are read from disk eagerly (they are small and local); the
    Environment is built lazily per task, because constructing one costs a BM25 index.
    """

    suite_id: ClassVar[str] = "tau2"
    suite_version: ClassVar[str] = "v1.0.1"
    corpus_id: ClassVar[str] = CORPUS_ID
    # THE UPSTREAM DOMAIN THIS SUITE IS GRADED IN. On the suite because there are two tau2
    # suites and one `tau2_runner`: the driver builds the Orchestrator and the grading
    # environment, and the only thing that knows which of the two worlds it is driving is the
    # suite it was handed. Reading banking's module constant there graded every retail
    # rollout against the banking database.
    domain: ClassVar[str] = DOMAIN
    # Exposed so `dialogue_view` can read it off the suite instead of importing this module's
    # module-level constant, which pinned every dialogue -- retail included -- to banking's
    # framing of the job.
    instructions: ClassVar[str] = INSTRUCTIONS

    documents_root: Path | None = None
    tasks_root: Path | None = None
    retrieval_variant: str = RETRIEVAL_VARIANT
    strict_counts: bool = True
    # Deliberately ugly to type. On all 97 banking tasks the only task text available is
    # the customer's roleplay script, which inverts the agent's role and hands over every
    # user-private fact. See view() for why refusing is the correct default.
    allow_user_script: bool = False

    corpus_hash: str = field(default="", init=False)
    _tasks: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _docs: tuple[dict, ...] = field(default=(), init=False, repr=False)
    # ONE Environment per task id, shared by the retriever, the actuator and tool_schemas.
    # Not a speed hack (though a BM25 index costs seconds): a rollout is ONE world. Building
    # a second Environment for the actuator would let the policy read from world A and mutate
    # world B, and the DB hash the reward is computed over would then be a hash of a world
    # nobody ever read from. A Tau2Suite is constructed per unit in pi_run.worker.load_suite,
    # so the cache never outlives a rollout.
    _envs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.documents_root is None or self.tasks_root is None:
            ok, why = available()
            if not ok:
                raise Tau2SuiteUnavailable(why)
        self._docs = load_documents(self.documents_root)
        records = load_task_records(self.tasks_root)
        self._tasks = {r["id"]: r for r in records}

        if self.strict_counts:
            # Assert, never assume: a silent upstream data change would move every
            # denominator in the paper without touching a line of code.
            if len(self._docs) != N_DOCUMENTS:
                raise ValueError(f"expected {N_DOCUMENTS} documents, found {len(self._docs)}")
            if len(self._tasks) != N_TASKS:
                raise ValueError(f"expected {N_TASKS} tasks, found {len(self._tasks)}")

        self.corpus_hash = _corpus_hash(
            (d["id"], d["title"], hashlib.sha256(d["content"].encode()).hexdigest())
            for d in self._docs
        )

    # ------------------------------------------------------------------ TaskSuite protocol

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(sorted(self._tasks))

    def view(self, tid: TaskId) -> TaskView:
        """The task, as the AGENT is allowed to see it.

        `user_scenario.instructions` is NOT that. On all 97 banking tasks it is a customer
        ROLEPLAY SCRIPT -- "You are playing the role of a customer... Your character is Sera
        Chen, a high ranking official at the EPA..." -- written for the user simulator. Handing
        it to the agent does two separate kinds of damage:

          * it inverts the role. Observed live: the agent answered as Sera Chen and never
            attempted the banking task at all.
          * it hands over every user-private fact for free, so the discoverable-from-KB versus
            user-private partition collapses and the ADR ceiling -- the number that bounds what
            ANY autonomous inquirer could reach -- becomes meaningless.

        `description.purpose` is not a substitute: it is literally "Task: task_002" on all 97.

        In tau2's own design the task arrives THROUGH THE USER SIMULATOR, turn by turn, via the
        Orchestrator; `pinq_adapters.tau2.agent` exists for exactly that. So the flat
        `pi run --suite tau2` path cannot construct an honest view, and refusing is the only
        correct answer -- a loud failure beats a wrong number that no one can see is wrong.

        `allow_user_script=True` restores the old behaviour for debugging and is deliberately
        ugly to type; it stamps the view so any run made that way is excluded from reported
        tables, exactly as gold-exposed runs are.
        """
        rec = self._tasks[tid]
        scenario = rec.get("user_scenario") or {}
        script = (scenario.get("instructions") or "").strip()
        if not self.allow_user_script:
            raise Tau2NeedsOrchestrator(
                f"tau2/{tid}: the only task text available is the customer's roleplay script, "
                "which inverts the agent's role and destroys the user-private partition. "
                "Drive tau2 through the Orchestrator (pinq_adapters.tau2.agent), or pass "
                "Tau2Suite(allow_user_script=True) for debugging -- runs made that way are "
                "excluded from reported tables."
            )
        return make_view(
            task_id=tid,
            suite_id=self.suite_id,
            question=script,
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=180,
        )

    def retriever(self, tid: TaskId) -> Tau2Retriever | RetailNullRetriever:
        """None when this retrieval variant exposes no KB tool.

        `golden_retrieval` puts the required documents in the environment's own prompt, so
        there is nothing to search and `pick_kb_tool` -- which raises rather than guessing --
        would refuse to construct. For bm25 a missing KB tool IS a misconfiguration and the
        raise is correct; in golden mode its absence is the condition being studied.

        A NULL RETRIEVER, NOT None. The inquirer arms call `retriever.search(...)`
        unconditionally; returning None killed both of them in the first golden pilot with
        `AttributeError: 'NoneType' object has no attribute 'search'`. Returning () keeps the
        empty channel visible in `n_retrieved` rather than silently plausible, which is the
        same reason `RetailNullRetriever` exists. The agent is not blinded either way: the
        documents arrive through the Orchestrator's system prompt.
        """
        env = self.environment(tid)
        if not _has_kb_tool(env):
            return RetailNullRetriever(corpus_id=self.corpus_id, corpus_hash=self.corpus_hash)
        return Tau2Retriever(
            env,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            tool_edges=self.tool_edges(env),
        )

    def actuator(self, tid: TaskId) -> Tau2Actuator:
        return Tau2Actuator(
            self.environment(tid),
            task=self._tasks[tid],
            domain=DOMAIN,
            env_kwargs=self.env_kwargs(tid),
        )

    # ------------------------------------------------------------------ tau2 specifics

    @property
    def documents(self) -> tuple[dict, ...]:
        return self._docs

    def task_record(self, tid: TaskId) -> dict:
        """The raw task JSON. GOLD-BEARING (`required_documents`, `evaluation_criteria`) and
        therefore read only by the gold builder and by the Actuator, never by `view()`."""
        return self._tasks[tid]

    def tau2_task_object(self, tid: TaskId) -> Any:
        """`self._tasks[tid]` as an upstream `Task`, parsed from the PER-TASK file.

        Deliberately not `tau2_task()`, which goes through `get_tasks()` and therefore reads
        the aggregate `tasks.json` that upstream ships stale for 13 of the 97 tasks (see
        `_probe.load_task_records`). Anything that grades a rollout must read the same bytes
        the graph was built from, or 13 tasks are scored against criteria the gold graph
        never saw.
        """
        ok, why = available()
        if not ok:
            raise Tau2SuiteUnavailable(why)
        from tau2.data_model.tasks import Task

        return Task.model_validate(self._tasks[tid])

    def env_kwargs(self, tid: TaskId | None = None) -> dict[str, Any]:
        """Constructor kwargs for this task's Environment.

        `read_log_allowlist` IS NOT OPTIONAL and is not a tuning knob. tau2's
        `call_discoverable_agent_tool` logs every call into the `agent_discoverable_tools`
        table, and that table is hashed into the DB reward. Without the allowlist a single
        extra READ — which the knowledge base actively tells the agent to perform — moves the
        hash and zeroes the reward; with a different allowlist than the grader's, the
        predicted and gold environments are not comparable at all. Upstream derives it in
        `tau2.runner.build._derive_read_log_allowlist`; we call that function rather than
        reimplementing it so the two cannot drift.

        `task=` IS PASSED ONLY FOR `golden_retrieval`, and never for the pinned `bm25`.

        The original rule here was to omit `task=` unconditionally, because it feeds
        `golden_retrieval`, which inlines the task's own required documents into the agent's
        context -- i.e. hands the policy the answer key -- and omitting it meant "a variant
        change can never SILENTLY turn the leak on". That reasoning is kept; the operative word
        is *silently*. `tau2_golden` exists to run that condition deliberately, and it is
        declared in `pi_run.worker.ORACLE_RETRIEVAL_SUITES`, which forces `gold_exposed=True`
        on every such run. So the leak is on, marked, and excluded from `report.ELIGIBLE`, from
        `assert_no_gold_exposed` and from the training export.

        Gating on the variant rather than passing it always preserves the original guarantee
        for bm25: a future variant change cannot turn the exposure on without also changing
        this condition and the suite id that declares it.

        MEASURED, and the reason this is not merely tidy: with `task=None` the golden template
        renders the literal "(No documents provided)". The first golden pilot ran that way and
        scored 0/28 against bm25's 6% -- banking with no search tool AND no documents, which
        looks exactly like "golden does not help".
        """
        kwargs: dict[str, Any] = {"retrieval_variant": self.retrieval_variant}
        if tid is None:
            return kwargs
        from tau2.runner.build import _derive_read_log_allowlist

        task = self.tau2_task_object(tid)
        kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
        if self.retrieval_variant == GOLDEN_RETRIEVAL_VARIANT:
            kwargs["task"] = task
        return kwargs

    def environment(self, tid: TaskId | None = None) -> Any:
        """The live banking_knowledge Environment for a task, pinned to the offline bm25
        variant and memoized per task id. See `_envs` for why memoization is load-bearing."""
        ok, why = available()
        if not ok:
            raise Tau2SuiteUnavailable(why)
        if tid is not None and tid in self._envs:
            return self._envs[tid]
        from tau2.runner import build_environment

        env = build_environment(DOMAIN, env_kwargs=self.env_kwargs(tid))
        if tid is not None:
            self._envs[tid] = env
        return env

    def tool_schemas(self, tid: TaskId) -> tuple[dict[str, Any], ...]:
        """The tool schemas a Drafter may plan against: EXACTLY the ADVERTISED tools.

        The 44 suffixed discoverable tools are excluded ON PURPOSE, and this is the single
        most important line in this file. Their four-digit suffix is unguessable, so
        `unlock_discoverable_agent_tool(name)` only succeeds for a policy that READ the naming
        document — and that prerequisite IS the mechanic this suite exists to measure. Handing
        the drafter `open_bank_account_4821` up front would satisfy the endpoint with no
        discovery at all and turn the headline result into an artifact of the harness.

        `env.get_tools()` is the honest set (15 tools). `unlock_discoverable_agent_tool` and
        `call_discoverable_agent_tool` are both in it, which is exactly how a policy that DID
        read gets from a document to an action.
        """
        env = self.environment(tid)
        return tuple(
            {
                "name": t.name,
                "description": _description_of(t),
                "parameters": _params_of(t),
            }
            for t in sorted(env.get_tools(), key=lambda t: t.name)
        )

    def template_id(self, tid: TaskId) -> str:
        """A DERIVED template id, because upstream ships none.

        THE PROBLEM. The tau2 primary endpoint is clustered at `template_id` (preregistration
        stage 1) because the 97 tasks are not 97 independent draws: they are instantiations of
        a much smaller set of banking scenarios, and resampling tasks rather than templates
        understates the SE. Upstream's only per-task label is `description.purpose`, which is
        the literal string "Task: task_017" — 97 distinct values, i.e. one cluster per task,
        i.e. no clustering at all.

        THE DERIVATION, in full, so it can be checked rather than trusted:

            template_id = "doc:" + <the sorted, deduplicated TOPIC STEMS of the task's
                          required_documents>, joined by "+"

        where the topic stem of `doc_credit_cards_gold_rewards_card_001` is `credit_cards`:
        the document id with its `doc_` prefix and its trailing `_NNN` serial removed, then
        truncated after the second underscore-delimited component. A task requiring no
        document falls back to "solo:<task_id>", its own cluster — the honest answer, since
        there is nothing to share.

        WHY THIS FIELD. What makes two tau2 tasks correlated is that they are answered out of
        the same corner of the knowledge base: same policy documents, same tool family, same
        failure modes. `required_documents` is exactly that corner, and it is the benchmark
        author's own assertion rather than our inference. Measured on v1.0.1 it yields 18
        clusters over 97 tasks (largest 27, median 3.5), which is the "small number of
        templates" the preregistration assumed.

        WHY IT IS NOT A LEAK. It is derived from a gold-bearing field, so it is minted here
        and travels in the RunManifest — which no agent-side object ever sees. It must never
        be routed through `view()`; `make_view` would raise, which is the point.
        """
        refs = tuple(self._tasks[tid].get("required_documents") or ())
        stems = sorted({s for s in (_topic_stem(r) for r in refs) if s})
        return ("doc:" + "+".join(stems)) if stems else f"solo:{tid}"

    def tool_edges(self, env: Any) -> tuple:
        """doc -> tool prerequisite edges, cross-referenced against this env's real tools."""
        return edges_for_env(self._docs, env)

    def tau2_task(self, tid: TaskId) -> Any:
        """The upstream Task object, for the Orchestrator. GOLD-BEARING — never viewed."""
        ok, why = available()
        if not ok:
            raise Tau2SuiteUnavailable(why)
        from tau2.runner import get_tasks

        for t in get_tasks(DOMAIN):
            if t.id == tid:
                return t
        raise KeyError(tid)


# ------------------------------------------------------------------------- pure helpers
#
# Module-level and dependency-free so they are unit-testable with tau2 ABSENT. An earlier
# draft of this file used both of them from methods without ever defining them; they are
# defined here, next to the tests that pin them.

# A trailing serial: `doc_credit_cards_gold_rewards_card_001` -> the `_001`. Anchored at the
# end so an id whose TOPIC contains digits keeps them.
_SERIAL = re.compile(r"_\d+$")

# How many underscore-delimited components of a document id make a topic. Two, measured:
# one ("credit") merges credit cards with credit lines; three splits `credit_cards_gold`
# from `credit_cards_silver` and gives back one cluster per task.
_STEM_PARTS = 2


def _topic_stem(doc_ref: str) -> str:
    """`doc_credit_cards_gold_rewards_card_001` -> `credit_cards`. See Tau2Suite.template_id."""
    s = doc_ref[4:] if doc_ref.startswith("doc_") else doc_ref
    s = _SERIAL.sub("", s)
    return "_".join(p for p in s.split("_")[:_STEM_PARTS] if p)


def _description_of(tool: Any) -> str:
    """A tool's human-readable description, whatever upstream calls the field this release.

    `Tool` has no `description` attribute — it is a pydantic model with `short_desc` /
    `long_desc` and a computed `openai_schema`. Reading `.description` raises AttributeError
    rather than returning None, so this asks the schema first and falls back to the two
    fields, and never guesses.
    """
    schema = getattr(tool, "openai_schema", None) or {}
    desc = (schema.get("function") or {}).get("description")
    if desc:
        return str(desc).strip()
    parts = [getattr(tool, "short_desc", "") or "", getattr(tool, "long_desc", "") or ""]
    return "\n".join(p for p in parts if p).strip()


def _params_of(tool: Any) -> dict[str, Any]:
    """The JSON-Schema parameter object upstream itself advertises to a model.

    Taken from `Tool.openai_schema` rather than rebuilt from `Tool.params`, because the schema
    is what a real tau2 agent is shown: rebuilding it would let our drafter be prompted with a
    different contract than upstream's baseline agent and make the comparison unfair in a way
    no test would see.
    """
    schema = getattr(tool, "openai_schema", None) or {}
    params = (schema.get("function") or {}).get("parameters")
    return dict(params) if isinstance(params, dict) else {}
