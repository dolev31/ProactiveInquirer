"""Drive one tau2 unit through the upstream Orchestrator, and grade it with upstream's grader.

`pinq_adapters.tau2` has had a suite, a retriever, an actuator and an agent class for a while.
What it has never had is the thing that CONSTRUCTS an Orchestrator, runs it, harvests the
result and writes the seven files `pi compact` reads -- so tau2 was reachable only through the
flat `pi run` path, which correctly refuses (`Tau2NeedsOrchestrator`) because the only task
text upstream ships is the customer's roleplay script. This module is that missing driver, and
it is what unblocks P1, S2, S11 and all of L2.

THREE DECISIONS, EACH OF WHICH IS THE DIFFERENCE BETWEEN A NUMBER AND A WRONG NUMBER.

1. ACTIONS TRAVEL AS MESSAGES, NEVER STRAIGHT INTO THE LIVE ENVIRONMENT.
   `EnvironmentEvaluator.calculate_reward` builds a FRESH environment and replays
   `message_history=full_trajectory` into it, then compares its DB hash against a gold
   environment's. A tool call executed against the live env but absent from the transcript is
   therefore invisible to the grader: the predicted DB would be pristine, and `tau_reward`
   would be 0 for every arm no matter what the policy actually did -- P1 reading as a flat null
   rather than as an unwired path. So the Drafter's tool plan is emitted as
   `AssistantMessage.tool_calls` and the Orchestrator executes it. `Tau2Actuator` is
   deliberately NOT passed to `run_loop` here; it remains the flat path's executor and this
   module's gold-replay oracle.

2. THE VIEW IS BUILT FROM WHAT THE USER HAS ACTUALLY SAID.
   Not `user_scenario.instructions` -- that is a roleplay script written for the simulator
   ("You are playing the role of a customer... Sera Chen... EPA"), and handing it to the agent
   both inverts the role (observed live: the agent answered as Sera Chen on a banking task) and
   gives away every user-private fact, which collapses the discoverable-from-KB vs user-private
   partition that the ADR ceiling is defined over. The user's own utterances are public by
   construction: the simulator says them out loud, into a transcript we store.

3. ONE ASK LEDGER PER UNIT, NOT PER ASSISTANT TURN -- AND ONE TOOL BUDGET PER UNIT BESIDE IT.
   `budget_cap` is 16 for a TASK. A policy handed a fresh 16 on every user turn would spend
   N x 16 against arms that spend 16, and the budget-parity precondition -- on which every
   comparison in the paper rests -- would be false in the treatment's favour. Both budgets are
   the unit's, each of `budget_cap`: the questioner's asks on the `BudgetLedger` (QA's semantics
   exactly, `run_loop`'s budget stop unchanged), the agent's tool calls on `BudgetGate`'s own
   counter (RULES amendment 7). The two never share a counter, so asking can never starve the
   agent of tool calls and acting can never stop the questioner.

4. THE TOOL BUDGET IS ENFORCED WHERE THE TOOL CALL EXECUTES, WHILE THE DIALOGUE IS RUNNING.
   The Orchestrator executed every tool call the moment it was emitted and `meter_env_calls`
   only charged them after the dialogue ended, so the cap was arithmetic done on a finished run:
   122 of 408 recorded fork runs spent past it, and every tau2 claim was withdrawn. See
   `BudgetGate`: once the unit's tool budget is spent, a further call is not executed, not
   counted, and answered with a tool result that says `retrieval budget exhausted`. GENERIC tools
   (`calculate`, `transfer_to_human_agents`) read no record and write no row, so they are not
   retrieval: never gated, never counted, and the agent can always still hand off. (From
   ba48ab0 to e5e8206 the gate charged tool calls to the ASK ledger -- one cap shared by both;
   `budget_enforcement` names which regime a run was made under.)

WHAT IS OPEN-LOOP, STATED PLAINLY. Within one assistant turn the tool plan is produced once,
by the Drafter, and then emitted step by step without re-planning on tool results. That is a
property of the architecture (the Drafter plans, the Inquirer asks) rather than a bug, but it
means a failed unlock does not stop the call that depended on it.

AND NO TOOL RESULT REACHES OUR POLICY AT ALL -- this paragraph used to say the opposite. Tool
messages land in `DriverState.messages`, and the only reader of that list is `dialogue_view`,
which keeps role == "user" and nothing else. So a result, executed or refused, is delivered to
the agent object and read by no Inquirer, Drafter or Answerer; the pinq policy's evidence comes
from its retriever, and the budget reaches it through `run_loop`'s own stop (`stop_reason ==
"budget"`) on the next rollout. Upstream's `LLMAgent` (the stock arm) does read tool messages,
so for it the refusal text IS the signal. `tests/test_tau2_budget_gate.py` pins both halves.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pinq import promptlib
from pinq.budget import HARD_CURRENCY, BudgetExceeded, BudgetLedger
from pinq.ids import canon, h
from pinq.types import EnvCall, Outcome, TaskView, Trajectory, Usage
from pinq.view import make_view

# tau2's max_steps counts ORCHESTRATOR steps (agent message, tool call, tool result, user
# message), not internal Inquirer turns -- the whole point of collapsing a rollout into one
# assistant message is that the Inquirer spends none of this. Upstream's default; identical
# for every arm, because a step budget that moved with the arm would confound the comparison
# with the harness.
DEFAULT_MAX_STEPS = 100

# UPSTREAM'S DEFAULT IS 10, AND IT COUNTS THE CUSTOMER'S MISTAKES AGAINST THE AGENT.
#
# `Orchestrator._execute_tool_calls` does `num_errors += 1` for any failed tool call, whatever
# the requestor, and a dialogue that trips the cap terminates as TOO_MANY_ERRORS. Measured on
# the first three live tau2 dialogues (drafter_only, gpt-oss-120b as both agent and user
# simulator):
#
#     task      msgs   agent errors   user errors   headroom
#     task_001    46              0             7         +3
#     task_003    42              0             8         +2
#     task_002    78              0             0        +10
#
# The AGENT made none. The user simulator repeatedly tried to call the agent's knowledge-base
# tools itself -- `search`, `get_document`, `knowledge_base_search` -- none of which are user
# tools, so each attempt failed and spent the agent's error budget. Two of three dialogues came
# within 2-3 errors of being terminated for the customer's mistakes.
#
# That is a confound, not an inconvenience, and it has a direction: a premature termination
# scores tau_reward = 0.0 (correctly -- the DB never reached the gold state), so a weak user
# simulator would be recorded as a policy failure. It is also arm-dependent, because arms that
# sustain longer dialogues give the customer more turns in which to flail.
#
# So the cap is raised to bound what it is FOR -- an agent making bad calls -- rather than the
# environment's noise, and `n_errors_agent` / `n_errors_user` are recorded on every run so the
# assumption stays auditable instead of buried in a constant.
DEFAULT_MAX_ERRORS = 30

# The suffix `pinq_` on tool-call ids is load-bearing only in that it must be STABLE: the id
# appears in the transcript the grader replays, and a uuid would make two identical replays
# diff.
_CALL_ID = "pinq_{:04d}"

# Emitted when a rollout produced no answer text. Deliberately not an empty string (which
# tau2 refuses) and deliberately not something a grader could mistake for a response.
NO_ANSWER = "(no answer produced)"

# THE BUDGET REGIME, recorded on every tau2 unit's status. THREE VALUES HAVE BEEN WRITTEN AND NO
# TWO MAY BE POOLED:
#   (absent)                 before ba48ab0: the cap was checked only after the dialogue;
#   "refuse_and_tell"        ba48ab0..e5e8206: tool calls refused in-dialogue, charged to the
#                            ASK ledger -- one cap shared by asks and tool calls;
#   "refuse_and_tell_split"  RULES amendment 7: asks and tool calls get `budget_cap` EACH, the
#                            tool budget on `BudgetGate`'s own counter.
# The current value is deliberately not a value any earlier run carries, so a reader matching it
# exactly cannot pick up a shared-cap run. A unit whose suite does not budget tool calls at all --
# banking, whose evidence arrives through a retriever -- records `TOOLS_UNCHARGED`.
BUDGET_ENFORCEMENT = "refuse_and_tell_split"
TOOLS_UNCHARGED = "tools_uncharged"

# The hand-off, counted on the status (`n_transfer_to_human`): the one tool the refusal text
# names, and the one a starved agent should still be able to reach.
HANDOFF_TOOL = "transfer_to_human_agents"

# THE EVIDENCE WINDOW OF THE CLAUDE ROLES ON tau2: how much of one unit's text the Drafter's
# resolve and draft prompts and the Answerer's prompt carry. `pinq_expt.components.UNIT_CHARS`
# (1200) is sized for QA paragraphs and cut EVERY airline flight record (median 4418
# characters, 100% over 1200), removing the later dates and their `available` status -- on
# HAT110 the Drafter saw 2024-05-01..05-10 and never the word "available", so no flight could
# be booked -- and 76% of retail products (median 1752) and 12% of orders.
#
# IMPORTED, NOT DEFINED HERE: it is ONE constant with the tool retriever's unit cap, which is
# where its value and the measurements behind it live (records max 4527, search results max
# 23,936 -> 32000). Two constants could disagree, and did: this window went to 6000 while the
# retriever still stored 4000 characters of every search result.
#
# THE QUESTIONER IS NOT WIDENED. `PromptedInquirer._prompt` renders evidence through the same
# `render_evidence`, but no Inquirer constructor accepts a window, so its view stays at 1200 on
# tau2 as on QA: it is the view the trained questioner learned on (`pi_run.cmd_train`), and the
# Qwen questioner serves in a 16384-token window.
from pinq_adapters.tau2.tool_retriever import TAU2_EVIDENCE_CHARS  # noqa: E402

# THE ONLY BUDGET SIGNAL THE AGENT EVER RECEIVES, and it arrives only at exhaustion. It names no
# number: a policy told its remaining budget would be a budget-aware policy, which is exactly
# what `Inquirer.act(s: State)` is forbidden to be. The phrase `retrieval budget exhausted` is
# load-bearing -- `split_refused` identifies a refusal by these exact bytes together with the
# tool-call id the gate recorded, so this constant is the one place the text may change.
#
# EVERY SENTENCE IN IT MUST BE TRUE, because an agent acts on it. GENERIC tools are exempt from
# the budget (see `budget_exempt_tools`), so "no further tool calls" would be false: the text
# says lookups and changes are over, and names the hand-off -- which is GENERIC in every gated
# domain of the committed map, and `tests/test_tau2_budget_gate.py` fails if it ever names a
# tool that is gated.
#
# WHO READS IT: upstream's `LLMAgent` (the stock arm) and nothing of ours. `dialogue_view` keeps
# user turns only, so on our arms the text reaches `DriverState.messages` and no Inquirer,
# Drafter or Answerer; those meet the budget as `run_loop`'s `budget` stop instead.
BUDGET_REFUSAL = (
    "retrieval budget exhausted: this tool call was not executed and nothing was changed. "
    "No further lookups or changes will be executed in this conversation. Reply to the customer "
    "with what you already have, or hand the conversation to a human agent with "
    "transfer_to_human_agents."
)

# THE TOOL-TYPE MAP THE EXEMPTION IS READ FROM: tau2-bench's own `@is_tool(ToolType.*)`
# declarations, committed. Resolved against THIS checkout, not the working directory, so the map
# a run used is the one at the commit its `code_version` names.
TOOL_TYPES_PATH = Path(__file__).resolve().parents[3] / "conf" / "tau2" / "tool_types.json"
EXEMPT_TOOL_TYPE = "GENERIC"


class Tau2DriverError(RuntimeError):
    """The tau2 driver could not be constructed. One unit's problem, never the sweep's."""


# ------------------------------------------------------ two prompt variants this driver alone uses

# WHY A SEPARATE TEMPLATE AND NOT A HOLE IN THE SHIPPED ONE. `inquirer_prompted.txt` is
# rendered by every QA suite, and its bytes are inside `prompt_hashes`, which is inside
# `semantic_hash`. Adding a `{{stop_rule}}` placeholder to it would move the run_id of every
# QA run ever made and of every one still to come, to change a paragraph only tau2 reads. A
# second FILE leaves the QA prompt byte-identical, which tests/test_tau2_stop_variant.py
# asserts against the committed bytes.
#
# WHY THERE ARE TWO, AND WHY THEY LAYER. Commit 3052546 ("teach the Inquirer that a question
# reaches records, never the customer") edited the shared template on branch `convlog-work`
# and was never merged, so this line of history renders the pre-3052546 bytes. MEASURED over
# every tau2 fork run on disk, holding arm and Inquirer model fixed (`inquirer_prompted`,
# claude-sonnet-5): the answered-ask rate is 0.873 retail / 0.908 airline under 3006f5bf and
# 0.443 / 0.485 under 814658ea -- the policy asks the CUSTOMER for facts no read-only tool can
# return, so the retrieval channel is effectively dead. The instruction is meaningless on the
# QA suites, which have no customer, and porting it into the shared template would move every
# held-out QA number in the paper. So the layering is
#
#     inquirer_prompted        the QA prompt, 814658ea, UNTOUCHED
#       + the 3052546 hunk  -> TAU2_BASE_TEMPLATE   routing only
#       + the stopping hunk -> TAU2_STOP_TEMPLATE   routing + stopping
#
# and the stop-vs-base contrast is still exactly the stopping paragraph.
#
# WHY THEY ARE OPT-IN. tau2's recorded fork campaigns rendered the shipped stopping paragraph. A
# variant that applied itself whenever the suite is tau2 would silently make every later run
# incomparable with them while sharing their arm_id, so it is selected by `UnitSpec.prompt_
# variant` and by nothing else -- an unset field renders exactly what the campaigns rendered.
#
# WHY THE LABEL CARRIES A DIGEST. `prompt_variant_id` is in `pinq.ids.SEMANTIC_FIELDS`, so the
# label separates variant runs from shipped-prompt ones in run identity and in the parquet
# column, and the digest means editing a variant's text cannot reuse the earlier label. That is
# load-bearing here: TAU2_STOP_TEMPLATE gained the routing paragraph in this same change, so
# `tau2_stop-0cde17cd` (the 15 runs of the killed campaign) and the label it carries now are
# two different prompts, and nothing can pool them.
#
# ALL FOUR NAMES ARE MODULE-LEVEL CONSTANTS, AND THAT IS LOAD-BEARING. tests/test_prompt_
# templates_exist.py walks the AST for `promptlib.<fn>(<name>)` and resolves a module-level
# `NAME = "literal"`; a name read out of a dict at the call site is reported as DYNAMIC and the
# template it names falls outside the gate that exists because `retriever_select.txt` went
# missing for four revisions with nothing noticing. A further variant belongs here as further
# constants, not as a lookup table.
TAU2_BASE_VARIANT = "tau2_base"
TAU2_BASE_TEMPLATE = "inquirer_prompted_tau2_base"
TAU2_STOP_VARIANT = "tau2_stop"
TAU2_STOP_TEMPLATE = "inquirer_prompted_tau2"

# variant name -> the template it points the Inquirer at. NOT the source of the names above:
# those stay module-level literals so the AST gate can resolve them, and this maps between two
# of them. `promptlib.<fn>()` is never called with a value read out of this dict.
_VARIANTS: dict[str, str] = {
    TAU2_BASE_VARIANT: TAU2_BASE_TEMPLATE,
    TAU2_STOP_VARIANT: TAU2_STOP_TEMPLATE,
}


def apply_prompt_variant(inquirer: Any, variant: str) -> str:
    """Point `inquirer` at the variant's template; return the label for the manifest.

    With no variant asked for this defers to `promptlib.variant_id()`, exactly as
    `worker.run_unit` does: "v1" on an un-overlaid tree, which is what every tau2 run on disk
    carries, and the overlay's own label when rung 0 is rendering a candidate.

    REFUSES RATHER THAN NO-OPS on an arm that renders a different template: silently leaving
    the shipped prompt in place would produce a run whose manifest names a variant and whose
    policy read the paragraph that variant exists to replace.
    """
    if not variant:
        return promptlib.variant_id()
    if variant not in _VARIANTS:
        raise Tau2DriverError(
            f"unknown prompt variant {variant!r}; known: "
            f"{TAU2_BASE_VARIANT!r}, {TAU2_STOP_VARIANT!r}"
        )
    base = str(getattr(inquirer, "prompt_name", ""))
    if base not in ("inquirer_prompted", TAU2_BASE_TEMPLATE, TAU2_STOP_TEMPLATE):
        raise Tau2DriverError(
            f"prompt variant {variant!r} rewrites blocks of inquirer_prompted, but this arm's "
            f"Inquirer renders {base or '(no template)'!r}"
        )
    template = _VARIANTS[variant]
    inquirer.prompt_name = template
    # The digest is of the template's OWN bytes, read through promptlib so an overlay is
    # reflected: two files, two digests, and an edit to either cannot reuse its old label.
    digest = (
        promptlib.sha(TAU2_BASE_TEMPLATE)[:8]
        if template == TAU2_BASE_TEMPLATE
        else promptlib.sha(TAU2_STOP_TEMPLATE)[:8]
    )
    return f"{variant}-{digest}"


# --------------------------------------------------------------------------- the view


def dialogue_view(suite: Any, tid: str, messages: Sequence[Any]) -> TaskView:
    """The task as the AGENT is allowed to see it: the user's own words, in order.

    Every field passed here is public. `make_view` refuses an unknown key, so a gold-bearing
    field cannot arrive by accident; that this function reads `messages` and never
    `suite.task_record(tid)` is the property that matters.

    An empty dialogue yields the opening prompt rather than an empty question, because a
    TaskView with no question is indistinguishable from a broken adapter three stages later.
    """
    said = [
        str(m.content).strip()
        for m in messages
        if getattr(m, "role", "") == "user" and getattr(m, "content", None)
    ]
    question = "\n".join(said) if said else "(the customer has not spoken yet)"

    return make_view(
        task_id=tid,
        # OFF THE SUITE, not a literal and not banking's module constant. A retail dialogue
        # stamped suite_id="tau2" lands in the eval-only namespace, where `assert_trainable`
        # refuses it -- so the rows vanish from the export with no error anyone can see.
        suite_id=suite.suite_id,
        question=question,
        instructions=suite.instructions,
        corpus_id=suite.corpus_id,
        corpus_hash=suite.corpus_hash,
        word_cap=180,
    )


# --------------------------------------------------------------------------- gold replay


# THE ONE `error` VALUE THAT MEANS "nothing to grade" RATHER THAN "could not grade it".
# Spelled once and compared against, because `pi verify tau2` has to tell the two apart and a
# repeated string literal is how they became one bucket in the first place.
NO_ACTIONS = "no evaluation actions"


@dataclass(frozen=True, slots=True)
class GoldReplay:
    task_id: str
    db_reward: float | None
    reward_basis: tuple[str, ...]
    n_actions: int
    n_failed_calls: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.db_reward == 1.0


def replay_gold(suite: Any, task_ids: Sequence[str] | None = None) -> list[GoldReplay]:
    """Execute each task's OWN `evaluation_criteria.actions` and check the gold hash comes back.

    Zero model calls, and it is the only check that can tell a broken harness from a bad
    policy. If replaying the answer key does not reproduce the answer, then `env_kwargs`, the
    `read_log_allowlist`, `initial_state` or the synthesized trajectory is wrong, and every
    `tau_reward` this repository has ever produced -- none of which had been validated against
    upstream even once -- is measuring our plumbing rather than the policy.

    A task whose criteria carry no actions returns `db_reward=None`, not 0.0: absent and
    failing must not look alike.
    """
    from pinq_adapters.tau2.actuator import Tau2Actuator

    out: list[GoldReplay] = []
    sink = io.StringIO()  # tau2 tools print their user-facing text to stdout
    for tid in task_ids if task_ids is not None else suite.task_ids():
        with contextlib.redirect_stdout(sink):
            try:
                task = suite.tau2_task_object(tid)
                crit = getattr(task, "evaluation_criteria", None)
                basis = tuple(
                    str(getattr(b, "value", b)) for b in (getattr(crit, "reward_basis", ()) or ())
                )
                actions = list(getattr(crit, "actions", None) or []) if crit else []
                if not actions:
                    out.append(GoldReplay(tid, None, basis, 0, 0, NO_ACTIONS))
                    continue
                act = Tau2Actuator(
                    suite.environment(tid),
                    task=suite.task_record(tid),
                    # OFF THE SUITE. The actuator's default is banking's domain, so replaying
                    # retail's answer key rebuilt BANKING to grade against -- which is the one
                    # thing a check that exists to separate a broken harness from a bad policy
                    # must not do.
                    domain=suite.domain,
                    env_kwargs=suite.env_kwargs(tid),
                )
                calls = act.execute(
                    [
                        {"name": a.name, "args": dict(a.arguments), "requestor": a.requestor}
                        for a in actions
                    ],
                    turn_idx=0,
                )
                native = act.native()
                out.append(
                    GoldReplay(
                        task_id=tid,
                        db_reward=native.get("db_reward"),
                        reward_basis=basis,
                        n_actions=len(actions),
                        n_failed_calls=sum(1 for c in calls if not c.ok),
                        error=act.reward_error,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one task's failure is a reported row
                out.append(GoldReplay(tid, None, (), 0, 0, f"{type(exc).__name__}: {exc}"))
    return out


# --------------------------------------------------------------------------- the agent


@dataclass
class DriverState:
    """Agent-side state for one simulation.

    `trajectories` accumulates one pinq rollout per USER message. It is never rendered into a
    tau2 message: the internal Inquirer/Drafter dialogue must not reach the transcript, or the
    user simulator would answer the internal questions and the Inquirer would be eliciting from
    the user -- the one thing the paper claims it does not need to do.
    """

    messages: list = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)  # tool plan steps not yet emitted
    answer: str = ""
    n_emitted: int = 0
    # Plan steps naming a tool the agent was never offered. A count, not a silence.
    rejected: int = 0


def rollout_branch(
    *, base: int, branch_at: int | None, branch_seed: int | None
) -> tuple[int | None, int | None]:
    """Map a GLOBAL fork turn onto THIS rollout's `run_loop` index. `(None, None)` when no fork.

    `candidate_specs` validates `branch_turn_idx` against the parent's `turns.jsonl`, whose
    turn_idx is renumbered sequentially across rollouts; `run_loop`'s `i` restarts at 0 for every
    user message. `base` is the turns already recorded before this rollout -- the same sum the
    driver sets as `ledger.turn_base`.

    A fork at or after `base` is `branch_at - base`; if this rollout ends before reaching it the
    loop never fires, and the next rollout recomputes with a larger base. A fork BEFORE `base` is
    local index 0: `build_parts()` rebuilds the Inquirer with the run seed every rollout, so a
    post-fork rollout must re-seed it or the candidate silently rejoins the parent.

    MEASURED before this existed: every candidate of every state sent the identical request
    (`request_sha dcc524774efe` for all four branch seeds), because the driver never passed the
    fork to `run_loop` at all -- the fourth field on this path that `run_unit` honours and the
    tau2 copy dropped.
    """
    if branch_at is None or branch_seed is None:
        return None, None
    local = int(branch_at) - int(base) if int(branch_at) >= int(base) else 0
    return local, int(branch_seed)


def make_driver_agent_class() -> type:
    """Build the HalfDuplexAgent subclass. Imports tau2; call only when it is installed."""
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, ToolCall

    class PinqDriverAgent(HalfDuplexAgent):
        def __init__(
            self,
            tools: list,
            domain_policy: str,
            *,
            suite: Any,
            task_id: str,
            build_parts,
            max_turns: int = 16,
            k: int = 5,
            seed: int = 0,
            branch_at: int | None = None,
            branch_seed: int | None = None,
        ) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)
            self._suite = suite
            self._tid = task_id
            self._build_parts = build_parts
            self._max_turns = max_turns
            self._k = k
            self._seed = seed
            # THE FORK, forwarded to every rollout's `run_loop`. See `rollout_branch`.
            self._branch_at = branch_at
            self._branch_seed = branch_seed

        def get_init_state(self, message_history: list | None = None) -> DriverState:
            return DriverState(messages=list(message_history or []))

        def generate_next_message(self, message: Any, state: DriverState):
            state.messages.append(message)

            # A USER message opens a new information need, so the policy is reset and one
            # whole rollout runs. A TOOL message is the environment answering a plan step we
            # already produced; it must not trigger a second rollout, or the arm would spend
            # a multiple of every other arm's budget on the same task.
            if getattr(message, "role", "") == "user":
                self._rollout(state)

            if state.pending:
                step = state.pending.pop(0)
                call = ToolCall(
                    id=_CALL_ID.format(state.n_emitted),
                    name=str(step.get("name") or step.get("tool_name") or ""),
                    arguments=dict(step.get("args") or step.get("kwargs") or {}),
                    requestor="assistant",
                )
                state.n_emitted += 1
                # cost=0.0, NOT unset. tau2's `get_cost` returns None if ANY non-tool message
                # lacks a cost, and `AssistantMessage.cost` defaults to None -- so one of our
                # hand-built messages poisoned the whole sum, `sim.user_cost` came back None,
                # and `user_sim_usd` was 0.0 on every unit. The user simulator is a SECOND,
                # uncached model conversation whose dollars are real, so that silently
                # under-reported `usd_billed` and blinded `--spend-cap` to it.
                #
                # 0.0 is the true value, not a placeholder: this turn came from our own
                # metered client and is already counted in BudgetLedger. Its incremental cost
                # in TAU2's ledger is zero, and saying so lets tau2 sum the user's turns.
                out = AssistantMessage(role="assistant", content=None, tool_calls=[call], cost=0.0)
            else:
                # AssistantMessage.validate() rejects a message with neither content nor tool
                # calls, and an empty answer is a real outcome of a rollout that stopped with
                # nothing to say. Raising there would turn one bad generation into a dead unit,
                # so the empty case gets a marker that is visibly not an answer.
                out = AssistantMessage(
                    role="assistant", content=state.answer or NO_ANSWER, cost=0.0
                )

            state.messages.append(out)
            return out, state

        def _rollout(self, state: DriverState) -> None:
            from pinq.loop import run_loop

            parts = self._build_parts()
            # Continue the turn axis rather than restarting it. `merge_trajectories` renumbers
            # Turn.turn_idx sequentially across rollouts and ledger.jsonl is joined to
            # turns.jsonl on that column, so a ledger that counted 0,1,2 again for the second
            # user message would attribute its spend to the first message's turns.
            base = sum(len(t.turns) for t in state.trajectories)
            parts["ledger"].turn_base = base
            local_at, local_seed = rollout_branch(
                base=base, branch_at=self._branch_at, branch_seed=self._branch_seed
            )
            view = dialogue_view(self._suite, self._tid, state.messages)
            traj = run_loop(
                view=view,
                inquirer=parts["inquirer"],
                retriever=parts["retriever"],
                drafter=parts["drafter"],
                answerer=parts["answerer"],
                ledger=parts["ledger"],
                # None ON PURPOSE. See the module docstring, decision 1: an action executed
                # here would never appear in the transcript the grader replays.
                actuator=None,
                max_turns=self._max_turns,
                k=self._k,
                seed=self._seed,
                # THE FORK, mapped onto this rollout. Absent, every candidate re-ran the
                # parent's seed and sent the identical request; see `rollout_branch`.
                branch_at=local_at,
                branch_seed=local_seed,
                # DERIVED FROM THE ARM, not pinned here. This line read `False`, so on the
                # only suite with a user simulator `inquirer_may_ask_user` could not open the
                # channel it exists to open: the prompt said "you may ask the user", the
                # policy emitted target="user", and the loop charged it as a wasted turn.
                # The arm had 0 runs on tau2 and the user-private ceiling was never measured.
                allow_user_target=allows_user_target(parts["inquirer"]),
            )
            state.trajectories.append(traj)
            draft = traj.final_draft
            plan = [dict(step) for step in (draft.tool_plan if draft else ()) if step]

            # Only tools the agent is ACTUALLY offered. Two reasons, both measured:
            #
            #  * 5 of the tools named in tau2's gold actions are USER tools
            #    (apply_for_credit_card, submit_referral, call_discoverable_user_tool, ...).
            #    102 of the 955 gold actions belong to the customer, and 7 of the 97 tasks are
            #    graded entirely on them. An agent emitting one is not doing the task, it is
            #    impersonating the customer.
            #  * A hallucinated name comes back "Tool 'x' not found", which counts against the
            #    Orchestrator's max_errors. Enough bad guesses END the simulation, so a
            #    policy that names tools badly would be scored on a truncated dialogue rather
            #    than on its actions -- a harness artifact wearing the shape of a result.
            #
            # Rejections are COUNTED, never silently dropped: "the policy planned tools it was
            # never offered" is itself a finding about the policy.
            allowed = {t["name"] for t in self._suite.tool_schemas(self._tid)}
            keep, drop = [], []
            for step in plan:
                name = str(step.get("name") or step.get("tool_name") or "")
                (keep if name in allowed else drop).append(step)
            state.pending = keep
            state.rejected += len(drop)
            answer = traj.outcome.answer
            state.answer = answer.text if answer else ""

    return PinqDriverAgent


# --------------------------------------------------------------------------- the budget gate


def env_calls_are_charged(suite: Any) -> bool:
    """Does this suite budget the dialogue's tool calls at all (`BudgetGate`, amendment 7)?

    The one predicate both halves read -- the gate that enforces the charge and
    `_attach_env_evidence`, which used to be its only author -- so they cannot disagree about
    which suites are metered. Banking has no index and gets its evidence through a retriever,
    which `run_loop` already charges; gating its tool calls would invent a charge that was never
    part of that suite's budget.
    """
    return getattr(suite, "index", None) is not None and hasattr(suite, "uids_for_calls")


def budget_exempt_tools(domain: str) -> frozenset[str]:
    """The tools `budget_cap` neither gates nor charges in `domain`: its GENERIC tools.

    WHY THEY ARE NOT RETRIEVAL. The budget meters looking something up and changing something.
    tau2 types every tool READ, WRITE, THINK or GENERIC, and GENERIC is the type of a tool that
    does neither -- `calculate` evaluates arithmetic, `transfer_to_human_agents` hands the customer
    over. Charging them would make "the agent can still answer or hand off", the reason the user
    chose refuse-and-tell, false at exactly the moment it matters.

    READ FROM THE COMMITTED MAP (`TOOL_TYPES_PATH`), never from a list of names here: a literal
    goes stale when upstream adds a tool, and the map is already the source the replay scripts
    classify by. ONLY an explicit GENERIC exempts. A domain the map does not cover, or a tool it
    does not name, exempts nothing -- gated and charged is the safe direction, because a lookup
    wrongly exempted is free retrieval and a GENERIC tool wrongly charged is merely a cost.

    A missing or unreadable map RAISES rather than exempting nothing: that would silently take
    the hand-off away again, on a checkout whose commit says it was given back.
    """
    try:
        table = json.loads(Path(TOOL_TYPES_PATH).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Tau2DriverError(
            f"cannot read the tool-type map {TOOL_TYPES_PATH}: {exc}. The budget exemption for "
            "GENERIC tools is read from it; running without it would gate the hand-off."
        ) from exc
    types = table.get(str(domain))
    if not isinstance(types, Mapping):
        return frozenset()
    return frozenset(str(n) for n, t in types.items() if str(t).upper() == EXEMPT_TOOL_TYPE)


def _declared_mutating(env: Any, name: str) -> bool:
    """The environment's own READ/WRITE declaration; a write when it cannot say.

    `Tau2Actuator._is_mutating`'s rule, for its reason: a read recorded as a write is a false
    alarm somebody investigates, a write recorded as a read is an unsafe action nobody sees.
    """
    probe = getattr(env, "_is_mutating_tool", None)
    if probe is None:
        return True
    try:
        return bool(probe(name))
    except Exception:  # noqa: BLE001 - an unanswerable probe is a write, not a crash
        return True


class BudgetGate:
    """The TOOL budget, enforced at the one place a dialogue's tool call executes.

    WHERE, AND WHY THERE. A tool call in a tau2 dialogue runs in exactly one place:
    `Orchestrator._execute_tool_calls` -> `Environment.get_response` -> `make_tool_call`.
    `Tau2Actuator` is not on this path (`run_loop` gets `actuator=None`, see decision 1). The gate
    wraps that one method on the ORCHESTRATOR INSTANCE, so it binds every arm identically -- our
    driver and upstream's `LLMAgent` alike -- and binds nothing else: upstream's `set_state`
    replay of a fork's prefix calls `get_response` directly and never passes through here, and
    the tool-backed retriever calls `make_tool_call` directly and is charged, as an Ask, by
    `run_loop`.

    ITS OWN COUNTER, NEVER THE LEDGER (RULES amendment 7). Asks and tool calls get `budget_cap`
    EACH. The ask budget is the unit's `BudgetLedger`, charged by `run_loop` exactly as on QA; this
    gate never checks it and never charges it. The tool budget is `n_charged` against `cap`. From
    ba48ab0 to e5e8206 the gate charged tool calls to the ask ledger -- one cap for both -- and a
    questioner that asked 16 times left the agent no tool call at all (measured: base Qwen3-8B at
    132e3e8 did so on 97 of 98 retail and 55 of 55 airline recorded forks). Split, asking cannot
    starve acting and acting cannot stop the questioner.

    REFUSE AND TELL. Once `cap` calls have executed ok, a further call is NOT executed (the world
    does not move), NOT counted, and is answered with a ToolMessage carrying `BUDGET_REFUSAL`, so
    the agent can still answer the customer or hand off. Every call in a multi-call message is
    decided in order, so the last unit of budget goes to the first call that asks for it.

    THE RULE, in full: a call to a tool in `exempt` (the domain's GENERIC tools, see
    `budget_exempt_tools`) is executed and never checked or counted; any other call is refused
    once `n_charged >= cap`, and otherwise executed and counted 1 if the environment accepted it.
    So on a gated unit `n_charged` == ok executed non-exempt calls AFTER the fork point, and
    `spent["retrieval_calls"]` == run_loop's charges alone (Asks, plus any query-expansion
    sub-retrievals). A fork's inherited prefix is replayed by upstream's `set_state`, not through
    here, so the tool cap is on the LIVE spend.

    ONE OWNER PER BUDGET. This gate owns the tool count; `meter_env_calls` no longer charges on a
    gated unit, and `run_loop` owns the ask ledger. Nothing is counted twice, and nothing is
    counted by an instrument that could not refuse it.

    A FAILED CALL IS NOT COUNTED, exactly as `meter_env_calls` has always said: it returned
    nothing, and billing for it would make a flaky environment read as an expensive policy.

    A REFUSAL IS NOT AN ERROR. The ToolMessage says `error=False`, so upstream does not count it
    toward `max_errors`. That cap bounds an agent making bad calls (see DEFAULT_MAX_ERRORS), and
    our driver is open-loop within a turn: it emits the rest of an already-planned tool list
    without re-planning, so counting refusals would end its dialogues as TOO_MANY_ERRORS -- a
    harness artifact, arm-dependent, wearing the shape of a result.

    AN EXEMPT TOOL MUST NOT WRITE. `exempt` comes from a committed copy of upstream's
    declarations; `install` asks the LIVE environment about each exempt tool and refuses the unit
    if any is a write, because a stale map would otherwise let a write through free.

    `ledger` is READ, once, for `spent_before_first_live_call`, and never written. `refused` maps
    each refused tool-call id to the environment's own READ/WRITE declaration for that tool,
    which is what `split_refused` stores.
    """

    def __init__(
        self, ledger: BudgetLedger, *, cap: int, exempt: frozenset[str] = frozenset()
    ) -> None:
        self.ledger = ledger
        self.cap = int(cap)
        self.exempt = frozenset(exempt)
        self.refused: dict[str, bool] = {}
        self.n_charged = 0
        # THE ASK LEDGER AT THE MOMENT THE FIRST LIVE NON-EXEMPT CALL EXECUTES (RULES amendment
        # 4): the questioner's spend when the agent first acted. Read here because nothing
        # recorded afterwards says when that was. None until such a call executes.
        self.spent_before_first_live_call: float | None = None

    @property
    def first_live_call_executed(self) -> bool:
        return self.spent_before_first_live_call is not None

    @property
    def n_refused(self) -> int:
        return len(self.refused)

    def install(self, orch: Any) -> None:
        execute = orch._execute_tool_calls  # upstream's bound method: it counts max_errors
        env = orch.environment
        writes = sorted(n for n in self.exempt if _declared_mutating(env, n))
        if writes:
            raise Tau2DriverError(
                f"the tool-type map types {writes} {EXEMPT_TOOL_TYPE}, but this environment "
                "declares them writes. Exempting them would let a write past the budget; the map "
                f"at {TOOL_TYPES_PATH} disagrees with the installed tau2."
            )

        def gated(tool_calls: list) -> list:
            out = []
            for tc in tool_calls:
                if str(tc.name) in self.exempt:
                    # Not retrieval: neither checked nor counted, spent or not.
                    out.extend(execute([tc]))
                    continue
                if self.n_charged >= self.cap:
                    out.append(self._refuse(env, tc))
                    continue
                if self.spent_before_first_live_call is None:
                    self.spent_before_first_live_call = float(
                        self.ledger.spent.get(HARD_CURRENCY, 0.0)
                    )
                (msg,) = execute([tc])
                if not bool(getattr(msg, "error", False)):
                    self.n_charged += 1
                out.append(msg)
            return out

        orch._execute_tool_calls = gated

    def _refuse(self, env: Any, tc: Any) -> Any:
        from tau2.data_model.message import ToolMessage

        self.refused[str(tc.id)] = _declared_mutating(env, str(tc.name))
        return ToolMessage(
            id=tc.id, role="tool", content=BUDGET_REFUSAL, requestor=tc.requestor, error=False
        )


def dialogue_secondaries(state: Any, gate: BudgetGate | None) -> dict[str, Any]:
    """RULES amendment 4's secondaries: recorded only, no behaviour depends on them.

    `rollout1_*` is the FIRST `run_loop` rollout of the live dialogue (a fork's is the one its
    last prefix user turn opens) -- `stop_reason` on the status has only ever been the LAST
    rollout's. `rollout1_n_asks` counts Ask turns exactly as `n_asks` does, so a rejected user
    Ask is counted and uncharged. None where there is no rollout to read: the stock arm runs
    none, and a unit killed inside the simulation has no agent state to read it from.

    The spend fields are the gate's (see `BudgetGate.spent_before_first_live_call`), and None on
    a unit that has no gate.
    """
    trajs = list(getattr(state, "trajectories", None) or ()) if state is not None else []
    first = trajs[0] if trajs else None
    return {
        "rollout1_stop_reason": first.stop_reason if first is not None else None,
        "rollout1_n_asks": first.n_asks if first is not None else None,
        "first_live_call_executed": gate.first_live_call_executed if gate is not None else None,
        "spent_before_first_live_call": (
            gate.spent_before_first_live_call if gate is not None else None
        ),
    }


def _answer_at(msgs: Sequence[Any], at: int, tc: Any) -> Any:
    """The ToolMessage answering `tc`, when it sits where upstream puts it, else None.

    Upstream's `_execute_tool_calls` answers a message's calls in order and the trajectory is
    flat, so the j-th call of message i is answered at i + 1 + j -- the adjacency `set_state`
    itself demands. A call with no answer there was never executed (the dialogue ended first).
    """
    answer = msgs[at] if at < len(msgs) else None
    if answer is None or getattr(answer, "role", "") != "tool":
        return None
    return answer if str(getattr(answer, "id", "")) == str(tc.id) else None


def _is_refusal(tc: Any, answer: Any, refused: Mapping[str, bool]) -> bool:
    """Refused only if BOTH halves say so; see `split_refused`."""
    return (
        answer is not None
        and str(tc.id) in refused
        and getattr(answer, "content", None) == BUDGET_REFUSAL
    )


def env_call_census(
    messages: Sequence[Any], *, boundary: int, refused: Mapping[str, bool], exempt: frozenset[str]
) -> dict[str, int]:
    """Count the dialogue's EXECUTED tool calls from the transcript alone.

    THE INSTRUMENT THAT CAN DISAGREE WITH THE GATE. On a gated unit `meter_env_calls` does not
    charge, so its overrun check cannot fire; and `BudgetGate.n_charged` is the gate's own
    counter, which cannot catch the gate. This reads neither: it walks the transcript the
    Orchestrator produced and counts what the environment answered. On a correct gate
    `n_live_env_calls_nongeneric_ok == BudgetGate.n_charged`, and the runner records whether
    that held (`gate_census_agrees`).

    THE BOUNDARY IS THE INHERITED HISTORY. `boundary` is the length of the task's
    `initial_state.message_history` (a fork's prefix): upstream puts exactly those messages first
    in the trajectory, timestamped before anything live. A call ANSWERED inside it was executed
    by upstream's `set_state` replay, never by the gate: it is the task's starting state and is
    counted in `n_prefix_env_calls`. A call whose answer falls AFTER the boundary -- a history
    ending on an assistant tool call, which upstream would then execute live through the gate --
    is LIVE, because live is where it was executed and charged. (At the installed tau2 that
    shape cannot start: `set_state` raises "Tool message expected" first, and `forked_task`
    refuses it before that. The rule is stated for the day either changes.)

    Refusals are not executed and are counted nowhere here. `exempt` is the domain's GENERIC
    set as read from the map -- passed in by the caller, not taken off the gate, so a gate that
    exempted more than the map says is caught rather than agreed with.

    `n_transfer_to_human` counts executed live hand-offs (`HANDOFF_TOOL`), ok or not: the one
    exit a budget-exhausted agent is told it still has.
    """
    msgs = list(messages)
    out = {
        "n_prefix_env_calls": 0,
        "n_live_env_calls": 0,
        "n_live_env_calls_generic": 0,
        "n_live_env_calls_nongeneric": 0,
        "n_live_env_calls_nongeneric_ok": 0,
        "n_transfer_to_human": 0,
    }
    for i, m in enumerate(msgs):
        for j, tc in enumerate(getattr(m, "tool_calls", None) or ()):
            at = i + 1 + j
            answer = _answer_at(msgs, at, tc)
            if answer is None or _is_refusal(tc, answer, refused):
                continue
            if at < boundary:
                out["n_prefix_env_calls"] += 1
                continue
            out["n_live_env_calls"] += 1
            if str(tc.name) == HANDOFF_TOOL:
                out["n_transfer_to_human"] += 1
            if str(tc.name) in exempt:
                out["n_live_env_calls_generic"] += 1
                continue
            out["n_live_env_calls_nongeneric"] += 1
            if not bool(getattr(answer, "error", False)):
                out["n_live_env_calls_nongeneric_ok"] += 1
    return out


def split_refused(
    messages: Sequence[Any], refused: Mapping[str, bool]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """`(the transcript the environment actually answered, one record per refused call)`.

    WHY THE GRADER MUST NOT SEE A REFUSAL. `EnvironmentEvaluator.calculate_reward` replays the
    trajectory through `Environment.set_state`, which EXECUTES every mutating tool call in it via
    `get_response` whatever its recorded ToolMessage says, and then compares contents strictly.
    A refused write left in would be performed on the predicted world -- a write the live world
    never saw -- and then fail against the refusal text, so the grade would raise and
    `tau_reward` would be absent. Measured on upstream's own evaluator in
    `test_t6_upstreams_grader_would_execute_a_refused_write_left_in_the_transcript`. The same
    holds for anyone recomputing a reward offline from `env_calls`, which is why the harvest
    derives `env_calls` from the executed transcript only.

    A CALL IS REFUSED ONLY IF BOTH HALVES SAY SO: its id is one the gate refused AND the
    ToolMessage that answers it -- the next messages, in order, which is the adjacency upstream's
    own `set_state` demands -- carries exactly `BUDGET_REFUSAL`. An id alone could collide with
    a call in a fork's foreign prefix. A message whose calls were all refused and that says
    nothing is dropped; one that keeps an executed call, or text, keeps them.

    `refused` is `BudgetGate.refused`: refused tool-call id -> the environment's own READ/WRITE
    declaration for that tool. Each record carries `transcript_idx` (1-based position among ALL
    tool calls in the full transcript, prefix included) and `turn_idx` (user messages so far --
    `EnvCall.turn_idx`'s definition), so it can be interleaved with the executed log.
    """
    mutating = {str(k): bool(v) for k, v in refused.items()}
    msgs = list(messages)
    drop: set[int] = set()  # message positions to remove: the refusal ToolMessages
    keep_calls: dict[int, list] = {}  # message position -> the tool calls it keeps
    records: list[dict[str, Any]] = []
    turn = 0
    idx = 0
    for i, m in enumerate(msgs):
        if getattr(m, "role", "") == "user":
            turn += 1
        calls = list(getattr(m, "tool_calls", None) or ())
        if not calls:
            continue
        kept = []
        for j, tc in enumerate(calls):
            idx += 1
            at = i + 1 + j
            if not _is_refusal(tc, _answer_at(msgs, at, tc), mutating):
                kept.append(tc)
                continue
            drop.add(at)
            records.append(
                {
                    "tool_call_id": str(tc.id),
                    "transcript_idx": idx,
                    "turn_idx": turn,
                    "requestor": "user" if str(tc.requestor) == "user" else "assistant",
                    "tool_name": str(tc.name),
                    "kwargs_json": canon(dict(tc.arguments or {})),
                    "mutating": bool(mutating[str(tc.id)]),
                }
            )
        if len(kept) != len(calls):
            keep_calls[i] = kept

    out: list[Any] = []
    for i, m in enumerate(msgs):
        if i in drop:
            continue
        if i in keep_calls:
            kept = keep_calls[i]
            if not kept and not getattr(m, "content", None):
                continue
            m = m.model_copy(update={"tool_calls": kept or None})
        out.append(m)
    return out, records


# --------------------------------------------------------------------------- harvest


def count_user_turns(messages: Sequence[Any]) -> int:
    """How many messages the USER produced. Role-filtered, not `len(messages)`.

    A tau2 transcript interleaves user, assistant and tool messages. The total length is
    dominated by the agent's tool calls, so it measures agent verbosity; only the user-role
    count measures how much the customer had to say.
    """
    return sum(1 for m in messages if getattr(m, "role", "") == "user")


def user_sim_usd(messages: Sequence[Any], model: str, *, table: Any = None) -> tuple[float, int]:
    """Price the user simulator's traffic from TOKENS, never from the provider's dollars.

    Returns `(usd, n_unpriced)`.

    `sim.user_cost` looks like the right field and is not. tau2's `get_response_cost`
    (`tau2/utils/llm_utils.py:127-131`) catches every exception out of litellm's
    `completion_cost` and returns 0.0; our user-sim pin is a proxy model name with no
    litellm price row, so it raises, is swallowed, and every user message carries
    `cost == 0.0`. Measured on the first live tau2 rollout: 43 user turns across 4 units,
    `user_sim_usd` 0.0 on every one. `usd_billed` then under-reports the invoice and
    `--spend-cap` cannot see the harness half of it.

    This is the rule the price table already states in its own `_why`: "USD is computed as
    tokens x these rates and NEVER read from a provider response field." tau2 records
    `usage` per message, so the user simulator is priced with the SAME `PriceTable` as
    every other call we bill -- one pricing path, one set of rates, one version pin.

    User-role only, on purpose. Assistant and tool tokens are the POLICY's spend and are
    already in the ledger; adding them here would double-bill them into `usd_billed` and
    corrupt the cross-arm token parity the ledger exists to protect.

    A message it cannot price -- no usage, or a model with no price row -- is COUNTED and
    returned, never raised and never silently dropped. The count is what makes a $0.00 that
    means "free" distinguishable from a $0.00 that means "unmeasured", which is the whole
    point; but it must not be an exception, because this runs at the END of a finished
    rollout. Raising here would discard a unit whose money is already spent, and (per the
    same escape path as `ReconcileError`) take the rest of the sweep with it. On a 2,037-unit
    grid that trades a visible accounting gap for a destroyed campaign invocation.
    """
    return role_usd(messages, model, role="user", table=table)


def role_usd(
    messages: Sequence[Any], model: str, *, role: str, table: Any = None
) -> tuple[float, int]:
    """`user_sim_usd`'s body, with the role as a parameter. Returns `(usd, n_unpriced)`.

    A SECOND ROLE EXISTS BECAUSE A SECOND PARTICIPANT BILLS OUTSIDE THE LEDGER. The stock arm
    is upstream's `LLMAgent`, which talks to litellm directly exactly as the user simulator
    does, so its ASSISTANT messages carry `usage` and no `BudgetLedger` row. Pricing them here
    is the same rule, applied to the other speaker: tokens times our own rates, never the
    provider's dollar field.

    Every caller must still pass ONE role. Summing assistant and user together would
    double-bill the augmented arms, whose assistant turns are already in the ledger.
    """
    from pinq_adapters.llm.pricing import LLMConfigError, PriceTable

    if table is None:
        table = PriceTable.load()
    total, unpriced = 0.0, 0
    for m in messages:
        if getattr(m, "role", "") != role:
            continue
        usage = getattr(m, "usage", None)
        if not usage:
            unpriced += 1
            continue
        get = usage.get if hasattr(usage, "get") else lambda k, d=0: getattr(usage, k, d)
        try:
            total += table.usd(
                model,
                tok_prompt=int(get("prompt_tokens", 0) or 0),
                tok_completion=int(get("completion_tokens", 0) or 0),
            )
        except LLMConfigError:
            unpriced += 1
    return total, unpriced


def role_tokens(messages: Sequence[Any], *, role: str) -> dict[str, int]:
    """Prompt/completion tokens for one role, summed off tau2's own `usage` records.

    The stock arm's tokens reach no `BudgetLedger`, so `traj.usage` is legitimately zero on it
    and a cost table built from the ledger alone would report the control as free. Recorded
    under their own names rather than folded into `usage`, because `reconcile` compares
    `traj.usage` against the ledger and a number added on one side would make a true
    reconciliation read as a break.
    """
    out = {"tok_prompt": 0, "tok_completion": 0, "n_messages": 0, "n_unmetered": 0}
    for m in messages:
        if getattr(m, "role", "") != role:
            continue
        out["n_messages"] += 1
        usage = getattr(m, "usage", None)
        if not usage:
            out["n_unmetered"] += 1
            continue
        get = usage.get if hasattr(usage, "get") else lambda k, d=0: getattr(usage, k, d)
        out["tok_prompt"] += int(get("prompt_tokens", 0) or 0)
        out["tok_completion"] += int(get("completion_tokens", 0) or 0)
    return out


def env_calls_from(messages: Sequence[Any]) -> tuple[EnvCall, ...]:
    """Derive the executed action log from the ORCHESTRATOR's transcript.

    Ground truth, and deliberately not `Tau2Actuator.env_calls`: the orchestrator is what
    actually executed these calls, and the transcript is what the grader replays. Deriving the
    log from anything else would let the stored `env_calls.parquet` disagree with the number in
    the reward column -- the two would then be two accounts of one rollout with no way to tell
    which was wrong.

    `unlock_satisfied` is tracked exactly as the environment does it: an unlock counts only
    once the call SUCCEEDED, because a failed unlock leaves the tool locked and crediting it
    would fabricate a crossed prerequisite edge.
    """
    from pinq_adapters.tau2.actuator import CALL_TOOL, UNLOCK_TOOL

    results: dict[str, Any] = {}
    for m in messages:
        if getattr(m, "role", "") == "tool" and getattr(m, "id", None):
            results[str(m.id)] = m

    unlocked: set[str] = set()
    out: list[EnvCall] = []
    seq = 0
    turn = 0
    for m in messages:
        if getattr(m, "role", "") == "user":
            turn += 1
        for tc in getattr(m, "tool_calls", None) or ():
            res = results.get(str(tc.id))
            ok = not bool(getattr(res, "error", False)) if res is not None else False
            args = dict(tc.arguments or {})
            gated = tc.name == CALL_TOOL
            target = str(args.get("agent_tool_name") or args.get("tool_name") or "")
            seq += 1
            out.append(
                EnvCall(
                    seq=seq,
                    turn_idx=turn,
                    requestor="user" if str(tc.requestor) == "user" else "assistant",
                    tool_name=str(tc.name),
                    kwargs_json=canon(args),
                    ok=ok,
                    # A digest, never the payload: a tool result here is a customer record.
                    result_digest=h("res", str(getattr(res, "content", "")))[:16],
                    mutating=True,  # conservatively; the orchestrator does not expose the flag
                    unlock_required=gated,
                    unlock_satisfied=(not gated) or (target in unlocked),
                )
            )
            if ok and tc.name == UNLOCK_TOOL:
                name = str(args.get("agent_tool_name") or "")
                if name:
                    unlocked.add(name)
    return tuple(out)


def meter_env_calls(
    index: Any, ledger: Any, calls: Sequence[Mapping[str, Any]], *, charge: bool = True
) -> tuple[tuple[str, ...], bool]:
    """Charge each tool call to the budget and return the records it read, in first-seen order.

    A TOOL CALL COSTS BUDGET. A proactive agent retrieves AND acts, and looking something up has
    to cost something or anticipation is worthless: an agent that can call tools for free calls
    them exhaustively, which is brute force wearing proactivity's name. Before RULES amendment 7
    it debited `budget_cap` on the same ledger counter a retrieval debits on musique; since, it
    debits a tool budget of its own (`BudgetGate`), and the ledger is the questioner's alone.

    ON A GATED UNIT THIS DOES NOT CHARGE (`charge=False`), AND NOTHING CHARGES THE LEDGER FOR A
    TOOL CALL AT ALL. Under RULES amendment 7 asks and tool calls have separate budgets of
    `budget_cap` each: the ledger is the ASK budget, charged by `run_loop` alone, and the tool
    budget is `BudgetGate`'s own counter, which refuses a call once `budget_cap` calls have
    executed -- so the cap BINDS on tool calls during the dialogue rather than being checked
    after it. What remains here is the half that was always this function's alone: turning the
    calls into the records they read. Every unit whose tool calls are budgeted is gated, so no
    gated unit ever reaches the charging path and no gated run can carry an overrun from it.
    The overrun check a gated run DOES get is `env_call_census`, in `run_tau2_unit`: it reads the
    transcript rather than this function's arithmetic, and it can disagree with the gate.

    `charge=True` IS THE PRE-GATE RULE, UNCHANGED, AND IT IS NOT THE GATE'S RULE. It charges every
    ok call to the SAME ledger the asks are on, GENERIC ones included (`calculate`,
    `transfer_to_human_agents`) -- one shared cap, where the gate counts only non-GENERIC calls,
    on a counter of their own. It is kept exactly as it ran, because its job is to reproduce what
    the recorded campaigns were charged: `tests/test_post_hoc_harvest_meter.py` replays them
    through it. Applied to a GATED run's calls it answers a question no gated run was run under;
    `tests/test_tau2_budget_gate.py` uses it only on a FRESH ledger, over the non-GENERIC calls,
    as an independent recount of the gate's tool count. Everything below describes that path.

    MUTATIONS ARE CHARGED AND CARRY NO EVIDENCE. 176 of retail's 550 required calls are writes
    (`cancel_pending_order`, the `modify_*` family). A write costs a turn and changes the world;
    it does not teach the agent a fact. So it debits `retrieval_calls` and adds no uid, which is
    also what keeps `reconcile`'s `ledger.unique_docs == len(evidence.doc_ids)` true.

    THE CALL IS CHARGED EVERY TIME, THE DOCUMENT ONLY ONCE. `note_docs` meters unique documents
    so a policy cannot free-ride on the cache by repeating a query; the repeat still cost the
    agent a turn, so the call itself is charged again.

    A FAILED CALL IS NOT CHARGED. It returned nothing, and billing for it would make a flaky
    environment read as an expensive policy.

    uids come from ARGUMENTS, never results: a retail tool result is a customer record and the
    runner stores only a `result_digest`.

    THE SPEND IS RECORDED, THE OVERRUN IS RETURNED, AND NOTHING IS RAISED. Returns
    `(uids, overrun)`. Inside `run_loop` the cap is a decision the policy is still making, and
    exceeding it stops the rollout with `stop_reason="budget"` -- a measurement, and that path is
    untouched (`pinq.loop` still calls `charge_retrieval`). Here the dialogue is already over and
    the Orchestrator has already executed the plan, so raising prevents no spend: it DELETES A
    COMPLETED RUN, via the caller's own `except Exception`, which writes `status: error` over a
    unit that finished and carries a real reward. (Preventing the spend is the gate's job, and it
    does it before the call runs; an overrun reported here on a gated unit would be a gate
    defect, which is why the tests replay gated runs through this path and require none.)

    AND IT DELETES SELECTIVELY, WHICH IS WHY THIS IS NOT A CONVENIENCE. Replayed over the 204
    recorded tau2_retail/test fork runs behind the published transfer result -- all 204
    `status: ok`, cap 16 -- raising here refuses 70: 57 of the 102 `self_ask` runs against 13 of
    the 102 `inquirer_prompted` runs, because `self_ask` charges 18.3 retrieval calls on average
    and `inquirer_prompted` 9.9. The arm that acts more loses more runs, and its expensive runs
    are its long dialogues, so the deletion filters the endpoint by its own value: 102 pairs
    collapse to 41, 32 recorded dialogues to 19, and the published $-4.3137$ follow-up turns read
    $-0.2927$ on what survives. Same direction-of-bias failure `_reward_of` records for premature
    termination, arriving through a different door. `tests/test_post_hoc_harvest_meter.py` holds
    the replay.

    An overspend is still never SILENT -- that property is real and is what the old raise was
    defending. It lands in `spent["retrieval_calls"]`, and the caller records
    `post_hoc_budget_overrun` on the run.
    """
    from pinq_adapters.tau2.retail_units import uids_for_call

    seen: list[str] = []
    known: set[str] = set()
    overrun = False
    for call in calls:
        if not call.get("ok", True):
            continue
        args = call.get("kwargs_json")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, Mapping):
            args = {}
        if charge:
            # RECORDED AND CHECKED, NEVER RAISED. `charge_retrieval` is check-then-bump and the
            # check raises; see the docstring for why raising here deletes a finished run
            # instead of preventing a spend. The two halves are done separately so the overrun
            # is still seen.
            try:
                ledger.check(1.0)
            except BudgetExceeded:
                overrun = True
            ledger.record(HARD_CURRENCY, 1.0)
        for uid in uids_for_call(index, str(call.get("tool_name") or ""), args):
            if uid not in known:
                known.add(uid)
                seen.append(uid)
    if seen:
        # DOC IDS, NOT UIDS. `BudgetLedger._docs` is one set and `run_loop` fills it with
        # `u.doc_id`; `reconcile` then compares its size against `len(evidence.doc_ids)`. Noting
        # uids here put a SECOND string for the same document into that set, so any record the
        # Inquirer retrieved and a tool call then re-read was counted twice and the run died on
        # `ReconcileError` -- after its dialogue had been billed, and selectively by arm. 15 of
        # this lane's 42 probe units went that way (25% of the airline ones), each leaving a run
        # directory containing manifest.json alone. See tests/test_tau2_reconcile_namespace.py.
        #
        # The inversion is total on both shipped indexes (retail 1550 doc keys -> 1550 distinct
        # uids, airline 2800 -> 2800; measured). A uid that does not invert -- `uids_for_call`
        # can synthesise one via `index.uid("products", pid)` -- falls back to itself, which is
        # the old behaviour for that uid alone rather than a silent doc_id of "".
        by_uid = {u: d for d, u in getattr(index, "uids", {}).items()}
        try:
            ledger.note_docs(by_uid.get(u, u) for u in seen)
        except BudgetExceeded:
            # The unique-document cap, for the same reason: it is not set on a tau2 unit
            # (`BudgetLedger(cap=cap)` leaves `unique_doc_cap` None), but if it ever is, the
            # documents have already been read by the time this runs.
            overrun = True
    return tuple(seen), overrun


def allows_user_target(inquirer: object) -> bool:
    """Does THIS arm's policy get to address the customer? Read off the policy, never set.

    A second switch here would be free to disagree with the arm table; the arm table is where
    "the user channel is closed by construction in every confirmatory arm" is already stated,
    so this reads that declaration rather than restating it. Anything that does not declare --
    every arm but `inquirer_may_ask_user`, and None -- is CLOSED.
    """
    return bool(getattr(inquirer, "may_ask_user", False))


def merge_trajectories(
    trajs: Sequence[Trajectory], *, view: TaskView, ledger: BudgetLedger
) -> Trajectory:
    """One Trajectory per unit, from the per-user-message rollouts.

    Turn indices are RENUMBERED sequentially across assistant turns. They are the x-index of
    the prefix ladder and `Trajectory.prefix(k)` slices on position, so three rollouts each
    starting at turn 0 would make `prefix(2)` mean "the first two turns of every rollout" --
    which is not a prefix of anything the policy ever ran.
    """
    from dataclasses import replace as _replace

    turns: list[Any] = []
    for t in trajs:
        for turn in t.turns:
            turns.append(_replace(turn, turn_idx=len(turns)))
    units: list[Any] = []
    seen: set[str] = set()
    for t in trajs:
        for u in t.evidence.units:
            if u.uid not in seen:
                seen.add(u.uid)
                units.append(u)
    last = trajs[-1] if trajs else None
    from pinq.types import Evidence

    return Trajectory(
        view=view,
        turns=tuple(turns),
        evidence=Evidence(units=tuple(units)),
        outcome=Outcome(),  # filled in by the driver, which owns the reward and the env log
        usage=ledger.usage,
        stop_reason=(last.stop_reason if last else "max_turns"),
        calls=tuple(ledger.calls),
        terminal_usage=_fold(t.terminal_usage for t in trajs),
        final_draft=(last.final_draft if last else None),
    )


def _fold(usages) -> Usage:
    total = Usage()
    for u in usages:
        total = total + u
    return total


# --------------------------------------------------------------------------- the unit


def _user_model() -> str:
    model = os.environ.get("PI_MODEL_USERSIM", "")
    if not model:
        raise Tau2DriverError(
            "tau2 needs a user simulator: set PI_MODEL_USERSIM. Defaulting it would let two "
            "arms be graded against two different customers and still be compared."
        )
    return model


# THE USER SIMULATOR'S RETRY LADDER, which our client's does not cover.
#
# `litellm_client` has an 8-attempt tenacity ladder spanning ~148s (three 60s windows) and
# treats `litellm.RateLimitError` as retryable. The user simulator never touches it: tau2 hands
# `_user_llm_args()` straight to `litellm.completion` inside its own code. Observed on the first
# live tau2_retail unit -- the run died on "Rate limit exceeded for team ... Remaining: 0" with
# `calls.jsonl` ABSENT, i.e. our client had not recorded a single call, which is what identifies
# the failure as upstream of the ladder rather than an exhaustion of it.
#
# Matched to our own ladder's attempt count so the two paths survive the same outage. litellm
# reads `Retry-After` for 429s (`litellm._calculate_retry_after`), so the wait tracks the
# window rather than a fixed backoff.
#
# TWO RETRY LAYERS ARE NOT A CONTRADICTION HERE. `litellm_client` passes `num_retries=0`
# because tenacity owns its retries and two layers would make its recorded `retries` a lie. On
# this path nothing else owns them.
USERSIM_NUM_RETRIES = 8

# A hung call must not stall a unit to its wall-clock cap. Generous rather than tight: the
# simulator is a full LLM turn and a slow response is not a failure.
USERSIM_TIMEOUT_S = 120.0


def _stock_model() -> str:
    """The model upstream's own agent loop runs on: the DRAFTER's role pin.

    Not a pin of its own. The augmented arms answer the customer with `PI_MODEL_DRAFTER`, so
    reading anything else here would make "stock versus augmented" a comparison of two models
    wearing a comparison of two architectures' name. Refuses rather than defaulting, exactly
    as `_user_model` does and for the same reason.
    """
    model = os.environ.get("PI_MODEL_DRAFTER", "")
    if not model:
        raise Tau2DriverError(
            "the stock tau2 agent runs on the Drafter's pin: set PI_MODEL_DRAFTER. Defaulting "
            "it would make stock-versus-augmented a model comparison."
        )
    return model


def _user_llm_args() -> dict[str, Any]:
    """tau2 forwards these straight to `litellm.completion`, so the proxy is reachable the same
    way our own client reaches it. Temperature 0.0 because the user simulator is part of the
    ENVIRONMENT: a customer who answers differently for two arms is a different task."""
    args: dict[str, Any] = {
        "temperature": 0.0,
        # Arrives through **kwargs -- it is absent from `litellm.completion`'s signature.
        # Verified honoured rather than assumed: against a dead host, num_retries=2 took 1.4s
        # against 0.3s at 0.
        "num_retries": USERSIM_NUM_RETRIES,
        "timeout": USERSIM_TIMEOUT_S,
    }
    if os.environ.get("LITELLM_BASE_URL"):
        args["api_base"] = os.environ["LITELLM_BASE_URL"]
    if os.environ.get("LITELLM_API_KEY"):
        args["api_key"] = os.environ["LITELLM_API_KEY"]
    return args


# BOTH tau2 DOMAINS ARE DIALOGUE BENCHMARKS and both need the Orchestrator: the only task text
# upstream ships is the customer's script, and the reward is over a transcript. Membership, not
# equality against "tau2" -- `spec.suite_id == "tau2"` sent retail down the flat path, where
# `view()` raises Tau2NeedsOrchestrator and every unit fails.
DIALOGUE_SUITES: frozenset[str] = frozenset({"tau2", "tau2_retail", "tau2_airline"})


# --------------------------------------------------------- the benchmark's own protocol

# WHY THERE IS A SECOND PROTOCOL AT ALL. Everything above is this repository's tau2 harness:
# the caps are ours (100/30, argued at DEFAULT_MAX_ERRORS), the reward is the DB-state check
# alone, and a prematurely terminated dialogue is still graded on what it did. Those choices
# are defended where they are made and they are the right ones for a within-repository
# contrast. They are not tau2-bench's, and a number produced under them may not be put beside
# a leaderboard's.
#
# So the benchmark's own protocol is a NAMED alternative rather than an edit: `""` is
# everything already on disk, `tau2_upstream` is upstream's. It differs in exactly three
# places and each one is checked by tests/test_tau2_full_protocol.py:
#
#   * max_steps / max_errors come from `tau2.config`, not from the constants above;
#   * the reward is `evaluate_simulation(..., EvaluationType.ALL, ...)`, the PRODUCT of every
#     component in the task's own `reward_basis` -- which on retail is DB x NL_ASSERTION and
#     on airline is DB x COMMUNICATE, so the `tau_reward` recorded everywhere else in this
#     repository (db_reward, whenever DB is in the basis) is an UPPER BOUND on it;
#   * upstream's termination gate applies, so a dialogue that ran out of steps scores 0.0.
#
# The name lands in `upstream_pins`, which is in `pinq.ids.SEMANTIC_FIELDS`, so the two
# protocols cannot collide on a run id, a run directory or a `--resume` sentinel.
UPSTREAM_PROTOCOL = "tau2_upstream"

# The upstream agent loop, run unchanged, with no Inquirer anywhere in it. It is a control
# and not an ablation: nothing of ours is switched off inside it, because none of ours is in
# it. See `build_stock_agent`.
STOCK_ARM = "tau2_stock"


def protocol_caps(protocol: str) -> tuple[int, int]:
    """`(max_steps, max_errors)` for a named protocol.

    UPSTREAM'S PAIR IS READ FROM `tau2.config`, NOT COPIED HERE. A literal would go stale the
    moment upstream moves its own defaults, and the failure mode is a number labelled
    leaderboard-comparable that was produced under caps the leaderboard does not use --
    invisible, because both values are plausible and both are recorded.

    REFUSES AN UNKNOWN NAME. A typo that fell back to the default would produce runs stamped
    with a protocol they did not run under, which is worse than a crash by exactly the margin
    that makes it publishable.
    """
    if not protocol:
        return DEFAULT_MAX_STEPS, DEFAULT_MAX_ERRORS
    if protocol != UPSTREAM_PROTOCOL:
        raise Tau2DriverError(f"unknown tau2 protocol {protocol!r}; known: {UPSTREAM_PROTOCOL!r}")
    from tau2 import config as tau2_config

    return int(tau2_config.DEFAULT_MAX_STEPS), int(tau2_config.DEFAULT_MAX_ERRORS)


def upstream_pins_for(
    protocol: str, *, user_sim: str, suite_version: str, agent_model: str = ""
) -> dict[str, str]:
    """The `upstream_pins` block, with the protocol and the caps it implies inside it.

    THE `protocol` KEY IS OMITTED WHEN EMPTY, for the same reason `semantic_hash` pops the
    branch triple rather than hashing it as None: `upstream_pins` is in `SEMANTIC_FIELDS` and
    `canon` sees a dict, so a key merely PRESENT with the empty value changes the hash of
    every tau2 run ever made -- renaming every run directory, making `--resume` re-roll a
    finished corpus, and leaving every recorded tau2 run id unreproducible. An absent key is
    exactly what the runs on disk carry, so the default protocol keeps their ids by
    construction rather than by a test that happens to pass.
    """
    steps, errors = protocol_caps(protocol)
    pins = {
        "suite_version": str(suite_version),
        # The user simulator IS part of the environment: a different customer model is a
        # different task, so it belongs in run identity rather than in a footnote.
        "user_sim": str(user_sim),
        "max_steps": str(steps),
        # In run identity: it bounds how long a dialogue may run, so two runs under
        # different caps are not the same experiment.
        "max_errors": str(errors),
    }
    if protocol:
        pins["protocol"] = str(protocol)
    # THE STOCK AGENT'S MODEL BELONGS HERE AND NOWHERE ELSE. `RunManifest.pins` is built by
    # `collect_pins` from OUR components' `llm_role`s, and the stock arm builds none of ours,
    # so its model would reach no field in run identity at all: a stock run on sonnet and a
    # stock run on gpt-oss-120b would share a run id, a directory and a resume sentinel.
    # Omitted when empty, for the reason `protocol` is.
    if agent_model:
        pins["agent_model"] = str(agent_model)
    return pins


def stock_agent_class() -> type:
    """Upstream's `LLMAgent`. Imported lazily, for the reason every tau2 import here is."""
    from tau2.agent.llm_agent import LLMAgent

    return LLMAgent


def build_stock_agent(env: Any, *, model: str, llm_args: Mapping[str, Any]) -> Any:
    """The upstream agent loop, constructed from the environment and nothing else.

    NO COMPONENT OF OURS IS CONSTRUCTED HERE, and that is the property that makes this a
    control rather than an ablation. It does not reach the arm table, it does not build an
    Inquirer, a Drafter or an Answerer, and it never calls `run_loop`. `tools` and
    `domain_policy` come off the same `Environment` the augmented arms read, so the two see
    the same world and differ only in what drives them.

    It bills through tau2's own litellm client rather than through `MeteredClient`, exactly as
    the user simulator does, so its dollars are priced from recorded tokens by the same
    `PriceTable` (see `user_sim_usd`) and never read from a provider field.
    """
    return stock_agent_class()(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm=str(model),
        llm_args=dict(llm_args),
    )


def _evaluate_simulation(**kwargs: Any) -> Any:
    """Indirection over upstream's `evaluate_simulation`, so a test can replace it.

    A module-level name rather than an inline import at the call site: a monkeypatch of the
    upstream module would also silence the flat path's grader, and a test that reaches into
    `tau2.evaluator.evaluator` cannot tell which of the two it just disabled.
    """
    from tau2.evaluator.evaluator import evaluate_simulation

    return evaluate_simulation(**kwargs)


def grade_upstream(sim: Any, task: Any, *, domain: str, env_kwargs: Mapping[str, Any]) -> dict:
    """tau2-bench's OWN reward for this dialogue: the product over the task's `reward_basis`.

    `EvaluationType.ALL` is what applies the basis. On retail that is DB x NL_ASSERTION (an
    LLM judge, upstream's `DEFAULT_LLM_NL_ASSERTIONS`, on the 40 of 114 tasks that carry
    assertions; a free 1.0 on the rest) and on airline DB x COMMUNICATE (a deterministic
    substring check). Neither is what `_reward_of` records, and the difference has a sign:
    `tau_reward` is `db_reward` whenever DB is in the basis, so it is an upper bound.

    THE TERMINATION GATE IS KEPT, unlike `_reward_of`. `evaluate_simulation` scores a
    dialogue that hit max_steps or max_errors 0.0 with no db_check at all, and `_reward_of`
    documents at length why bypassing that is right for OUR endpoint. It is not right here:
    under the benchmark's protocol the gate is the protocol, and a run that keeps its reward
    after running out of steps is not comparable with a leaderboard whose entries do not.

    `upstream_success` is upstream's own `is_successful`, not `reward > 0`: pass^k counts
    successes and a partial reward is not one.

    A FAILED GRADE IS DATA. Returns `upstream_error` and no reward rather than raising: the
    dialogue is over, its money is spent, and raising here deletes a finished run -- the same
    escape path `meter_env_calls` records at length.
    """
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.metrics.agent_metrics import is_successful

    try:
        info = _evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=str(domain),
            env_kwargs=dict(env_kwargs),
        )
    except Exception as exc:  # noqa: BLE001 - a failed grade is data, not a sweep abort
        return {"upstream_error": f"{type(exc).__name__}: {exc}"}

    reward = float(getattr(info, "reward", 0.0) or 0.0)
    out: dict[str, Any] = {
        "upstream_reward": reward,
        "upstream_success": 1.0 if is_successful(reward) else 0.0,
        "upstream_error": "",
    }
    for key, val in (getattr(info, "reward_breakdown", None) or {}).items():
        try:
            out[f"upstream.{getattr(key, 'value', key)}"] = float(val)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- forking

TRACE_ROOT = "data/traces/tau2"


def load_prefix(spec: Any, root: str = TRACE_ROOT) -> list[Any]:
    """The first `foreign_prefix_k` messages of the trace this fork was cut from.

    Empty when the spec names no trace, which is every ordinary rollout. The trace files are
    written by `pi traces import`, one JSONL row per trajectory, keyed by `trace_sha` -- the
    same sha that is part of run identity, so a run directory names the exact bytes it replayed.

    A named trace that cannot be found RAISES rather than silently running unforked: a fork that
    quietly becomes a fresh rollout would carry fork identity, land in the fork table, and be
    compared against real forks.
    """
    sha = getattr(spec, "foreign_trace_sha", None)
    k = getattr(spec, "foreign_prefix_k", None)
    if not sha or not k:
        return []
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    for path in sorted(Path(root).glob("*/*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip() or sha not in line:
                continue
            row = json.loads(line)
            if row.get("trace_sha") != sha:
                continue
            out = []
            for m in (row.get("messages") or [])[: int(k)]:
                role = str(m.get("role"))
                if role == "tool":
                    out.append(
                        ToolMessage(
                            id=str(m.get("id") or ""),
                            role="tool",
                            content=m.get("content"),
                            requestor=str(m.get("requestor") or "assistant"),
                            error=bool(m.get("error", False)),
                        )
                    )
                else:
                    cls = AssistantMessage if role == "assistant" else UserMessage
                    out.append(
                        cls(
                            role=role,
                            content=m.get("content"),
                            tool_calls=m.get("tool_calls") or None,
                            cost=0.0,
                        )
                    )
            return out
    raise FileNotFoundError(f"no trace {sha[:16]}... under {root}/; a fork cannot run unprefixed")


def forked_task(task: Any, prefix: Sequence[Any]) -> Any:
    """A deep copy of `task` whose `initial_state.message_history` is `prefix`.

    ONE OBJECT SERVES TWO CONSUMERS AND THAT IS THE POINT. `Orchestrator.initialize()` replays
    this field into the environment and seeds both participants from it;
    `EnvironmentEvaluator.calculate_reward` seeds the GOLD environment from the SAME field and
    then applies every gold action on top. Hand the Orchestrator a prefixed task and the
    evaluator the original and the two worlds start at different points -- every continuation is
    graded against a state it was never in and scores 0 for a reason that has nothing to do with
    the policy, with nothing in the output saying so.

    DEEP, AND THE ORIGINAL IS NEVER TOUCHED. `suite.tau2_task_object` may hand back the same
    object for the next seed of that task, so mutating in place leaks one fork's prefix into
    every later rollout of it in the same process.

    `initial_state` is EDITED, not replaced: all 114 telecom base tasks carry
    `initialization_actions` that set the handset state the scenario depends on, and replacing
    the object wholesale would start every telecom fork with a phone that was never
    misconfigured.

    Refuses a prefix that is empty (a fresh rollout wearing a fork's run identity) or that ends
    anywhere but a plain user turn (upstream routes on the LAST message, and an assistant turn
    leaves the USER next to speak, which is not the handover a fork is for).
    """
    if not prefix:
        raise ValueError("a fork needs a prefix; an empty one is a fresh rollout")
    last = prefix[-1]
    if str(getattr(last, "role", "")) != "user" or getattr(last, "tool_calls", None):
        raise ValueError(
            f"a prefix must end on a plain user turn so the AGENT speaks next; this one ends on "
            f"{getattr(last, 'role', '?')}"
            + (" with tool calls" if getattr(last, "tool_calls", None) else "")
        )

    from tau2.data_model.tasks import InitialState

    out = task.model_copy(deep=True)
    init = getattr(out, "initial_state", None)
    if init is None:
        out.initial_state = InitialState(message_history=list(prefix))
    else:
        init.message_history = list(prefix)
    return out


def _wire_retriever(suite, tid, *, llm, llm_free, seed, recorder):
    """Hand the suite's retriever its model, when the suite has one to give.

    NOT AN `except TypeError` FALLBACK. The published-era runner used one, and it would swallow a
    TypeError raised from ANY depth inside a suite's own construction and silently return an
    UNWIRED retriever -- reintroducing the exact failure. CHECKED: `Tau2Suite.retriever(tid)`
    (banking) takes neither a client nor a declaration and needs none, because banking retrieves
    documents by BM25 text search rather than by asking a model to pick a tool; retail, airline
    and telecom all accept both. So the branch is decided by the callee's SIGNATURE, which cannot
    be confused with a failure inside it.
    """
    import inspect

    params = inspect.signature(suite.retriever).parameters
    kw = {}
    if "llm" in params:
        kw["llm"] = llm
    if "llm_free" in params:
        kw["llm_free"] = llm_free
    if "seed" in params:
        kw["seed"] = seed
    if "recorder" in params:
        kw["recorder"] = recorder
    return suite.retriever(tid, **kw)


def run_tau2_unit(spec: Any, *, user: Any = None) -> dict[str, Any]:
    """One (task, arm, seed) tau2 unit: build, simulate, grade, write seven files.

    Mirrors `pi_run.worker.run_unit`'s contract exactly -- same status dict, same resume
    sentinel, same never-raise-for-a-task-level-failure rule -- because the sweep reassembles
    both kinds of unit by the same key and `summarize()` reads the same fields.

    `user` injects a HalfDuplexUser in place of the LLM-backed simulator. It exists so the
    driver can be tested END TO END with no network: the Orchestrator, the dialogue view, the
    tool-call channel, the env-call harvest, upstream's grader and all seven output files are
    exercised, and only the customer's choice of words is scripted. A driver whose only test
    needs a VPN is a driver that is tested when the VPN is up.
    """
    from pi_run.manifest import build_manifest, manifest_to_dict, template_id_of
    from pi_run.worker import (
        STATUS,
        UnitTimeout,
        _billed,
        _Watchdog,
        _write_jsonl,
        assert_firewall,
        call_rows,
        collect_pins,
        evidence_rows,
        ledger_rows,
        load_suite,
        questions_hash,
        read_status,
        turn_rows,
        unit_timeout_s,
        write_atomic,
    )
    from pinq_expt import arms as arm_table
    from pinq_expt.components import prompt_hashes_of

    assert_firewall()
    started = time.time()
    t0 = time.perf_counter()

    suite = load_suite(spec.suite_id, spec.corpus_dir)
    arm = arm_table.get(spec.arm_id)
    cap = spec.budget_cap if spec.budget_cap is not None else arm.budget_cap
    # REFUSED HERE, BEFORE A CENT IS SPENT, rather than at the first use. An unknown name is a
    # configuration fact about the whole campaign, and discovering it after the dialogue would
    # cost a unit to learn what a string comparison already knows.
    protocol = str(getattr(spec, "protocol", "") or "")
    protocol_caps(protocol)

    # ONE ASK ledger for the whole dialogue. See the module docstring, decision 3.
    ledger = BudgetLedger(cap=cap)
    # AND ONE TOOL BUDGET BESIDE IT, of the same size and on its own counter (RULES amendment 7),
    # enforced where the tool call executes (decision 4). Built here for every arm alike; absent
    # only where tool calls were never budgeted. The exempt (GENERIC) tools are read here, before
    # a cent is spent: a missing map raises.
    charged_suite = env_calls_are_charged(suite)
    exempt = budget_exempt_tools(suite.domain) if charged_suite else frozenset()
    gate = BudgetGate(ledger, cap=cap, exempt=exempt) if charged_suite else None
    budget_fields: dict[str, Any] = {
        "budget_enforcement": BUDGET_ENFORCEMENT if gate is not None else TOOLS_UNCHARGED,
        # The ASK budget (the ledger's cap), and the TOOL budget (the gate's). Equal by the
        # amendment, recorded apart so neither has to be inferred from the other.
        "budget_cap": cap,
        "tool_call_cap": gate.cap if gate is not None else None,
        # WHICH TOOLS THE RULE LEFT FREE, as read from the map at this commit -- so a change to
        # the map is visible in the record rather than only in the code history.
        "budget_exempt_tools": sorted(exempt),
    }

    # THE CLIENT IS BUILT BEFORE THE RETRIEVER, and that ordering is the fix. It used to be built
    # after, so `suite.retriever(spec.task_id)` could only ever be handed None and every Ask on a
    # tool-backed suite resolved to nothing while the run reported success -- see
    # tests/test_retriever_is_wired.py for the measurement and the deleted warning.
    llm = None
    if not arm.llm_free:
        from pi_run.cache import CachingClient, DiskCache
        from pinq_adapters.llm.litellm_client import MeteredClient

        llm = CachingClient(
            MeteredClient(ledger, temperature=0.0), DiskCache(spec.cache_root), ledger=ledger
        )
    retriever = _wire_retriever(
        suite, spec.task_id, llm=llm, llm_free=arm.llm_free, seed=spec.seed, recorder=ledger
    )

    def _fail(exc: BaseException, status_name: str = "error") -> dict[str, Any]:
        return {
            "run_id": "",
            "key": list(spec.key),
            "suite_id": spec.suite_id,
            "task_id": spec.task_id,
            "arm_id": spec.arm_id,
            "seed": spec.seed,
            "status": status_name,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at": started,
            "wall_ms": int((time.perf_counter() - t0) * 1000),
            "code_version": spec.code_version,
            "dirty": spec.dirty,
            "pilot": spec.pilot,
            "usd_billed": _billed(ledger),
        }

    def build_parts() -> dict[str, Any]:
        """Fresh components, one shared ledger. Components are rebuilt per user message so the
        Inquirer's own turn counter and any per-rollout memo start clean; the ledger is not,
        because the budget is the task's."""
        c = arm_table.build(
            arm,
            llm=llm,
            retriever=retriever,
            recorder=ledger,
            questions=spec.questions or None,
            ask_counts=spec.ask_counts or None,
            tools=suite.tool_schemas(spec.task_id),
            unit_chars=TAU2_EVIDENCE_CHARS,
        )
        # Components are rebuilt per user message, so the variant has to be re-applied here or
        # the first message would run the variant and every later one the shipped template.
        apply_prompt_variant(c.inquirer, str(getattr(spec, "prompt_variant", "") or ""))
        return {
            "inquirer": c.inquirer,
            "drafter": c.drafter,
            "answerer": c.answerer,
            "retriever": retriever,
            "ledger": ledger,
        }

    try:
        components = arm_table.build(
            arm,
            llm=llm,
            retriever=retriever,
            recorder=ledger,
            questions=spec.questions or None,
            ask_counts=spec.ask_counts or None,
            tools=suite.tool_schemas(spec.task_id),
            unit_chars=TAU2_EVIDENCE_CHARS,
        )
        prompt_variant_id = apply_prompt_variant(
            components.inquirer, str(getattr(spec, "prompt_variant", "") or "")
        )
    except Exception as exc:  # noqa: BLE001 - a configuration fact about THIS unit
        return _fail(exc)

    view = dialogue_view(suite, spec.task_id, ())
    manifest = build_manifest(
        suite_id=spec.suite_id,
        task_id=spec.task_id,
        arm_id=spec.arm_id,
        policy_id=str(getattr(components.inquirer, "policy_id", spec.arm_id)),
        seed=spec.seed,
        corpus_hash=suite.corpus_hash,
        # tau2 is self-sourced: there is no data/corpora/tau2/<hash>/ for `load_questions` to
        # open, and the corpus is an upstream checkout. Empty means "unknown", which the
        # gold/corpus mismatch guard treats as allowed rather than as a disagreement.
        corpus_dir="",
        k=spec.k,
        questions_hash=questions_hash(spec, components.inquirer),
        grid_sha256=spec.grid_sha256,
        grid_name=spec.grid_name,
        budget_cap=cap,
        max_turns=spec.max_turns,
        word_cap=view.word_cap,
        code_version=spec.code_version,
        dirty=spec.dirty,
        # This driver is a hand copy of `worker.run_unit` and four fields were once
        # silently dropped here. Every run in this project is tau2, so a provenance
        # field stamped only in run_unit is stamped nowhere that matters.
        dirty_files=spec.dirty_files,
        # WITHOUT THESE A FORK HAS NO IDENTITY. They are in SEMANTIC_FIELDS, but semantic_hash
        # reads the manifest's own dict -- so a runner that does not pass them leaves them
        # absent, the omit-when-None rule drops them, and two cut points of one task collapse
        # onto one run_id and one run directory. The fields also never reach disk, so a fork
        # is indistinguishable from a fresh rollout afterwards.
        foreign_trace_sha=getattr(spec, "foreign_trace_sha", None),
        foreign_prefix_k=getattr(spec, "foreign_prefix_k", None),
        template_id=template_id_of(suite, spec.task_id),
        prompt_hashes={
            "answerer": str(getattr(components.answerer, "prompt_hash", "")),
            **prompt_hashes_of(components.inquirer, components.drafter, components.answerer),
        },
        # WHICH PROMPT SET PRODUCED THE RUN. `worker.run_unit` has always stamped this and this
        # driver -- a hand copy of it -- never did, so every tau2 run on disk reads "v1" by
        # default rather than by measurement. `prompt_hashes` above already carries the
        # variant's template into run identity; this is the readable label to GROUP BY.
        prompt_variant_id=prompt_variant_id,
        upstream_pins=upstream_pins_for(
            protocol,
            user_sim=os.environ.get("PI_MODEL_USERSIM", ""),
            suite_version=str(getattr(suite, "suite_version", "")),
            agent_model=_stock_model() if spec.arm_id == STOCK_ARM else "",
        ),
        pins=collect_pins(llm, (components.inquirer, components.drafter, components.answerer))
        if llm is not None
        else {},
        pilot_flag=spec.pilot,
        exploratory=spec.exploratory,
        gold_exposed=arm.requires_gold,
        concurrency=spec.concurrency,
    )
    run_dir = Path(spec.runs_root) / manifest.run_id
    status_path = run_dir / STATUS

    prior = read_status(status_path) if (spec.resume and status_path.exists()) else None
    if prior is not None and str(prior.get("status")) != "ok":
        prior = None
    if prior is not None:
        return {
            **prior,
            "key": list(spec.key),
            "status": "resumed",
            "prior_status": prior.get("status"),
            "run_id": manifest.run_id,
        }

    status: dict[str, Any] = {
        "run_id": manifest.run_id,
        "key": list(spec.key),
        "suite_id": spec.suite_id,
        "task_id": spec.task_id,
        "arm_id": spec.arm_id,
        "seed": spec.seed,
        "started_at": started,
        "code_version": spec.code_version,
        "dirty": spec.dirty,
        "pilot": spec.pilot,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(run_dir / "manifest.json", json.dumps(manifest_to_dict(manifest), indent=2))

    sink = io.StringIO()  # tau2's tools print customer-facing text to stdout
    # Bound before the try so the failure branch can read the first rollout when the unit died
    # AFTER the simulation returned; it stays None when the simulation itself is what raised.
    state: Any = None
    try:
        with _Watchdog(unit_timeout_s(spec.timeout_s)), contextlib.redirect_stdout(sink):
            # Empty for every ordinary rollout; a named trace that is missing raises rather
            # than silently running unforked and landing in the fork table anyway.
            prefix = load_prefix(spec)
            sim, state, task_used = _simulate(
                suite, spec, build_parts, user=user, prefix=prefix, protocol=protocol, gate=gate
            )

        # WHAT THE ENVIRONMENT EXECUTED, SEPARATED FROM WHAT IT WAS NEVER ASKED TO. The full
        # transcript -- refusals included -- is what the dialogue was, and it stays the input to
        # `transcript_digest` and to the user simulator's bill. Everything that treats a tool call
        # as having HAPPENED reads `executed`: the env-call log, the evidence, and both graders,
        # which would otherwise perform a refused write on the predicted world (see
        # `split_refused`). A unit that never met the cap refuses nothing, and then `executed`
        # is the transcript and `graded` is `sim` itself -- graded exactly as before.
        executed, refused = split_refused(
            sim.messages or (), gate.refused if gate is not None else {}
        )
        graded = sim.model_copy(update={"messages": executed}) if refused else sim
        traj = merge_trajectories(state.trajectories, view=view, ledger=ledger)
        env_calls = env_calls_from(executed)
        # EVIDENCE FROM TOOL CALLS, on suites that have no retriever. Banking searches documents
        # through a retriever, so its evidence arrives through `run_loop` and is already metered.
        # Retail has no search surface at all -- 15 typed key-addressed tools -- so the records it
        # read are recoverable only from what its calls ASKED FOR, and nothing was charging or
        # attaching them: `retrieved_uids` came back empty and every gold uid was unsatisfiable.
        # NOT CHARGED HERE when the gate ran: tool calls are on the gate's own counter and never
        # reach the ledger (RULES amendment 7).
        traj = _attach_env_evidence(traj, suite, ledger, env_calls, charge=gate is None)
        # THE GATE, CHECKED FROM THE TRANSCRIPT. `env_call_census` reads what the environment
        # answered, not the gate's counters; see its docstring. Two ways it can disagree, and
        # both are RECORDED, never raised -- the dialogue is over and its money is spent:
        #   * `gate_census_agrees` False: the gate counted a different number of calls than the
        #     transcript says executed and were countable (a leak, or a double count);
        #   * `post_hoc_budget_overrun`, the same ledger row the post-hoc meter used to write:
        #     the transcript shows more live ok non-GENERIC calls than `tool_call_cap`, OR the
        #     ask ledger holds more than `budget_cap` -- either budget exceeded, each on its own.
        # On a correct gate neither can happen; each is pinned by a deliberately broken gate in
        # tests/test_tau2_budget_gate.py so that "never fired" is not the same as "cannot fire".
        init = getattr(task_used, "initial_state", None)
        census = env_call_census(
            sim.messages or (),
            boundary=len(getattr(init, "message_history", None) or ()),
            refused=gate.refused if gate is not None else {},
            exempt=exempt,
        )
        gate_census_agrees: bool | None = None
        if gate is not None:
            gate_census_agrees = gate.n_charged == census["n_live_env_calls_nongeneric_ok"]
            tools_over = census["n_live_env_calls_nongeneric_ok"] > gate.cap
            asks_over = ledger.spent.get(HARD_CURRENCY, 0.0) > cap
            if tools_over or asks_over:
                ledger.record("post_hoc_budget_overrun", 1.0)
        # THE TASK THE DIALOGUE ACTUALLY RAN, which on a fork is not the suite's. The evaluator
        # seeds the GOLD environment from `task.initial_state` and applies every gold action on
        # top, so grading the original while the Orchestrator ran the fork starts the two worlds
        # at different points and every continuation scores 0 for a reason that is not the
        # policy's.
        native, reward_error = _reward_of(graded, suite, spec.task_id, task=task_used)
        if protocol == UPSTREAM_PROTOCOL:
            # BESIDE the harness reward, never instead of it. The two answer different
            # questions -- upstream's is the benchmark's published quantity, ours is the
            # DB-state check with no termination gate -- and a campaign that recorded only one
            # of them could not say by how much they differ, which is the first thing a
            # reviewer comparing this table with a leaderboard will ask.
            native = {
                **native,
                **grade_upstream(
                    graded,
                    task_used,
                    domain=suite.domain,
                    env_kwargs=suite.env_kwargs(spec.task_id),
                ),
            }
        traj = _with_outcome(traj, state, env_calls, native, sim, suite.environment(spec.task_id))
    except Exception as exc:  # noqa: BLE001 - one task's failure is data, not a sweep abort
        # BUDGET EXHAUSTION IS A HARVEST-TIME FAILURE TOO, NOT ONLY A SIMULATE-TIME ONE. Retail
        # has no retriever, so its 15 typed tools are charged to the ledger only in
        # `_attach_env_evidence` (`meter_env_calls`), which used to sit AFTER this try/except
        # rather than inside it. Measured live: 3 of 3 retail units raised BudgetExceeded here
        # after 13-16 turns, and because nothing below this point wrote before propagating,
        # $30-47 of real spend left no status.json and no ledger.jsonl at all. The harvest no
        # longer raises at all -- it records the spend and reports the overrun, because raising
        # after the dialogue is over deletes a finished run rather than preventing a spend -- so
        # this branch now catches timeouts, grader errors and simulate-time failures, and the
        # reason it must still write is unchanged. `ledger` -- not
        # `traj`, which may not exist yet if `_simulate` itself is what raised -- is what a
        # killed unit's record must be built from: it is mutated in place by every successful
        # charge regardless of which line above raised.
        status.update(
            {
                "status": "timeout" if isinstance(exc, UnitTimeout) else "error",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": time.time(),
                "wall_ms": int((time.perf_counter() - t0) * 1000),
                "usd_billed": _billed(ledger),
                # SAME FIELD NAMES THE "ok" BRANCH WRITES BELOW. `pi_run.compact._run_row`
                # reads `usage.usd` and `spent.retrieval_calls`, never `usd_billed`, to build
                # the cost table -- so without these a killed unit compacts as a free run,
                # indistinguishable from one that never spent a cent.
                "usage": ledger.usage.as_dict(),
                "spent": dict(sorted(ledger.spent.items())),
                "unique_docs": ledger.unique_docs,
                # The regime and the refusals so far: a killed gated unit spent under the cap
                # and must say so as plainly as a finished one.
                **budget_fields,
                "n_refused_budget": gate.n_refused if gate is not None else 0,
                "n_gate_charged": gate.n_charged if gate is not None else 0,
                **dialogue_secondaries(state, gate),
            }
        )
        # Belt and suspenders, matching worker.run_unit's own failure branch: run_dir was
        # already created above, but a watchdog-killed unit re-asserts it rather than trust it.
        run_dir.mkdir(parents=True, exist_ok=True)
        # EVERY ROW ALREADY EARNED, flushed before the sentinel. `BudgetLedger.check` rejects
        # the overflowing call BEFORE `_bump` records it, so `ledger.rows` holds exactly the
        # calls the provider billed -- no more -- regardless of where in the block above the
        # exception came from.
        _write_jsonl(run_dir / "ledger.jsonl", ledger_rows(ledger))
        write_atomic(status_path, json.dumps(status, indent=2, sort_keys=True))
        return status

    from pi_run.worker import ReconcileError, reconcile

    rec = reconcile(traj, ledger)
    if not rec["docs_ok"]:
        # Identical to the flat path: a Drafter that retrieved privately and folded the result
        # into its own Evidence grows the trajectory's document set and not the ledger's.
        # Invisible in tokens, invisible in retrieval_calls, and fatal to a budget-parity
        # claim -- so it takes the sweep down rather than contributing a row.
        raise ReconcileError(
            f"{manifest.run_id}: ledger saw {rec['ledger_unique_docs']} unique docs but the "
            f"trajectory carries {rec['evidence_unique_docs']}. Unmetered nested retrieval."
        )

    _write_jsonl(run_dir / "turns.jsonl", turn_rows(traj))
    _write_jsonl(run_dir / "calls.jsonl", call_rows(traj))
    _write_jsonl(run_dir / "ledger.jsonl", ledger_rows(ledger))
    _write_jsonl(run_dir / "evidence.jsonl", evidence_rows(traj))
    from pi_run.worker import outcome_dict

    write_atomic(run_dir / "outcome.json", json.dumps(outcome_dict(traj), indent=2, sort_keys=True))

    # The user simulator's dollars are REAL money and are NOT the policy's spend. They stay out
    # of the ledger, so cross-arm token parity is not polluted by a harness cost that is
    # identical for every arm; they are added to usd_billed, so the campaign cap sees the
    # actual invoice rather than two thirds of it.
    # Priced against the pin THIS run recorded, not a fresh read of the environment: the
    # manifest is what the run actually used, and re-reading `PI_MODEL_USERSIM` here would
    # bill a rollout at whatever the env happened to say when the status was written.
    user_usd, user_unpriced = user_sim_usd(
        sim.messages or (), str(manifest.upstream_pins.get("user_sim") or "")
    )
    # THE STOCK ARM'S OWN DOLLARS, priced the same way and for the same reason. Its assistant
    # turns go through tau2's litellm client, so the ledger never sees them and `_billed`
    # returns 0 -- which would report the CONTROL as free, against treatment arms whose every
    # token is metered. Priced against the pin THIS run recorded, not a fresh env read. Empty
    # for every other arm, whose assistant turns are already in the ledger.
    agent_model = str(manifest.upstream_pins.get("agent_model") or "")
    agent_usd, agent_unpriced = (
        role_usd(sim.messages or (), agent_model, role="assistant") if agent_model else (0.0, 0)
    )
    agent_tok = role_tokens(sim.messages or (), role="assistant") if agent_model else {}
    status.update(
        {
            "status": "ok",
            "finished_at": time.time(),
            "wall_ms": int((time.perf_counter() - t0) * 1000),
            "n_turns": len(traj.turns),
            "n_asks": traj.n_asks,
            "n_calls": len(traj.calls),
            # Lost cache write races: calls we paid for whose answer was discarded for the
            # canonical one (see pi_run.cache). Nonzero is not an error -- it is the count of
            # places where this run and the cache would have disagreed and now do not.
            "cache_races": int(getattr(llm, "races", 0) or 0),
            "n_evidence": len(traj.evidence),
            "n_env_calls": len(env_calls),
            # Split by REQUESTOR, because upstream's max_errors counts both against the agent.
            # An agent error is the policy failing; a user error is the environment being
            # noisy, and conflating them turns a weak user simulator into a policy result.
            "n_errors_agent": sum(1 for c in env_calls if not c.ok and c.requestor == "assistant"),
            "n_errors_user": sum(1 for c in env_calls if not c.ok and c.requestor == "user"),
            "n_assistant_rollouts": len(state.trajectories),
            "n_plan_steps_rejected": state.rejected,
            "n_messages": len(sim.messages or ()),
            # THE ANTICIPATION ENDPOINT. Counted by role, because `n_messages` also counts
            # assistant and tool messages and therefore tracks the agent's tool-calling
            # verbosity rather than how much the USER had to say.
            #
            # This cannot be inflated by asking: the inner policy runs with
            # `allow_user_target=False`, so an ASK is charged as a wasted turn and never
            # reaches the transcript. Every user message here is the simulator speaking of
            # its own accord -- stating the task, or coming back because the answer did not
            # resolve the need.
            "n_user_turns": count_user_turns(sim.messages or ()),
            # HOW MUCH OF THAT WAS INHERITED. `fork_report.follow_ups` is
            # `n_user_turns - n_prefix_user_turns`, so without this a fork is charged for the
            # user turns of the dialogue it was handed -- on a k=34 retail prefix that is most
            # of the count, and it is charged identically to both arms, which flattens the
            # very contrast the protocol measures. 0 on every ordinary rollout, and a real 0
            # rather than a missing key (see compact.py).
            "n_prefix_user_turns": count_user_turns(prefix),
            # ON THE STATUS FILE TOO, not only the manifest: `report_forks.load_runs` reads
            # both, and a reader holding a status.json must be able to tell a continuation
            # from a full run without opening a second file.
            "foreign_trace_sha": getattr(spec, "foreign_trace_sha", None),
            "foreign_prefix_k": getattr(spec, "foreign_prefix_k", None),
            # Recorded beside the reward because a premature termination is a real and
            # arm-dependent event: max_errors counts failed tool calls, and the arms that
            # flail trip it most. The reward is still MEASURED (a truncated dialogue
            # did not reach the gold DB state, so 0.0 is what it earned) -- but an analyst
            # must be able to see how much of an arm ran out of dialogue rather than out
            # of ideas.
            "termination_reason": str(getattr(sim, "termination_reason", "")),
            "terminated_prematurely": str(getattr(sim, "termination_reason", "")).rsplit(".", 1)[-1]
            not in ("AGENT_STOP", "USER_STOP"),
            "stop_reason": traj.stop_reason,
            "usage": traj.usage.as_dict(),
            "usd_billed": round(_billed(ledger) + user_usd + agent_usd, 6),
            "user_sim_usd": round(user_usd, 6),
            # Nonzero means `user_sim_usd` is an UNDERCOUNT, not a cheap run. Recorded rather
            # than raised so the gap is auditable without costing the unit (see user_sim_usd).
            "user_sim_unpriced_msgs": user_unpriced,
            # 0.0 on every arm but the stock one, where it is the whole of the policy's spend.
            "stock_agent_usd": round(agent_usd, 6),
            "stock_agent_unpriced_msgs": agent_unpriced,
            "stock_agent_tokens": dict(agent_tok),
            "spent": dict(sorted(ledger.spent.items())),
            "unique_docs": ledger.unique_docs,
            # THE BUDGET REGIME AND WHAT IT REFUSED. `n_env_calls` above counts calls the
            # environment EXECUTED; a refusal is not one of them and is not an error either, so
            # it is counted here and nowhere else. `n_refused_budget` is the gate's own count and
            # `refused_budget_calls` is read back off the transcript by `split_refused` -- two
            # instruments, so a reader can check `len(refused_budget_calls)` against it.
            **budget_fields,
            "n_refused_budget": gate.n_refused if gate is not None else 0,
            "refused_budget_calls": refused,
            # THE LIVE TOOL SPEND, FROM THE TRANSCRIPT (see `env_call_census`), beside the gate's
            # own count. `n_env_calls` above is the WHOLE transcript, a fork's inherited prefix
            # included, so it can exceed the cap on a correct gate; `tool_call_cap` binds on
            # `n_live_env_calls_nongeneric_ok`, which must equal `n_gate_charged`. The ASK budget
            # is `spent.retrieval_calls` alone (amendment 7): `n_asks` less
            # `spent.rejected_user_asks`, plus any query-expansion sub-retrievals (which charge
            # the ledger with no turn of their own), and never a tool call.
            **census,
            "n_gate_charged": gate.n_charged if gate is not None else 0,
            "gate_census_agrees": gate_census_agrees,
            # RULES amendment 4: the FIRST rollout's stop and asks (`stop_reason` below is the
            # last rollout's) and the ledger when the first live non-GENERIC call executed.
            **dialogue_secondaries(state, gate),
            "native": dict(native),
            "reward_error": reward_error,
            # THE SAME reconciliation the flat path runs, reported the same way. This was a
            # bare bool with no `reconcile` sub-dict, and `pi_run.compact` reads
            # `status["reconcile"]` -- so every tau2 row compacted with
            # reconciled_tokens=False and reconciled_docs=False regardless of the truth, and
            # `docs_ok` (the detector for an unmetered nested retrieval, the only thing that
            # catches a Drafter retrieving privately) was never computed on tau2 at all.
            "reconciled": rec["tokens_ok"] and rec["docs_ok"],
            "reconcile": rec,
        }
    )
    write_atomic(status_path, json.dumps(status, indent=2, sort_keys=True))
    return status


def guard_silent_user_turn(user: Any) -> Any:
    """Make the user simulator's own empty completion end the episode, not crash it.

    `UserSimulator._generate_next_message` (tau2, vendored, `tau2/user/user_simulator.py`)
    hands back whatever `generate(...)` produced, unchecked: `content = assistant_message.content`,
    wrapped into a `UserMessage` with no fallback if that content is empty and no tool call
    came back either. `Orchestrator.step()`'s AGENT/ENV -> USER branch then calls
    `user_msg.validate()` UNCONDITIONALLY (`orchestrator.py:841-844`), before it ever asks
    `UserSimulator.is_stop`. That ordering assumes there is always something -- content or a
    tool call -- for the customer to hand back on every call, including the one right after
    its OWN tool call resolves: branch "AGENT/USER -> ENV" ends with
    `self.to_role = self.from_role; self.from_role = Role.ENV` (`orchestrator.py:891-892`),
    which for a user-initiated tool call hands control right back to Role.USER -- our agent
    is never invoked in between. Two banking tasks confirm this is reachable, not
    hypothetical: `task_007.json` and `task_010.json` each script the customer to resolve its
    own need with its OWN tool call (`apply_for_credit_card`, `submit_referral`) and then say,
    verbatim, "there is no need to respond to the agent." Immediately after that call
    executes, the customer's model is asked to speak again with nothing left to say, and the
    completion that comes back has neither text nor a tool call.

    Measured directly against the installed tau2: a `UserMessage(role="user", content=None)`
    -- exactly the shape `__str__` prints as `"UserMessage\nis_final_chunk: True"`, matching
    the two runs that actually crashed -- fails `.validate()` with precisely the observed
    error. `is_stop` is checked too late in that branch to save it, and once
    `generate_next_message` has returned, nothing our own agent does (an `is_stop` override on
    `PinqDriverAgent`, say) can run before it, because the crash never reaches our agent's
    role at all.

    So the correction sits on the way OUT of the call, at the only point that is reachable:
    `user.generate_next_message` is wrapped so that when (and only when) its result would
    fail `validate()` -- the exact predicate `validate()` itself uses, `has_content() or
    is_tool_call()`, not a looser guess -- the empty `UserMessage` is replaced with tau2's own
    end-of-conversation sentinel (`STOP`, `"###STOP###"`, the same marker
    `UserSimulator.is_stop` already recognizes at `user_simulator.py:183-196`). This is not a
    fabricated reply: no dialogue is invented, and nothing ever responds to it --
    `Orchestrator.run()`'s loop (`orchestrator.py:279-281`) checks `self.done` before every
    `step()`, and `is_stop` returning True on this sentinel sets `self.done = True` inside the
    very step that produced it, so the substituted message is the last thing either role
    says. A well-formed message -- real content, or a real tool call -- is returned
    completely unchanged.

    `_reward_of` (below) does not gate on `termination_reason` -- it always replays the
    transcript through `EnvironmentEvaluator.calculate_reward` -- so this substitution does
    not change how the unit is graded. It only stops the crash, on a transcript that, by
    construction, already contains whatever the customer's own required action was (the
    empty completion is the customer's SECOND consecutive turn; its tool call, if the task
    requires one, is the reason it had nothing left to add on the first).

    A `user` with no `generate_next_message` at all is returned completely unchanged, not
    wrapped. `tests/test_tau2_fork_wiring.py` deliberately hands `_simulate` a bare
    `object()` for `user`: its own module docstring says the `Orchestrator` is faked
    specifically so those tests exercise task/prefix wiring "before `orch.run()` is
    reached" without needing a real, callable user -- and before this guard existed,
    `_simulate` never touched `user` ahead of constructing that (faked) `Orchestrator`
    either. Wrapping unconditionally would newly demand an attribute those doubles were
    never designed to have, for a codepath that -- real or faked -- runs no dialogue at
    all. Declining changes no real risk: every `user` that will actually reach a live
    `Orchestrator.step()` already has `generate_next_message`, because `step()` itself
    calls it.
    """
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator_base import STOP

    if not hasattr(user, "generate_next_message"):
        return user

    original = user.generate_next_message

    def generate_next_message(message: Any, state: Any):
        user_msg, new_state = original(message, state)
        if user_msg.has_content() or user_msg.is_tool_call():
            return user_msg, new_state
        return UserMessage(role="user", content=STOP), new_state

    user.generate_next_message = generate_next_message
    return user


def _simulate(
    suite: Any,
    spec: Any,
    build_parts,
    *,
    user: Any = None,
    prefix=None,
    protocol: str = "",
    gate: BudgetGate | None = None,
):
    """Construct env + user + agent + Orchestrator and run to termination.

    Returns `(sim, agent_state, task)`. THE TASK IS RETURNED because a fork runs against a
    modified one, and `_reward_of` must grade against that same object -- re-fetching the
    original there would seed the gold environment from a world the policy was never in.

    `protocol` selects the step and error caps only. It does NOT select the agent: that is the
    ARM's job, so a stock run and an augmented run can be compared under either protocol and
    "which loop ran" never has to be inferred from a cap.

    `gate` is installed on the Orchestrator whichever agent was built, which is what makes the
    budget the same rule for every arm. None only on a suite whose tool calls are not charged.
    """
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_user

    # THE SAME Environment the retriever and the tool schemas came from. A rollout is one
    # world: building a second one for the Orchestrator would let the policy read from world A
    # while the graded mutations landed in world B.
    env = suite.environment(spec.task_id)
    task = suite.tau2_task_object(spec.task_id)
    if prefix:
        # The user simulator is seeded from this history too, and it holds no other
        # state, so a foreign prefix reads to it as its own past.
        task = forked_task(task, prefix)

    if user is None:
        user = build_user("user_simulator", env, task, llm=_user_model(), llm_args=_user_llm_args())
    # UNCONDITIONAL: a passed-in `user` (fork replay, tests) is just as able to run its
    # underlying model dry as a freshly built one -- the empty completion is a property of
    # the task script and the model, not of how this function obtained the object.
    user = guard_silent_user_turn(user)
    if str(getattr(spec, "arm_id", "")) == STOCK_ARM:
        # UPSTREAM'S OWN LOOP, on the same environment and the same customer. It is pinned to
        # the DRAFTER's role model deliberately: the augmented arms answer with that model, so
        # pinning the stock agent anywhere else would make "stock versus augmented" a model
        # comparison wearing an architecture comparison's name.
        agent = build_stock_agent(env, model=_stock_model(), llm_args=_user_llm_args())
    else:
        agent_cls = make_driver_agent_class()
        agent = agent_cls(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            suite=suite,
            task_id=spec.task_id,
            build_parts=build_parts,
            max_turns=spec.max_turns,
            k=spec.k,
            seed=spec.seed,
            branch_at=spec.branch_turn_idx,
            branch_seed=spec.branch_seed,
        )
    steps, errors = protocol_caps(protocol)
    orch = Orchestrator(
        # OFF THE SUITE, for the same reason `dialogue_view` reads `suite.suite_id` off it:
        # this driver serves both tau2 domains, so naming one of them here stamps retail
        # simulations as banking.
        domain=suite.domain,
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=steps,
        max_errors=errors,
        seed=spec.seed,
        solo_mode=False,
    )
    if gate is not None:
        gate.install(orch)
    sim = orch.run()
    state = orch.agent_state
    if str(getattr(spec, "arm_id", "")) == STOCK_ARM:
        # UPSTREAM'S AGENT STATE IS AN `LLMAgentState`, NOT A `DriverState`. The harvest reads
        # `state.trajectories` and `state.rejected`, neither of which exists there, so the
        # stock arm is handed an EMPTY DriverState -- which is the true value, not a
        # placeholder: this arm ran no pinq rollout, planned no tool steps of ours and had
        # none rejected. `merge_trajectories(())` then yields a trajectory with zero turns,
        # which is exactly what "no Inquirer, no Drafter, no Answerer" looks like.
        state = DriverState(messages=list(sim.messages or ()))
    return sim, state, task


def _reward_of(sim: Any, suite: Any, tid: str, *, task: Any = None) -> tuple[dict[str, float], str]:
    """Grade with UPSTREAM's EnvironmentEvaluator, over the transcript the Orchestrator produced.

    NOT `evaluate_simulation`, AND THIS IS THE WHOLE POINT. Its first statement is

        if simulation.termination_reason not in {AGENT_STOP, USER_STOP}:
            return RewardInfo(reward=0.0, reward_basis=None, ...)

    -- a RewardInfo with **no db_check**. `attach_reward` writes `tau_reward` only when a
    db_check is present, so a dialogue that hit `max_steps` or `max_errors` produced
    `native = {"reward": 0.0}` with no `tau_reward` at all, and `reward_error = ""` so the
    absence read as a clean grade. Downstream, `environment.tau_reward` returns NaN for a
    missing key and `score.py` emits the metric only when measured, so the run contributed NO
    ROW to the tau2 primary endpoint instead of the 0.0 it had earned.

    THE BIAS HAS A DIRECTION AND IT FLATTERS THE WORSE ARM. Premature termination is not
    random: `max_errors` counts failed tool calls, and 730 of tau2's 853 assistant gold
    actions are unlock/call pairs where a wrong guess errors. An arm that flails produces more
    premature terminations, so MORE of its failures are deleted from the denominator and its
    reported mean RISES with its failure rate. `report.estimate` pairs on the intersection of
    two arms' task sets, so the task vanishes from both arms of the contrast.

    Measured before the fix: the same policy on the same task scored `tau_reward = 0.0` when
    the customer said ###STOP### and contributed no row at all when the dialogue ran to
    max_steps.

    `EnvironmentEvaluator.calculate_reward` has no such gate -- it replays the transcript into
    a fresh environment and always returns a db_check -- and it is exactly what the flat path
    calls, which is what makes the two paths report through the same gate rather than merely
    claiming to. A truncated dialogue simply did not reach the gold DB state, so 0.0 is a
    MEASUREMENT of this rollout and not a default.
    """
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.runner import build_environment

    from pinq_adapters.tau2.actuator import Tau2Actuator

    # THE TASK THE DIALOGUE RAN, when the caller has one. On a fork that is the prefixed
    # copy: the evaluator seeds the GOLD environment from `task.initial_state` and then
    # applies every gold action, so grading the suite's original against a transcript
    # that began mid-dialogue compares the continuation with a world it was never in.
    task = task if task is not None else suite.tau2_task_object(tid)
    crit = getattr(task, "evaluation_criteria", None)
    if crit is None:
        return {}, "task has no evaluation_criteria"

    kwargs = dict(suite.env_kwargs(tid))
    holder = Tau2Actuator(
        suite.environment(tid),
        task=suite.task_record(tid),
        domain=suite.domain,
        env_kwargs=kwargs,
    )

    # THE DOMAIN IS THE SUITE'S, NOT THIS MODULE'S. Both tau2 suites route here, and this
    # constructor rebuilt banking for every one of them: a retail transcript was replayed
    # into the BANKING database and graded against a gold state from a world it never
    # touched. With no OPENAI_API_KEY the banking build raises while constructing its
    # document-embedding index, so `tau_reward` was merely absent -- measured, retail scored
    # `native = {}` with `reward_error = "OpenAIError: Missing credentials"` on every task.
    # With a key set it would have returned a number, and the number would have been wrong.
    def constructor(solo_mode: bool = False, **kw: Any) -> Any:
        return build_environment(suite.domain, solo_mode=solo_mode, env_kwargs=kw or kwargs)

    try:
        info = EnvironmentEvaluator.calculate_reward(
            constructor,
            task,
            list(sim.messages or ()),
            env_kwargs=kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a failed grade is data, not a sweep abort
        return {}, f"{type(exc).__name__}: {exc}"
    holder._reward_done = True  # attach only; never let it re-grade from its own empty log
    holder.attach_reward(info, reward_basis=getattr(crit, "reward_basis", ()) or ())
    return dict(holder.native()), ""


def _attach_env_evidence(
    traj: Trajectory, suite: Any, ledger: Any, env_calls, *, charge: bool = True
) -> Trajectory:
    """Charge this run's tool calls and fold the records they read into the trajectory.

    NO-OP ON A SUITE WITH A RETRIEVER. Banking's evidence already arrives through `run_loop`,
    already metered; charging it again here would double-bill it and break `reconcile`. The
    suite advertises its own capability via `uids_for_calls`, so the branch is a property of the
    adapter rather than a suite-id check that a third domain would have to be added to.

    `charge=False` ON A GATED UNIT: `BudgetGate` counted each accepted non-GENERIC call on its
    own tool budget as it executed, and this is then evidence only. See `meter_env_calls`.

    The units carry no text. `EvidenceUnit` needs one, and a retail record's content is a
    customer record the runner must not persist -- so the uid and doc_id are real and the text
    is empty. Matching is over uids, which is all `gold_ev_uids` compares.
    """
    from dataclasses import replace as _replace

    if not env_calls_are_charged(suite):
        return traj
    index = suite.index

    from pinq.types import Evidence, EvidenceUnit

    uids, overrun = meter_env_calls(
        index, ledger, [as_dict_call(c) for c in env_calls], charge=charge
    )
    if overrun:
        # REPORTED ON THE RUN, NOT THROWN. The dialogue is over; raising here would delete a
        # completed measurement, and would delete it preferentially from the arms that call the
        # most tools. Recorded, so the accounting gap is visible in the run's own ledger.
        ledger.record("post_hoc_budget_overrun", 1.0)
    if not uids:
        return traj
    by_uid = {u: d for d, u in index.uids.items()}
    units = tuple(
        EvidenceUnit(
            uid=u,
            corpus_id=suite.corpus_id,
            doc_id=by_uid.get(u, ""),
            span="",
            title=index.titles.get(by_uid.get(u, ""), ""),
            text="",
        )
        for u in uids
    )
    # THE TRAJECTORY'S UNIT WINS A SHARED uid: it carries the text the Drafter actually read,
    # and a text-less env unit of the same uid used to replace it (the later key wins), so
    # evidence.jsonl reported n_chars 0 for text the Drafter saw. Same uid, same doc_id -- the
    # index inverts totally -- so the uid set, the doc set and `reconcile` do not move.
    merged = tuple({u.uid: u for u in (*units, *traj.evidence.units)}.values())
    return _replace(traj, evidence=Evidence(units=merged))


def as_dict_call(call: Any) -> dict:
    """`EnvCall` -> the plain mapping `meter_env_calls` reads. Kept separate so the metering
    function stays testable without constructing dataclasses."""
    if isinstance(call, Mapping):
        return dict(call)
    return {
        "tool_name": getattr(call, "tool_name", ""),
        "kwargs_json": getattr(call, "kwargs_json", "{}"),
        "ok": bool(getattr(call, "ok", True)),
    }


def _with_outcome(traj: Trajectory, state: Any, env_calls, native, sim, env) -> Trajectory:
    """`env` is the LIVE Environment, passed in by the caller.

    It used to be `getattr(sim, "environment", None)`, and `SimulationRun` has no such field --
    so BOTH final hashes were the empty string on every tau2 run ever written. They are the
    substrate for recomputing the reward offline, and an empty hash compares equal to another
    empty hash, so "the DB matched" would be trivially true between any two runs that both
    recorded nothing. An empty hash is now a hard failure rather than a stored blank.
    """
    from dataclasses import replace as _replace

    last = state.trajectories[-1] if state.trajectories else None
    answer = last.outcome.answer if last is not None else None
    db = str(env.get_db_hash() or "") if env is not None else ""
    user_db = str(env.get_user_db_hash() or "") if env is not None else ""
    if not db:
        raise Tau2DriverError(
            "the tau2 environment produced no DB hash. That value is the reward substrate, and "
            "an empty one compares equal to every other empty one."
        )
    return _replace(
        traj,
        outcome=Outcome(
            answer=answer,
            env_calls=tuple(env_calls),
            env_final_hashes={"db": db, "user_db": user_db},
            native=dict(native),
            transcript_digest=h(
                "tau2_transcript", canon([_msg_digest(m) for m in sim.messages or ()])
            ),
        ),
    )


def _msg_digest(m: Any) -> Mapping[str, Any]:
    """A message reduced to what may safely be stored: roles, tool names, argument hashes.

    Not the content. A tau2 transcript is a customer-service dialogue containing account
    numbers and balances, and a run artifact is exactly the wrong place for it.
    """
    return {
        "role": str(getattr(m, "role", "")),
        "tools": [str(tc.name) for tc in (getattr(m, "tool_calls", None) or ())],
        "content_sha": h("c", str(getattr(m, "content", "") or ""))[:16],
    }
