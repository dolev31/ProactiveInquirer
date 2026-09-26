"""The bridge from a Draft's tool_plan to a stateful tau2 environment.

WHY `domain` IS A CONSTRUCTOR ARGUMENT AND NOT THIS MODULE'S CONSTANT
    There are two tau2 suites. Every caller inside `src/` passes the SUITE's domain
    (`Tau2Suite.domain` / `Tau2RetailSuite.domain`); the banking default survives only for
    the fixture-driven tests that never grade. It is read in exactly one place --
    `_compute_reward`'s constructor -- and getting it wrong there grades a rollout against a
    database it never touched, which is what happened to every retail run.

WHY final_hashes() RETURNS TWO HASHES
    tau2's reward replays the agent trajectory into a predicted environment, replays
    `evaluation_criteria.actions` into a gold environment, and compares BOTH
    `get_db_hash()` and `get_user_db_hash()`. A single hash would silently score every
    user-side mutation as correct, so the Outcome carries the pair under stable keys.

WHY THE UNLOCK PAIR IS TRACKED EXPLICITLY
    `call_discoverable_agent_tool` fails unless `unlock_discoverable_agent_tool` succeeded
    first, and that only succeeds once the naming document was read. Recording
    `unlock_required`/`unlock_satisfied` per EnvCall is what turns "did the policy cross
    the prerequisite edge?" into a groupby over stored rows instead of a re-run.

WHY THE REWARD IS COMPUTED BY UPSTREAM AND NOT HERE
    `tau_reward` is the paper's tau2 primary endpoint, so the ONE thing that must not happen
    is our own re-implementation of it drifting from the benchmark's. `native()` therefore
    hands a synthesized trajectory to `tau2.evaluator.evaluator_env.EnvironmentEvaluator`
    and reports what IT says. Everything in this file is plumbing to make that call
    well-formed; the definition of the number lives upstream.

    The replay is over a FRESH environment built from the same kwargs, not over the live one
    this actuator mutated. That is upstream's semantics, and it has a property worth having:
    the reward is a pure function of the stored EnvCall log, so it is recomputable offline
    from `env_calls.parquet` without re-running a single model call.

WHAT IS DELIBERATELY NOT SCORED
    9 of the 97 tasks carry `reward_basis == [ACTION]` and one carries `[DB, NL_ASSERTION]`.
    `EnvironmentEvaluator` only knows DB and ENV_ASSERTION, and for an ACTION-basis task it
    returns `reward=1.0` — a free pass, not a measurement. `tau_reward` is therefore emitted
    ONLY when DB is in the task's reward_basis, and is the DB check itself (agent-DB hash AND
    user-DB hash equality) rather than the evaluator's product. A missing measurement must
    look missing.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pinq.ids import NonCanonical, canon, h
from pinq.types import EnvCall

from ._probe import DOMAIN

UNLOCK_TOOL = "unlock_discoverable_agent_tool"
CALL_TOOL = "call_discoverable_agent_tool"

# Tools that gate on a prior unlock. Anything else is a plain call.
_GATED = frozenset({CALL_TOOL})

# The metric name the tau2 primary endpoint is declared on. Spelled once, here, because it
# has to agree with `pi_eval.score` across the firewall and neither side may import the
# other: a mismatch would make the endpoint read as absent rather than as broken.
TAU_REWARD = "tau_reward"


def _result_digest(raw: Any) -> str:
    """A 16-char digest of a tool result that EXECUTED, and never a lie about one.

    WHY THE REDUCTION IS HERE AND NOT IN `canon`
        `pinq.ids.canon` feeds `request_sha` (the per-call LLM cache key) and `semantic_hash`
        (run identity), and its contract is that anything it cannot serialize reproducibly
        RAISES -- an id that is not a function of its inputs must not be computed at all.
        Teaching it `model_dump` would make the module whose job is "reproducible from bytes
        alone" depend on pydantic, and would admit into cache keys a serialization whose
        stability is the model author's choice rather than ours: a plain `model_dump()`
        hands back enums, datetimes and sub-models, which are exactly the values canon
        exists to refuse. So the call site reduces the object to plain data -- which is what
        `_default`'s own error message instructs -- and canon still arbitrates whether the
        RESULT of that reduction is reproducible. `mode="json"` because the python mode
        returns those same non-JSON leaves.

    WHY THE THIRD BRANCH IS TAGGED RATHER THAN SILENT
        A value nothing can reduce still executed, so `ok` stays True; but its digest must
        not be mistakable for a digest of a real result. `h()` is domain-separated, so
        minting it under "res-uncanonical" puts it in a space no genuine digest can reach,
        and the type name is the most that can be said about it without inventing bytes.
        This is deliberately NOT a new `EnvCall` field: a column here changes
        `schema_hash()`, which changes `scorer_hash`, which re-labels every score row of the
        campaigns currently on disk -- a much larger event than the thing being recorded.

    THIS FUNCTION MUST NOT RAISE. It runs on the success path of a call the environment has
    already executed, so an exception escaping it would land in `_call`'s handler and record
    that call as a failure -- which is precisely the bug it exists to fix. Every branch
    therefore falls through to the tagged digest rather than out of the function.
    """
    if isinstance(raw, str):
        return h("res", raw)[:16]
    try:
        return h("res", canon(raw))[:16]
    except (NonCanonical, ValueError):
        # NonCanonical is the reproducibility refusal (a TypeError); ValueError is what
        # json.dumps raises on a circular structure.
        pass
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return h("res", canon(dump(mode="json")))[:16]
        except Exception:  # noqa: BLE001 - a third-party serializer; the tag is the fallback
            pass
    return h("res-uncanonical", type(raw).__name__)[:16]


class Tau2Actuator:
    """Satisfies pinq.protocols.Actuator over a tau2 Environment.

    The rollout EXECUTES here; pi_eval scores the stored log. Nothing in this class knows
    what the correct action sequence is: `evaluation_criteria` is read only inside
    `_compute_reward`, after the last plan step has run, and never reaches a prompt.
    """

    def __init__(
        self,
        env: Any,
        *,
        task: Any = None,
        domain: str = DOMAIN,
        env_kwargs: Mapping[str, Any] | None = None,
        initialize: bool = True,
    ) -> None:
        self._env = env
        self._task = task
        self._domain = domain
        self._env_kwargs = dict(env_kwargs or {})
        self._calls: list[EnvCall] = []
        # (tool_name, kwargs, requestor, raw_result, ok, content_snapshot) per executed step,
        # in order. Held in memory only: this is what the reward replay needs and what the
        # stored log deliberately does not carry, because a raw result is a customer record.
        # The snapshot is the result AS SERIALIZED AT THE MOMENT OF THE CALL -- `raw_result`
        # is a live reference into the environment's DB and does not stay still (see _call).
        self._results: list[tuple[str, dict[str, Any], str, Any, bool, str | None]] = []
        self._seq = 0
        self._unlocked: set[str] = set()
        self._reward: dict[str, float] = {}
        self._reward_done = False
        self._reward_error: str = ""
        self._task_obj: Any = None
        # `initialize=False` is for the GRADING actuator, which is built on the live
        # environment after the dialogue has finished only to fold in a RewardInfo -- and
        # `_with_outcome` reads `env.get_db_hash()` from that same environment straight after.
        # Re-applying `initial_state` there rewinds the world the run actually produced. Inert
        # on banking-style tasks and on airline and retail (both ship `initial_state: null` on
        # every task, MEASURED 0/50 and 0/114); NOT inert on telecom, where all 114 base tasks
        # carry `initialization_actions` and those actions are TOGGLES.
        if initialize:
            self._initialize_env()

    # ------------------------------------------------------------------ Actuator protocol

    def execute(self, plan: Sequence[Mapping[str, Any]], *, turn_idx: int) -> Sequence[EnvCall]:
        """Run a tool plan. Each entry is {name|tool_name, args|kwargs, requestor?}."""
        out: list[EnvCall] = []
        for step in plan:
            name = str(step.get("name") or step.get("tool_name") or "")
            if not name:
                continue
            kwargs = dict(step.get("args") or step.get("kwargs") or {})
            requestor = step.get("requestor", "assistant")
            out.append(self._call(name, kwargs, requestor, turn_idx))
        self._calls.extend(out)
        return tuple(out)

    def final_hashes(self) -> Mapping[str, str]:
        """The reward substrate. Missing hashes become "" rather than None so the parquet
        column stays a non-nullable string and a join never silently drops a run."""
        return {
            "db": self._env.get_db_hash() or "",
            "user_db": self._env.get_user_db_hash() or "",
        }

    def native(self) -> Mapping[str, float]:
        """Upstream's own reward fields.

        Computed HERE, on first call, rather than by the caller. `run_loop` calls
        `execute()`, then `final_hashes()`, then `native()`, so this is the first moment at
        which the action sequence is complete — and it is the only hook a suite-agnostic
        driver offers. Computing it lazily is what lets `attach_reward` have a caller
        without `pi_run.worker` needing to know that tau2 has a grader.
        """
        self._compute_reward()
        return dict(self._reward)

    # ------------------------------------------------------------------ tau2 specifics

    @property
    def env_calls(self) -> tuple[EnvCall, ...]:
        return tuple(self._calls)

    @property
    def unlocked(self) -> frozenset[str]:
        return frozenset(self._unlocked)

    @property
    def reward_error(self) -> str:
        """Why the reward is absent, when it is. Empty when it was computed."""
        return self._reward_error

    def attach_reward(self, reward_info: Any, *, reward_basis: Sequence[Any] = ()) -> None:
        """Fold a RewardInfo into native().

        Kept separate from `_compute_reward` so a caller that already has a RewardInfo — a
        full Orchestrator simulation driven through `pinq_adapters.tau2.agent`, or a test with
        a hand-built one — reports it through exactly the same path as the lazy computation.

        `reward_basis` gates `tau_reward`. See the module docstring: an ACTION-basis task
        evaluated by the ENVIRONMENT evaluator comes back as a free 1.0, and publishing that
        would be a fabricated success on 9 of 97 tasks.
        """
        if reward_info is None:
            return
        basis = {str(getattr(b, "value", b)).upper() for b in reward_basis}

        db = getattr(reward_info, "db_check", None)
        if db is not None:
            # The DB check IS the endpoint: agent-DB hash AND user-DB hash equality against a
            # gold environment. Recorded whatever the basis says, because it is a real
            # measurement of this rollout even where it is not the task's reward.
            self._reward["db_reward"] = float(getattr(db, "db_reward", 0.0) or 0.0)
            self._reward["db_match"] = 1.0 if getattr(db, "db_match", False) else 0.0
            if "DB" in basis:
                self._reward[TAU_REWARD] = self._reward["db_reward"]

        val = getattr(reward_info, "reward", None)
        if val is not None and basis and basis <= {"DB", "ENV_ASSERTION"}:
            # TELECOM HAS NO DB CHECK IN ITS BASIS, so the branch above never fires there and
            # the suite would report no `tau_reward` at all -- which `environment.tau_reward`
            # turns into NaN and `score.py` turns into no row, leaving a suite that looks
            # measured and contributes nothing. The condition is already exactly right: the
            # evaluator covers DB and ENV_ASSERTION and nothing else, so when the basis is a
            # subset of those, its scalar IS the task's reward. ACTION is excluded here for
            # the same reason it is excluded above -- a free 1.0 on telecom's 20 ACTION-basis
            # tasks (and banking's 9) would be a fabricated success.
            # Stored only when the evaluator that produced it covers EVERY component of the
            # basis. Otherwise this number is the product of the components upstream happened
            # to know about, which is not the task's reward.
            self._reward["reward"] = float(val)
            if "DB" not in basis:
                self._reward[TAU_REWARD] = float(val)

        breakdown = getattr(reward_info, "reward_breakdown", None) or {}
        try:
            items = breakdown.items()
        except AttributeError:
            items = ()
        for key, v in items:
            try:
                self._reward[f"breakdown.{getattr(key, 'value', key)}"] = float(v)
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------ the reward

    def _task_object(self) -> Any:
        """The raw task record as an upstream `Task`. None when no task was supplied."""
        if self._task is None:
            return None
        if self._task_obj is None:
            if isinstance(self._task, Mapping):
                from tau2.data_model.tasks import Task

                self._task_obj = Task.model_validate(dict(self._task))
            else:
                self._task_obj = self._task
        return self._task_obj

    def _initialize_env(self) -> None:
        """Apply `task.initial_state` to the live environment, exactly as the Orchestrator does.

        75 of the 97 tasks ship one. Skipping it would leave the policy reading a customer
        database that does not contain the customer, and would put the live environment in a
        different starting state than the gold environment the reward is compared against —
        so every one of those tasks would score 0 for a reason that has nothing to do with
        the policy.
        """
        if self._task is None or not hasattr(self._env, "set_state"):
            return
        task = self._task_object()
        init = getattr(task, "initial_state", None)
        if init is None:
            return
        self._env.set_state(
            initialization_data=getattr(init, "initialization_data", None),
            initialization_actions=getattr(init, "initialization_actions", None),
            message_history=list(getattr(init, "message_history", None) or []),
        )

    def _trajectory(self) -> list[Any]:
        """The executed action sequence as a tau2 message list.

        One tool-call message per step, each followed by its own ToolMessage, which is the
        shape `Environment.set_state` demands (it pops a ToolMessage for every ToolCall and
        raises on a mismatch). The task's own `initial_state.message_history` is prepended
        because `calculate_reward` replays the FULL history into the predicted environment
        while the gold environment gets only the initial part — omitting it would compare an
        environment that never saw the opening exchange against one that did.
        """
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

        task = self._task_object()
        init = getattr(task, "initial_state", None) if task is not None else None
        out: list[Any] = list(getattr(init, "message_history", None) or [])

        for i, (name, kwargs, requestor, raw, ok, snapshot) in enumerate(self._results):
            call_id = f"pinq_{i:04d}"
            tc = ToolCall(id=call_id, name=name, arguments=dict(kwargs), requestor=requestor)
            holder = UserMessage if requestor == "user" else AssistantMessage
            out.append(holder(role=requestor, content=None, tool_calls=[tc]))
            # `Environment.get_response` json-stringifies a successful result and formats a
            # failure as "Error: <exc>"; set_state compares against exactly those bytes, so
            # the recorded content has to be produced the same way -- AND AT THE SAME MOMENT.
            # `snapshot` was taken in `_call`; the fallback is for the one case that could not
            # be serialized then, and it raises here exactly as it did before, into
            # `_compute_reward`'s handler.
            content = (
                snapshot
                if snapshot is not None
                else (type(self._env).to_json_str(raw) if ok else f"Error: {raw}")
            )
            out.append(
                ToolMessage(
                    id=call_id, role="tool", content=content, requestor=requestor, error=not ok
                )
            )
        return out

    def _compute_reward(self) -> None:
        """Grade the executed sequence with upstream's own environment evaluator. Once.

        A failure here is recorded and NOT raised: a rollout that produced a real action log
        must still be stored. But it also must not produce a number — `_reward` stays empty,
        so the metric row is absent rather than zero.
        """
        if self._reward_done:
            return
        self._reward_done = True
        task = self._task_object()
        if task is None:
            return
        criteria = getattr(task, "evaluation_criteria", None)
        if criteria is None:
            self._reward_error = "task has no evaluation_criteria"
            return
        try:
            from tau2.evaluator.evaluator_env import EnvironmentEvaluator
            from tau2.runner import build_environment

            domain, kwargs = self._domain, dict(self._env_kwargs)

            def constructor(solo_mode: bool = False, **kw: Any) -> Any:
                return build_environment(domain, solo_mode=solo_mode, env_kwargs=kw or kwargs)

            info = EnvironmentEvaluator.calculate_reward(
                constructor,
                task,
                self._trajectory(),
                env_kwargs=kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - a failed grade is data, not a sweep abort
            self._reward_error = f"{type(exc).__name__}: {exc}"
            return
        self.attach_reward(info, reward_basis=getattr(criteria, "reward_basis", ()) or ())

    # ------------------------------------------------------------------ one tool call

    def _is_mutating(self, name: str) -> bool:
        """Ask the environment; assume mutation when it cannot say.

        Erring toward `True` is the safe direction: a read mislabelled as a write is a
        false tripwire hit that gets investigated, whereas a write mislabelled as a read
        is an unsafe mutation that is never noticed.
        """
        probe = getattr(self._env, "_is_mutating_tool", None)
        if probe is None:
            return True
        try:
            return bool(probe(name))
        except Exception:
            return True

    def _call(self, name: str, kwargs: dict[str, Any], requestor: str, turn_idx: int) -> EnvCall:
        gated = name in _GATED
        target = str(kwargs.get("agent_tool_name") or kwargs.get("tool_name") or "")
        satisfied = (not gated) or (target in self._unlocked)

        # THE TRY GUARDS THE ENVIRONMENT CALL AND NOTHING ELSE.
        #
        # `canon(raw)` used to sit inside it. Retail's 15 tools return pydantic models, canon
        # refuses anything it cannot serialize reproducibly, and the refusal landed in this
        # `except` -- so a call the environment had ALREADY EXECUTED AND MUTATED was recorded
        # `ok=False`, and `raw` was overwritten with the text of our own serialization error.
        # Measured on retail task 0: `find_user_id_by_name_zip` (a str return) ok, then
        # get_order_details, get_product_details x2 and the mutating
        # exchange_delivered_order_items all logged as failures while succeeding.
        #
        # The damage was not cosmetic. `_results` is what `_trajectory()` replays into the
        # gold environment, so every one of those results became `Error: <exc>` with
        # `error=True`, and the reward was computed over a trajectory in which the agent's
        # tool calls had all failed. `n_failed_calls` and `n_errors_agent` counted them too.
        ok = True
        raw: Any = ""
        try:
            raw = self._env.make_tool_call(name, requestor=requestor, **kwargs)
        except Exception as exc:
            ok, raw = False, str(exc)
        # SYNC THE TWO TOOLKITS, BECAUSE `make_tool_call` DOES NOT.
        #
        # Upstream says so in one line each: `make_tool_call` carries "Note: This does not call
        # sync_tools", and `get_response` -- what the Orchestrator uses -- carries "This also
        # calls sync_tools". `sync_tools` is what couples the agent's database to the user's
        # handset: whether the line is active, whether roaming is allowed, whether a payment
        # request exists.
        #
        # MEASURED on telecom's own gold. Replaying the answer key for the six tasks containing
        # `make_payment` executed `send_payment_request` (assistant) and then `make_payment`
        # (user), which returned "You do not have a payment request." The assistant's request
        # never reached the user's tools, so the gold sequence failed against itself and the
        # six reported as harness errors.
        #
        # In the failure path too: a tool that raises may have mutated before it raised, and
        # leaving the toolkits describing different worlds is the same bug one call later.
        #
        # `Environment.sync_tools` is `pass` in the base class and TELECOM IS THE ONLY DOMAIN
        # THAT OVERRIDES IT, so this is a strict no-op for banking, retail and airline and
        # cannot move a number any of them has produced.
        sync = getattr(self._env, "sync_tools", None)
        if callable(sync):
            try:
                sync()
            except Exception as exc:  # noqa: BLE001 - a failed sync is data, not a sweep abort
                if ok:
                    ok, raw = False, f"sync_tools: {exc}"
        digest = _result_digest(raw) if ok else h("res", f"ERROR: {raw}")[:16]

        # SERIALIZED NOW, NOT AT GRADING TIME. Retail's tools hand back a LIVE reference into
        # the environment's own DB -- measured on task 71, `_results[0][3] is _results[1][3]`
        # is True: one Order object, returned by the read and then mutated in place by
        # `modify_pending_order_items`. Serializing it later put the episode's FINAL state in
        # the transcript slot of an EARLIER call, so `set_state` re-executed, compared, and
        # raised `ValueError: Tool call: ... Returned: ... Expected: ...`. That killed the
        # grade outright on 15 of retail's 114 tasks, and `_reward_of` reports a failed grade
        # as an ABSENT number.
        #
        # Snapshotting is also simply what upstream does: `Environment.get_response`
        # stringifies the result at the moment of the call and it is those bytes set_state
        # compares against. Banking never saw this because every one of its tool returns is a
        # str, which cannot be mutated behind the log's back.
        snapshot: str | None
        try:
            snapshot = type(self._env).to_json_str(raw) if ok else f"Error: {raw}"
        except Exception:  # noqa: BLE001 - deferred to _trajectory, which raises as before
            snapshot = None
        self._results.append((name, dict(kwargs), requestor, raw, ok, snapshot))

        # An unlock only counts once the environment accepted it: a failed unlock leaves
        # the tool locked, and crediting it would fabricate a crossed prerequisite edge.
        if ok and name == UNLOCK_TOOL:
            unlocked = str(kwargs.get("agent_tool_name") or "")
            if unlocked:
                self._unlocked.add(unlocked)

        self._seq += 1
        return EnvCall(
            seq=self._seq,
            turn_idx=turn_idx,
            requestor="user" if requestor == "user" else "assistant",
            tool_name=name,
            kwargs_json=canon(kwargs),
            ok=ok,
            # digest, not payload: the log must be safe to store and diff without carrying
            # customer records around
            result_digest=digest,
            mutating=self._is_mutating(name),
            unlock_required=gated,
            unlock_satisfied=satisfied,
        )
