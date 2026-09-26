"""The seams. Every one of these is a structural-typing Protocol, so an adapter satisfies it
by shape and never has to import a base class from us."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .budget import BudgetLedger
from .types import (
    Action,
    Answer,
    Ask,
    CallTelemetry,
    Draft,
    EnvCall,
    Evidence,
    EvidenceUnit,
    State,
    TaskId,
    TaskView,
)


@runtime_checkable
class LLM(Protocol):
    """The single choke point for model traffic.

    Implementations debit the ledger internally and expose NO read access to it. That is how
    a policy stays budget-blind while still being able to call a model.
    """

    def complete(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, CallTelemetry]: ...


@runtime_checkable
class Recorder(Protocol):
    """A WRITE-ONLY counter sink. `BudgetLedger` satisfies it structurally.

    A policy that has to report something — a malformed generation, a parse failure — needs
    somewhere to put the count, and handing it the ledger would hand it `cap` and `spent`
    with it. It gets this instead: one method, no return value, nothing to read back. That
    is what lets `malformed_rate` be a reportable number without making `Inquirer.act`
    budget-aware, which would confound STOP with the cap the experiment announced.
    """

    def record(self, currency: str, amount: float = 1.0) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    corpus_id: str
    corpus_hash: str

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]: ...


@runtime_checkable
class Drafter(Protocol):
    """Two obligations, deliberately split.

    resolve() is agentic and impure: it may retrieve, and every retrieval it fires while
      answering an ASK debits the SHARED ledger.
    draft() MUST be a pure function of (view, evidence.subset_hash, seed). This is an
      ARCHITECTURAL constraint, not a metric detail: without it phi_LOO is undefined,
      the prefix ladder is not recomputable, and the stop test means nothing.
      Enforced by tests/test_arms.py::test_draft_is_a_pure_function_of_view_subset_hash_and_seed
      (five repeats through a replay client that RAISES on a cache miss, so five hits and one
      underlying call prove the drafter emitted identical request bytes) and by
      tests/test_synth_closed_form.py::test_draft_is_pure_in_subset_hash.

      This line used to cite `tests/test_purity.py` and claim "every results table prints
      purity_violation_rate". NEITHER EXISTS. The constraint is real and is genuinely tested --
      by the two tests named above -- but a docstring that cites a file nobody can open is how a
      reader concludes a property is checked when they cannot find the check, and how the next
      person deletes the real test believing the cited one covers it.
    """

    def resolve(
        self, view: TaskView, ask: Ask, ev: Evidence, *, seed: int, ledger: BudgetLedger
    ) -> tuple[str, Evidence]: ...

    def draft(self, view: TaskView, ev: Evidence, *, seed: int, ledger: BudgetLedger) -> Draft: ...


@runtime_checkable
class Answerer(Protocol):
    """FROZEN across all arms: one model pin, one prompt hash, one word cap, blind to the arm
    id and to the budget. This is what removes answer-length bias by construction rather
    than relying on a post-hoc regression alone."""

    prompt_hash: str

    def answer(
        self,
        view: TaskView,
        ev: Evidence,
        draft: Draft | None,
        *,
        seed: int,
        ledger: BudgetLedger,
    ) -> Answer: ...


@runtime_checkable
class Inquirer(Protocol):
    """Sees State ONLY. No ledger, no context object, no cap, no arm id, no gold.

    Budget-agnostic BY TYPE, which is what makes prefix-k of a long rollout exchangeable
    with an independent short run.
    """

    policy_id: str

    def reset(self, view: TaskView, seed: int) -> None: ...

    def act(self, s: State) -> Action: ...


@runtime_checkable
class Actuator(Protocol):
    """The bridge to a stateful world (tau2's DB, PARE's FSM apps).

    The rollout EXECUTES; pi_eval scores the stored LOG. This is what replaces the
    unimplementable native_score(task_id, answer: str).
    """

    def execute(self, plan: Sequence[Mapping[str, Any]], *, turn_idx: int) -> Sequence[EnvCall]: ...

    def final_hashes(self) -> Mapping[str, str]: ...

    def native(self) -> Mapping[str, float]: ...


@runtime_checkable
class TaskSuite(Protocol):
    suite_id: str
    suite_version: str

    def task_ids(self) -> tuple[TaskId, ...]: ...

    def view(self, tid: TaskId) -> TaskView: ...

    def retriever(self, tid: TaskId) -> Retriever: ...

    def actuator(self, tid: TaskId) -> Actuator | None: ...

    # NOTE: there is deliberately no native_score() here. Scoring lives in pi_eval and reads
    # the stored Outcome, because a hash over an executed action sequence against a stateful
    # environment is not computable from an answer string.
