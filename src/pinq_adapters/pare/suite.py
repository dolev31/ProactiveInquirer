"""PARE, VIEW SIDE — Grid E, the transfer experiment.

THE QUESTION THIS SUITE EXISTS TO ANSWER
    Does a policy prompted for proactive information ACQUISITION improve a proactive
    ACTION agent? To answer it and nothing else, we replace ONLY PARE's goal-inference
    stage with the Inquirer and hold its executor fixed. Swapping the executor too would
    make a win unattributable: the reader could not tell whether the Inquirer helped or
    whether we simply shipped a better actuator.

WHY THERE IS NO NEW GOLD AND NO NEW JUDGE HERE
    PARE ships oracle validation. Scoring on upstream's own verdict is what keeps Grid E
    at ~572 rollouts and about a day, and it is why this grid is exploratory: it carries no
    confirmatory test, so it cannot spend any of the multiplicity budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import EnvCall, EvidenceUnit, TaskId, TaskView
from pinq.view import make_view

from ._probe import N_SCENARIOS, SPLIT, app_of, available, load_scenario_ids
from .actuator import PareActuator

CORPUS_ID = "pare_apps_v1"

INSTRUCTIONS = (
    "You are a proactive assistant observing a user working across several apps. "
    "You may inspect app state before proposing an action. Propose only when the "
    "action is warranted; an unwanted proposal is a cost, not a neutral event."
)

# Grid E arm ids. Defined locally on purpose: pinq_expt/arms.py is owned elsewhere, and a
# duplicated three-line constant is cheaper than a cross-package coupling.
ARM_BASELINE = "pare_baseline"  # upstream Observe-Execute, unchanged
ARM_INQUIRER_PROMPTED = "pare_plus_inquirer_prompted"
ARM_VERBOSITY = "pare_plus_verbosity"  # the same kill switch as every other grid
GRID_E_ARMS = (ARM_BASELINE, ARM_INQUIRER_PROMPTED, ARM_VERBOSITY)


class PareSuiteUnavailable(RuntimeError):
    """Raised when a live-scenario operation is attempted without PARE installed."""


class AppStateRetriever:
    """'Retrieval' in PARE is reading app state, so a read is a read-only tool call.

    Modelled as a Retriever rather than as an Actuator call for the same reason tau2's KB
    search is: it cannot move app state, and logging it as mutating would permanently
    break the "mutations outside the reference actions must be 0" tripwire.
    """

    def __init__(self, apps: Sequence[Any], *, corpus_id: str, corpus_hash: str) -> None:
        self._apps = list(apps)
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash
        self._calls: list[EnvCall] = []
        self._seq = 0
        self.turn_idx = 0

    @property
    def env_calls(self) -> tuple[EnvCall, ...]:
        return tuple(self._calls)

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        """Rank apps by naive token overlap with the query and return their state.

        Deliberately model-free: a learned ranker here would put an untracked model inside
        the observation path, and every arm would silently inherit it.
        """
        from pinq.ids import canon

        terms = {t for t in query.lower().split() if len(t) > 2}
        scored: list[tuple[float, str, str]] = []
        for app in self._apps:
            label = str(getattr(app, "name", None) or type(app).__name__)
            try:
                state = app.get_state()
                text = state if isinstance(state, str) else canon(state)
            except Exception as exc:
                text = f"ERROR: {exc}"
            low = f"{label} {text}".lower()
            scored.append((float(sum(1 for t in terms if t in low)), label, text))
        scored.sort(key=lambda r: (-r[0], r[1]))
        hits = [s for s in scored[:k] if s[0] > 0] or scored[:1]

        self._seq += 1
        self._calls.append(
            EnvCall(
                seq=self._seq,
                turn_idx=self.turn_idx,
                requestor="assistant",
                tool_name="get_state",
                kwargs_json=canon({"query": query, "k": k}),
                ok=True,
                result_digest=",".join(label for _, label, _ in hits),
                mutating=False,
            )
        )
        return tuple(
            EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=label,
                span=f"0:{len(text)}",
                title=label,
                text=text,
                score=score,
            )
            for score, label, text in hits
        )


@dataclass
class PareSuite:
    """Satisfies pinq.protocols.TaskSuite over PARE's 143-scenario `full` split."""

    suite_id: ClassVar[str] = "pare"
    suite_version: ClassVar[str] = "v0.0.1"
    corpus_id: ClassVar[str] = CORPUS_ID

    splits_root: Path | None = None
    split: str = SPLIT
    strict_counts: bool = True

    corpus_hash: str = field(default="", init=False)
    _ids: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.splits_root is None:
            ok, why = available()
            if not ok:
                raise PareSuiteUnavailable(why)
        self._ids = load_scenario_ids(self.splits_root, self.split)
        if self.strict_counts and self.split == SPLIT and len(self._ids) != N_SCENARIOS:
            raise ValueError(f"expected {N_SCENARIOS} scenarios, found {len(self._ids)}")
        # The scenario id list IS the corpus identity here: app contents are generated per
        # run from the scenario, so there is no static document set to hash.
        self.corpus_hash = _corpus_hash((sid, app_of(sid), "") for sid in self._ids)

    # ------------------------------------------------------------------ TaskSuite protocol

    def task_ids(self) -> tuple[TaskId, ...]:
        return self._ids

    def view(self, tid: TaskId) -> TaskView:
        """PARE scenarios have no user question: the situation IS the task.

        The view therefore carries the scenario id and its app label and nothing else. The
        scenario's oracle events and expected actions stay on the gold side.
        """
        return make_view(
            task_id=tid,
            suite_id=self.suite_id,
            question=f"Observe the user's activity in the {app_of(tid)} app and act when warranted.",
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=180,
        )

    def retriever(self, tid: TaskId) -> AppStateRetriever:
        return AppStateRetriever(
            self.apps(tid), corpus_id=self.corpus_id, corpus_hash=self.corpus_hash
        )

    def actuator(self, tid: TaskId) -> PareActuator:
        return PareActuator(self.apps(tid), scenario_id=tid)

    # ------------------------------------------------------------------ PARE specifics

    def apps(self, tid: TaskId) -> tuple[Any, ...]:
        scenario = self.scenario(tid)
        return tuple(getattr(scenario, "apps", None) or ())

    def scenario(self, tid: TaskId) -> Any:
        """Load and initialize one scenario. Imports PARE; requires the extra."""
        ok, why = available()
        if not ok:
            raise PareSuiteUnavailable(why)
        from pare.benchmark.scenario_loader import load_scenarios_from_registry

        for sc in load_scenarios_from_registry(scenario_ids=[tid]):
            if not getattr(sc, "_initialized", False):
                sc.initialize()
            return sc
        raise KeyError(tid)

    def run_upstream(self, tid: TaskId, config: Any) -> Any:
        """Run a scenario through PARE's own runner and return its validation result.

        This is the `pare_baseline` arm: upstream Observe-Execute, untouched. Keeping the
        baseline on upstream's own code path is what makes the comparison a transfer
        result rather than a comparison between two of our own implementations.
        """
        ok, why = available()
        if not ok:
            raise PareSuiteUnavailable(why)
        from pare.scenario_runner import TwoAgentScenarioRunner

        return TwoAgentScenarioRunner().run(config, self.scenario(tid))
