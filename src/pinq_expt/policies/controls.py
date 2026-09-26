"""The nulls: policies that spend the same budget without the thing under test.

None of these calls a model. That is deliberate and it is not a saving — it is what makes
them clean controls. `RandomQInquirer` is the null for phi: if a question drawn from a
DIFFERENT task's pool moves the metric as much as the policy's own question did, phi is
measuring retrieval volume rather than question content, and every number downstream of it
is about how much was fetched. Matching volume is therefore not a nicety here; an unmatched
random arm cannot distinguish those two worlds at all.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pinq import promptlib
from pinq.ids import h
from pinq.types import Action, Ask, State, Stop, TaskView

CHECKLIST_PROMPT = "checklist_facets"


class MissingScript(KeyError):
    """A policy was run on a task the mined pool does not cover.

    Loud on purpose: a silent empty script turns `parallel_replay` into `drafter_only` and
    the two arms would then agree for a reason that has nothing to do with the graph.

    ALSO RAISED BY `RandomQInquirer`, which is not scripted but is volume-matched: without a
    reference run for the task there is no volume to match, and the fall-through used to
    resolve to 0 asks and report success. Same condition, same refusal, so the two arms cannot
    disagree about what a missing reference means.
    """


class EmptyPool(ValueError):
    """A random-question arm with no other task to draw from is not a control."""


# --------------------------------------------------------------------------- random_q


class RandomQInquirer:
    """The null for phi: real questions, wrong task, MATCHED VOLUME.

    Volume is matched by `ask_counts[task_id]` — the number of asks the reference arm made on
    THIS task — so the two arms issue the same number of retrieval calls against the same
    cap. Sampling is `random.Random` seeded from `(task, seed)`, so the draw is reproducible
    from the manifest and a re-run of the arm is byte-identical rather than a new sample.
    """

    policy_id = "random_q"
    # This arm draws from the questions of OTHER tasks, so its behaviour depends on the WHOLE
    # pool rather than on its own task's entry. `pi_run.worker` reads this to decide what to
    # hash into run identity: for a scripted arm the script is its own task's list, but for
    # this one, adding a task to the sweep changes what it can sample and therefore changes
    # the run.
    uses_question_pool = True

    def __init__(
        self,
        *,
        questions: Mapping[str, Sequence[str]] | None = None,
        ask_counts: Mapping[str, int] | None = None,
        n_asks: int | None = None,
    ) -> None:
        # `questions=None` constructs but cannot run: reset() raises EmptyPool. The arm
        # table declares requires_questions=True and build() refuses it earlier, so the only
        # way to reach that raise is to call the factory bare, which is what introspection
        # (policy_id, protocol conformance) does.
        self._pool = {k: tuple(v) for k, v in (questions or {}).items()}
        self._counts = dict(ask_counts or {})
        self._n_asks = n_asks

    def reset(self, view: TaskView, seed: int) -> None:
        others = sorted({q for tid, qs in self._pool.items() if tid != view.task_id for q in qs})
        if not others:
            raise EmptyPool(
                f"no questions from a task other than {view.task_id!r}: a random-question "
                "arm drawn from the task's own pool is not a null."
            )
        # VOLUME IS UNDEFINED WITHOUT A REFERENCE RUN FOR *THIS* TASK. This was a single
        # expression ending `or len(self._pool.get(view.task_id, ()))`, which resolves to 0 for
        # a task the pool does not cover -- and 0 is not a refusal, it is an empty queue, a STOP
        # on turn 0 and a `status: ok` row that asked nothing. MEASURED on musique test at cap 8
        # (2026-09-18): 26 absent tasks x 2 seeds = 50 such rows, billed $0.0847, each one a
        # `drafter_only` trajectory carrying this arm's label into a table as a matched null.
        # `parallel_replay` raised MissingScript on those same 26 task ids and billed $0.0000,
        # so the sibling arm was already doing this correctly on identical input.
        #
        # An EXPLICIT count is the one case where an absent task is still well defined: it is
        # the caller stating the volume rather than the pool implying it, so `ask_counts` and
        # `n_asks` are honoured (including a deliberate 0) and only the fall-through refuses.
        n = self._counts.get(view.task_id)
        if n is None:
            n = self._n_asks
        if n is None:
            reference = self._pool.get(view.task_id) or ()
            if not reference:
                raise MissingScript(
                    f"{type(self).__name__} has no reference run for task {view.task_id!r}, so "
                    "the matched volume is undefined. A volume-matched null that asks nothing "
                    "is a drafter_only trajectory wearing this arm's label. Seed it from the "
                    "reference run with recorded_questions(runs_root, arm_id=...), restricted "
                    "to the SAME budget_cap, or pass ask_counts to state the volume explicitly."
                )
            n = len(reference)
        rng = random.Random(h("random_q", view.key, str(seed)))
        self._queue = rng.sample(others, k=min(int(n), len(others)))

    def act(self, s: State) -> Action:
        if not self._queue:
            return Stop(reason="policy_stop")
        return Ask(text=self._queue.pop(0), rationale="random question from another task")


# --------------------------------------------------------------------------- checklist


class ChecklistInquirer:
    """A fixed facet checklist, walked in order, with NO conditioning on state.

    `act` reads nothing from `s`. That is the hypothesis it tests: if a policy that cannot
    see its own evidence, its own draft or its own history matches the one that can, then the
    system's advantage is a coverage template and not a decision procedure.
    """

    policy_id = "checklist"
    prompt_name = CHECKLIST_PROMPT

    def __init__(self, *, facets: Sequence[str] | None = None) -> None:
        self.facets = tuple(facets) if facets is not None else _load_facets()

    @property
    def prompt_hashes(self) -> dict[str, str]:
        return promptlib.hashes(self.prompt_name)

    def reset(self, view: TaskView, seed: int) -> None:
        self._queue = [f"{view.question} Specifically: {f}." for f in self.facets]

    def act(self, s: State) -> Action:
        if not self._queue:
            return Stop(reason="policy_stop")
        return Ask(text=self._queue.pop(0), rationale="fixed checklist facet")


def _load_facets() -> tuple[str, ...]:
    return tuple(
        line.strip() for line in promptlib.load(CHECKLIST_PROMPT).splitlines() if line.strip()
    )


# --------------------------------------------------------------------------- scripted


class ScriptedInquirer:
    """Issues a recorded question list, in order, conditioned on NOTHING.

    `act` ignores `s` entirely, which is the mechanism the arms below are built on: whatever
    dependency structure produced the recorded list, replaying it cannot reconstruct.
    """

    policy_id = "scripted"

    def __init__(self, *, questions: Mapping[str, Sequence[str]] | None = None) -> None:
        self._script = {k: tuple(v) for k, v in (questions or {}).items()}

    def reset(self, view: TaskView, seed: int) -> None:
        if view.task_id not in self._script:
            raise MissingScript(
                f"{type(self).__name__} has no recorded questions for task {view.task_id!r}. "
                "Seed it from the reference run with recorded_questions(runs_root, arm_id=...)."
            )
        self._queue = list(self._script[view.task_id])

    def act(self, s: State) -> Action:
        if not self._queue:
            return Stop(reason="policy_stop")
        return Ask(text=self._queue.pop(0), rationale=f"{self.policy_id}: replayed")


class ParallelReplayInquirer(ScriptedInquirer):
    """All of a recorded trajectory's questions AT ONCE, to a fresh Drafter.

    "At once" inside a sequential loop means: every question is emitted without looking at
    what the previous one returned. The retrieval is therefore identical to the sequential
    run by construction — the query strings are the same strings — and the ONLY thing that
    differs is that no answer informed any question.

    THEREFORE THIS ARM IS A DETERMINISM CHECK, NOT A SEQUENCING COUNTERFACTUAL, and the
    docstring above used to end "if this arm matches the sequential policy, the trajectory had
    no dependency structure to exploit and the graph framing goes". That conclusion does not
    follow from this arm. Proven LLM-free on the synth suite (see
    tests/test_prereg.py::test_parallel_replay_is_input_identical_to_the_arm_it_replays): the
    same question strings retrieve the same evidence uids, and because `draft()` and the
    Answerer are pure in the evidence subset hash, the ANSWER TEXT is identical too. Every
    metric that is a function of the evidence set is identically equal, so `evidence_coverage`
    has a delta of exactly zero before any data exists.

    What it IS good for: a non-zero delta here means the harness is impure -- a drafter that is
    not a pure function of (view, subset_hash, seed), a retriever with hidden state, or a
    leaking cache. That is worth having and worth running; it is just not a test of sequencing.

    The arm that DOES vary the sequencing channel is `inquirer_depth1`: it enumerates the
    questions from `x` alone in a single call and then walks them, so none of its questions
    saw an answer AND the strings genuinely differ from the treatment's.
    """

    policy_id = "parallel_replay"

    @classmethod
    def from_runs(
        cls,
        runs_root: str | Path,
        *,
        arm_id: str = "inquirer_prompted",
        suite_id: str | None = None,
    ) -> "ParallelReplayInquirer":
        return cls(questions=recorded_questions(runs_root, arm_id=arm_id, suite_id=suite_id))


class GoldEvidenceInquirer(ScriptedInquirer):
    """CEILING ARM. The questions are chosen on the gold side and handed in.

    Nothing in `pinq_expt` reads gold — the firewall forbids the import and this class does
    not attempt it. The scorer-side harness supplies `questions`, which is what keeps the
    ceiling computable without putting a gold reader on the rollout path.
    """

    policy_id = "gold_evidence"


class OracleVreqInquirer(ScriptedInquirer):
    """CEILING ARM. Only the questions with positive measured value, in order.

    Same construction and same firewall position as `GoldEvidenceInquirer`: the selection is
    made where phi is computed, and arrives here as a list of strings.
    """

    policy_id = "oracle_vreq"


# --------------------------------------------------------------------------- seeding


def recorded_questions(
    runs_root: str | Path,
    *,
    arm_id: str | None = None,
    suite_id: str | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Read `{task_id: [question, ...]}` out of a finished sweep's run directories.

    Reads `pi_run.worker`'s artifacts with plain json rather than importing it: the worker
    imports the arm table, and an import back the other way would be a cycle. The file format
    is the one stable thing between them (`manifest.json` + `turns.jsonl`), and it is already
    the format every scorer reads.

    THE PAIRING IS THE POINT. `parallel_replay`'s delta is "identically zero by construction"
    only if it replays the questions of the run it is PAIRED with, and a non-zero delta is
    documented as indicting the whole harness. This used to key on task_id alone and take
    whichever run sorted first by run_id -- an arbitrary hash order, across seeds, including
    inadmissible runs and rung-2 BRANCH runs, which are deliberate off-policy forks of a
    single turn. MEASURED on the first admissible musique sweep: 5 of 12 cells replayed a
    list that was not the treatment's, one recognisably a candidate-branch question, and the
    kill switch read parallel_replay delta 0.0208 instead of 0.

    So: `seed` selects the paired run; branch runs are never mined; an admissible run beats a
    `dev-` one; and what remains is chosen in sorted run_id order, deterministically. A task
    with no run at that seed is ABSENT rather than filled from another seed -- the replay arm
    then refuses loudly, which is the correct outcome for a control with nothing to replay.
    """
    # (task_id) -> (rank, run_id, questions); lower rank wins, run_id breaks ties.
    best: dict[str, tuple[int, str, list[str]]] = {}
    for manifest_path in sorted(Path(runs_root).glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if arm_id is not None and manifest.get("arm_id") != arm_id:
            continue
        if suite_id is not None and manifest.get("suite_id") != suite_id:
            continue
        if seed is not None and manifest.get("seed") != seed:
            continue
        # NEVER a branch: a rung-2 candidate forks one turn off-policy on purpose, so its
        # question list is not the trajectory any arm is paired against.
        if manifest.get("branch_of_run_id"):
            continue
        turns = manifest_path.parent / "turns.jsonl"
        if not turns.exists():
            continue
        questions = [
            str(row["question"])
            for row in _jsonl(turns)
            if row.get("action_kind") == "ask" and row.get("question")
        ]
        if not questions:
            continue
        task = str(manifest["task_id"])
        # An admissible run outranks a dev- one, but a dirty sweep must still be able to seed
        # its own replay arms, so dirty is a preference and not a filter.
        rank = 1 if manifest.get("dirty") else 0
        run_id = str(manifest.get("run_id") or manifest_path.parent.name)
        cand = (rank, run_id, questions)
        if task not in best or cand[:2] < best[task][:2]:
            best[task] = cand
    return {t: q for t, (_r, _i, q) in sorted(best.items())}


def _jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
