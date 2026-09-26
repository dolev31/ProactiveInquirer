"""The mode-collapse detector: distinct-3 over the questions a checkpoint actually emits.

WHY THIS GATE EXISTS AND WHY IT IS NOT OPTIONAL. Rejection-sampling SFT is trained on the
argmax candidate at every state. The argmax of a stochastic policy is systematically the most
GENERIC question -- the one phrasing that scores acceptably everywhere -- so the training set
is biased toward one template before a single gradient step is taken. The resulting failure is
not a crash and not a loss spike: the checkpoint asks a near-identical question at every state
and still scores respectably, because a generic question retrieves something. It would reach a
table as "trained, modest gain" while actually being a policy that stopped conditioning on
state at all. distinct-3 is the cheapest statistic that separates those two worlds.

WHY 0.55. It is a floor calibrated on the PROMPTED policy: the prompted Inquirer's own
questions must clear it comfortably, so a trained checkpoint that falls below has become
measurably more repetitive than the baseline it is supposed to beat. A checkpoint that fails
the gate is NOT shipped and NOT reported as a win; the written consequence is in
docs/TRAINING.md, and it is to fall back to rung 0's prompt and report the SFT attempt as a
negative result with this number attached.

WHY TRIGRAMS RATHER THAN EXACT DUPLICATES. Exact-duplicate rate is defeated by a policy that
varies one noun. Trigram diversity is not: "what is the founding date of X" and "what is the
founding date of Y" share every trigram but one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# The gate. Below this, the checkpoint is not shipped. See the module docstring.
# CALIBRATED, and the calibration says it is UNREACHABLE ON MUSIQUE. Measured on musique v1
# gold and the built 800-task corpus (tests/test_distinct3_calibration.py pins all four):
#
#     gold required-node texts (2,660)   0.2250   <- the ANSWER KEY scores this
#     one gold question per task (800)   0.4682
#     the 800 task questions themselves  0.4685   <- the corpus's own ceiling
#
# So a policy asking exactly the gold sub-questions -- perfectly state-dependent by
# construction -- is rejected at 0.225, and no sample drawn from musique reaches 0.55. A
# musique export failing this gate is therefore NOT evidence of the mode collapse the message
# below describes; the pooled statistic conflates "the policy always asks the same thing" with
# "many runs of one task converged on the right thing". Within a task our own export scores
# 0.777, which is the measure that separates them.
#
# The value is NOT changed here. It is a preregistered instrument choice, and moving it is a
# decision with a paper trail rather than a convenience; what is recorded is the evidence any
# such decision has to be made against.
DISTINCT3_FLOOR = 0.55

# Below this many questions, distinct-3 is not a measurement. Rejection-sampling SFT turns
# every below-floor state into a STOP target, so a collapsing dataset loses its questions
# first -- which means the diversity gate gets WEAKEST exactly as the thing it guards against
# arrives. Refusing a sample too small to measure is the same rule sigma_J follows with
# SIGMA_J_MIN_PAIRS: an estimate from too few observations is an anecdote, and an anecdote that
# clears a floor is worse than no estimate at all.
MIN_QUESTIONS_TO_MEASURE = 30

# STOP is one of two actions, so a dataset dominated by it teaches "always stop" -- and it does
# so while every per-example loss looks healthy, which is why this is a gate and not a log line.
# NOT YET CALIBRATED against a real export: no rung-1 dataset has been built. It is a ceiling
# chosen to catch collapse rather than to tune anything, and the first real export should
# either confirm it or move it in a visible diff.
STOP_SHARE_CEILING = 0.80

DEFAULT_N = 3


class ModeCollapse(RuntimeError):
    """A checkpoint whose questions stopped depending on the state.

    Raised rather than warned: a warning in a training log is a warning nobody reads, and the
    consequence of ignoring this one is a table row that says the opposite of what happened.
    """


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def ngrams(text: str, n: int = DEFAULT_N) -> list[tuple[str, ...]]:
    toks = _tokens(text)
    return [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]


def distinct_n(texts: Iterable[str], n: int = DEFAULT_N) -> float:
    """Unique n-grams / total n-grams, pooled across the whole sample.

    POOLED, not averaged per text. A per-text mean is near 1.0 even under total collapse,
    because each individual question has few repeated trigrams inside itself; the repetition
    this gate exists to catch is BETWEEN questions.
    """
    total = 0
    seen: set[tuple[str, ...]] = set()
    for t in texts:
        gs = ngrams(t, n)
        total += len(gs)
        seen.update(gs)
    if total == 0:
        return 0.0
    return len(seen) / total


@dataclass(frozen=True, slots=True)
class DiversityReport:
    n_texts: int
    n_unique_texts: int
    distinct_3: float
    floor: float
    ok: bool

    def render(self) -> str:
        verdict = "PASS" if self.ok else "FAIL -- mode collapse"
        return (
            f"distinct-3 = {self.distinct_3:.3f} (floor {self.floor:.2f})  "
            f"{self.n_unique_texts}/{self.n_texts} unique questions  {verdict}"
        )


def report(texts: Sequence[str], *, floor: float = DISTINCT3_FLOOR) -> DiversityReport:
    d3 = distinct_n(texts, DEFAULT_N)
    return DiversityReport(
        n_texts=len(texts),
        n_unique_texts=len(set(texts)),
        distinct_3=d3,
        floor=floor,
        ok=d3 >= floor,
    )


# THE GATE'S FLOOR, calibrated against four baselines rather than chosen. Measured on musique:
#
#     sample                 n   pooled d3   WITHIN-TASK d3
#     GOLD (perfect)      2660      0.2250           0.9985
#     ours                2254      0.4545           0.7787
#     semi-collapsed      2254      0.0028           0.6033
#     COLLAPSED           2254      0.0004           0.2571
#
# 0.65 sits above semi-collapsed -- ten canned templates emitted regardless of state, which IS
# collapse and must fail -- and below both our export and gold, with margin on each side.
WITHIN_TASK_FLOOR = 0.65

# Below this many questions a task's diversity is an anecdote, not a measurement.
MIN_PER_TASK = 2


def within_task_distinct_n(pairs: Sequence[tuple[str, str]], n: int = DEFAULT_N) -> float:
    """Mean distinct-n computed WITHIN each task, over tasks with at least MIN_PER_TASK
    questions.

    WHY WITHIN A TASK. The gate's own error message asks whether "the questions stopped
    depending on the state". Varying within a task IS state-dependence; repeating across runs
    of one task is convergence on the right question, which the pooled statistic cannot tell
    apart and penalises just the same.

    A task with a single question contributes NOTHING rather than 0.0 -- one question cannot be
    more or less varied than itself, and scoring it zero would punish a dataset for having
    short episodes.
    """
    from collections import defaultdict

    by_task: dict[str, list[str]] = defaultdict(list)
    for task_id, text in pairs:
        if text:
            by_task[str(task_id)].append(text)
    scores = [distinct_n(v, n) for v in by_task.values() if len(v) >= MIN_PER_TASK]
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def within_task_seed_distinct_n(
    triples: Sequence[tuple[str, object, str]], n: int = DEFAULT_N
) -> float:
    """Mean distinct-n computed WITHIN each (task, seed), over groups with at least
    MIN_PER_TASK questions.

    THE ARTIFACT THIS EXISTS TO AVOID. `within_task_distinct_n` groups every run of a task
    together regardless of seed. When an evaluation runs a checkpoint at more than one seed,
    two seeds of the same task land in the SAME group -- and a checkpoint that asks an
    identical, fully state-dependent set of questions under both seeds is scored as if it had
    become more repetitive, purely because it was evaluated twice. Concretely: duplicating a
    fixed set of k distinct questions once (seed 1 repeats seed 0 verbatim) doubles the n-gram
    total while adding no new n-gram, which HALVES distinct-n by construction. That is a
    property of how many seeds the evaluator happened to run, not of the policy -- see
    `tests/test_train_gate.py::test_distinct3_by_seed_is_not_halved_by_a_second_seed_that_repeats_the_first`,
    which pins the 1.0-vs-0.5 gap this function closes.

    Grouping by (task, seed) instead scores each rollout on its own: running the same task
    again under a second seed can no longer, by itself, move the statistic. `gate.py` reports
    this as `criteria["distinct3"]["by_seed"]`, alongside the unchanged pooled `"value"` --
    added rather than substituted, so an existing verdict's `value` never moves silently (see
    the call site in `pinq_train.gate`).
    """
    from collections import defaultdict

    by_group: dict[tuple[str, object], list[str]] = defaultdict(list)
    for task_id, seed, text in triples:
        if text:
            by_group[(str(task_id), seed)].append(text)
    scores = [distinct_n(v, n) for v in by_group.values() if len(v) >= MIN_PER_TASK]
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def assert_diverse(
    texts: Sequence[str],
    *,
    floor: float = DISTINCT3_FLOOR,
    min_texts: int = MIN_QUESTIONS_TO_MEASURE,
    task_ids: Sequence[str] | None = None,
) -> DiversityReport:
    """The gate. Returns the report on success, raises `ModeCollapse` on failure."""
    if not texts:
        raise ModeCollapse(
            "no questions to measure. An empty sample passes every diversity test ever "
            "written, so it is refused rather than scored."
        )
    if len(texts) < min_texts:
        raise ModeCollapse(
            f"only {len(texts)} question(s) to measure, below {min_texts}. distinct-3 over a "
            "handful of texts is an anecdote, and a collapsing dataset loses its questions "
            "FIRST -- so a sample this small is itself the symptom, not a reason to skip the "
            "check."
        )
    rep = report(texts, floor=floor)

    # WITHIN-TASK IS THE GATE; pooled distinct-3 is reported beside it as a diagnostic.
    # Pooled scores the GOLD answer key at 0.2250 -- a policy asking exactly the gold
    # sub-questions is perfectly state-dependent by construction, and pooled would reject it
    # while accepting weaker data. It measures how formulaic the question SURFACE is, not
    # whether questions depend on the state. See tests/test_within_task_diversity.py.
    if task_ids is not None:
        if len(task_ids) != len(texts):
            raise ValueError(
                f"{len(texts)} texts but {len(task_ids)} task ids: a within-task measure "
                "cannot be computed from a misaligned pairing, and zipping the shorter of the "
                "two would silently score a subset."
            )
        wt = within_task_distinct_n(list(zip(task_ids, texts)))
        if wt == wt and wt < WITHIN_TASK_FLOOR:  # NaN => no task had enough questions
            raise ModeCollapse(
                f"within-task distinct-3 = {wt:.3f} (floor {WITHIN_TASK_FLOOR:.2f}); pooled "
                f"{rep.distinct_3:.3f}. The questions stopped depending on the state -- they "
                "do not vary WITHIN a task, which is where variation would show. It would "
                "still score respectably, which is exactly why this is checked rather than "
                "eyeballed. Do not ship it: fall back to rung 0's prompt and report the SFT "
                "run as a negative result with this number."
            )
        return rep

    # No task ids supplied: fall back to the pooled check. Kept so a caller with a bare list
    # of strings still gets SOME collapse protection rather than none.
    if not rep.ok:
        raise ModeCollapse(
            f"{rep.render()}. The checkpoint's questions stopped depending on the state; it "
            "would still score respectably, which is exactly why this is checked rather than "
            "eyeballed. Do not ship it: fall back to rung 0's prompt and report the SFT run "
            "as a negative result with this number."
        )
    return rep
