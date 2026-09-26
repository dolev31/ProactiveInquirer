"""The judging STAGE: runs + gold -> Judgments -> judge-derived metric rows + diagnostics.

This is the seam between the three standalone graders (kpr, citation, quality) and
`pi_eval.score`. Everything here is a policy decision about how the instrument is USED, and
each one is written down because the alternative is a plausible-looking number.

EVERY PAIR IS GRADED IN BOTH ORDERS, ALWAYS. `judges.paired.both_orders` calls the grader
once per presentation order and `resolve_pair` forces an order-inconsistent pair to a TIE.
For an absolute grader the honest expectation is P(picks first) == 0.5 exactly, so the
diagnostic should never fire -- and that is the point. It fires the moment the grader stops
being order-invariant in practice (a judge handed both reports in one call, a reused
conversation, a batching layer that lets one verdict condition the next, a cache keyed on
something that carries order), all of which are real bugs that would otherwise be discovered
as an unexplained effect.

A DISQUALIFIED JUDGE EMITS NO METRIC. `judge_is_usable()` is consulted per (suite, criterion)
before any score row derived from that criterion is emitted. Above the position-bias
threshold the rows are NOT WRITTEN and the reason is recorded, rather than written with a
caveat in a `notes` column that nobody reads. The Judgments themselves are still written:
they are the evidence FOR the disqualification, and deleting them would remove the only proof
that the judge was measured at all.

WHICH PAIRS ARE JUDGED. Exactly the arm pairs the preregistration commits to for a
judge-derived endpoint on that suite. Judging every arm against every other is a multiplicity
problem bought with an LLM invoice, and the pairs that matter were named before the data
existed.

THE PER-RUN METRICS COME FROM THE 'ab' GRADING ONLY. Both orders are graded, but pooling both
into one run's recall would count every key point twice and make `keypoint_recall` a mean over
a multiset whose size depends on how many pairs the run happened to appear in. The 'ba'
grading exists to measure the INSTRUMENT, not the system.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from pi_eval.judges import citation, kpr, quality
from pi_eval.judges._llm import JudgeLLM, JudgeParseError
from pi_eval.judges.paired import (
    Order,
    both_orders,
    judge_is_usable,
    krippendorff_alpha_nominal,
    length_adjusted_effect,
    position_bias,
    ratings_by_family,
    resolve_pair,
    win_tie_loss,
)
from pi_eval.judges.types import Judgment
from pi_eval.metrics import kpr_incremental

NAN = float("nan")


class JudgeBudgetExceeded(RuntimeError):
    """The judge's spend cap was reached. Nothing partial is written; see _Grader.check_budget."""


def judge_spend_cap(explicit: float | None = None) -> float | None:
    """PI_JUDGE_SPEND_CAP_USD, in dollars. None means uncapped. An explicit argument wins --
    the same precedence rule as the sweep's --spend-cap and --concurrency."""
    import os

    if explicit is not None:
        return float(explicit) if float(explicit) > 0 else None
    raw = os.environ.get("PI_JUDGE_SPEND_CAP_USD", "").strip()
    if raw:
        try:
            v = float(raw)
            return v if v > 0 else None
        except ValueError:
            pass
    return None


# criterion -> the metric names derived from it. Used to suppress a disqualified judge's rows.
CRITERION_METRICS: dict[str, tuple[str, ...]] = {
    "keypoint": (
        "kpr_incremental",
        "keypoint_recall",
        "keypoint_contradiction_rate",
        "kpr_repetition_gap",
    ),
    "citation_support": ("citation_support",),
    **{k: (f"quality_{k}",) for k in quality.CRITERION_KEYS},
}

# The dead band `paired_from_absolute` applies, per criterion, as a MULTIPLE of sigma_J. On a
# 0-10 rubric a 0.17 mean difference is inside the judge's own noise and calling it a win
# manufactures wins out of sigma_J. The key-point criterion keeps a dead band of zero: its
# difference is a count of supported points, not a Likert mean, and a one-point difference is
# a real difference.
DEAD_BAND_SIGMAS: dict[str, float] = {"keypoint": 0.0, "citation_support": 0.0}
DEFAULT_DEAD_BAND_SIGMAS = 1.0


@dataclass(frozen=True, slots=True)
class TaskInput:
    """Everything the judges need about one task, and nothing gold-adjacent beyond it."""

    suite_id: str
    task_id: str
    question: str
    key_points: tuple[tuple[str, str], ...] = ()  # (gold_node_id, text)
    docs: Mapping[str, str] = field(default_factory=dict)  # url -> text, for the citation judge


@dataclass(frozen=True, slots=True)
class RunInput:
    run_id: str
    arm_id: str
    task: TaskInput
    report: str


@dataclass
class JudgeResult:
    judgments: list[Judgment] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # run_id -> name -> value
    # run_id -> name -> (n, note). THE DENOMINATOR EVERY JUDGE METRIC WAS PUBLISHED WITHOUT.
    #
    # `score.py` emitted judge rows with no `n=`, and `_emit` defaults n to 1 -- so
    # `kpr_incremental`, the drgym PRIMARY endpoint, reached scores.parquet claiming a sample
    # size of one. `KprResult` computes n_gold, n_points_judged and n_unjudged precisely so the
    # shrinkage is visible, and `as_dict()` returned the four floats and dropped all three.
    #
    # Concretely: 40 gold key points, the judge parses verdicts for 2, both Supported ->
    # keypoint_recall = 1.0 with n=1, and nothing downstream can see that 38 of 40 points were
    # never judged. The module's own docstring promises the opposite: "a missing measurement
    # must look missing. n_gold is carried alongside so the shrinkage is visible rather than
    # inferred."
    metric_n: dict[str, dict[str, tuple[int, str]]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    unusable: dict[str, str] = field(default_factory=dict)  # "suite/criterion" -> why
    unjudged: list[str] = field(default_factory=list)  # "run_id/criterion: reason"
    n_calls_failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_judgments": len(self.judgments),
            "n_runs_with_metrics": len(self.metrics),
            "judges_disqualified": dict(sorted(self.unusable.items())),
            "unjudged": sorted(self.unjudged),
            "n_parse_failures": self.n_calls_failed,
            "diagnostics": self.diagnostics,
        }


class _Grader:
    """Absolute grading, memoized on (run_id, criterion_family, order).

    Memoized ACROSS PAIRS but never across ORDERS: a run appears in several pairs and must not
    be re-billed for each, while collapsing the two orders would delete the very difference
    `position_bias` exists to detect.
    """

    def __init__(
        self,
        llm: JudgeLLM,
        *,
        judge_model: str,
        judge_family: str,
        seed: int,
        result: JudgeResult,
        cap_usd: float | None = None,
    ) -> None:
        self.cap_usd = cap_usd
        self._llm = llm
        self.judge_model = judge_model
        self.judge_family = judge_family
        self.seed = seed
        self._result = result
        self._cache: dict[tuple[str, str, str], tuple[Judgment, ...]] = {}

    def _grade(self, run: RunInput, family: str, order: Order) -> tuple[Judgment, ...]:
        t = run.task
        common = dict(
            suite_id=t.suite_id,
            task_id=t.task_id,
            run_id=run.run_id,
            judge_model=self.judge_model,
            judge_family=self.judge_family,
            seed=self.seed,
            order=order,
        )
        if family == "keypoint":
            if not t.key_points:
                return ()
            return kpr.grade(self._llm, key_points=t.key_points, report=run.report, **common)
        if family == "citation_support":
            if not t.docs:
                return ()
            js, _diag = citation.grade(self._llm, report=run.report, docs=t.docs, **common)
            return js
        if family == "quality":
            return quality.grade(self._llm, question=t.question, report=run.report, **common)
        raise ValueError(f"unknown judge family {family!r}")

    def check_budget(self) -> None:
        """Stop before the next call if the judge has spent its cap.

        `--spend-cap` is checked in `run_sweep` and covers ROLLOUTS. Judging is a SECOND
        provider bill: every eligible run with a report and gold, times every criterion, times
        both presentation orders, for every preregistered arm pair -- and it produces
        `kpr_incremental`, a primary endpoint. Nothing bounded it.

        Raising rather than truncating is deliberate. `pi_eval.score` writes its rows only
        after judging returns, so an abort leaves NOTHING partial on disk. A half-judged
        parquet is worse than an unjudged one: some arms would carry judge-derived rows and
        others would not, and the comparison between them would silently be over different
        subsets.
        """
        if self.cap_usd is None:
            return
        spend = getattr(self._llm, "spend", None)
        if not callable(spend):
            return
        billed = float(spend().get("usd_billed", 0.0))
        if billed >= self.cap_usd:
            raise JudgeBudgetExceeded(
                f"the judge has billed ${billed:.4f} against a ${self.cap_usd:.2f} cap "
                f"(PI_JUDGE_SPEND_CAP_USD / --judge-spend-cap). Nothing was written: a "
                f"half-judged parquet would give some arms judge-derived rows and not others."
            )

    def get(self, run: RunInput, family: str, order: Order) -> tuple[Judgment, ...]:
        key = (run.run_id, family, order)
        if key not in self._cache:
            self.check_budget()
            try:
                self._cache[key] = self._grade(run, family, order)
            except JudgeParseError as exc:
                # An unreadable verdict is an UNJUDGED item, never a passing grade. It is
                # counted and named; the run simply contributes no row for this criterion.
                self._result.n_calls_failed += 1
                self._result.unjudged.append(f"{run.run_id}/{family}[{order}]: {exc}")
                self._cache[key] = ()
        return self._cache[key]


def prefetch_gradings(
    keys: Sequence[tuple[str, str, str]],
    grade_one: Callable[[tuple[str, str, str]], Any],
    *,
    workers: int = 1,
) -> dict[tuple[str, str, str], tuple[Any, BaseException | None]]:
    """Grade distinct keys CONCURRENTLY, returning {key: (value, error)} in INPUT order.

    THREADS COMPUTE, THE CALLER MUTATES. Everything shared that `_Grader.get` touches is
    left alone here, because two of the three are unsafe to touch from a thread:

        self._cache[key] = ...      dict assignment, atomic under the GIL -- fine either way
        result.n_calls_failed += 1  read-modify-write -- NOT atomic, silently undercounts
        result.unjudged.append(..)  atomic, but ARRIVAL ORDER is nondeterministic

    `n_calls_failed` and `unjudged` are both REPORTED, so letting threads write them would
    make two identical runs disagree about how many items the judge failed to parse and in
    what order -- the precise shape of bug that becomes a wrong published number. So an
    exception is CAPTURED and handed back rather than raised or recorded, and the caller
    applies every result serially, in the order given here.

    The order of this mapping follows the INPUT, never completion, so a re-run with a
    different thread schedule produces identical downstream bytes.

    Duplicates are graded once: a run appears in several preregistered pairs, and grading it
    twice bills the provider twice for an answer that is memoized anyway.

    WHY IT IS WORTH THE CARE. Measured on this repository, the judge sustains ~7 live calls
    a minute. sigma_J at its 200-pair minimum is 3,000 live calls in the paraphrase pass --
    seven hours serially -- and drgym/P3 grades through this same function.
    """
    ordered: list[tuple[str, str, str]] = list(dict.fromkeys(keys))
    out: dict[tuple[str, str, str], tuple[Any, BaseException | None]] = {}
    if not ordered:
        return out
    n = max(1, int(workers))
    if n == 1 or len(ordered) == 1:
        for k in ordered:
            try:
                out[k] = (grade_one(k), None)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller, not swallowed
                out[k] = (None, exc)
        return out

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(n, len(ordered))) as ex:
        futures = {k: ex.submit(grade_one, k) for k in ordered}
    for k in ordered:  # INPUT order, not completion order
        try:
            out[k] = (futures[k].result(), None)
        except BaseException as exc:  # noqa: BLE001
            out[k] = (None, exc)
    return out


def judge_workers(env: Mapping[str, str] | None = None) -> int:
    """Concurrent judge dispatches. `PI_JUDGE_CONCURRENCY`, default 8.

    8 matches what the sweeps already sustain against this provider. Bounded and overridable
    because the judge is a shared service: a number that is right on one deployment is a
    denial of service on another.
    """
    import os

    e = os.environ if env is None else env
    try:
        return max(1, int(str(e.get("PI_JUDGE_CONCURRENCY", "8")).strip() or 8))
    except ValueError:
        return 8


def _recycle_every() -> int:
    """Live judge calls between client rebuilds. `PI_JUDGE_RECYCLE_CALLS`, 0 disables."""
    raw = os.environ.get("PI_JUDGE_RECYCLE_CALLS", "500").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 500


def warm_grader(
    grader: Any,
    items: Sequence[tuple[Any, str, str]],
    *,
    workers: int = 8,
    batch: int = 0,
) -> None:
    """Fill `grader._cache` for `keys` concurrently, so the serial loops become cache hits.

    THE BUDGET IS STILL CHECKED. `_Grader.get` calls `check_budget()` before every dispatch,
    and a prefetched key never reaches that branch -- so warming without checking would let
    the judge's spend cap be bypassed entirely by the very code meant to speed it up. Checked
    once per BATCH, which makes the cap a stop-loss with an overshoot of at most one batch:
    the same contract `run_sweep`'s cap already has, for the same reason (a call in flight
    cannot be recalled).

    FAILURES ARE APPLIED SERIALLY. `n_calls_failed` and `unjudged` are reported, and a thread
    writing them would undercount the first (read-modify-write) and scramble the order of the
    second. Threads only run `_grade`; this loop does every mutation, in input order.

    Already-cached keys are skipped, so warming twice does not re-bill.
    """
    from pi_eval.judges._llm import JudgeParseError

    # (run, family, order) -> the cache key `_Grader.get` would use. The RUN OBJECT is
    # carried because `_grade` needs it; the key is only its id.
    by_key = {(r.run_id, f, o): (r, f, o) for r, f, o in items}
    todo = [k for k in by_key if k not in grader._cache]
    if not todo:
        return
    result = getattr(grader, "_result", grader)  # tests may pass a grader that IS its result
    size = batch if batch and batch > 0 else max(1, int(workers))
    # MEASURED in-process decay: 47 -> 13 calls/min on one judging pass, 71 -> 22 on another,
    # while a fresh PROCESS at the same provider and cache held 71/min. Root cause unproven,
    # so the remedy is empirical and switchable: 0 disables it.
    #
    # Duck-typed on purpose. `pi_eval` is gold-only and may not import `pi_run`, where the
    # recyclable judge lives; a client without `recycle` (every test double, and the
    # cache-only `judge_replay`) is simply left alone.
    every = _recycle_every()
    since = 0
    for start in range(0, len(todo), size):
        chunk = todo[start : start + size]
        grader.check_budget()
        if every and since >= every:
            recycle = getattr(getattr(grader, "_llm", None), "recycle", None)
            if callable(recycle):
                recycle()
            since = 0
        since += len(chunk)
        got = prefetch_gradings(chunk, lambda k: grader._grade(*by_key[k]), workers=workers)
        for key in chunk:  # input order, on this thread only
            value, err = got[key]
            if err is None:
                grader._cache[key] = value
            elif isinstance(err, JudgeParseError):
                # An unreadable verdict is an UNJUDGED item, never a passing grade -- the
                # same rule `_Grader.get` applies, applied where it would have applied it.
                result.n_calls_failed += 1
                result.unjudged.append(f"{key[0]}/{key[1]}[{key[2]}]: {err}")
                grader._cache[key] = ()
            else:
                raise err


def _quality_of(js: Sequence[Judgment], criterion: str) -> tuple[Judgment, ...]:
    return tuple(j for j in js if j.criterion == criterion)


def _metrics_for(
    run: RunInput, ab: Mapping[str, tuple[Judgment, ...]]
) -> tuple[dict[str, float], dict[str, tuple[int, str]]]:
    """(values, denominators) for one run, from its canonical 'ab' grading.

    The second return is what every judge row was published without: the n each value was
    computed over, and a note when that n is smaller than the gold universe. See
    `JudgeResult.metric_n`.
    """
    out: dict[str, float] = {}
    ns: dict[str, tuple[int, str]] = {}
    kp = ab.get("keypoint", ())
    if kp:
        res = kpr_incremental.score(kp, [nid for nid, _ in run.task.key_points])
        vals = {k: v for k, v in res.as_dict().items() if not math.isnan(v)}
        out.update(vals)
        # THE DENOMINATOR EACH VALUE WAS ACTUALLY COMPUTED OVER, which is not n_gold for any of
        # them. `kpr_incremental.score` computes raw over JUDGMENT ROWS
        # (supported_judgments / len(js)) and incremental over JUDGED POINTS
        # (len(supported) / n_judged); nothing divides by n_gold. `n` feeds
        # `report.floor_flag`'s 2*sigma_J/sqrt(n), so using the gold universe would claim a
        # precision the measurement does not have -- n=40 where two points were judged.
        #
        # n_gold belongs in the NOTE: it is the universe the estimate is silent about, and the
        # module docstring's promise is that "the shrinkage is visible rather than inferred".
        note = (
            f"{res.n_unjudged} of {res.n_gold} gold key points carry no parsed verdict"
            if res.n_unjudged
            else ""
        )
        per_metric = {
            "keypoint_recall": res.n_judgments,
            "kpr_incremental": res.n_points_judged,
            "keypoint_contradiction_rate": res.n_points_judged,
            "kpr_repetition_gap": res.n_points_judged,
        }
        for name in vals:
            ns[name] = (per_metric.get(name, res.n_points_judged), note)
    cit = ab.get("citation_support", ())
    if cit:
        v = citation.mean_support(cit)
        if not math.isnan(v):
            out["citation_support"] = v
            ns["citation_support"] = (len(cit), "")
    qual = ab.get("quality", ())
    by_crit = quality.by_criterion(qual)
    for key, value in by_crit.items():
        out[f"quality_{key}"] = value
        ns[f"quality_{key}"] = (len(_quality_of(qual, key)) or len(qual), "")
    return out, ns


def _pair_meta(a: RunInput, b: RunInput, criterion: str, prompt_sha: str, grader: _Grader) -> dict:
    return dict(
        criterion=criterion,
        suite_id=a.task.suite_id,
        task_id=a.task.task_id,
        run_id_a=a.run_id,
        run_id_b=b.run_id,
        judge_family=grader.judge_family,
        judge_model=grader.judge_model,
        judge_prompt_sha=prompt_sha,
        len_a_words=len(a.report.split()),
        len_b_words=len(b.report.split()),
    )


_PROMPT_SHA = {
    "keypoint": kpr.PROMPT_SHA,
    "citation_support": citation.PROMPT_SHA,
    "quality": quality.PROMPT_SHA,
}


def _criteria_of(family: str) -> tuple[str, ...]:
    return quality.CRITERION_KEYS if family == "quality" else (family,)


def judge_runs(
    llm: JudgeLLM,
    runs: Sequence[RunInput],
    *,
    judge_model: str,
    judge_family: str = "openai",
    arm_pairs: Sequence[tuple[str, str]] = (),
    families: Sequence[str] = ("keypoint", "citation_support", "quality"),
    sigma_j: Mapping[str, float] | None = None,
    seed: int = 0,
    cap_usd: float | None = None,
) -> JudgeResult:
    """Grade every run, judge every preregistered pair in both orders, then diagnose.

    Returns the Judgments to write, the per-run judge-derived metric values to emit, and the
    instrument diagnostics. Nothing here touches parquet: `pi_eval.score` owns the append.
    """
    res = JudgeResult()
    grader = _Grader(
        llm,
        judge_model=judge_model,
        judge_family=judge_family,
        seed=seed,
        result=res,
        cap_usd=judge_spend_cap(cap_usd),
    )
    by_arm: dict[tuple[str, str], list[RunInput]] = defaultdict(list)
    for r in runs:
        by_arm[(r.task.task_id, r.arm_id)].append(r)

    # ---- 0. warm the grader concurrently. The loops below are unchanged and become cache
    # hits; only DISPATCH is parallel, so judgment ids, ordering and every downstream byte
    # are identical to the serial path. Measured before this: the judge sustained ~7 live
    # calls a minute, which made sigma_J a multi-hour job and P3 inherit it.
    warm_grader(
        grader,
        [(r, f, "ab") for r in runs for f in families],
        workers=judge_workers(),
    )

    # ---- 1. canonical 'ab' grading of every run, and the per-run metric values
    ab_by_run: dict[str, dict[str, tuple[Judgment, ...]]] = {}
    for r in runs:
        graded = {f: grader.get(r, f, "ab") for f in families}
        ab_by_run[r.run_id] = graded
        for js in graded.values():
            res.judgments.extend(js)
        res.metrics[r.run_id], res.metric_n[r.run_id] = _metrics_for(r, graded)

    # ---- 2. the preregistered pairs, both orders
    paired: dict[tuple[str, str], list[Judgment]] = defaultdict(list)
    outcomes: dict[tuple[str, str], list] = defaultdict(list)
    for task_id, treatment, comparator in _pairs(runs, arm_pairs):
        for a in by_arm[(task_id, treatment)]:
            for b in by_arm[(task_id, comparator)]:
                for family in families:
                    for criterion in _criteria_of(family):
                        tol = _dead_band(criterion, sigma_j)
                        meta = _pair_meta(a, b, criterion, _PROMPT_SHA[family], grader)
                        crit = criterion

                        def grade(order: Order, a=a, b=b, family=family, crit=crit):
                            ja = grader.get(a, family, order)
                            jb = grader.get(b, family, order)
                            if family == "quality":
                                return _quality_of(ja, crit), _quality_of(jb, crit)
                            return ja, jb

                        ab, ba = both_orders(grade, tol=tol, **meta)
                        res.judgments.extend((ab, ba))
                        paired[(a.task.suite_id, criterion)].extend((ab, ba))
                        outcomes[(a.task.suite_id, criterion)].append(resolve_pair(ab, ba))
        # 'ba' grading of the pair members is now in the memo; fold it into the record too.
        for arm in (treatment, comparator):
            for r in by_arm[(task_id, arm)]:
                for family in families:
                    res.judgments.extend(grader.get(r, family, "ba"))

    # ---- 3. the instrument diagnostics, and the disqualification gate
    diags: dict[str, Any] = {}
    for (suite_id, criterion), js in sorted(paired.items()):
        ok, why = judge_is_usable(js)
        key = f"{suite_id}/{criterion}"
        outs = outcomes[(suite_id, criterion)]
        diags[key] = {
            "position_bias": position_bias(js),
            "usable": ok,
            "why": why,
            "n_pairs": len(outs),
            **win_tie_loss(outs),
            "length_adjusted": length_adjusted_effect(outs),
        }
        if not ok:
            res.unusable[key] = why
    diags["krippendorff_alpha_nominal"] = krippendorff_alpha_nominal(
        ratings_by_family(res.judgments)
    )
    diags["judge_families"] = sorted({j.judge_family for j in res.judgments})
    res.diagnostics = diags

    # ---- 4. suppress the rows a disqualified judge would otherwise have produced
    for key in res.unusable:
        suite_id, criterion = key.split("/", 1)
        drop = set(CRITERION_METRICS.get(criterion, ()))
        for r in runs:
            if r.task.suite_id == suite_id:
                res.metrics[r.run_id] = {
                    k: v for k, v in res.metrics.get(r.run_id, {}).items() if k not in drop
                }
                res.metric_n[r.run_id] = {
                    k: v for k, v in res.metric_n.get(r.run_id, {}).items() if k not in drop
                }
    # Dedup: a run appears in several pairs, so its memoized gradings were extended more than
    # once. judgment_id is the identity; keeping the first occurrence keeps arrival order.
    seen: set[str] = set()
    deduped: list[Judgment] = []
    for j in res.judgments:
        if j.judgment_id in seen:
            continue
        seen.add(j.judgment_id)
        deduped.append(j)
    res.judgments = deduped
    return res


def _dead_band(criterion: str, sigma_j: Mapping[str, float] | None) -> float:
    """The dead band in SCORE UNITS, derived from the measured sigma_J for this criterion.

    Zero when sigma_J has not been measured: an invented dead band would silently convert
    real differences into ties, which is a bias with no measurement behind it. The
    conservative direction here is to leave the raw sign alone and let the noise floor in
    `pi_eval.report` refuse the effect instead.
    """
    mult = DEAD_BAND_SIGMAS.get(criterion, DEFAULT_DEAD_BAND_SIGMAS)
    if not mult or not sigma_j:
        return 0.0
    for name in CRITERION_METRICS.get(criterion, ()):
        if name in sigma_j:
            return mult * float(sigma_j[name])
    return 0.0


def _pairs(
    runs: Sequence[RunInput], arm_pairs: Sequence[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """(task_id, treatment, comparator) for every pair that actually has both arms present."""
    arms_by_task: dict[str, set[str]] = defaultdict(set)
    for r in runs:
        arms_by_task[r.task.task_id].add(r.arm_id)
    out = []
    for task_id, arms in sorted(arms_by_task.items()):
        for t, c in arm_pairs:
            if t in arms and c in arms:
                out.append((task_id, t, c))
    return out
