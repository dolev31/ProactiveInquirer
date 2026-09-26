"""Paired, order-randomized judging, plus the diagnostics that decide whether to trust it.

Why paired at all: for independent scores, Var[Q_a - Q_b] = 2*sigma_J^2*(1 - rho). With a
1-10 Likert rubric sigma_J is around 0.5-1.0 while the per-question effects being chased are
0.1-0.3, so an unpaired design is measuring noise. Showing one judge BOTH answers in a single
call drives rho up and collapses the variance of the difference.

Why both orders: position bias is real and large. Every pair is judged "ab" and "ba"; a pair
whose two verdicts disagree is recorded as a TIE rather than being silently resolved, and
P(judge picks first) is published as a diagnostic that can DISQUALIFY a judge outright.

Why length is always recorded: the null explanation for any judge-derived win is "the longer
answer won". It is controlled by construction (a frozen word-capped Answerer) and then again
post hoc here.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

from pinq.ids import h

from .types import Judgment

DISQUALIFY_POSITION_BIAS = 0.55


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    task_id: str
    sign: int  # +1 A wins, -1 B wins, 0 tie (including order-inconsistent)
    magnitude: float
    order_consistent: bool
    delta_log_len: float
    # False when either presentation order produced no judgment at all. Distinct from sign == 0,
    # which is a judge that looked and found them equal. Unjudged pairs are DROPPED by the
    # estimators; counting them as ties diluted every effect toward zero.
    judged: bool = True


def resolve_pair(ab: Judgment, ba: Judgment) -> PairedOutcome:
    """Combine the two orderings of one pair.

    An order-inconsistent pair is forced to a TIE. That is deliberately conservative: such a
    pair carries no information about which answer is better, only that the judge is
    order-sensitive on it, and counting it either way would import position bias into the
    effect estimate.
    """
    judged = ab.pref_sign is not None and ba.pref_sign is not None
    sa = ab.pref_sign if ab.pref_sign is not None else 0
    sb = ba.pref_sign if ba.pref_sign is not None else 0
    consistent = sa == sb
    sign = sa if consistent else 0
    mags = [m for m in (ab.magnitude, ba.magnitude) if m is not None]
    mag = statistics.fmean(mags) if (consistent and mags) else 0.0
    la, lb = max(ab.len_a_words, 1), max(ab.len_b_words, 1)
    return PairedOutcome(
        ab.task_id, sign, mag, consistent, math.log(la) - math.log(lb), judged=judged
    )


def position_bias(judgments: Iterable[Judgment]) -> float:
    """P(the judge prefers whichever answer was shown first).

    Above DISQUALIFY_POSITION_BIAS the judge is not usable for this comparison, and that is a
    finding about the instrument rather than about the systems.
    """
    first = total = 0
    for j in judgments:
        if j.pref_sign is None or j.pref_sign == 0:
            continue
        total += 1
        # In "ab" order A was shown first; in "ba" order B was.
        if (j.order == "ab" and j.pref_sign > 0) or (j.order == "ba" and j.pref_sign < 0):
            first += 1
    return first / total if total else float("nan")


def order_diagnostic_has_power(judgments: Sequence[Judgment]) -> bool:
    """Did grading in both orders ever produce a DIFFERENT verdict?

    If not, `position_bias` is 0.5 by construction rather than by measurement, and a gate at
    0.55 can never fire. That is the state this repository is in: all three graders
    (kpr, citation, quality) grade ONE report in isolation, and `order` reaches only the
    judgment_id hash and a stamped field -- never the prompt, never the request. So
    grade("ab") and grade("ba") return identical scores, every ab/ba pair carries the same
    pref_sign, and exactly one of each pair counts as "first".

    Measured over 200 pairs: position_bias == 0.5 exactly, against a 0.55 threshold.

    The double grading is not therefore pointless -- harness.py documents it as a tripwire for a
    future pairwise judge, a reused cache key, or anything else that carries order -- but it
    costs a second billed call per run per criterion (the memo key is
    (run_id, family, order)), and it must not be REPORTED as a passed position-bias check.
    """
    by_pair: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for j in judgments:
        if j.pref_sign is not None:
            by_pair[(j.task_id, j.criterion, j.judge_family)][j.order] = j.pref_sign
    return any(len(set(v.values())) > 1 for v in by_pair.values() if len(v) > 1)


def judge_is_usable(judgments: Sequence[Judgment]) -> tuple[bool, str]:
    """Is the INSTRUMENT usable -- which is not the same question as whether it found an effect.

    `position_bias` is NaN whenever no judgment is decisive, and this mapped that straight to
    "unusable". But there are two entirely different ways to have no decisive judgment, and only
    one of them is a fault in the judge:

      * every pair TIED. The judge read both answers and found them equal. That is a measured
        null -- the systems are indistinguishable on this criterion -- and it is the single most
        likely outcome for a criterion where the arms genuinely do not differ. Disqualifying
        here made `judge_is_usable` return False exactly when the answer was "no effect", and
        harness.py then STRIPPED every judge-derived metric for that suite/criterion, including
        kpr_incremental, a primary. A null result was deleted rather than reported, and the
        table showed a missing row where the honest reading is a zero.
      * nothing PARSED. Every judgment came back with pref_sign None. That is an instrument
        failure and the disqualification is right.

    Position bias is undefined in the first case, not zero and not damning: with no decisive
    judgment there is no evidence the judge favours a position, so the gate has nothing to fire
    on and says so.
    """
    decisive = [j for j in judgments if j.pref_sign is not None and j.pref_sign != 0]
    if not decisive:
        parsed = [j for j in judgments if j.pref_sign is not None]
        if not parsed:
            return False, f"no judgment parsed ({len(judgments)} attempted)"
        return True, f"position bias not estimable: all {len(parsed)} judgments tied"
    pb = position_bias(judgments)
    if pb > DISQUALIFY_POSITION_BIAS:
        return False, f"position bias {pb:.3f} > {DISQUALIFY_POSITION_BIAS}"
    if not order_diagnostic_has_power(judgments):
        # 0.5 because nothing could have made it anything else. Reporting that as a passed check
        # states a measurement that was not taken.
        return True, (
            f"position bias {pb:.3f}, NOT MEASURED: no pair's verdict differed between "
            "presentation orders, so the graders are order-invariant and this value is a "
            "property of the design rather than of the judge"
        )
    return True, f"position bias {pb:.3f}"


def sigma_j(judgments: Sequence[Judgment]) -> float:
    """Judge noise, estimated by TEST-RETEST over identical and paraphrased pairs.

    Published, frozen in prereg BEFORE any Inquirer arm runs, and thereafter the gate on what
    may be claimed at all: no effect below 2*sigma_J/sqrt(n) is reportable.

    POOLED, NOT AVERAGED. This took the mean of each group's standard deviation, and the mean
    of SDs is not an estimate of sigma -- E[s] = sigma*sqrt(2/pi) ~= 0.798*sigma for the
    two-replicate design this docstring documents. Measured against a known sigma=1.0 over 400
    groups x 200 datasets:

        replicates    mean-of-SDs        pooled
             2       0.7958 (-20.4%)   1.0012 (+0.1%)
             3       0.8872 (-11.3%)   0.9987 (-0.1%)
             5       0.9390  (-6.1%)   0.9999 (-0.0%)

    sigma_J is the gate on what may be CLAIMED -- no effect below 2*sigma_J/sqrt(n) is
    reportable -- so a 20% low estimate lowers that floor by 20% and admits effects that are
    inside the judge's own noise. The bias is toward reporting more.

    Pooling also brings this into agreement with scripts/measure_sigma_j.py, which computes
    SD(differences)/sqrt(2): over two-replicate groups s^2 = (x1-x2)^2 / 2, so pooling gives
    sqrt(mean(d^2)/2). The two are not algebraically identical -- `stdev` centres on the sample
    mean and divides by n-1, this does neither -- but they agree to O(1/n) and were measured at
    1.312 vs 1.313 on n=500. Two estimators of one FROZEN prereg constant that disagreed by 20%
    is how a published value comes to depend on which function produced it.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for j in judgments:
        if j.retest_group_id and j.magnitude is not None:
            groups[j.retest_group_id].append(j.magnitude)
    num = sum((len(v) - 1) * statistics.variance(v) for v in groups.values() if len(v) > 1)
    den = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    return math.sqrt(num / den) if den else float("nan")


def krippendorff_alpha_nominal(ratings: dict[str, dict[str, str]]) -> float:
    """Nominal-scale Krippendorff's alpha across judge families or human annotators.

    ratings: unit_id -> {rater_id: label}. Chosen over Cohen's kappa because it tolerates
    missing ratings, which is the normal case when three judge families are run over
    overlapping but not identical subsets.
    """
    units = [list(r.values()) for r in ratings.values() if len(r) > 1]
    if not units:
        return float("nan")
    labels = sorted({v for u in units for v in u})
    if len(labels) < 2:
        return 1.0

    # PER-UNIT NORMALISATION. Krippendorff's D_o divides each unit's pairwise disagreement by
    # (m_u - 1) and the total by n, the number of ratings -- NOT the total disagreement by the
    # total pair count, which is what this did. The two agree exactly when every unit has the
    # same number of raters, and this function's own docstring says the unbalanced case is the
    # normal one ("three judge families run over overlapping but not identical subsets"). Under
    # imbalance the old form over-weights the units with more raters by a factor of (m_u - 1).
    do = 0.0
    n_ratings = 0
    for u in units:
        m = len(u)
        pairs = sum(u[i] != u[j] for i in range(m) for j in range(m) if i != j)
        do += pairs / (m - 1)
        n_ratings += m
    do = do / n_ratings if n_ratings else 0.0

    counts: dict[str, int] = defaultdict(int)
    for u in units:
        for v in u:
            counts[v] += 1
    n = sum(counts.values())
    de = 1 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)) if n > 1 else 0.0
    return 1 - do / de if de else float("nan")


def length_adjusted_effect(outcomes: Sequence[PairedOutcome]) -> dict[str, float]:
    """Regress the paired sign on the log-length difference and report the intercept.

    The intercept is the arm effect at Delta-length = 0, i.e. what remains once "the longer
    answer won" has been accounted for. Both the raw and the adjusted number go in the paper;
    if the effect dies under adjustment, that sentence goes in the abstract.
    """
    outcomes = [o for o in outcomes if o.judged]  # an unjudged pair is not a measured tie
    xs = [o.delta_log_len for o in outcomes]
    ys = [float(o.sign) for o in outcomes]
    if len(xs) < 3:
        return {
            "raw": statistics.fmean(ys) if ys else float("nan"),
            "adjusted": float("nan"),
            "slope": float("nan"),
            "n": len(ys),
            "identifiable": 0.0,
        }
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if not sxx:
        # NO VARIANCE IN THE REGRESSOR: THE SLOPE IS UNDEFINED, NOT ZERO.
        #
        # This returned 0.0, which makes `adjusted == raw` -- and that is the STRONGEST reading
        # of the paper's claim ("the effect survives adjustment for length") produced by having
        # performed no adjustment at all. Indistinguishable in the table from a genuine finding
        # that length explained nothing.
        #
        # And it is the EXPECTED case here, not a corner: the frozen Answerer applies the same
        # word cap to every arm by construction, precisely so that length cannot be what a judge
        # rewards. Every pair having the same delta-log-length is what success looks like.
        #
        # report.py types NaN as NOT_APPLICABLE ("--"), so this reads as "not measured".
        return {
            "raw": my,
            "adjusted": float("nan"),
            "slope": float("nan"),
            "n": len(ys),
            "identifiable": 0.0,
        }
    slope = sxy / sxx
    return {
        "raw": my,
        "adjusted": my - slope * mx,
        "slope": slope,
        "n": len(ys),
        "identifiable": 1.0,
    }


def win_tie_loss(outcomes: Sequence[PairedOutcome]) -> dict[str, int]:
    out = {"win": 0, "tie": 0, "loss": 0, "order_inconsistent": 0, "unjudged": 0}
    for o in outcomes:
        if not o.judged:
            out["unjudged"] += 1
            continue
        out["win" if o.sign > 0 else "loss" if o.sign < 0 else "tie"] += 1
        if not o.order_consistent:
            out["order_inconsistent"] += 1
    return out


# --------------------------------------------------- absolute graders, run through the pair
#
# The three DeepResearchGym judges (kpr, citation, quality) are ABSOLUTE graders: each one
# scores ONE report against a rubric or a gold key point, and returns no preference. The
# machinery above needs a preference, so `paired_from_absolute` folds a system's per-item
# judgments into one signed comparison, and `both_orders` runs it in both presentations so
# the SAME position-bias gate, sigma_J estimate and alpha computation apply to them.
#
# WHAT THE POSITION-BIAS DIAGNOSTIC MEANS FOR AN ABSOLUTE GRADER, precisely. Grading each
# report in isolation SHOULD make the verdict independent of which report was presented
# first, so the honest expectation is exactly 0.5 and the gate should never fire. That is not
# a tautology and the check is not decoration: the diagnostic fires the moment the grader
# stops being order-invariant in practice -- a judge handed both reports in one call, a
# conversation reused across the pair, a batching layer that lets one item's verdict condition
# the next, a cache keyed on something that carries order. Every one of those is a real bug we
# would otherwise discover as an unexplained effect, and each shows up here as
# P(picks first) drifting off 0.5 and the judge being disqualified by judge_is_usable().

ABSOLUTE_CRITERIA: frozenset[str] = frozenset(
    {
        "keypoint",
        "citation_support",
        "clarity",
        "depth",
        "balance",
        "breadth",
        "support",
        "insightfulness",
    }
)

Order = Literal["ab", "ba"]

# Grades one presentation order and returns (judgments about A, judgments about B).
Grader = Callable[[Order], tuple[Sequence[Judgment], Sequence[Judgment]]]


def mean_score(judgments: Sequence[Judgment]) -> float:
    """Mean of the non-null scores. NaN on an empty set, because 0.0 would read as a verdict."""
    xs = [j.score for j in judgments if j.score is not None]
    return statistics.fmean(xs) if xs else float("nan")


def paired_from_absolute(
    *,
    a: Sequence[Judgment],
    b: Sequence[Judgment],
    order: Order,
    criterion: str,
    suite_id: str,
    task_id: str,
    run_id_a: str,
    run_id_b: str,
    judge_family: str,
    judge_model: str,
    judge_prompt_sha: str,
    len_a_words: int = 0,
    len_b_words: int = 0,
    retest_group_id: str | None = None,
    tol: float = 0.0,
) -> Judgment:
    """Two absolute gradings -> one signed, paired Judgment from A's point of view.

    `tol` is a DEAD BAND, not a rounding convenience: on a 0-10 rubric a 0.17 mean difference
    is inside the judge's own noise, and calling that a win manufactures wins out of sigma_J.
    Set it from the published sigma_J for the criterion; the default of 0 keeps the raw
    behaviour for the key-point criterion, where the difference is a count of supported
    points rather than a Likert mean.
    """
    ma, mb = mean_score(a), mean_score(b)
    if math.isnan(ma) or math.isnan(mb):
        # NOT A TIE. `mean_score` returns NaN precisely so that "0.0 would read as a verdict",
        # and this turned it straight back into one -- pref_sign 0, the same value a judge that
        # looked at both answers and found them equal produces. Two ways to reach it, both
        # silent: the judge emitted nothing parseable for a side, or that side had no judgments
        # at all. Every such pair then entered the effect estimate as evidence of no difference,
        # so the measured effect was diluted 1:1 with the judge's parse-failure rate and the
        # dilution grew as the instrument got worse.
        #
        # None means "this pair was not judged" and is dropped downstream, which is the only
        # reading that does not let an instrument failure masquerade as a finding.
        sign, mag = None, None
    else:
        d = ma - mb
        sign = 1 if d > tol else -1 if d < -tol else 0
        mag = abs(d)
    return Judgment(
        judgment_id=h(
            "paired", suite_id, task_id, criterion, run_id_a, run_id_b, order, judge_prompt_sha
        )[:32],
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        suite_id=suite_id,
        task_id=task_id,
        criterion=criterion,  # type: ignore[arg-type]
        order=order,
        judge_family=judge_family,
        judge_model=judge_model,
        judge_prompt_sha=judge_prompt_sha,
        pref_sign=sign,
        magnitude=mag,
        len_a_words=len_a_words,
        len_b_words=len_b_words,
        retest_group_id=retest_group_id,
        judge_pin=f"{judge_model}@{judge_prompt_sha[:12]}",
    )


def both_orders(grade: Grader, **meta: object) -> tuple[Judgment, Judgment]:
    """Run an absolute grader in both presentation orders. Feed the result to resolve_pair().

    `grade(order)` must GRADE, not merely relabel: it is called once per order so that an
    order-sensitive implementation produces two different verdicts and is caught, which is the
    entire point of calling it twice.
    """
    out: list[Judgment] = []
    for order in ("ab", "ba"):
        ja, jb = grade(order)  # type: ignore[arg-type]
        out.append(
            paired_from_absolute(
                a=ja,
                b=jb,
                order=order,  # type: ignore[arg-type]
                **meta,  # type: ignore[arg-type]
            )
        )
    return out[0], out[1]


def rating_unit_id(j: Judgment) -> str:
    """The UNIT two raters must be compared on: one task, one criterion, one key point.

    Key-point id is part of the identity: two judges that agree on a task's overall recall
    while disagreeing about which points were supported are not in agreement, and an alpha
    computed over task-level means would report that they are.
    """
    return f"{j.suite_id}/{j.task_id}/{j.criterion}/{j.key_point_id or '-'}"


def ratings_by_family(judgments: Iterable[Judgment]) -> dict[str, dict[str, str]]:
    """unit_id -> {judge_family: label}, ready for krippendorff_alpha_nominal().

    Scores are rendered as labels when a judgment carries no label (the quality rubric), so a
    nominal alpha over a 0-10 scale treats 6 and 7 as fully disagreeing. That is the
    conservative reading and it is deliberate: the ordinal alpha is not implemented here, and
    silently substituting a nominal one for it would overstate agreement.
    """
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for j in judgments:
        label = j.label if j.label is not None else (f"{j.score:g}" if j.score is not None else "")
        if label:
            out[rating_unit_id(j)][j.judge_family] = label
    return dict(out)
