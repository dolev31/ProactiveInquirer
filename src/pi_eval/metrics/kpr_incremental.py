"""Key-point recall, RAW and INCREMENTAL. The DeepResearchGym primary endpoint (P3).

WHAT IS COUNTED. The gold for a DRGym task is the benchmark authors' aggregated key-point
list (`key_point/<id>_aggregated.json`, one required GoldNode per point; see
pi_eval.build.drgym_build). The KPR judge labels each (report, key point) pair Supported,
Omitted or Contradicted. Recall is the Supported share -- but "share of WHAT" is the whole
question, and the two defensible answers are different numbers:

    raw KPR          = Supported JUDGMENTS / all parsed judgments
    incremental KPR  = key points with >= 1 Supported verdict / key points judged

WHY THE INCREMENTAL FORM IS THE ENDPOINT. The judgments are not one-per-key-point by
construction. A key point picks up several verdicts whenever the harness is doing the thing
it is supposed to do: both presentation orders are graded, a test-retest replicate is judged,
a second judge family grades the same unit, or a claim-driven pass checks each of the
report's claims against the point it covers. Take a run that accumulated 22 key-point
verdicts:

    report A -- 20 Supported verdicts that all name the SAME key point, then Omitted on two
                others:  raw = 20/22 = 0.909,  incremental = 1/3 = 0.333
    report B -- 20 different key points Supported once each, then Omitted on two others:
                raw = 20/22 = 0.909,  incremental = 20/22 = 0.909

Raw cannot tell restating one point from covering twenty; on a suite whose entire question is
"did the system discover the things it did not know to look for", that is the one distinction
which has to survive. Incremental credits a key point ONCE however many verdicts cover it, so
it separates the two reports by a factor of 2.7.

BOTH ARE EMITTED. Raw is kept because it is the number comparable to the published upstream
figure (which grades exactly once per point, where the two coincide), and because raw minus
incremental IS the repetition diagnostic: `repetition_gap` is zero exactly when every judged
key point received exactly one verdict.

THE DENOMINATOR IS THE JUDGED POINTS, NOT THE GOLD POINTS. A key point whose verdict could
not be parsed is UNJUDGED, and scoring it as unsupported would charge the system for the
judge's stutter. It is excluded from the denominator and counted in `n_unjudged`, which is
reported next to the metric -- a missing measurement must look missing. `n_gold` is carried
alongside so the shrinkage is visible rather than inferred.

A CONTRADICTION IS NOT PARTIAL CREDIT, and it is also not an omission. It scores 0 in both
recall forms and is counted separately: "this system contradicted 8% of the key points" and
"this system missed them" are different findings, and collapsing them at metric time makes
the second unrecoverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from pi_eval.judges.types import Judgment

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
OMITTED = "Omitted"

NAN = float("nan")


@dataclass(frozen=True, slots=True)
class KprResult:
    """Both recall forms plus everything needed to read them honestly."""

    raw: float
    incremental: float
    n_gold: int  # key points in the gold graph
    n_points_judged: int  # distinct key points carrying >= 1 parsed verdict
    n_points_supported: int  # distinct key points with >= 1 Supported verdict
    n_points_contradicted: int  # distinct key points contradicted and never supported
    n_judgments: int  # judgment rows consumed
    n_unjudged: int  # gold points with no parsed verdict at all

    @property
    def contradiction_rate(self) -> float:
        """Share of JUDGED key points the report contradicts. NaN when nothing was judged."""
        if not self.n_points_judged:
            return NAN
        return self.n_points_contradicted / self.n_points_judged

    @property
    def repetition_gap(self) -> float:
        """raw - incremental. Zero exactly when every judged key point got one verdict.

        Positive means the judgment multiset over-weights points that attracted several
        Supported verdicts, which is precisely what incremental counting removes.
        """
        if math.isnan(self.raw) or math.isnan(self.incremental):
            return NAN
        return self.raw - self.incremental

    def as_dict(self) -> dict[str, float]:
        """The metric names this result contributes to `scores.parquet`."""
        return {
            "kpr_incremental": self.incremental,
            "keypoint_recall": self.raw,
            "keypoint_contradiction_rate": self.contradiction_rate,
            "kpr_repetition_gap": self.repetition_gap,
        }


def keypoint_judgments(judgments: Iterable[Judgment]) -> tuple[Judgment, ...]:
    """The key-point verdicts, and only those.

    Filtered on BOTH criterion and a present label: a Judgment carrying `criterion="keypoint"`
    with `label=None` is a row the parser refused, not a verdict, and letting it through would
    put an unlabelled item in the denominator.
    """
    return tuple(j for j in judgments if j.criterion == "keypoint" and j.label is not None)


def by_key_point(judgments: Iterable[Judgment]) -> dict[str, list[str]]:
    """key_point_id -> the labels it received, in arrival order.

    A judgment with no `key_point_id` cannot be attributed to a point and is DROPPED here
    rather than pooled under a shared empty key, which would merge unrelated points into one
    unit and let a single Supported verdict credit all of them.
    """
    out: dict[str, list[str]] = {}
    for j in keypoint_judgments(judgments):
        if not j.key_point_id:
            continue
        out.setdefault(j.key_point_id, []).append(str(j.label))
    return out


def raw_recall(judgments: Iterable[Judgment]) -> float:
    """Supported judgments / all parsed judgments. NaN on an empty set, never 0.0.

    Zero would read as "the system supported nothing"; NaN reads as "nothing was judged".
    """
    js = keypoint_judgments(judgments)
    if not js:
        return NAN
    return sum(1 for j in js if j.label == SUPPORTED) / len(js)


def incremental_recall(
    judgments: Iterable[Judgment], gold_key_point_ids: Sequence[str] = ()
) -> float:
    """Key points with >= 1 Supported verdict / key points judged. NaN when nothing was judged.

    `gold_key_point_ids` is optional and bounds BOTH counts to points the gold graph actually
    contains: a judgment naming a key point the gold does not have is a join bug, and
    silently crediting it would inflate recall out of a typo.
    """
    return score(judgments, gold_key_point_ids).incremental


def score(judgments: Iterable[Judgment], gold_key_point_ids: Sequence[str] = ()) -> KprResult:
    """Both recall forms and their diagnostics, from one run's key-point judgments."""
    known = set(gold_key_point_ids)
    labels = by_key_point(judgments)
    if known:
        labels = {k: v for k, v in labels.items() if k in known}
    js = [
        j
        for j in keypoint_judgments(judgments)
        if j.key_point_id and (not known or j.key_point_id in known)
    ]

    supported = {k for k, v in labels.items() if SUPPORTED in v}
    contradicted = {k for k, v in labels.items() if CONTRADICTED in v} - supported
    n_judged = len(labels)
    n_gold = len(known) if known else n_judged
    return KprResult(
        raw=(sum(1 for j in js if j.label == SUPPORTED) / len(js)) if js else NAN,
        incremental=(len(supported) / n_judged) if n_judged else NAN,
        n_gold=n_gold,
        n_points_judged=n_judged,
        n_points_supported=len(supported),
        n_points_contradicted=len(contradicted),
        n_judgments=len(js),
        n_unjudged=max(0, n_gold - n_judged),
    )


def by_run(
    judgments: Iterable[Judgment], gold_key_point_ids: Mapping[str, Sequence[str]] | None = None
) -> dict[str, KprResult]:
    """run_id -> KprResult. Absolute key-point judgments carry the run in `run_id_a`.

    `gold_key_point_ids` is keyed by TASK id, because the gold key-point list belongs to the
    task and every arm's run of that task is scored against the same list.
    """
    grouped: dict[str, list[Judgment]] = {}
    tasks: dict[str, str] = {}
    for j in keypoint_judgments(judgments):
        grouped.setdefault(j.run_id_a, []).append(j)
        tasks[j.run_id_a] = j.task_id
    gold = gold_key_point_ids or {}
    return {
        run_id: score(js, gold.get(tasks.get(run_id, ""), ()))
        for run_id, js in sorted(grouped.items())
    }
