"""Gate B: whether the B labels are usable, and if not, which criterion failed.

WHY A GATE AND NOT A JUDGEMENT CALL. The A campaign's findings are the argument. One model
caught 99 of 99 planted foils and was still quarantined, because it was answering the SLOT
rather than the content. Another instrument's automatic margin turned out to measure retrieval
novelty rather than question quality, which would have trained the policy to ask vaguer
questions. Neither was visible without computing the diagnostics first and reading them before
believing any headline.

FOUR CRITERIA, EACH ABLE TO KILL THE STAGE.

  foil catch   a rater that misses items built to be answerable from the state is not reading
               the state, and its verdicts on the real items mean nothing;
  agreement    alpha per unit kind, ACROSS rater families. Two runs of one model agreeing says
               the model is deterministic, not that the instrument is legible;
  base rates   if nothing is wasted and nothing is missed there is no signal to train on --
               the logged agent was already doing the right thing, which is a finding and not
               a dataset;
  coverage     how many shown items produced a decided verdict at all. A rater that answers
               `cant_tell` everywhere passes agreement trivially.

ABSENT IS NOT ZERO, and it is not a pass either. A rater shown no foil has not passed the foil
check, it has not taken it, so it gets no score rather than a perfect one. An empty record set
fails rather than vacuously passing.

FOILS ARE EXCLUDED FROM THE BASE RATES. A foil is an item constructed to have a known answer;
counting it measures the plant rather than the corpus.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pi_eval.judges.paired import krippendorff_alpha_nominal
from pi_eval.metrics.convlog import (
    Rate,
    anticipation_miss_rate,
    necessary_ask_precision,
    over_action_rate,
    wasted_ask_rate,
)

# Thresholds, written down here rather than passed in, so a run cannot quietly clear a gate by
# choosing its own bar. Each is the plan's committed value.
MIN_FOIL_CATCH = 0.9
MIN_ALPHA = 0.6
MIN_WASTED_ASK_RATE = 0.15
MIN_MISS_RATE = 0.10
MIN_COVERAGE = 0.5
# OUTLIER RULE, fixed in code so it cannot be chosen after seeing which answer it gives. A
# rater whose base rate on a unit kind's target class sits more than this many median absolute
# deviations from the raters' median is NAMED. It is not dropped: excluding a rater is a
# decision a person makes and writes down -- the A campaign quarantined a model that caught 99
# of 99 foils, and did it on a stated statistic with the records kept as evidence.
OUTLIER_MADS = 2.0
MIN_RATERS_FOR_OUTLIER = 3

# The class each unit kind's base rate is about, which is what an outlier is an outlier ON.
_TARGET_CLASS: dict[str, frozenset[str]] = {
    "B1": frozenset({"answer_in_state", "answer_inferable", "default_existed"}),
    "B2": frozenset({"stated_already", "inferable_from_state"}),
    "B3": frozenset({"should_have_asked"}),
}

# A FOIL IS SCORED AGAINST THE CLASS, NOT THE WORD. The plant guarantees the answer is
# available from the state; which of the two "available" labels a rater reaches for is a
# vocabulary preference, and `metrics.convlog` already pools them into one numerator for the
# same reason. Measured on the pilot: 5 of one model's 6 misses were the adjacent label, so the
# exact-match check was reporting "did not read the state" for a rater that had. Both numbers
# are reported, so this can never hide a real miss.
_FOIL_CLASS: dict[str, frozenset[str]] = {
    "answer_in_state": frozenset({"answer_in_state", "answer_inferable", "default_existed"}),
    "stated_already": frozenset({"stated_already", "inferable_from_state"}),
}

_RATE_FNS = {
    "wasted_ask_rate": wasted_ask_rate,
    "necessary_ask_precision": necessary_ask_precision,
    "anticipation_miss_rate": anticipation_miss_rate,
    "over_action_rate": over_action_rate,
}


@dataclass(frozen=True, slots=True)
class GateB:
    foil_catch: dict[str, float]
    # The same measurement with exact-label matching. Reported beside the class-based one so a
    # reader can see what the relaxation bought and what it might have hidden.
    foil_catch_exact: dict[str, float]
    alpha: dict[str, float]
    rates: dict[str, Rate]
    coverage: dict[str, float]
    # Per unit kind, the raters whose base rate is an outlier from the others'. Named, never
    # excluded here.
    outliers: dict[str, list[str]]
    n_records: int
    n_raters: int
    passed: bool
    failures: tuple[str, ...] = field(default=())


def _labels_of(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(unit_id, unit_kind, label) for one record. Mirrors `annotate._units` for the B family
    without importing it: this runs over raw records, before any consensus exists."""
    tt = str(record.get("task_type") or "")
    iid = str(record.get("item_id") or "")
    resp = record.get("response") or {}
    if tt in ("B1", "B1b", "B3"):
        v = resp.get("verdict")
        # B1b IS REPORTED AS B1. `annotate._units` does the same, so the variant and the thing
        # it controls land in one column; a reader that did not know the variant scored an
        # entire pass as zero on every criterion, which is indistinguishable from two raters
        # failing and was in fact every row being dropped.
        return [(iid, "B1" if tt == "B1b" else tt, str(v))] if v else []
    if tt == "B2":
        return [(f"{iid}/{k}", "B2", str(v)) for k, v in (resp.get("verdicts") or {}).items() if v]
    return []


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _outliers(rates: Mapping[str, float]) -> list[str]:
    """Raters more than OUTLIER_MADS median absolute deviations from the median rate.

    Median and MAD rather than mean and SD because with a handful of raters one extreme value
    drags a mean far enough to hide itself. A zero MAD means every rater agrees exactly, and
    then nobody is an outlier no matter how the arithmetic is written.
    """
    if len(rates) < MIN_RATERS_FOR_OUTLIER:
        return []
    med = _median(list(rates.values()))
    mad = _median([abs(v - med) for v in rates.values()])
    if mad <= 0:
        # ZERO MAD MEANS THE MAJORITY IS UNANIMOUS. Dividing by it reports nobody, which loses
        # exactly the least ambiguous case: on the targeted B3 slice `should_have_asked` came
        # back 0%, 0% and 18%, and the median and the MAD were both zero. Differing at all from
        # a unanimous majority is what being an outlier means.
        return sorted(a for a, v in rates.items() if v != med)
    return sorted(a for a, v in rates.items() if abs(v - med) > OUTLIER_MADS * mad)


def gate_b(
    bundle: Mapping[str, Any],
    key: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    families: Mapping[str, str] | None = None,
) -> GateB:
    """Compute every Gate B number, then say which criteria failed.

    `families` maps annotator_id -> family. Two raters in one family never contribute a pair to
    alpha; without the mapping each annotator_id is its own family, which is the right default
    for a pilot run with one model per id.
    """
    key_items = dict(key.get("items") or {})
    foil_expect = {k: v["expected"] for k, v in key_items.items() if v.get("foil")}

    # ---- foil catch, per rater. Only raters actually shown a foil are scored.
    seen: defaultdict[str, int] = defaultdict(int)
    caught: defaultdict[str, int] = defaultdict(int)
    caught_exact: defaultdict[str, int] = defaultdict(int)
    for r in records:
        iid, ann = str(r.get("item_id") or ""), str(r.get("annotator_id") or "")
        if iid in foil_expect:
            seen[ann] += 1
            expected = foil_expect[iid]
            verdicts = {lbl for _, _, lbl in _labels_of(r)}
            if expected in verdicts:
                caught_exact[ann] += 1
            if verdicts & _FOIL_CLASS.get(expected, frozenset({expected})):
                caught[ann] += 1
    foil_catch = {a: caught[a] / n for a, n in seen.items() if n}
    foil_catch_exact = {a: caught_exact[a] / n for a, n in seen.items() if n}

    # ---- base rates, over the real items only.
    real = [r for r in records if str(r.get("item_id") or "") not in foil_expect]
    flat = [{"kind": kind, "label": label} for r in real for _uid, kind, label in _labels_of(r)]
    rates = {name: fn(flat) for name, fn in _RATE_FNS.items()}

    # ---- agreement, across families only.
    fam = dict(families or {})
    ratings: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in real:
        ann = str(r.get("annotator_id") or "")
        f = fam.get(ann, ann)
        for uid, kind, label in _labels_of(r):
            ratings[kind].setdefault(uid, {})[f] = label
    alpha = {k: krippendorff_alpha_nominal(v) for k, v in ratings.items()}

    # ---- coverage: shown items that produced a decided verdict.
    decided: defaultdict[str, set[str]] = defaultdict(set)
    for r in real:
        for _uid, kind, label in _labels_of(r):
            if label != "cant_tell":
                decided[kind].add(str(r.get("item_id") or ""))
    by_kind_shown: defaultdict[str, set[str]] = defaultdict(set)
    for i in bundle.get("items") or ():
        if str(i["item_id"]) not in foil_expect:
            tt = str(i["task_type"])
            by_kind_shown["B1" if tt == "B1b" else tt].add(str(i["item_id"]))
    coverage = {
        k: (len(decided.get(k, set())) / len(v) if v else math.nan)
        for k, v in by_kind_shown.items()
    }

    # ---- outlier raters, by the fixed rule above.
    outliers: dict[str, list[str]] = {}
    for kind, target in _TARGET_CLASS.items():
        hits: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in real:
            ann = str(r.get("annotator_id") or "")
            for _uid, k, label in _labels_of(r):
                if k != kind or label == "cant_tell":
                    continue
                hits[ann][1] += 1
                hits[ann][0] += label in target
        rates_by_rater = {a: h / n for a, (h, n) in hits.items() if n}
        outliers[kind] = _outliers(rates_by_rater)

    # ---- the verdict, naming every criterion that failed.
    failures: list[str] = []
    if not records:
        failures.append("no records: the pass produced nothing to gate")
    if not foil_catch:
        failures.append("no rater was shown a planted foil, so the foil check did not run")
    for ann, v in sorted(foil_catch.items()):
        if v < MIN_FOIL_CATCH:
            failures.append(f"foil catch {v:.2f} < {MIN_FOIL_CATCH} for {ann}")
    for kind, v in sorted(alpha.items()):
        if math.isnan(v):
            failures.append(f"alpha for {kind} is absent: no unit was labelled by two families")
        elif v < MIN_ALPHA:
            failures.append(f"alpha {v:.2f} < {MIN_ALPHA} for {kind}")
    w, m = rates["wasted_ask_rate"], rates["anticipation_miss_rate"]
    if not math.isnan(w.value) and w.value < MIN_WASTED_ASK_RATE:
        failures.append(f"wasted-ask base rate {w.value:.2f} < {MIN_WASTED_ASK_RATE}")
    if not math.isnan(m.value) and m.value < MIN_MISS_RATE:
        failures.append(f"anticipation-miss base rate {m.value:.2f} < {MIN_MISS_RATE}")
    if math.isnan(w.value) and math.isnan(m.value):
        failures.append("neither base rate could be computed")
    for kind, v in sorted(coverage.items()):
        if not math.isnan(v) and v < MIN_COVERAGE:
            failures.append(f"coverage {v:.2f} < {MIN_COVERAGE} for {kind}")
    for kind, names in sorted(outliers.items()):
        for name in names:
            failures.append(
                f"{name} is a base-rate outlier on {kind} "
                f"(>{OUTLIER_MADS} MADs from the raters' median)"
            )

    return GateB(
        foil_catch=foil_catch,
        foil_catch_exact=foil_catch_exact,
        alpha=alpha,
        rates=rates,
        coverage=coverage,
        outliers=outliers,
        n_records=len(records),
        n_raters=len({str(r.get("annotator_id") or "") for r in records}),
        passed=not failures,
        failures=tuple(failures),
    )
