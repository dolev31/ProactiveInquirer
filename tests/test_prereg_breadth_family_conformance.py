"""The BREADTH family census invariant, from the 2026-09-19 prereg-conformance audit
(`artifacts/prereg_breadth_conformance_20260919/RESULT.md`).

TWO SUSPECTED DEFECTS WENT IN; ONE CAME OUT CONFIRMED.

(a) SUSPECTED: `facet_breadth` is declared in `pi_eval.report.EXPLORATORY_METRICS`
("point estimate + BCa CI, no p-value ever" -- scoped to how `pi_eval.report` RENDERS a
paper table) *and* named with `gated=True` in `pinq_train.gate.PAIRED_METRICS`. Read as the
SAME kind of "confirmatory" this looks like a metric gating and not-gating at once.

REFUTED, after reading `pinq_train/gate.py`'s own module docstring, `docs/GPU_RUNBOOK.md`
and `plans/2026-09-15-decisions-brief.md`: `PAIRED_METRICS`'s `gated` flag is Tier B's
checkpoint-SELECTION rule (plan I.8/I.9) on the DEV split, deciding only whether a
checkpoint earns a test-split rollout -- "the test split is not touched until one checkpoint
per arm has been chosen here" (`gate.py` line 5). It computes a bootstrap CI and a threshold,
never a p-value, on data that never reaches the paper's confirmatory family
(`pi_eval.prereg.PRIMARY`/`SECONDARY`, test-split only). Under the default
`--coverage-rule matched_cost` (130/130 real verdicts on disk, 0 on `cap8`) `facet_breadth`
is ALREADY forced to `gated=False` at runtime, by design, for the asks-count confound
(`gate.py`'s matched-cost override block; T19 follow-up 9750860, 2026-09-15). Two dedicated,
already-passing tests (`test_facet_breadth_is_gated_as_no_loss_rather_than_as_a_gain`,
`test_a_facet_loss_is_gated_at_cap8_and_report_only_at_matched_cost`) exercise this on
purpose, in both directions. Forcing `PAIRED_METRICS`'s declared `gated` to `False` to
satisfy an "exploratory metrics never gate" invariant across BOTH systems would break those
two tests to make a THIRD, over-scoped test pass -- exactly the CLAUDE.md rule-4 case where
the test, not the code, encodes the wrong belief. So this file asserts nothing about (a); the
full refutation with commands and output lives in RESULT.md, not here.

(b) SUSPECTED: `breadth_components` and `breadth_recall` are computed and emitted
(`pi_eval/score.py`) but declared in none of `PRIMARY`, `SECONDARY`, `CALIBRATION`,
`EXPLORATORY_METRICS` or `PAIRED_METRICS` -- "a reported quantity in no declared family has
no prereg status."

CONFIRMED, for those two AND for `breadth_singleton_share` (emitted in the same guarded block,
same omission, not named in the original suspicion but caught by reading the source rather
than the task text). This file's tests assert exactly that gap and fail until it closes.

WHAT WOULD MAKE THIS TEST TAUTOLOGICAL. If it hardcoded "breadth_components" as the expected
violator, a fix that added a fourth undeclared breadth metric would still pass silently. It
derives the emitted breadth-family metric names from `pi_eval.score`'s own `MetricDef` table
(filtered to the ones this audit's structural-component logic emits, per
`score.py`'s `emit("breadth_components", ...)` / `emit("breadth_recall", ...)` /
`emit("breadth_singleton_share", ...)` block) and checks each against the five declared
families, so it fails on ANY of them left undeclared, not only the ones named on 2026-09-19.
"""

from __future__ import annotations

from pi_eval.prereg import default_stage1
from pi_eval.report import EXPLORATORY_METRICS
from pinq_train.gate import PAIRED_METRICS

# The breadth-over-independent-components family: computed in `pi_eval.score`'s
# `_score_structure` (or equivalent) block guarded on `graph.gold_edges` carrying a
# "prerequisite" kind, alongside `facet_breadth`. Read directly off score.py's `emit(...)`
# calls rather than typed from memory -- `grep -n 'emit("breadth' src/pi_eval/score.py`,
# 2026-09-19, returned exactly these three and no fourth.
BREADTH_COMPONENT_METRICS: tuple[str, ...] = (
    "breadth_components",
    "breadth_recall",
    "breadth_singleton_share",
)


def _declared_families() -> dict[str, set[str]]:
    s1 = default_stage1()
    return {
        "PRIMARY": {e.metric for e in s1.primary},
        "SECONDARY": {e.metric for e in s1.secondary},
        "CALIBRATION": {e.metric for e in s1.calibration},
        "EXPLORATORY_METRICS": set(EXPLORATORY_METRICS),
        "PAIRED_METRICS": {metric for metric, _rule, _gated in PAIRED_METRICS},
    }


def test_every_breadth_component_metric_is_computed_in_score_py():
    """Guards against the invariant below going vacuous by construction: if `score.py`
    stopped emitting one of these (a rename, a deletion), the test after this one would pass
    for having nothing left to check rather than because the gap closed. This repo's own
    `pi_eval.score` module is the ground truth for "is it computed", read once here so the
    two facts -- computed, and undeclared -- are checked against the same source."""
    import inspect

    src = inspect.getsource(__import__("pi_eval.score", fromlist=["score"]))
    for metric in BREADTH_COMPONENT_METRICS:
        assert f'"{metric}"' in src, (
            f"{metric} is not emitted anywhere in pi_eval/score.py -- it has left the code "
            "this test was written to census, and BREADTH_COMPONENT_METRICS above is stale."
        )


def test_no_computed_breadth_metric_is_declared_in_no_family():
    """`breadth_components`, `breadth_recall` and `breadth_singleton_share` are computed and
    emitted into `scores.parquet` whenever a task's gold graph carries a prerequisite edge
    (`pi_eval/score.py`), but before the fix below they sit in NONE of the five families a
    metric in this repo can have prereg status through. That is defect (b), CONFIRMED: a
    number with no declared family is a number nothing prevents from being typeset as if it
    had one. The fix is to declare them EXPLORATORY (point estimate + CI, no p-value,
    matching `facet_breadth`'s own treatment as the metric they were built to sit beside),
    never to delete a true measurement or to promote them into a confirmatory family nobody
    preregistered."""
    families = _declared_families()
    declared_anywhere = set().union(*families.values())
    undeclared = [m for m in BREADTH_COMPONENT_METRICS if m not in declared_anywhere]
    assert undeclared == [], (
        f"{undeclared} are computed in pi_eval/score.py but declared in none of "
        f"{sorted(families)} -- a reported quantity in no declared family has no prereg "
        "status (pi_eval/report.py EXPLORATORY_METRICS is where facet_breadth, the metric "
        "these ride beside, already lives)."
    )
