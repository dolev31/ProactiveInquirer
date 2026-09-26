"""The diversity gate now measures WITHIN-TASK variation, because pooled distinct-3 rejected a
perfect policy.

THE MEASUREMENT THAT SETTLED IT. Four samples, all musique, scored on four candidate metrics:

    sample                 n   pooled d3   within-task d3   cross-task%   uniq%
    GOLD (perfect)      2660      0.2250           0.9985         83.1%   29.8%
    ours                2254      0.4545           0.7787         12.9%   79.2%
    semi-collapsed      2254      0.0028           0.6033        100.0%    0.4%
    COLLAPSED           2254      0.0004           0.2571        100.0%    0.0%

GOLD is every required gold sub-question -- a policy asking exactly those is perfectly
state-dependent BY CONSTRUCTION. It scores 0.2250 pooled: below our own export, and far below
the 0.55 floor. So pooled distinct-3 does not merely set the bar too high, it is ANTI-CORRELATED
with quality at the top end -- it measures how formulaic the question SURFACE is, and gold
questions are extremely formulaic ("The Collegian >> owned by"). `cross-task%` and `uniq%` fail
the same way, both scoring gold badly.

Within-task distinct-3 is monotone across all four and ranks gold highest, which is what an
instrument is supposed to do. It also asks the question the gate's own error message asks --
"did the questions stop depending on the state?" -- because varying WITHIN a task is exactly
state-dependence, while repeating ACROSS runs of one task is convergence on the right question.

THE FLOOR is set from those baselines, not from taste: 0.65 sits above semi-collapsed (0.603,
ten canned templates regardless of state -- collapse, and must fail) and below our export
(0.779) and gold (0.999).
"""

from __future__ import annotations

import pytest

from pinq_train.rung1_sft.collapse import (
    WITHIN_TASK_FLOOR,
    ModeCollapse,
    assert_diverse,
    within_task_distinct_n,
)


def _pairs(mapping):
    return [(t, q) for t, qs in mapping.items() for q in qs]


def test_a_single_canned_question_collapses() -> None:
    d = within_task_distinct_n(_pairs({f"t{i}": ["tell me more"] * 4 for i in range(20)}))
    assert d < 0.30


def test_varying_within_a_task_scores_high() -> None:
    d = within_task_distinct_n(
        _pairs(
            {
                f"t{i}": [f"who founded {w} in {i}?" for w in "alpha beta gamma delta".split()]
                for i in range(20)
            }
        )
    )
    assert d > WITHIN_TASK_FLOOR


def test_repeating_ACROSS_tasks_is_not_penalised() -> None:
    """Two runs of one task converging on the right question is convergence, not collapse --
    and it is what dragged the pooled statistic under the floor."""
    same_q_everywhere = _pairs(
        {f"t{i}": [f"who founded it in {i}?", f"when did {i} open?"] for i in range(20)}
    )
    reused = _pairs({f"t{i}": ["who founded it?", "when did it open?"] for i in range(20)})
    assert within_task_distinct_n(reused) == pytest.approx(
        within_task_distinct_n(same_q_everywhere), abs=0.35
    ), "cross-task reuse moved a WITHIN-task metric"


def test_a_task_with_one_question_contributes_nothing_rather_than_zero() -> None:
    """One question cannot be more or less varied than itself. Scoring it 0 would punish a
    dataset for having short episodes."""
    a = within_task_distinct_n(_pairs({"t1": ["q one two three", "q four five six"]}))
    b = within_task_distinct_n(
        _pairs({"t1": ["q one two three", "q four five six"], "t2": ["solo"]})
    )
    assert a == pytest.approx(b)


def test_an_empty_sample_is_refused_not_passed() -> None:
    """'An empty sample passes every diversity test ever written.'"""
    with pytest.raises(ModeCollapse):
        assert_diverse([])


def test_assert_diverse_still_rejects_real_collapse() -> None:
    with pytest.raises(ModeCollapse):
        assert_diverse(["tell me more"] * 60, task_ids=[f"t{i // 3}" for i in range(60)])


def test_assert_diverse_accepts_a_state_dependent_sample() -> None:
    qs, tids = [], []
    for i in range(20):
        for w in "alpha beta gamma delta".split():
            qs.append(f"who founded {w} in year {i}?")
            tids.append(f"t{i}")
    assert_diverse(qs, task_ids=tids)  # must not raise
