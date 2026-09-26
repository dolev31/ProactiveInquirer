"""The four rates that are legal on a conversation log, and the one number that is not a claim.

Every test here is about a refusal. The metrics themselves are ratios of counts; what makes
them correct is what they decline to count.
"""

import math

import pytest

from pi_eval.metrics.convlog import (
    B1_LABELS,
    B2_LABELS,
    B3_LABELS,
    IllegalComparison,
    anticipation_miss_rate,
    compare_followups,
    followups_per_task,
    necessary_ask_precision,
    over_action_rate,
    wasted_ask_rate,
)


def rec(kind, label, dp="d0", item=None):
    return {"item_id": item or f"{kind}-{label}-{dp}", "kind": kind, "label": label, "dp_id": dp}


def test_label_vocabularies_are_closed():
    assert B1_LABELS == (
        "necessary",
        "answer_in_state",
        "answer_inferable",
        "default_existed",
        "cant_tell",
    )
    assert B2_LABELS == (
        "stated_already",
        "inferable_from_state",
        "user_private",
        "new_task",
        "cant_tell",
    )
    assert B3_LABELS == (
        "stop_was_right",
        "should_have_asked",
        "should_have_continued",
        "should_have_stopped_earlier",
        "cant_tell",
    )


def test_wasted_ask_rate_counts_the_three_wasted_verdicts():
    recs = [
        rec("B1", "necessary", "d1"),
        rec("B1", "answer_in_state", "d2"),
        rec("B1", "answer_inferable", "d3"),
        rec("B1", "default_existed", "d4"),
    ]
    r = wasted_ask_rate(recs)
    assert r.value == pytest.approx(0.75)
    assert r.n == 4


def test_cant_tell_leaves_both_the_numerator_and_the_denominator():
    """An undecided unit is not a decided negative. Putting it in the denominator would let an
    annotator lower the rate by giving up."""
    recs = [rec("B1", "necessary", "d1"), rec("B1", "cant_tell", "d2")]
    r = wasted_ask_rate(recs)
    assert r.n == 1
    assert r.value == pytest.approx(0.0)


def test_absent_is_not_zero():
    """The same rule ADR follows: with no labels of this kind the metric emits no number, so a
    table can print NOT RUN rather than a plausible 0.0."""
    for fn in (wasted_ask_rate, necessary_ask_precision, anticipation_miss_rate, over_action_rate):
        r = fn([])
        assert math.isnan(r.value)
        assert r.n == 0


def test_necessary_ask_precision_is_the_complement_on_decided_asks():
    recs = [rec("B1", "necessary", "d1"), rec("B1", "answer_in_state", "d2")]
    assert necessary_ask_precision(recs).value == pytest.approx(0.5)


def test_anticipation_miss_counts_only_what_the_state_could_have_yielded():
    """`user_private` and `new_task` are items no policy could have anticipated from the state.
    Counting them as misses would charge the agent for not reading a mind."""
    recs = [
        rec("B2", "stated_already", "d1"),
        rec("B2", "inferable_from_state", "d2"),
        rec("B2", "user_private", "d3"),
        rec("B2", "new_task", "d4"),
    ]
    r = anticipation_miss_rate(recs)
    assert r.value == pytest.approx(0.5)
    assert r.n == 4


def test_over_action_rate_is_about_interrupts_only():
    recs = [
        rec("B3", "should_have_asked", "d1"),
        rec("B3", "stop_was_right", "d2"),
        rec("B3", "should_have_continued", "d3"),
        rec("B3", "cant_tell", "d4"),
    ]
    r = over_action_rate(recs)
    assert r.n == 3
    assert r.value == pytest.approx(1 / 3)


def test_an_unknown_label_raises_rather_than_being_ignored():
    """A vocabulary drift that silently drops rows shrinks every denominator and moves every
    rate, with nothing in the output saying so."""
    with pytest.raises(ValueError, match="not_a_label"):
        wasted_ask_rate([rec("B1", "not_a_label")])


def test_followups_per_task_is_marked_uncomparable():
    d = followups_per_task([3, 1, 2])
    assert d.value == pytest.approx(2.0)
    assert d.comparable_across_arms is False
    assert "counterfactual" in d.why_not.lower()


def test_comparing_followups_across_arms_raises():
    """The number exists and is worth printing; the COMPARISON is the illegal act, because the
    user turns a different policy would have produced do not exist in any log."""
    with pytest.raises(IllegalComparison):
        compare_followups(followups_per_task([3, 1]), followups_per_task([1, 1]))
