"""A run with no question text is SKIPPED and counted -- unless its suite carries a
judge-derived endpoint, in which case judging still refuses.

The guard was right to exist and too coarse to use. Every judge takes the question, so
grading against an empty one measures a different instrument -- but the check refused the
WHOLE run when any run lacked a question, and on this repository that meant 78 stale
`synth` runs blocked judging drgym, the only suite with a judge-derived preregistered
endpoint (kpr_incremental, the P3 primary).

Measured: of 379 ok runs, the 78 without question text were all synth -- 65 recorded before
`corpus_dir` existed and 15 pointing at corpus e937e83d6c2b6e7a, which is no longer on disk
(only d13f5f4b7a52088d survives). Their corpus is gone; they can never be judged, and no
preregistered judge-derived endpoint reads synth.

So: skip them, COUNT them, name them -- the same rule `collect_rows` already applies ("a run
that cannot be exported is COUNTED, not hidden"). And keep the hard refusal exactly where it
protects a number: a suite whose endpoint IS judge-derived must never be judged over a
silently smaller denominator.
"""

from __future__ import annotations


def test_judge_derived_suites_are_identified_from_the_prereg() -> None:
    from pi_eval.score import judge_derived_suites

    s = judge_derived_suites()
    assert "drgym" in s, "P3's kpr_incremental is judge-derived"
    assert "synth" not in s, "the calibration suite carries no judge-derived endpoint"


def test_a_missing_question_on_a_non_judged_suite_is_skippable() -> None:
    from pi_eval.score import question_gap_verdict

    skip, err = question_gap_verdict({"synth": 78}, total=379)
    assert skip == 78
    assert err is None, "stale calibration runs must not block a primary endpoint"


def test_a_missing_question_on_a_JUDGED_suite_still_refuses() -> None:
    """The case the guard exists for: P3 computed over a silently smaller denominator."""
    from pi_eval.score import question_gap_verdict

    skip, err = question_gap_verdict({"drgym": 3}, total=300)
    assert err is not None and "drgym" in err


def test_a_mixed_gap_refuses_because_of_the_judged_suite() -> None:
    from pi_eval.score import question_gap_verdict

    _, err = question_gap_verdict({"synth": 78, "drgym": 1}, total=379)
    assert err is not None and "drgym" in err
    assert "synth" not in err.split("drgym")[0], "the refusal must name the suite that matters"


def test_no_gap_is_no_verdict() -> None:
    from pi_eval.score import question_gap_verdict

    assert question_gap_verdict({}, total=300) == (0, None)


def test_the_refusal_message_still_explains_why() -> None:
    from pi_eval.score import question_gap_verdict

    _, err = question_gap_verdict({"drgym": 2}, total=100)
    assert "question" in err.lower()
    assert "corpus" in err.lower(), "the usual cause must stay in the message"
