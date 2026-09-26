"""The hedge clause is a MEASUREMENT ARTIFACT, and until it is measured it is invisible.

`answerer_frozen` tells the Answerer to say so "in a clause rather than guessing" where the
evidence does not support part of the task. MEASURED over 448 musique runs: 50.2% of answers
take that instruction, and every answer-quality metric collapses on them --

    committed (n=223)  token_f1 0.302  recall 0.368  exact_match 0.233
    hedged    (n=225)  token_f1 0.003  recall 0.018  exact_match 0.000

-- while `evidence_coverage` is if anything HIGHER on the hedged half (0.561 vs 0.489). So
hedging is not a rational response to missing evidence: the Answerer holds the gold spans and
declines to name them. Hedge rate ranges 16.7%-83.3% ACROSS ARMS, which makes every
answer-quality contrast a contrast in prompt compliance.

`answer_hedged` exists so that can never again be true without a number saying so.
"""

from __future__ import annotations

import pytest

from pi_eval.metrics import quality

COMMITTED = [
    "Hassan Gouled Aptidon",
    "2003",
    "the Dominican Republic",
    "18,705",
    "Moscow",
]

# Verbatim from runs on disk (runs/*/outcome.json), not invented for the test.
HEDGED = [
    "The evidence does not specify which Caribbean source contributed the most legal "
    "immigrants; the Dominican Republic, a top Caribbean source, is 18,705",
    "The Dominican Republic, though the evidence only lists it among the top Caribbean "
    "sources and does not confirm it was the highest.",
    "The evidence does not state the founding date.",
    "It is unclear from the provided paragraphs.",
    "The paragraphs provide no information about the first president.",
    "Cannot be determined from the evidence given.",
    "The evidence is insufficient to identify the season.",
]


@pytest.mark.parametrize("text", COMMITTED)
def test_committed_answers_are_not_hedged(text: str) -> None:
    assert quality.is_hedged(text) is False, text


@pytest.mark.parametrize("text", HEDGED)
def test_hedged_answers_are_detected(text: str) -> None:
    assert quality.is_hedged(text) is True, text


def test_empty_answer_is_not_hedged() -> None:
    """An empty answer is a different failure and must not be folded into this one."""
    assert quality.is_hedged("") is False


def test_hedge_detector_does_not_fire_on_a_gold_answer_containing_a_negation() -> None:
    """A real answer may legitimately contain 'not'. The detector keys on the REFUSAL
    idiom ("the evidence does not ..."), never on negation alone."""
    assert quality.is_hedged("Not Guilty") is False
    assert quality.is_hedged("The No. 1 Ladies' Detective Agency") is False


# ---------------------------------------------------------------- answer_correct


def test_contains_answer_credits_a_hedged_answer_that_still_names_the_answer() -> None:
    """The whole point: a refusal clause must not erase a correct answer.

    Verbatim from a run on disk. token_f1 scores this 0.10; the answer is nonetheless there.
    """
    text = (
        "The evidence does not specify which Caribbean source contributed the most legal "
        "immigrants; the Dominican Republic, a top Caribbean source, is 18,705"
    )
    assert quality.contains_answer(text, "18,705") == 1.0
    assert quality.contains_answer(text, "the Dominican Republic") == 1.0


def test_contains_answer_is_zero_when_the_answer_is_absent() -> None:
    assert quality.contains_answer("Richard Allen", "Bishop Francis Asbury") == 0.0
    assert quality.contains_answer("Ecuador", "the Dominican Republic") == 0.0


def test_contains_answer_matches_aliases() -> None:
    assert quality.contains_answer("Born in Bombay.", "Mumbai", ("Bombay",)) == 1.0


def test_contains_answer_ignores_articles_and_case_like_token_f1_does() -> None:
    """Shares `normalize` with the F1 family: two normalisations would disagree silently."""
    assert quality.contains_answer("the DOMINICAN republic", "Dominican Republic") == 1.0


def test_contains_answer_requires_a_token_boundary() -> None:
    """Substring matching would credit 'Austria' for gold 'Australia'-style overlaps and
    every short numeric gold ('18' inside '1985'). Matching is over token sequences."""
    assert quality.contains_answer("He lived in Austrian lands", "Austria") == 0.0
    assert quality.contains_answer("Founded in 1985", "18") == 0.0


def test_contains_answer_empty_gold_is_zero_not_one() -> None:
    """`token_f1('', '')` is 1.0 by its own contract; a containment metric must not inherit
    that, or every task with no gold answer would score a free success."""
    assert quality.contains_answer("anything", "") == 0.0
    assert quality.contains_answer("", "Moscow") == 0.0


# ---------------------------------------------------------------- the prompt itself


def test_answerer_prompt_asks_for_the_answer_before_any_hedge() -> None:
    """The refusal must be an ADDITION to an answer, never a substitute for one.

    The prompt used to end "Where the evidence does not support a part of the task, say so in
    a clause rather than guessing", full stop. MEASURED on runs that had retrieved ALL gold
    evidence: 45.6% took the clause and named nothing, and every one of them scored 0 on
    token_f1 and on answer_correct. Refusal is correct when the evidence is genuinely absent;
    it is not correct with the gold spans in hand.

    This pins the ORDERING invariant -- answer first, caveat after -- rather than the wording,
    so the prompt can be reworded without silently losing the property.
    """
    from pinq import promptlib

    text = promptlib.load("answerer_frozen").lower()
    # the anti-guessing intent is preserved, not deleted
    assert "guess" in text, "the anti-hallucination instruction was dropped, not fixed"
    # and the answer is demanded first
    assert "first" in text, "nothing requires the answer to precede the caveat"
    hedge_at = min(
        (text.index(w) for w in ("unsupported", "does not", "not supported") if w in text),
        default=-1,
    )
    answer_at = text.index("first")
    assert hedge_at > answer_at, (
        "the caveat instruction precedes the answer instruction; the measured failure mode "
        "is the model emitting the caveat INSTEAD of the answer"
    )
