"""The paraphrase label, which is the only thing standing between the loss and eight
renderings of one question.

The three buckets are pinned on real candidate text taken from the corpus, not on invented
strings: the threshold is a judgement call and the test is what records which judgement was
made.
"""

from __future__ import annotations

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.distinctness import (
    content_tokens,
    distinctness,
    jaccard,
    question_of,
)

# Three of eight candidates sampled at one musique state, in the near-paraphrase group.
NAVY_A = "Which country has the world's oldest navy and what is its primary language?"
NAVY_B = "Which country possesses the world's oldest navy and what is the primary language?"


def test_two_renderings_of_one_question_are_a_paraphrase():
    j, bucket = distinctness(ask_action_json(NAVY_A), ask_action_json(NAVY_B))
    assert bucket == "paraphrase" and j >= 0.70


def test_two_questions_about_different_needs_are_different():
    j, bucket = distinctness(
        ask_action_json("who founded the company?"),
        ask_action_json("what year did the treaty end?"),
    )
    assert bucket == "different" and j < 0.40


def test_same_subject_different_attribute_is_related():
    """WAS "oldest navy" against "largest merchant fleet", asserted to be `related`. That
    belief was wrong: those two share only the token `country` and score 0.125, which is
    correctly `different` -- they pursue different needs. `related` is the narrower case of
    the SAME subject with a different attribute, which is what this now pins."""
    j, bucket = distinctness(
        ask_action_json("which country has the world's oldest navy?"),
        ask_action_json("which country has the world's largest navy?"),
    )
    assert bucket == "related" and 0.40 <= j < 0.70


def test_a_different_need_with_one_shared_noun_is_still_different():
    j, bucket = distinctness(
        ask_action_json("which country has the world's oldest navy?"),
        ask_action_json("which country has the largest merchant fleet today?"),
    )
    assert bucket == "different" and j < 0.40


def test_stopwords_do_not_floor_the_score():
    """Without the stopword list any two English questions share "the/of/is" and score high,
    which would call genuinely different questions paraphrases."""
    assert content_tokens("What is the name of the city?") == frozenset({"name", "city"})


def test_two_empty_questions_are_the_same_question_not_maximally_different():
    """Scoring them 0.0 would promote the most degenerate pairs to the top of any
    diversity-sorted selection."""
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_a_stop_parses_to_an_empty_question_rather_than_raising():
    assert question_of(STOP_ACTION_JSON) == ""
    assert question_of("not json at all") == ""
    j, bucket = distinctness(STOP_ACTION_JSON, ask_action_json("who founded it?"))
    assert bucket == "different" and j == 0.0
