"""The LLM pass over the B instruments: prompt, and the refusal to read a reply that is wrong.

The prompt is the instrument. Two properties matter more than wording, and both are tested:
the prompt must show the state and the question WITHOUT showing what the person actually did
next (except for B2, whose unit IS an item of that turn), and the parser must refuse anything
it cannot read rather than defaulting a verdict. A defaulted verdict here does not bias a rate
slightly; it fabricates a judgment nobody made.
"""

import pytest
from tests.test_annotate_b_instruments import b1_item, b2_item, b3_item

from pi_eval.annotate_llm import AnnotationParseError, build_prompt, parse_reply

# --------------------------------------------------------------------------- prompts


def test_the_b1_prompt_shows_the_state_and_the_question():
    p = build_prompt(b1_item())
    assert "Add a retry to the uploader." in p
    assert "Which timeout should I use?" in p
    for label in (
        "necessary",
        "answer_in_state",
        "answer_inferable",
        "default_existed",
        "cant_tell",
    ):
        assert label in p


def test_the_b1_prompt_never_shows_what_the_person_replied():
    """B1 asks whether the question was worth asking BEFORE the reply existed. The reply is
    the answer to that question."""
    item = b1_item()
    p = build_prompt(item)
    assert "30 seconds" not in p
    assert "changelog" not in p


def test_the_b2_prompt_lists_the_items_by_key():
    p = build_prompt(b2_item())
    assert "k0" in p and "k1" in p
    assert "add a timeout" in p
    for label in ("stated_already", "inferable_from_state", "user_private", "new_task"):
        assert label in p


def test_the_b3_prompt_shows_what_the_agent_said_last():
    p = build_prompt(b3_item())
    assert "Applying the change to the config loader now." in p
    for label in ("stop_was_right", "should_have_asked", "should_have_continued"):
        assert label in p


def test_every_b_prompt_asks_for_a_rationale():
    for item in (b1_item(), b2_item(), b3_item()):
        assert "rationale" in build_prompt(item)


# --------------------------------------------------------------------------- parsing


def test_a_well_formed_b1_reply_parses():
    r = parse_reply(b1_item(), '{"verdict": "necessary", "rationale": "no default existed"}')
    assert r.response == {"verdict": "necessary"}
    assert r.rationale


def test_b1_default_existed_must_name_the_default():
    """Identical rule to the human path, for the identical reason: the verdict lands in the
    wasted-ask numerator and without the default it is unfalsifiable."""
    with pytest.raises(AnnotationParseError):
        parse_reply(b1_item(), '{"verdict": "default_existed", "rationale": "obvious"}')
    ok = parse_reply(
        b1_item(),
        '{"verdict": "default_existed", "default_action": "30s", "rationale": "matches config"}',
    )
    assert ok.response["default_action"] == "30s"


def test_b3_should_have_asked_must_name_the_question():
    with pytest.raises(AnnotationParseError):
        parse_reply(b3_item(), '{"verdict": "should_have_asked", "rationale": "risky"}')
    ok = parse_reply(
        b3_item(),
        '{"verdict": "should_have_asked", "question": "Which config?", "rationale": "risky"}',
    )
    assert ok.response["question"] == "Which config?"


def test_a_verdict_outside_the_vocabulary_is_refused_not_coerced():
    with pytest.raises(AnnotationParseError):
        parse_reply(b1_item(), '{"verdict": "probably_fine", "rationale": "x"}')


def test_b2_refuses_a_verdict_for_an_item_that_is_not_on_this_bundle_item():
    with pytest.raises(AnnotationParseError):
        parse_reply(
            b2_item(),
            '{"verdicts": {"k0": "new_task", "kZ": "new_task"}, "rationale": "x"}',
        )


def test_b2_accepts_a_partial_answer_because_silence_is_not_a_verdict():
    """The model that labelled two of three items said nothing about the third. Forcing a
    label there would invent one; `_units` already drops unmentioned items."""
    r = parse_reply(b2_item(), '{"verdicts": {"k0": "stated_already"}, "rationale": "said it"}')
    assert r.response["verdicts"] == {"k0": "stated_already"}


def test_a_reply_with_no_rationale_is_refused():
    """A model can always say why in the same call for near-zero marginal cost, and a verdict
    with no stated reason is the one hardest to audit later."""
    with pytest.raises(AnnotationParseError):
        parse_reply(b1_item(), '{"verdict": "necessary"}')


def test_an_unreadable_reply_raises_rather_than_defaulting():
    with pytest.raises(AnnotationParseError):
        parse_reply(b1_item(), "I think it was probably necessary.")
