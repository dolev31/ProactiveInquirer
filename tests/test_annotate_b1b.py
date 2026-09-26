"""B1b: B1 with `default_existed` removed, as a controlled comparison.

WHY IT EXISTS AS A SEPARATE TASK TYPE RATHER THAN AN EDIT TO B1. The pilot's disagreement is
concentrated on one axis: of 19 disagreements between the two strong raters on B1, 15 are
`default_existed` against `necessary`. Whether a default was "obvious" is a judgment about risk
tolerance that the state does not settle, so it is the prime suspect.

Editing B1 in place would answer nothing -- the old and new numbers would not be comparable and
nobody could tell a real improvement from a relabelling. B1b is the same items, the same states
and the same prompt minus one option, so the two are a controlled pair and both sets of records
stay on disk under their own task type.
"""

import pytest
from tests.test_annotate_b_instruments import CTX

from pi_eval.annotate import _LABELS, TASK_TYPES, _response_errors, _units
from pi_eval.annotate_llm import AnnotationParseError, build_prompt, parse_reply


def b1b_item(question="Which timeout should I use?", iid="v1"):
    return {
        "task_type": "B1b",
        "item_id": iid,
        "provenance": {"session": "s1", "dp": 3},
        "context": {**CTX, "asked_question": question},
        "payload": {},
    }


def test_b1b_is_registered_and_drops_only_the_contested_option():
    assert "B1b" in TASK_TYPES
    assert _LABELS["B1b"] == ("necessary", "answer_in_state", "answer_inferable", "cant_tell")
    assert set(_LABELS["B1b"]) == set(_LABELS["B1"]) - {"default_existed"}


def test_its_units_are_scored_as_B1_so_the_two_are_comparable():
    """The unit KIND is what a rate is computed over. Emitting B1b units under their own kind
    would put the controlled comparison in a different column from the thing it controls."""
    units = _units(b1b_item(), {"verdict": "answer_inferable"}, None)
    assert [(u[1], u[2]) for u in units] == [("B1", "answer_inferable")]


def test_the_removed_option_is_refused_rather_than_silently_mapped():
    """Mapping it onto `answer_inferable` would hide the very behaviour under test."""
    assert _response_errors(b1b_item(), {"verdict": "default_existed"}, "w") != []
    with pytest.raises(AnnotationParseError):
        parse_reply(b1b_item(), '{"verdict": "default_existed", "rationale": "x"}')


def test_the_prompt_offers_no_default_option():
    p = build_prompt(b1b_item())
    assert "default" not in p.lower()
    for label in ("necessary", "answer_in_state", "answer_inferable", "cant_tell"):
        assert label in p
    assert "Which timeout should I use?" in p
    assert "30 seconds" not in p


def test_a_well_formed_reply_parses():
    r = parse_reply(b1b_item(), '{"verdict": "necessary", "rationale": "not derivable"}')
    assert r.response == {"verdict": "necessary"}
