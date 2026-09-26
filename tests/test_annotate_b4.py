"""B4: propose the question the agent should have asked, then validate it as if it were real.

WHY THIS INSTRUMENT EXISTS. B3 asks a recognition question — "should it have asked something
first?" — and three strong raters answered yes for a unanimous pair ONCE in 848 units. A
recognition task with a base rate near zero is the weakest design available, and it left the
dataset's ASK supervision resting entirely on questions the agent already thought to ask.

B4 inverts it. A strong model is shown the same state and PROPOSES the question, or says none
was needed. The proposal is then put through B1 unchanged, by different raters, and kept only
if they judge it necessary.

THE VALIDATION PROMPT IS B1 VERBATIM, ON PURPOSE. A proposed question and a logged one reach
the validators in identical form, with the provenance held in the key. That buys two things a
separate wording would lose: the blinding is real, and the two acceptance rates are directly
comparable — so "are invented questions accepted more often than real ones?" is a number rather
than a worry. A much higher rate would mean the proposer is producing agreeable noise; a much
lower one would mean it is inventing.

THE PROPOSER MUST BE ABLE TO DECLINE. Without `none_needed` a generator asked for a question
returns a question every time, and the acceptance rate would measure the validators' patience
rather than the proposer's judgement.
"""

import pytest
from tests.test_annotate_b_instruments import CTX

from pi_eval.annotate import _LABELS, TASK_TYPES, _response_errors, _units
from pi_eval.annotate_llm import AnnotationParseError, build_prompt, parse_reply


def b4_item(iid="p1", final="Rewriting the config loader now."):
    return {
        "task_type": "B4",
        "item_id": iid,
        "provenance": {"session": "s1", "dp": 3},
        "context": {**CTX, "final_assistant_text": final},
        "payload": {},
    }


def test_b4_is_registered_with_a_closed_vocabulary():
    assert "B4" in TASK_TYPES
    assert _LABELS["B4"] == ("question_needed", "none_needed", "cant_tell")


def test_a_proposal_carries_its_question_in_provenance_not_in_the_label():
    """The label is what two raters must agree on; the question is what the row is built from.
    Free text inside `response` would sit inside the equality comparison consensus depends on."""
    units = _units(
        b4_item(), {"verdict": "question_needed", "question": "Which config file?"}, None
    )
    assert [(u[1], u[2]) for u in units] == [("B4", "question_needed")]
    assert units[0][3]["question"] == "Which config file?"


def test_question_needed_without_a_question_is_refused():
    assert _response_errors(b4_item(), {"verdict": "question_needed"}, "w") != []
    ok = _response_errors(b4_item(), {"verdict": "question_needed", "question": "Which?"}, "w")
    assert ok == []


def test_declining_is_a_first_class_answer():
    """Without it a generator asked for a question returns one every time, and the acceptance
    rate would measure the validators' patience rather than the proposer's judgement."""
    assert _response_errors(b4_item(), {"verdict": "none_needed"}, "w") == []
    assert _units(b4_item(), {"verdict": "none_needed"}, None)[0][2] == "none_needed"


def test_a_question_supplied_alongside_none_needed_is_refused():
    assert _response_errors(b4_item(), {"verdict": "none_needed", "question": "x?"}, "w") != []


def test_the_prompt_shows_the_state_and_what_the_agent_actually_did():
    p = build_prompt(b4_item())
    assert "Add a retry to the uploader." in p
    assert "Rewriting the config loader now." in p
    assert "none_needed" in p and "question_needed" in p


def test_the_prompt_does_not_show_what_the_person_said_next():
    """The proposal must come from the state alone. Showing the reply turns proposing into
    copying, and every proposal would then be trivially necessary."""
    p = build_prompt(b4_item())
    assert "30 seconds" not in p
    assert "changelog" not in p


def test_a_reply_parses_and_refuses_the_two_incoherent_shapes():
    r = parse_reply(b4_item(), '{"verdict": "none_needed", "rationale": "nothing was open"}')
    assert r.response == {"verdict": "none_needed"}
    with pytest.raises(AnnotationParseError):
        parse_reply(b4_item(), '{"verdict": "question_needed", "rationale": "x"}')
    with pytest.raises(AnnotationParseError):
        parse_reply(b4_item(), '{"verdict": "none_needed", "question": "y?", "rationale": "x"}')
