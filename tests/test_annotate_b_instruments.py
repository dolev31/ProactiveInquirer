"""B1, B2 and B3: the three instruments that label a real conversation.

A1-A7 judge a rollout against a gold graph. These judge a LOG, where there is no graph and the
only ground truth is what the person actually did next. Everything below follows from that
difference, and from the rule that makes labelling a log legal at all: a label is a property of
one state and one observed object, never of a trajectory (see pi_eval/metrics/convlog.py).

  B1  was this question necessary given the state?      -> wasted_ask_rate, necessary_ask_precision
  B2  was this next-turn item inferable from the state? -> anticipation_miss_rate
  B3  should the agent have stopped and reported here?  -> over_action_rate, and the supervised
                                                           "should have asked X" target

None of the three may write anything gold-side, for the same reason A5 and A6 may not: they are
judgments about a conversation, not claims about a task's graph.
"""

import pytest

from pi_eval.annotate import (
    _LABELS,
    TASK_TYPES,
    _response_errors,
    _units,
    bundle_shape_errors,
    consensus,
    merge_into_graphs,
)

CTX = {
    "task": "Add a retry to the uploader.",
    "prior_turns": "PERSON: yes, also add a timeout\nASSISTANT: I can add a fixed retry.",
    "tool_trace": "Bash(cat upload.py) -> ok, 42 chars",
}


def b1_item(question="Which timeout should I use?", iid="i1"):
    return {
        "task_type": "B1",
        "item_id": iid,
        "provenance": {"session": "s1", "dp": 3},
        "context": {**CTX, "asked_question": question},
        "payload": {},
    }


def b2_item(items=(("k0", "add a timeout"), ("k1", "also update the changelog")), iid="i2"):
    return {
        "task_type": "B2",
        "item_id": iid,
        "provenance": {"session": "s1", "dp": 3},
        "context": {**CTX, "next_turn": "add a timeout, also update the changelog"},
        "payload": {"items": [{"item_key": k, "text": t} for k, t in items]},
    }


def b3_item(final="Applying the change to the config loader now.", iid="i3"):
    return {
        "task_type": "B3",
        "item_id": iid,
        "provenance": {"session": "s1", "dp": 3},
        "context": {**CTX, "final_assistant_text": final},
        "payload": {},
    }


# --------------------------------------------------------------------------- vocabularies


def test_the_three_instruments_are_registered():
    for tt in ("B1", "B2", "B3"):
        assert tt in TASK_TYPES


def test_the_label_vocabularies_are_closed_and_match_the_metrics_module():
    from pi_eval.metrics.convlog import B1_LABELS, B2_LABELS, B3_LABELS

    assert _LABELS["B1"] == B1_LABELS
    assert _LABELS["B2"] == B2_LABELS
    assert _LABELS["B3"] == B3_LABELS


# --------------------------------------------------------------------------- units


def test_b1_produces_one_unit_carrying_the_verdict():
    units = _units(b1_item(), {"verdict": "answer_in_state"}, None)
    assert [(u[1], u[2]) for u in units] == [("B1", "answer_in_state")]


def test_b2_produces_one_unit_per_item_not_one_over_the_set():
    """Same reasoning as A5_missing: a set-equality comparison would score two annotators who
    agree on three of four items as total disagreement."""
    units = _units(b2_item(), {"verdicts": {"k0": "inferable_from_state", "k1": "new_task"}}, None)
    assert sorted((u[1], u[2]) for u in units) == [
        ("B2", "inferable_from_state"),
        ("B2", "new_task"),
    ]
    assert len({u[0] for u in units}) == 2, "each item needs its own unit id"


def test_b2_scores_an_unmentioned_item_as_undecided_not_as_a_miss():
    """An annotator who skipped an item said nothing about it. Reading silence as
    `new_task` -- or as a miss -- invents a judgment and moves the rate either way."""
    units = _units(b2_item(), {"verdicts": {"k0": "stated_already"}}, None)
    assert [(u[1], u[2]) for u in units] == [("B2", "stated_already")]


def test_b3_produces_one_unit_and_carries_the_question_as_provenance():
    units = _units(
        b3_item(), {"verdict": "should_have_asked", "question": "Which config file?"}, None
    )
    assert [(u[1], u[2]) for u in units] == [("B3", "should_have_asked")]
    assert units[0][3]["question"] == "Which config file?"


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("tt,resp", [("B1", {"verdict": "nope"}), ("B3", {"verdict": "nope"})])
def test_a_verdict_outside_the_vocabulary_is_refused(tt, resp):
    item = b1_item() if tt == "B1" else b3_item()
    assert _response_errors(item, resp, "w") != []


def test_b1_default_existed_without_naming_the_default_is_refused():
    """`default_existed` is the claim 'you did not need to ask, the obvious choice was X'. With
    no X it is unfalsifiable, and it lands in the wasted-ask numerator. A7's reaches_unstated
    carries the same requirement for the same reason."""
    assert _response_errors(b1_item(), {"verdict": "default_existed"}, "w") != []
    ok = _response_errors(
        b1_item(), {"verdict": "default_existed", "default_action": "30s, as elsewhere"}, "w"
    )
    assert ok == []


def test_b3_should_have_asked_without_the_question_is_refused():
    """This verdict IS the supervised target: it becomes an ASK row whose question is the
    annotator's text. Without the text there is no training example, only a complaint."""
    assert _response_errors(b3_item(), {"verdict": "should_have_asked"}, "w") != []
    assert (
        _response_errors(b3_item(), {"verdict": "should_have_asked", "question": "Q?"}, "w") == []
    )


def test_b2_verdicts_for_items_not_on_this_bundle_item_are_refused():
    errs = _response_errors(b2_item(), {"verdicts": {"k0": "new_task", "kZ": "new_task"}}, "w")
    assert any("kZ" in e for e in errs)


def test_b2_rejects_a_label_outside_its_vocabulary():
    assert _response_errors(b2_item(), {"verdicts": {"k0": "necessary"}}, "w") != []


# --------------------------------------------------------------------------- bundle shape


def test_well_formed_items_pass_the_bundle_shape_check():
    assert bundle_shape_errors({"items": [b1_item(), b2_item(), b3_item()]}) == []


def test_an_item_missing_its_state_is_refused():
    """Every B judgment is 'given this state'. An item that does not show the state is not a
    harder item, it is a different question."""
    bad = b1_item()
    del bad["context"]["tool_trace"]
    assert bundle_shape_errors({"items": [bad]}) != []


def test_a_b1_item_may_not_carry_the_next_turn():
    """B1 asks whether the question was necessary BEFORE the person replied. Showing the reply
    is showing the answer."""
    bad = b1_item()
    bad["context"]["next_turn"] = "30 seconds"
    assert bundle_shape_errors({"items": [bad]}) != []


# --------------------------------------------------------------------------- the gold wall


@pytest.mark.parametrize("tt", ["B1", "B2", "B3"])
def test_no_b_instrument_ever_writes_a_gold_field(tt):
    """Same wall A5 and A6 sit behind. These judge a conversation; a gold graph describes a
    task. There is no defensible edit from one to the other, so there is no branch at all."""
    item = {"B1": b1_item(), "B2": b2_item(), "B3": b3_item()}[tt]
    resp = {
        "B1": {"verdict": "necessary"},
        "B2": {"verdicts": {"k0": "stated_already"}},
        "B3": {"verdict": "stop_was_right"},
    }[tt]
    records = [
        {
            "item_id": item["item_id"],
            "task_type": tt,
            "annotator_id": f"h{i}",
            "annotator_kind": "human",
            "response": resp,
        }
        for i in range(2)
    ]
    bundle = {"bundle_id": "b", "items": [item]}
    cons = consensus(bundle, records)
    # Two agreeing humans must actually have resolved, or this test proves nothing: an empty
    # consensus would make `merge_into_graphs` return [] for a reason that has nothing to do
    # with the wall being tested.
    assert cons.by_kind(tt), f"{tt} produced no resolved unit"
    assert merge_into_graphs({}, cons, out_version="v2") == []
