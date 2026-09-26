"""The offline stop 2x2 can be conditioned on a gold fact other than pooled completeness.

WHY THIS EXISTS. `stop_confusion` read `row.get("done_before")` and nothing else, so every
"does it stop when it should" number this programme has ever produced is keyed on POOLED
required-evidence completeness. Lane L6.1's question is about the ANSWER-BEARING node, and
re-exporting the dev rows under the answer-node STOP label does NOT change what the table
conditions on -- it changes which rows are STOP targets, and leaves both cells keyed on
`done_before`. Without this, the lane's offline number would answer the pooled question while
being reported under the answer-node arm's name.

SCORED ONCE, BUCKETED TWICE. The expensive half is two forward passes per row; the condition is
only a bucketing of rows already scored. Asking for a second condition therefore costs nothing,
which is what makes reporting both the honest default rather than a choice between them.

BACKWARD COMPATIBILITY IS A REQUIREMENT, NOT A COURTESY. Sixty Tier-A verdicts on the cluster
carry the flat `p_stop_given_done` / `p_ask_given_not_done` keys, and `scripts/figures_programme
.py` reads them. The first requested field keeps those keys with exactly their old meaning, so a
verdict written before and after this change means the same thing.
"""

from __future__ import annotations

import json

import pytest

# The SAME stubs the existing stop-2x2 tests use, imported rather than re-written: `render`
# returns a `Rendered(prompt_ids, completion_ids)` and `logprob` takes those id sequences, and a
# third private copy of that contract is a third thing to get wrong when it changes.
from tests.test_eval_offline import _is_stop, _render

from pinq.actions import STOP_ACTION_JSON
from pinq_train.eval_offline import stop_confusion

STOP = STOP_ACTION_JSON


def _ask(q: str) -> str:
    return json.dumps({"action": "ASK", "question": q, "rationale": ""})


def _prefers_stop(prompt_ids, completion_ids) -> float:
    """A scripted policy that always prefers STOP. A stub and not a model, because the thing
    under test is the BUCKETING: a test that needed a checkpoint to decide which cell a row
    lands in could not tell a bucketing bug from the checkpoint having an opinion."""
    per_token = -0.1 if _is_stop(completion_ids) else -9.0
    return per_token * len(completion_ids)


def _prefers_ask(prompt_ids, completion_ids) -> float:
    per_token = -9.0 if _is_stop(completion_ids) else -0.1
    return per_token * len(completion_ids)


def _row(*, done, answer, action_json, state="s"):
    r = {"state_text": state, "action_json": action_json}
    if done is not None:
        r["done_before"] = done
    if answer is not None:
        r["answer_node_covered_before"] = answer
    return r


def test_the_two_fields_bucket_the_same_rows_differently():
    """The whole point: one set of rows, two conditions, two different tables.

    Four ASK rows spanning every (done, answer-covered) combination the gold can produce. The
    model always prefers ASK here, so `p_ask_given_not_done` is 1.0 within whichever cell the
    condition puts a row in -- and the cells differ between the two conditions, which is what a
    hardcoded field made impossible to see.
    """
    rows = [
        _row(done=True, answer=True, action_json=_ask("q1")),
        _row(done=False, answer=True, action_json=_ask("q2")),
        _row(done=False, answer=False, action_json=_ask("q3")),
        _row(done=False, answer=False, action_json=_ask("q4")),
    ]
    out = stop_confusion(
        rows,
        render=_render,
        logprob=_prefers_ask,
        condition_fields=("done_before", "answer_node_covered_before"),
    )
    by = out["by_condition"]
    assert by["done_before"]["n_done"] == 1.0
    assert by["done_before"]["n_not_done"] == 3.0
    assert by["answer_node_covered_before"]["n_done"] == 2.0
    assert by["answer_node_covered_before"]["n_not_done"] == 2.0
    # The model prefers ASK everywhere, so every not-done cell reads 1.0 and every done cell 0.0
    # under BOTH conditions -- the denominators are the thing that moved.
    for f in ("done_before", "answer_node_covered_before"):
        assert by[f]["p_ask_given_not_done"] == pytest.approx(1.0)
        assert by[f]["p_stop_given_done"] == pytest.approx(0.0)


def test_the_first_field_keeps_the_flat_keys_with_their_old_meaning():
    """Sixty verdicts on disk and `figures_programme.py` read the flat keys. A verdict written
    before and after this change must mean the same thing."""
    rows = [
        _row(done=True, answer=True, action_json=_ask("q1")),
        _row(done=False, answer=False, action_json=_ask("q2")),
    ]
    out = stop_confusion(rows, render=_render, logprob=_prefers_ask)
    assert out["n_done"] == 1.0 and out["n_not_done"] == 1.0
    assert out["p_ask_given_not_done"] == pytest.approx(1.0)
    assert out["condition_field"] == "done_before"
    assert out["by_condition"]["done_before"]["n_done"] == out["n_done"]


def test_an_unknown_second_field_is_skipped_and_counted_not_read_as_false():
    """Unknown is not "not covered", by the same asymmetry `done_before is None` already has.

    A row with no `answer_node_covered_before` must fall into `n_skipped_no_label` for THAT
    condition and still be scored under `done_before`. Defaulting it to False would move every
    such row into the second cell and quietly inflate `p_ask_given_not_done`.
    """
    rows = [
        _row(done=False, answer=None, action_json=_ask("q1")),
        _row(done=False, answer=False, action_json=_ask("q2")),
    ]
    out = stop_confusion(
        rows,
        render=_render,
        logprob=_prefers_ask,
        condition_fields=("done_before", "answer_node_covered_before"),
    )
    assert out["by_condition"]["done_before"]["n_not_done"] == 2.0
    assert out["by_condition"]["answer_node_covered_before"]["n_not_done"] == 1.0
    assert out["by_condition"]["answer_node_covered_before"]["n_skipped_no_label"] == 1.0


def test_a_stop_row_without_a_reference_ask_is_skipped_under_every_condition():
    """The skip is a property of the ROW, not of the condition, so it must be counted the same
    way in each table. Counting it once would make the two tables' denominators incomparable."""
    rows = [_row(done=True, answer=True, action_json=STOP)]
    out = stop_confusion(
        rows,
        render=_render,
        logprob=_prefers_stop,
        condition_fields=("done_before", "answer_node_covered_before"),
    )
    for f in ("done_before", "answer_node_covered_before"):
        assert out["by_condition"][f]["n_skipped_no_reference"] == 1.0
        assert out["by_condition"][f]["n_done"] == 0.0


def test_a_stop_row_with_a_reference_ask_lands_in_both_tables():
    """With the reference ask supplied the row is scored, and the model preferring STOP must
    register as `p_stop_given_done == 1.0` under both conditions that call it done."""
    rows = [_row(done=True, answer=True, action_json=STOP)]
    out = stop_confusion(
        rows,
        render=_render,
        logprob=_prefers_stop,
        reference_ask=_ask("anything"),
        condition_fields=("done_before", "answer_node_covered_before"),
    )
    for f in ("done_before", "answer_node_covered_before"):
        assert out["by_condition"][f]["n_skipped_no_reference"] == 0.0
        assert out["by_condition"][f]["p_stop_given_done"] == pytest.approx(1.0)


def test_the_rows_are_scored_once_however_many_conditions_are_asked_for():
    """Two forward passes per row, not two per row per condition.

    Asserted because the whole argument for reporting both tables is that the second is free; if
    it were not, someone would reasonably drop one and the lane would be back to choosing.
    """
    calls: list[str] = []

    def counting_logprob(prompt_ids, completion_ids) -> float:
        calls.append(tuple(completion_ids))
        return -1.0 * len(completion_ids)

    rows = [_row(done=False, answer=False, action_json=_ask(f"q{i}")) for i in range(5)]
    stop_confusion(
        rows,
        render=_render,
        logprob=counting_logprob,
        condition_fields=("done_before", "answer_node_covered_before"),
    )
    assert len(calls) == 10, "5 rows x (STOP, ASK) = 10 scorings, regardless of condition count"


def test_an_empty_condition_list_is_refused():
    """A table with no condition is not a table. Returning an empty result would read as "no
    rows were labelled", which is a statement about the data rather than about the call."""
    with pytest.raises(ValueError, match="at least one"):
        stop_confusion([], render=_render, logprob=_prefers_ask, condition_fields=())
