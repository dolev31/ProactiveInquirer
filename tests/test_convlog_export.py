"""Consensus labels -> training rows.

WHAT A ROW IS. One decision point of one real conversation: the prompt the policy would be
shown at that moment, and the action two strong raters independently agreed was right there.
Nothing else in this repository has that shape, because nothing else has a person in it.

THE MAPPING, AND WHY EACH ARM POINTS WHERE IT DOES.

    B1 necessary                      -> ASK, with the question the agent actually asked
    B1 answer_in_state / _inferable
       / default_existed              -> STOP: the honest lesson from a question that bought
                                        nothing is "do not ask", exactly as `export_sft`
                                        turns a sub-threshold margin into STOP
    B3 should_have_asked              -> ASK, with the question the RATER supplied
    B3 stop_was_right                 -> STOP
    B3 should_have_continued          -> no row. "Keep working" is a decision for the tool
                                        loop, not for the Inquirer, and inventing a STOP
                                        there would teach the opposite of what was meant.
    B2 anything                       -> no row. B2 measures a MISS, and a miss names no
                                        action the policy could have taken instead.

UNANIMITY ONLY. A split item is not a weak label, it is two competent readers disagreeing, and
the pilot measured that at 22-35% of items. `annotate.consensus` already resolves only
unanimous units and queues the rest, so this consumes its output rather than re-deciding.
"""

import json

import pytest

from pi_eval.build.convlog_export import ExportedRow, rows_from_consensus
from pinq.actions import STOP_ACTION_JSON


def _key_entry(**over):
    dp = over.get("dp_index", 4)
    base = {
        "session_id": "s1",
        "dp_index": 4,
        "kind": "yield",
        "observed_action": "ASK_USER",
        "reaction": "short_yes",
        "repeats_earlier_turn": False,
        "asked_question": "Which timeout should I use?",
        "state_text": f"RENDERED PROMPT FOR s1:{dp}",
        "final_assistant_text": "Retry is in. Which timeout should I use?",
    }
    base.update(over)
    return base


def _unit(item_id, kind, label, **prov):
    return {
        "unit_id": item_id,
        "kind": kind,
        "label": label,
        "item_id": item_id,
        "provenance": prov,
    }


KEY = {
    "i-ask": _key_entry(),
    "i-waste": _key_entry(dp_index=5, asked_question="Should I use the config file?"),
    "i-stop": _key_entry(
        dp_index=6, kind="yield", observed_action="REPORT", asked_question="", reaction="other"
    ),
    "i-act": _key_entry(
        dp_index=7,
        kind="interrupt",
        observed_action="ACTING",
        asked_question="",
        reaction="interrupt",
    ),
}


def test_a_necessary_question_becomes_an_ask_row_carrying_the_real_question():
    rows = rows_from_consensus([_unit("i-ask", "B1", "necessary")], KEY)
    assert len(rows) == 1
    a = json.loads(rows[0].action_json)
    assert a == {
        "action": "ASK",
        "question": "Which timeout should I use?",
        "rationale": "",
        "target": "user",
    }
    assert rows[0].state_text == "RENDERED PROMPT FOR s1:4"


@pytest.mark.parametrize("label", ["answer_in_state", "answer_inferable", "default_existed"])
def test_a_question_that_bought_nothing_becomes_a_stop_row(label):
    rows = rows_from_consensus([_unit("i-waste", "B1", label)], KEY)
    # WAS {"action": "STOP", "rationale": "report"}. Belief corrected: "report" was a
    # rationale nobody wrote; a STOP target carries no rationale.
    assert rows[0].action_json == STOP_ACTION_JSON


def test_should_have_asked_becomes_an_ask_row_carrying_the_raters_question():
    """This is the only place a training target's TEXT comes from an annotator rather than
    from the log, which is the whole reason B3 refuses the verdict without it."""
    unit = _unit("i-act", "B3", "should_have_asked", question="Which config file did you mean?")
    rows = rows_from_consensus([unit], KEY)
    a = json.loads(rows[0].action_json)
    assert a["action"] == "ASK"
    assert a["question"] == "Which config file did you mean?"


def test_should_have_asked_without_a_question_yields_no_row_rather_than_an_empty_ask():
    rows = rows_from_consensus([_unit("i-act", "B3", "should_have_asked")], KEY)
    assert rows == []


def test_stop_was_right_becomes_a_stop_row():
    rows = rows_from_consensus([_unit("i-stop", "B3", "stop_was_right")], KEY)
    assert json.loads(rows[0].action_json)["action"] == "STOP"


def test_should_have_continued_yields_no_row():
    """Keep-working is the tool loop's decision, not the Inquirer's. A STOP row here would
    teach the opposite of what the rater said."""
    assert rows_from_consensus([_unit("i-stop", "B3", "should_have_continued")], KEY) == []


def test_b2_yields_no_row_at_all():
    """A miss names no action the policy could have taken instead, so it supervises nothing.
    B2 stays an evaluation metric."""
    assert rows_from_consensus([_unit("i-ask", "B2", "stated_already")], KEY) == []


def test_cant_tell_yields_no_row():
    assert rows_from_consensus([_unit("i-ask", "B1", "cant_tell")], KEY) == []


def test_every_row_carries_the_provenance_a_training_row_needs():
    rows = rows_from_consensus(
        [_unit("i-ask", "B1", "necessary")], KEY, scorer_hash="abc", graph_version="convlog-v1"
    )
    r = rows[0]
    assert (r.scorer_hash, r.graph_version, r.matcher_id) == ("abc", "convlog-v1", "human_b1b3")
    assert r.suite_id == "convlog"
    assert r.task_id == "s1:4"
    assert r.template_id == "s1"
    assert r.split in ("train", "dev", "test")


def test_a_repeated_prompt_is_carried_so_an_exporter_can_cap_it():
    """One looping session was 20% of every decision point mined. A consumer that cannot see
    which rows came from it cannot weight them down."""
    key = dict(KEY, **{"i-rep": _key_entry(dp_index=9, repeats_earlier_turn=True)})
    rows = rows_from_consensus([_unit("i-rep", "B1", "necessary")], key)
    assert rows[0].repeats_earlier_turn is True


def test_rows_are_deduplicated_by_decision_point():
    """B1 and B3 can both resolve at one decision point. Two contradictory rows at one state
    is a coin flip written into the dataset."""
    key = {"a": _key_entry(dp_index=4), "b": _key_entry(dp_index=4)}
    units = [_unit("a", "B1", "necessary"), _unit("b", "B3", "stop_was_right")]
    rows = rows_from_consensus(units, key)
    assert len(rows) == 1


def test_a_unit_with_no_key_entry_is_dropped_loudly_not_guessed():
    with pytest.raises(KeyError):
        rows_from_consensus([_unit("missing", "B1", "necessary")], KEY)


def test_exported_rows_serialise_into_the_shared_training_schema():
    """`pinq_train.export.dataset.Example` is what rung 1 loads. A convlog row that cannot
    become one is a row the trainer cannot read."""
    from pinq_train.export.dataset import Example

    rows = rows_from_consensus(
        [_unit("i-ask", "B1", "necessary")], KEY, scorer_hash="abc", graph_version="convlog-v1"
    )
    ex = Example(**rows[0].as_example_kwargs())
    assert isinstance(ex, Example)
    assert ex.suite_id == "convlog"
    assert isinstance(rows[0], ExportedRow)


def test_the_shard_manifest_records_what_was_dropped_and_why(tmp_path):
    """A guard that drops rows without saying how many is a guard nobody can audit -- the same
    reasoning `ExportManifest` carries for the benchmark exporter."""
    from pi_eval.build.convlog_export import write_shard

    units = [
        _unit("i-ask", "B1", "necessary"),
        _unit("i-waste", "B1", "answer_in_state"),
        _unit("i-ask", "B2", "stated_already"),
        _unit("i-stop", "B3", "should_have_continued"),
        _unit("i-act", "B3", "should_have_asked"),
    ]
    man = write_shard(units, KEY, out_dir=tmp_path, scorer_hash="abc")
    rows = [json.loads(x) for x in (tmp_path / "sft.jsonl").read_text().splitlines() if x]
    assert len(rows) == man["n_rows"] == 2
    assert man["n_b2_no_action"] == 1
    assert man["n_verdict_no_action"] == 1
    assert man["n_b3_missing_question"] == 1
    assert man["counts_by_verdict"]["necessary"] == 1
    assert set(man) >= {"scorer_hash", "graph_version", "matcher_id", "train_id_set_hash"}


def test_the_shard_splits_are_recorded_so_nobody_trains_on_test(tmp_path):
    from pi_eval.build.convlog_export import write_shard

    man = write_shard([_unit("i-ask", "B1", "necessary")], KEY, out_dir=tmp_path, scorer_hash="abc")
    assert set(man["counts_by_split"]) <= {"train", "dev", "test"}
    assert sum(man["counts_by_split"].values()) == man["n_rows"]


# --------------------------------------------------------------- the ASK target is a question


def test_the_ask_target_is_the_question_not_the_whole_report():
    """Measured on the first full export: the median ASK target was 2,368 characters and the
    longest 6,622, because a `yield` decision's `asked_question` is the agent's ENTIRE final
    message, which merely happens to end in a question mark.

    Training on that teaches a model to emit a status report where a question belongs. The
    label is still sound -- the rater judged whether asking was necessary, and they were shown
    the same block -- so the extraction happens here, at the target, not at parse time where it
    would desynchronise the labels from what the annotator read.
    """
    from pi_eval.build.convlog_export import extract_question

    report = (
        "**Both open questions are answered.** The cluster cert expired at 09:12 and I "
        "rotated it. Tests pass, 41 of 41.\n\n"
        "Do you want me to push this to the shared branch, or hold it for review?"
    )
    q = extract_question(report)
    assert q == "Do you want me to push this to the shared branch, or hold it for review?"


def test_a_trailing_question_split_over_two_sentences_keeps_only_the_interrogative():
    from pi_eval.build.convlog_export import extract_question

    assert extract_question("I fixed it. Should I commit?") == "Should I commit?"


def test_markdown_emphasis_around_the_question_is_stripped():
    from pi_eval.build.convlog_export import extract_question

    assert extract_question("Done. **Shall I continue?**") == "Shall I continue?"


def test_a_text_with_no_question_mark_yields_nothing():
    """Then there is no ASK target to build, and the row must not be invented."""
    from pi_eval.build.convlog_export import extract_question

    assert extract_question("I rewrote the loader and it builds.") == ""


def test_an_ask_row_carries_only_the_extracted_question():
    long_report = "I did four things. " * 40 + "Which timeout should I use?"
    key = dict(KEY, **{"i-long": _key_entry(dp_index=11, asked_question=long_report)})
    rows = rows_from_consensus([_unit("i-long", "B1", "necessary")], key)
    q = json.loads(rows[0].action_json)["question"]
    assert q == "Which timeout should I use?"
    assert "I did four things" not in q


def test_a_necessary_verdict_whose_text_holds_no_question_yields_no_row():
    key = dict(KEY, **{"i-noq": _key_entry(dp_index=12, asked_question="All four are green.")})
    assert rows_from_consensus([_unit("i-noq", "B1", "necessary")], key) == []


def test_the_split_items_are_written_as_an_adjudication_queue(tmp_path):
    """1,783 of 3,647 units in the full run were splits, and a split is the most informative
    thing the campaign produces: two strong readers looked at one state and disagreed.
    Discarding them silently throws away exactly the items a person should read, so they are
    written with the state attached -- a verdict pair without the state it was given for
    cannot be adjudicated."""
    from pi_eval.build.convlog_export import write_shard

    splits = [
        {
            "unit_id": "i-waste",
            "item_id": "i-waste",
            "kind": "B1",
            "votes": {"llm:a": "necessary", "llm:b": "answer_in_state"},
        }
    ]
    man = write_shard(
        [_unit("i-ask", "B1", "necessary")],
        KEY,
        out_dir=tmp_path,
        scorer_hash="abc",
        disagreements=splits,
    )
    rows = [json.loads(x) for x in (tmp_path / "adjudication.jsonl").read_text().splitlines() if x]
    assert man["n_adjudication"] == 1
    assert rows[0]["votes"] == {"llm:a": "necessary", "llm:b": "answer_in_state"}
    assert rows[0]["state_text"] == "RENDERED PROMPT FOR s1:5"


def test_the_manifest_reports_the_action_balance(tmp_path):
    """The full export came out 83% STOP. A dataset that imbalanced teaches "never ask", which
    is half of proactivity presented as the whole of it, so the number goes on the manifest
    rather than being discovered by whoever trains on it."""
    from pi_eval.build.convlog_export import write_shard

    units = [
        _unit("i-ask", "B1", "necessary"),
        _unit("i-waste", "B1", "answer_in_state"),
        _unit("i-stop", "B3", "stop_was_right"),
    ]
    man = write_shard(units, KEY, out_dir=tmp_path, scorer_hash="abc")
    assert man["ask_share"] == pytest.approx(1 / 3)


# ------------------------------------------------- how strongly the raters agreed on each row


def test_a_row_records_how_many_raters_agreed_and_how_many_looked():
    """Unanimity of two and a two-of-three majority are different evidence, and a shard that
    mixes them without saying which is which cannot be filtered by whoever trains on it.

    `annotate.consensus` already refuses a majority below three raters -- a majority of two is
    one person outvoting nobody -- so the choice only exists at three, and it is the consumer's
    to make rather than this exporter's.
    """
    u = _unit("i-ask", "B1", "necessary")
    u["votes"] = {"llm:a": "necessary", "llm:b": "necessary", "llm:c": "answer_in_state"}
    rows = rows_from_consensus([u], KEY)
    assert rows[0].n_agreeing == 2
    assert rows[0].n_raters == 3
    assert rows[0].unanimous is False


def test_a_unanimous_row_says_so():
    u = _unit("i-ask", "B1", "necessary")
    u["votes"] = {"llm:a": "necessary", "llm:b": "necessary"}
    rows = rows_from_consensus([u], KEY)
    assert (rows[0].n_agreeing, rows[0].n_raters, rows[0].unanimous) == (2, 2, True)


def test_votes_absent_leaves_the_counts_unknown_rather_than_asserting_unanimity():
    """A unit carrying no vote record is not evidence of agreement. Defaulting `unanimous` to
    True there would silently promote every legacy row to the stronger standard."""
    rows = rows_from_consensus([_unit("i-ask", "B1", "necessary")], KEY)
    assert (rows[0].n_agreeing, rows[0].n_raters, rows[0].unanimous) == (0, 0, False)


def test_the_manifest_splits_the_rows_by_agreement_standard(tmp_path):
    from pi_eval.build.convlog_export import write_shard

    a = _unit("i-ask", "B1", "necessary")
    a["votes"] = {"x": "necessary", "y": "necessary", "z": "necessary"}
    b = _unit("i-waste", "B1", "answer_in_state")
    b["votes"] = {"x": "answer_in_state", "y": "answer_in_state", "z": "necessary"}
    man = write_shard([a, b], KEY, out_dir=tmp_path, scorer_hash="s")
    assert man["n_unanimous_rows"] == 1
    assert man["n_majority_rows"] == 1
