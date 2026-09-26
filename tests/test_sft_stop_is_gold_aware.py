"""The SFT STOP target is a statement about the TASK, not about the sampler.

THE DEFECT THESE TESTS WERE WRITTEN FOR. `export_sft` labelled a state STOP whenever the
best-valued candidate fell below the noise floor: "not measurably better than not asking".
That reads a property of the SAMPLES as a property of the STATE. On a state where required
evidence is still missing and every one of eight candidates happened to ask badly, the
export taught the policy to stop -- exactly where it should have kept going -- and it did so
on 66.8% of the live SFT (`stop_share` 0.668). Whether stopping is right is a gold
question: was the required evidence already in hand BEFORE this decision? Rows now carry
that (`done_before`, from `rows_from_run`), so the rule can ask it.

THE RULE (`stop_rule = "gold_coverage_v1"`), per state:
  * done before the decision      -> STOP, `label_rule="stop_done"`, whatever the samples did;
  * not done, best ASK clears the floor -> that ASK, `label_rule="ask_clears_floor"`;
  * not done, no ASK clears the floor   -> NO ROW, counted as `n_no_target_dropped`. The
    reward's own stop term punishes the undershoot; the exporter has no target to teach.
A recorded STOP candidate (`is_stop` row, see B0) never enters the ASK argmax.

`done_before` ABSENT is UNKNOWN, not done. Rows from before the field existed carry no
gold-side "done" signal, so they can never yield a STOP target; the exporter counts them
(`n_done_before_unknown`) rather than raising, because the only rows that ever reach
`export_sft` without the field are fixtures and legacy files -- the CLI recomputes rows from
run directories and always stamps it.

`frontier_size == 0` is NOT "done": it is a matcher statement over uids, and it disagreed
with coverage on the optional-parent graphs before that fix. It is a cross-check only,
counted in `n_done_frontier_disagree`.

The first test failed before the change with `is_stop == False` on a done state, and the
third with a STOP where there was nothing to teach.
"""

from __future__ import annotations

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_sft


def _row(run_id, value, question, *, done=None, coverage=None, frontier=1, is_stop=False):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "frontier_size": frontier,
        "is_stop": is_stop,
    }
    if done is not None:
        r["done_before"] = done
        r["coverage_before"] = 1.0 if done else 0.5
    if coverage is not None:
        r["coverage_before"] = coverage
    return r


def test_a_done_state_is_labelled_stop_even_when_an_ask_clears_the_floor():
    """Gold says everything required was in hand; the ASK's gain is redundancy or noise."""
    rows = [
        _row("a", 0.9, "who?", done=True, frontier=0),
        _row("b", 0.1, "when?", done=True, frontier=0),
    ]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert len(ex) == 1 and ex[0].is_stop
    assert ex[0].action_json == STOP_ACTION_JSON
    assert ex[0].label_rule == "stop_done"
    assert ex[0].done_before is True and ex[0].coverage_before == 1.0
    assert man.n_stop_done_before_dedupe == 1 and man.n_no_target_dropped == 0


def test_a_not_done_state_whose_best_ask_clears_the_floor_keeps_that_ask():
    rows = [_row("a", 0.9, "who?", done=False), _row("b", 0.1, "when?", done=False)]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert len(ex) == 1 and not ex[0].is_stop
    assert ex[0].action_json == ask_action_json("who?")
    assert ex[0].label_rule == "ask_clears_floor"
    assert man.n_stop_done_before_dedupe == 0


def test_a_not_done_state_with_no_ask_above_the_floor_has_no_target():
    """Every sample missed. That is evidence about the sampler, not a reason to stop."""
    rows = [_row("a", 0.05, "who?", done=False), _row("b", 0.01, "when?", done=False)]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert ex == []
    assert man.n_no_target_dropped == 1 and man.n_stop_done_before_dedupe == 0


def test_a_recorded_stop_candidate_never_enters_the_ask_argmax():
    """A STOP row (value 0.0) outranks every negative-valued ASK on `value`; before, it would
    have been `best` and exported as an ASK-shaped target with a STOP body. It is neither the
    target nor a reason to stop on a not-done state."""
    rows = [
        _row("s", 0.0, "", done=False, is_stop=True),
        _row("a", -0.05, "who?", done=False),
    ]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert ex == [] and man.n_no_target_dropped == 1

    # and with an ASK that clears the floor, the ASK wins over the recorded STOP
    rows = [_row("s", 0.0, "", done=False, is_stop=True), _row("a", 0.5, "who?", done=False)]
    ex, _ = export_sft(rows, margin_threshold=0.2)
    assert len(ex) == 1 and ex[0].action_json == ask_action_json("who?")


def test_done_before_absent_is_unknown_and_never_yields_a_stop():
    """A legacy row set: no gold-side done signal, so no STOP target can be derived from it.
    Counted, not raised -- see the module docstring."""
    rows = [_row("a", 0.05, "who?"), _row("b", 0.01, "when?")]  # no done_before at all
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert ex == [] and man.n_no_target_dropped == 1
    assert man.n_done_before_unknown == 1

    rows = [_row("a", 0.9, "who?")]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert len(ex) == 1 and ex[0].label_rule == "ask_clears_floor" and ex[0].done_before is None


def test_a_done_state_with_a_nonempty_frontier_is_counted_as_a_disagreement():
    """Coverage says done, the matcher's frontier says something is still askable. Coverage
    wins (it is the quantity the reward and `_complete_at` use); the disagreement is counted so
    the two instruments can be reconciled, never silently."""
    rows = [_row("a", 0.9, "who?", done=True, frontier=2)]
    ex, man = export_sft(rows, margin_threshold=0.2)
    assert len(ex) == 1 and ex[0].is_stop
    assert man.n_done_frontier_disagree == 1

    rows = [_row("a", 0.9, "who?", done=True, frontier=0)]
    _, man = export_sft(rows, margin_threshold=0.2)
    assert man.n_done_frontier_disagree == 0


def test_the_manifest_names_the_rule_and_the_stop_shape():
    _, man = export_sft([_row("a", 0.9, "who?", done=True)], margin_threshold=0.2)
    assert man.stop_rule == "gold_coverage_v1"
    assert man.stop_action_json == STOP_ACTION_JSON


def test_a_state_where_rows_disagree_on_done_is_refused():
    """`done_before` is a property of the state; two candidates at one state that disagree
    were not rendered from the same evidence, and no label may be derived from them.

    THE INVARIANT IS FATAL TO THE STATE, NOT TO THE EXPORT -- and this test used to assert the
    second. The belief changed on evidence: a 206,088-row export died 57 minutes in on one
    such state, and re-reading its 32 candidates found them unanimous. The exporter had raced
    a fork worker writing those directories. `_state_done` still raises (the check is not
    weakened); `export_sft` catches it, counts it and carries on, the way `collect_rows`
    already treats a run it cannot export. See tests/test_export_state_disagreement_is_counted.
    """
    from pinq_train.export.dataset import _state_done

    rows = [_row("a", 0.9, "who?", done=True), _row("b", 0.1, "when?", done=False)]
    with pytest.raises(ValueError, match="done_before"):
        _state_done(rows)

    ex, man = export_sft(rows, margin_threshold=0.2)
    assert ex == [] and man.n_state_done_disagree == 1
