"""A pair whose two candidates were rendered from different evidence is not a same-state pair.

THIS GUARD HAS NEVER FIRED, AND IS KEPT ANYWAY. The honest version of how it got here:

An audit grouped the exported pairs by state key, found 47 states carrying two different
`state_text` values, and concluded the exporter was pairing across them. It was not. Those 47
states each hold two DISJOINT groups of pairs, and `pins_sha` already refuses every pair
between them -- measured on the full corpus, the number of (state, pins_sha) combinations
carrying more than one `state_text` is ZERO, and a re-export with this guard active dropped
nothing and produced byte-identical output.

WHY THE DRIFT IS REAL EVEN THOUGH THE BUG WAS NOT. A fork replays its parent's prefix and the
retriever does not guarantee the same evidence set twice. 13 candidates off one parent at one
turn rendered into prompts of 5,536 and 3,573 characters because they were forked in different
rounds. Every one of those rounds also changed the model/prompt pin, so `pins_sha` happened to
separate them.

WHY IT IS KEPT. `pins_sha` covers the pin, not the retrieval RESULT. The two coincide today
only because each fork round changed the pin. A round that re-forks an already-forked state
under an unchanged pin would drift with identical `pins_sha`, and nothing else in the exporter
looks at the rendered text: `render_state` verifies each row against its OWN recorded hash, so
every row is individually correct, and rung 2's `assert_same_state` returns early because the
exporter emits one `state_text` per pair. The cost is one string comparison per candidate
pair. The thing being protected is the only invariant that makes rung 2 valid -- every term
that is a function of the state alone appearing identically in both halves and subtracting to
zero.

`_pair` writes the WINNER's state, so without this the rejected action would be scored against
a prompt it never saw.
"""

from __future__ import annotations

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_pairs


def _c(run_id, value, question, *, state="S", is_stop=False, done=False):
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": state,
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
    }


def _ex(rows):
    return export_pairs(rows, margin_threshold=0.05, len_delta_max=1000)


def test_two_candidates_rendered_from_different_evidence_make_no_pair():
    rows = [_c("a", 0.9, "who?", state="EVIDENCE ONE"), _c("b", 0.1, "when?", state="EVIDENCE TWO")]
    pairs, man = _ex(rows)
    assert pairs == []
    assert man.n_cross_state_dropped == 1


def test_the_same_state_still_pairs():
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    pairs, man = _ex(rows)
    assert len(pairs) == 1 and man.n_cross_state_dropped == 0


def test_only_the_drifted_candidate_is_excluded_not_the_whole_state():
    """Two candidates agree and a third drifted. The agreeing pair survives; the two pairs
    involving the drifted one do not. Dropping the state entirely would throw away a valid
    contrast to punish a neighbour."""
    rows = [
        _c("a", 0.9, "who?", state="SAME"),
        _c("b", 0.5, "when?", state="SAME"),
        _c("c", 0.1, "where?", state="DRIFTED"),
    ]
    pairs, man = _ex(rows)
    assert len(pairs) == 1
    assert {pairs[0].chosen_run_id, pairs[0].rejected_run_id} == {"a", "b"}
    assert man.n_cross_state_dropped == 2


def test_an_ask_stop_pair_is_guarded_too():
    """A recorded STOP is a candidate like any other and replays the same prefix."""
    rows = [
        _c("a", 0.9, "who?", state="ONE", done=True),
        _c("s", 0.0, "", state="TWO", is_stop=True, done=True),
    ]
    pairs, man = _ex(rows)
    assert [p for p in pairs if p.pair_kind == "ask_stop"] == []
    assert man.n_cross_state_dropped >= 1


def test_the_synthetic_stop_uses_the_state_it_was_built_from():
    """The synthetic side is a constant, so there is no second state to disagree -- but the
    ASK it is paired against must still be one of the candidates that agreed."""
    rows = [
        _c("a", 0.9, "who?", state="SAME", done=True),
        _c("b", 0.1, "when?", state="SAME", done=True),
    ]
    pairs, _ = _ex(rows)
    synth = [p for p in pairs if p.pair_kind == "ask_stop_synth"]
    assert len(synth) == 1 and synth[0].state_text == "SAME"
