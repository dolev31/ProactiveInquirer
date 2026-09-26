"""ASK versus STOP is a preference, and until now the dataset could not express one.

THE GAP THESE TESTS WERE WRITTEN FOR. Every pair in `pairs.jsonl` is ASK-vs-ASK: two
questions at one state, ranked by which retrieved more. Nothing in the artifact ever says
"here, stopping was better than asking" or "here, asking was better than stopping" -- so a
DPO policy trained on it has no gradient at all on the decision the paper is about. Recorded
STOPs are now rows (see `tests/test_export_stop_rows.py`), so the pair exists in the data and
only the exporter had to learn to emit it.

THE RULE. `pair_kind` is on every pair: "ask_ask" (unchanged), "ask_stop" for a STOP a
candidate actually took, and "ask_stop_synth" for one the exporter derived from gold. The
third is a separate kind because it outnumbered the recorded STOPs 18,227 to 627 on the live
corpus and would otherwise be 73% of the dataset under one label; see
tests/test_synth_stop_is_its_own_kind.py.

  * STOP CHOSEN requires a gold reason, never a value comparison: `done_before` is True (the
    required evidence was already in hand), or the outcome key decides. `value` cannot do
    this job -- a STOP's value is 0.0 by construction, so on any state whose questions all
    have negative value STOP would "win" every pair for having paid no retrieval, and the
    policy would learn to stop whenever asking is expensive.
  * ASK CHOSEN requires `phi(ASK) > margin_threshold` -- the same floor `export_sft` uses.
    Below it the question was not measurably better than not asking, and the pair would
    assert a preference the measurement does not support.
  * STOP vs STOP is dropped before the identical-action guard, and counted separately: two
    STOPs are byte-identical by construction (one constant), so they would otherwise be
    filed as "both candidates asked the same question", which is not what happened.
  * The LENGTH GUARD DOES NOT APPLY to ask_stop. A STOP is ~18 bytes and a question ~40-120,
    so every ask_stop pair violates any sane cap. The asymmetry is real and unsolvable --
    it is what the two actions ARE -- so it is made visible (`pair_kind` on the row,
    per-direction counters in the manifest, `include_pair_kinds` in the rung-2 loader)
    rather than hidden behind a guard that would delete the whole category.

`stop_source` says whether the STOP was RECORDED (a candidate actually stopped there) or
SYNTHESISED (the state had no recorded STOP and gold says it was already done). A synthetic
side is not a rollout: it has no run id, and `stop_source` is how a consumer tells.
"""

from __future__ import annotations

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_pairs

LONG_Q = "who was the founder of the company that first manufactured the device in question?"


def _c(run_id, value, question, *, done=False, is_stop=False, correct=None, phi=None):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value if phi is None else phi,
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
    if correct is not None:
        r["answer_correct"] = correct
    return r


def _only(pairs):
    assert len(pairs) == 1, pairs
    return pairs[0]


def test_a_done_state_prefers_the_stop_over_a_high_gain_ask():
    """The ASK has the higher value and would have won on gain. Gold says the task was already
    finished, so what it retrieved was not needed."""
    rows = [_c("a", 0.9, "who?", done=True), _c("s", 0.0, "", done=True, is_stop=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    p = _only(pairs)
    assert p.chosen_json == STOP_ACTION_JSON and p.rejected_json == ask_action_json("who?")
    assert p.pair_kind == "ask_stop" and p.stop_source == "recorded"
    assert p.chosen_run_id == "s" and p.rejected_run_id == "a"
    assert man.n_stop_chosen == 1 and man.n_ask_stop_pairs == 1


def test_an_unfinished_state_prefers_the_ask_that_clears_the_floor():
    rows = [_c("a", 0.9, "who?", done=False), _c("s", 0.0, "", done=False, is_stop=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    p = _only(pairs)
    assert p.chosen_json == ask_action_json("who?") and p.rejected_json == STOP_ACTION_JSON
    assert p.pair_kind == "ask_stop" and man.n_ask_chosen_over_stop == 1


def test_an_unfinished_state_whose_ask_misses_the_floor_makes_no_pair():
    """Neither direction is supported: the task is not done, so STOP is not right; and the
    question was not measurably better than not asking, so ASK is not either."""
    rows = [_c("a", 0.02, "who?", done=False), _c("s", 0.0, "", done=False, is_stop=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert pairs == [] and man.n_ask_stop_undecided == 1


def test_stop_never_wins_on_value_alone():
    """The whole reason the STOP direction needs a gold key. Every question here has negative
    value (each cost a retrieval and gained nothing), so a value comparison would hand STOP
    the pair and teach the policy to stop whenever asking is expensive."""
    rows = [_c("a", -0.5, "who?", done=False), _c("s", 0.0, "", done=False, is_stop=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert pairs == [] and man.n_stop_chosen == 0


def test_the_outcome_key_can_decide_the_stop_direction():
    """Not done by coverage, but the STOP candidate's episode answered the task and the ASK's
    did not. Outcome outranks gain for ask_stop exactly as it does for ask_ask."""
    rows = [
        _c("a", 0.9, "who?", done=False, correct=0.0),
        _c("s", 0.0, "", done=False, is_stop=True, correct=1.0),
    ]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    p = _only(pairs)
    assert p.chosen_json == STOP_ACTION_JSON and man.n_stop_chosen == 1


def test_the_length_guard_does_not_delete_the_category():
    """A 79-char question against an 18-byte STOP is a delta of 79 -- twice any sane cap. The
    guard is skipped for ask_stop and the pair survives; `len_delta` is still recorded."""
    rows = [_c("a", 0.9, LONG_Q, done=True), _c("s", 0.0, "", done=True, is_stop=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    p = _only(pairs)
    assert p.len_delta > 40 and man.n_len_dropped == 0


def test_the_length_guard_still_applies_to_ask_ask():
    """The regression the exemption above could cause. Two questions still face the cap."""
    rows = [_c("a", 0.9, LONG_Q, done=False), _c("b", 0.1, "who?", done=False)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert pairs == [] and man.n_len_dropped == 1


def test_two_stops_are_dropped_as_stop_stop_not_as_identical():
    """Byte-identical by construction. Filing them under `n_identical_dropped` would read as
    'both candidates asked the same question', which is not what happened."""
    rows = [
        _c("s1", 0.0, "", done=True, is_stop=True),
        _c("s2", 0.0, "", done=True, is_stop=True),
    ]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert pairs == [] and man.n_stop_stop_dropped == 1 and man.n_identical_dropped == 0


def test_ask_ask_pairs_are_labelled_and_unchanged():
    rows = [_c("a", 0.9, "who?", done=False), _c("b", 0.1, "when?", done=False)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    p = _only(pairs)
    assert p.pair_kind == "ask_ask" and p.stop_source == ""
    assert man.n_ask_stop_pairs == 0 and p.margin == 0.8


def test_a_done_state_with_no_recorded_stop_gets_one_synthesised():
    """The commonest shape: gold says the state was finished and every candidate asked anyway.
    Without a synthetic side the state contributes nothing to the STOP decision at all."""
    rows = [_c("a", 0.9, "who?", done=True), _c("b", 0.1, "when?", done=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    synth = [p for p in pairs if p.pair_kind == "ask_stop_synth"]
    assert len(synth) == 1, "exactly one synthetic STOP pair per state, not one per candidate"
    s = synth[0]
    assert s.chosen_json == STOP_ACTION_JSON and s.rejected_json == ask_action_json("who?")
    assert s.stop_source == "synthesised" and s.chosen_run_id == ""
    assert man.n_stop_pairs_synth == 1 and man.n_stop_pairs_recorded_before_cap == 0
    # and it is added BESIDE the ask_ask pair, which is unaffected.
    # WAS `{"ask_ask", "ask_stop"}`. Belief corrected: a gold-derived STOP is not the same
    # evidence as one a candidate took, and on the live corpus it outnumbered the recorded
    # ones 18,227 to 627 -- 73% of the dataset. It carries its own kind so the rung-2 loader
    # can default to sampled contrasts. See tests/test_synth_stop_is_its_own_kind.py.
    assert {p.pair_kind for p in pairs} == {"ask_ask", "ask_stop_synth"}


def test_no_synthetic_stop_where_one_was_recorded():
    rows = [
        _c("a", 0.9, "who?", done=True),
        _c("b", 0.1, "when?", done=True),
        _c("s", 0.0, "", done=True, is_stop=True),
    ]
    _, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert man.n_stop_pairs_synth == 0 and man.n_stop_pairs_recorded_before_cap == 2


def test_no_synthetic_stop_on_an_unfinished_state():
    """A synthetic STOP is only ever justified by gold saying the task was done. Inventing one
    anywhere else would fabricate the very label the dataset is short of."""
    rows = [_c("a", 0.9, "who?", done=False), _c("b", 0.1, "when?", done=False)]
    _, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert man.n_stop_pairs_synth == 0


def test_the_recorded_stop_counter_says_which_stage_it_counts_before():
    """THE NAME IS THE CLAIM, and the old one was false.

    `n_stop_pairs_recorded` read as "recorded ask_stop pairs in this file" and was not: the
    counter is incremented while the pair sits in `here`, and `MAX_PAIRS_PER_STATE` then trims
    `here`. MEASURED on the shipped export -- manifest 1,958 against 1,592 rows carrying
    `pair_kind == "ask_stop"` in `data/rl/pairs.jsonl`, a 366-pair gap that two published
    inventory tables quote as if it were a row count.

    IT IS THE CAP AND NOT A DEDUPE. `export_pairs` runs no dedupe at all
    (`n_exact_duplicate_dropped` is 0 on that manifest), and `n_stop_pairs_synth` is 32,683 in
    the manifest and 32,683 on disk -- a synthetic pair is appended OUTSIDE the cap and loses
    nothing. So the stage the counter precedes is named `_before_cap`, which is what happens.
    """
    stop = _c("s", 0.0, "", done=False, is_stop=True)
    asks = [_c(f"a{i}", 0.9 - 0.01 * i, f"question number {i}?", done=False) for i in range(14)]
    pairs, man = export_pairs([*asks, stop], margin_threshold=-1.0, len_delta_max=40)

    kept = [p for p in pairs if p.pair_kind == "ask_stop"]
    assert man.n_over_cap_dropped > 0, "the fixture must actually exceed the per-state cap"
    assert man.n_stop_pairs_recorded_before_cap == 14, "one ask_stop pair per ASK candidate"
    assert len(kept) < man.n_stop_pairs_recorded_before_cap, (
        "the counter must exceed what survives, or the name makes a claim the code does not"
    )


def test_the_synthetic_stop_counter_needs_no_such_qualifier():
    """The asymmetry that proves the mechanism. A synthetic STOP pair is appended straight to
    `pairs`, never into `here`, so the cap cannot touch it and its counter equals its row count
    exactly -- 32,683 in both the shipped manifest and the shipped file."""
    states = []
    for t in range(20):
        # A DISTINCT STATE PER ITERATION: `state_key` is (suite, task, PARENT, branch turn), so
        # re-parenting is what makes these twenty states rather than one state with forty
        # candidates -- which would be capped, and would test the opposite of the point.
        for run, value, q in (("a", 0.9, "who founded"), ("b", 0.1, "when did it open")):
            row = _c(run, value, f"{q} number {t}?", done=True)
            row["branch_of_run_id"] = f"parent{t}"
            states.append(row)
    pairs, man = export_pairs(states, margin_threshold=0.05, len_delta_max=40)

    synth = [p for p in pairs if p.pair_kind == "ask_stop_synth"]
    assert len(synth) == man.n_stop_pairs_synth > 0
