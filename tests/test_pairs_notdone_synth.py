"""THE THIRD LESSON THE DATASET NEVER TAUGHT: "you are not done, so do not stop".

THE MEASUREMENT (docs/TRAINING.md 5.3.1-5.3.2, 2026-09-15). Every rung-1 checkpoint
over-stops: re-read per decision point on the three gate parquets, `P(ASK | not done)` is
0.83-0.91 against the prompted base's 0.96 -- the trained policies stop at 9-17% of states
where required evidence was still missing. What they lose at cap 8 they lose by stopping.

WHY THE DATASET COULD NOT HAVE TAUGHT OTHERWISE. Under `gold_coverage_v1` the SFT set holds
"STOP when done" (`stop_done`) and "ASK this good question" (`ask_clears_floor`) and nothing
else. The third case emits NO ROW: not done, and no sampled candidate cleared the noise
floor -- 24,538 states on `data/rl/sft.manifest.json` (`n_no_target_dropped`). That refusal
is right for an IMITATION target (the right question there is one none of the eight
candidates asked, so there is nothing to imitate) and it is silent about the decision the
paper is about. A PREFERENCE does not need a good question to be a true statement: whichever
of these questions was best, it beat stopping, because the task was not finished.

THE RULE (`stop_rule = "gold_coverage_v2"`, opt-in; v1 stays the default of record). At a
state gold says was NOT done, where at least one candidate asked and NO candidate stopped,
emit exactly ONE pair: the best-available ASK CHOSEN over `STOP_ACTION_JSON` REJECTED.

  * `pair_kind="ask_stop_synth"` -- the same kind as the done-state synthetic STOP, because
    the STOP side is the same derived constant and not a rollout. `stop_source` is what
    separates the two directions: "synthesised" (done, STOP chosen) against
    "synthesised_notdone" (not done, ASK chosen). `decided_by="not_done"` names the gold
    fact, which is `done_before is False` and nothing about the sample.
  * ONE per state, outside `MAX_PAIRS_PER_STATE`, for the reason the done-state synth is:
    the cap bounds C(n,2) ASK pairs from an accidental candidate count, and this is one pair
    carrying one gold fact.
  * NEVER where a candidate actually stopped. There the STOP on the table is a MEASURED one
    and the state's contrast is the recorded pair (or the rule's recorded refusal to order
    it); appending a side labelled "synthesised" whose bytes a rollout did emit would be a
    false provenance claim, and the state would carry two contrasts about one decision.
  * NEVER on `done_before is None`. Unknown is not "not done" -- the same asymmetry
    `_state_done` and `export_sft` already enforce.

WHY NOT INSTEAD LOWER THE FLOOR. Exporting the best sub-floor ASK as an SFT target would
teach the policy to imitate a question the measurement says was no better than silence. The
preference asserts strictly less: not "ask this", but "asking beat stopping here".
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_pairs

LONG_Q = "who was the founder of the company that first manufactured the device in question?"


def _c(run_id, value, question, *, done=False, is_stop=False, correct=None, phi=None):
    """One candidate at one state. Copied from tests/test_pairs_ask_stop.py deliberately: the
    two files are about the same decision and a shared helper that drifted would make them
    disagree about what a candidate is."""
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


V2 = "gold_coverage_v2"


def _v2(rows, **over):
    kw = {"margin_threshold": 0.05, "len_delta_max": 40, "stop_rule": V2}
    kw.update(over)
    return export_pairs(rows, **kw)


def _notdone(pairs):
    return [p for p in pairs if p.stop_source == "synthesised_notdone"]


# --------------------------------------------------------------------------- (a) the rule


def test_a_notdone_state_with_no_floor_clearing_candidate_yields_one_ask_chosen_pair():
    """THE STATE v1 DROPS ENTIRELY. Both questions sit under the noise floor, so `export_sft`
    counts the state in `n_no_target_dropped` and the ask_ask pair falls to the margin guard.
    Gold still says the task was unfinished, and that is a fact about the STATE."""
    rows = [_c("a", 0.02, "who?"), _c("b", 0.01, "when?")]
    pairs, man = _v2(rows)

    assert len(pairs) == 1, pairs
    p = pairs[0]
    assert p.chosen_json == ask_action_json("who?")
    assert p.rejected_json == STOP_ACTION_JSON
    assert p.pair_kind == "ask_stop_synth"
    assert p.stop_source == "synthesised_notdone"
    assert p.decided_by == "not_done"
    assert p.label_source == "rule"
    assert p.done_before is False
    # The ASK side is a real rollout and names it; the STOP side is a derived constant and
    # must not claim a run id it never had.
    assert p.chosen_run_id == "a" and p.rejected_run_id == ""
    assert man.n_stop_pairs_synth_notdone == 1
    assert man.n_ask_stop_pairs == 1 and man.n_ask_chosen_over_stop == 1
    # The DONE-state synthetic STOP is a different rule and must not fire here.
    assert man.n_stop_pairs_synth == 0 and man.n_stop_chosen == 0


def test_the_same_state_yields_nothing_under_v1():
    """v1 IS THE DEFAULT OF RECORD AND MUST BE BYTE-UNCHANGED. If this ever produces a pair,
    every shipped `pairs.jsonl` was exported under a rule its manifest does not name."""
    rows = [_c("a", 0.02, "who?"), _c("b", 0.01, "when?")]
    pairs, man = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert pairs == []
    assert man.stop_rule == "gold_coverage_v1"
    assert man.n_stop_pairs_synth_notdone == 0


def test_the_pair_is_added_beside_the_ask_ask_pairs_not_instead_of_them():
    """The v2 pair carries a different fact from the ask_ask contrasts and displaces none."""
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    pairs, man = _v2(rows)
    assert {p.pair_kind for p in pairs} == {"ask_ask", "ask_stop_synth"}
    assert len(_notdone(pairs)) == 1 and man.n_stop_pairs_synth_notdone == 1


def test_the_pair_sits_outside_the_per_state_cap():
    """OUTSIDE `MAX_PAIRS_PER_STATE`, like the done-state synth: the cap bounds C(n,2) ASK
    pairs from an accidental candidate count, and this is one pair carrying one gold fact.
    The counter equals the row count exactly, which is what `_before_cap` names elsewhere."""
    asks = [_c(f"a{i}", 0.9 - 0.01 * i, f"question number {i}?") for i in range(14)]
    pairs, man = _v2(asks, margin_threshold=-1.0)
    assert man.n_over_cap_dropped > 0, "the fixture must actually exceed the per-state cap"
    assert len(_notdone(pairs)) == man.n_stop_pairs_synth_notdone == 1


# --------------------------------------------------------------------------- (b) done states


def test_a_done_state_yields_no_notdone_pair():
    """A done state keeps the v1 synthetic STOP -- STOP chosen, `stop_source="synthesised"` --
    and gets no not-done contrast. Emitting both would assert opposite directions at one
    state on one gold fact."""
    rows = [_c("a", 0.9, "who?", done=True), _c("b", 0.1, "when?", done=True)]
    pairs, man = _v2(rows)
    assert _notdone(pairs) == [] and man.n_stop_pairs_synth_notdone == 0
    synth = [p for p in pairs if p.pair_kind == "ask_stop_synth"]
    assert len(synth) == 1 and synth[0].stop_source == "synthesised"
    assert synth[0].chosen_json == STOP_ACTION_JSON and man.n_stop_pairs_synth == 1


def test_an_unknown_done_before_yields_no_notdone_pair():
    """UNKNOWN IS NOT "NOT DONE" -- the asymmetry `_state_done` and `export_sft` already
    enforce. A legacy row carries no gold-side signal, and reading its absence as "not done"
    would fabricate the direction on exactly the rows nobody can check."""
    rows = [_c("a", 0.02, "who?"), _c("b", 0.01, "when?")]
    for r in rows:
        del r["done_before"]
    pairs, man = _v2(rows)
    assert pairs == [] and man.n_stop_pairs_synth_notdone == 0


# --------------------------------------------------------------------------- (c) which ASK


def test_the_chosen_ask_is_the_highest_value_candidate():
    """BEST AVAILABLE, not first seen. Every candidate here is under the floor, so `value` is
    the only ordering there is; the fixture feeds them out of order so a stable sort on input
    position would pick the wrong one."""
    rows = [_c("a", 0.01, "first?"), _c("b", 0.04, "second?"), _c("c", 0.02, "third?")]
    pairs, _ = _v2(rows)
    p = _notdone(pairs)[0]
    assert p.chosen_json == ask_action_json("second?") and p.chosen_run_id == "b"


def test_the_tie_break_does_not_depend_on_input_order():
    """Equal values must not leave the choice to whatever order the rows arrived in. Two
    candidates, two orderings, one answer -- the same determinism `MAX_PAIRS_PER_STATE`'s
    `pair_id` sort and `_task_cap_key` buy for their own selections."""
    a, b = _c("a", 0.02, "who?"), _c("b", 0.02, "when?")
    first, _ = _v2([a, b])
    second, _ = _v2([b, a])
    assert _notdone(first)[0].chosen_json == _notdone(second)[0].chosen_json
    assert _notdone(first)[0].pair_id == _notdone(second)[0].pair_id


# ------------------------------------------------------------- (d) a recorded STOP wins


def test_a_state_with_a_recorded_ask_stop_gets_no_synthesised_notdone_pair():
    """A CANDIDATE ACTUALLY STOPPED HERE. The recorded pair is the state's contrast; adding a
    side labelled "synthesised" whose bytes a rollout did emit is a false provenance claim,
    and the state would carry two contrasts about one decision."""
    rows = [
        _c("a", 0.9, "who?"),
        _c("b", 0.1, "when?"),
        _c("s", 0.0, "", is_stop=True),
    ]
    pairs, man = _v2(rows)
    assert _notdone(pairs) == [] and man.n_stop_pairs_synth_notdone == 0
    recorded = [p for p in pairs if p.stop_source == "recorded"]
    assert len(recorded) == 2 and man.n_stop_pairs_recorded_before_cap == 2
    assert man.n_ask_chosen_over_stop == 2


def test_a_recorded_stop_the_rule_could_not_order_still_blocks_the_synthesis():
    """The guard is on the CANDIDATE, not on the emitted pair. Here `_ask_stop_decide` refuses
    both directions (not done, and the question misses the floor) so no pair survives -- and
    the state still has a measured STOP in it, so nothing may be synthesised against it."""
    rows = [_c("a", 0.02, "who?"), _c("s", 0.0, "", is_stop=True)]
    pairs, man = _v2(rows)
    assert pairs == [] and man.n_ask_stop_undecided == 1
    assert man.n_stop_pairs_synth_notdone == 0


def test_a_notdone_state_with_no_ask_at_all_is_counted_not_emitted():
    """Every candidate stopped. There is no ASK to put on the chosen side, and inventing a
    question is the one thing a synthetic side may never do -- there is exactly one way to
    stop and no canonical way to ask."""
    rows = [_c("s1", 0.0, "", is_stop=True), _c("s2", 0.0, "", is_stop=True)]
    pairs, man = _v2(rows)
    assert pairs == [] and man.n_stop_pairs_synth_notdone == 0
    assert man.n_notdone_states_no_ask == 1 and man.n_stop_stop_dropped == 1


# ------------------------------------------------------- (e) the rung-2 loader accepts it


def test_the_length_guard_does_not_delete_the_new_category():
    """A 79-char question against an 18-byte STOP. The exemption is keyed on the KIND, and the
    kind is unchanged -- but the direction is reversed (ASK chosen), so this pins that the
    exporter applies no length guard on either side of the new pair."""
    rows = [_c("a", 0.02, LONG_Q), _c("b", 0.01, "when?")]
    pairs, man = _v2(rows)
    p = _notdone(pairs)[0]
    assert p.len_delta > 40 and man.n_len_dropped == 0


def test_the_rung2_loader_keeps_the_new_stop_source(tmp_path):
    """`pair_kind_of` re-derives the kind from the PAYLOADS and accepts `ask_stop_synth` as a
    refinement; `stop_source` is not consulted, so a new value must ride through untouched.
    This is the check the previous split missed: naming one member instead of the SET cost
    rung 2 its preflight on 18,227 pairs (tests/test_synth_stop_is_its_own_kind.py)."""
    from pinq_train.rung2_dpo import assert_length_guard, load_pairs, pair_kind_of

    row = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": ask_action_json(LONG_Q),
        "rejected_json": STOP_ACTION_JSON,
        "margin": 0.02,
        "pair_kind": "ask_stop_synth",
        "stop_source": "synthesised_notdone",
        "decided_by": "not_done",
        "label_source": "rule",
    }
    assert pair_kind_of(row) == "ask_stop_synth"
    assert_length_guard(row, len_delta_max=40)  # must not raise: the STOP kinds are exempt

    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(row) + "\n")
    kept, drops = load_pairs(p, include_pair_kinds=("ask_ask", "ask_stop_synth"))
    assert len(kept) == 1 and drops == {}
    assert kept[0]["stop_source"] == "synthesised_notdone"


def test_the_exported_pair_survives_a_round_trip_through_the_loader(tmp_path):
    """END TO END, on the exporter's own bytes rather than a hand-written row. The loader
    ASSERTS before it filters, so a pair the exporter can write and the loader cannot read
    kills rung 2's preflight on the whole file."""
    from dataclasses import asdict

    from pinq_train.rung2_dpo import load_pairs

    pairs, _ = _v2([_c("a", 0.02, LONG_Q), _c("b", 0.01, "when?")])
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(asdict(x), sort_keys=True) for x in pairs) + "\n")
    kept, _ = load_pairs(p, include_pair_kinds=("ask_ask", "ask_stop_synth"))
    assert [r["stop_source"] for r in kept] == ["synthesised_notdone"]


# --------------------------------------------------------------- (f) the manifest says so


def test_the_manifest_names_the_rule_that_ran():
    """A pairs file written under a different stop rule is a DIFFERENT DATASET, and the
    manifest is where a reader finds out which one they have."""
    rows = [_c("a", 0.02, "who?"), _c("b", 0.01, "when?")]
    _, v2 = _v2(rows)
    _, v1 = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert v2.stop_rule == "gold_coverage_v2"
    assert v1.stop_rule == "gold_coverage_v1"


def test_an_unknown_stop_rule_is_refused():
    """The same refusal `--rank` gets. A typo that silently selected the default would write a
    manifest naming a rule the export did not run."""
    with pytest.raises(ValueError, match="stop_rule"):
        export_pairs([_c("a", 0.02, "who?")], margin_threshold=0.05, stop_rule="v2")


def test_the_two_synth_counters_are_separate_populations():
    """One done state and one not-done state, each with no recorded STOP. They must land in
    different counters: `n_stop_pairs_synth` is STOP-chosen-because-done and
    `n_stop_pairs_synth_notdone` is ASK-chosen-because-not-done. Summing them into one number
    would report a stopping dataset and an anti-stopping one as the same evidence."""
    rows = []
    for t, done in ((0, True), (1, False)):
        for run, value, q in (("a", 0.9, "who founded"), ("b", 0.1, "when did it open")):
            r = _c(run, value, f"{q} number {t}?", done=done)
            # RE-PARENTED, not re-tasked: `state_key` is (suite, task, PARENT, branch turn),
            # so this is two states, and `task_id` stays on the one id the train split holds.
            r["branch_of_run_id"] = f"parent{t}"
            rows.append(r)
    pairs, man = _v2(rows)
    assert man.n_stop_pairs_synth == 1 and man.n_stop_pairs_synth_notdone == 1
    by_source = sorted(p.stop_source for p in pairs if p.pair_kind == "ask_stop_synth")
    assert by_source == ["synthesised", "synthesised_notdone"]
    assert man.n_stop_chosen == 1 and man.n_ask_chosen_over_stop == 1


def test_the_no_ask_counter_is_about_the_rule_that_ran():
    """`n_notdone_states_no_ask` explains why v2 emitted nothing at a state. Under v1 no
    not-done state was ever a candidate for synthesis, so a non-zero value there would
    describe a rule the export did not run."""
    rows = [_c("s1", 0.0, "", is_stop=True), _c("s2", 0.0, "", is_stop=True)]
    _, v1 = export_pairs(rows, margin_threshold=0.05, len_delta_max=40)
    assert v1.n_notdone_states_no_ask == 0


# ------------------------------------------------------------------------ (3) the CLI flag


def test_the_cli_exposes_the_rule_and_defaults_to_v1():
    from pi_run.cli import build_parser

    a = build_parser().parse_args(["train", "export"])
    assert a.stop_rule == "gold_coverage_v1", "v1 stays the default of record"
    a = build_parser().parse_args(["train", "export", "--stop-rule", "gold_coverage_v2"])
    assert a.stop_rule == "gold_coverage_v2"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "export", "--stop-rule", "v2"])


def test_the_flag_reaches_the_exporter():
    """A flag that is parsed and dropped writes a manifest naming a rule the export did not
    run -- the one failure this whole change must not have. Asserted on the SOURCE because
    `cmd_train_export` needs a runs tree and a gold root to execute, and neither belongs in a
    unit test; the call site is what carries the value."""
    from pi_run import cmd_train

    tree = ast.parse(Path(cmd_train.__file__).read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "cmd_train_export"
    )
    calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "export_pairs"
    ]
    assert len(calls) == 1, "cmd_train_export must call export_pairs exactly once"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "stop_rule" in kw, "--stop-rule is parsed and never passed to the exporter"
    assert isinstance(kw["stop_rule"], ast.Attribute) and kw["stop_rule"].attr == "stop_rule"


def test_the_export_help_renders_the_flag():
    """`pi train export --help` must render -- argparse interpolates `%` in help text, so a
    bare percent sign in a new help string is a format spec and a crash. See
    tests/test_train_cli.py::test_every_train_subcommand_renders_its_help."""
    from pi_run.cli import build_parser

    parser = build_parser()
    train = next(
        a.choices["train"]
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction) and "train" in a.choices
    )
    subs = next(a for a in train._actions if isinstance(a, argparse._SubParsersAction))
    rendered = subs.choices["export"].format_help()
    assert "--stop-rule" in rendered and "gold_coverage_v2" in rendered
