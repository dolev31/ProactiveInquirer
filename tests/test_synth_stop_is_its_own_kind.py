"""A gold-derived STOP statement is not a sampled preference, and must not pass for one.

WHAT THE FIRST FULL EXPORT SHOWED. 25,746 pairs over 19,047 states -- and 18,227 of those
states contributed exactly ONE pair each: a synthesised STOP. Only 761 states produced a real
ask_ask contrast. The synthetic pair had been written to fire at any state gold called
complete, and a state does not have to be a FORK: every ordinary recorded run contributes one
state per turn, so every over-asking turn in the corpus became a "preference pair". 73% of the
dataset was one 18-byte chosen side.

The statement is TRUE -- the task was already complete, so stopping beat that question, which
is the over-asking signal the work is about (A5 put over-asking at 23.5%). What it is NOT is a
contrast between two candidates SAMPLED at one state. Both sides still sit at the same state,
so the V(s_t) cancellation argument survives; what does not survive is calling it the same
kind of evidence. A default dataset that is three-quarters one synthetic template trains
toward STOP, which is exactly what rung 1's `STOP_SHARE_CEILING` exists to catch.

So it gets its own `pair_kind`. `ask_stop` means a STOP that a candidate actually took;
`ask_stop_synth` means one the exporter derived from gold. `DPOConfig.include_pair_kinds`
defaults to the two SAMPLED kinds, so the synthetic ones are opt-in and the ablation that
measures what they buy is a one-flag change.
"""

from __future__ import annotations

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_pairs
from pinq_train.rung2_dpo import DPOConfig, load_pairs


def _c(run_id, value, question, *, done, is_stop=False):
    return {
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
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
    }


def test_a_synthesised_stop_is_labelled_synth():
    pairs, man = export_pairs(
        [_c("a", 0.9, "who?", done=True), _c("b", 0.1, "when?", done=True)],
        margin_threshold=0.05,
        len_delta_max=1000,
    )
    synth = [p for p in pairs if p.stop_source == "synthesised"]
    assert len(synth) == 1 and synth[0].pair_kind == "ask_stop_synth"
    assert man.n_stop_pairs_synth == 1


def test_a_recorded_stop_keeps_the_sampled_kind():
    """The distinction is the ORIGIN of the STOP, not the direction of the preference."""
    pairs, _ = export_pairs(
        [_c("a", 0.9, "who?", done=True), _c("s", 0.0, "", done=True, is_stop=True)],
        margin_threshold=0.05,
        len_delta_max=1000,
    )
    assert [p.pair_kind for p in pairs] == ["ask_stop"]
    assert pairs[0].stop_source == "recorded"


def test_a_single_candidate_state_still_yields_the_synthetic_pair():
    """It is real supervision and is NOT dropped -- 18,227 states in the live corpus have
    exactly this shape. It is labelled, so a consumer chooses."""
    pairs, _ = export_pairs(
        [_c("a", 0.9, "who?", done=True)], margin_threshold=0.05, len_delta_max=1000
    )
    assert len(pairs) == 1 and pairs[0].pair_kind == "ask_stop_synth"


def test_the_loader_excludes_the_synthetic_kind_by_default(tmp_path):
    """The default is SAMPLED contrasts only. A run that wants the gold-derived statements
    asks for them, and `cfg.sha` then records that it did."""
    import json

    assert DPOConfig(base_model="m", adapter="a").include_pair_kinds == ("ask_ask", "ask_stop")
    p = tmp_path / "pairs.jsonl"
    rows = [
        {
            "suite_id": "musique",
            "task_id": "t",
            "run_id": "r",
            "turn_idx": 1,
            "state_text": "S",
            "chosen_json": ask_action_json("who?"),
            "rejected_json": ask_action_json("when?"),
            "margin": 0.5,
            "pair_kind": "ask_ask",
        },
        {
            "suite_id": "musique",
            "task_id": "t",
            "run_id": "r2",
            "turn_idx": 1,
            "state_text": "S",
            "chosen_json": STOP_ACTION_JSON,
            "rejected_json": ask_action_json("who?"),
            "margin": 0.5,
            "pair_kind": "ask_stop_synth",
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    kept, drops = load_pairs(p)
    assert [r["pair_kind"] for r in kept] == ["ask_ask"] and drops["pair_kind"] == 1

    kept, drops = load_pairs(p, include_pair_kinds=("ask_ask", "ask_stop", "ask_stop_synth"))
    assert len(kept) == 2 and drops == {}


def test_the_derived_kind_accepts_either_stop_label():
    """`pair_kind_of` recomputes from the payloads, which cannot distinguish a recorded STOP
    from a synthesised one -- both are the same constant. So it must accept the label the
    exporter gave rather than overwrite it, while still refusing a label the bytes contradict."""
    import pytest

    from pinq_train.rung2_dpo import pair_kind_of

    base = {"chosen_json": STOP_ACTION_JSON, "rejected_json": ask_action_json("q?")}
    assert pair_kind_of({**base, "pair_kind": "ask_stop_synth"}) == "ask_stop_synth"
    assert pair_kind_of({**base, "pair_kind": "ask_stop"}) == "ask_stop"
    assert pair_kind_of(base) == "ask_stop"  # unlabelled falls back to the derived kind
    with pytest.raises(ValueError, match="pair_kind"):
        pair_kind_of(
            {
                "chosen_json": ask_action_json("a"),
                "rejected_json": ask_action_json("b"),
                "pair_kind": "ask_stop_synth",
            }
        )


def test_the_length_guard_exempts_the_synthetic_kind_too(tmp_path):
    """THE REGRESSION THIS COMMIT FIXES. Splitting `ask_stop_synth` out of `ask_stop` left the
    length-guard exemption matching only the old name, so every synthesised pair -- 18,227 of
    them -- tripped the guard it exists to be exempt from. `load_pairs` asserts before it
    filters (a property every row must have is not something you filter away), so rung 2's
    preflight died on the first synthetic pair even with `include_pair_kinds` excluding them.

    Caught by running the real `pinq_train.rung2_dpo.preflight` on the shipped artifact:
        ValueError: |delta question len| = 57 > 40

    Both kinds put an 18-byte constant against a 40-120 character question. The asymmetry is
    identical and so is the exemption.
    """
    import json

    from pinq_train.rung2_dpo import assert_length_guard

    long_q = ask_action_json("who founded the company that first made the device described here?")
    for kind in ("ask_stop", "ask_stop_synth"):
        pair = {"chosen_json": STOP_ACTION_JSON, "rejected_json": long_q, "pair_kind": kind}
        assert_length_guard(pair, len_delta_max=40)  # must not raise for either

    p = tmp_path / "pairs.jsonl"
    rows = [
        {
            "suite_id": "musique",
            "task_id": "t",
            "run_id": "r",
            "turn_idx": 1,
            "state_text": "S",
            "chosen_json": STOP_ACTION_JSON,
            "rejected_json": long_q,
            "margin": 0.5,
            "pair_kind": "ask_stop_synth",
        }
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    kept, drops = load_pairs(p)  # default excludes the synthetic kind -- but must not RAISE
    assert kept == [] and drops["pair_kind"] == 1
