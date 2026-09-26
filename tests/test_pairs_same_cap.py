"""TWO CANDIDATES RUN UNDER DIFFERENT BUDGET CAPS ARE NOT A PREFERENCE PAIR.

A pair asserts "at this state, A beats B". With the cap now overridable on the fork path
(`sample-candidates --budget-cap/--max-turns`), a state can hold a cap-8 cohort and a cap-24
cohort of candidates. Their fork-turn ACTIONS are comparable -- same state, same seeds, same
request bytes -- but the LABEL that orders a pair is episode-level: `answer_correct`,
`turns_to_complete`, `evidence_coverage` all depend on how far the continuation was allowed to
run. A cap-24 candidate beats a cap-8 twin on outcome by construction, and the pair would
record that as a preference over the question.

`pins_sha` does not include the cap, so before this guard such a pair passed every check and
was exported. This is the same rule as `test_pairs_same_instrument.py`: a systematic
difference of known origin, ranked as if it were a difference in the question. The first test
here failed before the guard with one exported pair and `n_cross_cap_dropped == 0`.

The guard is COUNTED, not hidden in the state key: a reader of the manifest sees how many
pairs the two cohorts would have formed, and the cap rides on every row so a consumer can
filter on it later.
"""

from __future__ import annotations

from pinq_train.export.dataset import export_pairs, export_sft


def _c(run_id, value, action, *, cap=8, mt=16):
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": action,
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": cap,
        "max_turns": mt,
    }


ASK_X = '{"action": "ASK", "question": "x?", "rationale": ""}'
ASK_Y = '{"action": "ASK", "question": "y?", "rationale": ""}'
ASK_Z = '{"action": "ASK", "question": "z?", "rationale": ""}'


def test_candidates_under_different_caps_do_not_pair() -> None:
    pairs, man = export_pairs(
        [_c("a", 0.9, ASK_X, cap=8, mt=16), _c("b", 0.1, ASK_Y, cap=24, mt=24)],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert pairs == [], "a cross-cap pair reached the export"
    assert man.n_cross_cap_dropped == 1


def test_same_cap_still_pairs_and_carries_the_cap() -> None:
    pairs, man = export_pairs(
        [_c("a", 0.9, ASK_X), _c("b", 0.1, ASK_Y)], margin_threshold=0.0, len_delta_max=1000
    )
    assert len(pairs) == 1 and man.n_cross_cap_dropped == 0
    assert pairs[0].budget_cap == 8 and pairs[0].max_turns == 16


def test_a_cap_missing_on_one_side_refuses() -> None:
    """None versus an int is a cohort of unknown budget against a known one -- refuse, the
    same asymmetry the pins guard applies."""
    rows = [_c("a", 0.9, ASK_X), _c("b", 0.1, ASK_Y)]
    rows[1]["budget_cap"] = None
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert pairs == [] and man.n_cross_cap_dropped == 1


def test_both_caps_missing_still_pairs() -> None:
    """A legacy export whose rows predate the field is unchanged."""
    rows = [_c("a", 0.9, ASK_X), _c("b", 0.1, ASK_Y)]
    for r in rows:
        r.pop("budget_cap")
        r.pop("max_turns")
    pairs, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert len(pairs) == 1 and pairs[0].budget_cap is None


def test_three_candidates_keep_only_the_within_cap_pairs() -> None:
    pairs, man = export_pairs(
        [_c("a", 0.9, ASK_X), _c("b", 0.5, ASK_Y), _c("c", 0.1, ASK_Z, cap=24, mt=24)],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert {(p.chosen_run_id, p.rejected_run_id) for p in pairs} == {("a", "b")}
    assert man.n_cross_cap_dropped == 2


def test_sft_counts_a_state_with_mixed_caps() -> None:
    _, man = export_sft(
        [_c("a", 0.9, ASK_X), _c("b", 0.1, ASK_Y, cap=24, mt=24)], margin_threshold=0.0
    )
    assert man.n_states_mixed_cap == 1
