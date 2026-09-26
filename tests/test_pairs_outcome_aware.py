"""RANKING ON PER-TURN EVIDENCE GAIN ALONE ACCEPTS TRAJECTORIES THAT FAILED THE TASK.

`turn_values` is `w_phi*phi_tilde - w_red*rho - c_ret`. It contains no task term: `reward_of`
computes one (`w_task * q_terminal`) and `turn_values` uses only `br.turns`, dropping it. And
`q_terminal` is itself `evidence_coverage`, not answer correctness -- so nothing anywhere in
the pair ranking knew whether the episode was ultimately answered.

MEASURED on musique: retrieving every required gold span and still answering wrong is common,
not marginal. A pair whose winner did that teaches "ask this" about a question that led
nowhere.

The rule added here is deliberately narrow: episode outcome ORDERS a pair only when the two
candidates DISAGREE about it, and per-turn gain continues to order the rest. Two candidates
that both failed still pair on gain -- the state is informative even when the episode was
not -- and a bigger gain never overrides a task the other candidate actually answered.
"""

from __future__ import annotations

from pinq_train.export.dataset import export_pairs


def _c(run_id, value, correct, action='{"q":"a"}', **kw):
    r = {
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
        "answer_correct": correct,
    }
    r.update(kw)
    return r


def _one(rows):
    pairs, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    return pairs


def test_the_candidate_that_answered_wins_even_with_less_evidence_gain() -> None:
    """The case the old rule got backwards."""
    pairs = _one(
        [
            _c("lost", 0.9, 0.0, '{"q":"high gain, wrong answer"}'),
            _c("won", 0.1, 1.0, '{"q":"low gain, right answer"}'),
        ]
    )
    assert len(pairs) == 1
    assert pairs[0].chosen_run_id == "won"
    assert pairs[0].rejected_run_id == "lost"


def test_gain_still_orders_two_candidates_that_agree_on_outcome() -> None:
    """Both failed: the state is still informative about which question was better."""
    pairs = _one([_c("hi", 0.9, 0.0), _c("lo", 0.1, 0.0, '{"q":"b"}')])
    assert len(pairs) == 1
    assert pairs[0].chosen_run_id == "hi"


def test_gain_orders_two_successes_too() -> None:
    pairs = _one([_c("hi", 0.9, 1.0), _c("lo", 0.1, 1.0, '{"q":"b"}')])
    assert pairs[0].chosen_run_id == "hi"


def test_an_unknown_outcome_falls_back_to_gain_and_does_not_crash() -> None:
    """`answer_correct` is NaN on any suite with no gold answer. NaN must not be read as a
    failure -- that would silently invert every pair on those suites."""
    nan = float("nan")
    pairs = _one([_c("hi", 0.9, nan), _c("lo", 0.1, nan, '{"q":"b"}')])
    assert pairs and pairs[0].chosen_run_id == "hi"

    mixed = _one([_c("known", 0.1, 1.0), _c("unknown", 0.9, nan, '{"q":"b"}')])
    assert mixed and mixed[0].chosen_run_id == "unknown", "NaN was treated as a failure"


def test_the_margin_of_an_outcome_flip_is_not_negative() -> None:
    """`margin` feeds the accept threshold; a flipped pair must report the magnitude of the
    preference, not a negative gain difference that would be filtered away."""
    pairs = _one([_c("lost", 0.9, 0.0), _c("won", 0.1, 1.0, '{"q":"b"}')])
    assert pairs[0].margin > 0.0
