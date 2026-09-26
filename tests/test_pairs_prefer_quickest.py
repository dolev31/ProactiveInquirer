"""Among candidates that agree on the outcome, prefer the one that reaches gold SOONER.

Per-turn `phi_tilde` scores GAIN NOW. It cannot distinguish a question that opens the chain
from one that pays the same this turn and dead-ends -- and the difference is wide: measured
across runs of one musique task, the fastest reaches complete evidence at a median of 0.5 turns
against the slowest at 4.0.

`turns_to_complete` is that distinction, and until now it was carried on the row and read by
nothing.

THE PRECEDENCE, and why this order:
  1. OUTCOME   -- a candidate that answered the task beats one that did not, whatever it cost.
  2. QUICKEST  -- among candidates that agree on the outcome, fewer remaining turns wins.
  3. GAIN      -- otherwise, per-turn evidence gain, as before.

Each rule fires only when its signal is KNOWN on both sides and actually differs, so an
unlabelled row falls through to the next rule rather than being ranked on a missing value.
"""

from __future__ import annotations

from pinq_train.export.dataset import export_pairs


def _c(run_id, value, action, **kw):
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
    }
    r.update(kw)
    return r


def _one(rows):
    pairs, _ = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    return pairs


def test_the_faster_path_wins_when_gain_disagrees() -> None:
    """Equal outcome, lower gain, but it reaches gold in one turn instead of five."""
    p = _one(
        [
            _c(
                "slow",
                0.9,
                '{"action":"ASK","question":"a"}',
                answer_correct=1.0,
                turns_to_complete=5,
            ),
            _c(
                "fast",
                0.2,
                '{"action":"ASK","question":"b"}',
                answer_correct=1.0,
                turns_to_complete=1,
            ),
        ]
    )
    assert len(p) == 1 and p[0].chosen_run_id == "fast"
    assert p[0].margin > 0.0


def test_outcome_still_outranks_speed() -> None:
    """Fast and wrong loses to slow and right. Reaching gold sooner is worthless if the task
    was not answered."""
    p = _one(
        [
            _c(
                "fastwrong",
                0.9,
                '{"action":"ASK","question":"a"}',
                answer_correct=0.0,
                turns_to_complete=1,
            ),
            _c(
                "slowright",
                0.1,
                '{"action":"ASK","question":"b"}',
                answer_correct=1.0,
                turns_to_complete=6,
            ),
        ]
    )
    assert p and p[0].chosen_run_id == "slowright"


def test_gain_decides_when_speed_ties() -> None:
    p = _one(
        [
            _c(
                "hi",
                0.9,
                '{"action":"ASK","question":"a"}',
                answer_correct=1.0,
                turns_to_complete=2,
            ),
            _c(
                "lo",
                0.1,
                '{"action":"ASK","question":"b"}',
                answer_correct=1.0,
                turns_to_complete=2,
            ),
        ]
    )
    assert p and p[0].chosen_run_id == "hi"


def test_an_unknown_speed_falls_through_to_gain() -> None:
    """-1 means 'this run never reached complete evidence', which is not 'zero turns away'.
    Ranking on it would make a run that never finished look instant."""
    p = _one(
        [
            _c(
                "never",
                0.1,
                '{"action":"ASK","question":"a"}',
                answer_correct=1.0,
                turns_to_complete=-1,
            ),
            _c(
                "hi",
                0.9,
                '{"action":"ASK","question":"b"}',
                answer_correct=1.0,
                turns_to_complete=-1,
            ),
        ]
    )
    assert p and p[0].chosen_run_id == "hi"


def test_a_known_speed_does_not_beat_an_unknown_one() -> None:
    """One side unlabelled means the comparison is not available; fall through rather than
    treating the labelled side as automatically better."""
    p = _one(
        [
            _c(
                "known",
                0.1,
                '{"action":"ASK","question":"a"}',
                answer_correct=1.0,
                turns_to_complete=1,
            ),
            _c(
                "unknown",
                0.9,
                '{"action":"ASK","question":"b"}',
                answer_correct=1.0,
                turns_to_complete=-1,
            ),
        ]
    )
    assert p and p[0].chosen_run_id == "unknown", "ranked on a value one side does not have"
