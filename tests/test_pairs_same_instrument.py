"""TWO CANDIDATES GENERATED UNDER DIFFERENT INSTRUMENTS ARE NOT COMPARABLE.

A preference pair asserts "at this state, A beats B". That claim is only about the QUESTIONS if
everything else about the two rollouts was the same. When the two were produced under different
prompts or a different model pin, the pair also encodes whatever those differ by, and nothing
downstream can separate the two contributions.

THIS IS NOT HYPOTHETICAL. After `answerer_frozen` was fixed, the export contained 266 pairs
(18.9%) whose two sides came from different prompt eras -- states branched once before the fix
and once after. In those pairs the new-prompt side was chosen only 35.3% of the time
(p = 1e-6), while the outcome-driven flips among them were balanced at 48.9% -- so the skew was
NOT the answerer prompt acting through the outcome rule, but some other systematic difference
between the two cohorts. Which is exactly the problem: an uncontrolled difference, of unknown
origin, ranked as if it were a difference in the question.

The repo already refuses this pooling elsewhere -- `matcher_id` rides into `scorer_hash` so rows
scored under two matchers can never be compared. Same rule, same reason.
"""

from __future__ import annotations

from pinq_train.export.dataset import export_pairs


def _c(run_id, value, action, pins="pins-A"):
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
        "pins_sha": pins,
    }


def test_candidates_from_different_instruments_do_not_pair() -> None:
    pairs, man = export_pairs(
        [
            _c("a", 0.9, '{"action":"ASK","question":"x"}', pins="pins-A"),
            _c("b", 0.1, '{"action":"ASK","question":"y"}', pins="pins-B"),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert pairs == [], "a cross-instrument pair reached the export"
    assert man.n_cross_instrument_dropped == 1


def test_same_instrument_still_pairs() -> None:
    pairs, man = export_pairs(
        [
            _c("a", 0.9, '{"action":"ASK","question":"x"}'),
            _c("b", 0.1, '{"action":"ASK","question":"y"}'),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert len(pairs) == 1
    assert man.n_cross_instrument_dropped == 0


def test_a_missing_pin_does_not_silently_pair_with_anything() -> None:
    """An unstamped row cannot be shown to be comparable, so it is refused rather than assumed
    equal -- the same direction `_provenance` takes for scorer_hash."""
    a = _c("a", 0.9, '{"action":"ASK","question":"x"}')
    b = _c("b", 0.1, '{"action":"ASK","question":"y"}')
    b.pop("pins_sha")
    pairs, man = export_pairs([a, b], margin_threshold=0.0, len_delta_max=1000)
    assert pairs == []
    assert man.n_cross_instrument_dropped == 1


def test_two_unstamped_rows_still_pair() -> None:
    """Legacy rows with no pin at all are internally consistent with each other; refusing them
    would delete every pre-existing export for a difference that is not there."""
    a = _c("a", 0.9, '{"action":"ASK","question":"x"}')
    a.pop("pins_sha")
    b = _c("b", 0.1, '{"action":"ASK","question":"y"}')
    b.pop("pins_sha")
    pairs, _ = export_pairs([a, b], margin_threshold=0.0, len_delta_max=1000)
    assert len(pairs) == 1


def test_a_three_way_state_keeps_only_the_within_instrument_pairs() -> None:
    rows = [
        _c("a1", 0.9, '{"action":"ASK","question":"x"}', pins="A"),
        _c("a2", 0.5, '{"action":"ASK","question":"y"}', pins="A"),
        _c("b1", 0.1, '{"action":"ASK","question":"z"}', pins="B"),
    ]
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert len(pairs) == 1 and {pairs[0].chosen_run_id, pairs[0].rejected_run_id} == {"a1", "a2"}
    assert man.n_cross_instrument_dropped == 2
