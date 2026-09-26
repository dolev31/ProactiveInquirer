"""A PAIR OF IDENTICAL QUESTIONS IS NOT A PREFERENCE. It is label noise with a margin on it.

MEASURED on the 1,521-pair export: 95 of 159 states (60%) contained at least one pair of
candidates whose action text was byte-identical, and 329 of 1,134 unique candidates were exact
duplicates of another candidate at the same state.

They survived `margin_threshold` because their VALUES differ despite the text being the same:
`phi_tilde` is computed over the evidence a turn actually retrieved, and the retriever is not
guaranteed to return the identical set for two runs of one query (ordering, ties, cache state).
So the export contained pairs asserting "prefer X over X", with a margin large enough to clear
the floor -- which trains the preference model on a distinction that does not exist.

This is not the length guard. `len_delta_max` blocks a pair whose two sides differ by too MUCH
in length; this blocks a pair whose two sides do not differ at all.
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


def test_two_identical_actions_do_not_pair() -> None:
    same = '{"action":"ASK","question":"who founded it?"}'
    pairs, _ = export_pairs(
        [_c("a", 0.9, same), _c("b", 0.1, same)],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert pairs == [], "a pair asserting 'prefer X over X' reached the export"


def test_a_genuine_difference_still_pairs() -> None:
    pairs, _ = export_pairs(
        [
            _c("a", 0.9, '{"action":"ASK","question":"who founded it?"}'),
            _c("b", 0.1, '{"action":"ASK","question":"when was it founded?"}'),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert len(pairs) == 1 and pairs[0].chosen_run_id == "a"


def test_whitespace_only_differences_are_still_identical() -> None:
    """Two renderings of one question are one question. Tokenised, they differ by nothing the
    policy could learn."""
    pairs, _ = export_pairs(
        [
            _c("a", 0.9, '{"action":"ASK","question":"who founded it?"}'),
            _c("b", 0.1, '{"action":"ASK", "question":"who  founded   it?"}'),
        ],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert pairs == []


def test_the_refusal_is_counted_not_silent() -> None:
    """A guard that drops rows without saying how many is a guard nobody can audit."""
    same = '{"action":"ASK","question":"who founded it?"}'
    _, man = export_pairs(
        [_c("a", 0.9, same), _c("b", 0.1, same)],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert getattr(man, "n_identical_dropped", 0) == 1


def test_three_candidates_two_identical_keeps_only_the_informative_pairs() -> None:
    """C(3,2) = 3 pairs; the one between the twins is dropped and the other two survive."""
    same = '{"action":"ASK","question":"who founded it?"}'
    other = '{"action":"ASK","question":"where is it?"}'
    pairs, man = export_pairs(
        [_c("a", 0.9, same), _c("b", 0.5, same), _c("c", 0.1, other)],
        margin_threshold=0.0,
        len_delta_max=1000,
    )
    assert len(pairs) == 2
    assert man.n_identical_dropped == 1
