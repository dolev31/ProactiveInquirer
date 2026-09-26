"""Candidates from one state must group together, or no preference pair can exist.

`export_pairs` grouped on (suite, task, run_id, turn_idx). Every candidate branch is a
SEPARATE RUN with its own run_id -- that is what stops it overwriting its parent's directory
-- so under that key each candidate sat alone in its own group and C(1,2) = 0 pairs. This is
the second half of why `pairs.jsonl` has always been 0 lines: the first half was that nothing
produced candidates at all.

ONLY THE ROW AT THE BRANCH TURN PAIRS. A branch run also records the prefix it replayed
(turns before the fork, identical across candidates) and its own continuation (turns after
the fork, each downstream of a DIFFERENT decision). Pairing the prefix is comparing a turn to
itself; pairing the continuation is comparing two different states while claiming they are
one, which is precisely the error `state_at` refuses to make.
"""

from __future__ import annotations

from dataclasses import asdict

from pinq_train.export.dataset import export_pairs


def _row(**kw):
    base = dict(
        suite_id="musique",
        task_id="t1",
        template_id=None,
        run_id="r0",
        turn_idx=0,
        state_text="S",
        action_json='{"q":"a"}',
        value=0.0,
        phi_tilde=0.0,
        # see the note in tests/test_train_export.py::_row -- a row with no provenance is
        # refused by export_sft/export_pairs rather than defaulted.
        scorer_hash="sh",
        graph_version="v1",
    )
    base.update(kw)
    return base


def test_candidates_of_one_state_pair() -> None:
    rows = [
        _row(
            run_id="c1",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"aaa"}',
            value=0.9,
        ),
        _row(
            run_id="c2",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"bbb"}',
            value=0.1,
        ),
    ]
    pairs, _ = export_pairs(rows, margin_threshold=0.1)
    assert len(pairs) == 1, "candidates of one state did not group"
    assert pairs[0].chosen_json == '{"q":"aaa"}'
    assert pairs[0].rejected_json == '{"q":"bbb"}'


def test_branches_of_different_parents_do_not_pair() -> None:
    rows = [
        _row(run_id="c1", turn_idx=1, branch_of_run_id="pA", branch_turn_idx=1, value=0.9),
        _row(
            run_id="c2",
            turn_idx=1,
            branch_of_run_id="pB",
            branch_turn_idx=1,
            action_json='{"q":"b"}',
            value=0.1,
        ),
    ]
    assert export_pairs(rows, margin_threshold=0.1)[0] == []


def test_different_branch_points_do_not_pair() -> None:
    rows = [
        _row(run_id="c1", turn_idx=1, branch_of_run_id="p", branch_turn_idx=1, value=0.9),
        _row(
            run_id="c2",
            turn_idx=3,
            branch_of_run_id="p",
            branch_turn_idx=3,
            action_json='{"q":"b"}',
            value=0.1,
        ),
    ]
    assert export_pairs(rows, margin_threshold=0.1)[0] == []


def test_only_the_branch_turn_pairs() -> None:
    """Prefix and continuation rows of a branch must not be paired.

    The continuation is the dangerous one: those turns follow DIFFERENT decisions, so pairing
    them asserts a same-state comparison that is not one.
    """
    rows = [
        _row(
            run_id="c1",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"aaa"}',
            value=0.9,
        ),
        _row(
            run_id="c2",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"bbb"}',
            value=0.1,
        ),
        # continuation of each candidate: same turn index, different underlying state
        _row(
            run_id="c1",
            turn_idx=2,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"ccc"}',
            value=0.9,
        ),
        _row(
            run_id="c2",
            turn_idx=2,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"ddd"}',
            value=0.1,
        ),
    ]
    pairs, _ = export_pairs(rows, margin_threshold=0.1)
    assert len(pairs) == 1, f"paired a continuation turn as if it were the branch state: {pairs}"
    assert pairs[0].chosen_json == '{"q":"aaa"}'


def test_a_branch_turn_of_zero_is_not_treated_as_absent() -> None:
    """`branch_turn_idx or turn_idx` would silently mis-key turn 0, the commonest fork."""
    rows = [
        _row(
            run_id="c1",
            turn_idx=0,
            branch_of_run_id="p",
            branch_turn_idx=0,
            action_json='{"q":"aaa"}',
            value=0.9,
        ),
        _row(
            run_id="c2",
            turn_idx=0,
            branch_of_run_id="p",
            branch_turn_idx=0,
            action_json='{"q":"bbb"}',
            value=0.1,
        ),
    ]
    assert len(export_pairs(rows, margin_threshold=0.1)[0]) == 1


def test_unbranched_rows_group_exactly_as_before() -> None:
    """The existing path must not move: two turns of one run are not a pair."""
    rows = [
        _row(run_id="r0", turn_idx=0, value=0.9),
        _row(run_id="r0", turn_idx=1, action_json='{"q":"b"}', value=0.1),
    ]
    assert export_pairs(rows, margin_threshold=0.1)[0] == []


def test_pairs_carry_candidate_run_ids() -> None:
    """A preference pair must name the two candidate RUNS it came from.

    `judgments.parquet` is keyed on `run_id_a`/`run_id_b`; without the candidate run ids on
    `PreferencePair` a human-annotation pass has no join key back to that table.
    """
    rows = [
        _row(
            run_id="c1",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"aaa"}',
            value=0.9,
        ),
        _row(
            run_id="c2",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"bbb"}',
            value=0.1,
        ),
    ]
    pairs, _ = export_pairs(rows, margin_threshold=0.1)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.chosen_run_id == "c1"
    assert pair.rejected_run_id == "c2"
    assert pair.pair_id != ""

    pairs_again, _ = export_pairs(rows, margin_threshold=0.1)
    assert pairs_again[0].pair_id == pair.pair_id, "pair_id must be stable across identical runs"

    swapped_rows = [
        _row(
            run_id="c3",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"aaa"}',
            value=0.9,
        ),
        _row(
            run_id="c4",
            turn_idx=1,
            branch_of_run_id="p",
            branch_turn_idx=1,
            action_json='{"q":"bbb"}',
            value=0.1,
        ),
    ]
    pairs_diff, _ = export_pairs(swapped_rows, margin_threshold=0.1)
    assert pairs_diff[0].pair_id != pair.pair_id, "different candidate run ids must change pair_id"

    d = asdict(pair)
    assert "chosen_run_id" in d
    assert "rejected_run_id" in d
    assert "pair_id" in d
