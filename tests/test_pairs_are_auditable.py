"""A PAIR MUST DESCRIBE BOTH SIDES, or the dataset cannot be validated from the artifact.

`PreferencePair` carried `latent_depth`/`is_latent` for the CHOSEN candidate only, and
`export_sft` stamps the PARENT run id (the state key) rather than the candidate's. So on the
1,525-pair export, ZERO pairs could be joined back to their candidates' properties -- the
question "does the chosen candidate pursue a latent need more often than the rejected one?"
was unanswerable from the files a trainer consumes.

That question is the whole claim. A preference dataset whose central property cannot be checked
without re-deriving it from run directories is not a validated dataset; it is one nobody has
checked yet.
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


def _pair(a, b):
    pairs, _ = export_pairs([a, b], margin_threshold=0.0, len_delta_max=1000)
    assert pairs, "fixture produced no pair"
    return pairs[0]


def test_the_rejected_sides_latent_facts_are_carried() -> None:
    p = _pair(
        _c(
            "win",
            0.9,
            '{"action":"ASK","question":"deep one"}',
            is_latent=True,
            latent_depth=2,
            turns_to_complete=1,
        ),
        _c(
            "lose",
            0.1,
            '{"action":"ASK","question":"shallow one"}',
            is_latent=False,
            latent_depth=0,
            turns_to_complete=5,
        ),
    )
    assert p.is_latent is True and p.latent_depth == 2
    assert p.rejected_is_latent is False, "the rejected side's latency is not recorded"
    assert p.rejected_latent_depth == 0
    assert p.rejected_turns_to_complete == 5


def test_the_contrast_is_computable_from_the_pair_alone() -> None:
    """The validation the artifact must support without touching run directories."""
    p = _pair(
        _c("win", 0.9, '{"action":"ASK","question":"a"}', is_latent=True, turns_to_complete=1),
        _c("lose", 0.1, '{"action":"ASK","question":"b"}', is_latent=False, turns_to_complete=4),
    )
    assert p.is_latent and not p.rejected_is_latent  # chosen pursues the latent need
    assert p.turns_to_complete < p.rejected_turns_to_complete  # and reaches gold sooner


def test_the_rejected_outcome_is_carried_too() -> None:
    """Outcome ORDERS a pair when the two disagree; the loser's outcome is what makes that
    decision auditable rather than asserted."""
    p = _pair(
        _c("win", 0.1, '{"action":"ASK","question":"a"}', answer_correct=1.0),
        _c("lose", 0.9, '{"action":"ASK","question":"b"}', answer_correct=0.0),
    )
    assert p.chosen_run_id == "win", "outcome should have overridden the gain"
    assert p.answer_correct == 1.0
    assert p.rejected_answer_correct == 0.0


def test_defaults_are_absent_not_wrong_when_a_row_lacks_the_field() -> None:
    """A row with no latent label must not be recorded as 'not latent': -1/NaN says unknown,
    False says measured-and-negative, and the two must not be confused in an audit."""
    p = _pair(
        _c("win", 0.9, '{"action":"ASK","question":"a"}'),
        _c("lose", 0.1, '{"action":"ASK","question":"b"}'),
    )
    assert p.rejected_latent_depth == -1
    assert p.rejected_turns_to_complete == -1
    assert p.rejected_answer_correct != p.rejected_answer_correct  # NaN
