"""Ranking on anticipation is a CHOICE, and the export must say which one it made.

WHAT `_anticipates` IS. `newly_reachable` is a TRAJECTORY property, not a property of what a
turn retrieved: True iff the turn resolved a required need at depth > 0 whose last
prerequisite landed on the immediately previous turn. That is "took the need at the moment it
became nameable" -- the thesis's vertical claim, stated per turn. `is_latent` is the weaker
statement "the turn resolved something at depth > 0" and is a property of the retrieval alone,
which is why ranking on it would be near-tautological with phi.

WHY IT IS A FLAG AND NOT THE DEFAULT. `scripts/validate_pairs.py` reports "of discordant
pairs, the chosen side pursued the latent need N% of the time" as evidence that ranking on
outcome, speed and gain SELECTS FOR latent pursuit -- an emergent property. The moment
`_anticipates` enters `_prefers`, that check becomes true by construction on every pair it
decides, and quoting it would be circular. So there are two exports and they are different
artifacts: `pairs.jsonl` (rank_rule="outcome", the CONTROL, the only one the latent claim may
be measured on) and `pairs.anticipation.jsonl` (rank_rule="anticipation", the one to train
on). Two orderings must never share one filename or one manifest.

WHERE IT SITS IN THE PRECEDENCE: below outcome, above speed. Outcome is the task actually
being answered and nothing about anticipation outranks that. Speed is a proxy for the same
thing anticipation measures directly, so anticipation goes first when it is enabled.

WHERE IT IS A NO-OP: turn-0 forks. A depth-0 need has no prerequisite, so no turn-0 action is
ever `newly_reachable` and the key cannot fire. Since 46% of the live states are turn-0
forks, `--rank anticipation` changes roughly half the dataset not at all -- worth knowing
before reading a small delta as a small effect.

`rejected_newly_reachable` is emitted regardless of rank: without it the artifact records the
key's input on the winner only, and no audit can check a single decision the key made.
"""

from __future__ import annotations

import pytest

from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_pairs


def _c(run_id, value, question, *, reach=False, correct=None, ttc=None):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": False,
        "done_before": False,
        "newly_reachable": reach,
        "is_latent": reach,
        "latent_depth": 1 if reach else 0,
    }
    if correct is not None:
        r["answer_correct"] = correct
    if ttc is not None:
        r["turns_to_complete"] = ttc
    return r


def _one(rows, **kw):
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000, **kw)
    assert len(pairs) == 1, pairs
    return pairs[0], man


def test_anticipation_flips_a_pair_that_outcome_and_speed_cannot_decide():
    """Same outcome, same speed. The lower-gain candidate took the need at the moment it
    became reachable, and only `--rank anticipation` can see that."""
    rows = [_c("a", 0.9, "who?", reach=False), _c("b", 0.1, "when?", reach=True)]

    ctl, man_ctl = _one(rows, rank="outcome")
    assert ctl.chosen_run_id == "a" and man_ctl.rank_rule == "outcome"

    ant, man_ant = _one(rows, rank="anticipation")
    assert ant.chosen_run_id == "b" and man_ant.rank_rule == "anticipation"
    assert ant.newly_reachable is True and ant.rejected_newly_reachable is False
    assert ant.margin == pytest.approx(0.8), "the magnitude is reported, never negative"


def test_outcome_still_outranks_anticipation():
    """The task was answered. Nothing about when a need became nameable overrides that."""
    rows = [
        _c("a", 0.9, "who?", reach=False, correct=1.0),
        _c("b", 0.1, "when?", reach=True, correct=0.0),
    ]
    p, _ = _one(rows, rank="anticipation")
    assert p.chosen_run_id == "a"


def test_anticipation_outranks_speed():
    """Speed is a proxy for the thing anticipation measures directly, so it goes second."""
    rows = [
        _c("a", 0.9, "who?", reach=False, correct=1.0, ttc=1),
        _c("b", 0.1, "when?", reach=True, correct=1.0, ttc=3),
    ]
    assert _one(rows, rank="outcome")[0].chosen_run_id == "a"
    assert _one(rows, rank="anticipation")[0].chosen_run_id == "b"


def test_the_key_is_a_no_op_when_both_sides_agree():
    """Both reachable, or neither: gain decides, under either rank."""
    for reach in (True, False):
        rows = [_c("a", 0.9, "who?", reach=reach), _c("b", 0.1, "when?", reach=reach)]
        assert _one(rows, rank="anticipation")[0].chosen_run_id == "a"


def test_the_rejected_side_records_the_keys_input_under_either_rank():
    """Emitted regardless of rank: on the control export it is what makes the latent check
    auditable, and on the anticipation export it is what makes the key auditable."""
    rows = [_c("a", 0.9, "who?", reach=True), _c("b", 0.1, "when?", reach=False)]
    for rank in ("outcome", "anticipation"):
        p, _ = _one(rows, rank=rank)
        assert p.newly_reachable is True and p.rejected_newly_reachable is False


def test_a_row_that_never_carried_the_label_is_unknown_not_false():
    """None, never False: a row with no label recorded as measured-negative would let an audit
    read "the rejected side did not anticipate" off a row that was never scored for it."""
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    for r in rows:
        del r["newly_reachable"]
    p, _ = _one(rows, rank="anticipation")
    assert p.rejected_newly_reachable is None


def test_the_default_rank_is_the_control():
    """`export_pairs` called the old way produces the old ordering, byte for byte."""
    rows = [_c("a", 0.9, "who?", reach=False), _c("b", 0.1, "when?", reach=True)]
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert pairs[0].chosen_run_id == "a" and man.rank_rule == "outcome"


def test_an_unknown_rank_is_refused():
    with pytest.raises(ValueError, match="rank"):
        export_pairs([], margin_threshold=0.0, len_delta_max=40, rank="latent")
