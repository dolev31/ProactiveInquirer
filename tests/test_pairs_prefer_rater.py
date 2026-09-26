"""A rater majority can order a pair, and it is a THIRD artifact, never the default.

WHY THIS KEY EXISTS. `_prefers` orders a tie on outcome, then anticipation, then speed, then
per-turn gain. Every one of those is derived from the trajectory or the gold graph. A6 buys
something none of them are: a reader's ordering of the SAME candidates, at $0.0077 per
preference against A7's $0.0786.

WHY IT IS NOT THE DEFAULT, AND CANNOT BE. `scripts/validate_pairs.py` reports rater agreement
as evidence that the mechanical ordering tracks a careful reader. Rank on the rater and that
becomes true by construction, exactly as ranking on `newly_reachable` made the latent-pursuit
headline circular. So `pairs.jsonl` (rank_rule="outcome") stays the control, and this writes
`pairs.rater.jsonl` -- three orderings, three files, three manifests, and any comparison
between them is between artifacts that differ in one rule.

COVERAGE IS PARTIAL AND MUST STAY VISIBLE. A6 covers the states it was sampled over, not the
corpus. A pair with no rater verdict falls through to the mechanical precedence unchanged, and
the manifest counts how many pairs the key actually touched -- a key that silently did nothing
on 95% of the file would otherwise read as a ranking decision.
"""

from __future__ import annotations

import pytest

from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_pairs


def _c(run_id, value, question, *, correct=None, ttc=None):
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
        "coverage_before": 0.4,
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


def test_a_rater_majority_orders_a_pair_gain_would_order_the_other_way():
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    ctl, _ = _one(rows, rank="outcome")
    assert ctl.chosen_run_id == "a", "gain prefers the higher value"

    p, man = _one(rows, rank="rater", rater_prefs={frozenset(("a", "b")): "b"})
    assert p.chosen_run_id == "b" and p.rejected_run_id == "a"
    assert man.rank_rule == "rater" and man.n_rater_ordered == 1


def test_outcome_still_outranks_the_rater():
    """A candidate whose episode answered the task beats one whose did not, whatever a reader
    preferred. The rater key sits where anticipation sits: below outcome."""
    rows = [_c("a", 0.9, "who?", correct=1.0), _c("b", 0.1, "when?", correct=0.0)]
    p, _ = _one(rows, rank="rater", rater_prefs={frozenset(("a", "b")): "b"})
    assert p.chosen_run_id == "a"


def test_a_pair_with_no_verdict_falls_through_unchanged():
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    p, man = _one(rows, rank="rater", rater_prefs={frozenset(("x", "y")): "x"})
    assert p.chosen_run_id == "a", "no verdict for this pair -> the mechanical order stands"
    assert man.n_rater_ordered == 0


def test_the_manifest_counts_what_the_key_touched():
    """A key that silently did nothing on most of the file must not read as a decision."""
    rows = [_c("a", 0.9, "who?"), _c("b", 0.5, "when?"), _c("c", 0.1, "where?")]
    _, man = export_pairs(
        rows,
        margin_threshold=0.0,
        len_delta_max=1000,
        rank="rater",
        rater_prefs={frozenset(("b", "c")): "c"},
    )
    assert man.n_rater_ordered == 1, "one of the three C(3,2) pairs had a verdict"


def test_rater_rank_without_verdicts_is_refused():
    """Asking for the ordering and supplying nothing to order by is a silent no-op otherwise."""
    with pytest.raises(ValueError, match="rater_prefs"):
        export_pairs([], margin_threshold=0.0, len_delta_max=40, rank="rater")


def test_the_default_is_still_the_control():
    rows = [_c("a", 0.9, "who?"), _c("b", 0.1, "when?")]
    pairs, man = export_pairs(rows, margin_threshold=0.0, len_delta_max=1000)
    assert man.rank_rule == "outcome" and pairs[0].chosen_run_id == "a"
    assert man.n_rater_ordered == 0
