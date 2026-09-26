"""One corrupt state must not kill a 57-minute export.

WHAT HAPPENED. `_state_done` raises when candidates at one state disagree about
`done_before` -- correct, and it earned its keep: a `pi train export` over 206,088 rows died
on `musique/2hop__11804_11827@1` after 57 minutes of work. Re-reading those 32 candidates
afterwards showed them in perfect agreement (`done_before=True`, coverage 1.0 on every one).
The export had read the directories WHILE a fork worker was still writing them, so it saw a
truncated `turns.jsonl` and computed a coverage that no finished run has.

So the guard was right and its BLAST RADIUS was wrong. `collect_rows` already sets the
standard for this file's neighbours -- "a run that cannot be exported is COUNTED, not hidden"
-- and refuses one run at a time. A state whose candidates contradict each other is refused
the same way: skipped, counted in `n_state_done_disagree`, and the rest of the export
survives. Fatal-per-state, not fatal-per-export.

The operational rule this cost us is worth stating too: do not export while fork workers are
writing into the same runs root. The counter is what makes a violation visible rather than
silent -- a non-zero `n_state_done_disagree` on a quiet tree means something real.
"""

from __future__ import annotations

from pinq.actions import ask_action_json
from pinq_train.export.dataset import export_pairs, export_sft


def _c(run_id, value, question, *, done, task="t1"):
    return {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": f"p-{task}",
        "branch_turn_idx": 1,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": False,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.5,
    }


def _rows():
    """One corrupt state (t1) and one healthy state (t2), in that order."""
    return [
        _c("a", 0.9, "who?", done=True, task="t1"),
        _c("b", 0.1, "when?", done=False, task="t1"),
        _c("c", 0.9, "who?", done=False, task="t2"),
        _c("d", 0.1, "when?", done=False, task="t2"),
    ]


def test_export_pairs_skips_the_corrupt_state_and_keeps_the_rest():
    pairs, man = export_pairs(_rows(), margin_threshold=0.05, len_delta_max=1000)
    assert man.n_state_done_disagree == 1
    assert [p.task_id for p in pairs] == ["t2"], "the healthy state must still export"


def test_export_sft_skips_the_corrupt_state_and_keeps_the_rest():
    ex, man = export_sft(_rows(), margin_threshold=0.05)
    assert man.n_state_done_disagree == 1
    assert [e.task_id for e in ex] == ["t2"]


def test_a_clean_export_counts_zero():
    """The counter is only useful if it is silent when nothing is wrong: a non-zero value on a
    quiet tree is then real evidence of a concurrent writer."""
    rows = [r for r in _rows() if r["task_id"] == "t2"]
    _, man_p = export_pairs(rows, margin_threshold=0.05, len_delta_max=1000)
    _, man_s = export_sft(rows, margin_threshold=0.05)
    assert man_p.n_state_done_disagree == 0 and man_s.n_state_done_disagree == 0
