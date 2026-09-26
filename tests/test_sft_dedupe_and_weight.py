"""The SFT per-task cap was deduplicating by accident, and deleting the scarce class on purpose.

MEASURED on the uncapped export (2026-09-08): 25,734 ASK rows, of which 14,275 -- 55% -- are
exact (task, state_text, question) repeats. Not near-paraphrases: byte-identical prompts with
byte-identical targets, produced by parent runs repeated across days and by forks-of-forks
replaying one prefix, each landing under a different state KEY. The cap at 12 per task removed
16,675 ASK rows (65%), 9,306 of them at evidence-bearing states (70% of that population), and
every one was a state no kept row represented. It also happened to remove most of the
duplicates, which is why nobody noticed: the shipped file looked diverse because the cap had
thinned the repeats along with everything else.

THE TWO OPERATIONS ARE NOT THE SAME AND ONLY ONE IS FREE. Exact dedupe removes 14,275 rows and
destroys no variance whatsoever: within-task distinct-3 RISES from 0.6713 to 0.7919. The cap
then still cuts 3,336 rows from 188 tasks, three of which hold 107 ASK states each. Those are
real, distinct demonstrations on hard multi-hop tasks; flattening them to 12 buys nothing the
mode-collapse gate needs (0.79 against a 0.65 floor) and costs a quarter of the class.

Exposure is balanced by weight instead, as the peer's shard does: `sample_weight` =
1/sqrt(rows in the task), normalised to mean 1 within each kind, so a 107-row task's rows
carry ~0.5 each and a 2-row task's ~1.8, and a trainer that ignores the field trains on
everything -- the right default for the class the dataset is short of.

`max_examples_per_task` survives as an explicit option, default off, so the old behaviour is
reproducible and the ablation ("did the cap matter?") is one flag.
"""

from __future__ import annotations

import math

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.export.dataset import export_sft


def _row(run_id, value, question, *, task="t1", turn=1, state="S", done=False, is_stop=False):
    return {
        "suite_id": "musique",
        "task_id": task,
        "template_id": None,
        "run_id": run_id,
        "turn_idx": turn,
        "state_text": state,
        "action_json": STOP_ACTION_JSON if is_stop else ask_action_json(question),
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": f"p-{run_id}",
        "branch_turn_idx": turn,
        "pins_sha": "pins-A",
        "budget_cap": 24,
        "max_turns": 24,
        "is_stop": is_stop,
        "done_before": done,
        "coverage_before": 1.0 if done else 0.4,
        "frontier_size": 0 if done else 2,
    }


def test_an_exact_repeat_of_state_and_question_is_dropped_and_counted():
    """Two parents re-run on different days reached the same prompt and chose the same best
    question. That is one demonstration, not two."""
    rows = [
        _row("a", 0.9, "who founded it?", state="SAME"),
        _row("b", 0.9, "who founded it?", state="SAME"),
    ]
    ex, man = export_sft(rows, margin_threshold=0.05)
    assert len(ex) == 1
    assert man.n_exact_duplicate_dropped == 1


def test_a_different_question_at_the_same_prompt_is_kept():
    """The dedupe is on the TARGET too. Two demonstrations that disagree about what to ask from
    one state are exactly the variance the gate exists to protect."""
    rows = [
        _row("a", 0.9, "who founded it?", state="SAME"),
        _row("b", 0.9, "when was it founded?", state="SAME"),
    ]
    ex, man = export_sft(rows, margin_threshold=0.05)
    assert len(ex) == 2 and man.n_exact_duplicate_dropped == 0


def test_a_repeated_stop_at_one_prompt_is_one_row():
    rows = [
        _row("a", 0.9, "x?", state="DONE", done=True),
        _row("b", 0.9, "y?", state="DONE", done=True),
    ]
    ex, man = export_sft(rows, margin_threshold=0.05)
    assert [e.is_stop for e in ex] == [True] and man.n_exact_duplicate_dropped == 1


def test_dedupe_is_deterministic_in_which_copy_survives():
    """Whichever row the file order presents first must not decide it: two exports of the same
    rows in different orders keep the same representative, or the artifact changes under a
    reordering nothing records."""
    a = _row("a", 0.9, "who?", state="SAME")
    b = _row("b", 0.9, "who?", state="SAME")
    ex1, _ = export_sft([a, b], margin_threshold=0.05)
    ex2, _ = export_sft([b, a], margin_threshold=0.05)
    assert ex1[0].run_id == ex2[0].run_id


def test_sample_weight_favours_the_sparse_task_without_deleting_the_dense_one():
    """Task ids t7 (dense) and t8 (sparse) bucket to TRAIN under `pinq.splitting`; the first
    draft used "dense"/"sparse", which do not, and lost two rows to the split firewall."""
    rows = [
        _row(f"d{i}", 0.9, f"question number {i}?", task="t7", state=f"S{i}") for i in range(16)
    ]
    rows += [
        _row("s0", 0.9, "who?", task="t8", state="A"),
        _row("s1", 0.9, "when?", task="t8", state="B"),
    ]
    ex, man = export_sft(rows, margin_threshold=0.05)
    assert len(ex) == 18, "nothing deleted"
    dense = [e for e in ex if e.task_id == "t7"]
    sparse = [e for e in ex if e.task_id == "t8"]
    assert dense[0].sample_weight < sparse[0].sample_weight
    assert dense[0].sample_weight == pytest.approx(sparse[0].sample_weight / math.sqrt(8))
    assert sum(e.sample_weight for e in ex) == pytest.approx(len(ex)), "normalised to mean 1"


def test_the_cap_still_works_when_asked_for():
    rows = [
        _row(f"d{i}", 0.9, f"question number {i}?", task="t7", state=f"S{i}") for i in range(16)
    ]
    ex, man = export_sft(rows, margin_threshold=0.05, max_examples_per_task=12)
    assert len(ex) == 12 and man.n_over_task_cap_dropped == 4
