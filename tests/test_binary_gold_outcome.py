"""A hedged refusal scores CORRECT on every gold-"no" StrategyQA task, and the label ranks on it.

THE DEFECT. `pi_eval.metrics.quality.contains_answer` matches a normalised token sequence, and
`pi_eval.build.strategyqa_build` writes gold "no" with aliases ("false", "no") and gold "yes"
with ("true", "yes"). So the refusal

    "There is no evidence in the retrieved documents that this is the case."

contains the gold token on every gold-"no" task and scores 1.0 -- and the identical sentence
scores 0.0 on every gold-"yes" task. A grade that is a function of gold polarity alone is not
a grade of the answer.

THE STAKES, measured on the 1,110 strategyqa preference pairs: 250 of the 508 "correct"
gold-"no" labels are hedges, against 4 of 245 on gold-"yes". `_outcome_prefers` puts
`answer_correct` at precedence 1, above quickest and gain, so on those tasks a pair is ordered
by which candidate's rollout happened to hedge; the question it asked is irrelevant.

WHERE THE FIX LIVES, AND WHERE IT DOES NOT.

  * Not in `contains_answer` or `pi_eval.score`. The scorer's hash carries no answer term, so
    a changed definition would ship under an unchanged `scorer_hash` and one hash would denote
    two measurements (AGENTS.md rule 1). `contains_answer` is also right for a free-text span:
    the rescue case it exists for ("does not specify ...; 1960 is mentioned") is real.
  * Not in `pinq_train`. An earlier draft of this file demanded a
    `pinq_train.export.dataset.answer_correct_for_suite`, keyed on the SUITE id. That location
    was wrong twice over: `pinq_train` may never read gold (import-linter contract 1), so it
    could only guess binariness from a suite NAME rather than measure it from the gold VALUE;
    and a suite-wide blanket NaN would discard the 245 legitimately committed gold-"yes"
    answers to kill 4 false ones.
  * In `pi_run.cmd_train._outcome_fields`, gold-side, where `graph.answer` is readable and the
    exporter already recomputes `answer_correct` independently of the scorer. The gate is on
    the ANSWER, not the suite: gold normalises to a yes/no/true/false token AND the answer is
    hedged. A committed "No, Aristotle did not use a laptop." keeps its 1.0.

WHY NaN, AND WHY ON GOLD-"YES" TOO, when today's 0.0 there is not obviously wrong. A hedge is
a non-answer whatever the gold's polarity, and the exporter already has the right semantics
for a non-answer: NaN, on which `_outcome_prefers` abstains and gain decides. The 0.0 on
gold-"yes" is not a measured wrong; it is the same non-answer graded from the other side.
Blanking only the gold-"no" side would keep the label's treatment of an identical hedge
polarity-dependent -- the defect's own shape, one polarity narrower -- and would let outcome
rank a committed "yes" above a hedge on gold-"yes" tasks, which trains prompt compliance
(hedge rate varies 16.7%-83.3% across arms; see `is_hedged`), not question choice.

THE GUARD IS COUNTED. Every row it blanks carries `binary_gold_hedged=True`, and both exporters
sum that into `ExportManifest.n_binary_gold_hedged_nan`, once per row consumed: a guard that
drops a value without saying how often is a guard nobody can audit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

REFUSALS = (
    "There is no evidence in the retrieved documents that this is the case.",
    "The evidence does not say, so there is no way to answer.",
)
COMMITTED_NO = "No, Aristotle did not use a laptop."
COMMITTED_WRONG = "Yes, he did."


# --------------------------------------------------------------------------- fixtures


def _run_dir(root: Path, text: str | None) -> Path:
    """A run directory as `_outcome_fields` reads it: `outcome.json` carrying the Answerer's
    text at `answer.text`, the shape `pi_run.worker.outcome_dict` writes. `None` writes no
    outcome at all -- the incomplete-run case."""
    d = root / "run"
    d.mkdir(parents=True)
    if text is not None:
        (d / "outcome.json").write_text(json.dumps({"answer": {"text": text}}))
    return d


def _graph(answer: str, aliases: tuple[str, ...] = ()) -> SimpleNamespace:
    """The two attributes `_outcome_fields` reads off a `GoldGraph`. `GoldGraph.answer` is the
    canary-stripped property, so the stub carries the bare word."""
    return SimpleNamespace(answer=answer, gold_aliases=aliases)


def _outcome(root: Path, text: str | None, gold: str, aliases: tuple[str, ...] = ()) -> dict:
    from pi_run.cmd_train import _outcome_fields

    return _outcome_fields(_run_dir(root, text), _graph(gold, aliases))


# --------------------------------------------------------------------------- the scorer, pinned
# These document the defect and must keep passing: they are the reason the fix cannot live in
# `contains_answer`.


def test_a_refusal_scores_correct_on_a_gold_no_task():
    from pi_eval.metrics.quality import contains_answer

    for text in REFUSALS:
        assert contains_answer(text, "no", ("false", "no")) == 1.0, text


def test_the_same_refusal_scores_wrong_on_a_gold_yes_task():
    """The asymmetry: identical non-answers are graded differently by gold polarity alone."""
    from pi_eval.metrics.quality import contains_answer

    assert contains_answer(REFUSALS[0], "yes", ("true", "yes")) == 0.0


def test_outcome_abstains_when_answer_correct_is_unknown():
    """The mechanism the fix relies on already exists and is already correct."""
    from pinq_train.export.dataset import _outcome_prefers

    nan = float("nan")
    assert _outcome_prefers({"answer_correct": nan}, {"answer_correct": 0.0}) is False
    assert _outcome_prefers({"answer_correct": 1.0}, {"answer_correct": nan}) is False
    # and it still orders a pair when both sides are known
    assert _outcome_prefers({"answer_correct": 1.0}, {"answer_correct": 0.0}) is True


# --------------------------------------------------------------------------- the fix


@pytest.mark.parametrize("text", REFUSALS)
def test_a_hedge_on_gold_no_is_unknown_not_correct(tmp_path, text):
    """(a) The defect. Before the fix this returned 1.0 and carried no flag."""
    out = _outcome(tmp_path, text, "no", ("false", "no"))
    assert out["answer_hedged"] == 1.0, "fixture must be a hedge"
    assert math.isnan(out["answer_correct"]), out
    assert out["binary_gold_hedged"] is True


def test_a_committed_no_keeps_its_credit(tmp_path):
    """(b) Hedge-gated, not blanket: the guard fires on the hedge, never on the gold alone."""
    out = _outcome(tmp_path, COMMITTED_NO, "no", ("false", "no"))
    assert out["answer_hedged"] == 0.0, "fixture must not be a hedge"
    assert out["answer_correct"] == 1.0
    assert out["binary_gold_hedged"] is False


def test_a_committed_wrong_answer_stays_wrong(tmp_path):
    """(b, other side) A committed answer that names the other polarity is a measured 0.0,
    and stays one -- the guard must not turn a wrong answer into an unknown."""
    out = _outcome(tmp_path, COMMITTED_WRONG, "no", ("false", "no"))
    assert out["answer_hedged"] == 0.0
    assert out["answer_correct"] == 0.0
    assert out["binary_gold_hedged"] is False


def test_a_hedge_on_gold_yes_is_unknown_not_wrong(tmp_path):
    """(c) Before the fix this returned 0.0. NaN is right: see the module docstring -- the
    0.0 is the same non-answer graded by the other polarity, and blanking one side only would
    keep the label polarity-dependent."""
    out = _outcome(tmp_path, REFUSALS[0], "yes", ("true", "yes"))
    assert out["answer_hedged"] == 1.0
    assert math.isnan(out["answer_correct"]), out
    assert out["binary_gold_hedged"] is True


def test_free_text_gold_is_untouched(tmp_path):
    """(d) The gate is on the GOLD being binary, not on hedging. On a free-text gold a hedge
    keeps `contains_answer`'s value both ways: 0.0 when the span is absent, and 1.0 in the
    rescue case that function exists for."""
    miss = _outcome(tmp_path / "miss", "The evidence does not specify the year.", "1960")
    assert miss["answer_hedged"] == 1.0
    assert miss["answer_correct"] == 0.0
    assert miss["binary_gold_hedged"] is False

    rescue = _outcome(
        tmp_path / "rescue", "The evidence does not specify, but 1960 is mentioned once.", "1960"
    )
    assert rescue["answer_hedged"] == 1.0
    assert rescue["answer_correct"] == 1.0
    assert rescue["binary_gold_hedged"] is False


def test_the_flag_is_present_and_false_on_every_row(tmp_path):
    """The "else False" includes the early returns: a row with no outcome, or an empty answer,
    carries the key so the exporters' count can never KeyError on a partial run."""
    absent = _outcome(tmp_path / "absent", None, "no", ("false", "no"))
    assert absent["binary_gold_hedged"] is False
    assert math.isnan(absent["answer_correct"]) and math.isnan(absent["answer_hedged"])

    empty = _outcome(tmp_path / "empty", "", "no", ("false", "no"))
    assert empty["binary_gold_hedged"] is False
    assert math.isnan(empty["answer_correct"]) and math.isnan(empty["answer_hedged"])


# --------------------------------------------------------------------------- the count


def _row(run_id: str, value: float, **over) -> dict:
    """The minimum a row needs to be consumed by both exporters (same shape as
    tests/test_len_guard_on_question.py): trainable suite, a fork state, provenance."""
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "template_id": None,
        "run_id": run_id,
        "turn_idx": 1,
        "state_text": "S",
        "action_json": f'{{"action":"ASK","question":"who owned it, {run_id}","rationale":"r"}}',
        "value": value,
        "phi_tilde": value,
        "scorer_hash": "sh",
        "graph_version": "v1",
        "branch_of_run_id": "p",
        "branch_turn_idx": 1,
    }
    r.update(over)
    return r


def test_both_exporters_count_the_rows_the_guard_blanked():
    """(e) Once per row CONSUMED -- at the point each exporter reads its input, before split
    refusal, the leak drop and the same-state grouping -- so the number reconciles exactly
    with `sum(binary_gold_hedged)` over the rows file. The row keyed off its fork (`skipped`)
    never enters a pair and is still counted: the count says how much outcome signal the
    guard withheld from the input, not how many pairs happened to notice."""
    from pinq_train.export.dataset import export_pairs, export_sft

    nan = float("nan")
    rows = [
        _row("a", 0.9, answer_correct=nan, answer_hedged=1.0, binary_gold_hedged=True),
        _row("b", 0.5, answer_correct=nan, answer_hedged=1.0, binary_gold_hedged=True),
        _row("c", 0.1, answer_correct=1.0, answer_hedged=0.0, binary_gold_hedged=False),
        _row("d", 0.1, answer_correct=0.0, answer_hedged=1.0),  # free-text hedge: no flag
        _row("skipped", 0.3, answer_correct=nan, binary_gold_hedged=True, branch_turn_idx=0),
    ]
    _, man_sft = export_sft(rows, margin_threshold=0.0)
    _, man_pairs = export_pairs(rows, margin_threshold=0.0, len_delta_max=40)
    assert man_sft.n_binary_gold_hedged_nan == 3
    assert man_pairs.n_binary_gold_hedged_nan == 3

    _, clean_sft = export_sft(rows[2:4], margin_threshold=0.0)
    _, clean_pairs = export_pairs(rows[2:4], margin_threshold=0.0, len_delta_max=40)
    assert clean_sft.n_binary_gold_hedged_nan == 0
    assert clean_pairs.n_binary_gold_hedged_nan == 0
