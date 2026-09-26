"""`pi suites eligibility` must ask the same question the report asks, or it answers a
different one and reassures about a table it never looked at.

`cmd_suites._ELIGIBLE_CORE` is a copy of `pi_eval.report.ELIGIBLE` minus the two
reconciliation clauses, which need columns a plain parquet read does not carry. A duplication
defended by a comment is a duplication that drifts; this file is the test that makes it one.

THE SPLIT CLAUSE, AND WHAT USED TO BE HERE. This file's second test asserted, as an executable
statement, that the report's predicate contained **no** split clause -- the defect the command
existed to surface -- and instructed whoever added one to come here and record the decision
rather than let every table's n move unremarked. The decision was taken on 2026-09-15 (plan v4
SS8.6): `ELIGIBLE` gains `AND r.split = 'test'`, and the command shares it. The three tests
below are that record.

Measured the day it landed, over `scores/parquet/runs.parquet`: under the report's full
predicate 34,155 train and 1,976 dev rows left the eligible set and 1,537 test rows remain
(musique 14,152 train + 1,368 dev, strategyqa 14,056 + 360, wiki2 4,548 + 248, tau2_airline
493 train, tau2_retail 906 train); under the command's core predicate, which does not carry the
reconciliation clauses, drgym's 174 core-eligible train rows and 51 dev rows go with them.
Every table rendered before that date was computed over a different population and has to be
re-rendered; no grid was re-run, because the grids already ran the tasks -- what changed is
which of their rows may be reported.
"""

from __future__ import annotations

import argparse

import pytest

from pi_run.cmd_suites import (
    _BY_DESIGN_CLAUSES,
    _CORE_CLAUSES,
    _ELIGIBLE_CORE,
    _ELIGIBLE_CORE_ANY_SPLIT,
    _HELD_OUT,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _report_eligible() -> str:
    from pi_eval.report import ELIGIBLE

    return ELIGIBLE


def test_the_commands_predicate_is_a_subset_of_the_reports():
    """Every clause the command filters on must be a clause the report also filters on.

    Normalised on whitespace because `ELIGIBLE` is assembled from adjacent string literals
    with comments between them, so its spacing is an accident of formatting rather than a
    property worth asserting.
    """
    report = " ".join(_report_eligible().split())
    for clause in _ELIGIBLE_CORE.split(" AND "):
        assert " ".join(clause.split()) in report, clause


def test_the_reports_predicate_filters_on_the_held_out_split():
    """THE DECISION, AS AN ASSERTION, in the place the old finding was asserted.

    A predicate is enforced everywhere and cannot be forgotten by the next grid anyone writes;
    `split: test` per grid is narrower and one forgotten line away from recurring. `report.py`
    makes that argument for itself in its own first paragraph -- eligibility is a predicate,
    not a habit -- and this clause is that paragraph applied to the split.

    WHAT THE OLD FINDING MEASURED, kept because it is what the clause removes. Measured on the
    compacted runs on 2026-09-02: of the rows eligible to reach a table, musique was 1,467
    train against 26 test, drgym 174/51/81, and only tau2 was clean -- and tau2 only because it
    is eval-only, so `split_of` forces every task to test regardless of the grid. drgym joined
    `EVAL_ONLY_SUITES` on 2026-09-15 and did NOT become clean the way tau2 is: re-measured the
    same day it was still 174/51/81. drgym's train and dev stamps were written by campaigns
    that predate the decision and are left alone on purpose -- `split` is not in
    `pinq.ids.SEMANTIC_FIELDS`, so rewriting them would move no run_id and buy nothing.
    Eval-only stops those rows being MINED, by name, at both export doors; it never
    retroactively restamped them, and until this clause landed this predicate is what still let
    them reach a table. "Eval-only" and "reports only over held-out tasks" are two properties,
    and this suite is the case that separates them.
    """
    assert _HELD_OUT in _report_eligible(), (
        "pi_eval.report.ELIGIBLE no longer filters on split. A run on a task the policy may "
        "have trained on is then eligible to reach a published table, and nothing in the "
        "rendered output says which split it came from."
    )


def test_the_command_shares_the_clause_rather_than_stating_its_own():
    """One decision, one string. The command exists to say what the report WOULD include, so a
    split clause in one and not the other is precisely the drift this file was written to
    catch -- and it would be invisible, because both sides still print a plausible table."""
    assert _HELD_OUT in _ELIGIBLE_CORE
    assert _ELIGIBLE_CORE == f"{_ELIGIBLE_CORE_ANY_SPLIT} AND {_HELD_OUT}"


def test_the_pre_clause_population_is_kept_so_the_command_can_price_the_clause():
    """What the clause REMOVES is the number the decision record needs, and it is unrecoverable
    from the post-clause query alone: filtered on `split = 'test'`, the train and dev columns
    are zero by construction and the command would print a table that can only ever say
    "clean". `_ELIGIBLE_CORE_ANY_SPLIT` is that population, and it must carry no split clause
    of its own or it is the same query twice."""
    assert "split" not in _ELIGIBLE_CORE_ANY_SPLIT.lower()


def test_the_core_clause_list_matches_the_core_string():
    """`_CORE_CLAUSES` exists so the eligibility breakdown can attribute a removed row to one
    clause by name. Spelled out a second time next to `_ELIGIBLE_CORE_ANY_SPLIT` rather than
    parsed out of it, so this is the test that keeps the two from drifting apart -- the same
    defect this file's first test exists to catch between the command and the report."""
    assert " AND ".join(frag for _, frag in _CORE_CLAUSES) == _ELIGIBLE_CORE_ANY_SPLIT


def test_by_design_clauses_are_a_subset_of_the_core_clauses():
    assert set(_BY_DESIGN_CLAUSES) <= {lbl for lbl, _ in _CORE_CLAUSES}


# --------------------------------------------------------------- REPORTS NOTHING, two ways
#
# `pi suites eligibility` printed the SAME advisory -- re-run this suite on `split: test` --
# whether a suite had zero test rows ON DISK or had 344 of them, all excluded BY DESIGN
# because they are flagged `exploratory` (a forked rollout inherits a foreign prefix and must
# never pool with our own arms under one task_id; CLAUDE.md's firewall section and
# `pi_eval.fork_report`'s own docstring say the same thing about forks specifically). Re-running
# the by-design suite reproduces the identical exploratory-flagged rows for zero benefit -- a
# real lane was assigned exactly that re-run today. These two tests pin the fix: the by-design
# case must stop advising a re-run, and the genuinely-empty case must keep advising one, so the
# fix cannot be satisfied by deleting the advice everywhere.


def _row(rid: str, *, suite: str, split: str = "test", **overrides) -> dict:
    """One ELIGIBLE-shaped row. Every column `_ELIGIBLE_CORE_ANY_SPLIT` filters on is spelled
    out explicitly, the same discipline `test_suites_range.py::_run_row` uses, so a fixture
    that forgets a column cannot be mistaken for one that sets it correctly."""
    row = dict(
        run_id=rid,
        suite_id=suite,
        split=split,
        status="ok",
        gold_exposed=False,
        is_dev_run=False,
        dirty=False,
        pilot_flag=False,
        canary_hit=False,
        exploratory=False,
        firewall_ok=True,
        counterfactual_kind="none",
    )
    row.update(overrides)
    return row


def _build_runs(tmp_path, rows: list[dict]):
    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), d / "runs.parquet")
    return d


def _eligibility(tmp_path, rows: list[dict]) -> int:
    from pi_run.cmd_suites import cmd_suites_eligibility

    d = _build_runs(tmp_path, rows)
    a = argparse.Namespace(parquet=str(d), root=None, json=False)
    return cmd_suites_eligibility(a)


def test_a_suite_excluded_by_design_must_not_get_the_rerun_advisory(tmp_path, capsys):
    """THE CENTRAL TEST. `acme` has three test-split rows, every one of them `exploratory`
    (a fork campaign's shape), plus one train row so the suite surfaces in the table at all.
    Re-running `acme` on `split: test` would produce the identical exploratory-flagged rows --
    that clause does not read the split, it reads what the run IS. The fix must recognise
    that and stop printing the re-run advisory for this suite; today it prints it."""
    rows = [
        _row("acme-train", suite="acme", split="train"),
        _row("acme-fork-1", suite="acme", split="test", exploratory=True),
        _row("acme-fork-2", suite="acme", split="test", exploratory=True),
        _row("acme-fork-3", suite="acme", split="test", exploratory=True),
    ]
    code = _eligibility(tmp_path, rows)
    out = capsys.readouterr().out
    assert "acme" in out
    assert "Re-run" not in out, (
        "acme's test rows are excluded BY DESIGN (exploratory/fork), not absent -- re-running "
        "cannot admit them and the tool must not advise it"
    )
    assert code == 0


def test_a_suite_with_no_test_rows_at_all_still_gets_the_rerun_advisory(tmp_path, capsys):
    """THE GUARD AGAINST OVER-FIXING. `zeta` genuinely has zero test-split rows on disk --
    the grid never ran it on `split: test`. That is exactly the case the re-run advisory was
    written for, and it must survive the fix: a change that deletes the advisory everywhere
    would pass the test above for the wrong reason."""
    rows = [
        _row("zeta-train-1", suite="zeta", split="train"),
        _row("zeta-train-2", suite="zeta", split="train"),
        _row("zeta-dev-1", suite="zeta", split="dev"),
    ]
    code = _eligibility(tmp_path, rows)
    out = capsys.readouterr().out
    assert "zeta" in out
    assert "Re-run" in out, "a suite with no test rows on disk at all must still be told to re-run"
    assert code == 0


def test_the_per_clause_breakdown_names_the_clause_responsible(tmp_path, capsys):
    """The advisory must say WHICH clause removed the rows, not just that some clause did --
    that is the difference between a census and a verdict."""
    rows = [
        _row("acme-train", suite="acme", split="train"),
        _row("acme-fork-1", suite="acme", split="test", exploratory=True),
        _row("acme-fork-2", suite="acme", split="test", exploratory=True),
    ]
    _eligibility(tmp_path, rows)
    out = capsys.readouterr().out
    assert "exploratory" in out
    assert "2" in out
