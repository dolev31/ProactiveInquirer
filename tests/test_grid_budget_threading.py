"""The per-grid spend cap, tested by RUNNING the loop rather than by grepping its source.

`test_the_grid_spend_cap_is_shared_across_its_suites` asserts only that four strings appear
in `cmd_run`'s source -- "remaining = spend_cap(", "sub.spend_cap = remaining",
"_billed_usd", "spend cap is exhausted". It never executes the loop, so it passes whether or
not the arithmetic is right: `remaining` could go undecremented, or be decremented by the
wrong amount, and the assertion set would not move. CLAUDE.md rule 2: a test that passes
without the fix is not a test.

The defect it is meant to guard is real and was expensive to find: `--spend-cap` was passed
UNCHANGED to each suite, so tier1_confirmatory (musique, strategyqa, wiki2) could bill three
times the stated cap.

These tests drive the real `cmd_run` over a real multi-suite grid with the per-suite call
stubbed, so what is asserted is the budget that each suite actually receives.
"""

from __future__ import annotations

import argparse

import pytest


def _drive(cap, per_suite_cost, grid="conf/grids/tier1_confirmatory.yaml"):
    """Run cmd_run's grid loop, capturing the cap handed to each suite.

    `--allow-dirty` IS PART OF THE FIXTURE, not a relaxation. `cmd_run` refuses a `--sweep`
    on a dirty tree before dispatching any suite (`dirty_sweep_refusal`, 2026-09-17), so
    without this flag the tree state of whoever runs pytest decides whether the budget loop
    executes at all: on a dirty checkout `seen` stayed empty, three of these tests failed on
    a stale assertion and `test_the_cap_is_shared_not_repeated` raised a bare `IndexError`
    off `caps[0]` -- a failure that says nothing about any budget.

    Worse, `test_a_zero_cap_skips_every_suite` PASSED on a dirty tree while the refusal fired:
    "no suite dispatched, non-zero rc" is exactly what a refusal produces, so the assertion
    was satisfied by a code path that never consulted the cap. CLAUDE.md rule 2.

    The refusal keeps its own coverage in tests/test_dirty_sweep_refusal.py, including that
    `--allow-dirty` reaches the grid checks past it. These tests are about the arithmetic.
    """
    from pi_run import cli

    real = cli.cmd_run
    seen: list[tuple[str, float | None]] = []

    def _stub(sub: argparse.Namespace) -> int:
        seen.append((sub.suite, sub.spend_cap))
        sub._billed_usd = per_suite_cost
        return 0

    a = cli.build_parser().parse_args(
        ["run", "--sweep", grid, "--allow-dirty"]
        + ([] if cap is None else ["--spend-cap", str(cap)])
    )
    try:
        cli.cmd_run = _stub  # the loop's recursive call resolves from module globals
        rc = real(a)
    finally:
        cli.cmd_run = real
    return rc, seen


def test_the_grid_has_more_than_one_suite() -> None:
    """Otherwise the whole test proves nothing about sharing."""
    import yaml

    g = yaml.safe_load(open("conf/grids/tier1_confirmatory.yaml"))
    assert len(g["suites"]) >= 3, g["suites"]


def test_the_cap_is_shared_not_repeated() -> None:
    """The bug: each suite got the full cap, so a 3-suite grid could bill 3x."""
    _, seen = _drive(cap=1.00, per_suite_cost=0.40)
    caps = [c for _, c in seen]
    assert caps[0] == pytest.approx(1.00)
    assert caps[1] == pytest.approx(0.60), f"second suite got {caps[1]}, not the remainder"
    assert caps[2] == pytest.approx(0.20), f"third suite got {caps[2]}, not the remainder"
    assert sum(caps) < 3.0, "the cap is being repeated per suite"


def test_a_suite_reached_with_nothing_left_is_skipped() -> None:
    """And the run reports failure rather than pretending the grid completed."""
    rc, seen = _drive(cap=1.00, per_suite_cost=0.60)
    assert [s for s, _ in seen] == ["musique", "strategyqa"], seen
    assert rc != 0, "a skipped suite must not report success"


def test_an_uncapped_grid_threads_none_to_every_suite() -> None:
    _, seen = _drive(cap=None, per_suite_cost=5.0)
    assert [c for _, c in seen] == [None, None, None]


def test_the_budget_decrements_by_what_was_actually_billed() -> None:
    """Not by an estimate, and not by a constant: `_billed_usd` is the suite's own report."""
    _, seen = _drive(cap=2.00, per_suite_cost=0.25)
    assert [c for _, c in seen] == [
        pytest.approx(2.00),
        pytest.approx(1.75),
        pytest.approx(1.50),
    ]


def test_a_zero_cap_skips_every_suite() -> None:
    """--spend-cap 0 now means spend nothing; no suite may be dispatched."""
    rc, seen = _drive(cap=0, per_suite_cost=0.10)
    assert seen == [], f"a zero cap dispatched {seen}"
    assert rc != 0
