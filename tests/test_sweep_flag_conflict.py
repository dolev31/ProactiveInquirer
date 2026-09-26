"""`--sweep` must not SILENTLY discard the flags typed beside it.

FOUND BY THE CANARY. The plan's Phase 1 command was
`pi run --sweep conf/grids/tier0_canary.yaml --seeds 7 --max-turns 1 --budget-cap 1 --k 2`,
and every one of those flags would have been thrown away: cli.py's grid loop assigns
`sub.seeds = [str(x) for x in grid.seeds]` and the same for n, max_turns, k and budget_cap,
unconditionally. The canary would have run at the grid's values while its operator believed
otherwise -- and on a warm seed, which is the one thing a canary must not do.

This is the SAME CLASS as the bug that billed $0.7158: `--sweep` silently overrode `--arm`.
That one was fixed in 431bada and the fix was never generalised to the other five flags.

The rule here matches the one `--arm` already uses: a conflicting explicit flag is REFUSED,
not silently dropped and not silently honoured. "Run what you typed" and "run what the grid
says" are both wrong answers to a flag the operator did not mean to conflict.

SIXTH FLAG, 2026-09-17 (found by another lane, verified in the code): the grid loop also
assigns `sub.task = list(grid.task_ids) or []` unconditionally, and this check never looked
at `--task`. Unlike the other five, this one has no case where the explicit flag survives --
a grid that declares its own `task_ids` replaces `--task` with them, and a grid that declares
none replaces it with `[]`, which then falls through to `_select_task_ids`'s `n_tasks`-based
count in `cmd_run`. So `pi run --sweep <200-task grid> --task one_task` ran all 200 while its
operator believed they had pinned one -- worse than the other five, which at least change the
unit count in the banner; this one changes nothing an operator would think to look at, and it
is exactly the failure that would corrupt the one-unit validation sweep used to prove a tree
is clean before a large campaign.
"""

from __future__ import annotations


def _args(**kw):
    from pi_run.cli import build_parser

    argv = ["run", "--sweep", "conf/grids/tier0_canary.yaml"]
    for k, v in kw.items():
        argv.append(f"--{k.replace('_', '-')}")
        argv.extend(str(x) for x in (v if isinstance(v, list) else [v]))
    return build_parser().parse_args(argv)


def test_an_unpassed_flag_is_none_so_the_grid_wins() -> None:
    """The default must be distinguishable from an explicit value, or nothing can be checked."""
    a = _args()
    for flag in ("seeds", "n", "max_turns", "k", "budget_cap"):
        assert getattr(a, flag) is None, f"--{flag} default must be None to detect an explicit one"


def test_a_conflicting_seed_is_refused(capsys) -> None:
    from pi_run.cli import grid_flag_conflicts

    conflicts = grid_flag_conflicts(_args(seeds=[7]), seeds=(0, 1), n_tasks=200, max_turns=16, k=5)
    assert any("seeds" in c for c in conflicts), conflicts


def test_a_matching_value_is_not_a_conflict() -> None:
    """Typing the grid's own value is redundant, not wrong."""
    from pi_run.cli import grid_flag_conflicts

    assert grid_flag_conflicts(_args(seeds=[7]), seeds=(7,), n_tasks=3, max_turns=1, k=2) == ()


def test_every_overridden_flag_is_checked() -> None:
    """The bug was that five flags shared one blind spot; a partial fix repeats it. `task` was
    the sixth (2026-09-17).

    "This list must grow with the guard" was the instruction here, and it did not: `budget_cap`
    was the seventh and was never added. As of 2026-09-18 there is no list to grow -- the check
    diffs against the namespace the sweep would dispatch, so these six are a sample of its
    coverage rather than its definition. Kept as the regression test for the six, and the
    coverage of the whole class is asserted by the AST tests at the foot of this file."""
    from pi_run.cli import grid_flag_conflicts

    a = _args(seeds=[9], n=999, max_turns=99, k=9, task=["bogus_task"])
    conflicts = grid_flag_conflicts(a, seeds=(7,), n_tasks=3, max_turns=1, k=2, task_ids=("t0",))
    for flag in ("seeds", "n", "max-turns", "k", "task"):
        assert any(flag in c for c in conflicts), f"{flag} unchecked: {conflicts}"


def test_no_flags_means_no_conflicts() -> None:
    from pi_run.cli import grid_flag_conflicts

    assert grid_flag_conflicts(_args(), seeds=(7,), n_tasks=3, max_turns=1, k=2) == ()


def test_the_standalone_path_still_has_working_defaults() -> None:
    """Making the defaults None must not turn a plain `pi run` into a crash."""
    from pi_run.cli import build_parser, resolved_run_defaults

    a = build_parser().parse_args(["run", "--suite", "musique"])
    d = resolved_run_defaults(a)
    assert d["seeds"] == ["0"]
    assert d["n"] == 5
    assert d["max_turns"] == 16
    assert d["k"] == 5


def test_arm_keeps_its_existing_filter_behaviour() -> None:
    """--arm FILTERS a grid rather than conflicting with it; that fix must not regress."""
    from pi_run.cli import grid_flag_conflicts

    a = _args()
    a.arm = ["drafter_only"]
    assert grid_flag_conflicts(a, seeds=(7,), n_tasks=3, max_turns=1, k=2) == ()


def test_task_default_is_a_list_not_none() -> None:
    """`--task` uses `action="append", default=[]` (it is repeatable), not the `None`
    sentinel the other five flags use -- so the explicit-vs-default distinction for THIS flag
    is "non-empty list" vs "[]", not "not None" vs "None". Both are real sentinels; they are
    just not the same sentinel, and a check that assumed `None` would never fire."""
    a = _args()
    assert a.task == [], "the --task default must stay [] to detect an explicit one"


def test_a_conflicting_task_selection_is_refused() -> None:
    """The grid declares its own explicit task_ids, and they differ from what was passed."""
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier0_canary.yaml", "--task", "musique_0001"]
    )
    conflicts = grid_flag_conflicts(
        a, seeds=(0, 1), n_tasks=3, max_turns=1, k=2, task_ids=("t0", "t1", "t2")
    )
    assert any("task" in c for c in conflicts), conflicts
    assert any("musique_0001" in c for c in conflicts), conflicts
    assert any("t0" in c for c in conflicts), conflicts  # names what the grid declares too


def test_an_explicit_task_conflicts_even_when_the_grid_names_no_tasks_of_its_own() -> None:
    """THE DANGEROUS SHAPE. Most grids have no `task_ids` at all -- they select by `n_tasks`
    instead -- so before this check existed, `--task one_task` was silently discarded by
    `sub.task = list(grid.task_ids) or []` and the sweep ran all `n_tasks` (200 here) while
    its operator believed they had pinned one. There is no grid-declared task list to be
    redundant WITH here; an explicit --task must conflict on its own, because there is no
    shape in which the grid loop lets it survive."""
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier1_confirmatory.yaml", "--task", "musique_0001"]
    )
    conflicts = grid_flag_conflicts(a, seeds=(0, 1), n_tasks=200, max_turns=16, k=5, task_ids=())
    assert any("task" in c for c in conflicts), conflicts
    assert any("musique_0001" in c for c in conflicts), conflicts
    assert any("200" in c for c in conflicts), conflicts  # names what the grid declares instead


def test_a_matching_task_selection_is_not_a_conflict() -> None:
    """Typing the grid's own task_ids, in the grid's own order, is redundant, not wrong."""
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier0_canary.yaml", "--task", "t0", "--task", "t1"]
    )
    assert (
        grid_flag_conflicts(a, seeds=(0,), n_tasks=2, max_turns=1, k=2, task_ids=("t0", "t1")) == ()
    )


def test_no_task_flag_means_no_conflict_even_when_the_grid_declares_tasks() -> None:
    """A refusal that fired on an UNPASSED --task would break every legitimate sweep -- this
    is the same discipline `test_no_flags_means_no_conflicts` checks for the other five."""
    from pi_run.cli import build_parser, grid_flag_conflicts

    a = build_parser().parse_args(["run", "--sweep", "conf/grids/tier0_canary.yaml"])
    assert (
        grid_flag_conflicts(a, seeds=(0,), n_tasks=2, max_turns=1, k=2, task_ids=("t0", "t1")) == ()
    )


# --------------------------------------------------------------------------- the CLASS, not the
# instance
#
# SEVENTH FLAG, 2026-09-18: `sub.budget_cap = grid.budget_cap` is the same unconditional
# assignment and this file's checker never inspected `budget_cap` -- although
# `grid_flag_conflicts`' own docstring named it among the five the assignment broke. Measured
# at d7fa4e8: `grid_flag_conflicts(Namespace(budget_cap=4), <cap-8 grid>)` -> `()`, so
# `pi run --sweep conf/grids/frames_trained.yaml --budget-cap 4` was accepted, the flag was
# discarded, and the campaign ran at 8. Someone ends up with a directory of runs they believe
# are one cap and are not, and `budget_cap` is the axis of the whole spend curve
# (frames_trained_cap{4,8,12,24} differ in that field and in nothing else), so the wrong cap
# is not a slow run -- it is a mislabelled point on a published curve.
#
# The tests below drive the REAL `cmd_run` against a REAL grid, because that is the only
# surface that is handed a whole grid; `grid_flag_conflicts`' loose keywords cannot express a
# grid's budget_cap at all, which is why the blind spot survived a unit test per flag. The
# recursive `cmd_run(sub)` is replaced by a recorder, so nothing is dispatched and the child
# namespace the sweep WOULD have launched is available to assert on.

_CAP8_GRID = "conf/grids/frames_trained.yaml"  # budget_cap: 8, one suite


def _clean_git(monkeypatch):
    import pi_run.cli as cli
    from pi_run.manifest import GitInfo

    monkeypatch.setattr(
        cli, "git_info", lambda root: GitInfo(sha="deadbeef", dirty=False, dirty_files=())
    )


def _dispatch(tmp_path, monkeypatch, *flags, grid=_CAP8_GRID):
    """Run `cmd_run` on a sweep with `flags`, dispatching nothing. Returns (rc, launched)."""
    import pi_run.cli as cli

    _clean_git(monkeypatch)
    a = cli.build_parser().parse_args(["run", "--sweep", grid, "--root", str(tmp_path), *flags])
    launched: list = []
    real_cmd_run = cli.cmd_run

    def _recorder(sub):
        launched.append(sub)
        return 0

    monkeypatch.setattr(cli, "cmd_run", _recorder)  # the RECURSIVE call, not the one we invoke
    return real_cmd_run(a), launched


def test_a_conflicting_budget_cap_is_refused_before_anything_is_dispatched(
    tmp_path, monkeypatch, capsys
) -> None:
    """THE DEFECT. `--budget-cap 4` beside a cap-8 grid must be refused, not discarded."""
    rc, launched = _dispatch(tmp_path, monkeypatch, "--budget-cap", "4")
    err = capsys.readouterr().err

    assert launched == [], (
        "the sweep dispatched a suite while the operator's --budget-cap 4 was discarded; "
        f"it would have run at budget_cap={[getattr(s, 'budget_cap', None) for s in launched]}"
    )
    assert rc == 2, f"rc={rc}, stderr={err!r}"
    assert "budget-cap" in err, err
    assert "4" in err and "8" in err, err  # names what was typed AND what the grid declares


def test_a_budget_cap_that_agrees_with_the_grid_is_not_a_conflict(tmp_path, monkeypatch) -> None:
    """The refusal must not fire when the flag AGREES, or every legitimate sweep breaks."""
    rc, launched = _dispatch(tmp_path, monkeypatch, "--budget-cap", "8")
    assert rc == 0
    assert [s.budget_cap for s in launched] == [8]


def test_an_unpassed_budget_cap_is_not_a_conflict(tmp_path, monkeypatch) -> None:
    """`--budget-cap` default is None, so an unpassed flag is distinguishable from a typed one
    and must never be refused -- the same discipline as `test_no_flags_means_no_conflicts`."""
    rc, launched = _dispatch(tmp_path, monkeypatch)
    assert rc == 0
    assert [s.budget_cap for s in launched] == [8]


def test_a_conflicting_max_turns_is_still_refused_by_the_same_path(
    tmp_path, monkeypatch, capsys
) -> None:
    """A FIELD ALREADY COVERED, driven through the same surface, so a regression in either
    direction is caught: closing the class must not lose the six flags already checked."""
    rc, launched = _dispatch(tmp_path, monkeypatch, "--max-turns", "1")
    err = capsys.readouterr().err
    assert launched == []
    assert rc == 2, f"rc={rc}, stderr={err!r}"
    assert "max-turns" in err, err


# --------------------------------------------------------------------------- the drift guard
#
# The correspondence is STRUCTURAL: `grid_discarded_flags` diffs the operator's namespace
# against the one `grid_child_namespace` builds, so a field becomes checked at the moment it
# becomes assigned. That leaves exactly one way for the two to drift apart again -- an
# assignment made to the child namespace SOMEWHERE ELSE, where the diff cannot see it, which
# is what the sweep loop used to do for all seventeen fields. These walk the syntax tree
# rather than the prose, because the prose was right and the code was not: the checker's own
# docstring named `budget_cap` among the fields the assignment broke, for a day, while the
# code below it never inspected `budget_cap` at all.


def _sub_assignments(fn) -> set[str]:
    """Fields assigned onto the child `argparse.Namespace` inside `fn`, by AST."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    child = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in ("Namespace", "grid_child_namespace")
        for t in node.targets
        if isinstance(t, ast.Name)
    } | {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "grid_child_namespace"
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert child, f"no child namespace binding found in {fn.__name__}"
    return {
        t.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in child
    }


def test_the_sweep_loop_assigns_no_grid_field_outside_the_one_checked_function() -> None:
    """THE DRIFT GUARD. Every field the grid overrides must be assigned in
    `grid_child_namespace`, because that function's output is what the conflict check diffs.
    A `sub.<field> = grid.<field>` added to `cmd_run`'s loop instead is invisible to the diff
    and is a silent discard the moment it is written -- which is the defect this closes, for
    all seventeen fields at once.

    `cmd_run` may still assign the three fields that are NOT overrides, and each must be in
    `GRID_DIFF_EXEMPT` with its reason."""
    from pi_run import cli

    in_loop = _sub_assignments(cli.cmd_run)
    unexplained = in_loop - set(cli.GRID_DIFF_EXEMPT)
    assert unexplained == set(), (
        f"cmd_run assigns {sorted(unexplained)} onto the dispatched namespace outside "
        "grid_child_namespace, so grid_discarded_flags cannot see it. Move the assignment "
        "there, or add it to GRID_DIFF_EXEMPT with the reason it is not a discarded flag."
    )


def test_every_field_the_grid_overrides_is_a_field_the_check_can_reach() -> None:
    """The other direction. A field assigned in `grid_child_namespace` is only checked if it
    is a `pi run` dest -- `grid_discarded_flags` iterates the parser's own defaults, so a
    field that no flag can set is unreachable and unreachable is fine (nobody can type it).
    Anything that IS a flag must be either diffed or exempt, and this names which is which so
    the next reader does not have to re-derive it."""
    from pi_run import cli

    assigned = _sub_assignments(cli.grid_child_namespace)
    dests = set(cli.run_flag_defaults())

    typed_and_overridden = (assigned & dests) - set(cli.GRID_DIFF_EXEMPT)
    assert typed_and_overridden == {
        "suite",
        "seeds",
        "n",
        "task",
        "split",
        "task_offset",
        "max_turns",
        "k",
        "budget_cap",
        "pilot",
        "concurrency",
    }, sorted(typed_and_overridden)

    # Not `pi run` flags at all, so no operator value exists to discard: provenance, and
    # `exploratory`, which only `pi agg --exploratory` declares -- `pi run` has no such flag,
    # so the grid is its only source and there is nothing to contradict.
    assert (assigned - dests) == {
        "grid_sha256",
        "grid_name",
        "exploratory",
    }, sorted(assigned - dests)


def test_no_exemption_is_a_field_nothing_assigns_or_a_flag_nothing_declares() -> None:
    """A check that can never fire is its own defect, and so is an exemption for a field that
    does not exist -- both read as a clean pass. Every `GRID_DIFF_EXEMPT` entry must name a
    real `pi run` dest that something really does assign, and must carry its reason."""
    from pi_run import cli

    dests = set(cli.run_flag_defaults())
    assigned = _sub_assignments(cli.cmd_run) | _sub_assignments(cli.grid_child_namespace)
    for field, reason in cli.GRID_DIFF_EXEMPT.items():
        assert field in dests, f"{field} is exempted but is not a `pi run` flag"
        assert field in assigned, f"{field} is exempted but nothing assigns it"
        assert len(reason) > 40, f"{field} is exempted without a reason"


# ------------------------------------------------------- the rest of the class, once each
#
# `--split` and `--task-offset` were never named in any list; `--budget-cap` was named in the
# checker's docstring and never checked; `--pilot` and `--suite` are overrides too. None of
# these needed a new list entry -- they are covered because `grid_child_namespace` assigns
# them. One test each, so the coverage is visible rather than inferred from the diff.


def _conflicts(grid_path: str, *flags: str) -> tuple[str, ...]:
    from pi_run import grids
    from pi_run.cli import build_parser, grid_flag_conflicts

    grid = grids.load(grid_path)
    a = build_parser().parse_args(["run", "--sweep", grid_path, *flags])
    out: list[str] = []
    for suite_id in grid.suites:
        for m in grid_flag_conflicts(a, grid=grid, suite_id=suite_id):
            if m not in out:
                out.append(m)
    return tuple(out)


_SPLIT_TEST_GRID = "conf/grids/frames_trained.yaml"  # split: test, budget_cap: 8, one suite
_MULTI_SUITE_GRID = "conf/grids/tier1_confirmatory.yaml"  # 3 suites, k_by_suite, no split


def test_a_contradicting_split_is_refused_and_a_matching_one_is_not() -> None:
    """`--split` decides which rows are ELIGIBLE at all (report filters on `split = 'test'`),
    so a discarded `--split` is not a slower run, it is a table of the wrong population."""
    assert any("split" in c for c in _conflicts(_SPLIT_TEST_GRID, "--split", "train"))
    assert _conflicts(_SPLIT_TEST_GRID, "--split", "test") == ()


def test_a_contradicting_task_offset_is_refused() -> None:
    """`task_offset` is what makes grid slices disjoint; a discarded one silently re-runs the
    slice a pilot already burned."""
    assert any("task-offset" in c for c in _conflicts(_SPLIT_TEST_GRID, "--task-offset", "600"))


def test_pilot_is_refused_when_the_grid_is_not_a_pilot_grid() -> None:
    """`--pilot` marks runs pilot_flag=true (ids are burned). Discarding it makes a run that
    the operator believes is disposable count as a real one."""
    assert any("pilot" in c for c in _conflicts(_SPLIT_TEST_GRID, "--pilot"))


def test_a_suite_the_grid_does_not_run_alone_is_refused() -> None:
    """A grid runs ALL of `grid.suites`; `--suite` beside it neither filters nor survives.
    Naming the grid's only suite is redundant, not a conflict."""
    assert any("suite" in c for c in _conflicts(_MULTI_SUITE_GRID, "--suite", "musique"))
    assert _conflicts(_SPLIT_TEST_GRID, "--suite", "frames") == ()


def test_the_per_suite_k_is_what_is_compared_not_the_grid_wide_k() -> None:
    """A FIELD THAT WAS CHECKED AND STILL SILENTLY DISCARDED. The old check compared `--k`
    against `grid.k`; the loop assigned `grid.k_for(suite_id)`. tier1_confirmatory declares
    `k: 5` with `k_by_suite {strategyqa: 2, wiki2: 3}`, so `--k 5` was accepted grid-wide and
    strategyqa still ran at 2 -- measured on 4 grids on disk. This is why the diff is taken
    against the child namespace of EVERY suite, not against the grid's own fields."""
    from pi_run import grids

    grid = grids.load(_MULTI_SUITE_GRID)
    assert grid.k == 5 and grid.k_by_suite, "this test needs a grid with per-suite k"

    clashes = _conflicts(_MULTI_SUITE_GRID, "--k", "5")
    assert any("--k 5 but the grid makes it 2" in c for c in clashes), clashes
    assert any("--k 5 but the grid makes it 3" in c for c in clashes), clashes


def test_the_two_flags_whose_typed_value_argparse_cannot_distinguish() -> None:
    """WHAT THE PARSER CANNOT TELL US, recorded rather than assumed away.

    The refusal rests on comparing a dest against the parser's own default, which is why
    `--seeds/--n/--max-turns/--k/--budget-cap/--split` all default to None. Two flags cannot
    supply that sentinel:

      * `--task-offset` defaults to 0, so an explicit `--task-offset 0` is byte-identical to
        an unpassed one and cannot be refused. A non-zero one is caught.
      * `--pilot` is `store_true`, so only True is visible; `--pilot` cannot be typed as False.

    Both are blind in exactly the direction where the typed value already EQUALS the parser
    default -- so neither can produce a wrong refusal, and each can only miss one. That is a
    property of argparse, not of the check: giving `--task-offset` a None default would fix it
    and would also change `pi run`'s standalone behaviour, so it is a decision rather than a
    fix and is left as one. This test fails if someone changes those defaults, which is the
    point at which the blind spot could be closed."""
    from pi_run.cli import run_flag_defaults

    d = run_flag_defaults()
    assert d["task_offset"] == 0, "if this became None, --task-offset 0 becomes detectable"
    assert d["pilot"] is False, "store_true: only an explicit --pilot is visible"

    # ... and the six that CAN be told apart, so a default silently regaining a real value
    # (which is what made the flags undetectable in the first place) fails here.
    for flag in ("seeds", "n", "max_turns", "k", "budget_cap", "split", "suite", "concurrency"):
        assert d[flag] is None, f"--{flag} must default to None for an explicit one to be seen"
    assert d["task"] == [], "--task is action=append; its sentinel is the empty list"


def test_a_call_that_supplies_no_grid_and_no_values_is_refused_not_answered_with_clean() -> None:
    """A guard whose caller cannot tell "no conflicts" from "never compared anything" is the
    shape this whole file exists to remove, so the vacuous call raises instead."""
    import argparse

    import pytest

    from pi_run.cli import grid_flag_conflicts

    with pytest.raises(TypeError, match="pass grid="):
        grid_flag_conflicts(argparse.Namespace(budget_cap=4))


def test_every_grid_whose_per_suite_k_differs_refuses_the_grid_wide_k() -> None:
    """The per-suite discard, over WHATEVER grids are on disk rather than a stated count.

    Two different numbers were reported for this on 2026-09-18 and all three are real: NINE
    grids declare `k_by_suite` tracked at main, TEN in a working tree that also carries the
    untracked `frontier_trained_musique_cap24`, and on FOUR of them a per-suite value actually
    differs from the grid-wide `k`. The declaration count is the mechanism exposure (every one
    takes the `k_for` path); the four is where the old check and the loop's assignment could
    disagree and a wrong `k` could be recorded. The rest declare it redundantly, every entry
    equal to `k`. The first report gave the narrowest number in the widest number's sentence.

    Asserting either count would make this test a restatement that goes stale the next time a
    grid is added, so it asserts the PROPERTY over the grids that exist: wherever a per-suite k
    differs, typing the grid-wide k must be refused, and wherever it does not, typing it must
    not be. Both directions, because a refusal that fired on the redundant six would break
    every legitimate sweep on them."""
    import glob

    from pi_run import grids

    differing, redundant = [], []
    for path in sorted(glob.glob("conf/grids/*.yaml")):
        try:
            grid = grids.load(path)
        except Exception:  # a grid this checkout cannot resolve is not this test's subject
            continue
        if not grid.k_by_suite:
            continue
        clashes = _conflicts(path, "--k", str(grid.k))
        if any(grid.k_for(s) != grid.k for s in grid.suites):
            differing.append(grid.name)
            assert any("--k" in c for c in clashes), (
                f"{grid.name} resolves k per suite to "
                f"{ {s: grid.k_for(s) for s in grid.suites} } against a grid-wide k={grid.k}, "
                f"so --k {grid.k} is silently discarded for at least one suite: {clashes}"
            )
        else:
            redundant.append(grid.name)
            assert clashes == (), (
                f"{grid.name} declares k_by_suite but every entry equals k={grid.k}, so "
                f"--k {grid.k} agrees with what every suite will run and must not be refused: "
                f"{clashes}"
            )

    assert differing, "no grid on disk resolves k per suite; this test would be vacuous"
    assert redundant, "no grid declares k_by_suite redundantly; the second branch is untested"
