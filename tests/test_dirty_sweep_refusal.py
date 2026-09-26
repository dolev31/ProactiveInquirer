"""`pi run --sweep` must refuse a dirty tree rather than banner past it.

Incident, 2026-09-17: a 400-unit evaluation sweep was launched while three unrelated tracked
files sat uncommitted in this checkout. Every one of the 400 runs came out dev- stamped
(`RunManifest.dirty` follows `GitInfo.dirty` straight into the run_id) -- training-only,
inadmissible as an eval arm -- and the loss was invisible until someone read the manifests
after the fact. `pi run` already prints a DIRTY banner for this case, deep in the non-sweep
tail, once per suite, after real per-unit planning has already happened; the operator missed
it. A banner is not a control for a CAMPAIGN.

This mirrors `test_sweep_flag_conflict.py`'s two-layer shape: `dirty_sweep_refusal` is a pure
helper, unit-tested directly on a `GitInfo`, and `cmd_run` is proven to call it (and to refuse
early, before `grid_flag_conflicts` or any suite is dispatched) without actually launching
anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _gi(*, dirty: bool, files=()):
    from pi_run.manifest import GitInfo

    return GitInfo(sha="deadbeef", dirty=dirty, dirty_files=tuple(files))


# --------------------------------------------------------------------------- pure helper


def test_a_clean_tree_returns_none() -> None:
    from pi_run.cli import dirty_sweep_refusal

    assert dirty_sweep_refusal(_gi(dirty=False), allow_dirty=False, root=Path("/r")) is None


def test_a_dirty_tree_without_allow_dirty_refuses_and_names_the_files() -> None:
    from pi_run.cli import dirty_sweep_refusal

    msg = dirty_sweep_refusal(
        _gi(dirty=True, files=("src/a.py", "docs/b.md")), allow_dirty=False, root=Path("/r")
    )
    assert msg is not None
    assert "src/a.py" in msg
    assert "docs/b.md" in msg
    assert "dev-" in msg
    assert "training-only" in msg
    assert "--allow-dirty" in msg
    assert "worktree add --detach" in msg  # the pinned-clean-worktree remediation, named


def test_allow_dirty_restores_todays_prior_behaviour() -> None:
    from pi_run.cli import dirty_sweep_refusal

    assert (
        dirty_sweep_refusal(_gi(dirty=True, files=("src/a.py",)), allow_dirty=True, root=Path("/r"))
        is None
    )


def test_a_dirty_tree_with_no_file_list_says_so_rather_than_a_fabricated_count() -> None:
    """`GitInfo.dirty=True, dirty_files=()` happens when git itself failed (see GitInfo's
    docstring); the message must not print "0 files" beside a tree it is refusing as dirty."""
    from pi_run.cli import dirty_sweep_refusal

    msg = dirty_sweep_refusal(_gi(dirty=True, files=()), allow_dirty=False, root=Path("/r"))
    assert msg is not None
    assert "0 dirty file" not in msg
    assert "could not enumerate" in msg


# --------------------------------------------------------------------------- argparse wiring


def test_allow_dirty_defaults_to_false_and_is_settable() -> None:
    from pi_run.cli import build_parser

    a = build_parser().parse_args(["run", "--sweep", "conf/grids/tier0_canary.yaml"])
    assert a.allow_dirty is False

    a2 = build_parser().parse_args(
        ["run", "--sweep", "conf/grids/tier0_canary.yaml", "--allow-dirty"]
    )
    assert a2.allow_dirty is True


# --------------------------------------------------------------------------- cmd_run, end to end
#
# Each of these drives the real `cmd_run` against the real `conf/grids/tier0_canary.yaml` grid,
# with `git_info` monkeypatched so no subprocess runs and no real repo state matters, and with
# the very next call `cmd_run` would make after the dirty check -- `grid_flag_conflicts` for the
# sweep path, `flat_run_refusal` for the non-sweep path -- replaced by a sentinel. Reaching the
# sentinel proves control flow got PAST the dirty check; a bare `2` return with nothing launched
# proves it did not, mirroring `test_hpc_launch.py`'s early-return-detection pattern.


class _ReachedDownstream(Exception):
    pass


def _sentinel(*_a, **_kw):
    raise _ReachedDownstream


def _sweep_args(tmp_path, *, allow_dirty=False):
    from pi_run.cli import build_parser

    argv = ["run", "--sweep", "conf/grids/tier0_canary.yaml", "--root", str(tmp_path)]
    if allow_dirty:
        argv.append("--allow-dirty")
    return build_parser().parse_args(argv)


def test_sweep_on_a_dirty_tree_refuses_before_grid_flag_conflicts_and_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    import pi_run.cli as cli

    monkeypatch.setattr(cli, "git_info", lambda root: _gi(dirty=True, files=("x.py",)))
    monkeypatch.setattr(cli, "grid_flag_conflicts", _sentinel)

    rc = cli.cmd_run(_sweep_args(tmp_path))

    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSING --sweep on a dirty tree" in err
    assert "x.py" in err
    assert list(tmp_path.iterdir()) == [], "nothing must be written to --root on this path"


def test_sweep_on_a_dirty_tree_with_allow_dirty_reaches_grid_flag_conflicts(
    tmp_path, monkeypatch
) -> None:
    import pi_run.cli as cli

    monkeypatch.setattr(cli, "git_info", lambda root: _gi(dirty=True, files=("x.py",)))
    monkeypatch.setattr(cli, "grid_flag_conflicts", _sentinel)

    with pytest.raises(_ReachedDownstream):
        cli.cmd_run(_sweep_args(tmp_path, allow_dirty=True))


def test_sweep_on_a_clean_tree_reaches_grid_flag_conflicts_unaffected(
    tmp_path, monkeypatch
) -> None:
    import pi_run.cli as cli

    monkeypatch.setattr(cli, "git_info", lambda root: _gi(dirty=False))
    monkeypatch.setattr(cli, "grid_flag_conflicts", _sentinel)

    with pytest.raises(_ReachedDownstream):
        cli.cmd_run(_sweep_args(tmp_path))


def test_the_non_sweep_path_is_unaffected_by_a_dirty_tree(tmp_path, monkeypatch) -> None:
    """No `--sweep` at all: `dirty_sweep_refusal` must never be consulted, dirty tree or not --
    this is a campaign-launch refusal, not a rule about the tree. Proven by monkeypatching the
    next call `cmd_run` makes once it falls through the (absent) sweep branch."""
    import pi_run.cli as cli
    from pi_run.cli import build_parser

    monkeypatch.setattr(cli, "git_info", lambda root: _gi(dirty=True, files=("x.py",)))
    monkeypatch.setattr(cli, "flat_run_refusal", _sentinel)

    a = build_parser().parse_args(
        ["run", "--suite", "musique", "--root", str(tmp_path), "--n", "1"]
    )
    assert getattr(a, "sweep", None) is None

    with pytest.raises(_ReachedDownstream):
        cli.cmd_run(a)
