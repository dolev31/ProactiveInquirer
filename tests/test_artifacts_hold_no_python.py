"""The ruff exclusion of `artifacts/` rests on a measurement, so enforce the measurement.

`pyproject.toml` excludes `artifacts` from ruff because formatting rewrites code fences inside
results documents, and it justifies that exclusion by recording that no `.py` file lives there, so
nothing real loses linting. That premise is not self-maintaining. On 2026-09-18 a scanner was
committed to `artifacts/layer4_uncovered_suites/` and carried fourteen ruff errors that the gate
could not see, because the gate is configured not to look. The canonical `ruff check .` passed
throughout.

An exclusion whose safety argument is a fact about the tree needs that fact asserted, otherwise the
argument decays without any check failing. Results belong under `artifacts/`; the tooling that
produces them belongs somewhere ruff reads.
"""

from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise AssertionError("no pyproject.toml above this test")


def test_ruff_excludes_artifacts_only_because_it_holds_no_python() -> None:
    root = _repo_root()
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        return

    # The hazard EXISTS ONLY WHILE THE EXCLUSION DOES. If ruff is not configured to skip this
    # tree then it lints whatever python lives here and nothing is invisible, so there is nothing
    # to guard and this test must not fail. An earlier version asserted the exclusion was present,
    # which failed in exactly the configuration that is safe: measured 2026-09-18 in a clean
    # worktree, where the setting is not committed, this test was the lead failure of fifteen.
    # A guard that refuses a safe configuration is a guard that gets disabled.
    excluded = 'extend-exclude = ["artifacts"]' in (root / "pyproject.toml").read_text()
    if not excluded:
        return

    stray = sorted(p.relative_to(root).as_posix() for p in artifacts.rglob("*.py"))
    assert not stray, (
        "python under artifacts/ is invisible to `ruff check .`, which is configured to skip that "
        "tree on the recorded ground that no python lives there. Move these to scripts/ or src/ "
        f"where ruff reads them, or amend the justification in pyproject.toml: {stray}"
    )


def test_the_guard_can_actually_fail(tmp_path: Path) -> None:
    """Non-vacuity: the same glob must find a planted file, or the assertion above proves nothing."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "planted.py").write_text("x = 1\n")
    assert [p.name for p in tmp_path.rglob("*.py")] == ["planted.py"]
