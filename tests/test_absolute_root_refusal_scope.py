"""The absolute-root refusal must fire on a launch and not on a plan.

The refusal exists for a measured reason: a launcher validated an absolute runs root, sourced
`.env` for the gateway key, and the file's `./runs` silently replaced the approved value, so twelve
shards wrote 117 run directories into a pinned worktree where no reporting path looks. It was
added to `scripts/run_tau2_forks.py` with no test of its own, and its scope was wrong in a way no
test could report: it ran BEFORE the `--dry-run` return, so it refused plans too.

`--runs-root` defaults to the relative `"runs"`, so every caller that omits it -- which is every
caller that only wants to see a plan, including this repository's own launch preflight and 15 tests
in `test_run_tau2_forks.py` -- got exit 4 instead of a plan. A dry run writes nothing, so a
caller-relative root cannot mislocate anything, and refusing there buys no safety at all.

Both halves are pinned here, because a fix that only made the failures stop could have deleted the
guard outright and nothing would have said so:

    - a real launch with a relative runs root still refuses with exit 4;
    - a plan with the same relative root returns 0 and prints the plan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_tau2_forks as launcher  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "run_manifests"
FORKS = (
    "fork_tau2_retail_test_inquirer_prompted_s0.json",
    "fork_tau2_retail_test_self_ask_s0.json",
    "fork_tau2_retail_test_inquirer_prompted_s2.json",
)


@pytest.fixture
def runs_root(tmp_path):
    runs = tmp_path / "runs"
    for name in FORKS + ("nonfork_tau2_retail.json",):
        d = json.loads((FIXTURES / name).read_text())
        rd = runs / d["run_id"]
        rd.mkdir(parents=True)
        (rd / "manifest.json").write_text(json.dumps(d))
        (rd / "status.json").write_text(json.dumps({"status": "ok"}))
    return runs


def _argv(runs_root, *extra):
    return [
        "--suite",
        "tau2_retail",
        "--split",
        "test",
        "--forkpoints-from-runs",
        str(runs_root),
        "--arm",
        "inquirer_prompted",
        "--arm",
        "self_ask",
        "--seeds",
        "0",
        *extra,
    ]


def test_a_plan_with_the_default_relative_runs_root_is_not_refused(runs_root, monkeypatch, capsys):
    """The regression. Omitting --runs-root means the relative default, and a plan must still print."""

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("--dry-run must not launch a unit")

    monkeypatch.setattr(launcher, "_run_unit", _boom)
    rc = launcher.main(_argv(runs_root, "--dry-run"))
    err = capsys.readouterr().err
    assert rc == 0, f"a dry run was refused for a root it never writes to; stderr: {err}"
    assert "is not absolute" not in err


def test_a_real_launch_with_a_relative_runs_root_still_refuses(runs_root, monkeypatch, capsys):
    """Non-vacuity, and the property the guard was added for: a LAUNCH must still refuse."""

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("a relative runs root must be refused before any unit launches")

    monkeypatch.setattr(launcher, "_run_unit", _boom)
    rc = launcher.main(_argv(runs_root, "--runs-root", "runs"))
    err = capsys.readouterr().err
    assert rc == 4, "a relative runs root was accepted for a real launch"
    assert "is not absolute" in err
    assert "--runs-root" in err


def test_a_real_launch_with_both_roots_absolute_passes_the_root_check(
    runs_root, monkeypatch, tmp_path
):
    """The guard must not refuse the case it exists to permit, or it would block every launch.

    BOTH roots are set absolute here. `cache_root` has no flag -- it comes from `$PI_CACHE_ROOT`
    and otherwise defaults to the relative `"cache"` -- so a test that set only `--runs-root` would
    still be refused, and asserting that it was NOT refused would have quietly deleted the
    cache-root half of the guard. A relative cache root is its own measured defect: `./cache`
    resolves against the reader's cwd, and a probe once reported a 16% hit rate against a cache
    holding 97.1% for the same reason.

    `_run_unit` is replaced, so this reaches the launch path and stops there.
    """
    launched: list[object] = []
    monkeypatch.setenv("PI_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(launcher, "_run_unit", lambda spec: launched.append(spec))
    rc = launcher.main(_argv(runs_root, "--runs-root", str(tmp_path / "out")))
    assert rc != 4, "two absolute roots were refused as relative"
    assert launched, "nothing reached the launch path, so the root check was not exercised"


def test_a_real_launch_with_a_relative_cache_root_still_refuses(
    runs_root, monkeypatch, tmp_path, capsys
):
    """The other half of the guard: an absolute runs root does not excuse a relative cache root."""

    def _boom(spec):  # pragma: no cover - reaching it is the failure
        raise AssertionError("a relative cache root must be refused before any unit launches")

    monkeypatch.setenv("PI_CACHE_ROOT", "cache")
    monkeypatch.setattr(launcher, "_run_unit", _boom)
    rc = launcher.main(_argv(runs_root, "--runs-root", str(tmp_path / "out")))
    err = capsys.readouterr().err
    assert rc == 4, "a relative cache root was accepted because the runs root was absolute"
    assert "cache_root" in err, f"the refusal did not name the offending root; stderr: {err}"
