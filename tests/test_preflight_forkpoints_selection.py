"""Preflight cannot get ONE unit from an explicit-selection forkpoints file via `--n`.

THE DEFECT, MEASURED 2026-09-19. `phase5_launch.sh`'s preflight step sets `PINQ_PROBE_N=1`,
which `phase5_shard.sh` forwards as `--n 1` to `run_tau2_forks.py`, together with the SAME
fixed `conf/forks/tau2_airline_test.recovered34.json` the real campaign uses for every arm.
That file is a dict -- an EXPLICIT SELECTION, not a pool -- and `run_tau2_forks.py` refuses to
drop any point of an explicit selection (see its own `ap.error` at "--forkpoints names an
explicit selection..."). `--n 1` against 34 explicit points drops 33 of them, so the refusal
fires on EVERY arm, every time: preflight can never produce a manifest this way, on any
campaign that uses an explicit-selection forkpoints file.

Reproduced live during an L9 relaunch attempt (2026-09-19): after fixing a separate
`PI_CACHE_ROOT` environment defect, the very next preflight attempt got exactly this far and
died with rc=2 -- argparse's own error, before any unit was constructed. No manifest, no
ledger, no run directory: caught before billing, not after.

THE FIX preflight needs is not `--n`: it is a SEPARATE, scratch explicit-selection file
holding exactly one of the recorded points, passed with NO `--n`/`--min-k`/`--max-k` at all,
so the guard's own "no k flags, no --n -> use every point" branch fires and "every point" is
one. This file pins that: the current wiring's failure mode (regression-proofing the guard,
not weakening it), and the shape a fix must take.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_tau2_forks.py"
REAL_FORKPOINTS = ROOT / "conf" / "forks" / "tau2_airline_test.recovered34.json"


def _dry_run(forkpoints: Path, extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--suite",
            "tau2_airline",
            "--forkpoints",
            str(forkpoints),
            "--arm",
            "inquirer_prompted",
            "--seeds",
            "0",
            "--prompt-variant",
            "tau2_base",
            "--dry-run",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_the_real_forkpoints_file_exists_and_is_an_explicit_selection_of_34():
    """Without this, the tests below could pass vacuously against a missing/empty file."""
    assert REAL_FORKPOINTS.is_file(), f"{REAL_FORKPOINTS} missing; the tests below are vacuous"
    raw = json.loads(REAL_FORKPOINTS.read_text())
    assert isinstance(raw, dict), "the file changed shape; re-check the preflight fix's premise"
    assert len(raw["fork_points"]) == 34, "the file changed size; re-check the preflight fix"


def test_preflight_probe_n_1_against_the_real_file_is_refused_not_truncated():
    """THE REGRESSION preflight hits on every arm. `--n 1` must never silently launch 1 of 34
    named points -- it must refuse, because that is the explicit-selection guard doing its job.
    This pins the guard's behavior so the only available fix is to the CALLER (phase5_launch.sh
    / phase5_shard.sh's preflight construction), never to this guard.
    """
    r = _dry_run(REAL_FORKPOINTS, ["--n", "1"])
    assert r.returncode != 0, (
        "if this ever returns 0, the explicit-selection guard has been weakened -- CONTRIBUTING.md "
        f"rule 4 forbids that. stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    assert "explicit selection" in (r.stdout + r.stderr).lower(), (
        f"refused, but not for the reason under test:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


def test_a_scratch_one_point_explicit_selection_launches_without_any_n_flag(tmp_path):
    """THE FIX preflight needs: a real subset file of the same dict shape, holding exactly one
    of the 34 recorded points, with no `--n`/`--min-k`/`--max-k` -- so the guard's own default
    branch ("no k flags, no --n -> use every point") fires, and "every point" is exactly one.
    """
    raw = json.loads(REAL_FORKPOINTS.read_text())
    one = dict(raw)
    one["fork_points"] = raw["fork_points"][:1]
    one["n_fork_points"] = 1
    scratch = tmp_path / "preflight_one.json"
    scratch.write_text(json.dumps(one))

    r = _dry_run(scratch, [])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "would launch 1 units" in r.stdout, (
        f"expected exactly one unit, got:\n{r.stdout}\nstderr:\n{r.stderr}"
    )


def test_the_scratch_file_with_two_points_and_no_n_flag_launches_exactly_two(tmp_path):
    """NON-VACUITY. Proves the scratch file's `fork_points` length -- not some other default --
    is what decides the count, by varying it and checking the count moves with it."""
    raw = json.loads(REAL_FORKPOINTS.read_text())
    two = dict(raw)
    two["fork_points"] = raw["fork_points"][:2]
    two["n_fork_points"] = 2
    scratch = tmp_path / "preflight_two.json"
    scratch.write_text(json.dumps(two))

    r = _dry_run(scratch, [])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "would launch 2 units" in r.stdout, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
