"""The length-exploitation kill rule must be shown to FIRE, inside the collected suite.

`scripts/length_audit/test_kill_rule_can_fire.py` proves the rule is non-vacuous: it fires on a
planted verdict, on both sides of the boundary, on a perturbed real verdict, and through the real
file-based CLI. But that file is a SCRIPT with a `main()`, and `pyproject.toml` sets
`testpaths = ["tests"]`, so `pytest -q` never collected it despite the `test_` prefix in its name.

A non-vacuity check the suite does not run is one refactor away from silently becoming vacuous
again -- and the 0-KILL headline (0 among 112 EVALUABLE verdicts, not 216: 104 predate
`acc_by_len_sign` and cannot show either outcome) rests entirely on that rule being able to fire.
So this wrapper puts it inside the gate. It shells out rather than importing, so the script stays
usable standalone and there is exactly one copy of the assertions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "length_audit" / "test_kill_rule_can_fire.py"
)


def test_the_kill_rule_can_actually_fire() -> None:
    assert SCRIPT.is_file(), f"missing non-vacuity script: {SCRIPT}"
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"kill-rule non-vacuity check FAILED:\n{out}"
    # If the rule stops firing, the 0-KILL audit result is withdrawn, not merely stale.
    assert "all checks passed" in out.lower(), out
