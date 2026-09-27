"""The length-matched ladder's memoizing cache must be shown correct, inside the collected suite.

`scripts/length_matched/test_memoizing_cache.py` proves `score_checkpoint.py`'s cache (a) agrees
byte-for-byte with the uncached `pair_accuracy` path, (b) records an exact hit count rather than
merely a nonzero one, and (c) really short-circuits recomputation instead of coincidentally
agreeing with it. That file is a SCRIPT with a `main()`, and `pyproject.toml` sets
`testpaths = ["tests"]`, so `pytest -q` never collects it despite the `test_` prefix in its name.

This correctness proof is exactly the kind a refactor could silently break (the same script's
first version DID break it -- an `eo._score` lookup made inside the wrapper, at call time,
resolved to the wrapper itself once installed, and recursed until `RecursionError`; this test is
what caught that, per CONTRIBUTING.md rule 2, before any GPU time was spent on it). So this wrapper puts
it inside the gate. It shells out rather than importing, so the script stays usable standalone and
there is exactly one copy of the assertions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "length_matched" / "test_memoizing_cache.py"
)


def test_the_memoizing_cache_is_correct_and_actually_hits() -> None:
    assert SCRIPT.is_file(), f"missing non-vacuity script: {SCRIPT}"
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"memoizing-cache correctness check FAILED:\n{out}"
    assert "property 1 (cached == uncached baseline): ok" in out.lower(), out
    assert "property 2 (exact hit count): ok" in out.lower(), out
    assert "property 3" in out.lower() and "ok" in out.lower(), out
