#!/usr/bin/env python
"""Delete run directories that FAILED, so a resume actually re-runs them.

WHY THIS IS NEEDED AND WHY IT IS DANGEROUS TO OMIT. Resumption is by existence:
`sweep.py` — "a unit whose status.json exists returns status='resumed' without re-running" —
and `worker.py:855` deliberately carries the PRIOR outcome forward, so a resumed error stays
an error rather than reading as a fresh success. That is the right design for a crash: you
re-issue the identical command and keep everything that worked.

It is the wrong outcome for an ENVIRONMENT failure. A dropped network, an unset
PI_MODEL_INQUIRER, an expired key — these write `status: error` for every unit they touch,
and those units are then skipped by every future resume. The task list is silently consumed:
the sweep reports "resumed" forever and the data never arrives. Measured in this repository:
one unset environment variable produced 548 such runs in two minutes.

So before resuming a campaign that hit an environment failure, delete the failed directories.
This script does that, and NOTHING else:

  * only `status: error` directories, never `ok`;
  * only failures matching --error-contains, so a genuine model error is not swept away with
    an infrastructure one;
  * --dry-run by default, because deleting run directories is not reversible.

A run that failed for a REAL reason (a task the policy cannot do, a malformed graph) should
be left alone: re-running it burns tokens to reproduce the same error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument(
        "--error-contains",
        default="LLMConfigError",
        help="only delete errors whose message contains this. Use a specific infrastructure "
        "signature; the default is the unset-model-pin failure.",
    )
    ap.add_argument("--newer-than-min", type=float, default=0, help="0 = no age limit")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    root = Path(a.runs_root)
    now = time.time()
    hits: list[Path] = []
    reasons: Counter = Counter()
    kept_ok = kept_other = 0

    for d in sorted(root.iterdir()):
        st = d / "status.json"
        if not st.is_file():
            continue
        try:
            s = json.loads(st.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if s.get("status") != "error":
            kept_ok += 1
            continue
        err = str(s.get("error") or "")
        if a.error_contains and a.error_contains not in err:
            kept_other += 1
            reasons[err[:70]] += 1
            continue
        if a.newer_than_min and (now - (s.get("finished_at") or 0)) / 60 > a.newer_than_min:
            kept_other += 1
            continue
        hits.append(d)

    print(f"{len(hits)} error runs match {a.error_contains!r}")
    print(f"  left alone: {kept_ok} non-error, {kept_other} errors that do not match")
    for r, c in reasons.most_common(3):
        print(f"    kept ({c}): {r}")
    if not a.apply:
        print(
            "\nDRY RUN — nothing deleted. Re-run with --apply to remove them, then resume "
            "the campaign with the identical command."
        )
        return 0
    for d in hits:
        shutil.rmtree(d)
    print(f"\ndeleted {len(hits)} directories; a resume will now re-run those units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
