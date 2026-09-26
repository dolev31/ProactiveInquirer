"""`random_q` matches VOLUME, and the check is the delta against its reference arm.

`RandomQInquirer` takes `ask_counts[task_id]` -- the number of asks the reference arm made on THIS
task -- so its own `n_asks` must equal the reference's on every shared `(suite, task, seed)`. A mean
delta of zero over all rows is the arm doing its job. A PARTIAL match rate is not a weaker version
of that: it means the pairing is not what the name says, and the rate is then a property of which
runs finished rather than of the arm.

Every candidate reference arm at the same commit is reported, because a single match rate quoted
without its reference is unreadable -- and on the legacy killswitch cell only one reference shares
any keys at all, which is what makes that rate a measurement rather than a choice.

Usage, from the repo root:
    python scripts/launch_controls/random_q.py [RUN_ID_LIST | -]
`-` or no argument measures the live glob. `PI_RUNS_ROOT` overrides the runs root.
"""

import collections
import json
import os
import pathlib
import statistics
import sys

RUNS = pathlib.Path(os.environ.get("PI_RUNS_ROOT", "runs"))
NEW_CV = "107ef2221a55229ac7ba60a54ebfc9849971bde0"
REF_CV = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
LEGACY_GRID = "tier1_trained_killswitch"


def load(runs: pathlib.Path) -> list[tuple[dict, str | None, int | None]]:
    out = []
    for p in runs.glob("*/manifest.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        sp = p.parent / "status.json"
        st: dict = {}
        if sp.exists():
            try:
                st = json.loads(sp.read_text())
            except Exception:
                st = {}
        out.append((m, st.get("status"), st.get("n_asks")))
    return out


def refmap(rows, cv: str, arm: str) -> dict[tuple, int | None]:
    return {
        (m.get("suite_id"), m.get("task_id"), m.get("seed")): n_asks
        for m, status, n_asks in rows
        if m.get("arm_id") == arm
        and m.get("code_version") == cv
        and m.get("split") == "test"
        and status == "ok"
    }


def report(rows, pin, label: str, cv: str, grid: str | None, refs) -> None:
    per: dict[str, list[tuple]] = collections.defaultdict(list)
    nopair: collections.Counter = collections.Counter()
    for m, status, n_asks in rows:
        if m.get("arm_id") != "random_q" or m.get("code_version") != cv:
            continue
        if grid is not None and m.get("grid_name") != grid:
            continue
        if status != "ok" or m.get("split") != "test":
            continue
        if pin is not None and m.get("run_id") not in pin:
            continue
        key = (m.get("suite_id"), m.get("task_id"), m.get("seed"))
        hit = next(((name, rm[key]) for name, rm in refs if key in rm), None)
        if hit is None:
            nopair[key[0]] += 1
            continue
        per[key[0]].append((n_asks, hit[1], hit[0]))
    print(f"\n-- {label}")
    for suite in sorted(per):
        v = per[suite]
        deltas = [(a or 0) - (b or 0) for a, b, _ in v]
        matched = sum(1 for d in deltas if d == 0)
        names = collections.Counter(name for _, _, name in v)
        print(
            f"   {suite:11s} rows={len(v):4d} matched={matched}/{len(v)} "
            f"({100.0 * matched / len(v):.1f}%) "
            f"mean_delta={statistics.fmean(deltas):+.6f} refs={dict(names)}"
        )
    if nopair:
        print(f"   no paired reference: {dict(nopair)}")
    if not per:
        print("   (no rows)")


def main() -> int:
    pin = None
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        pin = {x.strip() for x in pathlib.Path(sys.argv[1]).read_text().split() if x.strip()}
    rows = load(RUNS)
    print(f"pin = {sys.argv[1] if pin else 'NONE (live glob)'}")
    refs_new = [
        ("prompted@3ae099d0", refmap(rows, REF_CV, "inquirer_prompted")),
        ("prompted@107ef222", refmap(rows, NEW_CV, "inquirer_prompted")),
    ]
    report(rows, pin, "random_q @107ef222 (grid_name EMPTY)", NEW_CV, None, refs_new)
    legacy_cvs = sorted(
        {
            m.get("code_version")
            for m, _, _ in rows
            if m.get("arm_id") == "random_q" and m.get("grid_name") == LEGACY_GRID
        }
    )
    for cv in legacy_cvs:
        refs = [
            (f"{arm}@{cv[:8]}", refmap(rows, cv, arm))
            for arm in ("inquirer_trained", "inquirer_prompted", "drafter_only", "checklist")
        ] + refs_new
        report(rows, pin, f"LEGACY random_q @{cv[:8]} grid={LEGACY_GRID}", cv, LEGACY_GRID, refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
