"""`parallel_replay` determinism, measured WITHOUT the scorer, over a PINNED run-id list.

Identical question strings must retrieve identical evidence uid sets. Every metric that is a
function of the evidence set then has a delta of exactly zero, and a non-zero here indicts the
harness as impure rather than any metric definition.

THE PIN IS NOT A CONVENIENCE. The campaign that writes these runs was still running when they were
first reported, so a live glob returns a growing n: 527 paired rows at 11:12 and 545 at 11:22 on
2026-09-18. A count over a live glob is a timestamp, not a quantity. These arms also carry an EMPTY
`grid_name` because their grid file was lost, so a run-id list is the only selector available.

Usage, from the repo root:
    python scripts/launch_controls/parallel_replay.py [RUN_ID_LIST]
Omit the list to measure the live glob deliberately. `PI_RUNS_ROOT` overrides the runs root.
"""

import collections
import json
import os
import pathlib
import sys

RUNS = pathlib.Path(os.environ.get("PI_RUNS_ROOT", "runs"))
CV = "107ef2221a55229ac7ba60a54ebfc9849971bde0"
SRC_CV = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"


def questions(d: pathlib.Path) -> list[str]:
    rows = (
        json.loads(line) for line in (d / "turns.jsonl").read_text().splitlines() if line.strip()
    )
    return [r["question"] for r in rows if r.get("action_kind") == "ask" and r.get("question")]


def evidence_uids(d: pathlib.Path) -> list[str]:
    f = d / "evidence.jsonl"
    if not f.exists():
        return []
    return sorted(
        {json.loads(line).get("uid") for line in f.read_text().splitlines() if line.strip()}
    )


def main() -> int:
    pin = None
    if len(sys.argv) > 1:
        pin = {x.strip() for x in pathlib.Path(sys.argv[1]).read_text().split() if x.strip()}

    src: dict[tuple, pathlib.Path] = {}
    for p in RUNS.glob("*/manifest.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if (
            m.get("arm_id") == "inquirer_prompted"
            and m.get("code_version") == SRC_CV
            and m.get("split") == "test"
            and m.get("budget_cap") == 8
            and not m.get("dirty")
        ):
            src[(m.get("suite_id"), m.get("task_id"), m.get("seed"))] = p.parent

    res: collections.Counter = collections.Counter()
    mismatch = []
    for p in RUNS.glob("*/manifest.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if m.get("arm_id") != "parallel_replay" or m.get("code_version") != CV:
            continue
        if pin is not None and m.get("run_id") not in pin:
            continue
        st = p.parent / "status.json"
        if not st.exists() or json.loads(st.read_text()).get("status") != "ok":
            continue
        key = (m.get("suite_id"), m.get("task_id"), m.get("seed"))
        if key not in src:
            res[(key[0], "no paired source")] += 1
            continue
        a, b = p.parent, src[key]
        q_same = questions(a) == questions(b)
        e_same = evidence_uids(a) == evidence_uids(b)
        res[(key[0], f"questions={q_same} evidence={e_same}")] += 1
        if not (q_same and e_same):
            mismatch.append((m.get("run_id"), key))

    label = sys.argv[1] if pin else "NONE (live glob)"
    print(f"pin = {label}   pinned ids = {len(pin) if pin else '-'}")
    total = 0
    for key in sorted(res):
        print(f"  {key[0]:11s} {key[1]:34s} n={res[key]}")
        total += res[key]
    print(f"  TOTAL paired rows = {total}")
    print(f"MISMATCHES (harness impurity would appear here): {len(mismatch)}")
    for r in mismatch[:5]:
        print("   ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
