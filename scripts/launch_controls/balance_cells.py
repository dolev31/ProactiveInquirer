"""Is the balanced prompted cell single-valued on commit, cap and PIN?

A balancing sweep that quietly mixes commits or pins is worse than the imbalance it fixes: an
unbalanced cell is visible in a task count and a mixed one is not. Run from the repo root.
"""

import collections
import json
import os
import pathlib

RUNS = pathlib.Path(os.environ.get("PI_RUNS_ROOT", "runs"))
CV = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
SUITES = {"musique", "strategyqa", "wiki2"}

cells: collections.Counter = collections.Counter()
pins: dict[tuple, set] = collections.defaultdict(set)
status: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
grids: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
usd: collections.Counter = collections.Counter()

for p in RUNS.glob("*/manifest.json"):
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    if m.get("code_version") != CV or m.get("split") != "test" or m.get("budget_cap") != 8:
        continue
    if m.get("suite_id") not in SUITES or m.get("arm_id") != "inquirer_prompted" or m.get("dirty"):
        continue
    sp = p.parent / "status.json"
    js = json.loads(sp.read_text()) if sp.exists() else {}
    key = (m["suite_id"], m["seed"])
    status[key][js.get("status")] += 1
    if js.get("status") != "ok":
        continue
    cells[key] += 1
    grids[key][m.get("grid_name")] += 1
    # Sum each run's OWN status.json usd_billed. Never a log regex: `_emit` spans lines, so a
    # regex over the launcher's log read $0.000000 and a cap parsed that way was inert.
    usd[key] += float(js.get("usd_billed") or 0.0)
    iq = (m.get("pins") or {}).get("inquirer") or {}
    an = (m.get("pins") or {}).get("answerer") or {}
    pins[key].add(
        (
            iq.get("model_id"),
            (iq.get("adapter_sha") or "none")[:10],
            (iq.get("base_url_sha") or "")[:10],
            an.get("model_id"),
            (an.get("base_url_sha") or "")[:10],
            (m.get("grid_sha256") or "")[:12],
            m.get("budget_cap"),
            m.get("code_version")[:7],
        )
    )

print("BALANCED CELL: inquirer_prompted @3ae099d, cap 8, split test\n")
bad = [k for k in cells if len(pins[k]) != 1]
for k in sorted(cells):
    print(f"  {k[0]:11s} seed={k[1]}  ok={cells[k]:4d}  statuses={dict(status[k])}")
    print(f"      grid_names={dict(grids[k])}  distinct pin tuples={len(pins[k])}")
    for t in sorted(pins[k]):
        print(f"        {t}")
print(f"\ncells at 200 ok: {sum(1 for k in cells if cells[k] == 200)} of {len(cells)}")
print(
    "VERDICT:",
    "SINGLE-VALUED on commit, cap and pin in every cell"
    if not bad
    else f"MIXED in {bad} -- do not report these as one cell",
)
