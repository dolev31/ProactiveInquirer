"""Lane L3: independent completeness check of a reanswer rows file against the planned requests,
rebuilt from the run lists and turns.jsonl WITHOUT importing reanswer.py.

Usage (repo root): .venv/bin/python -m scripts.answer_loss.completeness reanswer_rows.full.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path.cwd()
OUT = REPO / "artifacts/answer_loss_20260923"
rows_path = OUT / sys.argv[1]
SUITES = ("musique", "strategyqa", "wiki2")
REC = {"s1": "qwen3-8b-dpo-stacked-notdone-both-s1", "s2": "qwen3-8b-dpo-stacked-notdone-both-s2"}


def ids(p):
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
ok = {(r["condition"], r["sid"]) for r in rows if r.get("ok")}
err = [(r["condition"], r["sid"]) for r in rows if not r.get("ok")]
dup = [k for k, n in Counter((r["condition"], r["sid"]) for r in rows).items() if n > 1]
cb_tasks = {
    (r["suite"], r["task"]) for r in rows if r["condition"] == "closed_book" and r.get("ok")
}
print(f"rows={len(rows)} ok={len(ok)} errors={len(err)} duplicate (condition,sid)={len(dup)}")
tot_states = set()
for suite in SUITES:
    runs = {}
    for arm, lst in [
        ("s1", REPO / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.{REC['s1']}.{suite}.txt"),
        ("s2", REPO / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.{REC['s2']}.{suite}.txt"),
        ("cmp", REPO / f"artifacts/completed_cohort_20260922/cohort/run_ids.prompted.{suite}.txt"),
    ]:
        for rid in ids(lst):
            m = json.loads((REPO / "runs" / rid / "manifest.json").read_text())
            n = sum(
                1
                for x in (REPO / "runs" / rid / "turns.jsonl").read_text().splitlines()
                if x.strip()
            )
            runs[rid] = (arm, m["task_id"], int(m["seed"]), n)
    cmp_by = {(t, s): rid for rid, (a, t, s, n) in runs.items() if a == "cmp"}
    tasks = sorted({t for _, t, _, _ in runs.values()})
    own_states = {f"{rid}@final" for rid in runs}
    pref_states, pairs = set(), []
    for rid, (a, t, s, n) in runs.items():
        if a == "cmp":
            continue
        c = cmp_by[(t, s)]
        k = min(n, runs[c][3])
        sr = f"{rid}@final" if k == n else f"{rid}@k{k}"
        sc = f"{c}@final" if k == runs[c][3] else f"{c}@k{k}"
        pairs.append((f"{rid}@final", f"{c}@final", sr, sc))
        pref_states |= {sr, sc}
    tot_states |= own_states | pref_states
    for cond in ("control", "evidence_only"):
        own_pairs = sum(1 for p in pairs if (cond, p[0]) in ok and (cond, p[1]) in ok)
        pre_pairs = sum(1 for p in pairs if (cond, p[2]) in ok and (cond, p[3]) in ok)
        print(
            f"{suite:10s} {cond:13s} own: states {sum((cond, s) in ok for s in own_states)}/{len(own_states)} "
            f"pairs {own_pairs}/{len(pairs)} | prefix: states {sum((cond, s) in ok for s in pref_states)}/{len(pref_states)} "
            f"pairs {pre_pairs}/{len(pairs)}"
        )
    print(
        f"{suite:10s} closed_book   tasks {sum((suite, t) in cb_tasks for t in tasks)}/{len(tasks)}"
    )
print(
    f"distinct planned states {len(tot_states)}; planned requests = 2 x {len(tot_states)} + 600 = {2 * len(tot_states) + 600}"
)
