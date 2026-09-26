"""Compare per-run turn/evidence/call row counts in an isolated parquet against the run
directories it was compacted from, for an explicit run-id list.

WHY. `pi compact` has silently dropped turns/evidence/calls rows for 21,001 runs whose
`turns.jsonl` was intact on disk; the parquet then reads `n_turns 0 / coverage 0` and nothing
errors. `pi compact` now reports `n_runs_with_turns_on_disk_but_none_compacted`, but that field
is presence-only: it cannot see a run whose 14 turns became 3. This compares COUNTS.

NON-VACUITY. `--force-disagreement N` perturbs the on-disk count of the first N run ids before
comparing, so a clean run can be told apart from a check that cannot fail. A comparison that
has never been observed reporting a disagreement is not evidence of agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import duckdb


def disk_counts(runs_root: str, rid: str) -> dict[str, int]:
    out = {}
    for name, fn in (
        ("turns", "turns.jsonl"),
        ("evidence", "evidence.jsonl"),
        ("calls", "calls.jsonl"),
    ):
        path = os.path.join(runs_root, rid, fn)
        n = 0
        try:
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except OSError:
            n = 0
        out[name] = n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--ids", action="append", required=True)
    ap.add_argument(
        "--force-disagreement",
        type=int,
        default=0,
        help="perturb the DISK count of the first N ids to prove the check fires",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args()

    ids: list[str] = []
    for p in args.ids:
        ids.extend(x.strip() for x in open(p) if x.strip() and not x.startswith("#"))
    ids = sorted(set(ids))
    if not ids:
        print("REFUSING: empty id list", file=sys.stderr)
        return 2

    con = duckdb.connect()
    pq = os.path.abspath(args.parquet)
    tbl = {}
    for name in ("turns", "evidence", "calls"):
        rows = con.execute(
            f"SELECT run_id, COUNT(*) c FROM read_parquet('{pq}/{name}.parquet') GROUP BY run_id"
        ).fetchall()
        tbl[name] = {r[0]: int(r[1]) for r in rows}

    disagree = []
    checked = 0
    totals = {"turns": [0, 0], "evidence": [0, 0], "calls": [0, 0]}
    for i, rid in enumerate(ids):
        d = disk_counts(args.runs_root, rid)
        if i < args.force_disagreement:
            d["turns"] += 1  # deliberate perturbation
        checked += 1
        for name in ("turns", "evidence", "calls"):
            p = tbl[name].get(rid, 0)
            totals[name][0] += d[name]
            totals[name][1] += p
            if p != d[name]:
                disagree.append({"run_id": rid, "table": name, "disk": d[name], "parquet": p})

    report = {
        "n_ids": len(ids),
        "n_checked": checked,
        "forced_disagreements": args.force_disagreement,
        "totals_disk_vs_parquet": {k: {"disk": v[0], "parquet": v[1]} for k, v in totals.items()},
        "n_disagreements": len(disagree),
        "disagreements": disagree[:20],
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0 if len(disagree) == args.force_disagreement else 1


if __name__ == "__main__":
    raise SystemExit(main())
