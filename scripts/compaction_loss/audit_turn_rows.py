"""Which runs hold turn records on disk and have no turn rows in the parquet.

THE PREDICATE IS ON DISK, NOT ON A JOIN AGAINST AN OLD HASH. L0.6 found the loss by joining
old-hash score rows to new-hash ones, which can only see runs that HAD rows under some old
hash; on tau2/inquirer_prompted/test that reads 355 where the disk predicate reads 355 lost out
of 603 rows whose `runs.n_turns` is 0. The other 248 are genuinely empty on disk, and a run
whose policy never asks is not damage: `drafter_only` is 780 of 780 empty on tau2, 836 of 836
on musique, 733 of 733 on wiki2. Counting those as loss would make the number meaningless and
the check disableable, so a no-ask arm is excluded BY CONSTRUCTION -- it has no turn lines on
disk, so it is never in the numerator or the denominator.

    lost           turns.jsonl holds >= 1 non-empty line AND turns.parquet holds 0 rows for it
    empty_on_disk  turns.jsonl is absent or holds 0 non-empty lines AND 0 rows in the parquet
    ok             >= 1 row in turns.parquet

Usage:
    python scripts/compaction_loss/audit_turn_rows.py <runs_root> <parquet_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def n_turn_lines(run_dir: Path) -> int:
    p = run_dir / "turns.jsonl"
    if not p.exists():
        return 0
    with p.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def audit(runs_root: Path, parquet_dir: Path, since: float | None = None) -> dict:
    """`since` is the epoch second the compaction being audited STARTED its scan.

    On a live tree a run that finishes while the walker is somewhere else in the sorted order
    has turn lines on disk and no rows in the table, and that is not loss -- it is a run that
    arrived after the snapshot. Measured on the 22:36:46 pass: four tau2_retail runs, written
    at 22:37:50, 22:41:00, 22:43:54 and 22:44:00, all inside the scan window.

    `since` is the compaction's OWN start time, read from its log, never a value picked to
    make the count come out. A run whose turns.jsonl predates it stays LOST: the split is on
    the file's mtime, so the escape hatch cannot swallow a real loss, and both counts are
    reported either way.
    """
    import pyarrow.parquet as pq

    runs_tbl = pq.read_table(
        parquet_dir / "runs.parquet", columns=["run_id", "suite_id", "arm_id", "split", "n_turns"]
    )
    meta = {
        r["run_id"]: (r["suite_id"], r["arm_id"], r["split"], r["n_turns"])
        for r in runs_tbl.to_pylist()
    }
    turn_ids = {
        i
        for i in pq.read_table(parquet_dir / "turns.parquet", columns=["run_id"])
        .column("run_id")
        .to_pylist()
        if i
    }
    n_null = sum(
        1
        for i in pq.read_table(parquet_dir / "turns.parquet", columns=["run_id"])
        .column("run_id")
        .to_pylist()
        if i is None
    )

    lost: list[str] = []
    empty: list[str] = []
    later: list[str] = []
    by_cell: Counter = Counter()
    empty_by_cell: Counter = Counter()
    later_by_cell: Counter = Counter()
    n_lines = 0
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or not (d / "status.json").exists():
            continue
        rid = d.name
        if rid not in meta:
            continue
        if rid in turn_ids:
            continue
        n = n_turn_lines(d)
        cell = meta[rid][:3]
        if not n:
            empty.append(rid)
            empty_by_cell[cell] += 1
            continue
        if since is not None and (d / "turns.jsonl").stat().st_mtime >= since:
            later.append(rid)
            later_by_cell[cell] += 1
            continue
        lost.append(rid)
        by_cell[cell] += 1
        n_lines += n

    return {
        "runs_root": str(runs_root),
        "parquet_dir": str(parquet_dir),
        "since": since,
        "n_runs_in_parquet": len(meta),
        "n_runs_with_turn_rows": len(turn_ids),
        "n_turn_rows_with_null_run_id": n_null,
        "n_lost": len(lost),
        "n_turn_lines_lost": n_lines,
        "n_arrived_after_snapshot": len(later),
        "n_empty_on_disk": len(empty),
        "lost_by_cell": {"/".join(map(str, k)): v for k, v in sorted(by_cell.items())},
        "arrived_after_snapshot_by_cell": {
            "/".join(map(str, k)): v for k, v in sorted(later_by_cell.items())
        },
        "empty_by_cell": {"/".join(map(str, k)): v for k, v in sorted(empty_by_cell.items())},
        "lost_run_ids": sorted(lost),
        "arrived_after_snapshot_run_ids": sorted(later),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("parquet_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--since",
        type=float,
        default=None,
        help="Epoch second the audited compaction STARTED its scan, from its own log. A run "
        "whose turns.jsonl was written at or after it is counted as arrived_after_snapshot "
        "rather than lost; one written before it stays lost.",
    )
    a = ap.parse_args(argv)

    res = audit(a.runs_root, a.parquet_dir, since=a.since)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    printable = {
        k: v for k, v in res.items() if k not in ("lost_run_ids", "arrived_after_snapshot_run_ids")
    }
    print(json.dumps(printable, indent=1))
    # Non-zero when the store has lost turn rows, so a pipeline step can gate on it.
    return 1 if res["n_lost"] or res["n_turn_rows_with_null_run_id"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
