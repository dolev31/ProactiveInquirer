#!/usr/bin/env python3
"""A `run_id -> rehydrated` lookup, kept OUTSIDE the frozen `runs.parquet` contract on purpose.

A first attempt put `rehydrated` on `runs.parquet` itself (one line in `pi_eval.schema.RUNS`).
Measured before that merged, side by side, `HEAD^` (without the column) against `HEAD` (with
it), on identical inputs: `pi_eval.schema.schema_hash()` feeds `pi_eval.score.metric_defs_hash()`
feeds `pi_eval.score.scorer_hash()` (`score.py`'s own module docstring states the formula), and
all three differ. A schema addition is SUPPOSED to do that -- `schema_hash()`'s own comment says
so, "a column change writes new score rows instead of quietly re-labelling old ones" -- which
is exactly why it must not happen as a side effect of a census-only convenience column on a
Friday night. `graph_hash`, `matcher_hash` and `METRIC_DEFS_VERSION` are unaffected (they never
read `pi_eval.schema` at all), only the scorer-identity chain moves, and it moves for every
existing scored row in the shared store, not only the 786 this lane investigated.

So this file, not `pi_eval.schema.RUNS`. It reads `manifest.json` directly -- the same ground
truth `pi_run.compact._run_row` would have read `rehydrated_from` from -- and writes a small,
separate parquet with exactly two columns, built with raw pyarrow rather than through
`pi_eval.schema.to_table`/`validate_rows`. It is never registered in `pi_eval.schema.TABLES`,
so `schema_hash()` cannot see it and no identity hash moves when it is written, rewritten, or
deleted. The census (`artifacts/rehydrated_answers_20260918/RESULT.md`) reads this file, or
`scan_rehydrated` directly; nothing else needs to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - names the extra instead of the module
    raise SystemExit(f"pyarrow is required to write the sidecar parquet: {exc}")


def scan_rehydrated(runs_root: Path) -> dict[str, bool]:
    """`{run_id: rehydrated}` for every run directory under `runs_root` that has a readable
    `manifest.json`. Defaults to False for a manifest that never carries the key at all --
    every run this lane never touched -- and for an explicit `null`, matching `bool(None)`."""
    out: dict[str, bool] = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            manifest = json.loads((d / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        run_id = str(manifest.get("run_id", d.name))
        out[run_id] = bool(manifest.get("rehydrated_from"))
    return out


def write_sidecar(runs_root: Path, out_path: Path) -> int:
    """Writes `{out_path}` with columns (run_id: string, rehydrated: bool). Returns the row
    count. Raw pyarrow, deliberately not `pi_eval.schema.to_table`: this table is never
    declared in `pi_eval.schema.TABLES`, so it cannot contribute to `schema_hash()`."""
    mapping = scan_rehydrated(runs_root)
    table = pa.table(
        {
            "run_id": pa.array(list(mapping.keys()), type=pa.string()),
            "rehydrated": pa.array(list(mapping.values()), type=pa.bool_()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return table.num_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    n = write_sidecar(args.runs_root, args.out)
    print(json.dumps({"runs_root": str(args.runs_root), "out": str(args.out), "n_rows": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
