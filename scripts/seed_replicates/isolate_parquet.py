"""Isolate a run-id list into its own parquet directory.

WHY THIS EXISTS. `pinq_train.gate.run_gate` selects its checkpoint arm by `(arm_id, grid_name)`
alone -- `_select_runs(con, arm=checkpoint_arm, grids=[grid_name], model_id=None)` -- with NO
model filter. The dev-select grids (`dev_select_{musique,strategyqa,wiki2}`) are a fixed
`grid_name` reused by every checkpoint ever gated through them: MEASURED on the shared
`scores/parquet` (2026-09-18), `dev_select_musique` alone carries `inquirer_trained` rows under
9 distinct `code_version`s and dozens of distinct Inquirer models. Pointing `run_gate` straight
at that shared store after a NEW checkpoint's dev sweep lands would pool every checkpoint that
ever ran under that grid name into one "ckpt" set -- a verdict describing no single checkpoint.
The existing qwen3-8b-sft-headline (s0) verdicts
(`artifacts/gate/qwen3-8b-sft-headline.{musique,strategyqa}.json`) were computed against a
private `parquet_dir` for exactly this reason: their own `selection.parquet_dir` names a
scratch path, not `scores/parquet`.

The fix is not a new selector inside `gate.py` -- `_select_runs` already accepts `model_id`,
which is enough to pick the right checkpoint rows OUT of the shared store (verified: the
Inquirer `model` name in `calls.parquet` maps one-to-one to a served checkpoint, independent of
`code_version` or `grid_name`). What has no existing tool is turning that row selection into a
parquet directory `run_gate` can point `--parquet-dir` at, so its own (unfiltered) internal
selection only ever sees the rows chosen here. That is what `isolate_parquet` does: copy
exactly the given `run_id`s, and only those, out of every table a gate parquet directory needs,
via a `WHERE run_id IN (...)` passthrough so every column keeps the source's own type rather
than a reconstruction that could drift from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import duckdb

# The tables `pinq_train.gate._con` refuses to start without, plus the optional `native` table
# (fork-reward runs only) it reads if present and tolerates absent.
REQUIRED_TABLES = ("runs", "turns", "scores", "ledger", "calls")
OPTIONAL_TABLES = ("native",)


def isolate_parquet(parquet_dir: Path, run_ids: Sequence[str], out_dir: Path) -> Path:
    """Copy exactly `run_ids`'s rows of every gate-parquet table into `out_dir`.

    All five required tables are always written, empty or not, so `--parquet-dir <out_dir>` is
    readable by `run_gate` even when a table has no matching row -- an empty checkpoint arm
    must reach `EmptyArm`, not a missing-file crash that reads like an unrelated bug.
    """
    parquet_dir = Path(parquet_dir)
    out_dir = Path(out_dir)
    ids = sorted(set(run_ids))
    if not ids:
        raise ValueError(
            "isolate_parquet: run_ids is empty -- refusing to silently write an "
            "all-tables-empty directory"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    id_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
    written = []
    for name in (*REQUIRED_TABLES, *OPTIONAL_TABLES):
        src = parquet_dir / f"{name}.parquet"
        if not src.exists():
            if name in REQUIRED_TABLES:
                raise FileNotFoundError(
                    f"{src} is missing -- isolate_parquet needs all of {REQUIRED_TABLES}"
                )
            continue
        dst = out_dir / f"{name}.parquet"
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{src.as_posix()}') "
            f"WHERE run_id IN ({id_list})) TO '{dst.as_posix()}' (FORMAT PARQUET)"
        )
        written.append(name)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True, help="the shared compacted parquet dir")
    ap.add_argument("--run-ids-file", required=True, help="one run_id per line")
    ap.add_argument("--out", required=True, help="the isolated parquet dir to write")
    a = ap.parse_args(argv)
    ids = [line.strip() for line in Path(a.run_ids_file).read_text().splitlines() if line.strip()]
    out = isolate_parquet(Path(a.parquet_dir), ids, Path(a.out))
    print(f"wrote {len(ids)} run_ids' rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
