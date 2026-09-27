"""Read-only census of a score store: what CONTRIBUTING.md rule 1 asks for, without the parquet.

CONTRIBUTING.md: "Every reported value traces to a `run_id`, a `scorer_hash` and a `graph_version`."
`runs.parquet` carries `arm_id` / `suite_id` / `split` / `code_version` per run; `scores.parquet`
carries `scorer_hash` / `graph_version` per scored row. A census records the distinct values of
each, with counts, plus a row count and sha256 for every table in the store. That is enough for
a future reader to tell whether a store restored from backup is the one a published cell was
computed from -- same row counts, same hashes, same scorer_hash -- without needing the parquet
itself to check.

This module only reads. It never opens a connection with `PI_GOLD_ROOT` set and never writes to
the store it is given; the census is written by `backup.py`, beside the backup, not here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from scripts.evidence_backup.manifest import sha256_file

# Which table carries which census column. Both tables are optional: an empty `judgments.parquet`
# (0 rows) is normal (seen in `artifacts/banking/scores_parquet`), and a store built before a
# column existed should census as "column absent", never crash the whole run over one store.
_RUNS_TABLE = "runs"
_RUNS_COLUMNS = {
    "arms": "arm_id",
    "suites": "suite_id",
    "splits": "split",
    "code_versions": "code_version",
}
_SCORES_TABLE = "scores"
_SCORES_COLUMNS = {
    "scorer_hashes": "scorer_hash",
    "graph_versions": "graph_version",
}


def _value_counts(parquet_path: Path, column: str) -> dict[str, int]:
    """`{str(value): count}` for one column of one parquet file, or `{}` if the file or column
    does not exist -- a schema gap is data about the store (worth recording, per rule 1: name
    the triple or say which part is missing), never a reason to abort the census."""
    if not parquet_path.exists():
        return {}
    table = pq.read_table(parquet_path)
    if column not in table.column_names:
        return {}
    counts = Counter(table.column(column).to_pylist())
    return {str(value): count for value, count in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def compute_census(store_dir: Path) -> dict:
    """The full census for one store directory: row count and sha256 per `*.parquet` table
    directly inside it, plus distinct arms/suites/splits/code_versions (from `runs.parquet`) and
    scorer_hashes/graph_versions (from `scores.parquet`), each with counts.

    Deterministic in the store's own content: two calls against an unchanged directory return
    equal dicts, and changing one byte of one table changes only that table's entry in `sha256`
    (proven by `tests/test_evidence_backup.py`), never another table's.
    """
    store_dir = Path(store_dir)
    parquet_files = sorted(store_dir.glob("*.parquet"))

    tables: dict[str, int] = {}
    sha256: dict[str, str] = {}
    for p in parquet_files:
        tables[p.stem] = pq.read_metadata(p).num_rows
        sha256[p.name] = sha256_file(p)

    census: dict = {"n_tables": len(parquet_files), "tables": tables, "sha256": sha256}

    runs_path = store_dir / f"{_RUNS_TABLE}.parquet"
    for key, column in _RUNS_COLUMNS.items():
        census[key] = _value_counts(runs_path, column)

    scores_path = store_dir / f"{_SCORES_TABLE}.parquet"
    for key, column in _SCORES_COLUMNS.items():
        census[key] = _value_counts(scores_path, column)

    return census
