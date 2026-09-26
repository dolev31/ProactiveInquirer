"""Step 3 of 4: one scoring pass, three per-arm stores.

`pi train gate` selects the checkpoint side by ARM_ID, and all three label arms are
`inquirer_trained` under one `grid_name`. A store holding two of them would gate their UNION and
print a number that is not any arm's. Filtering by run_id after scoring is exact and re-scores
nothing, so all three stores carry one scorer_hash.
"""

import os
import pathlib

import duckdb

WORK = pathlib.Path(os.environ["PI_WORK"])
SRC = WORK / "parquet"
IDS = WORK / "run_ids"
TABLES = [
    "runs",
    "scores",
    "matches",
    "evidence",
    "turns",
    "calls",
    "ledger",
    "native",
    "judgments",
    "env_calls",
]
baseline = set((IDS / "qwen3-8b-base.txt").read_text().split())
con = duckdb.connect()
for arm in [
    "qwen3-8b-dpo-headline-control",
    "qwen3-8b-dpo-headline-rater",
    "qwen3-8b-dpo-headline-reaches",
]:
    keep = baseline | set((IDS / f"{arm}.txt").read_text().split())
    out = WORK / f"store_{arm}"
    out.mkdir(exist_ok=True)
    con.execute(
        "create or replace table k as select * from (values "
        + ",".join(f"('{r}')" for r in sorted(keep))
        + ") t(run_id)"
    )
    for t in TABLES:
        src = SRC / f"{t}.parquet"
        cols = [
            r[0] for r in con.execute(f"describe select * from read_parquet('{src}')").fetchall()
        ]
        where = " where run_id in (select run_id from k)" if "run_id" in cols else ""
        con.execute(
            f"copy (select * from read_parquet('{src}'){where}) "
            f"to '{out / (t + '.parquet')}' (format parquet)"
        )
    n = con.execute(f"select count(*) from read_parquet('{out / 'runs.parquet'}')").fetchone()[0]
    print(f"{arm}: runs={n} keep_ids={len(keep)}")
