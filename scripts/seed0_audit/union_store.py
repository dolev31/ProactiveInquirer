"""One read-only scratch store: the completed cohort (seed 0 and the completed prompted comparator)
plus seeds 1 and 2 from the seed-replicate store, so `pinq_train.gate.run_gate`, which reads one
parquet directory, can read the recipe against the comparator Table 1 reads.

Both source stores are scorer_hash e82c7458 and code_version 3ae099d0, and every run they share
carries identical coverage, ladder and n_asks (checked by heldout_methods_by_seed.py, 2,282 runs).
From the seed-replicate store ONLY seed 1 and seed 2's run ids are taken (its own copies of seed 0
and of the 1,082 short-comparator runs are left out), so no run enters twice; the script refuses
if any run id appears in both halves of the union.

Writes runs, turns, scores, ledger and calls parquet files into --out (a scratch directory, never
under the repo's scores/ store).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

S0 = "qwen3-8b-dpo-stacked-notdone-both"
TABLES = ("runs", "turns", "scores", "ledger", "calls")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cc = a.data_root / "artifacts/completed_cohort_20260922/scores_parquet"
    sr = a.data_root / "artifacts/seedrep_gate_20260919/scores_parquet"
    ids: set[str] = set()
    for m in (f"{S0}-s1", f"{S0}-s2"):
        for s in ("musique", "strategyqa", "wiki2"):
            p = a.data_root / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.{m}.{s}.txt"
            ids |= {x.strip() for x in p.read_text().split() if x.strip()}
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE keep (run_id VARCHAR)")
    con.executemany("INSERT INTO keep VALUES (?)", [(r,) for r in sorted(ids)])
    clash = con.execute(
        f"SELECT count(*) FROM read_parquet('{cc}/runs.parquet') WHERE run_id IN (SELECT run_id FROM keep)"
    ).fetchone()[0]
    if clash:
        raise SystemExit(f"{clash} seed-1/2 run ids already in the completed cohort store")
    a.out.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{cc}/{t}.parquet') UNION ALL BY NAME "
            f"SELECT * FROM read_parquet('{sr}/{t}.parquet') WHERE run_id IN (SELECT run_id FROM keep)) "
            f"TO '{a.out / (t + '.parquet')}' (FORMAT PARQUET)"
        )
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{a.out / (t + '.parquet')}')"
        ).fetchone()[0]
        print(f"{t}: {n} rows")
    print(f"runs from seed-replicate store: {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
