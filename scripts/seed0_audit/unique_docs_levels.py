"""Unmatched question and document levels per arm on the held-out split (App matchedcost's "third
currency" sentence), from runs.parquet's n_asks and unique_docs columns, mean over eligible runs.

LOCK: seed 0's own levels (a checkpoint-only quantity, so the comparator's completion cannot move
them) must equal the record the sentence cites, artifacts/testsplit_qa/CURRENCIES_AND_ESTIMATORS.md
section 2: unique_docs 9.1225 / 2.5400 / 4.3500 and n_asks 2.6250 / 1.5775 / 1.9650 on 400 runs
per suite. Store: the union store of union_store.py (completed comparator, seeds 0, 1 and 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

S0 = "qwen3-8b-dpo-stacked-notdone-both"
PRINTED_S0 = {
    "musique": ("9.1225", "2.6250"),
    "strategyqa": ("2.5400", "1.5775"),
    "wiki2": ("4.3500", "1.9650"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    u = a.store
    q = f"""
    WITH m AS (SELECT DISTINCT run_id, model FROM read_parquet('{u}/calls.parquet')
               WHERE actor = 'inquirer')
    SELECT r.suite_id,
           CASE WHEN m.model IN ('{S0}-s1', '{S0}-s2') THEN 'recipe'
                WHEN m.model = '{S0}' THEN 's0'
                WHEN m.model = 'qwen3-8b-base' THEN 'prompted' END AS arm,
           avg(r.unique_docs), avg(r.n_asks), count(*)
    FROM read_parquet('{u}/runs.parquet') r JOIN m USING (run_id)
    WHERE r.grid_name = 'tier1_trained_qa_base' AND r.status = 'ok'
    GROUP BY 1, 2 HAVING arm IS NOT NULL ORDER BY 1, 2
    """
    res: dict = {"levels": {}, "lock": {}}
    for suite, arm, docs, asks, n in duckdb.connect().execute(q).fetchall():
        res["levels"].setdefault(arm, {})[suite] = {
            "unique_docs": docs,
            "n_asks": asks,
            "n_runs": n,
        }
        print(f"{suite:10s} {arm:8s} unique_docs {docs:.4f} n_asks {asks:.4f} runs {n}")
    for suite, (d, k) in PRINTED_S0.items():
        got = res["levels"]["s0"][suite]
        rep = (f"{got['unique_docs']:.4f}", f"{got['n_asks']:.4f}")
        ok = rep == (d, k)
        res["lock"][suite] = {"printed": [d, k], "reproduced": list(rep), "ok": ok}
        print(f"LOCK s0 {suite}: printed {(d, k)} reproduced {rep}")
        if not ok:
            raise SystemExit("LOCK FAILED")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
