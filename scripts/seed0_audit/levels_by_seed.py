"""Cap-8 coverage LEVELS per arm on the held-out split, for App heldout's "levels behind the contrast".

Per task, the mean `evidence_coverage` over the arm's runs (rollout seeds, and for the recipe both
training seeds), then the mean over tasks: the estimand of the contrast, read one arm at a time. A
level, not a contrast: no comparator, no interval.

Store: the union store of scripts/seed0_audit/union_store.py (the completed cohort plus seeds 1 and 2),
scorer e82c7458. LOCK: seed 0's levels must equal the printed 0.83625 / 0.77630 / 0.92250
(tex:3509-3511 at da38372, from artifacts/testsplit_qa/TESTSPLIT_QA.md) to five decimals before the
recipe's are written. The prompted levels are the completed comparator's and are NOT the printed
short-comparator 0.82031 / 0.87573 / 0.94503 on MuSiQue and 2Wiki, where 24 and 34 tasks were added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

SC = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
S0 = "qwen3-8b-dpo-stacked-notdone-both"
PRINTED_S0 = {"musique": "0.83625", "strategyqa": "0.77630", "wiki2": "0.92250"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    u = a.store
    q = f"""
    WITH m AS (SELECT DISTINCT run_id, model FROM read_parquet('{u}/calls.parquet')
               WHERE actor = 'inquirer'),
    r AS (SELECT r.run_id, r.suite_id, r.task_id, m.model FROM read_parquet('{u}/runs.parquet') r
          JOIN m USING (run_id) WHERE r.grid_name = 'tier1_trained_qa_base' AND r.status = 'ok'),
    c AS (SELECT run_id, value FROM read_parquet('{u}/scores.parquet')
          WHERE metric_name = 'evidence_coverage' AND scorer_hash = '{SC}'),
    t AS (SELECT r.suite_id, r.task_id,
            CASE WHEN r.model IN ('{S0}-s1', '{S0}-s2') THEN 'recipe'
                 WHEN r.model = '{S0}' THEN 's0'
                 WHEN r.model = 'qwen3-8b-base' THEN 'prompted' END AS arm,
            avg(c.value) AS v, count(*) AS n_runs
          FROM r JOIN c USING (run_id) GROUP BY 1, 2, 3)
    SELECT suite_id, arm, avg(v), count(*), min(n_runs), max(n_runs)
    FROM t WHERE arm IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2
    """
    res: dict = {"scorer_hash": SC, "levels": {}, "lock": {}}
    for suite, arm, v, n, lo, hi in duckdb.connect().execute(q).fetchall():
        res["levels"].setdefault(arm, {})[suite] = {
            "level": v,
            "n_tasks": n,
            "runs_per_task": [lo, hi],
        }
        print(f"{suite:10s} {arm:8s} {v:.5f} tasks={n} runs/task={lo}-{hi}")
    for suite, want in PRINTED_S0.items():
        got = f"{res['levels']['s0'][suite]['level']:.5f}"
        res["lock"][suite] = {"printed": want, "reproduced": got, "ok": got == want}
        print(f"LOCK s0 {suite}: printed {want} reproduced {got}")
        if got != want:
            raise SystemExit("LOCK FAILED")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
