"""Within-task question diversity (`distinct3`) per checkpoint and suite.

Reported because it is the criterion that FLIPPED between development and test for s0 in
`artifacts/testsplit_qa/` -- 0.95135 dev to 0.57844 test on musique, against a 0.65 floor --
so a seed replicate of it is worth as much as the coverage cell. It is a LEVEL against a fixed
floor, not a contrast: no comparator, no interval, and it is not differenced here.

Both `value` and `by_seed` are printed. `value` pools a task's two evaluation seeds, so a
checkpoint that repeats its own questions across seeds is scored as if it had collapsed;
`by_seed` uses the (task, seed) key. Pooling across seeds measures determinism as much as
diversity, so the two are stated side by side rather than one being chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinq_train.gate import _con, _questions, _questions_with_seed, _select_runs
from pinq_train.rung1_sft.collapse import within_task_distinct_n, within_task_seed_distinct_n

S0 = "qwen3-8b-dpo-stacked-notdone-both"
ARMS = [
    (S0, "inquirer_trained"),
    (f"{S0}-s1", "inquirer_trained"),
    (f"{S0}-s2", "inquirer_trained"),
    ("qwen3-8b-base", "inquirer_prompted"),
]
GRID = "tier1_trained_qa_base"
FLOOR = 0.65


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = _con(Path(a.parquet))
    out: dict = {"floor": FLOOR, "cells": {}}
    for mid, arm in ARMS:
        runs = _select_runs(con, arm=arm, grids=[GRID], model_id=mid)
        for suite in ("musique", "strategyqa", "wiki2"):
            sub = [r for r in runs if r["suite_id"] == suite]
            pairs = _questions(con, sub)
            trip = _questions_with_seed(con, sub)
            d3 = within_task_distinct_n(pairs) if pairs else float("nan")
            d3s = within_task_seed_distinct_n(trip) if trip else float("nan")
            out["cells"][f"{mid}|{suite}"] = {
                "model_id": mid,
                "suite": suite,
                "n_runs": len(sub),
                "n_tasks_with_asks": len({t for t, _ in pairs}),
                "n_questions": len(pairs),
                "distinct3_value_seed_pooled": d3,
                "distinct3_by_seed": d3s,
                "passes_floor_value": bool(d3 == d3 and d3 >= FLOOR),
                "passes_floor_by_seed": bool(d3s == d3s and d3s >= FLOOR),
            }
    Path(a.out).write_text(json.dumps(out, indent=2))
    for k, v in out["cells"].items():
        print(
            f"{v['model_id']:40s} {v['suite']:11s} n_q={v['n_questions']:5d} "
            f"value={v['distinct3_value_seed_pooled']:.5f} (pass={v['passes_floor_value']}) "
            f"by_seed={v['distinct3_by_seed']:.5f} (pass={v['passes_floor_by_seed']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
