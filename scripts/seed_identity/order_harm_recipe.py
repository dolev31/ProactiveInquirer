"""Does resolving needs out of prerequisite order cost the answer, for the recipe?

scripts/order_harm/does_order_cost_answers.py answered this for seed 0 on the completed cohort
(artifacts/order_harm_20260922). This reads the same question for the recipe, training seeds 1 and 2,
against the SAME comparator runs, through the same function (`run_cells`): pairs within the task,
strata by whether the trained arm resolved more out of order than the comparator on that task, and
an independent-samples difference between the two strata's paired answer deltas.

The recipe's per-task value is Table 1's: each training seed's runs are averaged within the task,
then the two seeds are averaged. The out-of-order rate is averaged over the runs that define it, and
a task enters only where both arms define it (a seed that never defines it on a task drops out of
that task's average rather than counting as zero).

LOCK: seed 0 read through this path, from the same store and runs as the published record, must
reproduce artifacts/order_harm_20260922/order_vs_answers.json exactly before any recipe cell is
written. Arms are selected by the inquirer MODEL on calls.parquet, never by arm_id alone, since the
seed-replicate store holds three weights under one arm_id.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "order_harm"))
import does_order_cost_answers as oh  # noqa: E402

COHORT = REPO / "artifacts/completed_cohort_20260922/scores_parquet"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919/scores_parquet"
PUBLISHED = REPO / "artifacts/order_harm_20260922/order_vs_answers.json"
S0 = "qwen3-8b-dpo-stacked-notdone-both"
RECIPE = (f"{S0}-s1", f"{S0}-s2")
BASE = "qwen3-8b-base"
WANTED = ("precedence_violation_rate", *oh.ANSWER_METRICS)
SUITES = ("musique", "strategyqa", "wiki2")


def per_task(store: Path, model: str, arm: str) -> dict[tuple[str, str], dict[str, float]]:
    """(suite, task) -> metric mean over this model's eligible test runs of that arm on that task."""
    con = duckdb.connect()
    rows = con.execute(
        f"""
        WITH m AS (SELECT DISTINCT run_id, model FROM read_parquet('{store / "calls.parquet"}')
                   WHERE actor = 'inquirer')
        SELECT r.suite_id, r.task_id, s.metric_name, avg(s.value)
        FROM read_parquet('{store / "runs.parquet"}') r
        JOIN read_parquet('{store / "scores.parquet"}') s USING (run_id)
        JOIN m USING (run_id)
        WHERE r.split = 'test' AND r.status = 'ok' AND r.arm_id = ? AND m.model = ?
          AND s.metric_name IN {WANTED} AND s.value IS NOT NULL
        GROUP BY 1, 2, 3
        """,
        [arm, model],
    ).fetchall()
    out: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for suite, task, metric, value in rows:
        out[(suite, task)][metric] = float(value)
    return out


def cells_for(trained: list[dict], prompted: dict) -> dict[tuple[str, str, str], dict[str, float]]:
    cells: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, metrics in prompted.items():
        cells[(key[0], key[1], oh.PROMPTED)] = metrics
    keys = set().union(*[set(t) for t in trained])
    for key in keys:
        avg = {}
        for metric in WANTED:
            vals = [t[key][metric] for t in trained if key in t and metric in t[key]]
            if vals:
                avg[metric] = sum(vals) / len(vals)
        cells[(key[0], key[1], oh.TRAINED)] = avg
    return cells


def _same(a, b, path="") -> list[str]:
    if isinstance(a, dict):
        return [x for k in set(a) | set(b) for x in _same(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _same(u, v, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 else [path]
    return [] if a == b else [path]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    prompted = per_task(COHORT, BASE, oh.PROMPTED)
    s0 = cells_for([per_task(COHORT, S0, oh.TRAINED)], prompted)
    got0 = json.loads(json.dumps(oh.run_cells(s0, SUITES)))
    want0 = json.loads(PUBLISHED.read_text())
    diffs = _same(got0, want0)
    out: dict = {"lock": {"published": str(PUBLISHED.relative_to(REPO)), "differences": diffs}}
    if diffs:
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"LOCK FAILED: {len(diffs)} fields differ, e.g. {diffs[:5]}; no recipe cell written")
        return 1
    per_seed = [per_task(SEEDREP, m, oh.TRAINED) for m in RECIPE]
    out["recipe"] = oh.run_cells(cells_for(per_seed, prompted), SUITES)
    for lab, t in zip(("s1", "s2"), per_seed):
        out[lab] = oh.run_cells(cells_for([t], prompted), SUITES)
    out["stores"] = {
        "comparator_and_seed0": str(COHORT.relative_to(REPO)),
        "recipe": str(SEEDREP.relative_to(REPO)),
    }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; seed-0 lock exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
