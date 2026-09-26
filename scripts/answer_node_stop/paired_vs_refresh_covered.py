"""P(answer-bearing node covered): answer-node arm vs refresh control, paired in its own right.

The two published deltas in `artifacts/answer_node_stop_20260922/RESULT.md` --
answer-node-arm-vs-prompted and refresh-vs-prompted, on this same endpoint -- share a common
baseline (`qwen3-8b-base`), so they cannot be subtracted to recover the arm-vs-control
contrast: the comparator moves with each arm. This computes that missing contrast directly,
with its own `(suite, task, seed)` pairing, exactly the convention
`scripts/answer_node_stop/arm_vs_arm.py` uses for the total-coverage endpoint and
`scripts/answer_node_stop/stratify_by_depth.py` uses for THIS endpoint (against the prompted
base). Both arms here are `arm_id=inquirer_trained` with different `model_id`s, so
`scripts.stopping_answer_test.lib.load_arm_runs` (model as an explicit argument, not a module
constant) is what reads them -- the module-constant reader in this package's own `run.py`
cannot.

Reused, not reimplemented: `scripts.answer_node_coverage.lib.covered_map` for the coverage bar
(`match_kind in ('resolve', 'use')`, read from `matches.parquet`), `build_answer_nodes_for_suite`
for node identification, and `scripts.stopping_answer_test.lib.bca_paired_ratio_delta` /
`with_stability` for the estimator -- the same BCa-over-tasks, ratio-of-sums-per-replicate
family every other coverage delta in this lane uses. The pairing is `_task_ratio_rows`
(`scripts.answer_node_coverage.lib`): per task, `sum(indicator)/count(runs)` pools that task's
seeds within each arm (the "average seeds into the task" step), and only tasks present in BOTH
arms enter the bootstrap (the per-task pairing).

This lane's 2026-09-22 rule: report the bound at 1,000, 10,000 and 50,000 resamples (seed 0),
AND at 50,000 across three bootstrap seeds (101, 202, 303). A bound whose sign is not stable
across the three resample counts, or across the three 50k seeds, is UNDECIDED -- not a pass,
regardless of what any single count reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Package-form imports: `from run import ...` resolves to whichever `run.py` is first on
# sys.path, and both `scripts/answer_node_coverage` and `scripts/stopping_answer_test` have one.
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.answer_node_coverage import lib  # noqa: E402
from scripts.answer_node_coverage.lib import _task_ratio_rows  # noqa: E402
from scripts.answer_node_coverage.run import (  # noqa: E402
    _corpus_dir_for_suite,
    build_answer_nodes_for_suite,
)
from scripts.stopping_answer_test import lib as stopping_lib  # noqa: E402

load_arm_runs = stopping_lib.load_arm_runs

SUITES = ("musique", "strategyqa", "wiki2")
GRID = "tier1_trained_qa_base"
ARM_ID = "inquirer_trained"
RESAMPLE_COUNTS = (1_000, 10_000, 50_000)
STABILITY_SEEDS = (101, 202, 303)
STABILITY_N_BOOT = 50_000
NEAR_ZERO = 0.01


def _excludes_zero(row: dict[str, Any]) -> bool | None:
    if row["n_reps_used"] == 0:
        return None
    return row["lo"] > 0 or row["hi"] < 0


def resample_table(rows: list[tuple[str, float, float, float, float]]) -> dict[str, Any]:
    """The full resample ladder this lane's 2026-09-22 rule requires, computed by calling
    `bca_paired_ratio_delta` directly (the exact function `p_covered_delta` calls internally)
    at each resample count / seed -- `p_covered_delta`'s own `n_boot` kwarg is not honoured by
    `with_stability` (which always computes its first pass at the hardcoded default 10k), so it
    cannot itself produce the 1k/50k cells this rule asks for."""
    by_count = {
        str(nb): stopping_lib.bca_paired_ratio_delta(rows, n_boot=nb, seed=0)
        for nb in RESAMPLE_COUNTS
    }
    by_seed_50k = {
        str(sd): stopping_lib.bca_paired_ratio_delta(rows, n_boot=STABILITY_N_BOOT, seed=sd)
        for sd in STABILITY_SEEDS
    }

    excl_by_count = [_excludes_zero(by_count[str(nb)]) for nb in RESAMPLE_COUNTS]
    excl_by_seed = [_excludes_zero(by_seed_50k[str(sd)]) for sd in STABILITY_SEEDS]

    def signs(entries: list[dict[str, Any]]) -> set[int]:
        out = set()
        for e in entries:
            lo, hi = e["lo"], e["hi"]
            out.add(1 if lo > 0 else (-1 if hi < 0 else 0))
        return out

    count_entries = [by_count[str(nb)] for nb in RESAMPLE_COUNTS]
    seed_entries = [by_seed_50k[str(sd)] for sd in STABILITY_SEEDS]
    stable_across_counts = len(signs(count_entries)) == 1
    stable_across_seeds = len(signs(seed_entries)) == 1

    if any(e["n_reps_used"] == 0 for e in [*count_entries, *seed_entries]):
        verdict = "absent"
    elif stable_across_counts and stable_across_seeds:
        # same sign class (excludes-zero-positive / excludes-zero-negative / spans-zero) at
        # every count and every 50k seed
        pooled_sign = next(iter(signs(count_entries)))
        verdict = "DECIDED" if pooled_sign != 0 else "DECIDED_NULL"
    else:
        verdict = "UNDECIDED"

    return {
        "by_resample_count_seed0": by_count,
        "by_seed_at_50000": by_seed_50k,
        "excludes_zero_by_count": dict(zip((str(n) for n in RESAMPLE_COUNTS), excl_by_count)),
        "excludes_zero_by_seed_50k": dict(zip((str(s) for s in STABILITY_SEEDS), excl_by_seed)),
        "stable_across_counts": stable_across_counts,
        "stable_across_50k_seeds": stable_across_seeds,
        "verdict": verdict,
        "point_10k_seed0": by_count["10000"]["point"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population-dir", required=True, type=Path)
    ap.add_argument("--corpora-root", required=True, type=Path)
    ap.add_argument("--model-answernode", default="qwen3-8b-sft-headline-answernode")
    ap.add_argument("--model-refresh", default="qwen3-8b-sft-headline-refresh")
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    store = a.population_dir / "scores_parquet"
    con = lib.open_store(store)

    out: dict[str, Any] = {
        "store": str(store.resolve()),
        "arm_id": ARM_ID,
        "model_answernode": a.model_answernode,
        "model_refresh": a.model_refresh,
        "graph_version": a.graph_version,
        "grid": GRID,
    }
    per_suite: dict[str, Any] = {}
    strat_per_suite: dict[str, Any] = {}

    for suite in SUITES:
        an_runs = load_arm_runs(
            con, arm_id=ARM_ID, model_id=a.model_answernode, grid_name=GRID, suite=suite
        )
        rf_runs = load_arm_runs(
            con, arm_id=ARM_ID, model_id=a.model_refresh, grid_name=GRID, suite=suite
        )
        task_ids = sorted({r["task_id"] for r in [*an_runs, *rf_runs]})
        corpus_dir = _corpus_dir_for_suite(con, suite, [r["run_id"] for r in [*an_runs, *rf_runs]])
        graphs, results = build_answer_nodes_for_suite(
            suite=suite,
            task_ids=task_ids,
            corpora_root=a.corpora_root,
            corpus_dir=corpus_dir,
            graph_version=a.graph_version,
        )
        keyed = {(suite, t): r for t, r in results.items()}
        covered = lib.covered_map(con, [*an_runs, *rf_runs], keyed)

        ind = {rid: (1.0 if v else 0.0) if v is not None else None for rid, v in covered.items()}
        rows = _task_ratio_rows(an_runs, rf_runs, ind, ind)
        n_an_tasks = len({r["task_id"] for r in an_runs})
        n_rf_tasks = len({r["task_id"] for r in rf_runs})

        table = resample_table(rows)
        table["n_pairs"] = len(rows)
        table["n_tasks_answernode_arm"] = n_an_tasks
        table["n_tasks_refresh_arm"] = n_rf_tasks
        table["n_tasks_dropped_not_in_both"] = len(
            (set(r["task_id"] for r in an_runs) ^ set(r["task_id"] for r in rf_runs))
        )
        per_suite[suite] = table

        # ---- stratified by the answer node's own gold_depth (cheap, additional cut) ----
        stratum: dict[str, str] = {}
        for t, res in results.items():
            by_id = {n.gold_node_id: n for n in graphs[t].required()}
            depths = [
                by_id[nid].gold_depth
                for nid in res.node_ids
                if nid in by_id and by_id[nid].gold_depth is not None
            ]
            if not depths:
                stratum[t] = "no_depth_signal"
            elif max(depths) >= 1:
                stratum[t] = "behind_prerequisite"
            else:
                stratum[t] = "nameable_from_task"

        strat_row: dict[str, Any] = {"tally": {}}
        for name in ("nameable_from_task", "behind_prerequisite", "no_depth_signal"):
            tasks = {t for t, s in stratum.items() if s == name}
            strat_row["tally"][name] = len(tasks)
            if len(tasks) < 10:
                strat_row[name] = {"note": "stratum too small to bootstrap (n<10)", "n": len(tasks)}
                continue
            tr = [r for r in an_runs if r["task_id"] in tasks]
            pr = [r for r in rf_runs if r["task_id"] in tasks]
            strat_row[name] = lib.p_covered_delta(tr, pr, covered, n_boot=10_000)
        strat_per_suite[suite] = strat_row

        print(
            f"  {suite:11s} n_pairs={table['n_pairs']:3d} "
            f"10k={table['by_resample_count_seed0']['10000']['point']:+.4f} "
            f"[{table['by_resample_count_seed0']['10000']['lo']:+.4f},"
            f"{table['by_resample_count_seed0']['10000']['hi']:+.4f}] "
            f"verdict={table['verdict']}"
        )
        parts = []
        for name in ("nameable_from_task", "behind_prerequisite"):
            d = strat_row.get(name)
            if d and "point" in d:
                parts.append(
                    f"{name} {d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] n={d['n_tasks']}"
                )
            elif d:
                parts.append(f"{name} {d.get('note')} n={d.get('n')}")
        print(f"    stratified: tally={strat_row['tally']}  " + " | ".join(parts))

    out["paired_vs_refresh"] = per_suite
    out["stratified_by_gold_depth"] = strat_per_suite
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True, default=lambda o: None) + "\n")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
