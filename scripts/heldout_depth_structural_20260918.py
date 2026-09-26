"""Structural coarseness check for the two held-out matched-cost depth quantities.

paper/results.tex Section sub:heldoplan (labelled sub:heldoutplan) now reports dwr and
max_depth_reached at matched retrieval cost on the held-out split, against inquirer_prompted,
in artifacts/testsplit_qa/scores_parquet. Before either quantity is treated as evidence about the
POLICY, this checks whether the gold graphs it is read against let it vary at all: the depth
histogram of required nodes per task per suite (a property of the graphs, not of any arm), and the
correlation of each quantity against plain evidence_coverage on the same population, so a reading
that is just a monotone reflection of coverage is named as one rather than reported as separable
evidence.

Two things this script does NOT do, stated so the gap is not silently read as an oversight:
1. The correlation is computed on the STORED TERMINAL (unmatched, cap8) values, not the
   matched-cost reconstruction. No prefix ladder for dwr/max_depth_reached is stored anywhere,
   artifacts/testsplit_plan_metrics_20260918/RESULT.md built one by hand against
   matches.parquet, and rebuilding that ladder a second time here risks a second, divergent
   implementation of the same gold-side logic. The distinct-per-task-delta coarseness census AT
   matched cost is already in paper/results.tex's own comment on sub:heldoutplan (39/24/12 for
   dwr, 9/9/6 for max_depth_reached, musique/strategyqa/wiki2) and is not recomputed here.
2. It reads gold graphs (pi_eval.gold.load_graphs) under PI_GOLD_ROOT, which this script's own
   process must have set. It writes nothing under data/ or runs/ and touches no live store.

Lives under scripts/, not under artifacts/, for the same reason
scripts/depth1_contrast_20260918.py does: artifacts/ is excluded from ruff on the recorded ground
that no .py file lives there (tests/test_artifacts_hold_no_python.py enforces that ground). Its
output lands in artifacts/heldout_depth_structural_20260918/, where the results belong.
"""

import collections
import json
import os
from pathlib import Path

import duckdb
import numpy as np

from pi_eval.gold import load_graphs

# artifacts/testsplit_qa/ is untracked and lives only in the main checkout, not in every
# worktree, so this defaults to a repo-root-relative path (correct once this script is run
# from the main checkout, which is where it will live after merge) and accepts an override
# for a worktree session where the main checkout sits elsewhere. No literal home path here.
STORE = os.environ.get("HELDOUT_DEPTH_STRUCTURAL_STORE", "artifacts/testsplit_qa/scores_parquet")
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "heldout_depth_structural_20260918"
SUITES = ("musique", "strategyqa", "wiki2")
METRICS = ["evidence_coverage", "max_depth_reached", "dwr", "facet_breadth"]

con = duckdb.connect()

scorer_hash = con.execute(
    f"SELECT DISTINCT scorer_hash FROM read_parquet('{STORE}/scores.parquet')"
).fetchall()
assert len(scorer_hash) == 1, scorer_hash
scorer_hash = scorer_hash[0][0]
graph_version = con.execute(
    f"SELECT DISTINCT graph_version FROM read_parquet('{STORE}/scores.parquet')"
).fetchall()
assert len(graph_version) == 1, graph_version
graph_version = graph_version[0][0]

report = {"scorer_hash": scorer_hash, "graph_version": graph_version, "suites": {}}

for suite in SUITES:
    task_ids = (
        con.execute(
            f"""
        SELECT DISTINCT task_id FROM read_parquet('{STORE}/runs.parquet')
        WHERE suite_id = '{suite}' AND split = 'test'
        """
        )
        .fetchdf()["task_id"]
        .tolist()
    )

    graphs = load_graphs(suite, graph_version)
    assert all(t in graphs for t in task_ids), "gold_task_key must match run task_id 1:1"

    depth_hist = collections.Counter()
    tasks_with_ge2 = 0
    max_depth_per_task = []
    for t in task_ids:
        req = graphs[t].required()
        depths = [n.gold_depth for n in req]
        for d in depths:
            depth_hist[d] += 1
        if any(d is not None and d >= 2 for d in depths):
            tasks_with_ge2 += 1
        md = max([d for d in depths if d is not None], default=None)
        max_depth_per_task.append(md)
    max_depth_hist = collections.Counter(max_depth_per_task)

    runs = con.execute(
        f"""
        SELECT run_id, task_id FROM read_parquet('{STORE}/runs.parquet')
        WHERE suite_id = '{suite}' AND split = 'test' AND arm_id = 'inquirer_trained'
        """
    ).fetchdf()
    scores = con.execute(
        f"""
        SELECT run_id, metric_name, value FROM read_parquet('{STORE}/scores.parquet')
        WHERE metric_name IN ({",".join(f"'{m}'" for m in METRICS)})
        """
    ).fetchdf()
    wide = scores.pivot_table(
        index="run_id", columns="metric_name", values="value", aggfunc="first"
    )
    df = runs.merge(wide, left_on="run_id", right_index=True, how="left")
    task_level = df.groupby("task_id")[METRICS].mean()

    corr = {}
    for m in ("dwr", "max_depth_reached", "facet_breadth"):
        sub = task_level.dropna(subset=[m, "evidence_coverage"])
        pear = float(np.corrcoef(sub[m], sub["evidence_coverage"])[0, 1])
        spear = float(sub[m].rank().corr(sub["evidence_coverage"].rank()))
        corr[m] = {"n": len(sub), "pearson": pear, "spearman": spear}

    report["suites"][suite] = {
        "n_test_tasks": len(task_ids),
        "required_node_depth_histogram": dict(sorted(depth_hist.items())),
        "tasks_with_required_node_at_depth_ge2": tasks_with_ge2,
        "per_task_max_required_depth_histogram": {
            str(k): v
            for k, v in sorted(max_depth_hist.items(), key=lambda kv: (kv[0] is None, kv[0]))
        },
        "task_level_distinct_values_terminal_cap8": {
            m: int(task_level[m].round(6).nunique()) for m in METRICS
        },
        "correlation_vs_evidence_coverage_terminal_cap8": corr,
    }
    print(f"=== {suite} ===")
    print(
        f"  required-node depth histogram (n={len(task_ids)} tasks): {report['suites'][suite]['required_node_depth_histogram']}"
    )
    print(f"  tasks with a required node at depth>=2: {tasks_with_ge2}/{len(task_ids)}")
    print(
        f"  per-task max required depth: {report['suites'][suite]['per_task_max_required_depth_histogram']}"
    )
    for m in ("dwr", "max_depth_reached", "facet_breadth"):
        c = corr[m]
        print(
            f"  corr({m}, evidence_coverage) terminal cap8, n={c['n']}: pearson={c['pearson']:.4f} spearman={c['spearman']:.4f}"
        )
    print()

with open(OUT / "structural_check.json", "w") as f:
    json.dump(report, f, indent=1)
