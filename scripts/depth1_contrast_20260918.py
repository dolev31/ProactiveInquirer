"""Depth-one kill-switch contrast: inquirer_trained vs inquirer_depth1.

inquirer_depth1 enumerates all its questions from the initial task statement in a
single call and then walks them, so none of them ever saw an answer (the
sequencing/vertical-axis ablation; see src/pinq_expt/policies/controls.py and
src/pi_eval/prereg.py KILL_SWITCHES["inquirer_depth1"]).

This contrasts it against inquirer_trained on the depth-relevant criteria, on the
TEST split, in the already-scored isolated farm used by artifacts/killswitch/RESULT.md
(grid tier1_trained_killswitch, sha256 038dce8ad81091098273e2220867fbe80a6991b5535c63f41c4cf74a03d0c935).
Basis is UNMATCHED (cap8): both arms run to their own natural stop under the same
nominal budget_cap=8, exactly as artifacts/killswitch/RESULT.md's evidence_coverage
headline did. This is NOT a matched-cost (equal-retrieval-cost) reconstruction.

Lives under scripts/, not artifacts/depth1_contrast_20260918/, because artifacts/ is
excluded from ruff on the recorded ground that no .py file lives there
(tests/test_artifacts_hold_no_python.py enforces that ground). Its output (run_id
lists, contrasts.json) still lands in artifacts/depth1_contrast_20260918/, where the
results belong.
"""

import json
import math
from pathlib import Path

import duckdb

from pi_eval.stats.inference import paired_difference
from pinq_train.gate import cad_ge2_by_run

PQ = "/private/tmp/ks-scratch/pq_clean"
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "depth1_contrast_20260918"
con = duckdb.connect()

ELIGIBLE = (
    "r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE "
    "AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE "
    "AND r.exploratory = FALSE "
    "AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none' "
    "AND r.split = 'test' "
    "AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE"
)

scorer_hash = con.execute(
    f"SELECT DISTINCT scorer_hash FROM read_parquet('{PQ}/scores.parquet')"
).fetchall()
assert len(scorer_hash) == 1, scorer_hash
scorer_hash = scorer_hash[0][0]

graph_version = con.execute(
    f"SELECT DISTINCT graph_version FROM read_parquet('{PQ}/scores.parquet')"
).fetchall()
assert len(graph_version) == 1, graph_version
graph_version = graph_version[0][0]

runs = con.execute(
    f"""
    SELECT r.run_id, r.suite_id, r.task_id, r.arm_id, r.seed, r.template_id,
           r.budget_cap, r.n_asks
    FROM read_parquet('{PQ}/runs.parquet') r
    WHERE {ELIGIBLE} AND r.arm_id IN ('inquirer_depth1', 'inquirer_trained')
    """
).fetchdf()

for arm, label in (("inquirer_trained", "trained"), ("inquirer_depth1", "depth1")):
    ids = sorted(runs.loc[runs.arm_id == arm, "run_id"].tolist())
    path = OUT / f"run_ids.{label}.txt"
    with open(path, "w") as f:
        f.write(f"# artifacts/depth1_contrast_20260918/run_ids.{label}.txt\n")
        f.write(
            "# grid=tier1_trained_killswitch "
            "grid_sha256=038dce8ad81091098273e2220867fbe80a6991b5535c63f41c4cf74a03d0c935\n"
        )
        f.write(f"# scorer_hash={scorer_hash} graph_version={graph_version}\n")
        f.write(f"# store={PQ} split=test arm={arm}\n")
        f.write(f"# n={len(ids)}\n")
        for rid in ids:
            f.write(rid + "\n")

metrics = ["precedence_violation_rate", "max_depth_reached", "dwr", "facet_breadth", "facet_total"]
scores = con.execute(
    f"""
    SELECT run_id, metric_name, value FROM read_parquet('{PQ}/scores.parquet')
    WHERE scorer_hash = '{scorer_hash}' AND metric_name IN ({",".join(f"'{m}'" for m in metrics)})
    """
).fetchdf()

cad = cad_ge2_by_run(Path(PQ), scorer_hash=scorer_hash)

per_metric_run = {m: {} for m in metrics}
for _, row in scores.iterrows():
    per_metric_run[row["metric_name"]][row["run_id"]] = row["value"]
per_metric_run["cad_ge2"] = cad
metrics_all = metrics + ["cad_ge2"]


def run_level_defined_rate(metric_map, arm, suite):
    sub = runs[(runs.arm_id == arm) & (runs.suite_id == suite)]
    tot = sub.run_id.nunique()
    defined = sum(1 for r in sub.run_id if r in metric_map)
    return defined, tot


def task_level(metric_map, arm):
    sub = runs[runs["arm_id"] == arm]
    agg = {}
    for suite, task in sub[["suite_id", "task_id"]].drop_duplicates().itertuples(index=False):
        rids = sub[(sub["suite_id"] == suite) & (sub["task_id"] == task)]["run_id"].tolist()
        vals = [metric_map[r] for r in rids if r in metric_map]
        if vals:
            agg[f"{suite}::{task}"] = sum(vals) / len(vals)
    return agg


tmpl_map = {}
for suite, task, tid in (
    runs[["suite_id", "task_id", "template_id"]].drop_duplicates().itertuples(index=False)
):
    key = f"{suite}::{task}"
    is_nan = isinstance(tid, float) and math.isnan(tid)
    tmpl_map[key] = str(tid) if (tid is not None and not is_nan and str(tid) != "") else key

results = []
print(f"scorer_hash={scorer_hash} graph_version={graph_version}")
print("eligible run counts by arm/suite:")
print(runs.groupby(["arm_id", "suite_id"]).size())
print()

for metric in metrics_all:
    mp = per_metric_run[metric]
    a = task_level(mp, "inquirer_trained")
    b = task_level(mp, "inquirer_depth1")
    for suite in ("musique", "strategyqa"):
        a_s = {k: v for k, v in a.items() if k.startswith(suite + "::")}
        b_s = {k: v for k, v in b.items() if k.startswith(suite + "::")}
        n_a_tasks = runs.loc[
            (runs.arm_id == "inquirer_trained") & (runs.suite_id == suite), "task_id"
        ].nunique()
        n_b_tasks = runs.loc[
            (runs.arm_id == "inquirer_depth1") & (runs.suite_id == suite), "task_id"
        ].nunique()
        n_pairable_suite = max(n_a_tasks, n_b_tasks)
        clusters = {k: tmpl_map.get(k, k) for k in set(a_s) | set(b_s)}
        common = set(a_s) & set(b_s)
        pairs_dropped = n_pairable_suite - len(common)
        deltas = [a_s[k] - b_s[k] for k in common]
        delta_distinct = len(set(round(d, 6) for d in deltas))
        est = paired_difference(a_s, b_s, clusters=clusters, n_boot=1000, n_perm=10000, seed=0)
        rec = {
            "metric": metric,
            "suite": suite,
            "basis": "unmatched_cap8",
            "delta": est.point,
            "ci_lo": est.ci_lo,
            "ci_hi": est.ci_hi,
            "n": est.n,
            "n_pairable_tasks": n_pairable_suite,
            "pairs_dropped": pairs_dropped,
            "delta_distinct": delta_distinct,
            "p_value": est.p_value,
        }
        results.append(rec)
        print(
            f"{metric:28s} {suite:10s} delta={est.point:+.4f} "
            f"CI=[{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n} "
            f"(pairable={n_pairable_suite} dropped={pairs_dropped} distinct={delta_distinct}) "
            f"p={est.p_value:.4f}"
        )
    clusters = {k: tmpl_map.get(k, k) for k in set(a) | set(b)}
    est = paired_difference(a, b, clusters=clusters, n_boot=1000, n_perm=10000, seed=0)
    rec = {
        "metric": metric,
        "suite": "pooled",
        "basis": "unmatched_cap8",
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "n": est.n,
        "p_value": est.p_value,
    }
    results.append(rec)
    print(
        f"{metric:28s} {'pooled':10s} delta={est.point:+.4f} CI=[{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n} p={est.p_value:.4f}"
    )
    print()

with open(OUT / "contrasts.json", "w") as f:
    json.dump(results, f, indent=1)

print("run-level defined rates (metric emitted only where the arm's trajectory makes it defined):")
for metric in ("precedence_violation_rate",):
    mp = per_metric_run[metric]
    for suite in ("musique", "strategyqa"):
        for arm in ("inquirer_trained", "inquirer_depth1"):
            defined, tot = run_level_defined_rate(mp, arm, suite)
            print(f"  {metric} {suite} {arm}: {defined}/{tot} defined ({defined / tot:.1%})")

print()
print("budget_cap / n_asks by arm/suite:")
print(
    runs.groupby(["arm_id", "suite_id"]).agg(
        budget_cap=("budget_cap", "first"),
        avg_asks=("n_asks", "mean"),
        min_asks=("n_asks", "min"),
        max_asks=("n_asks", "max"),
    )
)
