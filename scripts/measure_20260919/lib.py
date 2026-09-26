"""
Shared helpers for the 2026-09-19 trained-vs-teacher measurement lane.
READ-ONLY: queries an existing isolated scoring store. No rollouts, no LLM calls.

Store        : built by scripts/contributions_on_test/teacher_token_charge.py (not re-run here)
scorer_hash  : e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7
graph_version: v1
TEACHER_PIN  : 5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf
               (arm_id=inquirer_prompted, model=gpt-oss-120b / 120B teacher, code_version=fc1def5dc0c7690558ae07d01fb70eb152e91915)
TRAINED_PIN  : 6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856
               (arm_id=inquirer_trained, code_version=3ae099d0e9f08f6654d5e259ffc0850232f8e70a)

Estimator: pi_eval.stats.inference.cluster_bootstrap / paired_difference, imported and
called directly (not reimplemented) -- task-clustered BCa, 10000 resamples, seed 0.
Seeds are averaged into the task BEFORE resampling via avg(s.value) GROUP BY task_id,
cluster_id, mirroring pi_eval.report.metric_query's own SQL exactly.
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pi_eval.stats.inference import cluster_bootstrap, paired_difference  # noqa: E402

STORE = "/private/tmp/claude-501/-Users-someone-PycharmProjects-ProactiveInquirer/3593f11f-056f-4800-a24f-c5870cda2e31/scratchpad/teacher_token_charge_store"
SCORER_HASH = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
GRAPH_VERSION = "v1"
TEACHER_PIN = "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf"
TRAINED_PIN = "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856"
SUITES = ("musique", "strategyqa", "wiki2")
N_BOOT = 10_000
SEED = 0

ELIGIBLE = """
    r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE
    AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE
    AND r.exploratory = FALSE
    AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none'
    AND r.split = 'test'
    AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE
"""

_con = duckdb.connect()
RUNS = f"read_parquet('{STORE}/runs.parquet')"
SCORES = f"read_parquet('{STORE}/scores.parquet')"


def per_task_values(metric_name: str, suite: str, pin: str) -> dict:
    """task_id -> (avg-over-seed value, cluster_id). Mirrors report.metric_query's GROUP BY (no seed)."""
    q = f"""
    SELECT r.task_id AS task_id,
           COALESCE(NULLIF(r.template_id,''), r.task_id) AS cluster_id,
           avg(s.value) AS value
    FROM {SCORES} s JOIN {RUNS} r ON r.run_id = s.run_id
    WHERE s.metric_name = ? AND s.scorer_hash = ?
      AND r.suite_id = ? AND r.model_pin_hash = ?
      AND ({ELIGIBLE})
    GROUP BY 1, 2
    """
    rows = _con.execute(q, [metric_name, SCORER_HASH, suite, pin]).fetchall()
    return {task_id: (value, cluster_id) for task_id, cluster_id, value in rows}


def level_point(values: dict, clusters: dict, n_boot=N_BOOT, seed=SEED):
    """Replicates pi_eval.report.level_estimate's grouping + cluster_bootstrap call exactly."""
    groups = defaultdict(list)
    for k, v in values.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        groups[clusters.get(k, k)].append(v)
    if not groups:
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n": 0,
            "n_clusters": 0,
        }
    point, lo, hi = cluster_bootstrap(list(groups.values()), n_boot=n_boot, seed=seed)
    n = sum(len(v) for v in groups.values())
    return {"point": point, "lo": lo, "hi": hi, "n": n, "n_clusters": len(groups)}


def paired(metric_name: str, suite: str):
    trained_raw = per_task_values(metric_name, suite, TRAINED_PIN)
    teacher_raw = per_task_values(metric_name, suite, TEACHER_PIN)
    trained_vals = {k: v for k, (v, _c) in trained_raw.items()}
    teacher_vals = {k: v for k, (v, _c) in teacher_raw.items()}
    clusters = {k: c for k, (_v, c) in trained_raw.items()}
    clusters.update({k: c for k, (_v, c) in teacher_raw.items()})

    keys = sorted(set(trained_vals) & set(teacher_vals))
    inter_clusters = {k: clusters[k] for k in keys}

    trained_level = level_point({k: trained_vals[k] for k in keys}, inter_clusters)
    teacher_level = level_point({k: teacher_vals[k] for k in keys}, inter_clusters)

    est = paired_difference(trained_vals, teacher_vals, clusters=clusters, n_boot=N_BOOT, seed=SEED)

    return {
        "metric": metric_name,
        "suite": suite,
        "n_trained_full": len(trained_vals),
        "n_teacher_full": len(teacher_vals),
        "n_paired": len(keys),
        "trained_level": trained_level,
        "teacher_level": teacher_level,
        "delta": est,  # trained - teacher
    }
