#!/usr/bin/env python3
"""Per-shape matched-cost analysis for the two-axis campaign.

COPIED VERBATIM from /tmp/two_axis_matched_20260922/analyze_shape.py (the script behind
artifacts/two_axis_matched_20260922/analysis_{2x4,3x6,4x8}.json) so the recipe reading runs the
published analysis unchanged. The one edit: WORKTREE is read from PINQ_TWO_AXIS_WORKTREE instead of
a hard-coded home path. Note the pairing rule it inherits from ladder._pair_keys at that commit:
every trained run of a task against every prompted run of it (a cross product at the task), not
Table 1's seed-matched pairing.

Reuses, never reimplements, the gold-touching pieces:
  * pi_eval.gold.load_graphs, pi_eval.score._gold_uids
  * scripts/plan_metrics_symmetric/ladder.py: RunLadder, load_run_ladders, METRIC_FNS,
    symmetric_contrast, _pair_keys
  * scripts/plan_metrics_symmetric/sweep.py: load_coverage_ladders, EXTRA_METRIC_FNS
    (evidence_coverage), DIAGNOSTIC_METRIC_FNS (facet_breadth_scorer -- the COUNT),
    registered_metrics, terminal_check, _stored_values

Only the per-task non-vacuity tabulation (distinct/discordant/tie counts) is new code, and it
is a literal copy of ladder.symmetric_contrast's own pairing loop so the numbers it reports
describe exactly the population the published contrast was computed over -- not a separate
guess at it.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# The campaign's pinned worktree (code 1979fb7ef321). Required, never defaulted: the ladder and
# scorer code this imports must be the code that produced the runs.
WORKTREE = os.environ["PINQ_TWO_AXIS_WORKTREE"]
sys.path.insert(0, f"{WORKTREE}/src")
sys.path.insert(0, f"{WORKTREE}/scripts")

import duckdb  # noqa: E402

os.environ.setdefault("PI_GOLD_ROOT", "/dev/null-unused")  # overwritten below per call

from plan_metrics_symmetric import ladder as ladder_mod  # noqa: E402
from plan_metrics_symmetric import sweep as sweep_mod  # noqa: E402

METRICS = [
    "evidence_coverage",
    "facet_breadth_scorer",
    "max_depth_reached",
    "dwr",
    "precedence_violation_rate",
]
COVERAGE_METRICS = {"evidence_coverage"}


def load_run_table(parquet_dir: Path) -> "list":
    con = duckdb.connect()
    return con.execute(
        f"SELECT run_id, arm_id, task_id, seed, n_asks, is_dev_run, dirty, code_version, "
        f"model_pin_hash, corpus_hash, canary_hit, firewall_ok "
        f"FROM read_parquet('{(parquet_dir / 'runs.parquet').as_posix()}')"
    ).fetchdf()


def analyze(shape: str, root_dir: Path, parquet_dir: Path) -> dict:
    os.environ["PI_GOLD_ROOT"] = str(root_dir / "data" / "gold")
    from pi_eval.gold import load_graphs  # local import: needs PI_GOLD_ROOT set first

    graphs = load_graphs("synth", "v1")
    runs = load_run_table(parquet_dir)
    if bool(runs["is_dev_run"].any()):
        dev_ids = runs.loc[runs["is_dev_run"], "run_id"].tolist()
        raise SystemExit(f"REFUSING: {shape} has dev- runs: {dev_ids}")
    if bool(runs["dirty"].any()):
        raise SystemExit(f"REFUSING: {shape} has dirty=True runs")
    if runs["code_version"].nunique() != 1:
        raise SystemExit(
            f"REFUSING: {shape} spans multiple code_version values: {runs['code_version'].unique()}"
        )
    if runs["corpus_hash"].nunique() != 1:
        raise SystemExit(
            f"REFUSING: {shape} spans multiple corpus_hash values: {runs['corpus_hash'].unique()}"
        )
    if "firewall_ok" in runs and not bool(runs["firewall_ok"].fillna(True).all()):
        raise SystemExit(f"REFUSING: {shape} has firewall_ok=False rows")
    if "canary_hit" in runs and bool(runs["canary_hit"].fillna(False).any()):
        raise SystemExit(f"REFUSING: {shape} has a canary hit (gold leak)")

    by_arm = {
        arm: runs.loc[runs["arm_id"] == arm, "run_id"].tolist()
        for arm in ["inquirer_trained", "inquirer_prompted", "shallow_wide", "deep_narrow"]
    }
    all_ids = runs["run_id"].tolist()

    run_ladders = {
        rid: lad for rid, lad in ladder_mod.load_run_ladders(parquet_dir, all_ids).items()
    }
    coverage_ladders = sweep_mod.load_coverage_ladders(parquet_dir, all_ids, graphs)
    stored = sweep_mod._stored_values(parquet_dir, all_ids)

    with sweep_mod.registered_metrics(
        {**sweep_mod.EXTRA_METRIC_FNS, **sweep_mod.DIAGNOSTIC_METRIC_FNS}
    ):
        # ---- instrument lock: does OUR ladder reproduce the SCORER's own stored value? ----
        instrument = {}
        for metric in METRICS:
            arms_map = coverage_ladders if metric in COVERAGE_METRICS else run_ladders
            instrument[metric] = sweep_mod.terminal_check(metric, arms_map, graphs, stored)

        # ---- matched-cost symmetric contrast: trained (ckpt) vs prompted (base) ----
        contrasts = {}
        nonvacuity = {}
        for metric in METRICS:
            arms_map = coverage_ladders if metric in COVERAGE_METRICS else run_ladders
            ckpt = {rid: arms_map[rid] for rid in by_arm["inquirer_trained"] if rid in arms_map}
            base = {rid: arms_map[rid] for rid in by_arm["inquirer_prompted"] if rid in arms_map}
            res = ladder_mod.symmetric_contrast(metric, ckpt, base, graphs, seed=0, n_boot=2000)

            # --- replicate the pairing loop only to expose per-task values for non-vacuity ---
            fn = ladder_mod.METRIC_FNS[metric]
            pairs = ladder_mod._pair_keys(ckpt, base)
            per_task_a, per_task_b = {}, {}
            for (_suite, task), (ck_ids, ba_ids) in pairs.items():
                a_vals, b_vals = [], []
                graph = graphs.get(task)
                if graph is None:
                    continue
                for c_rid in ck_ids:
                    c_lad = ckpt[c_rid]
                    for b_rid in ba_ids:
                        b_lad = base[b_rid]
                        k_common = min(c_lad.n_asks, b_lad.n_asks)
                        a_val = fn(c_lad.at(k_common), graph)
                        b_val = fn(b_lad.at(k_common), graph)
                        if math.isnan(a_val) or math.isnan(b_val):
                            continue
                        a_vals.append(a_val)
                        b_vals.append(b_val)
                if a_vals:
                    per_task_a[task] = sum(a_vals) / len(a_vals)
                    per_task_b[task] = sum(b_vals) / len(b_vals)

            n = len(per_task_a)
            distinct_vals = sorted(set(per_task_a.values()) | set(per_task_b.values()))
            wins = sum(1 for t in per_task_a if per_task_a[t] > per_task_b[t])
            losses = sum(1 for t in per_task_a if per_task_a[t] < per_task_b[t])
            ties = sum(1 for t in per_task_a if per_task_a[t] == per_task_b[t])
            withheld = n < 10
            contrasts[metric] = {
                "delta": None if withheld else res.delta,
                "ci_lo": None if withheld else res.ci_lo,
                "ci_hi": None if withheld else res.ci_hi,
                "n": res.n,
                "n_pairs_here": n,
                "withheld_interval": withheld,
                "sign_count": {"trained_gt": wins, "trained_lt": losses, "tie": ties},
            }
            nonvacuity[metric] = {
                "n_tasks_informative": n,
                "n_distinct_values": len(distinct_vals),
                "discordant": wins + losses,
                "ties": ties,
            }

    # ---- corner-policy levels (context, not the contrast) ----
    con = duckdb.connect()
    scores_df = con.execute(
        f"SELECT run_id, metric_name, value FROM read_parquet('{(parquet_dir / 'scores.parquet').as_posix()}')"
    ).fetchdf()
    corner_levels = {}
    for arm in ["shallow_wide", "deep_narrow", "inquirer_prompted", "inquirer_trained"]:
        ids = set(by_arm[arm])
        sub = runs[runs["arm_id"] == arm]
        n_asks_mean = float(sub["n_asks"].mean()) if len(sub) else float("nan")
        levels = {}
        for metric in ["max_depth_reached", "dwr", "precedence_violation_rate"]:
            vals = scores_df.loc[
                (scores_df["run_id"].isin(ids)) & (scores_df["metric_name"] == metric), "value"
            ]
            levels[metric] = float(vals.mean()) if len(vals) else None
        # facet_breadth_scorer and evidence_coverage aren't in scores.parquet under DIAGNOSTIC name;
        # read at full-k from the ladder dicts directly instead.
        arms_map = run_ladders
        fb_fn = None
        with sweep_mod.registered_metrics(
            {**sweep_mod.EXTRA_METRIC_FNS, **sweep_mod.DIAGNOSTIC_METRIC_FNS}
        ):
            fb_fn = ladder_mod.METRIC_FNS["facet_breadth_scorer"]
            fb_vals = []
            for rid in ids:
                lad = run_ladders.get(rid)
                if lad is None:
                    continue
                g = graphs.get(lad.task_id)
                if g is None:
                    continue
                fb_vals.append(fb_fn(lad.at(lad.n_asks), g))
            ec_fn = ladder_mod.METRIC_FNS["evidence_coverage"]
            ec_vals = []
            for rid in ids:
                lad = coverage_ladders.get(rid)
                if lad is None:
                    continue
                g = graphs.get(lad.task_id)
                if g is None:
                    continue
                ec_vals.append(ec_fn(lad.at(lad.n_asks), g))
        levels["facet_breadth_scorer"] = sum(fb_vals) / len(fb_vals) if fb_vals else None
        levels["evidence_coverage"] = sum(ec_vals) / len(ec_vals) if ec_vals else None
        corner_levels[arm] = {"n_asks_mean": n_asks_mean, "n_runs": len(ids), **levels}

    return {
        "shape": shape,
        "n_runs_total": len(all_ids),
        "n_runs_by_arm": {a: len(v) for a, v in by_arm.items()},
        "provenance": {
            "code_version": str(runs["code_version"].iloc[0]) if len(runs) else None,
            "corpus_hash": str(runs["corpus_hash"].iloc[0]) if len(runs) else None,
            "model_pin_hash_by_arm": {
                a: sorted(set(runs.loc[runs["arm_id"] == a, "model_pin_hash"])) for a in by_arm
            },
            "dev_runs": int(runs["is_dev_run"].sum()),
            "dirty_runs": int(runs["dirty"].sum()),
        },
        "instrument_lock": instrument,
        "contrasts": contrasts,
        "nonvacuity": nonvacuity,
        "corner_levels": corner_levels,
    }


if __name__ == "__main__":
    shape = sys.argv[1]
    root_dir = Path(sys.argv[2])
    parquet_dir = Path(sys.argv[3])
    out = analyze(shape, root_dir, parquet_dir)
    print(json.dumps(out, indent=2, default=str))
