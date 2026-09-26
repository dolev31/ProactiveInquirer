"""Per-arm task-level levels for the paper's tool-use figure, from the rerun's records.json.

The rerun's reader (scripts/tau2_concordance/transfer_rerun.py at 57eefed0) prints paired differences
but no per-arm levels. This recomputes, per suite x prompt variant, the task-level success rate and the
mean questions per dialogue of the prompted comparator and of the trained questioner (training seeds 1
and 2 averaged within the task), with a task-clustered BCa interval on each level.

LOCK FIRST: the trained-minus-comparator task-level success and follow-up-turn differences recomputed
here must equal the reader's eight pooled points in final/transfer_rerun.json to 1e-9, and the input
must hash to the records.json the reader read (SHA256SUMS).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from pi_eval.stats.inference import cluster_bootstrap

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "artifacts/tau2_rerun_20260923/final"
RECORDS_SHA256 = "63ff31a73c930ccb08f2551ed3ee2cd0e75423e2c1712eb3d71f8a15573c8059"
COMPARATOR = "qwen3-8b-base"
SEEDS = {"qwen3-8b-dpo-stacked-notdone-both-s1": "s1", "qwen3-8b-dpo-stacked-notdone-both-s2": "s2"}


def _arm(rec: dict) -> str:
    m = rec["_manifest"]["pins"]["inquirer"]["model_id"]
    if m == COMPARATOR:
        return "comparator"
    return SEEDS[m]


def _variant(rec: dict) -> str:
    return rec["_manifest"]["prompt_variant_id"].split("-")[0]


def _success(rec: dict) -> float:
    n = rec["native"]
    n = n if isinstance(n, dict) else ast.literal_eval(n)
    return float(n.get("tau_reward", 0.0))


def per_task(records: list[dict]) -> dict:
    """(suite, variant) -> arm -> task -> (success mean, asks mean, follow-up turns mean)."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        fu = int(r["n_user_turns"]) - int(r["n_prefix_user_turns"])
        acc[(r["suite_id"], _variant(r))][_arm(r)][r["task_id"]].append(
            (_success(r), float(r["n_asks"]), float(fu))
        )
    out: dict = {}
    for cell, arms in acc.items():
        out[cell] = {
            arm: {
                t: tuple(statistics.fmean(v[i] for v in vals) for i in range(3))
                for t, vals in tasks.items()
            }
            for arm, tasks in arms.items()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--records", type=Path, default=FINAL / "records.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    raw = args.records.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECORDS_SHA256:
        raise SystemExit(f"records.json sha256 {digest} is not the reader's {RECORDS_SHA256}")
    cells = per_task(json.loads(raw))
    reader = json.loads((FINAL / "transfer_rerun.json").read_text())
    out: dict = {"records_sha256": digest, "cells": {}, "lock": {}}
    for p in reader["pooled"]:
        key = (p["suite"], p["variant"])
        c = cells[key]
        tasks = sorted(c["comparator"])
        trained = {t: [(c["s1"][t][i] + c["s2"][t][i]) / 2 for i in range(3)] for t in tasks}
        d_succ = statistics.fmean(trained[t][0] - c["comparator"][t][0] for t in tasks)
        d_fu = statistics.fmean(trained[t][2] - c["comparator"][t][2] for t in tasks)
        ok = (
            abs(d_succ - p["guard"]["task_level"]["mean_delta"]) <= 1e-9
            and abs(d_fu - p["task_level"]["mean_delta"]) <= 1e-9
        )
        out["lock"][f"{key[0]}::{key[1]}"] = {
            "success_delta": [d_succ, p["guard"]["task_level"]["mean_delta"]],
            "followup_delta": [d_fu, p["task_level"]["mean_delta"]],
            "ok": ok,
        }
        if not ok:
            raise SystemExit(f"LOCK FAILED {key}")
        levels = {}
        for arm, vals in (
            ("comparator", {t: list(c["comparator"][t]) for t in tasks}),
            ("trained", trained),
        ):
            for i, name in ((0, "success"), (1, "asks")):
                units = [[vals[t][i]] for t in tasks]
                point, lo, hi = cluster_bootstrap(units, n_boot=10_000, seed=0)
                levels[f"{arm}_{name}"] = {
                    "point": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_tasks": len(tasks),
                }
        out["cells"][f"{key[0]}::{key[1]}"] = {
            "levels": levels,
            "success_delta": {
                "point": p["guard"]["task_level"]["mean_delta"],
                "ci_lo": p["guard"]["task_level"]["printed_interval"]["lo"],
                "ci_hi": p["guard"]["task_level"]["printed_interval"]["hi"],
                "decided": p["guard"]["task_level"]["decided"],
                "split": p["guard"]["task_level"]["split"],
            },
        }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k, v in out["cells"].items():
        lv = v["levels"]
        print(
            f"{k}: success {lv['comparator_success']['point']:.3f} -> {lv['trained_success']['point']:.3f}; "
            f"asks {lv['comparator_asks']['point']:.2f} -> {lv['trained_asks']['point']:.2f}"
        )
    print(
        f"wrote {args.out}; lock ok on {sum(v['ok'] for v in out['lock'].values())}/{len(out['lock'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
