"""Resample stability of the two-axis contrasts, for seed 0 and for the recipe.

analyze_shape.py reads every contrast at 2,000 resamples (its symmetric_contrast call passes
n_boot=2000; artifacts/two_axis_matched_20260922/RESULT.md says 10,000, which the code does not do).
This re-reads each contrast with the same ladders and the same contrast function at 10,000 resamples,
seed 0, and at 50,000 under seeds 101/202/303, the paper's decision rule: a cell is DECIDED only if
every 50,000-resample interval excludes zero.

LOCK: at n_boot=2000, seed 0, every contrast must equal its analysis record to 1e-12 before any other
reading is written. Seed 0 reads the published store under /tmp/two_axis_matched_20260922; the recipe
readings read the farm stores under artifacts/two_axis_recipe_20260923.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKTREE = os.environ["PINQ_TWO_AXIS_WORKTREE"]
sys.path.insert(0, f"{WORKTREE}/src")
sys.path.insert(0, f"{WORKTREE}/scripts")

import duckdb  # noqa: E402
from plan_metrics_symmetric import ladder as ladder_mod  # noqa: E402
from plan_metrics_symmetric import sweep as sweep_mod  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = Path("/tmp/two_axis_matched_20260922")
RECIPE = REPO / "artifacts" / "two_axis_recipe_20260923"
PUBLISHED = REPO / "artifacts" / "two_axis_matched_20260922"
METRICS = ("evidence_coverage", "facet_breadth_scorer", "max_depth_reached", "dwr")
COVERAGE = {"evidence_coverage"}
READINGS = (("s0", None), ("s1", "s1"), ("s2", "s2"), ("rec", "rec"))


def _read(shape: str, store: Path) -> dict:
    os.environ["PI_GOLD_ROOT"] = str(CAMPAIGN / "roots" / shape / "data" / "gold")
    from pi_eval.gold import load_graphs

    graphs = load_graphs("synth", "v1")
    con = duckdb.connect()
    runs = con.execute(
        f"SELECT run_id, arm_id FROM read_parquet('{(store / 'runs.parquet').as_posix()}')"
    ).fetchall()
    by_arm: dict[str, list[str]] = {}
    for rid, arm in runs:
        by_arm.setdefault(arm, []).append(rid)
    ids = [r for r, _ in runs]
    run_l = ladder_mod.load_run_ladders(store, ids)
    cov_l = sweep_mod.load_coverage_ladders(store, ids, graphs)
    return {"graphs": graphs, "by_arm": by_arm, "run": run_l, "cov": cov_l}


def contrast(d: dict, metric: str, n_boot: int, seed: int) -> dict:
    lad = d["cov"] if metric in COVERAGE else d["run"]
    a = {r: lad[r] for r in d["by_arm"]["inquirer_trained"] if r in lad}
    b = {r: lad[r] for r in d["by_arm"]["inquirer_prompted"] if r in lad}
    res = ladder_mod.symmetric_contrast(metric, a, b, d["graphs"], seed=seed, n_boot=n_boot)
    return {"delta": res.delta, "ci_lo": res.ci_lo, "ci_hi": res.ci_hi, "n": res.n}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out: dict = {"lock": {}, "cells": {}}
    bad = []
    with sweep_mod.registered_metrics(
        {**sweep_mod.EXTRA_METRIC_FNS, **sweep_mod.DIAGNOSTIC_METRIC_FNS}
    ):
        for shape in ("2x4", "3x6", "4x8"):
            for lab, reading in READINGS:
                if reading is None:
                    store = CAMPAIGN / "roots" / shape / "scores_parquet"
                    rec = json.loads((PUBLISHED / f"analysis_{shape}.json").read_text())
                else:
                    store = RECIPE / shape / f"store_{reading}"
                    rec = json.loads((RECIPE / shape / f"analysis_{reading}.json").read_text())
                d = _read(shape, store)
                for m in METRICS:
                    key = f"{shape}/{lab}/{m}"
                    c2k = contrast(d, m, 2000, 0)
                    want = rec["contrasts"][m]
                    ok = all(abs(c2k[k] - want[k]) <= 1e-12 for k in ("delta", "ci_lo", "ci_hi"))
                    out["lock"][key] = {
                        "ok": ok,
                        "got": c2k,
                        "want": {k: want[k] for k in ("delta", "ci_lo", "ci_hi")},
                    }
                    if not ok:
                        bad.append(key)
                        continue
                    cell = {"n2000_s0": c2k, "n10000_s0": contrast(d, m, 10_000, 0)}
                    for s in (101, 202, 303):
                        cell[f"n50000_s{s}"] = contrast(d, m, 50_000, s)
                    fifty = [cell[f"n50000_s{s}"] for s in (101, 202, 303)]
                    cell["decided"] = all(c["ci_lo"] > 0 for c in fifty) or all(
                        c["ci_hi"] < 0 for c in fifty
                    )
                    out["cells"][key] = cell
                    print(
                        f"{key:34s} {cell['n10000_s0']['delta']:+.4f} "
                        f"[{cell['n10000_s0']['ci_lo']:+.4f},{cell['n10000_s0']['ci_hi']:+.4f}] "
                        f"50k lo {min(c['ci_lo'] for c in fifty):+.4f} hi {max(c['ci_hi'] for c in fifty):+.4f} "
                        f"decided={cell['decided']}",
                        flush=True,
                    )
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; lock failures: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
