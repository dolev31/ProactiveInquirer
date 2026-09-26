"""The depth-one control on the depth quantities themselves, for each seed against its own ablation.

scripts/depth1_contrast_20260918.py read seed 0 against a depth-one control that runs seed 0's weights:
at the shared cap (unmatched), per-task means, paired_difference with clusters on template_id. This
reads the same three quantities (max_depth_reached, dwr, cad_ge2) for seeds 1 and 2 against depth-one
controls that run THEIR weights, then pools the two seeds within the task.

LOCK: seed 0 (artifacts/depth1_contrast_20260918 run-id lists, on the rebuilt controls store at the
published scorer hash) must reproduce the published multi-hop cells +0.2295 / +0.1401 / +0.1113 to
5e-4 before any seed-1/2 cell is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import paired_difference
from pinq_train.gate import cad_ge2_by_run

REPO = Path(__file__).resolve().parents[2]
METRICS = ("max_depth_reached", "dwr", "cad_ge2")
PUBLISHED_S0 = {"max_depth_reached": 0.2295, "dwr": 0.1401, "cad_ge2": 0.1113}


def _ids(p: Path) -> set[str]:
    return {
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    }


def per_task(store: Path, ids: set[str], suite: str) -> dict[str, dict[str, list[float]]]:
    con = duckdb.connect()
    sh = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT scorer_hash FROM read_parquet('{store}/scores.parquet')"
        ).fetchall()
    ]
    assert len(sh) == 1, sh
    runs = {
        r[0]: (r[1], r[2])
        for r in con.execute(
            f"SELECT r.run_id, r.task_id, COALESCE(NULLIF(r.template_id, ''), r.task_id) "
            f"FROM read_parquet('{store}/runs.parquet') r WHERE {ELIGIBLE} AND r.suite_id = ?",
            [suite],
        ).fetchall()
        if r[0] in ids
    }
    vals: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    for rid, name, v in con.execute(
        f"SELECT run_id, metric_name, value FROM read_parquet('{store}/scores.parquet') "
        "WHERE metric_name IN ('max_depth_reached', 'dwr')"
    ).fetchall():
        if rid in runs and v is not None:
            vals[name][rid] = float(v)
    vals["cad_ge2"] = {
        r: v for r, v in cad_ge2_by_run(store, scorer_hash=sh[0]).items() if r in runs
    }
    out: dict[str, dict[str, list[float]]] = {m: {} for m in METRICS}
    clusters: dict[str, str] = {}
    for m in METRICS:
        for rid, v in vals[m].items():
            task, tmpl = runs[rid]
            out[m].setdefault(task, []).append(v)
            clusters[task] = tmpl
    out["_clusters"] = clusters  # type: ignore[assignment]
    return out


def contrast(parts, metric):
    a_all: dict[str, list[float]] = {}
    b_all: dict[str, list[float]] = {}
    cl: dict[str, str] = {}
    for a, b in parts:
        for t in set(a[metric]) & set(b[metric]):
            a_all.setdefault(t, []).append(sum(a[metric][t]) / len(a[metric][t]))
            b_all.setdefault(t, []).append(sum(b[metric][t]) / len(b[metric][t]))
            cl[t] = a["_clusters"].get(t, t)
    keys = sorted(a_all)
    aa = {k: sum(a_all[k]) / len(a_all[k]) for k in keys}
    bb = {k: sum(b_all[k]) / len(b_all[k]) for k in keys}
    e = paired_difference(
        aa, bb, clusters={k: cl[k] for k in keys}, n_boot=1000, n_perm=10000, seed=0
    )
    return {"delta": e.point, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi, "n": e.n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    obase = REPO / "artifacts/structured_baselines_clean_20260922"
    sbase = REPO / "artifacts/seed_identity_20260923"
    ks = REPO / "artifacts/depth1_contrast_20260918"
    cstore = sbase / "controls_store"
    out: dict = {"lock": {}, "cells": {}}
    for suite in ("musique", "strategyqa"):
        s0 = (
            per_task(cstore, _ids(ks / "run_ids.trained.txt"), suite),
            per_task(cstore, _ids(ks / "run_ids.depth1.txt"), suite),
        )
        con = duckdb.connect()

        def arm_ids(store: Path, arm: str) -> set[str]:
            return {
                r[0]
                for r in con.execute(
                    f"SELECT run_id FROM read_parquet('{store}/runs.parquet') WHERE arm_id = ?",
                    [arm],
                ).fetchall()
            }

        seeds = []
        for s in ("s1", "s2"):
            t_store, k_store = obase / f"scores_parquet_{s}", sbase / f"ks_store_{s}"
            seeds.append(
                (
                    per_task(t_store, arm_ids(t_store, "inquirer_trained"), suite),
                    per_task(k_store, arm_ids(k_store, "inquirer_depth1"), suite),
                )
            )
        for m in METRICS:
            c0 = contrast([s0], m)
            out["cells"].setdefault("s0", {})[f"{m}::{suite}"] = c0
            out["cells"].setdefault("s1", {})[f"{m}::{suite}"] = contrast([seeds[0]], m)
            out["cells"].setdefault("s2", {})[f"{m}::{suite}"] = contrast([seeds[1]], m)
            out["cells"].setdefault("pooled", {})[f"{m}::{suite}"] = contrast(seeds, m)
            if suite == "musique":
                ok = abs(c0["delta"] - PUBLISHED_S0[m]) <= 5e-4
                out["lock"][m] = {"published": PUBLISHED_S0[m], "reproduced": c0["delta"], "ok": ok}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    bad = [m for m, v in out["lock"].items() if not v["ok"]]
    print(f"wrote {args.out}; lock failures: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
