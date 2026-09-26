"""Breadth on the one natural suite that branches: 2WikiMultiHopQA's held-out tasks with two or more
independent lines of need, recipe (seeds 1 and 2, averaged within the task) minus the same weights
prompted, under Table 1's symmetric seed-matched rule.

WHY. On MuSiQue and StrategyQA every held-out graph has exactly one line of need, so breadth there
equals "resolved anything past the surface". 2WikiMultiHopQA is the only suite where a task can have
several independent lines, so it is the only natural data on which breadth can differ from depth.

HOW. The same function Table 1's recipe row uses (`table1_by_seed.pooled_symmetric`), with the graph
dict restricted to tasks with len(gold_facets) >= 2; a task absent from the dict is skipped by that
function, so the rule, pairing and bootstrap are unchanged. LOCK FIRST: with the unrestricted graphs
the same call reproduces table1_by_seed.json's recipe cell for the same metric exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))

import table1_by_seed as t1  # noqa: E402

rc = t1.rc
METRICS = ("facet_breadth_scorer", "evidence_coverage", "dwr")
RESAMPLES = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from pi_eval.gold import load_graphs

    rec = json.loads((REPO / "artifacts/seed_identity_20260923/table1_by_seed.json").read_text())
    seeds = {
        **rc._seed_map(t1.COHORT / "scores_parquet"),
        **rc._seed_map(t1.SEEDREP / "scores_parquet"),
    }
    suite = "wiki2"
    graphs = load_graphs(suite, "v1")
    prompted = rc._ids(t1.COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
    out: dict = {"lock": {}, "cells": {}, "census": {}}
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        base = rc._ladders(t1.COHORT / "scores_parquet", prompted, graphs)
        arms = [
            rc._ladders(
                t1.SEEDREP / "scores_parquet",
                rc._ids(t1.SEEDREP / "run_ids" / f"run_ids.{t1.S_NAMES[k]}.{suite}.txt"),
                graphs,
            )
            for k in ("s1", "s2")
        ]
        tasks = {lad.task_id for lad in base[0].values()}
        multi = {t: g for t, g in graphs.items() if t in tasks and len(g.gold_facets) >= 2}
        out["census"] = {
            "held_out_tasks": len(tasks),
            "multi_facet_tasks": len(multi),
            "facet_counts": {
                str(n): sum(1 for t in tasks if len(graphs[t].gold_facets) == n)
                for n in sorted({len(graphs[t].gold_facets) for t in tasks})
            },
        }
        for metric in METRICS:
            idx = 1 if metric in rc.COVERAGE_METRICS else 0
            a = [arm[idx] for arm in arms]
            b = base[idx]
            full = t1.pooled_symmetric(metric, a, b, graphs, seeds, seed=0, n_boot=10000)
            pub = next(
                r for r in rec["cells"]["s1s2"][f"{metric}::{suite}"] if r["n_boot"] == 10000
            )
            ok = abs(full["delta"] - pub["delta"]) <= 1e-12 and full["n"] == pub["n"]
            out["lock"][metric] = {"published": pub["delta"], "reproduced": full["delta"], "ok": ok}
            if not ok:
                raise SystemExit(f"LOCK FAILED {metric}: {full['delta']} vs {pub['delta']}")
            out["cells"][metric] = [
                {
                    "n_boot": nb,
                    "seed": sd,
                    **t1.pooled_symmetric(metric, a, b, multi, seeds, seed=sd, n_boot=nb),
                }
                for nb, sd in RESAMPLES
            ]
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["census"]))
    for m, rows in out["cells"].items():
        ten = rows[0]
        dec = all(r["ci_lo"] > 0 for r in rows[1:]) or all(r["ci_hi"] < 0 for r in rows[1:])
        print(
            f"{m}: {ten['delta']:+.4f} [{ten['ci_lo']:+.4f},{ten['ci_hi']:+.4f}] n={ten['n']}"
            f" 50k lo {[round(r['ci_lo'], 4) for r in rows[1:]]} decided={dec}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
