"""Trained minus the 120B teacher with each arm at its own stop under the cap of eight, coverage,
per training seed. No matching rule. Store: artifacts/seed_identity_20260923/teacher_store. Seed 0
must reproduce the printed cells first: +0.02521 / -0.12393 / -0.04500, to 5e-5.

Seeds 1 and 2 are also read alone (added 2026-09-23 for the per-seed appendix table; the s0 and
s1s2 cells are unchanged by the addition, which the paper lane can check by re-running).

task_success is deliberately NOT read here: this per-task mean gave 0.0125 for seed 0 on MuSiQue
against the printed +0.0081 (scripts/measure_20260919/items_1_2_3.py), so it fails the lock and an
unlocked reader does not produce paper numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "artifacts/seed_identity_20260923/teacher_store"
SUITES = ("musique", "strategyqa", "wiki2")
PUB = {
    "evidence_coverage": {"musique": 0.02521, "strategyqa": -0.12393, "wiki2": -0.04500},
}


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def per_task(con, ids, metric):
    return {
        t: float(v)
        for t, v in con.execute(
            f"SELECT r.task_id, avg(s.value) FROM read_parquet('{(STORE / 'scores.parquet').as_posix()}') s "
            f"JOIN read_parquet('{(STORE / 'runs.parquet').as_posix()}') r USING (run_id) "
            "WHERE s.metric_name = ? AND s.value IS NOT NULL AND NOT isnan(s.value) "
            "AND s.run_id IN (SELECT unnest(?)) GROUP BY 1",
            [metric, ids],
        ).fetchall()
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    suite_of = dict(
        con.execute(
            f"SELECT run_id, suite_id FROM read_parquet('{(STORE / 'runs.parquet').as_posix()}')"
        ).fetchall()
    )
    teacher_all = _ids(REPO / "artifacts/n6/run_ids.inquirer_prompted.txt")
    out: dict = {"store": str(STORE.relative_to(REPO)), "cells": {}, "lock": {}}
    for suite in SUITES:
        teacher = [r for r in teacher_all if suite_of.get(r) == suite]
        s1 = _ids(
            REPO
            / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s1.{suite}.txt"
        )
        s2 = _ids(
            REPO
            / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s2.{suite}.txt"
        )
        arms = {
            "s0": _ids(REPO / "artifacts/testsplit_qa" / f"run_ids.trained.{suite}.txt"),
            "s1": s1,
            "s2": s2,
            "s1s2": s1 + s2,
        }
        for metric in PUB:
            t = per_task(con, teacher, metric)
            for k, ids in arms.items():
                a = per_task(con, ids, metric)
                shared = sorted(set(a) & set(t))
                est = paired_difference(
                    {x: a[x] for x in shared}, {x: t[x] for x in shared}, n_boot=10000, seed=0
                )
                out["cells"].setdefault(k, {}).setdefault(metric, {})[suite] = {
                    "delta": est.point,
                    "ci_lo": est.ci_lo,
                    "ci_hi": est.ci_hi,
                    "n": est.n,
                    "level_trained": sum(a[x] for x in shared) / len(shared),
                    "level_teacher": sum(t[x] for x in shared) / len(shared),
                }
            d0 = out["cells"]["s0"][metric][suite]["delta"]
            ok = abs(d0 - PUB[metric][suite]) <= 5e-5
            out["lock"][f"{metric}::{suite}"] = {
                "published": PUB[metric][suite],
                "reproduced": d0,
                "ok": ok,
            }
            if not ok:
                raise SystemExit(f"LOCK FAILED {metric} {suite}: {d0} vs {PUB[metric][suite]}")
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
