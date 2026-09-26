"""The fixed-budget (cap 8) coverage contrast and the question-count distribution, per training seed.

Each arm read at its own stop, no matching rule: per task, the mean evidence_coverage over that
arm's runs (rollout seeds averaged), paired with the completed cohort's prompted runs, bootstrap over
tasks (paired_difference, 10,000 resamples, seed 0). Seed 0 (the completed cohort's trained runs)
must reproduce three.json's cap8_coverage_delta to 1e-9 first. Also the share of runs finishing in
under three questions and the mean question count, per arm, from runs.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "artifacts/completed_cohort_20260922"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919"
SUITES = ("musique", "strategyqa", "wiki2")


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def per_task(con, store: Path, ids: list[str]) -> dict[str, float]:
    return {
        t: float(v)
        for t, v in con.execute(
            f"SELECT r.task_id, avg(s.value) FROM read_parquet('{(store / 'scores.parquet').as_posix()}') s "
            f"JOIN read_parquet('{(store / 'runs.parquet').as_posix()}') r USING (run_id) "
            "WHERE s.metric_name = 'evidence_coverage' AND s.run_id IN (SELECT unnest(?)) GROUP BY 1",
            [ids],
        ).fetchall()
    }


def asks(con, store: Path, ids: list[str]) -> dict[str, float]:
    n, mean, lt3 = con.execute(
        f"SELECT count(*), avg(n_asks), avg((n_asks < 3)::int) FROM read_parquet('{(store / 'runs.parquet').as_posix()}') "
        "WHERE run_id IN (SELECT unnest(?))",
        [ids],
    ).fetchone()
    return {"n_runs": int(n), "mean_n_asks": float(mean), "share_under_3": float(lt3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    pub = json.loads((REPO / "artifacts/baseline_completion_20260920/three.json").read_text())[
        "complete"
    ]["by_suite"]
    out: dict = {"cells": {}, "asks": {}, "lock": {}}
    for suite in SUITES:
        pr_ids = _ids(COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
        pr = per_task(con, COHORT / "scores_parquet", pr_ids)
        out["asks"].setdefault("prompted", {})[suite] = asks(con, COHORT / "scores_parquet", pr_ids)
        s1 = _ids(SEEDREP / "run_ids" / f"run_ids.qwen3-8b-dpo-stacked-notdone-both-s1.{suite}.txt")
        s2 = _ids(SEEDREP / "run_ids" / f"run_ids.qwen3-8b-dpo-stacked-notdone-both-s2.{suite}.txt")
        arms = {
            "s0": (
                COHORT / "scores_parquet",
                _ids(COHORT / "cohort" / f"run_ids.trained.{suite}.txt"),
            ),
            "s1": (SEEDREP / "scores_parquet", s1),
            "s2": (SEEDREP / "scores_parquet", s2),
            "s1s2": (SEEDREP / "scores_parquet", s1 + s2),
        }
        for k, (store, ids) in arms.items():
            a = per_task(con, store, ids)
            shared = sorted(set(a) & set(pr))
            est = paired_difference(
                {t: a[t] for t in shared}, {t: pr[t] for t in shared}, n_boot=10000, seed=0
            )
            out["cells"].setdefault(k, {})[suite] = {
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                # both arms' levels over the same paired tasks (added 2026-09-23 for the headline
                # table; delta, interval and n are unchanged by the addition)
                "level_trained": sum(a[t] for t in shared) / len(shared),
                "level_prompted": sum(pr[t] for t in shared) / len(shared),
            }
            out["asks"].setdefault(k, {})[suite] = asks(con, store, ids)
        d0 = out["cells"]["s0"][suite]["delta"]
        ok = abs(d0 - pub[suite]["cap8_coverage_delta"]) <= 1e-9
        out["lock"][suite] = {
            "published": pub[suite]["cap8_coverage_delta"],
            "reproduced": d0,
            "ok": ok,
        }
        if not ok:
            raise SystemExit(f"LOCK FAILED {suite}: {d0} vs {pub[suite]['cap8_coverage_delta']}")
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
