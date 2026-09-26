"""Tokens per useful need, trained minus prompted, per training seed. No matching rule enters.

Per arm, each task's value is the mean of `tokens_per_useful_need` over that arm's runs of the task
that carry one (the scorer emits none where a run found no required need). Tasks present in both
arms are paired, and the paired bootstrap is over tasks (`pi_eval.stats.inference.
paired_difference`, 10,000 resamples, seed 0). Store: the seed-replicate store (scorer e82c7458),
which holds s0, s1, s2 and the prompted comparator runs of artifacts/testsplit_qa. Seed 0 must
reproduce the paper's paired points (-11,036.8 / -8,838.6 / -2,189.2) to 0.1 first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "artifacts/seedrep_gate_20260919/scores_parquet"
SUITES = ("musique", "strategyqa", "wiki2")
PUBLISHED_S0 = {"musique": -11036.8, "strategyqa": -8838.6, "wiki2": -2189.2}
METRIC = "tokens_per_useful_need"


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def per_task(con, ids: list[str]) -> dict[str, float]:
    rows = con.execute(
        f"SELECT r.task_id, avg(s.value) FROM read_parquet('{(STORE / 'scores.parquet').as_posix()}') s "
        f"JOIN read_parquet('{(STORE / 'runs.parquet').as_posix()}') r USING (run_id) "
        f"WHERE s.metric_name = ? AND s.value IS NOT NULL AND NOT isnan(s.value) "
        f"AND s.run_id IN (SELECT unnest(?)) GROUP BY 1",
        [METRIC, ids],
    ).fetchall()
    return {t: float(v) for t, v in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    out: dict = {"store": str(STORE.relative_to(REPO)), "metric": METRIC, "cells": {}, "lock": {}}
    for suite in SUITES:
        pr = per_task(con, _ids(REPO / "artifacts/testsplit_qa" / f"run_ids.prompted.{suite}.txt"))
        arms = {
            "s0": _ids(REPO / "artifacts/testsplit_qa" / f"run_ids.trained.{suite}.txt"),
            "s1": _ids(
                REPO
                / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s1.{suite}.txt"
            ),
            "s2": _ids(
                REPO
                / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s2.{suite}.txt"
            ),
        }
        per = {k: per_task(con, v) for k, v in arms.items()}
        # the full recipe: both fresh seeds' runs averaged into the task
        per["s1s2"] = per_task(con, arms["s1"] + arms["s2"])
        for k, a in per.items():
            shared = sorted(set(a) & set(pr))
            est = paired_difference(
                {t: a[t] for t in shared}, {t: pr[t] for t in shared}, n_boot=10000, seed=0
            )
            out["cells"].setdefault(k, {})[suite] = {
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                "level_trained": sum(a[t] for t in shared) / len(shared),
                "level_prompted": sum(pr[t] for t in shared) / len(shared),
            }
        d0 = out["cells"]["s0"][suite]["delta"]
        ok = abs(d0 - PUBLISHED_S0[suite]) <= 0.1
        out["lock"][suite] = {"published": PUBLISHED_S0[suite], "reproduced": d0, "ok": ok}
        if not ok:
            raise SystemExit(f"LOCK FAILED {suite}: {d0} vs {PUBLISHED_S0[suite]}")
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
