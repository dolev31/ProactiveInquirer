"""Trained minus the prompted teacher (gpt-oss-120b), per training seed, both arms at the lower count.

The paper's "fifteen times its size" sentence rests on this contrast for seed 0 only
(artifacts/symmetric_matched_cost_20260919/RESULT.md:351-355: +0.0808 / +0.0310 / -0.0294). This
reads it for the recipe's two fresh seeds on ONE isolated store holding all four populations
(artifacts/seed_identity_20260923/teacher_store, scorer e82c7458, the published record's hash):
teacher = artifacts/n6/run_ids.inquirer_prompted.txt, s0 = artifacts/testsplit_qa trained ids,
s1/s2 = artifacts/seedrep_gate_20260919/run_ids. Seed 0 must reproduce the published points to
1e-4 (the record prints four decimals) before anything else is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "structured_baselines"))
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))

import read_contrasts as rc  # noqa: E402
from table1_by_seed import POOL_RESAMPLES, pooled_symmetric  # noqa: E402

SUITES = ("musique", "strategyqa", "wiki2")
STORE = REPO / "artifacts/seed_identity_20260923/teacher_store"
PUBLISHED_S0 = {"musique": 0.0808, "strategyqa": 0.0310, "wiki2": -0.0294}
SEEDS = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}


def _ids(path: Path) -> list[str]:
    return [
        ln.split()[0]
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from pi_eval.gold import load_graphs

    seeds = rc._seed_map(STORE)
    import duckdb

    suite_of = dict(
        duckdb.connect()
        .execute(
            f"SELECT run_id, suite_id FROM read_parquet('{(STORE / 'runs.parquet').as_posix()}')"
        )
        .fetchall()
    )
    teacher_all = _ids(REPO / "artifacts/n6/run_ids.inquirer_prompted.txt")
    out: dict = {"store": str(STORE.relative_to(REPO)), "cells": {}, "lock": {}, "arms": {}}
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        for suite in SUITES:
            graphs = load_graphs(suite, "v1")
            teacher = [r for r in teacher_all if suite_of.get(r) == suite]
            arms = {
                "s0": _ids(REPO / "artifacts/testsplit_qa" / f"run_ids.trained.{suite}.txt"),
                **{
                    k: _ids(
                        REPO
                        / "artifacts/seedrep_gate_20260919/run_ids"
                        / f"run_ids.{v}.{suite}.txt"
                    )
                    for k, v in SEEDS.items()
                },
            }
            _, t_cov = rc._ladders(STORE, teacher, graphs)
            out["arms"].setdefault("teacher", {})[suite] = {
                "n_runs": len(teacher),
                "mean_n_asks": sum(x.n_asks for x in t_cov.values()) / len(t_cov),
            }
            cov = {}
            for k, ids in arms.items():
                _, cov[k] = rc._ladders(STORE, ids, graphs)
                out["arms"].setdefault(k, {})[suite] = {
                    "n_runs": len(ids),
                    "mean_n_asks": sum(x.n_asks for x in cov[k].values()) / len(cov[k]),
                }
                got = rc.seed_matched_symmetric(
                    "evidence_coverage", cov[k], t_cov, graphs, seeds, seed=0, n_boot=10000
                )
                out["cells"].setdefault(k, {})[suite] = got
            d0 = out["cells"]["s0"][suite]["delta"]
            ok = abs(d0 - PUBLISHED_S0[suite]) <= 1e-4
            out["lock"][suite] = {"published": PUBLISHED_S0[suite], "reproduced": d0, "ok": ok}
            if not ok:
                raise SystemExit(f"LOCK FAILED {suite}: {d0} vs {PUBLISHED_S0[suite]}")
            out["cells"].setdefault("s1s2", {})[suite] = [
                {
                    "n_boot": nb,
                    "seed": sd,
                    **pooled_symmetric(
                        "evidence_coverage",
                        [cov["s1"], cov["s2"]],
                        t_cov,
                        graphs,
                        seeds,
                        seed=sd,
                        n_boot=nb,
                    ),
                }
                for nb, sd in POOL_RESAMPLES
            ]
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
