"""Table 1's five rows for each training seed of the selected recipe, under Table 1's own rule.

WHY. The selected arm's registered adapter (seed 0, `qwen3-8b-dpo-stacked-notdone-both`,
adapter_sha 80a56b74...) is the only one of its three seeds whose final preference stage RESUMED
from checkpoint-1800 (`resumed_from` in its rung2.manifest.json; seeds 1 and 2 record None). Lane
S2 measured its final weights ~11x closer to the stage's initialization than its own step 1800,
i.e. the resume re-loaded the init and ran only the last ~168 of 1,968 steps. If so, seed 0 is
not a sample of the recipe the paper names, and every Table 1 cell must be read on the seeds
that are.

WHAT. For each suite and each of Table 1's five metrics, the symmetric seed-matched contrast
(`symmetric_completed.seed_matched_symmetric`: pairing (suite, task, rollout seed), both arms at
min(k_a, k_b), rollout seeds averaged into the task, paired bootstrap over tasks) of:
  s0  the completed cohort's trained runs (the paper's Table 1 population)
  s1, s2  the seed-replicate runs (artifacts/seedrep_gate_20260919)
each against THE SAME comparator runs, the completed cohort's prompted arm, n = 200 per suite.

LOCK FIRST. The s0 column must reproduce every Table 1 cell the paper prints from its three
records to 1e-9 (point estimates; the bootstrap here is the same seed and count as those records)
before any s1 or s2 cell is written. The comparator is the same run set in every column, so a
difference between columns is a difference between trained seeds and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "structured_baselines"))

import read_contrasts as rc  # noqa: E402  (one ladder module, the registered metrics, the rule)

SUITES = ("musique", "strategyqa", "wiki2")
METRICS = (
    "evidence_coverage",
    "facet_breadth_scorer",
    "dwr",
    "max_depth_reached",
    "precedence_violation_rate",
)
COHORT = REPO / "artifacts/completed_cohort_20260922"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919"
S_NAMES = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}


def published_cells() -> dict[str, float]:
    """The point estimates Table 1 prints, read from the three records the figure script reads."""
    cov = json.loads((REPO / "artifacts/baseline_completion_20260920/three.json").read_text())
    sym = json.loads(
        (REPO / "artifacts/plan_metrics_completed_20260920/symmetric_completed.json").read_text()
    )
    two = json.loads(
        (REPO / "artifacts/table1_completed_20260922/two_more_symmetric.json").read_text()
    )
    out: dict[str, float] = {}
    for s in SUITES:
        out[f"evidence_coverage::{s}"] = cov["complete"]["by_suite"][s]["symmetric"]["delta"]
        for m in ("dwr", "precedence_violation_rate"):
            out[f"{m}::{s}"] = next(
                x for x in sym["completed"][f"{m}::{s}"] if x["n_boot"] == 10000
            )["delta"]
        for m in ("facet_breadth_scorer", "max_depth_reached"):
            out[f"{m}::{s}"] = next(
                x for x in two["completed"][f"{m}::{s}"] if x["n_boot"] == 10000
            )["delta"]
    return out


def pooled_symmetric(metric, arms, base, graphs, seeds, *, seed=0, n_boot=10000):
    """Several TRAINING seeds of one recipe against one comparator, averaged WITHIN the task.

    For each training seed separately this is exactly `seed_matched_symmetric`'s pairing: key
    (suite, task, rollout seed), both arms at min(k_a, k_b). The per-pair values of every training
    seed then go into ONE per-task mean, and the bootstrap is over tasks. The interval therefore
    reflects task sampling only; two training seeds cannot estimate training-seed variance, which
    is why each seed is also reported alone. A key seen twice within one arm raises, as there.
    """
    import math

    from pi_eval.stats.inference import paired_difference

    fn = rc.ladder.METRIC_FNS[metric]

    def keyed(arm, what):
        out = {}
        for rid, lad in arm.items():
            key = (lad.suite_id, lad.task_id, int(seeds[rid]))
            if key in out:
                raise ValueError(f"{what}: two runs at {key}")
            out[key] = lad
        return out

    b_by = keyed(base, "comparator")
    acc: dict[str, tuple[list[float], list[float]]] = {}
    n_pairs = 0
    for i, arm in enumerate(arms):
        a_by = keyed(arm, f"training seed #{i}")
        for key in sorted(set(a_by) & set(b_by)):
            graph = graphs.get(key[1])
            if graph is None:
                continue
            k = min(a_by[key].n_asks, b_by[key].n_asks)
            av, bv = fn(a_by[key].at(k), graph), fn(b_by[key].at(k), graph)
            if math.isnan(av) or math.isnan(bv):
                continue
            la, lb = acc.setdefault(key[1], ([], []))
            la.append(av)
            lb.append(bv)
            n_pairs += 1
    per_a = {t: sum(a) / len(a) for t, (a, _) in acc.items()}
    per_b = {t: sum(b) / len(b) for t, (_, b) in acc.items()}
    est = paired_difference(per_a, per_b, n_boot=n_boot, seed=seed)
    # The two arms' LEVELS over the same paired tasks the delta is taken over (added 2026-09-23 for
    # the headline table; delta, interval, n and n_pairs are unchanged by the addition).
    shared = sorted(set(per_a) & set(per_b))
    return {
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "n": est.n,
        "n_pairs": n_pairs,
        "level_a": sum(per_a[t] for t in shared) / len(shared),
        "level_b": sum(per_b[t] for t in shared) / len(shared),
    }


POOL_RESAMPLES = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    from pi_eval.gold import load_graphs

    pub = published_cells()
    seeds = {**rc._seed_map(COHORT / "scores_parquet"), **rc._seed_map(SEEDREP / "scores_parquet")}
    result: dict = {
        "rule": "symmetric seed-matched, both arms at min(k_a, k_b), n_boot 10000 seed 0",
        "lock": {},
        "cells": {},
        "arms": {},
    }
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        for suite in SUITES:
            graphs = load_graphs(suite, "v1")
            prompted = rc._ids(COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
            arms_ids = {
                "s0": (
                    COHORT / "scores_parquet",
                    rc._ids(COHORT / "cohort" / f"run_ids.trained.{suite}.txt"),
                ),
                **{
                    k: (
                        SEEDREP / "scores_parquet",
                        rc._ids(SEEDREP / "run_ids" / f"run_ids.{v}.{suite}.txt"),
                    )
                    for k, v in S_NAMES.items()
                },
            }
            base_run, base_cov = rc._ladders(COHORT / "scores_parquet", prompted, graphs)
            loaded = {}
            for arm, (store, ids) in arms_ids.items():
                run_l, cov_l = rc._ladders(store, ids, graphs)
                loaded[arm] = (run_l, cov_l)
                stored = rc.sweep._stored_values(store, ids)
                result["arms"].setdefault(arm, {})[suite] = {
                    "n_runs": len(ids),
                    "run_id_sha256": rc._digest(ids),
                    "mean_n_asks": sum(lad.n_asks for lad in run_l.values()) / len(run_l),
                }
                for metric in METRICS:
                    a = cov_l if metric in rc.COVERAGE_METRICS else run_l
                    b = base_cov if metric in rc.COVERAGE_METRICS else base_run
                    inst = rc.instrument_check(metric, a, graphs, stored)
                    if not rc.instrument_ok(inst):
                        raise SystemExit(f"instrument lock failed {arm} {suite} {metric}: {inst}")
                    got = rc.seed_matched_symmetric(
                        metric, a, b, graphs, seeds, seed=0, n_boot=10000
                    )
                    key = f"{metric}::{suite}"
                    result["cells"].setdefault(arm, {})[key] = got
                    if arm == "s0":
                        diff = abs(got["delta"] - pub[key])
                        result["lock"][key] = {
                            "published": pub[key],
                            "reproduced": got["delta"],
                            "abs_diff": diff,
                            "ok": diff <= args.tol,
                        }
                        if diff > args.tol:
                            raise SystemExit(
                                f"LOCK FAILED {key}: {got['delta']} vs published {pub[key]}"
                            )
            # The full recipe: seeds 1 and 2, averaged within the task. Its own lock: with ONE
            # training seed this function must reproduce seed_matched_symmetric exactly.
            for metric in METRICS:
                idx = 1 if metric in rc.COVERAGE_METRICS else 0
                b = base_cov if idx else base_run
                key = f"{metric}::{suite}"
                solo = pooled_symmetric(metric, [loaded["s1"][idx]], b, graphs, seeds)
                if abs(solo["delta"] - result["cells"]["s1"][key]["delta"]) > 1e-12:
                    raise SystemExit(f"POOLED-READER LOCK FAILED {key}: one-seed pool != s1 cell")
                result["cells"].setdefault("s1s2", {})[key] = [
                    {
                        "n_boot": nb,
                        "seed": sd,
                        **pooled_symmetric(
                            metric,
                            [loaded["s1"][idx], loaded["s2"][idx]],
                            b,
                            graphs,
                            seeds,
                            seed=sd,
                            n_boot=nb,
                        ),
                    }
                    for nb, sd in POOL_RESAMPLES
                ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
