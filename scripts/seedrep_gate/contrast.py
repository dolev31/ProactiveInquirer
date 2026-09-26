"""The seed-replicate contrast for the rung-2 DPO stage of `qwen3-8b-dpo-stacked-notdone-both`.

Everything here is computed in ONE isolated store holding all five populations (s0, s1, s2,
the prompted base comparator and the never-ask arm) scored in ONE `pi score` pass, so every
cell below carries the SAME `scorer_hash`. That is the only way s0 is comparable to s1/s2:
the published s0 cell is at `aa18b1fe`, from a pinned worktree predating `513c876` (the metric
registry change that moves `scorer_hash`) and `4b7b24b` (the `bca_ci` sort fix), and a delta
read out of one store cannot be differenced against a delta read out of another.

Reuses `scripts/decomposition_test/contrast.py` rather than reimplementing:
  * `select_and_contrast` -- matched-cost against a COMPARATOR, with an explicit
    `checkpoint_model_id`. Without it `_select_runs(model_id=None)` selects all 3,600
    `inquirer_trained` rows in this store and pools the three checkpoints into one average.
  * `select_and_contrast_symmetric` -- arm-vs-arm with BOTH sides read at `min(k_a, k_b)`.
    Used for s1-vs-s2 because MATCHED-COST DELTAS DO NOT SUBTRACT: the comparator rung moves
    with each arm, so `delta(s1) - delta(s2)` is not the s1-vs-s2 contrast.
  * `assert_non_vacuous`, `stability_recheck` -- the project's standing rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "decomposition_test"))
import duckdb  # noqa: E402
from contrast import (  # noqa: E402
    assert_non_vacuous,
    select_and_contrast,
    select_and_contrast_symmetric,
    stability_recheck,
)

from pinq_train.gate import (  # noqa: E402
    _by_key,
    _con,
    _mean,
    _metric_by_run,
    _select_runs,
    _with_stop,
)

S0 = "qwen3-8b-dpo-stacked-notdone-both"
S1 = f"{S0}-s1"
S2 = f"{S0}-s2"
BASE = "qwen3-8b-base"
GRID = "tier1_trained_qa_base"


def levels(parquet_dir: Path, *, model_id: str, arm: str, scorer_hash: str) -> dict:
    """Per-suite mean `evidence_coverage` at the arm's own cap-8 terminal value, plus mean
    asks. A LEVEL, not a contrast: no comparator, no truncation, no interval."""
    con = _con(parquet_dir)
    runs = _select_runs(con, arm=arm, grids=[GRID], model_id=model_id)
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    n_asks = {str(r["run_id"]): int(r["n_asks"] or 0) for r in _with_stop(con, runs)}
    by_key = _by_key(runs)
    out: dict[str, dict] = {}
    per_task: dict[tuple, list[float]] = {}
    per_task_k: dict[tuple, list[float]] = {}
    for (suite, task, _seed), rids in by_key.items():
        vals = [cov[r] for r in rids if r in cov]
        ks = [float(n_asks.get(r, 0)) for r in rids]
        if vals:
            per_task.setdefault((suite, task), []).append(_mean(vals))
        if ks:
            per_task_k.setdefault((suite, task), []).append(_mean(ks))
    for suite in sorted({k[0] for k in per_task}):
        vals = [v for k, v in per_task.items() if k[0] == suite]
        ks = [v for k, v in per_task_k.items() if k[0] == suite]
        out[suite] = {
            "n_tasks": len(vals),
            "n_runs": sum(len(v) for k, v in by_key.items() if k[0] == suite),
            "mean_coverage_cap8": _mean([_mean(v) for v in vals]),
            "mean_asks": _mean([_mean(v) for v in ks]),
        }
    return {"model_id": model_id, "arm": arm, "by_suite": out}


def task_overlap(parquet_dir: Path) -> dict:
    """Do s1/s2 cover the same (suite, task, seed) keys as the published s0 arm? If they do,
    the contamination question they inherit is s0's, which was answered (0 violations, with a
    negative control that produced 40). Stated because the guard itself is INERT on s1/s2:
    their manifests carry `train_id_set_hash: null`, so a guard keyed on it cannot fire."""
    con = _con(parquet_dir)
    keys = {}
    for mid in (S0, S1, S2):
        runs = _select_runs(con, arm="inquirer_trained", grids=[GRID], model_id=mid)
        keys[mid] = {(r["suite_id"], r["task_id"], r["seed"]) for r in runs}
    return {
        "n_keys": {m: len(k) for m, k in keys.items()},
        "s1_equals_s0": sorted(keys[S1]) == sorted(keys[S0]),
        "s2_equals_s0": sorted(keys[S2]) == sorted(keys[S0]),
        "s1_equals_s2": sorted(keys[S1]) == sorted(keys[S2]),
        "s1_minus_s0": len(keys[S1] - keys[S0]),
        "s0_minus_s1": len(keys[S0] - keys[S1]),
    }


def pooling_demo(parquet_dir: Path, *, scorer_hash: str) -> dict:
    """The measurement that justifies `--checkpoint-model-id`: the SAME call with the
    checkpoint filter dropped. Reported so the fix is evidenced, not asserted."""
    con = _con(parquet_dir)
    pooled = _select_runs(con, arm="inquirer_trained", grids=[GRID], model_id=None)
    r = select_and_contrast(
        parquet_dir,
        checkpoint_model_id=None,
        baseline_model_id=BASE,
        scorer_hash=scorer_hash,
        seed=0,
        n_resamples=2000,
    )
    return {
        "n_rows_selected_without_filter": len(pooled),
        "n_rows_selected_with_filter_s1": len(
            _select_runs(con, arm="inquirer_trained", grids=[GRID], model_id=S1)
        ),
        "pooled_delta_by_suite": {
            s: {"delta": c["delta"], "n_tasks": c["n_tasks"]} for s, c in r["by_suite"].items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    args = ap.parse_args()

    pq = Path(args.parquet)
    sh = args.scorer_hash
    con = duckdb.connect()
    out: dict = {
        "parquet_dir": str(pq),
        "scorer_hash": sh,
        "n_resamples": args.n_resamples,
        "graph_version": con.execute(
            f"SELECT DISTINCT graph_version FROM read_parquet('{pq}/scores.parquet')"
        ).fetchall(),
        "code_version": con.execute(
            f"SELECT DISTINCT code_version FROM read_parquet('{pq}/runs.parquet')"
        ).fetchall(),
    }

    # --- non-vacuity over MY 2,400 before any contrast is read -------------------------------
    nv = {}
    for mid in (S1, S2):
        runs = _select_runs(_con(pq), arm="inquirer_trained", grids=[GRID], model_id=mid)
        nv[mid] = assert_non_vacuous(
            pq, Path(args.runs_root), [r["run_id"] for r in runs], scorer_hash=sh
        )
    out["non_vacuity"] = nv

    out["task_key_overlap"] = task_overlap(pq)
    out["pooling_demo"] = pooling_demo(pq, scorer_hash=sh)

    # --- levels ------------------------------------------------------------------------------
    out["levels"] = {
        m: levels(pq, model_id=m, arm=a, scorer_hash=sh)
        for m, a in (
            (S0, "inquirer_trained"),
            (S1, "inquirer_trained"),
            (S2, "inquirer_trained"),
            (BASE, "inquirer_prompted"),
        )
    }
    out["levels"]["never_ask"] = levels(pq, model_id=None, arm="drafter_only", scorer_hash=sh)

    # --- each seed against the SHARED comparator ---------------------------------------------
    out["vs_comparator"] = {}
    for mid in (S0, S1, S2):
        r = select_and_contrast(
            pq,
            checkpoint_model_id=mid,
            baseline_model_id=BASE,
            scorer_hash=sh,
            seed=0,
            n_resamples=args.n_resamples,
        )
        r["stability"] = stability_recheck(
            pq,
            checkpoint_model_id=mid,
            baseline_model_id=BASE,
            by_suite=r["by_suite"],
            scorer_hash=sh,
        )
        out["vs_comparator"][mid] = r

    # --- the only CLEAN seed contrast, with its own per-task pairing ---------------------------
    out["arm_vs_arm"] = {}
    for a, b in ((S1, S2), (S1, S0), (S2, S0)):
        r = select_and_contrast_symmetric(
            pq,
            arm_a_model_id=a,
            arm_b_model_id=b,
            scorer_hash=sh,
            seed=0,
            n_resamples=args.n_resamples,
        )
        r["stability"] = stability_recheck_symmetric(pq, a, b, r["by_suite"], scorer_hash=sh)
        out["arm_vs_arm"][f"{a} vs {b}"] = r

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


def stability_recheck_symmetric(
    pq, a, b, by_suite, *, scorer_hash, tol=0.01, big_n=50_000, reseeds=(0, 1, 2)
) -> dict:
    """`stability_recheck` for the symmetric estimator. Same rule -- any 10k bound within
    `tol` of zero is re-read at `big_n` across three bootstrap seeds and its SIGN reported --
    but it must call the symmetric contrast, because the two estimators are different."""
    out: dict = {}
    for suite, cell in by_suite.items():
        lo, hi = cell.get("ci_lo"), cell.get("ci_hi")
        near = [x for x in (lo, hi) if x is not None and abs(x) <= tol]
        if not near:
            out[suite] = {"checked": False, "stable": True, "verdict": "as computed"}
            continue
        which = "ci_lo" if (lo is not None and abs(lo) <= tol) else "ci_hi"
        signs, reads = [], []
        for rs in reseeds:
            r = select_and_contrast_symmetric(
                pq,
                arm_a_model_id=a,
                arm_b_model_id=b,
                scorer_hash=scorer_hash,
                seed=rs,
                n_resamples=big_n,
            )["by_suite"][suite]
            reads.append(
                {"seed": rs, "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "delta": r["delta"]}
            )
            signs.append(1 if r[which] >= 0 else -1)
        out[suite] = {
            "checked": True,
            "near_zero_bound": which,
            "reads_at_50k": reads,
            "signs": signs,
            "stable": len(set(signs)) == 1,
            "verdict": "as computed" if len(set(signs)) == 1 else "undecided",
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
