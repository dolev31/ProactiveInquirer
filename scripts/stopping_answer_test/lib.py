"""Lane L1.5: does the trained policy stop right, and does the coverage gain reach the answer.

Reads the ISOLATED store `artifacts/testsplit_qa/scores_parquet` (never `scores/parquet`) and
the sibling `artifacts/testsplit_plan_metrics_20260918/contrasts.json`. Nothing here writes to
either store.

WHY THIS IMPORTS `pinq_train.gate` PRIVATE NAMES. `_stop_2x2`, `_coverage_ladder`, `_by_key`,
`_metric_by_run`, `_con`, `_select_runs`, `_mean`, `_phi`, `_phi_inv`, `_percentile` are the
functions that already compute -- and, for `_stop_2x2`, are already pinned equal to the scorer's
own `stop2x2_*` columns by `tests/test_stop_2x2_in_the_scorer.py` -- exactly the quantities this
lane needs. `scripts/select_checkpoint.py`, `scripts/recover_graph_version.py` and
`scripts/bca_era_attribution.py` already import private members of this module for the same
reason: a script is not one of the six packages import-linter's contracts govern
(`root_packages` in pyproject.toml), so nothing here is a firewall violation, and reusing the
vetted function beats a second hand-rolled copy that could quietly diverge from it.

THE NEW BOOTSTRAP CELLS -- P(STOP|done) and P(ASK|not done) trained-minus-prompted, and the
coverage-to-answer linkage -- have no existing estimator in this repo: `_stop_2x2` returns
pooled counts with no interval (the gate criterion is report-only), and `cluster_bootstrap` /
`bca_ci` both assume the statistic is a MEAN over per-cluster scalars, not a RATIO OF SUMS
pooled across clusters. A rate like P(STOP|done) is exactly the case where those differ: a
mean of per-task rates equal-weights every task regardless of how many decision points it
contributed and is undefined on a task with zero `done` states in one arm, while the pooled
ratio this repo's own `_stop_2x2` reports is a ratio of sums. So `bca_paired_ratio_delta` below
is a new function, written to the same BCa recipe `cluster_bootstrap` uses (bias correction from
the resample proportion below theta_hat, acceleration from a task-jackknife, endpoints via the
same `_phi`/`_phi_inv`/`_percentile`), but resampling TASKS and re-pooling their counts on every
replicate rather than averaging a per-task scalar.
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pinq_train.gate import (
    _by_key,
    _coverage_ladder,
    _mean,
    _metric_by_run,
    _percentile,
    _phi,
    _phi_inv,
    _select_runs,
    _stop_2x2,
)
from pinq_train.gate import (
    _con as gate_con,
)

__all__ = [
    "load_arm_runs",
    "population_report",
    "verify_turns_not_dropped",
    "verify_ladder_exists_for_asking_runs",
    "pooled_stop_cells",
    "per_task_stop_cells",
    "task_level_metric",
    "bca_paired_ratio_delta",
    "bca_two_sample_mean_delta",
    "with_stability",
    "matched_cost_coverage_by_task",
    "gate_con",
    "_select_runs",
    "_by_key",
    "_metric_by_run",
    "_coverage_ladder",
    "_stop_2x2",
    "_mean",
]

NEAR_ZERO = 0.01
"""A CI bound this close to 0 is read a second time (coordinator rule, 2026-09-18): 1k-resample
bootstraps flipped the sign of 3 of 7 such bounds across seeds, and one printed PASS died at
10k. See `with_stability`."""

STABILITY_SEEDS = (0, 1, 2)
STABILITY_N_BOOT = 50_000
DEFAULT_N_BOOT = 10_000


# --------------------------------------------------------------------------- population


def load_arm_runs(
    con, *, arm_id: str, model_id: str, grid_name: str, suite: str | None = None
) -> list[dict]:
    """The (run_id, suite_id, task_id, seed, n_asks, stop_reason) rows for one arm, optionally
    filtered to one suite.

    Wraps `pinq_train.gate._select_runs`, which already matches `model_id` through
    `calls.parquet` rather than `arm_id` alone -- see that function's docstring on why: an arm
    id can serve two model pins under the two-pin protocol. `_select_runs` itself returns only
    `(run_id, suite_id, task_id, seed)`; `n_asks` and `stop_reason` are joined in here because
    `_stop_2x2` requires both on every row it is handed.
    """
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    runs = _select_runs(con, arm=arm_id, grids=[grid_name], model_id=model_id)
    if suite is not None:
        runs = [r for r in runs if str(r["suite_id"]) == suite]
    ids = [r["run_id"] for r in runs]
    stop_rows = _rows(
        con, f"SELECT run_id, n_asks, stop_reason FROM runs WHERE run_id IN {_in(ids)}"
    )
    by_id = {r["run_id"]: r for r in stop_rows}
    for r in runs:
        extra = by_id.get(r["run_id"], {})
        r["n_asks"] = extra.get("n_asks")
        r["stop_reason"] = extra.get("stop_reason")
    return runs


def population_report(con, *, run_ids: Sequence[str], scorer_hash: str) -> dict[str, Any]:
    """Assert every `run_id` is present in `scores` at `scorer_hash`, and in `runs`.

    Per the analysis rules: a population must be checked present before it is read, not
    assumed present because a sibling document says it was scored once. Returns counts rather
    than raising, so the caller can print the command-and-output pair CLAUDE.md rule 3 asks
    for before deciding whether to trust anything downstream.
    """
    run_ids = sorted(set(run_ids))
    if not run_ids:
        return {
            "n_population": 0,
            "n_in_runs": 0,
            "n_scored": 0,
            "missing_from_runs": [],
            "missing_scores": [],
        }
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    in_runs = {
        r["run_id"] for r in _rows(con, f"SELECT run_id FROM runs WHERE run_id IN {_in(run_ids)}")
    }
    scored = {
        r["run_id"]
        for r in _rows(
            con,
            "SELECT DISTINCT run_id FROM scores WHERE scorer_hash = "
            f"'{scorer_hash}' AND run_id IN {_in(run_ids)}",
        )
    }
    return {
        "n_population": len(run_ids),
        "n_in_runs": len(in_runs),
        "n_scored": len(scored),
        "missing_from_runs": sorted(set(run_ids) - in_runs),
        "missing_scores": sorted(set(run_ids) - scored),
    }


def verify_turns_not_dropped(runs_root: Path, n_turns_by_run: Mapping[str, int]) -> dict[str, Any]:
    """Cross-check `runs.parquet.n_turns` against `runs/<run_id>/turns.jsonl` on disk.

    Coordinator alert, 2026-09-18: a compaction bug dropped the turn/evidence/call rows of
    21,001 runs whose `turns.jsonl` is intact on disk, so they read as `n_turns=0` in the
    parquet -- exactly what a policy that never asks would also produce, and exactly the shape
    `pooled_stop_cells` would silently fold into "asked nothing" rather than "not compacted".
    A run FAILS when its `turns.jsonl` is non-empty on disk but the parquet says `n_turns == 0`.
    """
    bad: list[str] = []
    missing_dir: list[str] = []
    checked = 0
    for run_id, n_turns in n_turns_by_run.items():
        p = runs_root / run_id / "turns.jsonl"
        if not p.exists():
            missing_dir.append(run_id)
            continue
        checked += 1
        if p.stat().st_size > 0 and int(n_turns) == 0:
            bad.append(run_id)
    return {
        "n_checked": checked,
        "n_missing_dir": len(missing_dir),
        "missing_dir": missing_dir,
        "n_bad": len(bad),
        "bad_run_ids": bad,
    }


def verify_ladder_exists_for_asking_runs(
    con, runs: Sequence[Mapping], *, scorer_hash: str
) -> dict[str, Any]:
    """Every run with `n_asks > 0` must have a non-empty `frontier_q#k` ladder in `scores` at
    `scorer_hash`. A run with asks but an absent ladder is the scores-side signature of the
    same dropped-rows bug `verify_turns_not_dropped` catches from the runs side."""
    ladders = _coverage_ladder(con, runs, scorer_hash=scorer_hash)
    bad = [
        str(r["run_id"])
        for r in runs
        if int(r.get("n_asks") or 0) > 0 and not ladders.get(str(r["run_id"]))
    ]
    return {"n_checked": len(runs), "n_bad": len(bad), "bad_run_ids": bad}


# --------------------------------------------------------------------------- stop 2x2


def per_task_stop_cells(
    con, runs: Sequence[Mapping], *, scorer_hash: str
) -> dict[str, dict[str, Any]]:
    """`_stop_2x2`, called once per task (both seeds of that task pooled).

    Reuses the exact gate function per task rather than re-deriving the ladder-walk, so a
    per-task cell can never disagree with the pooled arm-level cell computed the same way
    (`pooled_stop_cells` sums these same six counters across tasks; a test pins the two equal).
    """
    ladders = _coverage_ladder(con, runs, scorer_hash=scorer_hash)
    by_task: dict[str, list[dict]] = {}
    for r in runs:
        by_task.setdefault(str(r["task_id"]), []).append(r)
    return {task: _stop_2x2(con, rs, ladders) for task, rs in sorted(by_task.items())}


def pooled_stop_cells(con, runs: Sequence[Mapping], *, scorer_hash: str) -> dict[str, Any]:
    """`pinq_train.gate._stop_2x2` pooled over every decision point in `runs` -- one arm's
    (optionally one suite's) worth of runs, scorer_hash threaded explicitly."""
    ladders = _coverage_ladder(con, runs, scorer_hash=scorer_hash)
    return _stop_2x2(con, runs, ladders)


def task_level_metric(
    con, runs: Sequence[Mapping], metric: str, *, scorer_hash: str
) -> dict[tuple[str, str], float]:
    """`metric`, averaged over the seeds of each (suite, task) -- the same seed-into-task
    averaging `_paired_by_task_map` does for a delta, here for one arm's own level."""
    vals = _metric_by_run(con, metric, scorer_hash=scorer_hash)
    by_task: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        v = vals.get(str(r["run_id"]))
        if v is None:
            continue
        by_task.setdefault((str(r["suite_id"]), str(r["task_id"])), []).append(v)
    return {k: _mean(v) for k, v in by_task.items()}


# --------------------------------------------------------------------------- BCa: paired ratio delta


def bca_paired_ratio_delta(
    task_rows: Sequence[tuple[str, float, float, float, float]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """BCa interval for `trained_num/trained_den - prompted_num/prompted_den`, resampling TASKS
    and re-pooling both arms' numerators and denominators on every replicate.

    `task_rows`: `(task_id, trained_num, trained_den, prompted_num, prompted_den)`, ALREADY
    restricted to the paired population (a task present in both arms) and already pooled over
    seeds within the task -- the caller's job, mirroring `_paired_by_task_map`.

    NOT a mean of per-task rates: a task with `trained_den == 0` (no `done` decision point in
    that arm on that task) has an undefined per-task rate, and averaging it in as a skipped `nan`
    would silently reweight the tasks that stayed. Pooling the counts first and dividing once,
    per replicate, is what `_stop_2x2` itself does at the arm level, so the interval and the
    point estimate answer the same question.

    Sorted by `task_id` before resampling, for the reason `cluster_bootstrap` sorts: the RNG
    draws indices, so an unsorted caller order is a silent, undocumented input to the endpoints
    (measured example in that function's docstring: a 1/824 step from run-id order alone).
    """
    rows = sorted(task_rows, key=lambda t: t[0])
    if not rows:
        return _nan_result(n=0)

    def ratio_delta(sample: Sequence[tuple[str, float, float, float, float]]) -> float:
        tn = sum(r[1] for r in sample)
        td = sum(r[2] for r in sample)
        pn = sum(r[3] for r in sample)
        pd = sum(r[4] for r in sample)
        if td <= 0 or pd <= 0:
            return float("nan")
        return tn / td - pn / pd

    theta_hat = ratio_delta(rows)
    n = len(rows)
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        v = ratio_delta(sample)
        if not math.isnan(v):
            reps.append(v)
    if not reps or math.isnan(theta_hat):
        return _nan_result(n=n)

    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))

    jack: list[float] = []
    for i in range(n):
        rest = rows[:i] + rows[i + 1 :]
        if rest:
            v = ratio_delta(rest)
            if not math.isnan(v):
                jack.append(v)
    if len(jack) > 1:
        jbar = statistics.fmean(jack)
        num = sum((jbar - x) ** 3 for x in jack)
        den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
        acc = num / den if den else 0.0
    else:
        acc = 0.0

    def adj(q: float) -> float:
        z = _phi_inv(q)
        denom = 1 - acc * (z0 + z)
        return _phi(z0 + (z0 + z) / denom) if denom else q

    lo = _percentile(sorted(reps), adj(0.025))
    hi = _percentile(sorted(reps), adj(0.975))
    return {
        "point": theta_hat,
        "lo": lo,
        "hi": hi,
        "n": n,
        "n_boot": n_boot,
        "n_reps_used": len(reps),
    }


def _nan_result(*, n: int) -> dict[str, Any]:
    return {
        "point": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "n": n,
        "n_boot": 0,
        "n_reps_used": 0,
    }


# --------------------------------------------------------------------------- BCa: two independent samples


def bca_two_sample_mean_delta(
    group_a: Sequence[float], group_b: Sequence[float], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """BCa interval for `mean(group_a) - mean(group_b)`, two INDEPENDENT samples (disjoint
    tasks), each resampled with replacement from itself.

    Not `cluster_bootstrap` / `bca_ci`: those assume one paired population resampled once, and
    here group_a and group_b are disjoint task sets (the coverage-linkage split), so the two
    must be resampled independently. Acceleration is the standard two-sample generalisation:
    jackknife over the POOLED n_a + n_b leave-one-out replicates (Efron & Tibshirani 1993,
    ch. 14), each removing one unit from whichever group it belongs to.
    """
    a = sorted(float(x) for x in group_a if not math.isnan(x))
    b = sorted(float(x) for x in group_b if not math.isnan(x))
    if not a or not b:
        return _nan_result(n=len(a) + len(b))

    theta_hat = statistics.fmean(a) - statistics.fmean(b)
    na, nb = len(a), len(b)
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        ra = statistics.fmean([a[rng.randrange(na)] for _ in range(na)])
        rb = statistics.fmean([b[rng.randrange(nb)] for _ in range(nb)])
        reps.append(ra - rb)

    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))

    jack: list[float] = []
    for i in range(na):
        rest = a[:i] + a[i + 1 :]
        if rest:
            jack.append(statistics.fmean(rest) - statistics.fmean(b))
    for i in range(nb):
        rest = b[:i] + b[i + 1 :]
        if rest:
            jack.append(statistics.fmean(a) - statistics.fmean(rest))
    if len(jack) > 1:
        jbar = statistics.fmean(jack)
        num = sum((jbar - x) ** 3 for x in jack)
        den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
        acc = num / den if den else 0.0
    else:
        acc = 0.0

    def adj(q: float) -> float:
        z = _phi_inv(q)
        denom = 1 - acc * (z0 + z)
        return _phi(z0 + (z0 + z) / denom) if denom else q

    lo = _percentile(sorted(reps), adj(0.025))
    hi = _percentile(sorted(reps), adj(0.975))
    return {
        "point": theta_hat,
        "lo": lo,
        "hi": hi,
        "n": na + nb,
        "n_a": na,
        "n_b": nb,
        "n_boot": n_boot,
        "n_reps_used": len(reps),
    }


# --------------------------------------------------------------------------- stability wrapper


def with_stability(
    compute: Callable[[int, int], dict[str, Any]],
    *,
    seed: int = 0,
    near_zero: float = NEAR_ZERO,
) -> dict[str, Any]:
    """Run `compute(n_boot, seed)` at 10k. If either bound sits within `near_zero` of 0, rerun
    at 50k with three seeds and report whether that bound's SIGN is stable across them.

    Coordinator rule, 2026-09-18: a peer measured 3 of 7 gated cells whose bound sat within 0.01
    of zero flipping that bound's sign across bootstrap seeds at 1k resamples, and one printed
    PASS died at 10k. `verdict` is `"excludes_zero"` / `"includes_zero"` when no bound is that
    close, or when every close bound's sign held across all three 50k seeds; otherwise
    `"undecided"` -- never a pass/fail read off an unstable bound.
    """
    base = compute(DEFAULT_N_BOOT, seed)
    lo, hi = base["lo"], base["hi"]
    result = dict(base, n_boot_reported=DEFAULT_N_BOOT, stability_checked=False)
    if math.isnan(lo) or math.isnan(hi):
        result["verdict"] = "absent"
        return result

    lo_close = abs(lo) <= near_zero
    hi_close = abs(hi) <= near_zero
    if not (lo_close or hi_close):
        result["verdict"] = "excludes_zero" if (lo > 0 or hi < 0) else "includes_zero"
        return result

    reps = [compute(STABILITY_N_BOOT, s) for s in STABILITY_SEEDS]

    def sign(x: float) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    stable = True
    if lo_close:
        signs = {sign(r["lo"]) for r in reps}
        stable = stable and len(signs) == 1
    if hi_close:
        signs = {sign(r["hi"]) for r in reps}
        stable = stable and len(signs) == 1

    result["stability_checked"] = True
    result["stability_seeds"] = list(STABILITY_SEEDS)
    result["stability_n_boot"] = STABILITY_N_BOOT
    result["stability_reps"] = [{"lo": r["lo"], "hi": r["hi"]} for r in reps]
    result["stable"] = stable
    if not stable:
        result["verdict"] = "undecided"
    else:
        # Read the verdict off the 50k, seed-0 replicate (index 0), which is the same seed
        # convention as every other interval in this lane.
        r0 = reps[0]
        result["verdict"] = "excludes_zero" if (r0["lo"] > 0 or r0["hi"] < 0) else "includes_zero"
        result["lo"], result["hi"], result["point"] = r0["lo"], r0["hi"], r0["point"]
        result["n_boot_reported"] = STABILITY_N_BOOT
    return result


# --------------------------------------------------------------------------- matched-cost coverage, by task


def matched_cost_coverage_by_task(
    con, *, ck_runs: Sequence[Mapping], ba_runs: Sequence[Mapping], scorer_hash: str
) -> dict[tuple[str, str], float]:
    """Per-(suite, task) matched-cost `evidence_coverage` delta, seeds averaged within task.

    Reproduces the inner loop of `pinq_train.gate._matched_cost` (checkpoint's own terminal
    coverage minus the mean of the baseline's `frontier_q#min(k, baseline n_asks)` across the
    baseline runs sharing that key), which that function computes internally as `by_task` but
    does not return -- only the suite/pooled aggregates are public. Calling the same primitives
    (`_coverage_ladder`, `_metric_by_run`, `_by_key`) in the same order keeps this identical to
    the published headline number rather than a second, possibly-diverging implementation; the
    caller in `run.py` cross-checks the pooled mean of this function's output against
    `artifacts/testsplit_qa/verdicts/stacked-notdone.*.json` before trusting it.
    """
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    all_runs = [*ck_runs, *ba_runs]
    n_asks = {str(r["run_id"]): int(r.get("n_asks") or 0) for r in _n_asks_rows(con, all_runs)}
    ck_keys = _by_key(ck_runs)
    ba_keys = _by_key(ba_runs)
    shared = sorted(set(ck_keys) & set(ba_keys))
    base_used = sorted({rid for key in shared for rid in ba_keys[key]})
    ladders = _coverage_ladder(con, [{"run_id": rid} for rid in base_used], scorer_hash=scorer_hash)

    per_task: dict[tuple[str, str], list[float]] = {}
    for key in shared:
        deltas: list[float] = []
        for c_rid in ck_keys[key]:
            c_val = cov.get(c_rid)
            if c_val is None:
                continue
            k = n_asks.get(c_rid, 0)
            rungs = []
            for b_rid in ba_keys[key]:
                b_n = n_asks.get(b_rid, 0)
                at = min(k, b_n)
                rung = (ladders.get(b_rid) or {}).get(at)
                if rung is None:
                    continue
                rungs.append(float(rung))
            if not rungs:
                continue
            deltas.append(float(c_val) - _mean(rungs))
        if deltas:
            per_task.setdefault((str(key[0]), str(key[1])), []).append(_mean(deltas))
    return {k: _mean(v) for k, v in per_task.items()}


def _n_asks_rows(con, runs: Sequence[Mapping]) -> list[dict]:
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    ids = [r["run_id"] for r in runs]
    if not ids:
        return []
    return _rows(con, f"SELECT run_id, n_asks FROM runs WHERE run_id IN {_in(ids)}")
