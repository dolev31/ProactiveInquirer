#!/usr/bin/env python3
"""FRAMES cap-8 accuracy: `inquirer_trained` vs `inquirer_prompted`@Qwen3-8B-base, paired on
task, per seed and pooled over three seeds, task-clustered BCa.

WHY A SEPARATE SCRIPT RATHER THAN A ONE-OFF QUERY. The pooled-over-seeds contrast has one
easy way to get silently wrong: collapse `(task_id, seed)` to `task_id` before pairing, and a
task that only ONE arm completed at one seed quietly borrows the other arm's OTHER-seed value
instead of being dropped. `per_seed_and_pooled` below keys every row on the full
`(task_id, seed)` pair through `paired_difference`'s own intersection (`set(a) & set(b)`), and
only THEN maps each surviving key back to its task for clustering -- so a task with 2 of 3
seeds present contributes a 2-valued cluster, never a 3-valued one with a borrowed number in it.

WHY THE ESTIMATOR IS `pi_eval.stats.inference`, IMPORTED AND NOT REIMPLEMENTED. That module is
what `artifacts/frames/FRAMES.md` and `artifacts/frames_frontier/FRAMES_FRONTIER.md` were
computed with (`cluster_bootstrap`/`paired_difference`, post the `4b7b24b` order-invariance
fix), at `n_boot=10000, seed=0`. A second bootstrap implementation would make this file's
numbers uncomparable to those two even if it were correct.

FRAMES clusters equal its tasks (`template_id` is NULL on this suite -- see FRAMES.md), so the
per-seed contrast needs no cluster map at all: `paired_difference(a, b, clusters=None, ...)`
already puts each task in its own singleton cluster, because its `clusters.get(k, k) if
clusters else k` falls back to the key itself. Only the POOLED contrast needs an explicit
cluster map, to fold a task's up-to-three seed values back into one cluster.

Usage, from the repo root (read-only; writes nothing):
    .venv/bin/python scripts/frames_seeds/contrasts.py \\
        --snapshot artifacts/frames_frontier/scores_parquet.frames \\
        --snapshot artifacts/frames_seeds_20260918/scores_parquet.frames \\
        --cap 8 --metric answer_correct \\
        --trained-pin 6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856 \\
        --prompted-pin e9af49a4b5ff0a0b27a57f7c4feca06566cbfbec3cc0f810920f5f414862fcfe \\
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pi_eval.report import ELIGIBLE  # noqa: E402
from pi_eval.stats.inference import Estimate, paired_difference  # noqa: E402


def load_metric(
    snapshots: Sequence[str],
    *,
    cap: int,
    pin_hash: str,
    metric: str,
) -> dict[tuple[str, int], float]:
    """`{(task_id, seed): value}` for one model pin's scored runs at one budget_cap.

    Reads across every snapshot given (seed 0 and seeds 1/2 live in different isolated
    stores) and REFUSES on a conflicting duplicate -- the same `(task_id, seed)` scored to two
    different values in two snapshots means the snapshots overlap and silently averaging or
    last-write-wins would hide that.
    """
    import duckdb

    con = duckdb.connect()
    out: dict[tuple[str, int], float] = {}
    for snap in snapshots:
        runs_p = Path(snap) / "runs.parquet"
        scores_p = Path(snap) / "scores.parquet"
        if not runs_p.exists() or not scores_p.exists():
            raise FileNotFoundError(f"not a scored snapshot (missing parquet): {snap}")
        # Reuses `pi_eval.report.ELIGIBLE` verbatim rather than a hand-copied WHERE clause, so
        # this can never silently drift from the predicate every other reported table uses
        # (status=ok, not dev/dirty/pilot/exploratory/gold_exposed/canary_hit/counterfactual,
        # firewall_ok, split=test, both reconciliation flags true).
        q = f"""
            SELECT r.task_id, r.seed, s.value
            FROM read_parquet('{runs_p.as_posix()}') r
            JOIN read_parquet('{scores_p.as_posix()}') s USING (run_id)
            WHERE r.budget_cap = {int(cap)}
              AND r.model_pin_hash = '{pin_hash}'
              AND s.metric_name = '{metric}'
              AND {ELIGIBLE}
        """
        for task_id, seed, value in con.execute(q).fetchall():
            key = (task_id, int(seed))
            if key in out and out[key] != value:
                raise ValueError(
                    f"{snap}: {key} = {value} conflicts with an earlier snapshot's "
                    f"{out[key]} for the same pin+cap+metric -- snapshots overlap"
                )
            out[key] = value
    return out


def per_seed_and_pooled(
    trained: Mapping[tuple[str, int], float],
    prompted: Mapping[tuple[str, int], float],
    seeds: Sequence[int],
    *,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    boot_seed: int = 0,
) -> dict[str, Estimate]:
    """One `Estimate` per seed (singleton task clusters) plus one `'pooled'` Estimate.

    The pooled contrast keys on `"task_id|seed"` so `paired_difference`'s own
    `set(a) & set(b)` intersection decides inclusion PER (task, seed) row, exactly as the
    per-seed contrasts do -- then the cluster map folds surviving keys back onto their task,
    so a task contributes one cluster of up to `len(seeds)` values, not `len(seeds)`
    independent clusters.
    """
    results: dict[str, Estimate] = {}
    for seed in seeds:
        a = {tid: v for (tid, sd), v in trained.items() if sd == seed}
        b = {tid: v for (tid, sd), v in prompted.items() if sd == seed}
        results[str(seed)] = paired_difference(
            a, b, clusters=None, n_boot=n_boot, n_perm=n_perm, seed=boot_seed
        )

    a_pool = {f"{tid}|{sd}": v for (tid, sd), v in trained.items() if sd in seeds}
    b_pool = {f"{tid}|{sd}": v for (tid, sd), v in prompted.items() if sd in seeds}
    cluster_of = {k: k.rsplit("|", 1)[0] for k in set(a_pool) | set(b_pool)}
    results["pooled"] = paired_difference(
        a_pool, b_pool, clusters=cluster_of, n_boot=n_boot, n_perm=n_perm, seed=boot_seed
    )
    return results


def render_table(results: Mapping[str, Estimate], *, metric: str) -> str:
    lines = [
        f"| seed | n | delta {metric} | BCa 95% | zero |",
        "|---|---|---|---|---|",
    ]
    for key in [*(k for k in results if k != "pooled"), "pooled"]:
        est = results[key]
        zero = "excludes 0" if (est.ci_lo > 0 or est.ci_hi < 0) else "covers 0"
        label = "pooled (3 seeds)" if key == "pooled" else key
        lines.append(
            f"| {label} | {est.n} | {est.point:+.4f} | [{est.ci_lo:+.4f}, {est.ci_hi:+.4f}] | {zero} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--snapshot", action="append", required=True, help="repeatable")
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--metric", default="answer_correct")
    p.add_argument("--trained-pin", required=True)
    p.add_argument("--prompted-pin", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args(argv)

    trained = load_metric(a.snapshot, cap=a.cap, pin_hash=a.trained_pin, metric=a.metric)
    prompted = load_metric(a.snapshot, cap=a.cap, pin_hash=a.prompted_pin, metric=a.metric)
    for seed in a.seeds:
        nt = sum(1 for (_, sd) in trained if sd == seed)
        npr = sum(1 for (_, sd) in prompted if sd == seed)
        print(f"seed {seed}: trained n={nt} prompted n={npr}", file=sys.stderr)

    results = per_seed_and_pooled(trained, prompted, a.seeds)
    print(render_table(results, metric=a.metric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
