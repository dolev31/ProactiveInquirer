#!/usr/bin/env python
"""Score a cap-aware comparator campaign and contrast the two comparator arms.

Written BEFORE the campaign finished, and tested on the partial store, so the scoring step is not
improvised at the moment it matters.

WHAT IT REFUSES TO DO. It will not read a contrast until the campaign is complete and every unit is
ok, because a store whose comparator is short on a task set selected by which requests survived is
the failure this programme has already paid for twice.

HOW IT CONTRASTS. A direct paired difference on the tasks both arms share, with the rollout seeds
averaged into the task BEFORE resampling. Not folding seeds gives pseudo-units that narrow the
interval while leaving the point estimate bit-identical, so nothing looks wrong. It prints the
discordant and tie counts beside every interval, and when the per-task delta takes only a handful of
distinct values it prints a sign test too, because a bootstrap interval on a lattice-valued quantity
can exclude zero where a sign test over the informative pairs does not.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import subprocess
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
VENV = REPO / ".venv/bin/python"
A, B = "inquirer_prompted", "inquirer_prompted_capaware"


def outcomes(runs_root: Path) -> tuple[int, int, int]:
    ok = err = other = 0
    for s in runs_root.glob("*/status.json"):
        try:
            st = json.loads(s.read_text()).get("status")
        except Exception:
            other += 1
            continue
        if st == "ok":
            ok += 1
        elif st == "error":
            err += 1
        else:
            other += 1
    return ok, err, other


def run(cmd: list[str]) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd])


def bca_like(vals: list[float], *, seed: int, n_boot: int) -> tuple[float, float, float]:
    """Percentile bootstrap over TASKS on sorted input. Sorted because resampling draws indices,
    so the caller's order would otherwise be an input to the endpoints."""
    import random

    xs = sorted(float(v) for v in vals if not math.isnan(float(v)))
    if not xs:
        return (float("nan"),) * 3
    point = statistics.fmean(xs)
    rng = random.Random(seed)
    n = len(xs)
    reps = sorted(statistics.fmean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = reps[int(0.025 * (len(reps) - 1))]
    hi = reps[int(0.975 * (len(reps) - 1))]
    return point, lo, hi


def sign_test(deltas: list[float]) -> tuple[int, int, int, float]:
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n = pos + neg
    if n == 0:
        return pos, neg, ties, float("nan")
    k = min(pos, neg)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n))
    return pos, neg, ties, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--planned", type=int, required=True)
    ap.add_argument("--metric", default="evidence_coverage")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--allow-partial", action="store_true", help="diagnostics only, prints no contrast"
    )
    a = ap.parse_args()

    runs_root, store = Path(a.runs_root), Path(a.store)
    ok, err, other = outcomes(runs_root)
    total = ok + err + other
    print(f"outcomes: ok={ok} error={err} other={other} total={total} planned={a.planned}")
    complete = ok == a.planned and err == 0 and other == 0
    if not complete and not a.allow_partial:
        print("REFUSING: the campaign is not complete and clean. No contrast read.")
        return 1
    if not complete:
        print(
            "PARTIAL, diagnostics only. A contrast from here would be selected on which units finished."
        )

    if not a.skip_build:
        if run(
            [
                VENV,
                "-m",
                "pi_run.cli",
                "compact",
                "--runs-root",
                runs_root,
                "--out",
                store,
                "--exclude-dev",
            ]
        ):
            print("compact failed")
            return 1
        if run(
            [
                VENV,
                "-m",
                "pi_run.cli",
                "score",
                "--parquet",
                store,
                "--runs-root",
                runs_root,
                "--gold-root",
                REPO / "data/gold",
                "--corpora-root",
                REPO / "data/corpora",
                "--allow-no-judge",
                "--no-answers",
            ]
        ):
            print("score failed")
            return 1

    con = duckdb.connect()
    rows = con.execute(
        f"""SELECT r.suite_id, r.arm_id, r.task_id, s.value
            FROM read_parquet('{store}/scores.parquet') s
            JOIN read_parquet('{store}/runs.parquet') r USING (run_id)
            WHERE s.metric_name = '{a.metric}' AND r.arm_id IN ('{A}','{B}')"""
    ).fetchall()
    hashes = [
        h[0]
        for h in con.execute(
            f"SELECT DISTINCT scorer_hash FROM read_parquet('{store}/scores.parquet')"
        ).fetchall()
    ]
    gv = [
        g[0]
        for g in con.execute(
            f"SELECT DISTINCT graph_version FROM read_parquet('{store}/scores.parquet')"
        ).fetchall()
    ]
    print(f"scorer_hash(es): {[str(h)[:16] for h in hashes]}  graph_version(s): {gv}")
    if len(hashes) != 1:
        print("REFUSING: more than one scorer_hash in this store; cells must not be pooled.")
        return 1

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for suite, arm, task, val in rows:
        per[suite][(task, arm)].append(float(val))

    if not complete:
        print("(diagnostics only, contrast suppressed)")
        return 0

    for suite in sorted(per):
        cells = per[suite]
        tasks = {t for (t, _) in cells}
        deltas = []
        for t in sorted(tasks):
            va, vb = cells.get((t, A)), cells.get((t, B))
            if va and vb:
                deltas.append(statistics.fmean(vb) - statistics.fmean(va))
        if not deltas:
            print(f"{suite}: no shared tasks")
            continue
        point, lo, hi = bca_like(deltas, seed=0, n_boot=a.n_boot)
        pos, neg, ties, p = sign_test(deltas)
        distinct = len(set(round(d, 10) for d in deltas))
        verdict = "EXCLUDES ZERO" if (lo > 0 or hi < 0) else "spans zero"
        print(f"\n{suite}: {B} minus {A} on {a.metric}, paired over {len(deltas)} shared tasks")
        print(f"  delta {point:+.5f} [{lo:+.5f},{hi:+.5f}]  {verdict}")
        print(f"  discordant {pos + neg} (cap-aware better {pos}, worse {neg}), ties {ties}")
        print(f"  distinct per-task delta values: {distinct}")
        if distinct <= 8:
            print(f"  LATTICE-VALUED, so the sign test is what carries it: two-sided p={p:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
