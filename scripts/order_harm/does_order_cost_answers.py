"""Does resolving needs out of prerequisite order cost anything downstream?

WHY. The paper reports that the trained questioner takes needs out of prerequisite order more often
on the multi-hop suite, and frames that as an operational risk without quantifying it. A reviewer
asked directly: can the downstream harm be measured, for instance as wrong final answers conditioned
on unconfirmed premises. This answers that from the existing held-out population, with no new
rollouts.

THE DESIGN, AND THE CONFOUND IT IS BUILT AROUND. A raw association between out-of-order resolution
and answer failure is confounded by task difficulty: a harder task invites both. So nothing here
compares tasks to other tasks. Every reading is PAIRED WITHIN TASK -- the trained arm against the
same weights prompted on the same task -- and the tasks are then STRATIFIED by whether the trained
arm resolved MORE out of order than the comparator did on that same task. If out-of-order resolution
carries downstream harm, the arm's answer deficit should be worse in the stratum where it resolved
more out of order. Task difficulty is held fixed inside each pair, so it cannot produce that pattern.

WHAT WOULD FALSIFY THE HARM CLAIM. Equal answer deltas across the two strata. That is a real
possible outcome and is reported as such: the paper's existing claim is that the ORDER RATE is worse,
not that it costs anything, and this script is allowed to confirm that the cost is unmeasurable here.

READ THE NEGATIVE CASE CAREFULLY. An interval covering zero on a stratum with few tasks is not
evidence of no harm, it is absence of evidence. Both strata's task counts are printed beside every
cell for exactly that reason.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAINED, PROMPTED = "inquirer_trained", "inquirer_prompted"
ANSWER_METRICS = ("answer_correct", "answer_token_f1", "answer_token_recall", "task_success")
N_BOOT = 10000


def boot(values: Sequence[float], n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Sorted-input bootstrap. The sort is load-bearing: duckdb returns rows in no guaranteed
    order, and resampling indices into an order-dependent array gives identical point estimates
    with moving bounds, which this repository has recorded as a real defect."""
    a = np.sort(np.asarray(values, dtype=float))
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    m = np.sort(a[idx].mean(axis=1))
    return float(a.mean()), float(m[int(0.025 * n)]), float(m[int(0.975 * n)])


def load(store: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """(suite, task, arm) -> metric means over that arm's seeds on that task."""
    con = duckdb.connect()
    wanted = ("precedence_violation_rate", *ANSWER_METRICS)
    rows = con.execute(
        f"""
        select r.suite_id, r.task_id, r.arm_id, s.metric_name, avg(s.value)
        from '{(store / "runs.parquet").as_posix()}' r
        join '{(store / "scores.parquet").as_posix()}' s using (run_id)
        where s.metric_name in {wanted} and s.value is not null
        group by 1,2,3,4
        """
    ).fetchall()
    out: dict[tuple[str, str, str], dict[str, float]] = collections.defaultdict(dict)
    for suite, task, arm, metric, value in rows:
        out[(suite, task, arm)][metric] = float(value)
    return out


def run(store: Path, suites: Sequence[str]) -> dict[str, Any]:
    return run_cells(load(store), suites)


def run_cells(
    cells: dict[tuple[str, str, str], dict[str, float]], suites: Sequence[str]
) -> dict[str, Any]:
    """The reading itself, over (suite, task, arm) -> metric means however they were loaded, so a
    caller that averages training seeds within the task first reads through the same code."""
    result: dict[str, Any] = {}
    for suite in suites:
        tasks = sorted({t for (s, t, _a) in cells if s == suite})
        strata: dict[str, list[tuple[str, dict[str, float]]]] = {
            "more_out_of_order": [],
            "not_more": [],
        }
        n_undefined = 0
        for task in tasks:
            tr = cells.get((suite, task, TRAINED), {})
            pr = cells.get((suite, task, PROMPTED), {})
            if "precedence_violation_rate" not in tr or "precedence_violation_rate" not in pr:
                n_undefined += 1
                continue
            d_order = tr["precedence_violation_rate"] - pr["precedence_violation_rate"]
            deltas = {m: tr[m] - pr[m] for m in ANSWER_METRICS if m in tr and m in pr}
            key = "more_out_of_order" if d_order > 0 else "not_more"
            strata[key].append((task, deltas))

        suite_out: dict[str, Any] = {
            "n_tasks_with_order_defined_on_both_arms": sum(len(v) for v in strata.values()),
            "n_tasks_order_undefined": n_undefined,
            "strata_sizes": {k: len(v) for k, v in strata.items()},
            "answer_delta_by_stratum": {},
            "difference_between_strata": {},
        }
        for metric in ANSWER_METRICS:
            per_stratum = {}
            for name, items in strata.items():
                vals = [d[metric] for _t, d in items if metric in d]
                pt, lo, hi = boot(vals)
                per_stratum[name] = {
                    "n": len(vals),
                    "delta": pt,
                    "ci": [lo, hi],
                    "excludes_zero": bool(len(vals) and (lo > 0 or hi < 0)),
                }
            suite_out["answer_delta_by_stratum"][metric] = per_stratum
            # The harm claim lives in the DIFFERENCE between strata, not in either one alone.
            a = [d[metric] for _t, d in strata["more_out_of_order"] if metric in d]
            b = [d[metric] for _t, d in strata["not_more"] if metric in d]
            if len(a) >= 5 and len(b) >= 5:
                rng = np.random.default_rng(0)
                a_s, b_s = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
                ia = rng.integers(0, len(a_s), (N_BOOT, len(a_s)))
                ib = rng.integers(0, len(b_s), (N_BOOT, len(b_s)))
                diffs = np.sort(a_s[ia].mean(axis=1) - b_s[ib].mean(axis=1))
                pt = float(a_s.mean() - b_s.mean())
                lo, hi = float(diffs[int(0.025 * N_BOOT)]), float(diffs[int(0.975 * N_BOOT)])
                suite_out["difference_between_strata"][metric] = {
                    "delta": pt,
                    "ci": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_more": len(a_s),
                    "n_not_more": len(b_s),
                    "note": (
                        "INDEPENDENT-samples difference of two paired deltas: the two strata are "
                        "different task sets, so this is not itself a paired reading and its "
                        "interval is wider than a paired one would be."
                    ),
                }
            else:
                suite_out["difference_between_strata"][metric] = {
                    "verdict": "NOT_READ",
                    "reason": f"stratum too small: {len(a)} and {len(b)} tasks",
                }
        result[suite] = suite_out
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--store", type=Path, default=REPO / "artifacts/completed_cohort_20260922/scores_parquet"
    )
    ap.add_argument("--suites", nargs="+", default=["musique", "strategyqa", "wiki2"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rec = run(a.store, a.suites)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")

    for suite, s in rec.items():
        print(f"\n=== {suite} ===")
        print(
            f"  tasks with the order rate defined on both arms: "
            f"{s['n_tasks_with_order_defined_on_both_arms']}"
            f"   (undefined on {s['n_tasks_order_undefined']})"
        )
        print(
            f"  trained resolved MORE out of order on {s['strata_sizes']['more_out_of_order']} "
            f"tasks, not more on {s['strata_sizes']['not_more']}"
        )
        for metric in ANSWER_METRICS:
            per = s["answer_delta_by_stratum"][metric]
            m, nm = per["more_out_of_order"], per["not_more"]
            print(
                f"    {metric:20s} more:{m['delta']:+.4f} [{m['ci'][0]:+.4f},{m['ci'][1]:+.4f}] "
                f"n={m['n']:3d}   not-more:{nm['delta']:+.4f} "
                f"[{nm['ci'][0]:+.4f},{nm['ci'][1]:+.4f}] n={nm['n']:3d}"
            )
            d = s["difference_between_strata"][metric]
            if "delta" in d:
                verdict = "EXCLUDES 0" if d["excludes_zero"] else "spans 0"
                print(
                    f"      {'difference':20s} {d['delta']:+.4f} "
                    f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]  {verdict}"
                )
            else:
                print(f"      {'difference':20s} {d['verdict']}: {d['reason']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
