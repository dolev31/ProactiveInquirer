"""The two headline rows that have no symmetric reading on the completed comparator.

WHY THIS FILE EXISTS. The paper's headline table has five rows. Three of them
(`evidence_coverage`, `dwr`, `precedence_violation_rate`) are read on the SYMMETRIC rule, both arms
truncated to the shared lower question count. Two (`facet_breadth`, `max_depth_reached`) are read on
the published asymmetric helper rule, which reads the comparator at `min(k, its own asks)` and the
treatment at its own `k`. A reviewer objected, correctly, that the rows are then not one homogeneous
equal-spend experiment. This produces the missing symmetric readings so every row can be stated under
one rule on one population.

WHAT IT REUSES, AND WHY IT DOES NOT REIMPLEMENT. The pairing, truncation and bootstrap come from
`scripts/plan_metrics_symmetric/ladder.py` and from `scripts/plan_metrics_completed/
symmetric_completed.py::seed_matched_symmetric`, which belongs to another lane. Reimplementing the
truncation would make "our number differs" indistinguishable from "our code differs", which is the
whole reason the lock in that module exists.

THE LOCK IS WEAKER HERE AND SAYS SO. `contrasts.symmetric.json` records `symmetric_delta: null` for
both of these metrics, with a reason stating the symmetric form is computable but was "pending a
decision on whether the three-lock verification is worth doing for each of these metrics
individually". So there is no published symmetric float to lock against, and the DOUBLE lock the
sibling module enforces cannot be satisfied. Only the ASYMMETRIC half can be locked: the
reimplementation must reproduce the published asymmetric value before its symmetric number is
reported. Every cell this writes is therefore marked `SINGLE_LOCKED`, and the table must carry that
distinction rather than presenting these two rows as if they had the same standing as the other three.

ONE DEFINITION TRAP, MEASURED NOT ASSUMED. `facet_breadth` names two different functions.
`src/pi_eval/score.py` emits `float(touched)`, a COUNT, and puts the facet total in the row's `n`.
`ladder.METRIC_FNS["facet_breadth"]` is `touched/total`, a RATIO, NaN where the graph has no facets.
The published cells are the COUNT: on the completed comparator the ratio reading FAILS the lock
(`facet_breadth_probe.json`, abs diff 4.6e-3 on 2Wiki) while the count reading locks at 0.0 with
n = 200 on all three suites. So this reads `facet_breadth_scorer`, the count, registered at runtime
by `sweep.registered_metrics`. Using the bare name would silently compute the ratio and drop
facet-less tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "src",
    REPO / "scripts" / "plan_metrics_symmetric",
    REPO / "scripts" / "plan_metrics_completed",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ladder  # noqa: E402
import symmetric_completed  # noqa: E402
from sweep import (  # noqa: E402
    DIAGNOSTIC_METRIC_FNS,
    EXTRA_METRIC_FNS,
    seed_matched_contrast,
)
from symmetric_completed import _ids, _seed_map, seed_matched_symmetric  # noqa: E402


@contextmanager
def registered_everywhere(extra):
    """Register metric functions on EVERY module object that wraps `ladder.py`, then restore.

    ONE FILE, TWO MODULES. `sweep.py` does `from plan_metrics_symmetric import ladder` and
    `symmetric_completed.py` does a bare `import ladder`, so with both script directories on
    `sys.path` the same file is imported twice under two names -- `plan_metrics_symmetric.ladder`
    and `ladder` -- holding two DISTINCT `METRIC_FNS` dicts. `sweep.registered_metrics` updates
    one of them. `seed_matched_symmetric` reads the other. So the registration silently does
    nothing for the consumer that needs it, and the failure is a KeyError deep inside a reused
    function rather than at the registration site.

    It only bites for metrics that are NOT native to `ladder.METRIC_FNS`, which is why the sibling
    lane's `dwr` and `precedence_violation_rate` run never hit it: those need no registration.

    This registers on every distinct module object, refuses to shadow a name any of them already
    defines (the sibling module's rule, kept), restores all of them on the way out, and the caller
    asserts visibility before computing so a silent no-op cannot recur.
    """
    targets = []
    seen: set[int] = set()
    for mod in list(sys.modules.values()):
        table = getattr(mod, "METRIC_FNS", None)
        if isinstance(table, dict) and getattr(mod, "__file__", "") == ladder.__file__:
            if id(table) not in seen:
                seen.add(id(table))
                targets.append(table)
    if not targets:
        raise RuntimeError("no ladder METRIC_FNS table found to register into")
    for table in targets:
        clash = sorted(set(extra) & set(table))
        if clash:
            raise ValueError(f"METRIC_FNS already defines {clash}; refusing to shadow it")
    befores = [dict(table) for table in targets]
    for table in targets:
        table.update(extra)
    try:
        yield len(targets)
    finally:
        for table, before in zip(targets, befores):
            table.clear()
            table.update(before)


# (metric as the ladder knows it, the published cell's key, the label the paper uses)
TARGETS = (
    ("facet_breadth_scorer", "facet_breadth", "breadth"),
    ("max_depth_reached", "max_depth_reached", "deepest need resolved"),
)
RESAMPLES: tuple[tuple[int, int], ...] = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))
NEAR_ZERO = 0.01


def run(
    *,
    published_cohort: Path,
    completed_cohort: Path,
    published_asymmetric: Path,
    suites: Sequence[str],
    graph_version: str,
    tol: float,
) -> dict[str, Any]:
    from pi_eval.gold import load_graphs

    published = json.loads(published_asymmetric.read_text())
    pub_by_key = {
        (r["metric"], r["suite"], r["basis"]): r for r in published if isinstance(r, dict)
    }
    out: dict[str, Any] = {
        "lock": {},
        "completed": {},
        "stability": {},
        "tol": tol,
        "lock_kind": (
            "SINGLE: the asymmetric half only. contrasts.symmetric.json carries no published "
            "symmetric float for either metric, so the double lock the sibling module enforces "
            "cannot be satisfied and is not claimed."
        ),
    }

    for suite in suites:
        graphs = load_graphs(suite, graph_version)
        pub_store = published_cohort / "scores_parquet"
        cmp_store = completed_cohort / "scores_parquet"
        pub_seeds, cmp_seeds = _seed_map(pub_store), _seed_map(cmp_store)
        pub_ck = ladder.load_run_ladders(
            pub_store, _ids(published_cohort / f"run_ids.trained.{suite}.txt")
        )
        pub_ba = ladder.load_run_ladders(
            pub_store, _ids(published_cohort / f"run_ids.prompted.{suite}.txt")
        )
        cmp_ck = ladder.load_run_ladders(
            cmp_store, _ids(completed_cohort / f"run_ids.trained.{suite}.txt")
        )
        cmp_ba = ladder.load_run_ladders(
            cmp_store, _ids(completed_cohort / f"run_ids.prompted.{suite}.txt")
        )

        for metric, pub_metric, _label in TARGETS:
            key = f"{metric}::{suite}"
            cell = pub_by_key.get((pub_metric, suite, "matched"))
            if cell is None:
                out["lock"][key] = {"verdict": "NO_PUBLISHED_CELL"}
                continue
            pub_value = cell["task"]["delta"]

            # STAGE 1, the lock: reproduce the published ASYMMETRIC value on the published cohort.
            asym = seed_matched_contrast(
                metric, pub_ck, pub_ba, graphs, pub_seeds, seed=0, n_boot=1000
            )
            d_asym = abs(asym.delta - pub_value)
            locked = d_asym <= tol
            out["lock"][key] = {
                "verdict": "SINGLE_LOCKED" if locked else "FAILED",
                "published_asymmetric": pub_value,
                "reproduced_asymmetric": asym.delta,
                "abs_diff_asymmetric": d_asym,
                "published_n": cell["task"]["n"],
                "reproduced_n": asym.n,
                "published_symmetric": None,
                "note": "no published symmetric float exists for this metric; symmetric is single-locked",
            }
            if not locked:
                continue

            # STAGE 2, the reading: symmetric on the COMPLETED comparator.
            readings = []
            for n_boot, seed in RESAMPLES:
                r = seed_matched_symmetric(
                    metric, cmp_ck, cmp_ba, graphs, cmp_seeds, seed=seed, n_boot=n_boot
                )
                readings.append(
                    {
                        "n_boot": n_boot,
                        "seed": seed,
                        "delta": r["delta"],
                        "ci_lo": r["ci_lo"],
                        "ci_hi": r["ci_hi"],
                        "n": r["n"],
                        "dropped": r.get("dropped", 0),
                    }
                )
            out["completed"][key] = readings

            ten = next(x for x in readings if x["n_boot"] == 10000)
            fifty = [x for x in readings if x["n_boot"] == 50000]
            spans = [(x["ci_lo"] <= 0 <= x["ci_hi"]) for x in fifty]
            out["stability"][key] = {
                "verdict": (
                    "SPANS_ZERO"
                    if any(spans)
                    else ("DECIDED" if not (ten["ci_lo"] <= 0 <= ten["ci_hi"]) else "SPANS_ZERO")
                ),
                "near_zero_at_10k": min(abs(ten["ci_lo"]), abs(ten["ci_hi"])) < NEAR_ZERO,
                "ci_lo_range_at_50k": [
                    min(x["ci_lo"] for x in fifty),
                    max(x["ci_lo"] for x in fifty),
                ],
                "ci_hi_range_at_50k": [
                    min(x["ci_hi"] for x in fifty),
                    max(x["ci_hi"] for x in fifty),
                ],
                "disagrees_across_50k_seeds": len(set(spans)) > 1,
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published-cohort", type=Path, default=REPO / "artifacts/testsplit_qa")
    ap.add_argument(
        "--completed-cohort",
        type=Path,
        default=REPO / "artifacts/completed_cohort_20260922/cohort",
    )
    ap.add_argument(
        "--published-asymmetric",
        type=Path,
        default=REPO / "artifacts/testsplit_plan_metrics_20260918/contrasts.json",
    )
    ap.add_argument("--suites", nargs="+", default=["musique", "strategyqa", "wiki2"])
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    extra = {**EXTRA_METRIC_FNS, **DIAGNOSTIC_METRIC_FNS}
    with registered_everywhere(extra) as n_tables:
        # The guard: assert the metric is visible from the module the REUSED reader looks in,
        # not merely from the one we registered through.
        for metric, _pub, _label in TARGETS:
            if metric not in symmetric_completed.ladder.METRIC_FNS:
                raise RuntimeError(
                    f"{metric} is not visible to symmetric_completed's ladder view after "
                    f"registering on {n_tables} table(s). Refusing to run: this is the silent "
                    "no-op the context manager exists to prevent."
                )
        print(f"registered {len(extra)} metric functions on {n_tables} ladder module object(s)")
        rec = run(
            published_cohort=a.published_cohort,
            completed_cohort=a.completed_cohort,
            published_asymmetric=a.published_asymmetric,
            suites=a.suites,
            graph_version=a.graph_version,
            tol=a.tol,
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")

    for k, v in sorted(rec["lock"].items()):
        extra = (
            f"  asym |d|={v['abs_diff_asymmetric']:.2e}  n={v['reproduced_n']}"
            if "abs_diff_asymmetric" in v
            else ""
        )
        print(f"lock {k:34s} {v['verdict']}{extra}")
    print()
    for k, readings in sorted(rec["completed"].items()):
        ten = next(x for x in readings if x["n_boot"] == 10000)
        st = rec["stability"][k]
        print(
            f"{k:34s} symmetric@10k {ten['delta']:+.6f} [{ten['ci_lo']:+.6f},{ten['ci_hi']:+.6f}]"
            f" n={ten['n']}  {st['verdict']}"
        )
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
