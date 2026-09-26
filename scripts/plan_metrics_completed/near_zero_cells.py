"""Intervals for the near-zero cells on the completed comparator, on the PUBLISHED asymmetric basis.

WHY A SECOND SCRIPT. `symmetric_completed.py` covers the two metrics whose published cells are on
the symmetric rule. Three other cells sit close enough to zero that a point estimate is not a
verdict, and they are published on the asymmetric rule only:

    newly_reachable_share    musique   published +0.0058, completed point -0.0077 (a SIGN change)
    precedence_violation_rate wiki2    published +0.0199, completed point +0.0023
    stop_overshoot           wiki2     published -0.0120, completed point -0.0175

`sweep.py` records no `seedmatched_ci_*` field, so re-running it at higher resample counts moves
nothing: the point estimate is not a function of the resample count and the interval it does record
belongs to `ladder`'s cross-product pairing, which is a different estimand. Those runs therefore
cannot decide these cells, which is why this script exists rather than another sweep.

WHAT IT DOES. `sweep.seed_matched_contrast`, unchanged, on the published cohort as a lock against
the published value, then on the completed cohort at 10,000 resamples and at 50,000 across three
bootstrap seeds. Two of the three metrics are not native to `ladder.METRIC_FNS`, so the whole run
sits inside `sweep.registered_metrics`, exactly as `sweep.main` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "plan_metrics_symmetric"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import duckdb  # noqa: E402
import ladder  # noqa: E402
from sweep import (  # noqa: E402
    DIAGNOSTIC_METRIC_FNS,
    EXTRA_METRIC_FNS,
    load_coverage_ladders,
    published_matched_cells,
    registered_metrics,
    seed_matched_contrast,
)

COVERAGE_LADDER_METRICS = {"evidence_coverage", "stop_overshoot"}
CELLS: tuple[tuple[str, str], ...] = (
    ("newly_reachable_share", "musique"),
    ("precedence_violation_rate", "wiki2"),
    ("stop_overshoot", "wiki2"),
)
RESAMPLES: tuple[tuple[int, int], ...] = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))


def _ids(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def _seeds(store: Path) -> dict[str, int]:
    con = duckdb.connect()
    return {
        r[0]: int(r[1])
        for r in con.execute(
            f"SELECT run_id, seed FROM read_parquet('{(store / 'runs.parquet').as_posix()}')"
        ).fetchall()
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published-cohort", type=Path, default=REPO / "artifacts" / "testsplit_qa")
    ap.add_argument("--completed-cohort", type=Path, required=True)
    ap.add_argument(
        "--published",
        type=Path,
        default=REPO / "artifacts" / "testsplit_plan_metrics_20260918" / "contrasts.json",
    )
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        metavar="METRIC::SUITE",
        help="override the default near-zero cell list",
    )
    a = ap.parse_args(argv)

    from pi_eval.gold import load_graphs

    published = published_matched_cells(json.loads(a.published.read_text()))
    pub_store, cmp_store = (
        a.published_cohort / "scores_parquet",
        a.completed_cohort / "scores_parquet",
    )
    pub_seeds, cmp_seeds = _seeds(pub_store), _seeds(cmp_store)
    out: dict[str, Any] = {}

    with registered_metrics({**EXTRA_METRIC_FNS, **DIAGNOSTIC_METRIC_FNS}):
        cells = tuple(tuple(c.split("::", 1)) for c in a.cells) if a.cells else CELLS
        for metric, suite in cells:
            graphs = load_graphs(suite, a.graph_version)
            cov = metric in COVERAGE_LADDER_METRICS
            build = (
                (lambda st, ids: load_coverage_ladders(st, ids, graphs))
                if cov
                else ladder.load_run_ladders
            )
            pub_ck = build(pub_store, _ids(a.published_cohort / f"run_ids.trained.{suite}.txt"))
            pub_ba = build(pub_store, _ids(a.published_cohort / f"run_ids.prompted.{suite}.txt"))
            cmp_ck = build(cmp_store, _ids(a.completed_cohort / f"run_ids.trained.{suite}.txt"))
            cmp_ba = build(cmp_store, _ids(a.completed_cohort / f"run_ids.prompted.{suite}.txt"))

            # A `_scorer` variant computes the PUBLISHED definition of the same quantity
            # (pi_eval.score's count) while `ladder` holds a ratio under the bare name, so the
            # published cell to lock against is the bare one.
            published_name = metric[: -len("_scorer")] if metric.endswith("_scorer") else metric
            cell = published.get((published_name, suite))
            lock = seed_matched_contrast(
                metric, pub_ck, pub_ba, graphs, pub_seeds, seed=0, n_boot=1000
            )
            d = abs(lock.delta - cell.published_delta)
            key = f"{metric}::{suite}"
            out[key] = {
                "published_delta": cell.published_delta,
                "published_n": cell.published_n,
                "lock_reproduced": lock.delta,
                "lock_abs_diff": d,
                "lock_verdict": "LOCKED" if d <= a.tol else "FAILED",
                "readings": [],
            }
            if d > a.tol:
                print(f"{key:40s} LOCK FAILED |d|={d:.2e}")
                continue
            for nb, sd in RESAMPLES:
                r = seed_matched_contrast(
                    metric, cmp_ck, cmp_ba, graphs, cmp_seeds, seed=sd, n_boot=nb
                )
                out[key]["readings"].append(
                    {
                        "n_boot": nb,
                        "seed": sd,
                        "delta": r.delta,
                        "ci_lo": r.ci_lo,
                        "ci_hi": r.ci_hi,
                        "n": r.n,
                    }
                )
            fifty = [x for x in out[key]["readings"] if x["n_boot"] == 50000]
            los, his = [x["ci_lo"] for x in fifty], [x["ci_hi"] for x in fifty]
            decided = all(x > 0 for x in los) or all(x < 0 for x in his)
            out[key]["verdict"] = "DECIDED" if decided else "SPANS_ZERO"
            out[key]["ci_lo_range_at_50k"] = [min(los), max(los)]
            out[key]["ci_hi_range_at_50k"] = [min(his), max(his)]
            ten = out[key]["readings"][0]
            print(
                f"{key:40s} lock |d|={d:.0e}  published {cell.published_delta:+.6f} n={cell.published_n}"
                f"  ->  {ten['delta']:+.6f} [{ten['ci_lo']:+.6f},{ten['ci_hi']:+.6f}] n={ten['n']}"
                f"  {out[key]['verdict']}"
            )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
