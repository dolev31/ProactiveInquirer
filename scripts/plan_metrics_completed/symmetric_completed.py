"""The SYMMETRIC reading of `dwr` and `precedence_violation_rate` on a completed comparator.

WHY THIS FILE EXISTS. `artifacts/baseline_completion_20260920` showed the published cells were
paired against a comparator arm short by 24 tasks on musique and 34 on wiki2.
`artifacts/plan_metrics_completed_20260920` recomputed all twelve matched cells on the completed
comparator, but on the ASYMMETRIC rule those cells were published under. Three rows of the paper's
table are on the SYMMETRIC rule instead, both arms at the lower question count, and mixing the two
in one column is how a table survives review while being wrong twice: the completed asymmetric dwr
on musique is +0.1313 and the published SYMMETRIC dwr on the short comparator is +0.13116, a
difference of 0.0001 between two different quantities on two different populations.

THE PAIRING IS NOT LADDER'S. `artifacts/symmetric_matched_cost_20260919/contrasts.symmetric.json`
records, in each cell's own `reason`, that the published symmetric numbers were computed with the
seed-matched `(suite, task, seed)` pairing and that `ladder.py`'s `_pair_keys` was deliberately NOT
used, because it keys on `(suite_id, task_id)` alone and averages every checkpoint seed against
every baseline seed of that task, an N:M cross product. Using `ladder.symmetric_contrast` here
would therefore compute a different estimand: run against the published cells its asymmetric
counterpart fails the lock on dwr for all three suites and on musique precedence, which is that
defect showing rather than a finding.

So this module reuses `sweep.seed_matched_contrast` unchanged for the asymmetric side, and
implements the symmetric side as the two-line variant of that same function: `k` becomes
`min(k_checkpoint, k_baseline)` and BOTH arms are read at it. Pairing, graph lookup, NaN dropping,
seed averaging into the task and the bootstrap are all the same code path.

THE LOCK IS DOUBLE, and that is the point. Before any completed number is reported, on the
PUBLISHED cohort the asymmetric reading must reproduce `published_value` and the symmetric reading
must reproduce `symmetric_delta`, each to `tol`. Reproducing both means this implementation is the
one that produced the paper's figures, not merely something that agrees on one column.

Readings on the completed cohort are taken at the repository's resample rule rather than the
published `n_boot` of 1000: 10,000 for every cell, and 50,000 across three bootstrap seeds so a
bound within 0.01 of zero is reported with its sign stability instead of as a decided cell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "plan_metrics_symmetric"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import duckdb  # noqa: E402
import ladder  # noqa: E402
from sweep import seed_matched_contrast  # noqa: E402

from pi_eval.stats.inference import paired_difference  # noqa: E402

METRICS = ("dwr", "precedence_violation_rate")
RESAMPLES: tuple[tuple[int, int], ...] = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))
NEAR_ZERO = 0.01


def seed_matched_symmetric(
    metric: str,
    ckpt: Mapping[str, Any],
    base: Mapping[str, Any],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """`sweep.seed_matched_contrast` with BOTH arms read at `min(k_checkpoint, k_baseline)`.

    The only difference from that function is the two lines that choose `k` and read the baseline.
    Everything else is copied deliberately rather than abstracted, so a reader can diff the two
    bodies and see that the pairing and the bootstrap did not change with the cost basis.
    """
    fn = ladder.METRIC_FNS[metric]

    def keyed(arm: Mapping[str, Any], what: str) -> dict[tuple[str, str, int], Any]:
        out: dict[tuple[str, str, int], Any] = {}
        for rid, lad in arm.items():
            key = (lad.suite_id, lad.task_id, int(seeds[rid]))
            if key in out:
                raise ValueError(
                    f"{what} has more than one run at {key}: {out[key].run_id} and {rid}."
                )
            out[key] = lad
        return out

    ck_by, ba_by = keyed(ckpt, "checkpoint arm"), keyed(base, "baseline arm")
    shared = sorted(set(ck_by) & set(ba_by))
    acc: dict[str, tuple[list[float], list[float]]] = {}
    dropped = 0
    for key in shared:
        _, task, _ = key
        graph = graphs.get(task)
        if graph is None:
            dropped += 1
            continue
        c_lad, b_lad = ck_by[key], ba_by[key]
        k = min(c_lad.n_asks, b_lad.n_asks)  # SYMMETRIC: the shared, smaller budget
        a_val = fn(c_lad.at(k), graph)
        b_val = fn(b_lad.at(k), graph)  # baseline at the SAME k, not at its own
        if math.isnan(a_val) or math.isnan(b_val):
            dropped += 1
            continue
        a_list, b_list = acc.setdefault(task, ([], []))
        a_list.append(a_val)
        b_list.append(b_val)
    per_a = {t: sum(a) / len(a) for t, (a, _) in acc.items()}
    per_b = {t: sum(b) / len(b) for t, (_, b) in acc.items()}
    est = paired_difference(per_a, per_b, n_boot=n_boot, seed=seed)
    return {
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "n": est.n,
        "n_pairs": len(shared) - dropped,
        "pairs_dropped": dropped,
    }


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _seed_map(store: Path) -> dict[str, int]:
    con = duckdb.connect()
    return {
        r[0]: int(r[1])
        for r in con.execute(
            f"SELECT run_id, seed FROM read_parquet('{(store / 'runs.parquet').as_posix()}')"
        ).fetchall()
    }


def run(
    *,
    published_cohort: Path,
    completed_cohort: Path,
    published_symmetric: Path,
    suites: Sequence[str],
    graph_version: str,
    tol: float,
) -> dict[str, Any]:
    from pi_eval.gold import load_graphs

    pub_cells = json.loads(published_symmetric.read_text())
    out: dict[str, Any] = {"lock": {}, "completed": {}, "stability": {}, "tol": tol}

    for suite in suites:
        graphs = load_graphs(suite, graph_version)
        pub_store, cmp_store = (
            published_cohort / "scores_parquet",
            completed_cohort / "scores_parquet",
        )
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

        for metric in METRICS:
            key = f"{metric}::{suite}"
            cell = pub_cells.get(f"{metric}::{suite}::matched")
            if cell is None or cell.get("symmetric_delta") is None:
                out["lock"][key] = {"verdict": "NO_PUBLISHED_SYMMETRIC_CELL"}
                continue
            asym = seed_matched_contrast(
                metric, pub_ck, pub_ba, graphs, pub_seeds, seed=0, n_boot=1000
            )
            sym = seed_matched_symmetric(
                metric, pub_ck, pub_ba, graphs, pub_seeds, seed=0, n_boot=1000
            )
            d_asym = abs(asym.delta - cell["published_value"])
            d_sym = abs(sym["delta"] - cell["symmetric_delta"])
            locked = d_asym <= tol and d_sym <= tol
            out["lock"][key] = {
                "verdict": "LOCKED" if locked else "FAILED",
                "published_value": cell["published_value"],
                "published_symmetric_delta": cell["symmetric_delta"],
                "published_n": cell["n"],
                "reproduced_asymmetric": asym.delta,
                "reproduced_symmetric": sym["delta"],
                "abs_diff_asymmetric": d_asym,
                "abs_diff_symmetric": d_sym,
                "reproduced_n": sym["n"],
            }
            if not locked:
                continue
            readings = [
                {
                    "n_boot": nb,
                    "seed": sd,
                    **seed_matched_symmetric(
                        metric, cmp_ck, cmp_ba, graphs, cmp_seeds, seed=sd, n_boot=nb
                    ),
                }
                for nb, sd in RESAMPLES
            ]
            out["completed"][key] = readings
            fifty = [x for x in readings if x["n_boot"] == 50000]
            los = [x["ci_lo"] for x in fifty]
            his = [x["ci_hi"] for x in fifty]
            ten = next(x for x in readings if x["n_boot"] == 10000)
            excludes = all(x > 0 for x in los) or all(x < 0 for x in his)
            out["stability"][key] = {
                "near_zero_at_10k": abs(ten["ci_lo"]) < NEAR_ZERO or abs(ten["ci_hi"]) < NEAR_ZERO,
                "ci_lo_range_at_50k": [min(los), max(los)],
                "ci_hi_range_at_50k": [min(his), max(his)],
                "verdict": "DECIDED" if excludes else "SPANS_ZERO",
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--published-cohort", type=Path, default=REPO / "artifacts" / "testsplit_qa")
    ap.add_argument("--completed-cohort", type=Path, required=True)
    ap.add_argument(
        "--published-symmetric",
        type=Path,
        default=REPO / "artifacts" / "symmetric_matched_cost_20260919" / "contrasts.symmetric.json",
    )
    ap.add_argument("--suites", nargs="+", default=["musique", "strategyqa", "wiki2"])
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rec = run(
        published_cohort=a.published_cohort,
        completed_cohort=a.completed_cohort,
        published_symmetric=a.published_symmetric,
        suites=a.suites,
        graph_version=a.graph_version,
        tol=a.tol,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    for k, v in sorted(rec["lock"].items()):
        if v["verdict"] == "LOCKED":
            print(
                f"lock {k:38s} LOCKED  asym |d|={v['abs_diff_asymmetric']:.2e} "
                f"sym |d|={v['abs_diff_symmetric']:.2e}  n={v['reproduced_n']}"
            )
        else:
            print(
                f"lock {k:38s} {v['verdict']}"
                + (
                    f"  asym |d|={v['abs_diff_asymmetric']:.2e} sym |d|={v['abs_diff_symmetric']:.2e}"
                    if "abs_diff_asymmetric" in v
                    else ""
                )
            )
    print()
    for k, readings in sorted(rec["completed"].items()):
        ten = next(x for x in readings if x["n_boot"] == 10000)
        st = rec["stability"][k]
        print(
            f"{k:38s} symmetric@10k {ten['delta']:+.6f} [{ten['ci_lo']:+.6f},{ten['ci_hi']:+.6f}]"
            f" n={ten['n']}  {st['verdict']}"
        )
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
