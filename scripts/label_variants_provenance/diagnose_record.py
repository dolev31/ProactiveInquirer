"""Why lock (a) of `recompute.py` fails: what the label-variants record could have computed.

`artifacts/label_variants_heldout_20260919/RESULT.md` prints twelve symmetric matched-cost cells
that `recompute.py` does not reproduce on the population the record names, although the same
reader reproduces the paper's printed selected-arm cells exactly (lock b) and the gate's own
verdicts for three of these arms from another scoring pass (lock c). A failed point estimate --
not only a failed interval -- means the record's per-task values differ, so no bootstrap seed can
explain it. This script measures the candidate explanations on the same population and store and
prints POINT ESTIMATES ONLY: none of them is the estimand except the first, and none is a result.

  seed_matched     the estimand (= lock a's route A point)
  nxm_one_pin      every trained seed against every base seed of the task (`ladder._pair_keys`'s
                   N:M cross product), base pinned
  cross_seed_only  trained seed s against base seed 1-s only
  nxm_pooled_base  N:M, base selected by model id alone (`_select_runs`: 3,200 runs, 2 pins,
                   3 code versions)
  committed_pooled `scripts/decomposition_test/contrast.select_and_contrast_symmetric` as
                   committed, which selects by model id alone on both sides
  off_by_minus1 /  both arms read one rung below / above min(k_a, k_b)
  off_by_plus1
  rnr_resolve /    the matcher-record metrics `ladder.METRIC_FNS` defines, seed-matched
  rnr_ask / dwr

It also compares the record's quoted LEVELS (mean asks and mean coverage) with the named
population's, since a record whose levels do not match its population was not computed on it.

    PI_GOLD_ROOT="$PWD/data/gold" .venv/bin/python scripts/label_variants_provenance/diagnose_record.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompute as R  # noqa: E402
from decomposition_test.contrast import select_and_contrast_symmetric  # noqa: E402

from pinq_train import gate  # noqa: E402

# The record's quoted levels (musique asks/coverage; strategyqa asks), and its unmatched cell.
RECORD_LEVELS = {
    ("base", "musique"): {"asks": 6.59, "coverage": 0.8321},
    ("stopcontrast", "musique"): {"asks": 3.39, "coverage": 0.8967},
    ("stopcontrast", "strategyqa"): {"asks": 2.39},
    ("others", "musique"): {"asks": "2.63 to 2.67"},
    ("others", "strategyqa"): {"asks": "1.53 to 1.55"},
}
RECORD_UNMATCHED_STOPCONTRAST_MUSIQUE = 0.0652


def paired_points(
    ck: Sequence[Mapping[str, Any]],
    ba: Sequence[Mapping[str, Any]],
    ladder: Mapping[str, Mapping[int, float]],
    n_asks: Mapping[str, int],
    *,
    mode: str,
    off: int = 0,
) -> float:
    """Mean over tasks of the per-task mean paired difference, off the STORED frontier_q."""
    a_by, b_by = defaultdict(list), defaultdict(list)
    for r in ck:
        a_by[r["task_id"]].append(r)
    for r in ba:
        b_by[r["task_id"]].append(r)
    per_task = []
    for t in set(a_by) & set(b_by):
        ds = []
        for x in a_by[t]:
            for y in b_by[t]:
                if mode == "seed_matched" and x["seed"] != y["seed"]:
                    continue
                if mode == "cross_seed_only" and x["seed"] == y["seed"]:
                    continue
                k = max(0, min(n_asks[x["run_id"]], n_asks[y["run_id"]]) + off)
                lx, ly = ladder.get(x["run_id"], {}), ladder.get(y["run_id"], {})
                vx, vy = lx.get(min(k, max(lx, default=0))), ly.get(min(k, max(ly, default=0)))
                if vx is not None and vy is not None:
                    ds.append(vx - vy)
        if ds:
            per_task.append(sum(ds) / len(ds))
    return sum(per_task) / len(per_task) if per_task else float("nan")


def main() -> int:
    store = R.REPO / "scores" / "parquet"
    con = gate._con(store)
    runs = {
        label: R.select_arm(con, arm_id, model_id, split="test", code_prefix="107ef2221a55")[0]
        for label, (arm_id, model_id) in R.ARMS.items()
    }
    pooled_base = R.stop_lib.load_arm_runs(
        con, arm_id="inquirer_prompted", model_id="qwen3-8b-base", grid_name=R.GRID
    )
    every = [r for rs in runs.values() for r in rs] + pooled_base
    sh, gv = R.scorer_of(con, [r["run_id"] for r in every])
    ladder = gate._coverage_ladder(con, every, scorer_hash=sh)
    n_asks = {
        r["run_id"]: int(r["n_asks"] or 0)
        for r in gate._rows(
            con,
            "SELECT run_id, n_asks FROM runs WHERE run_id IN "
            + gate._in([r["run_id"] for r in every]),
        )
    }
    seeds = {r["run_id"]: int(r["seed"]) for r in every}
    print(f"store {R.rel(store)}  scorer_hash {sh}  graph_version {gv}")
    print(f"pinned base runs {len(runs['base'])}; base by model id alone {len(pooled_base)}")

    committed = {}
    for label in R.TRAINED:
        out = select_and_contrast_symmetric(
            store,
            arm_a_model_id=R.ARMS[label][1],
            arm_b_model_id="qwen3-8b-base",
            arm_b_arm="inquirer_prompted",
            scorer_hash=sh,
            seed=0,
            n_resamples=1,
        )
        committed[label] = {s: v["delta"] for s, v in out["by_suite"].items()}

    matches: dict[tuple[str, str], dict[str, float]] = {}
    lad_mod = R.symmetric_completed.ladder
    for s in R.SUITES:
        graphs = R.load_graphs(s, "v1")
        base_l = lad_mod.load_run_ladders(
            store, [r["run_id"] for r in runs["base"] if r["suite_id"] == s]
        )
        for label in R.TRAINED:
            ck_l = lad_mod.load_run_ladders(
                store, [r["run_id"] for r in runs[label] if r["suite_id"] == s]
            )
            matches[(label, s)] = {
                m: R.symmetric_completed.seed_matched_symmetric(
                    m, ck_l, base_l, graphs, seeds, seed=0, n_boot=1
                )["delta"]
                for m in ("rnr_resolve", "rnr_ask", "dwr")
            }

    cols = (
        "seed_matched nxm_one_pin cross_seed_only nxm_pooled_base committed_pooled "
        "off_by_minus1 off_by_plus1 rnr_resolve rnr_ask dwr"
    ).split()
    print("\n== point estimates, symmetric matched cost, arm minus prompted base")
    print(f"  {'cell':24s} {'record':>8s} " + " ".join(f"{c:>16s}" for c in cols))
    closest: list[float] = []
    for label in R.TRAINED:
        for s in R.SUITES:
            ck = [r for r in runs[label] if r["suite_id"] == s]
            ba = [r for r in runs["base"] if r["suite_id"] == s]
            bp = [r for r in pooled_base if r["suite_id"] == s]
            v = {
                "seed_matched": paired_points(ck, ba, ladder, n_asks, mode="seed_matched"),
                "nxm_one_pin": paired_points(ck, ba, ladder, n_asks, mode="nxm"),
                "cross_seed_only": paired_points(ck, ba, ladder, n_asks, mode="cross_seed_only"),
                "nxm_pooled_base": paired_points(ck, bp, ladder, n_asks, mode="nxm"),
                "committed_pooled": committed[label][s],
                "off_by_minus1": paired_points(ck, ba, ladder, n_asks, mode="seed_matched", off=-1),
                "off_by_plus1": paired_points(ck, ba, ladder, n_asks, mode="seed_matched", off=1),
                **matches[(label, s)],
            }
            rec = R.RECORD_CELLS[(label, s)][0]
            closest.append(min(abs(x - rec) for x in v.values()))
            print(
                f"  {label + '::' + s:24s} {rec:+8.4f} " + " ".join(f"{v[c]:+16.4f}" for c in cols)
            )
    n_hit = sum(1 for c in closest if c < 0.00005)
    print(
        f"\n  cells where ANY candidate matches the record at 4 dp: {n_hit} of {len(closest)}; "
        f"largest nearest-candidate gap {max(closest):.4f}"
    )

    print("\n== the record's quoted levels against the named population (mean over runs)")
    cov = gate._metric_by_run(con, R.METRIC, scorer_hash=sh)
    lv = {}
    for label in R.ARMS:
        for s in R.SUITES:
            rs = [r for r in runs[label] if r["suite_id"] == s]
            lv[(label, s)] = {
                "asks": sum(n_asks[r["run_id"]] for r in rs) / len(rs),
                "coverage": sum(cov[r["run_id"]] for r in rs) / len(rs),
            }
    for (label, s), want in RECORD_LEVELS.items():
        if label == "others":
            have = ", ".join(f"{lv[(x, s)]['asks']:.2f}" for x in ("control", "rater", "reaches"))
            print(f"  {label:12s} {s:10s} record asks {want['asks']:>12s}  here {have}")
            continue
        h = lv[(label, s)]
        line = f"  {label:12s} {s:10s} record asks {want['asks']:12.2f}  here {h['asks']:.2f}"
        if "coverage" in want:
            line += f"   record coverage {want['coverage']:.4f}  here {h['coverage']:.4f}"
        print(line)
    um = lv[("stopcontrast", "musique")]["coverage"] - lv[("base", "musique")]["coverage"]
    print(
        f"  unmatched stopcontrast - base, musique: record {RECORD_UNMATCHED_STOPCONTRAST_MUSIQUE:+.4f}"
        f"  here {um:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
