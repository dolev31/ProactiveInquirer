"""Composed MuSiQue pairs (musique_x2): the declared primary cells and the interference DiD.

WHAT IT READS. Three scored stores of the musique_x2 grid (conf/grids/musique_x2_20260923.yaml),
one per pi-run invocation: the recipe at training seeds 1 and 2
(`inquirer_trained` @ qwen3-8b-dpo-stacked-notdone-both-s1 / -s2) and Qwen3-8B base prompted
(`inquirer_prompted` @ qwen3-8b-base). Each arm is (store, runs root, arm_id, inquirer model),
`read_contrasts.parse_arm`'s spec, and passes that module's population and contrast checks:
one code_version, one base_url_sha, identical frozen prompts, no dev- or dirty run, no canary
hit, one run per (suite, task, seed), split test, and the inquirer model every run's manifest pins.

THE PRIMARY CELLS (declared in artifacts/composed_pairs_20260923/DECLARATION.md before any data):
  facet_breadth_scorer  facets advanced, of 2 (the scorer's count at the resolve level)
  evidence_coverage     required-evidence coverage
  dwr                   depth-weighted recall
each as recipe (s1 and s2 averaged within the task) minus base, at equal realized spend under
Table 3's symmetric seed-matched rule (both arms at min(k_a, k_b) per (task, rollout seed)),
PAIRED BOOTSTRAP CLUSTERED ON THE PAIR (both question orders one cluster), at 1k and 10k
resamples (seed 0) and 50k at seeds 101, 202, 303. DECIDED only if every 50k interval excludes 0
on the same side. Every cell is reported whatever its sign.

THE INTERFERENCE DiD, against scripts/composed_pairs/predict.py's output, per composed task:
    (recipe composed - recipe predicted) - (base composed - base predicted)
  = (composed delta) - (predicted delta),
bootstrapped with the same clustering, over tasks that HAVE a prediction (the rest are counted,
never imputed). Primary basis: symmetric (the declared primary rule on both sides). Secondary:
own stop, every run at k = n_asks on both sides.

THE GRAPHS are musique_x2's, from PI_GOLD_ROOT pointed at the composed build's gold tree.
`graphs_for` REFUSES an empty dict and one missing any task that has runs:
scripts/two_axis_recipe/analyze_shape.py hard-codes `load_graphs("synth")`, which returns {}
silently on any other suite, and every ladder metric then reads as "no graph" instead of failing.

THE LOCK, before any composed cell: this reader's contrast path (`predict.pooled_levels` +
`paired_difference`, clusters None) on the Table 3 musique stores must reproduce
table1_by_seed.json's pooled (s1+s2) musique evidence_coverage and dwr cells -- point and both
bounds at every recorded (n_boot, seed) -- to `--tol`. It needs musique's gold, so it loads it
from `--lock-gold-root` through the same `load_graphs`, then restores PI_GOLD_ROOT.

    PI_GOLD_ROOT=<x2 root>/data/gold python scripts/composed_pairs/analyze.py \\
      --arm s1=<store_s1>:<runs_s1>:inquirer_trained:qwen3-8b-dpo-stacked-notdone-both-s1 \\
      --arm s2=<store_s2>:<runs_s2>:inquirer_trained:qwen3-8b-dpo-stacked-notdone-both-s2 \\
      --arm base=<store_b>:<runs_b>:inquirer_prompted:qwen3-8b-base \\
      --prediction <dir>/prediction.json --lock-gold-root <main>/data/gold \\
      --artifacts-root <main> --out <dir>/composed.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts" / "structured_baselines", REPO / "scripts" / "composed_pairs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import predict  # noqa: E402
import read_contrasts as rc  # noqa: E402

from pi_eval.gold import load_graphs  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402
from pinq_adapters.musique_x2.suite import pair_of  # noqa: E402

SUITE = "musique_x2"
METRICS: tuple[str, ...] = ("facet_breadth_scorer", "evidence_coverage", "dwr")
DID_METRICS: tuple[str, ...] = ("evidence_coverage", "dwr")
RESAMPLES: tuple[tuple[int, int], ...] = rc.RESAMPLES
NEAR_ZERO = 0.01
Refusal = rc.Refusal


# ------------------------------------------------------------------ the pieces that decide


def pair_clusters(task_ids: Iterable[str]) -> dict[str, str]:
    """task -> pair id. A task id that is not a composed id is a refusal, never its own cluster:
    a silently singleton cluster is the same arithmetic as no clustering at all."""
    out: dict[str, str] = {}
    for t in task_ids:
        p = pair_of(str(t))
        if p is None:
            raise Refusal(f"{t!r} is not a musique_x2 task id; cannot cluster it on a pair")
        out[str(t)] = p
    return out


def graphs_for(suite: str, version: str, task_ids: Iterable[str]) -> dict[str, Any]:
    graphs = load_graphs(suite, version)
    if not graphs:
        raise Refusal(
            f"no graphs for {suite} {version} under PI_GOLD_ROOT={os.environ.get('PI_GOLD_ROOT')}"
        )
    missing = sorted(set(map(str, task_ids)) - set(graphs))
    if missing:
        raise Refusal(f"{len(missing)} task(s) with runs have no gold graph: {missing[:5]}")
    return graphs


def _readings(
    a: Mapping[str, float],
    b: Mapping[str, float],
    clusters: Mapping[str, str] | None,
    resamples: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    out = []
    for nb, sd in resamples:
        est = paired_difference(
            dict(sorted(a.items())),
            dict(sorted(b.items())),
            clusters=clusters,
            n_boot=nb,
            seed=sd,
        )
        out.append(
            {
                "n_boot": nb,
                "seed": sd,
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                "note": est.note,
            }
        )
    return out


def _side(x: Mapping[str, Any]) -> int:
    return (x["ci_lo"] > 0) - (x["ci_hi"] < 0)


def verdict(readings: Sequence[Mapping[str, Any]], *, n_clusters: int) -> dict[str, Any]:
    """DECIDED only if every 50k interval excludes 0 on the same side, AND the bootstrap had
    something to resample: at least two clusters and no zero-width 50k interval. Measured on the
    2-task smoke (one pair): every resample was identical, the interval was [-0.125, -0.125],
    and it read DECIDED."""
    fifty = [x for x in readings if x["n_boot"] == 50000]
    degenerate = n_clusters < 2 or any(x["ci_hi"] - x["ci_lo"] <= 0 for x in fifty)
    decided = (
        bool(fifty)
        and not degenerate
        and (all(x["ci_lo"] > 0 for x in fifty) or all(x["ci_hi"] < 0 for x in fifty))
    )
    near = [x for x in readings if min(abs(x["ci_lo"]), abs(x["ci_hi"])) <= NEAR_ZERO]
    one_k = [x for x in readings if x["n_boot"] == 1000]
    flips = None
    if near and one_k:
        flips = any(_side(x) != _side(one_k[0]) for x in readings)
    return {
        "verdict": "DECIDED" if decided else ("DEGENERATE" if degenerate else "SPANS_ZERO"),
        "bound_within_0.01_of_zero": bool(near),
        "exclusion_flips_across_resample_counts": flips,
    }


def did_cell(
    per_a: Mapping[str, float],
    per_b: Mapping[str, float],
    pred_recipe: Mapping[str, float],
    pred_base: Mapping[str, float],
    *,
    resamples: Sequence[tuple[int, int]] = RESAMPLES,
) -> dict[str, Any]:
    """(composed delta) - (predicted delta), per task with a prediction, clustered on the pair."""
    both = set(per_a) & set(per_b)
    tasks = sorted(t for t in both if t in pred_recipe and t in pred_base)
    comp = {t: per_a[t] - per_b[t] for t in tasks}
    pred = {t: pred_recipe[t] - pred_base[t] for t in tasks}
    clusters = pair_clusters(tasks)
    readings = _readings(comp, pred, clusters, resamples)
    return {
        "readings": readings,
        **verdict(readings, n_clusters=len(set(clusters.values()))),
        "n_tasks": len(tasks),
        "n_clusters": len(set(clusters.values())),
        "n_tasks_without_prediction": len(both) - len(tasks),
        "mean_composed_delta": sum(comp.values()) / len(comp) if comp else None,
        "mean_predicted_delta": sum(pred.values()) / len(pred) if pred else None,
    }


# ------------------------------------------------------------------ the lock


@contextlib.contextmanager
def _gold_root(path: Path) -> Iterator[None]:
    before = os.environ.get("PI_GOLD_ROOT")
    os.environ["PI_GOLD_ROOT"] = str(path)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PI_GOLD_ROOT", None)
        else:
            os.environ["PI_GOLD_ROOT"] = before


def _table3_ids(artifacts_root: Path) -> dict[str, tuple[Path, list[str]]]:
    cohort = artifacts_root / "artifacts/completed_cohort_20260922"
    seedrep = artifacts_root / "artifacts/seedrep_gate_20260919"
    return {
        "s1": (
            seedrep / "scores_parquet",
            rc._ids(seedrep / f"run_ids/run_ids.{predict.RECIPE[0]}.musique.txt"),
        ),
        "s2": (
            seedrep / "scores_parquet",
            rc._ids(seedrep / f"run_ids/run_ids.{predict.RECIPE[1]}.musique.txt"),
        ),
        "base": (
            cohort / "scores_parquet",
            rc._ids(cohort / "cohort/run_ids.prompted.musique.txt"),
        ),
    }


def table1_lock(artifacts_root: Path, lock_gold_root: Path, tol: float) -> dict[str, Any]:
    """This reader's contrast path on Table 3's musique stores, against table1_by_seed.json."""
    table1 = json.loads((artifacts_root / predict.TABLE1).read_text())["cells"]["s1s2"]
    with _gold_root(lock_gold_root):
        graphs = load_graphs("musique", "v1")
    if not graphs:
        raise Refusal(f"lock: no musique graphs under {lock_gold_root}")
    ids = _table3_ids(artifacts_root)
    seeds: dict[str, int] = {}
    for store in sorted({s for s, _ in ids.values()}):
        seeds.update(rc._seed_map(store))
    out: dict[str, Any] = {}
    extra = {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    with rc.sweep.registered_metrics(extra):
        lad = {arm: rc._ladders(store, run_ids, graphs) for arm, (store, run_ids) in ids.items()}
        for m in ("evidence_coverage", "dwr"):
            idx = 1 if m in rc.COVERAGE_METRICS else 0
            per_a, per_b, n_pairs = predict.pooled_levels(
                m, [lad["s1"][idx], lad["s2"][idx]], lad["base"][idx], graphs, seeds
            )
            rows = []
            for want in table1[f"{m}::musique"]:
                got = _readings(per_a, per_b, None, ((want["n_boot"], want["seed"]),))[0]
                diffs = {k: abs(got[k] - want[k]) for k in ("delta", "ci_lo", "ci_hi")}
                ok = (
                    max(diffs.values()) <= tol
                    and got["n"] == want["n"]
                    and n_pairs == want["n_pairs"]
                )
                rows.append(
                    {
                        "n_boot": want["n_boot"],
                        "seed": want["seed"],
                        "published": {k: want[k] for k in ("delta", "ci_lo", "ci_hi")},
                        "reproduced": {k: got[k] for k in ("delta", "ci_lo", "ci_hi")},
                        "n": got["n"],
                        "n_pairs": n_pairs,
                        "max_abs_diff": max(diffs.values()),
                        "ok": ok,
                    }
                )
            out[f"{m}::musique"] = rows
    failed = [k for k, rows in out.items() if not all(r["ok"] for r in rows)]
    if failed:
        raise Refusal(f"table1 lock FAILED on {failed}: {json.dumps(out, indent=1)}")
    return out


# ------------------------------------------------------------------ main


def _pred_maps(prediction: Mapping[str, Any], metric: str) -> dict[str, dict[str, dict]]:
    tasks = prediction["tasks"]
    return {
        basis: {
            arm: {t: e["pred"][arm][basis][metric] for t, e in tasks.items() if e["pred"]}
            for arm in predict.ARMS
        }
        for basis in predict.BASES
    }


def run(
    arms: Sequence[rc.ArmSpec],
    *,
    recipe: Sequence[str],
    base: str,
    prediction: Mapping[str, Any] | None,
    graph_version: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    rows = {a.label: rc.load_arm(a, (SUITE,)) for a in arms}
    fail = [f for a in arms for f in rc.population_failures(rows[a.label], a)]
    fail += rc.contrast_failures([r for m in recipe for r in rows[m]], rows[base])
    if fail:
        raise Refusal("population checks failed:\n  " + "\n  ".join(fail))
    task_ids = sorted({r["task_id"] for rr in rows.values() for r in rr})
    graphs = graphs_for(SUITE, graph_version, task_ids)
    clusters = pair_clusters(task_ids)
    seeds = {r["run_id"]: int(r["seed"]) for rr in rows.values() for r in rr}

    out: dict[str, Any] = {
        "rule": "seed-matched (task, rollout seed); both arms at min(k_a, k_b); recipe training "
        "seeds averaged within the task; paired bootstrap over tasks CLUSTERED ON THE PAIR",
        "table1_lock": lock,
        "graph_version": graph_version,
        "arms": {},
        "instrument": {},
        "cells": {},
        "did": {},
        "n_pairs_in_population": len(set(clusters.values())),
    }
    extra = {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    with rc.sweep.registered_metrics(extra):
        lad: dict[str, tuple[dict, dict]] = {}
        for a in arms:
            ids = [r["run_id"] for r in rows[a.label]]
            lad[a.label] = rc._ladders(a.store, ids, graphs)
            stored = rc.sweep._stored_values(a.store, ids)
            sub = rows[a.label]
            out["arms"][a.label] = {
                "arm_id": a.arm_id,
                "inquirer_model": a.inquirer_model,
                "n_runs": len(sub),
                "n_tasks": len({r["task_id"] for r in sub}),
                "run_id_sha256": rc._digest(ids),
                "code_version": sub[0]["code_version"],
                "base_url_sha": sub[0]["base_url_sha"],
                "scorer_hashes": predict._scorer_hashes(a.store, ids),
                "mean_n_asks": sum(r["n_asks"] for r in sub) / len(sub),
            }
            for m in METRICS:
                x = lad[a.label][1] if m in rc.COVERAGE_METRICS else lad[a.label][0]
                out["instrument"].setdefault(a.label, {})[m] = rc.instrument_check(
                    m, x, graphs, stored
                )

        for m in METRICS:
            if not all(rc.instrument_ok(out["instrument"][x][m]) for x in (*recipe, base)):
                out["cells"][m] = {"withheld": "instrument lock failed on one arm"}
                continue
            idx = 1 if m in rc.COVERAGE_METRICS else 0
            per_a, per_b, n_pairs = predict.pooled_levels(
                m, [lad[x][idx] for x in recipe], lad[base][idx], graphs, seeds
            )
            readings = _readings(per_a, per_b, clusters, RESAMPLES)
            own_a = predict.own_stop_levels(m, [lad[x][idx] for x in recipe], graphs)
            own_b = predict.own_stop_levels(m, [lad[base][idx]], graphs)
            out["cells"][m] = {
                "readings": readings,
                **verdict(readings, n_clusters=len({clusters[t] for t in per_a})),
                "n_tasks": len(per_a),
                "n_clusters": len({clusters[t] for t in per_a}),
                "n_pairs": n_pairs,
                "recipe_level_symmetric": sum(per_a.values()) / len(per_a),
                "base_level_symmetric": sum(per_b.values()) / len(per_b),
                "recipe_level_own_stop": sum(own_a.values()) / len(own_a),
                "base_level_own_stop": sum(own_b.values()) / len(own_b),
            }
            if m in DID_METRICS and prediction is not None:
                pr = _pred_maps(prediction, m)
                sym, own = pr["symmetric"], pr["own_stop"]
                out["did"][m] = {
                    "symmetric (primary)": did_cell(per_a, per_b, sym["recipe"], sym["base"]),
                    "own_stop (secondary)": did_cell(own_a, own_b, own["recipe"], own["base"]),
                }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--arm", action="append", type=rc.parse_arm, default=[])
    ap.add_argument("--recipe", default="s1,s2", help="labels of the recipe's training seeds")
    ap.add_argument("--base", default="base")
    ap.add_argument("--prediction", type=Path)
    ap.add_argument("--lock-gold-root", type=Path, required=True)
    ap.add_argument("--artifacts-root", type=Path, default=REPO)
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--lock-only", action="store_true", help="run the Table 1 lock and stop")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("PI_GOLD_ROOT") and not a.lock_only:
        print("analyze: REFUSING: PI_GOLD_ROOT unset (the composed gold)", file=sys.stderr)
        return 3
    try:
        lock = table1_lock(a.artifacts_root, a.lock_gold_root, a.tol)
        if a.lock_only:
            out: dict[str, Any] = {"table1_lock": lock}
        else:
            recipe = [x for x in a.recipe.split(",") if x]
            labels = {x.label for x in a.arm}
            if not recipe or a.base not in labels or any(x not in labels for x in recipe):
                raise Refusal(
                    f"--recipe {recipe} and --base {a.base} must name --arm labels {sorted(labels)}"
                )
            pred = json.loads(a.prediction.read_text()) if a.prediction else None
            out = run(
                a.arm,
                recipe=recipe,
                base=a.base,
                prediction=pred,
                graph_version=a.graph_version,
                lock=lock,
            )
    except Refusal as r:
        print(f"analyze: REFUSING: {r}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"analyze: wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
