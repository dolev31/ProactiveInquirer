"""The additivity prediction for musique_x2, from the constituents' SOLO runs in the Table 3 population.

WHAT IT PREDICTS. For each composed task (constituents A and B) and each arm, the coverage and the
depth-weighted recall that pooling A's and B's solo readings would give, weighted by their node
counts (declared in artifacts/composed_pairs_20260923/DECLARATION.md before any data):

    pred = (n_A * v_A + n_B * v_B) / (n_A + n_B)

THE SOLO POPULATION is the one Table 3 is read on, the same files `scripts/seed_identity/
table1_by_seed.py` reads:
    recipe  qwen3-8b-dpo-stacked-notdone-both-s1 and -s2, artifacts/seedrep_gate_20260919
            (run_ids/run_ids.<model>.musique.txt, scores_parquet/)
    base    the completed cohort's prompted arm (qwen3-8b-base), artifacts/completed_cohort_20260922
            (cohort/run_ids.prompted.musique.txt, scores_parquet/)
Both at cap 8, rollout seeds 0 and 1, test split, n = 200 tasks.

TWO BASES, both per task and per arm, rollout seeds (and the recipe's two training seeds)
averaged within the task:
    symmetric  Table 3's rule: each (task, rollout seed) pair of the recipe seed and the base read
               at min(k_recipe, k_base); `pooled_levels` is `table1_by_seed.pooled_symmetric`
               returning the two per-task LEVELS instead of their difference. This is the basis
               of the declared primary metrics ("at equal realized spend") and so of the DiD.
    own_stop   every run read at its own stop (k = n_asks), no matching.

A CONSTITUENT WITH NO SOLO RUN IS FLAGGED, NEVER IMPUTED. The Table 3 population covers 200 of
MuSiQue's 212 test tasks, and musique_x2 draws constituents from all 212; a composed task with
a constituent outside the 200 gets `pred: null` and is named.

THE LOCK, before any prediction is written: `pooled_levels` must reproduce table1_by_seed.json's
pooled (s1+s2) musique evidence_coverage and dwr points to 1e-12, and every arm's metric at its
own stop must equal the scorer's stored value run by run (`read_contrasts.instrument_check`).

    PI_GOLD_ROOT=<main>/data/gold python scripts/composed_pairs/predict.py \\
        --constituents <x2 root>/data/gold/graphs/musique_x2/constituents.jsonl \\
        --artifacts-root <main> --out <dir>/prediction.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "structured_baselines"))

import read_contrasts as rc  # noqa: E402  (one ladder module, the registered metrics, the rule)

METRICS: tuple[str, ...] = ("evidence_coverage", "dwr")
BASES: tuple[str, ...] = ("symmetric", "own_stop")
ARMS: tuple[str, ...] = ("recipe", "base")
RECIPE = ("qwen3-8b-dpo-stacked-notdone-both-s1", "qwen3-8b-dpo-stacked-notdone-both-s2")
SUITE = "musique"
TABLE1 = "artifacts/seed_identity_20260923/table1_by_seed.json"


def pooled_levels(
    metric: str,
    members: Sequence[Mapping[str, Any]],
    base: Mapping[str, Any],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
) -> tuple[dict[str, float], dict[str, float], int]:
    """`table1_by_seed.pooled_symmetric`'s loop, returning (per-task recipe level, per-task base
    level, n pairs) instead of their bootstrapped difference. Restated line for line so the two
    can be diffed; tests/test_composed_pairs_readers.py pins the difference to its delta."""
    fn = rc.ladder.METRIC_FNS[metric]

    def keyed(arm, what):
        out = {}
        for rid, lad in arm.items():
            key = (lad.suite_id, lad.task_id, int(seeds[rid]))
            if key in out:
                raise ValueError(f"{what}: two runs at {key}")
            out[key] = lad
        return out

    b_by = keyed(base, "comparator")
    acc: dict[str, tuple[list[float], list[float]]] = {}
    n_pairs = 0
    for i, arm in enumerate(members):
        a_by = keyed(arm, f"training seed #{i}")
        for key in sorted(set(a_by) & set(b_by)):
            graph = graphs.get(key[1])
            if graph is None:
                continue
            k = min(a_by[key].n_asks, b_by[key].n_asks)
            av, bv = fn(a_by[key].at(k), graph), fn(b_by[key].at(k), graph)
            if math.isnan(av) or math.isnan(bv):
                continue
            la, lb = acc.setdefault(key[1], ([], []))
            la.append(av)
            lb.append(bv)
            n_pairs += 1
    per_a = {t: sum(a) / len(a) for t, (a, _) in sorted(acc.items())}
    per_b = {t: sum(b) / len(b) for t, (_, b) in sorted(acc.items())}
    return per_a, per_b, n_pairs


def own_stop_levels(
    metric: str, arms: Sequence[Mapping[str, Any]], graphs: Mapping[str, Any]
) -> dict[str, float]:
    """Per task, the mean over every run in `arms` of the metric at that run's own stop."""
    fn = rc.ladder.METRIC_FNS[metric]
    acc: dict[str, list[float]] = {}
    for arm in arms:
        for lad in arm.values():
            graph = graphs.get(lad.task_id)
            if graph is None:
                continue
            v = fn(lad.at(lad.n_asks), graph)
            if not math.isnan(v):
                acc.setdefault(lad.task_id, []).append(v)
    return {t: sum(v) / len(v) for t, v in sorted(acc.items())}


def weighted(va: float, vb: float, na: int, nb: int) -> float:
    return (na * va + nb * vb) / (na + nb)


def predict(
    side: Sequence[Mapping[str, Any]],
    levels: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> dict[str, Any]:
    """`side` = constituents.jsonl rows; `levels[arm][basis][metric][constituent id]`.

    A constituent counts as having a solo reading only if it has one in EVERY (arm, basis,
    metric); a composed task is predicted only if both constituents do.
    """

    def has_solo(cid: str) -> bool:
        return all(cid in levels[arm][basis][m] for arm in ARMS for basis in BASES for m in METRICS)

    tasks: dict[str, Any] = {}
    constituents: set[str] = set()
    for r in side:
        a, b = str(r["a_id"]), str(r["b_id"])
        constituents |= {a, b}
        na, nb = int(r["n_nodes"]["a"]), int(r["n_nodes"]["b"])
        missing = sorted(c for c in (a, b) if not has_solo(c))
        entry: dict[str, Any] = {
            "pair_id": r["pair_id"],
            "order": r["order"],
            "a_id": a,
            "b_id": b,
            "n_nodes": {"a": na, "b": nb},
            "missing_solo": missing,
            "solo": {
                arm: {
                    basis: {
                        m: {
                            "a": levels[arm][basis][m].get(a),
                            "b": levels[arm][basis][m].get(b),
                        }
                        for m in METRICS
                    }
                    for basis in BASES
                }
                for arm in ARMS
            },
            "pred": None,
        }
        if not missing:
            entry["pred"] = {
                arm: {
                    basis: {
                        m: weighted(levels[arm][basis][m][a], levels[arm][basis][m][b], na, nb)
                        for m in METRICS
                    }
                    for basis in BASES
                }
                for arm in ARMS
            }
        tasks[str(r["task_id"])] = entry
    without = sorted(c for c in constituents if not has_solo(c))
    predicted = {t: e for t, e in tasks.items() if e["pred"] is not None}
    means = {
        arm: {
            basis: {
                m: (
                    sum(e["pred"][arm][basis][m] for e in predicted.values()) / len(predicted)
                    if predicted
                    else None
                )
                for m in METRICS
            }
            for basis in BASES
        }
        for arm in ARMS
    }
    return {
        "tasks": dict(sorted(tasks.items())),
        "summary": {
            "n_tasks": len(tasks),
            "n_tasks_predicted": len(predicted),
            "tasks_without_prediction": sorted(t for t in tasks if t not in predicted),
            "n_constituents": len(constituents),
            "n_constituents_with_solo": len(constituents) - len(without),
            "constituents_without_solo": without,
            "mean_pred": means,
        },
    }


# ------------------------------------------------------------------------------ main


def _load_side(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--constituents", type=Path, required=True)
    ap.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO,
        help="checkout holding the Table 3 stores (they are untracked; a worktree lacks them)",
    )
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("PI_GOLD_ROOT"):
        print("predict: REFUSING: PI_GOLD_ROOT unset (musique graphs are gold)", file=sys.stderr)
        return 3
    from pi_eval.gold import load_graphs

    root = a.artifacts_root
    cohort = root / "artifacts/completed_cohort_20260922"
    seedrep = root / "artifacts/seedrep_gate_20260919"
    graphs = load_graphs(SUITE, a.graph_version)
    if not graphs:
        print(f"predict: REFUSING: no {SUITE} graphs under PI_GOLD_ROOT", file=sys.stderr)
        return 2
    ids = {
        "s1": (
            seedrep / "scores_parquet",
            rc._ids(seedrep / f"run_ids/run_ids.{RECIPE[0]}.musique.txt"),
        ),
        "s2": (
            seedrep / "scores_parquet",
            rc._ids(seedrep / f"run_ids/run_ids.{RECIPE[1]}.musique.txt"),
        ),
        "base": (
            cohort / "scores_parquet",
            rc._ids(cohort / "cohort/run_ids.prompted.musique.txt"),
        ),
    }
    seeds = {**rc._seed_map(cohort / "scores_parquet"), **rc._seed_map(seedrep / "scores_parquet")}
    table1 = json.loads((root / TABLE1).read_text())["cells"]["s1s2"]
    out: dict[str, Any] = {
        "population": {},
        "instrument": {},
        "lock": {},
        "graph_version": a.graph_version,
        "rule": "node-count weighted mean of the constituents' solo per-task levels; "
        "symmetric = table1_by_seed.pooled_symmetric's pairing (both at min k), own_stop = k = "
        "n_asks; rollout seeds and the recipe's two training seeds averaged within the task",
    }
    levels: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        arm: {basis: {} for basis in BASES} for arm in ARMS
    }
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        lad: dict[str, tuple[dict, dict]] = {}
        for arm, (store, run_ids) in ids.items():
            lad[arm] = rc._ladders(store, run_ids, graphs)
            stored = rc.sweep._stored_values(store, run_ids)
            out["population"][arm] = {
                "store": str(store.relative_to(root)),
                "n_runs": len(run_ids),
                "run_id_sha256": rc._digest(run_ids),
                "scorer_hashes": _scorer_hashes(store, run_ids),
            }
            for m in METRICS:
                x = lad[arm][1] if m in rc.COVERAGE_METRICS else lad[arm][0]
                chk = rc.instrument_check(m, x, graphs, stored)
                out["instrument"].setdefault(arm, {})[m] = chk
                if not rc.instrument_ok(chk):
                    print(
                        f"predict: REFUSING: instrument lock failed {arm} {m}: {chk}",
                        file=sys.stderr,
                    )
                    return 2
        for m in METRICS:
            idx = 1 if m in rc.COVERAGE_METRICS else 0
            per_a, per_b, n_pairs = pooled_levels(
                m, [lad["s1"][idx], lad["s2"][idx]], lad["base"][idx], graphs, seeds
            )
            from pi_eval.stats.inference import paired_difference

            got = paired_difference(per_a, per_b, n_boot=10000, seed=0)
            want = next(
                x for x in table1[f"{m}::musique"] if x["n_boot"] == 10000 and x["seed"] == 0
            )
            diff = abs(got.point - want["delta"])
            out["lock"][f"{m}::musique"] = {
                "published": want["delta"],
                "reproduced": got.point,
                "abs_diff": diff,
                "n": got.n,
                "n_pairs": n_pairs,
                "ok": diff <= a.tol and got.n == want["n"] and n_pairs == want["n_pairs"],
            }
            if not out["lock"][f"{m}::musique"]["ok"]:
                print(f"predict: LOCK FAILED {m}: {out['lock'][f'{m}::musique']}", file=sys.stderr)
                return 2
            levels["recipe"]["symmetric"][m] = per_a
            levels["base"]["symmetric"][m] = per_b
            levels["recipe"]["own_stop"][m] = own_stop_levels(
                m, [lad["s1"][idx], lad["s2"][idx]], graphs
            )
            levels["base"]["own_stop"][m] = own_stop_levels(m, [lad["base"][idx]], graphs)
    side = _load_side(a.constituents)
    out.update(predict(side, levels))
    out["constituents_file"] = {"n_rows": len(side), "sha256": _sha(a.constituents)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    s = out["summary"]
    print(
        f"predict: wrote {a.out}; constituents with solo runs {s['n_constituents_with_solo']}/"
        f"{s['n_constituents']}; tasks predicted {s['n_tasks_predicted']}/{s['n_tasks']}"
    )
    return 0


def _scorer_hashes(store: Path, run_ids: Sequence[str]) -> list[str]:
    import duckdb

    con = duckdb.connect()
    return [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT scorer_hash FROM read_parquet('{(store / 'scores.parquet').as_posix()}') "
            "WHERE run_id IN (SELECT unnest(?)) ORDER BY 1",
            [list(run_ids)],
        ).fetchall()
    ]


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
