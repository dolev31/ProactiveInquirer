"""The four-seed table: the selected recipe's final stage at seeds 1, 2, 0 (fresh) and 3.

Declared in artifacts/seedrep4_20260924/RULES.md section 4 before either new seed finished
training. The headline stays seeds 1+2; this table is added beside it.

READERS, UNCHANGED, WITH THEIR LOCKS FIRST.
  table1_by_seed   the symmetric seed-matched contrast (both arms at min(k_a, k_b)) against the
                   completed cohort's prompted runs, five metrics. Its seed 0 lock (published cells
                   to 1e-9) and the s1/s2 cells of its committed record must reproduce exactly.
  cap8_by_seed     each arm at its own stop under the cap of eight against the same prompted runs.
  teacher_by_seed  against gpt-oss-120b prompted, both at the lower count.
  teacher_ownstop  against gpt-oss-120b prompted, each at its own stop.
Each of the last three reproduces its committed record for the seeds it already covers first.

THE NEW SEEDS are read from their own isolated store (--new-store) with their committed run-id
lists; every other arm is read from the store its reader already uses. Metrics are recomputed from
ladders by the reader, and the instrument check compares them to the store's stored values.

PER METRIC AND SUITE: each seed's cell (10,000 resamples, seed 0; a bound within 0.01 of zero is
re-read at 50,000 with seeds 101, 202, 303), the sign count, the between-seed SD (n-1) and range,
the within-seed bootstrap SE (SD of 10,000 bootstrap means of that seed's per-task differences,
seed 0), a crude interval over the seed points (mean +- t(0.975, 3) SD / 2), and the four-seed
pooled reading. For the primary metric, all six pairwise seed contrasts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import duckdb
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))
sys.path.insert(0, str(REPO / "scripts" / "structured_baselines"))

import cap8_by_seed as c8  # noqa: E402
import read_contrasts as rc  # noqa: E402
import table1_by_seed as t1  # noqa: E402
import teacher_ownstop_by_seed as tos  # noqa: E402

from pi_eval.stats.inference import paired_difference  # noqa: E402

SUITES = t1.SUITES
METRICS = t1.METRICS
PRIMARY = "evidence_coverage"
FOUR = ("s1", "s2", "s0fresh", "s3")
NEW_NAMES = {
    "s0fresh": "qwen3-8b-dpo-stacked-notdone-both-s0fresh",
    "s3": "qwen3-8b-dpo-stacked-notdone-both-s3",
}
T_975_DF3 = 3.182446305284263
NEAR = 0.01
STAB = ((50000, 101), (50000, 202), (50000, 303))
REC = REPO / "artifacts" / "seed_identity_20260923"
NEW_IDS = REPO / "artifacts" / "seedrep4_20260924" / "run_ids"


def _ids(p: Path) -> list[str]:
    return [ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and ln[0] != "#"]


def per_task_pairs(metric, a, b, graphs, seeds) -> dict[str, float]:
    """seed_matched_symmetric's pairing, returning the per-task differences it bootstraps."""
    fn = rc.ladder.METRIC_FNS[metric]
    a_by = {(x.suite_id, x.task_id, int(seeds[r])): x for r, x in a.items()}
    b_by = {(x.suite_id, x.task_id, int(seeds[r])): x for r, x in b.items()}
    acc: dict[str, tuple[list[float], list[float]]] = {}
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
    return {t: sum(x) / len(x) - sum(y) / len(y) for t, (x, y) in acc.items()}


def boot_se(diffs: dict[str, float], n_boot: int = 10000, seed: int = 0) -> float:
    v = np.array([diffs[t] for t in sorted(diffs)])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(means.std(ddof=1))


def with_stability(cell: dict, reread) -> dict:
    near = [b for b in (cell["ci_lo"], cell["ci_hi"]) if abs(b) <= NEAR]
    if not near:
        return {**cell, "stability": {"checked": False, "verdict": "as computed"}}
    which = "ci_lo" if abs(cell["ci_lo"]) <= NEAR else "ci_hi"
    reads = [{"n_boot": nb, "seed": sd, **reread(nb, sd)} for nb, sd in STAB]
    signs = {1 if r[which] >= 0 else -1 for r in reads} | {1 if cell[which] >= 0 else -1}
    verdict = "as computed" if len(signs) == 1 else "undecided"
    return {
        **cell,
        "stability": {"checked": True, "bound": which, "reads": reads, "verdict": verdict},
    }


def summarize(cells: dict[str, dict], ses: dict[str, float] | None) -> dict:
    pts = [cells[s]["delta"] for s in FOUR if s in cells]
    n = len(pts)
    sd = statistics.stdev(pts) if n > 1 else float("nan")
    mean = statistics.fmean(pts)
    t = T_975_DF3 if n == 4 else float("nan")
    out = {
        "n_seeds": n,
        "points": {s: cells[s]["delta"] for s in FOUR if s in cells},
        "sign_count_positive": sum(p > 0 for p in pts),
        "intervals_above_zero": sum(cells[s]["ci_lo"] > 0 for s in FOUR if s in cells),
        "intervals_below_zero": sum(cells[s]["ci_hi"] < 0 for s in FOUR if s in cells),
        "between_seed_sd": sd,
        "range": max(pts) - min(pts),
        "min": min(pts),
        "max": max(pts),
        "crude_t_interval": {
            "label": "crude",
            "mean": mean,
            "lo": mean - t * sd / math.sqrt(n),
            "hi": mean + t * sd / math.sqrt(n),
            "t": t,
            "df": n - 1,
        },
    }
    if ses:
        out["within_seed_bootstrap_se"] = ses
        out["mean_within_seed_bootstrap_se"] = statistics.fmean(ses.values())
    return out


def table1_part(new_store: Path) -> dict:
    from pi_eval.gold import load_graphs

    pub = t1.published_cells()
    record = json.loads((REC / "table1_by_seed.json").read_text())
    seeds = {
        **rc._seed_map(t1.COHORT / "scores_parquet"),
        **rc._seed_map(t1.SEEDREP / "scores_parquet"),
        **rc._seed_map(new_store),
    }
    out: dict = {"lock": {}, "cells": {}, "summary": {}, "pairwise": {}, "pooled4": {}, "arms": {}}
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        for suite in SUITES:
            graphs = load_graphs(suite, "v1")
            prompted = rc._ids(t1.COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
            base_run, base_cov = rc._ladders(t1.COHORT / "scores_parquet", prompted, graphs)
            src = {
                "s0": (
                    t1.COHORT / "scores_parquet",
                    t1.COHORT / "cohort" / f"run_ids.trained.{suite}.txt",
                ),
                **{
                    k: (
                        t1.SEEDREP / "scores_parquet",
                        t1.SEEDREP / "run_ids" / f"run_ids.{v}.{suite}.txt",
                    )
                    for k, v in t1.S_NAMES.items()
                },
                **{
                    k: (new_store, NEW_IDS / f"run_ids.{v}.{suite}.txt")
                    for k, v in NEW_NAMES.items()
                },
            }
            loaded = {}
            for arm, (store, idf) in src.items():
                ids = rc._ids(idf)
                run_l, cov_l = rc._ladders(store, ids, graphs)
                loaded[arm] = (run_l, cov_l)
                stored = rc.sweep._stored_values(store, ids)
                out["arms"].setdefault(arm, {})[suite] = {
                    "n_runs": len(ids),
                    "run_id_sha256": rc._digest(ids),
                    "mean_n_asks": sum(x.n_asks for x in run_l.values()) / len(run_l),
                }
                for metric in METRICS:
                    a = cov_l if metric in rc.COVERAGE_METRICS else run_l
                    inst = rc.instrument_check(metric, a, graphs, stored)
                    if not rc.instrument_ok(inst):
                        raise SystemExit(f"instrument lock failed {arm} {suite} {metric}: {inst}")
            for metric in METRICS:
                idx = 1 if metric in rc.COVERAGE_METRICS else 0
                b = base_cov if idx else base_run
                key = f"{metric}::{suite}"
                s0 = rc.seed_matched_symmetric(
                    metric, loaded["s0"][idx], b, graphs, seeds, seed=0, n_boot=10000
                )
                if abs(s0["delta"] - pub[key]) > 1e-9:
                    raise SystemExit(f"LOCK FAILED s0 {key}")
                out["lock"][f"s0::{key}"] = {"published": pub[key], "reproduced": s0["delta"]}
                cells, ses = {}, {}
                for arm in FOUR:
                    a = loaded[arm][idx]
                    cell = rc.seed_matched_symmetric(
                        metric, a, b, graphs, seeds, seed=0, n_boot=10000
                    )
                    if arm in ("s1", "s2") and cell != record["cells"][arm][key]:
                        raise SystemExit(
                            f"LOCK FAILED {arm} {key}: differs from table1_by_seed.json"
                        )
                    diffs = per_task_pairs(metric, a, b, graphs, seeds)
                    if abs(statistics.fmean(diffs.values()) - cell["delta"]) > 1e-12:
                        raise SystemExit(f"PER-TASK LOCK FAILED {arm} {key}")
                    ses[arm] = boot_se(diffs)
                    cells[arm] = with_stability(
                        cell,
                        lambda nb, sd, a=a: rc.seed_matched_symmetric(
                            metric, a, b, graphs, seeds, seed=sd, n_boot=nb
                        ),
                    )
                out["cells"][key] = cells
                out["summary"][key] = summarize(cells, ses)
                rec_pool = record["cells"]["s1s2"][key]
                got_pool = [
                    {
                        "n_boot": nb,
                        "seed": sd,
                        **t1.pooled_symmetric(
                            metric,
                            [loaded["s1"][idx], loaded["s2"][idx]],
                            b,
                            graphs,
                            seeds,
                            seed=sd,
                            n_boot=nb,
                        ),
                    }
                    for nb, sd in t1.POOL_RESAMPLES
                ]
                if got_pool != rec_pool:
                    raise SystemExit(f"POOLED LOCK FAILED {key}")
                out["pooled4"][key] = [
                    {
                        "n_boot": nb,
                        "seed": sd,
                        **t1.pooled_symmetric(
                            metric,
                            [loaded[s][idx] for s in FOUR],
                            b,
                            graphs,
                            seeds,
                            seed=sd,
                            n_boot=nb,
                        ),
                    }
                    for nb, sd in t1.POOL_RESAMPLES
                ]
                if metric == PRIMARY:
                    for x, y in itertools.combinations(FOUR, 2):
                        ax, ay = loaded[x][idx], loaded[y][idx]
                        pc = rc.seed_matched_symmetric(
                            metric, ax, ay, graphs, seeds, seed=0, n_boot=10000
                        )
                        out["pairwise"].setdefault(suite, {})[f"{x}-{y}"] = with_stability(
                            pc,
                            lambda nb, sd, ax=ax, ay=ay: rc.seed_matched_symmetric(
                                metric, ax, ay, graphs, seeds, seed=sd, n_boot=nb
                            ),
                        )
    return out


def cap8_part(new_store: Path) -> dict:
    con = duckdb.connect()
    record = json.loads((REC / "cap8_by_seed.json").read_text())
    pub = json.loads((REPO / "artifacts/baseline_completion_20260920/three.json").read_text())[
        "complete"
    ]["by_suite"]
    out: dict = {"lock": {}, "cells": {}, "summary": {}, "asks": {}}
    for suite in SUITES:
        pr = c8.per_task(
            con,
            c8.COHORT / "scores_parquet",
            c8._ids(c8.COHORT / "cohort" / f"run_ids.prompted.{suite}.txt"),
        )
        src = {
            "s0": (
                c8.COHORT / "scores_parquet",
                c8._ids(c8.COHORT / "cohort" / f"run_ids.trained.{suite}.txt"),
            ),
            **{
                k: (
                    c8.SEEDREP / "scores_parquet",
                    c8._ids(c8.SEEDREP / "run_ids" / f"run_ids.{v}.{suite}.txt"),
                )
                for k, v in t1.S_NAMES.items()
            },
            **{
                k: (new_store, c8._ids(NEW_IDS / f"run_ids.{v}.{suite}.txt"))
                for k, v in NEW_NAMES.items()
            },
        }
        cells = {}
        per = {}
        for k, (store, ids) in src.items():
            a = c8.per_task(con, store, ids)
            per[k] = a
            sh = sorted(set(a) & set(pr))
            est = paired_difference(
                {t: a[t] for t in sh}, {t: pr[t] for t in sh}, n_boot=10000, seed=0
            )
            cells[k] = {"delta": est.point, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi, "n": est.n}
            out["asks"].setdefault(k, {})[suite] = c8.asks(con, store, ids)
        if abs(cells["s0"]["delta"] - pub[suite]["cap8_coverage_delta"]) > 1e-9:
            raise SystemExit(f"LOCK FAILED cap8 s0 {suite}")
        for k in ("s1", "s2"):
            if cells[k] != record["cells"][k][suite]:
                raise SystemExit(f"LOCK FAILED cap8 {k} {suite}: differs from cap8_by_seed.json")
        out["lock"][suite] = {
            "s0": cells["s0"]["delta"],
            "published": pub[suite]["cap8_coverage_delta"],
        }
        out["cells"][suite] = {k: cells[k] for k in FOUR}
        out["summary"][suite] = summarize(cells, None)
        # the four-seed pooled cell: every run of the four seeds averaged within the task
        a4: dict[str, list[float]] = {}
        for k in FOUR:
            store, ids = src[k]
            for t, v in c8.per_task(con, store, ids).items():
                a4.setdefault(t, []).append(v)
        # per_task already averages rollout seeds within a seed; average the four seed means
        a4m = {t: sum(v) / len(v) for t, v in a4.items() if len(v) == len(FOUR)}
        sh = sorted(set(a4m) & set(pr))
        est = paired_difference(
            {t: a4m[t] for t in sh}, {t: pr[t] for t in sh}, n_boot=10000, seed=0
        )
        out["cells"][suite]["pooled4"] = {
            "delta": est.point,
            "ci_lo": est.ci_lo,
            "ci_hi": est.ci_hi,
            "n": est.n,
        }
    return out


def teacher_part(new_store: Path) -> dict:
    """teacher_by_seed's contrast (both at the lower count, evidence_coverage) per seed."""
    import teacher_by_seed as tb

    from pi_eval.gold import load_graphs

    record = json.loads((REC / "teacher_by_seed.json").read_text())
    seeds = {**rc._seed_map(tb.STORE), **rc._seed_map(new_store)}
    suite_of = dict(
        duckdb.connect()
        .execute(
            f"SELECT run_id, suite_id FROM read_parquet('{(tb.STORE / 'runs.parquet').as_posix()}')"
        )
        .fetchall()
    )
    teacher_all = tb._ids(REPO / "artifacts/n6/run_ids.inquirer_prompted.txt")
    out: dict = {"lock": {}, "cells": {}, "summary": {}, "pooled4": {}}
    with rc.sweep.registered_metrics(
        {**rc.sweep.EXTRA_METRIC_FNS, **rc.sweep.DIAGNOSTIC_METRIC_FNS}
    ):
        for suite in SUITES:
            graphs = load_graphs(suite, "v1")
            teacher = [r for r in teacher_all if suite_of.get(r) == suite]
            _, t_cov = rc._ladders(tb.STORE, teacher, graphs)
            src = {
                "s0": (
                    tb.STORE,
                    tb._ids(REPO / "artifacts/testsplit_qa" / f"run_ids.trained.{suite}.txt"),
                ),
                **{
                    k: (
                        tb.STORE,
                        tb._ids(
                            REPO
                            / "artifacts/seedrep_gate_20260919/run_ids"
                            / f"run_ids.{v}.{suite}.txt"
                        ),
                    )
                    for k, v in tb.SEEDS.items()
                },
                **{
                    k: (new_store, tb._ids(NEW_IDS / f"run_ids.{v}.{suite}.txt"))
                    for k, v in NEW_NAMES.items()
                },
            }
            cov, cells, ses = {}, {}, {}
            for k, (store, ids) in src.items():
                _, cov[k] = rc._ladders(store, ids, graphs)
                cells[k] = rc.seed_matched_symmetric(
                    "evidence_coverage", cov[k], t_cov, graphs, seeds, seed=0, n_boot=10000
                )
            if abs(cells["s0"]["delta"] - tb.PUBLISHED_S0[suite]) > 1e-4:
                raise SystemExit(f"LOCK FAILED teacher s0 {suite}")
            for k in ("s1", "s2"):
                if cells[k] != record["cells"][k][suite]:
                    raise SystemExit(
                        f"LOCK FAILED teacher {k} {suite}: differs from teacher_by_seed.json"
                    )
            for k in FOUR:
                diffs = per_task_pairs("evidence_coverage", cov[k], t_cov, graphs, seeds)
                ses[k] = boot_se(diffs)
                cells[k] = with_stability(
                    cells[k],
                    lambda nb, sd, a=cov[k]: rc.seed_matched_symmetric(
                        "evidence_coverage", a, t_cov, graphs, seeds, seed=sd, n_boot=nb
                    ),
                )
            out["lock"][suite] = {"s0": cells["s0"]["delta"], "published": tb.PUBLISHED_S0[suite]}
            out["cells"][suite] = {k: cells[k] for k in FOUR}
            out["summary"][suite] = summarize(cells, ses)
            out["pooled4"][suite] = [
                {
                    "n_boot": nb,
                    "seed": sd,
                    **t1.pooled_symmetric(
                        "evidence_coverage",
                        [cov[s] for s in FOUR],
                        t_cov,
                        graphs,
                        seeds,
                        seed=sd,
                        n_boot=nb,
                    ),
                }
                for nb, sd in t1.POOL_RESAMPLES
            ]
    return out


def teacher_ownstop_part(new_store: Path) -> dict:
    con = duckdb.connect()
    record = json.loads((REC / "teacher_ownstop_by_seed.json").read_text())
    suite_of = dict(
        con.execute(
            f"SELECT run_id, suite_id FROM read_parquet('{(tos.STORE / 'runs.parquet').as_posix()}')"
        ).fetchall()
    )
    teacher_all = tos._ids(REPO / "artifacts/n6/run_ids.inquirer_prompted.txt")
    metric = "evidence_coverage"

    def per_task_at(store: Path, ids: list[str]) -> dict[str, float]:
        return {
            t: float(v)
            for t, v in con.execute(
                f"SELECT r.task_id, avg(s.value) FROM read_parquet('{(store / 'scores.parquet').as_posix()}') s "
                f"JOIN read_parquet('{(store / 'runs.parquet').as_posix()}') r USING (run_id) "
                "WHERE s.metric_name = ? AND s.value IS NOT NULL AND NOT isnan(s.value) "
                "AND s.run_id IN (SELECT unnest(?)) GROUP BY 1",
                [metric, ids],
            ).fetchall()
        }

    out: dict = {"lock": {}, "cells": {}, "summary": {}}
    for suite in SUITES:
        teacher = [r for r in teacher_all if suite_of.get(r) == suite]
        t = tos.per_task(con, teacher, metric)
        sr = REPO / "artifacts/seedrep_gate_20260919/run_ids"
        ids = {
            "s0": (
                tos.STORE,
                tos._ids(REPO / "artifacts/testsplit_qa" / f"run_ids.trained.{suite}.txt"),
            ),
            **{
                k: (tos.STORE, tos._ids(sr / f"run_ids.{v}.{suite}.txt"))
                for k, v in t1.S_NAMES.items()
            },
            **{
                k: (new_store, tos._ids(NEW_IDS / f"run_ids.{v}.{suite}.txt"))
                for k, v in NEW_NAMES.items()
            },
        }
        ids["s1s2"] = (tos.STORE, ids["s1"][1] + ids["s2"][1])
        cells = {}
        per = {}
        for k, (store, rid) in ids.items():
            a = tos.per_task(con, rid, metric) if store == tos.STORE else per_task_at(store, rid)
            per[k] = a
            sh = sorted(set(a) & set(t))
            est = paired_difference(
                {x: a[x] for x in sh}, {x: t[x] for x in sh}, n_boot=10000, seed=0
            )
            cells[k] = {"delta": est.point, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi, "n": est.n}
        if abs(cells["s0"]["delta"] - tos.PUB[metric][suite]) > 5e-5:
            raise SystemExit(f"LOCK FAILED ownstop s0 {suite}")
        rec = record["cells"]["s1s2"][metric][suite]
        if any(cells["s1s2"][k] != rec[k] for k in ("delta", "ci_lo", "ci_hi", "n")):
            raise SystemExit(
                f"LOCK FAILED ownstop s1s2 {suite}: differs from teacher_ownstop_by_seed.json"
            )
        a4: dict[str, list[float]] = {}
        for k in FOUR:
            for x, v in per[k].items():
                a4.setdefault(x, []).append(v)
        a4m = {x: sum(v) / len(v) for x, v in a4.items() if len(v) == len(FOUR)}
        sh = sorted(set(a4m) & set(t))
        est = paired_difference(
            {x: a4m[x] for x in sh}, {x: t[x] for x in sh}, n_boot=10000, seed=0
        )
        out["lock"][suite] = {"s0": cells["s0"]["delta"], "published": tos.PUB[metric][suite]}
        out["cells"][suite] = {k: cells[k] for k in FOUR} | {
            "pooled4": {"delta": est.point, "ci_lo": est.ci_lo, "ci_hi": est.ci_hi, "n": est.n}
        }
        out["summary"][suite] = summarize(cells, None)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--new-store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--part", choices=("table1", "cap8", "teacher", "ownstop", "all"), default="all"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="plumbing test only: s0fresh/s3 := s1/s2's own runs, read from --new-store (the "
        "seedrep store). Every lock must pass and the aliased cells must equal s1/s2's.",
    )
    a = ap.parse_args(argv)
    global NEW_IDS
    if a.selftest:
        NEW_IDS = t1.SEEDREP / "run_ids"
        NEW_NAMES["s0fresh"], NEW_NAMES["s3"] = t1.S_NAMES["s1"], t1.S_NAMES["s2"]
    res: dict = {"new_store": str(a.new_store), "seeds": list(FOUR), "names": NEW_NAMES}
    res["selftest"] = bool(a.selftest)
    parts = {
        "table1": table1_part,
        "cap8": cap8_part,
        "teacher": teacher_part,
        "ownstop": teacher_ownstop_part,
    }
    for name, fn in parts.items():
        if a.part in (name, "all"):
            res[name] = fn(a.new_store)
            print(f"{name}: done", flush=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
