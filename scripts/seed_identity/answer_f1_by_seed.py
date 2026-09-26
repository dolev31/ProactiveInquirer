"""Answer token F1 at each arm's own stop under the shared ceiling of eight, the recipe against the
same weights prompted, per training seed and pooled: the answer row of the headline table.

POPULATION. The own-stop coverage row's (cap8_by_seed.py): the recipe's seeds 1 and 2
(artifacts/seedrep_gate_20260919), seed 0 (the completed cohort's trained runs), and the completed
cohort's prompted runs as the one comparator, 200 held-out tasks per suite. Per task, the mean of
`answer_token_f1` over that arm's runs (training seeds and rollout seeds averaged into the task),
paired over tasks, `paired_difference` unclustered, 10,000 resamples at seed 0; the pooled cell also
at 50,000 under seeds 101/202/303, DECIDED only if all three exclude zero on one side.

LOCKS, BEFORE ANY NEW NUMBER. The script refuses to write if any fails.
  L1  The per-seed values section 5.5 prints (+0.031/+0.022 MuSiQue, +0.033/+0.032 2WikiMultiHopQA,
      and the prov line's +0.004/-0.011 StrategyQA): on those values' own population (the
      stop_pop_s{1,2} stores, the seed's 400 runs against the pre-completion prompted runs,
      176/200/166 tasks) and with their record's estimator (per-task mean, paired_difference
      unclustered, 10,000 resamples at seed 0, or 50,000 at seed 0 where a bound sits within 0.01
      of zero: stopping_answer_test.lib.with_stability), point, interval and n reproduce
      stop_s{1,2}.json step2_answer_quality.<suite>.answer_token_f1 to 1e-12, and each point
      rounds to the printed value.
  L2  The population is the own-stop coverage row's: this reader's per-task function, run on
      evidence_coverage, reproduces cap8_by_seed.json cells.<s0|s1|s2|s1s2>.<suite> (point,
      interval, n) to 1e-12.
  L3  The pooled answer reading is the one decompose.py already recorded: with decompose's
      clusters (template_id on MuSiQue, the task elsewhere) the 10,000-resample reading
      reproduces decompose.json per_suite.<suite>.readings.answer_token_f1 to 1e-12. The printed
      reading is unclustered, the rule of every other cell of the table; on MuSiQue the two differ.
  Pairing on (task, rollout seed), each training seed separately, then averaging into the task,
  gives the same per-task differences as the per-task means (asserted to 1e-12), because every
  task carries both rollout seeds in every arm (asserted).

Usage:  .venv/bin/python scripts/seed_identity/answer_f1_by_seed.py \
            --out artifacts/seed_identity_20260923/answer_f1_by_seed.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SI = REPO / "artifacts/seed_identity_20260923"
COHORT = REPO / "artifacts/completed_cohort_20260922"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919"
DECOMPOSE = REPO / "artifacts/answer_loss_20260923/decompose.json"
SUITES = ("musique", "strategyqa", "wiki2")
METRIC = "answer_token_f1"
SCORER_HASH = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
RESAMPLES = ((10_000, 0), (50_000, 101), (50_000, 202), (50_000, 303))
TOL = 1e-12
RECIPE = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
# What the paper prints for these cells: section 5.5's prose (MuSiQue, 2WikiMultiHopQA) and its
# prov line (StrategyQA), per training seed, at three decimals.
PRINTED = {
    ("s1", "musique"): "+0.031",
    ("s2", "musique"): "+0.022",
    ("s1", "strategyqa"): "+0.004",
    ("s2", "strategyqa"): "-0.011",
    ("s1", "wiki2"): "+0.033",
    ("s2", "wiki2"): "+0.032",
}


# --------------------------------------------------------------------------- pure


def pair_on_rollout_seed(
    arms: Sequence[Mapping[tuple[str, int], float]], base: Mapping[tuple[str, int], float]
) -> tuple[dict[str, float], dict[str, float]]:
    """Each training seed paired with the comparator on (task, rollout seed), then every pair of
    every training seed averaged into one per-task value per arm. Refuses any key present on one
    side only, so an unbalanced population cannot silently re-weight a training seed."""
    acc: dict[str, tuple[list[float], list[float]]] = {}
    for i, arm in enumerate(arms):
        if set(arm) != set(base):
            raise ValueError(
                f"training seed #{i}: unpaired (task, rollout seed) keys, "
                f"{sorted(set(arm) ^ set(base))[:4]}"
            )
        for key in sorted(arm):
            la, lb = acc.setdefault(key[0], ([], []))
            la.append(float(arm[key]))
            lb.append(float(base[key]))
    keys = sorted(acc)
    return (
        {t: sum(acc[t][0]) / len(acc[t][0]) for t in keys},
        {t: sum(acc[t][1]) / len(acc[t][1]) for t in keys},
    )


def lock(label: str, got: Mapping[str, Any], want: Mapping[str, Any], tol: float = TOL) -> dict:
    """Point, interval and n of `got` equal `want`, or SystemExit naming both."""
    bad = [k for k in ("point", "lo", "hi") if abs(float(got[k]) - float(want[k])) > tol]
    if bad or int(got["n"]) != int(want["n"]):
        raise SystemExit(
            f"LOCK FAILED {label}: reproduced {dict((k, got[k]) for k in ('point', 'lo', 'hi', 'n'))} "
            f"against the record's {dict((k, want[k]) for k in ('point', 'lo', 'hi', 'n'))}. NOT WRITING."
        )
    return {
        **{k: got[k] for k in ("point", "lo", "hi", "n")},
        **{f"record_{k}": want[k] for k in ("point", "lo", "hi", "n")},
        "ok": True,
    }


def fmt3(x: float) -> str:
    """The paper's print rule: half up on the exact decimal the value reads as (its repr)."""
    d = Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(d, "+f")


def lock_printed(label: str, point: float, printed: str) -> None:
    if fmt3(point) != printed:
        raise SystemExit(
            f"LOCK FAILED {label}: {point!r} prints {fmt3(point)}, the paper prints {printed}. "
            "NOT WRITING."
        )


def decided(readings: Sequence[Mapping[str, Any]]) -> bool:
    big = [r for r in readings if int(r["n_boot"]) == 50_000]
    if len(big) != 3:
        raise ValueError(f"need exactly three 50k intervals, got {len(big)}")
    return all(r["ci_lo"] > 0 for r in big) or all(r["ci_hi"] < 0 for r in big)


# --------------------------------------------------------------------------- data


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def per_task(con, store: Path, ids: Sequence[str], metric: str) -> dict[str, float]:
    """cap8_by_seed.per_task's SQL with the metric as a parameter, after asserting that every run
    carries exactly one value under the one scorer hash."""
    sc, rn = (store / "scores.parquet").as_posix(), (store / "runs.parquet").as_posix()
    n_rows, n_runs, hashes = con.execute(
        f"SELECT count(*), count(DISTINCT run_id), list(DISTINCT scorer_hash) FROM read_parquet('{sc}') "
        "WHERE metric_name = ? AND run_id IN (SELECT unnest(?))",
        [metric, list(ids)],
    ).fetchone()
    if n_rows != len(set(ids)) or n_runs != len(set(ids)) or hashes != [SCORER_HASH]:
        raise SystemExit(
            f"{store}: {metric} has {n_rows} rows over {n_runs} runs for {len(set(ids))} ids, "
            f"scorer hashes {hashes}. NOT WRITING."
        )
    return {
        t: float(v)
        for t, v in con.execute(
            f"SELECT r.task_id, avg(s.value) FROM read_parquet('{sc}') s "
            f"JOIN read_parquet('{rn}') r USING (run_id) "
            "WHERE s.metric_name = ? AND s.run_id IN (SELECT unnest(?)) GROUP BY 1",
            [metric, list(ids)],
        ).fetchall()
    }


def per_key(con, store: Path, ids: Sequence[str], metric: str) -> dict[tuple[str, int], float]:
    sc, rn = (store / "scores.parquet").as_posix(), (store / "runs.parquet").as_posix()
    rows = con.execute(
        f"SELECT r.task_id, r.seed, s.value FROM read_parquet('{sc}') s "
        f"JOIN read_parquet('{rn}') r USING (run_id) "
        "WHERE s.metric_name = ? AND s.run_id IN (SELECT unnest(?))",
        [metric, list(ids)],
    ).fetchall()
    out: dict[tuple[str, int], float] = {}
    for t, sd, v in rows:
        if (t, int(sd)) in out:
            raise SystemExit(f"{store}: two runs at ({t}, {sd}). NOT WRITING.")
        out[(t, int(sd))] = float(v)
    return out


def contrast(a: Mapping[str, float], b: Mapping[str, float], resamples, clusters=None) -> list:
    from pi_eval.stats.inference import paired_difference

    keys = sorted(set(a) & set(b))
    aa, bb = {k: a[k] for k in keys}, {k: b[k] for k in keys}
    cl = {k: clusters[k] for k in keys} if clusters else None
    out = []
    for nb, sd in resamples:
        e = paired_difference(aa, bb, clusters=cl, n_boot=nb, seed=sd)
        out.append(
            {"n_boot": nb, "seed": sd, "point": e.point, "lo": e.ci_lo, "hi": e.ci_hi, "n": e.n}
        )
    return out


def record_estimator(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    """stop_s{1,2}.json's own estimator (scripts/stopping_answer_test/run.py step2): 10,000
    resamples at seed 0, and where a bound sits within 0.01 of zero the 50,000-resample seed-0
    interval, through that module's `with_stability`."""
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.stopping_answer_test import lib as st_lib

    from pi_eval.stats.inference import paired_difference

    def compute(n_boot: int, seed: int) -> dict[str, Any]:
        e = paired_difference(a, b, clusters=None, n_boot=n_boot, seed=seed)
        return {"point": e.point, "lo": e.ci_lo, "hi": e.ci_hi, "n": e.n}

    return st_lib.with_stability(compute, seed=0)


def _ten(readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return next(r for r in readings if r["n_boot"] == 10_000 and r["seed"] == 0)


def definition(con, store: Path, ids: Sequence[str]) -> dict[str, Any]:
    sc = (store / "scores.parquet").as_posix()
    n, n_val, zero, one, mean = con.execute(
        f"SELECT count(*), count(value), avg((value = 0)::int), avg((value = 1)::int), avg(value) "
        f"FROM read_parquet('{sc}') WHERE metric_name = ? AND run_id IN (SELECT unnest(?))",
        [METRIC, list(ids)],
    ).fetchone()
    return {
        "n_runs": int(n),
        "n_with_value": int(n_val),
        "share_zero": zero,
        "share_one": one,
        "mean": mean,
    }


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    import duckdb

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    cap8 = json.loads((SI / "cap8_by_seed.json").read_text())
    dec = json.loads(DECOMPOSE.read_text())
    out: dict[str, Any] = {
        "lock": {"L1_printed_per_seed": {}, "L2_own_stop_population": {}, "L3_decompose": {}}
    }

    # L1: the per-seed values the paper prints, on their own population and estimator
    for s in ("s1", "s2"):
        rec = json.loads((SI / f"stop_{s}.json").read_text())["step2_answer_quality"]
        pop = SI / f"stop_pop_{s}"
        for suite in SUITES:
            t = per_task(
                con, pop / "scores_parquet", _ids(pop / f"run_ids.trained.{suite}.txt"), METRIC
            )
            p = per_task(
                con, pop / "scores_parquet", _ids(pop / f"run_ids.prompted.{suite}.txt"), METRIC
            )
            got = record_estimator(t, p)
            want = rec[suite][METRIC]
            cell = lock(f"L1 stop_{s}.json {suite}", got, want)
            lock_printed(f"L1 {s} {suite}", got["point"], PRINTED[(s, suite)])
            cell["printed"] = PRINTED[(s, suite)]
            out["lock"]["L1_printed_per_seed"][f"{s}::{suite}"] = cell

    cells: dict[str, dict[str, Any]] = {}
    definitions: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for suite in SUITES:
        pr_path = COHORT / "cohort" / f"run_ids.prompted.{suite}.txt"
        s0_path = COHORT / "cohort" / f"run_ids.trained.{suite}.txt"
        seed_paths = {
            s: SEEDREP / "run_ids" / f"run_ids.{m}.{suite}.txt" for s, m in RECIPE.items()
        }
        for p in (pr_path, s0_path, *seed_paths.values()):
            sources[p.relative_to(REPO).as_posix()] = _sha(p)
        pr = _ids(pr_path)
        s1, s2 = _ids(seed_paths["s1"]), _ids(seed_paths["s2"])
        arms = {
            "s0": (COHORT / "scores_parquet", _ids(s0_path)),
            "s1": (SEEDREP / "scores_parquet", s1),
            "s2": (SEEDREP / "scores_parquet", s2),
            "s1s2": (SEEDREP / "scores_parquet", s1 + s2),
        }
        base_cov = per_task(con, COHORT / "scores_parquet", pr, "evidence_coverage")
        base_f1 = per_task(con, COHORT / "scores_parquet", pr, METRIC)
        for k, (store, ids) in arms.items():
            # L2: this per-task function on coverage is the own-stop coverage row's
            cov = _ten(
                contrast(per_task(con, store, ids, "evidence_coverage"), base_cov, ((10_000, 0),))
            )
            want = cap8["cells"][k][suite]
            out["lock"]["L2_own_stop_population"][f"{k}::{suite}"] = lock(
                f"L2 cap8_by_seed {k} {suite}",
                cov,
                {"point": want["delta"], "lo": want["ci_lo"], "hi": want["ci_hi"], "n": want["n"]},
            )
            a = per_task(con, store, ids, METRIC)
            readings = contrast(a, base_f1, RESAMPLES if k == "s1s2" else ((10_000, 0),))
            ten = _ten(readings)
            shared = sorted(set(a) & set(base_f1))
            cell = {
                "delta": ten["point"],
                "ci_lo": ten["lo"],
                "ci_hi": ten["hi"],
                "n": ten["n"],
                "level_trained": sum(a[x] for x in shared) / len(shared),
                "level_prompted": sum(base_f1[x] for x in shared) / len(shared),
            }
            if k == "s1s2":
                cell["readings"] = [
                    {
                        "n_boot": r["n_boot"],
                        "seed": r["seed"],
                        "delta": r["point"],
                        "ci_lo": r["lo"],
                        "ci_hi": r["hi"],
                        "n": r["n"],
                    }
                    for r in readings
                ]
                cell["decided"] = decided(cell["readings"])
                # (task, rollout seed) pairing gives the same per-task difference
                ka, kb = pair_on_rollout_seed(
                    [per_key(con, store, s1, METRIC), per_key(con, store, s2, METRIC)],
                    per_key(con, COHORT / "scores_parquet", pr, METRIC),
                )
                worst = max(abs((ka[t] - kb[t]) - (a[t] - base_f1[t])) for t in ka)
                if set(ka) != set(a) or worst > TOL:
                    raise SystemExit(
                        f"{suite}: pairing on rollout seed moves a task by {worst}. NOT WRITING."
                    )
                cell["pairing_check_max_abs_diff"] = worst
                # L3: decompose.json's pooled reading, under its clusters
                rn = (store / "runs.parquet").as_posix()
                tmpl = dict(
                    con.execute(
                        f"SELECT DISTINCT task_id, template_id FROM read_parquet('{rn}') "
                        "WHERE run_id IN (SELECT unnest(?))",
                        [list(ids)],
                    ).fetchall()
                )
                clusters = {t: str(v) for t, v in tmpl.items() if v} if any(tmpl.values()) else None
                d_rec = dec["per_suite"][suite]["readings"][METRIC]
                got3 = _ten(contrast(a, base_f1, ((10_000, 0),), clusters=clusters))
                out["lock"]["L3_decompose"][suite] = {
                    **lock(
                        f"L3 decompose.json {suite}",
                        got3,
                        {
                            "point": d_rec["point"],
                            "lo": d_rec["ci_lo"],
                            "hi": d_rec["ci_hi"],
                            "n": d_rec["n_tasks"],
                        },
                    ),
                    "clusters": "template_id" if clusters else "task",
                    "n_clusters": len(set(clusters.values())) if clusters else len(shared),
                }
            cells.setdefault(k, {})[suite] = cell
        definitions[suite] = {
            "recipe_s1s2": definition(con, SEEDREP / "scores_parquet", s1 + s2),
            "comparator": definition(con, COHORT / "scores_parquet", pr),
        }

    out.update(
        {
            "cells": cells,
            "definition": definitions,
            "rule": (
                "own stop under the shared ceiling of eight, no matching; per task, the mean "
                "answer_token_f1 over the arm's runs (training and rollout seeds averaged into the "
                "task); paired_difference over tasks, unclustered, 10,000 resamples at seed 0; the "
                "pooled cell also at 50,000 under seeds 101/202/303, decided only if all three "
                "exclude zero on one side"
            ),
            "definition_note": (
                "answer_token_f1 = max over the gold answer and its aliases of the SQuAD token F1 "
                "(src/pi_eval/metrics/quality.py token_f1, decompose_over_aliases), emitted by "
                "src/pi_eval/score.py whenever the graph carries a gold answer. On StrategyQA the "
                "gold answer is 'yes' or 'no' (aliases e.g. 'false'/'no'), so a run scores "
                "2/(n_answer_tokens + 1) when its answer contains the gold word and 0 otherwise."
            ),
            "sources": {
                **sources,
                "artifacts/seed_identity_20260923/cap8_by_seed.json": _sha(
                    SI / "cap8_by_seed.json"
                ),
                "artifacts/seed_identity_20260923/stop_s1.json": _sha(SI / "stop_s1.json"),
                "artifacts/seed_identity_20260923/stop_s2.json": _sha(SI / "stop_s2.json"),
                "artifacts/answer_loss_20260923/decompose.json": _sha(DECOMPOSE),
            },
            "scorer_hash": SCORER_HASH,
            "graph_version": "v1",
            "command": "scripts/seed_identity/answer_f1_by_seed.py --out artifacts/seed_identity_20260923/answer_f1_by_seed.json",
        }
    )
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k in ("s1", "s2", "s1s2"):
        print(
            k,
            "  ".join(
                f"{s} {cells[k][s]['delta']:+.6f} [{cells[k][s]['ci_lo']:+.6f},{cells[k][s]['ci_hi']:+.6f}]"
                + (f" {'D' if cells[k][s]['decided'] else 'span'}" if k == "s1s2" else "")
                for s in SUITES
            ),
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
