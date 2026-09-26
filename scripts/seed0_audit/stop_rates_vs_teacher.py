"""Stop-when-done and ask-when-not-done RATES against the 120B prompted model, for the recipe.

WHY. App stopping prints "Seed 0 carries the same calibration against the model fifteen times
larger" (+0.576 / +0.706 / +0.180 stop-when-done, -0.046 / -0.174 / -0.040 ask-when-not-done),
read on the registered adapter, the resumed run. This reads the recipe, training seeds 1 and 2.

WHAT. `scripts/measure_20260919/item4_stop2x2_rates.py`, replicated rather than imported because
its `lib.py` hardcodes a scratch store that no longer exists: per task, the seed-averaged
`stop2x2_n_stop_at_done` over the seed-averaged `stop2x2_n_done` (and the not-done pair), kept
where the arm's own denominator is positive, paired on the tasks both arms keep, and
`pi_eval.stats.inference.paired_difference` with template clusters, 10,000 resamples, seed 0. The
recipe pools seeds 1 and 2 WITHIN the task: its per-task averages run over both seeds' runs, as
Table 1 pools them.

STORE. `artifacts/seed_identity_20260923/teacher_store` (4,800 runs: the 120B teacher of
artifacts/n6 and seeds 0, 1 and 2, one scoring pass, scorer e82c7458).

LOCK. Seed 0 must reproduce the six printed cells and their n before any recipe cell is written.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb

from pi_eval.stats.inference import cluster_bootstrap, paired_difference

SCORER = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
TEACHER = "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf"
PINS = {
    "s0": ["6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856"],
    "s1": ["de269316d0de12fcb603ab55edfee535d715d37dc90824c47c8f3b35d055d438"],
    "s2": ["c46ea7ed802722d7d7a9f06eec6952fe4a16ee291626f9c46ab6b3ceb963d7f9"],
}
PINS["s1s2"] = PINS["s1"] + PINS["s2"]
SUITES = ("musique", "strategyqa", "wiki2")
RATES = {
    "stop_when_done": ("stop2x2_n_stop_at_done", "stop2x2_n_done"),
    "ask_when_not_done": ("stop2x2_n_ask_at_not_done", "stop2x2_n_not_done"),
}
# What the tex prints for seed 0 (tex:4835-4844 at da38372): (delta, lo, hi, n), four decimals.
PRINTED_S0 = {
    ("stop_when_done", "musique"): ("+0.5760", "+0.4860", "+0.6633", 101),
    ("stop_when_done", "strategyqa"): ("+0.7061", "+0.6258", "+0.7745", 104),
    ("stop_when_done", "wiki2"): ("+0.1797", "+0.1280", "+0.2381", 169),
    ("ask_when_not_done", "musique"): ("-0.0459", "-0.0729", "-0.0189", 200),
    ("ask_when_not_done", "strategyqa"): ("-0.1739", "-0.2032", "-0.1438", 200),
    ("ask_when_not_done", "wiki2"): ("-0.0396", "-0.0617", "-0.0225", 200),
}
ELIGIBLE = """
    r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE
    AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE
    AND r.exploratory = FALSE
    AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none'
    AND r.split = 'test'
    AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE
"""


def per_task(con, store: Path, metric: str, suite: str, pins: list[str]) -> dict:
    q = f"""
    SELECT r.task_id, COALESCE(NULLIF(r.template_id,''), r.task_id) AS cluster_id,
           avg(s.value) AS value, count(*) AS n_runs
    FROM read_parquet('{(store / "scores.parquet").as_posix()}') s
    JOIN read_parquet('{(store / "runs.parquet").as_posix()}') r ON r.run_id = s.run_id
    WHERE s.metric_name = ? AND s.scorer_hash = ? AND r.suite_id = ?
      AND r.model_pin_hash IN ({",".join("?" * len(pins))}) AND ({ELIGIBLE})
    GROUP BY 1, 2
    """
    rows = con.execute(q, [metric, SCORER, suite, *pins]).fetchall()
    return {t: (v, c, n) for t, c, v, n in rows}


def rate(con, store, suite, pins, num, den):
    a = per_task(con, store, num, suite, pins)
    b = per_task(con, store, den, suite, pins)
    out, clusters = {}, {}
    for k in set(a) & set(b):
        clusters[k] = b[k][1]
        if b[k][0] and b[k][0] > 0:
            out[k] = a[k][0] / b[k][0]
    runs_per_task = sorted({b[k][2] for k in b})
    return out, clusters, runs_per_task


def level(values: dict, clusters: dict) -> dict:
    groups: dict = {}
    for k, v in values.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        groups.setdefault(clusters.get(k, k), []).append(v)
    p, lo, hi = cluster_bootstrap(list(groups.values()), n_boot=10_000, seed=0)
    return {"point": p, "lo": lo, "hi": hi}


def contrast(con, store, suite, pins, which, *, n_boot=10_000, seed=0) -> dict:
    num, den = RATES[which]
    t, tc, tn = rate(con, store, suite, pins, num, den)
    e, ec, _ = rate(con, store, suite, [TEACHER], num, den)
    clusters = {**tc, **ec}
    keys = sorted(set(t) & set(e))
    cl = {k: clusters[k] for k in keys}
    est = paired_difference(
        {k: t[k] for k in keys}, {k: e[k] for k in keys}, clusters=cl, n_boot=n_boot, seed=seed
    )
    return {
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "n": est.n,
        "n_trained_denominator_positive": len(t),
        "n_teacher_denominator_positive": len(e),
        "trained_runs_per_task": tn,
        "trained_level": level({k: t[k] for k in keys}, cl),
        "teacher_level": level({k: e[k] for k in keys}, cl),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    con = duckdb.connect()
    res: dict = {
        "store": "artifacts/seed_identity_20260923/teacher_store",
        "scorer_hash": SCORER,
        "teacher_pin": TEACHER,
        "pins": PINS,
        "lock": {},
        "cells": {},
    }
    for which in RATES:
        for suite in SUITES:
            got = contrast(con, a.store, suite, PINS["s0"], which)
            want = PRINTED_S0[(which, suite)]
            printed = (
                f"{got['delta']:+.4f}",
                f"{got['ci_lo']:+.4f}",
                f"{got['ci_hi']:+.4f}",
                got["n"],
            )
            ok = printed == want
            res["lock"][f"{which}.{suite}"] = {"printed": want, "reproduced": printed, "ok": ok}
            print(f"LOCK s0 {which} {suite}: printed {want} reproduced {printed} ok={ok}")
            if not ok:
                raise SystemExit("LOCK FAILED")
            res["cells"].setdefault("s0", {})[f"{which}.{suite}"] = got
    for tag in ("s1", "s2", "s1s2"):
        for which in RATES:
            for suite in SUITES:
                got = contrast(con, a.store, suite, PINS[tag], which)
                if tag == "s1s2":
                    got["reads_50k"] = [
                        {
                            k: contrast(
                                con, a.store, suite, PINS[tag], which, n_boot=50_000, seed=s
                            )[k]
                            for k in ("ci_lo", "ci_hi")
                        }
                        | {"seed": s}
                        for s in (101, 202, 303)
                    ]
                res["cells"].setdefault(tag, {})[f"{which}.{suite}"] = got
                tl, el = got["trained_level"]["point"], got["teacher_level"]["point"]
                print(
                    f"{tag:4s} {which:17s} {suite:10s} {got['delta']:+.4f} [{got['ci_lo']:+.4f}, {got['ci_hi']:+.4f}] "
                    f"n={got['n']} levels {tl:.4f} vs {el:.4f} runs/task={got['trained_runs_per_task']}"
                    + (
                        f" 50k={[(round(r['ci_lo'], 4), round(r['ci_hi'], 4)) for r in got['reads_50k']]}"
                        if tag == "s1s2"
                        else ""
                    )
                )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
