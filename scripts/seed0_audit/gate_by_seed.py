"""The selection gate's own verdict for the recipe's two fresh seeds pooled, per suite.

WHY. The development tables of the paper (tab:app-matchedcoverage, tab:app-train-coverage,
tab:app-train-behavior, tab:app-stop-composition, tab:app-stop-instruments, the development and
held-out columns of tab:app-heldout-coverage and tab:app-heldout-gates) print the registered
adapter `qwen3-8b-dpo-stacked-notdone-both`, i.e. seed 0, whose final preference stage resumed
from checkpoint-1800 and kept ~9% of the stage (artifacts/seed_identity_20260923/RESULT.md).

WHAT. `pinq_train.gate.run_gate` itself, unchanged, the function every one of those cells came
from. Two things are supplied from outside, both by wrapping `gate._select_runs`, the one
selection call `run_gate` makes for each arm:
  * a SUITE filter, because the held-out verdicts were each built on a one-suite store and the
    held-out union store (union_store.py) holds all three suites;
  * a POOLED checkpoint id, `POOL`, that selects seed 1's runs and seed 2's runs together, so the
    two training seeds share each (suite, task, seed) key. Every criterion then pools them the
    way Table 1 does: `_matched_cost` averages within the key and then within the task, which is
    the flat mean over (training seed, rollout seed) when every key holds one run of each (checked
    and recorded, `balanced`), `_paired_by_task` does the same for the cap-8 criteria, and the
    stop 2x2 and length criteria count every decision point and question of both seeds.
  `distinct3` is the one criterion where pooling two training seeds changes WHAT is measured
  (two policies' questions in one task group); its pooled value is recorded and never used, and
  the per-seed values are the ones to read.

LOCKS, each a hard stop before a pooled number is written:
  dev      seeds 0, 1 and 2 re-gated here equal their committed verdicts
           (artifacts/seed_identity_20260923/dev_gate/*.json) on every compared field to 1e-12,
           and seed 0 equals what the paper prints for it.
  heldout  on the COMPLETED comparator (union_store.py; run_gate refuses the published short
           comparator with IncompleteArm), 10,000 resamples: seed 0's coverage fields equal
           artifacts/baseline_completion_20260920/three.json, and its checkpoint-only fields
           (diversity, length, malformed, stop 2x2 counts) equal the published held-out verdicts
           (artifacts/testsplit_qa/verdicts/stacked-notdone.*.json), all to 1e-12.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pinq_train.gate as gate

S0 = "qwen3-8b-dpo-stacked-notdone-both"
S1, S2 = f"{S0}-s1", f"{S0}-s2"
POOL = "POOL:s1+s2"
BASE = "qwen3-8b-base"
DEV_SCORER = "3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"
HO_SCORER = "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"

_orig_select = gate._select_runs


def _patched(suite: str | None):
    def select(con, *, arm, grids, model_id):
        if model_id == POOL:
            rows = _orig_select(con, arm=arm, grids=grids, model_id=S1)
            rows += _orig_select(con, arm=arm, grids=grids, model_id=S2)
            rows.sort(key=lambda r: (r["suite_id"], r["task_id"], r["seed"], r["run_id"]))
        else:
            rows = _orig_select(con, arm=arm, grids=grids, model_id=model_id)
        if suite is not None:
            rows = [r for r in rows if r["suite_id"] == suite]
        return rows

    return select


def gate_once(*, store, suite, grid, base_grid, model, scorer, grids_root, n_resamples=1000):
    gate._select_runs = _patched(suite)
    try:
        return gate.run_gate(
            parquet_dir=store,
            grid_name=grid,
            baseline_grid_names=[base_grid],
            baseline_model_id=BASE,
            checkpoint_model_id=model,
            scorer_hash=scorer,
            bootstrap_seed=0,
            n_resamples=n_resamples,
            grids_root=grids_root,
            coverage_rule="matched_cost",
        )
    finally:
        gate._select_runs = _orig_select


def fields(v: dict, suite: str) -> dict:
    """The fields the paper prints, flattened, so a lock can compare them one by one."""
    c, mc = v["criteria"], v["matched_cost"]["by_suite"][suite]
    st = c["stop_2x2"]["value"]
    out = {
        "mc.delta": mc["delta"],
        "mc.ci_lo": mc["ci_lo"],
        "mc.ci_hi": mc["ci_hi"],
        "mc.n_tasks": mc["n_tasks"],
        "mc.trained_mean_k": mc["trained_mean_k"],
        "mc.baseline_mean_k_charged": mc["baseline_mean_k_charged"],
        "mc.baseline_mean_n_asks": mc["baseline_mean_n_asks"],
        "mc.n_baseline_shorter_than_k": mc["n_baseline_shorter_than_k"],
        "mc.cap8_coverage_delta": mc["cap8_coverage_delta"],
        "mc.symmetric.delta": (mc.get("symmetric") or {}).get("delta"),
        "mc.symmetric.ci_lo": (mc.get("symmetric") or {}).get("ci_lo"),
        "mc.symmetric.ci_hi": (mc.get("symmetric") or {}).get("ci_hi"),
        "stop.p_stop_given_done": st["p_stop_given_done"],
        "stop.p_ask_given_not_done": st["p_ask_given_not_done"],
        "stop.n_done": st["n_done"],
        "stop.n_not_done": st["n_not_done"],
        "stop.n_forced_stops": st["n_forced_stops"],
        "stop.mean_asks_after_done": st["mean_asks_after_done"],
        "stop.n_runs": st["n_runs"],
        "distinct3.value": c["distinct3"]["value"],
        "length.value": c["length_equivalence"]["value"],
        "length.baseline": c["length_equivalence"].get("baseline"),
        "length.ci_lo": c["length_equivalence"]["ci_lo"],
        "length.ci_hi": c["length_equivalence"]["ci_hi"],
        "length.passed": c["length_equivalence"]["passed"],
        "length.n": c["length_equivalence"]["n"],
        "malformed.value": c["malformed"]["value"],
    }
    for k in ("cad_ge2", "facet_breadth", "newly_reachable_share"):
        if k in c:
            cc = c[k]
            out[f"{k}.value"] = cc.get("cap8_cad_ge2_delta", cc.get("value"))
            out[f"{k}.ci_lo"] = cc.get("ci_lo")
            out[f"{k}.ci_hi"] = cc.get("ci_hi")
            out[f"{k}.n"] = cc.get("n")
    if "by_seed" in c["distinct3"]:
        out["distinct3.by_seed"] = c["distinct3"]["by_seed"]
    return out


def compare(got: dict, want: dict, tol=1e-12) -> dict:
    diff = {}
    for k, w in want.items():
        g = got.get(k)
        if isinstance(w, bool) or w is None or g is None or isinstance(g, bool):
            ok = g == w
        else:
            ok = abs(float(g) - float(w)) <= tol
        diff[k] = {"got": g, "want": w, "ok": ok}
    return diff


def balanced(store, suite, grid) -> dict:
    con = gate._con(Path(store))
    k1 = gate._by_key(
        [
            r
            for r in _orig_select(con, arm="inquirer_trained", grids=[grid], model_id=S1)
            if r["suite_id"] == suite
        ]
    )
    k2 = gate._by_key(
        [
            r
            for r in _orig_select(con, arm="inquirer_trained", grids=[grid], model_id=S2)
            if r["suite_id"] == suite
        ]
    )
    return {
        "keys_s1": len(k1),
        "keys_s2": len(k2),
        "same_keys": set(k1) == set(k2),
        "one_run_per_key": all(len(v) == 1 for v in (*k1.values(), *k2.values())),
    }


# What the paper prints for seed 0 on development (tab:app-matchedcoverage, tab:app-train-coverage,
# tab:app-train-behavior), as (field, printed string, format).
DEV_PRINTED = {
    "musique": [
        ("mc.delta", "+0.1604", "{:+.4f}"),
        ("mc.ci_lo", "+0.1225", "{:+.4f}"),
        ("mc.ci_hi", "+0.2088", "{:+.4f}"),
        ("mc.cap8_coverage_delta", "+0.0379", "{:+.4f}"),
        ("mc.trained_mean_k", "2.45", "{:.2f}"),
        ("mc.baseline_mean_k_charged", "2.42", "{:.2f}"),
        ("cad_ge2.value", "+0.0588", "{:+.4f}"),
        ("stop.p_stop_given_done", "0.9375", "{:.4f}"),
        ("stop.p_ask_given_not_done", "0.8507", "{:.4f}"),
        ("distinct3.value", "0.9514", "{:.4f}"),
        ("length.value", "13.23", "{:.2f}"),
        ("stop.n_not_done", "375", "{:d}"),
        ("stop.n_done", "80", "{:d}"),
    ],
    "strategyqa": [
        ("mc.delta", "+0.1030", "{:+.4f}"),
        ("mc.ci_lo", "+0.0766", "{:+.4f}"),
        ("mc.ci_hi", "+0.1315", "{:+.4f}"),
        ("mc.cap8_coverage_delta", "-0.0382", "{:+.4f}"),
        ("mc.trained_mean_k", "1.31", "{:.2f}"),
        ("mc.baseline_mean_k_charged", "1.29", "{:.2f}"),
        ("cad_ge2.value", "-0.1022", "{:+.4f}"),
        ("stop.p_stop_given_done", "0.9225", "{:.4f}"),
        ("stop.p_ask_given_not_done", "0.8533", "{:.4f}"),
        ("distinct3.value", "0.9216", "{:.4f}"),
        ("length.value", "18.07", "{:.2f}"),
        ("stop.n_not_done", "484", "{:d}"),
        ("stop.n_done", "284", "{:d}"),
    ],
}


def run_dev(data: Path, grids_root: Path) -> dict:
    store = data / "artifacts/seed_identity_20260923/dev_gate/store"
    vdir = data / "artifacts/seed_identity_20260923/dev_gate"
    out: dict = {
        "store": "artifacts/seed_identity_20260923/dev_gate/store",
        "scorer_hash": DEV_SCORER,
        "n_resamples": 1000,
        "locks": {},
        "cells": {},
    }
    for suite in ("musique", "strategyqa"):
        grid, bgrid = f"dev_select_{suite}", f"dev_baseline_{suite}"
        for tag, model in (("s0", S0), ("s1", S1), ("s2", S2)):
            v = gate_once(
                store=store,
                suite=None,
                grid=grid,
                base_grid=bgrid,
                model=model,
                scorer=DEV_SCORER,
                grids_root=grids_root,
            )
            got = fields(v, suite)
            want = fields(json.loads((vdir / f"{model}.{suite}.json").read_text()), suite)
            cmp_ = compare(got, want)
            bad = [k for k, x in cmp_.items() if not x["ok"]]
            out["locks"][f"{tag}.{suite}.verdict"] = {"n_fields": len(cmp_), "mismatch": bad}
            print(
                f"LOCK dev {tag} {suite} vs committed verdict: {len(cmp_) - len(bad)}/{len(cmp_)} fields equal",
                bad,
            )
            if bad:
                raise SystemExit(f"DEV VERDICT LOCK FAILED {tag} {suite}: {bad}")
            out["cells"].setdefault(tag, {})[suite] = got
            if tag == "s0":
                pr = {f: (fmt.format(got[f]), p) for f, p, fmt in DEV_PRINTED[suite]}
                badp = {f: x for f, x in pr.items() if x[0] != x[1]}
                out["locks"][f"s0.{suite}.printed"] = {"checked": pr, "mismatch": badp}
                print(
                    f"LOCK dev s0 {suite} vs paper: {len(pr) - len(badp)}/{len(pr)} printed cells",
                    badp,
                )
                if badp:
                    raise SystemExit(f"DEV PRINTED LOCK FAILED {suite}: {badp}")
        out["locks"][f"pool.{suite}.balanced"] = b = balanced(store, suite, grid)
        if not (b["same_keys"] and b["one_run_per_key"]):
            raise SystemExit(f"POOL NOT BALANCED {suite}: {b}")
        v = gate_once(
            store=store,
            suite=None,
            grid=grid,
            base_grid=bgrid,
            model=POOL,
            scorer=DEV_SCORER,
            grids_root=grids_root,
        )
        out["cells"].setdefault("s1s2", {})[suite] = fields(v, suite)
    return out


def run_heldout(data: Path, store: Path) -> dict:
    """The held-out gate on the COMPLETED comparator (the population Table 1 reads), 10,000
    resamples, from a union store built by union_store.py. The published held-out verdicts were
    taken on the short comparator (176/200/166 tasks) and `run_gate` now refuses that population
    (IncompleteArm: 26 MuSiQue tasks have trained runs and no comparator run), so the lock is split:
    the coverage fields must equal three.json's completed-cohort matched-cost cell exactly (the
    same function at the same count), and every CHECKPOINT-ONLY field (diversity, question length,
    malformed rate, the stop 2x2 counts), which the comparator cannot move, must equal the
    published verdict."""
    three = json.loads((data / "artifacts/baseline_completion_20260920/three.json").read_text())
    vdir = data / "artifacts/testsplit_qa/verdicts"
    out: dict = {
        "store": "union of artifacts/completed_cohort_20260922/scores_parquet and the seed-1/2 runs "
        "of artifacts/seedrep_gate_20260919/scores_parquet (scripts/seed0_audit/union_store.py)",
        "scorer_hash": HO_SCORER,
        "n_resamples": 10000,
        "comparator": "the completed cohort's 1,200 prompted runs, 200 tasks per suite",
        "locks": {},
        "cells": {},
    }
    grid = "tier1_trained_qa_base"
    for suite in ("musique", "strategyqa", "wiki2"):
        v0 = gate_once(
            store=store,
            suite=suite,
            grid=grid,
            base_grid=grid,
            model=S0,
            scorer=HO_SCORER,
            grids_root=None,
            n_resamples=10000,
        )
        got = fields(v0, suite)
        t = three["complete"]["by_suite"][suite]
        want_cov = {
            "mc.delta": t["delta"],
            "mc.ci_lo": t["ci_lo"],
            "mc.ci_hi": t["ci_hi"],
            "mc.n_tasks": t["n_tasks"],
            "mc.trained_mean_k": t["trained_mean_k"],
            "mc.baseline_mean_k_charged": t["baseline_mean_k_charged"],
            "mc.n_baseline_shorter_than_k": t["n_baseline_shorter_than_k"],
            "mc.cap8_coverage_delta": t["cap8_coverage_delta"],
            "mc.symmetric.delta": t["symmetric"]["delta"],
        }
        pub = fields(json.loads((vdir / f"stacked-notdone.{suite}.json").read_text()), suite)
        want_ck = {
            k: pub[k]
            for k in (
                "distinct3.value",
                "length.value",
                "length.n",
                "malformed.value",
                "stop.p_stop_given_done",
                "stop.p_ask_given_not_done",
                "stop.n_done",
                "stop.n_not_done",
                "stop.n_forced_stops",
                "stop.n_runs",
            )
        }
        c1, c2 = compare(got, want_cov), compare(got, want_ck)
        bad = [k for k, x in {**c1, **c2}.items() if not x["ok"]]
        out["locks"][f"s0.{suite}"] = {"three_json": c1, "published_checkpoint_only": c2}
        print(
            f"LOCK heldout s0 {suite}: three.json {sum(x['ok'] for x in c1.values())}/{len(c1)}, "
            f"checkpoint-only {sum(x['ok'] for x in c2.values())}/{len(c2)} {bad}"
        )
        if bad:
            raise SystemExit(f"HELDOUT LOCK FAILED {suite}: {bad}")
        out["cells"].setdefault("s0", {})[suite] = got
        for tag, model in (("s1", S1), ("s2", S2), ("s1s2", POOL)):
            if tag == "s1s2":
                out["locks"][f"pool.{suite}.balanced"] = b = balanced(store, suite, grid)
                if not (b["same_keys"] and b["one_run_per_key"]):
                    raise SystemExit(f"POOL NOT BALANCED {suite}: {b}")
            v = gate_once(
                store=store,
                suite=suite,
                grid=grid,
                base_grid=grid,
                model=model,
                scorer=HO_SCORER,
                grids_root=None,
                n_resamples=10000,
            )
            out["cells"].setdefault(tag, {})[suite] = fields(v, suite)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("which", choices=("dev", "heldout"))
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--grids-root", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None, help="heldout: the union store")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = (
        run_dev(a.data_root, a.grids_root)
        if a.which == "dev"
        else run_heldout(a.data_root, a.store)
    )
    for tag, by in res["cells"].items():
        for suite, f in by.items():
            print(
                f"{a.which} {tag:4s} {suite:10s} mc {f['mc.delta']:+.4f} [{f['mc.ci_lo']:+.4f}, {f['mc.ci_hi']:+.4f}] "
                f"n={f['mc.n_tasks']} k={f['mc.trained_mean_k']:.3f} charged={f['mc.baseline_mean_k_charged']:.3f} "
                f"cap8={f['mc.cap8_coverage_delta']:+.4f} cad={f.get('cad_ge2.value')} "
                f"stop={f['stop.p_stop_given_done']:.4f} ask={f['stop.p_ask_given_not_done']:.4f} "
                f"done={f['stop.n_done']} notdone={f['stop.n_not_done']} d3={f['distinct3.value']:.4f} "
                f"words={f['length.value']:.2f} lenCI=[{f['length.ci_lo']:+.3f},{f['length.ci_hi']:+.3f}] "
                f"lenpass={f['length.passed']} mal={f['malformed.value']:.4f}"
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
