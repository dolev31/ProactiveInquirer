"""The eight controls per suite and pooled for the recipe, under the published controls' rule.

WHY. tab:app-controls-pooled and tab:app-controls-suite print the controls against seed 0 (the
resumed run) and do not say so. The recipe's POOLED model-bearing controls exist
(artifacts/seed_identity_20260923/controls_recipe.json) and its weight-free ones
(controls_by_seed.json s1s2); no per-suite recipe reading does.

WHAT. The published rule and estimator, from the two readers that already carry it:
  * weight-free controls (random question, state-blind template, answer length, compute budget):
    scripts/seed_identity/controls_by_seed.py's contrast, replicated with a SUITE filter, on its
    store (artifacts/seed_identity_20260923/controls_store: the published farm plus seeds 1 and 2);
  * model-bearing controls (depth one, evidence blind, single model): each seed's treatment against
    its OWN ablation, scripts/seed_identity/controls_recipe.py's Store / per_seed_values / contrast
    imported unchanged, with the per-task values filtered to one suite before the contrast.
  matched = the control at the largest prefix of its own trajectory within the paired trained run's
  retrieval_calls; unmatched = each arm's terminal coverage; paired_difference, template clusters,
  n_boot=1000, n_perm=10000, seed 0; the recipe averages both seeds within the task.
The determinism check (parallel_replay) reissues SEED 0's recorded questions, so it has no recipe
reading: its runs replay the resumed run, not the recipe. It is read for seed 0 only.

LOCK. Seed 0, per suite and pooled, must reproduce every printed cell of both tables (points and
bounds at four decimals, n where printed) before any recipe cell is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))
import controls_recipe as cr  # noqa: E402

S0 = "qwen3-8b-dpo-stacked-notdone-both"
FREE = ("random_q", "checklist", "verbosity", "compute_matched", "parallel_replay")
BEARING = ("inquirer_depth1", "inquirer_noevidence", "self_inquire")
SUITES = ("musique", "strategyqa")
# Printed seed-0 cells, (matched point, lo, hi) and (unmatched point, lo, hi); None where the tex
# prints only a point. pooled from tab:app-controls-pooled, per suite from tab:app-controls-suite.
P = {
    ("parallel_replay", "pooled"): (
        ("+0.0902", "+0.0615", "+0.1204"),
        ("-0.0440", "-0.0699", "-0.0190"),
    ),
    ("verbosity", "pooled"): (("+0.8006", "+0.7753", "+0.8252"), ("+0.8006", "+0.7753", "+0.8252")),
    ("compute_matched", "pooled"): (
        ("+0.8000", "+0.7748", "+0.8251"),
        ("+0.8000", "+0.7748", "+0.8251"),
    ),
    ("random_q", "pooled"): (("+0.5151", "+0.4801", "+0.5523"), ("+0.2848", "+0.2482", "+0.3241")),
    ("checklist", "pooled"): (("+0.1995", "+0.1686", "+0.2325"), ("+0.1064", "+0.0747", "+0.1362")),
    ("inquirer_noevidence", "pooled"): (
        ("+0.0662", "+0.0440", "+0.0886"),
        ("+0.0220", "+0.0008", "+0.0447"),
    ),
    ("inquirer_depth1", "pooled"): (
        ("+0.0986", "+0.0698", "+0.1267"),
        ("+0.0790", "+0.0495", "+0.1070"),
    ),
    ("self_inquire", "pooled"): (
        ("+0.0143", "-0.0023", "+0.0297"),
        ("-0.0121", "-0.0274", "+0.0027"),
    ),
    ("parallel_replay", "musique"): (
        ("+0.1241", "+0.0858", "+0.1655"),
        ("+0.0244", "-0.0114", "+0.0640"),
    ),
    ("verbosity", "musique"): (("+0.8338", "+0.8001", "+0.8646"), ("+0.8338", None, None)),
    ("compute_matched", "musique"): (("+0.8326", "+0.7988", "+0.8629"), ("+0.8326", None, None)),
    ("random_q", "musique"): (("+0.3584", "+0.3151", "+0.4030"), ("+0.1656", "+0.1223", "+0.2114")),
    ("checklist", "musique"): (
        ("+0.3200", "+0.2782", "+0.3627"),
        ("+0.2388", "+0.1994", "+0.2815"),
    ),
    ("inquirer_noevidence", "musique"): (
        ("+0.1074", "+0.0760", "+0.1448"),
        ("+0.0400", "+0.0043", "+0.0801"),
    ),
    ("inquirer_depth1", "musique"): (
        ("+0.1614", "+0.1264", "+0.1999"),
        ("+0.1293", "+0.0932", "+0.1693"),
    ),
    ("self_inquire", "musique"): (
        ("+0.0375", "+0.0164", "+0.0603"),
        ("+0.0073", "-0.0129", "+0.0279"),
    ),
    ("parallel_replay", "strategyqa"): (
        ("+0.0628", "+0.0247", "+0.1036"),
        ("-0.0994", "-0.1306", "-0.0710"),
    ),
    ("verbosity", "strategyqa"): (("+0.7713", "+0.7368", "+0.8044"), ("+0.7713", None, None)),
    ("compute_matched", "strategyqa"): (("+0.7713", "+0.7368", "+0.8044"), ("+0.7713", None, None)),
    ("random_q", "strategyqa"): (
        ("+0.6420", "+0.6016", "+0.6864"),
        ("+0.3813", "+0.3325", "+0.4338"),
    ),
    ("checklist", "strategyqa"): (
        ("+0.0873", "+0.0530", "+0.1261"),
        ("-0.0167", "-0.0526", "+0.0192"),
    ),
    ("inquirer_noevidence", "strategyqa"): (
        ("+0.0295", "+0.0072", "+0.0571"),
        ("+0.0061", "-0.0184", "+0.0328"),
    ),
    ("inquirer_depth1", "strategyqa"): (
        ("+0.0432", "+0.0074", "+0.0780"),
        ("+0.0346", "-0.0005", "+0.0697"),
    ),
    ("self_inquire", "strategyqa"): (
        ("-0.0063", "-0.0311", "+0.0142"),
        ("-0.0293", "-0.0545", "-0.0101"),
    ),
}


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def free_reader(data: Path):
    """controls_by_seed.py's runs/coverage/ladder load and contrast, with a suite filter."""
    store = data / "artifacts/seed_identity_20260923/controls_store"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW runs AS SELECT * FROM read_parquet('{store}/runs.parquet')")
    con.execute(f"CREATE VIEW scores AS SELECT * FROM read_parquet('{store}/scores.parquet')")
    excl = set(
        _ids(
            data / "artifacts/nulls_and_determinism_20260918/EXCLUDE_random_q_zero_ask.eligible.txt"
        )
    )
    runs = {
        r[0]: {
            "arm": r[1],
            "suite": r[2],
            "task": r[3],
            "seed": int(r[4]),
            "calls": float(r[5]),
            "cluster": r[6],
        }
        for r in con.execute(
            "SELECT r.run_id, r.arm_id, r.suite_id, r.task_id, r.seed, r.retrieval_calls, "
            "COALESCE(NULLIF(r.template_id, ''), r.task_id) "
            f"FROM runs r WHERE {ELIGIBLE} AND r.suite_id IN ('musique','strategyqa')"
        ).fetchall()
        if r[0] not in excl
    }
    cov, spend, qq = {}, {}, {}
    for rid, name, v in con.execute(
        "SELECT run_id, metric_name, value FROM scores WHERE metric_name = 'evidence_coverage' "
        "OR metric_name LIKE 'frontier_spend#%' OR metric_name LIKE 'frontier_q#%'"
    ).fetchall():
        if rid not in runs or v is None:
            continue
        if name == "evidence_coverage":
            cov[rid] = float(v)
        else:
            kind, k = name.split("#")
            (spend if kind == "frontier_spend" else qq).setdefault(rid, {})[int(k)] = float(v)

    def matched_value(rid, budget):
        sp, q = spend.get(rid), qq.get(rid)
        if not sp or not q:
            return None
        ks = [k for k in sorted(sp) if sp[k] <= budget + 1e-9]
        kstar = max(ks) if ks else 0
        while kstar not in q and kstar > 0:
            kstar -= 1
        return q.get(kstar)

    rd = data / "artifacts/seedrep_gate_20260919/run_ids"
    treat = {
        "s0": set(_ids(data / "artifacts/killswitch/run_ids.treatment.txt")),
        "s1": {i for s in SUITES for i in _ids(rd / f"run_ids.{S0}-s1.{s}.txt")},
        "s2": {i for s in SUITES for i in _ids(rd / f"run_ids.{S0}-s2.{s}.txt")},
    }

    def contrast(treats, switch, basis, suite):
        a_vals, b_vals, clusters = {}, {}, {}
        for t in treats:
            budget = {}
            for rid in treat[t]:
                r = runs.get(rid)
                if r is None or rid not in cov or (suite and r["suite"] != suite):
                    continue
                key = (r["suite"], r["task"], r["seed"])
                if key in budget:
                    raise SystemExit(f"two runs of {t} at {key}")
                budget[key] = r["calls"]
                tk = f"{r['suite']}/{r['task']}"
                a_vals.setdefault(tk, []).append(cov[rid])
                clusters[tk] = f"{r['suite']}/{r['cluster']}"
            for rid, r in runs.items():
                if r["arm"] != switch or (suite and r["suite"] != suite):
                    continue
                tk = f"{r['suite']}/{r['task']}"
                if basis == "unmatched":
                    if rid in cov:
                        b_vals.setdefault(tk, []).append(cov[rid])
                else:
                    c = budget.get((r["suite"], r["task"], r["seed"]))
                    v = None if c is None else matched_value(rid, c)
                    if v is not None:
                        b_vals.setdefault(tk, []).append(v)
        keys = sorted(set(a_vals) & set(b_vals))
        a = {k: sum(a_vals[k]) / len(a_vals[k]) for k in keys}
        b = {k: sum(b_vals[k]) / len(b_vals[k]) for k in keys}
        e = paired_difference(
            a, b, clusters={k: clusters[k] for k in keys}, n_boot=1000, n_perm=10000, seed=0
        )
        return {"delta": e.point, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi, "n": e.n}

    return contrast


def bearing_reader(data: Path):
    s0_store = cr.Store(data / "artifacts/seed_identity_20260923/controls_store")
    s0_ids = set(cr._ids(data / "artifacts/killswitch/run_ids.treatment.txt"))
    seeds = {}
    for tag in ("s1", "s2"):
        t = cr.Store(data / f"artifacts/structured_baselines_clean_20260922/scores_parquet_{tag}")
        c = cr.Store(data / f"artifacts/seed_identity_20260923/ks_store_{tag}")
        model = f"{S0}-{tag}"
        ids = {
            rid
            for rid, r in t.runs.items()
            if r["arm"] == "inquirer_trained" and t.model.get(rid) == {model}
        }
        seeds[tag] = (t, ids, c, model)

    def only(part, suite):
        a, b, cl = part
        if suite is None:
            return part
        keep = lambda d: {k: v for k, v in d.items() if k.startswith(suite + "/")}  # noqa: E731
        return keep(a), keep(b), keep(cl)

    def contrast(treats, switch, basis, suite):
        parts = []
        for t in treats:
            if t == "s0":
                parts.append(
                    only(cr.per_seed_values(s0_store, s0_ids, s0_store, S0, switch, basis), suite)
                )
            else:
                ts, ids, cs, model = seeds[t]
                parts.append(only(cr.per_seed_values(ts, ids, cs, model, switch, basis), suite))
        r = cr.contrast(parts)
        return {k: r.get(k) for k in ("delta", "ci_lo", "ci_hi", "n")}

    return contrast


def fmt(x):
    return None if x is None else f"{x:+.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    free, bearing = free_reader(a.data_root), bearing_reader(a.data_root)
    res: dict = {"lock": {}, "cells": {}}
    bad = []
    for (sw, scope), (pm, pu) in P.items():
        fn = bearing if sw in BEARING else free
        suite = None if scope == "pooled" else scope
        for basis, want in (("matched", pm), ("unmatched", pu)):
            got = fn(["s0"], sw, basis, suite)
            printed = (fmt(got["delta"]), fmt(got["ci_lo"]), fmt(got["ci_hi"]))
            ok = all(w is None or w == g for w, g in zip(want, printed))
            res["lock"][f"{sw}::{scope}::{basis}"] = {
                "printed": want,
                "reproduced": printed,
                "n": got["n"],
                "ok": ok,
            }
            res["cells"].setdefault("s0", {})[f"{sw}::{scope}::{basis}"] = got
            if not ok:
                bad.append((sw, scope, basis, want, printed))
    print(f"LOCK s0: {len(res['lock']) - len(bad)}/{len(res['lock'])} printed cells reproduced")
    if bad:
        for b in bad:
            print("  MISMATCH", b)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        raise SystemExit("LOCK FAILED")
    for sw in FREE + BEARING:
        if sw == "parallel_replay":
            continue  # replays seed 0's questions: no recipe reading exists
        fn = bearing if sw in BEARING else free
        for scope in ("pooled", *SUITES):
            suite = None if scope == "pooled" else scope
            for basis in ("matched", "unmatched"):
                got = fn(["s1", "s2"], sw, basis, suite)
                res["cells"].setdefault("s1s2", {})[f"{sw}::{scope}::{basis}"] = got
                print(
                    f"s1s2 {sw:20s} {scope:10s} {basis:9s} {fmt(got['delta'])} [{fmt(got['ci_lo'])}, {fmt(got['ci_hi'])}] n={got['n']}"
                )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
