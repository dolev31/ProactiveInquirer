"""The eight control contrasts against each training seed of the selected recipe.

The published controls (artifacts/killswitch_mechanism_20260918/RESULT.md) were read against seed 0,
the registered adapter that kept about a tenth of its final stage. This reads them against seeds 1
and 2 with the SAME rule and the SAME estimator:

  matched  the control's coverage at the largest prefix of its own trajectory whose cumulative
           retrieval_calls is within the paired trained run's terminal retrieval_calls, read off the
           frontier_spend#k / frontier_q#k ladder the scorer emits (scripts/killswitch_mechanism/
           mk_matched.py, verbatim in rule); NaN rungs step down.
  unmatched  each arm's own terminal evidence_coverage.
  estimator  per task, the mean over that arm's eligible runs (rollout seeds averaged); pooled over
             musique and strategyqa with keys suite/task; pi_eval.stats.inference.paired_difference
             with clusters on template_id (falling back to task_id), n_boot=1000 (prereg
             CI_RESAMPLES), n_perm=10000, seed=0. delta > 0 means the control stayed BELOW the
             trained arm. Eligibility is pi_eval.report.ELIGIBLE, plus the 52 zero-ask random_q ids
             of artifacts/nulls_and_determinism_20260918, exactly as the published table.

The treatment is selected by RUN-ID LIST, never by arm_id, because seeds 0, 1 and 2 share arm_id
inquirer_trained in this store. The budget key is (suite, task, rollout seed) PER TRAINING SEED, so two
training seeds never overwrite each other's budgets; the pooled recipe averages both seeds' trained
values and both seeds' matched control values within the task.

READ THIS BEFORE QUOTING A SEED-1/2 CELL. calls.parquet shows every model-bearing control
(inquirer_depth1, inquirer_noevidence, self_inquire) runs seed 0's weights with part of the protocol
removed, so each published contrast is a within-weights ablation of seed 0. The recipe minus those
arms mixes a weights difference into the protocol question and is NOT a control result. The recipe's
own ablations need the controls rolled out on seeds 1 and 2.

LOCK: seed 0 must reproduce the published pooled points (random_q +0.5151, inquirer_depth1 +0.0986,
inquirer_noevidence +0.0662 matched / +0.0220 unmatched, self_inquire +0.0143 matched / -0.0121
unmatched) to 5e-4 before any recipe cell is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "artifacts/seed_identity_20260923/controls_store"
SUITES = ("musique", "strategyqa")
SWITCHES = (
    "random_q",
    "inquirer_depth1",
    "inquirer_noevidence",
    "self_inquire",
    "parallel_replay",
    "checklist",
    "verbosity",
    "compute_matched",
)
PUBLISHED_S0 = {
    ("random_q", "matched"): 0.5151,
    ("inquirer_depth1", "matched"): 0.0986,
    ("inquirer_noevidence", "matched"): 0.0662,
    ("inquirer_noevidence", "unmatched"): 0.0220,
    ("self_inquire", "matched"): 0.0143,
    ("self_inquire", "unmatched"): -0.0121,
}


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW runs AS SELECT * FROM read_parquet('{(STORE / 'runs.parquet').as_posix()}')"
    )
    con.execute(
        f"CREATE VIEW scores AS SELECT * FROM read_parquet('{(STORE / 'scores.parquet').as_posix()}')"
    )
    hashes = [r[0] for r in con.execute("SELECT DISTINCT scorer_hash FROM scores").fetchall()]
    if len(hashes) != 1:
        raise SystemExit(f"REFUSING: {len(hashes)} scorer hashes in the store")
    excl = set(
        _ids(
            REPO / "artifacts/nulls_and_determinism_20260918/EXCLUDE_random_q_zero_ask.eligible.txt"
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
    cov: dict[str, float] = {}
    spend: dict[str, dict[int, float]] = {}
    qq: dict[str, dict[int, float]] = {}
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

    def matched_value(rid: str, budget: float) -> float | None:
        sp, q = spend.get(rid), qq.get(rid)
        if not sp or not q:
            return None
        ks = [k for k in sorted(sp) if sp[k] <= budget + 1e-9]
        kstar = max(ks) if ks else 0
        while kstar not in q and kstar > 0:
            kstar -= 1
        return q.get(kstar)

    treat_lists = {
        "s0": set(_ids(REPO / "artifacts/killswitch/run_ids.treatment.txt")),
        "s1": {
            i
            for s in SUITES
            for i in _ids(
                REPO
                / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s1.{s}.txt"
            )
        },
        "s2": {
            i
            for s in SUITES
            for i in _ids(
                REPO
                / f"artifacts/seedrep_gate_20260919/run_ids/run_ids.qwen3-8b-dpo-stacked-notdone-both-s2.{s}.txt"
            )
        },
    }
    for k, ids in treat_lists.items():
        present = [i for i in ids if i in runs]
        if not present:
            raise SystemExit(f"REFUSING: treatment {k} has no eligible runs in the store")
        arms = {runs[i]["arm"] for i in present}
        if arms != {"inquirer_trained"}:
            raise SystemExit(f"REFUSING: treatment {k} carries arms {arms}")

    def contrast(treats: list[str], switch: str, basis: str) -> dict:
        a_vals: dict[str, list[float]] = {}
        b_vals: dict[str, list[float]] = {}
        clusters: dict[str, str] = {}
        for t in treats:
            budget = {}
            for rid in treat_lists[t]:
                r = runs.get(rid)
                if r is None or rid not in cov:
                    continue
                key = (r["suite"], r["task"], r["seed"])
                if key in budget:
                    raise SystemExit(f"REFUSING: two runs of treatment {t} at {key}")
                budget[key] = r["calls"]
                tk = f"{r['suite']}/{r['task']}"
                a_vals.setdefault(tk, []).append(cov[rid])
                clusters[tk] = f"{r['suite']}/{r['cluster']}"
            for rid, r in runs.items():
                if r["arm"] != switch:
                    continue
                tk = f"{r['suite']}/{r['task']}"
                if basis == "unmatched":
                    if rid in cov:
                        b_vals.setdefault(tk, []).append(cov[rid])
                else:
                    c = budget.get((r["suite"], r["task"], r["seed"]))
                    if c is None:
                        continue
                    v = matched_value(rid, c)
                    if v is not None:
                        b_vals.setdefault(tk, []).append(v)
        keys = sorted(set(a_vals) & set(b_vals))
        if not keys:
            return {"n": 0}
        a = {k: sum(a_vals[k]) / len(a_vals[k]) for k in keys}
        b = {k: sum(b_vals[k]) / len(b_vals[k]) for k in keys}
        e = paired_difference(
            a, b, clusters={k: clusters[k] for k in keys}, n_boot=1000, n_perm=10000, seed=0
        )
        return {
            "delta": e.point,
            "ci_lo": e.ci_lo,
            "ci_hi": e.ci_hi,
            "n": e.n,
            "trained": sum(a.values()) / len(a),
            "switch": sum(b.values()) / len(b),
        }

    out: dict = {
        "store": str(STORE.relative_to(REPO)),
        "scorer_hash": hashes[0],
        "cells": {},
        "lock": {},
    }
    for label, treats in (("s0", ["s0"]), ("s1", ["s1"]), ("s2", ["s2"]), ("s1s2", ["s1", "s2"])):
        for sw in SWITCHES:
            for basis in ("matched", "unmatched"):
                out["cells"].setdefault(label, {})[f"{sw}::{basis}"] = contrast(treats, sw, basis)
    for (sw, basis), want in PUBLISHED_S0.items():
        got = out["cells"]["s0"][f"{sw}::{basis}"].get("delta")
        ok = got is not None and abs(got - want) <= 5e-4
        out["lock"][f"{sw}::{basis}"] = {"published": want, "reproduced": got, "ok": ok}
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    bad = [k for k, v in out["lock"].items() if not v["ok"]]
    print(f"wrote {args.out}; lock failures: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
