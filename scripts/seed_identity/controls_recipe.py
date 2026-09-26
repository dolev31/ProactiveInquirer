"""The model-bearing controls, each an ablation of ONE seed's weights, read against that seed.

`inquirer_depth1`, `inquirer_noevidence` and `self_inquire` run the inquirer's own weights with part of
the protocol removed, so a control is only meaningful against the trained runs of the SAME weights.
The published controls ablate seed 0 (the resumed run). This reads each seed's ablations against that
seed's own treatment runs, then pools seeds 1 and 2 within the task.

Rule and estimator are controls_by_seed.py's, i.e. the published table's: the control's coverage at the
largest prefix of its own trajectory within the paired treatment run's retrieval_calls (frontier ladder,
NaN rungs step down), per-task means, pooled over musique and strategyqa, paired_difference with
clusters on template_id, n_boot=1000, n_perm=10000, seed=0; plus the unmatched basis. ELIGIBLE applies.

Each seed is a (treatment store, treatment run ids, control store, inquirer model) tuple. The control
runs are selected by arm_id AND by the inquirer model read from calls.parquet (actor='inquirer'), so a
control store holding several weights can never pool them.

LOCK: seed 0 (treatment artifacts/killswitch/run_ids.treatment.txt, controls in controls_store at seed
0's weights) must reproduce the published points to 5e-4 before any seed 1/2 cell is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
SUITES = ("musique", "strategyqa")
SWITCHES = ("inquirer_depth1", "inquirer_noevidence", "self_inquire")
PUBLISHED_S0 = {
    ("inquirer_depth1", "matched"): 0.0986,
    ("inquirer_noevidence", "matched"): 0.0662,
    ("inquirer_noevidence", "unmatched"): 0.0220,
    ("self_inquire", "matched"): 0.0143,
    ("self_inquire", "unmatched"): -0.0121,
}
S0_MODEL = "qwen3-8b-dpo-stacked-notdone-both"


def _ids(p: Path) -> list[str]:
    return [
        ln.split()[0] for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


class Store:
    """runs, coverage and frontier ladders of one scored store, ELIGIBLE rows only."""

    def __init__(self, path: Path) -> None:
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW runs AS SELECT * FROM read_parquet('{(path / 'runs.parquet').as_posix()}')"
        )
        con.execute(
            f"CREATE VIEW scores AS SELECT * FROM read_parquet('{(path / 'scores.parquet').as_posix()}')"
        )
        con.execute(
            f"CREATE VIEW calls AS SELECT * FROM read_parquet('{(path / 'calls.parquet').as_posix()}')"
        )
        self.hashes = sorted(
            r[0] for r in con.execute("SELECT DISTINCT scorer_hash FROM scores").fetchall()
        )
        self.runs = {
            r[0]: {
                "arm": r[1],
                "suite": r[2],
                "task": r[3],
                "seed": int(r[4]),
                "calls": float(r[5]),
                "cluster": r[6],
                "code": r[7],
            }
            for r in con.execute(
                "SELECT r.run_id, r.arm_id, r.suite_id, r.task_id, r.seed, r.retrieval_calls, "
                "COALESCE(NULLIF(r.template_id, ''), r.task_id), r.code_version "
                f"FROM runs r WHERE {ELIGIBLE} AND r.suite_id IN ('musique','strategyqa')"
            ).fetchall()
        }
        self.model: dict[str, set[str]] = {}
        for rid, m in con.execute(
            "SELECT DISTINCT run_id, model FROM calls WHERE actor = 'inquirer'"
        ).fetchall():
            self.model.setdefault(rid, set()).add(m)
        self.cov: dict[str, float] = {}
        self.spend: dict[str, dict[int, float]] = {}
        self.qq: dict[str, dict[int, float]] = {}
        for rid, name, v in con.execute(
            "SELECT run_id, metric_name, value FROM scores WHERE metric_name = 'evidence_coverage' "
            "OR metric_name LIKE 'frontier_spend#%' OR metric_name LIKE 'frontier_q#%'"
        ).fetchall():
            if rid not in self.runs or v is None:
                continue
            if name == "evidence_coverage":
                self.cov[rid] = float(v)
            else:
                kind, k = name.split("#")
                (self.spend if kind == "frontier_spend" else self.qq).setdefault(rid, {})[
                    int(k)
                ] = float(v)

    def matched(self, rid: str, budget: float) -> float | None:
        sp, q = self.spend.get(rid), self.qq.get(rid)
        if not sp or not q:
            return None
        ks = [k for k in sorted(sp) if sp[k] <= budget + 1e-9]
        kstar = max(ks) if ks else 0
        while kstar not in q and kstar > 0:
            kstar -= 1
        return q.get(kstar)


def per_seed_values(
    treat: Store, t_ids: set[str], ctrl: Store, model: str, switch: str, basis: str
):
    """(a_vals, b_vals, clusters) per task for one seed: its treatment vs its own ablation."""
    a_vals: dict[str, list[float]] = {}
    b_vals: dict[str, list[float]] = {}
    clusters: dict[str, str] = {}
    budget: dict[tuple, float] = {}
    for rid in t_ids:
        r = treat.runs.get(rid)
        if r is None or rid not in treat.cov:
            continue
        if treat.model.get(rid) != {model}:
            raise SystemExit(
                f"REFUSING: treatment run {rid} ran inquirer {treat.model.get(rid)}, not {model}"
            )
        key = (r["suite"], r["task"], r["seed"])
        if key in budget:
            raise SystemExit(f"REFUSING: two treatment runs at {key}")
        budget[key] = r["calls"]
        tk = f"{r['suite']}/{r['task']}"
        a_vals.setdefault(tk, []).append(treat.cov[rid])
        clusters[tk] = f"{r['suite']}/{r['cluster']}"
    for rid, r in ctrl.runs.items():
        if r["arm"] != switch or ctrl.model.get(rid) != {model}:
            continue
        tk = f"{r['suite']}/{r['task']}"
        if basis == "unmatched":
            if rid in ctrl.cov:
                b_vals.setdefault(tk, []).append(ctrl.cov[rid])
            continue
        c = budget.get((r["suite"], r["task"], r["seed"]))
        if c is None:
            continue
        v = ctrl.matched(rid, c)
        if v is not None:
            b_vals.setdefault(tk, []).append(v)
    return a_vals, b_vals, clusters


def contrast(parts) -> dict:
    """parts: list of per-seed (a_vals, b_vals, clusters); pooled within the task."""
    a_all: dict[str, list[float]] = {}
    b_all: dict[str, list[float]] = {}
    cl: dict[str, str] = {}
    for a_vals, b_vals, clusters in parts:
        for tk in set(a_vals) & set(b_vals):
            a_all.setdefault(tk, []).append(sum(a_vals[tk]) / len(a_vals[tk]))
            b_all.setdefault(tk, []).append(sum(b_vals[tk]) / len(b_vals[tk]))
            cl[tk] = clusters[tk]
    keys = sorted(a_all)
    if not keys:
        return {"n": 0}
    a = {k: sum(a_all[k]) / len(a_all[k]) for k in keys}
    b = {k: sum(b_all[k]) / len(b_all[k]) for k in keys}
    e = paired_difference(
        a, b, clusters={k: cl[k] for k in keys}, n_boot=1000, n_perm=10000, seed=0
    )
    return {
        "delta": e.point,
        "ci_lo": e.ci_lo,
        "ci_hi": e.ci_hi,
        "n": e.n,
        "trained": sum(a.values()) / len(a),
        "switch": sum(b.values()) / len(b),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--controls-store-s0",
        type=Path,
        default=REPO / "artifacts/seed_identity_20260923/controls_store",
    )
    ap.add_argument(
        "--seed",
        action="append",
        default=[],
        help="label=treatment_store:control_store:inquirer_model (repeatable; treatment ids = every "
        "eligible inquirer_trained run of that model in the treatment store)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    s0 = Store(args.controls_store_s0)
    s0_ids = set(_ids(REPO / "artifacts/killswitch/run_ids.treatment.txt"))
    out: dict = {"lock": {}, "cells": {}, "stores": {"s0": s0.hashes}}
    for sw in SWITCHES:
        for basis in ("matched", "unmatched"):
            out["cells"].setdefault("s0", {})[f"{sw}::{basis}"] = contrast(
                [per_seed_values(s0, s0_ids, s0, S0_MODEL, sw, basis)]
            )
    bad = []
    for (sw, basis), want in PUBLISHED_S0.items():
        got = out["cells"]["s0"][f"{sw}::{basis}"].get("delta")
        ok = got is not None and abs(got - want) <= 5e-4
        out["lock"][f"{sw}::{basis}"] = {"published": want, "reproduced": got, "ok": ok}
        if not ok:
            bad.append(f"{sw}::{basis}")
    if bad:
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"LOCK FAILED {bad}; no seed cell written")
        return 1

    seeds = []
    for spec in args.seed:
        lab, _, rest = spec.partition("=")
        tstore, cstore, model = rest.split(":", 2)
        t, c = Store(REPO / tstore), Store(REPO / cstore)
        out["stores"][lab] = {"treatment": t.hashes, "controls": c.hashes}
        ids = {
            rid
            for rid, r in t.runs.items()
            if r["arm"] == "inquirer_trained" and t.model.get(rid) == {model}
        }
        seeds.append((lab, t, ids, c, model))
    for sw in SWITCHES:
        for basis in ("matched", "unmatched"):
            parts = []
            for lab, t, ids, c, model in seeds:
                p = per_seed_values(t, ids, c, model, sw, basis)
                parts.append(p)
                out["cells"].setdefault(lab, {})[f"{sw}::{basis}"] = contrast([p])
            if len(parts) > 1:
                out["cells"].setdefault("pooled", {})[f"{sw}::{basis}"] = contrast(parts)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}; lock ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
