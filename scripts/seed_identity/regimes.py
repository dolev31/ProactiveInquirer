"""Two questions, one table: equal realized spend (Q1) and each arm stopping on its own under a
shared ceiling of eight (Q2), for the recipe (seeds 1 and 2) against the same weights prompted and
against the 120B prompted model, required-evidence coverage, three held-out suites.

Q1 cells are read, not recomputed, from the records that already carry 10k intervals and 50k
verdicts: table1_by_seed.json (against the same weights) and teacher_by_seed.json (against 120B),
both under Table 1's symmetric seed-matched rule.

Q2 cells come from cap8_by_seed.py and teacher_ownstop_by_seed.py, whose records hold one 10k
interval at seed 0 and no verdict. This script recomputes each Q2 cell through THOSE scripts' own
per-task readers, LOCKS the recomputed 10k point and interval to the stored record to 1e-12, and
only then adds the 50k intervals at seeds 101/202/303. A cell is DECIDED when all three 50k
intervals exclude zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from pi_eval.stats.inference import paired_difference

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "seed_identity"))

import cap8_by_seed as c8  # noqa: E402
import teacher_ownstop_by_seed as tos  # noqa: E402

SI = REPO / "artifacts/seed_identity_20260923"
SUITES = ("musique", "strategyqa", "wiki2")
VERDICT_SEEDS = (101, 202, 303)


def _q1(rows: list[dict]) -> dict:
    ten = [r for r in rows if r["n_boot"] == 10000 and r["seed"] == 0]
    fifty = [r for r in rows if r["n_boot"] == 50000]
    assert len(ten) == 1 and sorted(r["seed"] for r in fifty) == list(VERDICT_SEEDS), rows
    t = ten[0]
    return {
        "delta": t["delta"],
        "ci_lo": t["ci_lo"],
        "ci_hi": t["ci_hi"],
        "n": t["n"],
        "b50": [
            [r["seed"], r["ci_lo"], r["ci_hi"]] for r in sorted(fifty, key=lambda r: r["seed"])
        ],
    }


def _q2(a: dict, b: dict, stored: dict) -> dict:
    shared = sorted(set(a) & set(b))
    aa, bb = {t: a[t] for t in shared}, {t: b[t] for t in shared}
    e = paired_difference(aa, bb, n_boot=10000, seed=0)
    for f, v in (("delta", e.point), ("ci_lo", e.ci_lo), ("ci_hi", e.ci_hi)):
        if abs(v - stored[f]) > 1e-12:
            raise SystemExit(f"LOCK FAILED: {f} {v} vs stored {stored[f]}")
    b50 = []
    for s in VERDICT_SEEDS:
        f = paired_difference(aa, bb, n_boot=50000, seed=s)
        b50.append([s, f.ci_lo, f.ci_hi])
    return {"delta": e.point, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi, "n": e.n, "b50": b50}


def decided(cell: dict) -> bool:
    return all(lo > 0 for _, lo, _ in cell["b50"]) or all(hi < 0 for _, _, hi in cell["b50"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    t1 = json.loads((SI / "table1_by_seed.json").read_text())
    tb = json.loads((SI / "teacher_by_seed.json").read_text())
    cap8 = json.loads((SI / "cap8_by_seed.json").read_text())
    tow = json.loads((SI / "teacher_ownstop_by_seed.json").read_text())
    for name, rec in (("table1_by_seed", t1), ("cap8_by_seed", cap8), ("teacher_by_seed", tb)):
        bad = [k for k, v in rec["lock"].items() if not v["ok"]]
        assert not bad, f"{name} seed-0 lock not held: {bad}"
    assert all(v["ok"] for v in tow["lock"].values()), "teacher_ownstop_by_seed lock not held"

    con = duckdb.connect()
    runs_pq = (tos.STORE / "runs.parquet").as_posix()
    suite_of = dict(
        con.execute(f"SELECT run_id, suite_id FROM read_parquet('{runs_pq}')").fetchall()
    )
    teacher_all = tos._ids(REPO / "artifacts/n6/run_ids.inquirer_prompted.txt")
    out: dict = {"cells": {}, "asks": {}, "sources": {}}
    rid = SI.parent / "seedrep_gate_20260919/run_ids"
    for suite in SUITES:
        s1 = c8._ids(rid / f"run_ids.qwen3-8b-dpo-stacked-notdone-both-s1.{suite}.txt")
        s2 = c8._ids(rid / f"run_ids.qwen3-8b-dpo-stacked-notdone-both-s2.{suite}.txt")
        pr_ids = c8._ids(c8.COHORT / "cohort" / f"run_ids.prompted.{suite}.txt")
        rec = c8.per_task(con, c8.SEEDREP / "scores_parquet", s1 + s2)
        pr = c8.per_task(con, c8.COHORT / "scores_parquet", pr_ids)
        teacher = [r for r in teacher_all if suite_of.get(r) == suite]
        rec_t = tos.per_task(con, s1 + s2, "evidence_coverage")
        tea = tos.per_task(con, teacher, "evidence_coverage")
        cells = {
            ("base", "q1"): _q1(t1["cells"]["s1s2"][f"evidence_coverage::{suite}"]),
            ("teacher", "q1"): _q1(tb["cells"]["s1s2"][suite]),
            ("base", "q2"): _q2(rec, pr, cap8["cells"]["s1s2"][suite]),
            ("teacher", "q2"): _q2(rec_t, tea, tow["cells"]["s1s2"]["evidence_coverage"][suite]),
        }
        for (comp, q), c in cells.items():
            c["decided"] = decided(c)
            out["cells"].setdefault(comp, {}).setdefault(q, {})[suite] = c
        out["asks"][suite] = {
            "recipe": cap8["asks"]["s1s2"][suite]["mean_n_asks"],
            "base": cap8["asks"]["prompted"][suite]["mean_n_asks"],
            "teacher": tb["arms"]["teacher"][suite]["mean_n_asks"],
        }
    out["sources"] = {
        "q1_base": "artifacts/seed_identity_20260923/table1_by_seed.json cells.s1s2",
        "q1_teacher": "artifacts/seed_identity_20260923/teacher_by_seed.json cells.s1s2",
        "q2_base": "scripts/seed_identity/cap8_by_seed.py per_task, locked to cap8_by_seed.json cells.s1s2",
        "q2_teacher": "scripts/seed_identity/teacher_ownstop_by_seed.py per_task, locked to "
        "teacher_ownstop_by_seed.json cells.s1s2",
    }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for comp in ("base", "teacher"):
        for q in ("q1", "q2"):
            row = out["cells"][comp][q]
            print(
                comp,
                q,
                "  ".join(
                    f"{s} {row[s]['delta']:+.4f} [{row[s]['ci_lo']:+.4f},{row[s]['ci_hi']:+.4f}]"
                    f"{' D' if row[s]['decided'] else ''}"
                    for s in SUITES
                ),
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
