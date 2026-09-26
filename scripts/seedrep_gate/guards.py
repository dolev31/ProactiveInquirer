"""Run the eligibility and contamination guards over the seed-replicate population, and say
which layer of the contamination guard could actually FIRE on it.

`pi_eval.report.train_split_violations` has two predicates. The first recomputes `split_of`
and demands `test` on both the recomputed and the stamped side; it fires on anything. The
second resolves the row's `train_id_set_hash` against the exporter's id file and fires if the
task is in it; its own docstring says a NULL hash is not a refusal. Every one of the 2,400
seed-replicate runs carries `train_id_set_hash: null`, because the pin `…-s1`/`…-s2` is not
the bare registry name the stamper keys on -- so the SECOND predicate is structurally inert
here and a zero from it is not evidence. This script reports the two separately and counts the
null stamps, rather than printing one clean zero that hides the inert half.

A NEGATIVE CONTROL is run as well: the same call over dev-split trained runs, which must
produce violations. A guard never observed firing is not a guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pi_eval.report import ELIGIBLE, assert_no_gold_exposed, open_agg, train_split_violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--ids", action="append", required=True)
    ap.add_argument("--ids-dir", default="data/rl")
    ap.add_argument(
        "--shared-parquet",
        default="scores/parquet",
        help="read-only, for the negative control's dev-split run ids",
    )
    ap.add_argument(
        "--control-scorer-hash",
        default=None,
        help="the shared store holds many; open_agg refuses without one. It is "
        "not used by the guard, only to open the context.",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids: list[str] = []
    for p in a.ids:
        ids.extend(x.strip() for x in open(p) if x.strip() and not x.startswith("#"))
    ids = sorted(set(ids))

    agg = open_agg(a.parquet, scorer_hash=a.scorer_hash)
    con = duckdb.connect()
    pq = Path(a.parquet).resolve()
    stamps = con.execute(
        f"SELECT train_id_set_hash, COUNT(*) FROM read_parquet('{pq}/runs.parquet') "
        f"WHERE run_id IN ({','.join(repr(i) for i in ids)}) GROUP BY 1"
    ).fetchall()
    n_eligible = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{pq}/runs.parquet') r "
        f"WHERE r.run_id IN ({','.join(repr(i) for i in ids)}) AND {ELIGIBLE}"
    ).fetchone()[0]

    viol = train_split_violations(agg, ids, ids_dir=a.ids_dir)
    gold = assert_no_gold_exposed(agg, ids)

    out = {
        "n_ids": len(ids),
        "eligibility_predicate": ELIGIBLE,
        "n_passing_ELIGIBLE": int(n_eligible),
        "train_id_set_hash_census": {str(s): int(c) for s, c in stamps},
        "train_split_violations": len(viol),
        "train_split_violations_sample": viol[:5],
        "gold_exposed_violations": len(gold),
        "second_predicate_live": any(s for s, _ in stamps),
        "note": (
            "second_predicate_live=False means the train_id_set membership half of the "
            "contamination guard COULD NOT FIRE on this population; only the recomputed-split "
            "half is evidence here."
        ),
    }

    # --- negative control: the guard must fire on dev-split trained runs -------------------
    try:
        shared = Path(a.shared_parquet).resolve()
        dev_ids = [
            r[0]
            for r in con.execute(
                f"SELECT run_id FROM read_parquet('{shared}/runs.parquet') "
                "WHERE arm_id = 'inquirer_trained' AND split = 'dev' AND status = 'ok' "
                "ORDER BY run_id LIMIT 40"
            ).fetchall()
        ]
        ctrl_agg = open_agg(a.shared_parquet, scorer_hash=a.control_scorer_hash)
        ctrl = train_split_violations(ctrl_agg, dev_ids, ids_dir=a.ids_dir)
        out["negative_control"] = {
            "population": "40 dev-split inquirer_trained runs from the SHARED store (read-only)",
            "n_ids": len(dev_ids),
            "n_violations": len(ctrl),
            "fires": len(ctrl) > 0,
            "reasons": sorted({str(r.get("reason")) for r in ctrl}),
        }
    except Exception as exc:  # the control is reported, never silently skipped
        out["negative_control"] = {"error": f"{type(exc).__name__}: {exc}"}

    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
