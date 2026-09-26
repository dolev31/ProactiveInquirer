"""Compute the symmetric matched-cost `evidence_coverage` cells against a scored store, using
`pinq_train.gate`'s own `_by_key`, `_paired_by_task_map` and `_matched_cost` (which itself calls
`bca_ci`). Never hand-rolls the pairing or the bootstrap.

The run selection is NOT `gate._select_runs` -- that helper filters only on `arm_id`,
`grid_name` and `status = 'ok'`, with no `code_version` or `split` filter, which is exactly the
pooling trap this reproduction was warned about. This script pins `code_version` explicitly and
applies `pi_eval.report.ELIGIBLE`, and prints the distinct `code_version` values its own
selection touched so a pooled read cannot pass as a clean one.

Usage: python3 scripts/baseline_repro/compute_cells.py <store_dir> <scorer_hash> [--seed N]
       [--n-resamples N] [--label L]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

TRAINED_PIN = "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856"
PROMPTED_PIN = "e9af49a4b5ff0a0b27a57f7c4feca06566cbfbec3cc0f810920f5f414862fcfe"
CODE_VERSION = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
GRID = "tier1_trained_qa_base"

ELIGIBLE = (
    "r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE "
    "AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE "
    "AND r.exploratory = FALSE "
    "AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none' "
    "AND r.split = 'test' "
    "AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE"
)


def _select(con, *, arm_id: str, pin: str, pin_code_version: bool) -> list[dict]:
    where = (
        f"r.arm_id = '{arm_id}' AND r.grid_name = '{GRID}' AND r.model_pin_hash = '{pin}' "
        f"AND {ELIGIBLE}"
    )
    if pin_code_version:
        where += f" AND r.code_version = '{CODE_VERSION}'"
    rows = con.execute(
        f"SELECT r.run_id, r.suite_id, r.task_id, r.seed, r.code_version FROM runs r "
        f"WHERE {where} ORDER BY r.suite_id, r.task_id, r.seed, r.run_id"
    ).fetchall()
    cols = ["run_id", "suite_id", "task_id", "seed", "code_version"]
    return [dict(zip(cols, r)) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store_dir")
    ap.add_argument("scorer_hash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--no-code-version-pin",
        action="store_true",
        help="Deliberately reproduce the pooling trap for demonstration only.",
    )
    args = ap.parse_args()

    from pinq_train import gate

    con = gate._con(Path(args.store_dir))
    pin_cv = not args.no_code_version_pin

    ckpt = _select(con, arm_id="inquirer_trained", pin=TRAINED_PIN, pin_code_version=pin_cv)
    base = _select(con, arm_id="inquirer_prompted", pin=PROMPTED_PIN, pin_code_version=pin_cv)

    ck_cvs = sorted({r["code_version"] for r in ckpt})
    ba_cvs = sorted({r["code_version"] for r in base})
    print(f"=== {args.label or args.store_dir} ===")
    print(f"scorer_hash: {args.scorer_hash}")
    print(f"checkpoint (trained) runs: {len(ckpt)}, distinct code_versions: {ck_cvs}")
    print(f"baseline (prompted) runs: {len(base)}, distinct code_versions: {ba_cvs}")

    ck_keys = gate._by_key(ckpt)
    ba_keys = gate._by_key(base)

    result = gate._matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=args.scorer_hash,
        seed=args.seed,
        n_resamples=args.n_resamples,
    )

    out = {"label": args.label, "seed": args.seed, "n_resamples": args.n_resamples, "by_suite": {}}
    for suite, cell in result["by_suite"].items():
        sym = cell["symmetric"]
        print(
            f"  suite={suite:12s} symmetric n={sym['n_tasks']:4d} "
            f"delta={sym['delta']:+.6f} ci=[{sym['ci_lo']:+.6f}, {sym['ci_hi']:+.6f}]"
        )
        out["by_suite"][suite] = {
            "n_tasks": sym["n_tasks"],
            "delta": sym["delta"],
            "ci_lo": sym["ci_lo"],
            "ci_hi": sym["ci_hi"],
            "asym_delta": cell["delta"],
            "asym_n_tasks": cell["n_tasks"],
        }

    out_path = Path(args.store_dir) / f"cells.{args.label or 'result'}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
