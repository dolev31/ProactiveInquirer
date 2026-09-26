"""Self-check: row-order invariance of the two bootstrap pipelines this reproduction depends on.

`pinq_train.gate.bca_ci` sorts its input before computing `theta_hat` and before drawing
resample indices; `pi_eval.stats.inference.paired_difference` does `sorted(set(a) & set(b))`
before building its resample units. Both claims are read from source, but a claim about code
is not a measurement -- this script shuffles the row order actually fed into each pipeline and
asserts the emitted point estimate and CI bounds are bit-identical to an unshuffled run.

Usage: python3 scripts/baseline_repro/selfcheck_shuffle.py <store_dir> <scorer_hash>
"""

from __future__ import annotations

import random
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


def _select(con, *, arm_id: str, pin: str, shuffle_seed: int | None) -> list[dict]:
    where = (
        f"r.arm_id = '{arm_id}' AND r.grid_name = '{GRID}' AND r.model_pin_hash = '{pin}' "
        f"AND r.code_version = '{CODE_VERSION}' AND {ELIGIBLE}"
    )
    rows = con.execute(
        f"SELECT r.run_id, r.suite_id, r.task_id, r.seed, r.code_version FROM runs r WHERE {where}"
    ).fetchall()
    cols = ["run_id", "suite_id", "task_id", "seed", "code_version"]
    out = [dict(zip(cols, r)) for r in rows]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(out)
    return out


def _coverage_cells(con, scorer_hash: str, *, shuffle_seed: int | None) -> dict:
    from pinq_train import gate

    ckpt = _select(con, arm_id="inquirer_trained", pin=TRAINED_PIN, shuffle_seed=shuffle_seed)
    base = _select(con, arm_id="inquirer_prompted", pin=PROMPTED_PIN, shuffle_seed=shuffle_seed)
    ck_keys = gate._by_key(ckpt)
    ba_keys = gate._by_key(base)
    result = gate._matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=scorer_hash,
        seed=0,
        n_resamples=10000,
    )
    return {
        suite: {
            "n_tasks": cell["symmetric"]["n_tasks"],
            "delta": cell["symmetric"]["delta"],
            "ci_lo": cell["symmetric"]["ci_lo"],
            "ci_hi": cell["symmetric"]["ci_hi"],
        }
        for suite, cell in result["by_suite"].items()
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    store_dir, scorer_hash = Path(sys.argv[1]), sys.argv[2]

    from pinq_train import gate

    con = gate._con(store_dir)
    unshuffled = _coverage_cells(con, scorer_hash, shuffle_seed=None)
    shuffled = _coverage_cells(con, scorer_hash, shuffle_seed=777)

    ok = True
    for suite in unshuffled:
        same = unshuffled[suite] == shuffled[suite]
        ok = ok and same
        print(f"coverage {suite:12s} shuffle-invariant: {same}  {unshuffled[suite]}")

    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
