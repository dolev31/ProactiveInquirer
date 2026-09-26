"""Build an isolated, read-only-source scoring store for the baseline-completion reproduction.

Selects the trained/prompted arms of `tier1_trained_qa_base` at the two published model pins,
pinned to a single `code_version`, filtered by `pi_eval.report.ELIGIBLE`, from the SHARED store
(`scores/parquet/`, read-only). Copies the selected rows for `runs/turns/evidence/env_calls/
ledger/native.parquet` into a fresh directory this script owns. Never writes to the shared store.

Usage: python3 scripts/baseline_repro/build_store.py <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_STORE = REPO_ROOT / "scores" / "parquet"

TRAINED_PIN = "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856"
PROMPTED_PIN = "e9af49a4b5ff0a0b27a57f7c4feca06566cbfbec3cc0f810920f5f414862fcfe"
CODE_VERSION = "3ae099d0e9f08f6654d5e259ffc0850232f8e70a"
GRID = "tier1_trained_qa_base"

# `pi_eval.report.ELIGIBLE`, restated here rather than imported: this script has no firewall
# reason to import pi_eval (it never touches gold), and restating keeps the selection readable
# in one place. Confirmed byte-identical against `src/pi_eval/report.py`'s own `ELIGIBLE`
# constant by direct comparison (see RESULT.md); no separate verifier script exists.
ELIGIBLE = (
    "r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE "
    "AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE "
    "AND r.exploratory = FALSE "
    "AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none' "
    "AND r.split = 'test' "
    "AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE"
)


def _where(arm_id: str, pin: str) -> str:
    return (
        f"r.arm_id = '{arm_id}' AND r.grid_name = '{GRID}' AND r.model_pin_hash = '{pin}' "
        f"AND r.code_version = '{CODE_VERSION}' AND {ELIGIBLE}"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    for name in ("runs", "turns", "evidence", "env_calls", "ledger", "native", "calls"):
        p = SHARED_STORE / f"{name}.parquet"
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")

    where_trained = _where("inquirer_trained", TRAINED_PIN)
    where_prompted = _where("inquirer_prompted", PROMPTED_PIN)

    con.execute(
        "CREATE TEMP TABLE selected_runs AS "
        f"SELECT r.* FROM runs r WHERE ({where_trained}) OR ({where_prompted})"
    )

    n_selected = con.execute("SELECT count(*) FROM selected_runs").fetchone()[0]
    n_distinct_cv = con.execute(
        "SELECT count(distinct code_version) FROM selected_runs"
    ).fetchone()[0]
    cvs = [r[0] for r in con.execute("SELECT DISTINCT code_version FROM selected_runs").fetchall()]
    print(f"selected runs: {n_selected}")
    print(f"distinct code_versions touched by this selection: {cvs} (n={n_distinct_cv})")
    if n_distinct_cv != 1:
        raise SystemExit(
            f"REFUSING: selection touched {n_distinct_cv} code_versions, not 1. "
            "This is exactly the pooling trap -- do not write a store off this selection."
        )

    per_suite = con.execute(
        "SELECT arm_id, suite_id, count(distinct task_id) AS n_tasks, count(*) AS n_runs "
        "FROM selected_runs GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    for row in per_suite:
        print(f"  arm={row[0]:18s} suite={row[1]:12s} n_tasks={row[2]:4d} n_runs={row[3]:5d}")

    con.execute(f"COPY selected_runs TO '{(out_dir / 'runs.parquet').as_posix()}' (FORMAT parquet)")
    for name in ("turns", "evidence", "env_calls", "ledger", "native", "calls"):
        dest = (out_dir / f"{name}.parquet").as_posix()
        con.execute(
            f"COPY (SELECT t.* FROM {name} t WHERE t.run_id IN "
            "(SELECT run_id FROM selected_runs)) "
            f"TO '{dest}' (FORMAT parquet)"
        )
        n = con.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[0]
        print(f"wrote {name}.parquet: {n} rows")

    run_ids = [r[0] for r in con.execute("SELECT run_id FROM selected_runs ORDER BY 1").fetchall()]
    (out_dir / "run_ids.txt").write_text("\n".join(run_ids) + "\n")
    print(f"wrote run_ids.txt: {len(run_ids)} ids")


if __name__ == "__main__":
    main()
