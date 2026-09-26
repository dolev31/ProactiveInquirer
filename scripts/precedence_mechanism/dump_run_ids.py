"""Provenance companion to `cli.py`: the exact `run_id` list the analysis population is built
from, one row per (suite, task_id, seed, arm, run_id, n_asks), plus the pinned scorer_hash and
graph_version read straight off the store. Every number in `RESULT.md` traces back to this
file by construction (CLAUDE.md rule 1: "a number without provenance is not a result").

    PYTHONPATH=src python -m scripts.precedence_mechanism.dump_run_ids \\
        --store <abs artifacts/testsplit_qa/scores_parquet> --suite musique \\
        --out artifacts/precedence_mechanism_20260918/run_ids.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from .events import run_arms


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True)
    p.add_argument("--suite", default="musique")
    p.add_argument("--out", default="artifacts/precedence_mechanism_20260918/run_ids.json")
    args = p.parse_args(argv)

    con = duckdb.connect()
    pairs = run_arms(con, args.store, args.suite)
    scorer_hashes = con.execute(
        f"select distinct scorer_hash from '{args.store}/scores.parquet'"
    ).fetchall()
    graph_versions = con.execute(
        f"select distinct graph_version from '{args.store}/matches.parquet'"
    ).fetchall()

    rows = []
    for (task_id, seed), arms in sorted(pairs.items()):
        for arm, ra in sorted(arms.items()):
            rows.append(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "seed": seed,
                    "arm": arm,
                    "run_id": ra.run_id,
                    "n_asks": ra.n_asks,
                }
            )

    out = {
        "store": args.store,
        "scorer_hash": [r[0] for r in scorer_hashes],
        "graph_version": [r[0] for r in graph_versions],
        "n_rows": len(rows),
        "n_paired_task_seed": len(pairs),
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {len(rows)} rows ({len(pairs)} paired task,seed) to {out_path}")
    print(f"scorer_hash: {out['scorer_hash']}")
    print(f"graph_version: {out['graph_version']}")


if __name__ == "__main__":
    main()
