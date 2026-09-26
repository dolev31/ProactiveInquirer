"""How much evidence the 1,200-character rendering cut hides, per QA suite.

Every prompt that shows evidence (resolve, draft, answer, and the questioner's own prompt, in every
arm and in the training export) renders each unit through pinq_expt.components.render_evidence,
which keeps its first UNIT_CHARS = 1200 characters. This counts the paragraphs longer than that in
each suite's corpus, as the Table 3 runs used it (corpus_dir from their runs.parquet), over all tasks
and over the held-out tasks those runs read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from pinq_expt.components import UNIT_CHARS

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "artifacts/completed_cohort_20260922/scores_parquet/runs.parquet"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    con = duckdb.connect()
    out: dict = {"unit_chars": UNIT_CHARS, "suites": {}}
    for suite, corpus_dir in sorted(
        con.execute(f"SELECT DISTINCT suite_id, corpus_dir FROM read_parquet('{STORE}')").fetchall()
    ):
        held = {
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT task_id FROM read_parquet('{STORE}') WHERE suite_id = ?", [suite]
            ).fetchall()
        }
        path = REPO / "data/corpora" / suite / corpus_dir / "tasks.jsonl"
        n_all = long_all = n_held = long_held = tasks_held = 0
        for line in path.read_text().splitlines():
            t = json.loads(line)
            lens = [len(p["text"] if isinstance(p, dict) else p) for p in t["paragraphs"]]
            n_all += len(lens)
            long_all += sum(x > UNIT_CHARS for x in lens)
            if t["id"] in held:
                tasks_held += 1
                n_held += len(lens)
                long_held += sum(x > UNIT_CHARS for x in lens)
        out["suites"][suite] = {
            "corpus_dir": corpus_dir,
            "paragraphs": n_all,
            "longer": long_all,
            "share": long_all / n_all,
            "held_out_tasks": tasks_held,
            "held_out_paragraphs": n_held,
            "held_out_longer": long_held,
            "held_out_share": long_held / n_held,
        }
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for s, v in out["suites"].items():
        print(f"{s}: {v['share']:.4f} of {v['paragraphs']}; held-out {v['held_out_share']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
