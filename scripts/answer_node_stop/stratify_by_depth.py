"""Does the answer-node coverage deficit concentrate where the answer node is NOT YET NAMEABLE?

A peer raised the confound and it is the right one to raise. The answer-node arm covers the
answer-bearing node less than its comparator on all three suites. If that deficit sits on tasks
whose answer node hides behind a prerequisite edge, then the endpoint is partly measuring "the node
could not be identified yet" rather than "the policy stopped wrongly", and the finding is narrower
and more interesting than "conditioning makes it worse".

Stratum is the answer node's own `gold_depth`: 0 means nameable from the task alone, >= 1 means it
sits behind at least one prerequisite edge. Everything else is reused rather than reimplemented:
`build_answer_nodes_for_suite` for the node identification, `covered_map` for the coverage bar
(match_kind in resolve/use, read from matches.parquet), and `p_covered_delta` for the
task-clustered BCa delta. A reimplementation of any of those would be indistinguishable from a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Package-form imports, not bare ones: `from run import ...` resolves to whichever `run.py` is
# first on sys.path, and both sibling lanes have one. The lane's own modules import themselves as
# `scripts.<lane>.<mod>`, so REPO must be on the path and this must match.
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.answer_node_coverage import lib  # noqa: E402
from scripts.answer_node_coverage.run import (  # noqa: E402
    _corpus_dir_for_suite,
    build_answer_nodes_for_suite,
)
from scripts.stopping_answer_test import lib as stopping_lib  # noqa: E402

SUITES = ("musique", "strategyqa", "wiki2")
GRID = "tier1_trained_qa_base"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population-dir", required=True, type=Path)
    ap.add_argument("--corpora-root", required=True, type=Path)
    ap.add_argument("--model-trained", required=True)
    ap.add_argument("--model-prompted", required=True)
    ap.add_argument("--graph-version", default="v1")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    con = lib.open_store(a.population_dir / "scores_parquet")
    out: dict[str, dict] = {}
    for suite in SUITES:
        t_runs = stopping_lib.load_arm_runs(
            con, arm_id="inquirer_trained", model_id=a.model_trained, grid_name=GRID, suite=suite
        )
        p_runs = stopping_lib.load_arm_runs(
            con, arm_id="inquirer_prompted", model_id=a.model_prompted, grid_name=GRID, suite=suite
        )
        task_ids = sorted({r["task_id"] for r in [*t_runs, *p_runs]})
        corpus_dir = _corpus_dir_for_suite(con, suite, [r["run_id"] for r in [*t_runs, *p_runs]])
        graphs, results = build_answer_nodes_for_suite(
            suite=suite,
            task_ids=task_ids,
            corpora_root=a.corpora_root,
            corpus_dir=corpus_dir,
            graph_version=a.graph_version,
        )
        # stratum: the answer node's own depth. None means gold gives no depth signal at all.
        stratum: dict[str, str] = {}
        for t, res in results.items():
            by_id = {n.gold_node_id: n for n in graphs[t].required()}
            depths = [
                by_id[nid].gold_depth
                for nid in res.node_ids
                if nid in by_id and by_id[nid].gold_depth is not None
            ]
            if not depths:
                stratum[t] = "no_depth_signal"
            elif max(depths) >= 1:
                stratum[t] = "behind_prerequisite"
            else:
                stratum[t] = "nameable_from_task"

        keyed = {(suite, t): r for t, r in results.items()}
        covered = lib.covered_map(con, [*t_runs, *p_runs], keyed)
        row: dict[str, object] = {"tally": {}}
        for name in ("nameable_from_task", "behind_prerequisite", "no_depth_signal"):
            tasks = {t for t, s in stratum.items() if s == name}
            row["tally"][name] = len(tasks)
            if len(tasks) < 10:
                continue
            tr = [r for r in t_runs if r["task_id"] in tasks]
            pr = [r for r in p_runs if r["task_id"] in tasks]
            row[name] = lib.p_covered_delta(tr, pr, covered, n_boot=a.n_boot)
        out[suite] = row
        parts = []
        for name in ("nameable_from_task", "behind_prerequisite"):
            d = row.get(name)
            if d:
                parts.append(f"{name} {d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] n={d['n']}")
        print(f"  {suite:11s} tally={row['tally']}  " + " | ".join(parts))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
