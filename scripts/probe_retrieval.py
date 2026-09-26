#!/usr/bin/env python
"""Run the retrieval-sensitivity probe. Zero tokens, zero dollars, seconds.

    python scripts/probe_retrieval.py --suite musique --n 50

Run this BEFORE spending anything on a suite. See src/pi_eval/probe.py for why.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_eval.probe import probe, probe_task  # noqa: E402

# "#1" placeholders are MuSiQue's dependency markers, not text a retriever should see.
_PLACEHOLDER = re.compile(r"#\d+")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--corpus")
    ap.add_argument("--gold-root", default="data/gold")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    gold_path = next((root / a.gold_root / "graphs" / a.suite).glob("*.jsonl"), None)
    if gold_path is None:
        print(f"no gold for {a.suite!r} under {a.gold_root}/graphs/{a.suite}. Build it first.")
        return 2

    corpus = (
        Path(a.corpus)
        if a.corpus
        else next(
            (
                d
                for d in sorted((root / "data" / "corpora" / a.suite).glob("*"))
                if (d / "tasks.jsonl").exists()
            ),
            None,
        )
    )
    if corpus is None:
        print(f"no corpus for {a.suite!r}. Build it first.")
        return 2

    from pi_run.worker import load_suite

    suite = load_suite(a.suite, str(corpus))
    graphs = {}
    for line in gold_path.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            graphs[g["gold_task_key"]] = g

    probes = []
    for tid in list(suite.task_ids())[: a.n]:
        g = graphs.get(tid)
        if not g:
            continue
        subqs = [
            _PLACEHOLDER.sub(" ", n.get("gold_text", "")).strip() for n in g.get("gold_nodes", [])
        ]
        gold_uids = frozenset(u for n in g.get("gold_nodes", []) for u in n.get("gold_ev_uids", []))
        if not subqs or not gold_uids:
            continue
        probes.append(
            probe_task(
                task_id=tid,
                question=suite.view(tid).question,
                subquestions=[s for s in subqs if s],
                gold_uids=gold_uids,
                retriever=suite.retriever(tid),
                k=a.k,
            )
        )

    res = probe(probes, suite_id=a.suite, k=a.k)
    if a.json:
        print(
            json.dumps(
                {
                    "suite": res.suite_id,
                    "n_tasks": res.n_tasks,
                    "k": res.k,
                    "median_distinct_sets": res.median_distinct_sets,
                    "frac_tasks_with_subq_only_evidence": res.frac_tasks_with_subq_only_evidence,
                    "passed": res.passed,
                    "verdict": res.verdict,
                },
                indent=2,
            )
        )
    else:
        print(f"retrieval sensitivity  suite={res.suite_id}  n={res.n_tasks}  k={res.k}")
        print(
            f"  median distinct top-{res.k} sets per task : {res.median_distinct_sets:.2f} "
            f"(need >= {res.min_median_distinct})"
        )
        print(
            f"  tasks with sub-question-only evidence   : "
            f"{res.frac_tasks_with_subq_only_evidence:.0%} "
            f"(need >= {res.min_frac_subq_only:.0%})"
        )
        print(f"  {res.verdict}")
        # An inapplicable instrument has no consequence for the suite; printing the FAIL
        # remedy there would tell you to demote a suite over a limitation of the probe.
        if not res.passed and res.applicable:
            print(f"  CONSEQUENCE: {res.consequence}")
    if not res.applicable and res.n_tasks:
        return 3  # distinct from FAIL(1): nothing was measured
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
