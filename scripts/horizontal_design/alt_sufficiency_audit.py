"""Audit: do this project's gold graphs ever admit ALTERNATIVE SUFFICIENT SETS of required
evidence, or is every required node conjunctively required everywhere?

Reads real gold graphs via `pi_eval.gold.load_graphs` (requires PI_GOLD_ROOT; this is a
gold-reading tool, run offline, $0, no scores, no sweeps). For each suite/graph_version it
reports, per task:

  - n_facets, n_components (same operationalizations as the required pre-reading census)
  - partition counts (required / optional / dropped), overall and per-facet
  - among tasks with n_facets >= 2: how many facets are "fully skippable" (every node in the
    facet is optional or dropped, i.e. dropping the whole facet costs zero required coverage)
  - a structural check that no edge ever crosses two different facet ids (facets are, by the
    `_components` construction in pi_eval/build/common.py, weakly-connected components of the
    depth>=1 subgraph, so this must read 0 always; measured rather than assumed, per project
    rule "measure, don't guess a schema fact")

This is a read-only audit tool. It does not import anything from pi_run/pinq_* rollout code,
does not compute any of the shipped metrics (facet_breadth, breadth_components,
evidence_coverage), and does not touch any scores/parquet store.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("PI_GOLD_ROOT", str(REPO_ROOT / "data" / "gold"))

from pi_eval.gold import load_graphs  # noqa: E402

SUITES_VERSIONS = [
    ("musique", "v1"),
    ("strategyqa", "v1"),
    ("wiki2", "v1"),
    ("drgym", "v1"),
    ("synth", "v1"),
    ("tau2", "v1"),
    ("tau2_airline", "v1"),
    ("tau2_retail", "v1"),
    ("tau2_telecom", "v1"),
    ("frames", "v1"),
]


def audit_suite(suite: str, version: str) -> dict:
    graphs = load_graphs(suite, version)
    n_tasks = len(graphs)
    partition_counts = {"required": 0, "optional": 0, "dropped": 0}
    n_facets_hist: dict[int, int] = {}
    tasks_with_ge2_facets = 0
    tasks_ge2_facets_all_required_no_skippable = 0
    tasks_ge2_facets_with_a_skippable_facet = 0
    cross_facet_edges = 0
    total_prereq_edges = 0
    skippable_facet_examples: list[dict] = []

    for task_key, g in graphs.items():
        for n in g.gold_nodes:
            partition_counts[n.gold_partition] = partition_counts.get(n.gold_partition, 0) + 1

        n_facets = len(g.gold_facets)
        n_facets_hist[n_facets] = n_facets_hist.get(n_facets, 0) + 1

        node_by_id = {n.gold_node_id: n for n in g.gold_nodes}
        facet_by_node = {n.gold_node_id: n.gold_facet_id for n in g.gold_nodes}

        for e in g.gold_edges:
            if e.gold_edge_kind != "prerequisite":
                continue
            total_prereq_edges += 1
            fa = facet_by_node.get(e.gold_src_node_id)
            fb = facet_by_node.get(e.gold_dst_node_id)
            if fa is not None and fb is not None and fa != fb:
                cross_facet_edges += 1

        if n_facets >= 2:
            tasks_with_ge2_facets += 1
            any_skippable = False
            for facet in g.gold_facets:
                partitions = {
                    node_by_id[nid].gold_partition
                    for nid in facet.gold_node_ids
                    if nid in node_by_id
                }
                if partitions and partitions <= {"optional", "dropped"}:
                    any_skippable = True
                    if len(skippable_facet_examples) < 5:
                        skippable_facet_examples.append(
                            {
                                "suite": suite,
                                "task_key": task_key,
                                "facet_id": facet.gold_facet_id,
                                "partitions": sorted(partitions),
                                "n_facets_in_task": n_facets,
                            }
                        )
            if any_skippable:
                tasks_ge2_facets_with_a_skippable_facet += 1
            else:
                tasks_ge2_facets_all_required_no_skippable += 1

    return {
        "suite": suite,
        "graph_version": version,
        "n_tasks": n_tasks,
        "partition_counts": partition_counts,
        "n_facets_hist": {str(k): v for k, v in sorted(n_facets_hist.items())},
        "share_ge2_facets": (tasks_with_ge2_facets / n_tasks) if n_tasks else None,
        "tasks_with_ge2_facets": tasks_with_ge2_facets,
        "tasks_ge2_facets_all_required_no_skippable_facet": tasks_ge2_facets_all_required_no_skippable,
        "tasks_ge2_facets_with_a_skippable_facet": tasks_ge2_facets_with_a_skippable_facet,
        "cross_facet_prerequisite_edges": cross_facet_edges,
        "total_prerequisite_edges": total_prereq_edges,
        "skippable_facet_examples": skippable_facet_examples,
    }


def main() -> None:
    out = []
    for suite, version in SUITES_VERSIONS:
        try:
            out.append(audit_suite(suite, version))
        except Exception as exc:  # noqa: BLE001 - report the load error, do not hide it
            out.append({"suite": suite, "graph_version": version, "load_error": repr(exc)})
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
