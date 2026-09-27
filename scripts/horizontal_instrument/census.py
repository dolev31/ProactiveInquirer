"""Q1 of artifacts/horizontal_axis_instrument_20260919/RESULT.md: a branching census taken
directly from gold graphs, not from a metric.

This reads `pi_eval.gold.load_graphs`, which raises `GoldAccessError` (via `gold_root()`) in
any process without `PI_GOLD_ROOT` set -- that raise is the firewall documented in CONTRIBUTING.md
working correctly, not a bug in this script. It is run with `PI_GOLD_ROOT` pointed at the
repo-local `data/gold` (see the README beside this file for the exact invocation). This module
is analysis tooling under `scripts/`, not one of the root packages
(`pinq`, `pinq_adapters`, `pinq_expt`, `pinq_train`) the import-linter forbidden contract
(`pyproject.toml` `[[tool.importlinter.contracts]]`, name "Nothing outside pi_eval may import
pi_eval (the gold firewall)") names as forbidden sources, so importing `pi_eval` here does not
touch that contract; `lint-imports` is re-run after this file is written to confirm it still
prints "Contracts: 4 kept, 0 broken".

For every suite with a gold graph file on disk, and for `DEFAULT_GRAPH_VERSION` ("v1", pinned
in `pi_eval.score`, the version this paper's stores score against) unless a suite ships no v1,
computes three DIFFERENT operationalizations of "the breadth denominator", per task, straight
off the `GoldGraph` object, deliberately not through `facet_breadth` or `breadth_components`
(which need a policy's resolved records to report anything -- the denominator alone does not):

  n_facets       len(graph.gold_facets): weakly-connected components of prerequisite-linked
                 nodes AFTER DELETING every depth-0 node (GoldFacet's own docstring). This is
                 exactly what `facet_breadth`'s denominator is.
  n_components   union-find over `prerequisite` edges across EVERY node (depth-0 included), so
                 a lone depth-0 seed with no prerequisite edge counts as its own
                 one-node component. This is exactly what `breadth_components`'s denominator is
                 (confirmed against `src/pi_eval/metrics/structure.py`'s own implementation).
  n_seed_nodes   len(graph.gold_seed_node_ids): the literal count of depth-0 entry points, i.e.
                 "how many independent things can be named from the task statement x alone
                 before resolving anything." This is the most literal reading of "root-level
                 branches" in the task brief, and is not reported by either shipped metric.

Reports, per suite, per denominator kind: n_tasks, median, mean, share >= 2, share >= 3, and
the share of tasks where the denominator is 0 (degenerate: no facets/components/seeds at all,
e.g. an empty gold graph). Never pools suites or graph_versions.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from pi_eval.gold import GoldGraph, load_graphs

GOLD_ROOT_REL = "data/gold"  # relative to repo root; caller sets PI_GOLD_ROOT to its absolute form

SUITES = (
    "musique",
    "strategyqa",
    "wiki2",
    "drgym",
    "tau2",
    "tau2_airline",
    "tau2_retail",
    "tau2_telecom",
    "frames",
    "synth",
)

DEFAULT_GRAPH_VERSION = "v1"


def _components_over_prerequisites(graph: GoldGraph) -> int:
    """Union-find over `prerequisite` edges across every node id (breadth_components' own
    denominator; see src/pi_eval/metrics/structure.py:breadth_components, read this session)."""
    ids = [n.gold_node_id for n in graph.gold_nodes]
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in graph.gold_edges:
        if e.gold_edge_kind != "prerequisite":
            continue
        u, v = e.gold_src_node_id, e.gold_dst_node_id
        if u in parent and v in parent:
            parent[find(u)] = find(v)

    return len({find(i) for i in ids})


def _dist(values: list[int]) -> dict:
    n = len(values)
    if n == 0:
        return {
            "n_tasks": 0,
            "median": None,
            "mean": None,
            "share_ge2": None,
            "share_ge3": None,
            "share_eq0": None,
        }
    return {
        "n_tasks": n,
        "median": statistics.median(values),
        "mean": sum(values) / n,
        "share_ge2": sum(1 for v in values if v >= 2) / n,
        "share_ge3": sum(1 for v in values if v >= 3) / n,
        "share_eq0": sum(1 for v in values if v == 0) / n,
    }


def census_one_suite(suite: str, graph_version: str) -> dict | None:
    try:
        graphs = load_graphs(suite, graph_version)
    except TypeError as exc:
        # A later annotation pass added a field (e.g. tau2_{airline,retail,telecom} v2+ carry
        # gold_h1_label) that the CURRENT GoldNode/GoldEdge schema in src/pi_eval/gold.py does
        # not declare. This is schema drift between on-disk gold and the code that reads it,
        # not something this census tool should paper over with **kwargs -- it is reported as
        # a load error for that version and the pinned DEFAULT_GRAPH_VERSION is used instead.
        return {"load_error": str(exc)}
    if not graphs:
        return None
    n_facets, n_components, n_seed_nodes, n_nodes, n_edges, n_prereq_edges = [], [], [], [], [], []
    for g in graphs.values():
        n_facets.append(len(g.gold_facets))
        n_components.append(_components_over_prerequisites(g))
        n_seed_nodes.append(len(g.gold_seed_node_ids))
        n_nodes.append(len(g.gold_nodes))
        n_edges.append(len(g.gold_edges))
        n_prereq_edges.append(sum(1 for e in g.gold_edges if e.gold_edge_kind == "prerequisite"))
    return {
        "suite": suite,
        "graph_version": graph_version,
        "n_tasks_total": len(graphs),
        "n_tasks_nonempty_graph": sum(1 for v in n_nodes if v > 0),
        "n_nodes": _dist(n_nodes),
        "n_prereq_edges": _dist(n_prereq_edges),
        "n_facets": _dist(n_facets),
        "n_components": _dist(n_components),
        "n_seed_nodes": _dist(n_seed_nodes),
    }


def main() -> None:
    import os

    if not os.environ.get("PI_GOLD_ROOT"):
        raise SystemExit(
            "PI_GOLD_ROOT is unset. Run as: "
            f'PI_GOLD_ROOT="$(pwd)/{GOLD_ROOT_REL}" .venv/bin/python '
            "scripts/horizontal_instrument/census.py"
        )
    out = {"default_graph_version": DEFAULT_GRAPH_VERSION, "suites": {}}
    for suite in SUITES:
        # Report every graph_version that exists on disk for this suite, not just v1, because
        # tau2_airline/retail/telecom carry v2..v7 and DEFAULT_GRAPH_VERSION alone would hide
        # whether later annotation passes changed the branching structure.
        suite_dir = Path(os.environ["PI_GOLD_ROOT"]) / "graphs" / suite
        versions = sorted(p.stem for p in suite_dir.glob("v*.jsonl")) if suite_dir.exists() else []
        if not versions:
            out["suites"][suite] = None
            continue
        by_version = {}
        for v in versions:
            res = census_one_suite(suite, v)
            by_version[v] = res
        out["suites"][suite] = {
            "versions_on_disk": versions,
            "pinned": by_version.get(DEFAULT_GRAPH_VERSION),
            "all_versions": by_version,
        }
    dest = Path("artifacts/horizontal_axis_instrument_20260919/census.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
