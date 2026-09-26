#!/usr/bin/env python
"""How far does the headline move if the need graph's edges are wrong?

The reviewer question this answers is "how sensitive are your metrics to graph errors". Edges are
the part of the graph that could be wrong: the nodes are the benchmark's own required evidence and
have been read by two raters, but no edge-precision number exists or can exist (both renderings of
an edge-annotation item are degenerate, see the paper's validity appendix). So we perturb the edges
instead of arguing about them.

METHOD. Depth is a function of the edge set alone, and `dwr` and `max_depth_reached` are functions
of depth and of which nodes a run resolved. Both of those are already on disk: `matches.parquet`
records per-run, per-node resolution. So the whole study is a recomputation over stored runs with
no model call, no rescore and no dollars.

THE LOCK COMES FIRST. Before any perturbed number is read, the reimplementation must reproduce the
PUBLISHED per-run `dwr` and `max_depth_reached` exactly. A sensitivity curve from a reader that does
not reproduce the baseline is a statement about the reader.
"""

from __future__ import annotations

import collections
import json
import random
import statistics
import sys
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "artifacts/testsplit_qa/scores_parquet"
GOLD = REPO / "data/gold/graphs"
DWR_WEIGHTS = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0}


def load_graph(suite: str) -> dict[str, tuple[list[str], list[tuple[str, str]]]]:
    """task_id -> (node ids, prerequisite edges as (src, dst))."""
    out: dict[str, tuple[list[str], list[tuple[str, str]]]] = {}
    for line in (GOLD / suite / "v1.jsonl").open():
        g = json.loads(line)
        nodes = [n["gold_node_id"] for n in g.get("gold_nodes", [])]
        edges = [
            (e["gold_src_node_id"], e["gold_dst_node_id"])
            for e in g.get("gold_edges", [])
            if e.get("gold_edge_kind") == "prerequisite"
        ]
        out[g["gold_task_key"]] = (nodes, edges)
    return out


def depths(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int | None]:
    """Longest prerequisite path, Kahn. Unreachable or cycle-trapped nodes get None, never 0."""
    parents: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    kids: dict[str, list[str]] = {n: [] for n in nodes}
    for s, d in edges:
        if s in parents and d in parents:
            parents[d].append(s)
            kids[s].append(d)
            indeg[d] += 1
    out: dict[str, int | None] = {n: (0 if indeg[n] == 0 else None) for n in nodes}
    queue = collections.deque(n for n in nodes if indeg[n] == 0)
    while queue:
        n = queue.popleft()
        for k in kids[n]:
            indeg[k] -= 1
            if out[n] is not None:
                cur = out[k]
                cand = out[n] + 1
                out[k] = cand if cur is None else max(cur, cand)
            if indeg[k] == 0:
                queue.append(k)
    return out


def metrics(resolved: set[str], node_depths: dict[str, int | None]) -> tuple[float, int]:
    """(dwr, max_depth_reached) exactly as pi_eval.metrics.structure computes them."""
    buckets: dict[int, list[str]] = {}
    for nid, d in node_depths.items():
        if d is not None:
            buckets.setdefault(d, []).append(nid)
    cad = {
        d: (sum(1 for i in ids if i in resolved) / len(ids), len(ids)) for d, ids in buckets.items()
    }
    num = sum(DWR_WEIGHTS.get(d, 0.0) * c for d, (c, _) in cad.items())
    den = sum(DWR_WEIGHTS.get(d, 0.0) for d in cad)
    dwr = num / den if den else float("nan")
    hit = [d for nid, d in node_depths.items() if d is not None and nid in resolved]
    return dwr, (max(hit) if hit else -1)


def main() -> int:
    suite = sys.argv[1] if len(sys.argv) > 1 else "musique"
    con = duckdb.connect()
    graphs = load_graph(suite)

    resolved_by_run: dict[str, set[str]] = collections.defaultdict(set)
    task_of_run: dict[str, str] = {}
    for run_id, task_id, node_id in con.execute(
        f"SELECT run_id, task_id, node_id FROM read_parquet('{STORE}/matches.parquet') "
        f"WHERE suite_id = '{suite}' AND match_kind = 'resolve'"
    ).fetchall():
        resolved_by_run[run_id].add(node_id)
        task_of_run[run_id] = task_id
    for run_id, task_id in con.execute(
        f"SELECT DISTINCT run_id, task_id FROM read_parquet('{STORE}/matches.parquet') "
        f"WHERE suite_id = '{suite}'"
    ).fetchall():
        task_of_run.setdefault(run_id, task_id)

    published = collections.defaultdict(dict)
    for run_id, m, v in con.execute(
        f"SELECT run_id, metric_name, value FROM read_parquet('{STORE}/scores.parquet') "
        f"WHERE metric_name IN ('dwr','max_depth_reached')"
    ).fetchall():
        published[run_id][m] = v

    arm_of_run = dict(
        con.execute(
            f"SELECT run_id, arm_id FROM read_parquet('{STORE}/runs.parquet') WHERE suite_id = '{suite}'"
        ).fetchall()
    )
    # ---- THE LOCK: reproduce the published per-run values before reading any perturbation.
    runs = [r for r in task_of_run if r in published and task_of_run[r] in graphs]
    bad_dwr = bad_dep = checked = 0
    base_depths = {t: depths(*graphs[t]) for t in {task_of_run[r] for r in runs}}
    for r in runs:
        nd = base_depths[task_of_run[r]]
        d, mx = metrics(resolved_by_run.get(r, set()), nd)
        p = published[r]
        if "dwr" in p:
            checked += 1
            if abs(d - p["dwr"]) > 1e-9:
                bad_dwr += 1
            if "max_depth_reached" in p and mx != int(p["max_depth_reached"]):
                bad_dep += 1
    print(
        f"LOCK on {suite}: {checked} runs checked, dwr mismatches {bad_dwr}, "
        f"max_depth mismatches {bad_dep}"
    )
    if bad_dwr or bad_dep or checked == 0:
        print("LOCK FAILED. Not reading any perturbed number.")
        return 1
    print("LOCK PASSED at |diff| = 0.\n")

    def paired_delta(nd_by_task) -> tuple[float, float, int]:
        per_task: dict[str, dict[str, list[tuple[float, int]]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for r in runs:
            a = arm_of_run.get(r)
            if a not in ("inquirer_trained", "inquirer_prompted"):
                continue
            t = task_of_run[r]
            per_task[t][a].append((*metrics(resolved_by_run.get(r, set()), nd_by_task[t]),))
        dd, md = [], []
        for t, arms in per_task.items():
            if "inquirer_trained" in arms and "inquirer_prompted" in arms:
                tr, pr = arms["inquirer_trained"], arms["inquirer_prompted"]
                dd.append(statistics.fmean(x[0] for x in tr) - statistics.fmean(x[0] for x in pr))
                md.append(statistics.fmean(x[1] for x in tr) - statistics.fmean(x[1] for x in pr))
        return statistics.fmean(dd), statistics.fmean(md), len(dd)

    b_dwr, b_dep, n = paired_delta(base_depths)
    print(
        f"baseline paired delta, {suite}: dwr {b_dwr:+.5f}  deepest-need {b_dep:+.5f}  n={n} tasks\n"
    )
    print(f"{'edges dropped':>14} {'dwr delta':>12} {'move':>9} {'deepest delta':>14} {'move':>9}")
    for frac in (0.05, 0.10, 0.20, 0.40):
        ds, ms = [], []
        for seed in (0, 1, 2, 3, 4):
            rng = random.Random(hash((seed, frac)) & 0xFFFFFFFF)
            nd = {}
            for t, (nodes, edges) in graphs.items():
                keep = [e for e in edges if rng.random() >= frac]
                nd[t] = depths(nodes, keep)
            d, m, _ = paired_delta(nd)
            ds.append(d)
            ms.append(m)
        md_, mm = statistics.fmean(ds), statistics.fmean(ms)
        print(f"{frac:>13.0%} {md_:>+12.5f} {md_ - b_dwr:>+9.5f} {mm:>+14.5f} {mm - b_dep:>+9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
