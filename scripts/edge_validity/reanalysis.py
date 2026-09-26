"""Edge-validity sensitivity of Table 3's depth cells: the same reader on pruned gold graphs.

WHAT. `artifacts/edge_validity_20260923/CRITERION.md` (committed c2bdf77, before any data)
classes every prerequisite edge u -> v of the Table 3 population VALIDATED, REFUTED or
UNTESTABLE. This module re-reads Table 3's two depth rows -- depth-weighted recall (`dwr`) and
deepest need resolved (`max_depth_reached`) -- for the recipe (seeds 1 and 2 averaged within the
task) minus the same weights prompted, on ALTERNATE graphs:

  primary  every REFUTED edge removed;
  strict   only VALIDATED edges kept.

Same nodes, a subset of edges, every depth recomputed by longest path from the pruned graph's
roots (`pi_eval.gold.compute_depths`, Kahn's order, semantics "all"), facets held FIXED to the
unpruned graph's. The match records do not change -- a node is resolved when its evidence was
retrieved, which no edge touches -- so only the depth each resolved node is credited at moves.

THE READER IS TABLE 3'S, IMPORTED. `scripts/seed_identity/table1_by_seed.pooled_symmetric`
(pairing on (suite, task, rollout seed), both arms at min(k_a, k_b), training seeds averaged
within the task, paired bootstrap over tasks) through `scripts/structured_baselines/
read_contrasts` (one `ladder` module object, the registered metric functions). Pruned graphs
are injected in memory as that function's `graphs` argument; nothing under `src/`, the reader
scripts or the gold root is edited or written.

THE LOCK, before any pruned number. On the UNMODIFIED graphs this path reproduces the recipe's
pooled cells in `artifacts/seed_identity_20260923/table1_by_seed.json` for both metrics on all
three suites -- point, both bounds, n and n_pairs, at each recorded resample count and seed --
and (a) identity pruning reproduces every stored `gold_depth`, (b) the identity-pruned graphs
give the same cell through the same reader, (c) the fast point used for the null equals the
reader's point, (d) the metric functions reproduce the scorer's stored per-run values. `run`
re-runs the whole lock and refuses on any failure.

THE NULL. Per suite x unpruned child depth, the number of edges a variant removes is drawn
uniformly (seeded, without replacement) from all population edges at that child depth, R times;
each draw is pruned and read at the point only. The band is the 2.5/97.5 percentiles. A gain
"holds" (CRITERION.md) if the variant's point lies inside or above the band.

THE COMPLEMENT. The recipe-minus-base gain in the probability of resolving v, over the children
v of REFUTED edges, beside the same over children of VALIDATED (and UNTESTABLE) edges, matched
prefix, same reader (the share is registered as a metric for the duration of the read).

    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python scripts/edge_validity/reanalysis.py lock
    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python scripts/edge_validity/reanalysis.py run \\
        --classes artifacts/edge_validity_20260923/edge_classes.jsonl \\
        --out artifacts/edge_validity_20260923/reanalysis.json
    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python scripts/edge_validity/reanalysis.py smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "scripts", REPO / "scripts" / "seed_identity"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plan_metrics_symmetric import ladder  # noqa: E402  (the one ladder module object)

from pi_eval.gold import GoldGraph, compute_depths  # noqa: E402
from pi_eval.matcher.base import MatchRecord  # noqa: E402
from pi_eval.metrics.structure import coverage_at_depth  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402

OUT_DIR = REPO / "artifacts/edge_validity_20260923"
SMOKE_DIR = OUT_DIR / "_smoke"
TABLE1 = REPO / "artifacts/seed_identity_20260923/table1_by_seed.json"
COHORT = REPO / "artifacts/completed_cohort_20260922"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919"
S_NAMES = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
SUITES: tuple[str, ...] = ("musique", "strategyqa", "wiki2")
METRICS: tuple[str, ...] = ("dwr", "max_depth_reached")
CLASSES: tuple[str, ...] = ("VALIDATED", "REFUTED", "UNTESTABLE")
VARIANTS: tuple[str, ...] = ("primary", "strict")
GRAPH_VERSION = "v1"
RESAMPLES: tuple[tuple[int, int], ...] = (
    (1000, 0),
    (10000, 0),
    (50000, 101),
    (50000, 202),
    (50000, 303),
)
N_NULL = 200
NULL_BASE_SEED = 20260923
TOL = 1e-9
PREREQ = "prerequisite"

Edge = tuple[str, str, str]  # (task_id, u, v)


class Refusal(SystemExit):
    """A population, class file or lock that fails a check. Exits 2, distinct from a crash."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg

    def __str__(self) -> str:
        return self.msg


# ------------------------------------------------------------------------------ graph pruning


def prune_graph(graph: GoldGraph, removed: Iterable[tuple[str, str]]) -> GoldGraph:
    """`graph` with the prerequisite edges `removed` ((u, v) pairs) deleted. Pure.

    Every node is kept. Depth is recomputed by `compute_depths` (longest path, Kahn's order)
    from the pruned graph's ROOTS -- every node left with no incoming prerequisite edge, so a
    child that lost its only parent sits at depth 0, never None. Facets and every node's
    `gold_facet_id` are held fixed to the unpruned graph's. Removing an edge the graph does not
    have is a refusal: it means the class file and the graph disagree.
    """
    rm = set(removed)
    have = {
        (e.gold_src_node_id, e.gold_dst_node_id)
        for e in graph.gold_edges
        if e.gold_edge_kind == PREREQ
    }
    unknown = sorted(rm - have)
    if unknown:
        raise ValueError(f"{graph.gold_task_key}: edges {unknown[:5]} not in the graph")
    kept = tuple(
        e
        for e in graph.gold_edges
        if not (e.gold_edge_kind == PREREQ and (e.gold_src_node_id, e.gold_dst_node_id) in rm)
    )
    ids = [n.gold_node_id for n in graph.gold_nodes]
    has_parent = {e.gold_dst_node_id for e in kept if e.gold_edge_kind == PREREQ}
    roots = [i for i in ids if i not in has_parent]
    depth = compute_depths(ids, list(kept), roots)
    lost = [i for i in ids if depth[i] is None]
    if lost:
        raise ValueError(f"{graph.gold_task_key}: nodes {lost[:5]} have no depth after pruning")
    nodes = tuple(
        n
        if n.gold_depth == depth[n.gold_node_id]
        else dataclasses.replace(n, gold_depth=depth[n.gold_node_id])
        for n in graph.gold_nodes
    )
    return dataclasses.replace(
        graph, gold_nodes=nodes, gold_edges=kept, gold_seed_node_ids=tuple(roots)
    )


def pruned_graphs(graphs: Mapping[str, GoldGraph], removed: Iterable[Edge]) -> dict[str, GoldGraph]:
    """Every graph, with the tasks that lose an edge replaced by their pruned copy."""
    by_task: dict[str, set[tuple[str, str]]] = {}
    for t, u, v in removed:
        by_task.setdefault(t, set()).add((u, v))
    out = dict(graphs)
    for t, rm in by_task.items():
        out[t] = prune_graph(graphs[t], rm)
    return out


# ------------------------------------------------------------------------------ class file


def load_class_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def population_edges(graphs: Mapping[str, GoldGraph], tasks: Iterable[str]) -> list[Edge]:
    return sorted(
        (t, e.gold_src_node_id, e.gold_dst_node_id)
        for t in tasks
        for e in graphs[t].gold_edges
        if e.gold_edge_kind == PREREQ
    )


def index_classes(
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, GoldGraph],
    tasks: Sequence[str],
    suite: str,
) -> dict[Edge, str]:
    """{(task, u, v): class} for `suite`, or a Refusal naming every way the file fails to
    cover the population: a missing edge, an edge the graph lacks, a duplicate, a child_depth
    that is not the unpruned graph's, a class outside CLASSES. Rows of other suites are
    ignored."""
    want = set(population_edges(graphs, tasks))
    depth = {(t, n.gold_node_id): n.gold_depth for t in tasks for n in graphs[t].gold_nodes}
    out: dict[Edge, str] = {}
    problems: dict[str, list[str]] = {}

    def bad(kind: str, what: str) -> None:
        problems.setdefault(kind, []).append(what)

    for r in rows:
        if r.get("suite") != suite:
            continue
        key: Edge = (str(r["task_id"]), str(r["u"]), str(r["v"]))
        if r.get("class") not in CLASSES:
            bad("class", f"{key}: class {r.get('class')!r}")
            continue
        if key not in want:
            bad("not a prerequisite edge of the population", str(key))
            continue
        if key in out:
            bad("duplicate", str(key))
            continue
        cd = r.get("child_depth")
        if cd is None or int(cd) != depth[(key[0], key[2])]:
            bad("child_depth", f"{key}: file {cd} graph {depth[(key[0], key[2])]}")
            continue
        out[key] = str(r["class"])
    missing = sorted(want - set(out))
    if missing:
        problems["missing"] = [str(m) for m in missing]
    if problems:
        summary = "; ".join(f"{k}: {len(v)} (e.g. {v[:3]})" for k, v in sorted(problems.items()))
        raise Refusal(f"{suite}: class file does not cover the population: {summary}")
    return out


def removed_for_variant(classes: Mapping[Edge, str], variant: str) -> set[Edge]:
    if variant == "primary":
        return {e for e, c in classes.items() if c == "REFUTED"}
    if variant == "strict":
        return {e for e, c in classes.items() if c != "VALIDATED"}
    raise ValueError(f"unknown variant {variant!r}")


def edges_by_child_depth(
    edges: Iterable[Edge], graphs: Mapping[str, GoldGraph]
) -> dict[int, list[Edge]]:
    """Edges grouped by the UNPRUNED depth of their child, each group sorted."""
    depth = {}
    out: dict[int, list[Edge]] = {}
    for t, u, v in edges:
        if t not in depth:
            depth[t] = {n.gold_node_id: n.gold_depth for n in graphs[t].gold_nodes}
        out.setdefault(int(depth[t][v]), []).append((t, u, v))
    return {d: sorted(es) for d, es in sorted(out.items())}


# ----------------------------------------------------------------------------------- null


def null_seed(variant: str, suite: str) -> int:
    h = hashlib.sha256(f"{NULL_BASE_SEED}|{variant}|{suite}".encode()).hexdigest()
    return int(h[:16], 16)


def draw_matched_removal(
    pool_by_depth: Mapping[int, Sequence[Edge]],
    counts_by_depth: Mapping[int, int],
    rng: random.Random,
) -> set[Edge]:
    """`counts_by_depth[d]` edges drawn without replacement from `pool_by_depth[d]`, per d."""
    out: set[Edge] = set()
    for d in sorted(counts_by_depth):
        n = counts_by_depth[d]
        pool = list(pool_by_depth.get(d, ()))
        if n > len(pool):
            raise ValueError(f"depth {d}: {n} to draw from a pool of {len(pool)}")
        out.update(rng.sample(pool, n))
    return out


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default rule)."""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# ------------------------------------------------------ the pairing (pooled_symmetric's, mirrored)


@dataclass(frozen=True)
class Pair:
    task_id: str
    a: tuple[MatchRecord, ...]  # the recipe run's records at k = min(k_a, k_b)
    b: tuple[MatchRecord, ...]  # the comparator's, at the same k


def metric_fn(name: str) -> Callable[[Sequence[MatchRecord], Any], float]:
    return ladder.METRIC_FNS[name]


def build_pairs(
    arms: Sequence[Mapping[str, Any]],
    base: Mapping[str, Any],
    graphs: Mapping[str, GoldGraph],
    seeds: Mapping[str, int],
) -> list[Pair]:
    """Exactly `table1_by_seed.pooled_symmetric`'s pairs, in its order, truncated once.

    Key (suite, task, rollout seed); a key seen twice within one arm raises; training seeds in
    `arms` order, keys sorted; a task with no graph is skipped. The truncation does not depend
    on the graph, so the pairs are built once and read under every pruned graph set.
    """

    def keyed(arm: Mapping[str, Any], what: str) -> dict[tuple[str, str, int], Any]:
        out = {}
        for rid, lad in arm.items():
            key = (lad.suite_id, lad.task_id, int(seeds[rid]))
            if key in out:
                raise ValueError(f"{what}: two runs at {key}")
            out[key] = lad
        return out

    b_by = keyed(base, "comparator")
    pairs: list[Pair] = []
    for i, arm in enumerate(arms):
        a_by = keyed(arm, f"training seed #{i}")
        for key in sorted(set(a_by) & set(b_by)):
            if graphs.get(key[1]) is None:
                continue
            k = min(a_by[key].n_asks, b_by[key].n_asks)
            pairs.append(Pair(key[1], tuple(a_by[key].at(k)), tuple(b_by[key].at(k))))
    return pairs


def per_task(
    fn: Callable[[Sequence[MatchRecord], Any], float],
    pairs: Sequence[Pair],
    graphs: Mapping[str, GoldGraph],
) -> tuple[dict[str, float], dict[str, float], int]:
    """pooled_symmetric's per-task means (same summation order), and the pair count."""
    acc: dict[str, tuple[list[float], list[float]]] = {}
    n_pairs = 0
    for p in pairs:
        g = graphs.get(p.task_id)
        if g is None:
            continue
        av, bv = fn(p.a, g), fn(p.b, g)
        if math.isnan(av) or math.isnan(bv):
            continue
        la, lb = acc.setdefault(p.task_id, ([], []))
        la.append(av)
        lb.append(bv)
        n_pairs += 1
    per_a = {t: sum(a) / len(a) for t, (a, _) in acc.items()}
    per_b = {t: sum(b) / len(b) for t, (_, b) in acc.items()}
    return per_a, per_b, n_pairs


def point(per_a: Mapping[str, float], per_b: Mapping[str, float]) -> float:
    """`paired_difference(per_a, per_b).point` without the bootstrap: fmean over shared tasks."""
    keys = sorted(set(per_a) & set(per_b))
    d = [per_a[k] - per_b[k] for k in keys if not (math.isnan(per_a[k]) or math.isnan(per_b[k]))]
    return statistics.fmean(d) if d else float("nan")


def depth_coverage(pairs: Sequence[Pair], graphs: Mapping[str, GoldGraph]) -> dict[int, dict]:
    """C@d (resolve level) for the recipe and the comparator at the matched prefix, averaged
    within the task over pairs and then over the tasks that HAVE depth d, printed beside |V_d|
    (nodes at depth d summed over those tasks) -- coverage at a depth holding two nodes is not
    comparable to one holding forty, and pruning moves nodes between depths."""
    acc: dict[int, dict[str, tuple[list[float], list[float]]]] = {}
    for p in pairs:
        g = graphs[p.task_id]
        ca = coverage_at_depth(p.a, g, level="resolve")
        cb = coverage_at_depth(p.b, g, level="resolve")
        for d in ca:
            la, lb = acc.setdefault(d, {}).setdefault(p.task_id, ([], []))
            la.append(ca[d][0])
            lb.append(cb[d][0])
    out: dict[int, dict] = {}
    for d, by_task in sorted(acc.items()):
        ra_ = statistics.fmean(sum(a) / len(a) for a, _ in by_task.values())
        rb_ = statistics.fmean(sum(b) / len(b) for _, b in by_task.values())
        n_nodes = sum(sum(1 for n in graphs[t].gold_nodes if n.gold_depth == d) for t in by_task)
        out[d] = {
            "n_nodes": n_nodes,
            "n_tasks": len(by_task),
            "recipe": ra_,
            "base": rb_,
            "diff": ra_ - rb_,
        }
    return out


# ----------------------------------------------------------------------------- complement


def child_sets(classes: Mapping[Edge, str], cls: str) -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for (t, _u, v), c in classes.items():
        if c == cls:
            out.setdefault(t, set()).add(v)
    return {t: frozenset(vs) for t, vs in out.items()}


def resolve_share_fn(
    sets: Mapping[str, frozenset[str]],
) -> Callable[[Sequence[MatchRecord], Any], float]:
    """Share of the task's class-c children resolved (rank >= resolve) in the records given;
    NaN where the task has none, so it drops out of the pairing exactly as a NaN metric does."""

    def fn(records: Sequence[MatchRecord], graph: Any) -> float:
        vs = sets.get(graph.gold_task_key)
        if not vs:
            return float("nan")
        by_id = {r.node_id: r for r in records}
        return sum(1 for v in vs if v in by_id and by_id[v].rank >= 2) / len(vs)

    return fn


# ------------------------------------------------------------------------------ population


@dataclass
class Population:
    suite: str
    graphs: dict[str, GoldGraph]
    tasks: list[str]
    base: dict[str, Any]
    s1: dict[str, Any]
    s2: dict[str, Any]
    seeds: dict[str, int]
    templates: dict[str, str]
    pairs: list[Pair]
    ids: dict[str, list[str]]
    stored: dict[str, dict[tuple[str, str], float]]


def _reader():
    """table1_by_seed (and through it read_contrasts), imported lazily: importing them aliases
    `sys.modules['ladder']`, which the pure functions above do not need."""
    import table1_by_seed as t1

    assert t1.rc.ladder is ladder, "the reader bound a different ladder module"
    return t1.rc, t1


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_population(suites: Sequence[str]) -> dict[str, Population]:
    import duckdb

    from pi_eval.gold import load_graphs

    rc, _t1 = _reader()
    seeds = {**rc._seed_map(COHORT / "scores_parquet"), **rc._seed_map(SEEDREP / "scores_parquet")}
    con = duckdb.connect()
    out: dict[str, Population] = {}
    for suite in suites:
        graphs = load_graphs(suite, GRAPH_VERSION)
        ids = {
            "base": _ids(COHORT / "cohort" / f"run_ids.prompted.{suite}.txt"),
            **{
                k: _ids(SEEDREP / "run_ids" / f"run_ids.{v}.{suite}.txt")
                for k, v in S_NAMES.items()
            },
        }
        stores = {"base": COHORT / "scores_parquet", "s1": SEEDREP / "scores_parquet"}
        stores["s2"] = stores["s1"]
        lad = {k: ladder.load_run_ladders(stores[k], ids[k]) for k in ids}
        stored = {k: rc.sweep._stored_values(stores[k], ids[k]) for k in ids}
        tmpl = dict(
            con.execute(
                "SELECT task_id, template_id FROM read_parquet(?) WHERE run_id IN "
                "(SELECT unnest(?)) AND template_id IS NOT NULL AND template_id <> ''",
                [str(COHORT / "scores_parquet" / "runs.parquet"), ids["base"]],
            ).fetchall()
        )
        tasks = sorted({x.task_id for x in lad["base"].values()})
        pairs = build_pairs([lad["s1"], lad["s2"]], lad["base"], graphs, seeds)
        out[suite] = Population(
            suite, graphs, tasks, lad["base"], lad["s1"], lad["s2"], seeds, tmpl, pairs, ids, stored
        )
    return out


def provenance(pops: Mapping[str, Population]) -> dict[str, Any]:
    import duckdb

    con = duckdb.connect()
    hashes = sorted(
        {
            r[0]
            for st in (COHORT, SEEDREP)
            for r in con.execute(
                "SELECT DISTINCT scorer_hash FROM read_parquet(?)",
                [str(st / "scores_parquet" / "scores.parquet")],
            ).fetchall()
        }
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    rc, _ = _reader()
    return {
        "scorer_hash": hashes,
        "graph_version": GRAPH_VERSION,
        "git_head": head,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stores": {
            "comparator": str((COHORT / "scores_parquet").relative_to(REPO)),
            "recipe": str((SEEDREP / "scores_parquet").relative_to(REPO)),
        },
        "run_id_sha256": {
            s: {arm: rc._digest(p.ids[arm]) for arm in ("base", "s1", "s2")}
            for s, p in pops.items()
        },
        "n_runs": {
            s: {arm: len(p.ids[arm]) for arm in ("base", "s1", "s2")} for s, p in pops.items()
        },
        "n_tasks": {s: len(p.tasks) for s, p in pops.items()},
        "n_pairs_built": {s: len(p.pairs) for s, p in pops.items()},
    }


# --------------------------------------------------------------------------------- reading


def _verdict(readings: Sequence[Mapping[str, Any]]) -> str:
    fifty = [x for x in readings if x["n_boot"] == 50000]
    if not fifty:
        return "NOT_READ_AT_50K"
    if all(x["ci_lo"] > 0 for x in fifty) or all(x["ci_hi"] < 0 for x in fifty):
        return "DECIDED"
    return "SPANS_ZERO"


def _near_zero(readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Any bound within 0.01 of zero, and whether the 1k reading's side differs from 10k's."""

    def side(x):
        return "pos" if x["ci_lo"] > 0 else "neg" if x["ci_hi"] < 0 else "spans"

    near = any(min(abs(x["ci_lo"]), abs(x["ci_hi"])) < 0.01 for x in readings)
    by = {(x["n_boot"], x["seed"]): side(x) for x in readings}
    return {
        "bound_within_0.01_of_zero": near,
        "sides": {f"{nb}@{sd}": s for (nb, sd), s in by.items()},
        "stable_across_counts": len(set(by.values())) == 1,
    }


def read_cell(t1, metric: str, pop: Population, graphs: Mapping[str, GoldGraph], resamples) -> dict:
    readings = [
        {
            "n_boot": nb,
            "seed": sd,
            **t1.pooled_symmetric(
                metric, [pop.s1, pop.s2], pop.base, graphs, pop.seeds, seed=sd, n_boot=nb
            ),
        }
        for nb, sd in resamples
    ]
    return {"readings": readings, "verdict": _verdict(readings), "near_zero": _near_zero(readings)}


def clustered_readings(fn, pop: Population, graphs, resamples) -> dict:
    """The same per-task means, bootstrapped over TEMPLATE clusters (MuSiQue carries
    template_id; 200 tasks in 186 clusters). A secondary reading: Table 3's cells are not
    clustered, so the lock and the primary cells are not either."""
    per_a, per_b, _ = per_task(fn, pop.pairs, graphs)
    rows = []
    for nb, sd in resamples:
        est = paired_difference(per_a, per_b, clusters=pop.templates, n_boot=nb, seed=sd)
        rows.append(
            {
                "n_boot": nb,
                "seed": sd,
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                "note": est.note,
            }
        )
    return {"readings": rows, "verdict": _verdict(rows)}


# ------------------------------------------------------------------------------------ lock


def run_lock(pops: Mapping[str, Population]) -> dict[str, Any]:
    rc, t1 = _reader()
    pub = json.loads(TABLE1.read_text())["cells"]["s1s2"]
    out: dict[str, Any] = {
        "target": str(TABLE1.relative_to(REPO)) + " cells.s1s2",
        "reader": "scripts/seed_identity/table1_by_seed.py::pooled_symmetric",
        "tol": TOL,
        "cells": {},
        "depth_identity": {},
        "identity_prune": {},
        "fast_path": {},
        "instrument": {},
    }
    ok = True
    for suite, pop in pops.items():
        n_nodes = n_same = 0
        for t in pop.tasks:
            g = pop.graphs[t]
            p = prune_graph(g, ())
            for x, y in zip(g.gold_nodes, p.gold_nodes, strict=True):
                n_nodes += 1
                n_same += x.gold_depth == y.gold_depth
        out["depth_identity"][suite] = {"n_nodes": n_nodes, "n_same": n_same}
        ok &= n_nodes > 0 and n_same == n_nodes
        ident = {**pop.graphs, **{t: prune_graph(pop.graphs[t], ()) for t in pop.tasks}}
        for metric in METRICS:
            key = f"{metric}::{suite}"
            rows = []
            for nb, sd in t1.POOL_RESAMPLES:
                got = t1.pooled_symmetric(
                    metric, [pop.s1, pop.s2], pop.base, pop.graphs, pop.seeds, seed=sd, n_boot=nb
                )
                want = next(x for x in pub[key] if x["n_boot"] == nb and x["seed"] == sd)
                diff = {f: abs(got[f] - want[f]) for f in ("delta", "ci_lo", "ci_hi")}
                row_ok = (
                    max(diff.values()) <= TOL
                    and got["n"] == want["n"]
                    and got["n_pairs"] == want["n_pairs"]
                )
                ok &= row_ok
                rows.append(
                    {
                        "n_boot": nb,
                        "seed": sd,
                        "published": {
                            f: want[f] for f in ("delta", "ci_lo", "ci_hi", "n", "n_pairs")
                        },
                        "reproduced": got,
                        "max_abs_diff": max(diff.values()),
                        "ok": row_ok,
                    }
                )
            out["cells"][key] = rows
            got_id = t1.pooled_symmetric(
                metric, [pop.s1, pop.s2], pop.base, ident, pop.seeds, seed=0, n_boot=10000
            )
            ref = rows[0]["reproduced"]
            id_ok = all(got_id[f] == ref[f] for f in ("delta", "ci_lo", "ci_hi", "n", "n_pairs"))
            out["identity_prune"][key] = {"ok": id_ok, **got_id}
            ok &= id_ok
            per_a, per_b, n_pairs = per_task(metric_fn(metric), pop.pairs, pop.graphs)
            est = paired_difference(per_a, per_b, n_boot=10000, seed=0)
            fp = point(per_a, per_b)
            fast_ok = (
                fp == ref["delta"]
                and (est.point, est.ci_lo, est.ci_hi) == (ref["delta"], ref["ci_lo"], ref["ci_hi"])
                and n_pairs == ref["n_pairs"]
            )
            out["fast_path"][key] = {
                "ok": fast_ok,
                "point": fp,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n_pairs": n_pairs,
            }
            ok &= fast_ok
            for arm in ("base", "s1", "s2"):
                lad = getattr(pop, arm)
                inst = rc.instrument_check(metric, lad, pop.graphs, pop.stored[arm])
                inst_ok = rc.instrument_ok(inst)
                out["instrument"].setdefault(key, {})[arm] = {"ok": inst_ok, **inst}
                ok &= inst_ok
    out["ok"] = bool(ok)
    return out


def lock_table(lock: Mapping[str, Any]) -> str:
    lines = ["cell | n_boot@seed | published delta [lo, hi] | reproduced | max|diff| | ok"]
    for key, rows in lock["cells"].items():
        for r in rows:
            p, g = r["published"], r["reproduced"]
            lines.append(
                f"{key} | {r['n_boot']}@{r['seed']} | {p['delta']!r} [{p['ci_lo']!r}, "
                f"{p['ci_hi']!r}] n={p['n']} | {g['delta']!r} [{g['ci_lo']!r}, {g['ci_hi']!r}] "
                f"n={g['n']} | {r['max_abs_diff']:.3g} | {r['ok']}"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- run


def run_reanalysis(
    class_file: Path,
    out: Path,
    suites: Sequence[str],
    *,
    n_null: int = N_NULL,
    resamples: Sequence[tuple[int, int]] = RESAMPLES,
    smoke: bool = False,
    pops: Mapping[str, Population] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    rows = load_class_rows(class_file)
    in_smoke = SMOKE_DIR in class_file.resolve().parents
    if not smoke and (in_smoke or any(r.get("smoke") for r in rows)):
        raise Refusal(f"{class_file} is a SMOKE class file; refusing to write a result from it")
    if smoke and SMOKE_DIR not in out.resolve().parents:
        raise Refusal(f"smoke output must live under {SMOKE_DIR.relative_to(REPO)}")
    rc, t1 = _reader()
    pops = pops if pops is not None else load_population(suites)
    lock = run_lock(pops)
    if not lock["ok"]:
        raise Refusal("LOCK FAILED; no pruned number is written:\n" + lock_table(lock))
    result: dict[str, Any] = {
        "SMOKE": smoke,
        "smoke_warning": "SYNTHETIC random class labels; these numbers are NOT results"
        if smoke
        else None,
        "criterion": "artifacts/edge_validity_20260923/CRITERION.md (commit c2bdf77)",
        "class_file": str(class_file.resolve().relative_to(REPO)),
        "class_file_sha256": hashlib.sha256(class_file.read_bytes()).hexdigest(),
        "rule": "table1_by_seed.pooled_symmetric: pair (suite, task, rollout seed), both arms at "
        "min(k_a, k_b), training seeds 1+2 averaged within the task, paired bootstrap over tasks",
        "provenance": provenance(pops),
        "lock_ok": lock["ok"],
        "lock_cells": {
            k: [
                {"n_boot": r["n_boot"], "seed": r["seed"], "max_abs_diff": r["max_abs_diff"]}
                for r in v
            ]
            for k, v in lock["cells"].items()
        },
        "n_null": n_null,
        "null_seeds": {v: {s: null_seed(v, s) for s in suites} for v in VARIANTS},
        "suites": {},
    }
    classes_by_suite = {s: index_classes(rows, pops[s].graphs, pops[s].tasks, s) for s in suites}
    extra = {
        f"edge_validity.resolve_share.{cls}.{s}": resolve_share_fn(
            child_sets(classes_by_suite[s], cls)
        )
        for s in suites
        for cls in CLASSES
    }
    with rc.sweep.registered_metrics(extra):
        for suite in suites:
            pop = pops[suite]
            classes = classes_by_suite[suite]
            all_edges = sorted(classes)
            pool = edges_by_child_depth(all_edges, pop.graphs)
            sres: dict[str, Any] = {
                "n_edges": {d: len(v) for d, v in pool.items()},
                "class_by_child_depth": {
                    cls: {
                        d: len(v)
                        for d, v in edges_by_child_depth(
                            [e for e in all_edges if classes[e] == cls], pop.graphs
                        ).items()
                    }
                    for cls in CLASSES
                },
                "unpruned": {
                    "points": {
                        m: point(*per_task(metric_fn(m), pop.pairs, pop.graphs)[:2])
                        for m in METRICS
                    },
                    "depth_coverage": depth_coverage(pop.pairs, pop.graphs),
                },
                "variants": {},
                "complement": {},
            }
            for variant in VARIANTS:
                removed = removed_for_variant(classes, variant)
                counts = {d: len(v) for d, v in edges_by_child_depth(removed, pop.graphs).items()}
                pg = pruned_graphs(pop.graphs, removed)
                vres: dict[str, Any] = {
                    "n_removed_by_child_depth": counts,
                    "n_removed": len(removed),
                    "depth_coverage": depth_coverage(pop.pairs, pg),
                    "cells": {},
                }
                rng = random.Random(null_seed(variant, suite))
                null_pts: dict[str, list[float]] = {m: [] for m in METRICS}
                for _ in range(n_null):
                    g_r = pruned_graphs(pop.graphs, draw_matched_removal(pool, counts, rng))
                    for m in METRICS:
                        null_pts[m].append(point(*per_task(metric_fn(m), pop.pairs, g_r)[:2]))
                for m in METRICS:
                    cell = read_cell(t1, m, pop, pg, resamples)
                    per_a, per_b, _ = per_task(metric_fn(m), pop.pairs, pg)
                    obs = point(per_a, per_b)
                    ref = next(r for r in cell["readings"] if r["n_boot"] == 10000)["delta"]
                    if obs != ref:
                        raise Refusal(f"{variant} {suite} {m}: fast point {obs} != reader {ref}")
                    lo, hi = percentile(null_pts[m], 0.025), percentile(null_pts[m], 0.975)
                    cell["arm_levels"] = {
                        "recipe": statistics.fmean(per_a[t] for t in sorted(per_a)),
                        "base": statistics.fmean(per_b[t] for t in sorted(per_b)),
                    }
                    cell["unpruned_point"] = sres["unpruned"]["points"][m]
                    cell["null"] = {
                        "R": n_null,
                        "lo_2.5": lo,
                        "hi_97.5": hi,
                        "mean": statistics.fmean(null_pts[m]) if null_pts[m] else float("nan"),
                        "share_of_draws_below_observed": sum(x < obs for x in null_pts[m])
                        / max(1, len(null_pts[m])),
                        "holds": obs >= lo,
                        "above_band": obs > hi,
                    }
                    if pop.templates:
                        cell["clustered_template"] = clustered_readings(
                            metric_fn(m), pop, pg, [r for r in resamples if r[0] >= 10000]
                        )
                    vres["cells"][m] = cell
                sres["variants"][variant] = vres
            for cls in CLASSES:
                name = f"edge_validity.resolve_share.{cls}.{suite}"
                sets = child_sets(classes, cls)
                per_a, per_b, n_pairs = per_task(metric_fn(name), pop.pairs, pop.graphs)
                entry: dict[str, Any] = {
                    "n_children": sum(len(v) for v in sets.values()),
                    "n_tasks_with_children": len(sets),
                    "n_tasks_read": len(per_a),
                }
                if per_a:
                    entry.update(read_cell(t1, name, pop, pop.graphs, resamples))
                    entry["arm_levels"] = {
                        "recipe": statistics.fmean(per_a[t] for t in sorted(per_a)),
                        "base": statistics.fmean(per_b[t] for t in sorted(per_b)),
                    }
                sres["complement"][cls] = entry
            result["suites"][suite] = sres
            print(f"[{time.time() - t0:.0f}s] {suite} done", flush=True)
    result["runtime_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    return result


def summary_table(result: Mapping[str, Any]) -> str:
    lines = [
        ("SMOKE -- SYNTHETIC LABELS, NOT A RESULT" if result.get("SMOKE") else "RESULT"),
        "suite | variant | metric | removed | point [10k lo, hi] | 50k verdict | null [2.5, 97.5] "
        "| holds | unpruned",
    ]
    for s, sres in result["suites"].items():
        for v, vres in sres["variants"].items():
            for m, c in vres["cells"].items():
                r10 = next(r for r in c["readings"] if r["n_boot"] == 10000)
                n = c["null"]
                lines.append(
                    f"{s} | {v} | {m} | {vres['n_removed']} | {r10['delta']:+.6f} "
                    f"[{r10['ci_lo']:+.6f}, {r10['ci_hi']:+.6f}] | {c['verdict']} | "
                    f"[{n['lo_2.5']:+.6f}, {n['hi_97.5']:+.6f}] | {n['holds']} | "
                    f"{c['unpruned_point']:+.6f}"
                )
        for cls, e in sres["complement"].items():
            if "readings" in e:
                r10 = next(r for r in e["readings"] if r["n_boot"] == 10000)
                lines.append(
                    f"{s} | complement {cls} children ({e['n_children']} in "
                    f"{e['n_tasks_read']} tasks) | P(resolve) | {r10['delta']:+.6f} "
                    f"[{r10['ci_lo']:+.6f}, {r10['ci_hi']:+.6f}] | {e['verdict']}"
                )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- smoke


def write_smoke_classes(pops: Mapping[str, Population], path: Path, seed: int = 7) -> int:
    """Random class labels for every population edge, marked `smoke`, under _smoke/ only."""
    if SMOKE_DIR not in path.resolve().parents:
        raise Refusal("smoke class files live under _smoke/ only")
    rng = random.Random(seed)
    lines = []
    for suite, pop in pops.items():
        depth = {
            (t, n.gold_node_id): n.gold_depth for t in pop.tasks for n in pop.graphs[t].gold_nodes
        }
        for t, u, v in population_edges(pop.graphs, pop.tasks):
            cls = rng.choices(CLASSES, weights=(0.3, 0.3, 0.4))[0]
            lines.append(
                json.dumps(
                    {
                        "suite": suite,
                        "task_id": t,
                        "u": u,
                        "v": v,
                        "child_depth": depth[(t, v)],
                        "class": cls,
                        "smoke": True,
                        "reasons": ["SYNTHETIC smoke label"],
                    }
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ------------------------------------------------------------------------------------ main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser(
        "lock", help="reproduce Table 3's pooled depth cells; write the lock record"
    )
    lk.add_argument("--out", type=Path, default=OUT_DIR / "reanalysis_lock.json")
    rn = sub.add_parser("run", help="the sensitivity read on a class file (re-runs the lock first)")
    rn.add_argument("--classes", type=Path, required=True)
    rn.add_argument("--out", type=Path, default=OUT_DIR / "reanalysis.json")
    rn.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    rn.add_argument("--n-null", type=int, default=N_NULL)
    sm = sub.add_parser("smoke", help="end to end on SYNTHETIC labels, written under _smoke/ only")
    sm.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args(argv)
    if not os.environ.get("PI_GOLD_ROOT"):
        print("reanalysis: REFUSING: PI_GOLD_ROOT unset (scoring-stage reader)", file=sys.stderr)
        return 3
    try:
        if args.cmd == "lock":
            t0 = time.time()
            pops = load_population(SUITES)
            lock = run_lock(pops)
            lock["provenance"] = provenance(pops)
            lock["runtime_s"] = time.time() - t0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(lock, indent=1, sort_keys=True, default=str) + "\n")
            print(lock_table(lock))
            print(f"LOCK {'OK' if lock['ok'] else 'FAILED'}; wrote {args.out}")
            return 0 if lock["ok"] else 2
        if args.cmd == "run":
            res = run_reanalysis(args.classes, args.out, args.suites, n_null=args.n_null)
            print(summary_table(res))
            print(f"wrote {args.out}")
            return 0
        pops = load_population(SUITES)
        cls_path = SMOKE_DIR / "edge_classes.SMOKE.jsonl"
        n = write_smoke_classes(pops, cls_path)
        print(f"wrote {n} SYNTHETIC labels to {cls_path.relative_to(REPO)}")
        out = SMOKE_DIR / "reanalysis.SMOKE.json"
        res = run_reanalysis(cls_path, out, SUITES, n_null=args.n_null, smoke=True, pops=pops)
        print(summary_table(res))
        print(f"wrote {out.relative_to(REPO)} (SMOKE; never a result)")
        return 0
    except Refusal as r:
        print(f"reanalysis: REFUSING: {r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
