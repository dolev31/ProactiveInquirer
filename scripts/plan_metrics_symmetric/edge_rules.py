"""Does the strategyqa/wiki2 reproduction lock in ladder.py have the POWER to distinguish one
aggregation rule from another, or does it pass for a structural reason unrelated to which rule is
correct? A lock that cannot fail is not evidence -- this module measures the precondition instead
of assuming it, and then runs the decisive test: try musique's reproduction under every plausible
aggregation rule named from the code, and see whether any of them hits the published value.

Nothing here touches the gateway; every number comes from `matches.parquet`/`runs.parquet` plus
gold graphs already on disk.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

from plan_metrics_symmetric.ladder import RunLadder, _pair_keys

from pi_eval.matcher.base import MatchRecord
from pi_eval.stats.inference import paired_difference


def edge_counts(records: Sequence[MatchRecord], graph) -> tuple[int, int]:
    """(violations, qualifying_edges), the raw numerator/denominator that
    `pi_eval.metrics.structure.precedence_violation_rate` divides. Duplicated rather than
    imported, because that function returns only the ratio and the aggregation question below
    needs the two counts separately; `tests/test_edge_rules.py` pins this to agree with the
    source function on every non-empty case, so the duplication cannot silently drift."""
    by_id = {r.node_id: r for r in records if r.matched_turn_idx is not None}
    total = viol = 0
    for e in graph.gold_edges:
        if e.gold_edge_kind != "prerequisite":
            continue
        u, v = by_id.get(e.gold_src_node_id), by_id.get(e.gold_dst_node_id)
        if u is None or v is None:
            continue
        total += 1
        if v.matched_turn_idx < u.matched_turn_idx:
            viol += 1
    return viol, total


def edge_histogram(graphs: Mapping[str, object]) -> dict[str, int]:
    """Structural, run-independent: for every task's graph in `graphs`, how many `prerequisite`
    edges does it declare. Buckets 0 / 1 / '2+'. This is the primary evidence for whether a
    suite's lock COULD have discriminated an aggregation rule -- a task can have depth 1 (no
    chain longer than one hop) and still carry several independent depth-1 edges, so this must
    be measured, not inferred from depth."""
    buckets = {"0": 0, "1": 0, "2+": 0}
    for graph in graphs.values():
        n = sum(1 for e in graph.gold_edges if e.gold_edge_kind == "prerequisite")
        buckets["2+" if n >= 2 else str(n)] += 1
    return buckets


Aggregator = Callable[[list[tuple[int, int]]], float]


def pooled_per_seed_mean(pairs: list[tuple[int, int]]) -> float:
    """RULE A -- the rule already in production (precedence_violation_rate + ladder.py's
    seed-then-task averaging): each seed's own viol/total ratio, arithmetic mean across seeds
    unweighted by how many edges that seed actually saw."""
    rates = [v / t for v, t in pairs if t > 0]
    return sum(rates) / len(rates) if rates else float("nan")


def pooled_across_seeds(pairs: list[tuple[int, int]]) -> float:
    """RULE B -- pool violations and qualifying edges across ALL seeds first, divide once.
    Differs from Rule A exactly when seeds see different numbers of qualifying edges: a seed
    that resolved more prerequisite pairs gets proportionally more weight here; equal weight
    under Rule A regardless of how many edges it resolved."""
    v_sum = sum(v for v, _ in pairs)
    t_sum = sum(t for _, t in pairs)
    return v_sum / t_sum if t_sum > 0 else float("nan")


def any_violation_indicator(pairs: list[tuple[int, int]]) -> float:
    """RULE C -- binary per seed (did this seed violate at least once), meaned across seeds.
    Discards magnitude: 1-of-5 edges violated reads identically to 4-of-5."""
    inds = [1.0 if v > 0 else 0.0 for v, t in pairs if t > 0]
    return sum(inds) / len(inds) if inds else float("nan")


AGGREGATORS: dict[str, Aggregator] = {
    "pooled_per_seed_mean": pooled_per_seed_mean,
    "pooled_across_seeds": pooled_across_seeds,
    "any_violation_indicator": any_violation_indicator,
}


def _per_task_values(
    aggregator: Aggregator,
    ckpt: Mapping[str, RunLadder],
    base: Mapping[str, RunLadder],
    graphs: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float]]:
    """`ladder.asymmetric_contrast`'s exact pairing, truncation, and seed-nesting (checkpoint at
    its own full k; baseline at min(k, its own n_asks), meaned per checkpoint seed, then
    checkpoint seeds meaned per task) with the ONE fixed reduction it uses (ratio, then
    arithmetic mean) replaced by a swappable `aggregator` over raw (violations, edges) pairs.
    Feeding `pooled_per_seed_mean` here must reproduce `asymmetric_contrast`'s own per-task
    values exactly -- the control that proves this refactor changed nothing but the reduction
    (see `test_rule_a_reproduces_the_production_ladder`)."""
    pairs = _pair_keys(ckpt, base)
    per_task_a: dict[str, float] = {}
    per_task_b: dict[str, float] = {}
    for (_suite, task), (ck_ids, ba_ids) in pairs.items():
        graph = graphs.get(task)
        if graph is None:
            continue
        a_vals, b_vals = [], []
        for c_rid in ck_ids:
            c_lad = ckpt[c_rid]
            k = c_lad.n_asks
            a_val = aggregator([edge_counts(c_lad.at(k), graph)])
            b_here = []
            for b_rid in ba_ids:
                b_lad = base[b_rid]
                at = min(k, b_lad.n_asks)
                b_here.append(edge_counts(b_lad.at(at), graph))
            b_mean = aggregator(b_here)
            if math.isnan(a_val) or math.isnan(b_mean):
                continue
            a_vals.append(a_val)
            b_vals.append(b_mean)
        if a_vals:
            per_task_a[task] = sum(a_vals) / len(a_vals)
            per_task_b[task] = sum(b_vals) / len(b_vals)
    return per_task_a, per_task_b


def per_task_values_rule(
    aggregator: Aggregator,
    ckpt: Mapping[str, RunLadder],
    base: Mapping[str, RunLadder],
    graphs: Mapping[str, object],
) -> tuple[dict[str, float], dict[str, float]]:
    """Public wrapper on `_per_task_values`, for callers (diagnostics, tests) that want the raw
    per-task pairs directly rather than a bootstrapped contrast -- e.g. to characterize whether a
    suite's zero delta is two-equal-nonzero values or both arms zero everywhere."""
    return _per_task_values(aggregator, ckpt, base, graphs)


def asymmetric_contrast_rule(
    aggregator: Aggregator,
    ckpt: Mapping[str, RunLadder],
    base: Mapping[str, RunLadder],
    graphs: Mapping[str, object],
    *,
    seed: int = 0,
    n_boot: int = 1000,
):
    """`per_task_values_rule` then `paired_difference` -- the full contrast under a swappable
    aggregation rule, for the decisive test: does any named rule reproduce a suite's published
    delta?"""
    per_task_a, per_task_b = _per_task_values(aggregator, ckpt, base, graphs)
    return paired_difference(per_task_a, per_task_b, n_boot=n_boot, seed=seed)
