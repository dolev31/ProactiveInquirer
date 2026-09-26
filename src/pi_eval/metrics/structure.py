"""Depth and dependency-structure metrics.

Anticipation Depth (mean depth over discovered nodes) is deliberately ABSENT: it is
non-monotone. A policy finding only v4 scores 2.0 and beats one finding {v1..v4} at 1.25.
What replaces it is per-depth recall with |V_d| printed, plus one preregistered scalar.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from pi_eval.matcher.base import MatchRecord

_RANK = {"none": 0, "ask": 1, "resolve": 2, "use": 3}


def coverage_at_depth(
    records: Sequence[MatchRecord], graph, level: str = "resolve"
) -> dict[int, tuple[float, int]]:
    """C@d -> (recall, |V_d|). The cardinality is returned WITH the recall, never separately:
    coverage at a depth holding two nodes is not comparable to one holding forty."""
    want = _RANK[level]
    by_id = {r.node_id: r for r in records}
    buckets: dict[int, list[str]] = {}
    for n in graph.gold_nodes:
        if n.gold_depth is not None:
            buckets.setdefault(n.gold_depth, []).append(n.gold_node_id)
    out: dict[int, tuple[float, int]] = {}
    for d, ids in sorted(buckets.items()):
        hit = sum(1 for i in ids if i in by_id and by_id[i].rank >= want)
        out[d] = (hit / len(ids), len(ids))
    return out


def depth_weighted_recall(
    cad: Mapping[int, tuple[float, int]], weights: Mapping[int, float]
) -> float:
    """DWR = sum_d w_d C@d, with w_d PREREGISTERED. Weights chosen after seeing the data
    would make this a free parameter, so they are sealed in prereg/weights_dwr.md."""
    num = sum(weights.get(d, 0.0) * c for d, (c, _) in cad.items())
    den = sum(weights.get(d, 0.0) for d in cad)
    return num / den if den else float("nan")


def max_depth_reached(records: Sequence[MatchRecord], graph, level: str = "resolve") -> int:
    want = _RANK[level]
    by_id = {r.node_id: r for r in records}
    depths = [
        n.gold_depth
        for n in graph.gold_nodes
        if n.gold_depth is not None
        and n.gold_node_id in by_id
        and by_id[n.gold_node_id].rank >= want
    ]
    return max(depths) if depths else -1


def precedence_violation_rate(records: Sequence[MatchRecord], graph) -> float:
    """Does the policy respect the prerequisite topology?

    For each prerequisite edge u->v where BOTH were resolved, a violation is v resolved at an
    earlier turn than u. A policy that stumbles on deep nodes by luck violates often; one
    that genuinely follows the chain cannot violate at all.
    """
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
    return viol / total if total else float("nan")


def facet_breadth(records: Sequence[MatchRecord], graph, level: str = "resolve") -> tuple[int, int]:
    """Horizontal proactivity: how many distinct facets were touched, out of how many exist.

    Separating this from depth is what lets the paper say which KIND of proactivity produced
    a gain, rather than asserting both and measuring neither.
    """
    want = _RANK[level]
    by_id = {r.node_id: r for r in records}
    node_facet = {n.gold_node_id: n.gold_facet_id for n in graph.gold_nodes}
    touched = {
        node_facet[i] for i, r in by_id.items() if r.rank >= want and node_facet.get(i) is not None
    }
    return len(touched), len(graph.gold_facets)


def breadth_components(
    records: Sequence[MatchRecord], graph, level: str = "resolve"
) -> tuple[int, int, int]:
    """Horizontal proactivity where there are no facets: independent lines of inquiry touched.

    `facet_breadth` is defined over `gold_facets`, and MEASURED, tau2 carries none -- 0 facets on
    43 of 43 airline graphs and 112 of 112 retail ones. The whole family therefore scores 0/0,
    which reads as "no breadth" rather than as "not measured here", while VERTICAL is perfectly
    well populated on the same graphs. Two axes the paper claims are both crucial, and only one
    of them measurable.

    A facet is a hand-declared grouping of needs; the prerequisite DAG already carries the same
    information structurally. Two needs joined by a chain are one line of inquiry -- you had to
    know the first to name the second -- and two needs in different weakly-connected components
    are independent things the agent had to think of separately. So breadth is the number of
    COMPONENTS the policy resolved a node in, and it needs no annotation tau2 does not have.

    NOT A COUNT OF RESOLVED ROOTS. Six needs down one chain and six across six chains give the
    same node count always, and can give the same root count; components are what separates
    depth from breadth, which is the distinction this axis exists to make.

    ONLY PREREQUISITE EDGES MERGE. Any other edge kind is an annotation about support or
    similarity, and merging on it would turn an annotation choice into a breadth claim.

    The denominator is the components that EXIST, not the ones touched: a rate over touched
    components scores 1.0 for a policy that found one line out of nine.
    """
    want = _RANK[level]
    ids = [n.gold_node_id for n in graph.gold_nodes]
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in graph.gold_edges:
        if getattr(e, "gold_edge_kind", "") != "prerequisite":
            continue
        u, v = e.gold_src_node_id, e.gold_dst_node_id
        if u in parent and v in parent:
            parent[find(u)] = find(v)

    members = Counter(find(i) for i in ids)
    total = len(members)
    by_id = {r.node_id: r for r in records}
    touched = {find(i) for i in ids if i in by_id and by_id[i].rank >= want}
    # SINGLETONS ARE REPORTED, not hidden. On tau2 gold 90% of airline components and 85% of
    # retail ones are lone nodes, mostly of `unknown` provenance -- so a breadth number that does
    # not say how fragmented its own denominator is describes the gold builder, not the policy.
    singletons = sum(1 for n in members.values() if n == 1)
    return len(touched), total, singletons
