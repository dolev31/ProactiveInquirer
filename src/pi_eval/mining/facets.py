"""S6b — facets: the horizontal axis.

A facet is a weakly-connected component of the dependency graph AFTER deleting the depth-0
frontier. Deleting the seeds first is what makes the decomposition meaningful: every need is
reachable from the question, so with the seeds left in, the graph is usually one component
and "breadth" would be a constant 1 on every task.

Facets are MATERIALIZED into the gold artifact rather than recomputed at score time, so that
breadth is a function of a pinned `graph_version` and not of whatever the graph happens to
look like when a table is rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Facet:
    facet_id: str
    node_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.node_ids)


def components(
    node_ids: Sequence[str],
    edges: Iterable[tuple[str, str]],
    seed_ids: Iterable[str],
    *,
    task_key: str = "",
) -> list[Facet]:
    seeds = set(seed_ids)
    live = [n for n in node_ids if n not in seeds]
    index = {n: i for i, n in enumerate(live)}
    parent = list(range(len(live)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for src, dst in edges:
        # Undirected on purpose: two needs that share a prerequisite are the same facet even
        # though neither depends on the other.
        if src in index and dst in index:
            union(index[src], index[dst])

    groups: dict[int, list[str]] = {}
    for n in live:
        groups.setdefault(find(index[n]), []).append(n)

    out = []
    for i, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: sorted(kv[1]))):
        out.append(Facet(f"f_{task_key}_{i}" if task_key else f"f{i}", tuple(sorted(members))))
    return out
