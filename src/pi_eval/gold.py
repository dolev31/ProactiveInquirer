"""Gold information-need graphs, and the process-level guard that keeps them out of rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Provenance = Literal[
    "human_composed",  # MuSiQue question_decomposition: the composition recipe, not post-hoc
    "bench_author",  # tau2 required_documents, DRGym key points
    "mechanical",  # tau2 doc->tool edges, MuSiQue #N placeholders: extracted by regex
    "trace_mined",  # mined from successful trajectories
    "llm_elicited",  # weak supervision
    "click_evidence",  # Researchy DocStream click distributions
    "human_rated",  # adjudicated usefulness
]

Verdict = Literal["NECESSARY", "CONTRIBUTORY", "INERT", "UNTESTABLE"]
EdgeKind = Literal["prerequisite", "relevance"]
EdgeVerification = Literal["mechanical", "intervention", "observational", "human"]


class GoldAccessError(RuntimeError):
    """Raised when a process without PI_GOLD_ROOT tries to read gold."""


def gold_root() -> Path:
    root = os.environ.get("PI_GOLD_ROOT")
    if not root:
        raise GoldAccessError(
            "PI_GOLD_ROOT is unset: this process is not permitted to read gold. "
            "If you are in a rollout worker, this exception is the firewall working correctly."
        )
    return Path(root)


@dataclass(frozen=True, slots=True)
class GoldNode:
    """One information need.

    Every field is gold_*-prefixed or otherwise disjoint from TaskView's field names, so a
    leak cannot typecheck. Asserted by tests/test_firewall.py.

    The distinction that carries the annotation methodology: `gold_support` is FREQUENCY
    across mined traces and only ever yields a candidate; `gold_ablation_verdict` is
    CAUSAL (drop ev(v), re-run, measure) and is the only thing that promotes a node to
    required.
    """

    gold_suite: str
    gold_task_key: str
    gold_node_id: str
    gold_text: str
    gold_aliases: tuple[str, ...] = ()
    gold_kind: Literal["fact", "tool_unlock", "action", "constraint", "preference"] = "fact"
    gold_provenance: tuple[Provenance, ...] = ()
    gold_provenance_primary: Provenance = "trace_mined"
    # mining statistics
    gold_support: int = 0
    gold_cell_support: int = 0
    gold_model_families: tuple[str, ...] = ()
    gold_policy_forms: tuple[str, ...] = ()
    gold_generator_entropy: float = 0.0
    gold_fail_support: int = 0
    gold_lift: float = 0.0
    gold_lift_p: float = 1.0
    # causal verification
    gold_ablation_verdict: Verdict = "UNTESTABLE"
    gold_ablation_delta: float = 0.0
    gold_ablation_ci_lo: float = 0.0
    gold_ablation_ci_hi: float = 0.0
    gold_ablation_n_seeds: int = 0
    # partition and structure
    gold_partition: Literal["required", "optional", "dropped"] = "dropped"
    gold_discoverability: Literal["kb", "user_private", "unknown"] = "unknown"
    gold_depth: int | None = None
    gold_depth_basis: Literal["prereq_only", "prereq_plus_relevance"] = "prereq_only"
    gold_facet_id: str | None = None
    gold_ev_uids: tuple[str, ...] = ()
    # human signal
    gold_human_asked: bool | None = None
    gold_usefulness_rating: float | None = None
    gold_human_adjudicated: bool = False
    # instrument pins
    gold_theta: float = 0.85
    gold_nli_pin: str = ""
    gold_extractor_pin: str = ""
    gold_confidence: float = 0.0
    gold_graph_version: str = ""
    gold_canary: str = ""  # nonce scanned in every serialized LLM request


@dataclass(frozen=True, slots=True)
class GoldEdge:
    """A dependency between needs.

    `prerequisite` means v_dst is UNANSWERABLE until v_src is resolved; `relevance` means
    v_dst only becomes worth asking once v_src is known. Conflating them would make Depth
    mean two different things at once, so they are separate kinds and Depth is computed on
    prerequisites alone by default.
    """

    gold_suite: str
    gold_task_key: str
    gold_src_node_id: str
    gold_dst_node_id: str
    gold_edge_kind: EdgeKind = "prerequisite"
    gold_verified: EdgeVerification = "observational"
    gold_provenance: Provenance = "trace_mined"
    # order-consistency screen, then the counterfactual that upgrades it
    gold_gamma: float = 0.0
    gold_n_both: int = 0
    gold_n_src_absent: int = 0
    gold_resolve_rate_denied: float | None = None
    gold_resolve_rate_control: float | None = None
    gold_confidence: float = 0.0
    gold_human_adjudicated: bool = False
    gold_graph_version: str = ""


@dataclass(frozen=True, slots=True)
class GoldFacet:
    """A weakly-connected component after deleting depth-0 nodes.

    Materialized rather than derived so that horizontal-breadth metrics are a function of a
    pinned graph_version, not of whatever the graph happens to look like at score time.
    """

    gold_suite: str
    gold_task_key: str
    gold_facet_id: str
    gold_node_ids: tuple[str, ...] = ()
    gold_label: str | None = None

    @property
    def size(self) -> int:
        return len(self.gold_node_ids)


@dataclass(frozen=True, slots=True)
class GoldGraph:
    """The per-task graph. Depth is a derived property, never an annotated one."""

    gold_suite: str
    gold_task_key: str
    gold_nodes: tuple[GoldNode, ...] = ()
    gold_edges: tuple[GoldEdge, ...] = ()
    gold_facets: tuple[GoldFacet, ...] = ()
    gold_seed_node_ids: tuple[str, ...] = ()  # depth-0 frontier: entailed by x alone
    gold_graph_version: str = ""
    gold_answer: str = ""
    gold_aliases: tuple[str, ...] = ()
    # Firewall layer 4. Stamped by pi_eval.build.common.write_graphs for every suite, so it
    # is a property of "gold was written" rather than of "a builder remembered to".
    gold_canary: str = ""
    # THE CORPUS THIS GRAPH WAS DERIVED FROM. The corpora tree is content-addressed; the gold
    # tree is NOT -- it is one file per (suite, graph_version), so the last build wins. Without
    # this field nothing connects the two, and gold built from corpus A can score runs rolled
    # against corpus B: same task ids, different evidence uids, every match silently missing,
    # and `evidence_coverage` collapsing to ~0 for a plumbing reason that looks like a result.
    # Empty means "unknown", which is not an error -- a suite whose corpus is upstream may
    # legitimately have none -- but a MISMATCH is.
    gold_corpus_hash: str = ""

    @property
    def answer(self) -> str:
        """The gold answer WITHOUT its canary nonce. Use this for every comparison.

        `gold_answer` carries the nonce on purpose -- it is the string a leak would carry, and
        embedding it there is what makes firewall layer 4 able to fire at all (see
        `pi_eval.build.common.write_graphs`). But the nonce must never enter a comparison: an
        extra reference token dilutes token-F1 and breaks exact match outright, which would
        trade a firewall for two wrong metrics.

        Reaching for the raw field is the exception and should be deliberate.
        """
        from pi_eval.canary import strip

        return strip(self.gold_answer).strip()

    def required(self) -> tuple[GoldNode, ...]:
        return tuple(n for n in self.gold_nodes if n.gold_partition == "required")

    def optional(self) -> tuple[GoldNode, ...]:
        return tuple(n for n in self.gold_nodes if n.gold_partition == "optional")

    def by_depth(self, d: int) -> tuple[GoldNode, ...]:
        return tuple(n for n in self.gold_nodes if n.gold_depth == d)

    def depth_histogram(self) -> dict[int, int]:
        """|V_d|. Always printed alongside C@d: coverage at a depth with two nodes in it is
        not comparable to coverage at a depth with forty."""
        out: dict[int, int] = {}
        for n in self.gold_nodes:
            if n.gold_depth is not None:
                out[n.gold_depth] = out.get(n.gold_depth, 0) + 1
        return dict(sorted(out.items()))


def compute_depths(
    node_ids: list[str],
    edges: list[GoldEdge],
    seed_ids: list[str],
    *,
    basis: Literal["prereq_only", "prereq_plus_relevance"] = "prereq_only",
    semantics: Literal["all", "any"] = "all",
) -> dict[str, int | None]:
    """Depth = dependency distance from information explicitly stated in x.

    SEMANTICS IS NOT A DETAIL. A `prerequisite` edge means "v_src must be resolved before
    v_dst is answerable". When v_dst has several prerequisites you need ALL of them, so its
    depth is 1 + MAX over parents (a longest path), not 1 + MIN (a shortest path / BFS).

    The difference is not hypothetical. Take v4 with prerequisites {v2, v3} where v2 is a
    depth-0 seed and v3 sits at depth 1. Shortest-path calls v4 depth 1, but v4 genuinely
    cannot be resolved until v3 has been, so it is depth 2. BFS therefore SYSTEMATICALLY
    UNDERSTATES depth, which drags genuinely-deep needs into shallow buckets, thins out the
    deep strata that the vertical-proactivity claim is demonstrated on, and blurs the
    depth-1-restricted ablation that the whole claim rests on.

    So `semantics="all"` (AND / longest path) is the default and is what every depth number
    in the paper uses. `semantics="any"` (OR / shortest path) is kept because a genuinely
    disjunctive need — satisfiable by any one of several routes — is a real thing, and
    because reporting both is how a reviewer sees the choice was made rather than defaulted.

    Nodes unreachable from any seed, and nodes trapped in a dependency cycle (which has no
    well-defined depth at all), get None. Their share is reported as orphan_rate rather than
    being quietly assigned a number.
    """
    kinds = ("prerequisite",) if basis == "prereq_only" else ("prerequisite", "relevance")
    nodes = set(node_ids)
    parents: dict[str, list[str]] = {n: [] for n in node_ids}
    children: dict[str, list[str]] = {n: [] for n in node_ids}
    for e in edges:
        if (
            e.gold_edge_kind in kinds
            and e.gold_src_node_id in nodes
            and e.gold_dst_node_id in nodes
        ):
            parents[e.gold_dst_node_id].append(e.gold_src_node_id)
            children[e.gold_src_node_id].append(e.gold_dst_node_id)

    depth: dict[str, int | None] = {n: None for n in node_ids}
    for s in seed_ids:
        if s in depth:
            depth[s] = 0

    if semantics == "any":
        # OR: reachable as soon as ANY parent is. Plain BFS.
        frontier = [s for s in seed_ids if s in depth]
        d = 0
        while frontier:
            d += 1
            nxt: list[str] = []
            for src in frontier:
                for dst in children.get(src, ()):
                    if depth[dst] is None:
                        depth[dst] = d
                        nxt.append(dst)
            frontier = nxt
        return depth

    # AND: a node is resolvable only once EVERY prerequisite is. Kahn's algorithm over the
    # dependency DAG, relaxing to the maximum parent depth. Anything still unresolved when
    # the queue drains is either unreachable or inside a cycle -> None, by construction.
    indeg = {n: len(parents[n]) for n in node_ids}
    queue = [n for n in node_ids if indeg[n] == 0 or depth[n] == 0]
    seen: set[str] = set()
    order: list[str] = []
    while queue:
        n = queue.pop()
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        for c in children.get(n, ()):
            indeg[c] -= 1
            if indeg[c] <= 0 and c not in seen:
                queue.append(c)

    for n in order:
        if depth[n] == 0:
            continue  # a seed stays at 0 even if it also has parents
        ps = [depth[p] for p in parents[n]]
        if ps and all(x is not None for x in ps):
            depth[n] = 1 + max(x for x in ps if x is not None)
        # no parents and not a seed -> unreachable, stays None
    return depth


def orphan_rate(depths: dict[str, int | None]) -> float:
    """Share of nodes with no well-defined depth. Always reported next to C@d, because a
    coverage table over a graph that is 40% orphaned means something very different."""
    if not depths:
        return float("nan")
    return sum(1 for v in depths.values() if v is None) / len(depths)


def load_graphs(suite: str, graph_version: str) -> dict[str, GoldGraph]:
    """Read gold from disk. Raises GoldAccessError in any process without PI_GOLD_ROOT."""
    path = gold_root() / "graphs" / suite / f"{graph_version}.jsonl"
    out: dict[str, GoldGraph] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        g = GoldGraph(
            gold_suite=d["gold_suite"],
            gold_task_key=d["gold_task_key"],
            gold_nodes=tuple(GoldNode(**_tuplify(n)) for n in d.get("gold_nodes", [])),
            gold_edges=tuple(GoldEdge(**_tuplify(e)) for e in d.get("gold_edges", [])),
            gold_facets=tuple(GoldFacet(**_tuplify(f)) for f in d.get("gold_facets", [])),
            gold_seed_node_ids=tuple(d.get("gold_seed_node_ids", [])),
            gold_graph_version=d.get("gold_graph_version", graph_version),
            gold_answer=d.get("gold_answer", ""),
            gold_aliases=tuple(d.get("gold_aliases", [])),
            gold_canary=str(d.get("gold_canary", "")),
            gold_corpus_hash=str(d.get("gold_corpus_hash", "")),
        )
        out[g.gold_task_key] = g
    return out


def _tuplify(d: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
