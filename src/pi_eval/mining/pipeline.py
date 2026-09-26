"""S0..S7 wired together, plus the gates that decide whether the output may be used at all.

The framing that makes this defensible, and which belongs in the paper's own words: the
mined graph is A MEASUREMENT INSTRUMENT WITH A CHARACTERIZED ERROR PROFILE, not a recovery of
the true latent graph G. Every objection below is fatal to the strong reading ("we recovered
G") and survivable under the weak one ("we built a lower-bound instrument, measured its bias,
and never let it carry a primary endpoint alone").

Consequently `mine()` returns a MinedGraph carrying its own diagnostics, and `gate_report()`
decides admissibility. A graph that fails a gate is still emitted — with `admissible=False`
and the reason — because a systematically-skipped stratum must stay visible rather than
silently vanishing from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from pi_eval.mining.partition import DISCOVERABILITY_PIN
from pinq.ids import h

from .aggregate import MinedNode, aggregate
from .canon import Candidate, EntailFn, cluster, theta_sweep
from .depth import SeedVerdict, seed_basis_shares, seed_set
from .edges import MinedEdge, counterfactual_edge_test, mechanical_edges, screen_order
from .facets import Facet, components
from .partition import PartitionVerdict, partition, partition_elasticity
from .pool import TracePool, pool_is_admissible

NLI_PIN_DEFAULT = "deberta-v3-large-mnli@unpinned"


@dataclass
class MinedGraph:
    suite: str
    task_key: str
    nodes: list[MinedNode] = field(default_factory=list)
    edges: list[MinedEdge] = field(default_factory=list)
    facets: list[Facet] = field(default_factory=list)
    seeds: list[SeedVerdict] = field(default_factory=list)
    partitions: list[PartitionVerdict] = field(default_factory=list)
    depths: dict[str, int | None] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    admissible: bool = True
    reason: str = "ok"
    theta: float = 0.85
    nli_pin: str = NLI_PIN_DEFAULT

    @property
    def graph_hash(self) -> str:
        """Identity of the graph as an instrument: config + thresholds + pins.

        Flows into scorer_hash, which is why a graph revision ADDS score rows under a new
        hash instead of invalidating the rollouts it was computed over.
        """
        return h(
            "graph",
            self.suite,
            self.task_key,
            f"{self.theta:.2f}",
            self.nli_pin,
            # The DISCOVERABILITY TOKENISER is part of the instrument too. It decides
            # kb-vs-user_private, and therefore `private_share` -- the ceiling on what any
            # autonomous inquirer could reach. Two graphs partitioned under different
            # tokenisers are two different measurements and must not share a hash.
            DISCOVERABILITY_PIN,
            str(sorted(n.node_id for n in self.nodes)),
            str(sorted((e.src, e.dst, e.edge_kind) for e in self.edges)),
        )

    @property
    def graph_version(self) -> str:
        return f"{self.suite}/v1.0+{self.graph_hash[:12]}"


def mine(
    *,
    suite: str,
    task_key: str,
    question: str,
    pool: TracePool,
    candidates: Sequence[Candidate],
    entail: EntailFn,
    documents: Sequence[Mapping[str, str]] = (),
    mechanical_pairs: Sequence[tuple[str, str]] = (),
    probe: Callable[[frozenset[str], int], set[str]] | None = None,
    ablations: Mapping[str, str] | None = None,
    theta: float = 0.85,
    nli_pin: str = NLI_PIN_DEFAULT,
    discoverability_threshold: float = 0.60,
) -> MinedGraph:
    g = MinedGraph(suite=suite, task_key=task_key, theta=theta, nli_pin=nli_pin)

    ok, why = pool_is_admissible(pool)
    if not ok:
        g.admissible, g.reason = False, why
        g.diagnostics["pool"] = {"n_success": pool.n_success, "n_failure": pool.n_failure}
        return g

    # S2 canon -> S3 aggregate
    clusters = cluster(candidates, entail, theta)
    g.nodes = aggregate(clusters, pool, suite=suite, task=task_key, theta=theta, nli_pin=nli_pin)
    promoted = [n for n in g.nodes if n.promoted]
    node_ev = {n.node_id: frozenset(n.ev_uids) for n in promoted}
    node_text = {n.node_id: n.text for n in promoted}

    # S4 ablation verdicts are supplied by the caller (they cost real generations)
    verdicts = dict(ablations or {})

    # S5 edges: mechanical first, then screened + intervened
    g.edges = list(mechanical_edges(mechanical_pairs))
    if probe is not None:
        for stat in screen_order(pool.successes, node_ev):
            edge = counterfactual_edge_test(stat, node_ev, probe)
            if edge is not None:
                g.edges.append(edge)

    # S6 seeds -> depth -> facets
    g.seeds = seed_set(question, node_text)
    seed_ids = [v.node_id for v in g.seeds if v.is_seed]
    from pi_eval.gold import GoldEdge, compute_depths, orphan_rate

    gold_edges = [
        GoldEdge(
            gold_suite=suite,
            gold_task_key=task_key,
            gold_src_node_id=e.src,
            gold_dst_node_id=e.dst,
            gold_edge_kind=e.edge_kind,
            gold_verified=e.verified,
        )
        for e in g.edges
    ]
    g.depths = compute_depths(list(node_text), gold_edges, seed_ids)
    g.facets = components(
        list(node_text), [(e.src, e.dst) for e in g.edges], seed_ids, task_key=task_key
    )

    # S7 partition + discoverability
    g.partitions = partition(
        [
            {
                "node_id": n.node_id,
                "text": n.text,
                "ablation_verdict": verdicts.get(n.node_id, "UNTESTABLE"),
            }
            for n in promoted
        ],
        list(documents),
        threshold=discoverability_threshold,
    )

    priv = sum(1 for p in g.partitions if p.discoverability == "user_private")
    g.diagnostics = {
        "n_candidates": len(g.nodes),
        "n_promoted": len(promoted),
        "promotion_rate": len(promoted) / len(g.nodes) if g.nodes else float("nan"),
        "orphan_rate": orphan_rate(g.depths),
        "user_private_share": priv / len(g.partitions) if g.partitions else float("nan"),
        "partition_elasticity": partition_elasticity(
            [{"node_id": n.node_id, "text": n.text} for n in promoted], list(documents)
        ),
        "theta_sweep": theta_sweep(candidates, entail, (0.60, 0.75, 0.85, 0.95)),
        "seed_basis": seed_basis_shares(g.seeds),
        "n_mechanical_edges": sum(1 for e in g.edges if e.verified == "mechanical"),
        "n_intervention_edges": sum(1 for e in g.edges if e.verified == "intervention"),
    }
    return g


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    passed: bool
    value: float
    threshold: float
    consequence: str


def gate_report(
    graphs: Sequence[MinedGraph],
    *,
    node_recall: float | None = None,
    edge_precision: float | None = None,
    matcher_kappa: float | None = None,
    contamination_delta: float | None = None,
    headline_effect: float | None = None,
    max_orphan_rate: float = 0.40,
) -> list[Gate]:
    """The preregistered admissibility gates. Each carries its WRITTEN CONSEQUENCE, because a
    gate whose failure has no stated consequence is a decoration."""
    gates: list[Gate] = []

    orph = [g.diagnostics.get("orphan_rate", 0.0) for g in graphs if g.admissible]
    mean_orph = sum(float(o) for o in orph) / len(orph) if orph else 0.0
    gates.append(
        Gate(
            "orphan_rate",
            mean_orph <= max_orphan_rate,
            mean_orph,
            max_orphan_rate,
            "Above the ceiling the depth axis is mostly undefined: report C@d for reachable nodes "
            "only and state the orphan share beside every depth claim.",
        )
    )

    if node_recall is not None:
        gates.append(
            Gate(
                "G-M1 node recall vs human gold",
                node_recall >= 0.75,
                node_recall,
                0.75,
                "Below 0.75 the miner is not a usable node instrument: mined graphs become "
                "DIAGNOSTIC-ONLY and may not carry a secondary endpoint.",
            )
        )
    if edge_precision is not None:
        gates.append(
            Gate(
                "G-M1 edge precision vs human gold",
                edge_precision >= 0.80,
                edge_precision,
                0.80,
                "Below 0.80 no depth claim may rest on mined edges; only mechanical edges count.",
            )
        )
    if matcher_kappa is not None:
        gates.append(
            Gate(
                "G-M2 matcher kappa",
                matcher_kappa >= 0.70,
                matcher_kappa,
                0.70,
                "Below 0.70 no matcher-dependent metric may carry a secondary endpoint.",
            )
        )
    if contamination_delta is not None and headline_effect:
        limit = 0.5 * abs(headline_effect)
        gates.append(
            Gate(
                "G-M3 contamination delta",
                abs(contamination_delta) <= limit,
                abs(contamination_delta),
                limit,
                "Above half the headline effect, report headline numbers on leave-one-family-out "
                "graphs ONLY and put the delta in the abstract.",
            )
        )
    return gates


def gates_pass(gates: Sequence[Gate]) -> bool:
    return all(g.passed for g in gates)
