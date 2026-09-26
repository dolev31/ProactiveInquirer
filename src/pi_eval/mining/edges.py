"""S5 — edge induction. The stage where it is easiest to be wrong and hardest to notice.

The temptation is to read a prerequisite edge off temporal order: v_i is resolved before
v_j in most traces, therefore v_i enables v_j. That is a co-occurrence, and it is satisfied
by any pair of needs whose documents happen to rank differently under BM25.

So order NEVER establishes an edge here. It can only VETO one:

  1. OBSERVATIONAL SCREEN (cheap, runs on every pair). gamma = P(v_i resolved before v_j |
     both resolved). A pair failing the screen is discarded. A pair PASSING it is still only
     a hypothesis. Additionally, if >= `veto_k` traces resolved v_j WITHOUT ever resolving
     v_i, then v_i is demonstrably not a prerequisite for v_j, and the pair is vetoed
     outright no matter how strong gamma is. That veto is the workhorse.

  2. INTERVENTION (expensive, runs only on survivors). Deny v_i's evidence to a probe policy
     and measure whether v_j still gets resolved. An edge is `prerequisite/intervention`
     only if the resolve rate collapses (>= hi under control, <= lo under denial). This is a
     counterfactual, which is the only thing that licenses the word "prerequisite".

The two edge kinds are kept apart because they mean different things: `prerequisite` is
"unanswerable until", `relevance` is "not worth asking until". Conflating them would make
Depth mean two things at once, so Depth is computed over prerequisites alone by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from .pool import Trace, order_index

EdgeKind = Literal["prerequisite", "relevance"]
Verification = Literal["mechanical", "intervention", "observational", "human"]

# (denied_uids, seed) -> set of node_ids the probe policy resolved
ProbeFn = Callable[[frozenset[str], int], set[str]]


@dataclass(frozen=True, slots=True)
class EdgeStat:
    src: str
    dst: str
    gamma: float
    n_both: int
    n_dst_without_src: int
    passed_screen: bool
    veto_reason: str


@dataclass(frozen=True, slots=True)
class MinedEdge:
    src: str
    dst: str
    edge_kind: EdgeKind
    verified: Verification
    gamma: float
    n_both: int
    n_src_absent: int
    resolve_rate_control: float | None
    resolve_rate_denied: float | None
    confidence: float


def screen_order(
    traces: Sequence[Trace],
    node_ev: dict[str, frozenset[str]],
    *,
    gamma_min: float = 0.8,
    min_both: int = 3,
    veto_k: int = 5,
) -> list[EdgeStat]:
    """Compute gamma and the counter-evidence veto for every ordered node pair."""
    resolved_at: list[dict[str, int]] = []
    for tr in traces:
        idx = order_index(tr)
        first: dict[str, int] = {}
        for node_id, uids in node_ev.items():
            if uids and uids <= set(idx):
                first[node_id] = min(idx[u] for u in uids)
        resolved_at.append(first)

    ids = sorted(node_ev)
    out: list[EdgeStat] = []
    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            both = before = dst_without_src = 0
            for first in resolved_at:
                has_src, has_dst = src in first, dst in first
                if has_dst and not has_src:
                    dst_without_src += 1
                if has_src and has_dst:
                    both += 1
                    if first[src] < first[dst]:
                        before += 1
            gamma = before / both if both else 0.0

            reason = ""
            if dst_without_src >= veto_k:
                reason = f"{dst_without_src} traces resolved dst without src: not a prerequisite"
            elif both < min_both:
                reason = f"only {both} traces resolved both (< {min_both})"
            elif gamma < gamma_min:
                reason = f"gamma {gamma:.2f} < {gamma_min}"

            out.append(EdgeStat(src, dst, gamma, both, dst_without_src, not reason, reason))
    return out


def counterfactual_edge_test(
    stat: EdgeStat,
    node_ev: dict[str, frozenset[str]],
    probe: ProbeFn,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    control_hi: float = 0.8,
    denied_lo: float = 0.2,
    relevance_hi: float = 0.5,
    relevance_lo: float = 0.15,
) -> MinedEdge | None:
    """Deny src's evidence; does dst still get resolved?

    Only a collapse licenses `prerequisite`. A partial drop licenses `relevance` at most,
    and no drop means there was never an edge — which is exactly the outcome the
    observational screen alone would have got wrong.
    """
    if not stat.passed_screen:
        return None

    control = sum(1 for s in seeds if stat.dst in probe(frozenset(), s)) / len(seeds)
    denied_uids = node_ev.get(stat.src, frozenset())
    denied = sum(1 for s in seeds if stat.dst in probe(denied_uids, s)) / len(seeds)

    if control >= control_hi and denied <= denied_lo:
        kind: EdgeKind = "prerequisite"
        verified: Verification = "intervention"
        conf = min(1.0, (control - denied))
    elif control >= relevance_hi and denied <= relevance_lo:
        kind, verified = "relevance", "intervention"
        conf = min(1.0, (control - denied))
    else:
        return None

    return MinedEdge(
        src=stat.src,
        dst=stat.dst,
        edge_kind=kind,
        verified=verified,
        gamma=stat.gamma,
        n_both=stat.n_both,
        n_src_absent=stat.n_dst_without_src,
        resolve_rate_control=control,
        resolve_rate_denied=denied,
        confidence=conf,
    )


def mechanical_edges(pairs: Sequence[tuple[str, str]]) -> list[MinedEdge]:
    """Edges extracted by regex from the environment itself.

    MuSiQue's "#N" placeholders and tau2's document->tool unlocks are causally enforced by
    the benchmark, not inferred by us: you literally cannot invoke `submit_dispute_0589`
    before reading the document that names it. These need no screening and no intervention,
    which is why they are the calibration set for everything above.
    """
    return [
        MinedEdge(
            src=s,
            dst=d,
            edge_kind="prerequisite",
            verified="mechanical",
            gamma=1.0,
            n_both=0,
            n_src_absent=0,
            resolve_rate_control=None,
            resolve_rate_denied=None,
            confidence=1.0,
        )
        for s, d in pairs
    ]


def acyclic(edges: Sequence[MinedEdge]) -> bool:
    """A need graph with a cycle has no well-defined depth, so this is checked before use."""
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.src, []).append(e.dst)
    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {}

    def visit(n: str) -> bool:
        color[n] = GREY
        for m in adj.get(n, ()):
            c = color.get(m, WHITE)
            if c == GREY:
                return False
            if c == WHITE and not visit(m):
                return False
        color[n] = BLACK
        return True

    return all(color.get(n, WHITE) != WHITE or visit(n) for n in list(adj))
