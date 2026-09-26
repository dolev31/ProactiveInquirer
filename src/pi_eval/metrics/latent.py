"""Latent-need discovery: did the policy ask about needs that only became visible?

THE CLAIM THIS MEASURES. The paper's thesis is vertical proactivity -- asking about a need
the task does not name, one that becomes visible only once earlier evidence has been read.
Every existing metric here is depth-STRATIFIED (`cad#d`) or depth-WEIGHTED (`dwr`); none of
them asks whether the policy took a need at the moment that need appeared, which is the
behaviour the thesis is about.

THE LABEL IS FREE AND ALREADY ON DISK. MEASURED over all 800 musique gold graphs: 1,464 of
2,660 required nodes (55%) are gated behind a prerequisite edge, at depths
{0:1196, 1:800, 2:530, 3:134}. Those edges are human-authored -- MuSiQue's `#N` placeholders
in `question_decomposition` -- and are stamped `mechanical` at confidence 1.0. Nothing here
needs an annotator, a judge or a model.

TWO READINGS, BOTH EMITTED, because they answer different questions:

  is_latent        the need sits at gold_depth >= 1. A property of the GRAPH -- a position
                   in an authored decomposition, and NOT the same predicate as "the task did
                   not name it". MEASURED 2026-09-04: of 99 depth-3 items, three independent
                   raters UNANIMOUSLY judged 21 to be askable from the task statement alone,
                   because musique's 4-hop questions refer to their intermediates
                   descriptively ("the city where The Killers formed"). Use
                   `nameable_from_task` for the visibility claim; this field is depth.
  newly_reachable  the need's last prerequisite resolved on the PREVIOUS turn, so it became
                   askable exactly now. A property of the TRAJECTORY, and the operational
                   reading of "asked about a need that had just become visible".

A policy can score well on the first by grinding through a chain in any order it likes; the
second is the one that says it was following the structure. Reporting only the first would
let "reached depth 2 eventually" pass as anticipation.

THE DENOMINATOR IS WHAT WAS REACHABLE, NOT WHAT EXISTS. A latent need whose prerequisite the
policy never resolved was never askable, and counting it as a miss makes the rate a function
of how far the policy got rather than of what it did with what it had. `n_latent_available`
is therefore the set of depth>=1 required needs that ever entered the frontier.

A task with no latent needs returns NaN, not 0.0. A flat graph cannot exhibit vertical
proactivity in either direction, and a zero there would read as "the policy found no latent
needs" on a task that has none -- the same absent-is-not-zero rule `tau_reward` and `adr`
already follow.

Judge-free throughout, so nothing here sits behind the sigma_J noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

RESOLVE_RANK = 2


@dataclass(frozen=True, slots=True)
class TurnLatent:
    """What one decision point did, in latent-need terms."""

    turn_idx: int
    latent_depth: int  # deepest gold node resolved at this turn; -1 if none
    is_latent: bool  # that node sits behind a prerequisite
    newly_reachable: bool  # ...and it had just become askable
    frontier_size: int  # how many required needs were askable at this turn
    # THE SENTINEL IS NOT A DEPTH. `latent_depth == -1` means no required node was resolved
    # here at all, and folding that into `is_latent=False` made it indistinguishable from a
    # genuine depth-0 root -- 54.1% of SFT rows carry -1. "No identified need" is not "the
    # need was stated in the task", and a consumer that cannot tell them apart is training on
    # a category error.
    has_gold_node: bool = False
    # Did the TASK TEXT name this need? None when no task text was supplied, because a caller
    # that passed none cannot distinguish "not nameable" from "not measured", and defaulting
    # to False would assert the stronger claim on every existing caller.
    nameable_from_task: bool | None = None


@dataclass(frozen=True, slots=True)
class LatentLabels:
    per_turn: dict[int, TurnLatent]
    n_latent_available: int
    n_latent_resolved: int

    @property
    def latent_discovery_rate(self) -> float:
        """Of the latent needs that ever became askable, the share the policy resolved."""
        if self.n_latent_available <= 0:
            return float("nan")
        return self.n_latent_resolved / self.n_latent_available


class _Labels(dict):
    """`per_turn` for a turn nothing was recorded at: a real absence, not a KeyError."""

    def __init__(self, data, frontier_at):
        super().__init__(data)
        self._frontier_at = frontier_at

    def __missing__(self, t: int) -> TurnLatent:
        return TurnLatent(
            turn_idx=int(t),
            latent_depth=-1,
            is_latent=False,
            newly_reachable=False,
            frontier_size=self._frontier_at(int(t)),
            has_gold_node=False,
        )


def _names_it(task_text: str, node) -> bool:
    """Does the task statement already name this need? Surface-form containment, normalised.

    A CONSERVATIVE FLOOR, NOT THE MEASURE. It fires on 447 of 43,609 exported rows (1.0%),
    and on 196 of 14,005 `is_latent` rows (1.4%) -- against three A4 raters who unanimously
    judged 21 of 99 depth-3 items nameable, roughly 21%. The two numbers are an order of
    magnitude apart and the raters are closer to right.

    The reason is that musique's `gold_text` is not a descriptive phrase. It is a relation
    triple (`"The Collegian >> owned by"`) or a sub-question carrying `#N` placeholders
    (`"When was #1 founded?"`), while the composed task refers to the same need in prose
    ("the city where The Killers formed"). Containment between those two forms almost never
    holds. VERIFIED on the Killers item: this returns True for the descriptive phrase and
    False for musique's actual sub-question form.

    Doing it properly needs the composition structure `musique_build` already has -- the task
    text is BUILT by substituting each sub-question's description, so the mapping from node to
    descriptive phrase exists upstream and is thrown away. Until that is wired, read a True
    here as "certainly nameable" and a False as "not established", never as "latent".

    False negatives are the safe direction: they leave a need looking latent, which is the
    existing behaviour, so nothing regresses on a miss.
    """
    hay = " ".join((task_text or "").lower().split())
    if not hay:
        return False
    forms = [getattr(node, "gold_text", "") or "", *(getattr(node, "gold_aliases", ()) or ())]
    for f in forms:
        f = " ".join(str(f).lower().split())
        # Strip the interrogative shell a gold_text often carries ("what is X?" -> "x").
        f = f.removeprefix("what is ").removeprefix("who is ").removeprefix("where is ")
        f = f.rstrip("?").strip()
        if len(f) >= 4 and f in hay:
            return True
    return False


def latent_labels(
    graph,
    records: Iterable,
    *,
    turn_idxs: Sequence[int] = (),
    task_text: str = "",
) -> LatentLabels:
    """Per-turn latent labels and the run-level discovery rate.

    `records` are `MatchRecord`s for this run; only RESOLVE or better counts, because ASK is
    "the cheapest to game" and a need the policy named but never retrieved was not discovered.
    """
    required = {n.gold_node_id: n for n in graph.gold_nodes if n.gold_partition == "required"}

    # THE FRONTIER IS REQUIRED-ONLY, SO ITS GATES MUST BE TOO. `done` can only ever hold
    # required ids (`resolved_at` filters on `required` below), so a required need whose
    # prerequisite was an OPTIONAL node could never satisfy `parents[nid] <= done`: it never
    # entered the frontier, never counted in `n_latent_available`, and could never be
    # `newly_reachable` -- the policy did what the thesis asks for and scored nothing for it.
    # MEASURED on data/gold v1: 720 of 11,807 gated required nodes on wiki2 and 11 of 1,770 on
    # strategyqa sit behind an optional prerequisite; 0 on musique, whose builder stamps every
    # node required.
    #
    # "Let optional nodes into `done` when they resolve" fixes nothing. A node is optional
    # precisely because it carries no evidence -- wiki2_build and strategyqa_build stamp
    # `required` iff `gold_ev_uids` is non-empty, and 1,022/1,022 and 1,247/1,247 optional
    # nodes on disk have none -- and `MechanicalMatcher.match` reaches RESOLVE only when `ev`
    # is non-empty. An optional node never resolves.
    #
    # Dropping the edge is wrong the other way: the child becomes a root, askable at t=0
    # before its real ancestor is in, charged as available on a run that never got there, and
    # never "newly" anything. The reading that is consistent with depth is `compute_depths`',
    # which every builder runs over EVERY node (`finalize_graph`), so a required node's depth
    # is computed THROUGH its optional parents: a link, not a wall. The frontier therefore
    # walks through optional nodes to the nearest REQUIRED ancestors and gates on those. A
    # chain with no required ancestor at all is, from the required-only view, a root -- which
    # on wiki2 is all 720 (depth-1 needs behind an evidence-free optional seed): they enter
    # the denominator from t=0 and are never `newly_reachable`. On strategyqa 8 of the 11
    # regain one or two required ancestors and are gated on them.
    prereq: dict[str, set[str]] = {}
    for e in graph.gold_edges:
        if getattr(e, "gold_edge_kind", "") == "prerequisite":
            prereq.setdefault(e.gold_dst_node_id, set()).add(e.gold_src_node_id)

    def required_ancestors(nid: str) -> set[str]:
        """Nearest required nodes above `nid`, looking through optional ones."""
        out: set[str] = set()
        seen: set[str] = set()
        stack = list(prereq.get(nid, ()))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            if p in required:
                out.add(p)
            else:
                stack.extend(prereq.get(p, ()))
        return out

    parents: dict[str, set[str]] = {nid: required_ancestors(nid) for nid in required}

    resolved_at: dict[str, int] = {}
    for r in records:
        if r.node_id in required and r.rank >= RESOLVE_RANK and r.matched_turn_idx is not None:
            t = int(r.matched_turn_idx)
            prev = resolved_at.get(r.node_id)
            if prev is None or t < prev:
                resolved_at[r.node_id] = t

    def resolved_before(t: int) -> set[str]:
        return {nid for nid, at in resolved_at.items() if at < t}

    def frontier(t: int) -> set[str]:
        """Required needs askable at turn t: unresolved, every required ancestor already in."""
        done = resolved_before(t)
        return {nid for nid in required if nid not in done and parents[nid] <= done}

    def is_latent_node(nid: str) -> bool:
        d = required[nid].gold_depth
        return d is not None and d > 0

    turns = sorted({*turn_idxs, *resolved_at.values()})
    by_turn: dict[int, TurnLatent] = {}
    for t in turns:
        here = [nid for nid, at in resolved_at.items() if at == t]
        depths = [required[nid].gold_depth for nid in here if required[nid].gold_depth is not None]
        deepest = max(depths) if depths else -1
        # NEWLY REACHABLE: on the frontier now and NOT on it a turn ago, i.e. the last
        # prerequisite landed on the previous turn. A root is on the frontier from t=0, so it
        # is never "newly" anything -- which is correct: nothing made it appear.
        appeared = frontier(t) - (frontier(t - 1) if t > 0 else set())
        by_turn[t] = TurnLatent(
            turn_idx=t,
            latent_depth=deepest,
            is_latent=deepest > 0,
            newly_reachable=any(nid in appeared and is_latent_node(nid) for nid in here),
            frontier_size=len(frontier(t)),
            has_gold_node=bool(here),
            nameable_from_task=(
                any(_names_it(task_text, required[nid]) for nid in here) if task_text else None
            ),
        )

    # AVAILABLE = ever entered the frontier. A need whose prerequisite was never resolved was
    # never askable; charging the policy for it charges it for how far it got.
    horizon = (max(turns) + 2) if turns else 1
    ever: set[str] = set()
    for t in range(horizon):
        ever |= frontier(t)
    latent_ever = {nid for nid in ever if is_latent_node(nid)}
    return LatentLabels(
        per_turn=_Labels(by_turn, lambda t: len(frontier(t))),
        n_latent_available=len(latent_ever),
        n_latent_resolved=len({nid for nid in resolved_at if nid in latent_ever}),
    )
