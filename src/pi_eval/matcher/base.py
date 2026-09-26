"""Deciding whether a trajectory 'discovered' a need is itself a measurement instrument.

Three distinct events are conflated by a naive RNR and must be kept apart:
  ASK      the question mentions the need
  RESOLVE  evidence entailing the need's value was actually retrieved
  USE      the need materially changed the final answer

Only USE supports the thesis; ASK is the cheapest to game. The identity
RNR_ask >= RNR_resolve >= RNR_use is compiled into the metric code as an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

MatchKind = Literal["none", "ask", "resolve", "use"]
_ORDER: dict[MatchKind, int] = {"none": 0, "ask": 1, "resolve": 2, "use": 3}


@dataclass(frozen=True, slots=True)
class MatchRecord:
    run_id: str
    suite_id: str
    task_id: str
    node_id: str
    match_kind: MatchKind
    matched_turn_idx: int | None
    matcher_id: str
    matcher_family: str
    matcher_score: float
    threshold: float
    graph_version: str
    cited_uids: tuple[str, ...] = ()

    @property
    def rank(self) -> int:
        return _ORDER[self.match_kind]


class Matcher(Protocol):
    matcher_id: str
    matcher_family: str

    def match(self, *, graph, trajectory, run_id: str) -> Sequence[MatchRecord]: ...


class MechanicalMatcher:
    """Zero-LLM matcher: a node is RESOLVED iff its gold evidence uids were retrieved, and
    USED iff those uids are also cited by the final answer.

    On the synthetic suite and on MuSiQue (where gold evidence is a paragraph index) this is
    exact, which is precisely why those suites are the calibration sets for the LLM matchers
    that the open-ended suites require.
    """

    # v3: the ASK rung became PER-TURN and learned to resolve MuSiQue's `#N` references;
    # RESOLVE now reports the turn a need became fully resolved rather than the turn its
    # first fragment arrived. Bumped because matcher_id rides into matcher_hash ->
    # scorer_hash, so rows scored under v2 can never be pooled with rows scored under v3.
    matcher_id = "mechanical_v3"
    matcher_family = "rule"

    def match(self, *, graph, trajectory, run_id: str) -> list[MatchRecord]:
        retrieved: dict[str, int] = {}
        for t in trajectory.turns:
            for uid in t.retrieved_uids:
                retrieved.setdefault(uid, t.turn_idx)
        answer = trajectory.outcome.answer
        cited = set(answer.cited_unit_ids) if answer else set()

        # IS THE CITATION SET INFORMATIVE? Both shipped Answerers set
        # `cited_unit_ids=tuple(u.uid for u in ev.units)` -- they cite EVERY retrieved unit,
        # unconditionally -- so `ev <= cited` holds whenever `ev <= retrieved` and `kind` is
        # always "use", never "resolve". The top rung of the ladder collapsed into the middle
        # one, exactly as the ASK rung had collapsed into it from below. Confirmed on live
        # musique rollouts: every matched need came back "use".
        #
        # A citation list that names everything carries no information about USE, so USE is not
        # measured and must not be claimed. The record is capped at RESOLVE and says why. When
        # an Answerer cites a strict subset -- attributing rather than listing -- the rung
        # becomes real again with no further change here.
        informative = bool(cited) and not set(retrieved) <= cited
        # PER TURN, NOT CONCATENATED. `asked_text` used to be every question in the
        # trajectory joined into one string, which had two consequences: `matched_turn_idx`
        # was None for every ASK match (288 of 288 on disk), so the rung could not say WHEN a
        # need was named; and two questions could jointly satisfy a need neither one asked
        # about. Matching each question alone fixes both, and is strictly more conservative.
        asked = [(t.turn_idx, getattr(t.action, "text", "").upper()) for t in trajectory.turns]

        out: list[MatchRecord] = []
        for node in graph.gold_nodes:
            ev = set(node.gold_ev_uids)
            kind: MatchKind = "none"
            turn = None
            if ev and ev <= set(retrieved):
                # MAX, NOT MIN. A need is resolved when ALL of its evidence is in; `min` is
                # the turn the first fragment arrived, which is not resolution. `min`
                # systematically understated discovery time for multi-uid needs -- which are
                # disproportionately the deep ones -- and it feeds precedence_violation_rate,
                # whose whole question is which of two needs was resolved first.
                turn = max(retrieved[u] for u in ev)
                kind = "use" if (informative and ev <= cited) else "resolve"
            else:
                core, terms = _terms(node.gold_text or ""), resolved_terms(node, graph)
                hit = next((i for i, text in asked if _matches(node, core, terms, text)), None)
                if hit is not None:
                    kind, turn = "ask", hit
            out.append(
                MatchRecord(
                    run_id=run_id,
                    suite_id=graph.gold_suite,
                    task_id=graph.gold_task_key,
                    node_id=node.gold_node_id,
                    match_kind=kind,
                    matched_turn_idx=turn,
                    matcher_id=self.matcher_id,
                    matcher_family=self.matcher_family,
                    matcher_score=1.0 if kind != "none" else 0.0,
                    threshold=1.0,
                    graph_version=node.gold_graph_version,
                    cited_uids=tuple(sorted(ev & cited)),
                )
            )
        return out


# THE ASK RUNG, AND WHY IT NEEDED A SECOND TEST.
#
# `_tokens` finds the SYNTHETIC suite's opaque ids (K1DEC49FC). On synth that is exactly right:
# the need text IS the token, so an exact match is unambiguous and ungameable. On every real
# suite a need is natural language -- "The Collegian >> owned by" -- and the regex finds
# nothing. Measured: it matches 0 of 59,986 non-synth gold nodes, so the `elif` never fired,
# `kind` was never "ask", and RNR_ask == RNR_resolve == RNR_use on every suite that carries a
# real corpus. The ladder this module's docstring is built around had ONE rung, and the
# identity RNR_ask >= RNR_resolve >= RNR_use held vacuously.
#
# ASK is "the cheapest to game", so the replacement is deliberately conservative: it fires only
# when at least half of a need's DISTINCTIVE terms appear in what was asked, and never on a
# single shared word. The threshold is a preregisterable instrument choice, so it is a named
# constant and it rides into `matcher_hash` -> `scorer_hash`; the matcher_id is bumped to v2 so
# rows scored under the old instrument can never be pooled with rows scored under this one.
ASK_MIN_COVERAGE = 0.5
ASK_MIN_TERMS = 2
_MIN_TERM_LEN = 4

# The stoplist lives in `pi_eval.text`, shared with `mining.partition`. Both instruments ask
# "which words carry the need and which are scaffolding?", and two stoplists for one question
# is the shape this repository keeps finding bugs in.


def _tokens(node) -> list[str]:
    """The synthetic suite's opaque ids. Exact, ungameable, and absent from every real suite."""
    import re

    return re.findall(r"\b[KD][0-9A-F]{8}\b", node.gold_text.upper())


def _terms(text: str) -> set[str]:
    """Distinctive content terms of a natural-language need."""
    from pi_eval.text import content_terms

    return {w.upper() for w in content_terms(text, min_len=_MIN_TERM_LEN)}


def resolved_terms(node, graph) -> set[str]:
    """A need's distinctive terms, with MuSiQue's `#N` references resolved first.

    `"When was #1 founded?"` reduces to {FOUNDED}: `#1` names hop 1's ANSWER and `min_len=4`
    drops the digit. MEASURED over all 2,660 musique gold nodes, the share of needs left with
    fewer than ASK_MIN_TERMS distinctive terms is 2.3% at depth 0 and 24.5% at depth 1 -- so
    one latent need in four fell into the degenerate branch where a SINGLE shared word
    ("founded") decides the match. The instrument was weakest exactly on the population the
    vertical-proactivity claim is about.

    `pi_run.cmd_gold._resolve_placeholders` already performs this substitution for the ceiling
    arms, with the same reasoning ("left unresolved, that string retrieves nothing useful").
    Reimplemented here rather than imported because `pi_eval` may not import `pi_run`; the
    convention -- `#N` indexes the graph's own prerequisite-topological order, 1-based -- is
    the one that function established and `tests/test_matcher_latent.py` pins.
    """
    text = node.gold_text or ""
    if "#" in text:
        text = _resolve_refs(text, graph)
    return _terms(text)


def _resolve_refs(text: str, graph) -> str:
    """`#N` -> the alias of the Nth node in prerequisite-topological order."""
    import re

    aliases = [next((a for a in n.gold_aliases if a and a.strip()), "") for n in _ordered(graph)]

    def sub(m):
        i = int(m.group(1)) - 1
        return aliases[i] if 0 <= i < len(aliases) and aliases[i] else m.group(0)

    return re.sub(r"#(\d+)", sub, text)


def _ordered(graph) -> list:
    """Nodes in prerequisite-topological order; ties and cycles fall back to graph order."""
    nodes = list(graph.gold_nodes)
    parents: dict[str, set[str]] = {n.gold_node_id: set() for n in nodes}
    for e in graph.gold_edges:
        if getattr(e, "gold_edge_kind", "") == "prerequisite":
            dst = e.gold_dst_node_id
            if dst in parents:
                parents[dst].add(e.gold_src_node_id)
    out, placed = [], set()
    remaining = list(nodes)
    while remaining:
        ready = [n for n in remaining if parents[n.gold_node_id] <= placed]
        if not ready:  # a cycle: emit what is left in graph order rather than looping
            out.extend(remaining)
            break
        out.extend(ready)
        placed |= {n.gold_node_id for n in ready}
        remaining = [n for n in remaining if n.gold_node_id not in placed]
    return out


def _matches(node, core: set[str], terms: set[str], asked_text: str) -> bool:
    """The ASK test against ONE question. `core` is the need's OWN terms, `terms` the resolved.

    RESOLUTION ADDS DISCRIMINATION, IT DOES NOT REPLACE IT. Substituting `#1` injects the
    PARENT's identity into the child's term set, so without a required core a question about
    the parent alone -- "tell me about Houston Baptist University" -- hits 3 of the 4 resolved
    terms of "When was #1 founded?" and is credited with asking about the founding date it
    never mentions. Requiring the node's own terms too means resolution can only ever make the
    test stricter: the child is credited when the question names BOTH the need ("founded") and
    enough of the referent, which is exactly what discovering a latent need looks like.

    For a node with no placeholder, `core == terms` and this is v2's rule unchanged.
    """
    ids = _tokens(node)
    if ids:
        return any(tok in asked_text for tok in ids)
    if not terms:
        return False
    if core and not all(t in asked_text for t in core):
        return False
    hit = {t for t in terms if t in asked_text}
    if len(terms) < ASK_MIN_TERMS:
        return len(hit) == len(terms)
    return len(hit) >= ASK_MIN_TERMS and len(hit) / len(terms) >= ASK_MIN_COVERAGE


def asked_about(node, asked_text: str) -> bool:
    """Did the policy's questions MENTION this need?

    Two instruments, one per kind of need, and neither is a fallback for the other:
      * a need whose text carries synth ids matches on those ids, exactly;
      * a natural-language need matches when at least ASK_MIN_COVERAGE of its distinctive
        terms appear, and never on fewer than ASK_MIN_TERMS of them.

    The floor on the COUNT is what stops one shared word ("credit", "account") from crediting
    an ask across an entire suite; the floor on the SHARE is what stops a long need being
    credited by two incidental words.
    """
    ids = _tokens(node)
    if ids:
        return any(tok in asked_text for tok in ids)
    terms = _terms(node.gold_text)
    if not terms:
        return False
    hit = {t for t in terms if t in asked_text}
    if len(terms) < ASK_MIN_TERMS:
        return len(hit) == len(terms)
    return len(hit) >= ASK_MIN_TERMS and len(hit) / len(terms) >= ASK_MIN_COVERAGE
