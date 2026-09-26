"""The ASK rung, measured on the needs the paper is actually about: the LATENT ones.

`mechanical_v2` fixed the ASK rung for natural language. It is still systematically weakest
exactly at depth >= 1, and it cannot say WHEN a need was asked about. Both matter here,
because a latent-need dataset is labelled off this instrument.

MEASURED over all 2,660 musique gold nodes:

    depth   n     mean distinctive terms   share with < ASK_MIN_TERMS
      0    1196          3.86                     2.3 %
      1     800          2.77                    24.5 %
      2     530          3.20                    24.3 %
      3     134          2.63                    21.6 %

The cause is MuSiQue's own notation. `"When was #1 founded?"` reduces to {FOUNDED}: `#1` is a
reference to hop 1's ANSWER, and `min_len=4` drops the digit. One in four latent needs
therefore lands in the degenerate `len(terms) < ASK_MIN_TERMS` branch, where a SINGLE shared
word decides the match -- and "founded" is shared by most of the corpus.

`pi_run.cmd_gold._resolve_placeholders` already substitutes `#N` with the prior hop's alias,
for the ceiling arms, for exactly this reason: left unresolved the string "retrieves nothing
useful". The matcher needs the same substitution before it counts terms.

Two further defects, both fatal to a per-turn label:

  * `asked_text` is the concatenation of EVERY question in the trajectory, so `matched_turn_idx`
    is None for every ASK match (288 of 288 on disk). The rung has no turn attribution at all.
  * RESOLVE takes `min` over the node's evidence uids -- the turn the EARLIEST piece arrived.
    A node is resolved when ALL of its evidence is in, so `max` is the resolution turn. `min`
    systematically understates discovery time for multi-uid needs, which are disproportionately
    the deep ones, and it feeds `precedence_violation_rate` directly.

Fixing the instrument bumps `matcher_id` to `mechanical_v3` -> `matcher_hash` -> `scorer_hash`,
so rows scored under v2 can never pool with rows scored under v3.
"""

from __future__ import annotations

from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_eval.matcher.base import ASK_MIN_TERMS, MechanicalMatcher, resolved_terms


def _node(nid, text, uids=(), aliases=()):
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t0",
        gold_node_id=nid,
        gold_text=text,
        gold_partition="required",
        gold_ev_uids=tuple(uids),
        gold_aliases=tuple(aliases),
    )


def _edge(src, dst):
    return GoldEdge(
        gold_suite="musique",
        gold_task_key="t0",
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind="prerequisite",
    )


class _Turn:
    def __init__(self, i, question, uids=()):
        self.turn_idx = i
        self.action = type("A", (), {"text": question})()
        self.retrieved_uids = tuple(uids)


class _Traj:
    def __init__(self, turns, cited=()):
        self.turns = turns
        answer = type("A", (), {"cited_unit_ids": tuple(cited)})()
        self.outcome = type("O", (), {"answer": answer})()


# The real shape, from runs/b45efc9b.../ and data/gold/graphs/musique/v1.jsonl.
S1 = _node("s1", "The Collegian >> owned by", uids=("u1",), aliases=("Houston Baptist University",))
S2 = _node("s2", "When was #1 founded?", uids=("u2",), aliases=("1960",))
GRAPH = GoldGraph(
    gold_suite="musique",
    gold_task_key="t0",
    gold_nodes=(S1, S2),
    gold_edges=(_edge("s1", "s2"),),
)


def _records(turns, cited=(), graph=GRAPH):
    return {
        r.node_id: r
        for r in MechanicalMatcher().match(graph=graph, trajectory=_Traj(turns, cited), run_id="r0")
    }


# ------------------------------------------------------------- the placeholder


def test_a_latent_need_is_degenerate_until_its_placeholder_is_resolved() -> None:
    """`"When was #1 founded?"` carries ONE distinctive term. That is the whole defect."""
    from pi_eval.matcher.base import _terms

    assert len(_terms(S2.gold_text)) < ASK_MIN_TERMS


def test_resolution_restores_the_parent_identity() -> None:
    """After substitution the need carries the terms a policy would have had to discover."""
    terms = resolved_terms(S2, GRAPH)
    assert "FOUNDED" in terms
    assert {"HOUSTON", "BAPTIST", "UNIVERSITY"} <= terms
    assert len(terms) >= ASK_MIN_TERMS, "no longer degenerate"


def test_a_node_without_a_placeholder_is_untouched() -> None:
    from pi_eval.matcher.base import _terms

    assert resolved_terms(S1, GRAPH) == _terms(S1.gold_text)


def test_one_shared_word_no_longer_credits_a_latent_ask() -> None:
    """The degenerate branch let "founded" alone match. It must not.

    This is the failure the depth-1 population is exposed to: 24.5% of those needs had
    fewer than two terms, so a single common word decided the label.
    """
    got = _records([_Turn(0, "When was it founded?")])
    assert got["s2"].match_kind == "none"


def test_a_genuine_latent_ask_still_matches() -> None:
    """The policy that actually resolved #1 and asked about it must be credited."""
    got = _records([_Turn(0, "When was Houston Baptist University founded?")])
    assert got["s2"].match_kind == "ask"


# ------------------------------------------------------------- turn attribution


def test_an_ask_match_carries_the_turn_it_was_asked_at() -> None:
    """288 of 288 ASK matches on disk have matched_turn_idx None. A per-turn label needs it."""
    got = _records(
        [
            _Turn(0, "something unrelated entirely"),
            _Turn(1, "who owned The Collegian"),
        ]
    )
    assert got["s1"].match_kind == "ask"
    assert got["s1"].matched_turn_idx == 1


def test_the_earliest_matching_turn_wins() -> None:
    """Asked twice, the label is the turn the policy first named the need."""
    got = _records(
        [
            _Turn(0, "who owned The Collegian"),
            _Turn(1, "who owned The Collegian, again"),
        ]
    )
    assert got["s1"].matched_turn_idx == 0


def test_a_question_is_matched_alone_not_concatenated() -> None:
    """Concatenation lets two questions jointly satisfy a need neither one asked about.

    "Houston Baptist" in turn 0 and "founded" in turn 1 must not add up to an ask for s2.
    """
    got = _records(
        [
            _Turn(0, "tell me about Houston Baptist University"),
            _Turn(1, "when was it founded"),
        ]
    )
    assert got["s2"].match_kind == "none"


# ------------------------------------------------------------------- resolution


def test_resolve_is_the_turn_the_need_became_resolved_not_the_first_fragment() -> None:
    """A node is resolved when ALL its evidence is in, so the turn is max, not min."""
    node = _node("m", "multi uid need here", uids=("a", "b"))
    g = GoldGraph(gold_suite="musique", gold_task_key="t0", gold_nodes=(node,))
    got = _records([_Turn(0, "q", uids=("a",)), _Turn(5, "q", uids=("b",))], graph=g)
    assert got["m"].match_kind == "resolve"
    assert got["m"].matched_turn_idx == 5


# ---------------------------------------------------------------- the instrument


def test_the_matcher_id_is_bumped() -> None:
    """v2 and v3 rows must never pool: matcher_id rides into matcher_hash -> scorer_hash."""
    assert MechanicalMatcher.matcher_id == "mechanical_v3"


def test_the_synth_token_instrument_is_unchanged() -> None:
    """Exact id matching on the calibration suite must not move."""
    from pi_eval.matcher.base import asked_about

    n = _node("n", "the value for facet 0 is K1DEC49FC")
    assert asked_about(n, "WHAT IS K1DEC49FC")
    assert not asked_about(n, "WHAT IS SOMETHING ELSE")
