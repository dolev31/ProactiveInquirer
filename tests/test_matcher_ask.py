"""The ASK rung of the RNR ladder, which was unreachable on every suite.

`pi_eval.matcher.base` opens: three distinct events must be kept apart -- ASK (the question
mentions the need), RESOLVE (evidence entailing it was retrieved), USE (it changed the answer)
-- and "the identity RNR_ask >= RNR_resolve >= RNR_use is compiled into the metric code as an
assertion".

The ASK branch tested `re.findall(r"\\b[KD][0-9A-F]{8}\\b", node.gold_text)`. Those opaque ids
are the SYNTHETIC suite's, and they live in the corpus documents and the task QUESTION -- not
in `gold_text`, which reads "the value for facet 0 at step 0 is V000". Measured: the regex
matches 0 nodes on synth, musique, strategyqa, wiki2 AND tau2. So `kind` was never "ask" on any
suite, the ladder had one rung, and the identity held vacuously.
"""

from __future__ import annotations

from pi_eval.gold import GoldGraph, GoldNode
from pi_eval.matcher.base import (
    ASK_MIN_COVERAGE,
    ASK_MIN_TERMS,
    MechanicalMatcher,
    _tokens,
    asked_about,
)


def _node(nid: str, text: str, uids=()):
    return GoldNode(
        gold_suite="musique",
        gold_task_key="t0",
        gold_node_id=nid,
        gold_text=text,
        gold_partition="required",
        gold_ev_uids=tuple(uids),
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


def _match(nodes, turns, cited=()):
    g = GoldGraph(gold_suite="musique", gold_task_key="t0", gold_nodes=tuple(nodes))
    return {
        r.node_id: r.match_kind
        for r in MechanicalMatcher().match(graph=g, trajectory=_Traj(turns, cited), run_id="r0")
    }


def test_the_old_instrument_matched_nothing_on_any_suite():
    """Including synth, whose ids it was written for: they are in the question, not the need."""
    assert _tokens(_node("a", "the value for facet 0 at step 0 is V000")) == []
    assert _tokens(_node("b", "The Collegian >> owned by")) == []
    # It still works where the ids really are in the need text, which is what it is kept for.
    assert _tokens(_node("c", "need K1DEC49FC then D0A1B2C3D")) == ["K1DEC49FC", "D0A1B2C3D"]


def test_a_need_that_was_asked_about_but_never_resolved_is_ASK_and_not_NONE():
    """The whole point of the rung: a policy that asked the right question and got nothing back
    is materially different from one that never asked, and the ladder exists to say so."""
    n = _node("a", "When was Houston Baptist University founded", uids=("u1",))
    got = _match([n], [_Turn(0, "when was Houston Baptist University founded?", uids=())])
    assert got["a"] == "ask"

    silent = _match([n], [_Turn(0, "what is the weather in Paris?", uids=())])
    assert silent["a"] == "none"


def test_resolve_and_use_still_outrank_ask():
    n = _node("a", "When was Houston Baptist University founded", uids=("u1",))
    q = "when was Houston Baptist University founded?"
    assert _match([n], [_Turn(0, q, uids=("u1",))])["a"] == "resolve"
    # USE needs an INFORMATIVE citation set -- a strict subset of what was retrieved. Citing
    # everything says nothing about which need mattered; see the top-rung tests below.
    assert _match([n], [_Turn(0, q, uids=("u1", "u2"))], cited=("u1",))["a"] == "use"


def test_the_ladder_identity_holds_over_a_mixed_graph():
    """RNR_ask >= RNR_resolve >= RNR_use, over needs in every state at once."""
    nodes = [
        _node("used", "Houston Baptist University founding year", uids=("u1",)),
        _node("resolved", "Collegian newspaper ownership record", uids=("u2",)),
        _node("asked", "annual revenue of the publisher", uids=("u3",)),
        _node("missed", "population of Reykjavik in 1970", uids=("u4",)),
    ]
    turns = [
        _Turn(0, "Houston Baptist University founding year?", uids=("u1",)),
        _Turn(1, "Collegian newspaper ownership record?", uids=("u2",)),
        _Turn(2, "what is the annual revenue of the publisher?", uids=()),
    ]
    got = _match(nodes, turns, cited=("u1",))
    assert got == {"used": "use", "resolved": "resolve", "asked": "ask", "missed": "none"}

    rank = {"none": 0, "ask": 1, "resolve": 2, "use": 3}
    n = len(got)
    ask = sum(1 for v in got.values() if rank[v] >= 1) / n
    res = sum(1 for v in got.values() if rank[v] >= 2) / n
    use = sum(1 for v in got.values() if rank[v] >= 3) / n
    assert ask >= res >= use
    assert ask > res > use, "the three rungs must be able to SEPARATE, not merely be ordered"


def test_ask_is_conservative_because_it_is_the_cheapest_rung_to_game():
    """One shared common word must not credit an ask, or a policy that says "account" once is
    recorded as having asked about every account-related need in the suite."""
    n = _node("a", "annual percentage rate on the platinum rewards credit card")
    assert not asked_about(n, "TELL ME ABOUT MY CREDIT")
    assert not asked_about(n, "WHAT IS THE ANNUAL FEE")
    assert asked_about(n, "WHAT IS THE ANNUAL PERCENTAGE RATE ON THE PLATINUM REWARDS CARD")

    assert ASK_MIN_TERMS >= 2
    assert 0.0 < ASK_MIN_COVERAGE <= 1.0


def test_a_short_need_must_be_matched_entirely():
    """Below the count floor, coverage alone would let a single word through."""
    n = _node("a", "Reykjavik population")
    assert asked_about(n, "WHAT IS THE REYKJAVIK POPULATION")
    assert not asked_about(n, "TELL ME ABOUT REYKJAVIK")


def test_a_need_with_no_distinctive_terms_never_fires():
    """Absent is absent: a need made entirely of stopwords is not evidence of an ask."""
    assert not asked_about(_node("a", "what is that"), "WHAT IS THAT")
    assert not asked_about(_node("b", ""), "ANYTHING AT ALL")


def test_the_matcher_id_moved_so_old_and_new_rows_cannot_be_pooled():
    """matcher_id rides into matcher_hash -> scorer_hash. Changing an instrument without
    changing its identity is how two different measurements get averaged into one cell."""
    from pi_eval.score import matcher_hash

    m = MechanicalMatcher()
    # v3: the ASK rung became per-turn and learned to resolve MuSiQue's `#N` references, and
    # RESOLVE moved from min to max over a need's evidence uids. Every generation must hash
    # apart from every other, so this guards both prior ids rather than only the last one.
    assert m.matcher_id == "mechanical_v3"

    class _V1(MechanicalMatcher):
        matcher_id = "mechanical_v1"

    class _V2(MechanicalMatcher):
        matcher_id = "mechanical_v2"

    hashes = {matcher_hash(m), matcher_hash(_V1()), matcher_hash(_V2())}
    assert len(hashes) == 3


# --------------------------------------------------------------------------- and the top rung
#
# Both shipped Answerers set `cited_unit_ids=tuple(u.uid for u in ev.units)` -- they cite EVERY
# retrieved unit, unconditionally. So `ev <= cited` holds whenever `ev <= retrieved`, `kind` is
# always "use", and the top rung collapsed into the middle one, exactly as ASK had collapsed
# into it from below. The ladder had one rung, in the middle.


def test_use_is_withheld_when_the_answerer_cites_everything_it_retrieved():
    """A citation list that names everything carries no information about USE. Reporting it as
    USE would put the ladder's strongest claim -- the need materially changed the answer -- on
    an instrument that says nothing."""
    nodes = [
        _node("a", "alpha beta gamma", uids=("u1",)),
        _node("b", "delta epsilon zeta", uids=("u2",)),
    ]
    turns = [_Turn(0, "alpha beta gamma?", uids=("u1", "u2"))]
    got = _match(nodes, turns, cited=("u1", "u2"))
    assert got == {"a": "resolve", "b": "resolve"}


def test_an_informative_citation_set_restores_the_rung():
    """Cite a strict subset -- attribute rather than list -- and USE becomes real again with no
    other change. The instrument was never the problem; the Answerer's behaviour was."""
    nodes = [
        _node("a", "alpha beta gamma", uids=("u1",)),
        _node("b", "delta epsilon zeta", uids=("u2",)),
    ]
    turns = [_Turn(0, "alpha beta gamma?", uids=("u1", "u2"))]
    got = _match(nodes, turns, cited=("u1",))
    assert got == {"a": "use", "b": "resolve"}


def test_both_shipped_answerers_currently_cite_everything():
    """The fact the cap follows from. If an Answerer ever attributes instead, this test fails
    and the cap should be revisited -- which is the point of asserting it."""
    import inspect

    from pinq_expt.components import FrozenLLMAnswerer
    from pinq_expt.fakes import FrozenAnswerer

    for cls in (FrozenLLMAnswerer, FrozenAnswerer):
        src = inspect.getsource(cls)
        assert "cited_unit_ids=tuple(u.uid for u in ev.units)" in src, cls.__name__


def test_no_citations_at_all_is_not_informative_either():
    """An empty citation list is not evidence that nothing was used."""
    nodes = [_node("a", "alpha beta gamma", uids=("u1",))]
    assert _match(nodes, [_Turn(0, "alpha beta gamma?", uids=("u1",))], cited=())["a"] == "resolve"
