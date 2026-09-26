"""Edge validity (lane L1): the pure functions behind `scripts/edge_validity/probe.py`.

Hermetic: hand-made tiny graphs, a fake retriever, no store, no proxy, no PI_GOLD_ROOT. The
binding rule is `artifacts/edge_validity_20260923/CRITERION.md`; these tests pin the parts of it
that a silent bug would move -- the class boundaries at exactly 0.2 / 0.4 / 0.6 (where naive float
subtraction gives 0.6 - 0.2 = 0.39999999999999997 < 0.4), the UNTESTABLE conditions, what context
A is allowed to contain, and that context C's random node is from another task, at u's depth,
type-matched where it can be, and reproducible.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.edge_validity.probe import (
    PROMPTS,
    Candidate,
    ancestors,
    assign_class,
    build_candidates,
    context_facts,
    gap,
    hit_rate,
    node_fact,
    parse_queries,
    pick_random_node,
    prerequisite_edges,
    query_hits,
    relation_of,
    render_prompt,
    seeded_rng,
)

from pi_eval.gold import GoldEdge, GoldGraph, GoldNode

F = Fraction


def _node(suite, task, nid, text, answer=None, depth=0, ev=()):
    return GoldNode(
        gold_suite=suite,
        gold_task_key=task,
        gold_node_id=nid,
        gold_text=text,
        gold_aliases=(answer,) if answer is not None else (),
        gold_depth=depth,
        gold_ev_uids=tuple(ev),
    )


def _edge(suite, task, src, dst, kind="prerequisite"):
    return GoldEdge(
        gold_suite=suite,
        gold_task_key=task,
        gold_src_node_id=src,
        gold_dst_node_id=dst,
        gold_edge_kind=kind,
    )


def _musique_chain(task="3hop__a_b_c"):
    """s1 -> s2 -> s3, plus a relevance edge that must NOT be enumerated."""
    s = "musique"
    nodes = (
        _node(s, task, "s1", "The Galaxy Kings >> performer", "Bob Schneider", 0, ("u1",)),
        _node(s, task, "s2", "#1 >> record label", "Kirtland Records", 1, ("u2",)),
        _node(s, task, "s3", "When was #2 created?", "2003", 2, ("u3",)),
    )
    edges = (
        _edge(s, task, "s1", "s2"),
        _edge(s, task, "s2", "s3"),
        _edge(s, task, "s1", "s3", kind="relevance"),
    )
    return GoldGraph(gold_suite=s, gold_task_key=task, gold_nodes=nodes, gold_edges=edges)


def _strategyqa_graph(task="sqa1"):
    s = "strategyqa"
    nodes = (
        _node(s, task, "s1", "How many times was Kublai Khan married?", None, 0, ("p1",)),
        _node(s, task, "s2", "Who was Kublai Khan's grandfather?", None, 0, ("p2",)),
        _node(s, task, "s3", "How many wives did #2 have?", None, 1, ("p3",)),
        _node(s, task, "s4", "Is #1 equal to 1 and is #3 equal 1?", None, 2, ()),
    )
    edges = (
        _edge(s, task, "s2", "s3"),
        _edge(s, task, "s1", "s4"),
        _edge(s, task, "s3", "s4"),
    )
    return GoldGraph(gold_suite=s, gold_task_key=task, gold_nodes=nodes, gold_edges=edges)


# ---------------------------------------------------------------- edge enumeration, ancestry


def test_prerequisite_edges_skips_relevance_and_is_sorted():
    g = _musique_chain()
    assert prerequisite_edges(g) == [("s1", "s2"), ("s2", "s3")]


def test_ancestors_are_every_node_with_a_path_to_u_and_never_u():
    g = _musique_chain()
    assert ancestors(g, "s1") == []
    assert ancestors(g, "s2") == ["s1"]
    assert ancestors(g, "s3") == ["s1", "s2"]
    sq = _strategyqa_graph()
    assert ancestors(sq, "s4") == ["s1", "s2", "s3"]
    assert "s3" not in ancestors(sq, "s3")


# ---------------------------------------------------------------- context construction


def test_context_a_never_contains_u_or_v_answer_or_text():
    g = _musique_chain()
    nodes = {n.gold_node_id: n for n in g.gold_nodes}
    q = "When was the record label of the performer of The Galaxy Kings created?"
    for u, v in prerequisite_edges(g):
        facts = context_facts("musique", g, u, "A", titles={})
        prompt_a = render_prompt("P1", q, facts)
        for forbidden in (nodes[u], nodes[v]):
            assert forbidden.gold_aliases[0] not in facts
            assert forbidden.gold_aliases[0] not in prompt_a
            assert forbidden.gold_text not in prompt_a


def test_context_a_is_the_question_alone_when_u_has_no_ancestors():
    g = _musique_chain()
    assert context_facts("musique", g, "s1", "A", titles={}) == ()


def test_context_a_holds_every_ancestor_answer_and_b_adds_u():
    g = _musique_chain()
    assert context_facts("musique", g, "s2", "A", titles={}) == ("Bob Schneider",)
    assert context_facts("musique", g, "s2", "B", titles={}) == (
        "Bob Schneider",
        "Kirtland Records",
    )


def test_context_c_is_a_plus_the_random_fact_and_never_u():
    g = _musique_chain()
    facts = context_facts("musique", g, "s2", "C", titles={}, random_fact=("Elio Petri",))
    assert facts == ("Bob Schneider", "Elio Petri")
    with pytest.raises(ValueError):
        context_facts("musique", g, "s2", "C", titles={})


def test_strategyqa_context_uses_evidence_titles_not_answers():
    sq = _strategyqa_graph()
    titles = {"p1": "Kublai Khan", "p2": "Kublai Khan", "p3": "Genghis Khan"}
    assert node_fact("strategyqa", {n.gold_node_id: n for n in sq.gold_nodes}["s2"], titles) == (
        "Kublai Khan",
    )
    # edge s2 -> s3: A = the question alone (s2 has no ancestors); B adds s2's evidence title
    assert context_facts("strategyqa", sq, "s2", "A", titles=titles) == ()
    assert context_facts("strategyqa", sq, "s2", "B", titles=titles) == ("Kublai Khan",)
    # edge s3 -> s4: A = titles of s3's ancestors (s2), deduplicated; v's title never enters A
    assert context_facts("strategyqa", sq, "s3", "A", titles=titles) == ("Kublai Khan",)
    assert "Genghis Khan" not in context_facts("strategyqa", sq, "s3", "A", titles=titles)


def test_prompt_renders_question_and_facts_and_the_two_prompts_differ():
    p1 = render_prompt("P1", "Q?", ("fact one",))
    p2 = render_prompt("P2", "Q?", ("fact one",))
    assert "Q?" in p1 and "fact one" in p1 and '"queries"' in p1
    assert p1 != p2
    assert set(PROMPTS) == {"P1", "P2"}


# ---------------------------------------------------------------- context C's random node


def _wiki_candidates():
    return [
        Candidate("t1", "e1", 0, "director", ("Luis Saslavsky",)),  # own task: never drawn
        Candidate("t2", "e1", 0, "director", ("Elio Petri",)),
        Candidate("t3", "e1", 0, "director", ("Andrzej Wajda",)),
        Candidate("t4", "e1", 0, "mother", ("Anna Kowalska",)),
        Candidate("t5", "e3", 1, "director", ("Deep Director",)),  # wrong depth
        Candidate("t6", "e1", 0, "director", ("luis saslavsky",)),  # same fact as u, casefolded
    ]


def test_random_node_is_from_another_task_same_depth_same_relation():
    cands = _wiki_candidates()
    for u in ("e1", "e2", "e7"):
        pick = pick_random_node(
            "wiki2",
            "t1",
            u,
            depth=0,
            relation="director",
            exclude_facts={"luis saslavsky"},
            candidates=cands,
        )
        assert pick.task_id != "t1"
        assert pick.depth == 0
        assert pick.relation == "director"
        assert pick.type_matched is True
        assert pick.task_id in {"t2", "t3"}


def test_random_node_falls_back_to_any_relation_and_says_so():
    cands = _wiki_candidates()
    pick = pick_random_node(
        "wiki2", "t1", "e1", depth=0, relation="spouse", exclude_facts=set(), candidates=cands
    )
    assert pick.type_matched is False
    assert pick.task_id != "t1" and pick.depth == 0


def test_random_node_is_deterministic_under_the_seed():
    cands = _wiki_candidates()
    kw = dict(depth=0, relation=None, exclude_facts=set(), candidates=cands)
    a = pick_random_node("musique", "t1", "s1", **kw)
    b = pick_random_node("musique", "t1", "s1", **kw)
    assert a == b
    # the seed is (suite, task, u): a different u may draw differently, but always reproducibly
    draws = {pick_random_node("musique", "t1", f"s{i}", **kw).task_id for i in range(40)}
    assert len(draws) > 1
    assert seeded_rng("a", "b").random() == seeded_rng("a", "b").random()
    assert seeded_rng("a", "b").random() != seeded_rng("a", "c").random()


def test_random_node_never_draws_an_excluded_fact():
    cands = _wiki_candidates()
    for i in range(30):
        pick = pick_random_node(
            "musique",
            "t1",
            f"s{i}",
            depth=0,
            relation=None,
            exclude_facts={"elio petri", "andrzej wajda", "luis saslavsky"},
            candidates=cands,
        )
        assert pick.fact == ("Anna Kowalska",)


def test_build_candidates_reads_depth_relation_and_fact():
    g = _musique_chain("t9")
    cands = build_candidates("musique", {"t9": g}, titles_by_task={"t9": {}})
    by = {c.node_id: c for c in cands}
    assert by["s1"].depth == 0 and by["s1"].fact == ("Bob Schneider",)
    assert by["s2"].depth == 1
    assert relation_of("Xawery Zulawski >> mother") == "mother"
    assert relation_of("When was #2 created?") is None


# ---------------------------------------------------------------- query parsing and hits


def test_parse_queries_accepts_fenced_json_and_truncates_to_five():
    qs, ok = parse_queries('```json\n{"queries": ["a", "b", "c", "d", "e", "f"]}\n```')
    assert ok and qs == ["a", "b", "c", "d", "e"]
    qs, ok = parse_queries('Sure: {"queries": ["a", "b"]} done')
    assert ok and qs == ["a", "b"]
    qs, ok = parse_queries("not json at all")
    assert not ok and qs == []


class _FakeRetriever:
    """Returns units whose uid is a word of the query, in query order. Top-k honoured."""

    def search(self, query, k):
        return tuple(SimpleNamespace(uid=w) for w in query.split()[:k])


def test_query_hits_require_all_v_uids_in_top5_and_record_any_and_best_rank():
    r = _FakeRetriever()
    rows = query_hits(r, ["x v1 v2", "v1 y", "a b c d e f v1", "none"], ("v1", "v2"))
    assert [x["hit"] for x in rows] == [True, False, False, False]
    assert [x["hit_any"] for x in rows] == [True, True, False, False]
    assert [x["best_rank"] for x in rows] == [2, 1, 7, None]


def test_hit_rate_denominator_is_always_five():
    assert hit_rate([True, False]) == F(1, 5)
    assert hit_rate([True] * 5) == F(1)
    assert hit_rate([]) == F(0)


# ---------------------------------------------------------------- the class rule


def _cls(**kw):
    base = dict(
        v_has_evidence=True,
        overlap=False,
        floor=0,
        hit_a=F(0),
        hit_b=F(3, 5),
        hit_c=F(0),
        cb_answers=False,
    )
    base.update(kw)
    hit_a, hit_b, hit_c = base.pop("hit_a"), base.pop("hit_b"), base.pop("hit_c")
    return assign_class(hit_a=hit_a, g=gap(hit_b, hit_a, hit_c), **base)[0]


def test_gap_is_exact_at_the_float_trap():
    assert gap(F(3, 5), F(1, 5), F(0)) == F(2, 5)
    assert float(gap(F(3, 5), F(1, 5), F(0))) == 0.4


def test_validated_at_exactly_hit_a_0_2_and_g_0_4():
    assert _cls(hit_a=F(1, 5), hit_b=F(3, 5), hit_c=F(0)) == "VALIDATED"


def test_hit_a_just_above_0_2_is_not_validated():
    assert _cls(hit_a=F(2, 5), hit_b=F(1), hit_c=F(0)) == "UNTESTABLE"


def test_g_just_below_0_4_is_not_validated():
    assert _cls(hit_a=F(1, 5), hit_b=F(2, 5), hit_c=F(0)) == "UNTESTABLE"


def test_g_uses_the_max_of_a_and_c():
    assert _cls(hit_a=F(0), hit_b=F(3, 5), hit_c=F(2, 5)) == "UNTESTABLE"


def test_refuted_at_exactly_hit_a_0_6_and_not_at_0_4():
    assert _cls(hit_a=F(3, 5), hit_b=F(1)) == "REFUTED"
    assert _cls(hit_a=F(2, 5), hit_b=F(1)) == "UNTESTABLE"


def test_nonzero_floor_blocks_validated():
    assert _cls(floor=1) == "UNTESTABLE"


def test_closed_book_answering_u_refutes_and_blocks_validated():
    assert _cls(cb_answers=True) == "REFUTED"


def test_closed_book_not_testable_leaves_the_other_conditions():
    assert _cls(cb_answers=None) == "VALIDATED"


def test_untestable_when_v_has_no_evidence_or_overlaps_u():
    assert _cls(v_has_evidence=False, hit_a=None, hit_b=None, hit_c=None) == "UNTESTABLE"
    assert _cls(overlap=True) == "UNTESTABLE"
    # the evidence conditions come first: the hit instrument cannot run on such an edge
    assert _cls(overlap=True, hit_a=F(1), hit_b=F(1)) == "UNTESTABLE"
    assert _cls(v_has_evidence=False, hit_a=None, hit_b=None, hit_c=None, cb_answers=True) == (
        "UNTESTABLE"
    )


def test_secondary_thresholds_are_parameters():
    kw = dict(
        v_has_evidence=True, overlap=False, floor=0, hit_a=F(1, 5), g=F(1, 5), cb_answers=False
    )
    assert assign_class(**kw)[0] == "UNTESTABLE"
    assert assign_class(**kw, thr_g=F(1, 5))[0] == "VALIDATED"
    assert assign_class(**kw, thr_a=F(0), thr_g=F(1, 5))[0] == "UNTESTABLE"


# ---------------------------------------------------------------- truncated replies


def _entry(content, finish):
    return {"response": {"choices": [{"message": {"content": content}, "finish_reason": finish}]}}


def test_a_reply_cut_by_the_token_budget_is_truncated_not_a_zero():
    """Opus 5 on this gateway spends hidden thinking inside max_tokens: at 512, 502 of 5,206
    formulator replies came back finish_reason=length, most with EMPTY content. Scored as
    0 of 5 they would have pulled hit_C (and so G) around silently. They must be refused."""
    from scripts.edge_validity.probe import reply_status

    assert reply_status(_entry("", "length")) == "truncated"
    assert reply_status(_entry('{"queries": ["a", "b"', "length")) == "truncated"
    # a complete reply that happens to end at the budget still parses and is scored
    assert reply_status(_entry('{"queries": ["a"]}', "length")) == "ok"
    assert reply_status(_entry('{"queries": ["a"]}', "stop")) == "ok"
    assert reply_status(_entry("no json here", "stop")) == "unparseable"


def test_formulator_and_opus_closed_book_budgets_leave_room_for_hidden_thinking():
    from scripts.edge_validity.probe import closed_book_body, formulator_body

    assert formulator_body("openai/aws/claude-opus-5", "x")["max_tokens"] >= 4096
    assert closed_book_body("openai/aws/claude-opus-5", "q?")["max_tokens"] >= 2048
    # the 8B closed-book request is byte-for-byte the published one's budget (the lock)
    assert closed_book_body("qwen3-8b-base", "q?")["max_tokens"] == 64
