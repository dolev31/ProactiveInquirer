"""2WikiMultihopQA: the firewall, type-gated chain edges, entity->paragraph matching, a rollout.

The fixture is one hand-written record of each reasoning type, in the authors' original JSON
shape, and each one pins a different property of the extractor:

    compositional        object -> subject linkage across two triples
    inference            the same, over a repeated relation (father of father)
    comparison           FLAT by construction: two seeds, no edges, no facets
    bridge_comparison    TWO independent chains, therefore TWO facets, plus the
                         "Charles Saunders" / "Charles Saunders (director)" disambiguation
                         that raw string equality would miss

Remember what a node is here: a Wikidata triple, not a sub-question. This is an ENTITY graph.
"""

import json
import shutil

import pytest

from pi_eval.build.common import read_graphs, unit_uid
from pi_eval.build.wiki2_build import CORPUS_ID, _key, build
from pi_eval.gold import GoldEdge, GoldNode
from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.view import LeakageError, make_view
from pinq_adapters.paragraphs import PARAGRAPH_KEYS, RECORD_KEYS
from pinq_adapters.wiki2.suite import Wiki2Suite
from pinq_expt.fakes import EchoDrafter, FrozenAnswerer, VerbatimInquirer

FIXTURE = "wiki2"
COMP = "fixture_compositional_1"
CMP = "fixture_comparison_1"
INF = "fixture_inference_1"
BRIDGE = "fixture_bridge_comparison_1"

UPSTREAM_GOLD_KEYS = ("_id", "type", "answer", "context", "supporting_facts", "evidences")


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("wiki2")
    raw = root / "raw"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    res = build(split="train", raw_dir=raw, root=root, allow_download=False, verify=False)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    return Wiki2Suite(res.corpus.parent), graphs, res


# ------------------------------------------------------------------ the firewall


def test_the_adapter_structurally_satisfies_the_suite_and_retriever_protocols(built):
    """Structural typing, checked at runtime: an adapter conforms by SHAPE and never has to
    import a base class from pinq. If it did, every suite would drag the core package's
    inheritance into its own release, which is the coupling the Protocol seam exists to
    prevent."""
    from pinq.protocols import Retriever, TaskSuite

    suite, _, _ = built
    assert isinstance(suite, TaskSuite)
    assert isinstance(suite.retriever(suite.task_ids()[0]), Retriever)
    assert suite.actuator(suite.task_ids()[0]) is None


def test_public_record_has_exactly_the_allowed_keys(built):
    _, _, res = built
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines()]
    assert rows
    for r in rows:
        assert set(r) == set(RECORD_KEYS) == {"id", "question", "paragraphs"}
        for p in r["paragraphs"]:
            assert set(p) == set(PARAGRAPH_KEYS)


def test_no_gold_key_name_appears_anywhere_in_the_public_corpus(built):
    """`type` matters as much as `answer` here: telling a policy up front that its question
    is a comparison hands it the shape of the graph it is supposed to discover."""
    _, _, res = built
    text = res.corpus.read_text()
    for key in UPSTREAM_GOLD_KEYS:
        assert f'"{key}"' not in text, key
    for field in list(GoldNode.__dataclass_fields__) + list(GoldEdge.__dataclass_fields__):
        assert f'"{field}"' not in text, field


def test_make_view_succeeds_for_every_task(built):
    suite, _, _ = built
    for tid in suite.task_ids():
        v = suite.view(tid)
        assert v.suite_id == "wiki2" and v.question and v.word_cap > 0


@pytest.mark.parametrize(
    "gold_field",
    sorted(set(GoldNode.__dataclass_fields__) | set(GoldEdge.__dataclass_fields__))
    + list(UPSTREAM_GOLD_KEYS),
)
def test_every_gold_field_name_is_rejected_by_make_view(built, gold_field):
    suite, _, _ = built
    v = suite.view(COMP)
    kw = {f: getattr(v, f) for f in type(v).__dataclass_fields__}
    kw[gold_field] = "leaked"
    with pytest.raises(LeakageError):
        make_view(**kw)


# ------------------------------------------------------------------ type-gated chain edges


def _edges(graph):
    return sorted((e.gold_src_node_id, e.gold_dst_node_id) for e in graph.gold_edges)


def test_a_chain_type_links_object_to_subject(built):
    """e1's object IS e2's subject, so e2 cannot be looked up until e1 is known."""
    _, g, _ = built
    assert _edges(g[COMP]) == [("e1", "e2")]
    assert _edges(g[INF]) == [("e1", "e2")]
    assert {n.gold_node_id: n.gold_depth for n in g[COMP].gold_nodes} == {"e1": 0, "e2": 1}


def test_a_comparison_is_flat_and_that_is_the_correct_shape(built):
    """Both publication dates are needed and neither depends on the other. Every node is a
    depth-0 seed and there are no facets — a missing edge here would be a fabricated one."""
    _, g, _ = built
    assert _edges(g[CMP]) == []
    assert g[CMP].gold_seed_node_ids == ("e1", "e2")
    assert g[CMP].depth_histogram() == {0: 2}
    assert g[CMP].gold_facets == ()


def test_a_bridge_comparison_is_two_chains_and_therefore_two_facets(built):
    """The shape that makes horizontal breadth and vertical depth separable: two independent
    strands, each one hop deep. facet_breadth reads 2 here where a single chain reads 1."""
    _, g, _ = built
    assert _edges(g[BRIDGE]) == [("e1", "e3"), ("e2", "e4")]
    assert g[BRIDGE].depth_histogram() == {0: 2, 1: 2}
    assert [f.gold_node_ids for f in g[BRIDGE].gold_facets] == [("e3",), ("e4",)]


def test_linkage_only_runs_forward_so_a_cycle_is_impossible(built):
    """Restricted to the annotators' listed order. A cycle would leave Depth undefined for
    every node in it, and the inference fixture (father-of-father, same relation twice) is
    exactly where an unordered rule could produce one."""
    _, g, _ = built
    for graph in g.values():
        for e in graph.gold_edges:
            assert e.gold_src_node_id < e.gold_dst_node_id
    assert all(n.gold_depth is not None for n in g[INF].gold_nodes)


def test_nodes_are_triples_written_in_musique_form(built):
    """entity >> relation, with the object as the alias, so one matcher prompt and one set of
    calibration thresholds serve both suites."""
    _, g, _ = built
    n1, n2 = g[COMP].gold_nodes
    assert n1.gold_text == "Harrow Lights >> director"
    assert n1.gold_aliases == ("Petra Vance",)
    assert n2.gold_text == "Petra Vance >> mother"
    assert all(n.gold_provenance_primary == "bench_author" for n in g[COMP].gold_nodes)


# ------------------------------------------------------------------ entity -> paragraph


def test_a_parenthetical_disambiguator_does_not_block_the_match(built):
    """The subject is "Harrow Lights"; the paragraph is titled "Harrow Lights (film)"."""
    suite, g, _ = built
    units = suite.units(COMP)
    assert units[0].title == "Harrow Lights (film)"
    want = unit_uid(CORPUS_ID, COMP, 0, units[0].text)
    assert g[COMP].gold_nodes[0].gold_ev_uids == (want,)


def test_supporting_facts_break_a_same_name_tie(built):
    """Both "Charles Saunders (director)" and "Charles Saunders (Royal Navy officer)"
    normalise to the same key. The annotators already named the paragraph that matters, so a
    coincidence with a distractor must never win."""
    suite, g, _ = built
    assert _key("Charles Saunders (director)") == _key("Charles Saunders")
    units = suite.units(BRIDGE)
    node = {n.gold_node_id: n for n in g[BRIDGE].gold_nodes}["e4"]
    idx = next(i for i, u in enumerate(units) if u.title == "Charles Saunders (director)")
    decoy = next(i for i, u in enumerate(units) if u.title.endswith("(Royal Navy officer)"))
    assert node.gold_ev_uids == (unit_uid(CORPUS_ID, BRIDGE, idx, units[idx].text),)
    assert units[decoy].uid not in node.gold_ev_uids


def test_gold_ev_uids_are_uids_the_retriever_can_actually_return(built):
    suite, g, _ = built
    for tid in suite.task_ids():
        available = {u.uid for u in suite.units(tid)}
        gold_uids = {u for n in g[tid].gold_nodes for u in n.gold_ev_uids}
        assert gold_uids and gold_uids <= available, tid


def test_sentences_are_rolled_up_into_one_paragraph_unit(built):
    """supporting_facts is [title, sent_id], but the retriever returns paragraphs. A gold uid
    the retriever can never emit is a node that can never be resolved."""
    suite, _, _ = built
    units = suite.units(COMP)
    assert len(units) == 6
    assert units[0].span == f"0:{len(units[0].text)}"
    joined = next(u for u in units if u.title == "Harrow Lights (film)")
    assert "1988 British drama film." in joined.text and "Petra Vance" in joined.text


# ------------------------------------------------------------------ retrieval and the loop


def test_bm25_surfaces_the_evidence_paragraph_for_most_triples(built):
    suite, g, _ = built
    hits = total = 0
    for tid in suite.task_ids():
        r = suite.retriever(tid)
        for n in g[tid].gold_nodes:
            if not n.gold_ev_uids:
                continue
            total += 1
            hits += set(n.gold_ev_uids) <= {u.uid for u in r.search(n.gold_text, 3)}
    assert total == 10
    assert hits / total >= 0.7, f"only {hits}/{total} evidence paragraphs in the top 3"


def test_run_loop_over_the_fixture_completes_and_produces_evidence(built):
    suite, _, _ = built
    for tid in suite.task_ids():
        traj = run_loop(
            view=suite.view(tid),
            inquirer=VerbatimInquirer(max_asks=3),
            retriever=suite.retriever(tid),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=BudgetLedger(cap=8),
            max_turns=8,
            k=4,
            seed=0,
        )
        assert traj.n_asks >= 1 and len(traj.evidence) > 0, tid
        assert traj.evidence.uids <= {u.uid for u in suite.units(tid)}
        assert traj.outcome.answer.n_words <= suite.view(tid).word_cap


def test_two_runs_of_the_same_rollout_are_identical(built):
    """No RNG anywhere in the retrieval path: same view, same policy, same evidence set."""
    suite, _, _ = built

    def once():
        return run_loop(
            view=suite.view(BRIDGE),
            inquirer=VerbatimInquirer(max_asks=3),
            retriever=suite.retriever(BRIDGE),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=BudgetLedger(cap=8),
            max_turns=8,
            k=4,
            seed=0,
        )

    a, b = once(), once()
    assert a.evidence.subset_hash == b.evidence.subset_hash
    assert [t.retrieved_uids for t in a.turns] == [t.retrieved_uids for t in b.turns]


# ------------------------------------------------------------------ the real download


@pytest.mark.integration
def test_real_wiki2_dev_builds_with_the_pinned_digest(tmp_path, request):
    src_dir = request.config.rootpath / "data" / "raw" / FIXTURE
    src = next((src_dir / n for n in ("dev.jsonl", "dev.parquet") if (src_dir / n).exists()), None)
    if src is None:
        pytest.skip("no warm data/raw/wiki2 cache")
    if src.suffix == ".parquet":
        pytest.importorskip("pyarrow")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / src.name).symlink_to(src)

    res = build(split="dev", raw_dir=raw, root=tmp_path, limit=25, allow_download=False)
    assert res.n_tasks == 25
    # A truncated conversion must never be cached as if it were the whole split.
    assert not (raw / "dev.jsonl").exists() or src.name == "dev.jsonl"
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    assert all(g.gold_nodes for g in graphs.values())
    suite = Wiki2Suite(res.corpus.parent)
    for tid in list(suite.task_ids())[:10]:
        available = {u.uid for u in suite.units(tid)}
        assert {u for n in graphs[tid].gold_nodes for u in n.gold_ev_uids} <= available
    assert any(g.gold_edges for g in graphs.values()), "dev is majority chain types"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Charles Saunders (director)", "charles saunders"),
        ("El extraño viaje", "el extra o viaje"),
        ("Polish-Russian War", "polish russian war"),
    ],
)
def test_entity_key_normalisation(raw, expected):
    assert _key(raw) == expected
