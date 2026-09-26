"""StrategyQA: the firewall, the two edge mechanisms, the constructed pool, and a rollout.

The fixture is four hand-written questions in the authors' exact JSON shape, chosen to cover
every branch of the edge extractor:

    a, b   two independent lookups then an operation over BOTH ("Is #2 before #1?")
    c      NO back-reference anywhere: the honest flat case, two depth-0 seeds, no facets
    d      a PROSE back-reference and no "#N": the observational path

The point of c is that it is allowed to be flat. Inventing a chain from step order would
manufacture exactly the structure this project claims to measure, so an absent edge is a
correct output here and this file pins it as one.
"""

import json
import shutil

import pytest

from pi_eval.build.common import read_graphs
from pi_eval.build.strategyqa_build import build, step_evidence
from pi_eval.gold import GoldEdge, GoldNode
from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.view import LeakageError, make_view
from pinq_adapters.paragraphs import PARAGRAPH_KEYS, RECORD_KEYS
from pinq_adapters.strategyqa.suite import StrategyQASuite
from pinq_expt.fakes import EchoDrafter, FrozenAnswerer, VerbatimInquirer

FIXTURE = "strategyqa"
POOL = 8
A = "fixture0000000000000a"  # Aristotle / laptop: s1,s2 seeds -> s3 (operation)
B = "fixture0000000000000b"  # Genghis / Caesar: same shape
C = "fixture0000000000000c"  # flat: no references at all
D = "fixture0000000000000d"  # prose back-reference

UPSTREAM_GOLD_KEYS = ("answer", "facts", "decomposition", "evidence", "term", "description", "qid")


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("strategyqa")
    raw = root / "raw"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    res = build(raw_dir=raw, root=root, pool_size=POOL, allow_download=False, verify=False, seed=17)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    return StrategyQASuite(res.corpus.parent), graphs, res


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
    """`facts` and `decomposition` are the dangerous ones here: the whole premise of the
    suite is that the reasoning steps are NOT stated, and either field states them."""
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
        assert v.suite_id == "strategyqa" and v.question and v.word_cap > 0


@pytest.mark.parametrize(
    "gold_field",
    sorted(set(GoldNode.__dataclass_fields__) | set(GoldEdge.__dataclass_fields__))
    + list(UPSTREAM_GOLD_KEYS),
)
def test_every_gold_field_name_is_rejected_by_make_view(built, gold_field):
    suite, _, _ = built
    v = suite.view(A)
    kw = {f: getattr(v, f) for f in type(v).__dataclass_fields__}
    kw[gold_field] = "leaked"
    with pytest.raises(LeakageError):
        make_view(**kw)


def test_the_question_alone_does_not_state_the_strategy(built):
    """The premise of the suite, asserted rather than assumed: no decomposition step's
    content words all appear in the question the agent sees."""
    suite, g, _ = built
    q = suite.view(A).question.lower()
    assert "laptop" in q
    assert "invented" not in q and "live" not in q


# ------------------------------------------------------------------ the two edge mechanisms


def _edges(graph):
    return sorted(
        (e.gold_src_node_id, e.gold_dst_node_id, e.gold_verified) for e in graph.gold_edges
    )


def test_hash_n_references_give_mechanical_edges(built):
    """A step reading "Is #2 before #1?" is as explicit as MuSiQue's, extracted the same way."""
    _, g, _ = built
    assert _edges(g[A]) == [("s1", "s3", "mechanical"), ("s2", "s3", "mechanical")]
    assert _edges(g[B]) == [("s1", "s3", "mechanical"), ("s2", "s3", "mechanical")]


def test_a_question_with_no_reference_stays_flat_and_facetless(built):
    """The honest lower bound. Every step is a depth-0 seed, and because facets are the
    components that remain after deleting the depth-0 frontier, there are none."""
    _, g, _ = built
    assert _edges(g[C]) == []
    assert g[C].gold_seed_node_ids == ("s1", "s2")
    assert g[C].depth_histogram() == {0: 2}
    assert g[C].gold_facets == ()


def test_a_prose_reference_gives_an_observational_edge_not_a_mechanical_one(built):
    """A step reading "Is the previous value below 40 degrees?" names a prior result but
    does not identify it. We attribute it to the immediately preceding step and record that
    nothing verified the attribution — which is what gold_verified='observational' means."""
    _, g, _ = built
    assert _edges(g[D]) == [("s1", "s2", "observational")]
    edge = g[D].gold_edges[0]
    assert edge.gold_confidence < 1.0
    assert edge.gold_provenance == "mechanical"  # a regex found it; a regex did not verify it


def test_nodes_are_human_composed_even_where_the_edge_is_only_inferred(built):
    """The two provenances are tracked separately precisely so that a weak edge cannot
    downgrade a strong node."""
    _, g, _ = built
    for graph in g.values():
        for n in graph.gold_nodes:
            assert n.gold_provenance_primary == "human_composed"


def test_operation_steps_are_optional_and_untestable(built):
    """A step with no paragraph evidence is a pure operation over earlier answers. It stays
    in the graph because it carries the edges, but scoring it as required would build a
    permanent ceiling below 1.0 into every RNR on this suite."""
    _, g, _ = built
    op = {n.gold_node_id: n for n in g[A].gold_nodes}["s3"]
    assert op.gold_ev_uids == ()
    assert op.gold_partition == "optional"
    assert op.gold_ablation_verdict == "UNTESTABLE"
    assert op.gold_kind == "constraint"
    for graph in g.values():
        for n in graph.gold_nodes:
            assert (n.gold_partition == "required") == bool(n.gold_ev_uids)


def test_depths_are_hand_countable(built):
    _, g, _ = built
    assert g[A].depth_histogram() == {0: 2, 1: 1}
    assert g[D].depth_histogram() == {0: 1, 1: 1}


def test_step_evidence_takes_one_annotator_not_the_union(built, request):
    """The matcher requires ALL of a node's gold uids to be retrieved, so unioning three
    annotators would demand a policy retrieve every alternative any of them happened to pick.
    Fixture task A's first step has ["Aristotle-1"] from annotator 0 and
    ["Aristotle-1", "Plato-1"] from annotator 1; only the first must survive.
    """
    questions = json.loads(
        (
            request.config.rootpath / "tests" / "fixtures" / FIXTURE / "strategyqa_train.json"
        ).read_text()
    )
    rec = next(r for r in questions if r["qid"] == A)
    assert step_evidence(rec, 0) == ["Aristotle-1"]
    assert step_evidence(rec, 2) == []  # "operation" everywhere


# ------------------------------------------------------------------ the constructed pool


def test_the_pool_is_padded_with_distractors_and_gold_is_not_at_a_fixed_index(built):
    """StrategyQA ships evidence paragraphs but no distractors. An all-gold pool would make
    retrieval a no-op and every discovery metric read ~1.0 for free."""
    suite, g, _ = built
    positions = set()
    for tid in suite.task_ids():
        units = suite.units(tid)
        assert len(units) == POOL
        gold_uids = {u for n in g[tid].gold_nodes for u in n.gold_ev_uids}
        assert 0 < len(gold_uids) < POOL, "the pool must contain distractors"
        positions |= {i for i, u in enumerate(units) if u.uid in gold_uids}
    assert len(positions) > 1, "gold must not sit at one systematic index"


def test_the_pool_is_reproducible_for_a_given_qid(tmp_path, request):
    """The distractor draw is seeded per task id, so a 4-task fixture build and a full build
    put the same pool in front of the same question — otherwise two corpora that differ only
    in how many tasks were built would not be comparable."""
    fixture = request.config.rootpath / "tests" / "fixtures" / FIXTURE
    hashes = []
    for n in (2, None):
        raw = tmp_path / f"raw{n}"
        shutil.copytree(fixture, raw)
        res = build(
            raw_dir=raw,
            root=tmp_path / f"root{n}",
            pool_size=POOL,
            limit=n,
            allow_download=False,
            verify=False,
        )
        rows = {json.loads(x)["id"]: json.loads(x) for x in res.corpus.read_text().splitlines()}
        hashes.append(rows[A]["paragraphs"])
    assert hashes[0] == hashes[1]


def test_gold_ev_uids_are_uids_the_retriever_can_actually_return(built):
    suite, g, _ = built
    for tid in suite.task_ids():
        available = {u.uid for u in suite.units(tid)}
        gold_uids = {u for n in g[tid].gold_nodes for u in n.gold_ev_uids}
        assert gold_uids and gold_uids <= available, tid


# ------------------------------------------------------------------ retrieval and the loop


def test_bm25_surfaces_the_evidence_paragraph_for_most_evidence_bearing_steps(built):
    """Queried with the gold step text — the best case a policy could hope to formulate.
    A floor, not an equality: this watches for the corpus or the tokenizer breaking, not for
    BM25 being good."""
    suite, g, _ = built
    hits = total = 0
    for tid in suite.task_ids():
        r = suite.retriever(tid)
        for n in g[tid].gold_nodes:
            if not n.gold_ev_uids:
                continue
            total += 1
            hits += set(n.gold_ev_uids) <= {u.uid for u in r.search(n.gold_text, 3)}
    assert total == 7
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
        assert traj.outcome.answer.n_words <= suite.view(tid).word_cap


def test_the_answer_is_a_word_not_a_bool(built):
    """The Answerer emits text, so a bool in gold_answer would never match anything."""
    _, g, _ = built
    # `.answer`, not `.gold_answer`: the raw field carries the canary nonce so that leaking
    # the answer key carries it too. Every comparison uses the stripped form.
    assert {g[t].answer for t in (A, B, C, D)} == {"yes", "no"}
    assert g[B].answer == "yes" and g[A].answer == "no"


# ------------------------------------------------------------------ the real download


@pytest.mark.integration
def test_real_strategyqa_zip_builds_with_the_pinned_digest(tmp_path, request):
    from pi_eval.build import strategyqa_build

    src = request.config.rootpath / "data" / "raw" / FIXTURE / strategyqa_build.ZIP_NAME
    if not src.exists():
        pytest.skip("no warm data/raw/strategyqa cache")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / strategyqa_build.ZIP_NAME).symlink_to(src)

    res = build(raw_dir=raw, root=tmp_path, limit=25, allow_download=False)
    assert res.n_tasks == 25
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines()]
    assert all(len(r["paragraphs"]) == strategyqa_build.POOL_SIZE for r in rows)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    # MEASURED on the full 2,290-question train split: 99.2% of questions carry at least one
    # "#N", so a 25-task sample with no edges at all would mean the extractor broke.
    assert sum(1 for g in graphs.values() if g.gold_edges) >= 20
    suite = StrategyQASuite(res.corpus.parent)
    for tid in list(suite.task_ids())[:10]:
        available = {u.uid for u in suite.units(tid)}
        assert {u for n in graphs[tid].gold_nodes for u in n.gold_ev_uids} <= available
