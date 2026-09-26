"""MuSiQue: the firewall, the #N edge extraction, BM25, and a full rollout.

The fixture in tests/fixtures/musique/ is four hand-written tasks in the exact upstream
JSONL shape, and its graph was drawn on paper before the code was written — a plain 2-hop
chain, a 3-hop chain, a BRANCHING 4-hop whose last step references two earlier ones, and a
fourth task whose first hop sits on the dev/test leakage list. Every structural expectation
below is therefore a hand count, not a golden file: if the extractor changes behaviour these
tests say which shape it got wrong, rather than "the bytes differ".
"""

import json
import re
import shutil

import pytest

from pi_eval.build.common import read_graphs
from pi_eval.build.musique_build import _norm_q, build, load_exclusions
from pi_eval.gold import GoldEdge, GoldNode, compute_depths
from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.view import LeakageError, make_view
from pinq_adapters.musique.suite import MusiqueSuite
from pinq_adapters.paragraphs import PARAGRAPH_KEYS, RECORD_KEYS
from pinq_expt.fakes import EchoDrafter, FrozenAnswerer, VerbatimInquirer

FIXTURE = "musique"
A, B, C = "2hop__1001_1002", "3hop1__2001_2002_2003", "4hop2__3001_3002_3003_3004"
EXCLUDED = "2hop__4001_4002"

# Keys that exist in the upstream record and must never survive into data/corpora/.
UPSTREAM_GOLD_KEYS = (
    "question_decomposition",
    "answer",
    "answer_aliases",
    "paragraph_support_idx",
    "is_supporting",
    "paragraph_text",
    "answerable",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("musique")
    raw = root / "raw"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    # verify=False: the fixture is a hand-written stand-in and cannot carry upstream's digest.
    res = build(split="train", raw_dir=raw, root=root, allow_download=False, verify=False)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    return MusiqueSuite(res.corpus.parent), graphs, res


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
    suite, _, res = built
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines()]
    assert rows, "the fixture must produce tasks"
    for r in rows:
        assert set(r) == set(RECORD_KEYS) == {"id", "question", "paragraphs"}
        for p in r["paragraphs"]:
            assert set(p) == set(PARAGRAPH_KEYS) == {"idx", "title", "text"}


def test_no_gold_key_name_appears_anywhere_in_the_public_corpus(built):
    """Not 'the parsed dict has no gold key' but 'the BYTES contain no gold key'.

    A nested structure could hide one where a shallow key check would not look, and
    is_supporting is the dangerous one: it labels 2-4 paragraphs out of 20, so a record
    carrying it would let a policy skip discovery and still score perfectly.
    """
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
        assert v.task_id == tid and v.suite_id == "musique" and v.question
        assert v.corpus_hash and v.word_cap > 0


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


def test_gold_lives_in_a_different_tree_from_the_corpus(built):
    _, _, res = built
    assert "corpora" in res.corpus.parts and "gold" not in res.corpus.parts
    assert "gold" in res.gold.parts and "corpora" not in res.gold.parts


# ------------------------------------------------------------------ the #N mechanic


def _edges(graph):
    return sorted((e.gold_src_node_id, e.gold_dst_node_id) for e in graph.gold_edges)


def test_hash_n_placeholders_become_exactly_these_prerequisite_edges(built):
    """Hand-counted from the fixture's decompositions.

    A  s2 says "#1 >> spouse"                            -> s1 -> s2
    B  s2 says "#1 ...", s3 says "#2 ..."                -> s1 -> s2 -> s3
    C  s3 says "#1 >> inception", s4 says "... #2 and #3" -> s1 -> s3, s2 -> s4, s3 -> s4
    """
    _, g, _ = built
    assert _edges(g[A]) == [("s1", "s2")]
    assert _edges(g[B]) == [("s1", "s2"), ("s2", "s3")]
    assert _edges(g[C]) == [("s1", "s3"), ("s2", "s4"), ("s3", "s4")]


def test_every_extracted_edge_is_mechanical_and_certain(built):
    """MuSiQue is the calibration suite precisely because nothing here was judged."""
    _, g, _ = built
    for graph in g.values():
        for e in graph.gold_edges:
            assert e.gold_edge_kind == "prerequisite"
            assert e.gold_verified == "mechanical"
            assert e.gold_provenance == "mechanical"
            assert e.gold_confidence == 1.0


def test_a_step_with_no_placeholder_is_a_depth_zero_seed(built):
    _, g, _ = built
    assert g[A].gold_seed_node_ids == ("s1",)
    assert g[B].gold_seed_node_ids == ("s1",)
    assert g[C].gold_seed_node_ids == ("s1", "s2")  # two independent openings


def test_depth_is_the_number_you_can_count_by_hand(built):
    """B is the plain case; C is the case worth pinning.

    C's s4 depends on BOTH s2 (a seed) and s3 (depth 1). A `prerequisite` edge means "all of
    these must be resolved first", so s4 is depth 2 — it genuinely cannot be resolved until
    s3 has been. An earlier version of compute_depths used shortest-path BFS and called it
    depth 1, which systematically understated depth at every multi-parent node and dragged
    genuinely deep needs into shallow buckets.

    Both semantics are asserted here on purpose. The guard is not "depth equals this
    number"; it is "the AND/OR divergence is exactly here and is a deliberate choice", so a
    future redefinition still cannot slip through as a silent re-scoring.
    """
    _, g, _ = built
    assert {n.gold_node_id: n.gold_depth for n in g[A].gold_nodes} == {"s1": 0, "s2": 1}
    assert {n.gold_node_id: n.gold_depth for n in g[B].gold_nodes} == {"s1": 0, "s2": 1, "s3": 2}
    assert {n.gold_node_id: n.gold_depth for n in g[C].gold_nodes} == {
        "s1": 0,
        "s2": 0,
        "s3": 1,
        "s4": 2,
    }

    # The divergence, pinned explicitly: OR semantics would reach s4 in one hop from s2.
    from pi_eval.gold import compute_depths

    nodes = [n.gold_node_id for n in g[C].gold_nodes]
    seeds = [n.gold_node_id for n in g[C].gold_nodes if n.gold_depth == 0]
    or_depths = compute_depths(nodes, list(g[C].gold_edges), seeds, semantics="any")
    assert or_depths["s4"] == 1, "OR semantics is the understating definition we rejected"
    assert g[B].depth_histogram() == {0: 1, 1: 1, 2: 1}
    # |V_d| shifts with the correction: s4 moves out of the depth-1 bucket into depth 2,
    # which is precisely the stratum the vertical-proactivity claim is demonstrated on.
    assert g[C].depth_histogram() == {0: 2, 1: 1, 2: 1}


def test_stored_depth_is_reproducible_from_nodes_edges_and_seeds(built):
    """Depth is DERIVED, never annotated: recomputing it must return the stored values."""
    _, g, _ = built
    for graph in g.values():
        assert compute_depths(
            [n.gold_node_id for n in graph.gold_nodes],
            list(graph.gold_edges),
            list(graph.gold_seed_node_ids),
        ) == {n.gold_node_id: n.gold_depth for n in graph.gold_nodes}


def test_facets_exclude_the_depth_zero_frontier(built):
    """Same convention as synth_build: a seed belongs to no facet, so horizontal breadth and
    vertical depth stay separable numbers."""
    _, g, _ = built
    for graph in g.values():
        in_a_facet = {n for f in graph.gold_facets for n in f.gold_node_ids}
        for n in graph.gold_nodes:
            assert (n.gold_facet_id is None) == (n.gold_depth == 0)
            assert (n.gold_node_id in in_a_facet) == (n.gold_depth != 0)
    assert [f.gold_node_ids for f in g[C].gold_facets] == [("s3", "s4")]


def test_nodes_carry_the_human_composed_provenance_and_the_hop_answer(built):
    _, g, _ = built
    for graph in g.values():
        for n in graph.gold_nodes:
            assert n.gold_provenance_primary == "human_composed"
            assert n.gold_partition == "required"
            assert n.gold_ablation_verdict == "NECESSARY"
            assert len(n.gold_ev_uids) == 1
    # `.answer` strips the canary nonce the raw field carries; see GoldGraph.answer.
    assert g[A].answer == "Tomas Varga"
    assert "twenty-seven" in g[C].gold_aliases


# ------------------------------------------------------------------ uid agreement


def test_gold_ev_uids_are_uids_the_retriever_can_actually_return(built):
    """The silent-failure guard.

    Gold says "need v is resolved by uid U" and the rollout says "the retriever returned uid
    U". The two are minted by different modules that may not import each other, so if the
    (doc_id, span) convention ever drifts, every node becomes unresolvable and RNR reads 0.0
    with nothing in any plot to say why. This is the assertion that turns that into a
    failing build.
    """
    suite, g, _ = built
    for tid in suite.task_ids():
        available = {u.uid for u in suite.units(tid)}
        gold_uids = {u for n in g[tid].gold_nodes for u in n.gold_ev_uids}
        assert gold_uids and gold_uids <= available, tid


# ------------------------------------------------------------------ retrieval


def _resolved(step_text, graph):
    """Substitute "#N" with step N's answer, as a competent policy would before querying."""
    answers = {
        n.gold_node_id: (n.gold_aliases[0] if n.gold_aliases else "") for n in graph.gold_nodes
    }
    return re.sub(r"#(\d+)", lambda m: answers.get(f"s{m.group(1)}", m.group(0)), step_text)


def test_bm25_surfaces_the_supporting_paragraph_for_most_hops(built):
    """Not a claim that BM25 is good — a claim that the pool is retrievable at all.

    A hop is queried the way a policy would: the sub-question with its "#N" placeholder
    replaced by the previous hop's answer, since an unresolved "#1 >> spouse" carries no
    content words to match on. A floor rather than an equality, because BM25 on a
    20-paragraph pool is allowed to lose a hop or two; a big drop means the corpus or the
    tokenizer broke, which is what this is watching for.
    """
    suite, g, _ = built
    hits = total = 0
    for tid in suite.task_ids():
        r = suite.retriever(tid)
        for n in g[tid].gold_nodes:
            total += 1
            got = {u.uid for u in r.search(_resolved(n.gold_text, g[tid]), 3)}
            hits += set(n.gold_ev_uids) <= got
    assert total == 9
    assert hits / total >= 0.75, f"only {hits}/{total} supporting paragraphs in the top 3"


def test_bm25_returns_nothing_rather_than_padding_for_an_unmatched_query(built):
    """Padding a miss with the k least-bad paragraphs would hand a policy free evidence and
    make every discovery metric a function of pool size."""
    suite, _, _ = built
    assert suite.retriever(A).search("zzzzqqq unmatchedtokenxyz", 5) == ()


def test_bm25_ranking_is_deterministic_and_scored(built):
    suite, _, _ = built
    r = suite.retriever(B)
    a = r.search("Harrow Lights director", 5)
    b = r.search("Harrow Lights director", 5)
    assert [u.uid for u in a] == [u.uid for u in b]
    assert all(a[i].score >= a[i + 1].score for i in range(len(a) - 1))
    assert a[0].score > 0.0


# ------------------------------------------------------------------ leakage exclusion


def test_the_blocked_task_is_absent_from_the_corpus_and_recorded_in_the_audit(built):
    suite, g, res = built
    assert EXCLUDED not in suite.task_ids()
    assert EXCLUDED not in g
    assert res.n_tasks == 3 and res.n_excluded == 1
    audit = json.loads((res.gold.parent / "excluded_train.json").read_text())
    assert audit["excluded_task_ids"] == [EXCLUDED]
    assert audit["split"] == "train"


def test_the_sep_fold_is_what_makes_the_exclusion_list_match_at_all(built, request):
    """The seed datasets write "Hurghada [SEP] capital of"; MuSiQue writes
    "Hurghada >> capital of". Without folding those together the list matches 0 of 19,938
    train records and the contamination control is a no-op that still looks like it ran."""
    assert _norm_q("Hurghada [SEP] capital of") == _norm_q("Hurghada >> capital of")
    blocked = load_exclusions(
        request.config.rootpath
        / "tests"
        / "fixtures"
        / FIXTURE
        / "dev_test_singlehop_questions_v1.0.json"
    )
    assert _norm_q("Hurghada >> capital of") in blocked


def test_exclusion_is_not_applied_to_dev(tmp_path, request):
    """The list enumerates the single-hop questions USED in dev/test, so applying it to dev
    would delete the dev split entirely (measured on the real data: 2,417 of 2,417). It is a
    guard on what may be TRAINED on, not a filter on what may be evaluated.

    Same four fixture tasks, presented as the dev split: the task that train excludes must
    survive here, and the audit must record that no list was applied.
    """
    from pi_eval.build import musique_build

    fixture = request.config.rootpath / "tests" / "fixtures" / FIXTURE
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / musique_build.FILES["dev"]).write_bytes(
        (fixture / musique_build.FILES["train"]).read_bytes()
    )
    res = build(split="dev", raw_dir=raw, root=tmp_path, allow_download=False, verify=False)
    assert res.n_tasks == 4 and res.n_excluded == 0
    ids = {json.loads(x)["id"] for x in res.corpus.read_text().splitlines()}
    assert EXCLUDED in ids
    audit = json.loads((res.gold.parent / "excluded_dev.json").read_text())
    assert audit["exclusion_list"] is None and audit["n_excluded_tasks"] == 0


# ------------------------------------------------------------------ the loop


def test_run_loop_over_the_fixture_completes_and_produces_evidence(built):
    suite, g, _ = built
    for tid in suite.task_ids():
        ledger = BudgetLedger(cap=8)
        traj = run_loop(
            view=suite.view(tid),
            inquirer=VerbatimInquirer(max_asks=3),
            retriever=suite.retriever(tid),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=ledger,
            max_turns=8,
            k=4,
            seed=0,
        )
        assert traj.n_asks >= 1
        assert len(traj.evidence) > 0, tid
        assert traj.outcome.answer is not None
        assert traj.outcome.answer.n_words <= suite.view(tid).word_cap
        assert traj.stop_reason in ("policy_stop", "budget", "max_turns")
        assert traj.evidence.uids <= {u.uid for u in suite.units(tid)}


def test_prefix_of_a_real_rollout_is_still_a_monotone_evidence_ladder(built):
    suite, _, _ = built
    traj = run_loop(
        view=suite.view(B),
        inquirer=VerbatimInquirer(max_asks=3),
        retriever=suite.retriever(B),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=8),
        max_turns=8,
        k=4,
        seed=0,
    )
    prev = frozenset()
    for k in range(len(traj.turns) + 1):
        p = traj.prefix(k)
        assert p.stop_reason == "truncated_prefix"
        assert prev <= p.evidence.uids
        prev = p.evidence.uids


# ------------------------------------------------------------------ the real download


@pytest.mark.integration
def test_real_musique_train_builds_with_the_pinned_digest(tmp_path, request):
    """Runs only against a warm data/raw cache; CI never depends on the network."""
    from pi_eval.build import musique_build

    raw_src = request.config.rootpath / "data" / "raw" / FIXTURE
    name = musique_build.FILES["train"]
    if not (raw_src / name).exists() or not (raw_src / musique_build.EXCLUSION_NAME).exists():
        pytest.skip("no warm data/raw/musique cache")
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in (name, musique_build.EXCLUSION_NAME):
        (raw / f).symlink_to(raw_src / f)

    res = build(split="train", raw_dir=raw, root=tmp_path, limit=40, allow_download=False)
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines()]
    assert len(rows) == 40
    # "20 paragraphs per task" is nearly, but not exactly, universal. MEASURED over the real
    # train split: 19,917 of 19,938 tasks ship exactly 20 and 21 tasks ship 16-19. An
    # equality here would be a test that encodes a claim the data does not support.
    counts = [len(r["paragraphs"]) for r in rows]
    assert all(16 <= n <= 20 for n in counts)
    assert sum(n == 20 for n in counts) / len(counts) >= 0.95
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    assert len(graphs) == 40
    # every real task has at least one #N, so no task is edgeless
    assert all(g.gold_edges for g in graphs.values())
    suite = MusiqueSuite(res.corpus.parent)
    for tid in list(suite.task_ids())[:10]:
        available = {u.uid for u in suite.units(tid)}
        assert {u for n in graphs[tid].gold_nodes for u in n.gold_ev_uids} <= available


def test_selection_is_hop_stratified_so_the_depth_axis_exists():
    """Regression, and the nastiest defect found so far.

    MuSiQue's files are HOP-SORTED: the first 13,900 train records and the first 1,179 dev
    records are all 2hop (verified against the real downloads). The builder truncated head-N,
    so every sample was all-2hop, and a 2hop decomposition graph is depth<=1. That made
    `coverage_at_depth_ge2` -- a preregistered secondary endpoint, and the very axis the
    paper's central claim rests on -- structurally unmeasurable at ANY n. A 40-task head-N
    build had depth histogram {0: 40, 1: 40}: not one depth-2 node, and nothing in the output
    said so.
    """
    import collections

    from pi_eval.build.musique_build import _stratified_order

    # Hop-sorted, exactly like the real file.
    ids = (
        [f"2hop__{i}" for i in range(20)]
        + [f"3hop1__{i}" for i in range(20)]
        + [f"4hop2__{i}" for i in range(20)]
    )
    assert {i.split("__")[0] for i in ids[:6]} == {"2hop"}, "fixture reproduces hop-sorting"

    order = _stratified_order(ids, limit=6)
    got = collections.Counter(ids[i].split("__")[0] for i in order)
    assert len(order) == 6
    assert got == {"2hop": 2, "3hop1": 2, "4hop2": 2}, got

    # And it must stay balanced as n grows, not just at the boundary.
    order30 = _stratified_order(ids, limit=30)
    got30 = collections.Counter(ids[i].split("__")[0] for i in order30)
    assert got30 == {"2hop": 10, "3hop1": 10, "4hop2": 10}, got30

    # AND BALANCED AT EVERY PREFIX, which is the property that actually matters: every
    # consumer takes a prefix (`task_ids()` returns file order, `pi run --n N` slices it), so a
    # selection that is balanced only in aggregate is not stratified where it is used.
    for n in (3, 6, 9, 15, 30):
        prefix = collections.Counter(ids[i].split("__")[0] for i in order30[:n])
        assert max(prefix.values()) - min(prefix.values()) <= 1, (n, prefix)

    # This used to assert `order == sorted(order)`, on the belief that file order was needed to
    # keep corpus and gold "aligned row-for-row". That belief is false -- `public` and `graphs`
    # are indexed by this SAME list, so they align under any permutation -- and the sort it
    # justified undid the stratification in the written file, which is where every consumer
    # reads it from. Alignment is asserted directly instead.
    assert sorted(order) == sorted(set(order)), "no index selected twice"
    assert all(0 <= i < len(ids) for i in order)


def test_stratified_order_degrades_gracefully_on_one_class():
    from pi_eval.build.musique_build import _stratified_order

    ids = [f"2hop__{i}" for i in range(5)]
    assert _stratified_order(ids, limit=3) == [0, 1, 2]
    assert _stratified_order(ids, limit=99) == [0, 1, 2, 3, 4]


def test_corpus_and_gold_stay_aligned_under_the_round_robin_order(tmp_path, request):
    """The property the `sorted()` was wrongly defending. `public` and `graphs` are indexed by
    the SAME selection, so a permutation cannot separate them -- asserted on a real build
    rather than on the argument."""
    import json

    from pi_eval.build.common import read_graphs
    from pi_eval.build.musique_build import build

    raw = tmp_path / "raw"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    res = build(
        split="train", raw_dir=raw, root=tmp_path, limit=12, allow_download=False, verify=False
    )
    tasks = [json.loads(x) for x in res.corpus.read_text().splitlines() if x.strip()]
    graphs = read_graphs(res.gold)
    assert [t["id"] for t in tasks] == [g.gold_task_key for g in graphs]
    # And the written order is the ROUND-ROBIN one, not the file order it was selected from.
    hops = [t["id"].split("__")[0] for t in tasks]
    assert len(set(hops[:6])) > 1, f"the written prefix is single-class: {hops[:6]}"


@pytest.mark.integration
def test_the_built_corpus_is_hop_balanced_at_every_prefix_a_grid_uses():
    """`tier1_pilot` takes n=120 and `tier1_confirmatory` n=200, both as a PREFIX of
    `task_ids()`. Measured before the fix: the first 40 were 100% 2hop and the first 200 held
    no 4hop task at all, on a corpus built specifically to make depth>=2 measurable."""
    import collections
    from pathlib import Path as _P

    from pinq_adapters.musique.suite import MusiqueSuite

    dirs = sorted(_P("data/corpora/musique").glob("*/tasks.jsonl"))
    if len(dirs) != 1:
        pytest.skip(f"expected one built musique corpus, found {len(dirs)}")
    ids = list(MusiqueSuite(dirs[0].parent).task_ids())
    assert len(ids) >= 500

    for n in (40, 120, 200, 500):
        hops = collections.Counter(i.split("__")[0] for i in ids[:n])
        assert len(hops) >= 5, f"prefix n={n} covers only {sorted(hops)}"
        assert max(hops.values()) - min(hops.values()) <= 2, (n, dict(hops))
