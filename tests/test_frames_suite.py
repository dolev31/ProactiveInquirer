"""FRAMES: an external, eval-only, gold-FREE suite, and the four things that can go wrong.

FRAMES (Google DeepMind, NAACL 2025, arXiv:2409.12941, Apache-2.0) is the second benchmark
in this project we did not build. 824 hand-written multi-hop questions, each shipping the set
of Wikipedia articles that answer it; the "oracle articles" setting turns that into a closed
per-question paragraph pool, which is the shape `ParagraphSuite` already serves.

Four properties are pinned here because each fails silently rather than loudly:

  * it may never train. `EVAL_ONLY_SUITES` is the only thing standing between "the second
    benchmark we did not build" and a number that merely looks external.
  * the corpus is fetched from a LIVE, MUTABLE third party. Every other suite in this repo
    reads bytes we downloaded once and pinned; Wikipedia moves under you. The per-page
    sha256 in the manifest, verified at load, is what makes "the bytes I scored are the bytes
    I fetched" checkable rather than assumed.
  * it has no need-graph, so every structural metric is UNDEFINED, not zero. Six families
    would otherwise reach scores.parquet as publishable-looking zeros.
  * the gold answer must never cross into the view. It is in the gold tree; the public record
    key set is asserted to be exactly RECORD_KEYS, the same assertion every other suite makes.

The fixture is four hand-written rows in upstream's TSV shape plus three canned page-cache
files, so the whole build runs offline. Row 3 names a page that is not in the cache: that is
the deliberate fetch failure, and it must be NAMED rather than dropped.
"""

import json
import shutil

import pytest

FIXTURE = "frames"

# Upstream's own column names. None of these may appear in the public corpus.
UPSTREAM_GOLD_KEYS = ("Answer", "answer", "gold_answer", "reasoning_types")

# Fixture task ids and the gold answers that must never be visible to the policy.
T_JANE = "frames_0000"
T_TABLE = "frames_0001"
T_TEMPORAL = "frames_0002"
T_MISSING = "frames_0003"


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    """Copy the raw fixture, run the REAL builder offline, return (suite, graphs, result)."""
    from pi_eval.build.common import read_graphs
    from pi_eval.build.frames_build import build
    from pinq_adapters.frames.suite import FramesSuite

    root = tmp_path_factory.mktemp("frames")
    raw = root / "data" / "raw" / "frames"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    # verify=False: the fixture is a hand-written stand-in and cannot carry upstream's digest.
    res = build(root=root, allow_download=False, verify=False)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    return FramesSuite(res.corpus.parent), graphs, res


def _manifest(res):
    return json.loads((res.corpus.parent / "manifest.json").read_text())


# ------------------------------------------------------------------ the registry and the split


def test_the_registry_audit_stays_clean_with_frames_registered():
    from pi_run.cmd_suites import _loadable
    from pi_run.suites import REGISTRY, audit

    assert "frames" in REGISTRY
    assert "frames" in _loadable(), "load_suite must be able to construct it"
    assert audit(_loadable()) == []


def test_frames_declares_itself_gold_free_and_the_scorer_agrees():
    """Two trees must agree on one fact without importing each other -- the same shape as
    `audit_against_prereg`. A registry that says gold-free while the scorer computes a need
    graph, or the reverse, is a silent wrong number in either direction."""
    from pi_eval.score import GOLD_FREE_SUITES
    from pi_run.suites import REGISTRY

    assert REGISTRY["frames"].gold_free is True
    declared = {s.suite_id for s in REGISTRY.values() if s.gold_free}
    assert declared == set(GOLD_FREE_SUITES)


def test_frames_is_eval_only_and_every_task_is_test():
    from pinq.splitting import EVAL_ONLY_SUITES, split_of

    assert "frames" in EVAL_ONLY_SUITES
    for i in range(500):
        assert split_of("frames", f"frames_{i:04d}") == "test"


def test_frames_refuses_to_produce_a_training_row():
    from pinq_train.split import SplitViolation, assert_trainable

    with pytest.raises(SplitViolation, match="eval-only"):
        assert_trainable("frames", T_JANE)


def test_the_registry_names_the_tasks_whose_pages_could_not_be_fetched(built):
    """NAMED, not counted. A count lets a new gap hide behind an old one -- the reason
    `gold_gap_tasks` is a tuple of ids everywhere else in this registry."""
    _, _, res = built
    man = _manifest(res)
    assert man["failures"], "the fixture contains a deliberately missing page"
    assert [f["title"] for f in man["failures"]] == ["Fixture Page That Fails"]
    assert man["tasks_with_missing_pages"] == [T_MISSING]


# ------------------------------------------------------------------ the firewall


def test_public_record_has_exactly_the_allowed_keys(built):
    from pinq_adapters.paragraphs import PARAGRAPH_KEYS, RECORD_KEYS

    _, _, res = built
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines() if x]
    assert rows
    for r in rows:
        assert set(r) == RECORD_KEYS
        for p in r["paragraphs"]:
            assert set(p) == PARAGRAPH_KEYS


def test_no_gold_key_name_appears_anywhere_in_the_public_corpus(built):
    _, _, res = built
    blob = res.corpus.read_text()
    for key in UPSTREAM_GOLD_KEYS:
        assert f'"{key}"' not in blob


def test_the_view_carries_no_answer_and_no_gold_answer_text(built):
    """The gold answer is in the gold tree and nowhere the policy can reach.

    Both halves matter: no FIELD called answer (the type wall would catch that), and no
    OCCURRENCE of the answer string in any field (only a string scan can catch that)."""
    import dataclasses

    suite, graphs, _ = built
    view = suite.view(T_JANE)
    assert not hasattr(view, "answer")
    rendered = json.dumps(dataclasses.asdict(view), ensure_ascii=False)
    assert graphs[T_JANE].answer == "Jane Eyre"
    assert "Jane Eyre" not in rendered


def test_the_gold_answer_is_not_in_the_paragraph_pool_metadata(built):
    """A weaker scan than the above and a different failure: the ANSWER may legitimately
    appear inside a paragraph (that is what the pool is for); what must not appear is an
    answer FIELD alongside it."""
    suite, _, _ = built
    for tid in suite.task_ids():
        for u in suite.units(tid):
            assert set(dataclasses_fields(u)) == {
                "uid",
                "corpus_id",
                "doc_id",
                "span",
                "title",
                "text",
                "score",
            }


def dataclasses_fields(obj):
    import dataclasses

    return {f.name for f in dataclasses.fields(obj)}


# ------------------------------------------------------------------ the mutable-corpus guard


def test_a_tampered_page_is_refused_at_load(built, tmp_path):
    """Wikipedia is the only corpus in this repo that can change under a finished run.

    `corpus_hash` pins the directory NAME, which catches a rebuild; it cannot catch a byte
    edited inside a corpus directory that keeps its name. The per-page sha256 can."""
    from pinq_adapters.frames.suite import CorpusTampered, FramesSuite

    _, _, res = built
    work = tmp_path / "tampered"
    shutil.copytree(res.corpus.parent, work)

    rows = [json.loads(x) for x in (work / "tasks.jsonl").read_text().splitlines() if x]
    rows[0]["paragraphs"][0]["text"] = rows[0]["paragraphs"][0]["text"] + " (edited)"
    (work / "tasks.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n"
    )

    with pytest.raises(CorpusTampered, match="sha256"):
        FramesSuite(work)


def test_an_untampered_corpus_loads(built):
    suite, _, res = built
    assert len(suite.task_ids()) == 4
    assert suite.corpus_hash


# ------------------------------------------------------------------ retrieval and uids


def test_the_retriever_returns_uids_of_the_frames_shape(built):
    """The adapter's uid and the gold side's `unit_uid` are minted by two different code
    paths and must agree to the character; a drift is invisible in every downstream number."""
    from pi_eval.build.common import unit_uid
    from pinq_adapters.frames.suite import CORPUS_ID

    suite, _, _ = built
    hits = suite.retriever(T_JANE).search("novel published in 1847 Currer Bell", k=3)
    assert hits, "BM25 over the task's own pool must find the 1847 paragraph"
    pool = {u.uid: u for u in suite.units(T_JANE)}
    for h in hits:
        assert h.uid in pool
        assert h.corpus_id == CORPUS_ID
        assert h.doc_id.startswith(f"{T_JANE}:")
        assert h.span == f"0:{len(h.text)}"
        idx = int(h.doc_id.split(":")[-1])
        assert h.uid == unit_uid(CORPUS_ID, T_JANE, idx, h.text)


def test_the_pool_is_the_tasks_own_articles_only(built):
    """Closed per-question pool, exactly as in musique: a task that never linked Jane Eyre
    must not be able to retrieve from it."""
    suite, _, _ = built
    titles = {u.title for u in suite.units(T_TEMPORAL)}
    assert titles == {"Charlotte Brontë"}
    assert {u.title for u in suite.units(T_JANE)} == {"Charlotte Brontë", "Jane Eyre"}


def test_a_table_row_survives_the_build_as_its_own_paragraph(built):
    """28.6% of FRAMES questions are tabular. A plain-text extract drops every table, which
    makes those questions unanswerable from the pool by construction rather than by policy."""
    suite, _, _ = built
    texts = [u.text for u in suite.units(T_TABLE)]
    assert any("Floors: 104" in t for t in texts)


# ------------------------------------------------------------------ gold-free scoring


def _gold_free_graph():
    from pi_eval.gold import GoldGraph

    return GoldGraph(
        gold_suite="frames",
        gold_task_key=T_JANE,
        gold_nodes=(),
        gold_answer="Jane Eyre",
        gold_graph_version="v1",
    )


def _graph_with_a_need():
    from pi_eval.gold import GoldGraph, GoldNode

    return GoldGraph(
        gold_suite="musique",
        gold_task_key="t",
        gold_nodes=(
            GoldNode(
                gold_suite="musique",
                gold_task_key="t",
                gold_node_id="n1",
                gold_text="n1",
                gold_partition="required",
                gold_discoverability="kb",
                gold_ev_uids=("u1",),
            ),
        ),
        gold_answer="Jane Eyre",
    )


def _rows(graph, *, gold_free, suite_id):
    from pi_eval.score import AnswerRecord, score_run

    run = {
        "run_id": "r",
        "suite_id": suite_id,
        "task_id": "t",
        "arm_id": "a",
        "usd": 0.25,
        "n_turns": 3,
        "wall_ms": 1234.0,
        "retrieval_calls": 3,
    }
    return score_run(
        run,
        graph=graph,
        turns=[{"turn_idx": 0, "retrieved_uids": ["u1"], "new_uids": ["u1"]}],
        evidence=[{"uid": "u1"}],
        env_calls=[],
        ledger=[],
        records=[],
        answer=AnswerRecord(text="The novel is Jane Eyre.", cited_uids=()),
        gold_free=gold_free,
    )


# Every structural family that a node-free graph would otherwise emit as a zero nobody
# measured. Each was verified present in the gold-bearing path before this flag existed.
FORBIDDEN_WITHOUT_A_NEED_GRAPH = (
    "n_needs_resolved",
    "max_depth_reached",
    "facet_breadth",
    "facet_total",
    "stop_overshoot",
    "stop_undershoot",
)


def test_gold_free_scoring_emits_the_answer_metrics(built):
    by = {
        r["metric_name"]: r["value"]
        for r in _rows(_gold_free_graph(), gold_free=True, suite_id="frames")
    }
    assert by["answer_correct"] == 1.0
    assert by["answer_token_f1"] > 0.0
    assert by["answer_token_recall"] == 1.0
    assert by["answer_hedged"] == 0.0


def test_gold_free_scoring_emits_the_cost_and_turn_counts(built):
    by = {
        r["metric_name"]: r["value"]
        for r in _rows(_gold_free_graph(), gold_free=True, suite_id="frames")
    }
    assert by["usd"] == 0.25
    assert by["n_turns"] == 3.0
    assert by["retrieval_calls"] == 3.0
    assert by["answer_n_words"] == 5.0


def test_gold_free_scoring_emits_no_structural_metric():
    names = {r["metric_name"] for r in _rows(_gold_free_graph(), gold_free=True, suite_id="frames")}
    for n in FORBIDDEN_WITHOUT_A_NEED_GRAPH:
        assert n not in names, f"{n} is undefined without a need-graph; a 0.0 there is a lie"
    assert not any(n.startswith("cad") for n in names)
    assert not any(n.startswith("frontier_") for n in names)
    assert not any(n.startswith("phi_") for n in names)
    assert not any(n.startswith("rnr_") for n in names)
    assert "evidence_coverage" not in names
    assert "task_success" not in names


def test_the_flag_is_off_by_default_and_changes_nothing_for_a_gold_bearing_suite():
    """The isolation claim, asserted rather than argued: the same graph scored with the flag
    absent and with it explicitly False must produce identical rows."""
    from pi_eval.score import score_run

    g = _graph_with_a_need()
    explicit = _rows(g, gold_free=False, suite_id="musique")
    run = {
        "run_id": "r",
        "suite_id": "musique",
        "task_id": "t",
        "arm_id": "a",
        "usd": 0.25,
        "n_turns": 3,
        "wall_ms": 1234.0,
        "retrieval_calls": 3,
    }
    from pi_eval.score import AnswerRecord

    default = score_run(
        run,
        graph=g,
        turns=[{"turn_idx": 0, "retrieved_uids": ["u1"], "new_uids": ["u1"]}],
        evidence=[{"uid": "u1"}],
        env_calls=[],
        ledger=[],
        records=[],
        answer=AnswerRecord(text="The novel is Jane Eyre.", cited_uids=()),
    )
    assert explicit == default
    names = {r["metric_name"] for r in default}
    for n in FORBIDDEN_WITHOUT_A_NEED_GRAPH:
        assert n in names, f"{n} must still be emitted where a need-graph exists"
    assert "evidence_coverage" in names


def test_gold_free_is_stamped_in_the_score_result():
    """`pi score` must SAY it took the answers-only path. A table whose provenance does not
    record which scorer branch produced it is a number without provenance."""
    from pi_eval.score import ScoreResult

    res = ScoreResult(
        parquet_dir="d",
        scorer_hash="h",
        metric_defs_hash="m",
        graph_hash="g",
        matcher_hash="x",
        judge_pins=(),
        graph_version="v1",
        n_runs_scored=1,
        n_runs_skipped_no_graph=0,
        scores_rows_added=1,
        scores_rows_total=1,
        matches_rows_added=0,
        matches_rows_total=0,
        metrics_emitted=("answer_correct",),
        answers_available=1,
        gold_free_suites=("frames",),
    )
    d = res.as_dict()
    assert d["gold_free"] is True
    assert d["gold_free_suites"] == ["frames"]


# ------------------------------------------------------------------ the gold side


def test_the_gold_graph_carries_an_answer_and_no_need_graph(built):
    _, graphs, _ = built
    assert set(graphs) == {T_JANE, T_TABLE, T_TEMPORAL, T_MISSING}
    g = graphs[T_JANE]
    assert g.gold_nodes == ()
    assert g.gold_edges == ()
    assert g.required() == ()
    assert g.answer == "Jane Eyre"


def test_the_gold_answer_carries_a_canary(built):
    """Firewall layer 4. Layers 1-3 stop a leak through CODE; only the nonce catches a leak
    through a STRING, and it is minted by `write_graphs` so it is a property of "gold was
    written" rather than of "the builder remembered to"."""
    _, graphs, _ = built
    g = graphs[T_JANE]
    assert g.gold_canary.startswith("PINQCANARY_")
    assert g.gold_canary in g.gold_answer
    assert g.gold_canary not in g.answer


def test_the_gold_names_the_corpus_it_was_built_from(built):
    _, graphs, res = built
    assert graphs[T_JANE].gold_corpus_hash == res.corpus_hash
    assert res.corpus.parent.name == res.corpus_hash


# ------------------------------------------------------------------ the grid


def test_the_frames_grid_loads_and_is_held_out():
    from pi_run import grids
    from pinq_expt import arms as arm_table

    g = grids.load("conf/grids/frames_trained.yaml")
    assert g.suites == ("frames",)
    assert g.split == "test"
    assert g.n_tasks == 824
    assert g.seeds == (0,)
    assert g.budget_cap == 8
    assert g.max_turns == 16
    assert g.k == 5
    assert set(g.arms) <= set(arm_table.arm_ids())
    assert "inquirer_trained" in g.arms
    assert "PI_MODEL_INQUIRER" in g.notes


# ------------------------------------------------------------------ the loader


def test_load_suite_constructs_frames_from_a_corpus_directory(built):
    from pi_run.worker import CORPUS_BACKED, load_suite

    assert "frames" in CORPUS_BACKED
    _, _, res = built
    s = load_suite("frames", str(res.corpus.parent))
    assert len(s.task_ids()) == 4
    assert s.view(T_JANE).word_cap == 50


# ------------------------------------------------------------------ the link column


def test_a_search_box_url_never_becomes_a_pool_entry():
    """frames_0088 links a SEARCH url whose query string carries `title=Special:Search`.

    An `index.php?title=` reader that does not know about namespaces resolves that to the
    article "Special:Search", fetches the search page, and puts it in that task's evidence
    pool as though upstream had linked it. Measured before the namespace guard existed:
    `title_of(...) == 'Special:Search'`. A pool entry the benchmark never pointed at is
    invisible in every downstream number, which is why this is a refusal and not a guess.
    """
    from pi_eval.build.frames_build import title_of

    assert (
        title_of(
            "https://en.wikipedia.org/w/index.php?search=Polytrichum+piliferum"
            "&title=Special:Search&profile=advanced&fulltext=1&ns0=1"
        )
        is None
    )


def test_the_two_recoverable_link_defects_are_recovered():
    """A stated title is recovered; an inferred one is not. That is the whole rule."""
    from pi_eval.build.frames_build import title_of

    assert title_of("en.wikipedia.org/wiki/Grazia_Deledda") == "Grazia Deledda"
    assert title_of("https://en.wikipedia.org/w/index.php?title=Bronco&redirect=no") == "Bronco"
    assert title_of("https://en.m.wikipedia.org/wiki/Jane_Eyre") == "Jane Eyre"
    assert title_of("https://en.wikipedia.org/wiki/Charlotte_Bront%C3%AB") == "Charlotte Brontë"
    # Refused, and named in the manifest instead.
    assert title_of("https://w.wiki/ASFv") is None
    assert title_of("https://simple.wikipedia.org/wiki/Video_assistant_referee") is None


def test_a_table_row_keeps_its_column_headers():
    """A cell without its column name is an unlabelled number. Headers are declared once per
    table and carried down every row, which is what makes a rank table answerable."""
    from pi_eval.build.frames_build import flatten_html

    doc = """
    <div><table class="wikitable"><caption>Tallest buildings</caption>
    <tr><th>Rank</th><th>Name</th><th>Floors</th></tr>
    <tr><td>37</td><td>56 Leonard Street, a residential tower in Tribeca</td><td>57</td></tr>
    </table></div>
    """
    paras = flatten_html(doc)
    assert paras == [
        "Tallest buildings | Rank: 37 | Name: 56 Leonard Street, a residential tower in "
        "Tribeca | Floors: 57"
    ]


def test_navigation_furniture_is_not_evidence():
    from pi_eval.build.frames_build import flatten_html

    doc = """
    <div><p>This paragraph is long enough to survive the minimum length floor that the
    builder applies to every candidate paragraph it produces.</p>
    <table class="navbox"><tr><td>Presidents of the United States - Washington - Adams -
    Jefferson - Madison - Monroe - Adams - Jackson - Van Buren</td></tr></table>
    <div class="reflist"><p>1. Some reference that is quite long and carries no facts at
    all about the subject of the article in question here.</p></div></div>
    """
    paras = flatten_html(doc)
    assert len(paras) == 1
    assert paras[0].startswith("This paragraph is long enough")


# ------------------------------------------------------------------ the contamination check


def _contam():
    """The check is a script, not a package. Import it by path rather than adding scripts/ to
    sys.path permanently -- it must stay importable with PI_GOLD_ROOT unset, which is half
    the reason it is a script in the first place."""
    import importlib.util
    import pathlib

    here = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_check_frames_contamination", here / "scripts" / "check_frames_contamination.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_contamination_normaliser_matches_the_scorers():
    """The check duplicates `pi_eval.metrics.quality.normalize` because a script outside
    pi_eval may not import it. A duplicated definition that drifts would answer a different
    question than the one the paper claims was asked, so the two are pinned equal here."""
    from pi_eval.metrics.quality import normalize as gold_side

    ours = _contam().normalize
    for s in (
        "Who is the spouse of the director of A Film?",
        "  THE  quick,  brown; fox -- an apple!  ",
        "What is the capital of the 15th state?",
        "",
        "A an the",
    ):
        assert ours(s) == gold_side(s), s


def test_an_identical_question_is_caught_as_exact_not_merely_near():
    m = _contam()
    q = "Who is the spouse of the director of the 1994 film?"
    assert m.normalize(q) == m.normalize("Who is the spouse of the director of the 1994 film?")
    a = frozenset(m.normalize(q).split())
    assert m.jaccard(a, a) == 1.0


def test_a_reordered_question_still_scores_high_jaccard():
    """Set, not sequence: a paraphrase that reorders clauses is contamination too."""
    m = _contam()
    a = frozenset(m.normalize("Who directed the film whose composer won in 1994?").split())
    b = frozenset(m.normalize("In 1994 whose composer won, and who directed that film?").split())
    assert m.jaccard(a, b) >= 0.8


# ------------------------------------------------------------------ an LLM-free rollout


def test_run_loop_over_the_fixture_completes_and_produces_evidence(built):
    """The whole point of choosing FRAMES: it runs through the EXISTING harness unchanged.

    No LLM and no money -- `pinq_expt.fakes` supplies all three policies -- but this is the
    real `run_loop`, the real BM25 retriever and the real budget ledger, so a suite that could
    not be driven by the harness fails here rather than on the first paid campaign.
    """
    from pinq.budget import BudgetLedger
    from pinq.loop import run_loop
    from pinq_expt.fakes import EchoDrafter, FrozenAnswerer, VerbatimInquirer

    suite, _, _ = built
    for tid in suite.task_ids():
        if not suite.units(tid):
            continue  # frames_0003: its only article failed to fetch. Named, not dropped.
        traj = run_loop(
            view=suite.view(tid),
            inquirer=VerbatimInquirer(max_asks=3),
            retriever=suite.retriever(tid),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=BudgetLedger(cap=8),
            max_turns=8,
            k=5,
            seed=0,
        )
        assert traj.n_asks >= 1 and len(traj.evidence) > 0, tid
        assert traj.evidence.uids <= {u.uid for u in suite.units(tid)}
        assert traj.outcome.answer.n_words <= suite.view(tid).word_cap


def test_a_task_whose_pages_all_failed_has_an_empty_pool_and_still_exists(built):
    """The denominator must not be a function of Wikipedia's availability on build day.

    frames_0003 links one article that the fixture cache does not hold. It is KEPT, with an
    empty pool, and named in the manifest -- so a downstream table reports 824 tasks of which
    N had an incomplete pool, rather than quietly reporting 824-N and calling it 824.
    """
    suite, graphs, _ = built
    assert T_MISSING in suite.task_ids()
    assert suite.units(T_MISSING) == ()
    assert graphs[T_MISSING].answer == "Nowhere"
    assert suite.retriever(T_MISSING).search("anything at all", k=5) == ()


# ------------------------------------------------------------------ pi suites validate


def test_validate_reports_the_gold_ANSWERS_rather_than_claiming_there_is_no_gold(
    built, monkeypatch, tmp_path
):
    """`uses_gold_graph=False` made validate print "scored inside its own environment, not
    against a need-graph" and return. True of userbench, FALSE of frames: frames is scored by
    `pi score` against a gold ANSWER file, and printing that it has no gold means nobody
    notices when that file is missing or describes a different corpus -- which would silently
    mis-score every answer metric, the only metrics the suite has.

    The corpus-hash join is the part that matters. `pi score` refuses a mismatch, but it
    refuses it after a campaign has been paid for; this is the same check, before.
    """
    from pi_run.cmd_suites import _validate_one

    _, _, res = built
    # tasks.jsonl / <hash> / frames / corpora / data / <root>
    root = res.corpus.parents[4]
    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    r = _validate_one(root, "frames", k=5, graph_version="v1")

    assert "own environment" not in str(r["gold_graphs"])
    assert r["gold_answers"] == 4
    assert not r["blockers"], r["blockers"]


def test_validate_blocks_when_the_gold_answers_describe_another_corpus(
    built, monkeypatch, tmp_path
):
    """The negative control for the test above: it must actually bite."""
    import json
    import shutil

    from pi_run.cmd_suites import _validate_one

    _, _, res = built
    src_root = res.corpus.parents[4]
    root = tmp_path / "stale"
    shutil.copytree(src_root / "data", root / "data")

    gp = root / "data" / "gold" / "graphs" / "frames" / "v1.jsonl"
    rows = [json.loads(x) for x in gp.read_text().splitlines() if x]
    for row in rows:
        row["gold_corpus_hash"] = "0123456789abcdef"
    gp.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows))

    monkeypatch.setenv("PI_GOLD_ROOT", str(root / "data" / "gold"))
    r = _validate_one(root, "frames", k=5, graph_version="v1")
    assert any("gold describes corpus" in b for b in r["blockers"]), r["blockers"]


# ------------------------------------------------------------------ upstream link defects
#
# All three were found by running the real build over all 824 tasks and reading the failure
# list, which is what the manifest's `failures` block is for. Each cost real articles: 7
# comma-packed elements hid ~15 articles, and without them a task's pool is silently short.


def test_one_list_element_holding_several_urls_is_split():
    """MEASURED upstream, 7 rows: a single `wiki_links` element packs two or three URLs
    separated by ", ". Read as one string it resolves to the title
    "American Family Field, https://en.wikipedia.org/wiki/LoanDepot Park, ..." which does not
    exist, so all three articles go missing and the task runs with an incomplete pool."""
    from pi_eval.build.frames_build import split_links

    packed = (
        "https://en.wikipedia.org/wiki/American_Family_Field, "
        "https://en.wikipedia.org/wiki/LoanDepot_Park, "
        "https://en.wikipedia.org/wiki/Globe_Life_Field, "
    )
    assert split_links(packed) == [
        "https://en.wikipedia.org/wiki/American_Family_Field",
        "https://en.wikipedia.org/wiki/LoanDepot_Park",
        "https://en.wikipedia.org/wiki/Globe_Life_Field",
    ]
    plain = "https://en.wikipedia.org/wiki/Jane_Eyre"
    assert split_links(plain) == [plain]


def test_a_double_encoded_title_is_decoded():
    """MEASURED upstream: `2021_French_Open_%E2%80%93_Men%2527s_singles`. One unquote leaves
    `Men%27s`, which is not an article. A second is applied only when a %XX escape survives
    the first, so a title that legitimately ends in a percent sign is untouched."""
    from pi_eval.build.frames_build import title_of

    assert (
        title_of("https://en.wikipedia.org/wiki/2021_French_Open_%E2%80%93_Men%2527s_singles")
        == "2021 French Open – Men's singles"
    )
    # Single-encoded, the common case, must be unchanged by the second pass.
    assert (
        title_of("https://en.wikipedia.org/wiki/2016_Australian_Open_%E2%80%93_Men%27s_singles")
        == "2016 Australian Open – Men's singles"
    )
    # A literal percent that is not an escape must survive.
    assert title_of("https://en.wikipedia.org/wiki/Percentage_%28%25%29") == "Percentage (%)"


def test_an_article_created_after_the_pin_date_falls_back_and_is_named(monkeypatch):
    """MEASURED, 5 articles: `Agnieszka Kotlarska` exists (pageid 77891779) and has no
    revision at or before 2024-09-01 -- it was created later. Refusing it drops an article the
    benchmark points at; taking it silently would hide that one page in the pool postdates the
    pin every other page is held to. So it is taken AND named in the manifest.
    """
    from pi_eval.build import frames_build as fb

    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if "rvdir=older" in url:  # the pinned lookup finds nothing
            return b'{"query":{"pages":[{"title":"X","revisions":[]}]}}'
        if "rvdir=newer" in url:  # the fallback: the oldest revision that exists
            return b'{"query":{"pages":[{"title":"X","revisions":[{"revid":9,"timestamp":"2025-06-01T00:00:00Z"}]}]}}'
        raise AssertionError(url)

    monkeypatch.setattr(fb, "_get", fake_get)
    revid, ts, after_pin = fb.resolve_revision("X")
    assert (revid, ts, after_pin) == (9, "2025-06-01T00:00:00Z", True)
    assert any("rvdir=older" in c for c in calls) and any("rvdir=newer" in c for c in calls)


def test_a_pinned_revision_is_not_flagged_as_after_the_pin(monkeypatch):
    from pi_eval.build import frames_build as fb

    monkeypatch.setattr(
        fb,
        "_get",
        lambda url, **kw: (
            b'{"query":{"pages":[{"title":"X","revisions":'
            b'[{"revid":7,"timestamp":"2024-08-31T00:00:00Z"}]}]}}'
        ),
    )
    assert fb.resolve_revision("X") == (7, "2024-08-31T00:00:00Z", False)
