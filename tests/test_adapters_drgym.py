"""DeepResearchGym: the envelope, the 401, the offline replay, the flat graph, the emitter.

EVERY TEST IN THIS FILE RUNS WITH NO NETWORK AND NO API KEY. That is not a convenience: this
is the only OPEN-corpus suite here, so "the tests pass" has to mean "the replay path is
sound" rather than "the service was up this afternoon". Anything that needs the real key or
the real service is marked `integration` and is excluded from the default run.

The fixture is three hand-written Researchy-shaped queries and their aggregated key points,
one of which ships EMPTY exactly as 24 of the 1000 upstream files do.
"""

import json
import shutil

import pytest

from pi_eval.build.common import read_graphs
from pi_eval.build.drgym_build import BENCH_COMMIT, QUERIES, build
from pi_eval.gold import GoldEdge, GoldNode
from pi_run.cache import DiskCache
from pinq.budget import BudgetLedger
from pinq.ids import evidence_uid
from pinq.loop import run_loop
from pinq.view import LeakageError, make_view
from pinq_adapters.drgym.cache import (
    CachingSearch,
    ReplaySearch,
    SearchCacheMiss,
    ShardedJsonCache,
    record_of,
    search_key,
)
from pinq_adapters.drgym.client import (
    CORPUS_IDS,
    KEY_REMEDY,
    DrGymAuthError,
    DrGymClient,
    DrGymEnvelopeError,
    DrGymHTTPError,
    Response,
    corpus_of,
    decode_envelope,
    docs_from_envelope,
)
from pinq_adapters.drgym.report import (
    ReportWithoutCitations,
    emit,
    extract_urls,
    read_system_dir,
    sources_block,
)
from pinq_adapters.drgym.suite import RECORD_KEYS, WORD_CAP, DrGymSuite
from pinq_expt.fakes import EchoDrafter, FrozenAnswerer, VerbatimInquirer

FIXTURE = "drgym"
Q1, Q2, Q3 = "900001", "900002", "900003"


@pytest.fixture(scope="module")
def fixture_dir(request):
    return request.config.rootpath / "tests" / "fixtures" / FIXTURE


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("drgym")
    raw = root / "raw"
    shutil.copytree(request.config.rootpath / "tests" / "fixtures" / FIXTURE, raw)
    res = build(raw_dir=raw, root=root, allow_download=False, verify=False)
    graphs = {g.gold_task_key: g for g in read_graphs(res.gold)}
    return res, graphs


def _envelope(fixture_dir, name):
    return json.loads((fixture_dir / "search" / name).read_text())


def _seed(cache, corpus_name, query, k, envelope):
    spec = corpus_of(corpus_name)
    key = search_key(corpus=spec.name, endpoint=spec.path, query=query, k=k)
    cache.put(key, record_of(corpus=spec, query=query, k=k, envelope=envelope))
    return key


# ------------------------------------------------------------------ the envelope


def test_the_envelope_is_base64_encoded_json_per_result(fixture_dir):
    docs = decode_envelope(_envelope(fixture_dir, "fineweb_k2.json"))
    assert len(docs) == 2
    assert docs[0]["url"] == "https://example.org/oslo-car-free"


def test_a_string_body_is_parsed_before_the_envelope_is_read(fixture_dir):
    raw = (fixture_dir / "search" / "fineweb_k2.json").read_text()
    assert len(decode_envelope(raw)) == 2


def test_fineweb_field_names_normalize(fixture_dir):
    docs = docs_from_envelope(_envelope(fixture_dir, "fineweb_k2.json"), corpus_of("fineweb"))
    assert docs[0].doc_id.startswith("<urn:uuid:")
    assert docs[0].url == "https://example.org/oslo-car-free"
    assert docs[0].language == "en"
    assert "NO2" in docs[0].text


def test_clueweb_field_names_normalize_to_the_same_shape(fixture_dir):
    """URL/ClueWeb22-ID/Clean-Text/Language must land in exactly the fields `url`, `doc_id`,
    `text`, `language`, or every downstream consumer needs a per-corpus branch."""
    docs = docs_from_envelope(_envelope(fixture_dir, "clueweb22_k2.json"), corpus_of("clueweb22"))
    assert docs[0].doc_id == "clueweb22-en0000-00-00000"
    assert docs[0].url == "https://example.org/oslo-car-free"
    assert docs[0].language == "en"
    assert "NO2" in docs[0].text


def test_the_two_corpora_yield_the_same_normalized_text_for_the_same_page(fixture_dir):
    fw = docs_from_envelope(_envelope(fixture_dir, "fineweb_k2.json"), corpus_of("fineweb"))
    cw = docs_from_envelope(_envelope(fixture_dir, "clueweb22_k2.json"), corpus_of("clueweb22"))
    assert fw[0].text == cw[0].text and fw[0].url == cw[0].url


def test_units_use_the_repo_wide_uid_convention(fixture_dir):
    """uid is h(corpus_id, doc_id, span) and span is `0:len(text)` — the same convention
    pi_eval.build.common.unit_uid mints. A one-character drift here is invisible downstream."""
    doc = docs_from_envelope(_envelope(fixture_dir, "fineweb_k2.json"), corpus_of("fineweb"))[0]
    unit = doc.as_unit(CORPUS_IDS["fineweb"])
    assert unit.uid == evidence_uid(CORPUS_IDS["fineweb"], doc.doc_id, f"0:{len(doc.text)}")
    # The URL rides in `title`: it is the only handle a FineWeb record has, and both the
    # citation extractor and the Support rubric key off literal URLs.
    assert unit.title == doc.url


@pytest.mark.parametrize(
    "payload",
    [
        {"results": ["!!!! not base64 !!!!"]},
        {"results": ["eyJub3RfanNvbiI6"]},  # truncated base64 of JSON
        {"results": "not a list"},
        {"nope": []},
        [],
    ],
)
def test_a_broken_envelope_raises_instead_of_dropping_results(payload):
    """A silently dropped result is a silently shortened evidence set, and every retrieval
    number downstream is a function of that set."""
    with pytest.raises(DrGymEnvelopeError):
        decode_envelope(payload)


def test_a_document_without_text_is_an_error_not_an_empty_string():
    import base64

    doc = {"id": "x", "url": "https://example.org/a", "text": "", "language": "en"}
    env = {"results": [base64.b64encode(json.dumps(doc).encode()).decode()]}
    with pytest.raises(DrGymEnvelopeError):
        docs_from_envelope(env, corpus_of("fineweb"))


# ------------------------------------------------------------------ the 401


def test_a_missing_key_is_reported_as_a_remedy_not_a_traceback():
    c = DrGymClient(api_key="", env={})
    ok, why = c.available()
    assert not ok
    assert why == KEY_REMEDY
    assert "DRGYM_API_KEY" in why
    assert "deepresearchgym@cmu.edu" in why
    assert "offline=True" in why


def test_a_401_from_the_service_raises_with_the_same_remedy():
    """Both endpoints 401 without a key (verified by direct call), so a 401 is the NORMAL
    state of an unconfigured machine and must never read like a bug in this code."""
    calls = []

    def transport(url, headers, timeout):
        calls.append(url)
        return Response(401, b'{"detail":"Invalid or missing API Key"}')

    c = DrGymClient(api_key="a-key", env={}, transport=transport, sleep=lambda _s: None)
    with pytest.raises(DrGymAuthError) as exc:
        c.fetch("anything", 3)
    assert "401" in str(exc.value)
    assert "deepresearchgym@cmu.edu" in str(exc.value)
    assert len(calls) == 1, "a 401 is a configuration state; retrying it is pointless"


def test_fetch_without_a_key_never_reaches_the_transport():
    def transport(url, headers, timeout):  # pragma: no cover - must not run
        raise AssertionError("the client dialled out without a key")

    c = DrGymClient(api_key="", env={}, transport=transport)
    with pytest.raises(DrGymAuthError):
        c.fetch("q", 1)


def test_a_5xx_is_retried_and_a_4xx_is_not():
    seq = [Response(503, b"busy"), Response(200, b'{"results": []}')]

    def transport(url, headers, timeout):
        return seq.pop(0)

    c = DrGymClient(api_key="k", env={}, transport=transport, sleep=lambda _s: None)
    assert c.fetch("q", 1) == {"results": []}

    def bad(url, headers, timeout):
        return Response(422, b"bad query")

    c2 = DrGymClient(api_key="k", env={}, transport=bad, sleep=lambda _s: None)
    with pytest.raises(DrGymHTTPError):
        c2.fetch("q", 1)


def test_a_200_that_is_not_json_names_the_service_instead_of_leaking_a_decode_error():
    """A proxy or an error page wearing a success code. Retrying it would hide that."""

    def transport(url, headers, timeout):
        return Response(200, b"<html>gateway</html>")

    c = DrGymClient(api_key="k", env={}, transport=transport, sleep=lambda _s: None)
    with pytest.raises(DrGymEnvelopeError) as exc:
        c.fetch("q", 1)
    assert "/fineweb/search" in str(exc.value)


def test_the_key_is_sent_as_a_header_and_the_query_is_url_encoded():
    seen = {}

    def transport(url, headers, timeout):
        seen["url"] = url
        seen["headers"] = dict(headers)
        return Response(200, b'{"results": []}')

    c = DrGymClient(api_key="secret", env={}, transport=transport, base_url="https://h.invalid")
    c.fetch("cars & bikes", 4)
    assert seen["headers"]["X-API-Key"] == "secret"
    assert seen["url"] == "https://h.invalid/fineweb/search?query=cars+%26+bikes&k=4"


def test_the_corpus_is_swappable_and_picks_the_other_endpoint():
    c = DrGymClient(corpus="clueweb22", api_key="k", env={}, base_url="https://h.invalid")
    assert c.url_for("q", 2).startswith("https://h.invalid/search?")


# ------------------------------------------------------------------ offline replay


def test_offline_replay_returns_the_recorded_units(tmp_path, fixture_dir):
    cache = ShardedJsonCache(tmp_path / "cache")
    _seed(cache, "fineweb", "q", 2, _envelope(fixture_dir, "fineweb_k2.json"))
    r = ReplaySearch(cache, corpus=corpus_of("fineweb"), corpus_id=CORPUS_IDS["fineweb"])
    units = r.search("q", 2)
    assert len(units) == 2
    assert units[0].title.startswith("https://")


def test_a_replay_miss_raises_rather_than_returning_nothing(tmp_path):
    """An empty result would score as 'the policy found nothing', which is a different fact
    from 'this rollout was never recorded'."""
    r = ReplaySearch(
        ShardedJsonCache(tmp_path / "cache"),
        corpus=corpus_of("fineweb"),
        corpus_id=CORPUS_IDS["fineweb"],
    )
    with pytest.raises(SearchCacheMiss):
        r.search("never recorded", 2)


def test_the_search_cache_is_byte_compatible_with_pi_run_disk_cache(tmp_path, fixture_dir):
    """The runner passes pi_run.cache.DiskCache straight in. This asserts that claim rather
    than restating it: write with one class, read with the other, and compare the bytes."""
    root = tmp_path / "cache"
    rec = record_of(
        corpus=corpus_of("fineweb"),
        query="q",
        k=2,
        envelope=_envelope(fixture_dir, "fineweb_k2.json"),
    )
    sha = search_key(corpus="fineweb", endpoint="/fineweb/search", query="q", k=2)
    DiskCache(root).put(sha, rec)
    from_ours = ShardedJsonCache(root).get(sha)
    assert from_ours == rec
    assert DiskCache(root).path(sha) == ShardedJsonCache(root).path(sha)

    other = tmp_path / "cache2"
    ShardedJsonCache(other).put(sha, rec)
    assert DiskCache(other).path(sha).read_bytes() == DiskCache(root).path(sha).read_bytes()


def test_the_cache_key_ignores_the_host_but_not_the_corpus_or_k():
    a = search_key(corpus="fineweb", endpoint="/fineweb/search", query="q", k=5)
    assert a == search_key(corpus="fineweb", endpoint="/fineweb/search", query="q", k=5)
    assert a != search_key(corpus="clueweb22", endpoint="/search", query="q", k=5)
    assert a != search_key(corpus="fineweb", endpoint="/fineweb/search", query="q", k=6)


def test_a_read_through_search_calls_the_service_once_and_replays_after(tmp_path, fixture_dir):
    env = _envelope(fixture_dir, "fineweb_k2.json")
    n = {"calls": 0}

    def transport(url, headers, timeout):
        n["calls"] += 1
        return Response(200, json.dumps(env).encode())

    client = DrGymClient(api_key="k", env={}, transport=transport)
    s = CachingSearch(client, ShardedJsonCache(tmp_path / "cache"))
    assert len(s.search("q", 2)) == 2
    assert len(s.search("q", 2)) == 2
    assert n["calls"] == 1 and s.hits == 1 and s.misses == 1


# ------------------------------------------------------------------ the suite


@pytest.fixture(scope="module")
def suite(built, tmp_path_factory, request):
    res, _ = built
    cache_root = tmp_path_factory.mktemp("drgym_cache")
    cache = ShardedJsonCache(cache_root)
    fixture_dir = request.config.rootpath / "tests" / "fixtures" / FIXTURE
    env = json.loads((fixture_dir / "search" / "fineweb_k2.json").read_text())
    s = DrGymSuite(res.corpus.parent, offline=True, cache=cache)
    for tid in s.task_ids():
        _seed(cache, "fineweb", s.view(tid).question, 10, env)
    return s


def test_the_adapter_structurally_satisfies_the_suite_and_retriever_protocols(suite):
    from pinq.protocols import Retriever, TaskSuite

    assert isinstance(suite, TaskSuite)
    assert isinstance(suite.retriever(suite.task_ids()[0]), Retriever)
    assert suite.actuator(suite.task_ids()[0]) is None


def test_offline_mode_is_usable_without_a_key(monkeypatch, suite):
    monkeypatch.delenv("DRGYM_API_KEY", raising=False)
    ok, why = suite.available()
    assert ok and "no key required" in why


def test_public_record_has_exactly_the_allowed_keys(built):
    res, _ = built
    rows = [json.loads(x) for x in res.corpus.read_text().splitlines() if x]
    assert rows
    for r in rows:
        assert set(r) == set(RECORD_KEYS) == {"id", "question"}


def test_no_key_point_text_reaches_the_public_corpus(built):
    """The key points ARE the gold: a corpus row carrying one would hand the policy the
    checklist it is being scored against."""
    res, graphs = built
    text = res.corpus.read_text()
    for g in graphs.values():
        for n in g.gold_nodes:
            assert n.gold_text not in text


def test_make_view_succeeds_for_every_task_and_caps_words(suite):
    for tid in suite.task_ids():
        v = suite.view(tid)
        assert v.word_cap == WORD_CAP == 1000
        assert v.corpus_id == CORPUS_IDS["fineweb"]
        assert v.question


@pytest.mark.parametrize(
    "gold_field",
    sorted(set(GoldNode.__dataclass_fields__) | set(GoldEdge.__dataclass_fields__)),
)
def test_every_gold_field_name_is_rejected_by_make_view(suite, gold_field):
    tid = suite.task_ids()[0]
    v = suite.view(tid)
    kw = {
        "task_id": v.task_id,
        "suite_id": v.suite_id,
        "question": v.question,
        "instructions": v.instructions,
        "corpus_id": v.corpus_id,
        "corpus_hash": v.corpus_hash,
        "word_cap": v.word_cap,
        gold_field: "leaked",
    }
    with pytest.raises(LeakageError):
        make_view(**kw)


def test_corpus_hash_pins_the_query_set_and_the_endpoint(built, tmp_path):
    """It cannot pin the documents — a hosted index changes under us and no client can tell.
    What it must do is change when the QUERY SET or the corpus does."""
    res, _ = built
    a = DrGymSuite(res.corpus.parent, offline=True, cache_root=tmp_path / "c1")
    b = DrGymSuite(res.corpus.parent, offline=True, cache_root=tmp_path / "c2")
    assert a.corpus_hash == b.corpus_hash
    other = DrGymSuite(res.corpus.parent, corpus="clueweb22", offline=True, cache_root=tmp_path)
    assert other.corpus_hash != a.corpus_hash


def test_a_rollout_runs_end_to_end_offline(suite):
    tid = suite.task_ids()[0]
    traj = run_loop(
        view=suite.view(tid),
        inquirer=VerbatimInquirer(max_asks=1),
        retriever=suite.retriever(tid),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=4),
        max_turns=4,
        k=10,
    )
    assert traj.n_asks == 1
    assert len(traj.evidence) == 2
    assert traj.stop_reason == "policy_stop"


# ------------------------------------------------------------------ the gold graph


def test_an_empty_key_point_file_is_filtered_and_counted(built):
    """24 of the 1000 committed aggregated files are `key_points: []`. A task with no key
    points has no recall denominator; filtering it silently would shrink the denominator
    with nothing on the record."""
    res, graphs = built
    assert res.n_tasks == 2
    assert res.n_excluded == 1
    assert Q2 not in graphs
    report = json.loads((res.gold.parent / "keypoint_filter_report.json").read_text())
    assert report["n_filtered_empty_key_points"] == 1
    assert report["filtered_task_ids"] == [Q2]
    assert report["bench_commit"] == BENCH_COMMIT


def test_merge_provenance_never_becomes_an_edge(built):
    """`original_point_number` is many-to-one MERGE provenance pointing back into the
    aggregation pipeline, not a dependency between needs. Shipping it as a DAG would make
    Depth a function of how aggressively the aggregator deduplicated."""
    _, graphs = built
    all_edges = [e for g in graphs.values() for e in g.gold_edges]
    assert all_edges == []
    for g in graphs.values():
        assert g.gold_edges == ()
        assert g.gold_facets == ()


def test_merge_provenance_is_kept_as_an_audit_sidecar(built):
    res, _ = built
    side = json.loads((res.gold.parent / "merge_provenance.json").read_text())
    assert side["by_task"][Q1]["kp1"] == [1, 4, 9]
    assert "not a dependency DAG" in side["what"]


def test_every_node_is_a_required_bench_author_seed(built):
    _, graphs = built
    g = graphs[Q1]
    assert len(g.gold_nodes) == 3
    assert set(g.gold_seed_node_ids) == {n.gold_node_id for n in g.gold_nodes}
    for n in g.gold_nodes:
        assert n.gold_partition == "required"
        assert n.gold_provenance_primary == "bench_author"
        assert n.gold_depth == 0
        # No ablation was run, so the causal verdict must not read as if one had been.
        assert n.gold_ablation_verdict == "UNTESTABLE"
        # No released document set: there is no uid a retriever is guaranteed to return.
        assert n.gold_ev_uids == ()


def test_node_ids_are_kp_plus_the_upstream_point_number(built):
    """The fidelity comparison joins our node ids to upstream's `point_number` by a prefix
    strip; anything else is a lookup table that can silently mismatch."""
    _, graphs = built
    assert [n.gold_node_id for n in graphs[Q1].gold_nodes] == ["kp1", "kp2", "kp3"]


def test_the_queries_file_is_pinned_by_name(built):
    assert QUERIES.endswith("researchy_queries_sample_doc_click.jsonl")


# ------------------------------------------------------------------ the report emitter


def test_a_report_with_no_urls_is_refused(tmp_path):
    """The Support rubric hard-zeros a URL-free report and the citation extractor finds no
    claims in one, so emitting it produces a number about the emitter, not the system."""
    with pytest.raises(ReportWithoutCitations):
        emit(tmp_path, Q1, "should city centres ban private cars", "A report with no sources.")
    assert list(tmp_path.iterdir()) == []


def test_a_report_with_urls_is_written_as_q_and_a(tmp_path):
    body = "Oslo saw NO2 fall (https://example.org/oslo-car-free).\n"
    q, a = emit(tmp_path, Q1, "should city centres ban private cars", body)
    assert q.name == f"{Q1}.q" and a.name == f"{Q1}.a"
    assert q.read_text().strip() == "should city centres ban private cars"
    assert "https://example.org/oslo-car-free" in a.read_text()
    assert read_system_dir(tmp_path)[Q1][1].startswith("Oslo saw NO2 fall")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("see https://a.example/x.", ("https://a.example/x",)),
        (
            "(https://a.example/x) and https://b.example/y,",
            ("https://a.example/x", "https://b.example/y"),
        ),
        ("[label](https://a.example/x)", ("https://a.example/x",)),
        ("https://a.example/x https://a.example/x", ("https://a.example/x",)),
        ("no links here", ()),
    ],
)
def test_url_extraction_strips_markdown_and_sentence_punctuation(text, expected):
    assert extract_urls(text) == expected


def test_the_sources_block_can_only_cite_retrieved_documents(fixture_dir):
    docs = docs_from_envelope(_envelope(fixture_dir, "fineweb_k2.json"), corpus_of("fineweb"))
    units = tuple(d.as_unit(CORPUS_IDS["fineweb"]) for d in docs)
    block = sources_block(units)
    assert "https://example.org/oslo-car-free" in block
    assert extract_urls(block) == tuple(d.url for d in docs)


# ------------------------------------------------------------------ integration


@pytest.mark.integration
def test_the_live_endpoint_answers_or_refuses_with_a_key_error():
    """Not skipped when the key is absent: a 401 is the DOCUMENTED behaviour of this service
    without one, so the honest assertion is 'it 401s, and our client says so clearly'."""
    import os

    c = DrGymClient(corpus="fineweb")
    ok, _why = c.available()
    if not ok:
        with pytest.raises(DrGymAuthError):
            DrGymClient(corpus="fineweb", api_key="not-a-real-key").fetch("test", 1)
        return
    assert os.environ.get("DRGYM_API_KEY")
    docs = c.search_docs("what is a chip shortage", 3)
    assert docs and all(d.text for d in docs)


@pytest.mark.integration
def test_the_real_query_list_and_key_points_build_at_the_pinned_commit(tmp_path):
    """Fetches the pinned query list (hard sha256) and the first few key-point files.

    Marked integration because it downloads. What it proves is the thing a fixture cannot:
    that the pinned commit still serves the bytes this builder was written against, and that
    the real files parse into the same flat, edge-free graph the fixture does.
    """
    res = build(raw_dir=tmp_path / "raw", root=tmp_path, limit=5, verify=True)
    assert (tmp_path / "raw" / QUERIES).exists()
    graphs = read_graphs(res.gold)
    assert graphs
    for g in graphs:
        assert g.gold_edges == ()
        assert all(n.gold_provenance_primary == "bench_author" for n in g.gold_nodes)
