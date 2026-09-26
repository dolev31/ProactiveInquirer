"""tau2 adapter: the doc->tool edge, the leakage firewall, and the user-private partition.

Everything except the `integration` block runs with tau2 ABSENT, against committed
fixtures whose expected edge set is small enough to check by hand.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from pi_eval import schema as sch
from pi_eval.build.common import read_graphs
from pi_eval.build.tau2_build import (
    GRAPH_VERSION,
    build,
    build_task_graph,
    corpus_hash_of,
    graph_row,
    ngrams,
    norm_tokens,
    partition_elasticity,
    partition_facts,
    partition_summary,
    resolve_required,
    segment_instructions,
)
from pi_eval.gold import GoldEdge, GoldNode
from pi_eval.metrics import environment as env_metrics
from pi_eval.score import BY_NAME, DEFAULT_GRAPH_VERSION
from pinq.budget import BudgetLedger
from pinq.types import Evidence, EvidenceUnit
from pinq.view import LeakageError, make_view
from pinq_adapters.tau2 import (
    DocToolEdge,
    available,
    documented_but_not_a_tool,
    extract_tool_edges,
    load_documents,
    load_task_records,
    parse_kb_results,
    pick_kb_tool,
    scan_tool_tokens,
    tools_never_documented,
)
from pinq_adapters.tau2.actuator import CALL_TOOL, UNLOCK_TOOL, Tau2Actuator
from pinq_expt import arms as arm_table
from pinq_expt.components import LLMDrafter, normalize_tool_plan, split_tool_plan

FIX = Path(__file__).parent / "fixtures" / "tau2"

# The live environment exposes these. `decoy_tool_9999` is deliberately ABSENT even though
# a fixture document names it; `lonely_tool_4242` is present but named by no document.
FIXTURE_TOOLS = ["foo_bar_1234", "baz_qux_5678", "lonely_tool_4242", "KB_search"]


@pytest.fixture
def docs():
    return load_documents(FIX / "documents")


@pytest.fixture
def task():
    recs = load_task_records(FIX / "tasks")
    assert len(recs) == 1
    return recs[0]


# --------------------------------------------------------------------- the doc->tool edge


def test_extracted_edge_set_is_exactly_this(docs):
    """Hand-checkable. Three edges, and every one is readable in the fixture text."""
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    assert edges == (
        DocToolEdge("doc_alpha_001", "Internal: Opening Alpha Accounts", "baz_qux_5678"),
        DocToolEdge("doc_alpha_001", "Internal: Opening Alpha Accounts", "foo_bar_1234"),
        DocToolEdge("doc_beta_002", "Internal: Beta Card Disputes", "foo_bar_1234"),
    )


def test_token_in_a_doc_but_not_a_tool_yields_no_edge(docs):
    """Direction 1. On the real corpus this drops `downgrade_credit_card_3847`."""
    assert "decoy_tool_9999" in scan_tool_tokens(docs[1]["content"])
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    assert "decoy_tool_9999" not in {e.tool_name for e in edges}
    assert documented_but_not_a_tool(docs, FIXTURE_TOOLS) == frozenset({"decoy_tool_9999"})


def test_tool_named_in_no_doc_yields_no_edge(docs):
    """Direction 2. On the real corpus this is `example_agent_tool_0000`."""
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    assert "lonely_tool_4242" not in {e.tool_name for e in edges}
    assert tools_never_documented(docs, FIXTURE_TOOLS) == frozenset({"lonely_tool_4242"})


def test_a_document_naming_no_tool_contributes_nothing(docs):
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    assert "doc_gamma_003" not in {e.doc_id for e in edges}


def test_edge_set_is_non_empty_and_deterministic(docs):
    a = extract_tool_edges(docs, FIXTURE_TOOLS)
    b = extract_tool_edges(list(reversed(docs)), list(reversed(FIXTURE_TOOLS)))
    assert a and a == b, "the edge set must be order-independent to be hashable into a version"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call open_bank_account_4821 now", {"open_bank_account_4821"}),
        ("trailing punctuation foo_bar_1234.", {"foo_bar_1234"}),
        ("too few digits foo_bar_123", set()),
        ("too many digits foo_bar_12345", set()),
        ("uppercase FOO_BAR_1234", set()),
        ("", set()),
    ],
)
def test_token_regex_boundaries(text, expected):
    assert scan_tool_tokens(text) == frozenset(expected)


# ------------------------------------------------------------------------- leakage firewall


def _view_kwargs(**over):
    kw = dict(
        task_id="fix_001",
        suite_id="tau2",
        question="You are playing the role of a customer.",
        instructions="You are a bank customer-service agent.",
        corpus_id="tau2_banking_knowledge",
        corpus_hash="deadbeef",
        word_cap=180,
    )
    kw.update(over)
    return kw


@pytest.mark.firewall
def test_required_documents_cannot_reach_a_view():
    """The single most important leak to stop: it is the benchmark's own answer key."""
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(required_documents=["doc_alpha_001"]))


@pytest.mark.firewall
@pytest.mark.parametrize(
    "leaky",
    ["required_documents", "evaluation_criteria", "user_tools", "initial_state", "annotations"],
)
def test_every_tau2_gold_bearing_task_field_raises(leaky):
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(**{leaky: "leaked"}))


@pytest.mark.firewall
@pytest.mark.parametrize("gold_field", sorted(GoldNode.__dataclass_fields__))
def test_every_gold_node_field_raises(gold_field):
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(**{gold_field: "leaked"}))


@pytest.mark.firewall
@pytest.mark.parametrize("gold_field", sorted(GoldEdge.__dataclass_fields__))
def test_every_gold_edge_field_raises(gold_field):
    with pytest.raises(LeakageError):
        make_view(**_view_kwargs(**{gold_field: "leaked"}))


@pytest.mark.firewall
def test_the_view_a_suite_builds_carries_no_gold(task):
    """Positive control: the fields we DO pass are exactly the allowlist."""
    v = make_view(**_view_kwargs(question=task["user_scenario"]["instructions"]))
    rendered = repr(v)
    assert "doc_alpha_001" not in rendered and "doc_beta_002" not in rendered


# -------------------------------------------------------------- the user-private partition


def test_segmentation_uses_headers_when_present(task):
    labels = [lab for lab, _ in segment_instructions(task["user_scenario"]["instructions"])]
    assert labels == ["preamble", "Phase 1: Background", "Phase 2: Request"]


def test_segmentation_falls_back_to_paragraphs_without_headers():
    """13 of the 97 real tasks have no '##' header at all; they must still segment."""
    blocks = segment_instructions("First para line.\n\nSecond para line.")
    assert [lab for lab, _ in blocks] == ["para0", "para1"]
    assert blocks[1][1] == "Second para line."


def test_partition_marks_the_documented_fact_kb_and_the_personal_fact_private(task, docs):
    """The fixture controls exactly which fact is written in a required document.

    'Disputes must be filed within 90 days...' is verbatim in doc_beta_002.
    'My cat is named Persimmon...' is in no document at all.
    """
    req, missing = resolve_required(task["required_documents"], docs)
    assert missing == ()
    facts = partition_facts(task["user_scenario"]["instructions"], req)
    by_text = {f.text: f for f in facts}

    cat = by_text["My cat is named Persimmon and she dislikes thunderstorms."]
    assert cat.discoverability == "user_private"
    assert cat.matched_doc_id is None

    dispute = by_text["Disputes must be filed within 90 days of the transaction date."]
    assert dispute.discoverability == "kb"
    assert dispute.matched_doc_id == "doc_beta_002"


def test_both_partition_instruments_agree_on_the_fixture(task, docs):
    """The fixture is engineered so the two instruments cannot disagree: one sentence is
    verbatim in a required document, the other shares no vocabulary with any."""
    req, _ = resolve_required(task["required_documents"], docs)
    for mode, param in (("overlap", 0.6), ("ngram", 4)):
        by_text = {
            f.text: f
            for f in partition_facts(
                task["user_scenario"]["instructions"],
                req,
                mode=mode,
                threshold=param if mode == "overlap" else 1.0,
                n=param if mode == "ngram" else 4,
            )
        }
        cat = by_text["My cat is named Persimmon and she dislikes thunderstorms."]
        dispute = by_text["Disputes must be filed within 90 days of the transaction date."]
        assert cat.discoverability == "user_private", mode
        assert dispute.discoverability == "kb", mode


def test_overlap_partition_is_monotone_in_the_threshold(task, docs):
    """A stricter threshold can only move facts toward user_private, never back.

    Non-monotonicity here would mean the ceiling could improve by tightening the
    instrument, which would make the elasticity curve uninterpretable.
    """
    req, _ = resolve_required(task["required_documents"], docs)
    shares = []
    for th in (0.2, 0.4, 0.6, 0.8, 1.0):
        facts = partition_facts(task["user_scenario"]["instructions"], req, threshold=th)
        shares.append(sum(1 for f in facts if f.discoverability == "user_private") / len(facts))
    assert shares == sorted(shares), shares


def test_partition_elasticity_returns_a_curve_not_a_scalar(task, docs):
    """The plan's rule: the scalar never appears without the curve."""
    curve = partition_elasticity([task], docs, grid=(0.4, 0.6, 0.8))
    assert [t for t, _ in curve] == [0.4, 0.6, 0.8]
    assert all(0.0 <= share <= 1.0 for _, share in curve)
    assert [s for _, s in curve] == sorted(s for _, s in curve)


def test_partition_is_empty_of_kb_hits_when_no_document_is_required(task):
    """With nothing required, nothing is discoverable — the ceiling is 100% private."""
    facts = partition_facts(task["user_scenario"]["instructions"], [])
    assert facts and all(f.discoverability == "user_private" for f in facts)


def test_norm_tokens_and_ngrams():
    assert norm_tokens("The, QUICK   brown-fox!") == ("quick", "brown", "fox")
    assert ngrams(("a", "b", "c"), 2) == frozenset({"a b", "b c"})
    # shorter than n degrades to the whole token run rather than vanishing
    assert ngrams(("a", "b"), 4) == frozenset({"a b"})
    assert ngrams((), 4) == frozenset()


# ------------------------------------------------------------------------- the gold graph


def test_build_task_graph_shape(task, docs):
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    g = build_task_graph(task, docs, edges)

    facts = [n for n in g.gold_nodes if n.gold_provenance_primary == "bench_author"]
    assert {n.gold_text for n in facts} == {
        "Internal: Opening Alpha Accounts",
        "Internal: Beta Card Disputes",
    }
    assert all(n.gold_partition == "required" and n.gold_depth == 0 for n in facts)

    unlocks = [n for n in g.gold_nodes if n.gold_kind == "tool_unlock"]
    assert {n.gold_text.split()[2] for n in unlocks} == {"foo_bar_1234", "baz_qux_5678"}
    assert all(n.gold_provenance_primary == "mechanical" for n in unlocks)
    # a tool is one hop from the document that names it
    assert all(n.gold_depth == 1 for n in unlocks)


def test_gold_edges_are_mechanical_prerequisites(task, docs):
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    assert g.gold_edges, "the mechanical edge set must not be empty"
    for e in g.gold_edges:
        assert e.gold_edge_kind == "prerequisite"
        assert e.gold_verified == "mechanical"
        assert e.gold_provenance == "mechanical"


def test_tools_unlocked_only_by_unrequired_docs_are_excluded(docs):
    """A tool named in a document this task does not require is not this task's prerequisite."""
    task = {
        "id": "only_gamma",
        "user_scenario": {"instructions": "Nothing relevant here."},
        "required_documents": ["doc_gamma_003"],
    }
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    assert [n for n in g.gold_nodes if n.gold_kind == "tool_unlock"] == []
    assert g.gold_edges == ()


def test_partition_summary_counts_only_mined_user_facts(task, docs):
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    s = partition_summary([g])
    assert s["n_kb"] >= 1 and s["n_user_private"] >= 1
    assert 0.0 < s["user_private_share"] < 1.0


def test_resolve_required_reports_misses(docs):
    found, missing = resolve_required(["doc_alpha_001", "nope_999"], docs)
    assert [d["id"] for d in found] == ["doc_alpha_001"]
    assert missing == ("nope_999",)


# ------------------------------------------------------------------------- KB result parsing


KB_BLOB = """1. Internal: Opening Alpha Accounts
   ID: doc_alpha_001
   Score: 12.5000
   Content: The customer must be verified before opening an alpha account.

2. Internal: Gamma Fees
   ID: doc_gamma_003
   Score: 3.2500
   Content: The monthly maintenance fee is waived.

[Timing: retrieval=4ms, total=5ms]"""


def test_parse_kb_results_matches_upstream_format():
    hits = parse_kb_results(KB_BLOB)
    assert [h["doc_id"] for h in hits] == ["doc_alpha_001", "doc_gamma_003"]
    assert hits[0]["title"] == "Internal: Opening Alpha Accounts"
    assert hits[0]["score"] == 12.5
    assert hits[0]["content"].endswith("alpha account.")
    # the timing footer must not leak into the last document's content
    assert "Timing" not in hits[1]["content"]


def test_parse_kb_results_on_no_hits():
    assert parse_kb_results("No relevant documents found.\n\n[Timing: retrieval=1ms]") == ()
    assert parse_kb_results("") == ()


def test_pick_kb_tool_prefers_bm25_then_plain_then_raises():
    assert pick_kb_tool(["KB_search", "KB_search_bm25"]) == "KB_search_bm25"
    assert pick_kb_tool(["KB_search", "grep"]) == "KB_search"
    with pytest.raises(RuntimeError):
        pick_kb_tool(["grep", "shell"])


# ------------------------------------------------------------------------- offline guarantee


def test_available_returns_a_reason_either_way():
    ok, why = available()
    assert isinstance(ok, bool) and isinstance(why, str) and why


def test_adapter_never_imports_pi_eval():
    """The import contract, asserted on the AST as well as by lint-imports.

    Parsed rather than grepped: these modules legitimately *mention* pi_eval in prose
    explaining which side of the firewall they sit on, and a substring test would forbid
    documenting the boundary at all.
    """
    import ast

    for path in (Path(__file__).parents[1] / "src" / "pinq_adapters" / "tau2").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("pi_eval"), f"{path.name} imports {name}"


def test_no_tau2_import_at_module_scope():
    """OFFLINE-FIRST, enforced. A module-scope `import tau2` would make the default test
    run unrunnable without the extra — which is most machines, most of the time."""
    import ast

    for path in (Path(__file__).parents[1] / "src" / "pinq_adapters" / "tau2").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope ONLY; function-body imports are the pattern
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("tau2"), f"{path.name} imports {name} at module scope"


# ------------------------------------------------------------------------------ integration

_ok, _why = available()
integration = pytest.mark.skipif(not _ok, reason=_why)


@pytest.mark.integration
@integration
def test_real_corpus_counts():
    from pinq_adapters.tau2 import N_DOCUMENTS, N_TASKS
    from pinq_adapters.tau2._probe import load_documents as ld
    from pinq_adapters.tau2._probe import load_task_records as lt

    assert len(ld()) == N_DOCUMENTS
    assert len(lt()) == N_TASKS


@pytest.mark.integration
@integration
def test_real_get_tasks_returns_97():
    from tau2.runner import get_tasks

    assert len(get_tasks("banking_knowledge")) == 97


@pytest.mark.integration
@integration
def test_every_extracted_tool_really_exists_in_the_environment():
    """The claim the paper rests on, checked against a live environment.

    Note the tool universe is NOT `get_tools()`: the suffixed tools are hidden from the
    advertised list by design, which is the discoverability mechanic itself.
    """
    from pinq_adapters.tau2 import Tau2Suite
    from pinq_adapters.tau2.retriever import env_tool_names

    suite = Tau2Suite()
    env = suite.environment(suite.task_ids()[0])
    universe = env_tool_names(env)
    edges = suite.tool_edges(env)

    assert edges, "the doc->tool edge set must be non-empty"
    for e in edges:
        assert e.tool_name in universe, f"{e.tool_name} was extracted but is not a real tool"
        # the stronger claim: it is specifically a DISCOVERABLE tool, i.e. one that cannot
        # be invoked until its naming document has been read
        assert env.tools.is_discoverable(e.tool_name), f"{e.tool_name} is not discoverable"


@pytest.mark.integration
@integration
def test_the_hidden_tools_are_really_hidden():
    """The premise of the whole edge: an unread tool must be unguessable, so it must not
    appear in the advertised tool list. If upstream ever exposed them, the prerequisite
    would evaporate and this test is how we would find out."""
    from pinq_adapters.tau2 import Tau2Suite
    from pinq_adapters.tau2.discoverable import TOOL_TOKEN_RE

    suite = Tau2Suite()
    env = suite.environment(suite.task_ids()[0])
    visible = {t.name for t in env.get_tools()}
    assert not [n for n in visible if TOOL_TOKEN_RE.fullmatch(n)]
    assert "unlock_discoverable_agent_tool" in visible


@pytest.mark.integration
@integration
def test_the_documented_decoy_is_not_a_real_tool():
    """`downgrade_credit_card_3847` is written in a document but backed by no tool.

    This is why the cross-reference is load-bearing rather than decorative.
    """
    from pinq_adapters.tau2 import Tau2Suite, scan_tool_tokens
    from pinq_adapters.tau2.retriever import env_tool_names

    suite = Tau2Suite()
    documented = set()
    for d in suite.documents:
        documented |= scan_tool_tokens(d["content"])
    assert "downgrade_credit_card_3847" in documented

    env = suite.environment(suite.task_ids()[0])
    assert "downgrade_credit_card_3847" not in env_tool_names(env)
    assert "downgrade_credit_card_3847" not in {e.tool_name for e in suite.tool_edges(env)}


# ======================================================================= THE ACTION PATH
#
# Everything below is the tau2 PRIMARY ENDPOINT (`tau_reward`) and its supporting machinery:
# a Drafter that can emit a tool_plan, an Actuator that executes it and is graded by tau2's
# own evaluator, the gold graph the discoverable-tool endpoint is scored against, and the
# parquet columns that carry the result. The endpoint was structurally 0 for every arm before
# this: no Drafter populated `Draft.tool_plan`, `attach_reward` had zero callers, and the
# graph version the builder wrote was not the one `pi score` asks for.

# --------------------------------------------------------------------- the graph version


def test_graph_version_is_what_pi_score_asks_for():
    """THE BUG THAT MADE P1 UNSCOREABLE, PINNED.

    `GRAPH_VERSION` was "tau2/v1" while `pi score --graph-version` defaults to "v1". The
    builder wrote data/gold/graphs/tau2/tau2/v1.jsonl; `load_graphs("tau2", "v1")` looked in
    data/gold/graphs/tau2/v1.jsonl, found nothing, and every tau2 run was counted in
    `n_runs_skipped_no_graph`. Nothing failed, and the endpoint read as ABSENT rather than as
    broken — which is the worst of the three possible outcomes.
    """
    assert GRAPH_VERSION == DEFAULT_GRAPH_VERSION
    assert "/" not in GRAPH_VERSION, "the suite is the DIRECTORY; the version is the filename"


def test_tau2_metrics_are_registered_under_the_names_they_are_emitted_by():
    """A metric emitted under a name no MetricDef knows is family-less in every table."""
    for name in ("tau_reward", "discoverable_tool_unlock_tau2", "unlock_and_invoke"):
        assert BY_NAME[name].name == name
    assert BY_NAME["tau_reward"].binary, "the DB check is 0/1; a graded mean of it is a rate"


# --------------------------------------------------------------------- the build driver


def test_build_writes_one_line_per_task_where_score_will_look(tmp_path, task, docs):
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    res = build(root=tmp_path, documents=docs, tasks=[task], tool_edges=edges)

    assert res.gold == tmp_path / "data" / "gold" / "graphs" / "tau2" / f"{GRAPH_VERSION}.jsonl"
    assert res.gold.exists() and res.n_tasks == 1 and res.n_excluded == 0
    graphs = read_graphs(res.gold)
    assert [g.gold_task_key for g in graphs] == ["fix_001"]
    assert graphs[0].gold_graph_version == GRAPH_VERSION
    # tau2 has no free-text answer, so the answer-text metrics must not be emitted for it
    assert graphs[0].gold_answer == ""


def test_build_round_trips_every_derived_field(tmp_path, task, docs):
    """A graph read back off disk must be the graph that was built. Depth and facets are
    DERIVED, and a serializer that dropped them would silently zero every structure metric."""
    edges = extract_tool_edges(docs, FIXTURE_TOOLS)
    built = build_task_graph(task, docs, edges)
    res = build(root=tmp_path, documents=docs, tasks=[task], tool_edges=edges)
    back = read_graphs(res.gold)[0]

    assert {(n.gold_node_id, n.gold_depth) for n in back.gold_nodes} == {
        (n.gold_node_id, n.gold_depth) for n in built.gold_nodes
    }
    assert len(back.gold_edges) == len(built.gold_edges)
    assert back.gold_seed_node_ids == built.gold_seed_node_ids
    assert graph_row(built)["gold_graph_version"] == GRAPH_VERSION


def test_gold_and_adapter_agree_on_the_corpus_hash(docs):
    """Gold says "need v is resolved by uid U"; the rollout says "the retriever returned U".
    The two are minted by two functions on two sides of the firewall, and a one-character
    drift between them reads as "the policy retrieved nothing relevant" in every table."""
    from pinq_adapters.tau2 import Tau2Suite

    suite = Tau2Suite(
        documents_root=FIX / "documents", tasks_root=FIX / "tasks", strict_counts=False
    )
    assert corpus_hash_of(docs) == suite.corpus_hash


def test_tool_unlock_nodes_carry_the_tool_name_as_a_machine_readable_alias(task, docs):
    """`discoverable_tool_unlock_tau2` joins gold against `env_calls.kwargs_json` on this.

    `gold_text` is an English sentence; splitting its third word out is a join key that a
    reworded docstring breaks silently, taking the secondary endpoint to 0 with it.
    """
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    assert env_metrics.required_tools(g) == ("baz_qux_5678", "foo_bar_1234")


# --------------------------------------------------------- the elasticity curve, un-flattened


def test_ngram_elasticity_is_a_curve_and_not_a_flat_ceiling(task, docs):
    """FAILS BEFORE THE FIX. `partition_elasticity` fed the grid value in as a THRESHOLD in
    ngram mode, where the score is only ever 1.0 or 0.0. `best >= 2` was false for every
    sentence in the corpus, so the curve was 1.00 at every n and reported the ceiling as
    100% user-private for the whole suite — a number that could not move whatever the data
    said. The knob in ngram mode is `n`; the threshold is 1.0 by construction."""
    curve = partition_elasticity([task], docs, mode="ngram", grid=(2, 3, 4, 5))
    shares = [s for _, s in curve]
    assert shares[0] < 1.0, "a sentence copied verbatim from a required document IS a 2-gram hit"
    assert shares == sorted(shares), "a longer n can only move facts toward user_private"


# --------------------------------------------------------------------- the drafted tool plan


class _ScriptLLM:
    """Deterministic and offline. Keeps every prompt, so the interesting assertions are about
    the REQUEST rather than about the reply."""

    model = "stub/script"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, *, role, messages, seed, max_tokens=None, **kw):
        from pinq.types import CallTelemetry

        self.prompts.append(messages[-1]["content"])
        return self.reply, CallTelemetry(
            call_id=f"c{len(self.prompts)}",
            actor=role,
            model=self.model,
            provider="stub",
            request_sha="r",
            response_sha="s",
        )


TOOLS = (
    {"name": UNLOCK_TOOL, "description": "unlock a discoverable tool", "parameters": {}},
    {"name": CALL_TOOL, "description": "call an unlocked tool", "parameters": {}},
)

PLAN_REPLY = (
    "The dispute window is 90 days.\n\n"
    "```json\n"
    '{"tool_plan": [{"name": "' + UNLOCK_TOOL + '", "args": {"agent_tool_name": "foo_bar_1234"}},'
    ' {"name": "' + CALL_TOOL + '", "args": {"agent_tool_name": "foo_bar_1234"}}]}\n'
    "```"
)


def _view():
    return make_view(**_view_kwargs())


def _ev():
    return Evidence.of(
        [
            EvidenceUnit.make(
                corpus_id="tau2_banking_knowledge",
                doc_id="doc_beta_002",
                span="0:20",
                title="Beta",
                text="use foo_bar_1234",
            )
        ]
    )


def test_a_drafter_with_no_tools_emits_no_plan():
    """Every non-environment suite must be byte-for-byte unaffected by this feature."""
    llm = _ScriptLLM(PLAN_REPLY)
    d = LLMDrafter(llm)
    draft = d.draft(_view(), _ev(), seed=0, ledger=BudgetLedger(cap=8))
    assert draft.tool_plan == ()
    assert draft.text == PLAN_REPLY, "the prose is untouched when no tools were offered"
    assert "TOOLS" not in llm.prompts[0], "no tool block reaches a suite without tools"


def test_a_drafter_with_tools_emits_an_executable_plan():
    llm = _ScriptLLM(PLAN_REPLY)
    d = LLMDrafter(llm, tools=TOOLS)
    draft = d.draft(_view(), _ev(), seed=0, ledger=BudgetLedger(cap=8))

    assert draft.tool_plan == (
        {"name": UNLOCK_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}},
        {"name": CALL_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}},
    )
    # the JSON block is stripped: the FROZEN answerer is shared by every arm and must never
    # be handed a block of tool JSON to paraphrase into an answer
    assert "tool_plan" not in draft.text
    assert draft.text.startswith("The dispute window")
    assert UNLOCK_TOOL in llm.prompts[0], "the advertised tools reach the draft prompt"


def test_draft_stays_pure_in_view_subset_hash_and_seed_with_tools():
    """THE ARCHITECTURAL CONSTRAINT. phi_LOO, the prefix ladder and the stop test are all
    statements about re-drafting over an evidence subset long after the rollout ended, and
    none of them is defined if a tool-bearing draft() is not a pure function.

    Five distinct Evidence objects with one subset_hash between them, built in five different
    orders and all held alive at once, through ONE drafter: per-instance hidden state would
    show up as a different request and a different plan.
    """
    units = list(_ev().units) + [
        EvidenceUnit.make(
            corpus_id="tau2_banking_knowledge",
            doc_id="doc_alpha_001",
            span="0:10",
            title="Alpha",
            text="alpha body",
        )
    ]
    subsets = [Evidence.of(units[i:] + units[:i]) for i in range(2)] * 3
    assert len({e.subset_hash for e in subsets}) == 1

    llm = _ScriptLLM(PLAN_REPLY)
    d = LLMDrafter(llm, tools=TOOLS)
    drafts = [d.draft(_view(), ev, seed=0, ledger=BudgetLedger(cap=8)) for ev in subsets]
    assert len({dr.sha for dr in drafts}) == 1
    assert len(set(llm.prompts)) == 1, "one request, six times: the bytes are a pure function"


def test_a_tool_plan_never_executes_inside_draft():
    """`draft()` DECIDES; the Actuator EXECUTES. Executing here would fire a real tool call on
    every phi_LOO re-draft and every rung of the prefix ladder, so the same evidence subset
    re-scored twice would produce two different worlds."""
    env = _FakeEnv()
    LLMDrafter(_ScriptLLM(PLAN_REPLY), tools=TOOLS).draft(
        _view(), _ev(), seed=0, ledger=BudgetLedger(cap=8)
    )
    assert env.calls == [], "no environment call may originate in draft()"


def test_tools_reach_the_drafter_through_arms_build_and_the_inquirer_never():
    """`build()` passes only what a constructor declares. A policy that could see the action
    space would start planning actions inside `act(s)`, where nothing meters them."""
    components = arm_table.build(arm_table.get("inquirer_prompted"), llm=None, tools=TOOLS)
    assert components.drafter._tools == TOOLS
    assert not hasattr(components.inquirer, "_tools")
    assert "fragment_tool_plan" in components.drafter.prompt_hashes


def test_every_llm_backed_arm_receives_the_same_action_channel():
    """THE CONFOUND THIS TEST EXISTS TO STOP.

    `arms.build` passes only what a constructor DECLARES, so a `tools` parameter absorbed
    into **kw is never passed at all. Before this was fixed, `drafter_only` and
    `inquirer_prompted` got the tool schemas while `verbosity`, `compute_matched`,
    `query_expansion` and — worst of all — `self_inquire` did not.

    `self_inquire` is the COMPARATOR of the tau2 primary endpoint. A treatment that can act
    measured against a comparator that structurally cannot would have produced a large,
    clean, entirely artefactual win on P1, and nothing in the pipeline would have flagged it:
    both arms run, both write rows, both get scored.
    """
    for arm_id in arm_table.arm_ids():
        arm = arm_table.get(arm_id)
        if arm.llm_free:
            continue  # the fake_* reference arms have no LLM and no draft prompt at all
        kw = {"llm": None, "tools": TOOLS}
        if arm.requires_questions:
            kw["questions"] = {"t": ["q"]}
        drafter = arm_table.build(arm, **kw).drafter
        assert getattr(drafter, "_tools", ()) == TOOLS, f"{arm_id} was not given the tools"


def test_every_drafter_class_splits_the_plan_out_of_its_generation():
    """`ComputeMatchedDrafter` overrides `draft()` and would otherwise return the raw
    majority text with the JSON block still in it and an empty tool_plan."""
    from pinq_expt.components import ComputeMatchedDrafter, VerbosityDrafter

    for cls in (LLMDrafter, VerbosityDrafter, ComputeMatchedDrafter):
        kw = {"n": 3} if cls is ComputeMatchedDrafter else {}
        d = cls(_ScriptLLM(PLAN_REPLY), tools=TOOLS, **kw)
        draft = d.draft(_view(), _ev(), seed=0, ledger=BudgetLedger(cap=8))
        assert draft.tool_plan, f"{cls.__name__} emitted no plan"
        assert "tool_plan" not in draft.text, f"{cls.__name__} left the JSON in the prose"


def test_a_tool_bearing_drafter_declares_the_fragment_it_renders():
    """A manifest that names a prompt the run did not render is a false provenance claim, and
    so is one that omits a prompt the run DID render."""
    assert "fragment_tool_plan" not in LLMDrafter(None).prompt_hashes
    assert "fragment_tool_plan" in LLMDrafter(None, tools=TOOLS).prompt_hashes


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"tool_plan": []}', ()),
        ("no json at all", ()),
        ('{"tool_plan": [{"name": "a"}]}', ({"name": "a", "args": {}},)),
        # `arguments` is the alias models reach for constantly
        (
            '{"tool_plan": [{"name": "a", "arguments": {"x": 1}}]}',
            ({"name": "a", "args": {"x": 1}},),
        ),
        # a step with no name has nowhere to go: DROPPED, never repaired
        ('{"tool_plan": [{"args": {}}, {"name": "b"}]}', ({"name": "b", "args": {}},)),
        ('{"tool_plan": "not a list"}', ()),
    ],
)
def test_split_tool_plan_normalizes_or_drops(text, expected):
    assert split_tool_plan(text)[1] == expected


def test_the_last_tool_plan_object_wins_not_the_first():
    """A draft that quotes a JSON example out of a knowledge-base document would otherwise
    have that example executed against a live bank."""
    text = (
        'The document shows an example: {"tool_plan": [{"name": "example_from_the_kb"}]}\n'
        '```json\n{"tool_plan": [{"name": "what_i_actually_chose"}]}\n```'
    )
    assert split_tool_plan(text)[1] == ({"name": "what_i_actually_chose", "args": {}},)


def test_a_drafter_may_not_act_as_the_customer():
    """tau2's user tools belong to the customer. `Tau2Actuator.execute` honours `requestor`
    because the gold-action replay needs it, but a drafted step must never carry one: a
    policy that could act as the customer would be scored for a mutation the agent under
    test never had the authority to make."""
    plan = normalize_tool_plan([{"name": "apply_for_credit_card", "requestor": "user", "args": {}}])
    assert plan == ({"name": "apply_for_credit_card", "args": {}},)


# --------------------------------------------------------------------- the actuator


class _FakeEnv:
    """The smallest object satisfying what Tau2Actuator touches. No tau2 import."""

    def __init__(self, fail: frozenset[str] = frozenset()) -> None:
        self.calls: list[tuple] = []
        self._fail = fail
        self.db = "db0"

    def make_tool_call(self, name, *, requestor="assistant", **kwargs):
        if name in self._fail:
            raise RuntimeError(f"{name} refused")
        self.calls.append((name, requestor, dict(kwargs)))
        if name == "mutate":
            self.db = "db1"
        return {"ok": name}

    def _is_mutating_tool(self, name):
        return name == "mutate"

    def get_db_hash(self):
        return self.db

    def get_user_db_hash(self):
        return "udb0"


def test_the_actuator_executes_a_plan_in_order_and_logs_every_call():
    env = _FakeEnv()
    act = Tau2Actuator(env)
    calls = act.execute(
        [
            {"name": UNLOCK_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}},
            {"name": CALL_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}},
        ],
        turn_idx=3,
    )
    assert [c.tool_name for c in calls] == [UNLOCK_TOOL, CALL_TOOL]
    assert [c.seq for c in calls] == [1, 2]
    assert all(c.turn_idx == 3 for c in calls)
    assert act.unlocked == frozenset({"foo_bar_1234"})
    assert act.final_hashes() == {"db": "db0", "user_db": "udb0"}


def test_a_gated_call_without_its_unlock_is_recorded_as_unsatisfied():
    """The prerequisite edge, as stored data. `unlock_satisfied` is what the discoverable-tool
    endpoint groups on, so it must record what the environment saw and not what was hoped."""
    act = Tau2Actuator(_FakeEnv())
    (call,) = act.execute(
        [{"name": CALL_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}}], turn_idx=0
    )
    assert call.unlock_required and not call.unlock_satisfied


def test_a_failed_unlock_does_not_credit_a_crossed_edge():
    act = Tau2Actuator(_FakeEnv(fail=frozenset({UNLOCK_TOOL})))
    (call,) = act.execute(
        [{"name": UNLOCK_TOOL, "args": {"agent_tool_name": "foo_bar_1234"}}], turn_idx=0
    )
    assert not call.ok
    assert act.unlocked == frozenset(), "a refused unlock leaves the tool locked"


class _Reward:
    """A RewardInfo-shaped stub. Duck-typed on purpose: the real one needs tau2 installed."""

    def __init__(self, reward, db_reward, db_match, breakdown=None):
        self.reward = reward
        self.db_check = type("DBCheck", (), {"db_reward": db_reward, "db_match": db_match})()
        self.reward_breakdown = breakdown or {}


def test_attach_reward_publishes_tau_reward_only_for_a_db_basis_task():
    act = Tau2Actuator(_FakeEnv())
    act.attach_reward(_Reward(1.0, 1.0, True, {"DB": 1.0}), reward_basis=["DB"])
    native = dict(act._reward)
    assert native["tau_reward"] == 1.0
    assert native["db_reward"] == 1.0 and native["db_match"] == 1.0
    assert native["breakdown.DB"] == 1.0


def test_an_action_basis_task_gets_no_tau_reward_and_no_free_pass():
    """9 of the 97 tasks are ACTION-basis. The ENVIRONMENT evaluator does not know that
    criterion and returns reward=1.0 for them — a free pass, not a measurement. Storing it
    would publish a fabricated success on a tenth of the suite."""
    act = Tau2Actuator(_FakeEnv())
    act.attach_reward(_Reward(1.0, 0.0, False), reward_basis=["ACTION"])
    native = dict(act._reward)
    assert "tau_reward" not in native
    assert "reward" not in native, "the evaluator did not cover this task's basis"
    assert native["db_reward"] == 0.0, "the DB check itself is still a real measurement"


def test_a_failed_db_check_is_a_zero_and_not_an_absence():
    act = Tau2Actuator(_FakeEnv())
    act.attach_reward(_Reward(0.0, 0.0, False, {"DB": 0.0}), reward_basis=["DB"])
    assert dict(act._reward)["tau_reward"] == 0.0


def test_native_is_empty_rather_than_zero_when_there_is_nothing_to_grade():
    """A missing measurement must LOOK missing. `env_metrics.tau_reward` turns an absent key
    into NaN, and score.py then emits no row at all."""
    act = Tau2Actuator(_FakeEnv())
    assert dict(act.native()) == {}
    assert env_metrics.is_measured(env_metrics.tau_reward(act.native())) is False


# ------------------------------------------------- the discoverable-tool secondary endpoint


def _env_call(tool, target, *, ok=True, gated=False, satisfied=False):
    return {
        "tool_name": tool,
        "kwargs_json": _json.dumps({"agent_tool_name": target}, sort_keys=True),
        "ok": ok,
        "unlock_required": gated,
        "unlock_satisfied": satisfied,
    }


def test_unlock_rate_denominator_is_gold_not_the_runs_own_attempts(task, docs):
    """A run that unlocked one tool and stopped must not score 1.0. The denominator is the
    set of tools this task's REQUIRED documents name, so a policy cannot improve the metric
    by attempting less."""
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    calls = [
        _env_call(UNLOCK_TOOL, "foo_bar_1234"),
        _env_call(CALL_TOOL, "foo_bar_1234", gated=True, satisfied=True),
    ]
    out = env_metrics.discoverable_tool_unlock(calls, g)
    assert out["n_required"] == 2.0 and out["n_invoked"] == 1.0
    assert out["rate"] == 0.5


def test_a_refused_gated_call_counts_for_nothing(task, docs):
    g = build_task_graph(task, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    refused = [_env_call(CALL_TOOL, "foo_bar_1234", ok=False, gated=True, satisfied=True)]
    unsatisfied = [_env_call(CALL_TOOL, "foo_bar_1234", gated=True, satisfied=False)]
    assert env_metrics.discoverable_tool_unlock(refused, g)["rate"] == 0.0
    assert env_metrics.discoverable_tool_unlock(unsatisfied, g)["rate"] == 0.0


def test_a_task_that_requires_no_discoverable_tool_is_not_scored_zero(docs):
    """48 of the 97 tasks name no discoverable tool. Scoring them 0 would report that every
    policy fails half the suite at something the suite never asked for."""
    solo = {
        "id": "only_gamma",
        "user_scenario": {"instructions": "Nothing relevant here."},
        "required_documents": ["doc_gamma_003"],
    }
    g = build_task_graph(solo, docs, extract_tool_edges(docs, FIXTURE_TOOLS))
    out = env_metrics.discoverable_tool_unlock([], g)
    assert out["n_required"] == 0.0
    assert not env_metrics.is_measured(out["rate"])


def test_the_gate_tool_names_agree_across_the_firewall():
    """pi_eval may not import an adapter, so the two tool-name constants are two copies. This
    is the test that keeps them one value."""
    from pinq_adapters.tau2 import actuator as adapter_side

    assert env_metrics.UNLOCK_TOOL == adapter_side.UNLOCK_TOOL
    assert env_metrics.CALL_TOOL == adapter_side.CALL_TOOL


# --------------------------------------------------------------------- the parquet columns


def test_the_reward_survives_compaction_into_parquet(tmp_path):
    """`Outcome.native` and `Outcome.env_final_hashes` were both written to outcome.json and
    both dropped on the floor by `pi compact`: no column carried them, so the primary endpoint
    could not have been scored even from a correct rollout."""
    from pi_run.compact import compact

    run = tmp_path / "runs" / "dev-abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        _json.dumps(
            {
                "run_id": "dev-abc",
                "suite_id": "tau2",
                "task_id": "task_001",
                "arm_id": "inquirer_prompted",
                "template_id": "doc:credit_cards",
            }
        )
    )
    (run / "status.json").write_text(_json.dumps({"status": "ok"}))
    (run / "outcome.json").write_text(
        _json.dumps(
            {
                "answer": None,
                "env_calls": [],
                "env_final_hashes": {"db": "AAA", "user_db": "BBB"},
                "native": {"tau_reward": 1.0, "db_match": 1.0, "not_a_number": "nope"},
            }
        )
    )
    res = compact(tmp_path / "runs", tmp_path / "parquet")

    import pyarrow.parquet as pq

    runs = pq.read_table(tmp_path / "parquet" / "runs.parquet").to_pylist()
    assert runs[0]["env_db_hash"] == "AAA" and runs[0]["env_user_db_hash"] == "BBB"
    assert runs[0]["template_id"] == "doc:credit_cards"

    native = pq.read_table(tmp_path / "parquet" / "native.parquet").to_pylist()
    assert {(r["key"], r["value"]) for r in native} == {("tau_reward", 1.0), ("db_match", 1.0)}
    assert all(r["suite_id"] == "tau2" for r in native)
    assert res.counts["native"] == 2, "a non-numeric native field is dropped, not coerced"


def test_the_native_table_is_declared_and_populated():
    assert "native" in sch.POPULATED
    assert set(sch.schema_for("native").names) == {"run_id", "suite_id", "key", "value"}
    assert {"env_db_hash", "env_user_db_hash"} <= set(sch.schema_for("runs").names)


# =============================================== the action path, against the LIVE environment


@pytest.fixture(scope="module")
def live_suite():
    """One Tau2Suite for the read-only live tests. Building an Environment costs a BM25
    index, and these tests only ever read from it."""
    ok, why = available()
    if not ok:
        pytest.skip(why)
    from pinq_adapters.tau2 import Tau2Suite

    return Tau2Suite()


@pytest.mark.integration
@integration
def test_tool_schemas_offer_the_advertised_tools_and_never_a_suffixed_one(live_suite):
    """THE MOST IMPORTANT ASSERTION IN THE SUITE.

    The 44 discoverable tools carry an unguessable four-digit suffix, so a policy can only
    unlock one by READING the document that names it — and that prerequisite IS the mechanic
    the paper measures. Handing the Drafter `open_bank_account_4821` up front would satisfy
    the endpoint with no discovery at all and make the headline result an artifact of the
    harness. The gate itself must be offered, or a policy that DID read has no way to act.
    """
    from pinq_adapters.tau2.discoverable import TOOL_TOKEN_RE

    schemas = live_suite.tool_schemas(live_suite.task_ids()[0])
    names = {s["name"] for s in schemas}
    assert not [n for n in names if TOOL_TOKEN_RE.fullmatch(n)]
    assert {"unlock_discoverable_agent_tool", "call_discoverable_agent_tool"} <= names
    assert all(isinstance(s["parameters"], dict) for s in schemas)
    # every schema is the one upstream advertises, so our drafter and upstream's baseline
    # agent are shown the same contract
    assert any(s["parameters"].get("properties") for s in schemas)


@pytest.mark.integration
@integration
def test_the_derived_template_id_actually_clusters(live_suite):
    """A clustered bootstrap over one-task clusters is not clustered at all.

    Upstream's only per-task label is `description.purpose` == "Task: task_017", 97 distinct
    values. The derived id must be materially coarser than that or the tau2 primary
    endpoint's SE is the unclustered one under a clustered name.
    """
    ids = live_suite.task_ids()
    clusters = {t: live_suite.template_id(t) for t in ids}
    distinct = set(clusters.values())
    assert len(ids) == 97
    assert 2 <= len(distinct) < len(ids) / 4, f"{len(distinct)} clusters over {len(ids)} tasks"
    assert clusters["task_001"].startswith("doc:")


@pytest.mark.integration
@integration
def test_the_retriever_and_the_actuator_share_one_world(live_suite):
    """A rollout is ONE environment. Two would let the policy read from world A and mutate
    world B, and the DB hash the reward is computed over would be a hash of a world nobody
    ever read from."""
    tid = live_suite.task_ids()[0]
    assert live_suite.environment(tid) is live_suite.environment(tid)
    assert live_suite.retriever(tid)._env is live_suite.actuator(tid)._env


@pytest.mark.integration
@integration
def test_the_read_log_allowlist_comes_from_upstreams_own_deriver(live_suite):
    """Not a tuning knob. `call_discoverable_agent_tool` logs into a table that is hashed
    into the reward, so a different allowlist than the grader's makes the predicted and gold
    environments incomparable."""
    from tau2.runner.build import _derive_read_log_allowlist

    tid = "task_046"
    kwargs = live_suite.env_kwargs(tid)
    assert kwargs["retrieval_variant"] == "bm25"
    assert kwargs["read_log_allowlist"] == _derive_read_log_allowlist(
        live_suite.tau2_task_object(tid)
    )


@pytest.mark.integration
@integration
@pytest.mark.parametrize("tid", ["task_036", "task_046"])
def test_replaying_the_gold_actions_scores_the_primary_endpoint_at_one(tid):
    """THE PATH TEST FOR P1. A rollout that scores 0 cannot distinguish "the policy was
    wrong" from "the reward is still structurally unreachable". Feeding the benchmark's OWN
    reference action sequence through the same Actuator must produce 1.0, or the endpoint is
    broken rather than hard.

    A fresh suite per parametrization: the actuator MUTATES the environment, so a shared one
    would carry task_036's replacement card into task_046's grading.
    """
    from pinq_adapters.tau2 import Tau2Suite

    suite = Tau2Suite()
    rec = suite.task_record(tid)
    gold = (rec["evaluation_criteria"] or {}).get("actions") or []
    act = suite.actuator(tid)
    calls = act.execute(
        [
            {
                "name": a["name"],
                "args": a.get("arguments") or {},
                "requestor": a.get("requestor", "assistant"),
            }
            for a in gold
        ],
        turn_idx=0,
    )
    assert calls and all(c.ok for c in calls)
    native = dict(act.native())
    assert act.reward_error == ""
    assert native["tau_reward"] == 1.0, native
    assert native["db_match"] == 1.0


@pytest.mark.integration
@integration
def test_doing_nothing_scores_the_primary_endpoint_at_zero():
    """The other half of the path test. If the empty plan also scored 1.0, the metric would be
    measuring nothing at all — which is exactly what an ACTION-basis task returns from the
    environment evaluator and why `tau_reward` is gated on the reward basis."""
    from pinq_adapters.tau2 import Tau2Suite

    act = Tau2Suite().actuator("task_036")
    act.execute([], turn_idx=0)
    assert dict(act.native())["tau_reward"] == 0.0


@pytest.mark.integration
@integration
def test_the_live_gold_graph_carries_the_mechanical_edge_set(tmp_path, live_suite):
    """The corpus-level claim, as it lands in gold. Measured on v1.0.1: 62 doc->tool edges
    over 43 distinct tools and 41 distinct documents, with 3 documented decoys excluded."""
    from pi_eval.build.tau2_build import build as tau2_build_gold
    from pinq_adapters.tau2 import documented_but_not_a_tool
    from pinq_adapters.tau2.retriever import env_tool_names

    env = live_suite.environment(live_suite.task_ids()[0])
    edges = live_suite.tool_edges(env)
    assert len(edges) == 62
    assert len({e.tool_name for e in edges}) == 43
    assert len({e.doc_id for e in edges}) == 41
    assert len(documented_but_not_a_tool(live_suite.documents, env_tool_names(env))) == 3

    res = tau2_build_gold(
        root=tmp_path,
        documents=list(live_suite.documents),
        tasks=[live_suite.task_record(t) for t in live_suite.task_ids()],
        tool_edges=list(edges),
    )
    graphs = read_graphs(res.gold)
    assert res.n_tasks == 97 and len(graphs) == 97
    tools = {a for g in graphs for a in env_metrics.required_tools(g)}
    assert tools == {e.tool_name for e in edges}
    assert res.corpus_hash == live_suite.corpus_hash


@integration
def test_the_view_refuses_to_hand_the_agent_the_customers_script():
    """The nastiest adapter bug found in this project, caught only by reading a live answer.

    On all 97 banking tasks `user_scenario.instructions` is a customer ROLEPLAY SCRIPT written
    for the user simulator -- "You are playing the role of a customer... Your character is Sera
    Chen, a high ranking official at the EPA...". Pasting it into the agent's view did two
    separate kinds of damage:

      * it inverted the role. Observed live: the agent answered AS Sera Chen and never
        attempted the banking task.
      * it handed over every user-private fact, collapsing the discoverable-from-KB versus
        user-private partition and making the ADR ceiling -- the bound on what ANY autonomous
        inquirer could reach -- meaningless.

    `description.purpose` is not a substitute: it is literally "Task: task_00N" on all 97.
    """
    from pinq_adapters.tau2.suite import Tau2NeedsOrchestrator, Tau2Suite

    suite = Tau2Suite()
    with pytest.raises(Tau2NeedsOrchestrator, match="roleplay script"):
        suite.view(suite.task_ids()[0])

    # The escape hatch exists, is ugly to type, and still yields the script -- so a run made
    # with it is identifiable and must never reach a reported table.
    leaky = Tau2Suite(allow_user_script=True)
    v = leaky.view(suite.task_ids()[0])
    assert len(v.question) > 0

    # And the leak really is a roleplay script, on every task -- not a one-off.
    n = sum(
        1
        for tid in suite.task_ids()
        if "playing the role" in leaky.view(tid).question.lower()
        or "your character" in leaky.view(tid).question.lower()
    )
    assert n == len(suite.task_ids()) == 97, f"{n} of {len(suite.task_ids())}"


# --------------------------------------------------------------------------- the driver
#
# tau2 had a suite, a retriever, an actuator and an agent class, and nothing that CONSTRUCTED
# an Orchestrator, ran it, harvested it and wrote the seven files `pi compact` reads -- so the
# suite was reachable only through the flat path, which correctly refuses. These pin the
# driver that closes that gap. Everything here runs with NO network: the customer is scripted
# and the arm is LLM-free, so what is exercised is the Orchestrator wiring, the dialogue view,
# the tool-call channel, the env-call harvest, upstream's grader and all seven files.


def _scripted_user_class():
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator_base import HalfDuplexUser

    class ScriptedUser(HalfDuplexUser):
        """A customer who says exactly these lines, then ###STOP###."""

        def __init__(self, lines):
            super().__init__(instructions=None, tools=None)
            self._lines = list(lines)

        def get_init_state(self, message_history=None):
            return {"i": 0}

        def set_seed(self, seed):
            return None

        def generate_next_message(self, message, state):
            i = state["i"]
            state["i"] = i + 1
            text = self._lines[i] if i < len(self._lines) else "Thanks. ###STOP###"
            return UserMessage(role="user", content=text), state

    return ScriptedUser


@pytest.mark.integration
def test_the_driver_runs_a_whole_tau2_unit_and_writes_seven_files(tmp_path):
    """Gate 1, offline. `pi compact` reads exactly seven files per run; a driver that writes
    six produces a run that is silently absent from every table rather than an error."""
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="testsha",
        dirty=False,
        max_turns=4,
        k=3,
    )
    user = _scripted_user_class()(["I would like to open a savings account, please."])
    res = run_tau2_unit(spec, user=user)
    assert res["status"] == "ok", res.get("error") or res

    d = Path(spec.runs_root) / res["run_id"]
    for f in (
        "manifest.json",
        "status.json",
        "outcome.json",
        "turns.jsonl",
        "calls.jsonl",
        "ledger.jsonl",
        "evidence.jsonl",
    ):
        assert (d / f).exists(), f

    manifest = _json.loads((d / "manifest.json").read_text())
    assert manifest["suite_id"] == "tau2"
    # The primary endpoint is CLUSTERED at template_id; a None here silently makes every task
    # its own cluster, which is the same arithmetic as no clustering at all.
    assert manifest["template_id"] and manifest["template_id"] != tid
    # The user simulator is part of the environment: a different customer model is a different
    # task, so it must be inside run identity rather than in a footnote.
    assert "user_sim" in manifest["upstream_pins"]
    assert "max_steps" in manifest["upstream_pins"]

    outcome = _json.loads((d / "outcome.json").read_text())
    assert set(outcome["env_final_hashes"]) == {"db", "user_db"}
    assert outcome["transcript_digest"]
    assert res["n_assistant_rollouts"] >= 1
    assert res["n_messages"] > 1


@pytest.mark.integration
def test_the_dialogue_view_carries_the_users_words_and_never_the_roleplay_script():
    """On all 97 banking tasks the only task text upstream ships is a customer ROLEPLAY SCRIPT
    written for the simulator. Handing it to the agent inverts the role -- observed live: the
    agent answered as Sera Chen on a banking task -- and gives away every user-private fact,
    which collapses the partition the ADR ceiling is defined over."""
    from pi_run.stages.tau2_runner import dialogue_view
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    from tau2.data_model.message import AssistantMessage, UserMessage

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    script = str((suite.task_record(tid).get("user_scenario") or {}).get("instructions") or "")
    assert script, "the fixture assumes this task ships a scenario script"

    msgs = [
        AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
        UserMessage(role="user", content="I need to dispute a charge."),
        AssistantMessage(role="assistant", content="Certainly."),
        UserMessage(role="user", content="It was on the 4th."),
    ]
    view = dialogue_view(suite, tid, msgs)
    assert "dispute a charge" in view.question and "on the 4th" in view.question
    assert "Hi! How can I help" not in view.question, "the AGENT's own words are not the task"
    assert script[:60] not in view.question
    assert view.suite_id == "tau2" and view.corpus_hash == suite.corpus_hash

    empty = dialogue_view(suite, tid, ())
    assert empty.question, "an empty question is indistinguishable from a broken adapter"


def test_env_calls_are_derived_from_the_transcript_the_grader_replays():
    """Ground truth is the ORCHESTRATOR's transcript, not a private actuator log.

    `EnvironmentEvaluator` replays `message_history=full_trajectory` into a fresh environment,
    so a call executed anywhere else is invisible to the reward. Deriving the stored log from
    the same transcript is what stops `env_calls.parquet` and the reward column from becoming
    two accounts of one rollout with no way to tell which is wrong.
    """
    from pi_run.stages.tau2_runner import env_calls_from

    class TC:
        def __init__(self, id, name, arguments, requestor="assistant"):
            self.id, self.name, self.arguments, self.requestor = id, name, arguments, requestor

    class M:
        def __init__(self, role, tool_calls=None, id=None, content="", error=False):
            self.role, self.tool_calls, self.id = role, tool_calls, id
            self.content, self.error = content, error

    msgs = [
        M("user", content="hello"),
        M("assistant", [TC("a0", UNLOCK_TOOL, {"agent_tool_name": "open_x_4821"})]),
        M("tool", id="a0", content="unlocked"),
        M("assistant", [TC("a1", CALL_TOOL, {"agent_tool_name": "open_x_4821", "args": {}})]),
        M("tool", id="a1", content="{}"),
        M("assistant", [TC("a2", CALL_TOOL, {"agent_tool_name": "never_unlocked_9999"})]),
        M("tool", id="a2", content="Error: locked", error=True),
    ]
    calls = env_calls_from(msgs)
    assert [c.tool_name for c in calls] == [UNLOCK_TOOL, CALL_TOOL, CALL_TOOL]
    assert [c.seq for c in calls] == [1, 2, 3]
    assert calls[1].unlock_required and calls[1].unlock_satisfied
    assert calls[2].unlock_required and not calls[2].unlock_satisfied
    assert calls[2].ok is False
    assert all(c.turn_idx == 1 for c in calls), "attributed to the user turn that opened them"
    # A digest, never the payload: a tau2 tool result is a customer record.
    assert all(len(c.result_digest) == 16 for c in calls)
    assert "unlocked" not in _json.dumps([c.result_digest for c in calls])


def test_a_failed_unlock_does_not_credit_a_crossed_prerequisite_edge():
    """The unlock -> call edge IS the mechanic this suite exists to measure. Crediting a failed
    unlock would fabricate the crossing."""
    from pi_run.stages.tau2_runner import env_calls_from

    class TC:
        def __init__(self, id, name, arguments):
            self.id, self.name, self.arguments, self.requestor = id, name, arguments, "assistant"

    class M:
        def __init__(self, role, tool_calls=None, id=None, content="", error=False):
            self.role, self.tool_calls, self.id = role, tool_calls, id
            self.content, self.error = content, error

    msgs = [
        M("user", content="hi"),
        M("assistant", [TC("a0", UNLOCK_TOOL, {"agent_tool_name": "t_1"})]),
        M("tool", id="a0", content="Error: not documented", error=True),
        M("assistant", [TC("a1", CALL_TOOL, {"agent_tool_name": "t_1"})]),
        M("tool", id="a1", content="Error: locked", error=True),
    ]
    calls = env_calls_from(msgs)
    assert calls[0].ok is False
    assert calls[1].unlock_required and not calls[1].unlock_satisfied


def test_merged_turn_indices_are_renumbered_across_assistant_turns():
    """turn_idx is the x-index of the prefix ladder and `Trajectory.prefix(k)` slices on
    position, so three rollouts each starting at 0 would make prefix(2) mean "the first two
    turns of every rollout" -- a prefix of nothing the policy ran."""
    from pi_run.stages.tau2_runner import merge_trajectories
    from pinq.types import Ask, Evidence, EvidenceUnit, Outcome, Trajectory, Turn, Usage

    def mk(n, tag):
        units = tuple(
            EvidenceUnit.make(corpus_id="c", doc_id=f"{tag}{i}", span="0:1", title="t", text="x")
            for i in range(n)
        )
        return Trajectory(
            view=None,
            turns=tuple(
                Turn(turn_idx=i, action=Ask(text=f"{tag}{i}"), usage=Usage(n_calls=1))
                for i in range(n)
            ),
            evidence=Evidence(units=units),
            outcome=Outcome(),
            usage=Usage(),
            stop_reason="policy_stop",
            terminal_usage=Usage(n_calls=1),
        )

    ledger = BudgetLedger(cap=16)
    merged = merge_trajectories([mk(2, "a"), mk(3, "b")], view=None, ledger=ledger)
    assert [t.turn_idx for t in merged.turns] == [0, 1, 2, 3, 4]
    assert [t.action.text for t in merged.turns] == ["a0", "a1", "b0", "b1", "b2"]
    assert len(merged.evidence.units) == 5, "evidence is unioned, not concatenated with dupes"
    assert merged.terminal_usage.n_calls == 2, "one terminal window per assistant turn"


def test_the_driver_is_the_only_tau2_path_out_of_the_worker():
    """A flat tau2 unit must reach the driver, not `Tau2Suite.view()`'s refusal."""
    import inspect

    from pi_run import worker

    src = inspect.getsource(worker.run_unit)
    # MEMBERSHIP, not equality. This line pinned `spec.suite_id == "tau2"`, which was the
    # routing MECHANISM rather than the property; the second tau2 domain (retail) is also a
    # dialogue benchmark and equality sent it down the flat path, where `view()` raises.
    assert "DIALOGUE_SUITES" in src
    assert "run_tau2_unit" in src
    # Before the generic path builds anything: a tau2 unit must not construct a flat view.
    assert src.index("run_tau2_unit") < src.index("load_suite(spec.suite_id")


@pytest.mark.integration
def test_a_policy_that_takes_the_gold_actions_scores_one_through_the_orchestrator(tmp_path):
    """THE TEST THAT MAKES A ZERO MEANINGFUL.

    `pi verify tau2 --replay-gold` proves the ACTUATOR path reproduces the gold DB hash on
    97/97. It says nothing about the driver, and the driver takes a completely different
    route: the Drafter's tool plan is emitted as `AssistantMessage.tool_calls`, the
    Orchestrator executes them, and `evaluate_simulation` replays the transcript. If any link
    in that chain is wrong, `tau_reward` is 0.0 for every arm forever -- and 0.0 is exactly
    what a policy that simply failed the task would produce, so nothing downstream could tell
    a broken harness from a bad policy. P1 would read as a flat null.

    So: a Drafter whose tool plan IS the answer key, driven through the real Orchestrator.
    Reward must come back 1.0.
    """
    import contextlib
    import io

    from pi_run.stages.tau2_runner import _reward_of, _simulate
    from pi_run.worker import UnitSpec
    from pinq.budget import BudgetLedger
    from pinq.types import Answer, Draft, Stop
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()

    def _actions(t):
        return list(getattr(suite.tau2_task_object(t).evaluation_criteria, "actions", None) or [])

    def _empty_plan_scores(t):
        """What this task scores when the agent does NOTHING. Measured, never assumed."""
        from pinq_adapters.tau2.actuator import Tau2Actuator

        with contextlib.redirect_stdout(io.StringIO()):
            a = Tau2Actuator(
                suite.environment(t), task=suite.task_record(t), env_kwargs=suite.env_kwargs(t)
            )
            a.execute([], turn_idx=0)
            return a.native().get("db_reward")

    # THREE CONDITIONS, AND THE THIRD IS THE ONE THIS TEST FORGOT.
    #
    #  * ASSISTANT-ONLY. 102 of the 955 gold actions belong to the CUSTOMER
    #    (apply_for_credit_card, submit_referral, ...) and 7 of the 97 tasks are graded
    #    entirely on them; with a scripted customer those are unreachable.
    #  * EXERCISES THE DISCOVERY MECHANIC. 730 of the 853 assistant gold actions are
    #    unlock/call pairs, so a task without one is not representative of what tau2 measures.
    #  * AN EMPTY PLAN MUST SCORE 0.0. The first assistant-only task is task_004, whose entire
    #    answer key is one NON-MUTATING `transfer_to_human_agents` -- so doing nothing also
    #    scores 1.0, and this test passed no matter what the driver did. A test that cannot
    #    fail is not a test. Measured: 21 of the first 25 candidates discriminate; task_004
    #    is one of the 4 that do not.
    tid = next(
        t
        for t in suite.task_ids()
        if _actions(t)
        and {str(a.requestor) for a in _actions(t)} == {"assistant"}
        and any("discoverable" in a.name for a in _actions(t))
        and _empty_plan_scores(t) == 0.0
    )
    actions = _actions(tid)
    assert _empty_plan_scores(tid) == 0.0, "the control: doing nothing must NOT score 1.0 here"
    plan = tuple(
        {"name": a.name, "args": dict(a.arguments), "requestor": a.requestor} for a in actions
    )

    class GoldDrafter:
        """Plans the answer key. Everything else about it is inert."""

        def resolve(self, view, action, ev, *, seed, ledger):
            return "", ev

        def draft(self, view, ev, *, seed, ledger):
            return Draft(text="", tool_plan=plan)

    class NeverAsk:
        policy_id = "gold_plan"

        def reset(self, view, seed):
            return None

        def act(self, s):
            return Stop(reason="policy_stop")

    class Answerer:
        def answer(self, view, ev, draft, *, seed, ledger):
            return Answer(text="Done.", evidence_hash=ev.subset_hash)

    ledger = BudgetLedger(cap=64)
    calls = {"n": 0}

    def build_parts():
        calls["n"] += 1
        return {
            "inquirer": NeverAsk(),
            "drafter": GoldDrafter(),
            "answerer": Answerer(),
            "retriever": suite.retriever(tid),
            "ledger": ledger,
        }

    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
    )
    # The customer says one thing and stops, so exactly ONE rollout produces the plan.
    user = _scripted_user_class()(["Please go ahead.", "Thanks. ###STOP###"])
    with contextlib.redirect_stdout(io.StringIO()):
        sim, state, _task = _simulate(suite, spec, build_parts, user=user)
        native, err = _reward_of(sim, suite, tid)

    assert err == "", err
    assert native.get("db_reward") == 1.0, (
        f"the gold action sequence must reproduce the gold DB hash through the Orchestrator; "
        f"got {native}"
    )
    assert native.get("db_match") == 1.0
    # Every planned action reached the transcript as a real tool call.
    emitted = [
        tc.name
        for m in sim.messages
        for tc in (getattr(m, "tool_calls", None) or ())
        if str(getattr(tc, "requestor", "assistant")) == "assistant"
    ]
    assert emitted == [a.name for a in actions if a.requestor == "assistant"]
    assert calls["n"] == 1, "one rollout per USER message, never one per tool result"
    # And the mechanic the paper is about really was exercised.
    assert any("unlock_discoverable" in n for n in emitted)
    assert any("call_discoverable" in n for n in emitted)


@pytest.mark.integration
def test_the_shape_of_tau2s_gold_actions_is_what_the_design_assumes():
    """Measured on v1.0.1, pinned here because three separate claims rest on it.

    * 456 `call_discoverable_agent_tool` + 274 `unlock_discoverable_agent_tool` out of 853
      assistant actions: **76% of everything the agent is graded on IS the discovery
      mechanic** -- read the naming document, unlock, then call. That is the prerequisite edge
      the paper is about, and it is why tau2 is the L2 suite rather than a nice-to-have.
    * 102 of 955 actions belong to the CUSTOMER, and 7 of 97 tasks are graded entirely on
      customer actions. On those the agent's only lever is informing the user (or
      `give_discoverable_user_tool`), so a driver that emitted them as agent calls would be
      impersonating the customer, and a scripted-customer test cannot reach them at all.
    * Every non-user gold tool IS in the agent's advertised set, which is what makes the
      driver's allow-list filter safe: it can only ever drop a hallucination or a user tool.

    An upstream data change that moved any of these would move the tau2 story without touching
    a line of code, which is the failure mode `strict_counts` exists to prevent elsewhere.
    """
    from collections import Counter

    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    by_requestor: Counter = Counter()
    by_tool: Counter = Counter()
    ownership: Counter = Counter()
    for tid in suite.task_ids():
        acts = list(getattr(suite.tau2_task_object(tid).evaluation_criteria, "actions", None) or [])
        roles = {str(a.requestor) for a in acts}
        ownership[
            "none"
            if not acts
            else "assistant-only"
            if roles == {"assistant"}
            else "user-only"
            if roles == {"user"}
            else "both"
        ] += 1
        for a in acts:
            by_requestor[str(a.requestor)] += 1
            by_tool[(str(a.requestor), a.name)] += 1

    assert dict(by_requestor) == {"assistant": 853, "user": 102}
    assert dict(ownership) == {"assistant-only": 48, "both": 42, "user-only": 7}
    discovery = (
        by_tool[("assistant", "call_discoverable_agent_tool")]
        + by_tool[("assistant", "unlock_discoverable_agent_tool")]
    )
    assert discovery == 730
    assert discovery / by_requestor["assistant"] > 0.85

    advertised = {t["name"] for t in suite.tool_schemas(suite.task_ids()[0])}
    unadvertised = {name for (req, name) in by_tool if name not in advertised}
    user_tools = {t.name for t in suite.environment(suite.task_ids()[0]).get_user_tools()}
    assert unadvertised <= user_tools, (
        f"a gold action the AGENT must take is not in its advertised tools: "
        f"{sorted(unadvertised - user_tools)}"
    )


@pytest.mark.integration
def test_the_driver_refuses_to_emit_a_tool_the_agent_was_never_offered(tmp_path):
    """Enough "Tool 'x' not found" errors end the simulation (max_errors), so a policy that
    names tools badly would be scored on a truncated dialogue rather than on its actions."""
    from pi_run.stages.tau2_runner import _simulate
    from pi_run.worker import UnitSpec
    from pinq.budget import BudgetLedger
    from pinq.types import Answer, Draft, Stop
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    plan = (
        {"name": "KB_search", "args": {"query": "savings"}},
        {"name": "apply_for_credit_card", "args": {}},  # a USER tool
        {"name": "definitely_not_a_tool", "args": {}},  # a hallucination
    )

    class Drafter:
        def resolve(self, view, action, ev, *, seed, ledger):
            return "", ev

        def draft(self, view, ev, *, seed, ledger):
            return Draft(text="", tool_plan=plan)

    class NeverAsk:
        policy_id = "x"

        def reset(self, view, seed):
            return None

        def act(self, s):
            return Stop(reason="policy_stop")

    class Answerer:
        def answer(self, view, ev, draft, *, seed, ledger):
            return Answer(text="Done.", evidence_hash=ev.subset_hash)

    ledger = BudgetLedger(cap=16)
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
    )
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        sim, state, _task = _simulate(
            suite,
            spec,
            lambda: {
                "inquirer": NeverAsk(),
                "drafter": Drafter(),
                "answerer": Answerer(),
                "retriever": suite.retriever(tid),
                "ledger": ledger,
            },
            user=_scripted_user_class()(["Hello.", "Thanks. ###STOP###"]),
        )

    emitted = [tc.name for m in sim.messages for tc in (getattr(m, "tool_calls", None) or ())]
    assert "definitely_not_a_tool" not in emitted
    assert "apply_for_credit_card" not in emitted, "the agent must not impersonate the customer"
    assert "KB_search" in emitted
    assert state.rejected == 2, "rejections are COUNTED, never a silence"


# --------------------------------------------------------------------------- driver defects
#
# Found by an adversarial review of the driver, each confirmed by reproduction. All four share
# a shape: a well-formed value that is wrong, or an absent one that reads as clean.


def _asking_parts(suite, tid, ledger, n_asks):
    """Components that ask `n_asks` times then stop, with no LLM and no tool plan."""
    from pinq.types import Answer, Ask, Draft, Stop

    class Inq:
        policy_id = "asker"

        def reset(self, view, seed):
            self.n = 0

        def act(self, s):
            self.n += 1
            return Ask(text=f"q{self.n}") if self.n <= n_asks else Stop(reason="policy_stop")

    class Draf:
        def resolve(self, view, action, ev, *, seed, ledger):
            return "", ev

        def draft(self, view, ev, *, seed, ledger):
            return Draft(text="d")

    class Ans:
        def answer(self, view, ev, draft, *, seed, ledger):
            return Answer(text="ok", evidence_hash=ev.subset_hash)

    return lambda: {
        "inquirer": Inq(),
        "drafter": Draf(),
        "answerer": Ans(),
        "retriever": suite.retriever(tid),
        "ledger": ledger,
    }


@pytest.mark.integration
def test_a_dialogue_that_hits_max_steps_still_earns_a_tau_reward(tmp_path):
    """THE ONE THAT WOULD HAVE MOVED A PUBLISHED NUMBER.

    `evaluate_simulation`'s first statement returns a RewardInfo with NO db_check when the
    termination reason is not AGENT_STOP/USER_STOP. `attach_reward` writes tau_reward only when
    a db_check exists, so a dialogue that ran to max_steps produced `{"reward": 0.0}` with no
    tau_reward -- and `reward_error` stayed "", so the absence read as a clean grade. Downstream
    the run contributed NO ROW to the tau2 primary endpoint instead of the 0.0 it had earned.

    The bias flatters the worse arm: max_errors counts failed tool calls, 730 of tau2's 853
    assistant gold actions are unlock/call pairs where a wrong guess errors, so an arm that
    flails has more of its failures DELETED from the denominator and its mean rises.
    """
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator_base import HalfDuplexUser

    class NeverStops(HalfDuplexUser):
        def __init__(self):
            super().__init__(instructions=None, tools=None)
            self.i = 0

        def get_init_state(self, message_history=None):
            return {}

        def set_seed(self, seed):
            return None

        def generate_next_message(self, message, state):
            self.i += 1
            return UserMessage(role="user", content=f"And another thing ({self.i})."), state

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
        timeout_s=900,
    )
    res = run_tau2_unit(spec, user=NeverStops())
    assert res["status"] == "ok", res.get("error")
    assert res["terminated_prematurely"] is True
    assert "MAX_STEPS" in res["termination_reason"]

    native = res["native"]
    assert "tau_reward" in native, (
        "a truncated dialogue did not reach the gold DB state, so 0.0 is a MEASUREMENT of this "
        "rollout; dropping the row deletes exactly the failing runs from the denominator"
    )
    assert native["tau_reward"] == 0.0
    assert native["db_match"] == 0.0


@pytest.mark.integration
def test_the_final_db_hashes_are_real_and_not_empty_strings(tmp_path):
    """`getattr(sim, "environment", None)` -- SimulationRun has no such field, so both hashes
    were "" on every tau2 run ever written. An empty hash compares equal to another empty hash,
    which makes "the DB matched" trivially true between any two runs that recorded nothing."""
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
    )
    res = run_tau2_unit(spec, user=_scripted_user_class()(["Hello."]))
    outcome = _json.loads((Path(spec.runs_root) / res["run_id"] / "outcome.json").read_text())
    for key in ("db", "user_db"):
        assert outcome["env_final_hashes"][key], f"{key} hash is empty"
        assert len(outcome["env_final_hashes"][key]) > 16


@pytest.mark.integration
def test_a_tau2_run_carries_the_same_reconcile_block_the_flat_path_writes(tmp_path):
    """`pi_run.compact` reads `status["reconcile"]`. The driver wrote only a bare
    `status["reconciled"]`, so EVERY tau2 row compacted with reconciled_tokens=False and
    reconciled_docs=False regardless of the truth -- and `docs_ok`, the one detector for an
    unmetered nested retrieval, was never computed on tau2 at all."""
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=suite.task_ids()[0],
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
    )
    res = run_tau2_unit(spec, user=_scripted_user_class()(["Hello."]))
    assert set(res["reconcile"]) >= {"tokens_ok", "docs_ok", "turn_usage", "terminal_usage"}
    assert res["reconciled"] is (res["reconcile"]["tokens_ok"] and res["reconcile"]["docs_ok"])

    from pi_run.compact import compact

    compact(Path(spec.runs_root), tmp_path / "parquet", include_dev=True)
    import pyarrow.parquet as pq

    row = pq.read_table(tmp_path / "parquet" / "runs.parquet").to_pylist()[0]
    assert row["reconciled_docs"] is True, "compact read an absent reconcile block as False"
    assert row["reconciled_tokens"] is True


@pytest.mark.integration
def test_the_ledger_turn_axis_continues_across_assistant_turns(tmp_path):
    """ledger.jsonl is joined to turns.jsonl on turn_idx. `run_loop` counts turns from 0, and a
    tau2 unit runs one run_loop per USER message against ONE shared ledger -- so the second
    rollout's spend was stamped 0,1,2 again and attributed to the FIRST message's turns."""
    from pi_run.stages.tau2_runner import _simulate, merge_trajectories
    from pi_run.worker import UnitSpec
    from pinq.budget import BudgetLedger
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    suite = Tau2Suite()
    tid = suite.task_ids()[0]
    ledger = BudgetLedger(cap=64)
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=tid,
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=3,
        k=2,
    )
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        _sim, state, _task = _simulate(
            suite,
            spec,
            _asking_parts(suite, tid, ledger, n_asks=2),
            user=_scripted_user_class()(["First.", "Second.", "Thanks. ###STOP###"]),
        )
    assert len(state.trajectories) >= 2, "the test needs more than one assistant rollout"

    merged = merge_trajectories(state.trajectories, view=None, ledger=ledger)
    turn_ids = [t.turn_idx for t in merged.turns]
    assert turn_ids == list(range(len(turn_ids))), turn_ids

    ledger_turns = sorted({r.turn_idx for r in ledger.rows})
    assert max(ledger_turns) >= 2, (
        f"the ledger restarted its turn axis: {ledger_turns} against {turn_ids}. "
        "Rows from the second rollout would join onto the first rollout's turns."
    )
    assert set(ledger_turns) <= set(turn_ids)


@pytest.mark.integration
def test_the_customers_failed_tool_calls_are_counted_separately(tmp_path):
    """Upstream's `max_errors` counts ANY failed tool call against the agent, whatever the
    requestor -- `Orchestrator._execute_tool_calls` does `num_errors += 1` unconditionally --
    and tripping it terminates the dialogue, which now scores tau_reward = 0.0.

    Measured on the first three live dialogues: the AGENT made zero errors while the user
    simulator made 7 and 8 of a 10-error budget, repeatedly trying to call the agent's
    knowledge-base tools itself. Two of three came within 2-3 errors of being killed for the
    customer's mistakes -- recorded as a policy failure. So the counts are split, and the cap
    bounds what it is for.
    """
    from pi_run.stages.tau2_runner import DEFAULT_MAX_ERRORS, run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.suite import Tau2Suite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    assert DEFAULT_MAX_ERRORS > 10, (
        "upstream's 10 is spent by the user simulator's own failures before the agent has made "
        "any; raising it bounds the agent rather than the environment's noise"
    )

    suite = Tau2Suite()
    spec = UnitSpec(
        suite_id="tau2",
        corpus_dir=str(tmp_path),
        task_id=suite.task_ids()[0],
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="t",
        dirty=False,
        max_turns=2,
        k=3,
    )
    res = run_tau2_unit(spec, user=_scripted_user_class()(["Hello."]))
    assert "n_errors_agent" in res and "n_errors_user" in res
    assert res["n_errors_agent"] == 0

    manifest = _json.loads((Path(spec.runs_root) / res["run_id"] / "manifest.json").read_text())
    assert manifest["upstream_pins"]["max_errors"] == str(DEFAULT_MAX_ERRORS), (
        "the cap bounds how long a dialogue may run, so two caps are two experiments"
    )


@pytest.mark.integration
def test_budget_exceeded_during_harvest_still_persists_ledger_and_failed_status(
    tmp_path, monkeypatch
):
    """On a live retail baseline, 3 of 3 units raised BudgetExceeded 13-16 turns in, during the
    post-simulate harvest step (`_attach_env_evidence` -> `meter_env_calls`), because retail
    has no retriever and its 15 typed tools are charged to the ledger only there -- outside the
    only try/except `run_tau2_unit` has. That region sits AFTER `_simulate` and BEFORE
    `reconcile()`, so nothing landed on disk: no status.json, no ledger.jsonl, for calls that
    had already been billed. $30-47 of real spend, zero trace.

    THE HARVEST NO LONGER RAISES, and this test does not assume it does. `meter_env_calls`
    records the spend and returns an overrun flag rather than throwing, because raising after the
    dialogue is over deletes a completed run instead of preventing a spend
    (`tests/test_post_hoc_harvest_meter.py` replays the 204 runs it deleted). What is under test
    here is the OTHER half of the same incident and it is still live: whatever kills a unit in
    that region -- a watchdog timeout, a grader error, a future charge -- must still leave a
    status.json and a ledger.jsonl behind. So the raise is INJECTED, by a fake
    `_attach_env_evidence` that charges the cap over on purpose, and the assertions below are
    about what reaches disk, not about what the real meter does.

    Reproduced here with N=3 FAKE charges (`ledger.charge_retrieval`, never a real gateway
    call) so the count is exact and the test costs nothing.
    """
    import pi_run.stages.tau2_runner as tau2_runner
    from pi_run.stages.tau2_runner import run_tau2_unit
    from pi_run.worker import UnitSpec
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite

    ok, why = available()
    if not ok:
        pytest.skip(why)

    n_billed = 3
    real_attach = tau2_runner._attach_env_evidence

    # `**kw` because the runner now passes `charge=` (False on a unit `BudgetGate` already
    # charged); the double must accept the real call's signature to inject the raise at all.
    def fake_attach(traj, suite_arg, ledger, env_calls, **kw):
        for _ in range(n_billed):
            ledger.charge_retrieval(1.0)  # calls the provider actually billed
        ledger.charge_retrieval(1.0)  # the (n+1)-th: this is the one that blows the cap
        return real_attach(traj, suite_arg, ledger, env_calls, **kw)  # never reached

    monkeypatch.setattr(tau2_runner, "_attach_env_evidence", fake_attach)

    suite = Tau2RetailSuite()
    spec = UnitSpec(
        suite_id="tau2_retail",
        corpus_dir=str(tmp_path),
        task_id=suite.task_ids()[0],
        arm_id="fake_chain",
        seed=0,
        runs_root=str(tmp_path / "runs"),
        cache_root=str(tmp_path / "cache"),
        code_version="testsha",
        dirty=False,
        max_turns=2,
        k=3,
        budget_cap=n_billed,
    )
    res = run_tau2_unit(spec, user=_scripted_user_class()([]))

    assert res["status"] in ("error", "timeout"), res
    assert "BudgetExceeded" in res.get("error", ""), res.get("error")

    run_dir = Path(spec.runs_root) / res["run_id"]
    assert res["run_id"], "a unit killed after the manifest was built still has a run_id"
    assert (run_dir / "status.json").exists(), "the sentinel a killed unit must still leave"

    ledger_path = run_dir / "ledger.jsonl"
    assert ledger_path.exists(), (
        f"{n_billed} calls were billed before the overflow and left no ledger.jsonl at all"
    )
    rows = [_json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(rows) == n_billed, (
        f"expected {n_billed} billed rows on disk, got {len(rows)}: {rows}"
    )

    status = _json.loads((run_dir / "status.json").read_text())
    assert status["status"] in ("error", "timeout")
    # The part that matters most for the cost appendix: a reader summing costs from status.json
    # alone must see the true partial spend, not a blank/zero that reads as a free run.
    assert status.get("spent", {}).get("retrieval_calls") == n_billed, status.get("spent")


def test_a_failed_units_status_json_cannot_be_summed_as_a_free_run_by_compact(tmp_path):
    """The second half of the persistence defect: even if a failure DID write a status.json,
    `pi compact`'s own cost columns (`usd`, `retrieval_calls`) are read from `status["usage"]`
    and `status["spent"]`, never from `status["usd_billed"]`. A failed record that leaves those
    two sub-dicts empty is compacted as a free run, indistinguishable from a unit that never
    spent a cent -- which is exactly the failure mode a cost appendix cannot tolerate.

    This is the schema-legibility half of the fix: `run_tau2_unit`'s failure branch must
    populate `usage` / `spent` / `unique_docs` from the ledger, using the SAME field names the
    success branch already writes, not a new field nobody downstream reads.
    """
    from pi_run.compact import compact

    runs = tmp_path / "runs"
    d = runs / "run_failed_1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        _json.dumps(
            {
                "run_id": "run_failed_1",
                "suite_id": "tau2_retail",
                "task_id": "t1",
                "arm_id": "inquirer_prompted",
            }
        )
    )
    # Shaped exactly as `run_tau2_unit`'s failure branch writes it AFTER the fix: status +
    # error + the ledger-derived usage/spent/unique_docs, no turns/calls/outcome files.
    (d / "status.json").write_text(
        _json.dumps(
            {
                "status": "error",
                "error": "BudgetExceeded: retrieval_calls: 4.0 > 3",
                "usage": {"usd": 0.09},
                "spent": {"retrieval_calls": 3.0},
                "unique_docs": 0,
            }
        )
    )
    (d / "ledger.jsonl").write_text(
        "".join(
            _json.dumps(
                {
                    "currency": "retrieval_calls",
                    "charged": 1.0,
                    "cumulative": float(i + 1),
                    "turn_idx": 0,
                    "cap": 3,
                    "hard": True,
                }
            )
            + "\n"
            for i in range(3)
        )
    )

    out = tmp_path / "out"
    compact(runs, out)

    import pyarrow.parquet as pq

    runs_tbl = pq.read_table(out / "runs.parquet").to_pylist()
    row = next(r for r in runs_tbl if r["run_id"] == "run_failed_1")
    assert row["status"] == "error", row
    assert row["usd"] > 0.0, "a partial-spend failure must not compact to a free run"
    assert row["retrieval_calls"] == 3.0, row

    ledger_tbl = pq.read_table(out / "ledger.parquet").to_pylist()
    ledger_rows_for_run = [r for r in ledger_tbl if r["run_id"] == "run_failed_1"]
    assert len(ledger_rows_for_run) == 3, (
        "compact() reads ledger.jsonl unconditionally on status -- the 3 billed rows must be "
        "there for a reader who bypasses `usage`/`spent` and sums ledger.jsonl directly"
    )
