"""S0..S7 end to end, plus the gates that decide admissibility.

The pipeline's job is not only to emit a graph: it is to emit a graph that KNOWS ITS OWN
ERROR PROFILE. So most of what is asserted here is the diagnostics, not the nodes.
"""

import pytest

from pi_eval.mining.canon import Candidate, exact_entail
from pi_eval.mining.depth import salient_tokens, seed_basis_shares, seed_set
from pi_eval.mining.facets import components
from pi_eval.mining.partition import (
    discoverability_of,
    partition_elasticity,
    partition_from_ablation,
)
from pi_eval.mining.pipeline import gate_report, gates_pass, mine
from pi_eval.mining.pool import Cell, Trace, build_pool

FAMS = ("anthropic", "openai", "qwen", "llama")


def _trace(tid, cell, turn_uids, success=True):
    flat = tuple(u for _, us in turn_uids for u in us)
    return Trace(
        tid, "synth", "t", cell, 0, success, 1.0 if success else 0.0, flat, tuple(turn_uids)
    )


def _cand(text, trace, cell, fam, uids, turn=0):
    return Candidate(text, trace, cell, fam, "chain", tuple(uids), turn, "llm_elicited")


@pytest.fixture
def pool_and_cands():
    cells = [Cell(f, "chain", "bm25") for f in FAMS]
    traces = [_trace(f"t{i}", cells[i % 4], [(0, ("uA",)), (1, ("uB",))]) for i in range(8)]
    pool = build_pool(traces, suite_id="synth", task_id="t")
    cands = []
    for i, tr in enumerate(traces):
        cands.append(
            _cand(
                "acme was founded in 1912",
                tr.trace_id,
                tr.cell.cell_id,
                tr.cell.model_family,
                ("uA",),
                0,
            )
        )
        cands.append(
            _cand(
                "the founder later moved to berlin",
                tr.trace_id,
                tr.cell.cell_id,
                tr.cell.model_family,
                ("uB",),
                1,
            )
        )
    return pool, cands


# ------------------------------------------------------------------ seeds


def test_entity_coverage_marks_a_seed_mechanically():
    """No model in the loop: the need's salient tokens are already in the question."""
    q = "When was Acme founded, and where did the founder move?"
    v = {
        x.node_id: x
        for x in seed_set(
            q,
            {
                "n1": "Acme founded",
                "n2": "the founder moved to Berlin in 1930",
            },
        )
    }
    assert v["n1"].is_seed and v["n1"].basis == "entity_coverage"
    assert not v["n2"].is_seed, "Berlin and 1930 are not in the question"


def test_early_resolve_is_a_flagged_fallback_not_a_free_pass():
    q = "What happened?"
    v = {x.node_id: x for x in seed_set(q, {"n1": "Berlin 1930"}, early_resolved={"n1": 0.95})}
    assert v["n1"].is_seed and v["n1"].basis == "early_resolve"
    shares = seed_basis_shares(list(v.values()))
    assert shares["early_resolve"] == 1.0, "the weaker basis must be visible in the report"


def test_salient_tokens_ignore_lowercase_filler():
    assert salient_tokens("the Acme corp had 1912 revenue") == {"acme", "1912"}


# ------------------------------------------------------------------ facets


def test_facets_exclude_the_seed_frontier():
    """With seeds left in, everything is one component and breadth is a constant 1."""
    nodes = ["s1", "s2", "a1", "a2", "b1"]
    edges = [("s1", "a1"), ("a1", "a2"), ("s2", "b1")]
    f = components(nodes, edges, ["s1", "s2"], task_key="t")
    assert len(f) == 2
    assert sorted(sorted(x.node_ids) for x in f) == [["a1", "a2"], ["b1"]]


def test_facets_join_siblings_that_share_a_prerequisite():
    """Two needs neither of which depends on the other are still the same facet."""
    f = components(["s", "a", "b"], [("s", "a"), ("s", "b"), ("a", "b")], ["s"], task_key="t")
    assert len(f) == 1 and set(f[0].node_ids) == {"a", "b"}


# ------------------------------------------------------------------ partition


def test_ablation_verdict_drives_the_partition_not_frequency():
    assert partition_from_ablation("NECESSARY") == ("required", True)
    assert partition_from_ablation("CONTRIBUTORY") == ("optional", True)
    assert partition_from_ablation("INERT") == ("dropped", False)
    # UNTESTABLE is kept but must never inflate a required denominator
    assert partition_from_ablation("UNTESTABLE") == ("optional", False)


def test_discoverability_is_per_document_not_corpus_wide():
    """Tokens scattered across many documents are not something one retrieval can hand you."""
    docs = [{"id": "d1", "content": "acme was founded"}, {"id": "d2", "content": "in berlin 1912"}]
    disc, cov, _ = discoverability_of("acme founded berlin 1912", docs, threshold=0.6)
    assert disc == "user_private", f"scattered tokens scored {cov}"
    one = [{"id": "d3", "content": "acme was founded in berlin in 1912"}]
    assert discoverability_of("acme founded berlin 1912", one, threshold=0.6)[0] == "kb"


def test_elasticity_curve_is_monotone_in_the_threshold():
    """A stricter bar can only move needs INTO user_private; a non-monotone curve is a bug."""
    docs = [{"id": "d", "content": "alpha beta gamma"}]
    nodes = [
        {"node_id": "n1", "text": "alpha beta gamma delta"},
        {"node_id": "n2", "text": "alpha zeta eta theta"},
    ]
    curve = partition_elasticity(nodes, docs)
    vals = [curve[t] for t in sorted(curve)]
    assert vals == sorted(vals), f"non-monotone elasticity {curve}"


# ------------------------------------------------------------------ pipeline


def test_mine_emits_a_graph_that_knows_its_own_error_profile(pool_and_cands):
    pool, cands = pool_and_cands
    docs = [{"id": "d1", "content": "acme was founded in 1912"}]
    g = mine(
        suite="synth",
        task_key="t",
        question="When was Acme founded?",
        pool=pool,
        candidates=cands,
        entail=exact_entail,
        documents=docs,
        ablations={},
    )
    assert g.admissible
    d = g.diagnostics
    for key in (
        "n_candidates",
        "n_promoted",
        "promotion_rate",
        "orphan_rate",
        "user_private_share",
        "partition_elasticity",
        "theta_sweep",
        "seed_basis",
    ):
        assert key in d, f"missing diagnostic {key}"
    assert d["n_promoted"] == 2, "both needs appear across 4 families and 4 cells"


def test_graph_version_changes_with_theta_but_not_with_reordering(pool_and_cands):
    """graph_hash flows into scorer_hash, so it must be stable under reordering and
    sensitive to the instrument's operating point."""
    pool, cands = pool_and_cands
    kw = dict(suite="synth", task_key="t", question="q", pool=pool, entail=exact_entail)
    a = mine(candidates=cands, theta=0.85, **kw)
    b = mine(candidates=list(reversed(cands)), theta=0.85, **kw)
    c = mine(candidates=cands, theta=0.95, **kw)
    assert a.graph_hash == b.graph_hash
    assert a.graph_hash != c.graph_hash
    assert a.graph_version.startswith("synth/v1.0+")


def test_a_thin_pool_is_emitted_as_inadmissible_not_silently_dropped():
    """A systematically-skipped stratum must stay visible in the denominator."""
    c = Cell("anthropic", "chain", "bm25")
    pool = build_pool([_trace("t1", c, [(0, ("u1",))])], suite_id="synth", task_id="t")
    g = mine(
        suite="synth", task_key="t", question="q", pool=pool, candidates=[], entail=exact_entail
    )
    assert not g.admissible and "cells" in g.reason
    assert g.diagnostics["pool"]["n_success"] == 1


def test_mechanical_edges_are_counted_separately_from_intervention_edges(pool_and_cands):
    pool, cands = pool_and_cands
    g = mine(
        suite="synth",
        task_key="t",
        question="q",
        pool=pool,
        candidates=cands,
        entail=exact_entail,
        mechanical_pairs=[("x", "y")],
    )
    assert g.diagnostics["n_mechanical_edges"] == 1
    assert g.diagnostics["n_intervention_edges"] == 0


# ------------------------------------------------------------------ gates


def test_every_gate_carries_a_written_consequence():
    """A gate whose failure has no stated consequence is a decoration."""
    gates = gate_report(
        [],
        node_recall=0.80,
        edge_precision=0.85,
        matcher_kappa=0.72,
        contamination_delta=0.01,
        headline_effect=0.10,
    )
    assert gates and all(g.consequence.strip() for g in gates)
    assert gates_pass(gates)


def test_gm1_failure_demotes_mined_graphs_to_diagnostic_only():
    gates = {g.name: g for g in gate_report([], node_recall=0.60, edge_precision=0.85)}
    g = gates["G-M1 node recall vs human gold"]
    assert not g.passed and "DIAGNOSTIC-ONLY" in g.consequence


def test_contamination_gate_scales_with_the_headline_effect():
    """delta=0.03 is fine against a 0.10 effect and fatal against a 0.04 one."""
    ok = gate_report([], contamination_delta=0.03, headline_effect=0.10)[-1]
    bad = gate_report([], contamination_delta=0.03, headline_effect=0.04)[-1]
    assert ok.passed and not bad.passed
    assert "leave-one-family-out" in bad.consequence
