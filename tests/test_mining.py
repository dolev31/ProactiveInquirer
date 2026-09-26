"""The trace-mining pipeline, tested where it is easiest to be wrong.

The load-bearing test is test_order_alone_cannot_mint_an_edge: two needs that are always
resolved in the same order but have NO dependency must not produce a prerequisite edge.
If that ever passes silently, every depth claim in the paper is fiction.
"""

import pytest

from pi_eval.mining.ablate import ablation_verdict, substitution_rate
from pi_eval.mining.aggregate import aggregate, generator_entropy, haldane_lift
from pi_eval.mining.canon import (
    Candidate,
    cluster,
    exact_entail,
    mutually_entails,
    numerals,
    theta_sweep,
)
from pi_eval.mining.edges import acyclic, counterfactual_edge_test, mechanical_edges, screen_order
from pi_eval.mining.pool import Cell, Trace, build_pool, cell_grid, pool_is_admissible
from pinq.types import Evidence, EvidenceUnit
from pinq.view import make_view

FAMS = ("anthropic", "openai", "qwen", "llama")
FORMS = ("chain", "breadth", "react", "plan_then_act")


def _view():
    return make_view(
        task_id="t",
        suite_id="synth",
        question="q",
        instructions="i",
        corpus_id="c",
        corpus_hash="0" * 16,
        word_cap=60,
    )


def _cand(text, trace, cell, fam, form="chain", uids=("u1",), turn=0):
    return Candidate(text, trace, cell, fam, form, tuple(uids), turn, "llm_elicited")


def _trace(tid, cell, turn_uids, success=True):
    flat = tuple(u for _, us in turn_uids for u in us)
    return Trace(
        tid, "synth", "t", cell, 0, success, 1.0 if success else 0.0, flat, tuple(turn_uids)
    )


# ------------------------------------------------------------------ pool


def test_cell_grid_is_the_full_factorial():
    assert len(cell_grid(FAMS, FORMS, ("bm25", "dense", "grep"))) == 4 * 4 * 3


def test_thin_pool_is_refused_with_a_reason():
    """A pool too thin to support the contamination controls must not yield gold at all,
    and must say WHY so a systematically-skipped stratum stays visible."""
    c = Cell("anthropic", "chain", "bm25")
    pool = build_pool([_trace("a", c, [(0, ("u1",))])], suite_id="synth", task_id="t")
    ok, why = pool_is_admissible(pool)
    assert not ok and "cells" in why

    pool2 = build_pool([], suite_id="synth", task_id="t")
    ok2, why2 = pool_is_admissible(pool2)
    assert not ok2 and "no successful traces" in why2


def test_admissible_pool_needs_two_families():
    a, b = Cell("anthropic", "chain", "bm25"), Cell("openai", "react", "bm25")
    pool = build_pool(
        [_trace("a", a, [(0, ("u1",))]), _trace("b", b, [(0, ("u1",))])],
        suite_id="synth",
        task_id="t",
    )
    assert pool_is_admissible(pool)[0]


# ------------------------------------------------------------------ canon


def test_numeral_guard_prevents_a_silent_merge():
    """'the fee is 3%' vs 'the fee is 30%' must never cluster: most entailment models say
    they entail, and merging them corrupts every recall number downstream."""
    a, b = "the fee is 3 percent", "the fee is 30 percent"
    assert numerals(a) != numerals(b)
    assert not mutually_entails(a, b, lambda x, y: 1.0, 0.85)


def test_clustering_merges_identical_and_separates_distinct():
    cands = [
        _cand("fee is waived", "t1", "c1", "anthropic"),
        _cand("Fee Is Waived", "t2", "c2", "openai"),
        _cand("limit is 5000", "t3", "c1", "anthropic"),
    ]
    cls = cluster(cands, exact_entail, 0.85)
    assert len(cls) == 2
    merged = [c for c in cls if len(c.members) == 2][0]
    assert merged.families == {"anthropic", "openai"}
    assert merged.trace_ids == {"t1", "t2"}


def test_node_id_is_stable_and_theta_sensitive():
    cl = cluster([_cand("fee is waived", "t1", "c1", "anthropic")], exact_entail)[0]
    a = cl.node_id("synth", "t", 0.85, "pin")
    b = cl.node_id("synth", "t", 0.85, "pin")
    c = cl.node_id("synth", "t", 0.90, "pin")
    assert a == b and a != c, "theta is part of node identity, so a sweep cannot alias"


def test_theta_sweep_reports_granularity_elasticity():
    cands = [_cand(f"fact {i}", f"t{i}", "c1", "anthropic") for i in range(4)]
    sweep = theta_sweep(cands, exact_entail, [0.75, 0.85, 0.95])
    assert set(sweep) == {0.75, 0.85, 0.95}
    assert all(v == 4 for v in sweep.values())


# ------------------------------------------------------------------ aggregate


def test_promotion_requires_two_families_and_two_cells():
    """One generator's habit must not be able to mint a need."""
    c1 = Cell("anthropic", "chain", "bm25")
    pool = build_pool(
        [_trace("t1", c1, [(0, ("u1",))]), _trace("t2", c1, [(0, ("u1",))])],
        suite_id="synth",
        task_id="t",
    )
    single = cluster(
        [_cand("x", "t1", c1.cell_id, "anthropic"), _cand("x", "t2", c1.cell_id, "anthropic")],
        exact_entail,
    )
    nodes = aggregate(single, pool, suite="synth", task="t", theta=0.85, nli_pin="p")
    assert not nodes[0].promoted
    assert "families" in nodes[0].reason or "cell_support" in nodes[0].reason


def test_promotion_succeeds_across_families_and_cells():
    a, b = Cell("anthropic", "chain", "bm25"), Cell("openai", "react", "bm25")
    pool = build_pool(
        [_trace("t1", a, [(0, ("u1",))]), _trace("t2", b, [(0, ("u1",))])],
        suite_id="synth",
        task_id="t",
    )
    cls = cluster(
        [_cand("x", "t1", a.cell_id, "anthropic"), _cand("x", "t2", b.cell_id, "openai")],
        exact_entail,
    )
    nodes = aggregate(cls, pool, suite="synth", task="t", theta=0.85, nli_pin="p")
    assert nodes[0].promoted, nodes[0].reason
    assert nodes[0].support == 2 and nodes[0].cell_support == 2


def test_generator_entropy_flags_single_cell_support():
    a, b = Cell("anthropic", "chain", "bm25"), Cell("openai", "react", "bm25")
    cells = [a.cell_id, b.cell_id]
    one = cluster(
        [_cand("x", "t1", a.cell_id, "anthropic"), _cand("x", "t2", a.cell_id, "anthropic")],
        exact_entail,
    )[0]
    both = cluster(
        [_cand("y", "t1", a.cell_id, "anthropic"), _cand("y", "t2", b.cell_id, "openai")],
        exact_entail,
    )[0]
    assert generator_entropy(one, cells) == 0.0
    assert generator_entropy(both, cells) == pytest.approx(1.0)


def test_haldane_lift_is_finite_on_a_perfect_split():
    """6/6 successes and 0/6 failures must not produce an infinity a bootstrap cannot use."""
    lift, p = haldane_lift(6, 6, 0, 6)
    assert lift > 1 and lift != float("inf")
    assert 0.0 <= p <= 1.0


# ------------------------------------------------------------------ ablate


def test_ablation_separates_necessary_from_inert():
    """Frequency yields a candidate; only this stage yields 'required'."""
    view = _view()
    u = [
        EvidenceUnit.make(corpus_id="c", doc_id=f"d{i}", span="0:1", title="t", text="x")
        for i in range(3)
    ]
    ref = Evidence.of(u)
    key = frozenset({u[0].uid})

    def score(v, ev, seed):  # only u[0] matters
        return 1.0 if u[0].uid in ev.uids else 0.0

    assert (
        ablation_verdict(node_id="n", view=view, reference=ref, ev_uids=key, score=score).verdict
        == "NECESSARY"
    )
    assert (
        ablation_verdict(
            node_id="n", view=view, reference=ref, ev_uids=frozenset({u[1].uid}), score=score
        ).verdict
        == "INERT"
    )


def test_ablation_with_no_evidence_is_untestable_not_inert():
    """The distinction matters: UNTESTABLE leaves a denominator, INERT drops the node."""
    r = ablation_verdict(
        node_id="n",
        view=_view(),
        reference=Evidence(),
        ev_uids=frozenset(),
        score=lambda v, e, s: 0.0,
    )
    assert r.verdict == "UNTESTABLE" and r.n_seeds == 0


def test_substitution_rate_detects_redundant_evidence():
    """Two units that each fully substitute for the other read ~0 individually under LOO;
    the rate is what stops that being read as 'neither mattered'."""
    view = _view()
    u = [
        EvidenceUnit.make(corpus_id="c", doc_id=f"d{i}", span="0:1", title="t", text="x")
        for i in range(2)
    ]
    ref = Evidence.of(u)

    def score(v, ev, seed):
        return 1.0 if ev.uids else 0.0

    rate = substitution_rate(
        view=view,
        reference=ref,
        group_uids=[frozenset({u[0].uid}), frozenset({u[1].uid})],
        score=score,
    )
    assert rate == pytest.approx(1.0)


# ------------------------------------------------------------------ edges: THE CRITICAL ONES


def test_order_alone_cannot_mint_an_edge():
    """THE load-bearing test.

    Two needs always resolved in the same order but with NO dependency. The observational
    screen passes them (gamma = 1.0). The intervention refuses them, because denying A still
    leaves B resolvable. If this ever returns an edge, depth is fiction.
    """
    cells = [Cell(f, "chain", "bm25") for f in FAMS]
    traces = [_trace(f"t{i}", cells[i % 4], [(0, ("a",)), (1, ("b",))]) for i in range(8)]
    node_ev = {"A": frozenset({"a"}), "B": frozenset({"b"})}

    stats = {(s.src, s.dst): s for s in screen_order(traces, node_ev)}
    ab = stats[("A", "B")]
    assert ab.gamma == 1.0 and ab.passed_screen, "order screen alone is fooled, as expected"

    # B is resolvable whether or not A's evidence is denied -> no dependency.
    def probe(denied, seed):
        return {"A", "B"} - {n for n, uids in node_ev.items() if uids & denied}

    assert counterfactual_edge_test(ab, node_ev, probe) is None


def test_intervention_confirms_a_real_prerequisite():
    """The same screen, but now denying A genuinely blocks B."""
    cells = [Cell(f, "chain", "bm25") for f in FAMS]
    traces = [_trace(f"t{i}", cells[i % 4], [(0, ("a",)), (1, ("b",))]) for i in range(8)]
    node_ev = {"A": frozenset({"a"}), "B": frozenset({"b"})}
    stat = {(s.src, s.dst): s for s in screen_order(traces, node_ev)}[("A", "B")]

    def probe(denied, seed):
        if node_ev["A"] & denied:
            return set()  # cannot reach B without A
        return {"A", "B"}

    edge = counterfactual_edge_test(stat, node_ev, probe)
    assert edge is not None
    assert edge.edge_kind == "prerequisite" and edge.verified == "intervention"
    assert edge.resolve_rate_control == 1.0 and edge.resolve_rate_denied == 0.0


def test_counter_evidence_vetoes_regardless_of_gamma():
    """If enough traces resolve dst WITHOUT src, src is demonstrably not a prerequisite,
    however lopsided the ordering happens to be."""
    cells = [Cell(f, "chain", "bm25") for f in FAMS]
    traces = [_trace(f"o{i}", cells[i % 4], [(0, ("a",)), (1, ("b",))]) for i in range(6)]
    traces += [_trace(f"s{i}", cells[i % 4], [(0, ("b",))]) for i in range(5)]
    node_ev = {"A": frozenset({"a"}), "B": frozenset({"b"})}
    stat = {(s.src, s.dst): s for s in screen_order(traces, node_ev)}[("A", "B")]
    assert not stat.passed_screen
    assert "without src" in stat.veto_reason
    assert counterfactual_edge_test(stat, node_ev, lambda d, s: {"A", "B"}) is None


def test_mechanical_edges_bypass_screening_with_full_confidence():
    """MuSiQue #N and tau2 doc->tool are enforced by the environment, not inferred by us."""
    es = mechanical_edges([("A", "B"), ("B", "C")])
    assert all(e.verified == "mechanical" and e.confidence == 1.0 for e in es)
    assert acyclic(es)


def test_cycle_is_detected():
    """A cyclic need graph has no well-defined depth, so it must be caught before use."""
    assert not acyclic(mechanical_edges([("A", "B"), ("B", "A")]))


# --------------------------------------------------------------------------- who generated it
#
# The admissibility gate demands >= 2 model families before a candidate need may be promoted,
# because one generator's habits would otherwise manufacture a phantom need. The label was
# derived from EVERY call in the run -- Inquirer, Drafter AND frozen Answerer -- so a single
# Inquirer paired with a differently-pinned Answerer already looked like two families.


def _run_dir(tmp_path, run_id, calls, asks=("what year?",)):
    import json

    d = tmp_path / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {"run_id": run_id, "suite_id": "musique", "task_id": "t0", "arm_id": "a", "seed": 0}
        )
    )
    (d / "status.json").write_text(json.dumps({"status": "ok"}))
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "turns.jsonl").write_text(
        "".join(
            json.dumps({"turn_idx": i, "action_kind": "ask", "question": q, "retrieved_uids": []})
            + "\n"
            for i, q in enumerate(asks)
        )
    )
    (d / "outcome.json").write_text(json.dumps({"answer": None}))
    return d


def test_the_model_family_is_the_inquirers_not_the_whole_runs(tmp_path):
    """`tier1_trained` runs exactly the confounding shape on purpose: a TRAINED Inquirer
    against the SAME frozen Drafter and Answerer as the prompted arm. Folding those in makes
    one generator look like two families and satisfies the diversity gate by itself."""
    from pi_eval.mining.from_runs import read_run

    d = _run_dir(
        tmp_path,
        "r1",
        [
            {"actor": "inquirer", "model": "openai/gpt-4o"},
            {"actor": "drafter", "model": "anthropic/claude-3"},
            {"actor": "answerer", "model": "anthropic/claude-3"},
        ],
    )
    run = read_run(d)
    assert run.model_family == "openai_gpt", (
        f"got {run.model_family!r}: the Drafter's and Answerer's models are in the label"
    )
    assert "mixed" not in run.model_family


def test_a_run_whose_inquirer_never_called_is_llm_free(tmp_path):
    """It generated no question, so it contributes no candidate -- and must not borrow another
    role's family to look like diversity."""
    from pi_eval.mining.from_runs import LLM_FREE_FAMILY, read_run

    d = _run_dir(
        tmp_path,
        "r2",
        [{"actor": "drafter", "model": "anthropic/claude-3"}],
        asks=(),
    )
    assert read_run(d).model_family == LLM_FREE_FAMILY


def test_two_different_inquirers_still_read_as_two_families(tmp_path):
    """The gate must still be satisfiable by real diversity, or the fix would make promotion
    impossible rather than honest."""
    from pi_eval.mining.from_runs import read_run

    a = read_run(_run_dir(tmp_path, "a", [{"actor": "inquirer", "model": "openai/gpt-4o"}]))
    b = read_run(_run_dir(tmp_path, "b", [{"actor": "inquirer", "model": "anthropic/claude-3"}]))
    assert a.model_family != b.model_family
    assert len({a.model_family, b.model_family}) == 2
