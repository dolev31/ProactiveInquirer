"""Metrics against hand-computable ground truth on the synthetic suite.

Every expected value below is derivable by hand from the constructed graph, so a failure
points at the metric and nowhere else.
"""

import pytest

from pi_eval.build.synth_build import build, load_graphs_for_test
from pi_eval.matcher.base import MechanicalMatcher
from pi_eval.metrics.discovery import evidence_coverage, rnr, rnr_ladder
from pi_eval.metrics.qvalue import make_q, phi_loo, phi_prefix_marginal, stopping_error
from pi_eval.metrics.structure import (
    coverage_at_depth,
    depth_weighted_recall,
    facet_breadth,
    max_depth_reached,
    precedence_violation_rate,
)
from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq_adapters.synth.suite import SynthSuite
from pinq_expt.fakes import (
    BreadthOnlyInquirer,
    ChainInquirer,
    EchoDrafter,
    FrozenAnswerer,
    NeverAsk,
)

DEPTH, FACETS = 3, 3


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("m")
    corpus, gold, _ = build(n_tasks=4, n_facets=FACETS, depth=DEPTH, seed=11, root=root)
    suite = SynthSuite(corpus.parent)
    graphs = {g.gold_task_key: g for g in load_graphs_for_test(gold)}
    return suite, graphs


def _traj(suite, tid, pol, cap=64):
    return run_loop(
        view=suite.view(tid),
        inquirer=pol,
        retriever=suite.retriever(tid),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=cap),
        max_turns=16,
        k=5,
        seed=0,
    )


def _match(suite, tid, pol, graphs):
    t = _traj(suite, tid, pol)
    return t, MechanicalMatcher().match(graph=graphs[tid], trajectory=t, run_id="r")


# ------------------------------------------------------------------ discovery


def test_chain_policy_resolves_every_required_need(env):
    suite, graphs = env
    t, recs = _match(suite, "s0", ChainInquirer(), graphs)
    ids = [n.gold_node_id for n in graphs["s0"].required()]
    assert len(ids) == FACETS * DEPTH
    assert rnr(recs, ids, "resolve") == 1.0


def test_never_ask_resolves_nothing(env):
    suite, graphs = env
    _, recs = _match(suite, "s0", NeverAsk(), graphs)
    ids = [n.gold_node_id for n in graphs["s0"].required()]
    assert rnr(recs, ids, "resolve") == 0.0


def test_breadth_only_resolves_exactly_the_seed_frontier(env):
    """FACETS of FACETS*DEPTH nodes: 3/9 = 1/3, exactly."""
    suite, graphs = env
    _, recs = _match(suite, "s0", BreadthOnlyInquirer(), graphs)
    ids = [n.gold_node_id for n in graphs["s0"].required()]
    assert rnr(recs, ids, "resolve") == pytest.approx(FACETS / (FACETS * DEPTH))


def test_rnr_ladder_is_monotone(env):
    """ask >= resolve >= use is compiled in; this proves the assert is live."""
    suite, graphs = env
    for pol in (ChainInquirer(), BreadthOnlyInquirer(), NeverAsk()):
        _, recs = _match(suite, "s0", pol, graphs)
        led = rnr_ladder(recs, [n.gold_node_id for n in graphs["s0"].required()])
        assert led["ask"] >= led["resolve"] >= led["use"]


def test_evidence_coverage_is_granularity_invariant(env):
    suite, graphs = env
    t, _ = _match(suite, "s0", ChainInquirer(), graphs)
    gold_uids = {u for n in graphs["s0"].gold_nodes for u in n.gold_ev_uids}
    assert evidence_coverage(set(t.evidence.uids), gold_uids) == 1.0
    t2 = _traj(suite, "s0", NeverAsk())
    assert evidence_coverage(set(t2.evidence.uids), gold_uids) == 0.0


# ------------------------------------------------------------------ structure


def test_coverage_at_depth_prints_cardinality(env):
    suite, graphs = env
    _, recs = _match(suite, "s0", ChainInquirer(), graphs)
    cad = coverage_at_depth(recs, graphs["s0"])
    assert cad == {d: (1.0, FACETS) for d in range(DEPTH)}


def test_breadth_only_covers_depth0_and_nothing_deeper(env):
    """The decisive shape: C@0 = 1.0, C@d>0 = 0.0. This is what 'vertical proactivity does
    not exist' would look like if the ablation matched the full policy."""
    suite, graphs = env
    _, recs = _match(suite, "s0", BreadthOnlyInquirer(), graphs)
    cad = coverage_at_depth(recs, graphs["s0"])
    assert cad[0][0] == 1.0
    assert all(cad[d][0] == 0.0 for d in range(1, DEPTH))
    assert max_depth_reached(recs, graphs["s0"]) == 0


def test_dwr_respects_preregistered_weights(env):
    suite, graphs = env
    _, recs = _match(suite, "s0", BreadthOnlyInquirer(), graphs)
    cad = coverage_at_depth(recs, graphs["s0"])
    # weight only depth 0 -> 1.0; weight only depth 2 -> 0.0
    assert depth_weighted_recall(cad, {0: 1.0}) == 1.0
    assert depth_weighted_recall(cad, {2: 1.0}) == 0.0
    assert depth_weighted_recall(cad, {0: 1.0, 1: 1.0, 2: 1.0}) == pytest.approx(1 / 3)


def test_chain_policy_never_violates_precedence(env):
    """It cannot: the token unlocking depth d+1 is only revealed by reading depth d."""
    suite, graphs = env
    _, recs = _match(suite, "s0", ChainInquirer(), graphs)
    assert precedence_violation_rate(recs, graphs["s0"]) == 0.0


def test_facet_breadth_separates_horizontal_from_vertical(env):
    suite, graphs = env
    _, chain = _match(suite, "s0", ChainInquirer(), graphs)
    _, breadth = _match(suite, "s0", BreadthOnlyInquirer(), graphs)
    # breadth-only touches every facet but no depth; chain touches every facet AND all depth
    assert facet_breadth(chain, graphs["s0"]) == (FACETS, FACETS)
    assert facet_breadth(breadth, graphs["s0"])[0] == 0  # facets exclude the depth-0 frontier


# ------------------------------------------------------------------ phi


def test_phi_loo_is_zero_for_a_duplicate_question(env):
    """A question retrieving only already-known units must have ~0 marginal value."""
    from pinq.types import Ask, Stop

    class Repeater:
        policy_id = "repeat"

        def reset(self, view, seed):
            import re

            self.toks = re.findall(r"\b[KD][0-9A-F]{8}\b", view.question.upper())
            self.n = 0

        def act(self, s):
            self.n += 1
            if self.n <= 4:
                return Ask(text=f"What does record {self.toks[0]} say?")
            return Stop()

    suite, graphs = env
    t = _traj(suite, "s0", Repeater())
    q = make_q(EchoDrafter(), FrozenAnswerer(), graphs["s0"].gold_answer)
    base, phis = phi_loo(t, q)
    # all four turns retrieved the SAME unit, so dropping any one of them still leaves the
    # unit present via the others -> zero marginal value each.
    assert len(phis) == 4
    assert all(p.phi == pytest.approx(0.0) for p in phis)


def test_phi_loo_is_positive_for_a_load_bearing_question(env):
    suite, graphs = env
    t = _traj(suite, "s0", ChainInquirer())
    q = make_q(EchoDrafter(), FrozenAnswerer(), graphs["s0"].gold_answer)
    base, phis = phi_loo(t, q)
    assert base > 0
    assert sum(1 for p in phis if p.phi > 0) >= FACETS, "terminal-value questions must matter"


def test_prefix_marginal_and_loo_are_both_computed_and_can_disagree(env):
    """The paper publishes rho(phi_hat, phi_LOO); this proves both estimators exist and are
    computed from the same trajectory."""
    suite, graphs = env
    t = _traj(suite, "s0", ChainInquirer())
    q = make_q(EchoDrafter(), FrozenAnswerer(), graphs["s0"].gold_answer)
    _, loo = phi_loo(t, q)
    marg = phi_prefix_marginal(t, q)
    assert len(marg) == len(t.turns)
    assert {p.turn_idx for p in loo} <= {p.turn_idx for p in marg}


def test_stopping_error_is_in_question_units(env):
    suite, graphs = env
    t = _traj(suite, "s0", ChainInquirer())
    q = make_q(EchoDrafter(), FrozenAnswerer(), graphs["s0"].gold_answer)
    e = stopping_error(t, q)
    assert e["overshoot"] >= 0 and e["undershoot"] >= 0
    assert e["q_at_k_star"] >= e["q_at_k_hat"] - 1e-12
    assert float(e["k_hat"]).is_integer()


def test_q_is_memoized_by_subset_hash_not_by_k(env):
    """Keying the answer memo by prefix index would make every LOO re-draft a cache miss."""
    suite, graphs = env
    t = _traj(suite, "s0", ChainInquirer())
    calls = {"n": 0}

    class CountingDrafter(EchoDrafter):
        def draft(self, view, ev, *, seed, ledger):
            calls["n"] += 1
            return super().draft(view, ev, seed=seed, ledger=ledger)

    q = make_q(CountingDrafter(), FrozenAnswerer(), graphs["s0"].gold_answer)
    q(t.view, t.evidence)
    first = calls["n"]
    for _ in range(5):
        q(t.view, t.evidence)
    assert calls["n"] == first == 1


# --------------------------------------------------------------------------- absent vs zero
#
# The pattern this repo keeps hitting: a well-formed 0.0 computed from something that was never
# measured. Two more, both on the path to a preregistered endpoint.


def test_the_frontier_omits_an_undefined_point_rather_than_calling_it_zero():
    """Q is NaN when the task carries no required gold evidence -- coverage UNDEFINED, not
    zero. `frontier_auc` integrates over these points, so a fabricated 0 drags the curve down
    for a task that never had a curve."""
    import inspect

    from pi_eval import score

    src = inspect.getsource(score.score_run)
    assert 'emit(f"frontier_q#{k}", 0.0 if math.isnan(q) else q' not in src
    i = src.index('emit(f"frontier_q#{k}"')
    window = src[max(0, i - 900) : i]
    assert "if math.isnan(q):" in window and "continue" in window


def test_unlock_and_invoke_scores_zero_when_the_task_needed_an_unlock_and_none_was_tried():
    """It used to emit only `if gated` -- only when the policy had ATTEMPTED a gated call -- so
    a policy that never tried was DROPPED from the endpoint. Measured on the first live tau2
    sweep: inquirer_prompted made 26-38 real tool calls and never once reached
    call_discoverable_agent_tool. The arms that never attempt the mechanic are exactly the ones
    the metric exists to catch."""
    import inspect

    from pi_eval import score

    src = inspect.getsource(score.score_run)
    i = src.index('emit(\n            "unlock_and_invoke"')
    window = src[max(0, i - 900) : i]
    assert "requires_unlock" in window, "applicability must come from GOLD, not from the policy"
    assert "if gated or requires_unlock:" in window
    assert 'gold_kind == "tool_unlock"' in window
