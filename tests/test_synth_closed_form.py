"""The synthetic suite, where every metric has a hand-computable answer.

If these fail, the bug is in the metric or the loop — never in model sampling. That is the
entire reason this suite exists and is built before any real adapter.
"""

import pytest

from pi_eval.build.synth_build import build, load_graphs_for_test
from pi_eval.gold import compute_depths
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

DEPTH = 3
FACETS = 3


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    corpus, gold, chash = build(n_tasks=6, n_facets=FACETS, depth=DEPTH, seed=7, root=root)
    return SynthSuite(corpus.parent), load_graphs_for_test(gold), chash


def _run(suite, tid, inquirer, cap=32, max_turns=16):
    ledger = BudgetLedger(cap=cap)
    return run_loop(
        view=suite.view(tid),
        inquirer=inquirer,
        retriever=suite.retriever(tid),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=ledger,
        max_turns=max_turns,
        k=5,
        seed=0,
    ), ledger


# ---------------------------------------------------------------- the graph itself


def test_graph_shape_is_exactly_as_constructed(built):
    _, graphs, _ = built
    g = graphs[0]
    assert len(g.gold_nodes) == FACETS * DEPTH
    assert len(g.gold_edges) == FACETS * (DEPTH - 1)
    assert len(g.gold_seed_node_ids) == FACETS
    assert all(e.gold_edge_kind == "prerequisite" for e in g.gold_edges)


def test_depth_histogram_is_closed_form(built):
    """|V_d| must be printed alongside C@d, so it had better be right."""
    _, graphs, _ = built
    assert graphs[0].depth_histogram() == {d: FACETS for d in range(DEPTH)}


def test_depth_is_derived_not_annotated(built):
    """Recomputing depth from (nodes, edges, seeds) must reproduce the stored values."""
    _, graphs, _ = built
    g = graphs[0]
    recomputed = compute_depths(
        [n.gold_node_id for n in g.gold_nodes], list(g.gold_edges), list(g.gold_seed_node_ids)
    )
    assert {n.gold_node_id: n.gold_depth for n in g.gold_nodes} == recomputed


def test_depth_is_invariant_to_node_id_permutation(built):
    """Depth is a graph property; shuffling the node list must not move it."""
    _, graphs, _ = built
    g = graphs[0]
    ids = list(reversed([n.gold_node_id for n in g.gold_nodes]))
    a = compute_depths(ids, list(g.gold_edges), list(g.gold_seed_node_ids))
    b = compute_depths(
        [n.gold_node_id for n in g.gold_nodes], list(g.gold_edges), list(g.gold_seed_node_ids)
    )
    assert a == b


# ---------------------------------------------------------------- the loop


def test_chain_policy_reaches_full_depth(built):
    """The reference policy must traverse every chain: FACETS*DEPTH records retrieved."""
    suite, graphs, _ = built
    for tid in suite.task_ids():
        traj, _ = _run(suite, tid, ChainInquirer())
        gold_docs = {f"{tid[1:]}:{f}:{d}" for f in range(FACETS) for d in range(DEPTH)}
        assert gold_docs <= traj.evidence.doc_ids, tid


def test_never_ask_retrieves_nothing(built):
    suite, _, _ = built
    traj, ledger = _run(suite, "s0", NeverAsk())
    assert len(traj.evidence) == 0
    assert traj.stop_reason == "policy_stop"
    assert ledger.spent.get("retrieval_calls", 0) == 0


def test_depth1_policy_is_strictly_dominated(built):
    """The decisive ablation, made free: breadth-only cannot reach depth >= 1, because the
    tokens that unlock it are only revealed by reading depth 0."""
    suite, _, _ = built
    chain, _ = _run(suite, "s0", ChainInquirer())
    breadth, _ = _run(suite, "s0", BreadthOnlyInquirer())
    assert len(breadth.evidence) == FACETS
    assert len(chain.evidence) == FACETS * DEPTH
    assert breadth.evidence.uids < chain.evidence.uids


def test_budget_cap_stops_the_loop(built):
    suite, _, _ = built
    traj, ledger = _run(suite, "s0", ChainInquirer(), cap=4)
    assert traj.stop_reason == "budget"
    assert ledger.spent["retrieval_calls"] == 4


def test_user_channel_is_closed_and_charged(built):
    """target='user' must be refused and billed as a wasted turn, which is what makes
    'the Inquirer does not elicit user utterances' provable rather than asserted."""
    from pinq.types import Ask, Stop

    class UserAsker:
        policy_id = "user_asker"

        def reset(self, view, seed):
            self.n = 0

        def act(self, s):
            self.n += 1
            return Ask(text="what is your budget?", target="user") if self.n <= 3 else Stop()

    suite, _, _ = built
    traj, ledger = _run(suite, "s0", UserAsker())
    assert ledger.spent["rejected_user_asks"] == 3
    assert ledger.spent.get("retrieval_calls", 0) == 0
    assert len(traj.evidence) == 0


# ---------------------------------------------------------------- prefix ladder


def test_prefix_is_monotone_and_labelled_truncated(built):
    """prefix(k) must be a strict evidence subset and must NEVER claim stop_reason='budget':
    the gap between the truncated curve and the true budget-conditioned curve is the
    evidence of budget-adaptivity, and merging them destroys it."""
    suite, _, _ = built
    traj, _ = _run(suite, "s0", ChainInquirer())
    prev = frozenset()
    for k in range(len(traj.turns) + 1):
        p = traj.prefix(k)
        assert p.stop_reason == "truncated_prefix"
        assert prev <= p.evidence.uids
        prev = p.evidence.uids
    assert traj.prefix(len(traj.turns)).evidence.uids == traj.evidence.uids


def test_draft_is_pure_in_subset_hash(built):
    """The architectural constraint phi_LOO rests on: same evidence set -> same draft."""
    suite, _, _ = built
    traj, _ = _run(suite, "s0", ChainInquirer())
    d = EchoDrafter()
    v = suite.view("s0")
    led = BudgetLedger(cap=99)
    a = d.draft(v, traj.evidence, seed=0, ledger=led)
    b = d.draft(v, traj.evidence.without(frozenset()), seed=0, ledger=led)
    assert a.text == b.text and a.sha == b.sha


def test_answer_respects_the_word_cap_for_every_policy(built):
    """Length control BY CONSTRUCTION: no arm can win the judge on verbosity alone."""
    suite, _, _ = built
    cap = suite.view("s0").word_cap
    for pol in (ChainInquirer(), NeverAsk(), BreadthOnlyInquirer()):
        traj, _ = _run(suite, "s0", pol)
        assert traj.outcome.answer.n_words <= cap


def test_the_inquirer_actually_sees_a_draft():
    """The paper's central object is s_t = (x, D_t, E_t, H_t). D_t was ALWAYS EMPTY.

    `drafter.draft()` ran after the loop, so `s.draft` was None on every call to `act()`, the
    Inquirer prompt rendered "(no draft yet)" every turn, and 0 of 186 recorded turns carried
    a draft_sha. The Inquirer was interrogating a RETRIEVER, not a Drafter -- the two-agent
    mechanism the paper is about was not implemented.
    """
    import tempfile
    from pathlib import Path

    from pi_eval.build.synth_build import build
    from pinq.budget import BudgetLedger
    from pinq.loop import run_loop
    from pinq_adapters.synth.suite import SynthSuite
    from pinq_expt.fakes import ChainInquirer, EchoDrafter, FrozenAnswerer

    root = Path(tempfile.mkdtemp())
    corpus, _, _ = build(n_tasks=1, n_facets=2, depth=2, root=root)
    suite = SynthSuite(corpus.parent)

    seen: list[object] = []

    class Watching(ChainInquirer):
        def act(self, s):
            seen.append(s.draft)
            return super().act(s)

    traj = run_loop(
        view=suite.view("s0"),
        inquirer=Watching(),
        retriever=suite.retriever("s0"),
        drafter=EchoDrafter(),
        answerer=FrozenAnswerer(),
        ledger=BudgetLedger(cap=32),
        max_turns=8,
        k=5,
        seed=0,
    )

    # Turn 0 legitimately has no draft; there is no evidence yet. Every later turn must.
    assert len(seen) >= 3, "need several turns to test this"
    assert seen[0] is None, "turn 0 has nothing to draft from"
    assert all(d is not None for d in seen[1:]), (
        f"the Inquirer saw no draft on {sum(1 for d in seen[1:] if d is None)} later turns"
    )

    # ...and the draft must be RECORDED, or nothing downstream can audit what it saw.
    later = [t for t in traj.turns if t.turn_idx > 0]
    assert later and all(t.draft_sha for t in later), "draft_sha not stamped on the turns"


def test_the_no_draft_ablation_is_available_and_distinguishable():
    """`draft_every_turn=False` restores the old behaviour as a cheap ablation -- and a run
    made that way must be identifiable, not silently equivalent."""
    import tempfile
    from pathlib import Path

    from pi_eval.build.synth_build import build
    from pinq.budget import BudgetLedger
    from pinq.loop import run_loop
    from pinq_adapters.synth.suite import SynthSuite
    from pinq_expt.fakes import ChainInquirer, EchoDrafter, FrozenAnswerer

    root = Path(tempfile.mkdtemp())
    corpus, _, _ = build(n_tasks=1, n_facets=2, depth=2, root=root)
    suite = SynthSuite(corpus.parent)

    def go(flag):
        return run_loop(
            view=suite.view("s0"),
            inquirer=ChainInquirer(),
            retriever=suite.retriever("s0"),
            drafter=EchoDrafter(),
            answerer=FrozenAnswerer(),
            ledger=BudgetLedger(cap=32),
            max_turns=8,
            k=5,
            seed=0,
            draft_every_turn=flag,
        )

    on, off = go(True), go(False)
    assert any(t.draft_sha for t in on.turns)
    assert not any(t.draft_sha for t in off.turns)
