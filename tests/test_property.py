"""Invariants that must hold for all inputs.

Each of these is cheap to state and expensive to discover violated in week three.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pinq.budget import HARD_CURRENCY, BudgetExceeded, BudgetLedger
from pinq.ids import evidence_uid, h, subset_hash
from pinq.types import CallTelemetry, Evidence, EvidenceUnit, Usage

pytestmark = pytest.mark.property


def _unit(i: int, doc: str | None = None) -> EvidenceUnit:
    return EvidenceUnit.make(
        corpus_id="c", doc_id=doc or f"d{i}", span=f"0:{i}", title=f"t{i}", text=f"x{i}"
    )


# ------------------------------------------------------------------ ids


def test_h_is_domain_separated():
    """h('ab','c') must not collide with h('a','bc'); naive concatenation would."""
    assert h("ab", "c") != h("a", "bc")


def test_evidence_uid_is_pure_location():
    """Identity must not depend on text: two readings of the same span are the same atom."""
    a = EvidenceUnit.make(corpus_id="c", doc_id="d", span="0:5", title="T", text="alpha")
    b = EvidenceUnit.make(corpus_id="c", doc_id="d", span="0:5", title="OTHER", text="beta")
    assert a.uid == b.uid == evidence_uid("c", "d", "0:5")


@given(st.lists(st.integers(0, 50), max_size=20))
def test_subset_hash_is_order_and_duplicate_insensitive(xs):
    uids = [evidence_uid("c", f"d{x}", "0:1") for x in xs]
    assert subset_hash(uids) == subset_hash(list(reversed(uids)))
    assert subset_hash(uids) == subset_hash(uids + uids)


# ------------------------------------------------------------------ Evidence


@given(st.lists(st.integers(0, 30), max_size=25))
@settings(max_examples=50)
def test_evidence_is_canonical_and_deduped(xs):
    ev = Evidence.of([_unit(x) for x in xs])
    uids = [u.uid for u in ev.units]
    assert uids == sorted(uids), "units must be canonically ordered"
    assert len(uids) == len(set(uids)), "units must be deduplicated"
    assert len(ev) == len(set(xs))


@given(st.lists(st.integers(0, 20), min_size=1, max_size=15))
@settings(max_examples=50)
def test_without_is_the_exact_complement(xs):
    ev = Evidence.of([_unit(x) for x in xs])
    drop = frozenset(list(ev.uids)[: max(1, len(ev) // 2)])
    kept = ev.without(drop)
    assert kept.uids == ev.uids - drop
    assert ev.only(drop).uids == drop
    # without() then re-adding must return to the same subset_hash
    restored = kept.with_units(ev.only(drop).units)
    assert restored.subset_hash == ev.subset_hash


@given(st.lists(st.integers(0, 20), max_size=15))
@settings(max_examples=50)
def test_subset_hash_depends_only_on_the_uid_set(xs):
    a = Evidence.of([_unit(x) for x in xs])
    b = Evidence.of([_unit(x) for x in reversed(xs)])
    assert a.subset_hash == b.subset_hash


def test_duplicate_unit_contributes_nothing():
    """The substrate of 'a duplicated EvidenceUnit yields phi ~ 0': adding a unit already
    present must not change the subset the drafter sees."""
    ev = Evidence.of([_unit(1), _unit(2)])
    assert ev.with_units([_unit(1)]).subset_hash == ev.subset_hash


# ------------------------------------------------------------------ Usage


@given(
    st.integers(0, 10_000), st.integers(0, 10_000), st.integers(0, 10_000), st.integers(0, 10_000)
)
def test_usage_addition_is_associative_with_identity(a, b, c, d):
    x, y, z = Usage(a, b), Usage(c, d), Usage(b, c)
    assert (x + y) + z == x + (y + z)
    assert x + Usage() == x
    assert (x + y).tok_total == x.tok_total + y.tok_total


# ------------------------------------------------------------------ BudgetLedger


def test_hard_cap_raises_before_dispatch():
    led = BudgetLedger(cap=2)
    led.charge_retrieval()
    led.charge_retrieval()
    with pytest.raises(BudgetExceeded):
        led.charge_retrieval()
    assert led.spent[HARD_CURRENCY] == 2, "a refused charge must not be recorded"


def test_unique_docs_are_metered_not_calls():
    """Repeating a query is free at the provider but still yields evidence; metering unique
    docs is what stops a policy free-riding on the cache."""
    led = BudgetLedger(cap=10)
    assert led.note_docs(["a", "b"]) == 2
    assert led.note_docs(["a", "b"]) == 0
    assert led.note_docs(["c"]) == 1
    assert led.unique_docs == 3


def test_ledger_reconciles_with_turn_usage():
    """sum(turn.usage) == ledger.spent, or an unmetered nested retrieval is hiding."""
    led = BudgetLedger(cap=5)
    total = Usage()
    for i in range(3):
        call = CallTelemetry(
            call_id=f"c{i}",
            actor="drafter",
            model="m",
            provider="p",
            request_sha="r",
            response_sha="s",
            tok_prompt=100,
            tok_completion=10,
            usd=0.01,
        )
        led.record_call(call)
        total = total + Usage.from_call(call)
    assert led.reconcile(total)
    assert not led.reconcile(total + Usage(tok_prompt=1, n_calls=1))


# ------------------------------------------------------------------ per-turn attribution


def test_turn_usage_is_real_not_a_zero_stamp():
    """Regression: run_loop used to stamp Usage() on every Turn.

    That made per-turn cost attribution impossible, left the budget frontier without its
    cumulative-spend x-axis, and — worst — made ledger.reconcile() tautological, because it
    was comparing the ledger against a sum of zeros and could never fail. This test fails
    against the old behaviour.
    """
    import tempfile
    from pathlib import Path

    from pi_eval.build.synth_build import build
    from pinq.loop import run_loop
    from pinq_adapters.synth.suite import SynthSuite
    from pinq_expt.fakes import ChainInquirer, EchoDrafter, FrozenAnswerer

    root = Path(tempfile.mkdtemp())
    corpus, _, _ = build(n_tasks=1, n_facets=2, depth=2, root=root)
    suite = SynthSuite(corpus.parent)

    class BillingDrafter(EchoDrafter):
        """Charges a call per resolve, like any real LLM-backed drafter."""

        def resolve(self, view, ask, ev, *, seed, ledger):
            ledger.record_call(
                CallTelemetry(
                    call_id=f"c{ledger.n_calls}",
                    actor="drafter",
                    model="m",
                    provider="p",
                    request_sha="r",
                    response_sha="s",
                    tok_prompt=100,
                    tok_completion=10,
                )
            )
            return ("", ev)

    led = BudgetLedger(cap=32)
    traj = run_loop(
        view=suite.view("s0"),
        inquirer=ChainInquirer(),
        retriever=suite.retriever("s0"),
        drafter=BillingDrafter(),
        answerer=FrozenAnswerer(),
        ledger=led,
        max_turns=8,
        k=5,
        seed=0,
    )

    per_turn = sum(t.usage.tok_total for t in traj.turns)
    assert per_turn > 0, "turns must carry real usage"
    # every charged call is attributed to exactly one turn, so the sum IS the ledger
    assert per_turn == led.usage.tok_total
    assert sum(t.usage.n_calls for t in traj.turns) == led.n_calls


def test_usage_since_partitions_the_ledger():
    """Windows must tile the call list exactly: no call counted twice, none dropped."""
    led = BudgetLedger(cap=99)
    marks = []
    for i in range(6):
        marks.append(led.n_calls)
        led.record_call(
            CallTelemetry(
                call_id=f"c{i}",
                actor="drafter",
                model="m",
                provider="p",
                request_sha="r",
                response_sha="s",
                tok_prompt=10 * (i + 1),
                tok_completion=1,
            )
        )
    windows = [led.usage_since(m).tok_total - led.usage_since(m + 1).tok_total for m in marks]
    assert sum(windows) == led.usage.tok_total
    assert windows == [11, 21, 31, 41, 51, 61]
