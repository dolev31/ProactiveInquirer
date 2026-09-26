"""Branching a rollout at turn t: same prefix, a different decision from there on.

Rung 2 needs several candidate continuations from ONE recorded state so their outcomes can
be compared as a preference pair. `pairs.jsonl` has been 0 lines because nothing ever
produced a second continuation: the sampling primitives in `pinq.sampling` have no
production caller and no driver re-enters the loop with a prefix.

THE PREFIX IS RE-EXECUTED, NOT REPLAYED INTO A SYNTHETIC STATE. Replaying recorded turns
into a hand-built State desynchronises the ledger from the trajectory -- the prefix's calls
would be in the trajectory's per-turn usage and absent from the ledger -- and `reconcile`
compares exactly those two sums, so every branch would die on ReconcileError. Re-executing
costs approximately nothing because the requests are byte-identical and therefore cache
hits, and it keeps every existing invariant (usage attribution, budget frontier, the
subset_hash chain) true by construction rather than by reconstruction.

So the ONLY difference between a branch and its parent is the seed the Inquirer is reset
with, and only from turn `branch_at` onward.
"""

from __future__ import annotations

from pinq.budget import BudgetLedger
from pinq.loop import run_loop
from pinq.types import Ask, Draft, EvidenceUnit, Stop, TaskView


def _unit(uid: str) -> EvidenceUnit:
    return EvidenceUnit(uid=uid, corpus_id="c", doc_id=uid, span=(0, 1), text=uid, title=uid)


class _SeedEchoInquirer:
    """Its question names the seed it was reset with, so a branch is visible in the record."""

    prompt_hash = "seed-echo-v1"

    def __init__(self, n_asks: int = 4) -> None:
        self._seed = 0
        self._n = n_asks
        self._i = 0

    def reset(self, view, seed):
        self._seed = seed
        # NOT reset: `self._i` is the turn counter. A branch continues the dialogue, it does
        # not restart it.

    def act(self, state):
        if self._i >= self._n:
            return Stop()
        self._i += 1
        return Ask(text=f"q{self._i}@seed{self._seed}", target="kb")


class _Retriever:
    def search(self, query: str, k: int):
        return (_unit(f"u:{query}"),)


class _Drafter:
    prompt_hash = "d-v1"

    def resolve(self, view, ask, ev, *, seed, ledger):
        return ("", ev)

    def draft(self, view, ev, *, seed, ledger):
        return Draft(text=f"d{len(ev.uids)}")


class _Answerer:
    prompt_hash = "a-v1"

    def answer(self, view, ev, draft, *, seed, ledger):
        return draft.text


VIEW = TaskView(
    suite_id="s",
    task_id="t",
    question="q",
    instructions="",
    corpus_id="c",
    corpus_hash="ch",
    word_cap=50,
)


def _run(**kw):
    return run_loop(
        view=VIEW,
        inquirer=_SeedEchoInquirer(),
        retriever=_Retriever(),
        drafter=_Drafter(),
        answerer=_Answerer(),
        ledger=BudgetLedger(cap=64),
        max_turns=4,
        k=5,
        seed=0,
        **kw,
    )


def _questions(traj):
    return [t.action.text for t in traj.turns]


def test_without_branching_every_turn_uses_the_run_seed() -> None:
    assert _questions(_run()) == [f"q{i}@seed0" for i in (1, 2, 3, 4)]


def test_the_prefix_is_identical_and_the_suffix_diverges() -> None:
    """The defining property. Turns before `branch_at` must match the parent EXACTLY.

    A branch whose prefix differs is a pair for a decision the parent never faced, and
    nothing downstream could tell.
    """
    parent = _run()
    branch = _run(branch_at=2, branch_seed=99)
    assert _questions(branch)[:2] == _questions(parent)[:2]
    assert _questions(branch)[2:] == ["q3@seed99", "q4@seed99"]


def test_branching_at_zero_changes_everything() -> None:
    assert _questions(_run(branch_at=0, branch_seed=99)) == [f"q{i}@seed99" for i in (1, 2, 3, 4)]


def test_the_subset_hash_chain_stays_intact() -> None:
    """Each turn's `subset_hash_before` must equal the previous turn's `_after`.

    `state_at` refuses to branch when this chain is broken, so a branch that broke it would
    be unusable as a parent in turn.
    """
    turns = _run(branch_at=2, branch_seed=99).turns
    for a, b in zip(turns, turns[1:]):
        assert b.subset_hash_before == a.subset_hash_after


def test_a_branch_seed_without_a_branch_point_is_refused() -> None:
    """Silently ignoring it would produce a run labelled as a branch that is not one."""
    import pytest

    with pytest.raises(ValueError):
        _run(branch_seed=99)


def test_branching_past_the_end_never_fires() -> None:
    """Not an error -- a policy may STOP before the branch point -- but it must not relabel.

    The caller checks the resulting trajectory; the loop's job is only to not misapply a seed.
    """
    assert _questions(_run(branch_at=99, branch_seed=99)) == [f"q{i}@seed0" for i in (1, 2, 3, 4)]
