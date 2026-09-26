"""Candidate branches must not collapse into one reassembly slot.

`run_sweep` collects results as `by_key[spec.key]` and returns
`[by_key[spec.key] for spec in specs]`. `UnitSpec.key` was
`(suite_id, task_id, arm_id, seed)` -- and `candidate_specs` gives every candidate the
PARENT's suite, task, arm AND seed, because branch identity lives entirely in
`branch_of_run_id` / `branch_turn_idx` / `branch_seed`.

So all `n_candidates x len(turns)` specs shared one slot. Measured consequences, all silent:

  * the returned list is `len(specs)` ALIASES of a single result, so `ok`/`failed` count
    copies of one status and a candidate that errored is invisible;
  * `pi train sample-candidates` reported ONE run_id where 8 branch directories had been
    written -- observed exactly that on the first live invocation;
  * `drain_uncollected` skips `if key in by_key`, so under `--spend-cap` the money spent by
    in-flight siblings is never added to the reported total. `sweep.py`'s own docstring calls
    an under-reporting cap "worse than no cap".

The run DIRECTORIES were always correct -- each branch has its own `semantic_hash` -- which is
why this never corrupted data and never showed up as a failure.
"""

from __future__ import annotations

from pi_run.worker import UnitSpec
from pinq.sampling import candidate_seed


def _spec(**over) -> UnitSpec:
    base = dict(
        suite_id="musique",
        corpus_dir="/tmp/c",
        task_id="t1",
        arm_id="inquirer_prompted",
        seed=7,
        runs_root="/tmp/r",
        cache_root="/tmp/cache",
    )
    base.update(over)
    return UnitSpec(**base)


def test_an_unbranched_spec_still_reassembles_by_its_cell() -> None:
    """The common path: two unbranched specs collide iff they name the same cell.

    This first asserted the literal 4-tuple, which the fix necessarily changes. That was a
    test of the shape, not of the behaviour -- and no production caller reads the key
    positionally: `run_sweep` uses it as an opaque dict key and `worker` writes `list(key)`
    into status.json, which nothing parses.
    """
    assert _spec().key == _spec().key
    assert _spec().key != _spec(seed=8).key
    assert _spec().key != _spec(arm_id="drafter_only").key
    assert _spec().key != _spec(task_id="t2").key


def test_two_candidates_at_one_turn_do_not_share_a_key() -> None:
    """The defect. Every candidate carries the parent's seed; only branch_seed separates them."""
    keys = {
        _spec(branch_of_run_id="p", branch_turn_idx=1, branch_seed=candidate_seed(7, i)).key
        for i in range(4)
    }
    assert len(keys) == 4, f"candidates collapsed into {len(keys)} slot(s)"


def test_the_same_candidate_at_two_turns_does_not_share_a_key() -> None:
    a = _spec(branch_of_run_id="p", branch_turn_idx=1, branch_seed=99).key
    b = _spec(branch_of_run_id="p", branch_turn_idx=3, branch_seed=99).key
    assert a != b


def test_branches_of_different_parents_do_not_share_a_key() -> None:
    a = _spec(branch_of_run_id="pA", branch_turn_idx=1, branch_seed=99).key
    b = _spec(branch_of_run_id="pB", branch_turn_idx=1, branch_seed=99).key
    assert a != b


def test_a_branch_never_collides_with_its_parent() -> None:
    """candidate_seed(s, 0) == s, so candidate 0 is the parent's own continuation."""
    assert candidate_seed(7, 0) == 7
    parent = _spec().key
    cand0 = _spec(branch_of_run_id="p", branch_turn_idx=0, branch_seed=candidate_seed(7, 0)).key
    assert parent != cand0


def test_the_sweep_returns_one_result_per_candidate(monkeypatch) -> None:
    """End to end through `run_sweep`: 8 specs must reassemble to 8 distinct results."""
    from pi_run import sweep as sw

    specs = [
        _spec(branch_of_run_id="p", branch_turn_idx=t, branch_seed=candidate_seed(7, i))
        for t in (1, 3)
        for i in range(4)
    ]
    seen = []

    def _fake(spec):
        seen.append(spec.key)
        return {"status": "ok", "run_id": f"r{spec.branch_turn_idx}_{spec.branch_seed}"}

    monkeypatch.setattr(sw, "run_unit", _fake)
    out = sw.run_sweep(specs, concurrency=1)
    assert len(out) == 8
    assert len({r["run_id"] for r in out}) == 8, "results aliased onto one slot"
