"""On tau2 the fork never fired: `branch_at`/`branch_seed` reached the manifest and not the loop.

MEASURED, after fixing the sampling temperature and re-running 12 states x 4 candidates: all four
candidates of every state sent the IDENTICAL request (`request_sha dcc524774efe` for every branch
seed, at both turn 0 and turn 1). The seed in the payload was the run's, not the branch's. With a
seed set the provider is near-deterministic even at temperature 1.0 -- three of four responses at
one state were byte-identical -- so there was still nothing to rank.

`pinq.loop.run_loop` does the fork: at `i == branch_at` it calls `inquirer.reset(view,
branch_seed)`, and `LLMPolicy._complete` then sends `seed=self._seed + self._n_asks`. The tau2
driver (`make_driver_agent_class`) builds the loop itself, once per user message, and its call
passed `seed=self._seed` and neither branch argument. The fourth field on this path that
`run_unit` honours and the tau2 copy dropped, after the foreign prefix, the branch identity and
the sampling temperature.

THE FORK TURN IS GLOBAL AND THE LOOP IS PER ROLLOUT. `candidate_specs` validates `branch_turn_idx`
against the parent's `turns.jsonl`, whose `turn_idx` is renumbered sequentially across rollouts;
`run_loop`'s `i` restarts at 0 for every user message. So the driver maps one onto the other with
the rollout's base -- the same `sum(len(t.turns))` it already sets as `ledger.turn_base`:

    fork at or after this rollout's base   -> local index `branch_at - base` (the loop simply
                                              never reaches it if this rollout ends first, and
                                              the next rollout recomputes with a larger base)
    fork before this rollout's base        -> local index 0, because `build_parts()` rebuilds
                                              the Inquirer every rollout with the RUN seed, and a
                                              post-fork rollout must re-seed it or the candidate
                                              silently rejoins the parent after the fork turn
"""

from __future__ import annotations

import inspect

from pi_run.stages import tau2_runner
from pi_run.stages.tau2_runner import rollout_branch


def test_no_fork_means_no_branch_arguments() -> None:
    assert rollout_branch(base=0, branch_at=None, branch_seed=None) == (None, None)
    assert rollout_branch(base=3, branch_at=5, branch_seed=None) == (None, None)


def test_a_fork_inside_this_rollout_is_relative_to_its_base() -> None:
    assert rollout_branch(base=0, branch_at=5, branch_seed=11) == (5, 11)
    assert rollout_branch(base=4, branch_at=5, branch_seed=11) == (1, 11)


def test_a_fork_before_this_rollout_reseeds_at_its_first_turn() -> None:
    """`build_parts()` rebuilds the Inquirer with the run seed every rollout."""
    assert rollout_branch(base=8, branch_at=5, branch_seed=11) == (0, 11)


def test_a_fork_beyond_this_rollout_is_left_for_a_later_one() -> None:
    """Local index 9 in a rollout that stops at 4 is never reached; that is correct, not an
    error -- the next rollout's base will have grown past it and it lands there."""
    assert rollout_branch(base=0, branch_at=9, branch_seed=11) == (9, 11)


def test_the_driver_passes_the_fork_into_run_loop() -> None:
    src = inspect.getsource(tau2_runner.make_driver_agent_class)
    assert "rollout_branch(" in src, "the driver must map the global fork turn onto its rollout"
    assert "branch_at=" in src and "branch_seed=" in src, "run_loop must receive both"


def test_the_simulation_hands_the_fork_to_the_driver() -> None:
    """The driver is constructed in `_simulate`, which `run_tau2_unit` calls -- the first
    version of this test looked in `run_tau2_unit` itself and failed against correct code. The
    test encoded a wrong belief about WHERE the construction lives, not the code about WHETHER
    the fork is passed."""
    src = inspect.getsource(tau2_runner._simulate)
    assert "branch_at=spec.branch_turn_idx" in src
    assert "branch_seed=spec.branch_seed" in src
    assert "_simulate(" in inspect.getsource(tau2_runner.run_tau2_unit)
