"""Can the GATE's `P(STOP | done)` take a value other than 1.0? Forced, not argued.

THE CLAIM UNDER TEST. A reading of 1.0 on a cell that could not have come out otherwise is an
identity, not a measurement, and this repository has a name for that. The gate's stop 2x2 was
reported as reading exactly 1.0 on every verdict examined, which would make it one.

THE ANSWER IS NO, AND THE REASON IS STRUCTURAL. `_stop_2x2` walks EVERY decision point of a
rollout, t = 0..n_asks, and puts each into the done or not-done row by that prefix's coverage.
A done state whose action was an ASK therefore lands in `n_done` while contributing nothing to
`n_done_stop` -- that is `n_done_ask`, and it is the only thing that can pull the ratio below
1.0. It is reachable whenever a policy completes its required evidence and then asks again.

These tests force exactly that state and assert the cell moves, then pin the two boundaries. The
observational half is in `artifacts/answer_node_stop_20260919/RESULT.md`: over the 209 gate
verdicts on disk the checkpoint cell spans 0.0971 to 1.0000 and only 10 read exactly 1.0 -- all
ten with `n_done_ask == 0`, i.e. 1.0 because those checkpoints never asked at a done state, not
because the statistic was pinned.

WHY THE OFFLINE CELL IS A DIFFERENT INSTRUMENT, and not this one seen from another side:
`eval_offline.stop_confusion` reads the condition off an EXPORT ROW's `done_before`, and
`export_sft`'s labelling rule emits STOP at every done state and an ASK only at not-done states.
Measured on `data/rl/dev/sft.dev.jsonl`: 4,531 rows are (done, STOP) and 2,444 are (not-done,
ASK), and the other two cells are EMPTY BY CONSTRUCTION. So the offline table scores each state
against a target chosen by the same fact that buckets it, while the gate reads the action the
policy actually took on a real trajectory. Same threshold, same quantity, different populations.
"""

from __future__ import annotations

import math

from pinq_train.gate import DONE_AT, _stop_2x2


def _run(run_id: str, n_asks: int, stop_reason: str = "policy_stop") -> dict:
    return {"run_id": run_id, "n_asks": n_asks, "stop_reason": stop_reason}


def test_a_done_state_the_policy_did_not_stop_at_pulls_the_cell_off_one():
    """THE FORCING EXPERIMENT. One run that completes its evidence and then keeps asking.

    Coverage reaches 1.0 before the decision at t=2, so t=2 and t=3 are both done states. The
    policy ASKED at t=2 and stopped at t=3. If the cell were an identity it would read 1.0
    regardless; it must read 1/2.
    """
    runs = [_run("r1", n_asks=3)]
    ladders = {"r1": {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0}}
    out = _stop_2x2(None, runs, ladders)

    assert out["n_done"] == 2, "t=2 and t=3 both sit above the done threshold"
    assert out["n_done_ask"] == 1, "the ask at t=2 is the state that makes the cell movable"
    assert out["n_done_stop"] == 1
    assert out["p_stop_given_done"] == 0.5, "NOT 1.0: the cell moved, so it is not an identity"
    assert out["mean_asks_after_done"] == 1.0


def test_the_cell_reads_one_only_when_no_done_state_was_asked_at():
    """The other boundary, so 1.0 is shown to be a REACHABLE value rather than the only one.

    Same shape, except coverage completes exactly at the final decision -- the policy never had
    a done state to ask at. This is the configuration every one of the ten 1.0 verdicts on disk
    is in (`n_done_ask == 0`), and it is a fact about those runs, not about the statistic.
    """
    runs = [_run("r1", n_asks=3)]
    ladders = {"r1": {0: 0.0, 1: 0.3, 2: 0.6, 3: 1.0}}
    out = _stop_2x2(None, runs, ladders)

    assert out["n_done"] == 1 and out["n_done_ask"] == 0
    assert out["p_stop_given_done"] == 1.0
    assert out["mean_asks_after_done"] == 0.0


def test_the_cell_can_read_zero_which_bounds_it_from_the_other_side():
    """A policy that completes its evidence and is then cut off by the harness never stops
    voluntarily at all, so the done row is all asks. Pinned because a statistic whose only
    demonstrated values are 1.0 and 0.5 has not been shown to span its range."""
    runs = [_run("r1", n_asks=3, stop_reason="budget")]
    ladders = {"r1": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}}
    out = _stop_2x2(None, runs, ladders)

    assert out["n_forced_stops"] == 1, "the harness's halt is not the policy's decision"
    assert out["n_done"] == 3 and out["n_done_stop"] == 0
    assert out["p_stop_given_done"] == 0.0


def test_a_forced_stop_is_excluded_rather_than_counted_as_either_action():
    """`n_forced_stops` is the count the peer's report pairs with the 1.0 readings, so what it
    means has to be exact: a budget or max_turns halt is the HARNESS stopping the episode, the
    policy was never consulted at that state, and counting it as an ASK would credit the cap
    with the policy's judgement."""
    runs = [_run("r1", n_asks=2, stop_reason="max_turns")]
    ladders = {"r1": {0: 0.0, 1: 0.5, 2: 0.9}}
    out = _stop_2x2(None, runs, ladders)

    assert out["n_forced_stops"] == 1
    assert out["n_states"] == 2, "only t=0 and t=1 carried a decision"
    assert math.isnan(out["p_stop_given_done"]), "no done state at all: nan, never 1.0"


def test_the_done_threshold_is_the_exports_own_and_a_hair_below_it_is_not_done():
    """`DONE_AT` is restated in `gate.py` because contract 3 forbids importing the export's
    `_done`. Two restatements of one threshold is exactly the thing that drifts, so the boundary
    is pinned on both sides rather than trusted."""
    assert DONE_AT == 1.0 - 1e-12
    runs = [_run("r1", n_asks=1)]
    just_under = {"r1": {0: 0.0, 1: 1.0 - 1e-9}}
    assert math.isnan(_stop_2x2(None, runs, just_under)["p_stop_given_done"])
    just_over = {"r1": {0: 0.0, 1: 1.0}}
    assert _stop_2x2(None, runs, just_over)["p_stop_given_done"] == 1.0


def test_the_two_cells_move_independently():
    """The docstring's own warning made executable: a checkpoint can buy P(STOP|done) by
    stopping everywhere, which also costs it P(ASK|not done). A test that only ever moved them
    together could not tell a real improvement from that trade."""
    runs = [_run("r1", n_asks=2)]
    # Stops at t=2 while NOT done -- the bad trade, in one run.
    out = _stop_2x2(None, runs, {"r1": {0: 0.0, 1: 0.2, 2: 0.4}})
    assert out["n_not_done"] == 3 and out["n_not_done_stop"] == 1
    assert out["p_ask_given_not_done"] == 2 / 3
    assert math.isnan(out["p_stop_given_done"]), "it never reached a done state"
