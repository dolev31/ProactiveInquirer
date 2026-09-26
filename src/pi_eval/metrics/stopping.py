"""The STOP decision, counted over DECISION POINTS rather than over runs.

WHY THE UNIT IS A STATE. One episode offers the policy `n_asks + 1` decisions, t = 0..n_asks,
and a run-level summary collapses all of them into the last one: a run that asked six
questions after it was already done and then stopped scores a perfect 1.0, with the six
wasted questions invisible. The cells below are per RUN but counted per STATE, so a table can
sum them over any grouping and get the same 2x2 the dev gate computes.

DONE AXIS: the coverage the policy HOLDS when it decides at t, i.e. `frontier_q#t`, against
`DONE_AT`. NOT `stop_undershoot`: that metric is max(0, k* - k_hat) with k* the argmax of a
ladder that is monotone in k and k_hat its last index, so k* <= k_hat identically and the
metric is 0 on every run ever scored -- 1,395/1,395 on each gate parquet. Its "not done" cell
was empty by construction.

STOP AXIS: ASK at every t < n_asks, because the episode has a turn there. STOP at t = n_asks
ONLY under `stop_reason = 'policy_stop'`. Under 'budget' or 'max_turns' the HARNESS halted the
episode and the policy was never consulted at that state, so it carries no decision: it is
EXCLUDED and counted as `n_forced_stops`. Counting it as an ASK credits the cap with the
policy's judgement.

A DELIBERATE SECOND IMPLEMENTATION. `pinq_train.gate._stop_2x2` computes the same cells over a
whole grid; import-linter contract 4 forbids `pi_eval` from importing `pinq_train`, and
contract 1 forbids the reverse, so the two cannot share code. They are pinned equal on a
fixture by tests/test_stop_2x2_in_the_scorer.py, which is the only thing that keeps "done"
from meaning two things in two tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DONE_AT = 1.0 - 1e-12
"""`pi_run.cmd_train._done`'s threshold and `pinq_train.gate.DONE_AT`, restated because the
imports between those packages are forbidden. The ONE threshold the export's `done_before`
also uses, so "done" cannot mean two things."""


@dataclass(frozen=True, slots=True)
class Stop2x2:
    """The four cells, the excluded states, and the cost the cells do not show."""

    n_done: int
    n_stop_at_done: int
    n_not_done: int
    n_ask_at_not_done: int
    n_forced_stops: int
    asks_after_done: int
    n_skipped_no_coverage: int

    @property
    def n_scored(self) -> int:
        """Decision points that carried a coverage reading, i.e. the 2x2's own n."""
        return self.n_done + self.n_not_done


def stop_2x2(ladder: Mapping[int, float], *, n_asks: int, stop_reason: str) -> Stop2x2:
    """The 2x2 for ONE run.

    `ladder` maps a prefix index t to the required-evidence coverage held at t -- exactly the
    `frontier_q#t` rows the scorer emits, with the NaN points ABSENT rather than zeroed. A t
    with no entry is a state whose coverage is undefined; it is skipped and counted, never
    filed as "not done", because "no required gold evidence" and "the policy has none of it"
    are opposite statements.

    `n_asks` comes from the run record rather than from a count of turn rows, so a truncated
    turn table shows up as skipped states rather than as a shorter episode.
    """
    policy_stop = stop_reason == "policy_stop"
    n_done = n_not = stop_done = ask_not = 0
    n_forced = n_skipped = after_done = 0
    for t in range(max(int(n_asks), 0) + 1):
        action_is_stop = t == n_asks
        if action_is_stop and not policy_stop:
            n_forced += 1
            continue
        cov = ladder.get(t)
        if cov is None:
            n_skipped += 1
            continue
        if cov >= DONE_AT:
            n_done += 1
            stop_done += int(action_is_stop)
            after_done += int(not action_is_stop)
        else:
            n_not += 1
            ask_not += int(not action_is_stop)
    return Stop2x2(
        n_done=n_done,
        n_stop_at_done=stop_done,
        n_not_done=n_not,
        n_ask_at_not_done=ask_not,
        n_forced_stops=n_forced,
        asks_after_done=after_done,
        n_skipped_no_coverage=n_skipped,
    )
