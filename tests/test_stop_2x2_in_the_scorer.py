"""The STOP 2x2, per run, in the SCORER -- and pinned equal to the dev gate's own version.

WHY THE SCORER NEEDS ITS OWN CELLS. `pinq_train.gate._stop_2x2` computes this 2x2 for a whole
grid at gate time and throws the per-run detail away. Nothing in `scores.parquet` carries it,
so no paper table, no arm contrast and no re-score can ever recover "how often did this policy
stop when it was already done" -- the question the stopping half of the thesis is about. The
gate's number is also computed under whatever `scorer_hash` the gate was pointed at and is not
itself a scored row, so it has no provenance in the sense CONTRIBUTING.md means.

WHY IT IS A SECOND IMPLEMENTATION AND NOT AN IMPORT. import-linter contract 4 forbids anything
(including `pi_eval`) from importing `pinq_train`, and contract 1 forbids `pinq_train` from
importing `pi_eval`. The two definitions therefore cannot share code, and two definitions of
"done" in two tiers is exactly the failure this file exists to prevent. So they are pinned
EQUAL, cell by cell, on a fixture built from the gate module's own docstring semantics.

A test that only exercised `pi_eval` would pass while the two drifted apart, which is why the
gate's function is called here directly. `_stop_2x2(con, runs, ladders)` never touches `con`,
so it is driven with None and no parquet at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pi_eval.metrics.stopping import DONE_AT, stop_2x2

# `run_dirs`, not `runs`: .gitignore line 10 is `runs/`, which is unanchored and would have
# silently excluded this fixture from the commit while every local run of the suite passed.
FIXTURE_RUN = Path(__file__).parent / "fixtures" / "run_dirs" / "922ccb7bdbaac0dae19c9c1fae7cbcc1"

# Every shape the gate's docstring distinguishes. `cov` is the coverage BEFORE the decision at
# t, so `len(cov) - 1` is n_asks; None is a point the scorer omitted because coverage is
# undefined there, which is NOT the same as 0.0.
CASES = (
    # kept asking after it was already done, then stopped by choice
    ("r_late", (0.0, 1.0, 1.0, 1.0), "policy_stop"),
    # stopped by choice while required evidence was still missing -- the cell that was empty
    ("r_early", (0.0, 0.4, 0.6), "policy_stop"),
    # the harness halted it: the final state carries no decision
    ("r_capped", (0.0, 0.5, 1.0), "budget"),
    ("r_maxturns", (0.0, 0.5), "max_turns"),
    # a task with no required gold evidence at one prefix: skipped, never filed as not-done
    ("r_gap", (0.0, None, 1.0), "policy_stop"),
    # the policy stopped on the empty state: one decision point, not zero
    ("r_immediate", (0.0,), "policy_stop"),
    # an episode whose ladder is missing entirely
    ("r_noladder", (None, None), "policy_stop"),
    # the empty stop_reason, which the gate files as forced
    ("r_blank", (0.0, 1.0), ""),
)


def _runs_and_ladders():
    """The gate reads its ladders out of `scores.parquet` and `_coverage_ladder` only ever
    creates an entry for a run that HAS a value-carrying row, so "no ladder" reaches
    `_stop_2x2` as a run id ABSENT from the mapping -- never as an empty dict. The fixture
    models the reachable shape, because the unreachable one has a different skip rule and
    pinning against it would pin a branch the pipeline cannot produce."""
    runs, ladders = [], {}
    for rid, cov, reason in CASES:
        runs.append({"run_id": rid, "n_asks": len(cov) - 1, "stop_reason": reason})
        lad = {k: float(c) for k, c in enumerate(cov) if c is not None}
        if lad:
            ladders[rid] = lad
    return runs, ladders


def _ladder_of(ladders, rid):
    return ladders.get(rid, {})


def _mine(runs, ladders):
    return {
        r["run_id"]: stop_2x2(
            _ladder_of(ladders, r["run_id"]), n_asks=r["n_asks"], stop_reason=r["stop_reason"]
        )
        for r in runs
    }


# ------------------------------------------------------------------ the two implementations


def test_the_scorer_and_the_gate_count_the_same_events():
    """Cell by cell, summed over the fixture. If this ever fails, one tier's "done" has moved
    and every stopping number in the paper is computed against two definitions."""
    gate = pytest.importorskip("pinq_train.gate")
    runs, ladders = _runs_and_ladders()

    theirs = gate._stop_2x2(None, runs, ladders)
    mine = _mine(runs, ladders)

    assert sum(c.n_done for c in mine.values()) == theirs["n_done"]
    assert sum(c.n_not_done for c in mine.values()) == theirs["n_not_done"]
    assert sum(c.n_stop_at_done for c in mine.values()) == theirs["n_done_stop"]
    assert sum(c.n_ask_at_not_done for c in mine.values()) == theirs["n_not_done_ask"]
    assert sum(c.n_forced_stops for c in mine.values()) == theirs["n_forced_stops"]
    assert sum(c.n_skipped_no_coverage for c in mine.values()) == theirs["n_skipped_no_coverage"]
    # the derived cells the gate reports and the scorer leaves to the table
    assert sum(c.n_done - c.n_stop_at_done for c in mine.values()) == theirs["n_done_ask"]
    assert (
        sum(c.n_not_done - c.n_ask_at_not_done for c in mine.values()) == theirs["n_not_done_stop"]
    )
    assert sum(c.n_scored for c in mine.values()) == theirs["n_states"]


def test_the_asks_after_done_mean_agrees_too():
    """The gate averages over RUNS, not over states, and the scorer's per-run column must
    reproduce that average exactly or the two tiers report different costs for the same
    behaviour.

    The runs the gate averages over are exactly the runs the scorer writes a row for: the
    gate drops a run whose ladder is absent, and the scorer emits no row when no decision
    point carried a coverage reading. Those are the same set, which is the property that
    makes a table over the scored column reproduce the gate's number.
    """
    gate = pytest.importorskip("pinq_train.gate")
    runs, ladders = _runs_and_ladders()
    mine = _mine(runs, ladders)
    counted = [
        c.asks_after_done for rid, c in mine.items() if mine[rid].n_scored or ladders.get(rid)
    ]
    assert {rid for rid in mine if mine[rid].n_scored} <= set(ladders)
    theirs = gate._stop_2x2(None, runs, ladders)
    assert theirs["n_runs"] == len(counted), "the two tiers must average over the same runs"
    assert sum(counted) / len(counted) == pytest.approx(theirs["mean_asks_after_done"])


def test_the_two_cells_are_both_non_empty_on_the_fixture():
    """A 2x2 with an empty column is not a measurement -- the exact defect that killed
    `stop_undershoot` as the done axis. The fixture must exercise both."""
    gate = pytest.importorskip("pinq_train.gate")
    runs, ladders = _runs_and_ladders()
    theirs = gate._stop_2x2(None, runs, ladders)
    assert theirs["n_done"] > 0 and theirs["n_not_done"] > 0


# ------------------------------------------------------------------------- the semantics


def test_a_forced_stop_is_excluded_and_counted_not_filed_as_an_ask():
    """Counting the capped state as an ASK credits the cap with the policy's judgement."""
    c = stop_2x2({0: 0.0, 1: 0.5, 2: 1.0}, n_asks=2, stop_reason="budget")
    assert c.n_forced_stops == 1
    assert c.n_scored == 2, "t=0 and t=1 remain; only the state at t=n_asks is excluded"
    assert c.n_done == 0 and c.n_not_done == 2


def test_asks_taken_after_coverage_is_complete_are_done_ask_states():
    c = stop_2x2({0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}, n_asks=3, stop_reason="policy_stop")
    assert (c.n_done, c.n_stop_at_done, c.asks_after_done) == (3, 1, 2)


def test_a_policy_stop_before_coverage_is_complete_is_a_not_done_stop():
    c = stop_2x2({0: 0.0, 1: 0.4, 2: 0.6}, n_asks=2, stop_reason="policy_stop")
    assert (c.n_not_done, c.n_ask_at_not_done) == (3, 2)
    assert c.n_done == 0


def test_an_undefined_coverage_point_is_skipped_and_never_filed_as_not_done():
    """ "the task carries no required gold evidence" and "the policy has none of it" are
    opposite statements, and a 0.0 would make them the same row."""
    c = stop_2x2({0: 0.0, 2: 1.0}, n_asks=2, stop_reason="policy_stop")
    assert c.n_skipped_no_coverage == 1
    assert c.n_not_done == 1 and c.n_done == 1


def test_done_is_the_same_threshold_the_export_uses():
    assert DONE_AT == 1.0 - 1e-12
    assert stop_2x2({0: DONE_AT}, n_asks=0, stop_reason="policy_stop").n_done == 1
    assert stop_2x2({0: 0.999999}, n_asks=0, stop_reason="policy_stop").n_not_done == 1


# ---------------------------------------------------------------- over a REAL run directory


def _fixture_run():
    status = json.loads((FIXTURE_RUN / "status.json").read_text())
    turns = [json.loads(ln) for ln in (FIXTURE_RUN / "turns.jsonl").read_text().splitlines() if ln]
    return status, turns


def test_the_field_spellings_the_scorer_reads_are_the_ones_the_harness_writes():
    """THE SILENT FAILURE THIS GUARDS. If `stop_reason` were spelled anything but
    'policy_stop' on disk, every stop in the repository would be counted as FORCED and both
    cells would read as a well-behaved policy that never chose to stop. Read off a real run
    directory rather than a fixture the test wrote itself."""
    status, turns = _fixture_run()
    assert status["stop_reason"] == "policy_stop"
    assert status["n_asks"] == 2 and len(turns) == 2
    assert status["status"] == "ok"


def test_a_real_run_that_stopped_with_nothing_retrieved_is_a_not_done_stop():
    """tau2_airline task 46, run 922ccb7b: a reference_agent trace that transfers to a human
    and stops. Its two turns retrieve NO evidence uids, so the coverage ladder the scorer
    builds from them is 0.0 at every prefix -- three decision points, all not-done, two asks
    and one stop taken while required coverage was still zero."""
    status, turns = _fixture_run()
    assert all(not t["retrieved_uids"] for t in turns), "the ladder below rests on this"
    ladder = {k: 0.0 for k in range(len(turns) + 1)}
    c = stop_2x2(ladder, n_asks=status["n_asks"], stop_reason=status["stop_reason"])
    assert (c.n_done, c.n_not_done) == (0, 3)
    assert (c.n_ask_at_not_done, c.n_stop_at_done) == (2, 0)
    assert c.n_forced_stops == 0 and c.n_skipped_no_coverage == 0


def test_the_scorer_emits_the_cells_for_that_real_run():
    """End to end through `score_run` on the real run record and its real turns, against a
    gold graph small enough to write down. The row must exist or the column is NaN."""
    from pi_eval.gold import GoldGraph, GoldNode
    from pi_eval.score import score_run

    status, turns = _fixture_run()
    node = GoldNode(
        gold_suite="tau2_airline",
        gold_task_key="46",
        gold_node_id="n0",
        gold_text="n0",
        gold_partition="required",
        gold_depth=0,
        gold_ev_uids=("tau2_airline:46:0-1",),
    )
    graph = GoldGraph(gold_suite="tau2_airline", gold_task_key="46", gold_nodes=(node,))
    rows = score_run(
        {
            "run_id": status["run_id"],
            "suite_id": "tau2_airline",
            "task_id": "46",
            "arm_id": status["arm_id"],
            "n_asks": status["n_asks"],
            "stop_reason": status["stop_reason"],
        },
        graph=graph,
        turns=turns,
        evidence=(),
        env_calls=(),
        ledger=(),
        records=(),
        answer=None,
    )
    by = {r["metric_name"]: r["value"] for r in rows}
    assert by["stop2x2_n_done"] == 0.0
    assert by["stop2x2_n_not_done"] == 3.0
    assert by["stop2x2_n_ask_at_not_done"] == 2.0
    assert by["stop2x2_n_stop_at_done"] == 0.0
    assert by["stop2x2_n_forced_stops"] == 0.0
    assert by["stop2x2_asks_after_done"] == 0.0


def test_no_stop_cells_are_written_when_the_ladder_is_undefined_everywhere():
    """A graph with no required gold evidence gives a ladder of NaN at every prefix. Six
    zeros there would read as a policy that made three correct decisions."""
    from pi_eval.gold import GoldGraph
    from pi_eval.score import score_run

    status, turns = _fixture_run()
    graph = GoldGraph(gold_suite="tau2_airline", gold_task_key="46", gold_nodes=())
    rows = score_run(
        {
            "run_id": status["run_id"],
            "suite_id": "tau2_airline",
            "task_id": "46",
            "arm_id": status["arm_id"],
            "n_asks": status["n_asks"],
            "stop_reason": status["stop_reason"],
        },
        graph=graph,
        turns=turns,
        evidence=(),
        env_calls=(),
        ledger=(),
        records=(),
        answer=None,
    )
    names = {r["metric_name"] for r in rows}
    assert not [n for n in names if n.startswith("stop2x2_")]
    assert not math.isnan(0.0), "sanity"
