"""A candidate that STOPPED is a decision the policy made, and the exporter must see it.

THE DEFECT THESE TESTS WERE WRITTEN FOR. `pinq.loop` breaks on a `Stop` without stamping a
`Turn` -- correct for the audited core, which records what was retrieved. But the exporter
only ever built rows from `turns.jsonl`, so a candidate that stopped at the fork turn
produced no row at that turn, `state_key` returned None for its prefix rows, and the
candidate vanished from both `export_sft` and `export_pairs`. Measured: 1,988 of 23,583 ok
candidates (8.4%) stopped at the fork and were invisible, uncounted, in a dataset whose
whole purpose is to teach when to stop.

`status.json` already carries `stop_reason`, `handle_score` already reads it into
`resp.stop_reason`, and the last recorded turn carries `subset_hash_after` -- exactly the
state the STOP decision was made in. So `rows_from_run` can emit the STOP row without
touching the loop: `state_text` rendered `upto=len(turns)` and verified against that hash,
`action_json` the one STOP constant, `value` 0.0 (no gain, no redundancy, no retrieval paid),
and the real episode outcome fields, because a recorded STOP has a real `outcome.json`.

Every ASK row also gains `coverage_before` (required-evidence coverage BEFORE the decision,
`potential[turn_idx]`) and `done_before`. The gold-aware STOP target in `export_sft` needs
"was the task already done when this decision was made", and the episode-final
`evidence_coverage` cannot say that: it labels a state done exactly when the ASK is what
completed it.

The first test failed before the change with one row instead of two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.test_train_cli import _synth, _turns  # the row builder's own fixtures

from pi_run import cmd_train
from pi_run.cmd_train import StateMismatch, SuiteCache, rows_from_run
from pinq.actions import STOP_ACTION_JSON


def _write(
    runs: Path,
    run_id: str,
    *,
    task: str,
    turns: list,
    stop_reason: str,
    branch_of: str | None = None,
    branch_turn: int | None = None,
    after_hash: str | None = None,
) -> Path:
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "synth",
                "task_id": task,
                "arm_id": "inquirer_prompted",
                "split": "train",
                "template_id": None,
                "branch_of_run_id": branch_of,
                "branch_turn_idx": branch_turn,
                "budget_cap": 8,
                "max_turns": 16,
            }
        )
    )
    if turns and after_hash is not None:
        turns[-1]["subset_hash_after"] = after_hash
    (d / "turns.jsonl").write_text("\n".join(json.dumps(t) for t in turns))
    (d / "ledger.jsonl").write_text(
        json.dumps({"currency": "retrieval_calls", "cumulative": float(len(turns))})
    )
    (d / "status.json").write_text(
        json.dumps({"stop_reason": stop_reason, "wall_ms": 5, "usage": {"tok_total": 50}})
    )
    (d / "outcome.json").write_text(json.dumps({"answer": {"text": "some answer"}}))
    return d


def _after(units, n: int) -> str:
    from pinq.types import Evidence

    return Evidence.of(tuple(units[:n])).subset_hash


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_GOLD_ROOT", str(tmp_path / "data" / "gold"))
    suite, tid, units = _synth(tmp_path)
    return tmp_path, tid, units, cmd_train._train("reward").RewardWeights()


def test_a_candidate_that_stopped_at_the_fork_gets_a_stop_row(env):
    root, tid, units, weights = env
    runs = root / "runs"
    prefix = _turns(units[:1])  # one recorded ask, then the policy stopped
    d = _write(
        runs,
        "cand",
        task=tid,
        turns=prefix,
        stop_reason="policy_stop",
        branch_of="parent",
        branch_turn=1,
        after_hash=_after(units, 1),
    )
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)

    assert [r["turn_idx"] for r in rows] == [0, 1], "the STOP at turn 1 must be a row"
    ask, stop = rows
    assert ask["is_stop"] is False and stop["is_stop"] is True
    assert stop["action_json"] == STOP_ACTION_JSON
    assert stop["value"] == 0.0 and stop["phi_tilde"] == 0.0
    assert stop["branch_turn_idx"] == 1  # groups with its ASK siblings under state_key
    assert isinstance(stop["coverage_before"], float)
    assert "Retrieve record 0." in stop["state_text"]  # the prefix it decided on
    assert stop["stop_reason"] == "policy_stop"


def test_a_budget_stop_is_not_a_decision_and_gets_no_row(env):
    """The harness cut it off; the policy did not choose. Teaching STOP there teaches the cap."""
    root, tid, units, weights = env
    d = _write(
        root / "runs",
        "cand",
        task=tid,
        turns=_turns(units[:1]),
        stop_reason="budget",
        branch_of="parent",
        branch_turn=1,
        after_hash=_after(units, 1),
    )
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
    assert [r["turn_idx"] for r in rows] == [0]


def test_a_zero_turn_policy_stop_is_one_stop_row_at_turn_zero(env):
    """Stopped before asking anything: a real decision on the empty state, previously `[]`."""
    root, tid, units, weights = env
    d = _write(
        root / "runs",
        "cand",
        task=tid,
        turns=[],
        stop_reason="policy_stop",
        branch_of="parent",
        branch_turn=0,
    )
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
    assert len(rows) == 1 and rows[0]["is_stop"] is True and rows[0]["turn_idx"] == 0
    assert "(nothing retrieved yet)" in rows[0]["state_text"]


def test_every_ask_row_carries_the_before_fields(env):
    root, tid, units, weights = env
    d = _write(root / "runs", "run", task=tid, turns=_turns(units[:3]), stop_reason="budget")
    rows = rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
    assert len(rows) == 3
    for r in rows:
        assert r["is_stop"] is False
        assert isinstance(r["coverage_before"], float)
        assert isinstance(r["done_before"], bool)
        assert r["stop_reason"] == "budget"
    # coverage before turn 0 is the empty-evidence coverage, and it is monotone after
    assert rows[0]["coverage_before"] <= rows[1]["coverage_before"] <= rows[2]["coverage_before"]


def test_a_stop_row_whose_state_drifted_is_refused(env):
    """The same guard the ASK rows have: a STOP rendered from evidence that does not hash to
    what the run recorded would train on a prompt the policy never saw."""
    root, tid, units, weights = env
    d = _write(
        root / "runs",
        "cand",
        task=tid,
        turns=_turns(units[:1]),
        stop_reason="policy_stop",
        branch_of="parent",
        branch_turn=1,
        after_hash="0" * 64,
    )
    with pytest.raises(StateMismatch):
        rows_from_run(d, SuiteCache(root), graph_version="v1", weights=weights)
