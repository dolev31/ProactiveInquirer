"""The auditor behind artifacts/compaction_loss_20260918: the predicate is on DISK.

The one thing this must not do is count a no-ask arm as damage. `drafter_only` writes an empty
turns.jsonl by construction -- 780 of 780 runs on tau2, 836 of 836 on musique, 733 of 733 on
wiki2 -- and every one of them carries `runs.n_turns = 0`. A check that flagged those would be
wrong on 3,480 runs on day one and would be switched off by the end of the week.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "compaction_loss"))

from audit_turn_rows import audit, main, n_turn_lines  # noqa: E402

from pi_run.compact import compact  # noqa: E402


def _run_dir(root: Path, run_id: str, *, arm: str, split: str, turns: list[dict]) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "tau2",
                "task_id": "t1",
                "arm_id": arm,
                "split": split,
            }
        )
    )
    (d / "status.json").write_text(json.dumps({"status": "ok", "usage": {}}))
    (d / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns))
    return d


_TURN = {
    "turn_idx": 0,
    "action_kind": "ask",
    "question": "q",
    "question_id": "qid",
    "rationale": "",
    "target": "kb",
    "parent_uids": [],
    "retrieved_uids": [],
    "new_uids": [],
    "n_retrieved": 0,
    "n_new": 0,
    "response_text": "",
    "draft_sha": "",
    "draft_text": "",
    "subset_hash_before": "",
    "subset_hash_after": "",
    "branch_of_run_id": None,
    "branch_turn_idx": None,
    "candidate_id": None,
    "candidate_rank": None,
    "forced": None,
    "policy_conf": None,
    "depth_pred": None,
    "bm25_hits": None,
    "spec_bits": None,
    "usage_tok_prompt": 0,
    "usage_tok_completion": 0,
    "usage_tok_reasoning": 0,
    "usage_tok_cached": 0,
    "usage_tok_total": 0,
    "usage_usd": 0.0,
    "usage_wall_ms": 0,
    "usage_n_calls": 0,
}


def test_n_turn_lines_reads_the_file_not_the_parse(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    assert n_turn_lines(d) == 0  # absent
    (d / "turns.jsonl").write_text("")
    assert n_turn_lines(d) == 0  # empty
    (d / "turns.jsonl").write_text("{}\n\n{}\n")
    assert n_turn_lines(d) == 2  # blank lines are not records


def test_a_no_ask_arm_is_empty_on_disk_and_never_counted_as_lost(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "p"
    _run_dir(runs, "a" * 32, arm="drafter_only", split="test", turns=[])
    _run_dir(runs, "b" * 32, arm="inquirer_prompted", split="test", turns=[_TURN])

    compact(runs, out)
    res = audit(runs, out)

    assert res["n_lost"] == 0
    assert res["n_empty_on_disk"] == 1
    assert res["empty_by_cell"] == {"tau2/drafter_only/test": 1}
    assert res["n_runs_with_turn_rows"] == 1


def test_a_run_with_lines_on_disk_and_no_rows_is_lost_and_grouped(tmp_path, monkeypatch):
    import pi_run.compact as mod

    runs, out = tmp_path / "runs", tmp_path / "p"
    _run_dir(runs, "c" * 32, arm="inquirer_prompted", split="test", turns=[_TURN])
    _run_dir(runs, "d" * 32, arm="drafter_only", split="test", turns=[])

    real = mod._read_jsonl
    monkeypatch.setattr(mod, "_read_jsonl", lambda p: [] if p.name == "turns.jsonl" else real(p))
    compact(runs, out)
    res = audit(runs, out)

    assert res["n_lost"] == 1
    assert res["n_turn_lines_lost"] == 1
    assert res["lost_by_cell"] == {"tau2/inquirer_prompted/test": 1}
    assert res["lost_run_ids"] == ["c" * 32]
    assert res["n_empty_on_disk"] == 1  # the no-ask arm, still not damage


def test_the_auditor_exits_non_zero_only_when_something_is_lost(tmp_path, monkeypatch):
    import pi_run.compact as mod

    runs, out = tmp_path / "runs", tmp_path / "p"
    _run_dir(runs, "e" * 32, arm="inquirer_prompted", split="test", turns=[_TURN])
    compact(runs, out)
    assert main([str(runs), str(out)]) == 0

    real = mod._read_jsonl
    monkeypatch.setattr(mod, "_read_jsonl", lambda p: [] if p.name == "turns.jsonl" else real(p))
    compact(runs, tmp_path / "p2")
    assert main([str(runs), str(tmp_path / "p2")]) == 1


def test_a_run_that_finished_during_the_scan_is_not_counted_as_lost(tmp_path, monkeypatch):
    """On a live tree a run that completes while the walker is elsewhere in the sorted order
    has lines on disk and no rows in the table, and that is not loss. Measured on the 22:36:46
    pass: four tau2_retail runs written at 22:37:50, 22:41:00, 22:43:54 and 22:44:00.

    The split is on the FILE's mtime against the compaction's own start, so it cannot swallow
    a real loss -- the second half of this test is the same tree with an older file."""
    import os

    import pi_run.compact as mod

    runs, out = tmp_path / "runs", tmp_path / "p"
    _run_dir(runs, "1" * 32, arm="inquirer_prompted", split="test", turns=[_TURN])

    started = 1_000_000.0
    real = mod._read_jsonl
    monkeypatch.setattr(mod, "_read_jsonl", lambda p: [] if p.name == "turns.jsonl" else real(p))
    compact(runs, out)

    # Written DURING the scan: arrived after the snapshot, not lost.
    os.utime(runs / ("1" * 32) / "turns.jsonl", (started + 60, started + 60))
    res = audit(runs, out, since=started)
    assert (res["n_lost"], res["n_arrived_after_snapshot"]) == (0, 1)
    assert res["arrived_after_snapshot_by_cell"] == {"tau2/inquirer_prompted/test": 1}
    assert main([str(runs), str(out), "--since", str(started)]) == 0

    # The same tree, the same table, one older file: still lost, and still exits non-zero.
    os.utime(runs / ("1" * 32) / "turns.jsonl", (started - 60, started - 60))
    res = audit(runs, out, since=started)
    assert (res["n_lost"], res["n_arrived_after_snapshot"]) == (1, 0)
    assert main([str(runs), str(out), "--since", str(started)]) == 1


def test_null_run_ids_are_counted_separately(tmp_path):
    """The shipped turns.parquet held 134,370 rows with a NULL run_id. They are in the table
    and in no join, so the auditor reports them as their own number."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    runs, out = tmp_path / "runs", tmp_path / "p"
    _run_dir(runs, "f" * 32, arm="inquirer_prompted", split="test", turns=[_TURN])
    compact(runs, out)

    t = pq.read_table(out / "turns.parquet")
    nulled = t.set_column(
        t.schema.get_field_index("run_id"), "run_id", pa.array([None] * t.num_rows, pa.string())
    )
    pq.write_table(nulled, out / "turns.parquet")

    res = audit(runs, out)
    assert res["n_turn_rows_with_null_run_id"] == 1
    assert res["n_lost"] == 1
    assert main([str(runs), str(out)]) == 1
