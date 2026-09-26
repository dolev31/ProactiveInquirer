"""21,001 runs' turn rows reached turns.parquet with a NULL run_id and appeared in no table.

WHAT HAPPENED. `pi compact` built every child row as

    rows["turns"].append({"run_id": run_id, **_migrate("turns", r)})

which reads as "take the record and add the run id" and is not that. A dict literal lets the
LATER key win, so a record that carries its own `run_id` overwrites the one the DIRECTORY
proves. 21,001 run directories on the shared tree are rehydrated ones -- their manifests say
`"rehydrated_from": "scores/parquet + cache"` -- and their turns.jsonl records are
parquet-shaped, first key `"run_id": null`. All 134,370 of their turn rows landed in
turns.parquet with run_id NULL.

WHY NOTHING RAISED. `pi_eval.schema.validate_rows` is strict about a key being PRESENT and
says nothing about its value, so a NULL primary key validates. Every reporting query
INNER JOINs turns to runs on run_id, so those rows are in the table and in no result:
`pi score` read the affected runs as n_turns=0, n_asks=0, n_evidence=0, evidence_coverage=0.0,
and the L0.6 equivalence join attributed 382,916 of its 656,301 inequalities to them. The
number of NULL-run_id rows in the shipped turns.parquet, 134,370, is exactly the number of
turn lines those directories hold on disk.

THE GATE. A count alone would be a quantity that cannot take the other value if it were
derived from the same parse the rows came from, so it is measured against the FILE: the
non-empty line count of turns.jsonl, read independently, versus the run ids present in the
committed turns table. `test_the_gate_can_fire` forces a known loss and asserts the count
moves, because an invariant check that cannot detect a change is indistinguishable from a
clean pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pi_eval.schema import SchemaViolation
from pi_run.compact import compact

# The exact shape written by the rehydration: a parquet row, run_id first and NULL.
_REHYDRATED_TURN = {
    "run_id": None,
    "turn_idx": 0,
    "action_kind": "ask",
    "question": "q?",
    "question_id": "qid",
    "rationale": "why",
    "target": "kb",
    "parent_uids": [],
    "retrieved_uids": ["u1"],
    "new_uids": ["u1"],
    "n_retrieved": 1,
    "n_new": 1,
    "response_text": "",
    "draft_sha": "sha",
    "draft_text": "draft",
    "subset_hash_before": "",
    "subset_hash_after": "h",
    "branch_of_run_id": None,
    "branch_turn_idx": None,
    "candidate_id": None,
    "candidate_rank": None,
    "forced": None,
    "policy_conf": None,
    "depth_pred": None,
    "bm25_hits": None,
    "spec_bits": None,
    "usage_tok_prompt": 1,
    "usage_tok_completion": 1,
    "usage_tok_reasoning": 0,
    "usage_tok_cached": 0,
    "usage_tok_total": 2,
    "usage_usd": 0.0,
    "usage_wall_ms": 1,
    "usage_n_calls": 1,
}


def _native_turn() -> dict:
    """What `pi_run.worker.turn_rows` writes: the same record with NO run_id key at all."""
    row = dict(_REHYDRATED_TURN)
    row.pop("run_id")
    return row


def _run_dir(
    root: Path, run_id: str, turns: list[dict], status: dict | None = None, **over
) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "suite_id": "strategyqa",
        "task_id": "t1",
        "arm_id": "self_inquire",
    }
    manifest.update(over)
    (d / "manifest.json").write_text(json.dumps(manifest))
    # The restore's status.json, verbatim: no n_turns, no n_asks, no n_evidence, no spent.
    (d / "status.json").write_text(
        json.dumps(status if status is not None else {"status": "ok", "usage": {}})
    )
    (d / "turns.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns))
    return d


def _runs(out: Path) -> list[dict]:
    return pq.read_table(out / "runs.parquet").to_pylist()


def _turns(out: Path) -> list[dict]:
    return pq.read_table(out / "turns.parquet").to_pylist()


# ------------------------------------------------------------------ the defect


def test_a_record_that_carries_a_null_run_id_is_keyed_to_its_directory(tmp_path):
    """The failing test. Before the fix this row lands with run_id=None and joins to nothing."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "a" * 32, [_REHYDRATED_TURN, dict(_REHYDRATED_TURN, turn_idx=1)])

    compact(runs, out)

    ids = [r["run_id"] for r in _turns(out)]
    assert ids == ["a" * 32, "a" * 32], ids
    assert None not in ids


def _schema_shaped(table: str, **over) -> dict:
    """A record with exactly the declared columns, built FROM the contract so the fixture
    cannot drift away from it, and with every owned key explicitly NULL."""
    import pyarrow as pa

    from pi_eval.schema import schema_for

    row: dict = {}
    for f in schema_for(table):
        if pa.types.is_string(f.type):
            row[f.name] = ""
        elif pa.types.is_boolean(f.type):
            row[f.name] = False
        elif pa.types.is_floating(f.type):
            row[f.name] = 0.0
        elif pa.types.is_list(f.type):
            row[f.name] = []
        else:
            row[f.name] = 0
    row["run_id"] = None
    row.update(over)
    return row


def test_no_child_row_may_reach_the_table_with_a_null_key(tmp_path):
    """The invariant the join depends on, stated over every run-side table."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    d = _run_dir(runs, "b" * 32, [_REHYDRATED_TURN])
    (d / "calls.jsonl").write_text(json.dumps(_schema_shaped("calls")) + "\n")
    (d / "evidence.jsonl").write_text(json.dumps(_schema_shaped("evidence")) + "\n")
    (d / "ledger.jsonl").write_text(json.dumps(_schema_shaped("ledger", row_idx=0)) + "\n")

    compact(runs, out)

    for name in ("turns", "calls", "evidence", "ledger"):
        ids = pq.read_table(out / f"{name}.parquet").column("run_id").to_pylist()
        assert ids and all(i == "b" * 32 for i in ids), (name, ids)


def test_a_record_naming_another_run_is_refused_not_relabelled(tmp_path):
    """A directory holding another run's rows is a corpus that mixes two runs. Overwriting the
    key quietly would make that undetectable, so it raises instead."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "c" * 32, [dict(_REHYDRATED_TURN, run_id="d" * 32)])

    with pytest.raises(SchemaViolation, match="run_id"):
        compact(runs, out)


# --------------------------------------------- n_turns: a missing key is not a measured zero


def test_a_status_without_n_turns_is_counted_from_the_records_not_defaulted_to_zero(tmp_path):
    """`status.get("n_turns", 0)` read all 21,001 restored runs as runs that took no turns,
    and that 0 agreed with the missing turn rows, so the loss looked like a property of the
    runs. `worker` writes n_turns = len(traj.turns) and one record per turn, so the count is
    exact, not an estimate."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(
        runs,
        "6" * 32,
        [_REHYDRATED_TURN, dict(_REHYDRATED_TURN, turn_idx=1, action_kind="stop")],
        status={
            "ok": True,
            "status": "ok",
            "error": "",
            "stop_reason": "budget",
            "wall_ms": 1,
            "usage": {},
        },
    )

    compact(runs, out)

    row = _runs(out)[0]
    assert row["n_turns"] == 2
    assert row["n_asks"] == 1  # only the first record is an ask
    # NOT repaired: those directories carry no evidence.jsonl and no calls.jsonl, so there is
    # nothing to count and a repaired value would be invented.
    assert row["n_evidence"] == 0
    assert row["n_calls"] == 0


def test_a_status_that_states_n_turns_still_wins(tmp_path):
    """A run that crashed between writing status and writing its turns is a real disagreement,
    not a rounding error, and the compaction must not paper over it."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(
        runs,
        "7" * 32,
        [_native_turn()],
        status={"status": "error", "usage": {}, "n_turns": 5, "n_asks": 4},
    )

    compact(runs, out)

    row = _runs(out)[0]
    assert (row["n_turns"], row["n_asks"]) == (5, 4)


# ------------------------------------------------------------------ the golden


def test_a_normal_run_directory_compacts_exactly_as_before(tmp_path):
    """GOLDEN. A record written by `pi_run.worker.turn_rows` carries no run_id key at all;
    the fix must leave that path byte-identical."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "e" * 32, [_native_turn(), dict(_native_turn(), turn_idx=1)])

    compact(runs, out)

    rows = _turns(out)
    assert [r["run_id"] for r in rows] == ["e" * 32, "e" * 32]
    assert [r["turn_idx"] for r in rows] == [0, 1]
    expected = dict(_REHYDRATED_TURN)
    expected["run_id"] = "e" * 32
    assert rows[0] == expected


def test_the_two_layouts_produce_the_same_table(tmp_path):
    """The rehydrated record differs from the native one in exactly one key, so after the fix
    the two must compact to the same row. This is the statement L0.6 could not make."""
    a, b = tmp_path / "a", tmp_path / "b"
    _run_dir(a / "runs", "f" * 32, [_native_turn()])
    _run_dir(b / "runs", "f" * 32, [_REHYDRATED_TURN])

    compact(a / "runs", a / "p")
    compact(b / "runs", b / "p")

    assert _turns(a / "p") == _turns(b / "p")


# ----------------------------------------- the merge cannot carry an unjoinable row forever


def test_a_prior_row_with_a_null_key_is_dropped_by_the_merge(tmp_path):
    """MEASURED ON THE LIVE STORE. The first re-compaction under the fix added all 134,370
    recovered rows AND kept all 134,370 NULL-keyed ones: the merge drops a prior row only when
    its run_id is in the set this pass saw, and no run is named `None`, so an unjoinable row
    survives every compaction there will ever be. The two sets are content-identical modulo
    run_id -- `except` in both directions returns 0 of 134,370 -- so the prior row carries
    nothing the fresh one does not."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "8" * 32, [_native_turn()])
    compact(runs, out)

    # Put the table back the way the defect left it: the same row, keyed to nothing.
    t = pq.read_table(out / "turns.parquet")
    nulled = t.set_column(
        t.schema.get_field_index("run_id"), "run_id", pa.array([None] * t.num_rows, pa.string())
    )
    pq.write_table(pa.concat_tables([nulled, t]), out / "turns.parquet")
    assert pq.read_table(out / "turns.parquet").num_rows == 2

    res = compact(runs, out)

    ids = pq.read_table(out / "turns.parquet").column("run_id").to_pylist()
    assert ids == ["8" * 32], ids
    assert res.as_dict()["prior_rows_dropped_with_no_run_id"] == {"turns": 1}


def test_a_prior_row_with_a_real_key_this_pass_did_not_see_is_still_kept(tmp_path):
    """The merge's whole reason for existing. Dropping only the UNJOINABLE rows must not turn
    it back into a replace: compacting a subset still keeps the runs it did not visit."""
    import pyarrow.parquet as pq

    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "parquet"
    _run_dir(a, "9" * 32, [_native_turn()])
    _run_dir(b, "c" * 32, [_native_turn()])
    compact(a, out)
    res = compact(b, out)

    ids = sorted(set(pq.read_table(out / "turns.parquet").column("run_id").to_pylist()))
    assert ids == ["9" * 32, "c" * 32]
    assert res.as_dict()["prior_rows_dropped_with_no_run_id"] == {}


# ------------------------------------------------------------------ the counted gate


def test_the_compactor_reports_the_loss_count_and_it_is_zero_when_clean(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "0" * 32, [_REHYDRATED_TURN])
    _run_dir(runs, "1" * 32, [_native_turn()])
    _run_dir(runs, "2" * 32, [])  # a no-ask arm: genuinely empty on disk, never counted

    res = compact(runs, out)

    assert res.as_dict()["n_runs_with_turns_on_disk_but_none_compacted"] == 0
    assert res.runs_with_turns_on_disk_but_none_compacted == ()


def test_the_gate_can_fire(tmp_path, monkeypatch):
    """NON-VACUITY. Force a known loss and assert the statistic moves. A check that cannot
    detect a change is indistinguishable from a clean pass."""
    import pi_run.compact as mod

    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "3" * 32, [_native_turn()])
    _run_dir(runs, "4" * 32, [])

    real = mod._read_jsonl
    monkeypatch.setattr(mod, "_read_jsonl", lambda p: [] if p.name == "turns.jsonl" else real(p))
    res = compact(runs, out)

    assert res.as_dict()["n_runs_with_turns_on_disk_but_none_compacted"] == 1
    assert res.runs_with_turns_on_disk_but_none_compacted == ("3" * 32,)


def test_the_cli_exits_non_zero_when_the_gate_fires(tmp_path):
    """The count is printed and a compaction that ends with a non-zero count exits non-zero."""
    runs, out = tmp_path / "runs", tmp_path / "parquet"
    _run_dir(runs, "5" * 32, [_native_turn()])

    ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "pi_run.cli",
            "compact",
            "--runs-root",
            str(runs),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["n_runs_with_turns_on_disk_but_none_compacted"] == 0

    # Now break it the only way the CLI can see: hide the rows from the reader. A SEPARATE out
    # dir, because the merge would otherwise carry the good pass's rows forward and the gate
    # would be right to stay quiet -- the loss is defined on the committed table, not the pass.
    broken = tmp_path / "broken.py"
    broken.write_text(
        "import sys, pi_run.compact as m\n"
        "_r = m._read_jsonl\n"
        "m._read_jsonl = lambda p: [] if p.name == 'turns.jsonl' else _r(p)\n"
        "from pi_run.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    bad = subprocess.run(
        [
            sys.executable,
            str(broken),
            "compact",
            "--runs-root",
            str(runs),
            "--out",
            str(tmp_path / "parquet2"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0, bad.stdout
    assert json.loads(bad.stdout)["n_runs_with_turns_on_disk_but_none_compacted"] == 1
    assert "turns.jsonl" in bad.stderr
