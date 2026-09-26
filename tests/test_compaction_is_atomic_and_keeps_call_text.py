"""Two defects that between them froze four tables for six days without one error message.

WHAT HAPPENED. `pi compact` on the shared `runs/` raised

    SchemaViolation: table 'evidence' row 251: unknown column(s) ['text']

and the compaction had ALREADY written `runs`, `turns` and `calls` before it got to
`evidence`. `sch.POPULATED` is walked in order and each table is written inside the loop, so a
violation on the fourth table leaves the first three fresh and the last four -- `evidence`,
`env_calls`, `ledger`, `native` -- exactly as they were. Measured on the live tree: runs /
turns / calls stamped 2026-09-15 11:11, evidence / env_calls / ledger / native stamped
2026-09-09 00:26. `pi score`, `pi train gate`, the contamination checks and `pi render` all
inner-join across that boundary, so for six days every one of them read a fresh `runs` against
a six-day-old `evidence` and `ledger` and reported numbers for a corpus that never existed.
Nothing errored, because each table on its own was internally valid.

THE COLUMN ITSELF IS LEGITIMATE CONTENT, NOT BLOAT. Measured over all 48,299 run directories
that carry an evidence.jsonl: 2,051 of them, 3,244 rows, carry `text`, and 3,244 of 3,244 have
a `doc_id` beginning `call:` -- a REQUEST-ADDRESSED unit, one that is in no corpus. The whole
reason `evidence_rows` does not persist text ("the corpus is content-addressed and frozen, so
a uid plus corpus_hash already identifies the bytes") is true of a corpus record and false of
a tool result: nothing else identifies those bytes, and this column is the only copy of them.
Total 1.54 MB, mean 476 chars, zero empty. Dropping it at compaction would silently delete the
only surviving copy of 3,244 tool results and make those 2,051 tau2 runs permanently
unexportable -- `render_state` rebuilds held evidence out of the corpus and drops a `call:`
uid it cannot find, which is exactly the `state_mismatch` that was measured on 8 of the first
14 fork runs.

So: the column is DECLARED (never silently lost), backfilled "" for every corpus-addressed
row the way `turns.draft_text` already is, and the write phase is made all-or-nothing so a
future unknown column cannot tear the tables apart again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pi_eval.schema import SchemaViolation
from pi_run.compact import compact

_CALL_TEXT = "sofia_hernandez_5364"


def _evidence_row(**over):
    row = {
        "uid": "u1",
        "corpus_id": "tau2_retail",
        "doc_id": "call:find_user_id_by_name_zip:70cb558a03b37148",
        "span": "0:20",
        "title": "",
        "score": 0.0,
        "n_chars": len(_CALL_TEXT),
        "first_turn_idx": 0,
    }
    row.update(over)
    return row


def _run_dir(root: Path, run_id: str, evidence: list[dict]) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "tau2_retail",
                "task_id": "t1",
                "arm_id": "inquirer_prompted",
            }
        )
    )
    (d / "status.json").write_text(json.dumps({"status": "ok", "usage": {}}))
    (d / "evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evidence))
    return d


def _col(out: Path, name: str, column: str) -> list:
    import pyarrow.parquet as pq

    return pq.read_table(out / f"{name}.parquet").column(column).to_pylist()


# --------------------------------------------------------------- the column that was refused


def test_a_request_addressed_units_text_survives_compaction(tmp_path):
    """3,244 rows on the live tree, every one of them a `call:` unit. A `call:<tool>:<hash>`
    doc_id is in no corpus, so uid + corpus_hash identifies nothing and this column is the only
    copy of the bytes. Before the fix the whole compaction died on it."""
    runs = tmp_path / "runs"
    _run_dir(runs, "legacy1", [_evidence_row(text=_CALL_TEXT)])

    res = compact(runs, tmp_path / "pq")

    assert res.counts["evidence"] == 1
    assert _col(tmp_path / "pq", "evidence", "text") == [_CALL_TEXT]


def test_a_corpus_addressed_row_carries_an_empty_text_not_a_null(tmp_path):
    """The structural half of the contract: every declared column present on every row, so no
    aggregate can silently skip the rows that predate the field. "" is the signal -- measured 0
    of 3,244 `call:` texts are empty, so an empty string cannot be mistaken for a lost one."""
    runs = tmp_path / "runs"
    _run_dir(
        runs,
        "corpus1",
        [_evidence_row(uid="u2", doc_id="doc42", span="3:9", n_chars=120)],
    )

    compact(runs, tmp_path / "pq")

    assert _col(tmp_path / "pq", "evidence", "text") == [""]


def test_a_previously_compacted_evidence_table_still_merges(tmp_path):
    """The merge re-validates every prior row against the CURRENT schema. A new column that is
    not backfilled on the read-back path turns an additive fix into a demand that the whole
    corpus be rebuilt -- which, on a `runs/` that is in practice the only backup, is the same
    failure wearing a different hat."""
    runs, out = tmp_path / "runs", tmp_path / "pq"
    _run_dir(runs, "older", [_evidence_row(uid="u_old")])
    compact(runs, out)

    _run_dir(runs, "newer", [_evidence_row(uid="u_new", text=_CALL_TEXT)])
    res = compact(runs, out)

    assert res.counts["evidence"] == 2
    assert sorted(_col(out, "evidence", "text")) == ["", _CALL_TEXT]


def test_a_genuinely_unknown_column_still_fails_loudly(tmp_path):
    """The allowlist is an allowlist. A field the writer invented and nobody declared must
    still stop the compaction: that is the guard, and a fix for one legacy column must not
    become a general amnesty."""
    runs = tmp_path / "runs"
    _run_dir(runs, "weird", [_evidence_row(embedding_blob="0.1,0.2")])

    with pytest.raises(SchemaViolation, match=r"embedding_blob"):
        compact(runs, tmp_path / "pq")


# ------------------------------------------------------------------------- the torn write


def test_a_schema_violation_leaves_every_table_exactly_as_it_was(tmp_path):
    """THE SIX-DAY DEFECT. `runs`, `turns` and `calls` are written before `evidence` is even
    validated, so a violation on `evidence` committed three tables and abandoned four. The
    result is not a failed compaction -- it is a parquet directory whose tables describe two
    different corpora six days apart, which every downstream inner join reads without
    complaint."""
    runs, out = tmp_path / "runs", tmp_path / "pq"
    _run_dir(runs, "good", [_evidence_row(uid="u_ok")])
    compact(runs, out)

    names = sorted(p.name for p in out.glob("*.parquet"))
    before = {n: (out / n).read_bytes() for n in names}

    _run_dir(runs, "bad", [_evidence_row(uid="u_bad", embedding_blob="0.1,0.2")])
    with pytest.raises(SchemaViolation):
        compact(runs, out)

    after = {n: (out / n).read_bytes() for n in sorted(p.name for p in out.glob("*.parquet"))}
    assert after.keys() == before.keys(), "a failed compaction must create no new table either"
    torn = sorted(n for n in before if before[n] != after[n])
    assert not torn, (
        f"{torn} were rewritten by a compaction that then raised. Every consumer inner-joins "
        "these tables; a directory where some are fresh and some are stale reports numbers for "
        "a corpus that never existed, and says nothing."
    )


def test_a_schema_violation_leaves_no_half_written_temp_files(tmp_path):
    """Whatever the all-or-nothing write uses to stage its output must not survive the
    failure: a `.parquet.tmp` left behind is a file a later glob or a later rename can pick
    up, which is the same tearing one indirection further out."""
    runs, out = tmp_path / "runs", tmp_path / "pq"
    _run_dir(runs, "bad", [_evidence_row(embedding_blob="0.1,0.2")])

    with pytest.raises(SchemaViolation):
        compact(runs, out)

    leftovers = sorted(p.name for p in out.iterdir() if not p.name.endswith(".parquet"))
    assert not leftovers, f"staging files survived a failed compaction: {leftovers}"


# ---------------------------------------------------------------------------- the exit code


def test_pi_compact_exits_non_zero_on_a_schema_violation(tmp_path):
    """`cmd_compact` ends `return 0` unconditionally, so the truthfulness of the exit code
    rests entirely on the SchemaViolation escaping to the `sys.exit(main())` entry point. This
    pins that: a supervisor that gates on rc must not be told a torn compaction succeeded.

    (The `rc=0` recorded in scratchpad/post/supervisor10.sh:90 was a bash artifact, not this
    process: `echo "[$(date)] rc=$?"` expands the command substitution first, and `date`
    resets `$?` to 0 before `$?` is read. The real rc was 1, and this test keeps it 1.)
    """
    runs = tmp_path / "runs"
    _run_dir(runs, "bad", [_evidence_row(embedding_blob="0.1,0.2")])

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pi_run.cli import main; sys.exit(main(sys.argv[1:]))",
            "compact",
            "--runs-root",
            str(runs),
            "--out",
            str(tmp_path / "pq"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout
    assert "embedding_blob" in proc.stderr


def test_pi_compact_exits_zero_on_a_legacy_text_column(tmp_path):
    """The other direction, or the test above passes for a corpus nobody can compact."""
    runs = tmp_path / "runs"
    _run_dir(runs, "legacy1", [_evidence_row(text=_CALL_TEXT)])

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pi_run.cli import main; sys.exit(main(sys.argv[1:]))",
            "compact",
            "--runs-root",
            str(runs),
            "--out",
            str(tmp_path / "pq"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
