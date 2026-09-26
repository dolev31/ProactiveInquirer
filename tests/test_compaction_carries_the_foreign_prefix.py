"""A fork's prefix must survive `pi compact`, or the paired-fork report is unbuildable from it.

THE RULE `test_compaction_keeps_the_fork_key` ALREADY ENCODES, applied to the other kind of
fork. That file is about `branch_turn_idx`: a field `state_key` depends on, dropped by the
compaction, discovered only when `runs/` was destroyed and `scores/parquet` turned out to be
the backup. These two are the same shape of mistake for a FOREIGN-trace fork:

  * `n_prefix_user_turns` is the subtrahend in `fork_report.follow_ups`
    (`n_user_turns - n_prefix_user_turns`). `score_run` reads the parquet and never the run
    directory, so a column the compaction does not carry is unreachable however faithfully
    status.json recorded it -- and `environment.prefix_user_turns` defaults a missing value to
    0.0, which charges every forked run for the prefix's user turns and says nothing.
  * `foreign_trace_sha` is the ONLY way a table reader can tell a continuation from a full run.
    `report.ELIGIBLE` refuses forks on exactly this column, because a fork inherits a foreign
    prefix and must never pool with our own arms under one task_id.

NULL AND NOT 0, for the counts. A fork that genuinely had no prefix user turns and a suite that
never records the field are different facts, and 0 is the BEST possible score on the engagement
endpoint -- the same argument `n_user_turns` already carries one line above it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_eval import schema as sch
from pi_run.compact import _run_row

FIXTURES = Path(__file__).parent / "fixtures" / "run_manifests"


def _manifest(**over):
    m = {
        "run_id": "r1",
        "suite_id": "tau2_retail",
        "task_id": "55",
        "arm_id": "inquirer_prompted",
        "semantic_hash": "sem",
        "model_pin_hash": "pin",
    }
    m.update(over)
    return m


def _row(manifest=None, status=None):
    return _run_row(_manifest(**(manifest or {})), {"usage": {}, **(status or {})}, {})


# ------------------------------------------------------------------------- the columns exist


def test_the_schema_declares_both_columns():
    """A row carrying a key the schema does not declare rejects the whole corpus, so the
    declaration and the writer have to land together."""
    names = {f.name for f in sch.RUNS}
    assert "n_prefix_user_turns" in names
    assert "foreign_trace_sha" in names


def test_the_prefix_count_is_an_integer_column_and_the_trace_a_string():
    by_name = {f.name: f for f in sch.RUNS}
    assert by_name["n_prefix_user_turns"].type == by_name["n_user_turns"].type
    assert by_name["foreign_trace_sha"].type == by_name["semantic_hash"].type


def test_both_columns_are_nullable():
    """Every run written before these existed has neither, and `runs/` is in practice the
    backup: a non-nullable column would make those directories unrecompactable."""
    by_name = {f.name: f for f in sch.RUNS}
    assert by_name["n_prefix_user_turns"].nullable
    assert by_name["foreign_trace_sha"].nullable


# ------------------------------------------------------------------------ the writer fills them


def test_the_prefix_user_turns_survive_compaction():
    assert _row(status={"n_prefix_user_turns": 12})["n_prefix_user_turns"] == 12


def test_an_absent_prefix_count_is_null_not_zero():
    """0 is the best possible score on the engagement endpoint. A run that never recorded the
    field must not be handed it."""
    assert _row()["n_prefix_user_turns"] is None


def test_a_real_zero_survives_as_zero():
    """And the distinction has to run both ways, or NULL-when-absent just moves the bug."""
    assert _row(status={"n_prefix_user_turns": 0})["n_prefix_user_turns"] == 0


def test_the_foreign_trace_survives_compaction():
    row = _row(manifest={"foreign_trace_sha": "d" * 64})
    assert row["foreign_trace_sha"] == "d" * 64


def test_a_run_of_our_own_carries_no_foreign_trace():
    """NULL, and `ELIGIBLE` admits a row on `IS NULL OR = ''` so either reads as 'ours'. NULL
    is written because it is the one value that cannot be confused with a measured sha."""
    assert _row()["foreign_trace_sha"] is None
    assert _row(manifest={"foreign_trace_sha": ""})["foreign_trace_sha"] is None
    assert _row(manifest={"foreign_trace_sha": None})["foreign_trace_sha"] is None


# --------------------------------------------------------------- against a real fork manifest


@pytest.mark.parametrize(
    "name",
    [
        "fork_tau2_retail_test_inquirer_prompted_s0.json",
        "fork_tau2_retail_test_self_ask_s0.json",
    ],
)
def test_a_recorded_fork_manifest_compacts_with_its_prefix_intact(name):
    """Not a hand-built dict: these are rows of `docs/reports/forks_tau2_retail_test.json`,
    copied from their run directories."""
    d = json.loads((FIXTURES / name).read_text())
    row = _run_row(d, {"usage": {}, "n_user_turns": 9, "n_prefix_user_turns": 7}, {})
    assert row["foreign_trace_sha"] == d["foreign_trace_sha"]
    assert row["n_prefix_user_turns"] == 7
    # THE NUMBER THE REPORT ACTUALLY PRINTS, reconstructed from the row alone.
    assert row["n_user_turns"] - row["n_prefix_user_turns"] == 2


def test_a_recorded_non_fork_manifest_compacts_as_one_of_ours():
    d = json.loads((FIXTURES / "nonfork_musique_dev.json").read_text())
    row = _run_row(d, {"usage": {}}, {})
    assert row["foreign_trace_sha"] is None
    assert row["n_prefix_user_turns"] is None
