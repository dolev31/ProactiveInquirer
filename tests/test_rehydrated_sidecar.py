"""The `rehydrated` flag lives OUTSIDE `runs.parquet`: `pi_eval.schema.schema_hash()` feeds
`pi_eval.score.metric_defs_hash()` feeds `scorer_hash()` (score.py's own module docstring
states the formula), and adding a column to `pi_eval.schema.RUNS` was measured to move all
three (`artifacts/rehydrated_answers_20260918/RESULT.md`, hash-identity check). This sidecar
is the replacement: same ground truth (`manifest.json["rehydrated_from"]`), a separate parquet
file `pi_eval.schema.TABLES` never declares, so it cannot move `schema_hash()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from scripts.rehydrated_answers.sidecar import scan_rehydrated, write_sidecar


def _run_dir(root: Path, run_id: str, **manifest_over) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    manifest = {"run_id": run_id, "suite_id": "tau2", "task_id": "t1", "arm_id": "drafter_only"}
    manifest.update(manifest_over)
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_a_manifest_with_rehydrated_from_scans_true(tmp_path):
    _run_dir(tmp_path, "a" * 32, rehydrated_from="scores/parquet + cache")

    assert scan_rehydrated(tmp_path) == {"a" * 32: True}


def test_a_manifest_without_rehydrated_from_scans_false(tmp_path):
    """Every run directory `restore_corpus.py` never touched carries no such key at all."""
    _run_dir(tmp_path, "b" * 32)

    assert scan_rehydrated(tmp_path) == {"b" * 32: False}


def test_an_explicit_null_rehydrated_from_is_also_false(tmp_path):
    _run_dir(tmp_path, "c" * 32, rehydrated_from=None)

    assert scan_rehydrated(tmp_path) == {"c" * 32: False}


def test_write_sidecar_produces_a_two_column_parquet_independent_of_pi_eval_schema(tmp_path):
    _run_dir(tmp_path / "runs", "d" * 32, rehydrated_from="scores/parquet + cache")
    _run_dir(tmp_path / "runs", "e" * 32)
    out = tmp_path / "sidecar.parquet"

    n = write_sidecar(tmp_path / "runs", out)

    assert n == 2
    table = pq.read_table(out)
    assert set(table.column_names) == {"run_id", "rehydrated"}
    got = dict(zip(table.column("run_id").to_pylist(), table.column("rehydrated").to_pylist()))
    assert got == {"d" * 32: True, "e" * 32: False}


def test_the_sidecar_module_never_imports_pi_eval():
    """The point of the sidecar: it must not be able to move `pi_eval.schema.schema_hash()`,
    and the mechanical guarantee of that is that it never IMPORTS `pi_eval` at all (the module
    docstring names `pi_eval.schema` in prose, explaining why, which is fine -- an import is
    what could reach `TABLES` and move the hash, so that is the exact thing checked)."""
    import ast

    import scripts.rehydrated_answers.sidecar as sidecar_mod

    tree = ast.parse(Path(sidecar_mod.__file__).read_text())
    imported_roots = {
        (n.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for n in [node]
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pi_eval" not in imported_roots, imported_roots
