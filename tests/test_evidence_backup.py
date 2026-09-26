"""Tests for `scripts/evidence_backup`: discovery, hashing, census and the backup pipeline.

Lane L0.10b backs up isolated per-campaign score stores that nothing else in the repo reaches
(`.gitignore`: "content-addressed and regenerable", but several exist in exactly one place on
one machine). These tests build small synthetic stores under `tmp_path` rather than reading a
real store under `artifacts/` -- the same reason `tests/test_train_gate.py` gives for never
reading the shared `scores/parquet/` in a test: a real store changes under a concurrent
campaign, and a test must not pass or fail on whatever it currently holds.

`test_changing_one_table_changes_only_that_tables_digest` and
`test_census_reproduces_on_an_unchanged_store` together are the two properties lane L0.10b's own
brief names explicitly: the census reproduces on an unchanged store, and a changed byte changes
its digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.evidence_backup import backup, census, discover, manifest  # noqa: E402


def _write_table(path: Path, columns: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _write_store(store_dir: Path, *, n_runs: int = 3) -> None:
    """A minimal but real two-table store: `runs.parquet` + `scores.parquet`, the two tables
    `census.compute_census` reads by name for the provenance triple."""
    run_ids = [f"r{i}" for i in range(n_runs)]
    _write_table(
        store_dir / "runs.parquet",
        {
            "run_id": run_ids,
            "arm_id": ["trained"] * n_runs,
            "suite_id": ["musique"] * n_runs,
            "split": ["test"] * n_runs,
            "code_version": ["abc123"] * n_runs,
        },
    )
    _write_table(
        store_dir / "scores.parquet",
        {
            "run_id": run_ids,
            "scorer_hash": ["deadbeef"] * n_runs,
            "graph_version": ["v1"] * n_runs,
        },
    )


# ---------------------------------------------------------------------------
# discover.find_score_stores
# ---------------------------------------------------------------------------


def test_finds_a_directory_that_directly_holds_a_parquet_file(tmp_path):
    _write_store(tmp_path / "store_a")
    assert discover.find_score_stores(tmp_path) == [tmp_path / "store_a"]


def test_finds_nested_stores_at_different_depths(tmp_path):
    shallow = tmp_path / "campaign" / "scores_parquet"
    deep = tmp_path / "campaign" / "gate" / "sub" / "parquet_x"
    _write_store(shallow)
    _write_store(deep)
    assert discover.find_score_stores(tmp_path) == sorted([shallow, deep])


def test_a_symlink_farm_of_raw_run_directories_is_not_a_store(tmp_path):
    # artifacts/gate_wiki2_stacked_20260918/farm's own shape: a directory of symlinks to run
    # directories, none of which is itself a parquet file.
    real_run = tmp_path / "runs" / "abc"
    real_run.mkdir(parents=True)
    (real_run / "manifest.json").write_text("{}")
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "abc").symlink_to(real_run, target_is_directory=True)
    assert discover.find_score_stores(tmp_path) == []


def test_a_symlinked_parquet_file_still_counts(tmp_path):
    _write_store(tmp_path / "real_store")
    linked_dir = tmp_path / "linked_store"
    linked_dir.mkdir()
    (linked_dir / "runs.parquet").symlink_to(tmp_path / "real_store" / "runs.parquet")
    assert linked_dir in discover.find_score_stores(tmp_path)


def test_never_descends_into_a_nested_git_checkout(tmp_path):
    # A real example from lane L0.10b: a scratch farm's own `wt/` was a full second git
    # worktree of this repo, with its own `artifacts/`. Its contents are a different backup
    # decision, not part of whatever farm was scanned to reach it.
    nested = tmp_path / "farm" / "wt"
    nested.mkdir(parents=True)
    (nested / ".git").write_text("gitdir: /elsewhere\n")
    _write_store(nested / "artifacts" / "some_store")
    assert discover.find_score_stores(tmp_path) == []


def test_missing_root_returns_empty_list(tmp_path):
    assert discover.find_score_stores(tmp_path / "does_not_exist") == []


def test_output_is_sorted(tmp_path):
    _write_store(tmp_path / "zzz")
    _write_store(tmp_path / "aaa")
    found = discover.find_score_stores(tmp_path)
    assert found == sorted(found)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    assert manifest.sha256_file(f) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_changes_when_one_byte_changes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    before = manifest.sha256_file(f)
    f.write_bytes(b"hello worlD")
    after = manifest.sha256_file(f)
    assert before != after


def test_sha256_manifest_uses_relative_posix_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_bytes(b"a")
    (tmp_path / "b.txt").write_bytes(b"b")
    assert set(manifest.sha256_manifest(tmp_path)) == {"sub/a.txt", "b.txt"}


def test_diff_manifests_all_empty_when_equal():
    m = {"a": "1", "b": "2"}
    diff = manifest.diff_manifests(m, dict(m))
    assert diff == {"missing_in_backup": [], "unexpected_in_backup": [], "sha_mismatch": []}


def test_diff_manifests_reports_all_three_kinds_of_disagreement():
    source = {"same": "1", "changed": "2", "only_in_source": "3"}
    backed_up = {"same": "1", "changed": "X", "only_in_backup": "4"}
    diff = manifest.diff_manifests(source, backed_up)
    assert diff == {
        "missing_in_backup": ["only_in_source"],
        "unexpected_in_backup": ["only_in_backup"],
        "sha_mismatch": ["changed"],
    }


# ---------------------------------------------------------------------------
# census.compute_census
# ---------------------------------------------------------------------------


def test_census_reproduces_on_an_unchanged_store(tmp_path):
    store = tmp_path / "store"
    _write_store(store, n_runs=4)
    assert census.compute_census(store) == census.compute_census(store)


def test_census_reads_the_provenance_triple_and_row_counts(tmp_path):
    store = tmp_path / "store"
    _write_store(store, n_runs=4)
    c = census.compute_census(store)
    assert c["n_tables"] == 2
    assert c["tables"] == {"runs": 4, "scores": 4}
    assert c["arms"] == {"trained": 4}
    assert c["suites"] == {"musique": 4}
    assert c["splits"] == {"test": 4}
    assert c["code_versions"] == {"abc123": 4}
    assert c["scorer_hashes"] == {"deadbeef": 4}
    assert c["graph_versions"] == {"v1": 4}
    assert set(c["sha256"]) == {"runs.parquet", "scores.parquet"}


def test_changing_one_table_changes_only_that_tables_digest(tmp_path):
    store = tmp_path / "store"
    _write_store(store, n_runs=3)
    before = census.compute_census(store)

    # Rewrite scores.parquet with different content ( +1 row); runs.parquet is untouched.
    _write_table(
        store / "scores.parquet",
        {
            "run_id": ["r0", "r1", "r2", "r3"],
            "scorer_hash": ["deadbeef"] * 4,
            "graph_version": ["v1"] * 4,
        },
    )
    after = census.compute_census(store)

    assert after["sha256"]["scores.parquet"] != before["sha256"]["scores.parquet"]
    assert after["sha256"]["runs.parquet"] == before["sha256"]["runs.parquet"]
    assert after["tables"]["scores"] == 4
    assert after["tables"]["runs"] == before["tables"]["runs"]


def test_a_missing_column_censuses_as_absent_not_a_crash(tmp_path):
    store = tmp_path / "store"
    _write_table(store / "runs.parquet", {"run_id": ["r0"], "arm_id": ["trained"]})
    # no scores.parquet at all -- a store scored before scorer_hash/graph_version existed.
    c = census.compute_census(store)
    assert c["arms"] == {"trained": 1}
    assert c["splits"] == {}  # column absent from runs.parquet
    assert c["scorer_hashes"] == {}  # scores.parquet absent entirely
    assert c["tables"] == {"runs": 1}


def test_an_empty_table_censuses_as_zero_rows_not_a_crash(tmp_path):
    # artifacts/banking/scores_parquet's own judgments.parquet: seen at 0 rows, normal.
    store = tmp_path / "store"
    _write_table(store / "runs.parquet", {"run_id": pa.array([], type=pa.string())})
    c = census.compute_census(store)
    assert c["tables"]["runs"] == 0


# ---------------------------------------------------------------------------
# backup: home-path sanitization
# ---------------------------------------------------------------------------


def test_safe_display_substitutes_the_given_home(tmp_path):
    home = tmp_path / "Users" / "someone"
    under_home = home / "pinq-ablation" / "c1_score" / "parquet_control"
    got = backup._safe_display(under_home, home=home)
    assert got == "$HOME/pinq-ablation/c1_score/parquet_control"
    assert str(home) not in got


def test_safe_display_leaves_unrelated_paths_alone(tmp_path):
    home = tmp_path / "Users" / "someone"
    elsewhere = tmp_path / "private" / "tmp" / "farm"
    assert backup._safe_display(elsewhere, home=home) == str(elsewhere)


def test_safe_display_handles_home_itself(tmp_path):
    home = tmp_path / "Users" / "someone"
    assert backup._safe_display(home, home=home) == "$HOME"


# ---------------------------------------------------------------------------
# backup: copy / verify
# ---------------------------------------------------------------------------


def test_verify_copy_ok_after_a_clean_copy(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_store(src)
    backup.copy_store(src, dst)
    result = backup.verify_copy(src, dst)
    assert result["ok"] is True
    assert result["n_files"] == 2
    assert result["missing_in_backup"] == []
    assert result["unexpected_in_backup"] == []
    assert result["sha_mismatch"] == []
    # Both raw manifests come back too, so backup_one can persist them without re-hashing.
    assert set(result["source_manifest"]) == {"runs.parquet", "scores.parquet"}
    assert result["source_manifest"] == result["backup_manifest"]


def test_verify_copy_catches_a_corrupted_backup_file(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_store(src)
    backup.copy_store(src, dst)
    (dst / "runs.parquet").write_bytes(b"corrupted")
    result = backup.verify_copy(src, dst)
    assert result["ok"] is False
    assert result["sha_mismatch"] == ["runs.parquet"]


def test_copy_store_never_deletes_extra_content_already_in_dst(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_store(src)
    dst.mkdir()
    (dst / "extra_from_a_previous_run.txt").write_text("keep me")
    backup.copy_store(src, dst)
    assert (dst / "extra_from_a_previous_run.txt").exists()


# ---------------------------------------------------------------------------
# backup: _rows_for_root naming
# ---------------------------------------------------------------------------


def test_rows_for_artifacts_root_relativizes_directly(tmp_path):
    store = tmp_path / "frontier" / "scores_parquet.trained"
    _write_store(store)
    assert backup._rows_for_root(tmp_path, strip_parent=False) == [
        (store, Path("frontier/scores_parquet.trained"))
    ]


def test_rows_for_extra_root_keeps_the_farms_own_name(tmp_path):
    farm_root = tmp_path / "some-farm"
    store = farm_root / "c1_score" / "parquet_control"
    _write_store(store)
    assert backup._rows_for_root(farm_root, strip_parent=True) == [
        (store, Path("some-farm/c1_score/parquet_control"))
    ]


# ---------------------------------------------------------------------------
# backup: backup_one
# ---------------------------------------------------------------------------


def test_backup_one_writes_a_verified_copy_and_a_census(tmp_path):
    store = tmp_path / "artifacts" / "banking" / "scores_parquet"
    _write_store(store, n_runs=5)
    backup_root = tmp_path / "backup_root"
    census_root = tmp_path / "census_root"

    result = backup.backup_one(
        store,
        rel=Path("banking/scores_parquet"),
        backup_root=backup_root,
        census_root=census_root,
    )

    assert result["verification"]["ok"] is True
    dst = backup_root / "banking" / "scores_parquet"
    assert (dst / "runs.parquet").read_bytes() == (store / "runs.parquet").read_bytes()

    written = json.loads((census_root / "banking" / "scores_parquet.census.json").read_text())
    assert written["tables"] == {"runs": 5, "scores": 5}
    assert written["backup_relative_path"] == "banking/scores_parquet"
    assert written["size_bytes"] == result["size_bytes"] > 0

    # Both sha256 manifests lane L0.10b asks for are persisted as siblings of the copied
    # store's own directory, under the backup root -- not inside it (see manifest_path_for's
    # docstring for why not inside it).
    source_manifest_path = backup_root / "banking" / "scores_parquet.source.sha256.json"
    backup_manifest_path = backup_root / "banking" / "scores_parquet.backup.sha256.json"
    assert result["source_manifest_path"] == backup._safe_display(source_manifest_path)
    assert result["backup_manifest_path"] == backup._safe_display(backup_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    backup_manifest = json.loads(backup_manifest_path.read_text())
    assert source_manifest == backup_manifest
    assert set(source_manifest) == {"runs.parquet", "scores.parquet"}
    # The manifest files themselves are not inside the copied store directory, so a later
    # verify_copy against the live source still comes back clean.
    assert backup.verify_copy(store, dst)["ok"] is True


def test_backup_one_raises_evidence_mismatch_when_verification_fails(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _write_store(store)

    def fake_verify_copy(_src, _dst):
        return {
            "ok": False,
            "n_files": 0,
            "missing_in_backup": ["oops"],
            "unexpected_in_backup": [],
            "sha_mismatch": [],
        }

    monkeypatch.setattr(backup, "verify_copy", fake_verify_copy)
    with pytest.raises(backup.EvidenceMismatch) as exc_info:
        backup.backup_one(
            store, rel=Path("x"), backup_root=tmp_path / "b", census_root=tmp_path / "c"
        )
    assert exc_info.value.verification["missing_in_backup"] == ["oops"]


# ---------------------------------------------------------------------------
# backup: main() CLI, end to end
# ---------------------------------------------------------------------------


def test_main_backs_up_every_store_and_reports_zero_failures(tmp_path):
    artifacts_root = tmp_path / "artifacts"
    _write_store(artifacts_root / "frontier" / "scores_parquet.trained", n_runs=2)
    _write_store(artifacts_root / "banking" / "scores_parquet", n_runs=3)
    census_root = tmp_path / "evidence_backup"
    out_path = tmp_path / "backup_run.json"

    exit_code = backup.main(
        [
            "--artifacts-root",
            str(artifacts_root),
            "--backup-root",
            str(tmp_path / "backup_root"),
            "--backup-name",
            "evidence-test",
            "--census-root",
            str(census_root),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(out_path.read_text())
    assert (summary["n_stores"], summary["n_verified_ok"], summary["n_failed"]) == (2, 2, 0)
    dst = tmp_path / "backup_root" / "evidence-test"
    assert (dst / "frontier" / "scores_parquet.trained" / "runs.parquet").exists()
    assert (census_root / "banking" / "scores_parquet.census.json").exists()


def test_main_is_idempotent_on_rerun(tmp_path):
    artifacts_root = tmp_path / "artifacts"
    _write_store(artifacts_root / "banking" / "scores_parquet")
    argv = [
        "--artifacts-root",
        str(artifacts_root),
        "--backup-root",
        str(tmp_path / "backup_root"),
        "--backup-name",
        "evidence-test",
        "--census-root",
        str(tmp_path / "census_root"),
    ]
    assert backup.main(argv) == 0
    assert backup.main(argv) == 0  # a rerun tops up an existing backup; it does not fail


def test_main_output_never_contains_the_raw_home_path_for_an_extra_root(tmp_path, monkeypatch):
    fake_home = tmp_path / "Users" / "researcher"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    _write_store(fake_home / "pinq-ablation" / "c1_score" / "parquet_control")
    empty_artifacts_root = tmp_path / "artifacts"
    empty_artifacts_root.mkdir()
    out_path = tmp_path / "out.json"

    exit_code = backup.main(
        [
            "--artifacts-root",
            str(empty_artifacts_root),
            "--extra-root",
            str(fake_home / "pinq-ablation"),
            "--backup-root",
            str(tmp_path / "backup_root"),
            "--backup-name",
            "evidence-test",
            "--census-root",
            str(tmp_path / "census_root"),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    raw = out_path.read_text()
    assert str(fake_home) not in raw
    assert "$HOME/pinq-ablation" in raw
