"""Two lanes wrote the shared store without the lock on one evening. Make it a guard.

WHAT HAPPENED, 2026-09-18. The written rule is "take the store lock before compacting". At
22:04:39 one lane started `pi compact --out scores/parquet` from a snapshot farm under its own
scratchpad; at 22:04:56 a second lane took the lock and started its own compaction of the same
directory. Both staged seven tables and both renamed them into place. `os.replace` is atomic
per file, so neither directory tore -- but the later batch of renames silently discarded the
earlier one's entire result, and the earlier lane was running the OLD compactor, so for the
twenty minutes between the two commits the store carried 134,370 turn rows keyed to NULL
again. Nothing errored. A rule that is only written down is a rule that gets raced.

WHY THE DEFAULT IS BESIDE THE STORE AND NOT IN A SCRATCHPAD. The lock we had been using lives
in a per-session scratchpad directory, which no other session can see or name -- so two lanes
"holding the lock" can be holding two different directories, which is exactly what a lock
cannot do. The default is `<out>/.store.lock`: one path, derived from the thing it protects,
visible to every process that can write the store. `PI_STORE_LOCK` overrides it so an existing
convention keeps working.

An isolated `--out` is NOT the shared store and needs no lock: the frames convention builds a
store outside the repository on purpose, and requiring a lock there would make the guard
something people route around.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pi_run.cli import _store_lock_state

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


def _repo(tmp_path: Path) -> Path:
    """A tree `repo_root()` will recognise, with a shared store inside it."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "scores" / "parquet").mkdir(parents=True)
    runs = root / "runs"
    d = runs / ("a" * 32)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"run_id": "a" * 32, "suite_id": "wiki2", "task_id": "t", "arm_id": "x"})
    )
    (d / "status.json").write_text(json.dumps({"status": "ok", "usage": {}}))
    (d / "turns.jsonl").write_text(json.dumps(_TURN) + "\n")
    return root


def _compact(root: Path, out: Path, env_extra: dict[str, str] | None = None):
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.pop("PI_STORE_LOCK", None)
    env.update(env_extra or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pi_run.cli",
            "compact",
            "--root",
            str(root),
            "--runs-root",
            str(root / "runs"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )


# --------------------------------------------------------------- the state helper


def test_no_lock_directory_at_all(tmp_path):
    held, owner, path = _store_lock_state(tmp_path / "parquet")
    assert held is False
    assert owner == ""
    assert path == tmp_path / "parquet" / ".store.lock"


def test_a_lock_directory_with_no_owner_file_is_not_held(tmp_path):
    """`mkdir` is the atomic step and writing the owner is a second one. A directory with no
    owner is a lock someone is still taking, or one a crash left behind -- not one to trust."""
    (tmp_path / "parquet" / ".store.lock").mkdir(parents=True)
    held, owner, _ = _store_lock_state(tmp_path / "parquet")
    assert held is False
    assert owner == ""


def test_a_lock_with_an_owner_is_held_and_names_its_owner(tmp_path):
    lock = tmp_path / "parquet" / ".store.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text("Fri Sep 18 22:04:56 IDT 2026 L0.7 recompact\n")
    held, owner, _ = _store_lock_state(tmp_path / "parquet")
    assert held is True
    assert owner == "Fri Sep 18 22:04:56 IDT 2026 L0.7 recompact"


def test_pi_store_lock_overrides_the_default_path(tmp_path, monkeypatch):
    elsewhere = tmp_path / "session" / "store.lock"
    elsewhere.mkdir(parents=True)
    (elsewhere / "owner").write_text("a peer")
    monkeypatch.setenv("PI_STORE_LOCK", str(elsewhere))
    held, owner, path = _store_lock_state(tmp_path / "parquet")
    assert (held, owner, path) == (True, "a peer", elsewhere)


# --------------------------------------------------------------- pi compact


def test_compacting_the_shared_store_without_the_lock_writes_nothing(tmp_path):
    root = _repo(tmp_path)
    out = root / "scores" / "parquet"

    res = _compact(root, out)

    assert res.returncode != 0, res.stdout
    assert "no store lock held" in res.stderr
    assert str(out / ".store.lock") in res.stderr
    # NOTHING WRITTEN. A refusal that has already replaced six tables is not a refusal.
    assert list(out.iterdir()) == []


def test_compacting_the_shared_store_with_the_lock_proceeds_and_names_the_owner(tmp_path):
    root = _repo(tmp_path)
    out = root / "scores" / "parquet"
    lock = out / ".store.lock"
    lock.mkdir()
    (lock / "owner").write_text("L0.7 recompact")

    res = _compact(root, out)

    assert res.returncode == 0, res.stderr
    assert "shared store is locked by L0.7 recompact" in res.stderr
    assert pq.read_table(out / "runs.parquet").num_rows == 1


def test_an_isolated_out_needs_no_lock(tmp_path):
    """The frames convention builds a store outside the repository on purpose. Requiring a
    lock there would teach people to route around the guard."""
    root = _repo(tmp_path)
    out = tmp_path / "isolated" / "parquet"

    res = _compact(root, out)

    assert res.returncode == 0, res.stderr
    assert "store lock" not in res.stderr
    assert pq.read_table(out / "runs.parquet").num_rows == 1


def test_the_default_out_is_the_shared_store_and_is_guarded(tmp_path):
    """No `--out` at all resolves to <root>/scores/parquet, which is the shared store."""
    import os

    root = _repo(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.pop("PI_STORE_LOCK", None)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "pi_run.cli",
            "compact",
            "--root",
            str(root),
            "--runs-root",
            str(root / "runs"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "no store lock held" in res.stderr
    assert list((root / "scores" / "parquet").iterdir()) == []


# --------------------------------------------------------------- pi score


def test_scoring_into_the_shared_store_without_the_lock_refuses_before_reading_gold(tmp_path):
    """`pi score` appends to scores.parquet in the same directory, so it races the same way.
    The refusal comes BEFORE any gold is read and before any model call is priced."""
    import os

    root = _repo(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env.pop("PI_STORE_LOCK", None)
    env["PI_GOLD_ROOT"] = str(tmp_path / "gold")
    res = subprocess.run(
        [sys.executable, "-m", "pi_run.cli", "score", "--root", str(root), "--allow-no-judge"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "no store lock held" in res.stderr
    assert list((root / "scores" / "parquet").iterdir()) == []


@pytest.mark.parametrize("held", [False, True])
def test_the_guard_can_take_both_values_on_the_same_tree(tmp_path, held):
    """NON-VACUITY. The same command, the same directory, one difference: whether the lock is
    there. A guard that only ever saw one of the two would be untested."""
    root = _repo(tmp_path)
    out = root / "scores" / "parquet"
    if held:
        (out / ".store.lock").mkdir()
        (out / ".store.lock" / "owner").write_text("owner")

    res = _compact(root, out)

    assert (res.returncode == 0) is held, (res.returncode, res.stderr)
