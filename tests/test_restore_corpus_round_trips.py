"""The corpus must be restorable from the archive, and that must be TESTED, not hoped for.

WHY. `runs/` was destroyed by an `rm -rf` whose argument came from a `find` that had matched
its own search root. There was no snapshot and no Time Machine destination. 22,607 run
directories came back only because `pi compact` had written every field somewhere else and
the response cache still held the answers -- and the restore itself was an ad-hoc script
written under pressure, with no test, whose first version silently rebuilt every fork as a
standalone run because it read `branch_turn_idx` off the wrong row.

A restore path that has never been exercised is not a backup. These tests round-trip the
whole loop -- write run dirs, compact them, delete them, restore, compare -- so the archive's
claim to be sufficient is checked by machine on every commit rather than discovered during
an incident.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.restore_corpus import restore

from pi_run.compact import compact


def _run_dir(root: Path, run_id: str, *, branch=None, branch_turn=None, n_turns=2) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "semantic_hash": f"sem-{run_id}",
                "model_pin_hash": "pin",
                "prompt_hashes": {"inquirer": "aaa", "drafter": "bbb"},
                "suite_id": "musique",
                "task_id": "t1",
                "arm_id": "inquirer_prompted",
                "policy_id": "prompted",
                "seed": 0,
                "split": "train",
                "corpus_hash": "c",
                "corpus_dir": str(root),
                "budget_cap": 24,
                "max_turns": 24,
                "word_cap": 30,
                "code_version": "deadbeef",
                "dirty": False,
                "branch_of_run_id": branch,
                "branch_turn_idx": branch_turn,
                "branch_seed": 3_229_923_555,  # exceeds int32; the live corpus has this value
            }
        )
    )
    (d / "turns.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "turn_idx": i,
                    "action_kind": "ask",
                    "question": f"q{i}?",
                    "question_id": f"qid{i}",
                    "rationale": "",
                    "target": "",
                    "parent_uids": [],
                    "retrieved_uids": [f"u{i}"],
                    "new_uids": [f"u{i}"],
                    "n_retrieved": 1,
                    "n_new": 1,
                    "response_text": f"answer {i}",
                    "draft_sha": "",
                    "draft_text": f"draft {i}",
                    "subset_hash_before": f"before{i}",
                    "subset_hash_after": f"after{i}",
                    "branch_of_run_id": branch,
                    "branch_turn_idx": branch_turn,
                    "candidate_id": None,
                    "candidate_rank": None,
                    "forced": False,
                    "policy_conf": None,
                    "depth_pred": None,
                    "bm25_hits": None,
                    "spec_bits": None,
                    "usage_tok_prompt": 10,
                    "usage_tok_completion": 5,
                    "usage_tok_reasoning": 0,
                    "usage_tok_cached": 0,
                    "usage_tok_total": 15,
                    "usage_usd": 0.001,
                    "usage_wall_ms": 12,
                    "usage_n_calls": 1,
                }
            )
            for i in range(n_turns)
        )
    )
    (d / "ledger.jsonl").write_text(
        json.dumps(
            {
                "currency": "retrieval_calls",
                "charged": 1.0,
                "cumulative": 1.0,
                "turn_idx": 0,
                "cap": 24.0,
                "hard": True,
            }
        )
    )
    (d / "status.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "stop_reason": "policy_stop",
                "wall_ms": 100,
                "n_turns": n_turns,
                "usage": {"tok_total": 30, "usd": 0.002},
            }
        )
    )
    (d / "outcome.json").write_text(
        json.dumps({"answer": {"text": "the answer", "cited_unit_ids": ["u0"], "n_words": 2}})
    )
    return d


def _corpus(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    _run_dir(runs, "parent1")
    _run_dir(runs, "cand1", branch="parent1", branch_turn=1)
    _run_dir(runs, "cand2", branch="parent1", branch_turn=1)
    return runs


def test_the_archive_carries_the_fork_state_key(tmp_path):
    """The field whose absence made the first restore useless. Without it every candidate
    rebuilds as a run of one, C(1,2)=0, and the preference dataset is unbuildable."""
    import pandas as pd

    runs = _corpus(tmp_path)
    compact(runs, tmp_path / "parquet")
    r = pd.read_parquet(tmp_path / "parquet" / "runs.parquet").set_index("run_id")
    assert int(r.loc["cand1", "branch_turn_idx"]) == 1
    assert r.loc["parent1", "branch_turn_idx"] is None or pd.isna(
        r.loc["parent1", "branch_turn_idx"]
    )
    assert int(r.loc["cand1", "branch_seed"]) == 3_229_923_555
    assert json.loads(r.loc["cand1", "prompt_hashes"]) == {"drafter": "bbb", "inquirer": "aaa"}


def test_a_deleted_corpus_restores_from_the_archive_alone(tmp_path):
    """The whole loop: compact, destroy, restore. No network, no cache needed for the fields
    that decide a state key."""
    import shutil

    runs = _corpus(tmp_path)
    compact(runs, tmp_path / "parquet")
    before = {d.name: json.loads((d / "manifest.json").read_text()) for d in sorted(runs.iterdir())}
    shutil.rmtree(runs)
    assert not runs.exists()

    res = restore(runs, parq=tmp_path / "parquet", cache=tmp_path / "no-cache")
    assert res["runs"] == 3
    after = {d.name: json.loads((d / "manifest.json").read_text()) for d in sorted(runs.iterdir())}
    assert set(after) == set(before)
    for rid in before:
        for k in (
            "suite_id",
            "task_id",
            "arm_id",
            "seed",
            "split",
            "budget_cap",
            "max_turns",
            "branch_of_run_id",
            "branch_turn_idx",
            "semantic_hash",
            "code_version",
        ):
            assert after[rid][k] == before[rid][k], (rid, k)


def test_the_restored_turns_are_the_recorded_ones(tmp_path):
    """`state_text` is re-rendered from `retrieved_uids` and verified against
    `subset_hash_before`, so those two fields returning wrong would poison every row."""
    import shutil

    runs = _corpus(tmp_path)
    compact(runs, tmp_path / "parquet")
    before = [json.loads(x) for x in (runs / "cand1" / "turns.jsonl").read_text().splitlines()]
    shutil.rmtree(runs)
    restore(runs, parq=tmp_path / "parquet", cache=tmp_path / "no-cache")
    after = [json.loads(x) for x in (runs / "cand1" / "turns.jsonl").read_text().splitlines()]
    assert len(after) == len(before)
    for a, b in zip(after, before):
        for k in (
            "turn_idx",
            "question",
            "retrieved_uids",
            "new_uids",
            "response_text",
            "draft_text",
            "subset_hash_before",
            "subset_hash_after",
        ):
            assert a[k] == b[k], k


def test_the_restored_corpus_still_groups_into_states(tmp_path):
    """The end that matters: two candidates at one state must still form a pairable group
    after a round trip. This is what the first restore silently lost."""
    import shutil

    from pinq_train.export.dataset import state_key

    runs = _corpus(tmp_path)
    compact(runs, tmp_path / "parquet")
    shutil.rmtree(runs)
    restore(runs, parq=tmp_path / "parquet", cache=tmp_path / "no-cache")
    keys = []
    for rid in ("cand1", "cand2"):
        m = json.loads((runs / rid / "manifest.json").read_text())
        keys.append(
            state_key(
                {
                    "suite_id": m["suite_id"],
                    "task_id": m["task_id"],
                    "run_id": rid,
                    "turn_idx": 1,
                    "branch_of_run_id": m["branch_of_run_id"],
                    "branch_turn_idx": m["branch_turn_idx"],
                }
            )
        )
    assert keys[0] == keys[1] is not None, "the two candidates must land in ONE state group"
