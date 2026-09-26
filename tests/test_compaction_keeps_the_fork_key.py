"""The compaction must be able to rebuild a fork's STATE KEY, and it could not.

WHAT THIS COST. `runs/` was destroyed by an `rm -rf` with a mis-resolved variable. There was
no snapshot and no backup, so `scores/parquet/` -- what `pi compact` had written -- was the
only copy of the corpus, and it very nearly worked: 22,607 run directories were rebuilt from
it with ZERO `state_mismatch` under `render_state`'s own hash guard, which is as strong a
fidelity check as exists here.

Three fields were missing, and one of them mattered. `branch_turn_idx` appears NOWHERE in the
compaction -- not on `runs.parquet`, and 0 of 134,378 rows of `turns.parquet`. It is half of
`pinq_train.export.dataset.state_key`, so without it every fork candidate rebuilds as a
standalone run, C(1,2) = 0, and the entire preference dataset is unbuildable from the
compaction. 16,080 forks came back; only the 8,695 whose key could be read out of a surviving
preference export were usable. Two derivations were tried and REJECTED on validation against
those known labels: "first turn whose question differs from the parent" was 31.6% exact (the
replay diverges from the parent), "longest common prefix with a sibling" 41.7% (siblings
agree past the fork turn, because several sample the same question there).

`branch_seed` and `prompt_hashes` are the other two. `branch_seed` identifies WHICH candidate
of a state a run is; `prompt_hashes` feeds `_pins_sha`, and the exporter refuses to pair
across instruments -- a guard measured to matter (266 cross-era pairs, newer side chosen
35.3%, p = 1e-6). Rebuilt runs had to carry `code_version` in its place, which splits some
cohorts that shared prompts and never merges two that differed: conservative, but not the
real thing.

THE RULE THIS ENCODES: a field that `state_key`, `_pins_sha` or run identity depends on must
survive compaction, because the compaction is in practice the backup. The archive is not a
summary; anything the exporter needs and the archive drops is data that only looks safe.
"""

from __future__ import annotations

from pi_run.compact import _run_row


def _manifest(**over):
    m = {
        "run_id": "cand1",
        "suite_id": "musique",
        "task_id": "t1",
        "arm_id": "inquirer_prompted",
        "semantic_hash": "sem",
        "model_pin_hash": "pin",
        "branch_of_run_id": "parent1",
        "branch_turn_idx": 2,
        "branch_seed": 7,
        "prompt_hashes": {"inquirer": "abc", "drafter": "def"},
    }
    m.update(over)
    return m


def _row(**over):
    return _run_row(_manifest(**over), {"usage": {}}, {})


def test_the_fork_turn_survives_compaction():
    """Half of `state_key`. Without it a rebuilt candidate is a run of one and forms no pair."""
    assert _row()["branch_turn_idx"] == 2


def test_a_non_fork_run_records_no_fork_turn():
    """None, not 0: turn 0 is the commonest fork, so a default of 0 would invent a state."""
    assert _row(branch_turn_idx=None, branch_of_run_id=None)["branch_turn_idx"] is None


def test_the_candidate_seed_survives_compaction():
    """Which candidate of the state this is. Two candidates that differ only here are the
    whole point of a same-state pair."""
    assert _row()["branch_seed"] == 7


def test_the_prompt_hashes_survive_compaction():
    """`_pins_sha` digests (model_pin_hash, prompt_hashes) and the exporter refuses to pair
    across instruments. Dropping the prompts leaves only the model, which silently re-admits
    the cross-era pairs that guard exists to refuse."""
    assert _row()["prompt_hashes"] == '{"drafter": "def", "inquirer": "abc"}'


def test_the_prompt_hashes_are_canonical_and_absence_is_empty():
    """Serialised sorted, so the column is comparable across runs; '' when a run carried none,
    which is how a legacy run reads and is distinguishable from a run that carried some."""
    assert _row(prompt_hashes={"b": "2", "a": "1"})["prompt_hashes"] == '{"a": "1", "b": "2"}'
    assert _row(prompt_hashes={})["prompt_hashes"] == ""
