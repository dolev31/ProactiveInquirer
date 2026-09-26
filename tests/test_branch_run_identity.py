"""A candidate branch is a different run from its parent, and run_id must say so.

`candidate_seed(s, 0) == s` BY CONSTRUCTION -- candidate 0 is the on-policy continuation and
reuses the parent's seed. VERIFIED: candidate_seed(1234, 0) == 1234.

`branch_of_run_id` was outside `semantic_hash`, so a candidate-0 branch of a run agreed with
its parent on every semantic field there is: same suite, task, arm, seed, corpus, pins,
prompts, caps. Same semantic_hash, therefore the same run_id, therefore the same run
DIRECTORY -- and `--resume` treats an existing directory as already done, so the branch is
silently skipped and `pairs.jsonl` gains nothing. Two runs that differ in what they DID
cannot share an identity; this is the same argument `pilot_flag` and `exploratory` already
won, for the same reason.

`branch_turn_idx` is in identity too: branching the same parent at turn 2 and at turn 5
produces two different experiments.
"""

from __future__ import annotations

from pi_run.manifest import build_manifest
from pinq.ids import SEMANTIC_FIELDS
from pinq.sampling import candidate_seed


def _m(**kw):
    base = dict(
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=1234,
        corpus_hash="c",
        budget_cap=16,
        max_turns=8,
        word_cap=300,
        code_version="v",
        dirty=False,
    )
    base.update(kw)
    return build_manifest(**base)


def test_candidate_zero_does_not_collide_with_its_parent() -> None:
    """The bug this exists for: same seed by construction, so identity must differ elsewhere."""
    assert candidate_seed(1234, 0) == 1234, "premise changed; re-read this test"
    parent = _m()
    branch = _m(branch_of_run_id=parent.run_id, branch_turn_idx=2)
    assert branch.run_id != parent.run_id, (
        "a candidate-0 branch shares its parent's run directory and --resume skips it"
    )


def test_the_branch_point_is_part_of_identity() -> None:
    """Branching at turn 2 and at turn 5 are different experiments."""
    parent = _m()
    a = _m(branch_of_run_id=parent.run_id, branch_turn_idx=2)
    b = _m(branch_of_run_id=parent.run_id, branch_turn_idx=5)
    assert a.run_id != b.run_id


def test_branches_of_different_parents_differ() -> None:
    p1, p2 = _m(task_id="t1"), _m(task_id="t2")
    a = _m(branch_of_run_id=p1.run_id, branch_turn_idx=1)
    b = _m(branch_of_run_id=p2.run_id, branch_turn_idx=1)
    assert a.run_id != b.run_id


def test_an_unbranched_run_is_unchanged() -> None:
    """Weak on its own -- both sides are built by the CURRENT code. See the frozen test."""
    assert _m().run_id == _m(branch_of_run_id=None, branch_turn_idx=None).run_id


# A REAL manifest from disk, and the semantic_hash it was WRITTEN with.
# runs/009097d0273e81e19f926d8976d09692/manifest.json, drgym/self_inquire.
FROZEN_FIELDS = {
    "arm_id": "self_inquire",
    "budget_cap": 16,
    "code_version": "bffae02cca5d04cf18528fc7bf541b0fde8c0290",
    "corpus_hash": "c79fb5fa8240c2fc2f9784992766b59a0f3c2a7d91ee81b4a8d369f799d5c799",
    "exploratory": False,
    "k": 5,
    "max_turns": 16,
    "model_pin_hash": "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf",
    "pilot_flag": False,
    "prompt_hashes": {
        "answerer": "446d45b65227a4fa302edba3b5a8192499584223e51cea68b777804318c915cd",
        "answerer_frozen": "446d45b65227a4fa302edba3b5a8192499584223e51cea68b777804318c915cd",
        "drafter_draft": "8a4a6b0218e798680ceb215831e0b0de3cc6137df9f45eb21db601badbf1fe35",
        "fragment_user_channel_placebo": "011360c10d0b63471e5d40fae43fcbd4ec9700aa8dddcbf37edfb3de95a3e85d",
        "inquirer_prompted": "814658eaa2f09eb798ab6f1019ea9ba728b68a3c00c4aa722f300e87f49b69f7",
        "self_inquire": "1bf17c20d6acae260340c4aec1b10929f65223ac4fbf0a0dce656819fad14f3d",
    },
    "prompt_variant_id": "v1",
    "questions_hash": "",
    "seed": 0,
    "suite_id": "drgym",
    "task_id": "787717",
    "upstream_pins": {"suite_version": "v1"},
}
FROZEN_HASH = "8a8968c725d025d45fbc099d2586087ad2d20a5d98068a7a5c014365eab0ab07"


def test_the_hash_of_an_existing_run_is_frozen() -> None:
    """Adding a field to SEMANTIC_FIELDS must not move the identity of runs already on disk.

    MEASURED when `branch_of_run_id`/`branch_turn_idx` were first added to the allowlist:
    736 of 736 stored manifests stopped recomputing to their own semantic_hash. A key present
    with value None still changes `canon`, so merely LISTING a field renames every run
    directory, makes --resume re-roll a completed corpus, and leaves every recorded run_id
    unreproducible. Omitting the branch keys when they are None fixed it -- 0 of 736 changed
    after -- because None means "not a branch", which is what every historical run is.

    The sibling test above could not catch this: it compares two manifests both built by the
    current code, so it moves WITH the bug. This one holds a hash captured from disk.
    """
    from pinq.ids import semantic_hash

    assert semantic_hash(dict(FROZEN_FIELDS)) == FROZEN_HASH


def test_a_branch_of_that_same_run_hashes_differently() -> None:
    """The other half: the omission must not make branches invisible."""
    from pinq.ids import semantic_hash

    branched = dict(FROZEN_FIELDS, branch_of_run_id="009097d0273e", branch_turn_idx=0)
    assert semantic_hash(branched) != FROZEN_HASH


def test_the_fields_are_in_the_allowlist() -> None:
    """SEMANTIC_FIELDS is a SECOND allowlist: a field in the dict but not here is inert.

    Adding a field to the manifest without adding it here changes nothing at all, silently,
    which is the specific trap this repo has hit before.
    """
    assert "branch_of_run_id" in SEMANTIC_FIELDS
    assert "branch_turn_idx" in SEMANTIC_FIELDS


def test_two_candidates_at_the_same_turn_are_different_runs() -> None:
    """Found by RUNNING it: 8 candidate units produced 2 run directories, not 8.

    `branch_of_run_id` and `branch_turn_idx` separate a branch from its parent and one branch
    point from another -- but every candidate at ONE turn shares both, and shares the parent's
    `seed` as well (the loop takes the parent seed and forks to the candidate seed only at the
    branch turn). So all four candidates at turn 1 hashed identically, three were skipped by
    --resume, and a preference pair -- which needs at least two continuations of one state --
    could never be formed.

    MEASURED before the fix: `pi train sample-candidates --turns 1,3 --n 4` reported 8 ok and
    left 2 run directories on disk.
    """
    parent = _m()
    ids = {
        _m(branch_of_run_id=parent.run_id, branch_turn_idx=1, branch_seed=s).run_id
        for s in (1234, 288579810, 971625, 44881)
    }
    assert len(ids) == 4, f"candidates collide: {len(ids)} distinct ids for 4 candidates"


def test_the_branch_seed_is_omitted_when_absent_too() -> None:
    """Same care as the other two: it must not move any existing run's identity."""
    from pinq.ids import semantic_hash

    assert semantic_hash(dict(FROZEN_FIELDS)) == FROZEN_HASH
    assert semantic_hash(dict(FROZEN_FIELDS, branch_seed=None)) == FROZEN_HASH
