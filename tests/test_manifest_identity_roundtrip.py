"""A manifest on disk must recompute its own semantic_hash.

MEASURED across the 736 recorded runs: 273 do not. Every one of them is missing
`exploratory` from its manifest.json (and 80 are also missing `k` and `questions_hash`) --
those fields entered run identity AFTER the runs were written, so recomputation inserts a
key the original hash never had and produces a different value.

The existing guard (`tests/test_runtime.py`) holds SEMANTIC_FIELDS equal to the keys built
inside `RunManifest.semantic_hash`. That is one of the two links in the chain. The other is
`manifest_to_dict`, which is what actually reaches disk: a field can be in the allowlist AND
in the property AND still be absent from the serialized file, at which point the run is
permanently un-auditable -- nobody can recompute its identity from its own record.

This test closes that link. It would have failed on the day `exploratory` was added.

WHAT IT DOES NOT CLAIM. It cannot make the 273 historical manifests recompute. Their hashes
were correct under the field set in force when they were written, and asserting otherwise
would be asserting that run identity never changed. Nor does staleness cost anything
operationally: `code_version` is itself in SEMANTIC_FIELDS, so a run from any earlier commit
gets a new run_id today regardless, and `--resume` never matches across a commit.
"""

from __future__ import annotations

from pi_run.manifest import build_manifest, manifest_to_dict
from pinq.ids import SEMANTIC_FIELDS, semantic_hash


def _manifest(**kw):
    base = dict(
        suite_id="musique",
        task_id="t1",
        arm_id="inquirer_prompted",
        policy_id="p",
        seed=3,
        corpus_hash="c0ffee",
        budget_cap=8,
        max_turns=16,
        word_cap=300,
        code_version="deadbeef",
        dirty=False,
        k=5,
        questions_hash="qh",
        exploratory=True,
        prompt_hashes={"inquirer_prompted": "abc"},
        upstream_pins={"suite_version": "v1"},
    )
    base.update(kw)
    return build_manifest(**base)


def test_a_serialized_manifest_recomputes_its_own_hash() -> None:
    """The link the old guard did not cover: allowlist -> property -> FILE."""
    m = _manifest()
    on_disk = manifest_to_dict(m)
    assert semantic_hash({k: on_disk.get(k) for k in SEMANTIC_FIELDS}) == m.semantic_hash


def test_every_identity_field_survives_serialization() -> None:
    """Names the missing field instead of just failing on a hash mismatch.

    A bare hash comparison says 'these differ'; this says WHICH key never reached disk,
    which is the difference between a five-minute fix and an afternoon.
    """
    on_disk = manifest_to_dict(_manifest())
    # Fields omitted from identity when None are legitimately absent -- see pinq.ids.
    optional = {"branch_of_run_id", "branch_turn_idx", "branch_seed"}
    missing = [k for k in SEMANTIC_FIELDS if k not in on_disk and k not in optional]
    assert not missing, f"in run identity but never written to manifest.json: {missing}"


def test_it_holds_for_a_branch_manifest_too() -> None:
    """The optional fields must round-trip when they ARE set."""
    parent = _manifest()
    m = _manifest(branch_of_run_id=parent.run_id, branch_turn_idx=2, branch_seed=99)
    on_disk = manifest_to_dict(m)
    assert semantic_hash({k: on_disk.get(k) for k in SEMANTIC_FIELDS}) == m.semantic_hash
    assert m.semantic_hash != parent.semantic_hash


def test_it_holds_for_an_llm_free_arm() -> None:
    """No pins, no prompt hashes -- the shape that has the most empty identity fields."""
    m = _manifest(arm_id="fake_chain", prompt_hashes={}, upstream_pins={})
    on_disk = manifest_to_dict(m)
    assert semantic_hash({k: on_disk.get(k) for k in SEMANTIC_FIELDS}) == m.semantic_hash
