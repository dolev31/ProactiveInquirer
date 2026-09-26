"""A launched unit's RECORDED pins must match what the launcher REQUESTED, or refuse.

WHY THIS EXISTS. The `PI_MODEL_DRAFTER`/`PI_MODEL_ANSWERER` precedence bug (see
`test_model_role_pin_precedence.py`) was caught by reading manifests back out of a completed
408-unit campaign, after the money was spent. Fixing `load_env.sh` stops THAT mechanism, but a
guard that only prevents one known mechanism is not a guard against the class: the next
regression will not look like `${VAR:-default}` after `. .env`. The only check that generalizes
is one that reads what actually got RECORDED in a unit's own manifest and refuses to proceed if
it does not match what was asked for. That is what this module is for -- it is deliberately
independent of `load_env.sh` and would catch a mis-pin from ANY cause.

TWO REAL SHAPES, MEASURED, not invented. `pins.{inquirer,drafter,answerer}` come from
`LitellmClient.pin(role)` / `collect_pins` and carry `model_id`. `user_sim` is NOT under `pins`
at all -- tau2's user simulator is upstream of pinq's own client, so it is recorded under
`upstream_pins.user_sim` as a raw string. A comparator that looked for `pins.user_sim` would
silently see nothing to compare, which reads as "no mismatch" -- the same false-negative shape
as `search-zeros-must-be-interrogated`. This module checks user_sim by the field that actually
holds it.

  - the BUGGY shape: runs/005e0b9debcf458f5bc98c8aed12a553/manifest.json (rescued campaign root),
    `pins.drafter.model_id == pins.answerer.model_id == "openai/aws/gpt-oss-120b"` where the
    requested value was `claude-sonnet-5` -- this is the exact 408-unit defect.
  - the CORRECT shape: runs/a1e7ac0217f92b975849ba5779dc2b0b/manifest.json (a genuine member of
    the published claude-sonnet-5 airline column, RESULT.md's `d7cd7be043bbc488`), where every
    role matches what was requested.
"""

from __future__ import annotations

from scripts.tau2_campaign.verify_pins import check_pins

# The buggy manifest's relevant fields, reproduced verbatim from
# runs/005e0b9debcf458f5bc98c8aed12a553/manifest.json.
BUGGY_MANIFEST = {
    "run_id": "005e0b9debcf458f5bc98c8aed12a553",
    "arm_id": "inquirer_trained",
    "pins": {
        "inquirer": {"role": "inquirer", "model_id": "qwen3-8b-dpo-stacked-notdone-both"},
        "drafter": {"role": "drafter", "model_id": "openai/aws/gpt-oss-120b"},
        "answerer": {"role": "answerer", "model_id": "openai/aws/gpt-oss-120b"},
    },
    "upstream_pins": {"user_sim": "openai/aws/gpt-oss-120b"},
}

# The correct manifest's relevant fields, reproduced verbatim from
# runs/a1e7ac0217f92b975849ba5779dc2b0b/manifest.json.
CORRECT_MANIFEST = {
    "run_id": "a1e7ac0217f92b975849ba5779dc2b0b",
    "arm_id": "inquirer_prompted",
    "pins": {
        "inquirer": {"role": "inquirer", "model_id": "openai/aws/claude-sonnet-5"},
        "drafter": {"role": "drafter", "model_id": "openai/aws/claude-sonnet-5"},
        "answerer": {"role": "answerer", "model_id": "openai/aws/claude-sonnet-5"},
    },
    "upstream_pins": {"user_sim": "openai/aws/gpt-oss-120b"},
}

REQUESTED_TRAINED = {
    "inquirer": "qwen3-8b-dpo-stacked-notdone-both",
    "drafter": "openai/aws/claude-sonnet-5",
    "answerer": "openai/aws/claude-sonnet-5",
    "user_sim": "openai/aws/gpt-oss-120b",
}

# CORRECT_MANIFEST is a genuine `inquirer_prompted` arm member -- that arm's inquirer is ALSO
# claude-sonnet-5 (there is no trained checkpoint in the loop), unlike REQUESTED_TRAINED's
# `inquirer_trained` arm. Comparing it against REQUESTED_TRAINED would compare one arm's
# manifest to a different arm's request and call the resulting inquirer mismatch a code defect;
# it would not be one. Each manifest is checked against the request its OWN arm actually makes.
REQUESTED_PROMPTED = {
    "inquirer": "openai/aws/claude-sonnet-5",
    "drafter": "openai/aws/claude-sonnet-5",
    "answerer": "openai/aws/claude-sonnet-5",
    "user_sim": "openai/aws/gpt-oss-120b",
}


def test_the_buggy_manifest_is_flagged_on_drafter_and_answerer():
    """THE REGRESSION, REPRODUCED FROM A REAL RUN. Before this guard existed, nothing read this
    manifest back and compared it to what phase5_shard.sh asked for; 408 units shipped this way."""
    mismatches = check_pins(BUGGY_MANIFEST, REQUESTED_TRAINED)
    roles = {m.role for m in mismatches}
    assert roles == {"drafter", "answerer"}, (
        f"expected exactly drafter+answerer flagged on the measured buggy manifest, got {roles}"
    )
    by_role = {m.role: m for m in mismatches}
    assert by_role["drafter"].requested == "openai/aws/claude-sonnet-5"
    assert by_role["drafter"].recorded == "openai/aws/gpt-oss-120b"


def test_the_correct_manifest_is_clean():
    """NON-VACUITY. A comparator that flags everything (or nothing) regardless of content would
    pass the test above too if it always returned every role. This is the real published-column
    member; it must come back clean against the same requested mapping shape."""
    mismatches = check_pins(CORRECT_MANIFEST, REQUESTED_PROMPTED)
    assert mismatches == [], f"a genuinely correctly-pinned manifest was flagged: {mismatches}"


def test_user_sim_is_read_from_upstream_pins_not_pins():
    """user_sim lives in `upstream_pins`, not `pins` -- a comparator that only looked under
    `pins` would find no `user_sim` key there and could silently treat that as clean, which is
    a false negative of exactly the shape `search-zeros-must-be-interrogated` warns about."""
    manifest = {
        "run_id": "x",
        "arm_id": "inquirer_prompted",
        "pins": {
            "inquirer": {"model_id": "a"},
            "drafter": {"model_id": "a"},
            "answerer": {"model_id": "a"},
        },
        "upstream_pins": {"user_sim": "openai/aws/gpt-oss-120b"},
    }
    requested = {
        "inquirer": "a",
        "drafter": "a",
        "answerer": "a",
        "user_sim": "openai/aws/claude-sonnet-5",
    }
    mismatches = check_pins(manifest, requested)
    roles = {m.role for m in mismatches}
    assert roles == {"user_sim"}, f"user_sim mismatch was not detected via upstream_pins: {roles}"


def test_a_role_absent_from_the_manifest_is_a_mismatch_not_a_pass():
    """A manifest missing a role entirely (a schema change, a partial write) must not read as
    'nothing to compare, so clean' -- that is a guessed-field-name-reads-as-absence failure."""
    manifest = {"run_id": "x", "arm_id": "inquirer_prompted", "pins": {}, "upstream_pins": {}}
    requested = {"inquirer": "a", "drafter": "b", "answerer": "c", "user_sim": "d"}
    mismatches = check_pins(manifest, requested)
    assert {m.role for m in mismatches} == {"inquirer", "drafter", "answerer", "user_sim"}


def test_a_role_not_requested_is_not_checked():
    """The launcher for a given arm config does not always freeze every role (e.g. judge is
    never frozen by any tau2 shard); requesting nothing for a role must not manufacture a
    mismatch against whatever the manifest happens to record for it."""
    manifest = {
        "run_id": "x",
        "arm_id": "inquirer_prompted",
        "pins": {
            "inquirer": {"model_id": "a"},
            "drafter": {"model_id": "b"},
            "answerer": {"model_id": "c"},
        },
        "upstream_pins": {"user_sim": "d"},
    }
    requested = {"inquirer": "a", "drafter": "b", "answerer": "c"}  # user_sim omitted on purpose
    mismatches = check_pins(manifest, requested)
    assert mismatches == []
