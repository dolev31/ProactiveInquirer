"""`pi_run.sweep.plan()`'s own `max_turns`/`k`/`budget_cap`/`concurrency` defaults used to
retype `UnitSpec`'s field defaults of the same name as a second, hand-maintained literal --
`max_turns: int = 16` beside `UnitSpec.max_turns: int = 16`, and so on for the other three.
`plan()` has exactly one live caller (`pi_run.cli`, which always passes all four explicitly),
so today this path resolves nothing in production; the fix is hygiene against a FUTURE caller
that omits one, not a fix for a live bug.

A NECESSARY WRINKLE, NOT A DEFICIENCY. `candidate_specs`'s and `_rung1_defaults`'s equivalent
fallbacks are read fresh on every CALL, so a bare `monkeypatch.setattr` on `UnitSpec`'s field
is visible immediately. `plan()`'s defaults are different in kind: they are function-SIGNATURE
defaults, and Python evaluates a `def` statement's default expressions exactly once, when the
`def` runs -- which for a module-level function is at IMPORT time. `_UNIT_SPEC_DEFAULTS` and
`plan.__defaults__` are therefore both baked in when `pi_run.sweep` is first imported; a bare
monkeypatch of `UnitSpec` afterwards changes nothing already baked in, and a test that only
did that would fail even against the fixed code -- a false negative, not evidence of anything.
The fix protects the SOURCE-EDIT-then-restart boundary (change `UnitSpec.max_turns`'s default
in the source, restart the process) rather than a hot-patch of an already-running one -- which
is the realistic threat model: nobody mutates a dataclass field's default on a live production
process. `importlib.reload(sweep)` re-runs the module body, which is what makes that boundary
observable inside one test process, and is exactly what the second test below does.
"""

from __future__ import annotations

import importlib

import pi_run.sweep as sweep
from pi_run.worker import UnitSpec


def _plan(**over):
    return sweep.plan(
        suite_id="s",
        corpus_dir="c",
        task_ids=["t0"],
        arm_ids=["a0"],
        seeds=[0],
        runs_root="r",
        cache_root="c2",
        **over,
    )


def test_omitting_every_cap_still_matches_unitspecs_own_defaults():
    """The prior behaviour is the default, so every existing caller is unchanged."""
    specs = _plan()
    assert specs[0].max_turns == 16
    assert specs[0].k == 5
    assert specs[0].budget_cap is None
    assert specs[0].concurrency == 1


def test_a_changed_unitspec_default_reaches_plan_only_after_a_reload():
    """Demonstrates the wrinkle from the module docstring directly: the SAME monkeypatch is
    invisible to `plan()` before a reload and followed after one, on the SAME running process.
    This is not two competing behaviours -- it is one, observed at both sides of the def-time
    boundary the fix is meant to survive."""
    field = UnitSpec.__dataclass_fields__["max_turns"]
    old_default = field.default
    try:
        field.default = 41

        specs_no_reload = _plan()
        assert specs_no_reload[0].max_turns == 16, (
            "still the value baked in when pi_run.sweep was first imported -- expected, "
            "not a bug: a signature default cannot see a change made after def-time"
        )

        importlib.reload(sweep)
        specs_reloaded = _plan()
        assert specs_reloaded[0].max_turns == 41, (
            "reload re-runs `_UNIT_SPEC_DEFAULTS = {f.name: f.default for f in "
            "dataclasses.fields(UnitSpec)}` and plan()'s own `def` line, so this is where "
            "the fix's real guarantee -- a source-edit-plus-restart follows UnitSpec -- shows up"
        )
    finally:
        field.default = old_default
        importlib.reload(sweep)  # leave the module clean for every test that runs after this one
