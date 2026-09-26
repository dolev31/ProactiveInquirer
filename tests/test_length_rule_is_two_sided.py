"""The length diagnostic must detect length in BOTH directions.

`docs/GPU_RUNBOOK.md` section 5 states the rule as `chosen_shorter < 0.50 AND
chosen_longer > 0.60`, and `pair_accuracy`'s docstring states its PURPOSE as saying
"whether the ORDERING read length". Those two are not the same test: the threshold is
one-sided and fires only on a long-preference, while the purpose is two-sided.

MEASURED 2026-09-19 on all three E39-8b seeds at both distinct checkpoints, over the 970
ask-vs-ask dev pairs (no STOP on either side, so the action-type length asymmetry is
excluded by construction): chosen_longer 0.360-0.364, chosen_shorter 0.611-0.621,
len_sign_gap -0.246 to -0.263. The ordering plainly read length -- it preferred the
SHORTER question -- and the declared rule returns PASS on every one of them, because the
bias runs in the direction the threshold does not test.

This test pins the two-sided reading against the real verdict values. It fails against a
one-sided `kill_rule`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "length_audit" / "audit_tierA.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("audit_tierA_undertest", _SRC)
    assert spec is not None and spec.loader is not None, f"cannot import {_SRC}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cells(shorter: float, longer: float) -> dict[str, dict[str, float | int]]:
    return {
        "chosen_shorter": {"acc": shorter, "n": 470},
        "chosen_longer": {"acc": longer, "n": 475},
        "equal": {"acc": 0.72, "n": 25},
    }


# (label, chosen_shorter, chosen_longer) -- the two distinct checkpoints x three seeds.
REAL_8B = [
    ("s0.ckpt-600", 0.6170212765957447, 0.36210526315789476),
    ("s0.ckpt-707", 0.6191489361702127, 0.36),
    ("s1.ckpt-600", 0.6212765957446809, 0.35789473684210527),
    ("s1.ckpt-707", 0.6212765957446809, 0.36210526315789476),
    ("s2.ckpt-600", 0.6106382978723405, 0.3642105263157895),
    ("s2.ckpt-707", 0.6191489361702127, 0.3642105263157895),
]


@pytest.mark.parametrize(("label", "shorter", "longer"), REAL_8B)
def test_a_short_preference_is_flagged(label: str, shorter: float, longer: float) -> None:
    """A |gap| this large is length-reading whichever side it falls on."""
    mod = _audit_module()
    verdict, got_short, got_long = mod.kill_rule(_cells(shorter, longer))
    assert got_short == shorter and got_long == longer, f"{label}: cells not read back"
    gap = longer - shorter
    assert gap < -0.20, f"{label}: fixture is not a short-preference (gap={gap:.4f})"
    assert verdict != "PASS", (
        f"{label}: chosen_shorter={shorter:.4f} chosen_longer={longer:.4f} "
        f"len_sign_gap={gap:.4f} -- the ordering read length, and the rule returned PASS"
    )


def test_a_long_preference_is_still_flagged() -> None:
    """The original one-sided zone must keep firing: this is not a threshold relaxation."""
    mod = _audit_module()
    verdict, _, _ = mod.kill_rule(_cells(0.30, 0.85))
    assert verdict != "PASS", "the declared long-preference kill zone stopped firing"


def test_a_length_blind_ranker_passes() -> None:
    """Non-vacuity: the rule must still be able to return PASS, or it certifies nothing."""
    mod = _audit_module()
    verdict, _, _ = mod.kill_rule(_cells(0.61, 0.63))
    assert verdict == "PASS", f"a length-blind ranker was flagged: {verdict}"


def test_an_empty_cell_is_not_a_pass() -> None:
    mod = _audit_module()
    verdict, _, _ = mod.kill_rule(
        {"chosen_shorter": {"acc": None, "n": 0}, "chosen_longer": {"acc": 0.4, "n": 475}}
    )
    assert verdict != "PASS", "an unscored cell must not read as a pass"
