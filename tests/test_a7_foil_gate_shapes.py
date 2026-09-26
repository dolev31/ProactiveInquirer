"""The foil gate must read both shapes of `attention_check.expected`, or it crashes on a
bundle that has attention checks in it.

WHAT HAPPENED. `pi annotate export` plants two kinds of check and they are not the same
shape. An A7 foil pins named response FIELDS:

    {"expected": {"b_reaches": "stays_stated"}}

while an A3_node attention check pins the VERDICT alone:

    {"expected": "not_a_need"}

`scripts/a7_artifacts.py` called `.items()` on that value, so a campaign whose bundle carried
`--n-attention` items died with `AttributeError: 'str' object has no attribute 'items'` --
AFTER the slot-bias gate had already run and printed, which is the worst place to fail: the
expensive pass is finished, the gate said the rater was clean, and the artifacts are still
not written.

The scalar form is compared against the record's `verdict`, which is the field A3_node
answers with. Both forms are counted into one `foil_catch`, because they measure one thing --
did the rater notice a planted item whose answer is known in advance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from a7_artifacts import foil_caught  # noqa: E402


def test_a_field_map_expectation_is_matched_field_by_field():
    exp = {"b_reaches": "stays_stated"}
    assert foil_caught({"b_reaches": "stays_stated", "preference": "a"}, exp) is True
    assert foil_caught({"b_reaches": "reaches_unstated", "preference": "a"}, exp) is False


def test_every_named_field_must_match():
    exp = {"a_reaches": "stays_stated", "b_reaches": "stays_stated"}
    assert foil_caught({"a_reaches": "stays_stated", "b_reaches": "stays_stated"}, exp) is True
    assert foil_caught({"a_reaches": "stays_stated", "b_reaches": "reaches_unstated"}, exp) is False


def test_a_scalar_expectation_is_matched_against_the_verdict():
    """A3_node answers with `verdict`; the planted item pins that value alone."""
    assert foil_caught({"discoverability": "kb", "verdict": "not_a_need"}, "not_a_need") is True
    assert foil_caught({"discoverability": "kb", "verdict": "required"}, "not_a_need") is False


def test_a_missing_field_is_a_miss_not_a_crash():
    """A rater that answered with a different schema failed the check; it did not break the
    build. This is the shape that killed a finished campaign."""
    assert foil_caught({}, {"b_reaches": "stays_stated"}) is False
    assert foil_caught({}, "not_a_need") is False
    assert foil_caught({"verdict": None}, "not_a_need") is False
