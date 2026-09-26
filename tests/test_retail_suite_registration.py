"""`tau2_retail` is TRAINABLE; `tau2` is not. The two must never be confused for each other.

The whole point of adding retail is to get a user simulator into TRAINING without spending
the zero-shot transfer target. `tau2` (banking_knowledge) stays in `EVAL_ONLY_SUITES` because
"transfers to an action-consequential environment it was never trained on" is a much stronger
claim than a within-suite gain, and 97 tasks is far too few to split.

That guarantee rests on `assert_trainable` testing EXACT membership. A refactor to
`suite_id.startswith("tau2")` -- which reads like a tidy-up -- would make retail eval-only and
delete the reason it was built; the reverse mistake, dropping the exact check, would make the
transfer target trainable and silently destroy the claim. Both directions are pinned here.
"""

from __future__ import annotations

import pytest

from pinq_train.split import EVAL_ONLY_SUITES, SplitViolation, assert_trainable, split_of

RETAIL = "tau2_retail"


def test_the_transfer_target_is_still_eval_only() -> None:
    assert "tau2" in EVAL_ONLY_SUITES
    with pytest.raises(SplitViolation):
        assert_trainable("tau2", "task_001")


def test_retail_is_not_eval_only() -> None:
    assert RETAIL not in EVAL_ONLY_SUITES, (
        "retail exists to be trainable; if it is eval-only there is no reason to have built it"
    )


def test_a_prefix_match_would_break_this_and_is_not_what_is_used() -> None:
    """The refactor this test exists to stop: `startswith('tau2')` catches both."""
    assert RETAIL.startswith("tau2"), "premise of the test"
    assert RETAIL not in EVAL_ONLY_SUITES


def test_a_retail_train_task_is_exportable() -> None:
    tid = next(t for t in (f"t{i}" for i in range(500)) if split_of(RETAIL, t) == "train")
    assert_trainable(RETAIL, tid)  # must not raise


def test_a_retail_test_task_is_still_refused() -> None:
    """Trainable as a SUITE does not mean trainable as a TASK: the split wall still applies."""
    tid = next(t for t in (f"t{i}" for i in range(500)) if split_of(RETAIL, t) == "test")
    with pytest.raises(SplitViolation):
        assert_trainable(RETAIL, tid)


def test_retail_and_banking_bucket_independently() -> None:
    """`bucket` hashes the suite id with the task id, so the same task key must not land in
    the same split on both suites by construction -- otherwise a retail training task could
    be predicted from the banking split and vice versa."""
    same = sum(1 for i in range(200) if split_of(RETAIL, f"t{i}") == split_of("tau2", f"t{i}"))
    assert same < 190, f"splits are suspiciously aligned across suites ({same}/200)"


def test_retail_has_gold_and_can_be_asked_for() -> None:
    """`pi gold stats --suite tau2` once answered `invalid choice`. Same failure, same fix."""
    from pi_run.cmd_gold import GOLD_SUITES

    assert RETAIL in GOLD_SUITES
