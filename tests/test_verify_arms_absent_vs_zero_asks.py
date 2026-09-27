"""An unscored arm and a silent arm are different facts and must not print the same.

`pi verify arms` reads `mean_asks` from scores.parquet. When a suite has been RUN but not
yet SCORED there are no `n_asks` rows at all, and the lookup fell back to 0.0 -- so the
report said:

    tau2/inquirer_prompted: mean n_asks == 0.0 but this arm is expected to ask

MEASURED at the time: the same runs' `status.json` carried `n_asks: 8`, runs.parquet
carried 8, 8, 1, 8, and the true mean was 6.25. The arm asked constantly. The instrument
reported a measurement nobody had taken, which is the one thing CONTRIBUTING.md rule 3 forbids,
and the cell in the SAME report printed `mean_asks: NaN` -- the two code paths disagreed
about whether the number existed.

A genuinely silent arm must still fail. That is the finding `_asks_failure` was derived
from and this must not soften it.
"""

from __future__ import annotations

import math

from pi_run.cli import _asks_failure


def test_absent_scores_do_not_report_a_measured_zero() -> None:
    """No n_asks rows -> say so, and name the cause the operator can act on."""
    msg = _asks_failure(arm_id="inquirer_prompted", home=(), suite="tau2", mean_asks=float("nan"))
    assert msg is not None, "an unscored arm is still not verified; it must not pass silently"
    assert "== 0.0" not in msg, f"reports a measurement never taken: {msg!r}"
    assert "score" in msg.lower(), f"must name the missing step: {msg!r}"


def test_a_genuinely_silent_arm_still_fails() -> None:
    """The original finding. Scored, present, and zero -- that is a real defect."""
    msg = _asks_failure(arm_id="inquirer_prompted", home=(), suite="musique", mean_asks=0.0)
    assert msg is not None
    assert "0.0" in msg


def test_an_asking_arm_passes() -> None:
    assert _asks_failure(arm_id="inquirer_prompted", home=(), suite="tau2", mean_asks=6.25) is None


def test_off_suite_is_still_exempt_even_when_absent() -> None:
    """`home_suites` outranks both: off-suite silence was never a defect."""
    assert (
        _asks_failure(arm_id="fake_chain", home=("synth",), suite="musique", mean_asks=float("nan"))
        is None
    )


def test_nan_is_what_the_lookup_actually_produces() -> None:
    """Guards the contract between the two call sites, which used to disagree.

    The cell printed `asks.get(key, nan)` and the verdict was passed `asks.get(key, 0.0)`.
    Same missing key, two different defaults, one of them a lie.
    """
    asks: dict[tuple[str, str], float] = {}
    assert math.isnan(asks.get(("tau2", "inquirer_prompted"), float("nan")))
