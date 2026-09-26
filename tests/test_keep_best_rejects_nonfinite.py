"""`by_stop2x2` must not select on a NaN.

`scripts/hpc/eval_checkpoints.sh` logs a warning and continues when `PI_REFERENCE_ASK` is
unset, unlike `eval_offline.sh` which dies. Every STOP row is then skipped, `n_done` is 0 and
`stop_confusion.p_stop_given_done` is written as NaN -- MEASURED 2026-09-19 on all six E39-8b
verdicts (n_done 0.0, n_not_done 350, n_skipped_no_reference 650).

`_stop2x2` guarded only `is None`. NaN is not None, so it returned `(nan + 1.0) / 2 = nan`, the
candidate passed the `is not None` filter, and `max()` over keys that are all NaN returns
whichever element came FIRST -- every NaN comparison is False. So the criterion silently
selected an arbitrary checkpoint while reporting a winner, and the winner depended on dict
order rather than on any measurement.

A selection that cannot be wrong about its own basis is worse than no selection: the registry
row would name weights chosen by iteration order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "hpc" / "keep_best.py"


def _mod():
    spec = importlib.util.spec_from_file_location("keep_best_undertest", _SRC)
    assert spec is not None and spec.loader is not None, f"cannot import {_SRC}"
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _verdict(p_stop, p_ask, nll):
    return {
        "stop_confusion": {"p_stop_given_done": p_stop, "p_ask_given_not_done": p_ask},
        "sft_nll": {"nll_per_token": nll},
    }


def test_stop2x2_rejects_a_nan_cell() -> None:
    m = _mod()
    assert m._stop2x2(_verdict(float("nan"), 1.0, 4.9)) is None, (
        "a NaN p_stop_given_done must disqualify the candidate, not become a NaN score"
    )


def test_stop2x2_rejects_a_nan_in_either_cell() -> None:
    m = _mod()
    assert m._stop2x2(_verdict(0.5, float("nan"), 4.9)) is None
    assert m._stop2x2(_verdict(float("nan"), float("nan"), 4.9)) is None


def test_stop2x2_still_scores_real_cells() -> None:
    """Non-vacuity: the criterion must still work, or it certifies nothing."""
    m = _mod()
    got = m._stop2x2(_verdict(0.4, 0.8, 4.9))
    assert got is not None
    assert abs(got - 0.6) < 1e-9, got


def test_select_best_returns_no_stop_winner_when_every_cell_is_nan() -> None:
    """The real failure: all-NaN candidates must yield None, not the first one in dict order."""
    m = _mod()
    nan = float("nan")
    verdicts = {
        "200": _verdict(nan, 1.0, 4.90),
        "400": _verdict(nan, 1.0, 4.80),
        "600": _verdict(nan, 1.0, 4.70),
        "final": _verdict(nan, 1.0, 4.60),
    }
    out = m.select_best(verdicts)
    assert out["by_stop2x2"] is None, (
        f"selected {out['by_stop2x2']!r} on four NaN scores -- that is dict order, not a measurement"
    )
    # by_nll is unaffected and must still pick the smallest NLL.
    assert out["by_nll"] == "final", out["by_nll"]


def test_select_best_is_order_independent_under_nan() -> None:
    """Same measurements, reversed insertion order, same verdict."""
    m = _mod()
    nan = float("nan")
    items = [
        ("200", _verdict(nan, 1.0, 4.90)),
        ("400", _verdict(nan, 1.0, 4.80)),
        ("600", _verdict(nan, 1.0, 4.70)),
    ]
    a = m.select_best(dict(items))["by_stop2x2"]
    b = m.select_best(dict(reversed(items)))["by_stop2x2"]
    assert a == b, f"order changed the winner: {a!r} vs {b!r}"
