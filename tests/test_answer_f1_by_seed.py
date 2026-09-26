"""Pure helpers and refusals of `scripts/seed_identity/answer_f1_by_seed.py`, on SYNTHETIC inputs.

The reader's numbers are locked on the real stores by the reader itself (it refuses to write when
a lock fails); what is pinned here is that each helper a lock is built from can fail:
- pairing on (task, rollout seed) reproduces the per-task mean difference on a balanced population
  and REFUSES an unbalanced one, instead of silently re-weighting a training seed;
- a lock refuses a value off by more than its tolerance and names both numbers;
- the printed-value lock compares against the three-decimal print, so a value that rounds to a
  different print is refused;
- the decision rule needs all three 50,000-resample intervals to exclude zero on the SAME side.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MOD = REPO / "scripts" / "seed_identity" / "answer_f1_by_seed.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("seed_identity_answer_f1_by_seed", MOD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _arm(values: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    return dict(values)


def test_pairing_on_rollout_seed_equals_the_task_mean_difference_when_balanced(m):
    s1 = _arm({("t1", 0): 0.2, ("t1", 1): 0.4, ("t2", 0): 1.0, ("t2", 1): 0.0})
    s2 = _arm({("t1", 0): 0.6, ("t1", 1): 0.8, ("t2", 0): 0.5, ("t2", 1): 0.5})
    base = _arm({("t1", 0): 0.1, ("t1", 1): 0.3, ("t2", 0): 0.25, ("t2", 1): 0.75})
    a, b = m.pair_on_rollout_seed([s1, s2], base)
    assert a == pytest.approx({"t1": 0.5, "t2": 0.5}, abs=1e-15)
    assert b == pytest.approx({"t1": 0.2, "t2": 0.5}, abs=1e-15)
    assert list(a) == sorted(a)


def test_pairing_refuses_a_comparator_missing_one_rollout_seed(m):
    s1 = _arm({("t1", 0): 0.2, ("t1", 1): 0.4})
    base = _arm({("t1", 0): 0.1})
    with pytest.raises(ValueError, match="unpaired"):
        m.pair_on_rollout_seed([s1], base)


def test_pairing_refuses_a_training_seed_missing_a_task(m):
    s1 = _arm({("t1", 0): 0.2, ("t1", 1): 0.4, ("t2", 0): 0.0, ("t2", 1): 0.0})
    s2 = _arm({("t1", 0): 0.2, ("t1", 1): 0.4})
    base = _arm({("t1", 0): 0.1, ("t1", 1): 0.3, ("t2", 0): 0.0, ("t2", 1): 0.0})
    with pytest.raises(ValueError, match="unpaired"):
        m.pair_on_rollout_seed([s1, s2], base)


def test_lock_refuses_a_value_off_by_more_than_the_tolerance_and_names_both(m):
    m.lock(
        "x",
        {"point": 0.1, "lo": -0.2, "hi": 0.3, "n": 5},
        {"point": 0.1, "lo": -0.2, "hi": 0.3, "n": 5},
    )
    with pytest.raises(SystemExit) as exc:
        m.lock(
            "cell",
            {"point": 0.1 + 1e-9, "lo": -0.2, "hi": 0.3, "n": 5},
            {"point": 0.1, "lo": -0.2, "hi": 0.3, "n": 5},
        )
    assert "cell" in str(exc.value) and "0.100000001" in str(exc.value) and "0.1" in str(exc.value)
    with pytest.raises(SystemExit):
        m.lock(
            "n",
            {"point": 0.1, "lo": -0.2, "hi": 0.3, "n": 4},
            {"point": 0.1, "lo": -0.2, "hi": 0.3, "n": 5},
        )


def test_printed_lock_compares_the_three_decimal_print(m):
    m.lock_printed("ok", 0.031107772355353624, "+0.031")
    m.lock_printed("neg", -0.010809277323983206, "-0.011")
    with pytest.raises(SystemExit) as exc:
        m.lock_printed("musique s1", 0.0314, "+0.032")
    assert "+0.031" in str(exc.value) and "+0.032" in str(exc.value)
    # half up on the exact decimal: 0.2175 prints +0.218, not binary formatting's +0.217
    m.lock_printed("tie", 0.2175, "+0.218")
    with pytest.raises(SystemExit):
        m.lock_printed("tie", 0.2175, "+0.217")


def test_decided_needs_three_50k_intervals_on_one_side(m):
    def r(lo, hi, nb=50_000):
        return {"n_boot": nb, "ci_lo": lo, "ci_hi": hi}

    assert m.decided([r(0.01, 0.2, 10_000), r(0.01, 0.2), r(0.02, 0.2), r(0.001, 0.2)]) is True
    assert m.decided([r(-0.3, -0.1), r(-0.3, -0.1), r(-0.3, -0.01)]) is True
    assert m.decided([r(0.01, 0.2), r(0.0, 0.2), r(0.02, 0.2)]) is False
    assert m.decided([r(0.01, 0.2), r(-0.3, -0.1), r(0.02, 0.2)]) is False
    with pytest.raises(ValueError):
        m.decided([r(0.01, 0.2), r(0.01, 0.2)])
