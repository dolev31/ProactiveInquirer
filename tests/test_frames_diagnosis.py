"""The FRAMES diagnosis reports intervals, so its bootstrap must not depend on row order.

This exists because the defect was real and was observed, not anticipated. duckdb returns rows in
no guaranteed order without an ORDER BY, and the first version of
`scripts/frames_diagnosis/why_the_answer_null.py` resampled indices into whatever order arrived.
Two consecutive runs on identical data produced identical point estimates and DIFFERENT bounds,
including one cell that read [+0.000, +2.500] on one run and [-0.021, +2.479] on the next. That is
this repository's recorded fingerprint for an unsorted bootstrap, and a bound that moves across
reruns can flip a verdict.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "frames_diagnosis" / "why_the_answer_null.py"


@pytest.fixture(scope="module")
def diag():
    if not MODULE.exists():
        pytest.skip(f"{MODULE} is not present")
    spec = importlib.util.spec_from_file_location("why_the_answer_null", MODULE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_boot_is_invariant_to_the_order_rows_arrive_in(diag):
    src = random.Random(7)
    values = [round(src.uniform(-1, 1), 6) for _ in range(300)]
    shuffled = list(values)
    random.Random(11).shuffle(shuffled)
    assert shuffled != values, "the shuffle must actually reorder, or this test proves nothing"
    assert diag.boot(values) == diag.boot(shuffled)


def test_boot_ratio_is_invariant_to_order_and_keeps_pairs_together(diag):
    rng = random.Random(3)
    pairs = [(round(rng.uniform(-1, 1), 6), round(rng.uniform(0.5, 2.0), 6)) for _ in range(200)]
    shuffled = list(pairs)
    random.Random(5).shuffle(shuffled)
    assert shuffled != pairs
    a = diag.boot_ratio([n for n, _ in pairs], [d for _, d in pairs])
    b = diag.boot_ratio([n for n, _ in shuffled], [d for _, d in shuffled])
    assert a == b


def test_boot_would_have_caught_the_defect(diag):
    """NON-VACUITY: an unsorted bootstrap on this input really does move its bounds.

    Without this, the two tests above could pass against an implementation that never had the bug.
    """
    import numpy as np

    src = random.Random(7)
    values = [round(src.uniform(-1, 1), 6) for _ in range(300)]
    shuffled = list(values)
    random.Random(11).shuffle(shuffled)
    assert shuffled != values

    def unsorted_boot(v, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        a = np.asarray(v, dtype=float)  # deliberately NOT sorted
        idx = rng.integers(0, len(a), (n, len(a)))
        m = np.sort(a[idx].mean(axis=1))
        return a.mean(), m[int(0.025 * n)], m[int(0.975 * n)]

    one, two = unsorted_boot(values), unsorted_boot(shuffled)
    assert one[0] == pytest.approx(two[0]), "the POINT estimate is order-invariant either way"
    assert (one[1], one[2]) != (two[1], two[2]), "the bounds must move, or the fixture is wrong"
