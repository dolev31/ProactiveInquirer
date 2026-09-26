"""Tests for `scripts/frames_seeds/contrasts.py`.

The one bug class this guards against: POOLING OVER SEEDS BY COLLAPSING BEFORE PAIRING. If a
task's seed values were averaged (or otherwise merged) before the trained/prompted arms are
paired, a task that only one arm completed at one seed would silently borrow the OTHER arm's
value from a DIFFERENT seed instead of being dropped from that seed's contrast. The tests below
construct a case where the wrong order (collapse, then pair) and the right order (pair on
`(task_id, seed)`, then cluster the survivors by task) provably disagree, and assert the
function returns the right one -- not just "a plausible-looking number".

`load_metric` needs `duckdb` and a real scored snapshot; that one test is skipped, naming the
absent path, on a checkout that does not carry `artifacts/frames_frontier/scores_parquet.frames`
(mirroring `tests/test_paper_figures_iclr.py`'s own skip convention).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "frames_seeds" / "contrasts.py"


def _load():
    """`scripts/` is not an importable package, so load the file by path."""
    spec = importlib.util.spec_from_file_location("frames_seeds_contrasts", GEN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    if not GEN.exists():
        pytest.skip(f"generator absent: {GEN}")
    return _load()


# --------------------------------------------------------------------- per_seed_and_pooled


def test_per_seed_recovers_a_constant_offset_and_excludes_nothing(mod):
    """3 tasks, 2 seeds, trained = prompted + 0.1 everywhere: every row pairs, every seed's
    point estimate is exactly +0.1 (zero variance -> the BCa interval collapses to a point)."""
    tasks = ["A", "B", "C"]
    seeds = [0, 1]
    trained = {(t, s): 0.6 for t in tasks for s in seeds}
    prompted = {(t, s): 0.5 for t in tasks for s in seeds}

    results = mod.per_seed_and_pooled(trained, prompted, seeds, n_boot=200, n_perm=200)

    for s in seeds:
        est = results[str(s)]
        assert est.n == 3, f"seed {s}: expected 3 paired tasks, got {est.n}"
        assert est.point == pytest.approx(0.1, abs=1e-9)
        assert est.ci_lo == pytest.approx(0.1, abs=1e-9)
        assert est.ci_hi == pytest.approx(0.1, abs=1e-9)

    pooled = results["pooled"]
    assert pooled.n == 6, "pooled should see all 3 tasks x 2 seeds = 6 paired rows"
    assert "3 clusters" in pooled.note, pooled.note
    assert pooled.point == pytest.approx(0.1, abs=1e-9)


def test_pooling_pairs_before_clustering_not_after(mod):
    """T1 is missing from `prompted` at seed 1; T2 is complete at both seeds.

    Pairing FIRST on (task, seed) then clustering survivors by task gives:
        T1 cluster = [1.0]            (only seed 0 paired; seed 1 correctly dropped)
        T2 cluster = [0.5, 0.5]
        cluster means = [1.0, 0.5] -> pooled point = 0.75

    Collapsing to a per-task mean BEFORE pairing would instead average T1's trained
    seed0=1.0/seed1=3.0 to 2.0 and pair it against prompted's lone seed0=0.0, giving a T1
    cluster value of 2.0 and a pooled point of 1.25 -- a different, and wrong, number. This
    test fails under that reordering and passes under the documented one.
    """
    trained = {
        ("T1", 0): 1.0,
        ("T1", 1): 3.0,  # must NOT be pooled with T1's seed-0 diff via borrowing
        ("T2", 0): 1.0,
        ("T2", 1): 1.0,
    }
    prompted = {
        ("T1", 0): 0.0,
        # ("T1", 1) deliberately absent from prompted.
        ("T2", 0): 0.5,
        ("T2", 1): 0.5,
    }

    results = mod.per_seed_and_pooled(trained, prompted, [0, 1], n_boot=200, n_perm=200)

    seed0 = results["0"]
    assert seed0.n == 2, "both tasks pair at seed 0"

    seed1 = results["1"]
    assert seed1.n == 1, "only T2 pairs at seed 1; T1 must be dropped, not borrowed"

    pooled = results["pooled"]
    assert pooled.n == 3, "3 paired (task, seed) rows total: T1|0, T2|0, T2|1"
    assert "2 clusters" in pooled.note, pooled.note
    assert pooled.point == pytest.approx(0.75, abs=1e-9), (
        "pooled point must be mean(1.0, 0.5) = 0.75 over the 2 TASK clusters, not 1.25 -- "
        "1.25 is what you get if seeds are averaged before pairing (the bug this guards)"
    )


def test_render_table_marks_an_interval_that_excludes_zero(mod):
    results = mod.per_seed_and_pooled(
        {("A", 0): 1.0, ("B", 0): 1.0},
        {("A", 0): 0.0, ("B", 0): 0.0},
        [0],
        n_boot=50,
        n_perm=50,
    )
    table = mod.render_table(results, metric="answer_correct")
    assert "excludes 0" in table
    assert "pooled (3 seeds)" in table


# --------------------------------------------------------------------- load_metric (real data)


def test_load_metric_reads_the_existing_cap8_seed0_snapshot(mod):
    """Smoke test against the real, already-promoted cap-8 snapshot: the `trained` and
    `base8b` pins from `artifacts/frames_frontier/FRAMES_FRONTIER.md` at budget_cap=8, seed 0.
    Skips, naming the path, on a checkout that has not built this artifact.
    """
    pytest.importorskip("duckdb")
    snap = REPO / "artifacts" / "frames_frontier" / "scores_parquet.frames"
    if not (snap / "runs.parquet").exists():
        pytest.skip(f"snapshot absent: {snap}")

    trained_pin = "6bc87062cbbdd37802120ffc97d2bcd63a924797b456a99115644dc7aae4f856"
    values = mod.load_metric([str(snap)], cap=8, pin_hash=trained_pin, metric="answer_correct")
    assert len(values) == 824, f"expected 824 trained@cap8@seed0 rows, got {len(values)}"
    assert all(seed == 0 for (_task, seed) in values), "this snapshot is seed-0 only"
    assert set(values.values()) <= {0.0, 1.0}, "answer_correct should be a 0/1 mechanical match"
