"""scripts/granite_family/contrasts.py: the per-seed disambiguation and the bootstrap-stability
recheck the coordinator added 2026-09-18 (any CI bound within 0.01 of zero at 10k resamples is
re-read at 50k under three seeds; a sign that disagrees across those reruns must report
"undecided", never a pass or fail).

`test_train_gate_is_fooled_without_a_model_id_filter` is written FIRST and asserts the FAILURE
mode CONTRIBUTING.md rule 2 asks for: `pinq_train.gate.run_gate` itself, unmodified, pools three
granite seeds sharing one grid_name into a single checkpoint arm when nothing selects a model,
which is exactly why lane L2.5 cannot call it directly for the checkpoint side and needs
`_select_runs(..., model_id=...)` in `contrasts.py` instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "granite_family"))

import contrasts as C  # noqa: E402

from pinq_train.gate import _by_key, _con, _matched_cost, _rows, _select_runs  # noqa: E402
from test_train_gate import _build, _spec  # noqa: E402

GRID = "tier1_trained_qa_base"
CKPT = "inquirer_trained"
BASE = "inquirer_prompted"


# --------------------------------------------------------------------------- stability logic
# Pure Python, no parquet: `bca_ci` is monkeypatched to a fixed lookup table keyed by seed, so
# the recorded per-seed sign is a fact about `stability_check`'s wiring, not about whether real
# bootstrap resampling happens to agree today.


def test_annotate_interval_skips_the_recheck_when_the_interval_is_not_near_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "bca_ci", lambda *a, **k: calls.append(k) or (0.30, 0.20, 0.40))
    cell = C.annotate_interval("t", 0.30, 0.20, 0.40, [0.2, 0.3, 0.4])
    assert cell["near_zero_boundary"] is False
    assert cell["stable"] is True
    assert cell["verdict"] == "positive"
    assert calls == []  # the 50k/three-seed recheck must not fire when there is nothing to check


def test_stability_check_calls_bca_ci_at_50k_under_exactly_the_three_recheck_seeds(monkeypatch):
    seen = []

    def fake(values, *, seed, n_resamples):
        seen.append((seed, n_resamples))
        return (0.01, 0.002, 0.03)

    monkeypatch.setattr(C, "bca_ci", fake)
    C.stability_check([0.01, 0.02])
    assert seen == [(s, C.RECHECK_N_RESAMPLES) for s in C.RECHECK_SEEDS]
    assert C.RECHECK_N_RESAMPLES == 50_000
    assert len(C.RECHECK_SEEDS) == 3


def test_a_lower_bound_that_flips_sign_across_seeds_is_reported_undecided(monkeypatch):
    table = {101: (0.02, -0.001, 0.05), 202: (0.02, 0.003, 0.05), 303: (0.02, 0.004, 0.05)}
    monkeypatch.setattr(C, "bca_ci", lambda values, *, seed, n_resamples: table[seed])
    # primary interval's own lower bound (0.004) is within STABILITY_BAND of zero, so the
    # recheck fires
    cell = C.annotate_interval("t", 0.02, 0.004, 0.05, [0.0])
    assert cell["near_zero_boundary"] is True
    assert cell["stability_recheck"]["lo_sign_stable"] is False
    assert cell["stable"] is False
    assert cell["verdict"] == "undecided", "a flipped-sign bound must never read as pass or fail"


def test_a_lower_bound_that_agrees_across_seeds_keeps_its_verdict(monkeypatch):
    table = {101: (0.02, 0.001, 0.05), 202: (0.02, 0.002, 0.05), 303: (0.02, 0.003, 0.05)}
    monkeypatch.setattr(C, "bca_ci", lambda values, *, seed, n_resamples: table[seed])
    cell = C.annotate_interval("t", 0.02, 0.004, 0.05, [0.0])
    assert cell["stability_recheck"]["lo_sign_stable"] is True
    assert cell["stability_recheck"]["hi_sign_stable"] is True
    assert cell["stable"] is True
    assert cell["verdict"] == "positive"


def test_the_stability_band_is_the_coordinators_0_01(monkeypatch):
    # exactly the documented threshold: this constant is what "within 0.01 of zero" means and a
    # silent change to it would not otherwise be caught by any of the tests above.
    assert C.STABILITY_BAND == 0.01


# --------------------------------------------------------------------------- checkpoint-side
# per-seed disambiguation, over the two-checkpoint-pin fixture shape `test_train_gate.py`
# already builds and hand-checks for the BASELINE side (`test_the_baseline_model_id_is_
# resolved_through_calls_parquet`). This exercises the CHECKPOINT side, which `run_gate` does
# not filter -- three granite seeds on the same grid_name would pool without it.


def _two_pin_specs():
    # Two "granite seeds", same grid_name and arm, disambiguated only by `calls.model` -- the
    # exact shape lane L2.5's three real seeds are in on `tier1_trained_qa_base`.
    return [
        _spec(
            arm=CKPT,
            task="t1",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.8,
            questions=("q1", "q2"),
            done_after="last",
        ),
        _spec(
            arm=CKPT,
            task="t2",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.6,
            questions=("q1",),
            done_after="last",
        ),
        _spec(
            arm=CKPT,
            task="t1",
            grid=GRID,
            model="granite33-8b-sft-headline-s1",
            coverage=0.2,
            questions=("q1", "q2"),
            done_after=None,
        ),
        _spec(
            arm=CKPT,
            task="t2",
            grid=GRID,
            model="granite33-8b-sft-headline-s1",
            coverage=0.1,
            questions=("q1",),
            done_after=None,
        ),
        _spec(
            arm=BASE,
            task="t1",
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.5,
            questions=("q1", "q2", "q3"),
            done_after="last",
        ),
        _spec(
            arm=BASE,
            task="t2",
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.3,
            questions=("q1", "q2"),
            done_after="last",
        ),
    ]


def test_train_gate_is_fooled_without_a_model_id_filter(tmp_path) -> None:
    """THE FAILING CASE FIRST (CONTRIBUTING.md rule 2). `run_gate`'s checkpoint selection is
    `_select_runs(..., model_id=None)`: unmodified, it pools seed s0's and s1's rows on
    `tier1_trained_qa_base` into one 4-run arm instead of two 2-run arms, which is exactly the
    tautological measurement lane L2.5 must not report.
    """
    d = _build(tmp_path, _two_pin_specs())
    con = _con(d)
    pooled = _select_runs(con, arm=CKPT, grids=[GRID], model_id=None)
    assert len(pooled) == 4, "both seeds landed in one arm -- the bug this repo must not ship"


def test_seed_contrast_separates_the_two_pins_by_model_id(tmp_path) -> None:
    d = _build(tmp_path, _two_pin_specs())
    con = _con(d)
    s0 = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s0")
    s1 = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s1")
    assert len(s0) == 2 and len(s1) == 2
    assert {r["task_id"] for r in s0} == {"t1", "t2"}
    assert {r["run_id"] for r in s0}.isdisjoint({r["run_id"] for r in s1})


def test_suite_matched_cost_deltas_reproduces_matched_costs_own_by_suite_mean(tmp_path) -> None:
    """`contrasts.suite_matched_cost_deltas` must average to exactly what `gate._matched_cost`
    (the headline's own function, imported verbatim) reports for that suite -- CONTRIBUTING.md's
    "same functions as the headline" is a testable claim, not a description.
    """
    d = _build(tmp_path, _two_pin_specs())
    con = _con(d)
    ckpt = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s0")
    base = _select_runs(con, arm=BASE, grids=[GRID], model_id="granite33-8b-base")
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)

    mc = _matched_cost(
        con,
        ckpt=ckpt,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash="sc0",
        seed=0,
        n_resamples=200,
    )
    vals = C.suite_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert vals, "the fixture's two tasks must both produce a per-task delta"
    got_mean = sum(vals) / len(vals)
    want_mean = mc["by_suite"]["musique"]["delta"]
    assert got_mean == pytest.approx(want_mean, abs=1e-12)
    assert len(vals) == mc["by_suite"]["musique"]["n_tasks"]


# --------------------------------------------------------------------------- symmetric pairing
# The coordinator's 2026-09-18 defect: the published pairing reads the checkpoint at its own
# terminal coverage and the baseline at min(k_ckpt, k_base) -- matched only while the checkpoint
# does not outspend the baseline. `test_published_reads_the_outspending_checkpoint_at_its_full_k`
# is written FIRST and pins the UNFIXED behaviour so the fix below is provably a change, not a
# restatement.


def _outspend_specs():
    # t1: checkpoint asks 2, baseline asks 1 -- the checkpoint OUTSPENDS the baseline (unsafe).
    # Ladder is deliberately non-monotone-looking in step size so terminal != prefix-at-1.
    return [
        _spec(
            arm=CKPT,
            task="t1",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            questions=("q1", "q2"),
            ladder=(0.0, 0.5, 1.0),
            coverage=1.0,
        ),
        _spec(
            arm=BASE,
            task="t1",
            grid=GRID,
            model="granite33-8b-base",
            questions=("q1",),
            ladder=(0.0, 0.4),
            coverage=0.4,
        ),
    ]


def _build_outspend(tmp_path):
    d = _build(tmp_path, _outspend_specs())
    con = _con(d)
    ckpt = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s0")
    base = _select_runs(con, arm=BASE, grids=[GRID], model_id="granite33-8b-base")
    return con, _by_key(ckpt), _by_key(base)


def test_published_reads_the_outspending_checkpoint_at_its_full_k(tmp_path) -> None:
    """THE UNFIXED BEHAVIOUR, PINNED. Checkpoint at k=2 reads its terminal 1.0, baseline capped
    at min(2,1)=1 reads ITS terminal 0.4 (it only has one rung past zero). Published delta is
    1.0 - 0.4 = 0.6, and the checkpoint's own SHORTER rung (0.5, what it held after one
    question) never enters the computation. That is the mismatch the coordinator reported.
    """
    con, ck_keys, ba_keys = _build_outspend(tmp_path)
    vals = C.suite_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert vals == pytest.approx([0.6])


def test_symmetric_reads_both_sides_at_the_shorter_length(tmp_path) -> None:
    """THE FIX. Both sides read at min(k_ckpt, k_base) = 1: checkpoint's OWN rung at k=1 (0.5,
    a prefix, not its terminal 1.0) minus baseline's rung at k=1 (0.4). Symmetric delta is
    0.5 - 0.4 = 0.1, not 0.6 -- a different number from the same fixture, which is what makes
    this a fix rather than a relabelling.
    """
    con, ck_keys, ba_keys = _build_outspend(tmp_path)
    vals, n_pairs, n_unsafe = C.symmetric_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert vals == pytest.approx([0.1])
    assert n_pairs == 1
    assert n_unsafe == 1, "baseline (n_asks=1) is shorter than the checkpoint's k=2"


def _two_seed_one_task_specs():
    # ONE task, TWO rollout seeds, both arms -- the shape that exposed the bug: `ck_keys`/
    # `ba_keys` are keyed by (suite, task, seed), so a naive loop over `shared` appends once per
    # SEED, not once per TASK.
    return [
        _spec(
            arm=CKPT,
            task="t1",
            seed=0,
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.6,
            questions=("q1",),
            done_after="last",
        ),
        _spec(
            arm=CKPT,
            task="t1",
            seed=1,
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.8,
            questions=("q1",),
            done_after="last",
        ),
        _spec(
            arm=BASE,
            task="t1",
            seed=0,
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.4,
            questions=("q1", "q2"),
            done_after="last",
        ),
        _spec(
            arm=BASE,
            task="t1",
            seed=1,
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.5,
            questions=("q1", "q2"),
            done_after="last",
        ),
    ]


def test_the_deltas_list_is_task_clustered_not_seed_flattened(tmp_path) -> None:
    """THE BUG, PINNED (found 2026-09-19 while building the symmetric-pairing check, not
    reported by the coordinator). One task at two seeds must contribute ONE entry to the
    bootstrap population, the mean of its two seeds, exactly as `_matched_cost`'s own
    `per_task.setdefault((suite, task), []).append(...)` does. A version that appends once per
    (task, seed) key instead would return two entries here, and the point mean happens to be
    identical either way (an average of two seed-level averages equals one flat average when
    both tasks carry the same seed count), so a mean-only test cannot see this. The length can.
    """
    d = _build(tmp_path, _two_seed_one_task_specs())
    con = _con(d)
    ckpt = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s0")
    base = _select_runs(con, arm=BASE, grids=[GRID], model_id="granite33-8b-base")
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)

    vals = C.suite_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert len(vals) == 1, "one task, two seeds, must cluster to ONE bootstrap unit, not two"

    sym_vals, _n_pairs, _n_unsafe = C.symmetric_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert len(sym_vals) == 1, "the symmetric list must be task-clustered the same way"


def test_symmetric_matches_published_when_nothing_outspends(tmp_path) -> None:
    """The safe case (what lane L2.5's real granite cells look like: the trained arm spends far
    less than the prompted base). Published and symmetric must agree exactly, and n_unsafe must
    be 0 -- a fix that changes the SAFE case would itself be a new bug.
    """
    d = _build(tmp_path, _two_pin_specs())
    con = _con(d)
    ckpt = _select_runs(con, arm=CKPT, grids=[GRID], model_id="granite33-8b-sft-headline-s0")
    base = _select_runs(con, arm=BASE, grids=[GRID], model_id="granite33-8b-base")
    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)

    published = C.suite_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    symmetric, n_pairs, n_unsafe = C.symmetric_matched_cost_deltas(
        con, ck_keys=ck_keys, ba_keys=ba_keys, scorer_hash="sc0", suite="musique"
    )
    assert symmetric == pytest.approx(published)
    assert n_unsafe == 0
    assert n_pairs == 2  # two tasks (t1, t2), one baseline run each


# --------------------------------------------------------------------------- non_vacuity
# The coordinator's third check (2026-09-18): a malformed-action rate that zeros an arm's
# coverage must be visible beside the table, not left for a reader to infer from a suspicious
# zero (`search-zeros-must-be-interrogated`), and a run whose `runs.n_turns` claims turns that
# `turns.parquet` does not hold (the exact shape of the compaction defect Lane L0.6 found: 21,001
# runs, 513 test-split) must be flagged rather than silently scored as coverage 0.


def test_non_vacuity_counts_malformed_events_per_ask_run(tmp_path) -> None:
    specs = [
        _spec(
            arm=CKPT,
            task="t1",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.2,
            questions=("q1", "q2"),
            done_after=None,
            n_malformed=2,
        ),
        _spec(
            arm=CKPT,
            task="t2",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.9,
            questions=("q1",),
            done_after="last",
            n_malformed=0,
        ),
        _spec(
            arm=BASE,
            task="t1",
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.5,
            questions=("q1", "q2"),
            done_after="last",
            n_malformed=0,
        ),
    ]
    d = _build(tmp_path, specs)
    con = _con(d)
    run_ids = [r["run_id"] for r in _rows(con, "select run_id from runs")]
    result = C.non_vacuity(con, run_ids)
    assert result["n_runs"] == 3
    assert result["n_runs_that_ask"] == 3
    assert result["n_malformed_events"] == 2
    assert result["malformed_rate_per_ask_run"] == pytest.approx(2 / 3)


def test_non_vacuity_flags_a_run_whose_turn_rows_are_missing(tmp_path) -> None:
    """Reproduces the exact shape of the Lane L0.6 compaction defect: `runs.n_turns` says 2 but
    `turns.parquet` holds 0 rows for that run_id, because the writer dropped them. Built by
    overwriting turns.parquet after `_build`'s normal write, which is otherwise internally
    consistent by construction (one turns row per `questions` entry) and so cannot express this
    case on its own.
    """
    specs = [
        _spec(
            arm=CKPT,
            task="t1",
            grid=GRID,
            model="granite33-8b-sft-headline-s0",
            coverage=0.6,
            questions=("q1", "q2"),
            done_after="last",
        ),
        _spec(
            arm=BASE,
            task="t1",
            grid=GRID,
            model="granite33-8b-base",
            coverage=0.5,
            questions=("q1", "q2"),
            done_after="last",
        ),
    ]
    d = _build(tmp_path, specs)
    turns = pq.read_table(d / "turns.parquet").to_pylist()
    victim = turns[0]["run_id"]
    surviving = [t for t in turns if t["run_id"] != victim]
    pq.write_table(
        pa.Table.from_pylist(surviving, schema=pq.read_table(d / "turns.parquet").schema),
        d / "turns.parquet",
    )

    con = _con(d)
    run_ids = [r["run_id"] for r in _rows(con, "select run_id from runs")]
    result = C.non_vacuity(con, run_ids)
    assert result["n_runs_with_lost_turn_rows"] == 1
    assert victim in result["lost_turn_run_ids_sample"]
