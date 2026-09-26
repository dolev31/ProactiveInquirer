"""Lane L1.5 tooling (`scripts/stopping_answer_test/lib.py`).

Hermetic: every fixture here is synthetic, built in-memory, never `artifacts/testsplit_qa` --
that directory is untracked (git status `??`) and a worktree does not carry untracked files
(see the repo's own `worktree-agents-autoclean-and-pythonpath` lesson), so a test that opened it
would pass on the machine that built it and fail, silently for the wrong reason, everywhere
else including CI.

CLAUDE.md rule 2: the aggregation-consistency tests below (`test_pooled_...`,
`test_matched_cost_coverage_by_task_...`) exist because `run.py` trusts two things that are not
obviously true until checked -- that summing per-task stop-2x2 cells reproduces the pooled
arm-level cells, and that this module's `matched_cost_coverage_by_task` (which exposes an
intermediate value `pinq_train.gate._matched_cost` computes but does not return) reproduces
that function's own pooled delta. Both are written to fail if the grouping key or the pooling
step is wrong, not just if the file fails to import.
"""

from __future__ import annotations

import math

import duckdb
import pytest
from scripts.stopping_answer_test import lib

# --------------------------------------------------------------------------- bca_paired_ratio_delta


def test_bca_paired_ratio_delta_point_estimate_is_the_pooled_ratio():
    # 2 tasks. Trained: 3/4 done-stops on task A, 1/2 on task B -> pooled 4/6.
    # Prompted: 1/4 on A, 0/2 on B -> pooled 1/6. Delta = 4/6 - 1/6 = 0.5 exactly.
    rows = [
        ("A", 3.0, 4.0, 1.0, 4.0),
        ("B", 1.0, 2.0, 0.0, 2.0),
    ]
    out = lib.bca_paired_ratio_delta(rows, n_boot=200, seed=0)
    assert out["point"] == pytest.approx(0.5)
    assert out["n"] == 2
    assert out["lo"] <= out["point"] <= out["hi"]


def test_bca_paired_ratio_delta_is_order_independent():
    rows = [
        ("A", 3.0, 4.0, 1.0, 4.0),
        ("B", 1.0, 2.0, 0.0, 2.0),
        ("C", 5.0, 5.0, 2.0, 5.0),
        ("D", 0.0, 3.0, 0.0, 3.0),
    ]
    forward = lib.bca_paired_ratio_delta(rows, n_boot=500, seed=0)
    backward = lib.bca_paired_ratio_delta(list(reversed(rows)), n_boot=500, seed=0)
    assert forward["point"] == backward["point"]
    assert forward["lo"] == backward["lo"]
    assert forward["hi"] == backward["hi"]


def test_bca_paired_ratio_delta_positive_control_excludes_zero():
    # Non-vacuity (memory: "a permutation test needs a non-vacuity check" applies here too):
    # trained stops at every done-state on every task, prompted never does. A real, large,
    # unambiguous effect the interval MUST catch, or the instrument cannot be trusted at all.
    rows = [(f"t{i}", 10.0, 10.0, 0.0, 10.0) for i in range(30)]
    out = lib.bca_paired_ratio_delta(rows, n_boot=2000, seed=0)
    assert out["lo"] > 0, f"positive control failed to exclude zero: {out}"


def test_bca_paired_ratio_delta_null_case_is_exactly_zero():
    # Identical cells in both arms on every task: the delta is 0 on every conceivable resample
    # (any subset of identical pairs still nets to zero), so the interval collapses to a point
    # rather than merely spanning zero -- the strongest null a deterministic fixture can assert.
    rows = [(f"t{i}", 5.0, 8.0, 5.0, 8.0) for i in range(10)]
    out = lib.bca_paired_ratio_delta(rows, n_boot=500, seed=0)
    assert out["point"] == 0.0
    assert out["lo"] == pytest.approx(0.0, abs=1e-12)
    assert out["hi"] == pytest.approx(0.0, abs=1e-12)


def test_bca_paired_ratio_delta_skips_tasks_with_zero_denominator_in_either_arm():
    # A task where the trained arm never reaches "done" contributes (0, 0, ...) to the pooled
    # sums -- it must not be silently coerced to a per-task rate of 0/0 -> nan -> dropped, which
    # would change n; it should simply add nothing to trained's sums while still counting once.
    rows = [
        ("A", 3.0, 4.0, 1.0, 4.0),
        ("B", 0.0, 0.0, 2.0, 4.0),  # trained never done on B
    ]
    out = lib.bca_paired_ratio_delta(rows, n_boot=200, seed=0)
    assert out["n"] == 2
    assert out["point"] == pytest.approx(3.0 / 4.0 - 3.0 / 8.0)


def test_bca_paired_ratio_delta_empty_is_absent():
    out = lib.bca_paired_ratio_delta([], n_boot=100, seed=0)
    assert math.isnan(out["point"])
    assert math.isnan(out["lo"])
    assert math.isnan(out["hi"])
    assert out["n"] == 0


# --------------------------------------------------------------------------- bca_two_sample_mean_delta


def test_bca_two_sample_mean_delta_point_estimate():
    a = [0.8, 0.9, 1.0, 0.7]
    b = [0.2, 0.3, 0.1, 0.4]
    out = lib.bca_two_sample_mean_delta(a, b, n_boot=500, seed=0)
    assert out["point"] == pytest.approx(sum(a) / len(a) - sum(b) / len(b))
    assert out["n_a"] == 4
    assert out["n_b"] == 4


def test_bca_two_sample_mean_delta_swap_negates_the_point_estimate():
    # The POINT estimate (mean(a)-mean(b)) negates under a swap with no randomness involved,
    # so that equality is exact. The INTERVAL is not: `fwd` and `bwd` independently resample
    # group_a before group_b from the same seeded RNG, so the two calls consume the same
    # per-call draw SEQUENCE against DIFFERENT data (group_a=b's first n_a draws are not the
    # mirror of group_a=a's), and BCa's bias correction is computed from each call's own
    # replicate distribution -- the two are estimating mirror-image populations, not
    # generating sample-path-identical bootstraps, so exact endpoint negation is not a
    # property this function has (or should be made to have by special-casing the draw order).
    # A large, unambiguous effect is what the swap SHOULD do to the sign of the whole interval.
    a = [0.9, 0.95, 0.85, 1.0, 0.8, 0.92, 0.88]
    b = [0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.18]
    fwd = lib.bca_two_sample_mean_delta(a, b, n_boot=4000, seed=0)
    bwd = lib.bca_two_sample_mean_delta(b, a, n_boot=4000, seed=0)
    assert fwd["point"] == pytest.approx(-bwd["point"])
    assert fwd["lo"] > 0 and fwd["hi"] > 0, f"fwd should be entirely positive: {fwd}"
    assert bwd["lo"] < 0 and bwd["hi"] < 0, f"bwd should be entirely negative: {bwd}"


def test_bca_two_sample_mean_delta_positive_control_excludes_zero():
    a = [1.0] * 20
    b = [0.0] * 20
    out = lib.bca_two_sample_mean_delta(a, b, n_boot=2000, seed=0)
    assert out["lo"] > 0, f"positive control failed to exclude zero: {out}"


# --------------------------------------------------------------------------- with_stability


def test_with_stability_skips_the_50k_path_when_bounds_are_far_from_zero():
    calls = []

    def compute(n_boot, seed):
        calls.append((n_boot, seed))
        return {
            "point": 0.3,
            "lo": 0.2,
            "hi": 0.4,
            "n": 10,
            "n_boot": n_boot,
            "n_reps_used": n_boot,
        }

    out = lib.with_stability(compute, seed=0)
    assert calls == [(lib.DEFAULT_N_BOOT, 0)]
    assert out["stability_checked"] is False
    assert out["verdict"] == "excludes_zero"


def test_with_stability_flags_undecided_on_an_unstable_near_zero_bound():
    # 10k reads lo just inside the 0.01 band; the three 50k seeds disagree on lo's sign.
    fifty_k_los = iter([-0.002, 0.001, -0.0005])

    def compute(n_boot, seed):
        if n_boot == lib.DEFAULT_N_BOOT:
            return {
                "point": 0.05,
                "lo": -0.004,
                "hi": 0.09,
                "n": 50,
                "n_boot": n_boot,
                "n_reps_used": n_boot,
            }
        return {
            "point": 0.05,
            "lo": next(fifty_k_los),
            "hi": 0.09,
            "n": 50,
            "n_boot": n_boot,
            "n_reps_used": n_boot,
        }

    out = lib.with_stability(compute, seed=0)
    assert out["stability_checked"] is True
    assert out["stable"] is False
    assert out["verdict"] == "undecided"


def test_with_stability_confirms_a_stable_near_zero_bound():
    def compute(n_boot, seed):
        # lo is consistently (barely) positive at every seed and every resample count.
        lo = 0.003 if n_boot == lib.DEFAULT_N_BOOT else 0.001 + 0.0001 * seed
        return {
            "point": 0.05,
            "lo": lo,
            "hi": 0.09,
            "n": 50,
            "n_boot": n_boot,
            "n_reps_used": n_boot,
        }

    out = lib.with_stability(compute, seed=0)
    assert out["stability_checked"] is True
    assert out["stable"] is True
    assert out["verdict"] == "excludes_zero"
    assert out["n_boot_reported"] == lib.STABILITY_N_BOOT


def test_with_stability_reports_absent_on_nan_bounds():
    def compute(n_boot, seed):
        return {
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n": 0,
            "n_boot": n_boot,
            "n_reps_used": 0,
        }

    out = lib.with_stability(compute, seed=0)
    assert out["verdict"] == "absent"
    assert out["stability_checked"] is False


# --------------------------------------------------------------------------- duckdb fixtures: stop 2x2


SCORER = "testscorer"

# run_id -> (n_asks, stop_reason, {k: frontier_q#k}, evidence_coverage)
# Task A: trained reaches DONE (>=1-1e-12) at k=1 and holds it -- s0 has one ask-at-done then a
# stop-at-done, s1 has two asks-at-done then a stop-at-done. prompted plateaus at 0.5 on A,
# never done. Task B: neither arm ever reaches done. Every number below is walked through in
# the tests that use it, by hand, before it is trusted.
_FIXTURE_RUNS: dict[str, dict] = {
    "t-A-s0": (2, "policy_stop", {0: 0.0, 1: 1.0, 2: 1.0}, 1.0),
    "t-A-s1": (3, "policy_stop", {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}, 1.0),
    "p-A-s0": (2, "policy_stop", {0: 0.0, 1: 0.3, 2: 0.5}, 0.5),
    "p-A-s1": (2, "policy_stop", {0: 0.0, 1: 0.3, 2: 0.5}, 0.5),
    "t-B-s0": (1, "policy_stop", {0: 0.0, 1: 0.0}, 0.0),
    "p-B-s0": (1, "policy_stop", {0: 0.0, 1: 0.0}, 0.0),
}
_TASK_OF = {
    "t-A-s0": "A",
    "t-A-s1": "A",
    "p-A-s0": "A",
    "p-A-s1": "A",
    "t-B-s0": "B",
    "p-B-s0": "B",
}
_SEED_OF = {"t-A-s0": 0, "t-A-s1": 1, "p-A-s0": 0, "p-A-s1": 1, "t-B-s0": 0, "p-B-s0": 0}


def _synthetic_store():
    """An in-memory duckdb connection carrying a minimal `scores` table and a `runs` table
    (run_id, n_asks, stop_reason -- the columns `_coverage_ladder` and `_with_stop`, which
    `pinq_train.gate._matched_cost` calls internally, actually select) built from
    `_FIXTURE_RUNS`."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE scores (run_id VARCHAR, metric_name VARCHAR, scorer_hash VARCHAR, "
        "value DOUBLE)"
    )
    con.execute("CREATE TABLE runs (run_id VARCHAR, n_asks INTEGER, stop_reason VARCHAR)")
    score_rows = []
    run_rows = []
    for run_id, (n_asks, stop_reason, ladder, coverage) in _FIXTURE_RUNS.items():
        run_rows.append((run_id, n_asks, stop_reason))
        for k, q in ladder.items():
            score_rows.append((run_id, f"frontier_q#{k}", SCORER, q))
        score_rows.append((run_id, "evidence_coverage", SCORER, coverage))
    con.executemany("INSERT INTO scores VALUES (?,?,?,?)", score_rows)
    con.executemany("INSERT INTO runs VALUES (?,?,?)", run_rows)
    return con


def _fixture_run_dicts(run_ids):
    return [
        {
            "run_id": rid,
            "task_id": _TASK_OF[rid],
            "seed": _SEED_OF[rid],
            "n_asks": _FIXTURE_RUNS[rid][0],
            "stop_reason": _FIXTURE_RUNS[rid][1],
        }
        for rid in run_ids
    ]


def test_pooled_stop_cells_matches_hand_computation_trained():
    con = _synthetic_store()
    runs = _fixture_run_dicts(["t-A-s0", "t-A-s1", "t-B-s0"])
    cells = lib.pooled_stop_cells(con, runs, scorer_hash=SCORER)
    # Walked by hand in the fixture docstring: A contributes n_done=5 (2 ask-at-done seed0's
    # 1 + seed1's 2, plus 2 stop-at-done), n_not_done=2; B contributes n_done=0, n_not_done=2.
    assert cells["n_done"] == 5
    assert cells["n_done_stop"] == 2
    assert cells["n_not_done"] == 4
    assert cells["n_not_done_ask"] == 3
    assert cells["p_stop_given_done"] == pytest.approx(2 / 5)
    assert cells["p_ask_given_not_done"] == pytest.approx(3 / 4)


def test_pooled_stop_cells_matches_hand_computation_prompted():
    con = _synthetic_store()
    runs = _fixture_run_dicts(["p-A-s0", "p-A-s1", "p-B-s0"])
    cells = lib.pooled_stop_cells(con, runs, scorer_hash=SCORER)
    assert cells["n_done"] == 0
    assert math.isnan(cells["p_stop_given_done"])
    assert cells["n_not_done"] == 8
    assert cells["n_not_done_ask"] == 5
    assert cells["p_ask_given_not_done"] == pytest.approx(5 / 8)


def test_per_task_stop_cells_sum_to_the_pooled_cells():
    con = _synthetic_store()
    trained = _fixture_run_dicts(["t-A-s0", "t-A-s1", "t-B-s0"])
    pooled = lib.pooled_stop_cells(con, trained, scorer_hash=SCORER)
    per_task = lib.per_task_stop_cells(con, trained, scorer_hash=SCORER)
    assert set(per_task) == {"A", "B"}
    for field in ("n_done", "n_done_stop", "n_not_done", "n_not_done_ask", "n_states"):
        assert sum(c[field] for c in per_task.values()) == pooled[field], field


def test_matched_cost_coverage_by_task_pooled_mean_matches_gate_matched_cost():
    """Cross-check against `pinq_train.gate._matched_cost` itself, on the same fixture: this
    module's `matched_cost_coverage_by_task` must reproduce, task for task, the `by_task`
    dict that function computes internally and does not return."""
    from pinq_train.gate import _matched_cost

    con = _synthetic_store()
    ck_runs = _fixture_run_dicts(["t-A-s0", "t-A-s1", "t-B-s0"])
    ba_runs = _fixture_run_dicts(["p-A-s0", "p-A-s1", "p-B-s0"])
    for r in [*ck_runs, *ba_runs]:
        r["suite_id"] = "musique"

    by_task = lib.matched_cost_coverage_by_task(
        con, ck_runs=ck_runs, ba_runs=ba_runs, scorer_hash=SCORER
    )
    ck_keys = lib._by_key(ck_runs)
    ba_keys = lib._by_key(ba_runs)
    official = _matched_cost(
        con,
        ckpt=ck_runs,
        base=ba_runs,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    assert official["pooled"]["n_tasks"] == len(by_task) == 2
    pooled_mean = sum(by_task.values()) / len(by_task)
    assert pooled_mean == pytest.approx(official["pooled"]["delta"])
    # Task A: trained terminal coverage 1.0 (both seeds) minus baseline's OWN prefix at the
    # trained's k (min(k, baseline n_asks)): s0 k=2 -> baseline prefix at min(2,2)=2 -> 0.5;
    # s1 k=3 -> baseline prefix at min(3,2)=2 (baseline s1 only ran 2 asks) -> 0.5. Averaged
    # over the 2 baseline runs sharing each key, then over the 2 seeds: delta = 1.0 - 0.5 = 0.5.
    assert by_task[("musique", "A")] == pytest.approx(0.5)
    # Task B: trained terminal 0.0 minus baseline prefix at min(1,1)=1 -> 0.0. delta = 0.0.
    assert by_task[("musique", "B")] == pytest.approx(0.0)


def test_verify_turns_not_dropped_catches_the_compaction_bug(tmp_path):
    # good: turns.jsonl has 2 lines, parquet agrees (n_turns=2)
    (tmp_path / "run-good").mkdir()
    (tmp_path / "run-good" / "turns.jsonl").write_text('{"a":1}\n{"a":2}\n')
    # bad: turns.jsonl is intact (non-empty) but the parquet says n_turns=0 -- the bug shape
    (tmp_path / "run-dropped").mkdir()
    (tmp_path / "run-dropped" / "turns.jsonl").write_text('{"a":1}\n')
    # empty run: genuinely asked nothing, turns.jsonl is empty, parquet n_turns=0 -- not a bug
    (tmp_path / "run-empty").mkdir()
    (tmp_path / "run-empty" / "turns.jsonl").write_text("")

    report = lib.verify_turns_not_dropped(
        tmp_path, {"run-good": 2, "run-dropped": 0, "run-empty": 0, "run-no-dir": 3}
    )
    assert report["n_checked"] == 3
    assert report["bad_run_ids"] == ["run-dropped"]
    assert report["missing_dir"] == ["run-no-dir"]


def test_verify_ladder_exists_for_asking_runs():
    con = _synthetic_store()
    runs = _fixture_run_dicts(["t-A-s0", "t-B-s0"])
    # both fixture runs have n_asks > 0 and a real ladder -- must pass clean.
    report = lib.verify_ladder_exists_for_asking_runs(con, runs, scorer_hash=SCORER)
    assert report["n_bad"] == 0

    # a run claiming n_asks > 0 but absent from `scores` entirely -- the bug signature.
    dropped = [
        {"run_id": "ghost", "task_id": "A", "seed": 0, "n_asks": 5, "stop_reason": "policy_stop"}
    ]
    report = lib.verify_ladder_exists_for_asking_runs(con, dropped, scorer_hash=SCORER)
    assert report["n_bad"] == 1
    assert report["bad_run_ids"] == ["ghost"]


def test_task_level_metric_averages_seeds_into_the_task():
    con = _synthetic_store()
    runs = _fixture_run_dicts(["t-A-s0", "t-A-s1", "t-B-s0"])
    for r in runs:
        r["suite_id"] = "musique"
    by_task = lib.task_level_metric(con, runs, "evidence_coverage", scorer_hash=SCORER)
    # task A: seed0=1.0, seed1=1.0 -> mean 1.0; task B: single seed, 0.0.
    assert by_task[("musique", "A")] == pytest.approx(1.0)
    assert by_task[("musique", "B")] == pytest.approx(0.0)


def test_population_report_flags_missing_scores():
    con = _synthetic_store()
    report = lib.population_report(
        con, run_ids=["t-A-s0", "t-A-s1", "does-not-exist"], scorer_hash=SCORER
    )
    assert report["n_population"] == 3
    assert report["missing_from_runs"] == ["does-not-exist"]
    assert report["missing_scores"] == ["does-not-exist"]
    assert report["n_scored"] == 2
