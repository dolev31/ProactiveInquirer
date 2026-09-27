"""scripts/decomposition_test/contrast.py, on fixtures small enough to hand-check.

Two things are tested, both because a wrong answer here would look exactly like a right one:

1. `select_and_contrast` isolates ONE checkpoint model's rows before pairing. Lane L2.3
   measured that `pinq_train.gate.run_gate` cannot do this against a SHARED `scores/parquet`
   store (its checkpoint-side `_select_runs` call is hard-coded `model_id=None`): pointed at a
   store where several trained checkpoints share one `grid_name`/`arm_id`, it pools every
   checkpoint sharing a (suite, task, seed) key into one "checkpoint" side, and `_matched_cost`
   silently averages across models that were never meant to be compared as one arm. The fixture
   below gives two checkpoint models opposite, hand-picked deltas against the same baseline
   (+0.3 and -0.1); the unfiltered pooling this module exists to avoid would report their
   midpoint, 0.1 -- neither model's own number. `test_the_pooling_bug_this_module_fixes` pins
   that midpoint as the wrong answer an unfiltered selection produces, and
   `test_select_and_contrast_isolates_the_named_checkpoint` pins the right one.

2. `stability_recheck` reports a cell as `undecided` exactly when a near-zero bound's sign is
   not the same across three reseeds at 50k resamples (the coordinator's rule added to this
   lane mid-task, mirroring appendix_training.tex's own "seven distinct cells carry a gated
   lower bound within 0.01 of zero ... four ... agree ... three do not"). Tested by patching
   `select_and_contrast` itself, so the assertion is about the sign-agreement LOGIC and does
   not depend on any real bootstrap RNG actually landing on an unstable draw.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from scripts.decomposition_test import contrast as dt

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCORER = "sc0"
GRAPH = "v1"
GRID = "tier1_trained_qa_base"
CKPT_ARM = "inquirer_trained"
BASE_ARM = "inquirer_prompted"
BASE_MODEL = "base_x"


def _run_row(run_id, *, suite, task, seed, arm, model, n_asks):
    return dict(
        run_id=run_id,
        suite_id=suite,
        task_id=task,
        arm_id=arm,
        seed=seed,
        split="test",
        grid_name=GRID,
        status="ok",
        stop_reason="policy_stop",
        n_asks=n_asks,
    )


def _ladder_rows(run_id, ladder):
    return [
        dict(
            run_id=run_id,
            metric_name=f"frontier_q#{k}",
            scorer_hash=SCORER,
            value=float(v),
            graph_version=GRAPH,
        )
        for k, v in enumerate(ladder)
    ]


def _build(tmp_path, *, checkpoint_rows, baseline_rows):
    """checkpoint_rows: list of (run_id, model, task, seed, n_asks, evidence_coverage).
    baseline_rows: list of (run_id, model, task, seed, ladder) -- ladder[k] = frontier_q#k,
    ladder[-1] is treated as the baseline's own evidence_coverage (the scorer's own invariant,
    see `_coverage_ladder`'s docstring: frontier_q#n_asks == evidence_coverage).
    """
    runs, scores, calls = [], [], []
    for run_id, model, task, seed, n_asks, cov in checkpoint_rows:
        runs.append(
            _run_row(
                run_id,
                suite="musique",
                task=task,
                seed=seed,
                arm=CKPT_ARM,
                model=model,
                n_asks=n_asks,
            )
        )
        scores.append(
            dict(
                run_id=run_id,
                metric_name="evidence_coverage",
                scorer_hash=SCORER,
                value=float(cov),
                graph_version=GRAPH,
            )
        )
        # A checkpoint run needs no ladder for _matched_cost (only its OWN evidence_coverage,
        # `_check_ladder_is_the_coverage_column` only walks the BASELINE side's ladders), but
        # `_with_stop` reads every run in `[*ckpt, *base]` off `runs` alone, which is already
        # covered by `_run_row`.
        calls.append(dict(run_id=run_id, actor="inquirer", model=model))
    for run_id, model, task, seed, ladder in baseline_rows:
        n_asks = len(ladder) - 1
        runs.append(
            _run_row(
                run_id,
                suite="musique",
                task=task,
                seed=seed,
                arm=BASE_ARM,
                model=model,
                n_asks=n_asks,
            )
        )
        scores += _ladder_rows(run_id, ladder)
        scores.append(
            dict(
                run_id=run_id,
                metric_name="evidence_coverage",
                scorer_hash=SCORER,
                value=float(ladder[-1]),
                graph_version=GRAPH,
            )
        )
        calls.append(dict(run_id=run_id, actor="inquirer", model=model))

    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs", runs), ("scores", scores), ("calls", calls)):
        pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")
    # turns/ledger are read by _con only to confirm they EXIST -- _matched_cost and
    # _select_runs never query them (see module docstring). An empty `Table.from_pylist([])`
    # has zero columns, and duckdb's read_parquet refuses a file with "at least one non-root
    # column" missing, so these need an explicit (unused) schema instead of a truly empty one.
    empty = pa.table({"run_id": pa.array([], type=pa.string())})
    for name in ("turns", "ledger"):
        pq.write_table(empty, d / f"{name}.parquet")
    return d


# Four tasks, two checkpoint models (`ckpt_a`, `ckpt_b`) sharing every (suite, task, seed) key,
# one baseline model. Both checkpoint models ask k=1; the baseline's frontier_q#1 = 0.4 on
# every task, so every task's delta is exactly (checkpoint's own coverage) - 0.4.
# ckpt_a's coverage is 0.7 on every task -> delta +0.3 on every task -> mean +0.3.
# ckpt_b's coverage is 0.3 on every task -> delta -0.1 on every task -> mean -0.1.
# Pooled (unfiltered) per task: mean(+0.3, -0.1) = +0.1 -- neither model's own number.
TASKS = ["t1", "t2", "t3", "t4"]


def _fixture(tmp_path):
    checkpoint_rows = []
    baseline_rows = []
    for t in TASKS:
        checkpoint_rows.append((f"a-{t}", "ckpt_a", t, 0, 1, 0.7))
        checkpoint_rows.append((f"b-{t}", "ckpt_b", t, 0, 1, 0.3))
        baseline_rows.append((f"base-{t}", BASE_MODEL, t, 0, [0.0, 0.4]))
    return _build(tmp_path, checkpoint_rows=checkpoint_rows, baseline_rows=baseline_rows)


def test_select_and_contrast_isolates_the_named_checkpoint(tmp_path) -> None:
    d = _fixture(tmp_path)
    res_a = dt.select_and_contrast(
        d,
        checkpoint_model_id="ckpt_a",
        baseline_model_id=BASE_MODEL,
        scorer_hash=SCORER,
        n_resamples=200,
    )
    assert res_a["selection"]["n_checkpoint_runs"] == len(TASKS)
    assert res_a["by_suite"]["musique"]["delta"] == pytest.approx(0.3)

    res_b = dt.select_and_contrast(
        d,
        checkpoint_model_id="ckpt_b",
        baseline_model_id=BASE_MODEL,
        scorer_hash=SCORER,
        n_resamples=200,
    )
    assert res_b["selection"]["n_checkpoint_runs"] == len(TASKS)
    assert res_b["by_suite"]["musique"]["delta"] == pytest.approx(-0.1)


def test_the_pooling_bug_this_module_fixes(tmp_path) -> None:
    """The unfiltered selection `run_gate` would perform against a SHARED store: both
    checkpoint models pooled into one "checkpoint" side. This pins the WRONG number
    (+0.1, the midpoint) so a future change that drops the checkpoint-side model filter shows
    up as this test passing when it should fail -- i.e. this is the failing-without-the-fix
    case CONTRIBUTING.md rule 2 asks for, expressed as "the old code's answer is provably not either
    model's own number".
    """
    from pinq_train.gate import _by_key, _con, _matched_cost, _select_runs

    d = _fixture(tmp_path)
    con = _con(d)
    # The exact call run_gate makes for the checkpoint side: model_id=None.
    ckpt_unfiltered = _select_runs(con, arm=CKPT_ARM, grids=[GRID], model_id=None)
    base = _select_runs(con, arm=BASE_ARM, grids=[GRID], model_id=BASE_MODEL)
    assert len(ckpt_unfiltered) == 2 * len(TASKS)  # both ckpt_a and ckpt_b, pooled

    ck_keys, ba_keys = _by_key(ckpt_unfiltered), _by_key(base)
    pooled = _matched_cost(
        con,
        ckpt=ckpt_unfiltered,
        base=base,
        ck_keys=ck_keys,
        ba_keys=ba_keys,
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    pooled_delta = pooled["by_suite"]["musique"]["delta"]
    assert pooled_delta == pytest.approx(0.1)
    assert pooled_delta != pytest.approx(0.3)
    assert pooled_delta != pytest.approx(-0.1)


# --------------------------------------------------------------------------------------------
# stability_recheck: the sign-agreement logic, tested by patching select_and_contrast so the
# assertion is about this module's decision rule and not about any real RNG draw.


def _stub(reads_by_seed):
    def _fake(
        parquet_dir,
        *,
        checkpoint_model_id,
        baseline_model_id,
        scorer_hash,
        seed,
        n_resamples=10_000,
        **kw,
    ):
        lo, hi = reads_by_seed[seed]
        return {"by_suite": {"musique": {"ci_lo": lo, "ci_hi": hi, "delta": (lo + hi) / 2}}}

    return _fake


def test_stability_recheck_skips_a_bound_that_is_not_near_zero(tmp_path, monkeypatch) -> None:
    calls = {"n": 0}

    def _counting_fake(*a, **kw):
        calls["n"] += 1
        raise AssertionError("should not be called: no bound is near zero")

    monkeypatch.setattr(dt, "select_and_contrast", _counting_fake)
    by_suite = {"musique": {"ci_lo": 0.05, "ci_hi": 0.20}}
    out = dt.stability_recheck(
        tmp_path,
        checkpoint_model_id="x",
        baseline_model_id="y",
        by_suite=by_suite,
        scorer_hash=SCORER,
    )
    assert out["musique"] == {"checked": False, "stable": True, "verdict": "as computed"}
    assert calls["n"] == 0


def test_stability_recheck_reports_undecided_when_the_sign_flips_across_reseeds(
    monkeypatch,
) -> None:
    # 10k reading: ci_lo = +0.002, within NEAR_ZERO_TOL (0.01) of zero.
    by_suite = {"musique": {"ci_lo": 0.002, "ci_hi": 0.25}}
    reads = {0: (0.001, 0.24), 1: (-0.003, 0.26), 2: (0.004, 0.25)}  # sign flips: +, -, +
    monkeypatch.setattr(dt, "select_and_contrast", _stub(reads))
    out = dt.stability_recheck(
        "unused",
        checkpoint_model_id="x",
        baseline_model_id="y",
        by_suite=by_suite,
        scorer_hash=SCORER,
    )
    cell = out["musique"]
    assert cell["checked"] is True
    assert cell["near_zero_bound"] == "ci_lo"
    assert cell["signs"] == [1, -1, 1]
    assert cell["stable"] is False
    assert cell["verdict"] == "undecided"


def test_stability_recheck_reports_stable_when_the_sign_agrees_across_reseeds(monkeypatch) -> None:
    by_suite = {"musique": {"ci_lo": 0.003, "ci_hi": 0.25}}
    reads = {0: (0.001, 0.24), 1: (0.002, 0.26), 2: (0.0005, 0.25)}  # all >= 0
    monkeypatch.setattr(dt, "select_and_contrast", _stub(reads))
    out = dt.stability_recheck(
        "unused",
        checkpoint_model_id="x",
        baseline_model_id="y",
        by_suite=by_suite,
        scorer_hash=SCORER,
    )
    cell = out["musique"]
    assert cell["stable"] is True
    assert cell["verdict"] == "as computed"


def test_arm_map_names_exactly_the_six_paper_rows() -> None:
    assert set(dt.ARM_MAP) == {
        "Reference",
        "Stop-weight",
        "Persistence-only",
        "Mixed",
        "Stacked",
        "Question-preference",
    }
    candidates = {name for name, v in dt.ARM_MAP.items() if v["is_candidate"]}
    assert candidates == {"Stop-weight", "Persistence-only", "Mixed", "Stacked"}


# --------------------------------------------------------------------------------------------
# assert_non_vacuous: the coordinator's 2026-09-18 rule, catching exactly the shape of the
# lost-rows compaction defect (a real turns.jsonl on disk next to a zero-stamped parquet row,
# or asks that score no evidence anywhere).


def _vacuity_fixture(tmp_path, *, n_turns_row, turns_file_nonempty, coverage_values):
    """One run, `n_asks=1` always (it "asks"). `n_turns_row` is what the parquet claims;
    `turns_file_nonempty` controls whether a real turns.jsonl sits next to it; `coverage_values`
    is the evidence_coverage rows to write for run 'x' plus any EXTRA runs (for the
    all-zero-among-many-asking case).
    """
    runs, scores, calls = [], [], []
    for rid, cov in coverage_values.items():
        runs.append(
            dict(
                run_id=rid,
                suite_id="musique",
                task_id=rid,
                arm_id=CKPT_ARM,
                seed=0,
                split="test",
                grid_name=GRID,
                status="ok",
                stop_reason="policy_stop",
                n_asks=1,
                n_turns=(n_turns_row if rid == "x" else 1),
            )
        )
        scores.append(
            dict(
                run_id=rid,
                metric_name="evidence_coverage",
                scorer_hash=SCORER,
                value=float(cov),
                graph_version=GRAPH,
            )
        )
        calls.append(dict(run_id=rid, actor="inquirer", model="m"))

    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs", runs), ("scores", scores), ("calls", calls)):
        pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")
    empty = pa.table({"run_id": pa.array([], type=pa.string())})
    for name in ("turns", "ledger"):
        pq.write_table(empty, d / f"{name}.parquet")

    runs_root = tmp_path / "runs"
    for rid in coverage_values:
        (runs_root / rid).mkdir(parents=True)
        if rid == "x" and turns_file_nonempty:
            (runs_root / rid / "turns.jsonl").write_text('{"turn_idx": 0}\n')
        elif rid != "x":
            (runs_root / rid / "turns.jsonl").write_text('{"turn_idx": 0}\n')
    return d, runs_root


def test_non_vacuity_passes_on_an_ordinary_population(tmp_path) -> None:
    d, runs_root = _vacuity_fixture(
        tmp_path, n_turns_row=1, turns_file_nonempty=True, coverage_values={"x": 0.5, "y": 0.7}
    )
    report = dt.assert_non_vacuous(d, runs_root, ["x", "y"], scorer_hash=SCORER)
    assert report["n_zero_turns_but_nonempty_file"] == 0
    assert report["n_nonzero_coverage_among_asking"] == 2


def test_non_vacuity_catches_a_real_turns_file_beside_a_zero_stamped_row(tmp_path) -> None:
    """The lost-rows compaction defect, exactly: turns.jsonl is real on disk, but the parquet
    row this lane would read says n_turns=0.
    """
    d, runs_root = _vacuity_fixture(
        tmp_path, n_turns_row=0, turns_file_nonempty=True, coverage_values={"x": 0.5}
    )
    with pytest.raises(dt.Vacuous, match="lost-rows compaction defect"):
        dt.assert_non_vacuous(d, runs_root, ["x"], scorer_hash=SCORER)


def test_non_vacuity_tolerates_a_zero_turns_row_when_the_file_is_genuinely_empty(tmp_path) -> None:
    """n_turns=0 is not itself a defect -- only n_turns=0 NEXT TO A REAL FILE is. A run whose
    turns.jsonl is genuinely empty (e.g. it errored before turn 0) must not be flagged.
    """
    d, runs_root = _vacuity_fixture(
        tmp_path, n_turns_row=0, turns_file_nonempty=False, coverage_values={"x": 0.5}
    )
    report = dt.assert_non_vacuous(d, runs_root, ["x"], scorer_hash=SCORER)
    assert report["n_zero_turns_but_nonempty_file"] == 0


def test_non_vacuity_catches_zero_coverage_across_every_asking_run(tmp_path) -> None:
    d, runs_root = _vacuity_fixture(
        tmp_path,
        n_turns_row=1,
        turns_file_nonempty=True,
        coverage_values={"x": 0.0, "y": 0.0, "z": 0.0},
    )
    with pytest.raises(dt.Vacuous, match="NONE has nonzero"):
        dt.assert_non_vacuous(d, runs_root, ["x", "y", "z"], scorer_hash=SCORER)


# --------------------------------------------------------------------------------------------
# build_farm: pure filesystem logic (scripts/decomposition_test/build_farm.py).

from scripts.decomposition_test.build_farm import build_farm, read_run_id_list  # noqa: E402


def test_build_farm_links_exactly_the_named_ids_and_reports_missing_ones(tmp_path) -> None:
    source = tmp_path / "runs"
    for rid in ("r1", "r2"):
        (source / rid).mkdir(parents=True)
    farm = tmp_path / "farm" / "runs"

    report = build_farm(farm, ["r1", "r2", "r-missing"], source_runs_root=source)

    assert sorted(report["linked"]) == ["r1", "r2"]
    assert report["missing"] == ["r-missing"]
    assert (farm / "r1").is_symlink()
    assert (farm / "r1").resolve() == (source / "r1").resolve()
    assert not (farm / "r-missing").exists()


def test_build_farm_with_a_relative_source_root_still_resolves(tmp_path, monkeypatch) -> None:
    """Measured 2026-09-18: called with `source_runs_root=Path("runs")` from a caller whose
    cwd was the repo root, every symlink's target TEXT was the literal string "runs/<id>" --
    correct relative to that cwd, wrong relative to the symlink's own directory, which is what
    a relative symlink actually resolves against. `pi compact` then walked the farm and found
    zero run directories, with no error. This pins the fix: a RELATIVE `source_runs_root`
    (the exact shape of the original call) must still produce a farm that resolves.
    """
    (tmp_path / "actual_runs" / "r1").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    farm = tmp_path / "elsewhere" / "farm" / "runs"

    report = build_farm(farm, ["r1"], source_runs_root=Path("actual_runs"))

    assert report["linked"] == ["r1"]
    target = os.readlink(farm / "r1")
    assert os.path.isabs(target), f"symlink target must be absolute, got {target!r}"
    assert (farm / "r1" / ".").exists(), "the symlink must actually resolve to something"


def test_read_run_id_list_skips_blanks_and_comments(tmp_path) -> None:
    p = tmp_path / "ids.txt"
    p.write_text("r1\n\n# a comment\nr2\n")
    assert read_run_id_list(p) == ["r1", "r2"]


# --------------------------------------------------------------------------------------------
# select_and_contrast_symmetric: both arms read at min(k_a, k_b), never one arm's raw terminal
# `evidence_coverage` against the other's truncated rung (the asymmetry Lane L2.3 found reaching
# a published cell: Persistence-only's own k, 6.4-6.9, sits at or above the untrained baseline's
# ~6, so `select_and_contrast` -- which always keeps the checkpoint's own full terminal value --
# is not a matched reading on the majority of its StrategyQA task-pairs).


def _ladder_fixture(tmp_path, *, a_ladder, b_ladder, model_a="arm_a", model_b="arm_b"):
    """One task, one seed. `a_ladder`/`b_ladder` are each arm's full frontier_q#k series;
    both arms are `inquirer_trained` so either can play checkpoint or baseline.
    """
    runs, scores, calls = [], [], []
    for run_id, model, ladder in (("a", model_a, a_ladder), ("b", model_b, b_ladder)):
        n_asks = len(ladder) - 1
        runs.append(
            _run_row(
                run_id, suite="musique", task="t1", seed=0, arm=CKPT_ARM, model=model, n_asks=n_asks
            )
        )
        scores += _ladder_rows(run_id, ladder)
        scores.append(
            dict(
                run_id=run_id,
                metric_name="evidence_coverage",
                scorer_hash=SCORER,
                value=float(ladder[-1]),
                graph_version=GRAPH,
            )
        )
        calls.append(dict(run_id=run_id, actor="inquirer", model=model))

    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs", runs), ("scores", scores), ("calls", calls)):
        pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")
    empty = pa.table({"run_id": pa.array([], type=pa.string())})
    for name in ("turns", "ledger"):
        pq.write_table(empty, d / f"{name}.parquet")
    return d


def test_symmetric_contrast_truncates_the_higher_asking_arm_too(tmp_path) -> None:
    """arm_a asks 6 (ladder to 0.8), arm_b asks 2 (ladder to 0.5) -- exactly
    tests/test_train_gate.py::test_the_matched_cost_rule_does_not_flag_a_checkpoint_that_
    outspends_its_baseline's fixture. The unmatched (`select_and_contrast`-style) reading would
    be arm_a's raw terminal 0.8 minus arm_b's truncated 0.5 = +0.3; the symmetric reading must
    read BOTH at min(6, 2) = 2: arm_a's rung there (0.4) minus arm_b's (0.5) = -0.1.
    """
    d = _ladder_fixture(
        tmp_path,
        a_ladder=[0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8],
        b_ladder=[0.0, 0.3, 0.5],
    )
    res = dt.select_and_contrast_symmetric(
        d, arm_a_model_id="arm_a", arm_b_model_id="arm_b", scorer_hash=SCORER, n_resamples=200
    )
    cell = res["by_suite"]["musique"]
    assert cell["delta"] == pytest.approx(-0.1)
    assert cell["mean_k_common"] == pytest.approx(2.0)

    # Antisymmetry: swapping which arm is "a" negates the delta exactly, unlike
    # select_and_contrast (checkpoint/baseline are not interchangeable there).
    swapped = dt.select_and_contrast_symmetric(
        d, arm_a_model_id="arm_b", arm_b_model_id="arm_a", scorer_hash=SCORER, n_resamples=200
    )
    assert swapped["by_suite"]["musique"]["delta"] == pytest.approx(0.1)
