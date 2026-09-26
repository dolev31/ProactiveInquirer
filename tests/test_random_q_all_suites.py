"""`scripts/random_q_all_suites/contrast.py`'s own selection logic, on an in-memory fixture.

WHAT THIS FILE IS FOR. `pinq_train.gate._matched_cost` and `bca_ci` already have their own
tests; this file does not re-test them. It tests the part this lane added: `select()`, which
reads `random_q`/`inquirer_trained`/`inquirer_prompted` rows OUTSIDE `gate._select_runs`'s
grid-name filter (the `random_q` rows this lane reads carry an EMPTY `grid_name`, launched by
explicit `pi run` flags -- a grid-name selector silently returns nothing for them, which is
exactly the class of bug `test_select_ignores_grid_name_and_reads_empty_string` pins).

Each test constructs its own tiny `runs`/`calls` tables with duckdb rather than reading the
shared `scores/parquet/` -- see `tests/test_train_gate.py`'s docstring for why a test must not
read the shared store: it would pass or fail on whatever a concurrent campaign left behind.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")

from scripts.random_q_all_suites.contrast import (  # noqa: E402
    is_near_zero,
    load_exclude_run_ids,
    select,
    sign_stability,
)


def _con(runs_rows, calls_rows=()):
    con = duckdb.connect()
    runs_df = pd.DataFrame(
        runs_rows,
        columns=[
            "run_id",
            "suite_id",
            "task_id",
            "arm_id",
            "seed",
            "split",
            "status",
            "dirty",
            "budget_cap",
            "code_version",
        ],
    )
    con.register("runs", runs_df)
    calls_df = pd.DataFrame(list(calls_rows), columns=["run_id", "actor", "model"])
    con.register("calls", calls_df)
    return con


def _row(
    run_id,
    *,
    task="t1",
    arm="random_q",
    seed=0,
    split="test",
    status="ok",
    dirty=False,
    cap=8,
    cv="cv1",
):
    return (run_id, "musique", task, arm, seed, split, status, dirty, cap, cv)


def test_select_filters_on_every_column_a_wrong_default_would_silently_pool():
    con = _con(
        [
            _row("keep", cv="cv1"),
            _row("wrong_cv", cv="cv2"),
            _row("wrong_split", split="train"),
            _row("wrong_status", status="error"),
            _row("dirty", dirty=True),
            _row("wrong_cap", cap=4),
            _row("wrong_seed", seed=7),
        ]
    )
    rows = select(con, arm_id="random_q", suite_id="musique", code_version="cv1")
    assert [r["run_id"] for r in rows] == ["keep"]


def test_select_ignores_grid_name_and_reads_empty_string():
    """The rows this lane reads have `grid_name=""` (explicit `pi run` flags, no `--sweep`).
    `select()` never mentions `grid_name` at all, unlike `gate._select_runs`, so this passing
    is the point: there is no column here for an empty grid name to be filtered out by."""
    con = _con([_row("keep", cv="cv1")])
    rows = select(con, arm_id="random_q", suite_id="musique", code_version="cv1")
    assert [r["run_id"] for r in rows] == ["keep"]


def test_select_excludes_the_named_run_ids():
    con = _con([_row("keep", cv="cv1"), _row("zero_ask", task="t2", cv="cv1")])
    rows = select(
        con,
        arm_id="random_q",
        suite_id="musique",
        code_version="cv1",
        exclude_run_ids=frozenset({"zero_ask"}),
    )
    assert [r["run_id"] for r in rows] == ["keep"]


def test_select_model_id_disambiguates_a_pooled_arm():
    """`inquirer_trained` can carry more than one checkpoint pin under one arm_id (measured:
    musique pools `qwen3-8b-sft-headline` and `qwen3-8b-dpo-stacked-notdone-both` at the same
    commit). `model_id` must read only the named pin's rows, via `calls`, `actor='inquirer'`."""
    con = _con(
        [
            _row("sft", arm="inquirer_trained", cv="cv1"),
            _row("dpo", arm="inquirer_trained", task="t2", cv="cv1"),
        ],
        calls_rows=[
            ("sft", "inquirer", "qwen3-8b-sft-headline"),
            ("dpo", "inquirer", "qwen3-8b-dpo-stacked-notdone-both"),
        ],
    )
    rows = select(
        con,
        arm_id="inquirer_trained",
        suite_id="musique",
        code_version="cv1",
        model_id="qwen3-8b-sft-headline",
    )
    assert [r["run_id"] for r in rows] == ["sft"]


def test_select_restricts_to_the_requested_seeds():
    con = _con(
        [
            _row("s0", seed=0, cv="cv1"),
            _row("s1", seed=1, task="t2", cv="cv1"),
            _row("s2", seed=2, task="t3", cv="cv1"),
        ]
    )
    rows = select(con, arm_id="random_q", suite_id="musique", code_version="cv1", seeds=(0, 1))
    assert {r["run_id"] for r in rows} == {"s0", "s1"}


def test_load_exclude_run_ids_reads_one_id_per_line(tmp_path):
    p = tmp_path / "exclude.txt"
    p.write_text("abc\ndef\n\nghi\n")
    assert load_exclude_run_ids(p) == frozenset({"abc", "def", "ghi"})


def test_load_exclude_run_ids_missing_file_is_empty_not_an_error(tmp_path):
    assert load_exclude_run_ids(tmp_path / "does_not_exist.txt") == frozenset()


def test_trained_model_by_suite_uses_the_headline_checkpoint_on_every_suite():
    """`qwen3-8b-dpo-stacked-notdone-both` is the `inquirer_trained` pin this lane's brief
    names for ALL THREE suites -- the same checkpoint `artifacts/testsplit_qa/TESTSPLIT_QA.md`
    reads against `qwen3-8b-base` with n=200 musique test tasks, n=200 strategyqa, n=166 wiki2.
    `qwen3-8b-sft-headline` is a REAL checkpoint with its own musique-only `inquirer_trained`
    rows (`TESTSPLIT_QA.md`'s two-pin warning: musique pools 482 sft-headline + 400
    dpo-stacked-notdone-both rows under one arm_id), but it is a DIFFERENT checkpoint from the
    one this lane's brief names, and substituting it for musique alone would silently contrast
    two different treatments under one row label."""
    from scripts.random_q_all_suites.contrast import TRAINED_MODEL_BY_SUITE

    assert TRAINED_MODEL_BY_SUITE == {
        "musique": "qwen3-8b-dpo-stacked-notdone-both",
        "strategyqa": "qwen3-8b-dpo-stacked-notdone-both",
        "wiki2": "qwen3-8b-dpo-stacked-notdone-both",
    }


# --------------------------------------------------------------------- near-zero robustness


def test_is_near_zero_true_when_the_ci_contains_zero():
    assert is_near_zero({"ci_lo": -0.01, "ci_hi": 0.02}) is True


def test_is_near_zero_false_when_the_ci_excludes_zero():
    assert is_near_zero({"ci_lo": 0.01, "ci_hi": 0.02}) is False
    assert is_near_zero({"ci_lo": -0.02, "ci_hi": -0.01}) is False


def test_is_near_zero_false_when_ci_is_missing_not_a_crash():
    assert is_near_zero({"ci_lo": None, "ci_hi": None}) is False


def test_sign_stability_agrees_when_every_seed_has_the_same_sign_and_verdict():
    by_seed = [
        {"seed": 0, "delta": 0.04, "ci_lo": 0.01, "ci_hi": 0.07},
        {"seed": 1, "delta": 0.05, "ci_lo": 0.02, "ci_hi": 0.08},
        {"seed": 2, "delta": 0.03, "ci_lo": 0.01, "ci_hi": 0.06},
    ]
    out = sign_stability(by_seed)
    assert out == {"sign_stable": True, "zero_exclusion_stable": True, "n_seeds": 3}


def test_sign_stability_catches_a_sign_flip_across_seeds():
    """The whole point of the 50k/three-seed re-check: one seed reading positive and another
    negative on a small effect must not be reported as one number that happens to be seed 0's."""
    by_seed = [
        {"seed": 0, "delta": 0.004, "ci_lo": -0.01, "ci_hi": 0.02},
        {"seed": 1, "delta": -0.002, "ci_lo": -0.015, "ci_hi": 0.01},
        {"seed": 2, "delta": 0.001, "ci_lo": -0.012, "ci_hi": 0.015},
    ]
    out = sign_stability(by_seed)
    assert out["sign_stable"] is False
    assert out["n_seeds"] == 3


def test_sign_stability_catches_a_zero_exclusion_flip_even_with_a_stable_sign():
    by_seed = [
        {"seed": 0, "delta": 0.015, "ci_lo": 0.001, "ci_hi": 0.03},
        {"seed": 1, "delta": 0.012, "ci_lo": -0.004, "ci_hi": 0.028},
        {"seed": 2, "delta": 0.017, "ci_lo": 0.002, "ci_hi": 0.032},
    ]
    out = sign_stability(by_seed)
    assert out["sign_stable"] is True
    assert out["zero_exclusion_stable"] is False
