"""The online STOP 2x2, counted over DECISION POINTS rather than over runs.

WHY THIS FILE EXISTS. The gate's DONE axis used to be `scores.stop_undershoot`, which
`pi_eval.score` defines as `max(0, k* - k_hat)` with `k*` the argmax of the coverage-by-prefix
ladder and `k_hat = len(turns)`. Coverage is MONOTONE in k -- the seen-uid set only grows -- so
the argmax is always attained at or before the last prefix and `k* <= k_hat` identically. The
metric is therefore 0 on every run that has ever been scored (1,395/1,395 in each of
`artifacts/gate/parquet_{sft,headline,headline_stop}/`), the "not done" cell was empty by
construction, `p_ask_given_not_done` was NaN, and the criterion could only ever fail with
"one cell has no runs". A 2x2 with an empty column is not a measurement.

WHAT REPLACES IT mirrors the offline Tier A definition (`pinq_train.eval_offline.stop_confusion`
over the export's `done_before`, i.e. `pi_run.cmd_train._done` of the coverage held BEFORE the
action is chosen). One run contributes n_asks+1 decision points t = 0..n_asks:

  * `done_before(t)` is `frontier_q#t >= 1 - 1e-12`. `frontier_q#k` is required-evidence
    coverage after k asks are complete, which is exactly the coverage the policy holds when it
    decides at t = k. It is read from the parquet and never fabricated: on all three gate
    parquets `frontier_q#0` is 0.0 on 1,395/1,395 runs and `frontier_q#n_asks` equals
    `evidence_coverage` to 0.0, so the ladder is the same quantity at both ends.
  * the action at t < n_asks is ASK. The action at t = n_asks is STOP only when the episode
    ended by `stop_reason = 'policy_stop'`; under `budget` / `max_turns` the harness halted the
    episode and the policy was never asked, so that state carries no decision and is EXCLUDED
    AND COUNTED as `n_forced_stops`. Counting it as an ASK would credit the cap with the
    policy's judgement -- which is what the old code did to every capped run.

Every fixture here is built with pyarrow in `tmp_path` for the reason `test_train_gate` gives:
the shared `scores/parquet` is rewritten by `pi compact`, so a test that reads it asserts
whatever last night's campaign happened to hold.
"""

from __future__ import annotations

import math

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCORER = "sc0"
GRAPH = "v1"
CKPT = "inquirer_trained"
BASE = "inquirer_prompted"
DONE = 1.0


# --------------------------------------------------------------------------- fixture builder


def _run(
    *,
    arm=CKPT,
    task="t0",
    seed=0,
    suite="musique",
    cov=(0.0,),
    stop_reason="policy_stop",
    status="ok",
    grid=None,
    undershoot=0.0,
    questions=None,
) -> dict:
    """One episode. `cov[k]` is required-evidence coverage AFTER k asks, so `len(cov) - 1` is
    `n_asks` and `cov` alone fixes both the length of the trajectory and the done axis.

    `undershoot` defaults to 0.0 because that is what the real scorer emits on every run: a
    fixture that set it to 1.0 to make the second cell non-empty would be testing a value the
    pipeline cannot produce.
    """
    n = len(cov) - 1
    return dict(
        arm=arm,
        task=task,
        seed=seed,
        suite=suite,
        cov=tuple(None if c is None else float(c) for c in cov),
        stop_reason=stop_reason,
        status=status,
        grid=grid or ("dev_select_musique" if arm == CKPT else "dev_baseline_musique"),
        undershoot=undershoot,
        questions=questions
        if questions is not None
        else tuple(f"{arm[:2]} {task} question number {i}" for i in range(n)),
    )


# An empty list gives pyarrow no columns to infer, and duckdb refuses a column-less parquet
# ("Need at least one non-root column"). A run with n_asks = 0 contributes no turn rows, which
# is a legitimate episode -- the policy stopped on the empty state -- so the table is written
# against an explicit schema rather than by inference.
_SCHEMAS = {
    "turns": pa.schema(
        [
            ("run_id", pa.string()),
            ("turn_idx", pa.int64()),
            ("action_kind", pa.string()),
            ("question", pa.string()),
        ]
    ),
}


def _build(tmp_path, specs):
    runs, turns, scores, ledger, calls = [], [], [], [], []
    for i, s in enumerate(specs):
        rid = f"r{i}"
        n_asks = len(s["cov"]) - 1
        assert len(s["questions"]) == n_asks, (
            f"{rid}: {len(s['questions'])} questions but a ladder of {n_asks} asks. The "
            "fixture would describe a run the loop cannot produce."
        )
        runs.append(
            dict(
                run_id=rid,
                suite_id=s["suite"],
                task_id=s["task"],
                arm_id=s["arm"],
                seed=s["seed"],
                split="dev",
                template_id="",
                grid_name=s["grid"],
                status=s["status"],
                stop_reason=s["stop_reason"],
                n_asks=n_asks,
                n_turns=n_asks,
                exploratory=True,
            )
        )
        for j, q in enumerate(s["questions"]):
            turns.append(dict(run_id=rid, turn_idx=j, action_kind="ask", question=q))
        for k, q in enumerate(s["cov"]):
            if q is None:  # the scorer OMITS the point when coverage is undefined
                continue
            scores.append(
                dict(run_id=rid, metric_name=f"frontier_q#{k}", scorer_hash=SCORER, value=float(q))
            )
        for name, val in (
            ("evidence_coverage", s["cov"][-1]),
            ("facet_breadth", 0.5),
            ("stop_undershoot", s["undershoot"]),
        ):
            if val is not None:
                scores.append(
                    dict(run_id=rid, metric_name=name, scorer_hash=SCORER, value=float(val))
                )
        scores.append(dict(run_id=rid, metric_name="cad#2", scorer_hash=SCORER, value=0.5))
        scores.append(dict(run_id=rid, metric_name="cad_n#2", scorer_hash=SCORER, value=4.0))
        ledger.append(dict(run_id=rid, row_idx=0, currency="llm_calls", charged=1.0))
        calls.append(dict(run_id=rid, actor="inquirer", model="qwen3-8b-sft"))

    # `pi_eval.schema.SCORES` makes `graph_version` a column of every scores table the real
    # scorer writes (64/64 gate parquets carry it, single-valued). `run_gate` reads it to stamp
    # the provenance triple's third element into the verdict, so a fixture without it describes
    # a scores table the project cannot produce.
    for row in scores:
        row.setdefault("graph_version", GRAPH)

    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("runs", runs),
        ("turns", turns),
        ("scores", scores),
        ("ledger", ledger),
        ("calls", calls),
    ):
        table = (
            pa.Table.from_pylist(rows, schema=_SCHEMAS[name])
            if name in _SCHEMAS
            else pa.Table.from_pylist(rows)
        )
        pq.write_table(table, d / f"{name}.parquet")
    return d


def _cells(tmp_path, specs, **kw):
    """`criteria.stop_2x2` for a fixture. Only the checkpoint side is asserted on unless a
    test reaches for `baseline` itself."""
    from pinq_train.gate import run_gate

    args = dict(
        parquet_dir=_build(tmp_path, specs),
        grid_name="dev_select_musique",
        baseline_grid_names=("dev_baseline_musique",),
        checkpoint_arm=CKPT,
        baseline_arm=BASE,
        baseline_model_id=None,
        scorer_hash=SCORER,
        bootstrap_seed=0,
        n_resamples=50,
        grids_root=None,
    )
    args.update(kw)
    return run_gate(**args)["criteria"]["stop_2x2"]


def _twin(spec: dict) -> dict:
    """The baseline run of the same (suite, task, seed). A 2x2 with no baseline cannot be
    compared, and the criterion's pass rule is a comparison."""
    return dict(spec, arm=BASE, grid="dev_baseline_musique")


def _pair(**kw) -> list[dict]:
    ck = _run(**kw)
    ba = _run(**{**kw, "arm": BASE})
    return [ck, ba]


# --------------------------------------------------------------------------- the not-done cell


def test_a_policy_stop_before_coverage_is_complete_is_a_not_done_stop(tmp_path) -> None:
    """THE CELL THAT WAS EMPTY. A run that halts by its own choice while required evidence is
    still missing is the STOP-too-early failure the second cell exists to catch, and under
    `stop_undershoot` it was filed as done: the metric is 0 on every run ever scored.

    One checkpoint run, ladder (0.0, 0.4, 0.6), policy_stop. Decision points:
        t=0 coverage 0.0 -> not done, ASK   (correct)
        t=1 coverage 0.4 -> not done, ASK   (correct)
        t=2 coverage 0.6 -> not done, STOP  (wrong)
    so P(ASK | not done) = 2/3 and the done column is empty for this run.
    """
    specs = [_run(cov=(0.0, 0.4, 0.6)), _run(arm=BASE, cov=(0.0, 0.4, 0.6))]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_not_done"] == 3
    assert c["value"]["n_not_done_ask"] == 2
    assert c["value"]["n_not_done_stop"] == 1
    assert c["value"]["p_ask_given_not_done"] == pytest.approx(2 / 3)
    assert not math.isnan(c["value"]["p_ask_given_not_done"])


def test_the_2x2_does_not_read_stop_undershoot_at_all(tmp_path) -> None:
    """NON-TAUTOLOGY. `stop_undershoot` is identically 0 -- `k*` is the argmax of a monotone
    ladder and `k_hat` is its last index -- so a 2x2 that still has both columns when the
    metric is absent entirely is a 2x2 that is not reading it."""
    specs = [
        _run(task="t0", cov=(0.0, DONE), undershoot=None),
        _run(task="t0", arm=BASE, cov=(0.0, DONE), undershoot=None),
        _run(task="t1", cov=(0.0, 0.5), undershoot=None),
        _run(task="t1", arm=BASE, cov=(0.0, 0.5), undershoot=None),
    ]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_done"] > 0 and c["value"]["n_not_done"] > 0


# --------------------------------------------------------------------------- the done cell


def test_asks_taken_after_coverage_is_complete_are_done_ask_states(tmp_path) -> None:
    """A run that kept asking after it already held every required span. Ladder
    (0.0, 1.0, 1.0, 1.0), policy_stop:
        t=0 not done, ASK ; t=1 done, ASK ; t=2 done, ASK ; t=3 done, STOP
    so P(STOP | done) = 1/3 and the run carries 2 asks after done. Under the old axis the whole
    run was one 'done' unit that stopped, i.e. 1.0 -- the two wasted questions were invisible.
    """
    specs = [_run(cov=(0.0, DONE, DONE, DONE)), _run(arm=BASE, cov=(0.0, DONE, DONE, DONE))]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_done"] == 3
    assert c["value"]["n_done_stop"] == 1
    assert c["value"]["n_done_ask"] == 2
    assert c["value"]["p_stop_given_done"] == pytest.approx(1 / 3)
    assert c["value"]["mean_asks_after_done"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- forced stops


def test_a_cap_forced_final_state_is_excluded_and_counted(tmp_path) -> None:
    """The harness halted the episode; the policy was never consulted at t = n_asks. Scoring
    that state as an ASK credits the retrieval cap with the policy's judgement.

    Ladder (0.0, 0.5, 0.7) under `budget`: t=0 and t=1 are real ASK decisions, t=2 is not a
    decision at all -- 2 states, not 3, and one forced stop on the record.
    """
    specs = [_run(cov=(0.0, 0.5, 0.7), stop_reason="budget"), _run(arm=BASE, cov=(0.0, 0.5, 0.7))]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_forced_stops"] == 1
    assert c["value"]["n_states"] == 2
    assert c["value"]["n_not_done"] == 2 and c["value"]["n_not_done_ask"] == 2
    assert c["value"]["p_ask_given_not_done"] == 1.0
    assert c["value"]["n_runs"] == 1


def test_max_turns_is_forced_too(tmp_path) -> None:
    """`pinq.loop` writes 'max_turns' when the for-loop runs out and 'budget' when the
    retrieval charge is refused. Neither is the policy choosing to halt."""
    specs = [
        _run(cov=(0.0, 0.5), stop_reason="max_turns"),
        _run(arm=BASE, cov=(0.0, 0.5), stop_reason="max_turns"),
    ]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_forced_stops"] == 1 and c["value"]["n_states"] == 1


# --------------------------------------------------------------------------- calibration


def test_a_perfect_stopper_scores_one_on_both_cells(tmp_path) -> None:
    """The calibration point, mirroring Tier A's `test_a_perfect_stopper_scores_one_on_both_...`.
    If this is not 1.0/1.0 the metric cannot be read at all.

    Each run asks while not done and stops on the turn that completes coverage: ladder
    (0.0, ..., 1.0) with `policy_stop`, so every t < n_asks is (not done, ASK) and t = n_asks is
    the only (done, STOP).
    """
    specs = []
    for t in range(4):
        specs += _pair(task=f"t{t}", cov=(0.0, 0.5, DONE))
    c = _cells(tmp_path, specs)
    assert c["value"]["p_stop_given_done"] == 1.0
    assert c["value"]["p_ask_given_not_done"] == 1.0
    assert c["value"]["n_done"] == 4 and c["value"]["n_not_done"] == 8
    assert c["value"]["mean_asks_after_done"] == 0.0


def test_an_always_stop_at_t0_policy_scores_one_and_zero(tmp_path) -> None:
    """The failure I.8 names, in the online form: a checkpoint buys P(STOP|done) by halting on
    the empty state, and the second cell is the only thing that says so. n_asks = 0 and
    `policy_stop`, so each run is exactly one decision point at t = 0.

    Coverage before any ask is 0 unless a task's required evidence was already in hand, so the
    single done run here is the seeded case; the three not-done runs stopped with everything
    still missing.
    """
    specs = [
        _run(task="t0", cov=(DONE,)),
        _run(task="t1", cov=(0.0,)),
        _run(task="t2", cov=(0.0,)),
        _run(task="t3", cov=(0.0,)),
    ]
    specs += [_twin(s) for s in list(specs)]
    c = _cells(tmp_path, specs)
    assert c["value"]["p_stop_given_done"] == 1.0
    assert c["value"]["p_ask_given_not_done"] == 0.0
    assert c["value"]["n_done"] == 1 and c["value"]["n_not_done"] == 3
    assert c["value"]["n_states"] == 4 and c["value"]["n_runs"] == 4


# --------------------------------------------------------------------------- the pass rule


def test_the_criterion_still_fails_when_one_cell_is_bought_with_the_other(tmp_path) -> None:
    """The pass rule is unchanged: BOTH cells no worse than the baseline. The checkpoint below
    stops the moment it is done (P(STOP|done) 1.0 vs the baseline's 1/2) and also stops while
    not done (P(ASK|not done) 1/2 vs the baseline's 1.0), which is the trade the second cell
    exists to expose.

    UNDER `cap8`, which is where that rule now lives. The 2026-09-15 follow-up to D9 makes the
    criterion REPORT-ONLY under `matched_cost`: the real prompted base almost never stops
    (P(STOP | done) 0.0066-0.1525 on the six gate parquets), so its P(ASK | not done) is ~1 by
    construction and "both cells >= the base" is unreachable by any policy that stops at all.
    The cells are identical under both rules and the second half of this test says so.
    """
    specs = []
    for t in range(4):
        # the checkpoint: stops the instant it is done (1/1), and also stops while not done
        specs.append(_run(task=f"t{t}", cov=(0.0, DONE)))
        specs.append(_run(task=f"t{t}", seed=1, cov=(0.0, 0.5)))
        # the baseline: asks one question past done (1/2), and never stops while not done
        specs.append(_run(task=f"t{t}", arm=BASE, cov=(0.0, DONE, DONE)))
        specs.append(_run(task=f"t{t}", arm=BASE, seed=1, cov=(0.0, 0.5), stop_reason="budget"))
    c = _cells(tmp_path, specs, coverage_rule="cap8")
    assert c["value"]["p_stop_given_done"] == 1.0
    assert c["baseline"]["p_stop_given_done"] == pytest.approx(0.5)
    assert c["value"]["p_ask_given_not_done"] == pytest.approx(2 / 3)
    assert c["baseline"]["p_ask_given_not_done"] == 1.0
    assert c["gated"] is True and c["passed"] is False

    m = _cells(tmp_path / "mc", specs, coverage_rule="matched_cost")
    assert m["gated"] is False and m["passed"] is True
    assert m["both_cells_ge_baseline"] is False, "the cap-8 verdict, kept rather than dropped"
    assert m["value"] == c["value"] and m["baseline"] == c["baseline"], "every cell survives"


# --------------------------------------------------------------------------- provenance


def test_the_verdict_records_the_done_axis_definition(tmp_path) -> None:
    """A reader must be able to tell which quantity was thresholded, at which point in the
    episode, without re-deriving it. The old string named a metric that is identically 0."""
    specs = _pair(cov=(0.0, DONE))
    c = _cells(tmp_path, specs)
    assert c["columns"]["done_axis"] == "frontier_q#t >= 1-1e-12 before the decision at t"
    assert c["columns"]["unit"] == "one decision point (run, t), t = 0..n_asks"
    assert "policy_stop" in c["columns"]["stop_means"]
    assert "n_forced_stops" in c["columns"]["stop_means"]
    assert c["columns"]["filter"] == "runs.status='ok'"


def test_a_state_with_no_coverage_row_is_skipped_rather_than_called_not_done(tmp_path) -> None:
    """`pi_eval.score` OMITS `frontier_q#k` when the task carries no required gold evidence --
    coverage is undefined there, not zero. Reading the absence as not-done would move every
    gold-free run into the second cell's denominator, which is exactly the fabrication the
    scorer refused to make."""
    specs = [
        _run(task="t0", cov=(None, None), questions=("who wrote it",)),
        _run(task="t0", arm=BASE, cov=(0.0, 0.5), questions=("who wrote it",)),
    ]
    c = _cells(tmp_path, specs)
    assert c["value"]["n_done"] == 0 and c["value"]["n_not_done"] == 0
    assert c["value"]["n_skipped_no_coverage"] == 2
