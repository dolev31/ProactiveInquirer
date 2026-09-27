"""The online dev gate, on parquet fixtures small enough to hand-check.

WHY FIXTURES AND NOT THE REAL TABLES. `scores/parquet/*.parquet` is shared and is rewritten by
`pi compact`; a test that reads it asserts whatever last night's campaign happened to contain,
and passes or fails for reasons that have nothing to do with the gate. Every table below is
built with pyarrow in `tmp_path`, so each criterion's pass case AND its fail case are
constructed rather than hoped for.

THE POINT OF THE FAIL CASES. A gate that has only ever been seen to pass is not known to be a
gate. Each criterion here is exercised in both directions, and `test_a_missing_criterion_fails`
pins the third direction: a criterion whose inputs are absent must FAIL, never pass. A gate
that passes for want of data is the exact silent bug this repo exists not to have.
"""

from __future__ import annotations

import json
import math
import random

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SCORER = "sc0"
# The `graph_version` column every real `scores.parquet` carries beside `scorer_hash`.
# Measured on artifacts/gate/8b2-t20/parquet_qwen3-8b-dpo-headline-rater/scores.parquet:
# one distinct scorer_hash, one distinct graph_version ("v1"), 64/64 gate parquets single-valued.
GRAPH = "v1"
CKPT = "inquirer_trained"
BASE = "inquirer_prompted"


# --------------------------------------------------------------------------- fixture builder


def _spec(
    *,
    arm,
    task,
    seed=0,
    suite="musique",
    grid=None,
    stop_reason="policy_stop",
    done_after=None,
    coverage=0.5,
    cad=((2, 0.5, 4.0),),
    facet=0.5,
    newly=0.5,
    questions=("who wrote it",),
    n_malformed=0,
    status="ok",
    model=None,
    ladder=None,
    # ---- fork grids only (T16b). None on every existing caller, so the runs.parquet these
    # produce carries the columns with value NULL everywhere -- `_select_fork_columns` treats
    # that exactly like a grid that never had a fork run: `_fork_criteria` returns None and
    # `criteria` is untouched. `foreign_prefix_k` is accepted for a test's own bookkeeping only
    # -- gate.py never reads it off runs.parquet (see `_fork_key`) -- and is not written by
    # `_build`.
    foreign_trace_sha=None,
    foreign_prefix_k=None,
    n_user_turns=None,
    n_prefix_user_turns=None,
    tau_reward=None,
) -> dict:
    return dict(
        arm=arm,
        task=task,
        seed=seed,
        suite=suite,
        grid=grid or ("dev_select_musique" if arm == CKPT else "dev_baseline_musique"),
        stop_reason=stop_reason,
        done_after=done_after,
        # `frontier_q#0..n_asks` verbatim, for the tests that need a ladder that RISES rather
        # than the step `done_after` describes: the matched-cost rule reads the baseline's
        # prefix off exactly these rungs, so a step ladder would make every rung below the top
        # 0.0 and the comparator trivial.
        ladder=None if ladder is None else tuple(float(x) for x in ladder),
        coverage=coverage,
        cad=cad,
        facet=facet,
        newly=newly,
        questions=questions,
        n_malformed=n_malformed,
        status=status,
        model=model or ("qwen3-8b-sft" if arm == CKPT else "openai/aws/gpt-oss-120b"),
        foreign_trace_sha=foreign_trace_sha,
        foreign_prefix_k=foreign_prefix_k,
        n_user_turns=n_user_turns,
        n_prefix_user_turns=n_prefix_user_turns,
        tau_reward=tau_reward,
    )


def _ladder(spec) -> list[float]:
    """Required-evidence coverage after k asks, k = 0..n_asks -- what `scores.frontier_q#k`
    holds, what the STOP 2x2's done axis thresholds, and what the matched-cost coverage rule
    reads the baseline's prefix off.

    `done_after` is the first k at which coverage is complete; "last" means the final ask is
    what completed it (the perfect-stopper shape) and None means the run never got there.
    Derived from the question list rather than written out, so a test that overrides
    `questions` cannot silently leave a ladder of the wrong length behind.

    THE LAST RUNG IS `evidence_coverage`, NOT A SECOND INDEPENDENT NUMBER. `pi_eval.score`
    builds the ladder by merging each turn's retrieved uids in order, so `frontier_q#n_asks`
    and `evidence_coverage` are one quantity read twice -- they agree on 1,395/1,395 runs of
    each gate parquet, and `tests/test_gate_stop_2x2.py` has always written its fixtures that
    way (`evidence_coverage = cov[-1]`). A fixture whose ladder ended somewhere else describes
    an instrument that does not exist, and `--coverage-rule matched_cost` refuses exactly that
    parquet, so such a fixture would exercise the refusal rather than the rule. A run whose
    evidence is COMPLETE therefore carries coverage 1.0: that is what "done" means.
    """
    n = len(spec["questions"])
    at = spec["done_after"]
    at = n if at == "last" else at
    rungs = [0.0 if at is None or k < at else 1.0 for k in range(n + 1)]
    if spec["coverage"] is not None:
        rungs[-1] = float(spec["coverage"])
    return rungs


def _build(tmp_path, specs):
    runs, turns, scores, ledger, calls, native = [], [], [], [], [], []
    for i, s in enumerate(specs):
        rid = f"r{i}"
        n_asks = len(s["questions"])
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
                # NULL on every fixture that never sets it (see `_spec`), so a grid built
                # without fork columns in mind carries the same "not a fork" signal a real
                # pre-migration parquet dir would.
                foreign_trace_sha=s.get("foreign_trace_sha"),
                n_prefix_user_turns=s.get("n_prefix_user_turns"),
                n_user_turns=s.get("n_user_turns"),
            )
        )
        if s.get("tau_reward") is not None:
            native.append(
                dict(
                    run_id=rid, suite_id=s["suite"], key="tau_reward", value=float(s["tau_reward"])
                )
            )
        for j, q in enumerate(s["questions"]):
            turns.append(dict(run_id=rid, turn_idx=j, action_kind="ask", question=q))
        for k, q in enumerate(s["ladder"] if s["ladder"] is not None else _ladder(s)):
            scores.append(
                dict(run_id=rid, metric_name=f"frontier_q#{k}", scorer_hash=SCORER, value=float(q))
            )
        for name, val in (
            ("evidence_coverage", s["coverage"]),
            ("facet_breadth", s["facet"]),
            ("newly_reachable_share", s["newly"]),
            # 0.0 UNCONDITIONALLY, which is what the real scorer emits on every run: k* is the
            # argmax of a monotone ladder and k_hat is its last index, so max(0, k* - k_hat) is
            # identically 0. Written so the fixture matches the table the gate really reads,
            # and so a regression to this column as the done axis would empty a cell here too.
            ("stop_undershoot", 0.0),
        ):
            if val is not None:
                scores.append(
                    dict(run_id=rid, metric_name=name, scorer_hash=SCORER, value=float(val))
                )
        for d, c, n in s["cad"]:
            scores.append(
                dict(run_id=rid, metric_name=f"cad#{d}", scorer_hash=SCORER, value=float(c))
            )
            scores.append(
                dict(run_id=rid, metric_name=f"cad_n#{d}", scorer_hash=SCORER, value=float(n))
            )
        ledger.append(dict(run_id=rid, row_idx=0, currency="llm_calls", charged=1.0))
        for k in range(s["n_malformed"]):
            ledger.append(dict(run_id=rid, row_idx=k + 1, currency="malformed", charged=1.0))
        calls.append(dict(run_id=rid, actor="inquirer", model=s["model"]))
        calls.append(dict(run_id=rid, actor="drafter", model="openai/aws/gpt-oss-120b"))

    # Every real `scores.parquet` carries `graph_version` beside `scorer_hash` (measured: 64/64
    # gate parquets, single-valued). Stamped here in one place rather than on each `dict(...)`
    # above so a test can rewrite one row's value to build the mixed-store case the gate refuses.
    for row in scores:
        row.setdefault("graph_version", GRAPH)

    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    tables = [
        ("runs", runs),
        ("turns", turns),
        ("scores", scores),
        ("ledger", ledger),
        ("calls", calls),
    ]
    # native.parquet is written ONLY when a spec carried a tau_reward, matching `_con`'s
    # existing tables: no test predating fork_reward ever produced one, and `_native_reward_by_
    # run` treats its absence as "no rewards recorded", not as a missing-table error.
    if native:
        tables.append(("native", native))
    for name, rows in tables:
        pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")
    return d


def _gate(parquet_dir, **kw):
    from pinq_train.gate import run_gate

    args = dict(
        parquet_dir=parquet_dir,
        grid_name="dev_select_musique",
        baseline_grid_names=("dev_baseline_musique",),
        checkpoint_arm=CKPT,
        baseline_arm=BASE,
        baseline_model_id=None,
        length_margin=0.10,
        distinct_floor=0.65,
        scorer_hash=SCORER,
        bootstrap_seed=0,
        n_resamples=200,
        grids_root=None,
    )
    args.update(kw)
    return run_gate(**args)


def _q(t):
    return (f"who wrote the {t} novel", f"when did {t} die")


def _pairs(n=12, *, ckpt_over=None, base_over=None):
    """n task-matched pairs in a configuration that PASSES every gated criterion.

    A passing default is what lets each test below vary exactly one thing: with a failing
    default, `v["passed"] is False` would be true for reasons the test never named, and an
    assertion on the verdict would pass without touching the criterion under test.

    Half the tasks complete their evidence on the last ask and stop there; half never complete
    it and are halted by the cap. So the 2x2 has both columns AND a forced stop to exclude,
    and both cells read 1.0 on both arms.

    A COMPLETED TASK CARRIES COVERAGE 1.0 ON BOTH ARMS, so the gain lives on the six tasks
    neither arm completed: `_ladder` makes `evidence_coverage` the ladder's last rung, and a
    run cannot both hold complete evidence and score 0.7 for it. The paired coverage delta of
    the default fixture is therefore (6*0.0 + 6*0.2)/12 = 0.1, still a gain whose interval
    excludes 0; the tests that assert a particular delta override `coverage` on every task and
    are untouched by this.
    """
    out = []
    for t in range(n):
        done = t % 2 == 0
        common = dict(
            task=f"t{t}",
            questions=_q(t),
            done_after="last" if done else None,
            stop_reason="policy_stop" if done else "max_turns",
            facet=0.5,
            newly=0.5,
        )
        ck = dict(common, arm=CKPT, coverage=1.0 if done else 0.7, cad=((2, 1.0, 4.0),))
        bs = dict(common, arm=BASE, coverage=1.0 if done else 0.5, cad=((2, 0.5, 4.0),))
        ck.update(ckpt_over or {})
        bs.update(base_over or {})
        out.append(_spec(**ck))
        out.append(_spec(**bs))
    return out


def _crit(v, name):
    return v["criteria"][name]


# --------------------------------------------------------------------------- malformed


def test_malformed_passes_when_the_checkpoint_is_no_worse(tmp_path) -> None:
    d = _build(tmp_path, _pairs(ckpt_over={"n_malformed": 0}, base_over={"n_malformed": 1}))
    c = _crit(_gate(d), "malformed")
    assert c["passed"] is True
    assert c["value"] == 0.0 and c["baseline"] == 1.0


def test_malformed_fails_when_the_checkpoint_emits_more_broken_json(tmp_path) -> None:
    d = _build(tmp_path, _pairs(ckpt_over={"n_malformed": 2}, base_over={"n_malformed": 0}))
    v = _gate(d)
    assert _crit(v, "malformed")["passed"] is False
    assert v["passed"] is False


def test_malformed_tolerates_one_rounding_level_event_against_a_clean_baseline(tmp_path) -> None:
    """Measured on `artifacts/gate/8b2`: six DPO checkpoints failed `malformed` on strategyqa on
    exactly one malformed event in 333 runs (rate 0.003) against a zero-event baseline. That is
    a rounding-level event, not a behaviour, and the gate must say so: `rate <= max(baseline,
    MALFORMED_TOLERANCE)` with `MALFORMED_TOLERANCE = 0.005` (mirrored here, not imported, so a
    drift between the two shows up as a test failure rather than a silently moved goalpost).
    """
    specs = _pairs(n=333)
    assert specs[0]["arm"] == CKPT and specs[1]["arm"] == BASE  # even index = checkpoint spec
    specs[0]["n_malformed"] = 1  # 1 event / 333 checkpoint runs = 0.003003...; baseline stays 0

    v = _gate(_build(tmp_path, specs), n_resamples=20)
    c = _crit(v, "malformed")
    assert c["passed"] is True
    assert c["value"] == pytest.approx(1 / 333)
    assert c["baseline"] == 0.0

    m = v["malformed_tolerance"]
    assert m["tolerance"] == 0.005
    assert m["n_events"] == 1
    assert m["n_runs"] == 333
    assert m["baseline_rate"] == 0.0
    assert m["rate"] == pytest.approx(1 / 333)


def test_malformed_fails_above_the_tolerance_against_a_clean_baseline(tmp_path) -> None:
    """One tick over the tolerance still fails: the floor is 0.005, not "close to it"."""
    specs = _pairs(n=500)
    for i in (0, 2, 4):  # three checkpoint specs (even index): 3 / 500 = 0.006
        assert specs[i]["arm"] == CKPT
        specs[i]["n_malformed"] = 1

    v = _gate(_build(tmp_path, specs), n_resamples=20)
    c = _crit(v, "malformed")
    assert c["passed"] is False
    assert c["value"] == pytest.approx(0.006)

    m = v["malformed_tolerance"]
    assert m["tolerance"] == 0.005
    assert m["n_events"] == 3
    assert m["n_runs"] == 500
    assert m["baseline_rate"] == 0.0
    assert m["rate"] == pytest.approx(0.006)


def test_malformed_passes_against_a_baseline_that_itself_exceeds_tolerance(tmp_path) -> None:
    """The baseline branch of `max(baseline_rate, MALFORMED_TOLERANCE)`: a checkpoint at 0.02
    against a baseline of 0.03 passes ON THE BASELINE, not on the floor -- 0.02 is four times
    the 0.005 tolerance, so a bound that dropped the baseline term would fail this checkpoint.
    """
    specs = _pairs(n=100)
    for i in (0, 2):  # two checkpoint specs: 2 / 100 = 0.02
        assert specs[i]["arm"] == CKPT
        specs[i]["n_malformed"] = 1
    for i in (1, 3, 5):  # three baseline specs: 3 / 100 = 0.03
        assert specs[i]["arm"] == BASE
        specs[i]["n_malformed"] = 1

    v = _gate(_build(tmp_path, specs), n_resamples=20)
    c = _crit(v, "malformed")
    assert c["passed"] is True
    assert c["value"] == pytest.approx(0.02)
    assert c["baseline"] == pytest.approx(0.03)

    m = v["malformed_tolerance"]
    assert m["tolerance"] == 0.005
    assert m["n_events"] == 2
    assert m["n_runs"] == 100
    assert m["baseline_rate"] == pytest.approx(0.03)
    assert m["rate"] == pytest.approx(0.02)


# --------------------------------------------------------------------------- distinct-3


def test_distinct3_passes_when_questions_vary_within_a_task(tmp_path) -> None:
    """WITHIN a task, not pooled: varying within a task IS state-dependence, while repeating
    across runs of one task is convergence on the right question."""
    c = _crit(_gate(_build(tmp_path, _pairs())), "distinct3")
    assert c["passed"] is True, c
    assert c["value"] >= 0.65


def test_distinct3_fails_when_the_policy_asks_one_question_forever(tmp_path) -> None:
    """Mode collapse: the questions stopped depending on the state.

    NON-VACUITY for the by-seed gate (`artifacts/degenerate_metrics_20260919/RESULT.md`):
    every spec here uses the default `seed=0` (see `_pairs`/`_spec`), so `by_seed` groups
    each task's questions by its single (task, seed) pair -- the same repeated-question
    collapse `value` sees, not a seed-pooling artifact. If gating `passed` on `by_seed`
    could never fail, "by-seed passes" would be indistinguishable from "by-seed cannot
    fail"; asserting `by_seed` itself (not just `passed`) below the floor closes that.
    """
    same = "who wrote the novel that inspired the film"
    v = _gate(_build(tmp_path, _pairs(ckpt_over={"questions": (same, same)})))
    c = _crit(v, "distinct3")
    assert c["by_seed"] < 0.65, c  # genuine within-(task, seed) collapse fails by_seed too
    assert c["passed"] is False
    assert v["passed"] is False


def test_distinct3_by_seed_is_not_halved_by_a_second_seed_that_repeats_the_first(tmp_path) -> None:
    """Lane L1.1: `within_task_distinct_n` groups every run of a task together regardless of
    seed. A checkpoint evaluated at two seeds that asks the SAME fully state-dependent
    questions under both is therefore scored as if it had collapsed, purely because it ran
    twice: duplicating a fixed two-question set verbatim doubles the trigram total while
    adding no new trigram, which halves distinct-n by construction (measured on
    `artifacts/testsplit_qa`: the trained checkpoint's test-split questions are close to
    byte-identical across its two seeds, and pooled `value` drops from 0.95 on development,
    one seed, to 0.58 on test, two seeds).

    `by_seed` groups by (task, seed) instead, so a second seed that repeats the first can no
    longer move it. `value` must stay exactly 0.5 -- the artifact, unchanged -- so this test
    also guards against "fixing" distinct3 by silently redefining what `value` means.
    """
    specs = []
    for i in range(12):
        task = f"t{i}"
        qs = _q(task)  # two trigram-disjoint questions: a single seed's own distinct-3 is 1.0
        for arm in (CKPT, BASE):
            for seed in (0, 1):
                specs.append(
                    _spec(
                        arm=arm,
                        task=task,
                        seed=seed,
                        questions=qs,
                        done_after="last",
                        stop_reason="policy_stop",
                        coverage=1.0,
                        cad=((2, 1.0, 4.0),),
                        facet=0.5,
                        newly=0.5,
                    )
                )
    v = _gate(_build(tmp_path, specs), n_resamples=20)
    c = _crit(v, "distinct3")
    assert c["n_seeds"] == 2, c
    assert c["value"] == pytest.approx(0.5), c  # the pooling artifact -- unchanged by this fix
    assert c["by_seed"] == pytest.approx(1.0), c  # the fix: a repeated seed no longer halves it


def test_distinct3_passes_on_by_seed_when_only_the_seed_pooled_value_fails_the_floor(
    tmp_path,
) -> None:
    """`artifacts/degenerate_metrics_20260919`: 9 of 9 trained cells (s0/s1/s2 x musique/
    strategyqa/wiki2) failed the 0.65 floor on `value` while `by_seed` passed on all nine
    (0.804-0.929) -- see `artifacts/seedrep_gate_20260919/distinct3.json`. That is exactly this
    fixture's shape: a second seed that repeats the first drags the seed-pooled `value` below
    the floor while `by_seed`, which cannot be moved by how many seeds were evaluated (see
    `within_task_seed_distinct_n`'s docstring), stays high. A gate whose `passed` bit still
    reads FAIL in that shape is reporting how many seeds ran, not whether the policy collapsed
    -- CONTRIBUTING.md's "never state a measurement you did not take" cuts against publishing that
    FAIL as a finding about the arm.

    NON-VACUITY: `test_distinct3_fails_when_the_policy_asks_one_question_forever` already
    proves this criterion's `passed` can still be False (genuine collapse, single seed, `value`
    and `by_seed` coincide and both fail) -- so this fix does not make `passed` vacuously True.
    """
    specs = []
    for i in range(12):
        task = f"t{i}"
        qs = _q(task)  # two trigram-disjoint questions: a single seed's own distinct-3 is 1.0
        for arm in (CKPT, BASE):
            for seed in (0, 1):
                specs.append(
                    _spec(
                        arm=arm,
                        task=task,
                        seed=seed,
                        questions=qs,
                        done_after="last",
                        stop_reason="policy_stop",
                        coverage=1.0,
                        cad=((2, 1.0, 4.0),),
                        facet=0.5,
                        newly=0.5,
                    )
                )
    v = _gate(_build(tmp_path, specs), n_resamples=20)
    c = _crit(v, "distinct3")
    assert c["value"] == pytest.approx(0.5) and c["value"] < 0.65, c  # value fails the floor
    assert c["by_seed"] == pytest.approx(1.0) and c["by_seed"] >= 0.65, c  # by_seed clears it
    assert c["passed"] is True, c  # the fix: gated on by_seed, not on how many seeds ran


def test_a_per_model_report_cannot_present_the_datasets_distinct_3_as_the_policys(tmp_path) -> None:
    """`within_task_distinct_3` used to name two different measurements under one identical
    label. `dataset_report`'s is a property of the TRAINING FILE: every arm trained on the same
    export prints the same number regardless of base model or seed, because it never reads a
    checkpoint. `criteria["distinct3"]` above is a property of the CHECKPOINT'S OWN generated
    dev questions at inference, measured fresh per checkpoint -- two arms trained on the exact
    same file can differ here. `docs/GPU_RUNBOOK.md` once printed the identical phrase
    "within-task distinct-3" for both, five lines apart, with no marker saying which was which.
    A per-model report built from `rung1.manifest.json` plus the gate verdict must not be able
    to repeat that: the dataset-scoped figure needs a name that says so, and the verdict needs
    to say, in words, that its own figure is not that one.
    """
    from pinq_train.rung1_sft.train import dataset_report

    rows = [
        {
            "is_stop": False,
            "action_json": json.dumps({"question": f"what is fact {i} of task {i % 8}"}),
            "task_id": f"t{i % 8}",
            "suite_id": "musique",
        }
        for i in range(40)
    ]
    rep = dataset_report(rows, floor=0.55)

    # The dataset-level figure must not be reachable under the bare name that gate.py's
    # checkpoint-level criterion also answers to in prose ("within-task distinct-3").
    assert "within_task_distinct_3" not in rep
    assert "within_task_distinct_3_of_dataset" in rep

    # The policy-level figure keeps its established key: `select.py` and this suite both key
    # off `criteria["distinct3"]`, and `test_the_cap8_rule_reproduces_the_previous_verdict_bit_
    # for_bit` hashes `criteria` byte for byte, so the fix cannot live inside it. It must live
    # beside `criteria`, saying plainly that this figure is the checkpoint's own and is not
    # `within_task_distinct_3_of_dataset`.
    v = _gate(_build(tmp_path, _pairs()))
    assert "within_task_distinct_3_of_dataset" in v["distinct3_scope"]
    assert "checkpoint" in v["distinct3_scope"].lower()


# --------------------------------------------------------------------------- question length


def test_length_equivalence_passes_when_the_two_arms_ask_similar_length_questions(tmp_path) -> None:
    c = _crit(_gate(_build(tmp_path, _pairs())), "length_equivalence")
    assert c["passed"] is True, c
    assert c["value"] == c["baseline"] == 4.5


def test_length_equivalence_fails_when_the_checkpoint_doubles_its_question_length(tmp_path) -> None:
    """The verbosity confound, caught before it reaches a table: a longer question retrieves
    more by accident, so an unchecked length gain reads as a question-quality gain."""
    d = _build(
        tmp_path,
        _pairs(ckpt_over={"questions": ("a b c d e f g h i j k l",)}),
    )
    v = _gate(d)
    c = _crit(v, "length_equivalence")
    assert c["passed"] is False
    assert c["value"] == 12.0 and c["baseline"] == 4.5
    assert v["passed"] is False


# --------------------------------------------------------------------------- the STOP 2x2


def _stop_specs(*, ckpt_stops_when_done, ckpt_asks_when_not_done):
    """Half the tasks completable, half not; the baseline is right at every decision point.

    Two questions per run, so each episode offers three decisions (t = 0, 1, 2). On a
    completable task the evidence is complete after the ask named by `done_after`; on the
    others it never is. The checkpoint's two failure modes are expressed as ladders and stop
    reasons, never as a metric the scorer does not produce:

      * not stopping when done  -> done at t = 1, then one more ask: (done, ASK) at t = 1.
      * not asking when not done -> `policy_stop` at t = 2 with coverage still short.

    A completable task carries coverage 1.0 and an uncompletable one 0.7 / 0.5, because
    `_ladder` makes the ladder's last rung `evidence_coverage`: the done axis and the coverage
    column are one instrument, so they cannot disagree about whether the evidence is complete.
    """
    out = []
    for t in range(12):
        done = t % 2 == 0
        if done:
            ck = dict(done_after="last" if ckpt_stops_when_done else 1, stop_reason="policy_stop")
        else:
            ck = dict(
                done_after=None,
                stop_reason="budget" if ckpt_asks_when_not_done else "policy_stop",
            )
        common = dict(task=f"t{t}", questions=_q(t))
        out.append(
            _spec(arm=CKPT, coverage=1.0 if done else 0.7, cad=((2, 1.0, 4.0),), **ck, **common)
        )
        out.append(
            _spec(
                arm=BASE,
                coverage=1.0 if done else 0.5,
                cad=((2, 0.5, 4.0),),
                done_after="last" if done else None,
                stop_reason="policy_stop" if done else "budget",
                **common,
            )
        )
    return out


def test_the_stop_2x2_passes_when_neither_cell_is_worse_than_the_baseline(tmp_path) -> None:
    d = _build(tmp_path, _stop_specs(ckpt_stops_when_done=True, ckpt_asks_when_not_done=True))
    c = _crit(_gate(d), "stop_2x2")
    assert c["passed"] is True, c
    assert c["value"]["p_stop_given_done"] == 1.0
    assert c["value"]["p_ask_given_not_done"] == 1.0


def test_the_stop_2x2_fails_when_one_cell_is_bought_with_the_other(tmp_path) -> None:
    """I.8's worked example: stopping MORE raises P(STOP|done) and lowers P(ASK|not done).
    Both cells are reported so exactly this trade is visible rather than flattering.

    Hand-checked. Six completable tasks contribute 2 (not done, ASK) each; six uncompletable
    ones contribute 2 (not done, ASK) plus, because this checkpoint halts by choice with
    coverage still short, 1 (not done, STOP). So P(ASK | not done) = 24/30 = 0.8 against the
    baseline's 24/24, while P(STOP | done) is a perfect 6/6.

    UNDER `cap8`, which is where the "both cells >= the baseline" rule now lives: the
    2026-09-15 follow-up to D9 makes the criterion report-only under `matched_cost`, because
    the prompted base almost never stops and that rule inherits the degeneracy. The rule itself
    is unchanged and is still exercised both ways here; the downgrade has its own test.
    """
    d = _build(tmp_path, _stop_specs(ckpt_stops_when_done=True, ckpt_asks_when_not_done=False))
    v = _gate(d, coverage_rule="cap8")
    c = _crit(v, "stop_2x2")
    assert c["value"]["p_stop_given_done"] == 1.0
    assert c["value"]["p_ask_given_not_done"] == pytest.approx(0.8)
    assert c["baseline"]["p_ask_given_not_done"] == 1.0
    assert c["value"]["n_not_done_stop"] == 6
    assert c["passed"] is False
    assert v["passed"] is False


def test_the_stop_2x2_fails_when_the_policy_keeps_asking_after_it_is_done(tmp_path) -> None:
    """The other direction, which a run-level 2x2 could not see at all: the episode stopped by
    choice, so it scored 1.0 no matter how many questions it bought after the evidence was
    already complete. Counted over decision points, the wasted ask is a (done, ASK) cell.

    At `cap8` for the reason the test above gives: the rule is gated there, report-only under
    `matched_cost`, and the cells are identical under both."""
    d = _build(tmp_path, _stop_specs(ckpt_stops_when_done=False, ckpt_asks_when_not_done=True))
    c = _crit(_gate(d, coverage_rule="cap8"), "stop_2x2")
    assert c["value"]["p_stop_given_done"] == pytest.approx(0.5)
    assert c["value"]["n_done_ask"] == 6
    assert c["value"]["mean_asks_after_done"] == pytest.approx(0.5)
    assert c["baseline"]["p_stop_given_done"] == 1.0
    assert c["passed"] is False


def test_the_stop_2x2_names_the_columns_it_read(tmp_path) -> None:
    """`runs.parquet` has no coverage column, so the two axes come from two tables. A reader
    must not have to reconstruct which cells were counted -- nor be left to assume the done
    axis is the metric named `stop_undershoot`, which is identically 0."""
    c = _crit(_gate(_build(tmp_path, _pairs())), "stop_2x2")
    assert c["columns"]["unit"] == "one decision point (run, t), t = 0..n_asks"
    assert c["columns"]["done_axis"] == "frontier_q#t >= 1-1e-12 before the decision at t"
    assert "runs.n_asks" in c["columns"]["stop_axis"]
    assert "policy_stop" in c["columns"]["stop_means"]
    assert "identically 0" in c["columns"]["not_stop_undershoot"]
    assert c["columns"]["filter"] == "runs.status='ok'"


def test_error_runs_are_excluded_from_the_2x2(tmp_path) -> None:
    """stop_reason is '' on exactly the error rows; counting them would put every failed unit
    in the ASK column."""
    specs = _pairs(n=6)
    specs.append(_spec(arm=CKPT, task="boom", status="error", stop_reason=""))
    c = _crit(_gate(_build(tmp_path, specs)), "stop_2x2")
    assert c["value"]["n_runs"] == 6, "the error run must not reach the 2x2"
    # three completable tasks contribute three decisions each; three capped ones contribute two
    # apiece, their forced final state excluded rather than filed as an ASK.
    assert c["value"]["n_states"] == 15
    assert c["value"]["n_forced_stops"] == 3


# --------------------------------------------------------------------------- paired deltas


def test_evidence_coverage_passes_when_the_checkpoint_gains(tmp_path) -> None:
    d = _build(tmp_path, _pairs(ckpt_over={"coverage": 0.7}, base_over={"coverage": 0.5}))
    c = _crit(_gate(d), "evidence_coverage")
    assert c["passed"] is True, c
    assert c["value"] == pytest.approx(0.2)
    assert c["ci_lo"] > 0


def test_evidence_coverage_fails_when_there_is_no_effect(tmp_path) -> None:
    """A CI that contains 0 is not a win, and `pi train gate` must not round it into one."""
    d = _build(tmp_path, _pairs(ckpt_over={"coverage": 0.5}, base_over={"coverage": 0.5}))
    v = _gate(d)
    c = _crit(v, "evidence_coverage")
    assert c["value"] == pytest.approx(0.0)
    assert c["ci_lo"] <= 0 <= c["ci_hi"]
    assert c["passed"] is False
    assert v["passed"] is False


def test_facet_breadth_is_gated_as_no_loss_rather_than_as_a_gain(tmp_path) -> None:
    """I.9 Tier B asks for 'no loss on facet breadth', not a gain: a policy that goes deeper on
    a narrower set of facets is still allowed through, and one that goes narrower is not.

    UNDER `cap8`, which is where that rule now lives: the 2026-09-15 follow-up to D9 makes the
    criterion report-only under `matched_cost`, and the test below pins that direction. The
    no-loss rule itself is unchanged and is still exercised both ways here.
    """
    flat = _build(tmp_path, _pairs(ckpt_over={"facet": 0.5}, base_over={"facet": 0.5}))
    assert _crit(_gate(flat, coverage_rule="cap8"), "facet_breadth")["passed"] is True

    lost = _build(tmp_path / "b", _pairs(ckpt_over={"facet": 0.2}, base_over={"facet": 0.5}))
    assert _crit(_gate(lost, coverage_rule="cap8"), "facet_breadth")["passed"] is False


def test_newly_reachable_share_is_reported_and_not_gated(tmp_path) -> None:
    """I.9 Tier B names coverage, coverage at depth >= 2 and facet breadth. Gating a fourth
    endpoint nobody preregistered would let this command reject a checkpoint on a rule the
    plan never stated."""
    d = _build(tmp_path, _pairs(ckpt_over={"newly": 0.1}, base_over={"newly": 0.9}))
    v = _gate(d)
    c = _crit(v, "newly_reachable_share")
    assert c["gated"] is False
    assert c["value"] == pytest.approx(-0.8)
    assert v["passed"] is True, "an ungated criterion must not decide the verdict"


# --------------------------------------------------------------------------- cad_ge2


def test_cad_ge2_equals_the_hand_computed_cardinality_weighted_value(tmp_path) -> None:
    """|V_d|-weighted, and d >= 2 only. Weighting is not a nicety: an unweighted mean over
    depths lets a depth holding two nodes outvote one holding forty.

        cad#1 = 1.00, |V_1| = 99   <- must be EXCLUDED
        cad#2 = 0.50, |V_2| =  4
        cad#3 = 1.00, |V_3| =  1
        (0.50*4 + 1.00*1) / (4 + 1) = 3.0 / 5 = 0.60
    """
    from pinq_train.gate import cad_ge2_by_run

    cad = ((1, 1.0, 99.0), (2, 0.5, 4.0), (3, 1.0, 1.0))
    d = _build(tmp_path, [_spec(arm=CKPT, task="t0", cad=cad)])
    got = cad_ge2_by_run(d, scorer_hash=SCORER)
    assert got == {"r0": pytest.approx(0.6)}, got


def test_cad_ge2_paired_delta_is_gated(tmp_path) -> None:
    """UNDER `cap8`, which is where the gain rule now lives: the 2026-09-15 follow-up to D9
    makes the criterion report-only under `matched_cost`, because C@d>=2 rises with the number
    of asks and `scores` carries no `cad#d` ladder to read it at matched k. The gain rule
    itself is unchanged and is still exercised here; the downgrade has its own test."""
    hi = ((2, 1.0, 4.0),)
    lo = ((2, 0.5, 4.0),)
    d = _build(tmp_path, _pairs(ckpt_over={"cad": hi}, base_over={"cad": lo}))
    c = _crit(_gate(d, coverage_rule="cap8"), "cad_ge2")
    assert c["gated"] is True and c["passed"] is True
    assert c["value"] == pytest.approx(0.5)


def test_a_run_with_no_depth_two_or_deeper_contributes_no_cad_ge2(tmp_path) -> None:
    """NOT 0.0. A task whose graph is one hop deep has no C@d>=2 to report, and a fabricated
    zero would drag the arm's mean down for a task that never had the quantity."""
    from pinq_train.gate import cad_ge2_by_run

    d = _build(tmp_path, [_spec(arm=CKPT, task="t0", cad=((0, 1.0, 3.0), (1, 1.0, 2.0)))])
    assert cad_ge2_by_run(d, scorer_hash=SCORER) == {}


# --------------------------------------------------------------------------- absence


def test_a_missing_criterion_fails_rather_than_passes(tmp_path) -> None:
    """A gate that passes for want of data is worse than no gate: it produces a PASS verdict
    with a provenance trail, which is exactly what someone will cite."""
    d = _build(tmp_path, _pairs(ckpt_over={"coverage": None}, base_over={"coverage": None}))
    v = _gate(d)
    c = _crit(v, "evidence_coverage")
    assert c["passed"] is False
    assert c["n"] == 0
    assert "no" in c["reason"].lower()
    assert v["passed"] is False


def test_unpaired_runs_are_dropped_and_counted(tmp_path) -> None:
    """Dropping is right -- an unpaired task cannot enter a paired delta -- but a silent drop
    means nobody can tell a 6-pair gate from a 12-pair one: `n_unpaired_baseline` is recorded
    right next to `n_paired_keys`, so this stays auditable rather than silent.

    Correction, 2026-09-17: the fixture used to put the unpaired key on the CHECKPOINT side
    ("lonely" ran only there) and this test asserted `IncompleteArm`. Measured against 156 real
    verdicts, that is not the shape this test protects: a checkpoint key the baseline lacks is
    always a dropped RUN and is refused, unconditionally -- see
    `test_a_checkpoint_key_the_baseline_lacks_is_refused_not_dropped` below, which is this
    test's original fixture, moved rather than deleted. The legitimate drop-and-count case --
    48 of those 156 verdicts -- has the extra key on the BASELINE side instead (an extra seed
    the checkpoint didn't need), which is what this fixture builds now."""
    specs = _pairs(n=6, ckpt_over={"coverage": 0.7}, base_over={"coverage": 0.5})
    specs.append(_spec(arm=BASE, task="lonely", coverage=0.9))
    v = _gate(_build(tmp_path, specs))
    assert v["pairing"]["n_unpaired_baseline"] == 1
    assert _crit(v, "evidence_coverage")["n"] == 6


def test_a_checkpoint_key_the_baseline_lacks_is_refused_not_dropped(tmp_path) -> None:
    """This is `test_unpaired_runs_are_dropped_and_counted`'s fixture before the 2026-09-17
    correction: the unpaired key ("lonely") sits on the CHECKPOINT side instead of the
    baseline's. That is not the same hazard as the test above -- the checkpoint run at "lonely"
    exists, and a silent drop would exclude it from every criterion with nothing in the verdict
    distinguishing that from a complete gate -- so unlike a baseline with extra keys, this
    refuses unconditionally."""
    from pinq_train.gate import IncompleteArm

    specs = _pairs(n=6, ckpt_over={"coverage": 0.7}, base_over={"coverage": 0.5})
    specs.append(_spec(arm=CKPT, task="lonely", coverage=0.9))
    d = _build(tmp_path, specs)
    with pytest.raises(IncompleteArm) as e:
        _gate(d)
    msg = str(e.value)
    assert CKPT in msg
    assert BASE in msg
    assert "lonely" in msg  # the one task only the checkpoint arm ran, named as an example


def test_the_baseline_model_id_is_resolved_through_calls_parquet(tmp_path) -> None:
    """runs.parquet carries model_pin_hash and no model NAME, so the two-pin protocol (the same
    arm run against two different base models) is only separable through calls.parquet."""
    specs = []
    for t in range(6):
        specs.append(_spec(arm=CKPT, task=f"t{t}", coverage=0.7))
        specs.append(_spec(arm=BASE, task=f"t{t}", coverage=0.5, model="openai/aws/gpt-oss-120b"))
        # the SAME arm, task and seed against the other pin: separable only by model name
        specs.append(_spec(arm=BASE, task=f"t{t}", coverage=0.1, model="qwen3-8b-base"))
    d = _build(tmp_path, specs)

    unfiltered = _gate(d)
    assert _crit(unfiltered, "evidence_coverage")["value"] == pytest.approx(0.4), (
        "without the filter both pins are averaged into one baseline, which is the failure "
        "this flag exists to prevent"
    )

    v = _gate(d, baseline_model_id="openai/aws/gpt-oss-120b")
    assert v["selection"]["baseline_model_id"] == "openai/aws/gpt-oss-120b"
    assert _crit(v, "evidence_coverage")["value"] == pytest.approx(0.2)


def test_the_checkpoint_model_id_is_resolved_through_calls_parquet(tmp_path) -> None:
    """The mirror of the test above, on the CHECKPOINT side.

    A curriculum sweep that serves several checkpoints on one vLLM job and compacts them into
    one isolated store produces exactly this shape: several checkpoints run under the SAME
    `arm_id` (`inquirer_trained`) and the SAME `grid_name` (`dev_select_<suite>`), separable
    only through calls.parquet, same as the two-pin baseline above. `_select_runs` already
    takes a `model_id` argument generically -- it does not know which side is "baseline" -- but
    `run_gate`'s checkpoint-side call passed `model_id=None` unconditionally, with no parameter
    or CLI flag to override it. Without a `--checkpoint-model-id` symmetric to
    `--baseline-model-id`, two checkpoints sharing an arm/grid are pooled into one average, and
    a verdict computed over that average describes neither checkpoint. This is the seed-blending
    failure the per-seed-column requirement exists to catch.
    """
    specs = []
    for t in range(6):
        specs.append(_spec(arm=BASE, task=f"t{t}", coverage=0.5))
        # two different checkpoints, same arm_id and grid_name, separable only by model name
        specs.append(
            _spec(
                arm=CKPT,
                task=f"t{t}",
                coverage=0.7,
                model="qwen3-4b-sft-headline-easyfirst-s0",
            )
        )
        specs.append(
            _spec(
                arm=CKPT,
                task=f"t{t}",
                coverage=0.3,
                model="qwen3-4b-sft-headline-hardfirst-s0",
            )
        )
    d = _build(tmp_path, specs)

    unfiltered = _gate(d)
    assert _crit(unfiltered, "evidence_coverage")["value"] == pytest.approx(0.0), (
        "without the filter both checkpoints are averaged into one arm (mean coverage 0.5, "
        "identical to the baseline's), which is the blending this flag exists to prevent"
    )

    v = _gate(d, checkpoint_model_id="qwen3-4b-sft-headline-easyfirst-s0")
    assert v["selection"]["checkpoint_model_id"] == "qwen3-4b-sft-headline-easyfirst-s0"
    assert _crit(v, "evidence_coverage")["value"] == pytest.approx(0.2)


# ----------------------------------------------------------------- incomplete arm (2026-09-17)
#
# Incident, 2026-09-17: six of eight gate arms were only partially run -- a staging step
# upstream had selected checkpoint runs for some but not all of the baseline's tasks -- and
# only a human noticing kept them out of a verdict. `EmptyArm` (below) refuses zero runs on one
# side; it does not refuse a PARTIAL arm, which is the more dangerous shape because every
# criterion still computes a value and the printed verdict is indistinguishable from one over a
# complete population.
#
# Correction, 2026-09-17: `IncompleteArm` originally refused on set INEQUALITY -- either arm
# missing a key the other has. Re-measured at 2026-09-17T16:51:10+0300 against every real
# (non-symlink) *.json under artifacts/gate, recursive and top-level (two subtrees are
# entirely symlink farms and would double-count if walked naively) -- there are 150 real
# verdicts, and ALL 150 have this shape (a baseline arm with strictly more keys than the
# checkpoint, e.g. an extra seed run baseline-only); ZERO have a checkpoint key the baseline
# lacks. An earlier pass reported 48 of 156 for the same shape; both numbers were wrong (wrong
# denominator from the symlink farms, wrong numerator from reading a counter not every verdict
# carries). Baselines are deliberately run at more seeds than the checkpoints compared against
# them -- that is the design working, not a defect this corpus happens to tolerate. Only a
# checkpoint key the baseline lacks is a hazard -- a checkpoint RUN that would be silently
# dropped -- so the check is now directional: it refuses when the checkpoint's key set is NOT A
# SUBSET of the baseline's, and never refuses merely because the baseline has extra keys, which
# is paired and counted instead. Checked once, before any criterion runs, the same place and
# for the same reason as `EmptyArm`.


def test_a_checkpoint_arm_of_132_keys_passes_and_pairs_132_against_a_baseline_of_264_that_contains_them(
    tmp_path,
) -> None:
    """Built from a real verdict's shape (2026-09-17), not a fixture designed to match the
    guard: pairing key (suite, task, seed); paired count 132; unpaired baseline count 132;
    unpaired checkpoint count 0 -- the baseline ran every checkpoint task again at a second
    seed. Must pass and pair all 132; this is the shape the pre-correction equality check
    wrongly refused, and all 150 real verdicts under the gate directory have it (re-measured
    2026-09-17T16:51:10+0300; zero have the reverse, checkpoint-over, shape)."""
    specs = _pairs(n=132, ckpt_over={"coverage": 0.7}, base_over={"coverage": 0.5})
    # The baseline's second seed: every task the checkpoint ran, run again at seed=1, on the
    # baseline arm only. The checkpoint's 132 keys stay a full subset of the baseline's 264.
    for t in range(132):
        specs.append(_spec(arm=BASE, task=f"t{t}", seed=1, coverage=0.5))
    d = _build(tmp_path, specs)
    v = _gate(d)  # must not raise
    assert v["pairing"]["n_checkpoint_runs"] == 132
    assert v["pairing"]["n_baseline_runs"] == 264
    assert v["pairing"]["n_paired_keys"] == 132
    assert v["pairing"]["n_unpaired_checkpoint"] == 0
    assert v["pairing"]["n_unpaired_baseline"] == 132


def test_a_checkpoint_key_the_264_key_baseline_still_lacks_still_refuses(tmp_path) -> None:
    """Same 132/264 backdrop as the pass case above, plus one checkpoint run ("t132") the
    baseline never ran at any seed: the checkpoint's key set is no longer a subset of the
    baseline's, and this refuses regardless of how large or otherwise-legitimate the rest of
    the arm is."""
    from pinq_train.gate import IncompleteArm

    specs = _pairs(n=132, ckpt_over={"coverage": 0.7}, base_over={"coverage": 0.5})
    for t in range(132):
        specs.append(_spec(arm=BASE, task=f"t{t}", seed=1, coverage=0.5))
    specs.append(_spec(arm=CKPT, task="t132", coverage=0.9))
    d = _build(tmp_path, specs)
    with pytest.raises(IncompleteArm) as e:
        _gate(d)
    assert "t132" in str(e.value)


def test_underfilled_arm_refuses_a_checkpoint_still_short_of_its_own_grids_declared_tasks(
    tmp_path,
) -> None:
    """This is the ORIGINAL 2026-09-17 incident's shape (a staging step selected checkpoint
    runs for 40 of the baseline's 50 tasks), and it used to raise `IncompleteArm` under the
    set-inequality check. That check refused a legitimate shape too (see the section comment
    above) and was replaced with a directional subset rule -- and the checkpoint's 40 keys ARE
    a subset of the baseline's 50, so the rule that replaced it does not fire here either.

    `UnderfilledArm` (2026-09-17, later the same day) catches this shape instead: given
    `--grids-root`, the checkpoint's grid declares `n_tasks: 50` for suite `musique`, the paired
    count is 40, and 40 < 50 refuses. The original incident was a shortfall in HOW MANY tasks
    ran, not a question of WHICH, so the count alone is enough to catch it without ever naming
    the missing ten -- see `UnderfilledArm`'s docstring for what still cannot be named."""
    from pinq_train.gate import EmptyArm, IncompleteArm, UnderfilledArm, UnscoredArm

    specs = _pairs(n=50)
    # Drop the checkpoint side of 10 of the 50 tasks: the checkpoint arm now covers 40 of the
    # baseline's 50 tasks, still a SUBSET -- the shape of the original incident.
    dropped = {f"t{t}" for t in range(40, 50)}
    specs = [s for s in specs if not (s["arm"] == CKPT and s["task"] in dropped)]
    d = _build(tmp_path, specs)
    groot = tmp_path / "grids"
    groot.mkdir()
    (groot / "a.yaml").write_text(
        "name: dev_select_musique\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 50\n"
    )
    (groot / "b.yaml").write_text(
        "name: dev_baseline_musique\nsuites: [musique]\narms: [inquirer_prompted]\nn_tasks: 50\n"
    )
    with pytest.raises(UnderfilledArm) as e:
        _gate(d, grids_root=groot)
    assert "40" in str(e.value) and "50" in str(e.value) and "musique" in str(e.value)
    assert not issubclass(UnderfilledArm, EmptyArm)
    assert not issubclass(UnderfilledArm, IncompleteArm)
    assert not issubclass(UnderfilledArm, UnscoredArm)


def test_without_grids_root_the_underfilled_shape_still_passes_silently(tmp_path) -> None:
    """The residual `UnderfilledArm` documents rather than closes: the check requires
    `--grids-root`, and without it the exact shape the previous test now refuses still passes
    here -- opting in is what changed, not the default -- while `task_count` records WHY with a
    reason rather than pretending nothing was missed."""
    specs = _pairs(n=50)
    dropped = {f"t{t}" for t in range(40, 50)}
    specs = [s for s in specs if not (s["arm"] == CKPT and s["task"] in dropped)]
    d = _build(tmp_path, specs)
    v = _gate(d)  # grids_root defaults to None -- see docstring for the disclosed residual
    assert v["pairing"]["n_paired_keys"] == 40
    assert v["pairing"]["n_unpaired_checkpoint"] == 0
    assert v["pairing"]["n_unpaired_baseline"] == 10
    assert v["task_count"]["declared"] is None
    assert "grids-root" in v["task_count"]["reason"]


def test_underfilled_arm_does_not_raise_on_an_exact_match_with_the_declared_n_tasks(
    tmp_path,
) -> None:
    """Boundary: `n_tasks: 50` and 50 paired is a match, not a shortfall -- `UnderfilledArm`
    refuses when the paired count FALLS BELOW the declared one, never on equality."""
    specs = _pairs(n=50)
    d = _build(tmp_path, specs)
    groot = tmp_path / "grids"
    groot.mkdir()
    (groot / "a.yaml").write_text(
        "name: dev_select_musique\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 50\n"
    )
    (groot / "b.yaml").write_text(
        "name: dev_baseline_musique\nsuites: [musique]\narms: [inquirer_prompted]\nn_tasks: 50\n"
    )
    v = _gate(d, grids_root=groot)
    assert v["task_count"] == {
        "grid_name": "dev_select_musique",
        "declared": 50,
        "reason": "",
        "by_suite": {"musique": 50},
    }


def test_task_count_is_null_stamped_with_a_reason_when_the_grid_is_not_found(tmp_path) -> None:
    """Same discipline `_k_by_suite` already uses for `k_matches_baseline`: a grid that cannot
    be found under `--grids-root` stamps `declared: None` with a reason, never a guess, and does
    not raise -- there is nothing to compare the paired count against."""
    specs = _pairs(n=50)
    d = _build(tmp_path, specs)
    groot = tmp_path / "grids"
    groot.mkdir()
    (groot / "unrelated.yaml").write_text(
        "name: some_other_grid\nsuites: [musique]\narms: [inquirer_trained]\nn_tasks: 50\n"
    )
    v = _gate(d, grids_root=groot)
    assert v["task_count"]["declared"] is None
    assert "not found" in v["task_count"]["reason"]


def test_task_count_is_null_stamped_with_a_reason_when_the_grid_declares_no_cap(
    tmp_path,
) -> None:
    """The one honest residual: a grid selecting by `split` alone, with neither `n_tasks` nor
    `task_ids`, declares no count to check the paired total against. No grid under
    `conf/grids/` has this shape today (all 32 checked 2026-09-17 declare one or the other); if
    one ever does, this is the stamp it gets -- null, with this reason, and no raise -- not a
    new gap to design around."""
    specs = _pairs(n=50)
    d = _build(tmp_path, specs)
    groot = tmp_path / "grids"
    groot.mkdir()
    (groot / "a.yaml").write_text(
        "name: dev_select_musique\nsuites: [musique]\narms: [inquirer_trained]\nk: 5\n"
    )
    v = _gate(d, grids_root=groot)
    assert v["task_count"]["declared"] is None
    assert "n_tasks" in v["task_count"]["reason"] and "task_ids" in v["task_count"]["reason"]


def test_equal_task_coverage_between_the_arms_does_not_raise_incomplete_arm(tmp_path) -> None:
    """The ordinary, complete-run shape every other test in this file assumes: both arms cover
    exactly the same task set, so the paired population equals each arm's own population and
    `IncompleteArm` must not fire."""
    specs = _pairs(n=12)
    d = _build(tmp_path, specs)
    v = _gate(d)  # must not raise
    assert v["pairing"]["n_unpaired_checkpoint"] == 0
    assert v["pairing"]["n_unpaired_baseline"] == 0


def test_an_empty_checkpoint_arm_still_raises_empty_arm_not_incomplete_arm(tmp_path) -> None:
    """An empty checkpoint key set is trivially a SUBSET of any baseline key set, so since the
    2026-09-17 correction this fixture would NOT ALSO satisfy IncompleteArm's condition even if
    checked -- but `EmptyArm` must still run first (T19d), because a fully-empty arm is its own
    incident with its own message, not a 0-vs-N instance of a partial one."""
    from pinq_train.gate import EmptyArm, IncompleteArm

    specs = [s for s in _pairs(n=12) if s["arm"] == BASE]  # no checkpoint-arm rows at all
    d = _build(tmp_path, specs)
    with pytest.raises(EmptyArm):
        _gate(d)
    assert not issubclass(IncompleteArm, EmptyArm)


def _drop_scores_for_arm(d, arm):
    """Rewrites scores.parquet with every row belonging to `arm`'s runs removed -- the shape of
    today's near-miss: the score step never ran for one arm, so its runs exist in runs.parquet
    (EmptyArm does not fire) and cover the same tasks as the other arm (IncompleteArm does not
    fire) but have zero rows in scores.parquet."""
    runs = pq.read_table(d / "runs.parquet").to_pylist()
    arm_ids = {r["run_id"] for r in runs if r["arm_id"] == arm}
    scores = pq.read_table(d / "scores.parquet").to_pylist()
    kept = [r for r in scores if r["run_id"] not in arm_ids]
    pq.write_table(pa.Table.from_pylist(kept), d / "scores.parquet")
    return d


# ------------------------------------------------------------------------ unscored arm (2026-09-17)
#
# Near-miss, 2026-09-17, an hour before write-up: eight gate verdicts came back `passed: false`
# on every criterion, and the cause was not the checkpoints -- their `scores.parquet` held zero
# rows because the score step had never run, against 107,135 rows in the reference. Every
# criterion still computed a value anyway, and the printed verdict was indistinguishable from
# "these checkpoints failed" -- it said so, on every criterion, eight times. Only a human
# checking row counts before reporting caught it. Neither `EmptyArm` nor `IncompleteArm` (above)
# catches this: the arm's runs are present and cover the same tasks as the other arm: the
# failure is one level deeper, in `scores.parquet` specifically.


def test_run_gate_refuses_when_the_checkpoint_arms_scores_are_empty(tmp_path) -> None:
    from pinq_train.gate import UnscoredArm

    specs = _pairs(n=8)
    d = _build(tmp_path, specs)
    _drop_scores_for_arm(d, CKPT)
    with pytest.raises(UnscoredArm) as e:
        _gate(d)
    msg = str(e.value)
    assert CKPT in msg
    assert "8 run(s)" in msg  # the checkpoint's own run count
    assert "0 score row(s)" in msg  # the score-row count found: zero, not inferred
    assert "pi score" in msg  # told what to do, not left to infer it


def test_run_gate_refuses_when_the_baseline_arms_scores_are_empty(tmp_path) -> None:
    from pinq_train.gate import UnscoredArm

    specs = _pairs(n=8)
    d = _build(tmp_path, specs)
    _drop_scores_for_arm(d, BASE)
    with pytest.raises(UnscoredArm) as e:
        _gate(d)
    msg = str(e.value)
    assert BASE in msg
    assert "8 run(s)" in msg
    assert "0 score row(s)" in msg
    assert "pi score" in msg


def test_an_ordinary_fixture_is_unaffected_by_the_unscored_arm_check(tmp_path) -> None:
    """Every other test in this file builds a fixture this way; this one exists so the
    unscored-arm check itself has a pinned pass case, not just an incidental one."""
    specs = _pairs(n=6)
    d = _build(tmp_path, specs)
    v = _gate(d)  # must not raise
    assert "passed" in v


def test_an_unscored_checkpoint_arm_does_not_raise_empty_arm_or_incomplete_arm(tmp_path) -> None:
    """The checkpoint arm here has 8 real runs covering the same 8 tasks as the baseline --
    `EmptyArm` and `IncompleteArm` both see a normal, complete, paired arm -- so only the
    scores-are-empty check can catch this, and it must raise ITS OWN exception rather than
    either of theirs."""
    from pinq_train.gate import EmptyArm, IncompleteArm, UnscoredArm

    specs = _pairs(n=8)
    d = _build(tmp_path, specs)
    _drop_scores_for_arm(d, CKPT)
    with pytest.raises(UnscoredArm):
        _gate(d)
    assert not issubclass(UnscoredArm, EmptyArm)
    assert not issubclass(UnscoredArm, IncompleteArm)


# --------------------------------------------------------------------------- empty arm (T19d)
#
# Incident, 2026-09-16: `pi train gate` wrote two complete verdicts when its checkpoint arm had
# ZERO runs in the parquet -- a staging step upstream had matched nothing -- so every criterion
# was computed on an empty arm against the base alone. This is NOT the case
# `test_a_missing_criterion_fails_rather_than_passes` covers: there, the arm still has real runs
# and one metric's score is absent. Here `_select_runs` itself returns nothing, which is not a
# criterion that can fail on its own terms -- there is no arm to measure -- so `run_gate` must
# refuse before it computes anything, not let each criterion discover the emptiness separately
# and disagree about what it means. A verdict on an empty arm is a wrong published number
# waiting to happen.


def test_run_gate_raises_empty_arm_when_the_checkpoint_selects_zero_runs(tmp_path) -> None:
    from pinq_train.gate import EmptyArm

    specs = [s for s in _pairs(n=12) if s["arm"] == BASE]  # no checkpoint-arm rows at all
    d = _build(tmp_path, specs)
    with pytest.raises(EmptyArm) as e:
        _gate(d)
    msg = str(e.value)
    assert "inquirer_trained" in msg  # the empty arm, named
    assert "dev_select_musique" in msg  # the grid it was selected from, named
    assert str(d) in msg  # the parquet dir, named
    assert "musique" in msg  # the suite (read off the baseline, which is not empty), named


def test_run_gate_raises_empty_arm_when_the_baseline_selects_zero_runs(tmp_path) -> None:
    from pinq_train.gate import EmptyArm

    specs = [s for s in _pairs(n=12) if s["arm"] == CKPT]  # no baseline-arm rows at all
    d = _build(tmp_path, specs)
    with pytest.raises(EmptyArm) as e:
        _gate(d)
    msg = str(e.value)
    assert "inquirer_prompted" in msg  # the empty arm, named
    assert "dev_baseline_musique" in msg  # the grid it was selected from, named
    assert str(d) in msg  # the parquet dir, named
    assert "musique" in msg  # the suite (read off the checkpoint, which is not empty), named


def test_the_gate_cli_refuses_and_writes_nothing_when_the_checkpoint_arm_is_empty(
    tmp_path, capsys
) -> None:
    from pi_run.cli import build_parser

    specs = [s for s in _pairs(n=12) if s["arm"] == BASE]
    d = _build(tmp_path, specs)
    out = tmp_path / "verdict.json"
    args = build_parser().parse_args(
        [
            "train",
            "gate",
            "--parquet-dir",
            str(d),
            "--grid-name",
            "dev_select_musique",
            "--baseline-grid-name",
            "dev_baseline_musique",
            "--scorer-hash",
            SCORER,
            "--out",
            str(out),
        ]
    )
    rc = args.fn(args)
    assert rc != 0
    assert not out.exists(), "a refused gate must write nothing to --out"
    err = capsys.readouterr().err
    assert "GATE REFUSED" in err
    assert str(d) in err
    assert "dev_select_musique" in err
    assert "inquirer_trained" in err
    assert "musique" in err


def test_the_gate_cli_refuses_and_writes_nothing_when_the_baseline_arm_is_empty(
    tmp_path, capsys
) -> None:
    from pi_run.cli import build_parser

    specs = [s for s in _pairs(n=12) if s["arm"] == CKPT]
    d = _build(tmp_path, specs)
    out = tmp_path / "verdict.json"
    args = build_parser().parse_args(
        [
            "train",
            "gate",
            "--parquet-dir",
            str(d),
            "--grid-name",
            "dev_select_musique",
            "--baseline-grid-name",
            "dev_baseline_musique",
            "--scorer-hash",
            SCORER,
            "--out",
            str(out),
        ]
    )
    rc = args.fn(args)
    assert rc != 0
    assert not out.exists(), "a refused gate must write nothing to --out"
    err = capsys.readouterr().err
    assert "GATE REFUSED" in err
    assert str(d) in err
    assert "dev_baseline_musique" in err
    assert "inquirer_prompted" in err
    assert "musique" in err


def test_the_verdict_records_both_arms_run_counts_in_selection(tmp_path) -> None:
    """The fix: a reader of `selection` alone -- without cross-referencing `pairing` -- can see
    how many runs each arm actually contributed, which is exactly the number `run_gate` itself
    must check before computing anything. `pairing`'s counts (kept, unchanged) describe what
    survived task-pairing; these are the raw per-arm counts from selection.
    """
    v = _gate(_build(tmp_path, _pairs()))
    assert v["selection"]["n_checkpoint_runs"] == 12
    assert v["selection"]["n_baseline_runs"] == 12
    assert v["selection"]["n_checkpoint_runs"] == v["pairing"]["n_checkpoint_runs"]
    assert v["selection"]["n_baseline_runs"] == v["pairing"]["n_baseline_runs"]


# --------------------------------------------------------------------------- the verdict


def test_the_gate_exits_zero_when_every_gated_criterion_passes(tmp_path) -> None:
    v = _gate(_build(tmp_path, _pairs()))
    assert v["passed"] is True, {k: c for k, c in v["criteria"].items() if not c["passed"]}


def test_the_gate_cli_exits_one_on_a_failure_and_writes_its_verdict(tmp_path) -> None:
    from pi_run.cli import build_parser

    d = _build(tmp_path, _pairs(ckpt_over={"coverage": 0.5}, base_over={"coverage": 0.5}))
    out = tmp_path / "verdict.json"
    args = build_parser().parse_args(
        [
            "train",
            "gate",
            "--parquet-dir",
            str(d),
            "--grid-name",
            "dev_select_musique",
            "--baseline-grid-name",
            "dev_baseline_musique",
            "--scorer-hash",
            SCORER,
            "--out",
            str(out),
        ]
    )
    assert args.fn(args) == 1
    v = json.loads(out.read_text())
    assert v["passed"] is False
    assert v["criteria"]["evidence_coverage"]["passed"] is False


def test_the_verdict_records_its_bootstrap_seed_and_resample_count(tmp_path) -> None:
    """A CI nobody can reproduce is not a CI.

    Still an EXACT comparison of the three reproducibility fields: `clustering` is lifted out
    and checked on its own below, and any OTHER new key would still break the equality, which
    is the drift this test exists to catch.
    """
    v = _gate(_build(tmp_path, _pairs()), bootstrap_seed=7, n_resamples=250)
    b = dict(v["bootstrap"])
    clustering = b.pop("clustering")
    assert b == {"seed": 7, "n_resamples": 250, "unit": "(suite_id, task_id)"}
    assert clustering.startswith("TASK-LEVEL, AND EXPLICITLY NOT TEMPLATE-CLUSTERED")


def test_the_verdict_says_its_unit_is_not_the_template_clustering_and_where_that_one_lives(
    tmp_path,
) -> None:
    """`unit: "(suite_id, task_id)"` is accurate and was still misread. "task-clustered" reads
    as "clustered" to anyone who does not happen to know that one suite templates several of
    its tasks, and a pooled number under a clustered heading is the mislabelling this project
    has already had to correct once. So the verdict has to deny the reading in words.

    The gate's task-level unit is CORRECT for a selection gate choosing between checkpoints at
    task level, and nothing here changes it -- `pairing["key"]` and `bootstrap["unit"]` above
    pin that behaviour. What is checked here is that the verdict cannot be mistaken for the
    other estimand, and that it names where the other one is computed properly:
    `pi_eval.report` passes the template cluster map unconditionally, so a figure taken through
    the reporting path is template-clustered without anyone having to remember to ask.
    """
    c = _gate(_build(tmp_path, _pairs()))["bootstrap"]["clustering"]
    assert "NOT TEMPLATE-CLUSTERED" in c
    # names the estimand it is NOT, and the suites where the two actually differ
    assert "cluster_bootstrap" in c and "template_id" in c
    for suite in ("musique", "tau2", "tau2_golden"):
        assert suite in c, f"{suite} clusters non-trivially and must be named"
    # gives the magnitude on both ends, so no reader bounds it from the cluster ratio
    assert "0.000816" in c and "0.011803" in c
    # and says where the clustered number comes from correctly
    assert "pi_eval.report" in c


def test_k_is_stamped_per_suite_from_the_grid_files(tmp_path) -> None:
    """`k` is NOT a column of runs.parquet, so whether the two arms ran at the same retrieval
    budget can only be read off the grids. It is REPORTED and never gated: a mismatch makes the
    delta a k contrast too, which a reader must be told rather than have decided for them."""
    groot = tmp_path / "grids"
    groot.mkdir()
    (groot / "a.yaml").write_text(
        "name: dev_select_musique\nsuites: [musique]\narms: [inquirer_trained]\nk: 5\n"
    )
    (groot / "b.yaml").write_text(
        "name: dev_baseline_musique\nsuites: [musique]\narms: [inquirer_prompted]\nk: 5\n"
    )
    v = _gate(_build(tmp_path, _pairs()), grids_root=groot)
    assert v["k_matches_baseline"] == {"musique": True}

    (groot / "a.yaml").write_text(
        "name: dev_select_musique\nsuites: [musique]\narms: [inquirer_trained]\nk: 2\n"
    )
    v2 = _gate(_build(tmp_path / "x", _pairs()), grids_root=groot)
    assert v2["k_matches_baseline"] == {"musique": False}
    assert v2["passed"] == v["passed"], "the k stamp is reported, never gated"


# --------------------------------------------------------------------------- the coverage rule
#
# D9 (2026-09-15): the COVERAGE criterion is the matched-cost contrast -- the checkpoint at its
# own natural stop k against the BASELINE'S OWN PREFIX LADDER at the same k, read from
# `scores.frontier_q#k`. The cap-8 delta stays in the verdict beside it as the "asks less" cost
# line and is not gated. `--coverage-rule cap8` restores the old rule exactly.
#
# WHY THE PREFIX IS A LEGITIMATE COMPARATOR: `Inquirer.act(s: State)` takes no budget, so the
# base's k-th question is a function of the state alone and the first k questions of a cap-8 run
# are what a cap-k run would have asked. A budget-aware policy would break this, which is why
# that parameter does not exist.

_TEMPLATES = (
    "who wrote the {t} novel first",
    "when did that {t} author die",
    "where was this {t} book printed",
    "which press issued {t} in paperback",
    "how many {t} editions exist today",
    "why did critics call {t} difficult",
    "what prize did {t} eventually win",
    "whose translation of {t} sold best",
)
"""Eight questions of SIX WORDS EACH, sharing no trigram. Both arms draw from this list, so
within-task distinct-3 is 1.0 and the length TOST sees a relative difference of exactly 0 --
neither of them can be what decides a verdict about coverage."""


def _qn(t, n):
    return tuple(s.format(t=t) for s in _TEMPLATES[:n])


def _prefix_pairs(ck_rungs, ba_rungs, *, n=12, ck_stop="policy_stop", ba_stop="policy_stop"):
    """`n` task-matched pairs, each arm's whole coverage ladder written out.

    `evidence_coverage` is the last rung on both arms, because it is the same quantity (see
    `_ladder`). The base's ladder MUST be monotone -- the seen-uid set only grows, so
    `frontier_q#k` cannot fall -- and that is also why no fixture can make the cap-8 rule pass
    where the matched-cost rule fails: base@k <= base@8, so a run above the terminal is above
    every rung.
    """
    out = []
    for t in range(n):
        out.append(
            _spec(
                arm=CKPT,
                task=f"t{t}",
                questions=_qn(t, len(ck_rungs) - 1),
                ladder=ck_rungs,
                coverage=ck_rungs[-1],
                stop_reason=ck_stop,
                cad=((2, 1.0, 4.0),),
            )
        )
        out.append(
            _spec(
                arm=BASE,
                task=f"t{t}",
                questions=_qn(t, len(ba_rungs) - 1),
                ladder=ba_rungs,
                coverage=ba_rungs[-1],
                stop_reason=ba_stop,
                cad=((2, 0.5, 4.0),),
            )
        )
    return out


# The perfect stopper: complete evidence after three questions, against a base that reaches the
# same place in eight. At cap 8 the two arms tie and the gain rule fails; at matched cost the
# checkpoint is 0.40 ahead of where the base stood after three questions.
STOPS_EARLY = ((0.0, 0.4, 0.7, 1.0), (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
# The mirror: it stops at three holding 0.50, where the base already held 0.70.
STOPS_SHORT = ((0.0, 0.2, 0.3, 0.5), (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0))


def test_the_matched_cost_rule_passes_the_early_stopper_the_cap8_rule_fails(tmp_path) -> None:
    """(D9) The checkpoint holds 1.00 after three questions; the base holds 0.60 there and
    needs eight to reach 1.00. The cap-8 contrast scores that 0.00 -- it charges the checkpoint
    for stopping and credits it nothing for the five questions it did not buy -- and the gain
    rule fails on it. The matched-cost contrast scores it +0.40.
    """
    d = _build(tmp_path, _prefix_pairs(*STOPS_EARLY))

    m = _crit(_gate(d), "evidence_coverage")
    assert m["value"] == pytest.approx(0.4), m
    assert m["ci_lo"] > 0 and m["passed"] is True
    assert m["comparator"] == "matched_k"

    c8 = _crit(_gate(d, coverage_rule="cap8"), "evidence_coverage")
    assert c8["value"] == pytest.approx(0.0)
    assert c8["passed"] is False


def test_the_matched_cost_rule_still_fails_a_checkpoint_behind_at_its_own_k(tmp_path) -> None:
    """The mirror, and the whole point of gating it: the checkpoint stops at three holding
    0.50 where the base already held 0.70, so buying fewer questions bought worse ones. Both
    rules fail it -- a cap-8 pass with a matched-cost failure is impossible on a monotone
    ladder, so this is the only mirror there is.
    """
    d = _build(tmp_path, _prefix_pairs(*STOPS_SHORT))

    m = _crit(_gate(d), "evidence_coverage")
    assert m["value"] == pytest.approx(-0.2), m
    assert m["passed"] is False
    assert _gate(d)["passed"] is False

    c8 = _crit(_gate(d, coverage_rule="cap8"), "evidence_coverage")
    assert c8["value"] == pytest.approx(-0.5)
    assert c8["passed"] is False


# The checkpoint asks SIX questions; the baseline asks only TWO. `_matched_cost` reads the
# checkpoint at its own full terminal value (gate.py:988 `k = n_asks.get(c_rid, 0)`, gate.py:1004
# `deltas.append(float(c_val) - _mean(rungs))` -- `c_val` is never truncated) and the baseline at
# `min(k, b_n)` (gate.py:994). That is a matched reading only when the baseline asks AT LEAST
# `k` -- every real headline contrast (trained ~2-3 against prompted ~6 or teacher ~5.3). Here
# the baseline is the SHORTER arm, so `min(6, 2)` resolves to the baseline's own two-question
# terminal while the checkpoint keeps its six-question terminal: two different budgets, reported
# as one delta with no flag.
CK_OUTSPENDS_BASELINE = ((0.0, 0.2, 0.4, 0.6, 0.7, 0.75, 0.8), (0.0, 0.3, 0.5))


def test_the_matched_cost_rule_refuses_a_checkpoint_that_outspends_its_baseline(
    tmp_path,
) -> None:
    """Pins the asymmetry Lane L2.3 found reaching a published cell (Persistence-only asks
    6.4-6.9 questions against the prompted baseline's ~6, so a majority of its StrategyQA
    task-pairs hit exactly this branch: `n_baseline_shorter_than_k` 138 of 200).

    The coordinator's convention (2026-09-19): keep the ORIGINAL arithmetic exactly as it was
    (`0.8 - 0.5 = +0.3`, the checkpoint's own six-question terminal minus the baseline's own
    two-question terminal -- still computed, still the `value` on this criterion, so every
    already-published verdict reproduces bit for bit), ADD a symmetric reading beside it under
    its own key (both arms at the shared, smaller budget: `0.4 - 0.5 = -0.1`), and make the GATE
    REFUSE to emit a pass or fail on the pooled criterion rather than silently trusting the
    unmatched `+0.3`.
    """
    d = _build(tmp_path, _prefix_pairs(*CK_OUTSPENDS_BASELINE))
    v = _gate(d)
    m = _crit(v, "evidence_coverage")

    # The original arithmetic is untouched: `value` is still the unmatched +0.3.
    assert m["value"] == pytest.approx(0.3)
    # But it is REFUSED, not passed or failed.
    assert m["passed"] is None
    assert m["refused"] is True
    assert "musique" in m["outspent_suites"]
    assert "not matched" in m["reason"] and "symmetric" in m["reason"]
    assert v["passed"] is False, "a refused gated criterion must not let the overall gate pass"

    # Both readings are present. The symmetric one lives beside `value`, in `matched_cost`.
    cell = v["matched_cost"]["by_suite"]["musique"]
    assert cell["outspent_comparator"] is True
    assert cell["symmetric"]["delta"] == pytest.approx(-0.1)
    assert cell["symmetric"]["n_tasks"] == cell["n_tasks"]
    assert cell["symmetric"]["mean_k_common"] == pytest.approx(2.0)


def test_a_safe_arms_matched_cost_verdict_is_byte_identical_before_and_after_the_symmetric_addition(
    tmp_path,
) -> None:
    """The property that protects every already-published number: for an arm that spends LESS
    than its comparator (every headline arm), adding the symmetric reading and the refusal
    check must not move a single pre-existing value, and must not refuse it either.
    """
    v = _gate(_build(tmp_path, _prefix_pairs(*STOPS_EARLY)))
    m = _crit(v, "evidence_coverage")

    # Exactly the values `test_the_matched_cost_rule_passes_the_early_stopper_the_cap8_rule_fails`
    # and `test_the_verdict_records_the_rule_and_reports_the_cap8_delta_beside_it` already pin.
    assert m["value"] == pytest.approx(0.4)
    assert m["passed"] is True
    assert "refused" not in m
    assert "outspent_suites" not in m
    assert v["passed"] is True

    cell = v["matched_cost"]["by_suite"]["musique"]
    assert cell["outspent_comparator"] is False
    assert cell["trained_mean_k"] == pytest.approx(3.0)
    assert cell["baseline_mean_k_charged"] == pytest.approx(3.0)
    assert cell["delta"] == pytest.approx(0.4)
    assert cell["cap8_coverage_delta"] == pytest.approx(0.0)
    # The new, additive key is present and correct, but changes nothing above it.
    assert cell["symmetric"]["delta"] == pytest.approx(0.4)


def test_the_verdict_records_the_rule_and_reports_the_cap8_delta_beside_it(tmp_path) -> None:
    """The cost line. "Fewer questions for the same evidence" is an ORDERED PAIR, so the cap-8
    delta the rule replaces is reported per suite next to the matched one, with the mean k that
    matched them -- ungated, because D9 gates the matched contrast and nothing else."""
    v = _gate(_build(tmp_path, _prefix_pairs(*STOPS_EARLY)))
    assert v["coverage_rule"] == "matched_cost"

    mc = v["matched_cost"]["by_suite"]["musique"]
    assert mc["n_tasks"] == 12
    assert mc["trained_mean_k"] == pytest.approx(3.0)
    assert mc["baseline_mean_k_charged"] == pytest.approx(3.0)
    assert mc["delta"] == pytest.approx(0.4)
    assert mc["ci_lo"] > 0 and mc["ci_hi"] > 0
    assert mc["cap8_coverage_delta"] == pytest.approx(0.0)
    assert mc["trained_mean_n_asks"] == pytest.approx(3.0)
    assert mc["baseline_mean_n_asks"] == pytest.approx(8.0)
    assert v["matched_cost"]["gated"] is True


def test_the_cap8_rule_reports_no_matched_cost_block(tmp_path) -> None:
    """Under `cap8` the matched-cost block is ABSENT, not null-filled: computing it would let a
    rule the operator did not select refuse the run on its ladder check."""
    v = _gate(_build(tmp_path, _prefix_pairs(*STOPS_EARLY)), coverage_rule="cap8")
    assert v["coverage_rule"] == "cap8"
    assert "matched_cost" not in v


def test_a_baseline_run_shorter_than_k_is_charged_its_own_length(tmp_path) -> None:
    """`min(k, n_asks)`, not k. A base run that stopped at two under cap 8 stops at two under
    cap 3 as well -- it is budget-blind, so the cap cannot make it ask more -- and charging it
    three would invent spend it never made in the very comparison whose subject is spend."""
    specs = _prefix_pairs(*STOPS_EARLY, n=12)
    short = (0.0, 0.3, 0.55)  # two asks, then it stopped by itself
    for s in specs:
        if s["arm"] == BASE and s["task"] in ("t0", "t1", "t2"):
            s["ladder"], s["coverage"], s["questions"] = short, short[-1], _qn(s["task"], 2)
    v = _gate(_build(tmp_path, specs))
    mc = v["matched_cost"]["by_suite"]["musique"]
    assert mc["n_baseline_shorter_than_k"] == 3
    assert mc["baseline_mean_k_charged"] == pytest.approx((9 * 3 + 3 * 2) / 12)
    # nine tasks at 1.00 - 0.60, three at 1.00 - 0.55
    assert mc["delta"] == pytest.approx((9 * 0.4 + 3 * 0.45) / 12)


def test_the_gate_refuses_a_ladder_whose_last_rung_is_not_the_runs_own_coverage(tmp_path) -> None:
    """THE GATE'S HALF OF THE INSTRUMENT LOCK. `scripts/matched_cost.py` proves the ladder is
    the verdicts' instrument by re-running the real matcher against gold; contract 3 forbids
    this module the imports that takes, so the gate checks the one thing parquet alone can
    decide: `frontier_q#n_asks` IS `evidence_coverage`, the same number read twice. Where they
    disagree the prefix is not the quantity the delta is in, and no matched-cost number is
    reportable -- a refusal, not a warning.
    """
    from pinq_train.gate import LadderInconsistent

    specs = _prefix_pairs(*STOPS_EARLY)
    for s in specs:
        if s["arm"] == BASE and s["task"] == "t4":
            s["coverage"] = 0.42  # its ladder still ends at 1.0

    d = _build(tmp_path, specs)
    with pytest.raises(LadderInconsistent) as e:
        _gate(d)
    msg = str(e.value)
    assert "frontier_q#8" in msg and "0.42" in msg and "evidence_coverage" in msg

    # ... and the rule that never reads the ladder is not refused by it.
    assert _crit(_gate(d, coverage_rule="cap8"), "evidence_coverage")["n"] == 12


CAP8_CRITERIA_SHA256 = "cf1e6268b3d2d83ac36f724cb5bd6e9d07f0fb202fd5e71ee7f5af6c69da8a0b"
"""sha256 of `json.dumps(run_gate(...)["criteria"], sort_keys=True)` over `_pairs()` at
`n_resamples=200`, measured on commit f3f3f0f -- the gate as it stood before `--coverage-rule`
existed. A digest and not a transcription because the criteria dict carries whole paragraphs of
column prose: the claim is that `cap8` moves NOTHING, the prose included."""
# 2026-09-16, T19c: re-pinned. The malformed threshold string changed (built from
# MALFORMED_TOLERANCE, "<= max(baseline, 0.005)"); every other byte of the criteria block is unchanged.
# 2026-09-18: re-pinned again, from 0d4a0ea0e38d5515 to c36ee26c8f30baa8. WHICH BELIEF CHANGED.
# The old value encoded "the recorded verdict is the correct verdict, so cap8 must reproduce it
# byte for byte". That belief is now false for ONE field: `length_equivalence.ci_lo`/`ci_hi` came
# from `_length` resampling its questions in `_questions`' arrival order, and that order is
# canonical over KEYS (`ORDER BY task_id, question`) and says nothing about VALUES, so the
# endpoints were a property of what one query happened to return. `_length` now sorts before
# resampling, as `bca_ci` and `pi_eval.stats.inference.cluster_bootstrap` already did. On this
# fixture (12 four-word and 12 five-word questions per arm) that moves the interval from
# [-0.05454545454545455, +0.04807692307692322] to [-0.06250000000000006, +0.047619047619047554].
# PROVEN to be the only change, not assumed: substituting those two old endpoints back into the
# new criteria dict and re-hashing reproduces 0d4a0ea0e38d5515 exactly, so every other byte of the
# block -- every value, every verdict, every paragraph of column prose -- is untouched. Re-pinned
# rather than relaxed to a tolerance: a golden test with a corrected golden value is still a
# golden test.
# 2026-09-18, lane L1.1: re-pinned again, from c36ee26c8f30baa8 to 5a2fb787e57cd3a2. THE SHAPE OF
# `criteria["distinct3"]` CHANGED, not any recorded value: it now also carries `by_seed` (within
# -task-AND-seed distinct-3, see `within_task_seed_distinct_n`) and `n_seeds`, added because
# `value` pools every seed of a task together and a checkpoint evaluated at N seeds that repeats
# its own questions across them is scored as if it had collapsed by a factor of N -- see
# `test_distinct3_by_seed_is_not_halved_by_a_second_seed_that_repeats_the_first`. `value` keeps
# the exact float it always had (`_pairs()` runs one seed, so `by_seed` also happens to equal
# `value` here -- 1.0 -- which is expected, not a coincidence: with one seed the two groupings
# coincide). PROVEN to be the only change: popping `distinct3.by_seed` (1.0) and `distinct3.n_seeds`
# (1) back out of the new criteria dict and re-hashing reproduces c36ee26c8f30baa8 exactly, so
# every other byte -- every other criterion, every interval, every paragraph of prose -- is
# untouched.
# 2026-09-19, `artifacts/degenerate_metrics_20260919`: re-pinned again, from
# 5a2fb787e57cd3a2fbd1550d692855aa626e34122271daa72ee513712de565e5 to
# cf1e6268b3d2d83ac36f724cb5bd6e9d07f0fb202fd5e71ee7f5af6c69da8a0b. `criteria["distinct3"]
# ["passed"]` now GATES ON `by_seed` (falling back to `value` only when `by_seed` cannot be
# measured), not on the seed-pooled `value` -- see
# `test_distinct3_passes_on_by_seed_when_only_the_seed_pooled_value_fails_the_floor`. On THIS
# fixture (`_pairs()` runs one seed) `by_seed` equals `value`, so `passed` does not move here;
# the only byte that moved is the `measure` string, which now says "GATED ON by_seed". PROVEN
# to be the only change: substituting that string back to its pre-fix text and re-hashing
# reproduces 5a2fb787e57cd3a2fbd1550d692855aa626e34122271daa72ee513712de565e5 exactly, so every
# recorded value, every other criterion and every paragraph of prose besides that one string is
# untouched. Verdicts already on disk that failed `distinct3` on seed-pooled grounds alone do
# NOT retroactively become correct; this only changes what a FUTURE `run_gate` call computes.


def test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit(tmp_path) -> None:
    """The old rule is still the old rule: every criterion, every interval, every string."""
    import hashlib

    v = _gate(_build(tmp_path, _pairs()), coverage_rule="cap8")
    blob = json.dumps(v["criteria"], sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == CAP8_CRITERIA_SHA256, blob
    c = _crit(v, "evidence_coverage")
    assert (c["value"], c["ci_lo"], c["ci_hi"], c["n"]) == (
        0.09999999999999998,
        0.04999999999999999,
        0.14999999999999997,
        12,
    )
    assert v["passed"] is True


def test_a_verdict_carries_the_graph_version_beside_the_scorer_hash(tmp_path) -> None:
    """CONTRIBUTING.md rule 1: the provenance triple is (run_id, scorer_hash, graph_version), and a
    number missing any element is not a result. `scores.parquet` carries `graph_version` on
    every scored row -- measured on
    artifacts/gate/8b2-t20/parquet_qwen3-8b-dpo-headline-rater/scores.parquet, one distinct
    value "v1" -- and the verdict writer copied only `scorer_hash`, so no verdict on disk
    satisfied the rule and every table sourced from one was an element short.

    It belongs in `selection`, which is unhashed, and NOT in `criteria`, which
    `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` hashes byte for byte:
    adding a field there would move that hash and change the identity of every verdict ever
    written. `selection` is also where the module docstring says a reader looks to learn
    "which rows", and a graph version is a property of the rows, not of a threshold.
    """
    v = _gate(_build(tmp_path, _pairs()))
    assert v["selection"]["graph_version"] == GRAPH
    assert v["selection"]["scorer_hash"] == SCORER
    # All three elements readable from the verdict alone, without opening the parquet.
    assert v["selection"]["parquet_dir"]
    # The hashed block must not have acquired the field.
    assert "graph_version" not in json.dumps(v["criteria"])


def test_two_graph_versions_in_one_parquet_are_refused_not_averaged(tmp_path) -> None:
    """A graph version is an identity, not a measurement: there is no mean of "v1" and "v2"
    and no defensible rule for picking one. Two distinct versions over the runs a verdict
    selected means the parquet was scored under two different graphs, so the verdict cannot
    name the graph its numbers came from -- the same reason `EmptyArm` and `LadderInconsistent`
    raise instead of writing a verdict with a footnote.

    Measured before writing this: 0 of 64 gate `scores.parquet` files carry more than one
    `graph_version`, so this refusal is a guard against a future mixed store, not a
    description of one on disk today.
    """
    from pinq_train.gate import GraphVersionAmbiguous

    d = _build(tmp_path, _pairs())
    t = pq.read_table(d / "scores.parquet").to_pylist()
    assert {r["graph_version"] for r in t} == {GRAPH}
    t[0]["graph_version"] = "v2"
    pq.write_table(pa.Table.from_pylist(t), d / "scores.parquet")

    with pytest.raises(GraphVersionAmbiguous, match="graph_version"):
        _gate(d)


def test_an_unknown_coverage_rule_is_refused_rather_than_defaulted(tmp_path) -> None:
    """Silently falling back to a default would put a verdict in `artifacts/` whose
    `coverage_rule` field says one thing and whose number says another."""
    with pytest.raises(ValueError, match="coverage_rule"):
        _gate(_build(tmp_path, _pairs()), coverage_rule="matched-cost")


# The intervals `pi_eval.stats.inference.cluster_bootstrap` returns for singleton clusters at
# n_boot=1000, seed=0, measured directly:
#
#   .venv/bin/python -c "from pi_eval.stats.inference import cluster_bootstrap as cb; \
#       print(cb([[0.1*i] for i in range(10)], n_boot=1000, seed=0))"
#
# `pinq_train` may not import that module (contract 3), so the BCa is replicated in the gate and
# pinned here against what the original produces -- the same relationship `cad_ge2` has with
# `pi_eval`'s. Not a hand computation: a BCa endpoint is a bootstrap distribution, a bias
# correction and a jackknife acceleration, and a "hand" value would be this code's own
# arithmetic written out twice. What a hand value CAN pin is the degenerate case, which the
# third row is: every unit identical, so every resample is that same number.
BCA_REFERENCE = (
    ([0.1 * i for i in range(10)], (0.45, 0.27, 0.6095765495386787)),
    ([0.0] * 7 + [0.1, 0.2, 1.7], (0.2, 0.01, 0.8162937036395408)),
    ([0.2] * 12, (0.20000000000000004, 0.20000000000000004, 0.20000000000000004)),
)


def test_the_replicated_bca_does_not_depend_on_the_order_the_values_arrive_in() -> None:
    """The replica inherits every property of the original, this one included.

    `bca_ci` resamples by INDEX, so a permutation of the same values moved its endpoints -- and
    its inputs here are built from parquet row order (`by_point.values()` in `_fork_report`), not
    from a sorted key. See `pi_eval.stats.inference.cluster_bootstrap` for the measurement that
    found this on FRAMES. Exact equality: a one-step difference is the defect, not the tolerance.
    """
    from pinq_train.gate import bca_ci

    rng = random.Random(31)
    values = [float(rng.random() < 0.35) for _ in range(150)]
    want = bca_ci(values, seed=0, n_resamples=2000)
    for perm_seed in (1, 2, 3):
        shuffled = list(values)
        random.Random(perm_seed).shuffle(shuffled)
        assert bca_ci(shuffled, seed=0, n_resamples=2000) == want, f"shuffle {perm_seed}"


def test_the_length_bootstrap_does_not_depend_on_the_order_the_questions_arrive_in(
    monkeypatch,
) -> None:
    """Same defect class as `bca_ci` above, in `_length`, and fixed the same way.

    MEASURED BEFORE THE FIX, 2026-09-18. All 182 recorded `criteria.length_equivalence` intervals
    under `artifacts/` reproduce bit for bit under the arrival-order rule, and ALL 182 move under
    the value-sorted rule; zero of them arrive already value-sorted, and the two means and `n`
    move on none of them. The audit note that opened this claimed `qwen3-8b-sft-sw01.musique`
    reproduced [-0.134063, -0.066874] identically either way. It does not: sorted, that cell is
    [-0.134664, -0.068256]. So unlike `bca_ci`'s sort, this one is a PUBLISHED CORRECTION, and the
    numbers it moves are listed on the sort itself in `_length`.

    `_length` resamples question INDICES, so the arrival order of its per-arm lists was an input
    to its endpoints. Those lists come from `_questions`, whose `ORDER BY task_id, question` is
    INCIDENTAL: it is the order that one query happens to return, not the value order the
    endpoints depend on. Add a column to that ORDER BY, change the join, or feed it a suite whose
    task ids sort differently, and the published bounds move with no code change and no signal.
    That is exactly the surprise `cluster_bootstrap` and `bca_ci` produced before they sorted.

    A resampling unit here is ONE question's word count -- a scalar -- so there is no inner level
    to canonicalise: permuting arrival reorders units and cannot reorder values within a unit.
    The three order-insensitive facts are asserted beside the endpoints to pin that: both means
    and `n` are properties of the multiset and must not move either.

    Exact equality on all four numbers. A one-step difference is the defect, not the tolerance.
    """
    from pinq_train import gate

    rng = random.Random(7)
    ck = [(f"t{i:03d}", " ".join(["w"] * rng.randint(4, 30))) for i in range(120)]
    ba = [(f"t{i:03d}", " ".join(["w"] * rng.randint(4, 30))) for i in range(120)]

    def _stub(order_seed):
        c, b = list(ck), list(ba)
        if order_seed is not None:
            random.Random(order_seed).shuffle(c)
            random.Random(order_seed + 1000).shuffle(b)
        # The permutation must not change WHAT was asked, only when it arrived.
        assert sorted(c) == sorted(ck) and sorted(b) == sorted(ba)
        return lambda con, runs: c if runs == ["CK"] else b

    def _run():
        return gate._length(
            None, ["CK"], ["BA"], margin=0.10, seed=0, n_resamples=2000, rule="not_longer"
        )

    monkeypatch.setattr(gate, "_questions", _stub(None))
    want = _run()
    assert not math.isnan(want["ci_lo"]) and not math.isnan(want["ci_hi"])
    for order_seed in (1, 2, 3):
        monkeypatch.setattr(gate, "_questions", _stub(order_seed))
        got = _run()
        keys = ("ci_lo", "ci_hi", "value", "baseline", "n")
        assert [got[k] for k in keys] == [want[k] for k in keys], (
            f"permutation {order_seed}: "
            + ", ".join(f"{k} {got[k]!r} != {want[k]!r}" for k in keys if got[k] != want[k])
        )


def test_the_replicated_bca_is_the_interval_pi_eval_would_have_returned() -> None:
    from pinq_train.gate import bca_ci

    for values, want in BCA_REFERENCE:
        got = bca_ci(values, seed=0, n_resamples=1000)
        assert got == want, f"{values[:3]}...: {got} != {want}"


def test_the_coverage_rule_reaches_the_cli(tmp_path) -> None:
    """The flag, the default, and the field a reader of `artifacts/gate/*.json` checks first."""
    from pi_run.cli import build_parser

    d = _build(tmp_path, _prefix_pairs(*STOPS_EARLY))

    def run(*extra):
        out = tmp_path / f"verdict{len(extra)}.json"
        args = build_parser().parse_args(
            [
                "train",
                "gate",
                "--parquet-dir",
                str(d),
                "--grid-name",
                "dev_select_musique",
                "--baseline-grid-name",
                "dev_baseline_musique",
                "--scorer-hash",
                SCORER,
                "--out",
                str(out),
                *extra,
            ]
        )
        return args.fn(args), json.loads(out.read_text())

    code, v = run()
    assert v["coverage_rule"] == "matched_cost", "the default is the rule D9 decided"
    assert v["criteria"]["evidence_coverage"]["value"] == pytest.approx(0.4)
    assert v["matched_cost"]["by_suite"]["musique"]["cap8_coverage_delta"] == pytest.approx(0.0)
    assert code == 0

    code8, v8 = run("--coverage-rule", "cap8")
    assert v8["coverage_rule"] == "cap8"
    assert v8["criteria"]["evidence_coverage"]["value"] == pytest.approx(0.0)
    assert code8 == 1, "the same parquet, failed on the rule the operator asked for"


def test_a_facet_loss_is_gated_at_cap8_and_report_only_at_matched_cost(tmp_path) -> None:
    """The D9 follow-up (2026-09-15). `facet_breadth` scales with the NUMBER OF ASKS exactly as
    cap-8 coverage does, so gating it fails every trained checkpoint for asking less -- the very
    confound the matched-cost rule removes -- and unlike coverage it cannot be moved to matched
    k, because `scores` carries `facet_breadth` as one terminal metric and no `facet_breadth#k`
    ladder. Under `matched_cost` it is therefore REPORTED, with its value, interval, n and the
    no-loss outcome it would have had, and it does not decide the verdict. Tier B is checkpoint
    SELECTION; the Tier-C preregistered endpoint is untouched.

    The fixture is the default pair set, where the two arms ask the same number of questions, so
    the coverage criterion passes under BOTH rules and facet breadth is the only thing that can
    move the verdict.
    """
    d = _build(tmp_path, _pairs(ckpt_over={"facet": 0.2}, base_over={"facet": 0.5}))

    v8 = _gate(d, coverage_rule="cap8")
    f8 = _crit(v8, "facet_breadth")
    assert (f8["gated"], f8["passed"]) == (True, False)
    assert v8["facet_rule"] == "no_loss"
    assert v8["passed"] is False, "at cap 8 a facet loss still fails the gate"

    v = _gate(d)
    f = _crit(v, "facet_breadth")
    assert f["gated"] is False
    assert f["value"] == pytest.approx(-0.3), "the number stays in the JSON; only the rule moves"
    assert f["ci_lo"] <= f["value"] <= f["ci_hi"] and f["n"] == 12
    assert f["no_loss"] is False, "what the cap-8 rule would have said, kept rather than dropped"
    assert v["facet_rule"] == "report_only"
    assert v["passed"] is True, {
        k: c for k, c in v["criteria"].items() if c["gated"] and not c["passed"]
    }


# ------------------------------------------------- the three matched-cost follow-ups (T19b)
#
# D9 follow-up, 2026-09-15. Under `--coverage-rule matched_cost` ONLY, three criteria stop
# deciding the verdict on a comparator the rule has already been shown to confound:
#
#   * `cad_ge2` -- report-only, for the reason `facet_breadth` already is. Measured at matched
#     k by `scripts/matched_cost.py` on the gold side, all six 8B intervals SPAN 0, so the
#     cap-8 loss of -0.035..-0.177 is the asks-count confound and not a depth loss.
#   * `length_equivalence` -- ONE-SIDED. A shorter question cannot manufacture the retrieval
#     the criterion exists to catch, so only "longer than the base by more than the margin"
#     fails.
#   * `stop_2x2` -- report-only. The prompted base almost never stops, so its P(ASK|not done)
#     is ~1 by construction and "both cells >= the base" is unreachable by a real stopper.
#
# `cap8` is unchanged in all three, and `test_the_cap8_rule_reproduces_the_previous_verdict_
# bit_for_bit` is what says so.


def test_a_depth_loss_is_gated_at_cap8_and_report_only_at_matched_cost(tmp_path) -> None:
    """`cad_ge2` scales with the NUMBER OF ASKS exactly as cap-8 coverage and facet breadth do.

    AND IT CANNOT BE MOVED TO MATCHED k HERE. `scores.parquet` carries `cad#d` / `cad_n#d` as
    TERMINAL per-depth rows and no `cad#d` ladder over prefixes; only `frontier_q` is indexed by
    k. A prefix C@d>=2 needs the node -> gold_depth map, which is gold and which contract 3
    forbids this module. So the number, its interval and the no-loss verdict all stay in the
    JSON, `gated` is false, and `cad_rule` says which rule was in force.
    """
    d = _build(
        tmp_path, _pairs(ckpt_over={"cad": ((2, 0.2, 4.0),)}, base_over={"cad": ((2, 0.5, 4.0),)})
    )

    v8 = _gate(d, coverage_rule="cap8")
    c8 = _crit(v8, "cad_ge2")
    assert (c8["gated"], c8["passed"]) == (True, False)
    assert v8["cad_rule"] == "gain"
    assert v8["passed"] is False, "at cap 8 a depth loss still fails the gate"

    v = _gate(d)
    c = _crit(v, "cad_ge2")
    assert c["gated"] is False
    assert c["value"] == pytest.approx(-0.3), "the number stays in the JSON; only the rule moves"
    assert c["ci_lo"] <= c["value"] <= c["ci_hi"] and c["n"] == 12
    assert c["cap8_cad_ge2_delta"] == c["value"], "what the reported delta is denominated in"
    assert c["no_loss"] is False, "the no-loss verdict, kept rather than dropped"
    assert v["cad_rule"] == "report_only"
    assert v["passed"] is True, {
        k: cc for k, cc in v["criteria"].items() if cc["gated"] and not cc["passed"]
    }


def test_the_reported_cad_no_loss_verdict_is_true_on_a_tie(tmp_path) -> None:
    """The other direction of the kept verdict: a tie is a no-loss pass, a loss is not. A field
    a later rule change would be read off is exercised both ways or it is not pinned."""
    d = _build(
        tmp_path, _pairs(ckpt_over={"cad": ((2, 0.5, 4.0),)}, base_over={"cad": ((2, 0.5, 4.0),)})
    )
    c = _crit(_gate(d), "cad_ge2")
    assert c["value"] == pytest.approx(0.0)
    assert c["no_loss"] is True and c["gated"] is False


def test_a_shorter_checkpoint_passes_at_matched_cost_and_fails_the_tost_at_cap8(tmp_path) -> None:
    """ONE-SIDED, in the confound's direction. The criterion exists because a LONGER question
    retrieves more by accident, so an unchecked length gain reads as a question-quality gain. A
    shorter question cannot do that; it can only make the checkpoint's own numbers harder to
    earn. Two-sided TOST fails it anyway, and on the real gate parquets it did: 8 of 12
    verdicts, every one of them because the trained questions are SHORTER.
    """
    d = _build(tmp_path, _pairs(ckpt_over={"questions": ("who wrote it", "when he died")}))

    v8 = _gate(d, coverage_rule="cap8")
    c8 = _crit(v8, "length_equivalence")
    assert c8["value"] == 3.0 and c8["baseline"] == 4.5
    assert c8["ci_lo"] < -0.10, "the trained questions are a third shorter"
    assert c8["passed"] is False and v8["length_rule"] == "tost"

    v = _gate(d)
    c = _crit(v, "length_equivalence")
    assert (c["value"], c["baseline"]) == (c8["value"], c8["baseline"])
    assert c["passed"] is True, c
    assert c["tost_equivalent"] is False, "the two-sided verdict, kept for the record"
    assert c["gated"] is True, "one-sided, not ungated"
    assert v["length_rule"] == "not_longer"
    assert v["passed"] is True, {
        k: cc for k, cc in v["criteria"].items() if cc["gated"] and not cc["passed"]
    }


def test_a_checkpoint_longer_than_the_margin_fails_under_both_rules(tmp_path) -> None:
    """The direction the criterion is actually for: verbosity, caught before it reaches a
    table. One-sided is not ungated."""
    long_qs = ("a b c d e f g h i j k l", "m n o p q r s t u v w x")
    d = _build(tmp_path, _pairs(ckpt_over={"questions": long_qs}))

    for rule in ("cap8", "matched_cost"):
        v = _gate(d, coverage_rule=rule)
        c = _crit(v, "length_equivalence")
        assert c["value"] == 12.0 and c["baseline"] == 4.5
        assert c["ci_hi"] > 0.10
        assert c["passed"] is False, rule
        assert c["gated"] is True
        assert v["passed"] is False, rule
    assert _crit(_gate(d), "length_equivalence")["tost_equivalent"] is False


def test_a_stop_2x2_loss_is_gated_at_cap8_and_report_only_at_matched_cost(tmp_path) -> None:
    """The prompted base is a DEGENERATE comparator on this axis. Measured on the six gate
    parquets: P(STOP | done) is 0.1525 / 0.1460 at 8B and 0.0066 / 0.0081 at 4B, while
    P(ASK | not done) is 0.9642 / 0.9633 and 0.9985 / 0.9994 -- the second cell is ~1 BECAUSE
    the base almost never stops, so "both cells >= the base" asks a real stopper to keep asking
    as often as a policy that cannot stop. All twelve verdicts failed it.

    Stopping is still selected: the N1 rule (P(ASK | not done) rise >= 0.03 against the
    reference checkpoint AND P(STOP | done) >= 0.85) decides it OUTSIDE the gate. So every cell
    is kept here and none of them decides the verdict.
    """
    d = _build(tmp_path, _stop_specs(ckpt_stops_when_done=True, ckpt_asks_when_not_done=False))

    v8 = _gate(d, coverage_rule="cap8")
    s8 = _crit(v8, "stop_2x2")
    assert (s8["gated"], s8["passed"]) == (True, False)
    assert v8["stop_rule"] == "both_cells_ge_baseline"
    assert v8["passed"] is False, "at cap 8 a stop loss still fails the gate"

    v = _gate(d)
    s = _crit(v, "stop_2x2")
    assert s["gated"] is False
    assert s["both_cells_ge_baseline"] is False, "the cap-8 verdict, kept rather than dropped"
    assert s["value"]["p_stop_given_done"] == 1.0
    assert s["value"]["p_ask_given_not_done"] == pytest.approx(0.8)
    assert s["baseline"]["p_ask_given_not_done"] == 1.0
    assert s["value"]["n_not_done_stop"] == 6
    # 12 runs x 3 decisions: 30 not-done states and 6 done ones, unchanged by the downgrade.
    assert s["value"]["n_states"] == s8["value"]["n_states"] == 36
    assert (s["value"]["n_done"], s["value"]["n_not_done"]) == (6, 30)
    assert s["columns"] == s8["columns"], "every cell and column survives the downgrade"
    assert v["stop_rule"] == "report_only"
    assert v["passed"] is True, {
        k: cc for k, cc in v["criteria"].items() if cc["gated"] and not cc["passed"]
    }


# --------------------------------------------------------------------------- fork grids (T16b)
#
# A FORK GRID is detected when every selected CHECKPOINT run carries a non-empty
# `foreign_trace_sha`. These fixtures use a fresh, minimal spec-builder rather than `_pairs()`:
# `_pairs()`'s `ckpt_over`/`base_over` apply one static override to every task and cannot vary
# `foreign_trace_sha` per fork point, which every test below needs.


def _fork_point_specs(deltas, *, n_prefix=7, base_fu=5, tau_reward=(1.0, 0.0)):
    """One checkpoint/baseline pair per entry of `deltas` (checkpoint follow_ups minus baseline
    follow_ups), each at its OWN fork point: distinct `foreign_trace_sha`, same `seed=0`. Every
    OTHER gated criterion passes trivially (facet/cad/newly ties, matched coverage, an episode
    that completes on its last ask and stops), so these fixtures isolate the fork criteria --
    `distinct3` is the exception (each fork point reuses `_q`'s per-index question pair, so it
    is exercised too, not held fixed)."""
    ck_reward, ba_reward = tau_reward
    specs = []
    for i, d in enumerate(deltas):
        common = dict(
            task=f"fork{i}",
            questions=_q(i),
            done_after="last",
            stop_reason="policy_stop",
            coverage=1.0,
            cad=((2, 1.0, 4.0),),
            facet=0.5,
            newly=0.5,
            foreign_trace_sha=f"fork-trace-{i}",
            foreign_prefix_k=6,
            n_prefix_user_turns=n_prefix,
        )
        specs.append(
            _spec(
                arm=CKPT,
                seed=0,
                n_user_turns=n_prefix + base_fu + d,
                tau_reward=ck_reward,
                **common,
            )
        )
        specs.append(
            _spec(arm=BASE, seed=0, n_user_turns=n_prefix + base_fu, tau_reward=ba_reward, **common)
        )
    return specs


def test_fork_followups_fails_when_the_checkpoint_needs_more_turns_at_every_fork_point(
    tmp_path,
) -> None:
    """At the tip (before this criterion exists) there is no `fork_followups` key at all, so
    accessing it raises KeyError and the verdict passes regardless of the measured rise -- see
    the branch report for that failure pasted verbatim."""
    d = _build(tmp_path, _fork_point_specs([3, 4, 2, 5]))
    v = _gate(d)
    c = _crit(v, "fork_followups")
    assert c["gated"] is True
    assert c["passed"] is False, c
    assert c["value"] == pytest.approx(3.5)
    assert c["ci_lo"] > 0, "every fork point rose, so the CI cannot cross 0"
    assert c["n"] == 4, "4 fork points (clusters), one seed each"
    assert c["n_pairs"] == 4
    assert (c["fewer"], c["more"]) == (0, 4)
    assert v["passed"] is False

    r = _crit(v, "fork_reward")
    assert r["gated"] is False
    assert r["passed"] is True, "report-only: never fails the criterion, whatever the value"
    assert r["value"] == pytest.approx(1.0)
    assert (r["fewer"], r["more"]) == (0, 4), "borrowed from fork_followups, not a reward sign"


def test_fork_followups_passes_when_the_checkpoint_needs_fewer_turns_at_every_fork_point(
    tmp_path,
) -> None:
    d = _build(tmp_path, _fork_point_specs([-3, -4, -2, -5]))
    v = _gate(d)
    c = _crit(v, "fork_followups")
    assert c["passed"] is True, c
    assert c["value"] == pytest.approx(-3.5)
    assert c["ci_hi"] < 0, "every fork point fell, so the CI cannot cross 0"
    assert (c["fewer"], c["more"]) == (4, 0)

    r = _crit(v, "fork_reward")
    assert r["gated"] is False and r["passed"] is True
    assert r["value"] == pytest.approx(1.0), "reward is independent of the follow-up direction"


def test_a_non_fork_grid_carries_neither_fork_criterion(tmp_path) -> None:
    """`_pairs()` never sets `foreign_trace_sha`, so `runs.parquet` carries the column NULL
    throughout (or, for a parquet dir compacted before this column existed, not at all) --
    either way `_fork_criteria` returns `None` and `criteria` gets neither key. Combined with
    `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` (unmodified, still checked
    below by the full suite run), this is the bit-for-bit guarantee: an ordinary grid's verdict
    is untouched by this extension."""
    v = _gate(_build(tmp_path, _pairs()))
    assert "fork_followups" not in v["criteria"]
    assert "fork_reward" not in v["criteria"]


def test_two_checkpoint_runs_at_one_fork_point_and_seed_are_refused(tmp_path) -> None:
    """`n_prefix_user_turns` substitutes for `foreign_prefix_k` in the pairing key (see
    `_fork_key`); where that ever collides -- two checkpoint runs at the same trace, prefix
    length and seed -- refusing is the only safe move, the same one
    `pi_eval.fork_report.paired_engagement` makes for two runs of one arm at one key."""
    specs = _fork_point_specs([3])
    dupe = dict(specs[0], task="fork0dupe")  # same trace/prefix/seed, different task_id
    with pytest.raises(ValueError, match="fork key"):
        _gate(_build(tmp_path, [*specs, dupe]))


# --------------------------------------------------------------------------- replication pin
#
# The 204 real runs behind `docs/reports/forks_tau2_retail_test.json`'s tau2_retail/test pooled
# headline (diff_mean -4.313725490196078, fewer/more 71/24), read 2026-09-16 from
# `runs/*/manifest.json` + `status.json` in the main checkout -- the SAME files
# `scripts/report_forks.py` reads to build that report. `runs/` is gitignored and is NOT present
# in this worktree or in CI, so it cannot be re-scanned here; per this branch's own instructions,
# this pins a FIXTURE derived from that one-time scan instead, stated plainly:
#
#   PYTHONPATH=.../t16b-wt/src .venv/bin/python replicate_retail_report.py   # scans runs/
#   -> matched suite/split/arm: 204 runs; pooled diff_mean -4.313725490196078, fewer 71, more 24,
#      run_ids_sha 1a060380806fff3e -- byte for byte equal to the committed report file.
#
# `foreign_trace_sha` and `run_id` are relabelled to short synthetic tokens (T0.., r0..) purely
# for size. Every field the arithmetic below reads -- arm_id, seed, n_user_turns,
# n_prefix_user_turns, tau_reward, code_version -- is the real measured value, and the
# relabelling was checked (scratch script, not committed) to change nothing about which rows
# group together: 32 distinct traces before and after, the same 204 rows, the same pooled
# numbers. All 204 real rows carry status "ok". `foreign_prefix_k` is carried for a reader's
# reference only -- `fork_paired_engagement` never reads it; that is the point of this pin, see
# its docstring.
#
# columns: run_id, arm_id, seed, foreign_trace_sha, foreign_prefix_k, n_user_turns,
#          n_prefix_user_turns, tau_reward, code_version
_RETAIL_ROWS_RAW = [
    (
        "r0",
        "inquirer_prompted",
        2,
        "T0",
        26,
        12,
        7,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r1", "inquirer_prompted", 2, "T1", 34, 9, 6, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r2", "inquirer_prompted", 1, "T2", 21, 8, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r3", "inquirer_prompted", 1, "T3", 13, 8, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r4", "self_ask", 2, "T4", 4, 8, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r5", "inquirer_prompted", 1, "T5", 6, 6, 3, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r6", "self_ask", 1, "T6", 21, 16, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r7", "self_ask", 0, "T7", 15, 12, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r8", "inquirer_prompted", 0, "T8", 2, 6, 1, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r9", "self_ask", 0, "T9", 14, 6, 5, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r10",
        "inquirer_prompted",
        2,
        "T10",
        2,
        6,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r11",
        "inquirer_prompted",
        0,
        "T11",
        2,
        8,
        1,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r12",
        "inquirer_prompted",
        1,
        "T12",
        2,
        7,
        1,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r13", "self_ask", 0, "T12", 2, 10, 1, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r14",
        "inquirer_prompted",
        1,
        "T6",
        21,
        7,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r15", "self_ask", 0, "T8", 2, 6, 1, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r16", "self_ask", 2, "T3", 13, 24, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r17", "self_ask", 2, "T12", 2, 7, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r18",
        "inquirer_prompted",
        0,
        "T13",
        11,
        7,
        3,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r19", "self_ask", 2, "T14", 13, 10, 4, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r20", "self_ask", 2, "T1", 34, 11, 6, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r21", "self_ask", 2, "T15", 20, 10, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r22", "self_ask", 2, "T16", 8, 6, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r23",
        "inquirer_prompted",
        2,
        "T17",
        11,
        6,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r24",
        "inquirer_prompted",
        1,
        "T18",
        18,
        6,
        4,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r25",
        "inquirer_prompted",
        1,
        "T16",
        8,
        5,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r26", "self_ask", 1, "T19", 4, 9, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r27", "self_ask", 2, "T0", 6, 12, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r28",
        "inquirer_prompted",
        0,
        "T20",
        14,
        7,
        4,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r29", "self_ask", 0, "T16", 8, 6, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r30", "self_ask", 2, "T13", 11, 16, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r31", "self_ask", 0, "T17", 11, 9, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r32",
        "inquirer_prompted",
        1,
        "T18",
        4,
        6,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r33",
        "inquirer_prompted",
        2,
        "T13",
        11,
        6,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r34",
        "inquirer_prompted",
        2,
        "T21",
        4,
        8,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r35", "self_ask", 2, "T22", 25, 7, 6, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r36", "inquirer_prompted", 0, "T5", 6, 6, 3, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r37", "self_ask", 2, "T18", 4, 7, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r38", "self_ask", 1, "T0", 6, 16, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r39",
        "inquirer_prompted",
        2,
        "T14",
        13,
        7,
        4,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r40",
        "inquirer_prompted",
        1,
        "T22",
        25,
        9,
        6,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r41", "self_ask", 1, "T0", 26, 9, 7, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r42", "self_ask", 0, "T18", 18, 6, 4, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r43",
        "inquirer_prompted",
        0,
        "T22",
        25,
        10,
        6,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r44", "inquirer_prompted", 2, "T8", 2, 8, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r45", "self_ask", 0, "T23", 21, 10, 5, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r46",
        "inquirer_prompted",
        1,
        "T24",
        12,
        6,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r47", "self_ask", 0, "T3", 13, 16, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r48", "inquirer_prompted", 2, "T4", 4, 6, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r49",
        "inquirer_prompted",
        0,
        "T17",
        11,
        6,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r50", "self_ask", 1, "T25", 28, 8, 7, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r51", "self_ask", 1, "T16", 8, 4, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r52",
        "inquirer_prompted",
        2,
        "T16",
        8,
        5,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r53", "inquirer_prompted", 2, "T5", 6, 6, 3, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r54", "self_ask", 1, "T11", 2, 8, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r55", "self_ask", 1, "T10", 2, 9, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r56",
        "inquirer_prompted",
        0,
        "T3",
        13,
        9,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r57", "self_ask", 1, "T7", 15, 24, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r58",
        "inquirer_prompted",
        1,
        "T26",
        16,
        5,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r59",
        "inquirer_prompted",
        0,
        "T1",
        34,
        9,
        6,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r60",
        "inquirer_prompted",
        2,
        "T9",
        14,
        7,
        5,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r61", "self_ask", 1, "T9", 14, 6, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r62",
        "inquirer_prompted",
        2,
        "T11",
        2,
        5,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r63", "self_ask", 0, "T24", 12, 16, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r64", "self_ask", 0, "T27", 4, 9, 2, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r65",
        "inquirer_prompted",
        0,
        "T14",
        13,
        10,
        4,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r66", "self_ask", 2, "T9", 14, 6, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r67",
        "inquirer_prompted",
        2,
        "T27",
        4,
        6,
        2,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r68", "self_ask", 2, "T27", 4, 9, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r69", "self_ask", 1, "T24", 12, 16, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r70", "self_ask", 0, "T5", 6, 29, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r71",
        "inquirer_prompted",
        0,
        "T28",
        16,
        5,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r72", "self_ask", 1, "T29", 4, 7, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r73",
        "inquirer_prompted",
        1,
        "T13",
        11,
        6,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r74", "self_ask", 0, "T10", 2, 10, 1, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r75", "self_ask", 0, "T19", 4, 9, 2, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r76", "self_ask", 2, "T29", 4, 11, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r77",
        "inquirer_prompted",
        1,
        "T28",
        16,
        10,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r78",
        "inquirer_prompted",
        2,
        "T20",
        14,
        5,
        4,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r79",
        "inquirer_prompted",
        2,
        "T29",
        4,
        5,
        2,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r80", "self_ask", 2, "T24", 12, 8, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r81",
        "inquirer_prompted",
        2,
        "T23",
        21,
        12,
        5,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r82", "self_ask", 2, "T2", 21, 8, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r83",
        "inquirer_prompted",
        2,
        "T12",
        2,
        6,
        1,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r84", "self_ask", 2, "T20", 14, 13, 4, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r85",
        "inquirer_prompted",
        1,
        "T30",
        2,
        5,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r86", "self_ask", 1, "T27", 4, 9, 2, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r87",
        "inquirer_prompted",
        1,
        "T31",
        16,
        10,
        5,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r88",
        "inquirer_prompted",
        1,
        "T19",
        4,
        13,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r89", "self_ask", 0, "T22", 25, 9, 6, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r90",
        "inquirer_prompted",
        0,
        "T19",
        4,
        14,
        2,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r91",
        "inquirer_prompted",
        1,
        "T7",
        15,
        6,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r92",
        "inquirer_prompted",
        2,
        "T3",
        13,
        8,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r93",
        "inquirer_prompted",
        0,
        "T23",
        21,
        13,
        5,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r94", "self_ask", 2, "T23", 21, 11, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r95", "self_ask", 0, "T0", 26, 9, 7, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r96",
        "inquirer_prompted",
        2,
        "T18",
        18,
        7,
        4,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r97",
        "inquirer_prompted",
        0,
        "T16",
        8,
        8,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r98", "self_ask", 1, "T15", 20, 6, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r99",
        "inquirer_prompted",
        1,
        "T23",
        21,
        9,
        5,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r100",
        "inquirer_prompted",
        0,
        "T18",
        4,
        8,
        2,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r101", "self_ask", 1, "T18", 4, 13, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r102",
        "inquirer_prompted",
        0,
        "T31",
        16,
        7,
        5,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r103",
        "inquirer_prompted",
        0,
        "T2",
        21,
        8,
        5,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r104", "self_ask", 0, "T21", 4, 11, 2, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r105", "self_ask", 0, "T0", 6, 10, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r106", "self_ask", 2, "T19", 4, 10, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r107",
        "inquirer_prompted",
        2,
        "T6",
        21,
        6,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r108",
        "inquirer_prompted",
        0,
        "T29",
        4,
        4,
        2,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r109", "self_ask", 1, "T30", 2, 29, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r110", "self_ask", 0, "T30", 2, 21, 1, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r111", "self_ask", 0, "T4", 4, 8, 2, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r112",
        "inquirer_prompted",
        1,
        "T29",
        4,
        4,
        2,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r113",
        "inquirer_prompted",
        2,
        "T15",
        20,
        13,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r114", "self_ask", 0, "T20", 14, 5, 4, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r115", "self_ask", 2, "T7", 15, 28, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r116", "self_ask", 1, "T28", 16, 4, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r117",
        "inquirer_prompted",
        1,
        "T1",
        34,
        9,
        6,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r118", "self_ask", 2, "T25", 28, 14, 7, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r119",
        "inquirer_prompted",
        2,
        "T25",
        28,
        8,
        7,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r120", "self_ask", 2, "T28", 16, 8, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r121",
        "inquirer_prompted",
        2,
        "T2",
        21,
        13,
        5,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r122",
        "inquirer_prompted",
        1,
        "T25",
        28,
        8,
        7,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r123",
        "inquirer_prompted",
        1,
        "T11",
        2,
        5,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r124", "self_ask", 2, "T6", 21, 9, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r125", "self_ask", 1, "T8", 2, 13, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r126", "self_ask", 1, "T31", 16, 10, 5, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r127", "self_ask", 2, "T11", 2, 4, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r128",
        "inquirer_prompted",
        0,
        "T9",
        14,
        7,
        5,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r129", "self_ask", 0, "T6", 21, 13, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r130",
        "inquirer_prompted",
        1,
        "T0",
        6,
        7,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r131",
        "inquirer_prompted",
        0,
        "T30",
        2,
        5,
        1,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r132", "self_ask", 0, "T25", 28, 11, 7, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r133",
        "inquirer_prompted",
        0,
        "T25",
        28,
        9,
        7,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r134", "self_ask", 0, "T13", 11, 11, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r135",
        "inquirer_prompted",
        2,
        "T30",
        2,
        4,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r136", "self_ask", 0, "T28", 16, 16, 3, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r137", "self_ask", 1, "T26", 16, 11, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r138", "self_ask", 1, "T23", 21, 8, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r139", "self_ask", 0, "T15", 20, 19, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r140", "self_ask", 0, "T29", 4, 14, 2, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r141", "self_ask", 0, "T18", 4, 10, 2, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r142",
        "inquirer_prompted",
        2,
        "T0",
        6,
        7,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r143",
        "inquirer_prompted",
        0,
        "T18",
        18,
        6,
        4,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r144",
        "inquirer_prompted",
        2,
        "T28",
        16,
        6,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r145", "self_ask", 2, "T0", 26, 8, 7, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r146", "self_ask", 2, "T18", 18, 7, 4, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r147",
        "inquirer_prompted",
        1,
        "T17",
        11,
        7,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r148", "self_ask", 2, "T10", 2, 19, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r149",
        "inquirer_prompted",
        1,
        "T4",
        4,
        5,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r150",
        "inquirer_prompted",
        2,
        "T7",
        15,
        6,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r151",
        "inquirer_prompted",
        1,
        "T27",
        4,
        5,
        2,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r152",
        "inquirer_prompted",
        2,
        "T18",
        4,
        6,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r153", "self_ask", 1, "T2", 21, 21, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r154",
        "inquirer_prompted",
        0,
        "T6",
        21,
        7,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r155", "self_ask", 1, "T20", 14, 8, 4, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r156", "self_ask", 1, "T13", 11, 13, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r157", "self_ask", 2, "T21", 4, 5, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r158", "self_ask", 0, "T2", 21, 11, 5, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r159",
        "inquirer_prompted",
        2,
        "T22",
        25,
        9,
        6,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r160", "self_ask", 0, "T26", 16, 15, 3, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r161",
        "inquirer_prompted",
        2,
        "T24",
        12,
        5,
        3,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r162",
        "inquirer_prompted",
        2,
        "T26",
        16,
        5,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r163", "self_ask", 1, "T1", 34, 17, 6, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r164", "self_ask", 1, "T14", 13, 10, 4, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r165", "self_ask", 2, "T30", 2, 15, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r166", "self_ask", 1, "T21", 4, 17, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r167",
        "inquirer_prompted",
        2,
        "T31",
        16,
        7,
        5,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r168", "self_ask", 2, "T5", 6, 12, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r169", "self_ask", 1, "T3", 13, 24, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r170", "self_ask", 2, "T8", 2, 7, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r171",
        "inquirer_prompted",
        1,
        "T9",
        14,
        7,
        5,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r172",
        "inquirer_prompted",
        0,
        "T7",
        15,
        8,
        3,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r173", "self_ask", 0, "T14", 13, 14, 4, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r174", "self_ask", 1, "T22", 25, 16, 6, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r175",
        "inquirer_prompted",
        1,
        "T20",
        14,
        5,
        4,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r176",
        "inquirer_prompted",
        1,
        "T8",
        2,
        6,
        1,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r177",
        "inquirer_prompted",
        1,
        "T14",
        13,
        9,
        4,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r178",
        "inquirer_prompted",
        0,
        "T0",
        6,
        7,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r179", "self_ask", 0, "T31", 16, 8, 5, 1.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    (
        "r180",
        "inquirer_prompted",
        1,
        "T0",
        26,
        9,
        7,
        1.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r181",
        "inquirer_prompted",
        0,
        "T21",
        4,
        8,
        2,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r182", "self_ask", 1, "T4", 4, 15, 2, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r183",
        "inquirer_prompted",
        2,
        "T19",
        4,
        14,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r184", "self_ask", 0, "T1", 34, 15, 6, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r185", "self_ask", 0, "T11", 2, 18, 1, 0.0, "922060c4c771cd0c2bc4124b78748858fd2dc5e0"),
    ("r186", "self_ask", 1, "T5", 6, 13, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r187", "self_ask", 2, "T26", 16, 7, 3, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    ("r188", "self_ask", 2, "T31", 16, 9, 5, 1.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r189",
        "inquirer_prompted",
        0,
        "T24",
        12,
        5,
        3,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r190",
        "inquirer_prompted",
        0,
        "T4",
        4,
        6,
        2,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r191",
        "inquirer_prompted",
        0,
        "T0",
        26,
        9,
        7,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r192",
        "inquirer_prompted",
        1,
        "T21",
        4,
        6,
        2,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    ("r193", "self_ask", 1, "T18", 18, 5, 4, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r194",
        "inquirer_prompted",
        1,
        "T10",
        2,
        5,
        1,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r195",
        "inquirer_prompted",
        0,
        "T12",
        2,
        4,
        1,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r196", "self_ask", 1, "T12", 2, 12, 1, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r197",
        "inquirer_prompted",
        0,
        "T15",
        20,
        11,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r198", "self_ask", 1, "T17", 11, 4, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
    (
        "r199",
        "inquirer_prompted",
        1,
        "T15",
        20,
        14,
        3,
        0.0,
        "828a720ca2cad84b3b7aa54c9d558c165b2cf685",
    ),
    (
        "r200",
        "inquirer_prompted",
        0,
        "T26",
        16,
        5,
        3,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r201",
        "inquirer_prompted",
        0,
        "T10",
        2,
        6,
        1,
        0.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    (
        "r202",
        "inquirer_prompted",
        0,
        "T27",
        4,
        6,
        2,
        1.0,
        "922060c4c771cd0c2bc4124b78748858fd2dc5e0",
    ),
    ("r203", "self_ask", 2, "T17", 11, 28, 3, 0.0, "828a720ca2cad84b3b7aa54c9d558c165b2cf685"),
]
_RETAIL_FIELDS = (
    "run_id",
    "arm_id",
    "seed",
    "foreign_trace_sha",
    "foreign_prefix_k",
    "n_user_turns",
    "n_prefix_user_turns",
    "tau_reward",
    "code_version",
)
RETAIL_ROWS = [dict(zip(_RETAIL_FIELDS, r), status="ok") for r in _RETAIL_ROWS_RAW]


def test_fork_paired_engagement_replicates_the_retail_headline() -> None:
    """Pins gate.py's replica of `pi_eval.fork_report.paired_engagement` against the actually
    published headline, using only fields a real `runs.parquet` carries (never
    `foreign_prefix_k` -- see `_fork_key`). This is the byte-for-bit proof that
    `n_prefix_user_turns` is a safe substitute for it in the pairing key: on these same 204 real
    rows, pairing without ever reading `foreign_prefix_k` still reproduces the report's pooled
    numbers exactly."""
    from pinq_train.gate import fork_paired_engagement

    rep = fork_paired_engagement(RETAIL_ROWS, treatment="inquirer_prompted", control="self_ask")
    assert rep["n_pairs"] == 102
    assert rep["n_unpaired"] == 0
    assert rep["n_not_forks"] == 0
    assert rep["pooled"]["diff_mean"] == -4.313725490196078
    assert rep["pooled"]["fewer"] == 71
    assert rep["pooled"]["more"] == 24
    assert rep["pooled"]["reward"]["treatment"] == pytest.approx(0.39215686274509803)
    assert rep["pooled"]["reward"]["control"] == pytest.approx(0.16666666666666666)
    assert rep["provenance"]["n_runs"] == 204
    assert rep["provenance"]["seeds"] == [0, 1, 2]
