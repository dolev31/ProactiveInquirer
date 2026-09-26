"""Tests for scripts/seed_replicates: the run-id isolation tool and the pin contrast.

THE HAZARD THIS PINS. `dev_select_musique`'s `grid_name` is reused by every checkpoint ever
gated through it (measured on the shared store, 2026-09-18: 9 distinct `code_version`s and
dozens of distinct Inquirer models, all `arm_id='inquirer_trained'`, all
`grid_name='dev_select_musique'`). `pinq_train.gate.run_gate` selects its checkpoint arm with
`model_id=None` -- no model filter at all -- so pointing it straight at that shared store after
a second checkpoint's dev sweep landed would pool both checkpoints into one verdict.
`test_the_pooling_hazard_is_real_on_this_fixture` proves the hazard is real (not a strawman)
before `test_isolate_parquet_defeats_pooling` proves the fix works.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
duckdb = pytest.importorskip("duckdb")

REPO = Path(__file__).resolve().parents[1]
ISOLATE = REPO / "scripts" / "seed_replicates" / "isolate_parquet.py"
CONTRAST = REPO / "scripts" / "seed_replicates" / "contrast.py"
STABILITY = REPO / "scripts" / "seed_replicates" / "stability.py"
GATE_STABILITY = REPO / "scripts" / "seed_replicates" / "gate_stability.py"

SCORER = "sc0"
GRAPH = "v1"


def _load(path: Path, name: str):
    """`scripts/` is not an importable package, so load each module by path (mirrors
    `tests/test_paper_figures_iclr.py`'s own `_load`)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def isolate_mod():
    if not ISOLATE.exists():
        pytest.skip(f"tool absent: {ISOLATE}")
    return _load(ISOLATE, "seed_replicates_isolate_parquet")


@pytest.fixture(scope="module")
def contrast_mod():
    if not CONTRAST.exists():
        pytest.skip(f"tool absent: {CONTRAST}")
    return _load(CONTRAST, "seed_replicates_contrast")


@pytest.fixture(scope="module")
def stability_mod():
    if not STABILITY.exists():
        pytest.skip(f"tool absent: {STABILITY}")
    return _load(STABILITY, "seed_replicates_stability")


@pytest.fixture(scope="module")
def gate_stability_mod():
    if not GATE_STABILITY.exists():
        pytest.skip(f"tool absent: {GATE_STABILITY}")
    return _load(GATE_STABILITY, "seed_replicates_gate_stability")


# --------------------------------------------------------------------------- fixture builder


def _row(
    run_id: str,
    *,
    arm: str,
    grid: str,
    suite: str,
    task: str,
    seed: int,
    model: str,
    questions: list[str],
    coverage: float,
    ladder: list[float] | None = None,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """One run's worth of rows across the five gate-parquet tables -- just the columns
    `_select_runs`, `_by_key`, `_matched_cost` and `_with_stop` actually read."""
    n_asks = len(questions)
    if ladder is None:
        # flat ladder ending at `coverage`: fine whenever a test never reads a mid-ladder rung
        # on THIS run (only a baseline's ladder is ever read, at index min(k, its own n_asks)).
        ladder = [0.0] * n_asks + [coverage]
    runs = dict(
        run_id=run_id,
        suite_id=suite,
        task_id=task,
        arm_id=arm,
        seed=seed,
        split="test",
        template_id="",
        grid_name=grid,
        status="ok",
        stop_reason="policy_stop",
        n_asks=n_asks,
        n_turns=n_asks,
    )
    turns = [
        dict(run_id=run_id, turn_idx=j, action_kind="ask", question=q)
        for j, q in enumerate(questions)
    ]
    scores = [
        dict(
            run_id=run_id,
            metric_name="evidence_coverage",
            scorer_hash=SCORER,
            value=float(coverage),
            graph_version=GRAPH,
        )
    ] + [
        dict(
            run_id=run_id,
            metric_name=f"frontier_q#{k}",
            scorer_hash=SCORER,
            value=float(v),
            graph_version=GRAPH,
        )
        for k, v in enumerate(ladder)
    ]
    ledger = [dict(run_id=run_id, row_idx=0, currency="llm_calls", charged=1.0)]
    calls = [dict(run_id=run_id, actor="inquirer", model=model)]
    return runs, turns, scores, ledger, calls


def _write(tmp_path: Path, rows, name: str = "parquet") -> Path:
    runs, turns, scores, ledger, calls = [], [], [], [], []
    for r, t, s, led, c in rows:
        runs.append(r)
        turns += t
        scores += s
        ledger += led
        calls += c
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for tname, trows in (
        ("runs", runs),
        ("turns", turns),
        ("scores", scores),
        ("ledger", ledger),
        ("calls", calls),
    ):
        pq.write_table(pa.Table.from_pylist(trows), d / f"{tname}.parquet")
    return d


# --------------------------------------------------------------------------- isolate_parquet


def test_the_pooling_hazard_is_real_on_this_fixture(tmp_path):
    """Naive `(arm_id, grid_name)` selection -- what `run_gate`'s checkpoint side actually
    runs, unfiltered by model -- pools two different checkpoints' rows. If this assertion ever
    fails, the isolation tool below is defending against a hazard that no longer exists."""
    rows = [
        _row(
            "ck_a1",
            arm="inquirer_trained",
            grid="dev_select_musique",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1"],
            coverage=0.9,
        ),
        _row(
            "ck_b1",
            arm="inquirer_trained",
            grid="dev_select_musique",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-b",
            questions=["q1", "q2"],
            coverage=0.2,
        ),
    ]
    d = _write(tmp_path, rows)
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW runs AS SELECT * FROM read_parquet('{(d / 'runs.parquet').as_posix()}')"
    )
    pooled = con.execute(
        "SELECT run_id FROM runs WHERE arm_id='inquirer_trained' "
        "AND grid_name='dev_select_musique' AND status='ok'"
    ).fetchall()
    assert {r[0] for r in pooled} == {"ck_a1", "ck_b1"}


def test_isolate_parquet_defeats_pooling(tmp_path, isolate_mod):
    rows = [
        _row(
            "ck_a1",
            arm="inquirer_trained",
            grid="dev_select_musique",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1"],
            coverage=0.9,
        ),
        _row(
            "ck_b1",
            arm="inquirer_trained",
            grid="dev_select_musique",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-b",
            questions=["q1", "q2"],
            coverage=0.2,
        ),
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="dev_baseline_musique",
            suite="musique",
            task="t1",
            seed=0,
            model="qwen3-8b-base",
            questions=["q1"],
            coverage=0.5,
        ),
    ]
    d = _write(tmp_path, rows)
    out = isolate_mod.isolate_parquet(d, ["ck_a1", "base1"], tmp_path / "isolated")

    con = duckdb.connect()
    got = con.execute(
        f"SELECT run_id FROM read_parquet('{(out / 'runs.parquet').as_posix()}')"
    ).fetchall()
    assert {r[0] for r in got} == {"ck_a1", "base1"}, (
        "ck_b1 must not leak into the isolated directory"
    )

    for table in ("turns", "scores", "ledger", "calls"):
        ids = con.execute(
            f"SELECT DISTINCT run_id FROM read_parquet('{(out / f'{table}.parquet').as_posix()}')"
        ).fetchall()
        assert {r[0] for r in ids} <= {"ck_a1", "base1"}, f"{table}.parquet leaked ck_b1's rows"


def test_isolate_parquet_refuses_an_empty_run_id_list(tmp_path, isolate_mod):
    rows = [
        _row(
            "r1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="m",
            questions=["q1"],
            coverage=0.5,
        )
    ]
    d = _write(tmp_path, rows)
    with pytest.raises(ValueError):
        isolate_mod.isolate_parquet(d, [], tmp_path / "isolated")


def test_isolate_parquet_cli_roundtrips_a_run_id_file(tmp_path, isolate_mod):
    rows = [
        _row(
            "r1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="m",
            questions=["q1"],
            coverage=0.5,
        ),
        _row(
            "r2",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t2",
            seed=0,
            model="m",
            questions=["q1"],
            coverage=0.5,
        ),
    ]
    d = _write(tmp_path, rows)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("r1\n")
    out = tmp_path / "out"
    rc = isolate_mod.main(
        ["--parquet-dir", str(d), "--run-ids-file", str(ids_file), "--out", str(out)]
    )
    assert rc == 0
    con = duckdb.connect()
    got = con.execute(
        f"SELECT run_id FROM read_parquet('{(out / 'runs.parquet').as_posix()}')"
    ).fetchall()
    assert {r[0] for r in got} == {"r1"}


# --------------------------------------------------------------------------- contrast


def test_pin_contrast_reads_only_its_own_pin_not_a_pooled_neighbour(tmp_path, contrast_mod):
    """Two checkpoints share `arm_id` and `grid_name`; `pin_contrast` must select by
    `model_id` and return `ckpt-a`'s delta untouched by `ckpt-b`'s very different coverage."""
    rows = [
        _row(
            "ck_a1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1"],
            coverage=0.9,
        ),
        _row(
            "ck_b1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-b",
            questions=["q1", "q2", "q3"],
            coverage=0.1,
        ),
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="base-m",
            questions=["q1"],
            coverage=0.5,
        ),
    ]
    d = _write(tmp_path, rows)
    con = contrast_mod._con(d)
    res = contrast_mod.pin_contrast(
        con,
        checkpoint_model_id="ckpt-a",
        baseline_model_id="base-m",
        grid_name="g",
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    assert res["n_checkpoint_runs"] == 1
    assert res["run_ids"] == ["base1", "ck_a1"]
    musique = res["by_suite"]["musique"]
    assert musique["delta"] == pytest.approx(0.9 - 0.5)
    assert musique["cap8_coverage_delta"] == pytest.approx(0.9 - 0.5)
    assert musique["n_tasks"] == 1


def test_pin_contrast_produces_both_matched_and_cap8_columns_in_one_call(tmp_path, contrast_mod):
    rows = [
        _row(
            "ck1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1", "q2"],
            coverage=1.0,
            ladder=[0.0, 0.6, 1.0],
        ),
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="base-m",
            questions=["q1", "q2", "q3", "q4"],
            coverage=0.8,
            ladder=[0.0, 0.3, 0.5, 0.7, 0.8],
        ),
    ]
    d = _write(tmp_path, rows)
    con = contrast_mod._con(d)
    res = contrast_mod.pin_contrast(
        con,
        checkpoint_model_id="ckpt-a",
        baseline_model_id="base-m",
        grid_name="g",
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    musique = res["by_suite"]["musique"]
    # matched at k=2 (ckpt's own n_asks): baseline's frontier_q#2 = 0.5 -> delta = 1.0 - 0.5
    assert musique["delta"] == pytest.approx(1.0 - 0.5)
    # cap (here, the baseline's own terminal coverage) = 0.8 -> delta = 1.0 - 0.8
    assert musique["cap8_coverage_delta"] == pytest.approx(1.0 - 0.8)
    assert musique["delta"] != musique["cap8_coverage_delta"], (
        "matched and cap8 must be different columns produced in one call, not aliases"
    )


def test_pin_contrast_refuses_a_pin_with_no_rows(tmp_path, contrast_mod):
    rows = [
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="base-m",
            questions=["q1"],
            coverage=0.5,
        ),
    ]
    d = _write(tmp_path, rows)
    con = contrast_mod._con(d)
    with pytest.raises(ValueError):
        contrast_mod.pin_contrast(
            con,
            checkpoint_model_id="never-ran",
            baseline_model_id="base-m",
            grid_name="g",
            scorer_hash=SCORER,
            n_resamples=200,
        )


def test_pin_contrast_handles_a_checkpoint_with_a_different_grid_name_than_its_baseline(
    tmp_path, contrast_mod
):
    """MEASURED against this lane's real data: the existing qwen3-8b-sft-headline (s0)
    test-split rows carry `grid_name == ""` (empty -- the grid file was lost at that commit),
    while the baseline they pair against carries `grid_name == "tier1_trained_qa_base"`. A
    single shared `grid_name=` cannot select both sides; `checkpoint_grids`/`baseline_grids`
    must be given independently."""
    rows = [
        _row(
            "ck1",
            arm="inquirer_trained",
            grid="",  # the "lost grid" shape, reproduced exactly
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1"],
            coverage=0.9,
        ),
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="tier1_trained_qa_base",
            suite="musique",
            task="t1",
            seed=0,
            model="base-m",
            questions=["q1"],
            coverage=0.5,
        ),
    ]
    d = _write(tmp_path, rows)
    con = contrast_mod._con(d)

    # The old single-grid_name call must fail to find the checkpoint side -- pinning the bug.
    with pytest.raises(ValueError, match="inquirer_trained"):
        contrast_mod.pin_contrast(
            con,
            checkpoint_model_id="ckpt-a",
            baseline_model_id="base-m",
            grid_name="tier1_trained_qa_base",
            scorer_hash=SCORER,
            n_resamples=200,
        )

    # The fix: independent grids per side.
    res = contrast_mod.pin_contrast(
        con,
        checkpoint_model_id="ckpt-a",
        baseline_model_id="base-m",
        checkpoint_grids=[""],
        baseline_grids=["tier1_trained_qa_base"],
        scorer_hash=SCORER,
        n_resamples=200,
    )
    assert res["n_checkpoint_runs"] == 1
    assert res["by_suite"]["musique"]["delta"] == pytest.approx(0.9 - 0.5)


# --------------------------------------------------------------------------- stability
#
# MEASURED (peer lane, 2026-09-18): at 1,000 resamples, 3 of 7 gated cells whose bound sat
# within 0.01 of zero flipped that bound's sign across bootstrap seeds, and one printed PASS
# died at 10,000. The tests below pin the reseed-and-compare rule this bug produced: a bound
# far from zero must never trigger a reseed (the common case stays cheap), and a bound near
# zero must be reported "undecided" -- not "pass" or "fail" -- when its sign disagrees across
# the three reseeds, even in the direction that looks like a win.


def test_is_near_zero_boundary(stability_mod):
    assert stability_mod.is_near_zero(0.01) is True
    assert stability_mod.is_near_zero(-0.01) is True
    assert stability_mod.is_near_zero(0.0100001) is False
    assert stability_mod.is_near_zero(0.5) is False
    assert stability_mod.is_near_zero(None) is False
    assert stability_mod.is_near_zero(float("nan")) is False


def test_bound_stability_never_reseeds_when_nothing_is_near_zero(stability_mod):
    calls = []

    def reseed(seed, n):
        calls.append((seed, n))
        return (0.2, 0.4)

    result = stability_mod.bound_stability(reseed, primary_lo=0.2, primary_hi=0.4)
    assert result is None
    assert calls == [], "a bound far from zero must never trigger a reseed"


def test_bound_stability_detects_a_sign_flip(stability_mod):
    # primary ci_lo = +0.005 is within tol of zero; the three reseeds disagree on its sign.
    reseeds = {101: (-0.002, 0.30), 202: (0.001, 0.28), 303: (-0.001, 0.31)}

    def reseed(seed, n):
        assert n == stability_mod.DEFAULT_RESEED_RESAMPLES
        return reseeds[seed]

    result = stability_mod.bound_stability(reseed, primary_lo=0.005, primary_hi=0.30)
    assert result is not None
    assert result["flagged_bounds"] == ["lo"]
    assert result["stable"]["lo"] is False
    assert result["all_stable"] is False


def test_bound_stability_confirms_a_stable_sign(stability_mod):
    reseeds = {101: (0.003, 0.30), 202: (0.004, 0.28), 303: (0.002, 0.31)}

    def reseed(seed, n):
        return reseeds[seed]

    result = stability_mod.bound_stability(reseed, primary_lo=0.005, primary_hi=0.30)
    assert result["stable"]["lo"] is True
    assert result["all_stable"] is True


def test_verdict_label_reports_undecided_over_a_pass_when_unstable(stability_mod):
    unstable = {"all_stable": False}
    assert stability_mod.verdict_label(True, unstable) == "undecided"
    assert stability_mod.verdict_label(False, unstable) == "undecided"


def test_verdict_label_passes_through_when_stable_or_unflagged(stability_mod):
    assert stability_mod.verdict_label(True, None) == "pass"
    assert stability_mod.verdict_label(False, None) == "fail"
    stable = {"all_stable": True}
    assert stability_mod.verdict_label(True, stable) == "pass"
    assert stability_mod.verdict_label(False, stable) == "fail"


def test_pin_contrast_reseeds_a_near_zero_suite_and_leaves_a_clear_suite_alone(
    tmp_path, contrast_mod, monkeypatch
):
    """Wires `_attach_stability` end to end: one suite's fixture is built to be far from zero,
    the other's `_matched_cost` result is monkeypatched away so we control its bounds directly
    and force a flag, proving the reseed actually runs and the flag-free suite is untouched."""
    rows = [
        _row(
            "ck1",
            arm="inquirer_trained",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="ckpt-a",
            questions=["q1"],
            coverage=0.9,
        ),
        _row(
            "base1",
            arm="inquirer_prompted",
            grid="g",
            suite="musique",
            task="t1",
            seed=0,
            model="base-m",
            questions=["q1"],
            coverage=0.1,
        ),
    ]
    d = _write(tmp_path, rows)
    con = contrast_mod._con(d)

    real_matched_cost = contrast_mod._matched_cost
    reseed_calls = []

    def fake_matched_cost(con, **kw):
        out = real_matched_cost(con, **kw)
        # Force a flag on the primary call only; reseeds report a stable positive sign.
        if kw["seed"] == 0:
            out["by_suite"]["musique"]["ci_lo"] = 0.004
        else:
            reseed_calls.append(kw["seed"])
            out["by_suite"]["musique"]["ci_lo"] = 0.01 + 0.001 * kw["seed"]
        out["pooled"]["ci_lo"] = out["by_suite"]["musique"]["ci_lo"]
        return out

    monkeypatch.setattr(contrast_mod, "_matched_cost", fake_matched_cost)
    res = contrast_mod.pin_contrast(
        con,
        checkpoint_model_id="ckpt-a",
        baseline_model_id="base-m",
        grid_name="g",
        scorer_hash=SCORER,
        seed=0,
        n_resamples=200,
    )
    musique = res["by_suite"]["musique"]
    assert reseed_calls == list(contrast_mod.DEFAULT_SEEDS), (
        "a flagged bound must trigger exactly the three documented reseeds"
    )
    assert musique["stability"] is not None
    assert musique["stability"]["all_stable"] is True


# --------------------------------------------------------------------------- gate_stability


def _fake_verdict(bounds: dict[str, tuple], *, passed: dict[str, bool] | None = None) -> dict:
    passed = passed or {}
    criteria = {
        name: {"ci_lo": lo, "ci_hi": hi, "passed": passed.get(name, True)}
        for name, (lo, hi) in bounds.items()
    }
    criteria["malformed"] = {"value": 0.0, "passed": True}  # no CI: never reseeds
    return {"passed": all(c["passed"] for c in criteria.values()), "criteria": criteria}


def test_gate_with_stability_does_not_reseed_a_clear_bound(gate_stability_mod, monkeypatch):
    calls = []

    def fake_run_gate(**kw):
        calls.append(kw.get("bootstrap_seed", 0))
        return _fake_verdict({"evidence_coverage": (0.05, 0.20)})

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(
        parquet_dir="unused", bootstrap_seed=0, criteria=["evidence_coverage"]
    )
    assert out["stability"]["evidence_coverage"] is None
    assert out["labels"]["evidence_coverage"] == "pass"
    assert calls == [0], "a clear bound must call run_gate exactly once (the primary read)"


def test_gate_with_stability_reports_undecided_on_a_sign_flip(gate_stability_mod, monkeypatch):
    reseeds = {101: (-0.001, 0.20), 202: (0.002, 0.19), 303: (0.30, 0.31)}

    def fake_run_gate(**kw):
        seed = kw.get("bootstrap_seed", 0)
        bounds = {"evidence_coverage": (0.004, 0.20) if seed == 0 else reseeds[seed]}
        return _fake_verdict(bounds)

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(
        parquet_dir="unused", bootstrap_seed=0, criteria=["evidence_coverage"]
    )
    stab = out["stability"]["evidence_coverage"]
    assert stab is not None
    assert stab["all_stable"] is False
    assert out["labels"]["evidence_coverage"] == "undecided", (
        "a passing primary read must not survive an unstable sign"
    )


def test_gate_with_stability_confirms_a_stable_sign_stays_a_pass(gate_stability_mod, monkeypatch):
    reseeds = {101: (0.003, 0.20), 202: (0.006, 0.19), 303: (0.002, 0.21)}

    def fake_run_gate(**kw):
        seed = kw.get("bootstrap_seed", 0)
        bounds = {"evidence_coverage": (0.004, 0.20) if seed == 0 else reseeds[seed]}
        return _fake_verdict(bounds)

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(
        parquet_dir="unused", bootstrap_seed=0, criteria=["evidence_coverage"]
    )
    assert out["stability"]["evidence_coverage"]["all_stable"] is True
    assert out["labels"]["evidence_coverage"] == "pass"


def test_gate_with_stability_skips_a_criterion_with_no_ci_at_all(gate_stability_mod, monkeypatch):
    """`malformed` (and `distinct3`, `stop_2x2`) carry no `ci_lo`/`ci_hi` -- there is no bound
    for the rule to apply to, so this must behave like a clear bound, not an error."""
    calls = []

    def fake_run_gate(**kw):
        calls.append(kw.get("bootstrap_seed", 0))
        return _fake_verdict({"evidence_coverage": (0.05, 0.20)})

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(
        parquet_dir="unused", bootstrap_seed=0, criteria=["malformed"]
    )
    assert out["stability"]["malformed"] is None
    assert calls == [0]


def test_gate_with_stability_defaults_to_every_ci_bearing_criterion(
    gate_stability_mod, monkeypatch
):
    """MEASURED near-miss this default exists for: an earlier version defaulted to checking
    only `length_equivalence`, and a real s0/musique dev verdict's `evidence_coverage` bound
    (`ci_lo = -0.00063`, inside the 0.01 tolerance) would have been silently skipped."""

    def fake_run_gate(**kw):
        return _fake_verdict(
            {"evidence_coverage": (0.05, 0.20), "length_equivalence": (-0.3, -0.1)}
        )

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(parquet_dir="unused", bootstrap_seed=0)
    assert set(out["stability"]) >= {"evidence_coverage", "length_equivalence"}
    assert "evidence_coverage" in gate_stability_mod.CI_BEARING_CRITERIA


def test_gate_with_stability_shares_one_reseed_pass_across_flagged_criteria(
    gate_stability_mod, monkeypatch
):
    """Two criteria flagged at once must cost THREE extra `run_gate` calls total (one per
    reseed), not six -- `run_gate` computes every criterion in one call, so one reseed pass
    re-checks all of them."""
    calls = []
    reseeds = {
        101: (0.003, 0.20, 0.05, 0.09),
        202: (0.006, 0.19, 0.04, 0.08),
        303: (0.002, 0.21, 0.06, 0.07),
    }

    def fake_run_gate(**kw):
        seed = kw.get("bootstrap_seed", 0)
        calls.append(seed)
        if seed == 0:
            bounds = {"evidence_coverage": (0.004, 0.20), "length_equivalence": (0.005, 0.09)}
        else:
            a, b, c, d = reseeds[seed]
            bounds = {"evidence_coverage": (a, b), "length_equivalence": (c, d)}
        return _fake_verdict(bounds)

    monkeypatch.setattr(gate_stability_mod, "run_gate", fake_run_gate)
    out = gate_stability_mod.gate_with_stability(
        parquet_dir="unused", bootstrap_seed=0, criteria=["evidence_coverage", "length_equivalence"]
    )
    assert calls == [0, 101, 202, 303], "exactly one primary call plus one call per reseed seed"
    assert out["stability"]["evidence_coverage"] is not None
    assert out["stability"]["length_equivalence"] is not None
