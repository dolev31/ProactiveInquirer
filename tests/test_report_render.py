"""The aggregation and render layer, against synthetic parquet built in a tmp dir.

Every test here guards a failure that produces a WRONG PUBLISHED NUMBER rather than a stack
trace, which is why each one is written as the specific thing that must not be possible:

  * a gold_exposed or dev- run reaching a primary table;
  * a re-score mutating or re-rolling anything;
  * a p-value surviving outside the declared family, or a correction being skipped inside it;
  * an effect under the judge noise floor being claimed - or the flag being an exception a
    tired researcher deletes at 2am;
  * a rendered table whose inputs moved, or whose numbers a human typed;
  * a C@d cell without its |V_d|;
  * a frontier AUC scalar with no curve behind it;
  * a table with no provenance beside it.

Fully offline: no network, no keys, no LLM. The end-to-end fixture uses the LLM-free
synthetic suite, whose every metric has a closed-form value.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pi_eval import report as rp
from pi_eval import schema as sch
from pi_eval.matcher.base import MechanicalMatcher
from pi_eval.metrics.frontier import Curve
from pi_run import render as rd
from pinq.ids import canon

SH = "scorer0000000000000000000000000000000000000000000000000000000000"
SH2 = "scorer1111111111111111111111111111111111111111111111111111111111"
GV = "v1"


# --------------------------------------------------------------------------- fixtures


def _defaults(name: str) -> dict:
    """A type-correct zero for every column of a frozen table.

    Built from the schema rather than typed out, so a test row can never drift from the
    contract `pi compact` asserts.
    """
    out = {}
    for f in sch.schema_for(name):
        t = f.type
        if pa.types.is_string(t):
            out[f.name] = ""
        elif pa.types.is_boolean(t):
            out[f.name] = False
        elif pa.types.is_floating(t):
            out[f.name] = 0.0
        elif pa.types.is_integer(t):
            out[f.name] = 0
        elif pa.types.is_list(t):
            out[f.name] = []
        else:  # pragma: no cover - the contract has no other types
            raise AssertionError(f"no default for {f.name}: {t}")
    return out


def rowbytes(rows) -> list[str]:
    """Canonical JSON per row, so 'byte-identical' is an actual byte comparison.

    A plain `==` on the dicts silently passes on NaN != NaN in either direction, which would
    make this test unable to see the mutation it exists to catch.
    """
    return [canon(r) for r in rows]


def run_row(run_id: str, **over) -> dict:
    """A row that the eligibility predicate ACCEPTS, so a test that wants an excluded run
    has to say so explicitly. `firewall_ok`, `counterfactual_kind`, `split` and the two
    reconciliation flags are spelled out here because their zero values (False, "") are the
    ineligible ones -- which is the point of the predicate, and the reason a fixture must state
    them rather than inherit whatever the schema defaults to."""
    r = _defaults("runs")
    r.update(
        {
            "run_id": run_id,
            "status": "ok",
            "code_version": "testcode",
            "seed": 0,
            "firewall_ok": True,
            "counterfactual_kind": "none",
            # HELD OUT, because `ELIGIBLE` filters on split since 2026-09-15 and the schema
            # default is "". Same argument as `dirty=False` in the e2e fixture below: a fixture
            # that stamps a split no table may report produces empty tables and tests that pass
            # by measuring nothing.
            "split": "test",
            "reconciled": True,
            # A run whose accounting is known to be wrong must not reach a table. See
            # report.ELIGIBLE: docs_ok is the only detector for an unmetered nested retrieval.
            "reconciled_tokens": True,
            "reconciled_docs": True,
        }
    )
    r.update(over)
    r["is_dev_run"] = bool(r["is_dev_run"] or run_id.startswith("dev-"))
    return r


def match_row(run_id: str, node_id: str, kind: str, **over) -> dict:
    """One (run, gold node) match. T4 counts these, so a test about T4's arithmetic has to
    produce them rather than assert on a shape."""
    r = _defaults("matches")
    r.update(
        {
            "run_id": run_id,
            "node_id": node_id,
            "match_kind": kind,
            # The LIVE instrument, not a literal: `report.MATCHER_ID` filters the match join
            # on it, so a hardcoded id silently stops matching the moment the matcher is
            # bumped -- and the fixture's rows vanish from a table that then reads 0.0.
            "matcher_id": MechanicalMatcher.matcher_id,
            "matcher_family": "rule",
            "graph_version": "v1",
        }
    )
    r.update(over)
    return r


def score_row(run_id: str, metric: str, value: float, *, scorer_hash: str = SH, **over) -> dict:
    r = _defaults("scores")
    r.update(
        {
            "run_id": run_id,
            "metric_name": metric,
            "scorer_hash": scorer_hash,
            "value": float(value),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 1,
            "graph_version": GV,
        }
    )
    r.update(over)
    return r


def write_parquet(d: Path, **tables) -> Path:
    """Write the named tables and an EMPTY file for every other declared one.

    `open_agg` requires all nine to exist, exactly as `pi compact` leaves them: a downstream
    query must be a SELECT returning zero rows, never a missing file.
    """
    d.mkdir(parents=True, exist_ok=True)
    for name in sch.TABLES:
        pq.write_table(sch.to_table(name, list(tables.get(name, []))), d / f"{name}.parquet")
    return d


def _pair(
    runs: list,
    scores: list,
    *,
    suite: str,
    task: str,
    arm: str,
    metrics: dict,
    run_id: str | None = None,
    **run_over,
) -> str:
    rid = run_id or f"{suite}-{task}-{arm}"
    runs.append(run_row(rid, suite_id=suite, task_id=task, arm_id=arm, **run_over))
    for m, v in metrics.items():
        scores.append(score_row(rid, m, v))
    return rid


@pytest.fixture
def contaminated_parquet(tmp_path: Path) -> Path:
    """Six clean paired tasks plus one gold_exposed run and one dev- run on the same arm."""
    runs: list = []
    scores: list = []
    for i in range(6):
        _pair(
            runs,
            scores,
            suite="synth",
            task=f"t{i}",
            arm=rp.SYNTH_TREATMENT,
            metrics={"evidence_coverage": 1.0},
        )
        _pair(
            runs,
            scores,
            suite="synth",
            task=f"t{i}",
            arm=rp.SYNTH_COMPARATOR,
            metrics={"evidence_coverage": 0.0},
        )
    # The two that must never reach a reported number.
    _pair(
        runs,
        scores,
        suite="synth",
        task="t6",
        arm=rp.SYNTH_TREATMENT,
        metrics={"evidence_coverage": 1.0},
        run_id="ORACLE_LEAK",
        gold_exposed=True,
    )
    _pair(
        runs,
        scores,
        suite="synth",
        task="t6",
        arm=rp.SYNTH_COMPARATOR,
        metrics={"evidence_coverage": 0.0},
        run_id="dev-DIRTYTREE",
    )
    return write_parquet(tmp_path / "scores" / "parquet", runs=runs, scores=scores)


@pytest.fixture
def bh_parquet(tmp_path: Path) -> Path:
    """EXACTLY two live secondary contrasts, with deliberately different sign patterns.

    `answer_token_f1` is uniformly positive over six tasks (a small p); `rnr_resolve` has one
    task going the other way (a large one). That is what makes BH's step-up behaviour
    observable instead of a formality: the small p clears alpha=0.05 and still fails the
    family threshold.
    """
    runs: list = []
    scores: list = []
    for i in range(6):
        flip = -1.0 if i == 0 else 1.0
        _pair(
            runs,
            scores,
            suite="musique",
            task=f"t{i}",
            arm="inquirer_prompted",
            metrics={
                "answer_token_f1": 0.9,
                "rnr_resolve": 0.5 + 0.4 * flip,
                # not in the declared family: it must come out with a CI and no p at all
                "rnr_ask": 0.95,
            },
        )
        _pair(
            runs,
            scores,
            suite="musique",
            task=f"t{i}",
            arm="drafter_only",
            metrics={"answer_token_f1": 0.5, "rnr_resolve": 0.5, "rnr_ask": 0.55},
        )
    return write_parquet(tmp_path / "scores" / "parquet", runs=runs, scores=scores)


# The drgym PRIMARY endpoint is `kpr_incremental` (pi_eval.prereg.PRIMARY), so that is the
# metric name these fixtures emit. It was `keypoint_recall` here while prereg said
# `kpr_incremental`, which is the exact two-declarations bug tests/test_prereg.py now forbids.
# Both names are judge_derived, so the noise-floor behaviour under test is unchanged.
DRGYM_PRIMARY = "kpr_incremental"


def _drgym_parquet(tmp_path: Path, effect: float, sigma_j: float | None) -> Path:
    runs: list = []
    scores: list = []
    for i in range(6):
        _pair(
            runs,
            scores,
            suite="drgym",
            task=f"t{i}",
            arm="inquirer_prompted",
            metrics={DRGYM_PRIMARY: 0.60 + effect},
        )
        _pair(
            runs,
            scores,
            suite="drgym",
            task=f"t{i}",
            arm="self_inquire",
            metrics={DRGYM_PRIMARY: 0.60},
        )
    if sigma_j is not None:
        (tmp_path / "prereg").mkdir(parents=True, exist_ok=True)
        (tmp_path / "prereg" / "sigma_j.json").write_text(json.dumps({DRGYM_PRIMARY: sigma_j}))
    return write_parquet(tmp_path / "scores" / "parquet", runs=runs, scores=scores)


@pytest.fixture
def cad_parquet(tmp_path: Path) -> Path:
    """C@d rows for depths 0-2, and one depth whose |V_d| is missing on purpose."""
    runs: list = []
    scores: list = []
    for i in range(4):
        rid = _pair(
            runs,
            scores,
            suite="synth",
            task=f"t{i}",
            arm=rp.SYNTH_TREATMENT,
            metrics={"cad#0": 1.0, "cad_n#0": 3.0, "cad#1": 0.8, "cad_n#1": 5.0, "cad#2": 0.5},
        )
        assert rid
        _pair(
            runs,
            scores,
            suite="synth",
            task=f"t{i}",
            arm=rp.SYNTH_COMPARATOR,
            metrics={"cad#0": 0.0, "cad_n#0": 3.0, "cad#1": 0.0, "cad_n#1": 5.0},
        )
    return write_parquet(tmp_path / "scores" / "parquet", runs=runs, scores=scores)


@pytest.fixture
def frontier_parquet(tmp_path: Path) -> Path:
    # musique, NOT synth. `frontier_auc` is a POOLED endpoint and its sealed
    # `pool_suites` deliberately excludes synth -- the calibration suite was
    # contributing 27.4% of pooled rows to confirmatory endpoints. A synth fixture is
    # now filtered out before it reaches the table, so it exercises nothing.
    runs: list = []
    scores: list = []
    for i in range(5):
        ladder = {}
        for k in range(4):
            ladder[f"frontier_spend#{k}"] = float(k)
            ladder[f"frontier_q#{k}"] = min(1.0, 0.25 * k + 0.02 * i)
        _pair(
            runs,
            scores,
            suite="musique",
            task=f"t{i}",
            # the DECLARED secondary frontier contrast is inquirer_prompted vs drafter_only,
            # so the fixture has to speak those arm ids to exercise it
            arm="inquirer_prompted",
            metrics=ladder,
        )
        _pair(
            runs,
            scores,
            suite="musique",
            task=f"t{i}",
            arm="drafter_only",
            metrics={"frontier_spend#0": 0.0, "frontier_q#0": 0.0},
        )
    return write_parquet(tmp_path / "scores" / "parquet", runs=runs, scores=scores)


def agg_of(parquet: Path, **kw):
    kw.setdefault("n_boot", 2000)
    kw.setdefault("n_perm", 10_000)
    return rp.open_agg(parquet, **kw)


# ------------------------------------------------- the end-to-end (real rollouts) fixture


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> dict:
    """Real synthetic rollouts -> compact -> score. LLM-free, so this is CI-cheap."""
    from pi_eval.build.synth_build import build
    from pi_run.compact import compact
    from pi_run.sweep import plan, run_sweep

    root = tmp_path_factory.mktemp("e2e")
    corpus, gold, _ = build(n_tasks=12, n_facets=2, depth=3, seed=11, root=root)
    # THE FOUR HELD-OUT ONES, not the first four. `ELIGIBLE` filters on split since 2026-09-15,
    # and `manifest.split_of` stamps each run from `pinq.splitting` -- so s0, s1 and s3 are
    # train-split and every table built from them would be empty. Exactly the reason `dirty` is
    # False below, one column over. Twelve tasks are built because s2, s5, s8 and s9 are the
    # first four the bucket function calls test.
    specs = plan(
        suite_id="synth",
        corpus_dir=str(corpus.parent),
        task_ids=[f"s{i}" for i in (2, 5, 8, 9)],
        arm_ids=[rp.SYNTH_TREATMENT, rp.SYNTH_COMPARATOR],
        seeds=[0],
        runs_root=str(root / "runs"),
        cache_root=str(root / "cache"),
        # dirty=False is the whole point: a dirty tree yields dev- ids, and dev- ids are
        # mechanically excluded, so an e2e fixture built dirty would produce empty tables.
        code_version="testcode",
        dirty=False,
        concurrency=1,
    )
    run_sweep(specs, concurrency=1)
    compact(root / "runs", root / "scores" / "parquet")
    return {"root": root, "parquet": root / "scores" / "parquet", "gold": gold.parents[2]}


def _score(scored: dict, **kw):
    from pi_eval.score import score

    prior = os.environ.get("PI_GOLD_ROOT")
    os.environ["PI_GOLD_ROOT"] = str(scored["gold"])
    try:
        return score(scored["parquet"], runs_root=scored["root"] / "runs", **kw)
    finally:
        if prior is None:
            os.environ.pop("PI_GOLD_ROOT", None)
        else:
            os.environ["PI_GOLD_ROOT"] = prior


# --------------------------------------------------------------- 1. contamination


def test_gold_exposed_and_dev_runs_are_both_excluded_from_a_primary_table(contaminated_parquet):
    agg = agg_of(contaminated_parquet)
    t = rp.primary_table(agg)
    reached = set(t.provenance["run_ids"])
    assert "ORACLE_LEAK" not in reached, "an oracle arm reached a primary table"
    assert "dev-DIRTYTREE" not in reached, "a dirty-tree run reached a primary table"
    assert len(reached) == 12
    row = next(r for r in t.rows if r["suite"] == "synth")
    assert row["n"] == 6, "the contaminated task must not add a paired unit"
    assert row["delta"] == pytest.approx(1.0)
    assert rp.assert_no_gold_exposed(agg, t.provenance["run_ids"]) == []


def test_assert_no_gold_exposed_exits_non_zero_when_one_sneaks_in(contaminated_parquet, tmp_path):
    """The mechanical guarantee has to survive someone widening the eligibility predicate."""
    from pi_run.cli import main

    leaky = agg_of(contaminated_parquet, allow_contaminated=True)
    t = rp.primary_table(leaky)
    assert "ORACLE_LEAK" in set(t.provenance["run_ids"])
    offenders = rp.assert_no_gold_exposed(leaky, t.provenance["run_ids"])
    assert {o["run_id"] for o in offenders} == {"ORACLE_LEAK", "dev-DIRTYTREE"}

    argv = [
        "agg",
        "--parquet",
        str(contaminated_parquet),
        "--tables",
        str(tmp_path / "tables"),
        "--primary",
        "--assert-no-gold-exposed",
        "--n-boot",
        "200",
        "--n-perm",
        "200",
    ]
    assert main(argv) == 0
    assert main(argv + ["--allow-contaminated"]) == 2


# --------------------------------------------------------------- 2. append-only re-scoring


def test_rescoring_under_a_new_scorer_hash_adds_rows_and_leaves_the_old_ones_untouched(scored):
    first = _score(scored)
    path = scored["parquet"] / "scores.parquet"
    before = rowbytes(pq.read_table(path).to_pylist())
    matches_before = rowbytes(pq.read_table(scored["parquet"] / "matches.parquet").to_pylist())
    status_mtimes = {
        p: p.stat().st_mtime_ns for p in sorted((scored["root"] / "runs").rglob("status.json"))
    }
    assert before and first.scores_rows_added == len(before)

    # A different judge pin is a different scorer, so it must WRITE, not overwrite.
    second = _score(scored, judge_pins=["judge-J1@2026-08-24"])
    assert second.scorer_hash != first.scorer_hash

    after_rows = pq.read_table(path).to_pylist()
    after = rowbytes(after_rows)
    assert len(after) == len(before) + second.scores_rows_added
    assert after[: len(before)] == before, "an existing score row was mutated"
    assert rowbytes(r for r in after_rows if r["scorer_hash"] == first.scorer_hash) == before

    # matches are keyed by (run, node, matcher, graph) and the matcher did not change,
    # so re-scoring must add none of them.
    assert (
        rowbytes(pq.read_table(scored["parquet"] / "matches.parquet").to_pylist()) == matches_before
    )

    # and no rollout was re-rolled: scoring is a groupby, not an experiment.
    assert {
        p: p.stat().st_mtime_ns for p in sorted((scored["root"] / "runs").rglob("status.json"))
    } == status_mtimes

    # re-running the IDENTICAL scorer is a no-op, not a doubling
    again = _score(scored, judge_pins=["judge-J1@2026-08-24"])
    assert again.scores_rows_added == 0
    assert rowbytes(pq.read_table(path).to_pylist()) == after


# --------------------------------------------------------------- 3. multiplicity


def test_bh_fdr_is_applied_to_the_declared_family_and_never_to_exploratory(bh_parquet):
    agg = agg_of(bh_parquet)
    sec = rp.secondary_table(agg)
    live = [r for r in sec.rows if r["n"] > 0]
    assert len(live) == 2, [r["metric"] for r in live]
    assert all(r["bh_family_size"] == 2 for r in live)

    small = min(live, key=lambda r: r["p"])
    assert 0.025 < small["p"] < 0.05, small["p"]
    assert small["bh_reject"] is False, (
        "a p below alpha=0.05 was reported as a discovery without passing the family "
        "threshold: BH was not applied"
    )
    assert "bh_reject" in sec.columns and "p" in sec.columns

    expl = rp.exploratory_table(agg)
    assert "p" not in expl.columns, "an exploratory metric was given a p-value"
    assert "bh_reject" not in expl.columns, "BH was applied to an undeclared set"
    assert expl.rows, "the exploratory table should not be empty for this fixture"
    assert all(r["label"] == "exploratory" for r in expl.rows)
    assert all("exploratory" in r["method"] for r in expl.rows)


# --------------------------------------------------------------- 4. noise floor


def test_below_noise_floor_is_a_column_and_never_an_exception(tmp_path):
    """An effect under 2*sigma_J/sqrt(n) prints. It does not raise: a linter that throws at
    2am is a linter that is disabled by 2:05am."""
    small = _drgym_parquet(tmp_path, effect=0.001, sigma_j=0.15)
    t = rp.primary_table(agg_of(small))  # must NOT raise
    assert "below_noise_floor" in t.columns
    row = next(r for r in t.rows if r["metric"] == DRGYM_PRIMARY)
    assert row["below_noise_floor"] is True
    assert row["noise_floor"] == pytest.approx(2 * 0.15 / 6**0.5)
    assert any("BELOW NOISE FLOOR" in w for w in t.warnings)
    tex = rd.to_latex(t)
    assert r"below\_noise\_floor" in tex, "the flag must survive into the rendered table"
    assert "WARNING" in tex and "BELOW NOISE FLOOR" in tex

    shutil.rmtree(tmp_path / "scores")
    big = _drgym_parquet(tmp_path, effect=0.5, sigma_j=0.15)
    row = next(r for r in rp.primary_table(agg_of(big)).rows if r["metric"] == DRGYM_PRIMARY)
    assert row["below_noise_floor"] is False


def test_an_unmeasured_sigma_j_flags_rather_than_claims(tmp_path):
    """No sigma_J yet means the floor is +inf, so nothing judge-derived is reportable. The
    conservative direction is to flag; silently claiming is the failure being prevented."""
    p = _drgym_parquet(tmp_path, effect=0.5, sigma_j=None)
    row = next(r for r in rp.primary_table(agg_of(p)).rows if r["metric"] == DRGYM_PRIMARY)
    assert row["below_noise_floor"] is True
    assert "sigma_J not yet measured" in row["notes"]


def test_a_mechanical_metric_is_never_flagged_below_a_judge_floor(contaminated_parquet):
    row = next(r for r in rp.primary_table(agg_of(contaminated_parquet)).rows)
    assert row["metric"] == "evidence_coverage"
    assert row["below_noise_floor"] is False and row["noise_floor"] == 0.0


# --------------------------------------------------------------- 5. render refusals


def test_render_refuses_when_an_input_hash_changed(contaminated_parquet, tmp_path):
    tables = tmp_path / "tables"
    agg = agg_of(contaminated_parquet)
    rd.render_table(rp.primary_table(agg), tables)
    assert (tables / "T1_primary.tex").exists()

    # An input file moves. Anything downstream of it is no longer the table that was
    # aggregated, so rendering must stop rather than quietly restate it.
    runs = contaminated_parquet / "runs.parquet"
    rows = pq.read_table(runs).to_pylist()
    rows.append(run_row("LATE_ARRIVAL", suite_id="synth", task_id="t9", arm_id="drafter_only"))
    pq.write_table(sch.to_table("runs", rows), runs)

    fresh = agg_of(contaminated_parquet)
    with pytest.raises(rd.RenderRefused, match="input hash changed"):
        rd.render_table(rp.primary_table(fresh), tables)

    res = rd.render_all(fresh, tables_dir=tables, figures_dir=tmp_path / "figs", figure_ids=[])
    assert any("input hash changed" in m for m in res.refused)
    assert res.as_dict()["ok"] is False

    from pi_run.cli import main

    assert (
        main(
            [
                "render",
                "--parquet",
                str(contaminated_parquet),
                "--tables",
                str(tables),
                "--figures",
                str(tmp_path / "figs"),
                "--table",
                "T1_primary",
                "--figure",
                "F2_coverage_at_depth",
                "--n-boot",
                "200",
            ]
        )
        == 2
    )


def test_render_refuses_to_overwrite_a_hand_edited_table(contaminated_parquet, tmp_path):
    tables = tmp_path / "tables"
    agg = agg_of(contaminated_parquet)
    tex = rd.render_table(rp.primary_table(agg), tables)

    body = tex.read_text()
    assert "1.0000" in body
    tex.write_text(body.replace("1.0000", "9.9999"))

    with pytest.raises(rd.RenderRefused, match="edited by hand"):
        rd.render_table(rp.primary_table(agg), tables)
    # --force is the only way past it, and it exists so the refusal is never "fixed" by
    # deleting the check.
    rd.render_table(rp.primary_table(agg), tables, force=True)
    assert "9.9999" not in tex.read_text()


# --------------------------------------------------------------- 6. C@d carries |V_d|


def test_cad_cells_always_carry_the_cardinality(cad_parquet):
    t = rp.cad_table(agg_of(cad_parquet))
    assert t.columns[:3] == ("suite", "arm", "n_tasks")
    depth_cols = t.columns[3:]
    assert depth_cols and all(re.fullmatch(r"C@\d+ \(\|V_\d+\|\)", c) for c in depth_cols)
    seen = 0
    for row in t.rows:
        for c in depth_cols:
            cell = row[c]
            if cell == "-":
                continue
            assert re.fullmatch(r"\d\.\d{3} \(\d+\)", cell), f"{c}={cell!r} has no |V_d|"
            seen += 1
    assert seen, "no populated C@d cell in the fixture"
    # depth 2 has a recall but no cardinality, so it must print as unavailable rather than
    # as a bare number a reader would compare against a depth with forty nodes in it.
    assert all(row["C@2 (|V_2|)"] == "-" for row in t.rows)


def test_cad_survives_the_round_trip_to_latex(cad_parquet):
    tex = rd.to_latex(rp.cad_table(agg_of(cad_parquet)))
    assert r"C@0 (\|V\_0\|)" in tex or "C@0" in tex
    assert "1.000 (3)" in tex


# --------------------------------------------------------------- 7. the frontier


def test_frontier_figure_renders_and_carries_its_curve(frontier_parquet, tmp_path):
    agg = agg_of(frontier_parquet)
    fig = rd.f1_frontier(agg, tmp_path / "figures", n_boot=200)
    assert fig.pdf.exists() and fig.pdf.stat().st_size > 0
    assert fig.png.exists() and fig.png.stat().st_size > 0
    for arm, blob in fig.summary["arms"].items():
        assert "auc" in blob, arm
        # the scalar is NEVER alone: the curve and both bands travel with it
        for k in ("mean", "lo_pointwise", "hi_pointwise", "lo_simultaneous", "hi_simultaneous"):
            assert len(blob[k]) == len(fig.summary["grid"]), (arm, k)
    assert json.loads((tmp_path / "figures" / "F1_frontier" / "provenance.json").read_text())


def test_a_scalar_auc_without_a_curve_raises(frontier_parquet):
    nan = float("nan")
    naked = Curve(
        grid=(1.0, 2.0, 4.0),
        mean=(nan, nan, nan),
        lo_pointwise=(nan,) * 3,
        hi_pointwise=(nan,) * 3,
        lo_simultaneous=(nan,) * 3,
        hi_simultaneous=(nan,) * 3,
        auc=0.42,
        auc_lo=0.40,
        auc_hi=0.44,
        n_tasks=0,
    )
    with pytest.raises(rp.ScalarAucWithoutCurve):
        rp.auc_row(naked)
    # and the real curve does hand one back
    curves, _ = rp.frontier_curves(agg_of(frontier_parquet), n_boot=200)
    assert rp.auc_row(curves["inquirer_prompted"])["auc"] > 0


def test_render_refuses_a_delta_auc_row_with_no_figure(frontier_parquet, tmp_path):
    """The scalar hides curve crossings, so it may not be published on its own."""
    agg = agg_of(frontier_parquet)
    assert any(r["metric"] == "frontier_auc" and r["n"] > 0 for r in rp.secondary_table(agg).rows)
    with pytest.raises(rd.RenderRefused, match="never be emitted"):
        rd.render_all(
            agg,
            tables_dir=tmp_path / "tables",
            figures_dir=tmp_path / "figures",
            table_ids=["T1b_secondary"],
            figure_ids=[],
        )


# --------------------------------------------------------------- 8. provenance


def test_every_generated_table_has_a_provenance_json_naming_its_inputs(scored, tmp_path):
    res = _score(scored)
    agg = agg_of(scored["parquet"], scorer_hash=res.scorer_hash)
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    res = rd.render_all(agg, tables_dir=tables, figures_dir=figures, n_boot=200)
    assert not res.refused, res.refused
    assert set(res.tables) == set(rp.ALL_TABLES)
    assert set(res.figures) == set(rd.FIGURES)

    for tid in res.tables:
        prov = json.loads((tables / tid / "provenance.json").read_text())
        assert isinstance(prov["run_ids"], list)
        assert prov["scorer_hash"] == agg.scorer_hash
        assert prov["graph_version"] == GV
        assert prov["code_version"] == ["testcode"]
        assert prov["query"].strip()
        assert prov["inputs"]["runs.parquet"] != "ABSENT"
        assert prov["inputs_hash"] and prov["provenance_digest"]
        assert (tables / f"{tid}.tex").stat().st_size > 0
    for fid in res.figures:
        assert (figures / f"{fid}.pdf").stat().st_size > 0
        assert json.loads((figures / fid / "provenance.json").read_text())["scorer_hash"]

    # a table with no run behind it would be a number with no provenance
    t1 = json.loads((tables / "T1_primary" / "table.json").read_text())
    assert t1["rows"] and t1["digest"]
    assert json.loads((tables / "T1_primary" / "provenance.json").read_text())["n_run_ids"] == 8


def test_compact_never_clobbers_the_score_rows_it_does_not_own(scored):
    """Compaction owns the six run-side tables and nothing else.

    Rewriting matches/scores empty on every compaction would destroy the append-only history
    the moment one new rollout landed - and it would do it silently, leaving an aggregation
    that raises 'scores.parquet is empty' hours after the rows were lost.
    """
    from pi_run.compact import compact

    _score(scored)
    path = scored["parquet"] / "scores.parquet"
    before = rowbytes(pq.read_table(path).to_pylist())
    assert before

    res = compact(scored["root"] / "runs", scored["parquet"])
    assert res.counts["scores"] == len(before)
    assert rowbytes(pq.read_table(path).to_pylist()) == before


def test_open_agg_refuses_to_guess_between_two_scorer_hashes(contaminated_parquet):
    """A number must name exactly one scorer_hash. Picking 'the latest' is how a table ends
    up half-scored under one matcher and half under another."""
    path = contaminated_parquet / "scores.parquet"
    rows = pq.read_table(path).to_pylist()
    rows += [{**r, "scorer_hash": SH2} for r in rows]
    pq.write_table(sch.to_table("scores", rows), path)
    with pytest.raises(rp.AggregationError, match="several scorer_hashes"):
        agg_of(contaminated_parquet)
    agg = agg_of(contaminated_parquet, scorer_hash=SH2)
    assert rp.primary_table(agg).rows


def test_killswitch_prints_a_rule_for_every_comparator_including_unrun_ones(contaminated_parquet):
    """An arm that was never run must say so. A silent absence in the one command that tells
    you whether to stop is the worst possible failure mode."""
    t = rp.killswitch_table(agg_of(contaminated_parquet), tasks=6)
    assert [r["comparator"] for r in t.rows] == [a for a, _ in rp.KILLSWITCH_RULES]
    assert all(r["verdict"] == "NOT RUN" for r in t.rows)
    for arm, rule in rp.KILLSWITCH_RULES:
        assert any(arm in n and rule in n for n in t.notes)
    assert t.provenance["scorer_hash"] == SH


def test_a_run_that_failed_reconciliation_never_reaches_a_table(tmp_path):
    """`reconciled_docs` is the ONLY detector for an unmetered nested retrieval: a Drafter that
    retrieves inside resolve() and folds the result into its returned Evidence is invisible in
    tokens, invisible in retrieval_calls, and fatal to a budget-parity claim.

    Both flags were computed, written to status.json and compacted into these columns -- and
    read by nothing. `worker.run_unit` raises on a token mismatch only for llm_free arms, i.e.
    only where the ledger is empty and the check was already trivially true. So a run KNOWN to
    be mis-metered contributed to every reported table, and the field documented the failure
    instead of preventing it.
    """
    d = write_parquet(
        tmp_path / "pq",
        runs=[
            run_row("ok1", arm_id="inquirer_prompted", task_id="t1"),
            run_row("ok2", arm_id="drafter_only", task_id="t1"),
            run_row("bad1", arm_id="inquirer_prompted", task_id="t2", reconciled_docs=False),
            run_row("bad2", arm_id="drafter_only", task_id="t2", reconciled_tokens=False),
        ],
        scores=[
            score_row("ok1", "evidence_coverage", 0.9),
            score_row("ok2", "evidence_coverage", 0.4),
            score_row("bad1", "evidence_coverage", 0.9),
            score_row("bad2", "evidence_coverage", 0.4),
        ],
    )
    agg = agg_of(d)
    pull = rp.pull(agg, "evidence_coverage")
    assert set(pull.run_ids) == {"ok1", "ok2"}, sorted(pull.run_ids)
    assert "reconciled_docs" in pull.query and "reconciled_tokens" in pull.query, (
        "the predicate is inlined into provenance verbatim so a reviewer can paste it"
    )


# --------------------------------------------------------------- eligibility filters on split
#
# THE DECISION, TAKEN 2026-09-15 (plan v4 SS8.6), AS TESTS. `ELIGIBLE` filtered on status, gold
# exposure, dirty trees, pilots, canaries, the firewall and reconciliation -- and NOT on split,
# so a run on a task the policy may have trained on was eligible to reach a published table.
# Harmless while every arm was prompted, and unrecoverable at the first trained arm: the table
# renders, the CI is tight, and nothing in the output says which split it came from.
#
# The alternative was `split: test` in each confirmatory grid: narrower, and one forgotten line
# away from recurring in the next grid anyone writes. A predicate is enforced everywhere and
# cannot be forgotten, which is the argument `report.py` already makes for itself in its own
# first paragraph.
#
# Measured on the compacted store the day the clause landed: 34,155 train and 1,976 dev rows
# left the eligible set and 1,537 test rows remain, so every table rendered before it must be
# re-rendered.


def _split_parquet(tmp_path: Path) -> Path:
    """One run per split, identical in every other respect.

    Same arm and same metric on all three, so the ONLY thing that can separate them is the
    split clause -- a fixture that also varied the arm would pass for the wrong reason.
    """
    return write_parquet(
        tmp_path / "pq",
        runs=[
            run_row("held_out", arm_id="inquirer_prompted", task_id="t1", split="test"),
            run_row("fitted", arm_id="inquirer_prompted", task_id="t2", split="train"),
            run_row("selected_on", arm_id="inquirer_prompted", task_id="t3", split="dev"),
        ],
        scores=[
            score_row("held_out", "evidence_coverage", 0.5),
            score_row("fitted", "evidence_coverage", 0.5),
            score_row("selected_on", "evidence_coverage", 0.5),
        ],
    )


def test_a_train_split_run_never_reaches_a_table(tmp_path):
    """The failure the clause exists for: a number computed over tasks a policy trained on."""
    pull = rp.pull(agg_of(_split_parquet(tmp_path)), "evidence_coverage")
    assert "fitted" not in set(pull.run_ids), sorted(pull.run_ids)


def test_a_dev_split_run_never_reaches_a_table(tmp_path):
    """DEV IS NOT HELD OUT for a checkpoint that was SELECTED on it -- `pi train gate` reads
    the dev rollouts to choose which checkpoint to promote, so a dev number is one the
    selection already optimised. The same reason `train_split_violations` demands `test` on
    both sides rather than merely refusing `train`."""
    pull = rp.pull(agg_of(_split_parquet(tmp_path)), "evidence_coverage")
    assert "selected_on" not in set(pull.run_ids), sorted(pull.run_ids)


def test_a_test_split_run_with_every_other_flag_clean_is_still_eligible(tmp_path):
    """The other half, and the one that makes the two above mean something: the clause must
    remove exactly the non-held-out rows and nothing else. A predicate that excluded everything
    would satisfy both tests above and report no numbers at all."""
    pull = rp.pull(agg_of(_split_parquet(tmp_path)), "evidence_coverage")
    assert set(pull.run_ids) == {"held_out"}, sorted(pull.run_ids)


def test_the_predicate_inlined_into_a_table_carries_the_split_clause(contaminated_parquet):
    """Rule 1: a number traces to what produced it. The predicate is recorded VERBATIM in every
    provenance record so a reviewer can paste it into duckdb and get the same rows back -- so
    the split filter has to be IN that text, not applied somewhere a reader cannot see."""
    agg = agg_of(contaminated_parquet)
    t = rp.primary_table(agg)
    assert "r.split = 'test'" in agg.predicate
    assert "r.split = 'test'" in t.provenance["eligibility_predicate"]
    assert "r.split = 'test'" in t.provenance["query"]


# --------------------------------------------------------------------------- orphaned scores
#
# `pi compact` preserved scores.parquet deliberately -- an append-only guarantee -- and
# REPLACED runs.parquet with whatever --runs-root held. Every reporting query INNER JOINs
# scores -> runs, so compacting a different runs-root into the same directory evicted the runs
# and left their score rows joinable to nothing. Measured: 228 score rows, 0 joinable, and no
# error anywhere.


def _synth_sweep(tmp_path, name, arm):
    from pi_eval.build.synth_build import build
    from pi_run.worker import UnitSpec, run_unit

    corpus, _gold, _h = build(n_tasks=4, n_facets=2, depth=2, root=tmp_path)
    runs = tmp_path / name
    for tid in ("s0", "s1"):
        run_unit(
            UnitSpec(
                suite_id="synth",
                corpus_dir=str(corpus.parent),
                task_id=tid,
                arm_id=arm,
                seed=0,
                runs_root=str(runs),
                cache_root=str(tmp_path / "cache"),
                code_version="t",
                dirty=False,
            )
        )
    return runs


def test_compacting_a_second_runs_root_does_not_orphan_the_first_ones_scores(tmp_path):
    """The run side now behaves like the score side: additive, keyed on run_id, idempotent."""
    import pyarrow.parquet as _pq

    from pi_run.compact import compact

    a = _synth_sweep(tmp_path, "A", "fake_chain")
    out = tmp_path / "pq"
    compact(a, out, include_dev=True)
    ids_a = {r["run_id"] for r in _pq.read_table(out / "runs.parquet").to_pylist()}
    assert ids_a

    b = _synth_sweep(tmp_path, "B", "fake_drafter_only")
    res = compact(b, out, include_dev=True)
    ids = {r["run_id"] for r in _pq.read_table(out / "runs.parquet").to_pylist()}

    assert ids_a <= ids, "compacting B evicted A's runs"
    assert len(ids) == len(ids_a) * 2
    assert res.orphaned_score_rows == 0


def test_recompacting_the_same_runs_root_is_still_idempotent(tmp_path):
    """Merging must not duplicate. A row for a run this pass DID see is replaced by the fresh
    one; only rows for runs it did not see are carried forward."""
    from pi_run.compact import compact

    a = _synth_sweep(tmp_path, "A", "fake_chain")
    out = tmp_path / "pq"
    first = compact(a, out, include_dev=True)
    second = compact(a, out, include_dev=True)
    assert first.counts["runs"] == second.counts["runs"]
    assert first.counts["turns"] == second.counts["turns"]


def test_replace_still_drops_runs_and_counts_what_it_orphaned(tmp_path):
    """Deliberately dropping runs is a real need. Doing it silently is not: an orphan is a row
    that exists on disk and appears in NO table."""
    import pyarrow.parquet as _pq

    from pi_run.compact import compact

    a = _synth_sweep(tmp_path, "A", "fake_chain")
    out = tmp_path / "pq"
    compact(a, out, include_dev=True)

    # A score row for one of A's runs, which --replace is about to strand.
    run_id = _pq.read_table(out / "runs.parquet").to_pylist()[0]["run_id"]
    write_parquet(out, scores=[score_row(run_id, "evidence_coverage", 0.5)])

    b = _synth_sweep(tmp_path, "B", "fake_drafter_only")
    # write_parquet blanked the run side, so re-establish B and then replace with B only.
    compact(b, out, include_dev=True)
    res = compact(b, out, include_dev=True, replace=True)
    assert res.orphaned_score_rows >= 1, "a dropped run's score rows must be counted"


def test_the_cli_warns_on_stderr_when_score_rows_are_orphaned():
    """A count buried in JSON is a count nobody reads on the day it matters."""
    import inspect

    from pi_run import cli

    src = inspect.getsource(cli.cmd_compact)
    assert "orphaned_score_rows" in src
    assert "appear in" in src and "NO table" in src
    assert "file=sys.stderr" in src


def test_f2_keys_its_series_by_suite_and_arm_so_no_suite_overwrites_another(tmp_path):
    """`cad_query` groups by (suite_id, arm_id, depth, fam); the accumulator keyed on
    (arm, depth) alone, so with more than one suite in the parquet the LAST suite in
    `ORDER BY suite_id` silently overwrote the others and the figure showed one suite's numbers
    under a label naming only the arm. tier1_confirmatory runs THREE suites at once, so that is
    the ordinary case rather than an edge one."""
    pytest.importorskip("matplotlib")
    from pi_run.render import f2_coverage_at_depth

    runs, scores = [], []
    for suite, cov, card in (("musique", 0.9, 10.0), ("strategyqa", 0.1, 4.0)):
        rid = f"r_{suite}"
        runs.append(run_row(rid, suite_id=suite, arm_id="inquirer_prompted", task_id="t1"))
        scores.append(score_row(rid, "cad#2", cov))
        scores.append(score_row(rid, "cad_n#2", card))
    d = write_parquet(tmp_path / "pq", runs=runs, scores=scores)
    fig = f2_coverage_at_depth(agg_of(d), tmp_path / "figs")

    series = fig.summary["series"]
    assert set(series) == {"musique/inquirer_prompted", "strategyqa/inquirer_prompted"}
    assert series["musique/inquirer_prompted"][2] == pytest.approx(0.9)
    assert series["strategyqa/inquirer_prompted"][2] == pytest.approx(0.1)
    # |V_d| is per (suite, depth) and SUMMED for the tick label, not max'd -- max reported one
    # suite's cardinality as everyone's.
    assert sum(fig.summary["cardinality"].values()) == pytest.approx(14.0)
    assert set(fig.summary["cardinality"]) == {"musique/2", "strategyqa/2"}


def test_t4_reports_needs_per_run_not_a_total_in_a_table_of_means(tmp_path):
    """T4's title is "per-run means" and every other column is one. `needs_discovered` was a
    COUNT over match rows across every run in the cell, so an arm that contributed more runs
    showed a bigger number for identical behaviour and a reader comparing the column across
    arms was comparing run counts."""
    runs, scores, matches = [], [], []
    # Two arms, identical per-run behaviour, different numbers of runs.
    for arm, n in (("inquirer_prompted", 4), ("drafter_only", 2)):
        for i in range(n):
            rid = f"{arm}_{i}"
            runs.append(run_row(rid, suite_id="musique", arm_id=arm, task_id=f"t{i}"))
            scores.append(score_row(rid, "usd", 0.01))
            for node in ("n1", "n2"):  # exactly two needs resolved per run, in both arms
                matches.append(match_row(rid, node, "resolve", suite_id="musique", task_id=f"t{i}"))
    d = write_parquet(tmp_path / "pq", runs=runs, scores=scores, matches=matches)
    table = rp.ops_table(agg_of(d))

    by_arm = {r["arm"]: r for r in table.rows}
    assert "needs_per_run" in table.columns
    assert "needs_discovered" not in table.columns
    assert by_arm["inquirer_prompted"]["needs_per_run"] == pytest.approx(2.0)
    assert by_arm["drafter_only"]["needs_per_run"] == pytest.approx(2.0), (
        "identical per-run behaviour must give identical numbers whatever the run count"
    )


def test_t4_multiplies_the_usd_mean_by_the_usd_run_count(tmp_path):
    """`n_runs` is count(DISTINCT run_id) WITHIN (suite, arm, metric), and metrics are not
    emitted by the same runs -- tau_reward lands on 88 of tau2's 97 tasks. Keeping one n_runs
    per cell meant whichever metric the loop saw LAST supplied the multiplier for usd_total."""
    runs, scores, matches = [], [], []
    for i in range(4):
        rid = f"r{i}"
        runs.append(run_row(rid, suite_id="tau2", arm_id="a", task_id=f"t{i}"))
        scores.append(score_row(rid, "usd", 0.10))
        matches.append(match_row(rid, "n1", "resolve", suite_id="tau2", task_id=f"t{i}"))
    # A second metric emitted by only ONE of the four runs -- the shape tau_reward has.
    scores.append(score_row("r0", "tok_total", 1234.0))
    d = write_parquet(tmp_path / "pq", runs=runs, scores=scores, matches=matches)

    row = rp.ops_table(agg_of(d)).rows[0]
    # 4 runs x $0.10 = $0.40 over 4 resolved needs.
    assert row["usd_per_need*"] == pytest.approx(0.10)


def test_provenance_records_how_the_interval_was_resampled(tmp_path):
    """Rule 1: a reported value traces to what produced it. Every CI here is a bootstrap and
    every p a permutation, so the same rows under `--seed 1` give a different interval -- and
    without seed/n_boot/n_perm the artifact cannot say which one it is."""
    d = write_parquet(
        tmp_path / "pq",
        runs=[run_row("r1", arm_id="inquirer_prompted"), run_row("r2", arm_id="drafter_only")],
        scores=[
            score_row("r1", "evidence_coverage", 0.8),
            score_row("r2", "evidence_coverage", 0.4),
        ],
    )
    prov = rp.arm_ladder_table(agg_of(d, seed=7, n_boot=321, n_perm=654)).provenance
    assert prov["resampling"] == {"seed": 7, "n_boot": 321, "n_perm": 654}
    # And the rest of rule 1 is still there.
    for k in ("run_ids", "scorer_hash", "graph_version", "eligibility_predicate", "query"):
        assert k in prov


def test_an_absent_value_typesets_as_not_applicable_and_never_as_a_number(tmp_path):
    """The distinction this codebase keeps fighting for, at the last step where it can be lost.

    `usd_per_need` is NaN for an arm that resolved no needs -- ABSENT, not zero. Rendering it
    as 0.00 would claim the arm found needs for free; rendering it as the word "nan" leaves a
    reader to decide whether that is a bug or a value. The stored JSON keeps the NaN, so
    nothing downstream is told a different story from the reader.
    """
    import json as _json
    import math

    from pi_run.render import to_latex

    runs = [run_row("r1", suite_id="synth", arm_id="a", task_id="t1")]
    scores = [score_row("r1", "usd", 0.0)]
    d = write_parquet(tmp_path / "pq", runs=runs, scores=scores)
    table = rp.ops_table(agg_of(d))
    tex = to_latex(table)

    body = [ln for ln in tex.splitlines() if "&" in ln and ln.rstrip().endswith(r"\\")]
    assert body, tex
    nan_cells = [
        (r, k) for r in table.rows for k, v in r.items() if isinstance(v, float) and math.isnan(v)
    ]
    assert nan_cells, "this test needs an absent value to be meaningful"

    data_rows = [ln for ln in body if "textbf" not in ln]
    assert rp.NOT_APPLICABLE in " ".join(data_rows)
    assert " nan " not in " ".join(data_rows)

    # And the JSON is untouched: the reader's rendering is not the record.
    assert any(
        isinstance(v, float) and math.isnan(v)
        for r in _json.loads(_json.dumps(table.rows, default=str)) or table.rows
        for v in (r.values() if isinstance(r, dict) else [])
    )
