"""Whether the two breadth-over-independent-needs instruments are a second axis, or coverage
restated. Every number in artifacts/horizontal_axis_separability/RESULT.md and separability.json
comes from running this file, unchanged, against the two read-only stores it names. Nothing here
writes to either store. This file lives under scripts/, not artifacts/, because artifacts/ is
excluded from ruff on the recorded ground that no python lives there (tests/test_artifacts_hold_no_python.py).

Run from the repository root (the checkout that holds scores/parquet, artifacts/gate and
artifacts/testsplit_qa, none of which are copied into a fresh worktree):

    python3 scripts/horizontal_axis_separability/build_separability.py

Three questions, each answered against a named population with its own scorer_hash and
graph_version, never pooled across them:

  1. STRUCTURAL CENSUS. What is the denominator (total facets, total prerequisite components)
     on every suite, in the exact population behind the paper's 46-checkpoint gate table
     (scorer_hash 02d44f6984948372faab9c8ff0b1253653d73f75ab9ca131dc2ecbe68b680114) and in the
     full corpus. A metric whose denominator never varies cannot record more than one bit.

  2. IDENTITY CHECK. On MuSiQue and StrategyQA, is the facet reading a row-for-row restatement
     of a quantity the paper already reports under another name (max_depth_reached >= 1), on the
     held-out matched-cost population (scorer_hash aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28,
     the store artifacts/testsplit_qa/scores_parquet and the population
     artifacts/testsplit_plan_metrics_20260918 already built and locked).

  3. SEPARABILITY. On the exact paired population inquirer_trained/inquirer_prompted share
     (seed-averaged, inner-joined on task_id, matching the peer's own 166/119-task wiki2
     population to the row), does a task's breadth level predict its coverage level, and does
     the CHANGE in breadth between the two arms on a task predict the change in coverage on
     that task. Both are computed on all three suites so wiki2 (where breadth is not fixed at
     one bit) can be read against musique and strategyqa (where Section 2 already proves it a
     restatement). Also, on the 26-checkpoint development gate table, how strongly does the
     cross-checkpoint facet delta correlate with the cross-checkpoint coverage delta, per suite.

  4. THE DEPTH-BREADTH READING AT LINE ~208 OF paper/results.tex, AND WHETHER A HORIZONTAL
     CONTROL EXISTS. Reads the two fields the paper's own text says get confused (evidence_coverage
     at matched cost and cad_ge2, which cannot be read at matched cost and falls back to a
     cap8 value stored under the same field name) directly out of the verdict JSON, and reports
     whether facet_breadth and cad_ge2 agree in sign once both are read on the one basis they are
     actually both computed on. Separately, reads inquirer_depth1's own note and kill-switch text
     to check whether a symmetric horizontal-only control exists anywhere in the arm registry.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import duckdb

HEADLINE_SCORER_HASH = "02d44f6984948372faab9c8ff0b1253653d73f75ab9ca131dc2ecbe68b680114"
TESTSPLIT_SCORER_HASH = "aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28b594a5ab1d0cc5b930e0ba7"

MAIN_RUNS = "scores/parquet/runs.parquet"
MAIN_SCORES = "scores/parquet/scores.parquet"
HELD_RUNS = "artifacts/testsplit_qa/scores_parquet/runs.parquet"
HELD_SCORES = "artifacts/testsplit_qa/scores_parquet/scores.parquet"

GATE_DIR_GLOBS = ("artifacts/gate/*-t20/parquet_*", "artifacts/gate/n1/parquet_*")


def _require(path: str) -> None:
    if not Path(path).exists():
        raise SystemExit(
            f"missing {path}. This script reads the main checkout's data, gitignored or "
            "untracked stores that a fresh worktree does not carry. Run it from the checkout "
            "that has scores/parquet, artifacts/gate and artifacts/testsplit_qa."
        )


def structural_census(con: duckdb.DuckDBPyConnection) -> dict:
    _require(MAIN_RUNS)
    _require(MAIN_SCORES)
    con.execute(f"create or replace view main_runs as select * from read_parquet('{MAIN_RUNS}')")
    con.execute(
        f"create or replace view main_scores as select * from read_parquet('{MAIN_SCORES}')"
    )
    out: dict = {}
    for metric in ("facet_breadth", "breadth_components"):
        rows = con.execute(
            f"""
            select r.suite_id, r.split, s.n, count(*) as rows
            from main_scores s join main_runs r on s.run_id = r.run_id
            where s.metric_name = '{metric}'
            group by 1,2,3 order by 1,2,3
            """
        ).fetchall()
        out[metric] = [{"suite_id": a, "split": b, "n": c, "rows": d} for a, b, c, d in rows]
    return out


def gated_population_census(con: duckdb.DuckDBPyConnection) -> dict:
    dirs = sorted({p for g in GATE_DIR_GLOBS for p in glob.glob(g)})
    runs_files = json.dumps([f"{d}/runs.parquet" for d in dirs])
    scores_files = json.dumps([f"{d}/scores.parquet" for d in dirs])
    con.execute(f"create or replace view gate_runs as select * from read_parquet({runs_files})")
    con.execute(f"create or replace view gate_scores as select * from read_parquet({scores_files})")
    out: dict = {"n_verdict_dirs": len(dirs), "scorer_hashes": None}
    out["scorer_hashes"] = [
        r[0] for r in con.execute("select distinct scorer_hash from gate_scores").fetchall()
    ]
    for metric in ("facet_breadth", "breadth_components"):
        rows = con.execute(
            f"""
            select r.suite_id, count(distinct r.task_id) as n_tasks, s.n,
                   count(*) as rows, min(s.value) as vmin, max(s.value) as vmax,
                   avg(s.value) as vavg, count(distinct s.graph_version) as n_graph_version
            from gate_scores s join gate_runs r on s.run_id = r.run_id
            where s.metric_name = '{metric}' and s.scorer_hash = '{HEADLINE_SCORER_HASH}'
            group by 1,3 order by 1,3
            """
        ).fetchall()
        out[metric] = [
            {
                "suite_id": a,
                "n_tasks": b,
                "n": c,
                "rows": d,
                "value_min": e,
                "value_max": f,
                "value_mean": g,
                "n_graph_version": h,
            }
            for a, b, c, d, e, f, g, h in rows
        ]
    return out


def identity_check(con: duckdb.DuckDBPyConnection) -> dict:
    _require(HELD_RUNS)
    _require(HELD_SCORES)
    con.execute(f"create or replace view held_runs as select * from read_parquet('{HELD_RUNS}')")
    con.execute(
        f"create or replace view held_scores as select * from read_parquet('{HELD_SCORES}')"
    )
    con.execute(
        """
        create or replace view held_join as
        with fb as (select run_id, value as fb from held_scores where metric_name='facet_breadth'),
             mdr as (select run_id, value as mdr from held_scores where metric_name='max_depth_reached'),
             bc as (select run_id, value as bc from held_scores where metric_name='breadth_components')
        select r.suite_id, fb.fb, mdr.mdr, bc.bc
        from fb join mdr on mdr.run_id = fb.run_id
        join bc on bc.run_id = fb.run_id
        join held_runs r on r.run_id = fb.run_id
        where r.suite_id in ('musique', 'strategyqa')
        """
    )
    cross = con.execute(
        "select suite_id, fb, (mdr >= 1) as depth_ge1, count(*) "
        "from held_join group by 1,2,3 order by 1,2,3"
    ).fetchall()
    agree_bc = con.execute(
        "select suite_id, (fb = bc) as agree, count(*) from held_join group by 1,2 order by 1,2"
    ).fetchall()
    return {
        "facet_breadth_vs_max_depth_reached_ge1": [
            {"suite_id": a, "facet_breadth": b, "depth_ge1": c, "rows": d} for a, b, c, d in cross
        ],
        "facet_breadth_vs_breadth_components_agreement": [
            {"suite_id": a, "agree": b, "rows": c} for a, b, c in agree_bc
        ],
    }


def separability(con: duckdb.DuckDBPyConnection) -> dict:
    """Corrected methodology (superseded a naive `arm_id in (...)` filter that UNIONED
    inquirer_trained's 200 distinct wiki2 test tasks with inquirer_prompted's 166, instead of
    intersecting them, and so reported a correlation on a population no gate verdict actually
    uses). Seeds are averaged into (suite_id, arm_id, task_id) first, then inquirer_trained is
    INNER JOINed against inquirer_prompted on task_id. On wiki2 this reproduces the peer's own
    paired population exactly: 166 tasks, 119 with facet_total > 0 -- confirmed against
    artifacts/testsplit_plan_metrics_20260918/contrasts.json's wiki2 facet_breadth matched
    entry (n_pairs=332 runs, task.n=166) before trusting this function's own count.

    Reports two different relationships, because they answer different questions:
      * level-level: does a task's breadth VALUE predict its coverage value, within one arm.
      * delta-level: does the CHANGE in breadth between the trained and prompted arm, on a
        task, predict the change in coverage on that same task. This is the direct test of
        "is a breadth gain the coverage gain restated in another unit" and is computed on all
        three suites so the wiki2 reading (the only place breadth has more than one bit, see
        `identity_check`) can be compared against musique and strategyqa (where it is proved
        identical to `max_depth_reached >= 1`, so its delta-correlation is a second look at
        that same identity, not an independent fact).
    """
    _require(HELD_RUNS)
    _require(HELD_SCORES)
    con.execute(
        """
        create or replace view run_metrics as
        with fb as (select run_id, value as fb_value, n as fb_n from held_scores where metric_name = 'facet_breadth'),
             cov as (select run_id, value as coverage from held_scores where metric_name = 'evidence_coverage')
        select r.suite_id, r.arm_id, r.task_id, fb.fb_value, fb.fb_n, cov.coverage
        from fb join cov on cov.run_id = fb.run_id
        join held_runs r on r.run_id = fb.run_id
        where r.suite_id in ('musique', 'strategyqa', 'wiki2')
        """
    )
    con.execute(
        """
        create or replace view task_arm as
        select suite_id, arm_id, task_id,
               avg(fb_value) as fb_value, avg(fb_n) as fb_n, avg(coverage) as coverage
        from run_metrics group by 1, 2, 3
        """
    )
    con.execute(
        """
        create or replace view paired as
        select t.suite_id, t.task_id,
               t.fb_value as fb_trained, t.fb_n as fbn_trained, t.coverage as cov_trained,
               p.fb_value as fb_prompted, p.coverage as cov_prompted
        from (select * from task_arm where arm_id = 'inquirer_trained') t
        join (select * from task_arm where arm_id = 'inquirer_prompted') p
        on t.suite_id = p.suite_id and t.task_id = p.task_id
        """
    )
    out: dict = {}
    for suite in ("musique", "strategyqa", "wiki2"):
        n_pairs = con.execute(f"select count(*) from paired where suite_id = '{suite}'").fetchone()[
            0
        ]
        n_restricted = con.execute(
            f"select count(*) from paired where suite_id = '{suite}' and fbn_trained > 0"
        ).fetchone()[0]
        level_trained_all = con.execute(
            f"select corr(fb_trained, cov_trained) from paired where suite_id = '{suite}'"
        ).fetchone()[0]
        level_prompted_all = con.execute(
            f"select corr(fb_prompted, cov_prompted) from paired where suite_id = '{suite}'"
        ).fetchone()[0]
        level_trained_restricted = con.execute(
            f"select corr(fb_trained, cov_trained) from paired "
            f"where suite_id = '{suite}' and fbn_trained > 0"
        ).fetchone()[0]
        level_prompted_restricted = con.execute(
            f"select corr(fb_prompted, cov_prompted) from paired "
            f"where suite_id = '{suite}' and fbn_trained > 0"
        ).fetchone()[0]
        delta_pearson_all = con.execute(
            f"select corr(fb_trained - fb_prompted, cov_trained - cov_prompted) from paired "
            f"where suite_id = '{suite}'"
        ).fetchone()[0]
        delta_pearson_restricted = con.execute(
            f"select corr(fb_trained - fb_prompted, cov_trained - cov_prompted) from paired "
            f"where suite_id = '{suite}' and fbn_trained > 0"
        ).fetchone()[0]
        spearman_all = con.execute(
            f"""
            with d as (
                select fb_trained - fb_prompted as d_fb, cov_trained - cov_prompted as d_cov
                from paired where suite_id = '{suite}'
            ), r as (
                select rank() over (order by d_fb) as rk_fb, rank() over (order by d_cov) as rk_cov
                from d
            )
            select corr(rk_fb, rk_cov) from r
            """
        ).fetchone()[0]
        spearman_restricted = con.execute(
            f"""
            with d as (
                select fb_trained - fb_prompted as d_fb, cov_trained - cov_prompted as d_cov
                from paired where suite_id = '{suite}' and fbn_trained > 0
            ), r as (
                select rank() over (order by d_fb) as rk_fb, rank() over (order by d_cov) as rk_cov
                from d
            )
            select corr(rk_fb, rk_cov) from r
            """
        ).fetchone()[0]
        out[suite] = {
            "n_pairs": n_pairs,
            "n_pairs_facet_total_gt_0": n_restricted,
            "level_pearson_trained_all": level_trained_all,
            "level_pearson_prompted_all": level_prompted_all,
            "level_pearson_trained_restricted": level_trained_restricted,
            "level_pearson_prompted_restricted": level_prompted_restricted,
            "delta_pearson_all": delta_pearson_all,
            "delta_pearson_restricted": delta_pearson_restricted,
            "delta_spearman_all": spearman_all,
            "delta_spearman_restricted": spearman_restricted,
        }
    return out


def depth_breadth_same_basis(con: duckdb.DuckDBPyConnection) -> dict:
    """Whether facet_breadth and cad_ge2 trade off, on the one basis both are actually computed
    on. cad_ge2's own verdict record says it CANNOT be read at matched cost (the gold
    node->depth map contract forbids it, see the criterion's own "reason" text) and falls back
    to the fixed-budget (cap8) delta while still being stored in the "value" field of a record
    whose coverage_rule says matched_cost -- this is the ambiguity paper/results.tex lines
    ~204-212 names directly. facet_breadth's own "value" field in the same records is, by
    inspection, also the cap8/unmatched number (it reproduces
    artifacts/testsplit_plan_metrics_20260918/contrasts.json's "unmatched" row exactly, not the
    "matched" row), because facet_rule is report_only in these verdicts too. So the two fields
    that are actually comparable on a shared basis are both cap8. This function reads that
    basis directly out of the verdict JSON rather than asserting it, and reports sign agreement.
    """
    verdicts = {
        "musique": "artifacts/testsplit_qa/verdicts/stacked-notdone.musique.json",
        "strategyqa": "artifacts/testsplit_qa/verdicts/stacked-notdone.strategyqa.json",
    }
    out: dict = {}
    for suite, path in verdicts.items():
        _require(path)
        d = json.loads(Path(path).read_text())
        cad = d["criteria"].get("cad_ge2", {})
        fb = d["criteria"].get("facet_breadth", {})
        out[suite] = {
            "coverage_rule": d.get("coverage_rule"),
            "cad_rule": d.get("cad_rule"),
            "facet_rule": d.get("facet_rule"),
            "cad_ge2_value": cad.get("value"),
            "cad_ge2_comparator": cad.get("comparator"),
            "cad_ge2_n": cad.get("n"),
            "facet_breadth_value": fb.get("value"),
            "facet_breadth_n": fb.get("n"),
            "same_sign": (
                (cad.get("value") > 0) == (fb.get("value") > 0)
                if cad.get("value") is not None and fb.get("value") is not None
                else None
            ),
        }
    return out


def horizontal_control_census() -> dict:
    """Is there an arm that isolates the horizontal claim the way inquirer_depth1 isolates the
    vertical one. Reads the note/kill-switch text directly out of the two source files that
    carry it rather than asserting a conclusion, so a change to either file changes this
    function's output rather than silently going stale.
    """
    import re

    arms_src = Path("src/pinq_expt/arms.py").read_text()
    prereg_src = Path("src/pi_eval/prereg.py").read_text()
    m = re.search(r'"inquirer_depth1":\s*Arm\((.*?)\n    \),', arms_src, re.S)
    depth1_note = None
    if m:
        note_m = re.search(r'note="([^"]*(?:"\s*\n\s*"[^"]*)*)"', m.group(1))
        if note_m:
            depth1_note = re.sub(r'"\s*\n\s*"', "", note_m.group(1))
    m2 = re.search(r'"inquirer_depth1":\s*\(\s*(.*?)\s*\),', prereg_src, re.S)
    depth1_switch = None
    if m2:
        depth1_switch = re.sub(r'"\s*\n\s*"', "", m2.group(1)).strip().strip('"')
    candidates = re.findall(
        r"single_facet|one_line|sequential_only|single_component|breadth_kill|"
        r"kill.*breadth|BreadthKill|DepthOnly|depth_only",
        arms_src + prereg_src,
    )
    return {
        "inquirer_depth1_arm_note": depth1_note,
        "inquirer_depth1_kill_switch_text": depth1_switch,
        "candidate_breadth_only_kill_switch_names_found": sorted(set(candidates)),
    }


def cross_checkpoint_correlation() -> dict:
    import json as _json
    import statistics

    files = sorted(
        glob.glob("artifacts/gate/8b1-rescored-t20/*.json")
        + glob.glob("artifacts/gate/8b2-t20/*.json")
        + glob.glob("artifacts/gate/17b-t20/*.json")
        + glob.glob("artifacts/gate/4b-t20/*.json")
        + glob.glob("artifacts/gate/4b-rescored-t20/*.json")
        + glob.glob("artifacts/gate/n1/*.json")
    )
    files = [f for f in files if "t19c" not in f]
    out: dict = {"n_verdict_files": len(files)}
    for suite in ("musique", "strategyqa"):
        facet_deltas, cov_deltas = [], []
        for f in files:
            d = _json.load(open(f))
            crit = d.get("criteria", {})
            mc = d.get("matched_cost", {}).get("by_suite", {}).get(suite)
            fb = crit.get("facet_breadth")
            if mc is None or fb is None:
                continue
            fbv, cov = fb.get("value"), mc.get("delta")
            if fbv is None or cov is None:
                continue
            facet_deltas.append(fbv)
            cov_deltas.append(cov)
        n = len(facet_deltas)
        out[suite] = {
            "n_checkpoints": n,
            "r_facet_delta_vs_coverage_delta": statistics.correlation(facet_deltas, cov_deltas)
            if n > 2
            else None,
        }
    return out


def reporting_framework_check() -> dict:
    import re

    def _strip_comments(block: str) -> str:
        return "\n".join(line for line in block.splitlines() if not line.strip().startswith("#"))

    report_src = Path("src/pi_eval/report.py").read_text()
    gate_src = Path("src/pinq_train/gate.py").read_text()
    m = re.search(r"EXPLORATORY_METRICS.*?=\s*\((.*?)\n\)", report_src, re.S)
    exploratory = (
        re.findall(r'^\s*"([a-z0-9_]+)",?\s*$', _strip_comments(m.group(1)), re.M) if m else []
    )
    m2 = re.search(r"PAIRED_METRICS.*?=\s*\((.*?)\n\)", gate_src, re.S)
    paired = re.findall(r'\("([a-z0-9_]+)"', _strip_comments(m2.group(1))) if m2 else []
    return {
        "facet_breadth_in_EXPLORATORY_METRICS": "facet_breadth" in exploratory,
        "facet_breadth_in_PAIRED_METRICS": "facet_breadth" in paired,
        "breadth_components_in_EXPLORATORY_METRICS": "breadth_components" in exploratory,
        "breadth_recall_in_EXPLORATORY_METRICS": "breadth_recall" in exploratory,
        "breadth_components_in_PAIRED_METRICS": "breadth_components" in paired,
        "EXPLORATORY_METRICS": exploratory,
        "PAIRED_METRICS": paired,
    }


def main() -> None:
    con = duckdb.connect(":memory:")
    result = {
        "provenance": {
            "headline_scorer_hash": HEADLINE_SCORER_HASH,
            "testsplit_scorer_hash": TESTSPLIT_SCORER_HASH,
            "graph_version": "v1",
        },
        "structural_census_main_store": structural_census(con),
        "gated_population_census": gated_population_census(con),
        "identity_check_held_out": identity_check(con),
        "separability_held_out": separability(con),
        "depth_breadth_same_basis": depth_breadth_same_basis(con),
        "horizontal_control_census": horizontal_control_census(),
        "cross_checkpoint_correlation_dev_gate": cross_checkpoint_correlation(),
        "reporting_framework_membership": reporting_framework_check(),
    }
    out_path = (
        Path(__file__).resolve().parent.parent.parent
        / "artifacts"
        / "horizontal_axis_separability"
        / "separability.json"
    )
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
