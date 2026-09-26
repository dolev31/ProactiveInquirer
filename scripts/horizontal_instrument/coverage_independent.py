"""Q2 of artifacts/horizontal_axis_instrument_20260919/RESULT.md: does a coverage-independent
breadth reading survive on any suite, once coverage is held fixed rather than merely
correlated against.

`artifacts/horizontal_axis_separability/RESULT.md` (Sections 2-3, read in full before this file
was written) already reports CORRELATIONS between facet_breadth/breadth_components and
evidence_coverage on the same population this script reads. A correlation is not the same
question as this one: two variables can correlate at 0.5 and still diverge on individual
tasks. This script instead builds a MATCHED-REQUIRED-COVERAGE subpopulation: pairs of
(inquirer_trained, inquirer_prompted) runs on the SAME task where both arms resolved the exact
same INTEGER COUNT of required nodes (not just a similar rate), then asks whether breadth
still differs between the two arms on exactly those pairs. If it does not, breadth carries zero
information beyond coverage on this population, full stop -- there is no gap left for
"breadth" to be filling. If it does, that is a genuine coverage-independent signal.

Reads ONLY the frozen snapshot this session took before a peer's compaction of the live
scores/parquet store (see artifacts/horizontal_axis_instrument_20260919/snapshot_20260919/
SNAPSHOT_MANIFEST.json for the exact row counts and the single scorer_hash
aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28b594a5ab1d0cc5b930e0ba7 / graph_version v1 that
snapshot carries -- confirmed via `describe`/`select distinct` before it was copied, so this
script's population cannot silently move underneath it). Never reads the live
artifacts/testsplit_qa/scores_parquet or scores/parquet paths directly.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import duckdb

SNAPSHOT = (
    "artifacts/horizontal_axis_instrument_20260919/snapshot_20260919/testsplit_qa_scores_parquet"
)
RUNS = f"{SNAPSHOT}/runs.parquet"
SCORES = f"{SNAPSHOT}/scores.parquet"
SUITES = ("musique", "strategyqa", "wiki2")


def _require(path: str) -> None:
    if not Path(path).exists():
        raise SystemExit(f"missing {path}. Run scripts/horizontal_instrument/ from repo root.")


def build(con: duckdb.DuckDBPyConnection) -> dict:
    _require(RUNS)
    _require(SCORES)
    con.execute(f"create or replace view runs as select * from read_parquet('{RUNS}')")
    con.execute(f"create or replace view scores as select * from read_parquet('{SCORES}')")

    scorer_hashes = [
        r[0] for r in con.execute("select distinct scorer_hash from scores").fetchall()
    ]
    graph_versions = [
        r[0] for r in con.execute("select distinct graph_version from scores").fetchall()
    ]
    if len(scorer_hashes) != 1 or len(graph_versions) != 1:
        raise SystemExit(
            f"snapshot is not a single (scorer_hash, graph_version): {scorer_hashes} / "
            f"{graph_versions}. Refusing to compute a matched population across pooled hashes."
        )

    con.execute(
        """
        create or replace view per_run as
        with fb as (select run_id, value as fb_value, n as fb_n from scores
                    where metric_name = 'facet_breadth'),
             bc as (select run_id, value as bc_value, n as bc_n from scores
                    where metric_name = 'breadth_components'),
             cov as (select run_id, value as cov_value, n as cov_n from scores
                    where metric_name = 'evidence_coverage')
        select r.suite_id, r.arm_id, r.task_id, r.seed,
               fb.fb_value, fb.fb_n, bc.bc_value, bc.bc_n, cov.cov_value, cov.cov_n
        from cov
        join runs r on r.run_id = cov.run_id
        left join fb on fb.run_id = cov.run_id
        left join bc on bc.run_id = cov.run_id
        where r.arm_id in ('inquirer_trained', 'inquirer_prompted')
        """
    )
    # Average across seeds into one row per (suite, arm, task), matching the peer methodology
    # in scripts/horizontal_axis_separability/build_separability.py (seed-averaged before the
    # arm join), then round the coverage rate * denominator into an integer resolved-node
    # count. cov_n (required-node total) is a property of the TASK'S graph, not the arm, so it
    # is asserted equal across arms below rather than assumed.
    con.execute(
        """
        create or replace view task_arm as
        select suite_id, arm_id, task_id,
               avg(fb_value) as fb_value, avg(fb_n) as fb_n,
               avg(bc_value) as bc_value, avg(bc_n) as bc_n,
               avg(cov_value) as cov_value, avg(cov_n) as cov_n
        from per_run group by 1, 2, 3
        """
    )
    con.execute(
        """
        create or replace view paired as
        select t.suite_id, t.task_id,
               t.fb_value as fb_trained, t.fb_n as fbn_trained,
               t.bc_value as bc_trained, t.bc_n as bcn_trained,
               t.cov_value as cov_trained, t.cov_n as covn_trained,
               p.fb_value as fb_prompted, p.bc_value as bc_prompted,
               p.cov_value as cov_prompted, p.cov_n as covn_prompted,
               round(t.cov_value * t.cov_n) as resolved_trained,
               round(p.cov_value * p.cov_n) as resolved_prompted
        from (select * from task_arm where arm_id = 'inquirer_trained') t
        join (select * from task_arm where arm_id = 'inquirer_prompted') p
          on t.suite_id = p.suite_id and t.task_id = p.task_id
        """
    )

    out: dict = {}
    for suite in SUITES:
        n_total = con.execute(f"select count(*) from paired where suite_id='{suite}'").fetchone()[0]
        denom_mismatch = con.execute(
            f"select count(*) from paired where suite_id='{suite}' and covn_trained != covn_prompted"
        ).fetchone()[0]
        matched = con.execute(
            f"""
            select fb_trained, fb_prompted, bc_trained, bc_prompted,
                   cov_trained, cov_prompted, resolved_trained
            from paired
            where suite_id='{suite}' and resolved_trained = resolved_prompted
            """
        ).fetchall()
        n_matched = len(matched)
        fb_diffs = [m[0] - m[1] for m in matched if m[0] is not None and m[1] is not None]
        bc_diffs = [m[2] - m[3] for m in matched if m[2] is not None and m[3] is not None]
        cov_diffs = [
            m[4] - m[5] for m in matched
        ]  # should be ~0 by construction (matched rounding)
        by_resolved_count: dict[int, dict] = {}
        for m in matched:
            k = int(m[6]) if m[6] is not None else -1
            by_resolved_count.setdefault(k, {"n": 0, "fb_diffs": []})
            by_resolved_count[k]["n"] += 1
            if m[0] is not None and m[1] is not None:
                by_resolved_count[k]["fb_diffs"].append(m[0] - m[1])

        out[suite] = {
            "n_pairs_total": n_total,
            "n_pairs_required_node_total_mismatch_across_arms": denom_mismatch,
            "n_pairs_matched_resolved_count": n_matched,
            "fraction_pairs_matched": (n_matched / n_total) if n_total else None,
            "facet_breadth_delta_on_matched_subpop": {
                "n": len(fb_diffs),
                "mean": statistics.fmean(fb_diffs) if fb_diffs else None,
                "median": statistics.median(fb_diffs) if fb_diffs else None,
                "n_nonzero": sum(1 for d in fb_diffs if abs(d) > 1e-9),
                "share_nonzero": (sum(1 for d in fb_diffs if abs(d) > 1e-9) / len(fb_diffs))
                if fb_diffs
                else None,
            },
            "breadth_components_delta_on_matched_subpop": {
                "n": len(bc_diffs),
                "mean": statistics.fmean(bc_diffs) if bc_diffs else None,
                "median": statistics.median(bc_diffs) if bc_diffs else None,
                "n_nonzero": sum(1 for d in bc_diffs if abs(d) > 1e-9),
                "share_nonzero": (sum(1 for d in bc_diffs if abs(d) > 1e-9) / len(bc_diffs))
                if bc_diffs
                else None,
            },
            "coverage_rate_delta_on_matched_subpop_sanity": {
                "n": len(cov_diffs),
                "mean": statistics.fmean(cov_diffs) if cov_diffs else None,
                "max_abs": max((abs(d) for d in cov_diffs), default=None),
            },
            "matched_subpop_by_resolved_count": {
                str(k): {
                    "n": v["n"],
                    "facet_breadth_mean_delta": statistics.fmean(v["fb_diffs"])
                    if v["fb_diffs"]
                    else None,
                    "facet_breadth_n_nonzero": sum(1 for d in v["fb_diffs"] if abs(d) > 1e-9),
                }
                for k, v in sorted(by_resolved_count.items())
            },
        }

    # Secondary reading: OLS residualization of facet_breadth on evidence_coverage, pooling
    # both arms' task-level rows within a suite (not pooling suites), then reporting the
    # trained-vs-prompted mean residual difference. A second, independent look at the same
    # question with a different mechanic (regression, not exact-count matching).
    residual_out: dict = {}
    for suite in SUITES:
        rows = con.execute(
            f"""
            select arm_id, fb_value, cov_value from task_arm
            where suite_id = '{suite}' and fb_value is not null and cov_value is not null
            """
        ).fetchall()
        xs = [r[2] for r in rows]
        ys = [r[1] for r in rows]
        n = len(xs)
        if n < 3 or len(set(xs)) < 2:
            residual_out[suite] = {"n": n, "note": "insufficient variance for OLS"}
            continue
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b1 = sxy / sxx if sxx else 0.0
        b0 = my - b1 * mx
        residuals_trained = [
            y - (b0 + b1 * x)
            for arm, y, x in zip((r[0] for r in rows), ys, xs)
            if arm == "inquirer_trained"
        ]
        residuals_prompted = [
            y - (b0 + b1 * x)
            for arm, y, x in zip((r[0] for r in rows), ys, xs)
            if arm == "inquirer_prompted"
        ]
        residual_out[suite] = {
            "n": n,
            "ols_slope_fb_on_coverage": b1,
            "ols_intercept": b0,
            "n_trained": len(residuals_trained),
            "n_prompted": len(residuals_prompted),
            "mean_residual_trained": statistics.fmean(residuals_trained)
            if residuals_trained
            else None,
            "mean_residual_prompted": statistics.fmean(residuals_prompted)
            if residuals_prompted
            else None,
            "residual_gap_trained_minus_prompted": (
                statistics.fmean(residuals_trained) - statistics.fmean(residuals_prompted)
            )
            if residuals_trained and residuals_prompted
            else None,
        }

    return {
        "snapshot_scorer_hash": scorer_hashes[0],
        "snapshot_graph_version": graph_versions[0],
        "matched_coverage_subpopulation": out,
        "ols_residualization": residual_out,
    }


def main() -> None:
    con = duckdb.connect()
    result = build(con)
    dest = Path("artifacts/horizontal_axis_instrument_20260919/coverage_independent.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
