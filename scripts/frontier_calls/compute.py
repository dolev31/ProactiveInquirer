#!/usr/bin/env python3
"""Lane L1.4: the budget frontiers on the CALLS axis, not the cap axis, for musique,
strategyqa and frames -- plus the calls-axis dominance test of `trained` against `base8b` and
`teacher`.

WHY THIS EXISTS. A cap is an allowance; `Inquirer.act(s: State)` takes no budget, so a policy
that stops itself realizes fewer calls than its cap. Comparing two arms at the same CAP
compares unequal SPENDS whenever their ceiling-hit rates differ -- and on all three suites the
trained arm's ceiling-hit rate is in the single digits or low single percent while the prompted
arms' is 20-70%. This file re-reads the three already-verified, already-isolated frontier
snapshots (never `scores/parquet`, which none of the three campaigns ever wrote to -- see each
promoted FRONTIER*.md's "Scoring isolation" section) and reports both axes side by side.

Run from the repository root that holds `artifacts/frontier/`, `artifacts/frontier_strategyqa/`
and `artifacts/frames_frontier/` -- a fresh worktree has none of the three, because they are
untracked (`git status` shows them `??`) and worktrees do not carry untracked files:

    .venv/bin/python scripts/frontier_calls/compute.py \\
        --repo-root /absolute/path/to/the/main/checkout \\
        --out artifacts/frontier_calls_20260918

Writes `<out>/data.json` (every cell and every dominance test, each carrying its own run_id
count + a cross-check against the promoted run_ids.*.txt/tsv list, `scorer_hash` and
`graph_version` -- rule 1) and `<out>/tables.md` (the same data as markdown tables, no prose).
`<out>` is interpreted relative to `--repo-root` unless it is itself absolute.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
from scripts.frontier_calls.interp import (
    MatchedDelta,
    check_near_zero_bounds,
    dominance_at_matched_calls,
    verdict_for,
)
from scripts.frontier_calls.suites import ALL_SUITES, COMPARATORS, ArmSpec, SuiteSpec

from pi_eval.report import ELIGIBLE
from pi_eval.stats.inference import cluster_bootstrap

N_BOOT = 10_000
SEED = 0


@dataclass(frozen=True, slots=True)
class CellResult:
    suite: str
    arm: str
    cap: int
    n: int
    mean_retrieval_calls: float
    mean_n_asks: float
    n_ceiling_hit: int
    share_stopped_before_cap: float
    metric_name: str
    metric_point: float
    metric_ci_lo: float
    metric_ci_hi: float
    n_clusters: int
    scorer_hash: str
    graph_version: str
    run_id_list_file: str
    run_ids_match_promoted_list: bool
    retrieval_calls_equals_n_asks: bool
    metric_verdict: str
    metric_near_zero_checked: bool
    metric_lo_stable: bool | None
    metric_hi_stable: bool | None


def _cell_where(cap: int, arm: ArmSpec) -> str:
    clauses = [ELIGIBLE, f"r.budget_cap = {cap}"]
    if arm.model_pin_hash is not None:
        clauses.append(f"r.model_pin_hash = '{arm.model_pin_hash}'")
    return " AND ".join(clauses)


def pull_cell(
    con: duckdb.DuckDBPyConnection, repo_root: Path, suite: SuiteSpec, arm_name: str, cap: int
):
    arm = suite.arms[arm_name]
    d = repo_root / arm.runs_dir
    where = _cell_where(cap, arm)
    sql = f"""
        SELECT r.run_id, r.task_id, r.seed, r.template_id, r.retrieval_calls, r.n_asks,
               r.stop_reason, s.value AS metric_value, s.scorer_hash, s.graph_version
        FROM read_parquet('{d / "runs.parquet"}') r
        JOIN read_parquet('{d / "scores.parquet"}') s
          ON s.run_id = r.run_id AND s.metric_name = '{suite.metric}'
        WHERE {where}
        ORDER BY r.task_id, r.seed
    """
    return con.execute(sql).fetchdf()


def _run_id_list_path(repo_root: Path, suite: SuiteSpec, arm_name: str, cap: int) -> Path:
    ext = "tsv" if suite.name in ("strategyqa", "frames") else "txt"
    sub = {"musique": "frontier", "strategyqa": "frontier_strategyqa", "frames": "frames_frontier"}
    return repo_root / "artifacts" / sub[suite.name] / f"run_ids.{arm_name}.cap{cap}.{ext}"


_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _promoted_run_ids(path: Path) -> set[str]:
    """The run_id column's POSITION is not the same file to file: musique's
    `run_ids.*.cap*.txt` is `run_id\\ttask_id\\tseed` (run_id first) while strategyqa/frames'
    `run_ids.*.cap*.tsv` is `task_id\\tseed\\trun_id` (run_id last) -- confirmed by reading both
    formats' header comments and a sample data line, not assumed from the file extension.
    Picking a fixed column position silently read `seed` ("0"/"1") as musique's run_id in an
    earlier version of this function, which is exactly the class of bug CLAUDE.md's rule 3
    exists to catch: it produced a plausible-looking 2-element set instead of an error. Every
    run_id in this codebase is a 32-character lowercase-hex digest (see `pinq.ids.h`), so the
    field is identified by matching that shape rather than by column position.
    """
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("\t")]
        hits = [f for f in fields if _RUN_ID_RE.match(f)]
        if len(hits) != 1:
            raise RuntimeError(
                f"{path}: line {line!r} has {len(hits)} run_id-shaped fields, want 1"
            )
        ids.add(hits[0])
    return ids


def cell_stats(
    repo_root: Path, suite: SuiteSpec, arm_name: str, cap: int, df
) -> tuple[CellResult, dict]:
    n = len(df)
    if n == 0:
        raise RuntimeError(
            f"{suite.name}/{arm_name}/cap{cap}: 0 eligible rows -- refusing to report a mean of nothing"
        )

    scorer_hashes = df["scorer_hash"].unique().tolist()
    graph_versions = df["graph_version"].unique().tolist()
    if len(scorer_hashes) != 1:
        raise RuntimeError(
            f"{suite.name}/{arm_name}/cap{cap}: scorer_hash not single-valued: {scorer_hashes}"
        )
    if len(graph_versions) != 1:
        raise RuntimeError(
            f"{suite.name}/{arm_name}/cap{cap}: graph_version not single-valued: {graph_versions}"
        )

    calls_ok = bool((df["retrieval_calls"] == df["n_asks"].astype(float)).all())

    ceiling = df["stop_reason"].isin(["budget", "max_turns"])
    n_ceiling = int(ceiling.sum())

    cluster_of = df["template_id"].where(df["template_id"].astype(bool), df["task_id"])
    units = df.groupby(cluster_of)["metric_value"].apply(list).tolist()
    point, lo, hi = cluster_bootstrap(units, n_boot=N_BOOT, seed=SEED)
    robust = check_near_zero_bounds(units, lo, hi)
    verdict = verdict_for(lo, hi, robust)

    keys = (df["task_id"].astype(str) + "|" + df["seed"].astype(str)).tolist()
    values_by_key: dict[str, float] = dict(zip(keys, df["metric_value"].tolist()))
    clusters_by_key: dict[str, str] = dict(zip(keys, cluster_of.tolist()))

    list_path = _run_id_list_path(repo_root, suite, arm_name, cap)
    promoted = _promoted_run_ids(list_path) if list_path.exists() else None
    mine = set(df["run_id"].tolist())
    match = (promoted == mine) if promoted is not None else False

    result = CellResult(
        suite=suite.name,
        arm=arm_name,
        cap=cap,
        n=n,
        mean_retrieval_calls=float(df["retrieval_calls"].mean()),
        mean_n_asks=float(df["n_asks"].mean()),
        n_ceiling_hit=n_ceiling,
        share_stopped_before_cap=1.0 - n_ceiling / n,
        metric_name=suite.metric,
        metric_point=point,
        metric_ci_lo=lo,
        metric_ci_hi=hi,
        n_clusters=len(units),
        scorer_hash=scorer_hashes[0],
        graph_version=graph_versions[0],
        run_id_list_file=str(list_path.relative_to(repo_root)) if list_path.exists() else "MISSING",
        run_ids_match_promoted_list=match,
        retrieval_calls_equals_n_asks=calls_ok,
        metric_verdict=verdict,
        metric_near_zero_checked=robust.checked,
        metric_lo_stable=robust.lo_stable,
        metric_hi_stable=robust.hi_stable,
    )
    extra = {"values_by_key": values_by_key, "clusters_by_key": clusters_by_key}
    return result, extra


def run_dominance_tests(suite: SuiteSpec, cells: dict, extras: dict) -> list[dict]:
    out = []
    for comp_name in COMPARATORS:
        comp_curve = [
            (
                cap,
                cells[(comp_name, cap)].mean_retrieval_calls,
                extras[(comp_name, cap)]["values_by_key"],
            )
            for cap in suite.caps
        ]
        for cap in suite.caps:
            trained_cell = cells[("trained", cap)]
            trained_extra = extras[("trained", cap)]
            delta: MatchedDelta = dominance_at_matched_calls(
                trained_cell.mean_retrieval_calls,
                trained_extra["values_by_key"],
                comp_curve,
                clusters=trained_extra["clusters_by_key"],
                n_boot=N_BOOT,
                n_perm=N_BOOT,
                seed=SEED,
            )
            comp_calls_used = [
                cells[(comp_name, c)].mean_retrieval_calls for c in delta.comp_caps_used
            ]
            out.append(
                {
                    "suite": suite.name,
                    "trained_cap": cap,
                    "trained_mean_calls": trained_cell.mean_retrieval_calls,
                    "comparator": comp_name,
                    "mode": delta.mode,
                    "comp_caps_used": list(delta.comp_caps_used),
                    "comp_calls_used": comp_calls_used,
                    "weight_hi": delta.weight_hi,
                    "delta_point": delta.point,
                    "delta_ci_lo": delta.ci_lo,
                    "delta_ci_hi": delta.ci_hi,
                    "p_value": delta.p_value,
                    "n": delta.n,
                    "n_clusters": delta.n_clusters,
                    "verdict": delta.verdict,
                    "near_zero_checked": delta.robust.checked,
                    "lo_stable": delta.robust.lo_stable,
                    "hi_stable": delta.robust.hi_stable,
                    "check_seeds": list(delta.robust.check_seeds),
                    "check_n_boot": delta.robust.check_n_boot,
                    "check_los": list(delta.robust.check_los),
                    "check_his": list(delta.robust.check_his),
                }
            )
    return out


def to_markdown_cells(all_cells: list[CellResult]) -> str:
    lines = [
        "| suite | arm | cap | n | mean n_asks | mean retrieval_calls | share stopped "
        "before cap | metric | value [BCa 95%] | ceiling-hit | scorer_hash (12) | run_ids "
        "match promoted list | near-zero re-check (50k x3 seeds) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in all_cells:
        recheck = (
            "n/a (not near 0)"
            if not c.metric_near_zero_checked
            else (f"lo_stable={c.metric_lo_stable} hi_stable={c.metric_hi_stable}")
        )
        lines.append(
            f"| {c.suite} | {c.arm} | {c.cap} | {c.n} | {c.mean_n_asks:.4f} | "
            f"{c.mean_retrieval_calls:.4f} | {c.share_stopped_before_cap:.4f} | {c.metric_name} | "
            f"{c.metric_point:.4f} [{c.metric_ci_lo:.4f}, {c.metric_ci_hi:.4f}] | "
            f"{c.n_ceiling_hit}/{c.n} ({100 * c.n_ceiling_hit / c.n:.2f}%) | "
            f"{c.scorer_hash[:12]} | {c.run_ids_match_promoted_list} | {recheck} |"
        )
    return "\n".join(lines)


def to_markdown_dominance(rows: list[dict]) -> str:
    lines = [
        "| suite | comparator | trained cap | trained mean calls | mode | comp cap(s) used | "
        "comp calls used | delta | BCa 95% (10k) | p | n | clusters | verdict | "
        "near-zero re-check (50k x3 seeds) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        comp_calls = ", ".join(f"{v:.4f}" for v in r["comp_calls_used"])
        comp_caps = ",".join(str(c) for c in r["comp_caps_used"])
        if not r["near_zero_checked"]:
            recheck = "n/a (no bound within 0.01 of 0)"
        else:
            los = ", ".join(f"{v:+.4f}" for v in r["check_los"])
            his = ", ".join(f"{v:+.4f}" for v in r["check_his"])
            recheck = (
                f"lo_stable={r['lo_stable']} (seeds -> [{los}]); "
                f"hi_stable={r['hi_stable']} (seeds -> [{his}])"
            )
        lines.append(
            f"| {r['suite']} | {r['comparator']} | {r['trained_cap']} | "
            f"{r['trained_mean_calls']:.4f} | {r['mode']} | {comp_caps} | {comp_calls} | "
            f"{r['delta_point']:+.4f} | [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | "
            f"{r['p_value']:.4f} | {r['n']} | {r['n_clusters']} | {r['verdict']} | {recheck} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo-root", required=True, help="the main checkout holding artifacts/frontier*"
    )
    ap.add_argument("--out", default="artifacts/frontier_calls_20260918")
    a = ap.parse_args(argv)

    repo_root = Path(a.repo_root).resolve()
    out = Path(a.out)
    if not out.is_absolute():
        out = repo_root / out
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    all_cells: list[CellResult] = []
    cells_by_suite: dict[str, dict] = {}
    dominance_rows: list[dict] = []

    for suite in ALL_SUITES:
        cells: dict[tuple[str, int], CellResult] = {}
        extras: dict[tuple[str, int], dict] = {}
        for arm_name in suite.arms:
            for cap in suite.caps:
                df = pull_cell(con, repo_root, suite, arm_name, cap)
                if len(df) != suite.n_per_cell:
                    raise RuntimeError(
                        f"{suite.name}/{arm_name}/cap{cap}: pulled {len(df)} rows, "
                        f"expected {suite.n_per_cell}"
                    )
                result, extra = cell_stats(repo_root, suite, arm_name, cap, df)
                cells[(arm_name, cap)] = result
                extras[(arm_name, cap)] = extra
                all_cells.append(result)
        cells_by_suite[suite.name] = cells
        dominance_rows += run_dominance_tests(suite, cells, extras)

    (out / "data.json").write_text(
        json.dumps(
            {
                "cells": [asdict(c) for c in all_cells],
                "dominance": dominance_rows,
                "n_boot": N_BOOT,
                "seed": SEED,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / "tables.md").write_text(
        "# Lane L1.4 -- computed tables (auto-generated by scripts/frontier_calls/compute.py)\n\n"
        "## Per-cell calls-axis table\n\n"
        + to_markdown_cells(all_cells)
        + "\n\n## Calls-axis dominance test (trained vs. nearest/interpolated comparator)\n\n"
        + to_markdown_dominance(dominance_rows)
        + "\n"
    )

    print(to_markdown_cells(all_cells))
    print()
    print(to_markdown_dominance(dominance_rows))
    print(f"\nwrote {out / 'data.json'}\nwrote {out / 'tables.md'}", file=sys.stderr)

    try:
        from scripts.frontier_calls.plot import plot_all

        plot_all(cells_by_suite, out)
        print(f"wrote figures under {out}", file=sys.stderr)
    except Exception as exc:  # plotting is best-effort; the tables are the measurement
        print(f"PLOTTING FAILED (tables above are unaffected): {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
