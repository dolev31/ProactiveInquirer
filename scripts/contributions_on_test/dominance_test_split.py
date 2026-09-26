#!/usr/bin/env python3
"""Lane L1.2, contribution B on test: trained-vs-teacher dominance at every shared cap.

`artifacts/frontier/FRONTIER.md` (musique) and `artifacts/frontier_strategyqa/FRONTIER_STRATEGYQA.md`
(strategyqa) already hold promoted, canonical, `split='test'` numbers for this contrast --
git-blaming `paper/sections/mechanism.tex` and `paper/sections/introduction.tex` shows the
current `+0.0572` / `323 of 400` prose was written FROM `artifacts/frontier/{FRONTIER,RESULT}.md`
on 2026-09-17/18, i.e. contribution B's headline numbers are already test-split, not dev, in the
CURRENT paper text (see this lane's RESULT.md for the git-blame evidence). This script does not
change that; it independently RECOMPUTES the requested cap-4/8/12/16 slice directly from the two
isolated snapshots (never `scores/parquet/`), as its own measurement rather than a copy of the
promoted tables, and cross-checks against them.

    PYTHONPATH=<worktree>/src <main checkout>/.venv/bin/python \\
        scripts/contributions_on_test/dominance_test_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from contributions_on_test.lib import (  # noqa: E402
    PopulationIncomplete,
    TurnsDropped,
    assert_population_scored,
    assert_turns_not_dropped,
    elects_to_stop,
    main_checkout,
    near_zero_bounds,
    read_run_ids,
    stability_verdict,
)

from pi_eval.stats.inference import paired_difference  # noqa: E402

MAIN_CHECKOUT = main_checkout()
RUNS_ROOT = MAIN_CHECKOUT / "runs"
CAPS = (4, 8, 12, 16)

# PRIMARY read: every interval this script prints is read at 10k resamples (coordinator rule,
# added mid-task). N_BOOT_XCHECK/SEED_XCHECK additionally reproduce FRONTIER.md's/
# FRONTIER_STRATEGYQA.md's own canonical 1k-resample numbers as a pipeline cross-check --
# that reproduction is NOT the reported interval.
N_BOOT_PRIMARY = 10_000
N_PERM_PRIMARY = 10_000
SEED_PRIMARY = 0

N_BOOT_XCHECK = 1000
N_PERM_XCHECK = 10_000
SEED_XCHECK = 0

# A bound within this of zero is re-read at N_BOOT_STABILITY resamples under each of
# SEEDS_STABILITY, and the sign must agree across all of them or the cell is UNDECIDED, not a
# pass or a fail (measured today by a peer: 3 of 7 such cells flip at 1k, one pass dies at 10k).
NEAR_ZERO_THRESHOLD = 0.01
N_BOOT_STABILITY = 50_000
SEEDS_STABILITY = (0, 1, 2)

MUSIQUE = {
    "suite": "musique",
    "base": MAIN_CHECKOUT / "artifacts" / "frontier",
    "arm_dir": lambda arm: MAIN_CHECKOUT / "artifacts" / "frontier" / f"scores_parquet.{arm}",
    "run_ids": lambda arm, cap: (
        MAIN_CHECKOUT / "artifacts" / "frontier" / f"run_ids.{arm}.cap{cap}.txt"
    ),
    "expected_scorer_hash": "4e70a6b3c28647f629faca848b469d08d786e63872b3abc58c1bcc2b81aa6b80",
    "cluster_col": "template_id",
}
STRATEGYQA = {
    "suite": "strategyqa",
    "base": MAIN_CHECKOUT / "artifacts" / "frontier_strategyqa",
    "arm_dir": lambda arm: (
        MAIN_CHECKOUT / "artifacts" / "frontier_strategyqa" / "scores_parquet.strategyqa"
    ),
    "run_ids": lambda arm, cap: (
        MAIN_CHECKOUT / "artifacts" / "frontier_strategyqa" / f"run_ids.{arm}.cap{cap}.tsv"
    ),
    "expected_scorer_hash": "82f2717596478621172d25237786008e5638d618c61d9afb779d21128dd3988c",
    "cluster_col": None,  # 200 singleton clusters; cluster key = task_id itself
}

# Cross-checks: FRONTIER.md's canonical post-fix "Trained vs the prompted 120B teacher" table,
# and FRONTIER_STRATEGYQA.md's "trained minus teacher" table (both 2026-09-1{7,8}).
PUBLISHED_TRAINED_VS_TEACHER = {
    "musique": {
        4: (+0.0572, +0.0210, +0.0912),
        8: (+0.0245, -0.0146, +0.0596),
        12: (+0.0087, -0.0284, +0.0418),
        16: (+0.0031, -0.0325, +0.0370),
    },
    "strategyqa": {
        4: (-0.0737, -0.1021, -0.0441),
        8: (-0.1143, -0.1440, -0.0854),
        12: (-0.1277, -0.1579, -0.0980),
        16: (-0.1373, -0.1689, -0.1066),
    },
}
PUBLISHED_CEILING_HIT_BASE8B = {
    "musique": {4: 264, 8: 201, 12: 162, 16: 148},
    "strategyqa": {4: 277, 8: 169, 12: 111, 16: 86},
}


def load_cell(con, parquet_dir: Path, run_ids: list[str]) -> list[dict[str, Any]]:
    lst = ",".join(f"'{i}'" for i in run_ids)
    cur = con.execute(
        f"""
        SELECT r.run_id, r.task_id, r.seed, r.template_id, r.stop_reason, r.retrieval_calls,
               s.value AS evidence_coverage
        FROM read_parquet('{parquet_dir}/runs.parquet') r
        JOIN read_parquet('{parquet_dir}/scores.parquet') s
          ON s.run_id = r.run_id AND s.metric_name = 'evidence_coverage'
        WHERE r.run_id IN ({lst})
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_suite(cfg: dict[str, Any]) -> dict[str, Any]:
    con = duckdb.connect()
    suite = cfg["suite"]
    print(f"\n######## {suite} ({cfg['base']}) ########")
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for arm in ("teacher", "base8b", "trained"):
        for cap in CAPS:
            f = cfg["run_ids"](arm, cap)
            ids = read_run_ids(f)
            report = assert_population_scored(
                cfg["arm_dir"](arm),
                ids,
                expected_scorer_hash=cfg["expected_scorer_hash"],
                expected_graph_version="v1",
            )
            turns_report = assert_turns_not_dropped(cfg["arm_dir"](arm), ids, runs_root=RUNS_ROOT)
            rows = load_cell(con, cfg["arm_dir"](arm), ids)
            assert len(rows) == len(ids), (
                f"{arm} cap{cap}: {len(rows)} scored rows for {len(ids)} run_ids"
            )
            cells[(arm, cap)] = rows
            print(
                f"  population OK: {f.name} n={report.n_checked} scorer_hash={report.scorer_hash[:8]}"
                f" turns_checked={turns_report.n_nonempty_on_disk} zeroed=0 missing_calls=0"
            )

    print(
        f"\n{'arm':<8} {'cap':>4} {'n':>4} {'mean_coverage':>14} {'mean_calls':>11} {'ceiling_hit':>12} {'elects_to_stop':>15}"
    )
    levels: dict[tuple[str, int], dict[str, float]] = {}
    for arm in ("teacher", "base8b", "trained"):
        for cap in CAPS:
            rows = cells[(arm, cap)]
            n = len(rows)
            mean_cov = sum(r["evidence_coverage"] for r in rows) / n
            mean_calls = sum(r["retrieval_calls"] for r in rows) / n
            ceiling_hit = sum(1 for r in rows if r["stop_reason"] in ("budget", "max_turns"))
            stop_early = elects_to_stop(n, ceiling_hit)
            levels[(arm, cap)] = {
                "n": n,
                "mean_coverage": mean_cov,
                "mean_calls": mean_calls,
                "ceiling_hit": ceiling_hit,
                "elects_to_stop": stop_early,
            }
            print(
                f"{arm:<8} {cap:>4} {n:>4} {mean_cov:>14.5f} {mean_calls:>11.4f} {ceiling_hit:>12} {stop_early:>15}"
            )

    def pair_and_clusters(cap: int) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        trained_rows, teacher_rows = cells[("trained", cap)], cells[("teacher", cap)]
        a = {f"{r['task_id']}|{r['seed']}": r["evidence_coverage"] for r in trained_rows}
        b = {f"{r['task_id']}|{r['seed']}": r["evidence_coverage"] for r in teacher_rows}
        if cfg["cluster_col"] == "template_id":
            clusters = {
                f"{r['task_id']}|{r['seed']}": (r["template_id"] or r["task_id"])
                for r in trained_rows
            }
        else:
            clusters = {f"{r['task_id']}|{r['seed']}": r["task_id"] for r in trained_rows}
        return a, b, clusters

    print(
        f"\n=== PRIMARY (N_BOOT={N_BOOT_PRIMARY}, N_PERM={N_PERM_PRIMARY}, seed={SEED_PRIMARY}):"
        " trained - teacher, evidence_coverage ==="
    )
    print(
        f"{'cap':>4} {'delta':>10} {'95% CI':>24} {'p':>8} {'n_clusters':>10} {'near_zero':>10} {'stability':>10}"
    )
    deltas: dict[int, Any] = {}
    stability: dict[int, str] = {}
    for cap in CAPS:
        a, b, clusters = pair_and_clusters(cap)
        est = paired_difference(
            a, b, clusters=clusters, n_boot=N_BOOT_PRIMARY, n_perm=N_PERM_PRIMARY, seed=SEED_PRIMARY
        )
        deltas[cap] = est
        n_clusters = len(set(clusters.values()))
        flagged = near_zero_bounds(est.ci_lo, est.ci_hi, NEAR_ZERO_THRESHOLD)
        if flagged:
            bounds_by_seed = {}
            for sd in SEEDS_STABILITY:
                est50 = paired_difference(
                    a, b, clusters=clusters, n_boot=N_BOOT_STABILITY, n_perm=N_PERM_PRIMARY, seed=sd
                )
                bounds_by_seed[sd] = (est50.ci_lo, est50.ci_hi)
            verdict = stability_verdict(bounds_by_seed, flagged)
            stability[cap] = verdict
            print(
                f"  [stability check] cap{cap} flagged={flagged} 50k/seed readings: {bounds_by_seed} -> {verdict}"
            )
        else:
            stability[cap] = "n/a (no bound near zero)"
        print(
            f"{cap:>4} {est.point:>+10.4f} [{est.ci_lo:>+9.4f}, {est.ci_hi:>+9.4f}] {est.p_value:>8.4f}"
            f" {n_clusters:>10} {str(flagged):>10} {stability[cap]:>10}"
        )

    print(
        f"\n--- cross-check against the promoted artifact (N_BOOT={N_BOOT_XCHECK}, its own canonical resample count) ---"
    )
    all_ok = True
    for cap in CAPS:
        a, b, clusters = pair_and_clusters(cap)
        xcheck = paired_difference(
            a, b, clusters=clusters, n_boot=N_BOOT_XCHECK, n_perm=N_PERM_XCHECK, seed=SEED_XCHECK
        )
        pub_pt, pub_lo, pub_hi = PUBLISHED_TRAINED_VS_TEACHER[suite][cap]
        ok_pt = abs(xcheck.point - pub_pt) < 5e-4
        ok_lo = abs(xcheck.ci_lo - pub_lo) < 5e-3
        ok_hi = abs(xcheck.ci_hi - pub_hi) < 5e-3
        ok = ok_pt and ok_lo and ok_hi
        all_ok &= ok
        print(
            f"  cap{cap}: recomputed@1k {xcheck.point:+.4f} [{xcheck.ci_lo:+.4f},{xcheck.ci_hi:+.4f}]"
            f" published {pub_pt:+.4f} [{pub_lo:+.4f},{pub_hi:+.4f}] {'OK' if ok else 'MISMATCH'}"
        )
        pub_ch = PUBLISHED_CEILING_HIT_BASE8B[suite][cap]
        got_ch = levels[("base8b", cap)]["ceiling_hit"]
        ok_ch = pub_ch == got_ch
        all_ok &= ok_ch
        print(
            f"    base8b ceiling_hit cap{cap}: recomputed {got_ch} published {pub_ch} {'OK' if ok_ch else 'MISMATCH'}"
        )
    print(f"  ALL CROSS-CHECKS ({suite}): {'PASS' if all_ok else 'FAIL'}")

    return {"levels": levels, "deltas": deltas, "stability": stability, "all_ok": all_ok}


def main() -> int:
    try:
        m = run_suite(MUSIQUE)
        s = run_suite(STRATEGYQA)
    except (PopulationIncomplete, TurnsDropped) as e:
        print(f"REFUSED: {e}")
        return 1
    ok = m["all_ok"] and s["all_ok"]
    print(f"\n=== overall cross-check: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
