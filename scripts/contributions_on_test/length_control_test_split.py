#!/usr/bin/env python3
"""Lane L1.2, contribution A on test: the token-charged matched-cost length control.

`artifacts/length_tokens/RESULT.md` and `artifacts/length_control_tokens/RESULT.md` computed
the question-TOKEN-charged matched-cost contrast (`scripts/matched_cost.py`'s
`matched_question_tokens` basis) for the selected arm (`qwen3-8b-dpo-stacked-notdone-both`)
against `qwen3-8b-base`, but only on `artifacts/gate/n1` -- `split = 'dev'`, stated explicitly
in both files. No test-split token-charged number has ever been computed: until commit
`3475b943` (2026-09-18, today) `scripts/matched_cost.py`'s `contrast()` silently collapsed a
two-seed population's second seed into its first (see that commit's docstring on `contrast()`),
and `artifacts/testsplit_qa` runs both `inquirer_trained` and `inquirer_prompted` at seeds 0
AND 1 on all three suites, unlike the dev gate populations. This script is the first
token-charged computation against the held-out population, run against the now-fixed code.

Reuses `scripts/matched_cost.py` UNCHANGED: `read_tables`, `turn_shims`, `question_words`,
`question_tokens`, `active_cost_bases`, `read_inquirer_calls`, `build_ladders`,
`verify_instrument`, `contrast`. The only override is `N_BOOT`/`N_PERM`, raised from the
module's default 1000 to the 10,000 this lane's brief specifies, by reassigning the module
globals `contrast()` itself reads -- not by copying `contrast()`'s body.

    PI_GOLD_ROOT=<repo>/data/gold PYTHONPATH=<worktree>/src <main checkout>/.venv/bin/python \\
        scripts/contributions_on_test/length_control_test_split.py

Writes nothing; prints the three-column table (token-charged | calls-charged matched | cap-8)
per suite, the cross-check against the already-published calls-charged and cap-8 numbers in
`artifacts/testsplit_qa/TESTSPLIT_QA.md`, and the teacher-population check for the second half
of the brief.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../scripts/contributions_on_test/this.py
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matched_cost as mc  # noqa: E402  (scripts/matched_cost.py, path-inserted above)
from contributions_on_test.lib import (  # noqa: E402
    PopulationIncomplete,
    TurnsDropped,
    assert_population_scored,
    assert_turns_not_dropped,
    main_checkout,
    near_zero_bounds,
    read_run_ids,
    stability_verdict,
)

from pi_eval.gold import load_graphs  # noqa: E402

MAIN_CHECKOUT = main_checkout()
TESTSPLIT_QA = MAIN_CHECKOUT / "artifacts" / "testsplit_qa"
PARQUET_DIR = TESTSPLIT_QA / "scores_parquet"
RUNS_ROOT = MAIN_CHECKOUT / "runs"
SHARED_PARQUET_DIR = MAIN_CHECKOUT / "scores" / "parquet"

SUITES: tuple[str, ...] = ("musique", "strategyqa", "wiki2")
CHECKPOINT = "qwen3-8b-dpo-stacked-notdone-both"
EXPECTED_SCORER_HASH = "aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28b594a5ab1d0cc5b930e0ba7"
EXPECTED_GRAPH_VERSION = "v1"
N_BOOT = 10_000
N_PERM = 10_000

# Coordinator rule (added mid-task): a bound within this of zero is re-read at N_BOOT_STABILITY
# resamples under each of SEEDS_STABILITY, and the sign must agree across all of them or the
# cell is UNDECIDED, not a pass or a fail.
NEAR_ZERO_THRESHOLD = 0.01
N_BOOT_STABILITY = 50_000
SEEDS_STABILITY = (0, 1, 2)

# Cross-checks: already-published test-split numbers this run must reproduce (TESTSPLIT_QA.md,
# "The cells" and "Levels behind the contrast" / "cap-8 delta" column, both dated 2026-09-18).
PUBLISHED_MATCHED_K: dict[str, float] = {
    "musique": 0.12381628787878789,
    "strategyqa": 0.06280952380952382,
    "wiki2": 0.07756024096385543,
}
PUBLISHED_CAP8: dict[str, float] = {
    "musique": 0.02628,
    "strategyqa": -0.09942,
    "wiki2": -0.02560,
}

# The teacher's model_pin_hash, named in artifacts/frontier/FRONTIER.md's provenance table.
TEACHER_PIN = "5a1322e03e1b7588c214529a36ba718ca96aab243ac9c82c83944ece3c096ebf"


def load_population() -> tuple[dict[str, Any], list[str]]:
    """Population-assert both arms on all three suites, then return the filtered run rows
    `build_ladders` needs (trained + prompted only) plus the run_id union, in that order."""
    run_ids: list[str] = []
    for suite in SUITES:
        for arm in ("trained", "prompted"):
            f = TESTSPLIT_QA / f"run_ids.{arm}.{suite}.txt"
            ids = read_run_ids(f)
            report = assert_population_scored(
                PARQUET_DIR,
                ids,
                expected_scorer_hash=EXPECTED_SCORER_HASH,
                expected_graph_version=EXPECTED_GRAPH_VERSION,
            )
            print(
                f"  population OK: {f.name} n={report.n_checked} scorer_hash={report.scorer_hash[:8]}"
            )
            run_ids += ids
    tabs = mc.read_tables(PARQUET_DIR)
    keep = {"inquirer_trained", "inquirer_prompted"}
    runs = [r for r in tabs["runs"] if r["arm_id"] in keep]
    return {"runs": runs, "turns": tabs["turns"], "scores": tabs["scores"]}, sorted(set(run_ids))


def check_teacher_population() -> dict[str, Any]:
    """Item 1's second half: does a gpt-oss-120b (teacher) `inquirer_prompted` population exist
    on `split='test'` for the same grid/cap as `artifacts/testsplit_qa`, and is it USABLE (i.e.
    scored at a scorer_hash this repo currently treats as valid)? Queried live against the
    shared `scores/parquet/` by absolute path -- read-only, never written to."""
    runs_pq = MAIN_CHECKOUT / "scores" / "parquet" / "runs.parquet"
    scores_pq = MAIN_CHECKOUT / "scores" / "parquet" / "scores.parquet"
    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT suite_id, count(*) n, count(DISTINCT task_id) n_tasks, count(DISTINCT seed) n_seeds
        FROM read_parquet('{runs_pq}')
        WHERE model_pin_hash = '{TEACHER_PIN}' AND split = 'test'
          AND grid_name = 'tier1_trained_qa_teacher' AND arm_id = 'inquirer_prompted'
          AND suite_id IN {SUITES} AND status = 'ok'
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    run_ids = con.execute(
        f"""
        SELECT run_id FROM read_parquet('{runs_pq}')
        WHERE model_pin_hash = '{TEACHER_PIN}' AND split = 'test'
          AND grid_name = 'tier1_trained_qa_teacher' AND arm_id = 'inquirer_prompted'
          AND suite_id IN {SUITES} AND status = 'ok'
        """
    ).fetchall()
    ids = [r[0] for r in run_ids]
    lst = ",".join(f"'{i}'" for i in ids)
    hashes = (
        con.execute(
            f"""
        SELECT scorer_hash, count(*) FROM read_parquet('{scores_pq}')
        WHERE run_id IN ({lst}) AND metric_name = 'evidence_coverage'
        GROUP BY 1
        """
        ).fetchall()
        if ids
        else []
    )
    return {"by_suite": rows, "n_total": len(ids), "scorer_hash_counts": hashes, "run_ids": ids}


def main() -> int:
    print(
        "=== 1. population assertion (ANALYSIS RULES: every run_id in runs.parquet, scored"
        f" at scorer_hash {EXPECTED_SCORER_HASH[:8]}...) ==="
    )
    try:
        tabs, run_id_union = load_population()
    except PopulationIncomplete as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"  total distinct run_ids in population: {len(run_id_union)}")

    print()
    print(
        "=== 1b. turns-not-dropped assertion (coordinator rule: a run whose turns.jsonl is"
        " intact on disk must not have compacted to n_turns=0 / missing Inquirer calls) ==="
    )
    try:
        turns_report = assert_turns_not_dropped(PARQUET_DIR, run_id_union, runs_root=RUNS_ROOT)
    except TurnsDropped as e:
        print(f"REFUSED: {e}")
        return 1
    print(
        f"  checked {turns_report.n_checked_total} run_ids, {turns_report.n_nonempty_on_disk}"
        f" have a non-empty turns.jsonl on disk, 0 zeroed n_turns, 0 missing Inquirer calls"
    )

    print()
    print("=== 2. gold graphs, ladders, and the instrument lock ===")
    graphs = {suite: load_graphs(suite, mc.GRAPH_VERSION) for suite in SUITES}
    shims = mc.turn_shims(tabs["turns"])
    tokenizer = mc.find_question_tokenizer()
    if tokenizer is None:
        print(
            "REFUSED: no local Qwen3-8B tokenizer found; matched_question_tokens is not computable"
        )
        return 1
    print(f"  tokenizer: {tokenizer}")
    print(f"  sha256: {mc.sha256_file(tokenizer)}")
    bases = mc.active_cost_bases(tokenizer)
    print(f"  active cost bases: {bases}")

    qw = mc.question_words(shims)
    qt = mc.question_tokens(shims, tokenizer)
    gt = mc.read_inquirer_calls(PARQUET_DIR)

    built = mc.build_ladders(
        tabs["runs"], shims, graphs, inquirer_gen_tokens=gt, question_words=qw, question_tokens=qt
    )
    checked = mc.verify_instrument(built, mc.stored_scores(tabs["scores"]))
    print(f"  verify_instrument locked: {checked}  (InstrumentMismatch not raised)")

    code_versions = sorted({str(r["code_version"]) for r in tabs["runs"]})
    print(f"  code_version(s) in this population: {code_versions}")

    print()
    print(
        f"=== 3. contrasts, N_BOOT={N_BOOT} N_PERM={N_PERM} (patched from module default 1000) ==="
    )
    mc.N_BOOT = N_BOOT
    mc.N_PERM = N_PERM

    base_all = {
        (lad.suite_id, lad.task_id, lad.seed): lad
        for lad in built.values()
        if lad.arm_id == mc.BASE_ARM
    }

    results: dict[str, dict[str, mc.Contrast]] = {}
    for suite in SUITES:
        trained = sorted(
            (
                lad
                for lad in built.values()
                if lad.arm_id == mc.TRAINED_ARM and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        row: dict[str, mc.Contrast] = {}
        for comparator, offset in (
            ("matched_question_tokens", 0),
            ("matched_k", 0),
            ("cap8", 0),
        ):
            row[comparator] = mc.contrast(
                trained,
                base_all,
                suite_id=suite,
                checkpoint=CHECKPOINT,
                metric="evidence_coverage",
                comparator=comparator,
                offset=offset,
            )
        results[suite] = row

    print()
    print(
        f"=== 3b. near-zero bound stability (coordinator rule: any bound within "
        f"{NEAR_ZERO_THRESHOLD} of zero is re-read at {N_BOOT_STABILITY} resamples under seeds "
        f"{SEEDS_STABILITY}; a sign that is not stable across them is UNDECIDED) ==="
    )
    stability: dict[tuple[str, str], str] = {}
    for suite in SUITES:
        trained = sorted(
            (
                lad
                for lad in built.values()
                if lad.arm_id == mc.TRAINED_ARM and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        for comparator, offset in (
            ("matched_question_tokens", 0),
            ("matched_k", 0),
            ("cap8", 0),
        ):
            c = results[suite][comparator]
            flagged = near_zero_bounds(c.ci_lo, c.ci_hi, NEAR_ZERO_THRESHOLD)
            if not flagged:
                stability[(suite, comparator)] = "n/a (no bound near zero)"
                continue
            mc.N_BOOT = N_BOOT_STABILITY
            bounds_by_seed: dict[int, tuple[float, float]] = {}
            for sd in SEEDS_STABILITY:
                mc.SEED = sd
                c50 = mc.contrast(
                    trained,
                    base_all,
                    suite_id=suite,
                    checkpoint=CHECKPOINT,
                    metric="evidence_coverage",
                    comparator=comparator,
                    offset=offset,
                )
                bounds_by_seed[sd] = (c50.ci_lo, c50.ci_hi)
            mc.N_BOOT, mc.SEED = N_BOOT, 0
            verdict = stability_verdict(bounds_by_seed, flagged)
            stability[(suite, comparator)] = verdict
            print(
                f"  {suite:<11} {comparator:<24} flagged={flagged} "
                f"50k readings: {bounds_by_seed} -> {verdict}"
            )

    print()
    print(
        f"{'suite':<11} {'comparator':<24} {'n':>4} {'trained':>9} {'base':>9} {'delta':>9} "
        f"{'ci_lo':>9} {'ci_hi':>9} {'trained_asks':>13} {'base_asks':>10} {'base_short':>11} {'stability':>10}"
    )
    for suite in SUITES:
        for comparator in ("matched_question_tokens", "matched_k", "cap8"):
            c = results[suite][comparator]
            print(
                f"{suite:<11} {comparator:<24} {c.n_tasks:>4} {c.trained_mean:>9.5f} "
                f"{c.base_mean:>9.5f} {c.delta:>+9.5f} {c.ci_lo:>+9.5f} {c.ci_hi:>+9.5f} "
                f"{c.trained_asks:>13.4f} {c.base_asks:>10.4f} {c.n_base_short:>11}"
                f" {stability[(suite, comparator)]:>10}"
            )

    print()
    print("=== 4. cross-check against artifacts/testsplit_qa/TESTSPLIT_QA.md (2026-09-18) ===")
    all_ok = True
    for suite in SUITES:
        mk = results[suite]["matched_k"]
        pub = PUBLISHED_MATCHED_K[suite]
        diff = mk.delta - pub
        ok = abs(diff) < 1e-9
        all_ok &= ok
        print(
            f"  matched_k  {suite:<11} recomputed={mk.delta!r} published={pub!r} absdiff={diff:.2e} {'OK' if ok else 'MISMATCH'}"
        )
    for suite in SUITES:
        c8 = results[suite]["cap8"]
        pub = PUBLISHED_CAP8[suite]
        diff = c8.delta - pub
        ok = abs(diff) < 5e-5  # published to 5 dp only
        all_ok &= ok
        print(
            f"  cap8       {suite:<11} recomputed={c8.delta:.5f} published={pub!r} absdiff={diff:.2e} {'OK' if ok else 'MISMATCH'}"
        )
    print(f"  ALL CROSS-CHECKS: {'PASS' if all_ok else 'FAIL'}")

    print()
    print("=== 5. teacher test-row check (item 1, second half) ===")
    teacher = check_teacher_population()
    print(
        f"  teacher (gpt-oss-120b) inquirer_prompted, split=test, grid=tier1_trained_qa_teacher, "
        f"status=ok: n_total={teacher['n_total']}"
    )
    for row in teacher["by_suite"]:
        print(f"    {row}")
    print(
        f"  scorer_hash distribution for evidence_coverage on these run_ids: {teacher['scorer_hash_counts']}"
    )

    print()
    print(
        "=== 5b. turns-not-dropped assertion on the teacher-pin rows (flagged explicitly:"
        " these were NOT covered by the 1b check, which only covers artifacts/testsplit_qa) ==="
    )
    try:
        teacher_turns_report = assert_turns_not_dropped(
            SHARED_PARQUET_DIR, teacher["run_ids"], runs_root=RUNS_ROOT
        )
    except TurnsDropped as e:
        print(f"REFUSED: {e}")
        print(
            "  (moot for this lane's reported numbers: the teacher contrast is already not"
            " reportable, see section 5's scorer_hash finding -- but printed per the"
            " coordinator's rule regardless of the population's own usability)"
        )
    else:
        print(
            f"  checked {teacher_turns_report.n_checked_total} run_ids,"
            f" {teacher_turns_report.n_nonempty_on_disk} have a non-empty turns.jsonl on disk,"
            " 0 zeroed n_turns, 0 missing Inquirer calls"
        )

    print()
    print("=== provenance ===")
    print(f"  scorer_hash: {EXPECTED_SCORER_HASH}")
    print(f"  graph_version: {EXPECTED_GRAPH_VERSION}")
    print(f"  code_version(s): {code_versions}")
    print("  matched_cost.py git identity: see `git log -1 -- scripts/matched_cost.py`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
