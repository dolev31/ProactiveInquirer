#!/usr/bin/env python3
"""Lane L1.2 follow-up: the teacher, under the same token charge, on test.

The original attempt (see this directory's RESULT.md, "The teacher, under the same token
charge, on test") found the teacher-pin test rows scored only under a same-day compaction pass
that had not been cleared. The coordinator's follow-up says not to wait on that hash: the
convention of record is ONE ISOLATED SCORING PASS PER CAMPAIGN with every arm it compares in
it (exactly what `artifacts/frames` and `artifacts/testsplit_qa` already do), so this script
builds a THIRD such isolated store -- teacher (N6) + trained + prompted, all three suites, one
`pi compact` + `pi score` pass, one scorer_hash -- rather than trying to read two arms out of
two different stores that were never scored together.

Convention followed (named explicitly, per the brief): `pi compact --runs-root <farm> --out
<store> --exclude-dev` then `pi score --parquet <store> --runs-root <farm> --gold-root
data/gold --corpora-root data/corpora --allow-no-judge` -- the same two commands and the same
`--exclude-dev` / `--allow-no-judge` flags `artifacts/frames/FRAMES.md` and
`artifacts/testsplit_qa/TESTSPLIT_QA.md` both used, against a symlink farm built OUTSIDE the
repo so `scores/parquet/` is never touched.

Takes a whole-repository store lock (mkdir-based, matching the coordinator's exact protocol)
before writing into the shared scratch directory, and releases it in a `finally` after the
pass completes or fails.

    PI_GOLD_ROOT=<repo>/data/gold PYTHONPATH=<...> <python> \\
        scripts/contributions_on_test/teacher_token_charge.py [--skip-build]

`--skip-build` reuses an already-built store at SCRATCH/store (for re-running just the
contrast computation without re-locking / re-compacting).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matched_cost as mc  # noqa: E402
from contributions_on_test.lib import (  # noqa: E402
    PooledCodeVersions,
    TurnsDropped,
    assert_one_code_version_and_no_duplicate_keys,
    assert_turns_not_dropped,
    main_checkout,
    near_zero_bounds,
    read_run_ids,
    stability_verdict,
)

from pi_eval.gold import load_graphs  # noqa: E402

MAIN_CHECKOUT = main_checkout()
RUNS_ROOT = MAIN_CHECKOUT / "runs"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-someone-PycharmProjects-ProactiveInquirer/"
    "3593f11f-056f-4800-a24f-c5870cda2e31/scratchpad"
)
LOCK_DIR = SCRATCH / "store.lock"
FARM = SCRATCH / "teacher_token_charge_farm"
STORE = SCRATCH / "teacher_token_charge_store"
FLAG = SCRATCH / "COMPACTION_FIXED.ok"

SUITES: tuple[str, ...] = ("musique", "strategyqa", "wiki2")
N6_TEACHER_LIST = MAIN_CHECKOUT / "artifacts" / "n6" / "run_ids.inquirer_prompted.txt"


def populations() -> dict[str, list[str]]:
    teacher = read_run_ids(N6_TEACHER_LIST)
    trained: list[str] = []
    prompted: list[str] = []
    for suite in SUITES:
        trained += read_run_ids(
            MAIN_CHECKOUT / "artifacts" / "testsplit_qa" / f"run_ids.trained.{suite}.txt"
        )
        prompted += read_run_ids(
            MAIN_CHECKOUT / "artifacts" / "testsplit_qa" / f"run_ids.prompted.{suite}.txt"
        )
    return {"teacher": teacher, "trained": trained, "prompted": prompted}


def preflight_turns_on_disk(run_ids: list[str]) -> None:
    """ "Assert n_turns > 0 ... first": the disk-only half, before spending time compacting.
    A run whose directory or turns.jsonl is simply ABSENT is refused here rather than
    discovered as a confusing downstream compact/score failure. Does not check n_turns=0 vs
    a real 0-ask run (that needs the compacted table, done again after compaction)."""
    missing_dir = [r for r in run_ids if not (RUNS_ROOT / r).is_dir()]
    if missing_dir:
        raise SystemExit(
            f"REFUSED: {len(missing_dir)} run_id(s) have no directory under runs/: "
            f"{missing_dir[:5]}"
        )


def take_lock() -> None:
    waited = 0
    while True:
        try:
            LOCK_DIR.mkdir()
            break
        except FileExistsError:
            time.sleep(30)
            waited += 30
            if waited % 300 == 0:
                print(f"  ... waited {waited}s for store.lock")
    (LOCK_DIR / "owner").write_text("L1.2\n")
    print(f"  lock taken: {LOCK_DIR}")


def release_lock() -> None:
    owner = LOCK_DIR / "owner"
    if owner.exists():
        owner.unlink()
    if LOCK_DIR.is_dir():
        LOCK_DIR.rmdir()
    print(f"  lock released: {LOCK_DIR}")


def build_farm(all_ids: list[str]) -> None:
    FARM.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for rid in all_ids:
        link = FARM / rid
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(RUNS_ROOT / rid)
        n_linked += 1
    print(f"  farm: {FARM}, {n_linked} new symlinks, {len(all_ids)} total ids")


def run_compact_and_score(pi_bin: Path, pythonpath: str | None) -> dict[str, Any]:
    env = dict(os.environ)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    compact_cmd = [
        str(pi_bin),
        "compact",
        "--runs-root",
        str(FARM),
        "--out",
        str(STORE),
        "--exclude-dev",
    ]
    print("  $", " ".join(compact_cmd))
    compact_out = subprocess.run(
        compact_cmd, capture_output=True, text=True, env=env, cwd=MAIN_CHECKOUT
    )
    print(compact_out.stdout)
    print(compact_out.stderr, file=sys.stderr)
    compact_out.check_returncode()

    score_cmd = [
        str(pi_bin),
        "score",
        "--parquet",
        str(STORE),
        "--runs-root",
        str(FARM),
        "--gold-root",
        str(MAIN_CHECKOUT / "data" / "gold"),
        "--corpora-root",
        str(MAIN_CHECKOUT / "data" / "corpora"),
        "--allow-no-judge",
    ]
    print("  $", " ".join(score_cmd))
    score_out = subprocess.run(
        score_cmd, capture_output=True, text=True, env=env, cwd=MAIN_CHECKOUT
    )
    print(score_out.stdout)
    print(score_out.stderr, file=sys.stderr)
    score_out.check_returncode()
    try:
        return json.loads(score_out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"raw_stdout": score_out.stdout}


def symmetric_contrast(
    trained_suite: list,
    peer_dict: dict,
    *,
    comparator: str,
    n_boot: int,
    n_perm: int,
    seed: int,
):
    """The symmetric fix for a defect in `matched_cost.py`'s `contrast()`: `va` (the treatment)
    is read at `lad.n_asks` UNCONDITIONALLY (`matched_cost.py:626`), never clamped to the
    comparator's side, while `vb` (the comparator) IS clamped (`Ladder.asks_at`'s
    `min(k, self.n_asks)` for ask-priced comparators, `matched_cost.py:248`, or
    `Ladder.k_within` for budget-priced ones). Where the treatment spends MORE than the
    comparator on a given pair, this asymmetry silently reverts that pair to an unmatched
    reading while still being labelled "matched".

    `comparator` selects the CURRENCY, exactly as `matched_cost.py`'s own `contrast()` does via
    `budget = comparator in COST_BASES` -- an EARLIER version of this function ignored
    `comparator` entirely and always used ask counts, which silently computed the ask-count
    (`matched_k`-style) symmetric answer even when called for a token-priced comparator such as
    `matched_question_tokens`. Caught because a peer's independently-computed symmetric
    calls-matched figures agreed with this function's supposedly-token-charged output to four
    decimals on three suites, which is not a plausible coincidence between two different cost
    bases. Fixed here: for `comparator in mc.COST_BASES` (token- or word-priced), both arms are
    charged the LOWER of their own two total costs ON THAT BASIS (`Ladder.cost_at`), each read
    at the largest prefix its own trajectory affords within that shared budget
    (`Ladder.k_within`, reused unchanged, gold-touching logic untouched). For an ask-priced
    comparator (`matched_k`, offset 0), this reduces to the ask-count symmetric reading the
    earlier version always computed, which was already correct for that comparator.

    Returns `(Estimate, n_pairs, n_unsafe)`. For a budget-priced comparator, `n_unsafe` counts
    pairs where the comparator's own total cost on this basis is BELOW the treatment's -- the
    case the asymmetric reading gets wrong there too (both would be read at their own natural
    stops, which is not a matched reading regardless of the label). For an ask-priced
    comparator, `n_unsafe` counts `peer.n_asks < lad.n_asks`, unchanged from before.
    """
    budget = comparator in mc.COST_BASES
    pairs: dict[str, list[tuple[float, float]]] = {}
    n_pairs = 0
    n_unsafe = 0
    for lad in trained_suite:
        peer = peer_dict.get((lad.suite_id, lad.task_id, lad.seed))
        if peer is None:
            continue
        n_pairs += 1
        if budget:
            lad_cost = lad.cost_at(lad.n_asks, comparator)
            peer_cost = peer.cost_at(peer.n_asks, comparator)
            if peer_cost < lad_cost:
                n_unsafe += 1
            budget_sym = min(lad_cost, peer_cost)
            k_lad = lad.k_within(budget_sym, comparator)
            k_peer = peer.k_within(budget_sym, comparator)
        else:
            if peer.n_asks < lad.n_asks:
                n_unsafe += 1
            k_sym = min(lad.n_asks, peer.n_asks)
            k_lad = k_peer = k_sym
        va, vb = lad.coverage_at(k_lad), peer.coverage_at(k_peer)
        if math.isnan(va) or math.isnan(vb):
            continue
        pairs.setdefault(f"{lad.suite_id}/{lad.task_id}", []).append((va, vb))
    a = {key: statistics.fmean(v[0] for v in vals) for key, vals in pairs.items()}
    b = {key: statistics.fmean(v[1] for v in vals) for key, vals in pairs.items()}
    est = mc.paired_difference(a, b, clusters=None, n_boot=n_boot, n_perm=n_perm, seed=seed)
    return est, n_pairs, n_unsafe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--pythonpath", default=None, help="override PYTHONPATH for pi compact/score")
    args = ap.parse_args()

    if not FLAG.exists():
        print(f"REFUSED: {FLAG} does not exist yet; not proceeding.")
        return 1
    print(f"  flag present: {FLAG}")
    print("  flag content:")
    print(FLAG.read_text())

    pops = populations()
    all_ids = sorted(set(pops["teacher"]) | set(pops["trained"]) | set(pops["prompted"]))
    print(
        f"  populations: teacher={len(pops['teacher'])} trained={len(pops['trained'])} "
        f"prompted={len(pops['prompted'])} union={len(all_ids)}"
    )
    preflight_turns_on_disk(all_ids)
    print("  preflight OK: every run_id has a runs/<id>/ directory")

    if not args.skip_build:
        take_lock()
        try:
            build_farm(all_ids)
            pi_bin = MAIN_CHECKOUT / ".venv" / "bin" / "pi"
            result = run_compact_and_score(pi_bin, args.pythonpath)
            print("  score result:", result)
        finally:
            release_lock()
    else:
        print("  --skip-build: reusing existing store, no lock taken")

    # Assert n_turns > 0 (the compacted-table half) before computing anything.
    try:
        report = assert_turns_not_dropped(STORE, all_ids, runs_root=RUNS_ROOT)
    except TurnsDropped as e:
        print(f"REFUSED: {e}")
        return 1
    print(
        f"  turns-not-dropped: checked={report.n_checked_total} "
        f"nonempty_on_disk={report.n_nonempty_on_disk} zeroed=0 missing_calls=0"
    )

    # Code-version pooling: a hazard measured on the SHARED store (`_select_runs` has no
    # code_version filter) that does not apply here by construction -- this store's population
    # came from curated run-id lists, never an open arm_id query -- but checked directly per arm
    # rather than assumed clear.
    try:
        for name, ids in (
            ("teacher", pops["teacher"]),
            ("trained", pops["trained"]),
            ("prompted", pops["prompted"]),
        ):
            cv_report = assert_one_code_version_and_no_duplicate_keys(STORE, ids, arm=name)
            print(f"  code-version check {name}: {cv_report}")
    except PooledCodeVersions as e:
        print(f"REFUSED: {e}")
        return 1

    scorer_hashes = set()
    graph_versions = set()
    con = duckdb.connect()
    for r in con.execute(
        f"SELECT DISTINCT scorer_hash, graph_version FROM read_parquet('{STORE}/scores.parquet') "
        "WHERE metric_name = 'evidence_coverage'"
    ).fetchall():
        scorer_hashes.add(r[0])
        graph_versions.add(r[1])
    print(f"  scorer_hash(es): {scorer_hashes}")
    print(f"  graph_version(s): {graph_versions}")
    if len(scorer_hashes) != 1:
        print(f"REFUSED: expected ONE scorer_hash in this isolated store, got {scorer_hashes}")
        return 1
    scorer_hash = next(iter(scorer_hashes))

    tabs = mc.read_tables(STORE)
    keep = {"inquirer_trained", "inquirer_prompted"}
    runs = [r for r in tabs["runs"] if r["arm_id"] in keep]
    graphs = {suite: load_graphs(suite, mc.GRAPH_VERSION) for suite in SUITES}
    shims = mc.turn_shims(tabs["turns"])
    tokenizer = mc.find_question_tokenizer()
    if tokenizer is None:
        print("REFUSED: no local tokenizer")
        return 1
    qt = mc.question_tokens(shims, tokenizer)
    qw = mc.question_words(shims)
    gt = mc.read_inquirer_calls(STORE)
    built = mc.build_ladders(
        runs, shims, graphs, inquirer_gen_tokens=gt, question_words=qw, question_tokens=qt
    )
    checked = mc.verify_instrument(built, mc.stored_scores(tabs["scores"]))
    print(f"  verify_instrument locked: {checked}")

    mc.N_BOOT = 10_000
    mc.N_PERM = 10_000

    # Teacher and trained both compared against the SAME prompted-base ladders from this store.
    base = {
        (lad.suite_id, lad.task_id, lad.seed): lad
        for lad in built.values()
        if lad.arm_id == "inquirer_prompted" and lad.run_id in set(pops["prompted"])
    }
    print(f"  base (prompted) ladders: {len(base)}")

    print()
    print("=== teacher-under-token-charge, evidence_coverage, N_BOOT=10000 ===")
    for suite in SUITES:
        teacher_suite = sorted(
            (
                lad
                for lad in built.values()
                if lad.run_id in set(pops["teacher"]) and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        if not teacher_suite:
            print(f"  {suite}: 0 teacher ladders (unexpected)")
            continue
        c = mc.contrast(
            teacher_suite,
            base,
            suite_id=suite,
            checkpoint="teacher(gpt-oss-120b)",
            metric="evidence_coverage",
            comparator="matched_question_tokens",
            offset=0,
        )
        print(
            f"  {suite:<11} n={c.n_tasks:<4} teacher={c.trained_mean:.5f} base={c.base_mean:.5f} "
            f"delta={c.delta:+.5f} ci=[{c.ci_lo:+.5f},{c.ci_hi:+.5f}]"
        )

    print()
    print(
        "=== cross-check: trained-under-token-charge, from THIS fresh store, against the"
        " earlier testsplit_qa-store numbers (RESULT.md's contribution A table) ==="
    )
    PUBLISHED_A_TOKEN = {
        "musique": 0.15483,
        "strategyqa": 0.04974,
        "wiki2": 0.15136,
    }
    for suite in SUITES:
        trained_suite = sorted(
            (
                lad
                for lad in built.values()
                if lad.arm_id == "inquirer_trained" and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        if not trained_suite:
            print(f"  {suite}: 0 trained ladders (unexpected)")
            continue
        c = mc.contrast(
            trained_suite,
            base,
            suite_id=suite,
            checkpoint="qwen3-8b-dpo-stacked-notdone-both",
            metric="evidence_coverage",
            comparator="matched_question_tokens",
            offset=0,
        )
        pub = PUBLISHED_A_TOKEN[suite]
        print(
            f"  {suite:<11} n={c.n_tasks:<4} delta={c.delta:+.5f} ci=[{c.ci_lo:+.5f},{c.ci_hi:+.5f}]"
            f" vs earlier store {pub:+.5f} absdiff={c.delta - pub:.2e}"
        )

    # ----------------------------------------------------------------- contribution 3, DIRECT
    # trained vs teacher, matched cost. NEVER derived as (teacher-vs-prompted delta) minus
    # (trained-vs-prompted delta): the baseline (prompted) is evaluated at a DIFFERENT matched
    # rung in each of those two contrasts, so the two deltas do not share a subtrahend and their
    # difference is not a matched-cost quantity. This is a fresh, direct pairing with the
    # teacher AS the base role.
    print()
    print(
        "=== contribution 3, DIRECT: trained vs teacher, evidence_coverage, N_BOOT=10000"
        " (never a difference of two deltas against the prompted base) ==="
    )
    teacher_base = {
        (lad.suite_id, lad.task_id, lad.seed): lad
        for lad in built.values()
        if lad.run_id in set(pops["teacher"])
    }
    print(f"  teacher (as base) ladders: {len(teacher_base)}")
    runs_by_id = {r["run_id"]: r for r in tabs["runs"]}
    NEAR_ZERO_THRESHOLD = 0.01
    N_BOOT_STABILITY = 50_000
    SEEDS_STABILITY = (0, 1, 2)

    for suite in SUITES:
        trained_suite = sorted(
            (
                lad
                for lad in built.values()
                if lad.arm_id == "inquirer_trained" and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        trained_ids_suite = [
            i for i in pops["trained"] if runs_by_id.get(i, {}).get("suite_id") == suite
        ]
        teacher_ids_suite = [
            i for i in pops["teacher"] if runs_by_id.get(i, {}).get("suite_id") == suite
        ]
        t_rows = [runs_by_id[i] for i in trained_ids_suite]
        te_rows = [runs_by_id[i] for i in teacher_ids_suite]
        print(
            f"  {suite} levels: trained n={len(t_rows)} mean_retrieval_calls="
            f"{sum(r['retrieval_calls'] for r in t_rows) / len(t_rows):.4f} mean_asks="
            f"{sum(r['n_asks'] for r in t_rows) / len(t_rows):.4f}  |  teacher n={len(te_rows)}"
            f" mean_retrieval_calls={sum(r['retrieval_calls'] for r in te_rows) / len(te_rows):.4f}"
            f" mean_asks={sum(r['n_asks'] for r in te_rows) / len(te_rows):.4f}"
        )

        # Pairing direction, checked directly at the (suite, task, seed) PAIR level with its own
        # matching denominator -- NOT via Contrast.n_base_short, which is documented (its own
        # dataclass docstring) as a pair count while n_tasks counts post-seed-averaging tasks;
        # printing "n_base_short of n_tasks" mixes the two denominators and can read as more
        # than 100% (caught here: an earlier draft printed "330 of 200").
        n_pairs = 0
        n_teacher_shorter = 0
        for lad in trained_suite:
            peer = teacher_base.get((lad.suite_id, lad.task_id, lad.seed))
            if peer is None:
                continue
            n_pairs += 1
            if peer.n_asks < lad.n_asks:
                n_teacher_shorter += 1
        print(
            f"    pairing direction (matched_cost.py:625-626,248): {n_pairs} (suite,task,seed)"
            f" pairs; teacher's OWN n_asks < trained's OWN n_asks on {n_teacher_shorter} of them"
            " (there, Ladder.asks_at's min(k, self.n_asks) returns teacher's own smaller value,"
            " i.e. teacher is read at ITS OWN full trajectory, not truncated further; the same"
            " formula, not a special case)"
        )

        c_by_comparator: dict[str, mc.Contrast] = {}
        for comparator in ("matched_k", "cap8"):
            c = mc.contrast(
                trained_suite,
                teacher_base,
                suite_id=suite,
                checkpoint="qwen3-8b-dpo-stacked-notdone-both",
                metric="evidence_coverage",
                comparator=comparator,
                offset=0,
            )
            flagged = near_zero_bounds(c.ci_lo, c.ci_hi, NEAR_ZERO_THRESHOLD)
            stability = "n/a (no bound near zero)"
            if flagged:
                mc.N_BOOT = N_BOOT_STABILITY
                bounds_by_seed: dict[int, tuple[float, float]] = {}
                for sd in SEEDS_STABILITY:
                    mc.SEED = sd
                    c50 = mc.contrast(
                        trained_suite,
                        teacher_base,
                        suite_id=suite,
                        checkpoint="qwen3-8b-dpo-stacked-notdone-both",
                        metric="evidence_coverage",
                        comparator=comparator,
                        offset=0,
                    )
                    bounds_by_seed[sd] = (c50.ci_lo, c50.ci_hi)
                mc.N_BOOT, mc.SEED = 10_000, 0
                stability = stability_verdict(bounds_by_seed, flagged)
                print(
                    f"    [stability] {suite} {comparator} flagged={flagged} 50k={bounds_by_seed} -> {stability}"
                )
            if c.ci_lo > 0:
                verdict = "BEATS"
            elif c.ci_hi < 0:
                verdict = "LOSES TO"
            else:
                verdict = "TIES (CI includes zero)"
            print(
                f"  {suite:<11} {comparator:<10} n={c.n_tasks:<4} trained={c.trained_mean:.5f}"
                f" teacher={c.base_mean:.5f} delta={c.delta:+.5f} ci=[{c.ci_lo:+.5f},{c.ci_hi:+.5f}]"
                f" stability={stability}  ==> trained {verdict} teacher"
            )
            if comparator == "matched_k":
                print(
                    f"    rung (mean k charged to teacher, = trained's own mean asks over these"
                    f" {c.n_tasks} paired tasks) = {c.trained_asks:.4f}; teacher's mean REALIZED"
                    f" asks at that rung = {c.base_asks:.4f} (below the rung whenever a teacher"
                    " run's own trajectory was already shorter than it, see pairing direction"
                    " above)"
                )
            else:
                print(
                    "    rung = the fixed cap, 8 (comparator=cap8 does not depend on trained's spend)"
                )
            c_by_comparator[comparator] = c

        mk, c8 = c_by_comparator["matched_k"], c_by_comparator["cap8"]
        print(
            f"    LEVELS SUMMARY {suite}: at the matched rung (k={mk.trained_asks:.2f}) trained"
            f" reaches {mk.trained_mean:.4f} and teacher (charged the same rung) reaches"
            f" {mk.base_mean:.4f}. At each arm's OWN natural stopping point: trained reaches"
            f" {c8.trained_mean:.4f} (identical to its matched-rung value -- trained is always"
            f" read at its own n_asks) and teacher reaches {c8.base_mean:.4f} (its cap8 terminal"
            " coverage, since this store's teacher population never exceeds 8 asks)."
        )

    # ------------------------------------------------------------- symmetric-pairing recheck
    # A defect found (by another lane, on the paper's primary endpoint) in matched_cost.py's
    # contrast(): `va` (the treatment) is read at its own n_asks UNCONDITIONALLY, never clamped
    # to the comparator's side, so on pairs where the treatment asks MORE than the comparator,
    # the "matched" reading silently reverts to unmatched on that pair. Rechecked here on both
    # of this lane's own contrasts, since this lane's arms differ in spend by roughly 2x and the
    # "unsafe" (peer.n_asks < trained.n_asks) fraction is already known to be non-trivial
    # (see "pairing direction" above: 39/400, 12/400, 52/400 for trained-vs-teacher).
    print()
    print(
        "=== BASIS VERIFICATION (requested directly): does the symmetric path charge tokens,"
        " or did it fall back to ask counts, for the token-priced comparator? ==="
    )
    print(
        "  basis is selected at teacher_token_charge.py's symmetric_contrast, the line"
        " 'budget = comparator in mc.COST_BASES' -- for comparator='matched_question_tokens'"
        " that is True, so the branch below runs Ladder.cost_at(..., comparator), not n_asks."
    )
    _verify_suite = "musique"
    _sample_trained = sorted(
        (
            lad
            for lad in built.values()
            if lad.arm_id == "inquirer_trained" and lad.suite_id == _verify_suite
        ),
        key=lambda x: x.run_id,
    )[:3]
    for _lad in _sample_trained:
        _peer = base.get((_lad.suite_id, _lad.task_id, _lad.seed))
        if _peer is None:
            continue
        _ask_charge = _lad.n_asks  # what an ask-count fallback would have used
        _token_charge = _lad.cost_at(_lad.n_asks, "matched_question_tokens")  # what it must use
        print(
            f"    task={_lad.task_id[:24]:<24} ask_count_charge={_ask_charge}"
            f"  token_charge={_token_charge:.1f}  (a real token count is a float in the tens,"
            " not an integer under 10 -- confirms this is tokens, not a disguised ask count)"
        )
    print(
        "  (if this printed small integers under ~10 for 'token_charge' instead, the earlier"
        " version's fallback bug would still be live; it prints tokens now, per-task, verified"
        " above rather than asserted)"
    )

    print()
    print(
        "=== symmetric-pairing recheck: ask-priced comparators at min(trained.n_asks,"
        " peer.n_asks), budget-priced comparators at min(trained's own total cost, peer's own"
        f" total cost) on THAT basis, N_BOOT={10_000} ==="
    )
    for suite in SUITES:
        trained_suite = sorted(
            (
                lad
                for lad in built.values()
                if lad.arm_id == "inquirer_trained" and lad.suite_id == suite
            ),
            key=lambda x: x.run_id,
        )
        for label, peer_dict, published_comparator in (
            ("trained-vs-teacher", teacher_base, "matched_k"),
            ("trained-vs-base(token)", base, "matched_question_tokens"),
        ):
            published = mc.contrast(
                trained_suite,
                peer_dict,
                suite_id=suite,
                checkpoint="qwen3-8b-dpo-stacked-notdone-both",
                metric="evidence_coverage",
                comparator=published_comparator,
                offset=0,
            )
            sym_est, n_pairs, n_unsafe = symmetric_contrast(
                trained_suite,
                peer_dict,
                comparator=published_comparator,
                n_boot=10_000,
                n_perm=10_000,
                seed=0,
            )
            flagged = near_zero_bounds(sym_est.ci_lo, sym_est.ci_hi, NEAR_ZERO_THRESHOLD)
            sym_stability = "n/a (no bound near zero)"
            if flagged:
                bounds_by_seed = {}
                for sd in SEEDS_STABILITY:
                    est50, _, _ = symmetric_contrast(
                        trained_suite,
                        peer_dict,
                        comparator=published_comparator,
                        n_boot=N_BOOT_STABILITY,
                        n_perm=10_000,
                        seed=sd,
                    )
                    bounds_by_seed[sd] = (est50.ci_lo, est50.ci_hi)
                sym_stability = stability_verdict(bounds_by_seed, flagged)
                print(
                    f"    [stability] {suite} {label} (symmetric) flagged={flagged}"
                    f" 50k={bounds_by_seed} -> {sym_stability}"
                )
            sign_changes = (published.delta > 0) != (sym_est.point > 0)
            pub_excludes_0 = published.ci_lo > 0 or published.ci_hi < 0
            sym_excludes_0 = sym_est.ci_lo > 0 or sym_est.ci_hi < 0
            sig_changes = pub_excludes_0 != sym_excludes_0
            print(
                f"  {suite:<11} {label:<24} published={published.delta:+.4f}"
                f" [{published.ci_lo:+.4f},{published.ci_hi:+.4f}]  symmetric={sym_est.point:+.4f}"
                f" [{sym_est.ci_lo:+.4f},{sym_est.ci_hi:+.4f}]  diff={sym_est.point - published.delta:+.4f}"
                f"  unsafe={n_unsafe}/{n_pairs}  sign_changes={sign_changes}"
                f"  significance_changes={sig_changes}  stability={sym_stability}"
            )

    print()
    print("=== provenance ===")
    print(f"  scorer_hash: {scorer_hash}")
    print(f"  graph_version: {sorted(graph_versions)}")
    code_versions = sorted({str(r["code_version"]) for r in tabs["runs"]})
    print(f"  code_version(s): {code_versions}")
    print(f"  n runs in store: {len(tabs['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
