#!/usr/bin/env python
"""
Item 4 of the 2026-09-19 trained-vs-teacher measurement lane: screen every
needs_answer=False, non-indexed metric in pi_eval.score.METRICS that is actually
defined on both TRAINED_PIN and TEACHER_PIN in the isolated store, and classify
whether it favours trained with a CI excluding zero.
READ-ONLY. See scripts/measure_20260919/lib.py for store/scorer_hash/pins.
"""

import sys

sys.path.insert(0, "scripts/measure_20260919")
from lib import SUITES, paired, score_mod

ALREADY_REPORTED = {"evidence_coverage", "task_success", "n_asks", "tokens_per_useful_need"}
ARTIFACT_METRICS = score_mod.ARTIFACT_METRICS
# Pure spend/operational counters that are NOT in ARTIFACT_METRICS but whose
# higher_is_better=True is the MetricDef dataclass DEFAULT (never set deliberately
# for a descriptive spend counter) -- treating "fewer tokens/calls/turns" as an
# "adverse" or "favouring" QUALITY finding via that default would misrepresent a
# spend signal as a quality claim, the same category error CLAUDE.md calls out for
# wall_ms/usd. Reported for completeness, never classified as qualifying/adverse.
SPEND_NOT_QUALITY = {
    "tok_prompt",
    "tok_completion",
    "tok_cached",
    "tok_total",
    "retrieval_calls",
    "unique_docs",
    "n_turns",
    "n_evidence",
}

BY_NAME = {m.name: m for m in score_mod.METRICS}
cands = [
    m.name
    for m in score_mod.METRICS
    if not m.needs_answer and not m.indexed and m.name not in ALREADY_REPORTED
]

rows = []  # (metric, suite, status, detail_str, est)
for name in sorted(cands):
    hib = BY_NAME[name].higher_is_better
    bucket = (
        "ARTIFACT"
        if name in ARTIFACT_METRICS
        else ("SPEND" if name in SPEND_NOT_QUALITY else "CLAIMABLE")
    )
    for suite in SUITES:
        r = paired(name, suite)
        if r["n_trained_full"] == 0 and r["n_teacher_full"] == 0:
            rows.append((name, suite, bucket, "NOT_EMITTED", None, r))
            continue
        if r["n_paired"] == 0:
            rows.append((name, suite, bucket, "NO_OVERLAP", None, r))
            continue
        # structural-constant check: recompute raw diffs on the intersection
        est = r["delta"]
        # pull the actual per-task diffs to check constancy (paired_difference doesn't return them)
        from lib import per_task_values

        tv = {
            k: v
            for k, (v, _c) in per_task_values(name, suite, __import__("lib").TRAINED_PIN).items()
        }
        te = {
            k: v
            for k, (v, _c) in per_task_values(name, suite, __import__("lib").TEACHER_PIN).items()
        }
        keys = sorted(set(tv) & set(te))
        diffs = [tv[k] - te[k] for k in keys]
        span = (max(diffs) - min(diffs)) if diffs else 0.0
        if span < 1e-9:
            rows.append(
                (
                    name,
                    suite,
                    bucket,
                    "STRUCTURALLY_CONSTANT",
                    diffs[0] if diffs else float("nan"),
                    r,
                )
            )
            continue
        excludes_zero = not (est.ci_lo <= 0 <= est.ci_hi)
        if not excludes_zero:
            status = "TIE"
        else:
            favors_trained = (est.point > 0) if hib else (est.point < 0)
            status = "FAVORS_TRAINED" if favors_trained else "ADVERSE"
        rows.append((name, suite, bucket, status, None, r))

print(
    f"{'metric':30s} {'suite':11s} {'bucket':10s} {'status':22s} {'delta':>12s} {'ci_lo':>12s} {'ci_hi':>12s} {'n':>4s}"
)
for name, suite, bucket, status, const_val, r in rows:
    est = r["delta"]
    if status == "NOT_EMITTED":
        print(
            f"{name:30s} {suite:11s} {bucket:10s} {status:22s} {'--':>12s} {'--':>12s} {'--':>12s} {0:4d}"
        )
    elif status == "NO_OVERLAP":
        print(
            f"{name:30s} {suite:11s} {bucket:10s} {status:22s} {'--':>12s} {'--':>12s} {'--':>12s} {0:4d}"
        )
    elif status == "STRUCTURALLY_CONSTANT":
        print(
            f"{name:30s} {suite:11s} {bucket:10s} {status:22s} {const_val:12.6f} {'--':>12s} {'--':>12s} {r['n_paired']:4d}"
        )
    else:
        print(
            f"{name:30s} {suite:11s} {bucket:10s} {status:22s} {est.point:12.4f} {est.ci_lo:12.4f} {est.ci_hi:12.4f} {est.n:4d}"
        )

print()
print("=== SUMMARY: qualifying (CLAIMABLE bucket, FAVORS_TRAINED) ===")
for name, suite, bucket, status, const_val, r in rows:
    if bucket == "CLAIMABLE" and status == "FAVORS_TRAINED":
        est = r["delta"]
        print(f"  {name} / {suite}: {est.point:+.4f} [{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n}")

print()
print("=== SUMMARY: adverse (CLAIMABLE bucket, ADVERSE) ===")
for name, suite, bucket, status, const_val, r in rows:
    if bucket == "CLAIMABLE" and status == "ADVERSE":
        est = r["delta"]
        print(f"  {name} / {suite}: {est.point:+.4f} [{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n}")

print()
print("=== SUMMARY: structurally constant (any bucket) ===")
for name, suite, bucket, status, const_val, r in rows:
    if status == "STRUCTURALLY_CONSTANT":
        print(
            f"  {name} / {suite}: constant diff = {const_val:.6f} over n={r['n_paired']} paired tasks"
        )

print()
print("=== SUMMARY: not emitted at all on this suite (both arms zero rows) ===")
seen = set()
for name, suite, bucket, status, const_val, r in rows:
    if status == "NOT_EMITTED":
        seen.add(name)
if seen:
    for name in sorted(seen):
        print(f"  {name}")
else:
    print("  (none -- all NOT_EMITTED metrics were already screened out before this table)")

print()
print(
    "=== SUMMARY: ARTIFACT bucket (excluded from qualifying per CLAUDE.md/ARTIFACT_METRICS regardless of CI) ==="
)
for name, suite, bucket, status, const_val, r in rows:
    if bucket == "ARTIFACT" and status not in ("NOT_EMITTED", "NO_OVERLAP"):
        est = r["delta"]
        print(
            f"  {name} / {suite}: {est.point:+.4f} [{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n}  (mechanical status would be {status})"
        )

print()
print(
    "=== SUMMARY: SPEND bucket (operational counters; higher_is_better is the dataclass default, not an editorial claim -- not scored as qualifying/adverse) ==="
)
for name, suite, bucket, status, const_val, r in rows:
    if bucket == "SPEND" and status not in ("NOT_EMITTED", "NO_OVERLAP", "STRUCTURALLY_CONSTANT"):
        est = r["delta"]
        print(
            f"  {name} / {suite}: {est.point:+.4f} [{est.ci_lo:+.4f},{est.ci_hi:+.4f}] n={est.n}  (mechanical status would be {status})"
        )
