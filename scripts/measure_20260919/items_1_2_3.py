#!/usr/bin/env python
"""
Items 1-3 of the 2026-09-19 trained-vs-teacher measurement lane.
READ-ONLY. scripts/measure_20260919/lib.py, beside this file,
documents the store, scorer_hash, graph_version and pins. Run with:
    .venv/bin/python scripts/measure_20260919/items_1_2_3.py
"""

import sys

sys.path.insert(0, "scripts/measure_20260919")
from lib import (
    GRAPH_VERSION,
    SCORER_HASH,
    STORE,
    SUITES,
    TEACHER_PIN,
    TRAINED_PIN,
    level_point,
    paired,
    per_task_values,
)


def fmt_est(e):
    return f"{e.point:+.4f} [{e.ci_lo:+.4f},{e.ci_hi:+.4f}] p={e.p_value:.5f} n={e.n} ({e.note})"


def fmt_level(d):
    return f"{d['point']:.4f} [{d['lo']:.4f},{d['hi']:.4f}] n={d['n']} clusters={d['n_clusters']}"


print("=" * 100)
print(
    "ITEM 1: tokens_per_useful_need (tok_total / n_resolved; LOWER is better), trained vs teacher"
)
print("=" * 100)
for suite in SUITES:
    r = paired("tokens_per_useful_need", suite)
    # full (UNPAIRED, each arm's own population) levels -- shown ONLY to make the hazard concrete
    trained_full_raw = per_task_values("tokens_per_useful_need", suite, TRAINED_PIN)
    teacher_full_raw = per_task_values("tokens_per_useful_need", suite, TEACHER_PIN)
    trained_full_vals = {k: v for k, (v, _c) in trained_full_raw.items()}
    teacher_full_vals = {k: v for k, (v, _c) in teacher_full_raw.items()}
    trained_full_clusters = {k: c for k, (_v, c) in trained_full_raw.items()}
    teacher_full_clusters = {k: c for k, (_v, c) in teacher_full_raw.items()}
    trained_full_level = level_point(trained_full_vals, trained_full_clusters)
    teacher_full_level = level_point(teacher_full_vals, teacher_full_clusters)

    print(f"\n--- {suite} ---")
    print(
        f"n_res>0 present: trained {r['n_trained_full']}/200 tasks, teacher {r['n_teacher_full']}/200 tasks "
        f"(n_res=0 runs are OMITTED, not zeroed, per src/pi_eval/score.py's emission gate)"
    )
    print(
        f"PAIRED task intersection n = {r['n_paired']} (this is the n every number below is computed on)"
    )
    print(f"  trained level on INTERSECTION : {fmt_level(r['trained_level'])}")
    print(f"  teacher level on INTERSECTION : {fmt_level(r['teacher_level'])}")
    print(f"  paired delta (trained-teacher): {fmt_est(r['delta'])}")
    print(
        f"  [hazard check] trained level on trained's OWN full {r['n_trained_full']}-task set : {fmt_level(trained_full_level)}"
    )
    print(
        f"  [hazard check] teacher level on teacher's OWN full {r['n_teacher_full']}-task set : {fmt_level(teacher_full_level)}"
    )
    if (trained_full_level["point"], teacher_full_level["point"]) != (
        r["trained_level"]["point"],
        r["teacher_level"]["point"],
    ):
        print(
            "  --> CONFIRMED: unpaired per-arm levels differ from the paired-intersection levels; "
            "a level RATIO taken from the two unpaired numbers is NOT the paired ratio."
        )

print()
print("=" * 100)
print("ITEM 2: task_success (evidence_coverage >= 1.0 - 1e-12), trained vs teacher, paired")
print("=" * 100)
for suite in SUITES:
    r = paired("task_success", suite)
    print(f"\n--- {suite} ---")
    print(
        f"n_paired = {r['n_paired']} (task_success has 0 NaN rows on this population; full n_trained={r['n_trained_full']}, n_teacher={r['n_teacher_full']})"
    )
    print(f"  trained level on intersection : {fmt_level(r['trained_level'])}")
    print(f"  teacher level on intersection : {fmt_level(r['teacher_level'])}")
    print(f"  paired delta (trained-teacher): {fmt_est(r['delta'])}")

print()
print("=" * 100)
print(
    "ITEM 3: n_asks, trained vs teacher, paired per-task difference (not a naive mean difference)"
)
print("=" * 100)
for suite in SUITES:
    r = paired("n_asks", suite)
    print(f"\n--- {suite} ---")
    print(f"n_paired = {r['n_paired']}")
    print(f"  trained level on intersection : {fmt_level(r['trained_level'])}")
    print(f"  teacher level on intersection : {fmt_level(r['teacher_level'])}")
    print(f"  paired delta (trained-teacher): {fmt_est(r['delta'])}")

print()
print(f"STORE = {STORE}")
print(f"scorer_hash = {SCORER_HASH}")
print(f"graph_version = {GRAPH_VERSION}")
print(f"TRAINED_PIN = {TRAINED_PIN}")
print(f"TEACHER_PIN = {TEACHER_PIN}")
