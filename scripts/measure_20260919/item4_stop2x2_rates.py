#!/usr/bin/env python
"""
Follow-up check on item 4: stop2x2_n_stop_at_done and stop2x2_n_ask_at_not_done are RAW
COUNTS of decision points, not rates -- and item 3 already shows trained asks far fewer
questions per task than teacher (musique 2.57 vs 5.24, strategyqa 1.58 vs 6.14, wiki2 1.97
vs 3.08), so trained structurally has fewer decision points of every kind. A raw count is
therefore confounded with spend exactly the way tok_total/n_turns are (CONTRIBUTING.md: "wall_ms
and usd are machine artifacts ... never compared across arms as evidence"; same logic
applies to any unnormalized per-run count). This computes the RATE instead:
  rate_stop_at_done   = stop2x2_n_stop_at_done / stop2x2_n_done        (of the chances to
                         stop-because-done, the share actually taken; higher is better)
  rate_ask_at_not_done= stop2x2_n_ask_at_not_done / stop2x2_n_not_done (of the chances to
                         ask-because-not-done, the share actually taken; higher is better)
both restricted to tasks where the arm's OWN denominator (seed-averaged) is > 0, then paired
on the intersection. READ-ONLY, same store/pins/estimator as lib.py.
"""

import sys

sys.path.insert(0, "scripts/measure_20260919")
from lib import N_BOOT, SEED, SUITES, TEACHER_PIN, TRAINED_PIN, level_point, per_task_values

from pi_eval.stats.inference import paired_difference


def rate_series(suite, pin, num_name, den_name):
    num = {k: v for k, (v, _c) in per_task_values(num_name, suite, pin).items()}
    den = {k: v for k, (v, _c) in per_task_values(den_name, suite, pin).items()}
    clusters = {k: c for k, (_v, c) in per_task_values(den_name, suite, pin).items()}
    out = {}
    for k in set(num) & set(den):
        if den[k] and den[k] > 0:
            out[k] = num[k] / den[k]
    return out, clusters


for label, num_name, den_name in [
    (
        "rate_stop_at_done (of done-chances, share stopped)",
        "stop2x2_n_stop_at_done",
        "stop2x2_n_done",
    ),
    (
        "rate_ask_at_not_done (of not-done-chances, share asked)",
        "stop2x2_n_ask_at_not_done",
        "stop2x2_n_not_done",
    ),
]:
    print("=" * 100)
    print(f"ITEM 4 FOLLOW-UP: {label}")
    print("=" * 100)
    for suite in SUITES:
        t_rate, t_clusters = rate_series(suite, TRAINED_PIN, num_name, den_name)
        e_rate, e_clusters = rate_series(suite, TEACHER_PIN, num_name, den_name)
        clusters = {**t_clusters, **e_clusters}
        keys = sorted(set(t_rate) & set(e_rate))
        print(
            f"--- {suite} --- trained denom>0: {len(t_rate)}/200 tasks, teacher denom>0: {len(e_rate)}/200 tasks, paired n={len(keys)}"
        )
        if not keys:
            print(
                "    no paired tasks with both denominators > 0 -- cannot compute a rate contrast here."
            )
            continue
        inter_clusters = {k: clusters[k] for k in keys}
        tl = level_point({k: t_rate[k] for k in keys}, inter_clusters)
        el = level_point({k: e_rate[k] for k in keys}, inter_clusters)
        est = paired_difference(
            {k: t_rate[k] for k in keys},
            {k: e_rate[k] for k in keys},
            clusters=inter_clusters,
            n_boot=N_BOOT,
            seed=SEED,
        )
        print(f"    trained rate: {tl['point']:.4f} [{tl['lo']:.4f},{tl['hi']:.4f}]")
        print(f"    teacher rate: {el['point']:.4f} [{el['lo']:.4f},{el['hi']:.4f}]")
        print(
            f"    paired delta: {est.point:+.4f} [{est.ci_lo:+.4f},{est.ci_hi:+.4f}] p={est.p_value:.5f} n={est.n}"
        )
