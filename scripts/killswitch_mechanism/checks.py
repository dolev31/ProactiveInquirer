# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import dataclasses
import os

from pi_eval.prereg import CI_RESAMPLES, KILL_SWITCH_MARGINS
from pi_eval.report import open_agg, pull
from pi_eval.stats.inference import paired_difference

PI_REPO = os.environ.get("PI_REPO", os.getcwd()).rstrip("/")


SH = "3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"
R = PI_REPO + "/"
EXCL = [
    ln.strip()
    for ln in open(
        R + "artifacts/nulls_and_determinism_20260918/EXCLUDE_random_q_zero_ask.eligible.txt"
    )
    if ln.strip()
]
SW = [
    "parallel_replay",
    "verbosity",
    "compute_matched",
    "random_q",
    "checklist",
    "inquirer_noevidence",
    "inquirer_depth1",
    "self_inquire",
]
agg = dataclasses.replace(
    open_agg(
        "/private/tmp/mech-ks-idl/pq_m", scorer_hash=SH, n_boot=CI_RESAMPLES, n_perm=10_000, seed=0
    ),
    excluded_run_ids=frozenset(EXCL),
)


def show(tag, a, b, cl, m=0.0):
    e = paired_difference(a, b, clusters=cl, n_boot=agg.n_boot, n_perm=agg.n_perm, seed=agg.seed)
    pv = "nan" if e.p_value is None else f"{e.p_value:.4f}"
    print(f"  {tag:44s} n={e.n:4d} delta={e.point:+.4f} CI[{e.ci_lo:+.4f},{e.ci_hi:+.4f}] p={pv}")
    return e


print(
    "### 1. parallel_replay AGAINST ITS OWN REFERENCE inquirer_prompted (the zero that is claimed)"
)
for suite in ("musique", "strategyqa", "POOLED"):
    kw = dict(
        suite=None if suite == "POOLED" else suite,
        pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
    )
    for metric in ("evidence_coverage", "coverage_matched_calls"):
        p = pull(agg, metric, **kw)
        keys = sorted(set(p.by_arm["inquirer_prompted"]) & set(p.by_arm["parallel_replay"]))
        a = {k: p.by_arm["inquirer_prompted"][k] for k in keys}
        b = {k: p.by_arm["parallel_replay"][k] for k in keys}
        ident = sum(1 for k in keys if a[k] == b[k])
        show(f"{suite:10s} {metric:22s} prompted-replay", a, b, p.clusters)
        print(f"      identical task values: {ident}/{len(keys)}")

print("\n### 2. STRUCTURAL DETERMINATION: distinct values the instrument took, per arm and suite")
for suite in ("musique", "strategyqa"):
    p = pull(agg, "evidence_coverage", suite=suite)
    pm = pull(agg, "coverage_matched_calls", suite=suite)
    for arm in ["inquirer_trained"] + SW:
        v = list(p.by_arm[arm].values())
        vm = list(pm.by_arm[arm].values())
        print(
            f"  {suite:11s} {arm:20s} n={len(v):4d} distinct(unmatched)={len(set(v)):4d} "
            f"distinct(matched)={len(set(vm)):4d} mean={sum(v) / len(v):.4f} zeros={sum(1 for x in v if x == 0.0):4d}"
        )

print("\n### 3. ASK / RETRIEVAL VOLUME per arm (why unmatched is not readable alone)")
for suite in ("musique", "strategyqa"):
    pa = pull(agg, "n_asks", suite=suite)
    pr = pull(agg, "retrieval_calls", suite=suite)
    for arm in ["inquirer_trained", "inquirer_prompted", "drafter_only"] + SW:
        va = list(pa.by_arm[arm].values())
        vr = list(pr.by_arm[arm].values())
        print(
            f"  {suite:11s} {arm:20s} mean n_asks={sum(va) / len(va):6.3f}  mean retrieval_calls={sum(vr) / len(vr):6.3f}"
        )

print("\n### 4. REDUNDANCY: is any switch the SAME measurement as another?")
for suite in ("musique", "strategyqa"):
    p = pull(agg, "evidence_coverage", suite=suite)
    for i, a1 in enumerate(SW):
        for a2 in SW[i + 1 :]:
            keys = sorted(set(p.by_arm[a1]) & set(p.by_arm[a2]))
            same = sum(1 for k in keys if p.by_arm[a1][k] == p.by_arm[a2][k])
            if keys and same / len(keys) > 0.5:
                print(f"  {suite:11s} {a1:20s} vs {a2:20s} identical on {same}/{len(keys)} tasks")
    for arm in SW:
        keys = sorted(set(p.by_arm[arm]) & set(p.by_arm["drafter_only"]))
        same = sum(1 for k in keys if p.by_arm[arm][k] == p.by_arm["drafter_only"][k])
        if keys and same / len(keys) > 0.5:
            print(
                f"  {suite:11s} {arm:20s} vs drafter_only        identical on {same}/{len(keys)} tasks"
            )

print("\n### 5. MATCHED vs UNMATCHED are the same measurement for which arms?")
for suite in ("musique", "strategyqa"):
    p = pull(agg, "evidence_coverage", suite=suite)
    pm = pull(agg, "coverage_matched_calls", suite=suite)
    for arm in ["inquirer_trained"] + SW:
        keys = sorted(set(p.by_arm[arm]) & set(pm.by_arm[arm]))
        same = sum(1 for k in keys if p.by_arm[arm][k] == pm.by_arm[arm][k])
        print(f"  {suite:11s} {arm:20s} matched==unmatched on {same}/{len(keys)} tasks")

print("\n### 6. NON-VACUITY of the estimator on every EQUIVALENT / near-null cell")
for suite, arm, metric in (
    ("POOLED", "self_inquire", "coverage_matched_calls"),
    ("POOLED", "self_inquire", "evidence_coverage"),
    ("strategyqa", "self_inquire", "coverage_matched_calls"),
    ("POOLED", "verbosity", "coverage_matched_calls"),
    ("POOLED", "compute_matched", "coverage_matched_calls"),
):
    kw = dict(
        suite=None if suite == "POOLED" else suite,
        pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
    )
    p = pull(agg, metric, **kw)
    keys = sorted(set(p.by_arm["inquirer_trained"]) & set(p.by_arm[arm]))
    a = {k: p.by_arm["inquirer_trained"][k] for k in keys}
    print(f"  -- {suite} {arm} {metric}")
    for shift in (0.0, +0.02, +0.06, -0.06):
        b = {k: p.by_arm[arm][k] + shift for k in keys}
        show(f"     comparator {shift:+.2f}", a, b, p.clusters)

print("\n### 7. COMMON task set across ALL EIGHT switches and the treatment")
for suite in ("musique", "strategyqa", "POOLED"):
    kw = dict(
        suite=None if suite == "POOLED" else suite,
        pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
    )
    pu = pull(agg, "evidence_coverage", **kw)
    pm = pull(agg, "coverage_matched_calls", **kw)
    common = set(pu.by_arm["inquirer_trained"])
    for a1 in SW:
        common &= set(pu.by_arm[a1])
    common = sorted(common)
    print(f"\n  suite={suite}  COMMON n={len(common)}")
    for arm in SW:
        for tag, p in (("UNMATCHED", pu), ("MATCHED  ", pm)):
            a = {k: p.by_arm["inquirer_trained"][k] for k in common}
            b = {k: p.by_arm[arm][k] for k in common}
            m = float(KILL_SWITCH_MARGINS.get(arm, 0.0))
            e = paired_difference(
                a, b, clusters=p.clusters, n_boot=agg.n_boot, n_perm=agg.n_perm, seed=agg.seed
            )
            pv = "nan" if e.p_value is None else f"{e.p_value:.4f}"
            print(
                f"    {arm:20s} {tag} n={e.n:4d} sw={sum(b.values()) / len(b):.4f} "
                f"delta={e.point:+.4f} CI[{e.ci_lo:+.4f},{e.ci_hi:+.4f}] p={pv}"
            )
