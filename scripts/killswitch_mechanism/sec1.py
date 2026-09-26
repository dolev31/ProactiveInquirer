# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import dataclasses
import os

from pi_eval.prereg import CI_RESAMPLES
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
agg = dataclasses.replace(
    open_agg(
        "/private/tmp/mech-ks-idl/pq_m", scorer_hash=SH, n_boot=CI_RESAMPLES, n_perm=10_000, seed=0
    ),
    excluded_run_ids=frozenset(EXCL),
)
print("parallel_replay AGAINST ITS OWN REFERENCE inquirer_prompted")
for suite in ("musique", "strategyqa", "POOLED"):
    kw = dict(
        suite=None if suite == "POOLED" else suite,
        pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
    )
    for metric in (
        "evidence_coverage",
        "coverage_matched_calls",
        "n_asks",
        "retrieval_calls",
        "n_evidence",
        "task_success",
    ):
        p = pull(agg, metric, **kw)
        keys = sorted(set(p.by_arm["inquirer_prompted"]) & set(p.by_arm["parallel_replay"]))
        a = {k: p.by_arm["inquirer_prompted"][k] for k in keys}
        b = {k: p.by_arm["parallel_replay"][k] for k in keys}
        ident = sum(1 for k in keys if a[k] == b[k])
        e = paired_difference(
            a, b, clusters=p.clusters, n_boot=agg.n_boot, n_perm=agg.n_perm, seed=agg.seed
        )
        pv = "nan" if e.p_value is None else f"{e.p_value:.4f}"
        print(
            f"  {suite:10s} {metric:24s} n={e.n:4d} delta={e.point:+.6f} CI[{e.ci_lo:+.6f},{e.ci_hi:+.6f}] p={pv}  identical {ident}/{len(keys)}"
        )
# and random_q volume identity against its own reference
print("\nrandom_q volume against inquirer_prompted (its mining reference)")
for suite in ("musique", "strategyqa"):
    p = pull(agg, "n_asks", suite=suite)
    keys = sorted(set(p.by_arm["inquirer_prompted"]) & set(p.by_arm["random_q"]))
    same = sum(1 for k in keys if p.by_arm["inquirer_prompted"][k] == p.by_arm["random_q"][k])
    d = sum(p.by_arm["random_q"][k] - p.by_arm["inquirer_prompted"][k] for k in keys) / len(keys)
    print(f"  {suite:10s} n_asks identical on {same}/{len(keys)} tasks, mean delta {d:+.6f}")
