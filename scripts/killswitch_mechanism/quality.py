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
print("The two structurally-zero arms, read on a metric that CAN vary (exploratory, no p licence)")
for metric in ("answer_correct", "answer_token_f1"):
    for suite in ("musique", "strategyqa", "POOLED"):
        kw = dict(
            suite=None if suite == "POOLED" else suite,
            pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
        )
        p = pull(agg, metric, **kw)
        for arm in ("verbosity", "compute_matched", "drafter_only"):
            keys = sorted(set(p.by_arm["inquirer_trained"]) & set(p.by_arm[arm]))
            a = {k: p.by_arm["inquirer_trained"][k] for k in keys}
            b = {k: p.by_arm[arm][k] for k in keys}
            e = paired_difference(
                a, b, clusters=p.clusters, n_boot=agg.n_boot, n_perm=agg.n_perm, seed=agg.seed
            )
            ident = (
                sum(1 for k in keys if p.by_arm["verbosity"].get(k) == p.by_arm[arm].get(k))
                if arm != "verbosity"
                else len(keys)
            )
            print(
                f"  {metric:17s} {suite:10s} {arm:16s} n={e.n:4d} switch={sum(b.values()) / len(b):.4f} "
                f"delta={e.point:+.4f} CI[{e.ci_lo:+.4f},{e.ci_hi:+.4f}] p={e.p_value:.4f} distinct={len(set(b.values()))} same_as_verbosity={ident}/{len(keys)}"
            )
