# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import dataclasses
import json
import math
import os

from pi_eval.prereg import CI_RESAMPLES, KILL_SWITCH_MARGINS
from pi_eval.report import open_agg, pull
from pi_eval.stats.inference import paired_difference

PI_REPO = os.environ.get("PI_REPO", os.getcwd()).rstrip("/")


SH = "3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"
EXCL = [
    ln.strip()
    for ln in open(
        PI_REPO + "/artifacts/nulls_and_determinism_20260918/EXCLUDE_random_q_zero_ask.eligible.txt"
    )
    if ln.strip()
]
SWITCHES = [
    "parallel_replay",
    "verbosity",
    "compute_matched",
    "random_q",
    "checklist",
    "inquirer_noevidence",
    "inquirer_depth1",
    "self_inquire",
]
TREAT = "inquirer_trained"
agg0 = open_agg(
    "/private/tmp/mech-ks-idl/pq_m", scorer_hash=SH, n_boot=CI_RESAMPLES, n_perm=10_000, seed=0
)
agg = dataclasses.replace(agg0, excluded_run_ids=frozenset(EXCL))
print(
    "n_boot", agg.n_boot, "n_perm", agg.n_perm, "seed", agg.seed, "| excluded run ids:", len(EXCL)
)
print("PREDICATE:", agg.predicate[:200], "...")


def verd(pt, lo, hi, n, m):
    if n == 0:
        return "NOT RUN"
    if pt == 0.0 and lo == 0.0 and hi == 0.0:
        return "DEGENERATE (cannot differ)"
    if math.isnan(lo) or math.isnan(hi):
        return "INDETERMINATE"
    if lo > 0:
        return "SEPARATED (switch below trained)"
    if hi < 0:
        return "INVERTED (switch above trained)"
    if m > 0 and lo >= -m and hi <= m:
        return f"EQUIVALENT (|CI|<={m:.2f})"
    if m > 0:
        return f"INCONCLUSIVE (CI exceeds +/-{m:.2f})"
    return "MATCHES"


out = {}
for suite in ("musique", "strategyqa", "POOLED"):
    kw = dict(
        suite=None if suite == "POOLED" else suite,
        pool_suites=("musique", "strategyqa") if suite == "POOLED" else (),
    )
    pu = pull(agg, "evidence_coverage", **kw)
    pm = pull(agg, "coverage_matched_calls", **kw)
    print(f"\n{'=' * 118}\nSUITE {suite}   arms present: {pu.arms()}")
    for arm in SWITCHES:
        if arm not in pu.by_arm:
            print(f"  {arm:20s}  NOT PRESENT")
            continue
        m = float(KILL_SWITCH_MARGINS.get(arm, 0.0))
        line = {}
        for tag, p in (("UNMATCHED", pu), ("MATCHED  ", pm)):
            keys = sorted(set(p.by_arm[TREAT]) & set(p.by_arm[arm]))
            a = {k: p.by_arm[TREAT][k] for k in keys}
            b = {k: p.by_arm[arm][k] for k in keys}
            e = paired_difference(
                a, b, clusters=p.clusters, n_boot=agg.n_boot, n_perm=agg.n_perm, seed=agg.seed
            )
            pv = "nan" if e.p_value is None else f"{e.p_value:.4f}"
            lvl = sum(b.values()) / len(b) if b else float("nan")
            tl = sum(a.values()) / len(a) if a else float("nan")
            print(
                f"  {arm:20s} {tag} n={e.n:4d} trained={tl:.4f} switch={lvl:.4f} "
                f"delta={e.point:+.4f} CI[{e.ci_lo:+.4f},{e.ci_hi:+.4f}] p={pv:>7s} "
                f"{verd(e.point, e.ci_lo, e.ci_hi, e.n, m)}"
            )
            line[tag.strip()] = dict(
                n=e.n,
                trained=tl,
                switch=lvl,
                delta=e.point,
                lo=e.ci_lo,
                hi=e.ci_hi,
                p=e.p_value,
                verdict=verd(e.point, e.ci_lo, e.ci_hi, e.n, m),
                margin=m,
            )
        out[(suite, arm)] = line
json.dump(
    {f"{s}|{a}": v for (s, a), v in out.items()},
    open("/private/tmp/mech-ks-idl/table.json", "w"),
    indent=1,
)
