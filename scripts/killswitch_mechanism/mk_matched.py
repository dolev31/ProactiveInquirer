# Repo root and scratch dir come from the environment: a committed absolute home path is
# how a research repo stops being reproducible for anyone but its author.

import os
import pathlib
import shutil

import numpy as np
import pandas as pd

PI_REPO = os.environ.get("PI_REPO", os.getcwd()).rstrip("/")

"""Emit coverage_matched_calls: comparator coverage at the trained run's own retrieval spend."""

SRC = pathlib.Path("/private/tmp/mech-ks-idl/pq")
DST = pathlib.Path("/private/tmp/mech-ks-idl/pq_m")
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
SH = "3f88ac3527ed6c2fcf78b1538f6c44a3c14838aec93a583c072a770fb70e8162"
runs = pd.read_parquet(
    DST / "runs.parquet",
    columns=[
        "run_id",
        "arm_id",
        "suite_id",
        "task_id",
        "seed",
        "retrieval_calls",
        "n_asks",
        "status",
        "code_version",
    ],
)
sc = pd.read_parquet(DST / "scores.parquet")
assert set(sc.scorer_hash.unique()) == {SH}, sc.scorer_hash.unique()

lad = sc[sc.metric_name.str.startswith(("frontier_spend#", "frontier_q#"))].copy()
lad["k"] = lad.metric_name.str.split("#").str[1].astype(int)
lad["kind"] = lad.metric_name.str.split("#").str[0]
spend = {}
qq = {}
for rid, k, kind, v in zip(lad.run_id, lad.k, lad.kind, lad.value):
    (spend if kind == "frontier_spend" else qq).setdefault(rid, {})[k] = v

term = sc[sc.metric_name == "evidence_coverage"].set_index("run_id").value.to_dict()
# terminal frontier spend must equal recorded retrieval_calls -- checked, not assumed
info = runs.set_index("run_id")
bad = 0
for rid, sp in spend.items():
    if abs(max(sp.values()) - float(info.at[rid, "retrieval_calls"])) > 1e-9:
        bad += 1
print(
    f"CHECK terminal frontier_spend == retrieval_calls: {len(spend) - bad}/{len(spend)} agree, {bad} disagree"
)

trained = runs[(runs.arm_id == "inquirer_trained") & (runs.status == "ok")]
budget = {
    (s, t, d): float(c)
    for s, t, d, c in zip(trained.suite_id, trained.task_id, trained.seed, trained.retrieval_calls)
}
print("trained budget cells:", len(budget))

rows = []
stats = {}
for rid, arm, s, t, d, rc in zip(
    runs.run_id, runs.arm_id, runs.suite_id, runs.task_id, runs.seed, runs.retrieval_calls
):
    if rid not in spend or rid not in qq:
        continue
    c = budget.get((s, t, d))
    if c is None:
        continue
    sp, q = spend[rid], qq[rid]
    ks = [k for k in sorted(sp) if sp[k] <= c + 1e-9]
    kstar = max(ks) if ks else 0
    while kstar not in q and kstar > 0:
        kstar -= 1  # NaN rungs are absent, step down
    if kstar not in q:
        continue
    truncated = kstar < max(sp)
    rows.append(
        {
            "run_id": rid,
            "metric_name": "coverage_matched_calls",
            "scorer_hash": SH,
            "value": float(q[kstar]),
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "n": 1,
            "graph_version": "v1",
            "notes": f"prefix k={kstar} within trained spend {c:g}",
        }
    )
    a = stats.setdefault(arm, {"n": 0, "trunc": 0, "spend": [], "tspend": []})
    a["n"] += 1
    a["trunc"] += int(truncated)
    a["spend"].append(float(rc))
    a["tspend"].append(c)
print(
    f"\n{'arm':22s} {'n':>5s} {'truncated by the match':>22s}  mean own spend  mean trained spend"
)
for arm in sorted(stats):
    a = stats[arm]
    print(
        f"{arm:22s} {a['n']:5d} {a['trunc']:10d} ({100 * a['trunc'] / a['n']:5.1f}%)   "
        f"{np.mean(a['spend']):8.3f}        {np.mean(a['tspend']):8.3f}"
    )
new = pd.concat(
    [sc, pd.DataFrame(rows).astype(sc.dtypes[list(rows[0])].to_dict(), errors="ignore")],
    ignore_index=True,
)
new.to_parquet(DST / "scores.parquet", index=False)
print("\nwrote", len(rows), "coverage_matched_calls rows ->", DST / "scores.parquet")
