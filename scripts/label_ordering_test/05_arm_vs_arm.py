"""Step 4b of 4: the arm-vs-arm contrast, as a test of the DIFFERENCE.

Overlapping intervals are not a difference test: two intervals can both contain a value and the
paired difference still exclude zero, and the converse. So each arm's per-task matched-cost value
is re-derived here from `pinq_train.gate`'s own pieces -- the same function that produced the nine
published cells, so the pooled mean of these per-task values must equal the verdict's
`evidence_coverage.value` to the last digit, and that equality is the check that this
re-derivation is the gate's quantity and not a lookalike (it holds: 0.09267262 against 0.092673).

The contrast estimator is `pi_eval.stats.inference.paired_difference` (task-clustered BCa, 1000
resamples, seed 0; cluster-level sign-flip permutation, 10,000, seed 0) -- the estimator the
development reading of these arms used, so the two readings are comparable.

`pinq_train` may not be imported from inside `src/` (import-linter contract 4). This file is an
analysis script under `artifacts/`, not a package module, so it is outside that contract; it is
also why the gate could not compute this contrast itself.
"""

import json
import os
import pathlib
import sys

REPO = pathlib.Path(os.environ.get("PI_REPO", "."))
WORK = pathlib.Path(os.environ["PI_WORK"])
SCORER = os.environ.get(
    "PI_SCORER_HASH", "e82c7458ee8efa11660445bbf434b4c1dc74253253e8f190cc36e92fc50f40c7"
)
sys.path.insert(0, str(REPO / "src"))

from pi_eval.stats.inference import paired_difference  # noqa: E402
from pinq_train import gate  # noqa: E402

ARMS = [
    "qwen3-8b-dpo-headline-control",
    "qwen3-8b-dpo-headline-rater",
    "qwen3-8b-dpo-headline-reaches",
]
GRIDS = ["tier1_trained_qa_base"]

by_arm: dict[str, dict[tuple, float]] = {}
for arm in ARMS:
    con = gate._con(WORK / f"store_{arm}")
    ck = gate._select_runs(con, arm="inquirer_trained", grids=GRIDS, model_id=None)
    ba = gate._select_runs(con, arm="inquirer_prompted", grids=GRIDS, model_id="qwen3-8b-base")
    ckk, bak = gate._by_key(ck), gate._by_key(ba)
    cov = gate._metric_by_run(con, "evidence_coverage", scorer_hash=SCORER)
    n_asks = {str(r["run_id"]): int(r["n_asks"] or 0) for r in gate._with_stop(con, [*ck, *ba])}
    shared = sorted(set(ckk) & set(bak))
    base_used = sorted({r for k in shared for r in bak[k]})
    ladders = gate._coverage_ladder(con, [{"run_id": r} for r in base_used], scorer_hash=SCORER)
    per: dict[tuple, list[float]] = {}
    for key in shared:
        deltas = []
        for c in ckk[key]:
            cv = cov.get(c)
            if cv is None:
                continue
            k = n_asks.get(c, 0)
            # min(k, its own n_asks), NOT k: a budget-blind baseline that stopped at two under
            # cap 8 stops at two under cap 3, and charging it k invents spend it never made.
            rungs = []
            for b in bak[key]:
                rung = (ladders.get(b) or {}).get(min(k, n_asks.get(b, 0)))
                if rung is not None:
                    rungs.append(float(rung))
            if rungs:
                deltas.append(float(cv) - gate._mean(rungs))
        if deltas:
            per.setdefault((key[0], key[1]), []).append(gate._mean(deltas))
    by_arm[arm] = {k: gate._mean(v) for k, v in sorted(per.items())}
    print(
        f"{arm}: tasks={len(by_arm[arm])} pooled_mean={gate._mean(list(by_arm[arm].values())):+.8f}"
    )

out: dict = {"scorer_hash": SCORER, "contrasts": []}
hdr = f"\n{'contrast':22s} {'suite':11s} {'n':>4s} {'diff':>11s} {'ci_lo':>11s} {'ci_hi':>11s} {'p':>7s} excl0"
print(hdr)
for i in range(len(ARMS)):
    for j in range(i + 1, len(ARMS)):
        A, B = ARMS[i], ARMS[j]
        ka, kb = set(by_arm[A]), set(by_arm[B])
        shared_keys = sorted(ka & kb)
        name = f"{A.split('-')[-1]} - {B.split('-')[-1]}"
        for suite in ["musique", "strategyqa", "wiki2", "POOLED"]:
            keys = (
                shared_keys if suite == "POOLED" else [k for k in shared_keys if str(k[0]) == suite]
            )
            am = {"|".join(map(str, k)): by_arm[A][k] for k in keys}
            bm = {"|".join(map(str, k)): by_arm[B][k] for k in keys}
            e = paired_difference(am, bm, clusters=None, n_boot=1000, n_perm=10000, seed=0)
            excl = e.ci_lo > 0 or e.ci_hi < 0
            print(
                f"{name:22s} {suite:11s} {len(keys):4d} {e.point:+11.6f} {e.ci_lo:+11.6f} "
                f"{e.ci_hi:+11.6f} {e.p_value:7.3f} {'yes' if excl else 'no':>5s}"
            )
            out["contrasts"].append(
                {
                    "contrast": name,
                    "suite": suite,
                    "n": len(keys),
                    "diff": e.point,
                    "ci_lo": e.ci_lo,
                    "ci_hi": e.ci_hi,
                    "p": e.p_value,
                    "excludes_zero": excl,
                    "half_width": (e.ci_hi - e.ci_lo) / 2.0,
                }
            )
        print(
            f"{'':22s} shared keys {len(shared_keys)} of {len(ka)} / {len(kb)}; "
            f"unmatched A-only {len(ka - kb)} B-only {len(kb - ka)}"
        )
        out["contrasts"].append(
            {
                "contrast": name,
                "shared_keys": len(shared_keys),
                "n_A": len(ka),
                "n_B": len(kb),
                "unmatched_A_only": len(ka - kb),
                "unmatched_B_only": len(kb - ka),
            }
        )
(WORK / "arm_vs_arm.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
