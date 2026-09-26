"""answernode vs refresh, paired in its own right.

Two matched-cost deltas against a COMMON baseline cannot be subtracted: the matched-cost
comparator rung moves with each arm, so arm-vs-arm needs its own per-task pairing. This reads the
two trained arms directly against each other, on the seed-matched (suite, task, seed) pairing, in
both cost bases: asymmetric (treatment at its own k, comparator truncated to min) and symmetric
(BOTH arms at min(k_a, k_b)). No published cell exists for this contrast, so there is nothing to
lock against; the same reader double-locked at |d|=0 on 36 published cells, which is the only
provenance available for it.
"""

import json
import math
import os
import sys
from pathlib import Path

# Derived, never hardcoded: an absolute home path here reddens check_no_home_paths for every
# lane in the shared checkout, which is how this file first failed the gate.
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "scripts" / "plan_metrics_symmetric"):
    sys.path.insert(0, str(p))
import duckdb  # noqa: E402
import sweep as _sweep  # noqa: E402
from sweep import (  # noqa: E402
    DIAGNOSTIC_METRIC_FNS,
    EXTRA_METRIC_FNS,
    load_coverage_ladders,
    registered_metrics,
    seed_matched_contrast,
)

from pi_eval.gold import load_graphs  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402

STORE = Path(os.environ.get("L61_STORE", "/tmp/l61-score/p"))  # the isolated scored pass
CK, BA = "qwen3-8b-sft-headline-answernode", "qwen3-8b-sft-headline-refresh"


def ids(model, suite):
    c = duckdb.connect()
    return [
        r[0]
        for r in c.execute(
            f"""select distinct r.run_id from '{(STORE / "runs.parquet").as_posix()}' r
            join '{(STORE / "calls.parquet").as_posix()}' k using(run_id)
            where r.suite_id=? and k.model=?""",
            [suite, model],
        ).fetchall()
    ]


def seeds():
    c = duckdb.connect()
    return {
        r[0]: int(r[1])
        for r in c.execute(
            f"select run_id, seed from '{(STORE / 'runs.parquet').as_posix()}'"
        ).fetchall()
    }


def symmetric(metric, ck, ba, graphs, sd, seed, n_boot):
    fn = _sweep.ladder.METRIC_FNS[metric]

    def key(arm):
        return {(lad.suite_id, lad.task_id, int(sd[rid])): lad for rid, lad in arm.items()}

    A, B = key(ck), key(ba)
    acc = {}
    drop = 0
    for k in sorted(set(A) & set(B)):
        g = graphs.get(k[1])
        if g is None:
            drop += 1
            continue
        a, b = A[k], B[k]
        kk = min(a.n_asks, b.n_asks)
        va, vb = fn(a.at(kk), g), fn(b.at(kk), g)
        if math.isnan(va) or math.isnan(vb):
            drop += 1
            continue
        acc.setdefault(k[1], ([], []))[0].append(va)
        acc[k[1]][1].append(vb)
    pa = {t: sum(x) / len(x) for t, (x, _) in acc.items()}
    pb = {t: sum(y) / len(y) for t, (_, y) in acc.items()}
    e = paired_difference(pa, pb, n_boot=n_boot, seed=seed)
    return {"delta": e.point, "ci_lo": e.ci_lo, "ci_hi": e.ci_hi, "n": e.n, "dropped": drop}


out = {}
sd = seeds()
_ctx = registered_metrics({**EXTRA_METRIC_FNS, **DIAGNOSTIC_METRIC_FNS})
_ctx.__enter__()
for suite in ("musique", "strategyqa", "wiki2"):
    g = load_graphs(suite, "v1")
    ck = load_coverage_ladders(STORE, ids(CK, suite), g)
    ba = load_coverage_ladders(STORE, ids(BA, suite), g)
    asym = seed_matched_contrast("evidence_coverage", ck, ba, g, sd, seed=0, n_boot=10000)
    rows = {
        "asymmetric": {"delta": asym.delta, "ci_lo": asym.ci_lo, "ci_hi": asym.ci_hi, "n": asym.n},
        "symmetric": symmetric("evidence_coverage", ck, ba, g, sd, 0, 10000),
    }
    near = min(abs(rows["symmetric"]["ci_lo"]), abs(rows["symmetric"]["ci_hi"])) < 0.01
    if near:
        rows["symmetric_50k"] = [
            symmetric("evidence_coverage", ck, ba, g, sd, s, 50000) for s in (101, 202, 303)
        ]
    out[suite] = rows
    s = rows["symmetric"]
    print(
        f"  {suite:11s} asym {asym.delta:+.4f} [{asym.ci_lo:+.4f},{asym.ci_hi:+.4f}] n={asym.n}"
        f" | sym {s['delta']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] n={s['n']}"
        + ("  NEAR-ZERO->50k" if near else "")
    )
_ctx.__exit__(None, None, None)
Path(os.environ.get("L61_OUT", "/tmp/l61-score/out/arm_vs_arm.json")).write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n"
)
print("  wrote arm_vs_arm.json")
