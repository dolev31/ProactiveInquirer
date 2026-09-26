"""Read every cell of the seed-replicate gate at 10k AND 50k resamples across three bootstrap
seeds, and re-derive the PUBLISHED s0 cell at the published resample count.

TWO THINGS THIS SETTLES.

1. `A pass decided by the resample count is a property of the count.` The standing rule is
   10k, with 50k across three seeds for any bound within 0.01 of zero. One bound here came out
   at `0.010000000000000002` -- inside the tolerance by intent and outside it by float dust --
   so the tolerance test is dropped entirely and EVERY cell is read at both counts.

2. Whether the published s0 cell is reproducible in THIS store. The published cell was computed
   at `n_resamples=1000`, `seed=0`, in a store whose `scorer_hash` is `aa18b1fe`; this store's
   is different. Reading s0 here at 1000/0 makes the two directly comparable, so the
   store-vs-store question is answered by a number rather than by an argument about hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "decomposition_test"))
from contrast import select_and_contrast, select_and_contrast_symmetric  # noqa: E402

S0 = "qwen3-8b-dpo-stacked-notdone-both"
S1, S2 = f"{S0}-s1", f"{S0}-s2"
BASE = "qwen3-8b-base"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--scorer-hash", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pq, sh = Path(a.parquet), a.scorer_hash
    out: dict = {
        "scorer_hash": sh,
        "vs_comparator": {},
        "arm_vs_arm": {},
        "published_reproduction": {},
    }

    for mid in (S0, S1, S2):
        rows = {}
        for n in (10_000, 50_000):
            for seed in (0, 1, 2):
                r = select_and_contrast(
                    pq,
                    checkpoint_model_id=mid,
                    baseline_model_id=BASE,
                    scorer_hash=sh,
                    seed=seed,
                    n_resamples=n,
                )
                for s, c in r["by_suite"].items():
                    rows.setdefault(s, []).append(
                        {
                            "n_resamples": n,
                            "seed": seed,
                            "delta": c["delta"],
                            "ci_lo": c["ci_lo"],
                            "ci_hi": c["ci_hi"],
                        }
                    )
        out["vs_comparator"][mid] = rows

    for x, y in ((S1, S2), (S1, S0), (S2, S0)):
        rows = {}
        for n in (10_000, 50_000):
            for seed in (0, 1, 2):
                r = select_and_contrast_symmetric(
                    pq, arm_a_model_id=x, arm_b_model_id=y, scorer_hash=sh, seed=seed, n_resamples=n
                )
                for s, c in r["by_suite"].items():
                    rows.setdefault(s, []).append(
                        {
                            "n_resamples": n,
                            "seed": seed,
                            "delta": c["delta"],
                            "ci_lo": c["ci_lo"],
                            "ci_hi": c["ci_hi"],
                        }
                    )
        out["arm_vs_arm"][f"{x} vs {y}"] = rows

    r = select_and_contrast(
        pq, checkpoint_model_id=S0, baseline_model_id=BASE, scorer_hash=sh, seed=0, n_resamples=1000
    )
    out["published_reproduction"] = {
        s: {
            "delta": c["delta"],
            "ci_lo": c["ci_lo"],
            "ci_hi": c["ci_hi"],
            "cap8_coverage_delta": c["cap8_coverage_delta"],
            "n_tasks": c["n_tasks"],
            "trained_mean_k": c["trained_mean_k"],
            "baseline_mean_k_charged": c["baseline_mean_k_charged"],
            "n_baseline_shorter_than_k": c["n_baseline_shorter_than_k"],
        }
        for s, c in r["by_suite"].items()
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["published_reproduction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
