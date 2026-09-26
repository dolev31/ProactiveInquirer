"""Any interval bound within 0.01 of zero is re-read at 50,000 resamples, three seeds.

WHY. Measured on this repo 2026-09-18: 4 of 7 near-zero bounds were UNSTABLE at 1,000
resamples, and one PASS (8b-sft-headline/musique) is really a fail at 10k/50k -- typeset in
four places before anyone re-read it. A bound near zero decides `passed`, so at 10k it is the
resample count being reported, not the data. Sign stability is stated per criterion.
"""

import json
import subprocess
from pathlib import Path

MAIN = Path.home() / "PycharmProjects/ProactiveInquirer"
WT = Path.home() / "PycharmProjects/pinq-wt-rung32b-20260919"
A = MAIN / "artifacts/rung32b_gate_20260919"
FARM = A / "farm"
PI = str(MAIN / ".venv/bin/pi")
NEAR = 0.01
SEEDS = (0, 1, 2)
N_BIG = 50000


def gate(suite, n, seed, out):
    cmd = [
        PI,
        "train",
        "gate",
        "--parquet-dir",
        str(FARM / "scores_parquet"),
        "--grid-name",
        f"dev_select_{suite}",
        "--baseline-grid-name",
        f"dev_baseline_{suite}",
        "--checkpoint-arm",
        "inquirer_trained",
        "--baseline-arm",
        "inquirer_prompted",
        "--baseline-model-id",
        "qwen3-32b-base",
        "--checkpoint-model-id",
        "qwen3-32b-sft-headline",
        "--grids-root",
        str(WT / "conf/grids"),
        "--n-resamples",
        str(n),
        "--bootstrap-seed",
        str(seed),
        "--out",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r, cmd


report = {}
for suite in ("musique", "strategyqa"):
    base = A / f"qwen3-32b-sft-headline.{suite}.n10000.json"
    if not base.exists():
        print(f"SKIP {suite}: no n=10000 verdict")
        continue
    d = json.loads(base.read_text())
    near = {}
    for name, c in d["criteria"].items():
        if not isinstance(c, dict):
            continue
        for b in ("ci_lo", "ci_hi"):
            v = c.get(b)
            if isinstance(v, (int, float)) and abs(v) < NEAR:
                near.setdefault(name, []).append((b, v))
    print(f"\n=== {suite}: bounds within {NEAR} of zero at n=10,000 ===")
    if not near:
        print("  none -- no 50,000-resample re-read required")
        report[suite] = {"near_zero": {}, "rereads": {}}
        continue
    for k, v in near.items():
        print(f"  {k}: {v}")

    runs = {}
    for seed in SEEDS:
        out = A / f"qwen3-32b-sft-headline.{suite}.n{N_BIG}.seed{seed}.json"
        r, cmd = gate(suite, N_BIG, seed, out)
        print(f"  [{suite} n={N_BIG} seed={seed}] exit={r.returncode}")
        if not out.exists():
            print("   ", r.stdout[-500:], r.stderr[-500:])
            continue
        runs[seed] = json.loads(out.read_text())

    tbl = {}
    for name in near:
        row = {
            "n10000": {b: d["criteria"][name].get(b) for b in ("ci_lo", "ci_hi")},
            "n10000_passed": d["criteria"][name].get("passed"),
            "gated": d["criteria"][name].get("gated"),
        }
        for seed, dd in runs.items():
            c = dd["criteria"].get(name, {})
            row[f"n{N_BIG}_seed{seed}"] = {b: c.get(b) for b in ("ci_lo", "ci_hi")}
            row[f"n{N_BIG}_seed{seed}_passed"] = c.get("passed")
        # sign stability: does the interval exclude zero, and on which side, in every reading?
        sides = set()
        for key in [("n10000",)] + [(f"n{N_BIG}_seed{s}",) for s in runs]:
            lo, hi = row[key[0]]["ci_lo"], row[key[0]]["ci_hi"]
            if lo is None or hi is None:
                sides.add("na")
            elif lo > 0:
                sides.add("excludes_zero_positive")
            elif hi < 0:
                sides.add("excludes_zero_negative")
            else:
                sides.add("straddles_zero")
        passes = {row.get(f"n{N_BIG}_seed{s}_passed") for s in runs} | {row["n10000_passed"]}
        row["sign_stability"] = sorted(sides)
        row["stable"] = len(sides) == 1
        row["passed_stability"] = sorted(str(p) for p in passes)
        row["passed_stable"] = len(passes) == 1
        tbl[name] = row
        print(
            f"  {name}: sign={sorted(sides)} stable={len(sides) == 1} "
            f"passed={sorted(str(p) for p in passes)} stable={len(passes) == 1}"
        )
    report[suite] = {"near_zero": {k: v for k, v in near.items()}, "rereads": tbl}

(A / "resample_stability.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {A / 'resample_stability.json'}")
