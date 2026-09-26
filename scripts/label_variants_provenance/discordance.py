"""How many tasks carry each arm-vs-arm cell of `recompute.py`.

A paired per-task difference on coverage is lattice-valued and mostly zero, so an interval that
excludes zero can rest on a handful of tasks. For every (arm pair, suite) this rebuilds the
per-task symmetric matched-cost difference from the STORED `frontier_q#k` ladder with
`pinq_train.gate`'s own primitives -- per (suite, task, seed) both arms at `min(k_a, k_b)`, seeds
averaged into the task -- and counts tasks that are zero, positive and negative. It REFUSES unless
the mean of those per-task values equals the route-B point recorded in `numbers.json` for the
same cell to 1e-12, so the counts describe the published cells and not a lookalike.

    .venv/bin/python scripts/label_variants_provenance/discordance.py \\
        --numbers artifacts/label_variants_provenance_20260922/numbers.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import recompute as R  # noqa: E402

from pinq_train import gate  # noqa: E402


def per_task(
    a: Sequence[Mapping[str, Any]],
    b: Sequence[Mapping[str, Any]],
    ladder: Mapping[str, Mapping[int, float]],
) -> dict[str, float]:
    a_by = {(r["task_id"], int(r["seed"])): r for r in a}
    b_by = {(r["task_id"], int(r["seed"])): r for r in b}
    acc: dict[str, list[float]] = {}
    for key in sorted(set(a_by) & set(b_by)):
        x, y = a_by[key], b_by[key]
        k = min(int(x["n_asks"] or 0), int(y["n_asks"] or 0))
        vx, vy = ladder.get(x["run_id"], {}).get(k), ladder.get(y["run_id"], {}).get(k)
        if vx is not None and vy is not None:
            acc.setdefault(key[0], []).append(vx - vy)
    return {t: sum(v) / len(v) for t, v in acc.items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numbers", type=Path, required=True)
    a = ap.parse_args(argv)
    rec = json.loads(a.numbers.read_text())
    store = R.REPO / rec["provenance"]["store"]
    sh = rec["provenance"]["scorer_hash"]
    con = gate._con(store)
    runs = {}
    for label, (arm_id, model_id) in R.ARMS.items():
        runs[label], census = R.select_arm(
            con, arm_id, model_id, split="test", code_prefix="107ef2221a55"
        )
        if census["run_id_digest_sha256"] != rec["population"][label]["run_id_digest_sha256"]:
            raise SystemExit(f"{label}: selection differs from the population in {a.numbers}")
    ladder = gate._coverage_ladder(con, [r for rs in runs.values() for r in rs], scorer_hash=sh)
    print(f"store {rec['provenance']['store']}  scorer_hash {sh}")
    print(
        f"{'contrast':22s} {'suite':10s} {'n':>4s} {'zero':>5s} {'>0':>4s} {'<0':>4s}  mean (= numbers.json route B point)"
    )
    for x, y in itertools.combinations(R.TRAINED, 2):
        name = f"{x}-{y}"
        for s in R.SUITES:
            v = per_task(
                [r for r in runs[x] if r["suite_id"] == s],
                [r for r in runs[y] if r["suite_id"] == s],
                ladder,
            )
            mean = sum(v.values()) / len(v)
            want = next(
                q["delta"]
                for q in rec["arm_vs_arm"][name][s]["readings"]
                if (q["n_boot"], q["seed"]) == (10000, 0)
            )
            if abs(mean - want) > 1e-12:
                raise SystemExit(f"{name}/{s}: per-task mean {mean!r} != recorded point {want!r}")
            pos = sum(1 for d in v.values() if d > 0)
            neg = sum(1 for d in v.values() if d < 0)
            print(
                f"{name:22s} {s:10s} {len(v):4d} {len(v) - pos - neg:5d} {pos:4d} {neg:4d}  "
                f"{mean:+.6f} (= {want:+.6f})  {rec['arm_vs_arm'][name][s]['verdict']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
