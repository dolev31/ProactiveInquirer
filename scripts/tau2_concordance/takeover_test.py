"""Does the trained questioner TAKE OVER the downstream agent's lookups? A mechanism test.

On the tau2 fork campaigns the trained questioner asks more on retail (+4.35 / +4.91 at the task
level) while the downstream agent -- byte-identical in both arms -- makes fewer lookup calls. The
tempting reading is substitution: the questioner front-loads the information gathering the agent
would otherwise do. Substitution predicts that WITHIN pairs, more questioner asks go with FEWER
agent lookups -- a NEGATIVE correlation between d_asks and d_agent_reads.

Measured 2026-09-22 the correlation is POSITIVE in all four suite x prompt cells (task-level rho
+0.48, +0.78, +0.62, +0.21; permutation p 0.017, <0.0001, 0.009, 0.42). Where the questioner asks
more, the agent also looks up more. The two counts share a common factor -- how long the fork runs
-- and substitution is REFUTED. The average shifts stand; the mechanism linking them does not.

Agent lookups are READ calls by tau2's own declared tool types (conf/tau2/tool_types.json), not the
record's `mutating` field, which is True on every call.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scipy.stats import spearmanr  # noqa: E402

from pi_eval.fork_report import _pair_forks  # noqa: E402

TYPES = json.loads(
    (Path(__file__).resolve().parents[2] / "conf" / "tau2" / "tool_types.json").read_text()
)


def domain(r: dict) -> str:
    return str((r.get("key") or [""])[0]).removeprefix("tau2_")


def agent_reads(r: dict) -> int:
    counts = r.get("_env_tool_counts")
    if counts is None:
        raise SystemExit("records carry no _env_tool_counts; re-extract with extract_records.py")
    return sum(n for t, n in counts.items() if TYPES[domain(r)][t] == "READ")


def perm_p(xs: list, ys: list, n: int = 20000, seed: int = 0) -> float:
    rng = random.Random(seed)
    obs = abs(spearmanr(xs, ys)[0])
    ys = list(ys)
    hit = 0
    for _ in range(n):
        rng.shuffle(ys)
        if abs(spearmanr(xs, ys)[0]) >= obs:
            hit += 1
    return (hit + 1) / (n + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--treatment", default="inquirer_trained")
    ap.add_argument("--control", default="inquirer_prompted")
    a = ap.parse_args()

    runs = [r for r in json.loads(Path(a.records).read_text()) if r.get("status") == "ok"]
    if not runs:
        print("REFUSING: no ok runs")
        return 3
    print("substitution predicts NEGATIVE; a shared episode-length factor predicts POSITIVE")
    for variant in sorted({(r.get("key") or ["?"] * 10)[9] for r in runs}):
        sub = [r for r in runs if (r.get("key") or ["?"] * 10)[9] == variant]
        pairs, _, _ = _pair_forks(sub, treatment=a.treatment, control=a.control)
        if len(pairs) < 3:
            print(f"  {variant}: {len(pairs)} pairs -- too few to correlate, REFUSED")
            continue
        per_task: dict = defaultdict(lambda: ([], []))
        da, dr = [], []
        for v in pairs.values():
            x = (v[a.treatment].get("n_asks") or 0) - (v[a.control].get("n_asks") or 0)
            y = agent_reads(v[a.treatment]) - agent_reads(v[a.control])
            da.append(x)
            dr.append(y)
            t = str(v[a.treatment].get("task_id"))
            per_task[t][0].append(x)
            per_task[t][1].append(y)
        tx = [statistics.fmean(p) for p, _ in per_task.values()]
        ty = [statistics.fmean(q) for _, q in per_task.values()]
        print(
            f"  {variant:<10} pair rho={spearmanr(da, dr)[0]:+.3f} (n={len(da)})  "
            f"TASK rho={spearmanr(tx, ty)[0]:+.3f} (n={len(tx)})  perm p={perm_p(tx, ty):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
