"""Does the trained questioner respect the retrieval budget more often than the prompted one?

WHY THIS AND NOT THE TURNS CONTRAST. tau2 charges tool calls only AFTER the dialogue terminates, so the
per-turn retrieval check is structurally blind and the prereg's `budget_cap` is not enforced during a
run. Overruns are ARM-CORRELATED, so a follow-up-turns difference is entangled with a runaway-rate
difference: restricting to cap-respecting pairs cut the airline turns effect about 3.4x and lost its
interval. Filtering does not rescue it either -- it conditions on a variable that is arm- and
outcome-correlated, so it selects short dialogues and deletes the comparator's failures preferentially.
This design cannot separate the two post hoc.

What it CAN measure cleanly is the overrun itself, as a first-class endpoint: how often each arm stays
inside the budget it was given. That is a cost-control quantity, it needs no separation from runaways
because it IS the runaway rate, and it is paired by construction.

TWO ENDPOINTS, BOTH PAIRED ON THE FORK POINT:
  binary  -- within budget or not, compared with an exact McNemar on discordant pairs only. Concordant
             pairs carry no information about a difference and inflating n with them is the
             anticonservative error this repo has on record.
  count   -- retrieval calls, paired difference, reported with a sign test as well as a mean, because a
             mean over a runaway-heavy distribution is carried by its tail.

RETRIEVAL CALLS come from each unit's own `outcome.json` `env_calls`, which is what the environment
actually executed, not a model's self-report and not a planned count.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _sign_test(xs: list[float]) -> float:
    """Two-sided exact sign test over non-ties."""
    from math import comb

    neg = sum(1 for x in xs if x < 0)
    pos = sum(1 for x in xs if x > 0)
    n = neg + pos
    if n == 0:
        return float("nan")
    k = min(neg, pos)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar over the b+c discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--treatment", default="inquirer_trained")
    ap.add_argument("--control", default="inquirer_prompted")
    ap.add_argument("--out")
    a = ap.parse_args()

    cells: dict = defaultdict(dict)
    for st in Path(a.runs_root).rglob("status.json"):
        if "/.preflight_" in str(st):
            continue
        try:
            s = json.loads(st.read_text())
        except Exception:
            continue
        if s.get("status") != "ok":
            continue
        try:
            o = json.loads((st.parent / "outcome.json").read_text())
        except Exception:
            continue
        kt = s.get("key") or []
        variant = kt[9] if len(kt) > 9 else "?"
        cells[(str(s.get("task_id")), str(s.get("seed")), variant)][s.get("arm_id")] = {
            "calls": len(o.get("env_calls") or []),
            "run_id": s.get("run_id"),
        }

    report = {"cap": a.cap, "runs_root": a.runs_root, "variants": {}}
    for variant in sorted({k[2] for k in cells}):
        pairs = [
            v for k, v in cells.items() if k[2] == variant and a.treatment in v and a.control in v
        ]
        t_in = c_in = 0
        b = c = 0  # b: treatment in / control over ; c: treatment over / control in
        diffs = []
        for v in pairs:
            ti = v[a.treatment]["calls"] <= a.cap
            ci = v[a.control]["calls"] <= a.cap
            t_in += ti
            c_in += ci
            if ti and not ci:
                b += 1
            elif ci and not ti:
                c += 1
            diffs.append(v[a.treatment]["calls"] - v[a.control]["calls"])
        n = len(pairs)
        if n == 0:
            continue
        rec = {
            "n_pairs": n,
            "treatment_within_budget": t_in,
            "control_within_budget": c_in,
            "treatment_rate": round(t_in / n, 4),
            "control_rate": round(c_in / n, 4),
            "mcnemar_b_treatment_only_within": b,
            "mcnemar_c_control_only_within": c,
            "mcnemar_exact_p": round(_mcnemar(b, c), 6),
            "calls_diff_mean": round(statistics.fmean(diffs), 3),
            "calls_diff_median": statistics.median(diffs),
            "calls_sign_test_p": round(_sign_test(diffs), 6),
            "calls_fewer_more": [sum(1 for d in diffs if d < 0), sum(1 for d in diffs if d > 0)],
        }
        report["variants"][variant] = rec
        print(
            "  {:<10} pairs={:<3} within-budget {}/{} ({:.0%}) vs {}/{} ({:.0%})  "
            "McNemar b={} c={} p={:.4g}".format(
                variant, n, t_in, n, t_in / n, c_in, n, c_in / n, b, c, rec["mcnemar_exact_p"]
            )
        )
        print(
            "             calls diff mean={:+.2f} median={:+.1f} fewer/more={}/{} sign p={:.4g}".format(
                rec["calls_diff_mean"],
                rec["calls_diff_median"],
                rec["calls_fewer_more"][0],
                rec["calls_fewer_more"][1],
                rec["calls_sign_test_p"],
            )
        )
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print("  wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
