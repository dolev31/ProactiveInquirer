"""Score one tau2-bench campaign under the benchmark's own metric, from the run directories.

WHY NOT `pi score`. `pi_eval`'s scorer computes THIS repository's endpoints against a gold
graph. The benchmark's own endpoint is already in each run's `outcome.json` --
`native.upstream_reward`, the product over the task's `reward_basis`, written by
`tau2_runner.grade_upstream` at rollout time because the transcript it needs is deliberately
not persisted. So this reads run directories, and the only arithmetic it does is upstream's:

    pass_hat_k(n, c, k) = C(c, k) / C(n, k)          -- tau2/metrics/agent_metrics.py
    pass^k              = that, averaged over tasks
    success             = is_successful(reward), i.e. reward within 1e-6 of 1.0

IT REFUSES A TASK WITH FEWER THAN k TRIALS rather than scoring it at the trials it has.
`pass_hat_k` raises on `num_trials < k` and upstream's own reader silently lowers k to the
smallest task's trial count, which turns one truncated task into a different metric for the
whole table. Here the task is DROPPED from k and the drop is counted and printed, so a partial
campaign reports a smaller population rather than a different number.

INTERVALS ARE CLUSTERED ON THE TASK, because the four trials of one task are not independent
draws: they share the customer's script, the database and the gold action set. A bootstrap over
simulations would divide the standard error by two for free.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- loading


def load_runs(runs_root: str, run_ids: Iterable[str]) -> list[dict[str, Any]]:
    """One row per run id, from `manifest.json`, `status.json` and `outcome.json`.

    A run id that is named and missing RAISES. A silently skipped run is the difference
    between "this cell has 456 units" and "this cell has the 431 units that happened to be
    readable", and only one of those is a population.
    """
    out: list[dict[str, Any]] = []
    root = Path(runs_root)
    for rid in run_ids:
        d = root / rid
        man = json.loads((d / "manifest.json").read_text())
        st = json.loads((d / "status.json").read_text())
        native: dict[str, Any] = {}
        oc = d / "outcome.json"
        if oc.is_file():
            native = dict((json.loads(oc.read_text()).get("native") or {}))
        out.append(
            {
                "run_id": rid,
                "suite_id": man.get("suite_id"),
                "task_id": str(man.get("task_id")),
                "arm_id": man.get("arm_id"),
                "seed": int(man.get("seed", 0)),
                "split": man.get("split"),
                "code_version": man.get("code_version"),
                "model_pin_hash": man.get("model_pin_hash"),
                "upstream_pins": man.get("upstream_pins") or {},
                "status": st.get("status"),
                "n_user_turns": st.get("n_user_turns"),
                "n_asks": st.get("n_asks"),
                "n_turns": st.get("n_turns"),
                "n_env_calls": st.get("n_env_calls"),
                "n_messages": st.get("n_messages"),
                "termination_reason": st.get("termination_reason"),
                "usd_billed": float(st.get("usd_billed") or 0.0),
                "usage_usd": float((st.get("usage") or {}).get("usd") or 0.0),
                "user_sim_usd": float(st.get("user_sim_usd") or 0.0),
                "stock_agent_usd": float(st.get("stock_agent_usd") or 0.0),
                "native": native,
            }
        )
    return out


# --------------------------------------------------------------------------- upstream's metric


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Upstream's formula, transcribed from `tau2/metrics/agent_metrics.py`."""
    if num_trials < k:
        raise ValueError(f"num_trials {num_trials} < k {k}")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def by_task(rows: Sequence[dict[str, Any]], field: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r["native"].get(field)
        if v is None:
            continue
        out[r["task_id"]].append(float(v))
    return dict(out)


def pass_k_table(rows: Sequence[dict[str, Any]], field: str, kmax: int = 4) -> dict[str, Any]:
    """pass^1..pass^kmax over the tasks that carry EXACTLY the trials each k needs.

    `n_tasks_at_k` is reported per k and is what any interval is computed over.
    """
    per_task = by_task(rows, field)
    out: dict[str, Any] = {"n_tasks_seen": len(per_task)}
    for k in range(1, kmax + 1):
        vals = [
            pass_hat_k(len(v), sum(1 for x in v if x >= 1.0 - 1e-6), k)
            for v in per_task.values()
            if len(v) >= k
        ]
        out[f"pass^{k}"] = statistics.fmean(vals) if vals else float("nan")
        out[f"n_tasks_at_{k}"] = len(vals)
    out["avg_reward"] = (
        statistics.fmean([x for v in per_task.values() for x in v]) if per_task else float("nan")
    )
    return out


# --------------------------------------------------------------------------- intervals


def boot_mean(values: Sequence[float], *, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap of a mean over INDEPENDENT units (here, tasks)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(statistics.fmean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return statistics.fmean(values), lo, hi


def paired_boot(
    a: dict[str, float], b: dict[str, float], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Paired difference a - b over the tasks present in BOTH, clustered on the task.

    Paired on the task and not on (task, trial): the trials of one task share a script and a
    database, so pairing them as independent units would halve the standard error for free.
    """
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    mean, lo, hi = boot_mean(diffs, n_boot=n_boot, seed=seed)
    return {
        "n_tasks_paired": len(keys),
        "difference": mean,
        "ci95": [lo, hi],
        "covers_zero": bool(lo <= 0.0 <= hi),
    }


def per_task_mean(rows: Sequence[dict[str, Any]], key) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is None:
            continue
        acc[r["task_id"]].append(float(v))
    return {t: statistics.fmean(v) for t, v in acc.items()}


def per_task_pass_k(rows: Sequence[dict[str, Any]], field: str, k: int) -> dict[str, float]:
    return {
        t: pass_hat_k(len(v), sum(1 for x in v if x >= 1.0 - 1e-6), k)
        for t, v in by_task(rows, field).items()
        if len(v) >= k
    }


# --------------------------------------------------------------------------- the report


def ask_targets(runs_root: str, rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Where this cell's ASKs were addressed: the customer, or the knowledge base.

    Read from `turns.jsonl`, whose `target` column is the policy's own declaration. It is the
    quantity the `inquirer_may_ask_user` arm exists to produce: every other arm runs with
    `allow_user_target=False`, so an ask aimed at the customer is rewritten to the index and
    the share is 0 by construction -- which is a fact about the arm table, not about the
    policy, and is why it is reported per cell rather than pooled.
    """
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        p = Path(runs_root) / r["run_id"] / "turns.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if str(t.get("action_kind")) != "ask":
                continue
            out[str(t.get("target") or "(unset)")] += 1
    return dict(sorted(out.items()))


def cell_report(rows: Sequence[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    rep: dict[str, Any] = {
        "n_units": len(rows),
        "n_ok": len(ok),
        "n_error": sum(1 for r in rows if r["status"] != "ok"),
    }
    rep["upstream"] = pass_k_table(ok, "upstream_reward")
    rep["db_only"] = pass_k_table(ok, "tau_reward")
    turns = per_task_mean(ok, lambda r: r["n_user_turns"])
    m, lo, hi = boot_mean(sorted(turns.values()), n_boot=n_boot, seed=seed)
    rep["user_turns_per_task"] = {"mean": m, "ci95": [lo, hi], "n_tasks": len(turns)}
    for name, key in (
        ("asks_per_task", lambda r: r["n_asks"]),
        ("env_calls_per_task", lambda r: r["n_env_calls"]),
        ("usage_usd_per_unit", lambda r: r["usage_usd"]),
        ("usd_billed_per_unit", lambda r: r["usd_billed"]),
        ("user_sim_usd_per_unit", lambda r: r["user_sim_usd"]),
        ("stock_agent_usd_per_unit", lambda r: r["stock_agent_usd"]),
    ):
        vals = [float(key(r) or 0.0) for r in ok]
        rep[name] = statistics.fmean(vals) if vals else float("nan")
    rep["usage_usd_total"] = sum(r["usage_usd"] for r in rows)
    rep["usd_billed_total"] = sum(r["usd_billed"] for r in rows)
    term: dict[str, int] = defaultdict(int)
    for r in ok:
        term[str(r["termination_reason"]).rsplit(".", 1)[-1]] += 1
    rep["termination"] = dict(sorted(term.items()))
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--run-ids", required=True, help="JSON: {cell_id: {domain: [run_id, ...]}}")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default="stock")
    p.add_argument("--second-baseline", default="prompted_8b_base")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    index = json.loads(Path(args.run_ids).read_text())
    report: dict[str, Any] = {"cells": {}, "contrasts": {}}
    loaded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell_id, per_domain in sorted(index.items()):
        for domain, ids in sorted(per_domain.items()):
            rows = load_runs(args.runs_root, ids)
            loaded[(cell_id, domain)] = rows
            rep = cell_report(rows, n_boot=args.n_boot, seed=args.seed)
            rep["ask_targets"] = ask_targets(
                args.runs_root, [r for r in rows if r["status"] == "ok"]
            )
            rep["pins"] = sorted(
                {
                    (
                        str(r["model_pin_hash"]),
                        str(r["upstream_pins"].get("user_sim", "")),
                        str(r["upstream_pins"].get("agent_model", "")),
                        str(r["upstream_pins"].get("protocol", "")),
                        str(r["upstream_pins"].get("max_steps", "")),
                        str(r["upstream_pins"].get("max_errors", "")),
                        str(r["code_version"]),
                    )
                    for r in rows
                }
            )
            rep["splits"] = dict(
                sorted(
                    {
                        s: sum(1 for r in rows if r["split"] == s)
                        for s in {str(r["split"]) for r in rows}
                    }.items()
                )
            )
            report["cells"].setdefault(domain, {})[cell_id] = rep

    for (cell_id, domain), rows in sorted(loaded.items()):
        for base in (args.baseline, args.second_baseline):
            if cell_id == base or (base, domain) not in loaded:
                continue
            b = [r for r in loaded[(base, domain)] if r["status"] == "ok"]
            a = [r for r in rows if r["status"] == "ok"]
            key = f"{domain}:{cell_id}-minus-{base}"
            report["contrasts"][key] = {
                "pass^1": paired_boot(
                    per_task_pass_k(a, "upstream_reward", 1),
                    per_task_pass_k(b, "upstream_reward", 1),
                    n_boot=args.n_boot,
                    seed=args.seed,
                ),
                "pass^4": paired_boot(
                    per_task_pass_k(a, "upstream_reward", 4),
                    per_task_pass_k(b, "upstream_reward", 4),
                    n_boot=args.n_boot,
                    seed=args.seed,
                ),
                "user_turns": paired_boot(
                    per_task_mean(a, lambda r: r["n_user_turns"]),
                    per_task_mean(b, lambda r: r["n_user_turns"]),
                    n_boot=args.n_boot,
                    seed=args.seed,
                ),
            }

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
