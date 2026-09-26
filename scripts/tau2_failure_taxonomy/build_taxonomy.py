"""Lane L4.3b: the tau2 failure taxonomy over the four populations already on disk.

    .venv/bin/python scripts/tau2_failure_taxonomy/build_taxonomy.py [--out FILE.json]

Reads `runs/` and `artifacts/{forks_test,banking}/` by absolute path from the main checkout
(read-only -- see `populations.py`'s `MAIN_CHECKOUT`), classifies every Inquirer ask and every
unit's database outcome, reproduces the published retail/airline follow-up decomposition, and
verifies the banking gold-action/tool-registry claim. Prints every table to stdout (so the
command and its output can be pasted verbatim into RESULT.md, per this repository's rule that a
number without a pasted command is not a result) and, with `--out`, also writes the same
numbers as JSON.

No population is scored, compacted, or launched here: every number below is read from files
`pi run`/`pi score` already wrote.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.tau2_failure_taxonomy import gold_actions
from scripts.tau2_failure_taxonomy import populations as P
from scripts.tau2_failure_taxonomy import turn_accounting as TA
from scripts.tau2_failure_taxonomy.ask_classifier import classify_ask
from scripts.tau2_failure_taxonomy.outcome_classifier import (
    classify_outcome,
    mutating_flag_is_informative,
)
from scripts.tau2_failure_taxonomy.runs_io import RunRecord


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------------- Task 1


def ask_table(populations: list[P.Population]) -> dict[str, Any]:
    """Per (population, arm, domain): the share in each ask class, the answered rate per
    class ("answered" = the ask's `n_retrieved` > 0 -- the retrieval channel returned at least
    one evidence unit), and the mean asks per unit."""
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pop in populations:
        by_key: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
        for r in pop.runs:
            by_key[(pop.name, r.arm_id, P.domain_of(r.suite_id))].append(r)
        for key, runs in by_key.items():
            class_counts: Counter[str] = Counter()
            answered_counts: Counter[str] = Counter()
            n_asks_total = 0
            for r in runs:
                asks = [t for t in r.turns if t.get("action_kind") == "ask"]
                n_asks_total += len(asks)
                for t in asks:
                    cls = classify_ask(t.get("question"))
                    class_counts[cls] += 1
                    if int(t.get("n_retrieved") or 0) > 0:
                        answered_counts[cls] += 1
            n_asks = sum(class_counts.values())
            rows[key] = {
                "n_units": len(runs),
                "n_asks": n_asks,
                "mean_asks_per_unit": _mean(
                    [len([t for t in r.turns if t.get("action_kind") == "ask"]) for r in runs]
                ),
                "share": {
                    c: class_counts[c] / n_asks if n_asks else float("nan")
                    for c in ("customer", "tool", "unclear")
                },
                "answered_rate": {
                    c: (answered_counts[c] / class_counts[c]) if class_counts[c] else float("nan")
                    for c in ("customer", "tool", "unclear")
                },
                "n_by_class": dict(class_counts),
            }
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(rows.items())}


def hand_validation_worksheet(seed: int = 42, n_per_pool: int = 10) -> list[dict[str, Any]]:
    """The 40-ask worksheet: 10 each from published retail, published airline, degraded
    retail, degraded airline, classified and printed for a human to check by eye. This is the
    validation `ask_classifier.py`'s module docstring points at."""
    pub_r = P.published_retail()
    pub_a = P.published_airline()
    deg = P.degraded()
    deg_r = [r for r in deg.runs if r.suite_id == "tau2_retail"]
    deg_a = [r for r in deg.runs if r.suite_id == "tau2_airline"]

    rng = random.Random(seed)
    pools = [
        ("published_retail", pub_r.runs),
        ("published_airline", pub_a.runs),
        ("degraded_retail", deg_r),
        ("degraded_airline", deg_a),
    ]
    rows = []
    for label, pool in pools:
        shuffled = pool[:]
        rng.shuffle(shuffled)
        picked = 0
        for r in shuffled:
            asks = [t.get("question") for t in r.turns if t.get("action_kind") == "ask"]
            if not asks:
                continue
            q = rng.choice(asks)
            rows.append(
                {"population": label, "run_id": r.run_id, "question": q, "class": classify_ask(q)}
            )
            picked += 1
            if picked >= n_per_pool:
                break
    return rows


# --------------------------------------------------------------------------------- Task 2


def _trust_mutating_by_suite(populations: list[P.Population]) -> dict[str, bool]:
    """Measured once per suite, over every `ok` env call this lane has on disk for it -- see
    `outcome_classifier.mutating_flag_is_informative`'s docstring for why banking fails this
    and retail/airline do not."""
    calls_by_suite: dict[str, list[dict]] = defaultdict(list)
    for pop in populations:
        for r in pop.runs:
            calls_by_suite[r.suite_id].extend(r.outcome.get("env_calls") or [])
    return {suite: mutating_flag_is_informative(calls) for suite, calls in calls_by_suite.items()}


def outcome_table(populations: list[P.Population]) -> dict[str, Any]:
    trust_mutating_by_suite = _trust_mutating_by_suite(populations)
    print(f"trust_mutating by suite (measured): {trust_mutating_by_suite}")
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pop in populations:
        by_key: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
        for r in pop.runs:
            by_key[(pop.name, r.arm_id, P.domain_of(r.suite_id))].append(r)
        for key, runs in by_key.items():
            n_scored = n_pass = n_fail = 0
            fail_classes: Counter[str] = Counter()
            preventable = 0
            for r in runs:
                trust_mutating = trust_mutating_by_suite.get(r.suite_id, True)
                result = classify_outcome(
                    r.manifest, r.status, r.outcome, r.turns, trust_mutating=trust_mutating
                )
                if not result["scored"]:
                    continue
                n_scored += 1
                if result["passed"]:
                    n_pass += 1
                    continue
                n_fail += 1
                fc = result["fail_class"]
                fail_classes[fc] += 1
                # THE COUNTERFACTUAL RULE (stated so its limits are visible, not just its
                # trigger): a fail is counted "plausibly preventable by retargeting" when its
                # class is one retargeting could plausibly touch -- `info_never_obtained`
                # (the channel came back with nothing) or `over_cap_runaway` (the Inquirer
                # burned its whole budget without stopping, which a channel that kept
                # returning nothing produces by construction) -- AND more than half of the
                # unit's own asks were phrased customer-addressed, i.e. sent to a channel with
                # no customer to answer them. NECESSARY, NOT SUFFICIENT: retargeting a
                # customer-style ask to a tool-style one does not guarantee the record exists
                # or that it contains the fact needed, only that the channel COULD have
                # answered. `policy_violation` and `wrong_or_missing_tool_action` are excluded
                # by construction -- those are about which action executed, not about whether
                # the ask reached a channel able to inform it.
                if fc in ("info_never_obtained", "over_cap_runaway"):
                    asks = [t for t in r.turns if t.get("action_kind") == "ask"]
                    if asks:
                        customer_share = sum(
                            1 for t in asks if classify_ask(t.get("question")) == "customer"
                        ) / len(asks)
                        if customer_share > 0.5:
                            preventable += 1
            rows[key] = {
                "n_units": len(runs),
                "n_scored": n_scored,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "pass_rate": (n_pass / n_scored) if n_scored else float("nan"),
                "fail_classes": dict(fail_classes),
                "n_fail_plausibly_preventable_by_retargeting": preventable,
                "share_of_fails_preventable": (preventable / n_fail) if n_fail else float("nan"),
            }
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(rows.items())}


# --------------------------------------------------------------------------------- Task 3


def task3_decomposition() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain, pop_fn in (("retail", P.published_retail), ("airline", P.published_airline)):
        pop = pop_fn()
        dec = TA.reproduce_decomposition(pop.runs)
        out[domain] = {
            "sha_ok": pop.sha_ok,
            "run_ids_sha": pop.run_ids_sha,
            "n_runs": len(pop.runs),
            "pooled": dec["pooled"]["pooled"],
            "clustered_trace": dec["clustered"]["levels"]["trace"],
            "over_cap": TA.over_cap_table(pop.runs),
            "closing_share": TA.closing_share(pop.runs),
        }
    return out


def pick_worked_examples(
    pop: P.Population, *, treatment: str = "inquirer_prompted", control: str = "self_ask"
) -> dict[str, Any]:
    """One paired unit where the questioner saved turns, one wasted ask, one failed action --
    named by run_id, with the structural trace (no transcript text -- see turn_accounting.py's
    module docstring for why the outer dialogue's content is not recoverable)."""
    by_key: dict[tuple, dict[str, RunRecord]] = {}
    for r in pop.runs:
        key = (r.manifest.get("foreign_trace_sha"), r.manifest.get("foreign_prefix_k"), r.seed)
        by_key.setdefault(key, {})[r.arm_id] = r

    def followups(r: RunRecord) -> int:
        return int(r.status.get("n_user_turns") or 0) - int(
            r.status.get("n_prefix_user_turns") or 0
        )

    pairs = [
        (k, v[treatment], v[control]) for k, v in by_key.items() if treatment in v and control in v
    ]

    def trace(r: RunRecord) -> dict[str, Any]:
        asks = [t for t in r.turns if t.get("action_kind") == "ask"]
        return {
            "run_id": r.run_id,
            "task_id": r.task_id,
            "n_asks": len(asks),
            "n_retrieved_total": sum(int(t.get("n_retrieved") or 0) for t in asks),
            "follow_ups": followups(r),
            "questions": [t.get("question") for t in asks],
            "outcome": classify_outcome(r.manifest, r.status, r.outcome, r.turns),
        }

    saved = max(pairs, key=lambda p: followups(p[2]) - followups(p[1]), default=None)
    wasted = None
    best_waste = -1
    for _, t, _c in pairs:
        asks = [a for a in t.turns if a.get("action_kind") == "ask"]
        empty = sum(1 for a in asks if int(a.get("n_retrieved") or 0) == 0)
        if empty > best_waste and asks:
            best_waste = empty
            wasted = t
    failed = None
    for _, t, _c in pairs:
        res = classify_outcome(t.manifest, t.status, t.outcome, t.turns)
        if res["scored"] and not res["passed"]:
            failed = t
            break

    return {
        "saved_turns": {"treatment": trace(saved[1]), "control": trace(saved[2])}
        if saved
        else None,
        "wasted_ask": trace(wasted) if wasted else None,
        "failed_action": trace(failed) if failed else None,
    }


# --------------------------------------------------------------------------------- Task 4


def banking_tool_verification(n_sample: int = 20, seed: int = 0) -> dict[str, Any]:
    task_ids = list(gold_actions.all_task_ids("tau2"))
    agent_tools = None
    user_tools = None
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from tau2.registry import registry

        from pinq_adapters.tau2._probe import RETRIEVAL_VARIANT

        env = registry.get_env_constructor("banking_knowledge")(retrieval_variant=RETRIEVAL_VARIANT)
        agent_tools = sorted(t.name for t in env.get_tools())
        user_tools = sorted(t.name for t in env.get_user_tools())
    except Exception as exc:  # noqa: BLE001
        agent_tools = user_tools = f"unavailable: {exc}"

    per_task = {}
    n_with_gap = 0
    for tid in task_ids:
        gap = gold_actions.unresolvable_gold_actions("tau2", tid)
        per_task[tid] = gap
        if gap:
            n_with_gap += 1

    rng = random.Random(seed)
    sample = rng.sample(task_ids, min(n_sample, len(task_ids)))
    sample_report = {
        tid: {
            "n_actions": len(gold_actions.gold_action_names("tau2", tid) or ()),
            "unresolvable": gold_actions.unresolvable_gold_actions("tau2", tid),
        }
        for tid in sorted(sample)
    }

    return {
        "n_tasks": len(task_ids),
        "n_tasks_with_a_gold_action_no_toolkit_registers": n_with_gap,
        "registered_agent_tools": agent_tools,
        "registered_user_tools": user_tools,
        "random_20_sample": sample_report,
    }


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="also write the full result set as JSON")
    ap.add_argument(
        "--skip-task3", action="store_true", help="dev: skip the fork-report reproduction"
    )
    a = ap.parse_args(argv)

    print("=== populations ===")
    pops = P.all_populations()
    for pop in pops:
        print(
            f"{pop.name}: n={len(pop.runs)} run_ids_sha={pop.run_ids_sha} expected={pop.expected_sha} sha_ok={pop.sha_ok}"
        )

    print("\n=== Task 1: ask classification (per population|arm|domain) ===")
    t1 = ask_table(pops)
    print(json.dumps(t1, indent=2, default=str))

    print("\n=== Task 1b: 40-ask hand-validation worksheet ===")
    worksheet = hand_validation_worksheet()
    for i, row in enumerate(worksheet, 1):
        print(
            f"{i:2d}. [{row['population']:17s} {row['run_id'][:10]}] ({row['class']:8s}) {row['question']}"
        )

    print("\n=== Task 2: outcome classification (per population|arm|domain) ===")
    t2 = outcome_table(pops)
    print(json.dumps(t2, indent=2, default=str))

    t3 = {}
    examples = {}
    if not a.skip_task3:
        print("\n=== Task 3: published decomposition reproduction ===")
        t3 = task3_decomposition()
        print(json.dumps(t3, indent=2, default=str))

        print("\n=== Task 3b: worked examples ===")
        examples["retail"] = pick_worked_examples(P.published_retail())
        examples["airline"] = pick_worked_examples(P.published_airline())
        print(json.dumps(examples, indent=2, default=str))

    print("\n=== Task 4: banking tool-registry verification ===")
    t4 = banking_tool_verification()
    print(json.dumps(t4, indent=2, default=str))

    if a.out:
        Path(a.out).write_text(
            json.dumps(
                {
                    "populations": {
                        p.name: {"n": len(p.runs), "sha": p.run_ids_sha, "sha_ok": p.sha_ok}
                        for p in pops
                    },
                    "task1_ask_table": t1,
                    "task1b_worksheet": worksheet,
                    "task2_outcome_table": t2,
                    "task3_decomposition": t3,
                    "task3b_examples": examples,
                    "task4_banking": t4,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
