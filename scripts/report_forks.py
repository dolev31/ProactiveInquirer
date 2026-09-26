"""Paired-fork engagement report over runs on disk: the paper's headline, by command.

    python scripts/report_forks.py --suite tau2_retail --split test \
        --treatment inquirer_prompted --control self_ask [--seed 0 --seed 1 ...] [--out report.json]

    # the same contrast at every clustering resolution, published unit beside recomputed unit
    python scripts/report_forks.py --suite tau2_retail --split test --clusters

    # the airline confirmatory population: duplicate-cell selection record + the MDE at 20 tasks
    python scripts/report_forks.py --cells --suite tau2_airline \
        --prefer-code-version 5800f8db --endpoint n_user_turns \
        --arm drafter_only --arm inquirer_prompted

Definitions and provenance are in `pi_eval.fork_report`. Prints one line per seed and the pooled
line; writes a JSON with run ids digest, code versions and every number printed.

`--clusters` EXISTS BECAUSE THE POOLED p WAS ANTICONSERVATIVE. It counted 102 pairs as 102
independent observations of 32 recorded dialogues. Both numbers are printed, labelled, in one
table: a recomputation that replaces a published value in silence is indistinguishable from a
recomputation that was never checked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pi_eval.fork_report import (  # noqa: E402
    clustered_engagement,
    minimum_detectable_effect,
    paired_engagement,
    select_one_per_cell,
)


def load_runs(
    runs_root: Path, *, suite: str, split: str, arms: set[str], seeds: set[int] | None
) -> list[dict]:
    out: list[dict] = []
    for m in runs_root.glob("*/manifest.json"):
        d = json.loads(m.read_text())
        if (
            d.get("suite_id") != suite
            or str(d.get("split")) != split
            or d.get("arm_id") not in arms
        ):
            continue
        if seeds is not None and d.get("seed") not in seeds:
            continue
        st = m.parent / "status.json"
        if not st.is_file():
            continue
        s = json.loads(st.read_text())
        out.append(
            {
                "run_id": d["run_id"],
                "arm_id": d["arm_id"],
                "seed": d.get("seed"),
                "foreign_trace_sha": d.get("foreign_trace_sha"),
                "foreign_prefix_k": d.get("foreign_prefix_k"),
                "task_id": d.get("task_id"),
                "code_version": d.get("code_version", ""),
                "status": s.get("status"),
                "n_user_turns": s.get("n_user_turns"),
                "n_prefix_user_turns": s.get("n_prefix_user_turns"),
                "tau_reward": (s.get("native") or {}).get("tau_reward"),
            }
        )
    return out


ELIGIBLE_COLS = (
    "status",
    "gold_exposed",
    "is_dev_run",
    "dirty",
    "pilot_flag",
    "canary_hit",
    "exploratory",
    "firewall_ok",
    "counterfactual_kind",
    "split",
    "reconciled_tokens",
    "reconciled_docs",
)


def load_eligible(parquet: Path, *, suite: str, arms: set[str], endpoint: str) -> list[dict]:
    """`pi_eval.report.ELIGIBLE`, evaluated in Python over runs.parquet.

    Re-expressed rather than imported because `report.ELIGIBLE` is a SQL string meant for duckdb;
    every clause is reproduced here term for term and the predicate is echoed into the record, so
    a reader can diff the two by eye. If they ever drift, the record says which one ran.
    """
    import pyarrow.parquet as pq

    cols = [
        "run_id",
        "suite_id",
        "task_id",
        "arm_id",
        "seed",
        "code_version",
        "model_pin_hash",
        "grid_name",
        endpoint,
        *ELIGIBLE_COLS,
    ]
    t = pq.read_table(parquet, columns=sorted(set(cols))).to_pydict()
    out = []
    for i in range(len(t["run_id"])):
        if t["suite_id"][i] != suite or t["arm_id"][i] not in arms:
            continue
        if not (
            t["status"][i] == "ok"
            and t["gold_exposed"][i] is False
            and t["is_dev_run"][i] is False
            and t["dirty"][i] is False
            and t["pilot_flag"][i] is False
            and t["canary_hit"][i] is False
            and t["exploratory"][i] is False
            and t["firewall_ok"][i] is True
            and t["counterfactual_kind"][i] == "none"
            and t["split"][i] == "test"
            and t["reconciled_tokens"][i] is True
            and t["reconciled_docs"][i] is True
        ):
            continue
        out.append({c: t[c][i] for c in cols})
    return out


def cells_report(a) -> dict:
    """The duplicate-cell selection record, the residual arm balance, and the MDE at the number
    of CLUSTERS the endpoint actually has -- not the number of rows it appears to have."""
    arms = set(a.arm or ["drafter_only", "inquirer_prompted", "inquirer_may_ask_user"])
    rows = load_eligible(Path(a.parquet), suite=a.suite, arms=arms, endpoint=a.endpoint)
    sel = select_one_per_cell(rows, prefer_code_version=a.prefer_code_version)
    kept = sel.pop("kept")
    per_arm = Counter(r["arm_id"] for r in kept)
    # the population RESTRICTED to one code version, printed beside the deduplicated one so the
    # difference between the two rules is a measurement rather than an argument
    restricted = [r for r in kept if str(r["code_version"]).startswith(a.prefer_code_version)]
    # paired differences at the CLUSTER the endpoint rests on: the task
    by = defaultdict(dict)
    for r in kept:
        by[(r["task_id"], r["seed"])][r["arm_id"]] = r
    t_arm, c_arm = a.treatment_arm, a.control_arm
    per_task = defaultdict(list)
    for (task, _), v in by.items():
        if t_arm in v and c_arm in v:
            per_task[task].append(float(v[t_arm][a.endpoint]) - float(v[c_arm][a.endpoint]))
    diffs = {k: statistics.fmean(v) for k, v in per_task.items()}
    mde = minimum_detectable_effect(diffs)
    return {
        "suite_id": a.suite,
        "endpoint": a.endpoint,
        "arms": sorted(arms),
        "eligibility_predicate": "pi_eval.report.ELIGIBLE, re-expressed in load_eligible()",
        "selection": sel,
        "population_after_selection": {
            "n_rows": len(kept),
            "per_arm": dict(sorted(per_arm.items())),
            "n_tasks": len(set(r["task_id"] for r in kept)),
            "code_versions": dict(
                sorted(Counter(str(r["code_version"])[:8] for r in kept).items())
            ),
            "balanced": len(set(per_arm.values())) == 1,
        },
        "population_if_restricted_to_one_code_version": {
            "rule": f"drop every row whose code_version is not {a.prefer_code_version}",
            "n_rows": len(restricted),
            "per_arm": dict(sorted(Counter(r["arm_id"] for r in restricted).items())),
            "n_tasks": len(set(r["task_id"] for r in restricted)),
            "cells_deleted_outright": len(kept) - len(restricted),
            "why_this_is_the_worse_rule": (
                "it deletes whole cells rather than choosing within them, and it deletes them "
                "unevenly across arms; the arm imbalance IS the deletion"
            ),
        },
        "clusters": {
            "unit": "task_id",
            "n_clusters": len(diffs),
            "n_rows_behind_them": len(kept),
            "n_task_seed_cells_paired": sum(1 for v in by.values() if t_arm in v and c_arm in v),
            "why": (
                "replicate seeds sharpen a task's own mean; they do not add tasks. The "
                "confirmatory n is the task count."
            ),
        },
        "mde": mde,
        "contrast": f"{t_arm} - {c_arm} on {a.endpoint}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--treatment", default="inquirer_prompted")
    ap.add_argument("--control", default="self_ask")
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--out")
    ap.add_argument(
        "--clusters",
        action="store_true",
        help="also recompute the contrast at fork-point, trace and task resolution",
    )
    ap.add_argument(
        "--cells",
        action="store_true",
        help="duplicate-cell selection record + MDE over runs.parquet (not the fork protocol)",
    )
    ap.add_argument("--parquet", default=str(ROOT / "scores" / "parquet" / "runs.parquet"))
    ap.add_argument("--prefer-code-version", default="5800f8db")
    ap.add_argument("--endpoint", default="n_user_turns")
    ap.add_argument("--arm", action="append")
    ap.add_argument("--treatment-arm", default="inquirer_prompted")
    ap.add_argument("--control-arm", default="drafter_only")
    a = ap.parse_args(argv)
    if a.cells:
        rep = cells_report(a)
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
            print(f"  wrote {a.out}")
        return 0
    runs = load_runs(
        Path(a.runs_root),
        suite=a.suite,
        split=a.split,
        arms={a.treatment, a.control},
        seeds=set(a.seed) if a.seed else None,
    )
    rep = paired_engagement(runs, treatment=a.treatment, control=a.control)
    rep["suite_id"], rep["split"] = a.suite, a.split
    print(
        f"{a.suite} {a.split}: {a.treatment} vs {a.control}  pairs {rep['n_pairs']}  unpaired runs {rep['n_unpaired']}"
    )
    for seed, s in rep["per_seed"].items():
        print(
            f"  seed {seed}: n={s['n_pairs']}  follow-ups {s['follow_ups']['treatment']:.2f} vs {s['follow_ups']['control']:.2f}"
            f"  diff {s['diff_mean']:+.2f}  fewer/more {s['fewer']}/{s['more']}  p={s['sign_test_p']:.2g}"
            f"  reward {s['reward']['treatment']:.2f} vs {s['reward']['control']:.2f}"
        )
    p = rep["pooled"]
    print(
        f"  pooled: n={p['n_pairs']}  diff {p['diff_mean']:+.2f}  fewer/more {p['fewer']}/{p['more']}  p={p['sign_test_p']:.2g}  reward {p['reward']['treatment']:.2f} vs {p['reward']['control']:.2f}"
    )
    print(
        f"  provenance: code_versions {rep['provenance']['code_versions']}  runs {rep['provenance']['n_runs']}  run_ids_sha {rep['provenance']['run_ids_sha']}"
    )
    if a.clusters:
        cl = clustered_engagement(runs, treatment=a.treatment, control=a.control)
        rep["clustered"] = cl
        pv = cl["provenance"]
        print(
            f"  prefix field present on every run: {pv['prefix_field_present']}   "
            f"arms share the prefix at every fork point: "
            f"{pv['arms_share_the_prefix_at_every_fork_point']}"
        )
        print(
            "  clustering unit                                 n   diff   fewer/more/ties"
            "   exact-sign p   sign-flip p   BCa 95% CI"
        )
        for name, lv in cl["levels"].items():
            e = lv["estimate"]
            flag = (
                "  <- AS PUBLISHED"
                if name == cl["published_unit"]
                else ("  <- RECOMPUTED" if name == cl["recomputed_unit"] else "")
            )
            print(
                f"  {name:11s} {lv['unit'][:34]:34s} {lv['n_units']:4d} {lv['diff_mean']:+6.2f}"
                f"   {lv['fewer']:3d}/{lv['more']:3d}/{lv['ties']:3d}"
                f"   {lv['sign_test_p']:11.3g}   {e['sign_flip_p']:11.3g}"
                f"   [{e['ci_lo']:+.2f}, {e['ci_hi']:+.2f}]{flag}"
            )
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
