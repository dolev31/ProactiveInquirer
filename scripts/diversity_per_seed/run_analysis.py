#!/usr/bin/env python3
"""Lane L1.1 runner: pull the real dev and test populations and recompute distinct-3 four ways.

Reuses the gate's own selection code (`pinq_train.gate._con`/`_select_runs`) so the run
populations here are constructed the same way `pi train gate` constructs them, not
re-derived. Dev populations are reconstructed from each dev verdict's own `selection` block
(the same replay pattern `scripts/bca_era_attribution.py` uses); test populations are the
explicit run-id list files under `artifacts/testsplit_qa/`, cross-checked against
`gate._select_runs` on the same store as a population-identity self-check.

Writes one JSON report. No number in it is asserted anywhere else -- `RESULT.md` quotes this
file's own output.

WHY `--main-checkout`, SEPARATE FROM THIS SCRIPT'S OWN REPO ROOT. This lane runs from an
isolated worktree, which has no working copy of `artifacts/testsplit_qa/`, `artifacts/gate/`
or `runs/` -- all three are untracked-by-construction (per-arm provenance and raw rollouts are
not committed) and therefore exist only in whichever checkout produced them. Source data is
read by absolute path from the main checkout; this script's own code still comes from wherever
it was invoked (`sys.path` below), which is the point of running it from a worktree at all.

Usage (main checkout's venv, absolute PI_CACHE_ROOT, any worktree):
    PYTHONPATH=<this checkout>/src <main checkout>/.venv/bin/python \\
        scripts/diversity_per_seed/run_analysis.py \\
        --main-checkout ~/PycharmProjects/ProactiveInquirer \\
        --out artifacts/diversity_per_seed_20260918/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_THIS_CHECKOUT = Path(__file__).resolve().parents[2]
if str(_THIS_CHECKOUT / "src") not in sys.path:
    sys.path.insert(0, str(_THIS_CHECKOUT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diversity_variants import (  # noqa: E402
    all_variants,
    cross_seed_determinism,
    resolve_temperature,
)

from pinq_train import gate  # noqa: E402

DEV_SUITES = ("musique", "strategyqa")
TEST_SUITES = ("musique", "strategyqa", "wiki2")


def _load_run_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _ask_rows(con, run_ids: list[str]) -> list[dict[str, Any]]:
    """(task_id, seed, turn_idx, question) for every ASK turn of these runs -- the same filter
    `pinq_train.gate._questions` uses, with `seed` and `turn_idx` kept."""
    if not run_ids:
        return []
    rows = gate._rows(
        con,
        "SELECT r.task_id, r.seed, t.turn_idx, t.question "
        "FROM turns t JOIN runs r ON r.run_id = t.run_id "
        f"WHERE t.run_id IN {gate._in(run_ids)} AND lower(t.action_kind) = 'ask' "
        "AND t.question IS NOT NULL AND t.question <> '' ORDER BY 1, 2, 3",
    )
    return [
        {
            "task_id": str(r["task_id"]),
            "seed": r["seed"],
            "turn_idx": int(r["turn_idx"]),
            "question": str(r["question"]),
        }
        for r in rows
    ]


def _dev_cell(con, verdict: dict, *, role: str) -> tuple[list[str], dict]:
    """`role` is "checkpoint" or "baseline". Replays `gate._select_runs` off the verdict's own
    `selection` block, exactly as `scripts/bca_era_attribution.py::replay_verdict` does."""
    sel = verdict["selection"]
    if role == "checkpoint":
        runs = gate._select_runs(
            con, arm=sel["checkpoint_arm"], grids=[sel["grid_name"]], model_id=None
        )
    else:
        runs = gate._select_runs(
            con,
            arm=sel["baseline_arm"],
            grids=list(sel["baseline_grid_names"]),
            model_id=sel.get("baseline_model_id"),
        )
    return [r["run_id"] for r in runs], sel


def build_dev(report: dict, *, main_checkout: Path) -> None:
    dev_verdicts = main_checkout / "artifacts" / "testsplit_qa" / "verdicts"
    for suite in DEV_SUITES:
        verdict = json.loads((dev_verdicts / f"dev.{suite}.json").read_text())
        gate_reported_value = verdict["criteria"]["distinct3"]["value"]
        gate_reported_baseline = verdict["criteria"]["distinct3"]["baseline"]
        gate_reported_n = verdict["criteria"]["distinct3"]["n"]

        # `selection.parquet_dir` is the verdict's own record of where it was computed from
        # (relative to the main checkout) -- read it rather than hand-typing the path, so a
        # verdict that ever points somewhere else is followed, not silently missed.
        con = gate._con(main_checkout / verdict["selection"]["parquet_dir"])
        ck_ids, ck_sel = _dev_cell(con, verdict, role="checkpoint")
        ba_ids, _ = _dev_cell(con, verdict, role="baseline")
        ck_rows = _ask_rows(con, ck_ids)
        ba_rows = _ask_rows(con, ba_ids)

        ck_summary = all_variants(ck_rows)
        ba_summary = all_variants(ba_rows)

        replay_ok = math.isclose(
            ck_summary["a_pooled_by_task"]["value"], gate_reported_value, rel_tol=0, abs_tol=1e-9
        )
        replay_baseline_ok = math.isclose(
            ba_summary["a_pooled_by_task"]["value"], gate_reported_baseline, rel_tol=0, abs_tol=1e-9
        )
        report["dev"][suite] = {
            "selection": ck_sel,
            "gate_verdict_reported": {
                "value": gate_reported_value,
                "baseline": gate_reported_baseline,
                "n": gate_reported_n,
            },
            "replay_reproduces_gate_value": replay_ok,
            "replay_reproduces_gate_baseline": replay_baseline_ok,
            "trained": ck_summary,
            "prompted": ba_summary,
        }
        if not (replay_ok and replay_baseline_ok):
            report.setdefault("_warnings", []).append(
                f"dev/{suite}: replay did NOT reproduce the gate's own recorded distinct3 "
                f"value/baseline -- see dev.{suite}.json vs this cell"
            )
        con.close()


def build_test(report: dict, *, main_checkout: Path) -> None:
    testsplit = main_checkout / "artifacts" / "testsplit_qa"
    dev_verdicts = testsplit / "verdicts"
    runs_root = main_checkout / "runs"
    con = gate._con(testsplit / "scores_parquet")

    # population-identity self-check: the explicit run-id-list files vs gate._select_runs
    # replayed off stacked-notdone.<suite>.json's own selection block (musique/strategyqa only
    # -- wiki2 has no dev verdict but does have a stacked-notdone verdict with a selection block).
    for suite in TEST_SUITES:
        trained_ids = _load_run_ids(testsplit / f"run_ids.trained.{suite}.txt")
        prompted_ids = _load_run_ids(testsplit / f"run_ids.prompted.{suite}.txt")

        verdict_path = dev_verdicts / f"stacked-notdone.{suite}.json"
        verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else None
        selection_check: dict[str, Any] = {"checked": False}
        if verdict is not None:
            sel = verdict["selection"]
            # `grid_name` is `tier1_trained_qa_base` for EVERY suite on test (unlike dev, which
            # has one grid per suite), so `_select_runs` alone pools all three suites' runs --
            # restrict to this suite's own `suite_id`, which `_select_runs` already returns.
            replay_ck = gate._select_runs(
                con, arm=sel["checkpoint_arm"], grids=[sel["grid_name"]], model_id=None
            )
            replay_ba = gate._select_runs(
                con,
                arm=sel["baseline_arm"],
                grids=list(sel["baseline_grid_names"]),
                model_id=sel.get("baseline_model_id"),
            )
            replay_ck_ids = {r["run_id"] for r in replay_ck if r["suite_id"] == suite}
            replay_ba_ids = {r["run_id"] for r in replay_ba if r["suite_id"] == suite}
            selection_check = {
                "checked": True,
                "trained_run_id_list_matches_select_runs": replay_ck_ids == set(trained_ids),
                "prompted_run_id_list_matches_select_runs": replay_ba_ids == set(prompted_ids),
                "n_trained_list": len(trained_ids),
                "n_trained_select_runs": len(replay_ck_ids),
                "n_prompted_list": len(prompted_ids),
                "n_prompted_select_runs": len(replay_ba_ids),
            }
            gate_reported_value = verdict["criteria"]["distinct3"]["value"]
            gate_reported_baseline = verdict["criteria"]["distinct3"]["baseline"]
            gate_reported_n = verdict["criteria"]["distinct3"]["n"]
        else:
            gate_reported_value = gate_reported_baseline = gate_reported_n = None

        ck_rows = _ask_rows(con, trained_ids)
        ba_rows = _ask_rows(con, prompted_ids)
        ck_summary = all_variants(ck_rows)
        ba_summary = all_variants(ba_rows)

        replay_ok = gate_reported_value is not None and math.isclose(
            ck_summary["a_pooled_by_task"]["value"], gate_reported_value, rel_tol=0, abs_tol=1e-9
        )
        replay_baseline_ok = gate_reported_baseline is not None and math.isclose(
            ba_summary["a_pooled_by_task"]["value"], gate_reported_baseline, rel_tol=0, abs_tol=1e-9
        )

        report["test"][suite] = {
            "population_identity_check": selection_check,
            "gate_verdict_reported": {
                "value": gate_reported_value,
                "baseline": gate_reported_baseline,
                "n": gate_reported_n,
            },
            "replay_reproduces_gate_value": replay_ok,
            "replay_reproduces_gate_baseline": replay_baseline_ok,
            "trained": ck_summary,
            "prompted": ba_summary,
            "trained_cross_seed_determinism": cross_seed_determinism(ck_rows),
            "prompted_cross_seed_determinism": cross_seed_determinism(ba_rows),
        }
        if not (replay_ok and replay_baseline_ok):
            report.setdefault("_warnings", []).append(
                f"test/{suite}: replay did NOT reproduce the gate's own recorded distinct3 "
                f"value/baseline -- see stacked-notdone.{suite}.json vs this cell"
            )
        if not selection_check.get("checked") or not (
            selection_check.get("trained_run_id_list_matches_select_runs")
            and selection_check.get("prompted_run_id_list_matches_select_runs")
        ):
            report.setdefault("_warnings", []).append(
                f"test/{suite}: run_id list file does not match gate._select_runs replay -- "
                "population identity is not self-consistent"
            )

    # sampling temperature: one manifest per arm, any suite (musique is present in both arms)
    trained_sample = _load_run_ids(testsplit / "run_ids.trained.musique.txt")[0]
    prompted_sample = _load_run_ids(testsplit / "run_ids.prompted.musique.txt")[0]
    for label, run_id in (("trained", trained_sample), ("prompted", prompted_sample)):
        manifest = json.loads((runs_root / run_id / "manifest.json").read_text())
        pin = manifest["pins"]["inquirer"]
        sha = pin["sampling_sha"]
        report["sampling_temperature"][label] = {
            "sample_run_id": run_id,
            "model_id": pin["model_id"],
            "sampling_sha": sha,
            "resolved_temperature": resolve_temperature(sha),
        }
    con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--main-checkout",
        type=Path,
        required=True,
        help="absolute path to the main checkout that holds artifacts/testsplit_qa, "
        "artifacts/gate and runs/ -- all untracked, so a worktree does not have them. "
        "Required rather than defaulted: a hardcoded home path in committed source is exactly "
        "what scripts/check_no_home_paths.sh exists to catch.",
    )
    args = ap.parse_args()

    report: dict[str, Any] = {"dev": {}, "test": {}, "sampling_temperature": {}}
    build_dev(report, main_checkout=args.main_checkout)
    build_test(report, main_checkout=args.main_checkout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")
    if report.get("_warnings"):
        print("WARNINGS:")
        for w in report["_warnings"]:
            print(" -", w)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
