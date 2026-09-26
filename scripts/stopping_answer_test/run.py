"""Lane L1.5 driver: stop 2x2, answer quality, and coverage-to-answer linkage on the held-out
test split, trained vs prompted.

Reads ONLY `<population-dir>/scores_parquet` (an isolated snapshot -- see its own
`provenance.{trained,prompted}.json`) and `<plan-metrics-dir>/contrasts.json`. Writes nothing to
either. All paths are CLI arguments with no default: `scripts/check_no_home_paths.sh` bans a
committed absolute home path, and the population lives in an untracked, main-checkout-only
directory a worktree does not carry (see `tests/test_stopping_answer_test.py`'s module
docstring), so the caller must always pass it explicitly.

Usage:
    python -m scripts.stopping_answer_test.run \\
        --population-dir /path/to/artifacts/testsplit_qa \\
        --plan-metrics-dir /path/to/artifacts/testsplit_plan_metrics_20260918 \\
        --runs-root /path/to/runs \\
        --out /path/to/scratch/result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from scripts.stopping_answer_test import lib

GRID_NAME = "tier1_trained_qa_base"
ARM_TRAINED = "inquirer_trained"
ARM_PROMPTED = "inquirer_prompted"
MODEL_TRAINED = "qwen3-8b-dpo-stacked-notdone-both"
MODEL_PROMPTED = "qwen3-8b-base"
SUITES = ("musique", "strategyqa", "wiki2")


def _load_provenance(
    population_dir: Path,
    *,
    model_trained: str = MODEL_TRAINED,
    model_prompted: str = MODEL_PROMPTED,
) -> dict[str, Any]:
    """The arm models are ARGUMENTS with the old constants as defaults.

    They were module constants, so this reader could only ever read one checkpoint pair and any
    finding it produced was silently scoped to that pair. Lane L6.1 serves
    `qwen3-8b-sft-headline-answernode` against `-refresh`, and could not be read at all. The
    defaults are unchanged, so every existing caller keeps its behaviour including the refusal
    when a provenance file names an unexpected model.
    """
    trained = json.loads((population_dir / "provenance.trained.json").read_text())
    prompted = json.loads((population_dir / "provenance.prompted.json").read_text())
    t_hash, p_hash = trained["scorer_hash"], prompted["scorer_hash"]
    t_gv, p_gv = trained["graph_version"], prompted["graph_version"]
    if t_hash != p_hash:
        raise SystemExit(f"scorer_hash disagrees between arms: {t_hash!r} vs {p_hash!r}")
    if t_gv != p_gv:
        raise SystemExit(f"graph_version disagrees between arms: {t_gv!r} vs {p_gv!r}")
    if trained["inquirer_model_id"] != model_trained:
        raise SystemExit(
            f"provenance.trained.json names {trained['inquirer_model_id']!r}, "
            f"not the expected {model_trained!r}"
        )
    if prompted["inquirer_model_id"] != model_prompted:
        raise SystemExit(
            f"provenance.prompted.json names {prompted['inquirer_model_id']!r}, "
            f"not the expected {model_prompted!r}"
        )
    return {
        "scorer_hash": t_hash,
        "graph_version": t_gv,
        "code_version": trained["code_version"],
        "trained_model_pin_hash": trained["model_pin_hash"],
        "prompted_model_pin_hash": prompted["model_pin_hash"],
    }


def _read_run_id_files(population_dir: Path, arm: str) -> dict[str, list[str]]:
    out = {}
    for suite in SUITES:
        p = population_dir / f"run_ids.{arm}.{suite}.txt"
        out[suite] = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return out


def step0_verify_population(
    con, population_dir: Path, runs_root: Path, scorer_hash: str
) -> dict[str, Any]:
    trained_ids = _read_run_id_files(population_dir, "trained")
    prompted_ids = _read_run_id_files(population_dir, "prompted")
    all_ids = sorted(
        {rid for ids in [*trained_ids.values(), *prompted_ids.values()] for rid in ids}
    )

    presence = lib.population_report(con, run_ids=all_ids, scorer_hash=scorer_hash)

    from pinq_train.gate import _in, _rows  # noqa: PLC0415 -- kept local, verification-only path

    n_turns_rows = _rows(con, f"SELECT run_id, n_turns FROM runs WHERE run_id IN {_in(all_ids)}")
    n_turns_by_run = {r["run_id"]: r["n_turns"] for r in n_turns_rows}
    turns_check = lib.verify_turns_not_dropped(runs_root, n_turns_by_run)

    trained_runs = lib.load_arm_runs(
        con, arm_id=ARM_TRAINED, model_id=MODEL_TRAINED, grid_name=GRID_NAME
    )
    prompted_runs = lib.load_arm_runs(
        con, arm_id=ARM_PROMPTED, model_id=MODEL_PROMPTED, grid_name=GRID_NAME
    )
    ladder_check = lib.verify_ladder_exists_for_asking_runs(
        con, [*trained_runs, *prompted_runs], scorer_hash=scorer_hash
    )

    return {
        "n_population": len(all_ids),
        "presence": presence,
        "turns_not_dropped": turns_check,
        "ladder_exists_for_asking_runs": ladder_check,
        "clean": (
            not presence["missing_from_runs"]
            and not presence["missing_scores"]
            and turns_check["n_bad"] == 0
            and ladder_check["n_bad"] == 0
        ),
    }


def _stability_from_pooled(
    per_task_trained: dict[str, dict],
    per_task_prompted: dict[str, dict],
    num_key: str,
    den_key: str,
) -> dict[str, Any]:
    tasks = sorted(set(per_task_trained) & set(per_task_prompted))
    rows = [
        (
            t,
            float(per_task_trained[t][num_key]),
            float(per_task_trained[t][den_key]),
            float(per_task_prompted[t][num_key]),
            float(per_task_prompted[t][den_key]),
        )
        for t in tasks
    ]

    def compute(n_boot: int, seed: int) -> dict[str, Any]:
        return lib.bca_paired_ratio_delta(rows, n_boot=n_boot, seed=seed)

    return lib.with_stability(compute, seed=0)


def step1_stop_2x2(con, scorer_hash: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suite in SUITES:
        trained_runs = lib.load_arm_runs(
            con, arm_id=ARM_TRAINED, model_id=MODEL_TRAINED, grid_name=GRID_NAME, suite=suite
        )
        prompted_runs = lib.load_arm_runs(
            con, arm_id=ARM_PROMPTED, model_id=MODEL_PROMPTED, grid_name=GRID_NAME, suite=suite
        )
        pooled_t = lib.pooled_stop_cells(con, trained_runs, scorer_hash=scorer_hash)
        pooled_p = lib.pooled_stop_cells(con, prompted_runs, scorer_hash=scorer_hash)
        per_task_t = lib.per_task_stop_cells(con, trained_runs, scorer_hash=scorer_hash)
        per_task_p = lib.per_task_stop_cells(con, prompted_runs, scorer_hash=scorer_hash)

        delta_stop_given_done = _stability_from_pooled(
            per_task_t, per_task_p, "n_done_stop", "n_done"
        )
        delta_ask_given_not_done = _stability_from_pooled(
            per_task_t, per_task_p, "n_not_done_ask", "n_not_done"
        )

        # cross-check: method A (sum of the scorer's own stop2x2_* columns) vs method B
        # (calling _stop_2x2 directly, which is what pooled_t/pooled_p above already are).
        # This reproduces the scorer's own numbers from the raw metric rows independently.
        cross_t = _sum_scored_stop_cells(con, [r["run_id"] for r in trained_runs], scorer_hash)
        cross_p = _sum_scored_stop_cells(con, [r["run_id"] for r in prompted_runs], scorer_hash)

        cap_t = _cap_share(con, [r["run_id"] for r in trained_runs])
        cap_p = _cap_share(con, [r["run_id"] for r in prompted_runs])
        overshoot_t = _mean_metric(
            con, [r["run_id"] for r in trained_runs], "stop_overshoot", scorer_hash
        )
        overshoot_p = _mean_metric(
            con, [r["run_id"] for r in prompted_runs], "stop_overshoot", scorer_hash
        )

        out[suite] = {
            "trained": pooled_t,
            "prompted": pooled_p,
            "cross_check_trained_matches_scored_columns": _cells_match(pooled_t, cross_t),
            "cross_check_prompted_matches_scored_columns": _cells_match(pooled_p, cross_p),
            "delta_p_stop_given_done": delta_stop_given_done,
            "delta_p_ask_given_not_done": delta_ask_given_not_done,
            "stop_overshoot_mean": {"trained": overshoot_t, "prompted": overshoot_p},
            "share_hitting_cap": {"trained": cap_t, "prompted": cap_p},
        }
    return out


def _cells_match(pooled: dict, scored_sum: dict) -> bool:
    keys = ("n_done", "n_done_stop", "n_not_done", "n_not_done_ask", "n_forced_stops")
    return all(pooled[k] == scored_sum[k] for k in keys)


def _sum_scored_stop_cells(con, run_ids: list[str], scorer_hash: str) -> dict[str, int]:
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    if not run_ids:
        return {
            "n_done": 0,
            "n_done_stop": 0,
            "n_not_done": 0,
            "n_not_done_ask": 0,
            "n_forced_stops": 0,
        }
    rows = _rows(
        con,
        "SELECT metric_name, sum(value) AS total FROM scores WHERE scorer_hash = "
        f"'{scorer_hash}' AND run_id IN {_in(run_ids)} AND metric_name IN "
        "('stop2x2_n_done','stop2x2_n_stop_at_done','stop2x2_n_not_done',"
        "'stop2x2_n_ask_at_not_done','stop2x2_n_forced_stops') GROUP BY 1",
    )
    by_name = {r["metric_name"]: int(r["total"]) for r in rows}
    n_done = by_name.get("stop2x2_n_done", 0)
    n_stop_at_done = by_name.get("stop2x2_n_stop_at_done", 0)
    n_not_done = by_name.get("stop2x2_n_not_done", 0)
    n_ask_at_not_done = by_name.get("stop2x2_n_ask_at_not_done", 0)
    return {
        "n_done": n_done,
        "n_done_stop": n_stop_at_done,
        "n_not_done": n_not_done,
        "n_not_done_ask": n_ask_at_not_done,
        "n_forced_stops": by_name.get("stop2x2_n_forced_stops", 0),
    }


def _cap_share(con, run_ids: list[str]) -> dict[str, Any]:
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    if not run_ids:
        return {"n": 0, "n_hit_cap": 0, "share": float("nan")}
    rows = _rows(
        con,
        f"SELECT stop_reason, count(*) AS n FROM runs WHERE run_id IN {_in(run_ids)} GROUP BY 1",
    )
    total = sum(r["n"] for r in rows)
    hit = sum(r["n"] for r in rows if r["stop_reason"] != "policy_stop")
    return {"n": total, "n_hit_cap": hit, "share": hit / total if total else float("nan")}


def _mean_metric(con, run_ids: list[str], metric: str, scorer_hash: str) -> float:
    from pinq_train.gate import _mean

    vals = lib._metric_by_run(con, metric, scorer_hash=scorer_hash)
    xs = [vals[r] for r in run_ids if r in vals]
    return _mean(xs)


def step2_answer_quality(con, scorer_hash: str, plan_metrics_dir: Path) -> dict[str, Any]:
    from pi_eval.stats.inference import paired_difference  # noqa: PLC0415

    out: dict[str, Any] = {}
    for suite in SUITES:
        trained_runs = lib.load_arm_runs(
            con, arm_id=ARM_TRAINED, model_id=MODEL_TRAINED, grid_name=GRID_NAME, suite=suite
        )
        prompted_runs = lib.load_arm_runs(
            con, arm_id=ARM_PROMPTED, model_id=MODEL_PROMPTED, grid_name=GRID_NAME, suite=suite
        )
        suite_out: dict[str, Any] = {}
        for metric in ("answer_token_f1", "answer_token_recall"):
            a = lib.task_level_metric(con, trained_runs, metric, scorer_hash=scorer_hash)
            b = lib.task_level_metric(con, prompted_runs, metric, scorer_hash=scorer_hash)

            def compute(n_boot: int, seed: int, a=a, b=b) -> dict[str, Any]:
                e = paired_difference(a, b, clusters=None, n_boot=n_boot, seed=seed)
                return {"point": e.point, "lo": e.ci_lo, "hi": e.ci_hi, "n": e.n}

            suite_out[metric] = lib.with_stability(compute, seed=0)
            suite_out[metric]["level_trained"] = lib._mean(list(a.values()))
            suite_out[metric]["level_prompted"] = lib._mean(list(b.values()))
        out[suite] = suite_out

    # answer_correct: presence + definition, no interval (not requested as a contrast; see
    # RESULT.md -- rule-based, not judge-derived, so "judge cost" for it is not applicable).
    all_run_ids = [
        r["run_id"]
        for suite in SUITES
        for r in lib.load_arm_runs(
            con, arm_id=ARM_TRAINED, model_id=MODEL_TRAINED, grid_name=GRID_NAME, suite=suite
        )
        + lib.load_arm_runs(
            con, arm_id=ARM_PROMPTED, model_id=MODEL_PROMPTED, grid_name=GRID_NAME, suite=suite
        )
    ]
    from pinq_train.gate import _in, _rows  # noqa: PLC0415

    ac = _rows(
        con,
        "SELECT count(*) AS n, count(distinct notes) AS n_notes FROM scores WHERE "
        f"metric_name = 'answer_correct' AND scorer_hash = '{scorer_hash}' "
        f"AND run_id IN {_in(all_run_ids)}",
    )[0]
    out["answer_correct_presence"] = {
        "n_population": len(all_run_ids),
        "n_scored": ac["n"],
        "definition": "quality.contains_answer(answer.text, gold_answer, gold_aliases) -- "
        "rule-based alias/substring containment, computed in pi_eval.score.score_run; not "
        "judge-derived (judgments.parquet has 0 rows in this store, scored with "
        "--allow-no-judge). No judge_pin exists for it because no judge produced it.",
    }

    out["dev_null_reference"] = {
        "answer_token_f1": {"delta": -0.0134, "lo": -0.0377, "hi": 0.0112},
        "answer_token_recall": {"delta": 0.0237, "lo": -0.0048, "hi": 0.0530},
        "suite": "frames (824 tasks, cap-8), trained minus same base",
        "source": "paper/sections/scaling_and_family.tex:606-613",
    }
    out["plan_metrics_cap8_cross_check"] = _cross_check_plan_metrics(plan_metrics_dir)
    return out


def _cross_check_plan_metrics(plan_metrics_dir: Path) -> dict[str, Any]:
    rows = json.loads((plan_metrics_dir / "contrasts.json").read_text())
    return {
        f"{r['metric']}/{r['suite']}": {"delta": r["task"]["delta"], "n": r["task"]["n"]}
        for r in rows
        if r["metric"] in ("answer_token_f1", "answer_token_recall") and r["basis"] == "unmatched"
    }


def step3_coverage_to_answer(con, scorer_hash: str, plan_metrics_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    verdicts_dir = plan_metrics_dir.parent / "testsplit_qa" / "verdicts"
    for suite in SUITES:
        trained_runs = lib.load_arm_runs(
            con, arm_id=ARM_TRAINED, model_id=MODEL_TRAINED, grid_name=GRID_NAME, suite=suite
        )
        prompted_runs = lib.load_arm_runs(
            con, arm_id=ARM_PROMPTED, model_id=MODEL_PROMPTED, grid_name=GRID_NAME, suite=suite
        )
        mc_by_task = lib.matched_cost_coverage_by_task(
            con, ck_runs=trained_runs, ba_runs=prompted_runs, scorer_hash=scorer_hash
        )
        recall_t = lib.task_level_metric(
            con, trained_runs, "answer_token_recall", scorer_hash=scorer_hash
        )
        recall_p = lib.task_level_metric(
            con, prompted_runs, "answer_token_recall", scorer_hash=scorer_hash
        )

        high = [k for k, v in mc_by_task.items() if v > 0]
        low = [k for k, v in mc_by_task.items() if v <= 0]

        def group(vals: dict, keys: list) -> list[float]:
            return [vals[k] for k in keys if k in vals]

        trained_high, trained_low = group(recall_t, high), group(recall_t, low)
        prompted_high, prompted_low = group(recall_p, high), group(recall_p, low)

        def compute_trained(n_boot: int, seed: int) -> dict[str, Any]:
            return lib.bca_two_sample_mean_delta(
                trained_high, trained_low, n_boot=n_boot, seed=seed
            )

        def compute_prompted(n_boot: int, seed: int) -> dict[str, Any]:
            return lib.bca_two_sample_mean_delta(
                prompted_high, prompted_low, n_boot=n_boot, seed=seed
            )

        out[suite] = {
            "n_tasks_matched_cost": len(mc_by_task),
            "n_high": len(high),
            "n_low": len(low),
            "trained_recall_high_minus_low": lib.with_stability(compute_trained, seed=0),
            "prompted_recall_high_minus_low": lib.with_stability(compute_prompted, seed=0),
            "pooled_mean_mc_delta": lib._mean(list(mc_by_task.values())),
        }
    published_musique = json.loads((verdicts_dir / "stacked-notdone.musique.json").read_text())
    out["_published_cross_check_note"] = (
        "matched_cost_coverage_by_task's pooled mean on musique cross-checked against "
        "verdicts/stacked-notdone.musique.json's evidence_coverage.delta in RESULT.md"
    )
    out["_published_musique_evidence_coverage_delta"] = _find_evidence_coverage_delta(
        published_musique
    )
    return out


def _find_evidence_coverage_delta(verdict: dict) -> float | None:
    crit = verdict.get("criteria", {}).get("evidence_coverage")
    if crit is None:
        return None
    return crit.get("value")


def main() -> int:
    global MODEL_TRAINED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population-dir", required=True, type=Path)
    ap.add_argument("--plan-metrics-dir", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--model-trained",
        default=MODEL_TRAINED,
        help="the trained arm's inquirer model_id, matched through calls.parquet. Added "
        "2026-09-23 so the same driver reads the selected recipe's other training seeds "
        "(-s1, -s2); the default is the value every earlier reading used.",
    )
    args = ap.parse_args()
    # Every step selects the trained arm by this module global; rebinding it here, before any
    # step runs, is the whole change. step0 checks provenance against the same value.
    MODEL_TRAINED = args.model_trained

    prov = _load_provenance(args.population_dir, model_trained=MODEL_TRAINED)
    store = args.population_dir / "scores_parquet"
    con = lib.gate_con(store)

    result: dict[str, Any] = {"provenance": prov}
    result["step0_population"] = step0_verify_population(
        con, args.population_dir, args.runs_root, prov["scorer_hash"]
    )
    if not result["step0_population"]["clean"]:
        print("POPULATION CHECK FAILED -- see step0_population in the output JSON", file=sys.stderr)

    result["step1_stop_2x2"] = step1_stop_2x2(con, prov["scorer_hash"])
    result["step2_answer_quality"] = step2_answer_quality(
        con, prov["scorer_hash"], args.plan_metrics_dir
    )
    result["step3_coverage_to_answer"] = step3_coverage_to_answer(
        con, prov["scorer_hash"], args.plan_metrics_dir
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"wrote {args.out}")
    print(f"population clean: {result['step0_population']['clean']}")
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"not JSON serialisable: {o!r}")


if __name__ == "__main__":
    raise SystemExit(main())
