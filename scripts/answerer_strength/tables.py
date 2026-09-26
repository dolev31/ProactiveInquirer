"""Lane L1.10, stage 2: turn scored counterfactual rows into the contrast tables.

Separate from `run.py` so the statistics (BCa resample counts, the coverage-linkage split) can
be re-run without re-issuing a single LLM call: everything here reads `rows.jsonl` (already
scored by `run.py`) and the isolated `artifacts/testsplit_qa/scores_parquet` store.

Usage:
    python -m scripts.answerer_strength.tables \\
        --rows /path/to/artifacts/answerer_strength_20260918/rows.jsonl \\
        --matched-cost-parquet-dir /path/to/artifacts/testsplit_qa/scores_parquet \\
        --scorer-hash aa18b1fefd3b9b50d3d4922b86b00b4328d9a3a28b594a5ab1d0cc5b930e0ba7 \\
        --out /path/to/artifacts/answerer_strength_20260918/result.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from scripts.answerer_strength import lib

METRICS = ("answer_token_f1", "answer_token_recall", "contains_answer")
PIN_ORDER = ("gpt_oss_120b", "qwen3_8b_base", "granite33_8b_base")  # strongest -> weakest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--matched-cost-parquet-dir", required=True, type=Path)
    p.add_argument("--scorer-hash", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ok(rows: list[dict[str, Any]], **filt: Any) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not r.get("ok"):
            continue
        if all(r.get(k) == v for k, v in filt.items()):
            out.append(r)
    return out


def main_contrast_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per suite x pin x metric: trained-minus-prompted, paired on (task_id, seed)."""
    out: dict[str, Any] = {}
    for suite in lib.SUITES:
        out[suite] = {}
        for pin in PIN_ORDER:
            out[suite][pin] = {}
            trained_rows = _ok(
                rows, suite=suite, pin=pin, condition="counterfactual", arm="trained"
            )
            prompted_rows = _ok(
                rows, suite=suite, pin=pin, condition="counterfactual", arm="prompted"
            )
            for metric in METRICS:
                tmap = {
                    lib._key(r["task_id"], r["seed"]): r[metric]
                    for r in trained_rows
                    if metric in r
                }
                pmap = {
                    lib._key(r["task_id"], r["seed"]): r[metric]
                    for r in prompted_rows
                    if metric in r
                }
                out[suite][pin][metric] = lib.paired_contrast(tmap, pmap)
    return out


def closed_book_floor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per suite x pin x metric: one-sample level over de-duplicated (task_id, seed) closed-book
    answers. `condition == closed_book` values are, by construction, identical for a given
    (suite, task_id, seed, pin) regardless of which arm's row produced them (the request bytes
    do not carry the arm), so the arm that happens to answer first is kept and the other
    dropped -- never double counted."""
    out: dict[str, Any] = {}
    for suite in lib.SUITES:
        out[suite] = {}
        for pin in PIN_ORDER:
            cb_rows = _ok(rows, suite=suite, pin=pin, condition="closed_book")
            dedup: dict[str, dict[str, Any]] = {}
            for r in cb_rows:
                dedup.setdefault(lib._key(r["task_id"], r["seed"]), r)
            out[suite][pin] = {}
            for metric in METRICS:
                vals = [r[metric] for r in dedup.values() if metric in r]
                out[suite][pin][metric] = lib.one_sample_level(vals)
    return out


def _task_level_recall(rows: list[dict[str, Any]], *, metric: str) -> dict[str, float]:
    by_task: dict[str, list[float]] = {}
    for r in rows:
        if metric in r:
            by_task.setdefault(r["task_id"], []).append(r[metric])
    return {t: statistics.fmean(v) for t, v in by_task.items()}


def coverage_linkage(
    rows: list[dict[str, Any]], *, matched_cost_parquet_dir: Path, scorer_hash: str
) -> dict[str, Any]:
    """Within each arm, per suite per pin: recall(HIGH) - recall(LOW), where HIGH/LOW is the
    trained arm's own matched-cost coverage margin over its prompted pair (identical split for
    every pin, since coverage does not depend on which Answerer re-read the evidence)."""
    from pinq_train.gate import _con

    con = _con(matched_cost_parquet_dir)

    ck_runs_by_suite: dict[str, list[dict]] = {s: [] for s in lib.SUITES}
    ba_runs_by_suite: dict[str, list[dict]] = {s: [] for s in lib.SUITES}
    seen: set[str] = set()
    for r in rows:
        if r["run_id"] in seen or r["condition"] != "counterfactual":
            continue
        seen.add(r["run_id"])
        d = {
            "run_id": r["run_id"],
            "suite_id": r["suite"],
            "task_id": r["task_id"],
            "seed": r["seed"],
        }
        (ck_runs_by_suite if r["arm"] == "trained" else ba_runs_by_suite)[r["suite"]].append(d)

    out: dict[str, Any] = {}
    for suite in lib.SUITES:
        by_task = lib.matched_cost_coverage_by_task(
            con,
            ck_runs=ck_runs_by_suite[suite],
            ba_runs=ba_runs_by_suite[suite],
            scorer_hash=scorer_hash,
        )
        high, low = lib.high_low_split(by_task, suite=suite)
        out[suite] = {"n_high_tasks": len(high), "n_low_tasks": len(low), "by_pin": {}}
        for pin in PIN_ORDER:
            out[suite]["by_pin"][pin] = {}
            for arm in lib.ARMS:
                arm_rows = _ok(rows, suite=suite, pin=pin, condition="counterfactual", arm=arm)
                task_recall = _task_level_recall(arm_rows, metric="answer_token_recall")
                group_high = [v for t, v in task_recall.items() if t in high]
                group_low = [v for t, v in task_recall.items() if t in low]
                out[suite]["by_pin"][pin][arm] = lib.two_sample_contrast(group_high, group_low)
    return out


def interaction_summary(main_tables: dict[str, Any]) -> dict[str, Any]:
    """Per suite x metric, the trained-minus-prompted point estimate and zero-exclusion by pin,
    in strongest-to-weakest pin order -- the raw numbers the interaction verdict is read off,
    not a hardcoded pass/fail."""
    out: dict[str, Any] = {}
    for suite in lib.SUITES:
        out[suite] = {}
        for metric in METRICS:
            cells = [main_tables[suite][pin][metric] for pin in PIN_ORDER]
            points = [c["point"] for c in cells]
            excludes_zero = [c["verdict"] == "excludes_zero" for c in cells]
            monotone_increasing = all(b >= a - 1e-12 for a, b in zip(points, points[1:]))
            out[suite][metric] = {
                "points_by_pin": dict(zip(PIN_ORDER, points)),
                "excludes_zero_by_pin": dict(zip(PIN_ORDER, excludes_zero)),
                "monotone_nondecreasing_strong_to_weak": monotone_increasing,
            }
    return out


def spend_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pin: dict[str, dict[str, float]] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        pin = r["pin"]
        d = by_pin.setdefault(
            pin,
            {
                "usd": 0.0,
                "tok_prompt": 0,
                "tok_completion": 0,
                "tok_reasoning": 0,
                "n_calls": 0,
                "n_cache_hit": 0,
            },
        )
        d["usd"] += float(r.get("usd") or 0.0)
        d["tok_prompt"] += int(r.get("tok_prompt") or 0)
        d["tok_completion"] += int(r.get("tok_completion") or 0)
        d["tok_reasoning"] += int(r.get("tok_reasoning") or 0)
        d["n_calls"] += 1
        d["n_cache_hit"] += 1 if r.get("cache_hit") else 0
    return {"by_pin": by_pin, "total_usd": sum(d["usd"] for d in by_pin.values())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.rows)
    main_tables = main_contrast_tables(rows)
    result = {
        "scorer_hash_of_source_population": args.scorer_hash,
        "graph_version": "v1",
        "main_contrast": main_tables,
        "closed_book_floor": closed_book_floor(rows),
        "coverage_linkage": coverage_linkage(
            rows,
            matched_cost_parquet_dir=args.matched_cost_parquet_dir,
            scorer_hash=args.scorer_hash,
        ),
        "interaction_summary": interaction_summary(main_tables),
        "spend": spend_from_rows(rows),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
