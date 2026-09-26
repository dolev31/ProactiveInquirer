"""Lane L1.10 driver: does the evidence advantage reach the answer when the Answerer cannot
compensate from its own knowledge.

For every run in the held-out population (`artifacts/testsplit_qa`, trained vs prompted,
musique/strategyqa/wiki2, 2,282 runs), rebuilds the FINAL recorded evidence set and re-answers
it under three Answerer pins (gpt-oss-120b -- the recorded pin, as a reproduction control --
qwen3-8b-base, granite33-8b-base), plus a closed-book (empty-evidence) call per pin per run.
Scores every answer with `pi_eval.metrics.quality`, against the same gold graphs the harness
itself scores against. Writes counterfactual ROWS, never runs, under `--out-dir`.

Usage:
    python -m scripts.answerer_strength.run \\
        --population-dir /path/to/artifacts/testsplit_qa \\
        --runs-root /path/to/runs \\
        --corpora-root /path/to/data/corpora \\
        --matched-cost-parquet-dir /path/to/artifacts/testsplit_qa/scores_parquet \\
        --env-file /path/to/.env \\
        --cache-root /path/to/cache \\
        --out-dir /path/to/artifacts/answerer_strength_20260918
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from scripts.answerer_strength import lib

GRAPH_VERSION = "v1"
CONDITIONS = ("counterfactual", "closed_book")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-dir", required=True, type=Path)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--corpora-root", required=True, type=Path)
    p.add_argument("--matched-cost-parquet-dir", required=True, type=Path)
    p.add_argument("--env-file", required=True, type=Path)
    p.add_argument(
        "--gold-root",
        required=True,
        type=Path,
        help="Overrides .env's PI_GOLD_ROOT (blank there): the operator states it explicitly.",
    )
    p.add_argument("--cache-root", required=True, type=Path)
    p.add_argument("--price-table", default=None, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:4000",
        help="The LOCAL proxy, not .env's LITELLM_BASE_URL (that is the team gateway itself).",
    )
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pins", nargs="+", default=list(lib.PIN_IDS), choices=list(lib.PIN_IDS))
    p.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Pilot mode: cap each (arm, suite) run_id list to the first N ids.",
    )
    p.add_argument("--skip-closed-book", action="store_true")
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- population assembly


def _load_records(
    population: dict[str, dict[str, list[str]]], runs_root: Path, limit_per_group: int | None
) -> tuple[list[lib.RunRecord], list[dict[str, str]]]:
    records: list[lib.RunRecord] = []
    load_failures: list[dict[str, str]] = []
    for arm in lib.ARMS:
        for suite in lib.SUITES:
            ids = population[arm][suite]
            if limit_per_group is not None:
                ids = ids[:limit_per_group]
            for run_id in ids:
                try:
                    records.append(lib.load_run_record(runs_root, run_id, arm=arm))
                except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                    load_failures.append(
                        {"run_id": run_id, "arm": arm, "suite": suite, "error": repr(exc)}
                    )
    return records, load_failures


def _prebuild_views(
    records: list[lib.RunRecord], corpora_root: Path
) -> tuple[dict[str, tuple[Any, Any]], list[dict[str, str]]]:
    """{run_id: (view, evidence)} for every record whose reconstruction succeeds; the rest are
    reported as reconstruction failures, never silently dropped."""
    ok: dict[str, tuple[Any, Any]] = {}
    failures: list[dict[str, str]] = []
    for rec in records:
        try:
            ok[rec.run_id] = lib.build_view_and_evidence(rec, corpora_root=corpora_root)
        except lib.EvidenceReconstructionError as exc:
            failures.append({"run_id": rec.run_id, "error": str(exc)})
    return ok, failures


def _fill_missing_drafts(
    records: list[lib.RunRecord], views: dict[str, tuple[Any, Any]], gw: lib.Gateway
) -> dict[str, str]:
    """0-ask runs never persisted `draft_text` (loop.py:the final draft is computed AFTER the
    turn loop, unconditionally, but only WRITTEN to disk per-turn). Recomputed once, through the
    frozen Drafter pin, shared across all three Answerer pins for that run."""
    zero_turn = [r for r in records if r.n_turns == 0 and r.run_id in views]
    out: dict[str, str] = {}
    if not zero_turn:
        return out
    _log(
        f"recovering final draft text for {len(zero_turn)} zero-ask run(s) via the frozen Drafter pin"
    )
    for rec in zero_turn:
        view, ev = views[rec.run_id]
        draft, _meta = lib.draft_call("gpt_oss_120b", view, ev, rec.seed, gw)
        out[rec.run_id] = draft.text
    return out


# --------------------------------------------------------------------------- the call matrix


def _row_base(rec: lib.RunRecord, pin: str, condition: str) -> dict[str, Any]:
    return {
        "run_id": rec.run_id,
        "arm": rec.arm,
        "suite": rec.suite_id,
        "task_id": rec.task_id,
        "seed": rec.seed,
        "pin": pin,
        "condition": condition,
    }


def _do_one(
    rec: lib.RunRecord,
    pin: str,
    condition: str,
    view: Any,
    ev: Any,
    draft_text: str,
    gw: lib.Gateway,
) -> dict[str, Any]:
    from pinq.types import Draft, Evidence

    row = _row_base(rec, pin, condition)
    try:
        if condition == "counterfactual":
            draft = Draft(text=draft_text) if draft_text else None
            ans, meta = lib.answer_call(pin, view, ev, draft, rec.seed, gw)
            row["evidence_subset_hash"] = ev.subset_hash
            row["n_evidence_units"] = len(ev.units)
        else:
            ans, meta = lib.answer_call(pin, view, Evidence(), None, rec.seed, gw)
            row["evidence_subset_hash"] = Evidence().subset_hash
            row["n_evidence_units"] = 0
        row["answer_text"] = ans.text
        row["n_words"] = ans.n_words
        row.update(meta)
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- a failed call is a row, not a crashed sweep
        row["ok"] = False
        row["error"] = repr(exc)
        row["answer_text"] = ""
    return row


def run_matrix(
    records: list[lib.RunRecord],
    views: dict[str, tuple[Any, Any]],
    missing_drafts: dict[str, str],
    gw: lib.Gateway,
    *,
    pins: list[str],
    max_workers: int,
    include_closed_book: bool,
    rows_out: Path,
) -> dict[str, int]:
    tasks = []
    for rec in records:
        if rec.run_id not in views:
            continue
        view, ev = views[rec.run_id]
        draft_text = rec.draft_text or missing_drafts.get(rec.run_id, "")
        for pin in pins:
            tasks.append((rec, pin, "counterfactual", view, ev, draft_text))
            if include_closed_book:
                tasks.append((rec, pin, "closed_book", view, ev, draft_text))

    _log(f"submitting {len(tasks)} answerer calls across {max_workers} workers")
    counts = {"ok": 0, "error": 0, "cache_hit": 0, "used_fallback": 0}
    lock = Lock()
    n_done = 0
    t0 = time.time()
    with rows_out.open("w") as fh, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_do_one, rec, pin, cond, view, ev, draft_text, gw): (rec.run_id, pin, cond)
            for rec, pin, cond, view, ev, draft_text in tasks
        }
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row) + "\n")
                n_done += 1
                counts["ok" if row.get("ok") else "error"] += 1
                if row.get("cache_hit"):
                    counts["cache_hit"] += 1
                if row.get("used_fallback"):
                    counts["used_fallback"] += 1
                if n_done % 200 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t0
                    _log(
                        f"{n_done}/{len(tasks)} done "
                        f"(ok={counts['ok']} error={counts['error']} "
                        f"cache_hit={counts['cache_hit']} fallback={counts['used_fallback']}, "
                        f"{elapsed:.0f}s elapsed)"
                    )
    return counts


# --------------------------------------------------------------------------- scoring + tables


def score_rows(rows_path: Path, scored_out: Path) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    graphs_cache: dict[str, Any] = {}
    n_no_graph = 0
    with rows_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if not row.get("ok"):
                scored.append(row)
                continue
            suite = row["suite"]
            if suite not in graphs_cache:
                graphs_cache[suite] = lib.load_gold_graphs(suite, GRAPH_VERSION)
            graph = graphs_cache[suite].get(row["task_id"])
            if graph is None:
                row["ok"] = False
                row["error"] = f"no gold graph for {suite}/{row['task_id']}"
                n_no_graph += 1
                scored.append(row)
                continue
            row.update(lib.score_answer(row["answer_text"], graph))
            scored.append(row)
    with scored_out.open("w") as fh:
        for row in scored:
            fh.write(json.dumps(row) + "\n")
    if n_no_graph:
        _log(f"WARNING: {n_no_graph} row(s) had no gold graph and were marked not-ok")
    return scored


def reproduction_control(scored: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in scored if r["pin"] == "gpt_oss_120b" and r["condition"] == "counterfactual"]
    ok_rows = [r for r in rows if r.get("ok")]
    n = len(rows)
    n_ok = len(ok_rows)
    exact = [r for r in ok_rows if r["answer_text"] == r.get("recorded_answer_text")]
    return {
        "n_rows": n,
        "n_ok": n_ok,
        "n_errors": n - n_ok,
        "n_cache_hit": sum(1 for r in ok_rows if r.get("cache_hit")),
        "n_used_fallback": sum(1 for r in ok_rows if r.get("used_fallback")),
        "n_exact_match": len(exact),
        "exact_match_rate_over_ok": (len(exact) / n_ok) if n_ok else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    env = lib.load_env_file(args.env_file)
    for k, v in env.items():
        os.environ.setdefault(k, v)
    # PI_GOLD_ROOT is blank in .env (an operator states it explicitly, never a rollout-worker
    # default) and LITELLM_BASE_URL in .env is the team GATEWAY itself, not this local proxy
    # (see artifacts' PROXY.md) -- both are overridden here from CLI args, never taken as-is.
    os.environ["PI_GOLD_ROOT"] = str(args.gold_root)
    base_url = args.base_url
    api_key = os.environ.get("LITELLM_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("--base-url / LITELLM_API_KEY not set (checked CLI arg and .env)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    price_table = lib.PriceTable.load(args.price_table, strict=True)
    gw = lib.Gateway(
        base_url=base_url,
        api_key=api_key,
        cache_root=str(args.cache_root),
        price_table=price_table,
        env=dict(os.environ),
    )

    _log(f"population dir: {args.population_dir}")
    population = lib.read_population(args.population_dir)
    n_pop = sum(len(ids) for arm in population.values() for ids in arm.values())
    _log(f"population: {n_pop} run_ids across {len(lib.ARMS)} arms x {len(lib.SUITES)} suites")

    records, load_failures = _load_records(population, args.runs_root, args.limit_per_group)
    _log(f"loaded {len(records)} run records ({len(load_failures)} load failures)")

    views, recon_failures = _prebuild_views(records, args.corpora_root)
    _log(
        f"reconstructed {len(views)} (view, evidence) pairs ({len(recon_failures)} reconstruction failures)"
    )
    if recon_failures:
        for f in recon_failures[:5]:
            _log(f"  RECON FAIL {f['run_id']}: {f['error']}")

    missing_drafts = _fill_missing_drafts(records, views, gw)

    rows_path = args.out_dir / "rows.raw.jsonl"
    counts = run_matrix(
        records,
        views,
        missing_drafts,
        gw,
        pins=args.pins,
        max_workers=args.max_workers,
        include_closed_book=not args.skip_closed_book,
        rows_out=rows_path,
    )
    _log(f"call matrix done: {counts}")

    scored_path = args.out_dir / "rows.jsonl"
    # recorded_answer_text is needed by reproduction_control; stitch it back in from records
    recorded_by_run = {r.run_id: r.recorded_answer_text for r in records}
    tmp_rows = []
    with rows_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            row["recorded_answer_text"] = recorded_by_run.get(row["run_id"], "")
            tmp_rows.append(row)
    with rows_path.open("w") as fh:
        for row in tmp_rows:
            fh.write(json.dumps(row) + "\n")

    scored = score_rows(rows_path, scored_path)
    _log(f"scored {len(scored)} rows -> {scored_path}")

    repro = reproduction_control(scored)
    _log(f"reproduction control: {repro}")

    summary = {
        "graph_version": GRAPH_VERSION,
        "population_n": n_pop,
        "n_records_loaded": len(records),
        "load_failures": load_failures,
        "n_reconstructed": len(views),
        "reconstruction_failures": recon_failures,
        "call_counts": counts,
        "reproduction_control": repro,
        "spend": spend_summary(scored),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"wrote {args.out_dir / 'summary.json'}")
    return 0


def spend_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_pin: dict[str, dict[str, float]] = {}
    for r in scored:
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
    total_usd = sum(d["usd"] for d in by_pin.values())
    return {"by_pin": by_pin, "total_usd": total_usd}


if __name__ == "__main__":
    sys.exit(main())
