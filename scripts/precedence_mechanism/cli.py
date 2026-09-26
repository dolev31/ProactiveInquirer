"""Orchestrates Lane L1.3: why the trained policy violates prerequisite order more than the
prompted one, on MuSiQue test at matched cost.

Run from the repository root, store and gold root absolute (see module docstrings in
`events.py` and `probe.py` for why). Writes everything under `--out-dir`
(`artifacts/precedence_mechanism_20260918/` by default) and never under `runs/`.

    PYTHONPATH=src PI_GOLD_ROOT=<abs data/gold> python -m scripts.precedence_mechanism.cli \\
        --store <abs artifacts/testsplit_qa/scores_parquet> --suite musique \\
        --out-dir artifacts/precedence_mechanism_20260918

PROBE POPULATION IS EVERY QUALIFYING EDGE'S PARENT, NOT ONLY VIOLATED ONES. An earlier pass
probed only the parents of violated edges, which makes "answerable" true only where a
violation already happened by construction -- selected on its own outcome, the class of bug
`docs/`'s own operating notes name "a quantity that cannot take the other value". Answerability
has to be measured on the full matched population (every parent of every `QualifyingEdge`,
violated or not) for the stratification to test anything. `probe_cache.json` is loaded first
and only the (task, node) x model cells still missing are called -- deterministic at
temperature 0, so nothing already answered is re-spent.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import duckdb

from pi_eval.canary import load as load_canaries
from pi_eval.gold import load_graphs

from .events import (
    ARM_PROMPTED,
    ARM_TRAINED,
    build_all_events,
    build_all_qualifying_edges,
    lock_check,
    run_arms,
    terminal_scores,
)
from .probe import (
    assert_canary_clean,
    call_model,
    closed_book_question,
    score_answer,
)
from .stratify import interaction_bca, logistic_violation_model, stratified_contrast

QWEN_MODEL = "qwen3-8b-base"
# NOT "openai/aws/gpt-oss-120b". The `litellm` SDK strips a leading "openai/" provider prefix
# before putting a model name on the wire (`.env`'s own comment on the frozen roles says so);
# `call_model` now POSTs to the proxy directly and does not strip anything, so the wire form
# is required here. MEASURED: sending the unstripped string 403s with "team not allowed to
# access model", listing "aws/gpt-oss-120b" (no prefix) as what the team may actually reach.
OSS_MODEL = "aws/gpt-oss-120b"
MODEL_LABELS = (("qwen3_8b_base", QWEN_MODEL), ("gpt_oss_120b", OSS_MODEL))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, help="absolute path to the scores_parquet dir")
    p.add_argument("--suite", default="musique")
    p.add_argument("--graph-version", default="v1")
    p.add_argument("--out-dir", default="artifacts/precedence_mechanism_20260918")
    # The LOCAL LiteLLM proxy, not `LITELLM_BASE_URL` (that .env var is the remote
    # gateway itself, which has never heard of "qwen3-8b-base" -- only the local proxy's own
    # `conf/serving/litellm.yaml` maps that served name to the vLLM deployment). MEASURED:
    # asking the remote gateway for "qwen3-8b-base" 403s with "team not allowed to access
    # model", listing only the gateway's own hosted ids.
    p.add_argument("--base-url", default="http://127.0.0.1:4000")
    p.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY", ""))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--canary-root", default=os.environ.get("MAIN", "."))
    return p.parse_args(argv)


def cache_key(task_id: str, node_id: str) -> str:
    return f"{task_id}::{node_id}"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    graphs = load_graphs(args.suite, args.graph_version)
    print(f"loaded {len(graphs)} {args.suite} graphs at {args.graph_version}")

    n_checked, n_mismatch, n_matched_events = lock_check(con, args.store, graphs, args.suite)
    print(
        f"lock_check: {n_checked} runs, {n_mismatch} mismatches, {n_matched_events} matched violations"
    )
    if n_mismatch:
        raise SystemExit(
            f"REFUSING to proceed: {n_mismatch}/{n_checked} runs disagree with the real metric"
        )

    events = build_all_events(con, args.store, graphs, args.suite)
    edges = build_all_qualifying_edges(con, args.store, graphs, args.suite)
    print(f"events (skip superset): {len(events)}; qualifying edges: {len(edges)}")

    # ---- probe population: EVERY qualifying edge's parent, both arms, viol and non-viol ----
    # KEYED (task_id, node_id), NEVER A BARE node_id: MuSiQue ids ("s1", "s2", ...) are local
    # to a task and reused across all 200 tasks.
    all_parent_keys = sorted({(e.task_id, e.parent_node_id) for e in edges})
    violated_keys = sorted({(e.task_id, e.parent_node_id) for e in edges if e.is_violation})
    print(
        f"distinct (task, parent_node) pairs: {len(all_parent_keys)} total, "
        f"{len(violated_keys)} appear in a violated edge (some arm)"
    )

    canaries = frozenset(load_canaries(root=Path(args.canary_root)))
    print(f"canary registry: {len(canaries)} nonces")

    cache_path = out_dir / "probe_cache.json"
    probe_rows: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"probe cache on disk: {len(probe_rows)} (task, node) pairs")

    node_of_id = {
        (g.gold_task_key, n.gold_node_id): n for g in graphs.values() for n in g.gold_nodes
    }
    for task_id, node_id in all_parent_keys:
        key = cache_key(task_id, node_id)
        if key in probe_rows:
            continue
        graph = graphs.get(task_id)
        node = node_of_id.get((task_id, node_id))
        if graph is None or node is None:
            continue
        question = closed_book_question(graph, node)
        if question is None:
            probe_rows[key] = {
                "task_id": task_id,
                "node_id": node_id,
                "question": None,
                "skipped": "unresolved_placeholder",
            }
            continue
        assert_canary_clean(question, canaries, where=f"probe question for {key}")
        probe_rows[key] = {
            "task_id": task_id,
            "node_id": node_id,
            "question": question,
            "gold": node.gold_aliases[0],
        }

    # ---- the network part, fanned out: independent, idempotent, temperature-0 calls --------
    # Only cells actually missing a SCORE are dispatched, so a cache already holding earlier
    # successes costs nothing again. A cell with a recorded error IS retried (not permanently
    # skipped): MEASURED, the 2026-09-18 run's 72 errors were concentrated in its first ~330
    # calls and zero after, consistent with the VPN outage the coordinator named, not with a
    # persistent per-cell failure -- so a stale timeout note must not block a clean rerun once
    # connectivity is back. A cell that fails again just gets its error note overwritten.
    import concurrent.futures as cf

    jobs = [
        (key, label, model)
        for key, row in probe_rows.items()
        if row.get("question") is not None
        for label, model in MODEL_LABELS
        if label not in row
    ]
    print(f"dispatching {len(jobs)} new probe calls ({args.concurrency} concurrent)...", flush=True)

    def _one(job: tuple[str, str, str]) -> tuple[str, str, dict | None, str | None]:
        key, label, model = job
        try:
            ans = call_model(
                base_url=args.base_url,
                api_key=args.api_key,
                model=model,
                question=probe_rows[key]["question"],
            )
            node = node_of_id[(probe_rows[key]["task_id"], probe_rows[key]["node_id"])]
            cell = {
                "answer": ans.answer_text,
                "score": score_answer(ans.answer_text, node),
                "tok_prompt": ans.tok_prompt,
                "tok_completion": ans.tok_completion,
            }
            return key, label, cell, None
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return key, label, None, f"{type(exc).__name__}: {exc}"

    # HARD BACKSTOP: a bounded-size wave, not one unbounded `pool.map`/`as_completed` over all
    # 524. MEASURED twice now: a call can sit well past `call_model`'s own `timeout=30` without
    # litellm's client-side timeout firing (a background run held at the same CPU-time for
    # minutes with zero output at concurrency=16). `as_completed` has no PER-FUTURE timeout
    # (only a whole-iterator one), and a thread that has not returned cannot be force-killed in
    # Python -- so the only reliable bound is at the WAVE level: submit a small batch, wait on
    # the whole batch with a hard ceiling, and record anything still pending as a timeout
    # without ever blocking on it again. Lower concurrency than the first (successful) pass,
    # on the working theory that 16 concurrent requests queued the local vLLM server past
    # where its own or litellm's timeout behaves.
    WAVE_TIMEOUT_S = 90.0
    n_done, n_err = 0, 0
    remaining = list(jobs)
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while remaining:
            wave, remaining = remaining[: args.concurrency], remaining[args.concurrency :]
            futures = {pool.submit(_one, job): job for job in wave}
            done, not_done = cf.wait(futures, timeout=WAVE_TIMEOUT_S)
            for fut in done:
                key, label, cell, err = fut.result()
                n_done += 1
                if cell is not None:
                    probe_rows[key][label] = cell
                    probe_rows[key].get("errors", {}).pop(label, None)  # clear a stale retry note
                else:
                    probe_rows[key].setdefault("errors", {})[label] = err
                    n_err += 1
            for fut in not_done:
                key, label, _model = futures[fut]
                probe_rows[key].setdefault("errors", {})[label] = (
                    f"TimeoutError: exceeded {WAVE_TIMEOUT_S}s wave backstop"
                )
                n_done += 1
                n_err += 1
            print(f"  probe progress: {n_done}/{len(jobs)} ({n_err} errors)", flush=True)
            cache_path.write_text(json.dumps(probe_rows, indent=1, sort_keys=True))
    cache_path.write_text(json.dumps(probe_rows, indent=1, sort_keys=True))
    print(f"probe done: {n_done} new calls, {n_err} errors", flush=True)

    from pinq_adapters.llm.pricing import PriceTable

    prices = PriceTable.load(strict=False)
    usd_total = 0.0
    for row in probe_rows.values():
        for label, model in MODEL_LABELS:
            cell = row.get(label)
            if not cell:
                continue
            usd_total += prices.usd(
                model, tok_prompt=cell["tok_prompt"], tok_completion=cell["tok_completion"]
            )
    print(
        f"probe cost (cumulative, all cells ever fetched): ${usd_total:.4f} "
        f"({prices.version} price table, qwen3-8b-base self-hosted = $0)"
    )

    # ---- answerability, over ALL parents and, separately, over violated-only (reported) ----
    violated_set = set(violated_keys)
    for label, _model in MODEL_LABELS:
        scored = [
            (row["task_id"], row["node_id"], row[label]["score"])
            for row in probe_rows.values()
            if label in row
        ]
        n_all = len(scored)
        n_all_correct = sum(1 for *_, s in scored if s >= 1.0)
        viol_scored = [(t, n, s) for t, n, s in scored if (t, n) in violated_set]
        n_v = len(viol_scored)
        n_v_correct = sum(1 for *_, s in viol_scored if s >= 1.0)
        rate_all = n_all_correct / n_all if n_all else float("nan")
        rate_v = n_v_correct / n_v if n_v else float("nan")
        print(
            f"  {label}: answerable over ALL parents {n_all_correct}/{n_all} ({rate_all:.4f}); "
            f"over VIOLATED-only parents {n_v_correct}/{n_v} ({rate_v:.4f})"
        )

    answerable = frozenset(
        (row["task_id"], row["node_id"])
        for row in probe_rows.values()
        if row.get("qwen3_8b_base", {}).get("score", 0.0) >= 1.0
    )
    print(
        f"answerable by qwen3-8b-base (of {len(all_parent_keys)} total parents): {len(answerable)}"
    )

    strata = stratified_contrast(edges, answerable)
    interaction = interaction_bca(edges, answerable)
    for name, res in strata.items():
        stab = res.stability
        stab_note = (
            ""
            if not stab.checked
            else f" [stability: {'STABLE' if stab.stable else 'UNSTABLE -> UNDECIDED'}; {stab.note}]"
        )
        print(
            f"  {name}: {res.estimate} (edges trained={res.n_edges_trained} prompted={res.n_edges_prompted}){stab_note}"
        )
    istab = interaction.stability
    istab_note = (
        ""
        if not istab.checked
        else f" [stability: {'STABLE' if istab.stable else 'UNSTABLE -> UNDECIDED'}; {istab.note}]"
    )
    print(
        f"  interaction (answerable - not_answerable): {interaction.point:+.4f} [{interaction.ci_lo:+.4f}, {interaction.ci_hi:+.4f}] n_tasks={interaction.n_tasks}{istab_note}"
    )

    # ---- secondary: logistic model of violation on arm x answerable, task-clustered SEs ----
    logit = logistic_violation_model(edges, answerable)
    print(f"logistic (task-clustered SE): {logit}")

    # ---- recovery share among trained violations, and answer_correct by violation status --
    trained_viol = [e for e in events if e.arm == ARM_TRAINED and e.counts_in_matched_metric]
    n_recovered = sum(1 for e in trained_viol if e.parent_resolved_later)
    n_never = sum(1 for e in trained_viol if e.parent_never_resolved)
    print(
        f"trained formal violations: {len(trained_viol)}; parent resolved later (within run)={n_recovered}; never={n_never}"
    )

    # The trained arm's matched window IS its own full run (basis_k = its own n_asks, and
    # MEASURED: 0 of 400 trained runs carry a match turn >= n_asks), so `parent_resolved_later`
    # and `counts_in_matched_metric` coincide on every trained event -- but that does NOT make
    # this set equal to `trained_viol` above: `trained_viol` is already restricted to
    # counts_in_matched_metric=True, so it is the resolved-later slice alone. This is the
    # FULL superset, resolved-later and never-resolved together.
    trained_skip = [e for e in events if e.arm == ARM_TRAINED]
    n_recovered_skip = sum(1 for e in trained_skip if e.parent_resolved_later)
    n_never_skip = sum(1 for e in trained_skip if e.parent_never_resolved)
    print(
        f"trained ALL skip events: {len(trained_skip)}; recovered_later={n_recovered_skip}; never_resolved={n_never_skip}"
    )

    prompted_skip = [e for e in events if e.arm == ARM_PROMPTED]
    n_recovered_p = sum(1 for e in prompted_skip if e.parent_resolved_later)
    n_never_p = sum(1 for e in prompted_skip if e.parent_never_resolved)
    n_beyond_window_p = sum(
        1 for e in prompted_skip if e.parent_resolved_later and not e.counts_in_matched_metric
    )
    print(
        f"prompted skip events (matched window truncated): {len(prompted_skip)}; recovered={n_recovered_p} (of which beyond the matched window={n_beyond_window_p}); never={n_never_p}"
    )

    # answer_correct: violating vs non-violating runs, per arm
    by_run: dict[str, list[int]] = defaultdict(list)
    for e in edges:
        by_run[e.run_id].append(1 if e.is_violation else 0)
    pairs = run_arms(con, args.store, args.suite)
    run_ids = [ra.run_id for arms in pairs.values() for ra in arms.values()]
    scores = terminal_scores(con, args.store, run_ids)
    arm_of_run = {ra.run_id: arm for arms in pairs.values() for arm, ra in arms.items()}
    for arm in (ARM_TRAINED, ARM_PROMPTED):
        viol_correct, nonviol_correct = [], []
        for run_id, viols in by_run.items():
            if arm_of_run.get(run_id) != arm:
                continue
            ac = scores.get(run_id, {}).get("answer_correct")
            if ac is None:
                continue
            (viol_correct if sum(viols) > 0 else nonviol_correct).append(ac)
        vm = statistics.fmean(viol_correct) if viol_correct else float("nan")
        nm = statistics.fmean(nonviol_correct) if nonviol_correct else float("nan")
        print(
            f"  {arm}: answer_correct violating={vm:.4f} (n={len(viol_correct)}) non_violating={nm:.4f} (n={len(nonviol_correct)})"
        )

    summary = {
        "suite": args.suite,
        "graph_version": args.graph_version,
        "lock_check": {
            "n_runs_checked": n_checked,
            "n_mismatch": n_mismatch,
            "n_matched_events": n_matched_events,
        },
        "n_events": len(events),
        "n_qualifying_edges": len(edges),
        "all_parent_task_node_keys": [f"{t}::{n}" for t, n in all_parent_keys],
        "violated_task_node_keys": [f"{t}::{n}" for t, n in violated_keys],
        "answerable_by_qwen3_8b_base": [f"{t}::{n}" for t, n in sorted(answerable)],
        "probe_cost_usd_cumulative": usd_total,
        "strata": {
            name: {
                "point": res.estimate.point,
                "ci_lo": res.estimate.ci_lo,
                "ci_hi": res.estimate.ci_hi,
                "n_tasks": res.estimate.n,
                "n_edges_trained": res.n_edges_trained,
                "n_edges_prompted": res.n_edges_prompted,
                "stability_checked": res.stability.checked,
                "stability_stable": res.stability.stable,
                "stability_note": res.stability.note,
            }
            for name, res in strata.items()
        },
        "interaction": {
            "point": interaction.point,
            "ci_lo": interaction.ci_lo,
            "ci_hi": interaction.ci_hi,
            "n_tasks": interaction.n_tasks,
            "stability_checked": interaction.stability.checked,
            "stability_stable": interaction.stability.stable,
            "stability_note": interaction.stability.note,
        },
        "logistic": logit.to_dict(),
        "trained_recovery": {
            "n_violations": len(trained_viol),
            "n_recovered": n_recovered,
            "n_never": n_never,
        },
        "prompted_recovery": {
            "n_events": len(prompted_skip),
            "n_recovered": n_recovered_p,
            "n_recovered_beyond_matched_window": n_beyond_window_p,
            "n_never": n_never_p,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
