"""Lane L1.11 driver: does the trained arm's evidence-coverage gain reach the ANSWER-BEARING
node, or does it land on prerequisites while the answer node stays uncovered more often.

Reads ONLY the isolated store `<population-dir>/scores_parquet` (never `scores/parquet`), the
suite's public corpus under `<corpora-root>/<suite>/<corpus_dir>/tasks.jsonl` (never
`data/gold/`), and gold graphs through `pi_eval.gold.load_graphs` (requires `PI_GOLD_ROOT`,
operator-only). Writes nothing to any of them.

Usage:
    PI_GOLD_ROOT=<repo>/data/gold PYTHONPATH=<worktree>/src <venv>/bin/python \\
        -m scripts.answer_node_coverage.run \\
        --population-dir <repo>/artifacts/testsplit_qa \\
        --corpora-root <repo>/data/corpora \\
        --runs-root <repo>/runs \\
        --out <scratch>/result.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from scripts.answer_node_coverage import answer_node, corpus_text, lib
from scripts.stopping_answer_test import lib as stopping_lib
from scripts.stopping_answer_test.run import (
    ARM_PROMPTED,
    ARM_TRAINED,
    GRID_NAME,
    MODEL_PROMPTED,
    MODEL_TRAINED,
    SUITES,
    _load_provenance,
)

from pi_eval.gold import load_graphs


def _corpus_dir_for_suite(con, suite: str, run_ids: list[str]) -> str:
    rows = lib._rows(
        con,
        "SELECT DISTINCT corpus_dir FROM runs WHERE suite_id = "
        f"'{suite}' AND run_id IN {lib._in(run_ids)}",
    )
    dirs = sorted({r["corpus_dir"] for r in rows if r["corpus_dir"]})
    if len(dirs) != 1:
        raise SystemExit(
            f"{suite}: expected exactly one corpus_dir across this population, got {dirs}"
        )
    return dirs[0]


def build_answer_nodes_for_suite(
    *, suite: str, task_ids: list[str], corpora_root: Path, corpus_dir: str, graph_version: str
) -> tuple[dict[str, Any], dict[str, answer_node.AnswerNodeResult]]:
    graphs = load_graphs(suite, graph_version)
    missing = [t for t in task_ids if t not in graphs]
    if missing:
        raise SystemExit(
            f"{suite}: {len(missing)} population task(s) absent from gold graphs at "
            f"graph_version={graph_version!r}, e.g. {missing[:5]}"
        )
    restricted = {t: graphs[t] for t in task_ids}
    uid_text_by_task = corpus_text.uid_text_maps_for_suite(
        corpora_root, suite, corpus_dir, task_ids
    )
    results = answer_node.answer_nodes_for_suite(restricted, uid_text_by_task)
    return restricted, results


def hand_verification_sample(
    graphs_by_suite: dict[str, dict[str, Any]],
    results_by_suite: dict[str, dict[str, answer_node.AnswerNodeResult]],
    uid_text_by_suite: dict[str, dict[str, dict[str, str]]],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """`n` tasks sampled across suites (proportional to how many tasks each contributes),
    with the chosen answer node's own text, its resolved evidence text, and the gold answer,
    for a human to eyeball before trusting the rule at scale."""
    all_keys = [(s, t) for s, d in results_by_suite.items() for t in d]
    rng = random.Random(seed)
    sample = sorted(rng.sample(all_keys, min(n, len(all_keys))))
    out = []
    for suite, task in sample:
        graph = graphs_by_suite[suite][task]
        res = results_by_suite[suite][task]
        uid_text = uid_text_by_suite[suite].get(task, {})
        node_by_id = {nd.gold_node_id: nd for nd in graph.gold_nodes}
        node_views = []
        for nid in res.node_ids:
            nd = node_by_id[nid]
            text, _ = answer_node.node_evidence_text(nd, uid_text)
            node_views.append(
                {
                    "node_id": nid,
                    "depth": nd.gold_depth,
                    "node_text": nd.gold_text,
                    "evidence_text_head": text[:220],
                }
            )
        out.append(
            {
                "suite": suite,
                "task": task,
                "answer": graph.answer,
                "aliases": list(graph.gold_aliases),
                "rule": res.rule,
                "nodes": node_views,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population-dir", required=True, type=Path)
    ap.add_argument("--corpora-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--sample-n", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, default=0)
    # The arm MODELS are arguments because this reader was pinned to one checkpoint pair, which
    # silently scoped every finding it produced to that pair. Defaults preserve old behaviour.
    ap.add_argument("--model-trained", default=MODEL_TRAINED)
    ap.add_argument("--model-prompted", default=MODEL_PROMPTED)
    args = ap.parse_args()

    prov = _load_provenance(
        args.population_dir,
        model_trained=args.model_trained,
        model_prompted=args.model_prompted,
    )
    store = args.population_dir / "scores_parquet"
    con = lib.open_store(store)
    scorer_hash = prov["scorer_hash"]
    graph_version = prov["graph_version"]

    result: dict[str, Any] = {"provenance": prov}

    # ---- step 0: population presence, same checks L1.5's step0 runs, reused not re-derived.
    trained_ids = {
        suite: [
            r["run_id"]
            for r in stopping_lib.load_arm_runs(
                con,
                arm_id=ARM_TRAINED,
                model_id=args.model_trained,
                grid_name=GRID_NAME,
                suite=suite,
            )
        ]
        for suite in SUITES
    }
    prompted_ids = {
        suite: [
            r["run_id"]
            for r in stopping_lib.load_arm_runs(
                con,
                arm_id=ARM_PROMPTED,
                model_id=args.model_prompted,
                grid_name=GRID_NAME,
                suite=suite,
            )
        ]
        for suite in SUITES
    }
    all_ids = sorted(
        {rid for ids in [*trained_ids.values(), *prompted_ids.values()] for rid in ids}
    )
    presence = stopping_lib.population_report(con, run_ids=all_ids, scorer_hash=scorer_hash)
    n_turns_rows = lib._rows(
        con, f"SELECT run_id, n_turns FROM runs WHERE run_id IN {lib._in(all_ids)}"
    )
    turns_check = stopping_lib.verify_turns_not_dropped(
        args.runs_root, {r["run_id"]: r["n_turns"] for r in n_turns_rows}
    )
    result["step0_population"] = {
        "n_population": len(all_ids),
        "presence": presence,
        "turns_not_dropped": turns_check,
        "clean": not presence["missing_from_runs"]
        and not presence["missing_scores"]
        and turns_check["n_bad"] == 0,
    }
    if not result["step0_population"]["clean"]:
        print("POPULATION CHECK FAILED -- see step0_population", file=sys.stderr)

    graphs_by_suite: dict[str, dict[str, Any]] = {}
    results_by_suite: dict[str, dict[str, answer_node.AnswerNodeResult]] = {}
    uid_text_by_suite: dict[str, dict[str, dict[str, str]]] = {}
    per_suite: dict[str, Any] = {}

    for suite in SUITES:
        t_runs = stopping_lib.load_arm_runs(
            con, arm_id=ARM_TRAINED, model_id=args.model_trained, grid_name=GRID_NAME, suite=suite
        )
        p_runs = stopping_lib.load_arm_runs(
            con, arm_id=ARM_PROMPTED, model_id=args.model_prompted, grid_name=GRID_NAME, suite=suite
        )
        task_ids = sorted({r["task_id"] for r in [*t_runs, *p_runs]})
        corpus_dir = _corpus_dir_for_suite(con, suite, [r["run_id"] for r in [*t_runs, *p_runs]])

        graphs, results = build_answer_nodes_for_suite(
            suite=suite,
            task_ids=task_ids,
            corpora_root=args.corpora_root,
            corpus_dir=corpus_dir,
            graph_version=graph_version,
        )
        graphs_by_suite[suite] = graphs
        results_by_suite[suite] = results
        uid_text_by_suite[suite] = corpus_text.uid_text_maps_for_suite(
            args.corpora_root, suite, corpus_dir, task_ids
        )

        answer_nodes_keyed = {(suite, t): r for t, r in results.items()}
        gold_partition = {
            (suite, t): answer_node.gold_uid_partition(graphs[t], r.node_ids)
            for t, r in results.items()
        }

        covered = lib.covered_map(con, [*t_runs, *p_runs], answer_nodes_keyed)
        hit_counts = lib.span_hit_counts(con, [*t_runs, *p_runs], gold_partition)
        xcheck = lib.cross_check_evidence_coverage(
            con, [*t_runs, *p_runs], hit_counts, scorer_hash=scorer_hash
        )
        stop_reason_by_run = {r["run_id"]: r["stop_reason"] for r in [*t_runs, *p_runs]}

        p_cov = lib.p_covered_delta(t_runs, p_runs, covered, n_boot=args.n_boot)
        p_stop_unc = lib.p_stop_given_uncovered_delta(
            t_runs, p_runs, covered, stop_reason_by_run, n_boot=args.n_boot
        )
        recall_trained = lib.recall_by_coverage_group(
            con, t_runs, covered, scorer_hash=scorer_hash, n_boot=args.n_boot
        )
        recall_prompted = lib.recall_by_coverage_group(
            con, p_runs, covered, scorer_hash=scorer_hash, n_boot=args.n_boot
        )
        within_both = lib.within_both_covered_recall_delta(
            con, t_runs, p_runs, covered, scorer_hash=scorer_hash, n_boot=args.n_boot
        )
        gain_share = lib.coverage_gain_share(
            con, t_runs, p_runs, hit_counts, gold_partition, n_boot=args.n_boot
        )

        per_suite[suite] = {
            "n_trained_runs": len(t_runs),
            "n_prompted_runs": len(p_runs),
            "n_tasks": len(task_ids),
            "corpus_dir": corpus_dir,
            "answer_node_rule_tally": answer_node.rule_tally(results),
            "evidence_coverage_cross_check": xcheck,
            "p_answer_node_covered": p_cov,
            "p_stop_given_answer_uncovered": p_stop_unc,
            "recall_by_coverage_group_trained": recall_trained,
            "recall_by_coverage_group_prompted": recall_prompted,
            "recall_delta_within_both_covered": within_both,
            "coverage_gain_share_nonanswer": gain_share,
        }

    result["per_suite"] = per_suite
    result["hand_verification_sample"] = hand_verification_sample(
        graphs_by_suite,
        results_by_suite,
        uid_text_by_suite,
        n=args.sample_n,
        seed=args.sample_seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"wrote {args.out}")
    print(f"population clean: {result['step0_population']['clean']}")
    for suite in SUITES:
        s = per_suite[suite]
        print(
            f"{suite}: P(covered) trained={s['p_answer_node_covered']['level_trained']:.4f} "
            f"prompted={s['p_answer_node_covered']['level_prompted']:.4f} "
            f"delta={s['p_answer_node_covered']['point']:+.4f} "
            f"[{s['p_answer_node_covered']['lo']:+.4f},{s['p_answer_node_covered']['hi']:+.4f}] "
            f"verdict={s['p_answer_node_covered']['verdict']}"
        )
    return 0


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, frozenset):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {o!r}")


if __name__ == "__main__":
    raise SystemExit(main())
