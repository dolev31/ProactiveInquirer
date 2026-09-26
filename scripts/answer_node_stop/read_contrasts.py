"""The ANSWER-NODE reading, for an arbitrary set of arms in ONE isolated scoring pass.

WHAT THIS ADDS TO `pi train gate`, AND WHAT IT DELIBERATELY DOES NOT DUPLICATE. The gate already
computes matched-cost and cap-8 evidence coverage, the stop 2x2 and `n_baseline_shorter_than_k`,
and it is the instrument the paper's tables are built from -- so those are read from its verdict
JSONs, not recomputed here. What the gate cannot say is anything about the ANSWER-BEARING node,
which is the whole point of lane L6.1. That is what this computes, by calling L1.11's already
tested estimators (`scripts.answer_node_coverage.lib`) rather than writing a second copy:

    P(answer node covered)                  per arm, and the paired BCa delta
    P(stop | answer node uncovered)         per arm, and the paired BCa delta
    answer recall / token-F1                per arm, and the paired BCa delta

THREE ARMS, NOT TWO. L1.11's own driver hardcodes one trained/prompted pair from its brief;
every function it calls takes run lists, so this one takes `--arm label:arm_id:model_id` and
contrasts each arm against every named comparator. The brief asks for both comparators on every
quantity, and a table with one comparator is how a checkpoint gets reported as better than the
base while being worse than the checkpoint it was supposed to improve on.

THE COVERAGE INSTRUMENT HERE IS THE MATCHER, NOT THE LABEL'S. `covered_map` reads
`match_kind in ('resolve','use')` from `matches.parquet` -- L1.11's published instrument, so
these numbers sit beside its -0.0900 / -0.0452 / +0.0179 on the same scale. The SFT LABEL
(`answer_node_covered_before`) uses uid containment instead, because a training label must be a
property of the state every candidate shares. The two answer different questions and this file
reports the matcher one; a reader comparing them should know they are two instruments.

USAGE (gold-side, operator only; the store must be an ISOLATED pass containing every arm):

    PI_GOLD_ROOT=<repo>/data/gold PYTHONPATH=<worktree>/src <venv>/bin/python \\
      -m scripts.answer_node_stop.read_contrasts \\
      --parquet <iso>/parquet --corpora-root <repo>/data/corpora \\
      --grid-name tier1_trained_qa_base \\
      --arm answernode:inquirer_trained:qwen3-8b-sft-headline-answernode \\
      --arm headline:inquirer_trained:qwen3-8b-sft-headline \\
      --arm base:inquirer_prompted:qwen3-8b-base \\
      --focus answernode --n-boot 10000 --out <out>/answer_node_contrasts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SUITES = ("musique", "strategyqa", "wiki2")
#: A bound this close to zero is re-read at 50,000 resamples over three seeds before it is
#: called a null or a difference. The campaign's own rule, applied here rather than by hand.
NEAR_ZERO = 0.01


def parse_arm(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--arm {spec!r}: expected label:arm_id:model_id. `arm_id` alone cannot name a "
            "checkpoint -- one arm id serves every checkpoint under the two-pin protocol -- so "
            "the model id is not optional."
        )
    return (parts[0], parts[1], parts[2])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parquet", required=True, help="the ISOLATED store, never scores/parquet")
    p.add_argument("--corpora-root", required=True)
    p.add_argument("--grid-name", default="tier1_trained_qa_base")
    p.add_argument("--graph-version", default="v1")
    p.add_argument("--arm", action="append", type=parse_arm, required=True)
    p.add_argument("--focus", required=True, help="the label of the arm every contrast is FOR")
    p.add_argument("--suite", action="append", default=None)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p


def _answer_nodes(
    *, suite: str, task_ids: Sequence[str], corpora_root: Path, corpus_dir: str, graph_version: str
):
    """L1.11's own builder, imported rather than restated -- it is the rule the diagnosis was
    published from and re-deriving it here is how the dataset's answer node and the reading's
    answer node would drift apart under one name."""
    from scripts.answer_node_coverage.run import build_answer_nodes_for_suite

    return build_answer_nodes_for_suite(
        suite=suite,
        task_ids=list(task_ids),
        corpora_root=Path(corpora_root),
        corpus_dir=corpus_dir,
        graph_version=graph_version,
    )


def scorer_hash_of(con) -> str:
    """The ONE `scorer_hash` in this store, or a refusal.

    `scorer_hash` is global over the gold set a pass touched, so two hashes in one store means
    two scoring passes were pooled and no cross-arm number in it is comparable. The campaign's
    convention is one fresh isolated pass per campaign with every arm in it; this is the check
    that the store in hand is one.
    """
    from scripts.answer_node_coverage.lib import _rows

    hashes = sorted(
        {str(r["scorer_hash"]) for r in _rows(con, "SELECT DISTINCT scorer_hash FROM scores")}
    )
    if len(hashes) != 1:
        raise SystemExit(
            f"this store carries {len(hashes)} scorer hashes {hashes}. Every arm must be scored "
            "in ONE isolated pass or the contrasts below are across instruments."
        )
    return hashes[0]


def answer_quality_contrast(
    con,
    focus_runs: Sequence[Mapping[str, Any]],
    other_runs: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    scorer_hash: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """focus-minus-comparator on one answer-quality metric, paired on task.

    BOTH F1 AND RECALL, ALWAYS TOGETHER. `answer_token_f1` is rank-correlated with answer
    LENGTH at -0.677 pooled and -0.934 within one arm, so an F1 gap between arms whose answers
    differ in length is not separable from a brevity gap; recall is the length-robust half and
    is what makes the primary interpretable rather than arguable. `pi_eval.score`'s own metric
    table says so in as many words.
    """
    from scripts.answer_node_coverage import lib
    from scripts.stopping_answer_test import lib as stopping_lib

    a = {
        k[1]: v
        for k, v in stopping_lib.task_level_metric(
            con, focus_runs, metric, scorer_hash=scorer_hash
        ).items()
    }
    b = {
        k[1]: v
        for k, v in stopping_lib.task_level_metric(
            con, other_runs, metric, scorer_hash=scorer_hash
        ).items()
    }
    out = lib._paired_with_stability(a, b, n_boot=n_boot, seed=seed)
    out["mean_focus"] = stopping_lib._mean(list(a.values()))
    out["mean_comparator"] = stopping_lib._mean(list(b.values()))
    out["metric"] = metric
    return out


def _near_zero(d: Mapping[str, Any]) -> bool:
    """Does either bound sit within 0.01 of zero? Then the interval gets re-read at 50k."""
    lo, hi = d.get("lo"), d.get("hi")
    return any(v is not None and abs(float(v)) < NEAR_ZERO for v in (lo, hi))


def contrasts_for_suite(
    con,
    *,
    suite: str,
    arms: Sequence[tuple[str, str, str]],
    focus: str,
    grid_name: str,
    corpora_root: Path,
    graph_version: str,
    scorer_hash: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    from scripts.answer_node_coverage import lib
    from scripts.answer_node_coverage.run import _corpus_dir_for_suite
    from scripts.stopping_answer_test.lib import load_arm_runs

    runs_by_label: dict[str, list[dict]] = {}
    for label, arm_id, model_id in arms:
        runs_by_label[label] = load_arm_runs(
            con, arm_id=arm_id, model_id=model_id, grid_name=grid_name, suite=suite
        )
    if focus not in runs_by_label:
        raise SystemExit(
            f"--focus {focus!r} is not one of the --arm labels {sorted(runs_by_label)}"
        )
    empty = [k for k, v in runs_by_label.items() if not v]
    if empty:
        raise SystemExit(
            f"{suite}: no runs selected for {empty} at grid {grid_name!r}. An arm that selects "
            "nothing reads as a clean zero in every contrast below; refused rather than reported."
        )

    all_ids = [r["run_id"] for rs in runs_by_label.values() for r in rs]
    corpus_dir = _corpus_dir_for_suite(con, suite, all_ids)
    task_ids = sorted({str(r["task_id"]) for rs in runs_by_label.values() for r in rs})
    _graphs, nodes = _answer_nodes(
        suite=suite,
        task_ids=task_ids,
        corpora_root=corpora_root,
        corpus_dir=corpus_dir,
        graph_version=graph_version,
    )
    answer_nodes = {(suite, tid): res for tid, res in nodes.items()}

    covered = lib.covered_map(con, [r for rs in runs_by_label.values() for r in rs], answer_nodes)
    stop_reason = {
        str(r["run_id"]): str(r.get("stop_reason") or "")
        for rs in runs_by_label.values()
        for r in rs
    }

    out: dict[str, Any] = {
        "suite": suite,
        "corpus_dir": corpus_dir,
        "n_runs": {k: len(v) for k, v in runs_by_label.items()},
        "rule_tally": _tally(nodes),
        "levels": {},
        "contrasts": {},
    }
    for label, rs in runs_by_label.items():
        vals = [covered[str(r["run_id"])] for r in rs]
        known = [v for v in vals if v is not None]
        out["levels"][label] = {
            "n_runs": len(rs),
            "n_answer_node_unknown": len(vals) - len(known),
            "p_answer_node_covered": (sum(known) / len(known)) if known else float("nan"),
        }

    for label in runs_by_label:
        if label == focus:
            continue
        cov = lib.p_covered_delta(
            runs_by_label[focus], runs_by_label[label], covered, n_boot=n_boot, seed=seed
        )
        stop = lib.p_stop_given_uncovered_delta(
            runs_by_label[focus],
            runs_by_label[label],
            covered,
            stop_reason,
            n_boot=n_boot,
            seed=seed,
        )
        entry = {
            "p_answer_node_covered": cov | {"reread_at_50k": _near_zero(cov)},
            "p_stop_given_answer_node_uncovered": stop | {"reread_at_50k": _near_zero(stop)},
        }
        for metric in ("answer_token_f1", "answer_token_recall"):
            q = answer_quality_contrast(
                con,
                runs_by_label[focus],
                runs_by_label[label],
                metric=metric,
                scorer_hash=scorer_hash,
                n_boot=n_boot,
                seed=seed,
            )
            entry[metric] = q | {"reread_at_50k": _near_zero(q)}
        out["contrasts"][f"{focus}_vs_{label}"] = entry
    return out


def _tally(nodes) -> dict[str, int]:
    from scripts.answer_node_coverage.answer_node import rule_tally

    return rule_tally(nodes)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    from scripts.answer_node_coverage import lib

    con = lib.open_store(Path(a.parquet))
    scorer_hash = scorer_hash_of(con)
    suites = tuple(a.suite) if a.suite else SUITES
    report = {
        "parquet": str(Path(a.parquet).resolve()),
        "scorer_hash": scorer_hash,
        "grid_name": a.grid_name,
        "graph_version": a.graph_version,
        "arms": [{"label": x[0], "arm_id": x[1], "model_id": x[2]} for x in a.arm],
        "focus": a.focus,
        "n_boot": a.n_boot,
        "bootstrap_seed": a.seed,
        "by_suite": {},
    }
    for suite in suites:
        report["by_suite"][suite] = contrasts_for_suite(
            con,
            suite=suite,
            arms=a.arm,
            focus=a.focus,
            grid_name=a.grid_name,
            corpora_root=Path(a.corpora_root),
            graph_version=a.graph_version,
            scorer_hash=scorer_hash,
            n_boot=a.n_boot,
            seed=a.seed,
        )
        r = report["by_suite"][suite]
        print(f"\n=== {suite}  n={r['n_runs']} ===", flush=True)
        for label, lv in r["levels"].items():
            print(
                f"  P(answer node covered) {label:<12} {lv['p_answer_node_covered']:.4f}"
                f"  (unknown {lv['n_answer_node_unknown']})",
                flush=True,
            )
        for name, c in r["contrasts"].items():
            for metric, d in c.items():
                flag = "  REREAD@50k" if d.get("reread_at_50k") else ""
                print(
                    f"  {name} {metric}: {d.get('delta'):+.4f} "
                    f"[{d.get('lo'):+.4f},{d.get('hi'):+.4f}] {d.get('verdict', '')}{flag}",
                    flush=True,
                )
    Path(a.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - a driver
    raise SystemExit(main())
