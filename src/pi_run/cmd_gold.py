"""`pi gold` — build, mine, validate and inventory the gold need graphs.

WHY THIS IS A SEPARATE COMMAND FROM `pi data`. They share one builder pass, because a corpus
and its gold graph are two halves of a single split and producing them separately is how the
two drift. What differs is what a reader is asking for. `pi data` answers "is the input
present and verified"; `pi gold` answers "what is in the graph, was it mined admissibly, and
which gates does it pass". Those are different audiences and different failure modes, so they
are different commands over the same pass rather than one command with a mode flag.

THE FOUR SUBCOMMANDS

  build     drive a suite's builder and report the GOLD side: graphs, nodes, edges, depth.
  mine      drive S0..S7 (pi_eval.mining) from THIS repository's recorded rollouts and emit a
            MinedGraph per task, each carrying its own diagnostics.
  validate  run the admissibility gates that are actually runnable, and print each verdict
            WITH its written consequence. A gate whose input does not exist is printed as
            NOT RUN with the input it needs — never silently omitted, because a gate list
            that shrinks when data is missing is a gate list that always passes.
  stats     nodes / edges / depth histogram / provenance / partition / orphan rate per suite.

WHAT `mine` REFUSES, AND WHY IT SAYS SO OUT LOUD
    `pi_eval.mining.pool.pool_is_admissible` demands >= 2 generator cells and >= 2 model
    families before a candidate need may be promoted, because one generator's habits (a
    planner that always dumps the schema first) would otherwise manufacture a phantom need.
    A task whose pool is thinner than that is emitted with `admissible=false` and the reason,
    and its refusal is printed and counted. A systematically-skipped stratum has to stay
    visible; a miner that quietly returned fewer tasks would hide exactly the bias the
    factorial exists to detect.

NO NLI MODEL IS DOWNLOADED HERE. S2 canonicalization takes its entailment function by
injection, and the only one shipped in the library is `canon.exact_entail` (normalized string
identity). That is a WEAKER canonicalizer than the pinned NLI model the design calls for: it
splits needs a real model would merge, so node counts run high and promotion rates run low.
It is recorded in the graph pin as `exact_entail@stdlib` so no mined graph can ever be
mistaken for one produced under a real entailment model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pi_run.cmd_data import GRAPH_VERSION, SUITES, build_suite

# THE SUITES THAT HAVE GOLD, which is not the same list as the suites `pi data` BUILDS.
#
# `SUITES` names what the fetch/build pipeline drives. tau2 and pare are SELF-SOURCED -- their
# corpora come from an upstream checkout -- so they are not in it. But tau2 HAS a gold graph
# (data/gold/graphs/tau2/v1.jsonl, 5,790 nodes) and it carries the primary endpoint, and
# `pi gold stats --suite tau2` answered `invalid choice: 'tau2'`. `gold_stats` computes its
# numbers perfectly well; nothing could ask for them.
#
# That also silenced a real measurement: tau2's graph is 63.9% depth-orphaned (3,637 of 5,790
# nodes unreachable), which no command printed and no table shows.
#
# `pi gold questions` reads the same list, so the two CEILING arms and the pre-spend headroom
# gate could not be seeded on tau2 either.
# `tau2_retail` joins them for the same reason and with the OPPOSITE split status: it is
# self-sourced from the same upstream checkout, it has gold (112 graphs, 1,015 nodes), and it
# is TRAINABLE -- it exists so a user simulator can reach the training set without spending
# `tau2` itself, which stays the zero-shot transfer target. The two are distinguished by exact
# membership in `EVAL_ONLY_SUITES`; see tests/test_retail_suite_registration.py for why a
# `startswith("tau2")` tidy-up would silently destroy one claim or the other.
GOLD_SUITES: tuple[str, ...] = tuple(sorted(set(SUITES) | {"tau2", "tau2_retail"}))

# The retriever each suite actually runs, recorded on every mined cell. It is a property of
# the SUITE and not of the run, which is why the miner takes it from here rather than trying
# to infer it from a trajectory that never wrote it down.
RETRIEVAL_VARIANT: dict[str, str] = {
    "synth": "token",
    "musique": "bm25",
    "strategyqa": "bm25",
    "wiki2": "bm25",
    "drgym": "clueweb22",
    "tau2": "bm25",
}

# The entailment oracle used by S2 when no NLI model is available. Named, not defaulted: it
# rides into `MinedGraph.graph_hash`, so a graph mined under string identity can never be
# confused with one mined under a real model.
EXACT_ENTAIL_PIN = "exact_entail@stdlib"

MINED_DIRNAME = "mined"


def _jsonable(obj: Any) -> Any:
    """Dataclasses -> dicts, non-string mapping keys -> strings, tuples -> lists.

    `theta_sweep` and `partition_elasticity` are keyed by FLOAT thresholds, which json cannot
    encode as object keys. Rounding them to a fixed width here keeps the artifact stable
    across platforms rather than emitting `0.6000000000000001`.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {
            (f"{k:.2f}" if isinstance(k, float) else str(k)): _jsonable(v) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return obj


def _root(a: argparse.Namespace) -> Path:
    from pi_run.manifest import repo_root

    return Path(a.root).resolve() if getattr(a, "root", None) else repo_root()


# --------------------------------------------------------------------------- gold inventory


def _graph_path(root: Path, suite: str, version: str) -> Path:
    return root / "data" / "gold" / "graphs" / suite / f"{version}.jsonl"


def gold_stats(root: Path, suite: str, *, graph_version: str = GRAPH_VERSION) -> dict[str, Any]:
    """Read one suite's gold graphs and summarise them. Reads only; never builds."""
    from pi_eval.build.common import read_graphs
    from pi_eval.gold import compute_depths, orphan_rate

    path = _graph_path(root, suite, graph_version)
    out: dict[str, Any] = {
        "suite": suite,
        "graph_version": graph_version,
        "path": str(path),
        "present": path.exists(),
        "n_graphs": 0,
        "n_nodes": 0,
        "n_edges": 0,
        "n_facets": 0,
        "n_seeds": 0,
        "depth_histogram": {},
        "orphan_rate": float("nan"),
        "provenance": {},
        "partition": {},
        "edge_verified": {},
    }
    if not path.exists():
        return out

    graphs = read_graphs(path)
    depth_hist: dict[str, int] = {}
    prov: dict[str, int] = {}
    part: dict[str, int] = {}
    verified: dict[str, int] = {}
    orphans: list[float] = []

    for g in graphs:
        out["n_graphs"] += 1
        out["n_nodes"] += len(g.gold_nodes)
        out["n_edges"] += len(g.gold_edges)
        out["n_facets"] += len(g.gold_facets)
        out["n_seeds"] += len(g.gold_seed_node_ids)
        for n in g.gold_nodes:
            # Depth is RECOMPUTED from the stored edges rather than trusted off the record,
            # so a stamped depth that disagrees with the graph it was stamped from shows up
            # here rather than in a C@d table.
            prov[n.gold_provenance_primary] = prov.get(n.gold_provenance_primary, 0) + 1
            part[n.gold_partition] = part.get(n.gold_partition, 0) + 1
        for e in g.gold_edges:
            verified[e.gold_verified] = verified.get(e.gold_verified, 0) + 1
        depths = compute_depths(
            [n.gold_node_id for n in g.gold_nodes],
            list(g.gold_edges),
            list(g.gold_seed_node_ids),
        )
        orphans.append(orphan_rate(depths))
        for d in depths.values():
            key = "unreachable" if d is None else str(d)
            depth_hist[key] = depth_hist.get(key, 0) + 1

    out["depth_histogram"] = {k: depth_hist[k] for k in sorted(depth_hist, key=_depth_key)}
    out["orphan_rate"] = sum(orphans) / len(orphans) if orphans else float("nan")
    out["provenance"] = dict(sorted(prov.items()))
    out["partition"] = dict(sorted(part.items()))
    out["edge_verified"] = dict(sorted(verified.items()))
    return out


def _depth_key(k: str) -> tuple[int, int]:
    return (1, 0) if k == "unreachable" else (0, int(k))


# --------------------------------------------------------------------------- mining


def _scores(parquet_dir: Path, metric: str) -> dict[str, float]:
    """run_id -> the named metric, from `scores/parquet/scores.parquet`.

    A run with no row is ABSENT from this mapping, not zero. `pool.failures` is the negative
    control that decides whether a need is diagnostic at all, so defaulting an unscored run
    to failure would bias every lift statistic toward "everything is diagnostic".
    """
    path = parquet_dir / "scores.parquet"
    if not path.exists():
        raise SystemExit(
            f"no scores at {path}. Mining needs a per-run success verdict, so it needs a "
            "scored parquet:\n"
            "  .venv/bin/pi compact\n"
            '  .venv/bin/pi score --gold-root "$PWD/data/gold"'
        )
    from pinq.extras import require

    require("pandas", why="gold statistics")
    import pandas as pd  # noqa: E402

    df = pd.read_parquet(path)
    rows = df[df["metric_name"] == metric]
    if rows.empty:
        known = sorted(df["metric_name"].unique())
        raise SystemExit(f"no metric {metric!r} in {path}. Present: {known}")
    return {str(r.run_id): float(r.value) for r in rows.itertuples()}


def mine_suite(
    *,
    root: Path,
    suite: str,
    runs_root: Path,
    parquet_dir: Path,
    corpus_dir: Path | None,
    metric: str,
    theta: float,
    task_ids: list[str] | None = None,
    success_threshold: float | None = None,
) -> dict[str, Any]:
    """Drive S0..S7 over every recorded run of one suite and return the artifact.

    Returns the whole report rather than writing it, so a caller can assert on it without a
    filesystem round trip and `--json` is a print rather than a second code path.
    """
    from pi_eval.mining.canon import exact_entail
    from pi_eval.mining.from_runs import build_pools, corpus_documents, read_runs
    from pi_eval.mining.pipeline import mine

    variant = RETRIEVAL_VARIANT.get(suite, "unknown")
    runs = read_runs(runs_root, suite_id=suite, task_ids=task_ids) if runs_root.is_dir() else []
    if not runs:
        raise SystemExit(
            f"no completed {suite!r} runs under {runs_root}. S0 is assembled from recorded "
            "rollouts, so there is nothing to mine until a sweep has run:\n"
            f"  .venv/bin/pi run --suite {suite} --arm fake_chain --arm fake_depth1 --n 5"
        )
    scores = _scores(parquet_dir, metric)
    pools = build_pools(
        runs, scores, suite_id=suite, retrieval_variant=variant, success_threshold=success_threshold
    )

    graphs: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    for pb in pools:
        question = _question_of(corpus_dir, pb.task_id) if corpus_dir else ""
        docs = corpus_documents(corpus_dir, pb.task_id) if corpus_dir else []
        g = mine(
            suite=suite,
            task_key=pb.task_id,
            question=question,
            pool=pb.pool,
            candidates=pb.candidates,
            entail=exact_entail,
            documents=docs,
            # No probe policy is wired here, so NO intervention edge can be minted. Order
            # alone may only veto an edge (edges.py), so a mined graph with no probe has
            # mechanical edges or none at all -- which is the honest outcome, not a gap.
            probe=None,
            theta=theta,
            nli_pin=EXACT_ENTAIL_PIN,
        )
        row = _jsonable(g)
        row["graph_hash"] = g.graph_hash
        row["graph_version"] = g.graph_version
        row["pool"] = {
            "n_success": pb.pool.n_success,
            "n_failure": pb.pool.n_failure,
            "cells": sorted(pb.pool.cells),
            "families": sorted(pb.pool.families),
            "policy_forms": sorted(pb.pool.policy_forms),
            "n_runs": pb.n_runs,
            "n_unscored": pb.n_unscored,
            "n_non_generator": pb.n_non_generator,
        }
        graphs.append(row)
        if not g.admissible:
            refused.append({"task_key": pb.task_id, "reason": g.reason})

    return {
        "suite": suite,
        "retrieval_variant": variant,
        "metric": metric,
        "theta": theta,
        "nli_pin": EXACT_ENTAIL_PIN,
        "runs_root": str(runs_root),
        "n_runs": len(runs),
        "n_tasks": len(pools),
        "n_admissible": sum(1 for g in graphs if g["admissible"]),
        "n_refused": len(refused),
        "refusals": refused,
        "graphs": graphs,
    }


def _question_of(corpus_dir: Path, task_id: str) -> str:
    """The task's public question, for S6's entity-coverage seed test.

    Read from the PUBLIC corpus. The seed test asks "is this need already stated in x", and
    the only honest x is the one the policy saw.
    """
    path = Path(corpus_dir) / "tasks.jsonl"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if str(rec.get("id")) == str(task_id):
            return str(rec.get("question") or "")
    return ""


def write_mined(root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Two files, deliberately: the graphs and the accounting for what did not become one."""
    d = root / "data" / "gold" / MINED_DIRNAME / report["suite"]
    d.mkdir(parents=True, exist_ok=True)
    graphs_path = d / "mined.jsonl"
    graphs_path.write_text(
        "\n".join(json.dumps(g, sort_keys=True) for g in report["graphs"]) + "\n"
    )
    diag_path = d / "diagnostics.json"
    diag_path.write_text(
        json.dumps({k: v for k, v in report.items() if k != "graphs"}, indent=1, sort_keys=True)
        + "\n"
    )
    return graphs_path, diag_path


def read_mined(root: Path, suite: str) -> list[dict[str, Any]]:
    path = root / "data" / "gold" / MINED_DIRNAME / suite / "mined.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# --------------------------------------------------------------------------- handlers


def cmd_gold_build(a: argparse.Namespace) -> int:
    root = _root(a)
    suites = [a.suite] if a.suite else list(GOLD_SUITES)
    rc = 0
    for suite in suites:
        try:
            built = build_suite(
                suite,
                root=root,
                split=a.split,
                limit=a.limit,
                offline=a.offline,
                raw_dir=Path(a.raw_dir) if a.raw_dir else None,
            )
        except Exception as exc:  # noqa: BLE001 - an actionable message, not a traceback
            print(f"{suite}: FAILED {type(exc).__name__}: {exc}")
            rc = 1
            continue
        st = gold_stats(root, suite, graph_version=a.graph_version)
        print(
            f"{suite}: graphs={st['n_graphs']} nodes={st['n_nodes']} edges={st['n_edges']} "
            f"facets={st['n_facets']} orphan_rate={st['orphan_rate']:.3f}"
        )
        print(f"    gold   {built.gold.relative_to(root)}")
        print(f"    depth  {st['depth_histogram']}")
    return rc


def cmd_gold_stats(a: argparse.Namespace) -> int:
    root = _root(a)
    suites = [a.suite] if a.suite else list(GOLD_SUITES)
    rows = [gold_stats(root, s, graph_version=a.graph_version) for s in suites]
    if a.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    head = (
        f"{'suite':<11} {'graphs':>7} {'nodes':>7} {'edges':>7} {'facets':>7} "
        f"{'orphan':>7}  {'depth histogram':<28} provenance"
    )
    print(head)
    print("-" * len(head))
    for st in rows:
        if not st["present"]:
            print(f"{st['suite']:<11} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}  not built")
            continue
        prov = ", ".join(f"{k}={v}" for k, v in st["provenance"].items())
        hist = ", ".join(f"{k}:{v}" for k, v in st["depth_histogram"].items())
        print(
            f"{st['suite']:<11} {st['n_graphs']:>7} {st['n_nodes']:>7} {st['n_edges']:>7} "
            f"{st['n_facets']:>7} {st['orphan_rate']:>7.3f}  {hist:<28} {prov}"
        )
    print("-" * len(head))
    for st in rows:
        if st["present"] and st["edge_verified"]:
            ev = ", ".join(f"{k}={v}" for k, v in st["edge_verified"].items())
            print(f"{st['suite']:<11} edges by verification: {ev}")
    print(
        "\norphan_rate is the share of nodes unreachable from the depth-0 frontier; depth is "
        "RECOMPUTED here from the stored edges, never read off the record."
    )
    return 0


# --------------------------------------------------------------------------- gold questions
#
# WHY THIS COMMAND EXISTS. `tier1_oracle` carries the two ceiling arms, and both of them were
# seeded by `recorded_questions(runs_root, arm_id="inquirer_prompted")` -- THE TREATMENT'S OWN
# QUESTIONS. That made `gold_evidence` input-identical to `parallel_replay`, made
# `oracle_vreq` a second copy of it, and made the pre-spend headroom gate a tautology: it
# compared the treatment's questions against the treatment's questions and could only ever
# report that the treatment had no headroom over itself. A ceiling arm seeded from the arm it
# is supposed to bound is not a ceiling.
#
# The two modes are deliberately DIFFERENT ceilings, because they bound different things:
#
#   oracle_vreq    Perfect SELECTION and perfect SEQUENCING, ordinary retrieval. One question
#                  per REQUIRED node, decontextualized, in prerequisite-topological order. No
#                  answer text. It answers: "if the Inquirer had asked exactly the right
#                  questions in exactly the right order, how far would it get?"
#
#   gold_evidence  The above, plus each node's own answer aliases appended to force retrieval
#                  onto the gold paragraph. It answers: "if the evidence were simply handed
#                  over, how far would it get?" -- an upper bound on the ANSWERER, above which
#                  no inquiry policy can reach.
#
# Both are gold-derived, both stamp gold_exposed=true through `Arm.requires_gold`, and the
# aggregator refuses either in a primary table.

_PLACEHOLDER = None  # compiled lazily; see _resolve_placeholders


def _topological(nodes: list, edges: list) -> list:
    """Required nodes in prerequisite order (Kahn), ties broken by node id for determinism.

    Prerequisite means the destination is UNANSWERABLE until the source is resolved, so an
    oracle that asked them in any other order would be asking a question it could not yet
    have phrased -- which is the very thing `parallel_replay` exists to show is different.
    A cycle (there should be none) degrades to id order rather than dropping nodes.
    """
    ids = [n.gold_node_id for n in nodes]
    keep = set(ids)
    incoming: dict[str, set[str]] = {i: set() for i in ids}
    outgoing: dict[str, set[str]] = {i: set() for i in ids}
    for e in edges:
        if e.gold_edge_kind != "prerequisite":
            continue
        src, dst = e.gold_src_node_id, e.gold_dst_node_id
        if src in keep and dst in keep:
            incoming[dst].add(src)
            outgoing[src].add(dst)
    order: list[str] = []
    ready = sorted(i for i in ids if not incoming[i])
    seen: set[str] = set()
    while ready:
        cur = ready.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        for nxt in sorted(outgoing[cur]):
            incoming[nxt].discard(cur)
            if not incoming[nxt] and nxt not in seen:
                ready.append(nxt)
                ready.sort()
    order.extend(i for i in ids if i not in seen)  # cycle fallback: never drop a node
    by_id = {n.gold_node_id: n for n in nodes}
    return [by_id[i] for i in order]


def _resolve_placeholders(text: str, answered: list[str]) -> str:
    """MuSiQue writes later hops as `When was #1 founded?`, where `#1` is hop 1's ANSWER.

    Left unresolved, that string retrieves nothing useful and the "ceiling" would score below
    the treatment -- a ceiling that loses to the arm it bounds is not measuring headroom, it
    is measuring a bug. Substituting the prior hop's answer is exactly what a policy with
    perfect selection would have in hand at that point in the sequence, so it is the honest
    decontextualization rather than a shortcut.
    """
    import re

    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        _PLACEHOLDER = re.compile(r"#(\d+)")

    def sub(m):
        i = int(m.group(1)) - 1
        return answered[i] if 0 <= i < len(answered) else m.group(0)

    return _PLACEHOLDER.sub(sub, text)


def questions_for(graph: Any, *, mode: str) -> list[str]:
    """The question list for one task, in the order an oracle would ask them."""
    required = [n for n in graph.gold_nodes if n.gold_partition == "required"]
    if not required:
        return []
    ordered = _topological(required, list(graph.gold_edges))

    out: list[str] = []
    answered: list[str] = []
    for n in ordered:
        text = (n.gold_text or "").strip()
        if not text:
            continue
        text = _resolve_placeholders(text, answered)
        if mode == "gold_evidence":
            # The ANSWER appended. This arm bounds the Answerer, not the Inquirer: it exists
            # to say what is reachable when the evidence is simply handed over, so forcing
            # retrieval onto the gold paragraph is the point rather than a leak. It is why the
            # arm declares requires_gold and is refused in every primary table.
            alias = next((a for a in n.gold_aliases if a and a.strip()), "")
            if alias:
                text = f"{text} {alias}"
        out.append(text)
        answered.append(next((a for a in n.gold_aliases if a and a.strip()), ""))
    return out


def gold_questions(
    root: Path, suite: str, *, mode: str, graph_version: str = GRAPH_VERSION
) -> dict[str, Any]:
    from pi_eval.build.common import read_graphs

    path = _graph_path(root, suite, graph_version)
    if not path.exists():
        raise SystemExit(f"no gold for {suite!r} at {path}. Build it first: pi gold build.")
    graphs = read_graphs(path)
    questions: dict[str, list[str]] = {}
    empty: list[str] = []
    for g in graphs:
        qs = questions_for(g, mode=mode)
        if qs:
            questions[g.gold_task_key] = qs
        else:
            empty.append(g.gold_task_key)
    counts = [len(v) for v in questions.values()]
    return {
        # Provenance rides WITH the payload. `pi run --questions-from-file` accepts either a
        # bare mapping or this wrapper, so the file that seeded a ceiling arm can always say
        # which graph version and which mode produced it.
        "suite": suite,
        "mode": mode,
        "graph_version": graph_version,
        "source": str(path),
        "n_tasks": len(questions),
        "n_without_required_nodes": len(empty),
        "questions_per_task": {
            "min": min(counts) if counts else 0,
            "median": sorted(counts)[len(counts) // 2] if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "questions": questions,
    }


def verify_recall(
    root: Path, suite: str, questions: Mapping[str, Sequence[str]], *, k: int, n: int | None = None
) -> dict[str, Any]:
    """What share of each task's REQUIRED gold evidence these questions actually retrieve.

    THIS IS THE PRE-SPEND HEADROOM MEASUREMENT, and it costs zero tokens. A "ceiling" arm whose
    questions do not reach the gold evidence is not a ceiling: it would score near the
    treatment, Gate 2 would read "no headroom", and the musique track would be cancelled for a
    retrieval-plumbing reason rather than a scientific one. Reported against the FULL QUESTION
    baseline, because the number that matters is the gap: how much gold evidence a perfectly
    chosen question sequence reaches that the task's own question does not.

    Uses the suite's real retriever at the grid's k, so it measures the retrieval the sweep
    will actually perform rather than an idealisation of it.
    """
    import statistics

    from pi_eval.build.common import read_graphs
    from pi_run.worker import load_suite

    corpora = sorted(
        (root / "data" / "corpora" / suite).glob("*/tasks.jsonl"), key=lambda p: p.parent.name
    )
    if len(corpora) != 1:
        raise SystemExit(
            f"expected exactly one built corpus for {suite!r}, found {len(corpora)}. "
            "Two corpus hashes are two different frozen corpora."
        )
    adapter = load_suite(suite, str(corpora[0].parent))
    graphs = {g.gold_task_key: g for g in read_graphs(_graph_path(root, suite, GRAPH_VERSION))}

    ids = [t for t in adapter.task_ids() if t in graphs and t in questions]
    if n:
        ids = ids[:n]

    ours: list[float] = []
    baseline: list[float] = []
    full = 0
    for tid in ids:
        gold = {
            u
            for node in graphs[tid].gold_nodes
            if node.gold_partition == "required"
            for u in node.gold_ev_uids
        }
        if not gold:
            continue
        retriever = adapter.retriever(tid)
        got: set[str] = set()
        for q in questions[tid]:
            got |= {u.uid for u in retriever.search(q, k)}
        rec = len(gold & got) / len(gold)
        ours.append(rec)
        full += rec == 1.0
        base = {u.uid for u in retriever.search(adapter.view(tid).question, k)}
        baseline.append(len(gold & base) / len(gold))

    if not ours:
        return {"n_scored": 0, "note": "no task carries required gold evidence uids"}
    return {
        "n_scored": len(ours),
        "k": k,
        "mean_recall": round(statistics.mean(ours), 4),
        "median_recall": round(statistics.median(ours), 4),
        "tasks_with_full_recall": full,
        "baseline_full_question_mean_recall": round(statistics.mean(baseline), 4),
        # The whole point. A ceiling that does not clear the baseline is not a ceiling.
        "headroom_over_baseline": round(statistics.mean(ours) - statistics.mean(baseline), 4),
    }


def cmd_gold_questions(a: argparse.Namespace) -> int:
    root = _root(a)
    payload = gold_questions(root, a.suite, mode=a.mode, graph_version=a.graph_version)
    if a.verify_recall:
        payload["recall"] = verify_recall(
            root, a.suite, payload["questions"], k=a.verify_recall, n=a.n
        )
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary = {k: v for k, v in payload.items() if k != "questions"}
        summary["out"] = str(out)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["n_tasks"] else 1


def cmd_gold_mine(a: argparse.Namespace) -> int:
    root = _root(a)
    runs_root = Path(a.runs_root) if a.runs_root else root / "runs"
    parquet = Path(a.parquet) if a.parquet else root / "scores" / "parquet"
    corpus = Path(a.corpus) if a.corpus else _default_corpus(root, a.suite)

    report = mine_suite(
        root=root,
        suite=a.suite,
        runs_root=runs_root,
        parquet_dir=parquet,
        corpus_dir=corpus,
        metric=a.metric,
        theta=a.theta,
        task_ids=list(a.task) or None,
        success_threshold=a.success_threshold,
    )
    graphs_path, diag_path = write_mined(root, report)

    print(
        f"suite={report['suite']} runs={report['n_runs']} tasks={report['n_tasks']} "
        f"variant={report['retrieval_variant']} metric={report['metric']} "
        f"theta={report['theta']} nli_pin={report['nli_pin']}"
    )
    for g in report["graphs"]:
        if not g["admissible"]:
            print(f"  {g['task_key']:<10} REFUSED: {g['reason']}")
            continue
        d = g["diagnostics"]
        print(
            f"  {g['task_key']:<10} nodes={d['n_candidates']} promoted={d['n_promoted']} "
            f"edges={len(g['edges'])} facets={len(g['facets'])} "
            f"seeds={int(d['seed_basis']['n_seeds'])} orphan={d['orphan_rate']:.3f} "
            f"graph_version={g['graph_version']}"
        )
    print(
        f"\nadmissible={report['n_admissible']}/{report['n_tasks']} refused={report['n_refused']}"
    )
    print(f"graphs      -> {graphs_path.relative_to(root)}")
    print(f"diagnostics -> {diag_path.relative_to(root)}")
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))

    if report["n_admissible"] == 0:
        # REFUSAL IS THE POINT, not an error to be smoothed over. Exiting non-zero is what
        # stops a pipeline from carrying an empty mined graph forward as if it were gold.
        reasons = sorted({r["reason"] for r in report["refusals"]})
        print(
            "\nEXIT 1: no task produced an admissible pool. "
            f"Reasons: {reasons}\n"
            "A pool needs >= 2 generator cells and >= 2 model families before a candidate "
            "need may be promoted; below that, one generator's habits are indistinguishable "
            "from a real need. Widen the factorial (another model family, another policy "
            "form) and re-run the sweep."
        )
        return 1
    return 0


def _default_corpus(root: Path, suite: str) -> Path | None:
    """The suite's built corpus, if there is exactly one.

    Ambiguity is left to the caller: two corpus hashes are two different frozen corpora, and
    guessing between them would silently change the documents the discoverability cut is
    measured against.
    """
    base = root / "data" / "corpora" / suite
    cands = sorted(d for d in base.glob("*") if (d / "tasks.jsonl").exists())
    return cands[0] if len(cands) == 1 else None


def cmd_gold_validate(a: argparse.Namespace) -> int:
    """Run the gates that CAN be run, and name the input every other one is waiting on."""
    from pi_eval.mining.pipeline import MinedGraph, gate_report

    root = _root(a)
    rows = read_mined(root, a.suite)
    if not rows:
        raise SystemExit(
            f"no mined graphs for {a.suite!r}. Run `pi gold mine --suite {a.suite}` first."
        )

    # Only the diagnostics are needed by gate_report, so the graphs are rehydrated as the
    # thin shells that carry them rather than fully reconstructed.
    graphs = [
        MinedGraph(
            suite=r["suite"],
            task_key=r["task_key"],
            diagnostics=r["diagnostics"],
            admissible=bool(r["admissible"]),
            reason=r["reason"],
        )
        for r in rows
    ]
    gates = gate_report(
        graphs,
        node_recall=a.node_recall,
        edge_precision=a.edge_precision,
        matcher_kappa=a.matcher_kappa,
        contamination_delta=a.contamination_delta,
        headline_effect=a.headline_effect,
    )
    n_adm = sum(1 for g in graphs if g.admissible)
    print(f"suite={a.suite} mined_tasks={len(graphs)} admissible={n_adm}")
    print()
    for g in gates:
        verdict = "PASS" if g.passed else "FAIL"
        print(f"[{verdict}] {g.name}: {g.value:.4f} (threshold {g.threshold:.4f})")
        print(f"         consequence: {g.consequence}")
    for name, flag, need in _NOT_RUN:
        if getattr(a, flag) is None:
            print(f"[NOT RUN] {name}")
            print(f"         needs: {need}")
    print()
    failed = [g.name for g in gates if not g.passed]
    if failed:
        print(f"EXIT 1: {len(failed)} gate(s) FAILED: {failed}. Apply the consequence above.")
        return 1
    print(f"{len(gates)} gate(s) ran, all passed; {len(_NOT_RUN) - _n_supplied(a)} not run.")
    return 0


# Gates whose input is a MEASUREMENT this repository cannot take on its own. Printed as NOT
# RUN with the input they need, because a gate list that silently shrinks when the data is
# missing is a gate list that always passes.
_NOT_RUN: tuple[tuple[str, str, str], ...] = (
    (
        "G-M1 node recall vs human gold",
        "node_recall",
        "--node-recall: the mined node set scored against a human-annotated sample of the "
        "same tasks. No human gold sample exists in this repository yet.",
    ),
    (
        "G-M1 edge precision vs human gold",
        "edge_precision",
        "--edge-precision: mined edges adjudicated against human-annotated prerequisites.",
    ),
    (
        "G-M2 matcher kappa",
        "matcher_kappa",
        "--matcher-kappa: inter-annotator agreement between the matcher and a human on the "
        "same (ask, node) pairs.",
    ),
    (
        "G-M3 contamination delta",
        "contamination_delta",
        "--contamination-delta together with --headline-effect: the headline contrast "
        "recomputed on leave-one-family-out mined graphs, minus the all-families number.",
    ),
)


def _n_supplied(a: argparse.Namespace) -> int:
    return sum(1 for _, flag, _ in _NOT_RUN if getattr(a, flag) is not None)


# --------------------------------------------------------------------------- registration


def register(sub: argparse._SubParsersAction) -> None:
    """Attach `pi gold` to the top-level dispatch table."""
    p = sub.add_parser("gold", help="build, mine, validate and inventory the gold graphs")
    s = p.add_subparsers(dest="sub", required=True)

    b = s.add_parser("build", help="drive a builder and report the gold side")
    b.add_argument("--suite", choices=sorted(GOLD_SUITES))
    b.add_argument("--split")
    b.add_argument("--limit", type=int)
    b.add_argument("--root")
    b.add_argument(
        "--raw-dir", help="read pinned inputs from here (default <root>/data/raw/<suite>)"
    )
    b.add_argument("--graph-version", default=GRAPH_VERSION)
    b.add_argument("--offline", action="store_true", help="fail rather than download")
    b.set_defaults(fn=cmd_gold_build)

    m = s.add_parser("mine", help="S0..S7 over recorded runs -> a MinedGraph per task")
    m.add_argument("--suite", required=True, choices=sorted(RETRIEVAL_VARIANT))
    m.add_argument("--task", action="append", default=[], help="repeatable; default = all")
    m.add_argument("--root")
    m.add_argument("--runs-root")
    m.add_argument("--parquet", help="scores/parquet dir")
    m.add_argument("--corpus", help="the built corpus dir; needed for the discoverability cut")
    m.add_argument(
        "--metric",
        default="task_success",
        help="the scores.parquet metric that decides success (default: task_success)",
    )
    m.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="override the suite's preregistered threshold (pi_eval.mining.pool)",
    )
    m.add_argument("--theta", type=float, default=0.85, help="S2 canonicalization threshold")
    m.add_argument("--json", action="store_true")
    m.set_defaults(fn=cmd_gold_mine)

    v = s.add_parser("validate", help="the admissibility gates, each with its consequence")
    v.add_argument("--suite", required=True, choices=sorted(RETRIEVAL_VARIANT))
    v.add_argument("--root")
    v.add_argument("--node-recall", type=float, default=None)
    v.add_argument("--edge-precision", type=float, default=None)
    v.add_argument("--matcher-kappa", type=float, default=None)
    v.add_argument("--contamination-delta", type=float, default=None)
    v.add_argument("--headline-effect", type=float, default=None)
    v.set_defaults(fn=cmd_gold_validate)

    q = s.add_parser(
        "questions",
        help="gold-derived question lists for the CEILING arms (writes a --questions-from-file)",
    )
    q.add_argument("--suite", required=True, choices=sorted(GOLD_SUITES))
    q.add_argument(
        "--mode",
        required=True,
        choices=["gold_evidence", "oracle_vreq"],
        help="oracle_vreq: perfect selection+sequencing, ordinary retrieval. "
        "gold_evidence: the same plus the answer aliases, forcing retrieval onto gold.",
    )
    q.add_argument("--out", help="write JSON here; otherwise print it")
    q.add_argument(
        "--verify-recall",
        type=int,
        metavar="K",
        help="ALSO measure, for zero tokens, what share of required gold evidence these "
        "questions retrieve at k=K, against the full-question baseline. A ceiling arm "
        "whose questions miss the gold evidence is not a ceiling.",
    )
    q.add_argument("--n", type=int, default=None, help="first N task ids, for --verify-recall")
    q.add_argument("--root")
    q.add_argument("--graph-version", default=GRAPH_VERSION)
    q.set_defaults(fn=cmd_gold_questions)

    st = s.add_parser("stats", help="nodes, edges, depth, provenance and orphan rate per suite")
    st.add_argument("--suite", choices=sorted(GOLD_SUITES))
    st.add_argument("--root")
    st.add_argument("--graph-version", default=GRAPH_VERSION)
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_gold_stats)
