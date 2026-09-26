"""Is the recipe's out-of-order excess TARGETED or INCIDENTAL? precedence_violation_rate, split.

WHY. Table 3 reads the recipe (seeds 1 and 2 averaged within the task) minus the same weights
prompted as MORE out of order on MuSiQue (+0.0566, n = 178, decided). The metric
(`pi_eval.metrics.structure.precedence_violation_rate`) counts an edge u -> v as violated
when v's record carries an earlier turn than u's, whatever put it there. A child can be
resolved early because the policy ASKED FOR IT (skipping u: a targeted jump), or because a
question aimed elsewhere -- often at u itself -- happened to retrieve v's paragraph too (a
co-retrieval the policy never chose). The two say opposite things about the policy.

THE SPLIT. For every violation the metric counts, the question at the child's matched turn
(`matched_turn_idx`: for a resolved child, the turn whose retrieval completed v's evidence --
MuSiQue nodes carry exactly one evidence uid, so the turn that first surfaced it) is tested
against v with the repository's own ASK instrument, `pi_eval.matcher.base._matches`
(ASK_MIN_TERMS = 2, ASK_MIN_COVERAGE = 0.5, terms of >= 4 characters, stoplist from
`pi_eval.text`, substring test on the uppercased question, v's own terms REQUIRED). v's terms
are resolved first on MuSiQue with `pi_eval.build.musique_build.resolve_placeholders` over
`scripts/precedence_mechanism/probe.node_answers` (`#N` -> node sN's gold answer), never with
`matcher.base._resolve_refs`. StrategyQA nodes carry no answers and 2Wiki nodes no `#N`, so
there the node's own text is used as is. TARGETED if it matches, else INCIDENTAL; the two
rates sum to the metric run by run, over the metric's own denominator (edges with both ends
present in the window).

A SENSITIVITY, reported beside it: `targeted_core`, where the question contains every one of
v's OWN (unresolved) terms -- it credits "who owns X and when was that founded?", which names
v's relation without its referent, and which the primary instrument calls incidental.

THE RULE is Table 3's: pair on (suite, task, rollout seed), both arms at min(k_a, k_b),
training seeds averaged within the task, paired bootstrap over tasks (mirrored from
`scripts/seed_identity/table1_by_seed.pooled_symmetric`, and locked to it below).

THE EVENTS ARE `scripts/precedence_mechanism/events.py`'s: `events_for_arm` (the
`counts_in_matched_metric` rows) and `qualifying_edges_for_arm` (the denominator), over
`full_records`. `events.lock_check` itself pairs two arm_ids inside ONE store with the trained
arm untruncated; this population spans two stores (recipe: artifacts/seedrep_gate_20260919,
comparator: artifacts/completed_cohort_20260922), so its per-run body is applied run by run at
k = n_asks (`run_lock_row`) and compared with the scorer's STORED value.

LOCKS, before any split number:
  1. per run, for every run of the population: the event rebuild at k = n_asks equals the
     stored `precedence_violation_rate` (value and presence) and the metric function;
  2. the unsplit recipe-minus-base difference reproduces table1_by_seed.json's pooled
     precedence cell on every suite -- point, both bounds, n and n_pairs, at every recorded
     resample count and seed.

    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python scripts/edge_validity/order_split.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.precedence_mechanism import events as ev  # noqa: E402
from scripts.precedence_mechanism.probe import node_answers, score_answer  # noqa: E402

from pi_eval.build.musique_build import resolve_placeholders  # noqa: E402
from pi_eval.gold import GoldGraph, GoldNode  # noqa: E402
from pi_eval.matcher import base as mb  # noqa: E402
from pi_eval.matcher.base import MatchRecord  # noqa: E402
from pi_eval.metrics.structure import precedence_violation_rate  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402

OUT_DIR = REPO / "artifacts/edge_validity_20260923"
TABLE1 = REPO / "artifacts/seed_identity_20260923/table1_by_seed.json"
COHORT = REPO / "artifacts/completed_cohort_20260922"
SEEDREP = REPO / "artifacts/seedrep_gate_20260919"
S_NAMES = {
    "s1": "qwen3-8b-dpo-stacked-notdone-both-s1",
    "s2": "qwen3-8b-dpo-stacked-notdone-both-s2",
}
SUITES: tuple[str, ...] = ("musique", "strategyqa", "wiki2")
RATES: tuple[str, ...] = (
    "viol",
    "targeted",
    "incidental",
    "targeted_cov50",
    "incidental_cov50",
    "targeted_core",
    "incidental_core",
)
RESAMPLES: tuple[tuple[int, int], ...] = (
    (1000, 0),
    (10000, 0),
    (50000, 101),
    (50000, 202),
    (50000, 303),
)
LOCK_RESAMPLES: tuple[tuple[int, int], ...] = ((10000, 0), (50000, 101), (50000, 202), (50000, 303))
TOL = 1e-9
N_EXAMPLES = 4

MATCHER_SPEC = {
    "function": "pi_eval.matcher.base._matches(node, core, terms, question.upper())",
    "core": "pi_eval.matcher.base._terms(v.gold_text)",
    "terms": "musique: _terms(resolve_placeholders(v.gold_text, probe.node_answers(graph))); "
    "strategyqa and wiki2: core (no per-node answers / no #N)",
    "ASK_MIN_TERMS": mb.ASK_MIN_TERMS,
    "ASK_MIN_COVERAGE": mb.ASK_MIN_COVERAGE,
    "min_term_len": mb._MIN_TERM_LEN,
    "question": "turns.parquet `question` at turn_idx == the child's matched_turn_idx (the text "
    "the scorer's ASK rung reads)",
    "sensitivity_targeted_cov50": "pi_eval.matcher.base.asked_about(v with gold_text replaced "
    "by its resolved text, question.upper()): >= ASK_MIN_TERMS hits and >= ASK_MIN_COVERAGE of "
    "the resolved terms, no all-core requirement",
    "sensitivity_targeted_core": "core non-empty and every core term a substring of the "
    "uppercased question",
    "measured_property_of_the_primary": "with no #N in v's text, core == terms and `_matches` "
    "requires EVERY term (100% coverage), although its docstring says v2's >=50% rule is "
    "unchanged for such nodes; the cov50 sensitivity applies the documented rule",
    "question_names_child_answer": "probe.score_answer(question, v): contains_answer over "
    "token sequences against v's gold answer (descriptive; musique and wiki2 carry answers)",
}


# ------------------------------------------------------------------------------ the matcher


def node_terms(graph: GoldGraph, node: GoldNode, suite: str) -> tuple[set[str], set[str]]:
    """(v's own terms, v's terms with `#N` resolved). Resolution on MuSiQue only, by node id."""
    text = node.gold_text or ""
    core = mb._terms(text)
    if suite == "musique" and "#" in text:
        return core, mb._terms(resolve_placeholders(text, node_answers(graph)))
    return core, core


def names_node(graph: GoldGraph, node: GoldNode, suite: str, question: str | None) -> bool:
    core, terms = node_terms(graph, node, suite)
    return mb._matches(node, core, terms, (question or "").upper())


def names_node_cov50(graph: GoldGraph, node: GoldNode, suite: str, question: str | None) -> bool:
    """The documented v2 rule (`asked_about`: >= 2 hits, >= 50% of terms) on v's resolved text."""
    text = node.gold_text or ""
    if suite == "musique" and "#" in text:
        node = dataclasses.replace(node, gold_text=resolve_placeholders(text, node_answers(graph)))
    return mb.asked_about(node, (question or "").upper())


def names_answer(node: GoldNode, question: str | None) -> bool:
    """The question names v's own gold ANSWER (a jump past v). False where v carries none."""
    return score_answer(question or "", node) == 1.0


def names_core(node: GoldNode, question: str | None) -> bool:
    core = mb._terms(node.gold_text or "")
    q = (question or "").upper()
    return bool(core) and all(t in q for t in core)


# ------------------------------------------------------------------------------- per arm


@dataclass
class ArmRates:
    total: int = 0
    n_viol: int = 0
    n_targeted: int = 0
    n_incidental: int = 0
    n_targeted_cov50: int = 0
    n_targeted_core: int = 0
    n_incidental_names_parent: int = 0
    n_incidental_names_child_answer: int = 0
    child_kinds: Counter = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def rate(self, which: str) -> float:
        num = {
            "viol": self.n_viol,
            "targeted": self.n_targeted,
            "incidental": self.n_incidental,
            "targeted_cov50": self.n_targeted_cov50,
            "incidental_cov50": self.n_viol - self.n_targeted_cov50,
            "targeted_core": self.n_targeted_core,
            "incidental_core": self.n_viol - self.n_targeted_core,
        }[which]
        return num / self.total if self.total else float("nan")


def arm_rates(
    *,
    suite: str,
    task_id: str,
    seed: int,
    arm: str,
    run: ev.RunArm,
    basis_k: int,
    full: list[ev.NodeMatch],
    graph: GoldGraph,
    questions: Mapping[int, str],
) -> ArmRates:
    """One run in its matched window (turns < basis_k): the metric's violations, each labelled."""
    kw = dict(suite=suite, task_id=task_id, seed=seed, arm=arm, run=run, basis_k=basis_k)
    events = ev.events_for_arm(**kw, full=full, graph=graph, scores={})
    total = len(ev.qualifying_edges_for_arm(**kw, full=full, graph=graph))
    nodes = {n.gold_node_id: n for n in graph.gold_nodes}
    out = ArmRates(total=total)
    for e in events:
        if not e.counts_in_matched_metric:
            continue
        q = questions.get(e.child_turn, "")
        child, parent = nodes[e.child_node_id], nodes[e.parent_node_id]
        tgt = names_node(graph, child, suite, q)
        par = names_node(graph, parent, suite, q)
        core = names_core(child, q)
        cov50 = names_node_cov50(graph, child, suite, q)
        ans = names_answer(child, q)
        out.n_viol += 1
        out.n_targeted += tgt
        out.n_incidental += not tgt
        out.n_targeted_core += core
        out.n_targeted_cov50 += cov50
        out.n_incidental_names_parent += (not tgt) and par
        out.n_incidental_names_child_answer += (not tgt) and ans
        out.child_kinds[e.child_match_kind] += 1
        out.events.append(
            {
                "run_id": run.run_id,
                "task_id": task_id,
                "seed": seed,
                "basis_k": basis_k,
                "parent": e.parent_node_id,
                "child": e.child_node_id,
                "child_turn": e.child_turn,
                "child_kind": e.child_match_kind,
                "question": q,
                "targeted": tgt,
                "targeted_core": core,
                "targeted_cov50": cov50,
                "question_names_parent": par,
                "question_names_child_answer": ans,
            }
        )
    return out


def run_lock_row(
    *, full: list[ev.NodeMatch], n_asks: int, graph: GoldGraph, suite: str
) -> tuple[float, float]:
    """`events.lock_check`'s per-run body at basis k = n_asks: (event rebuild, metric fn)."""
    run = ev.RunArm("lock", n_asks)
    truncated = [m for m in full if m.turn < n_asks]
    records = [
        MatchRecord(
            run_id=run.run_id,
            suite_id=suite,
            task_id=graph.gold_task_key,
            node_id=m.node_id,
            match_kind=m.match_kind,
            matched_turn_idx=m.turn,
            matcher_id="mechanical_v3",
            matcher_family="mechanical",
            matcher_score=1.0,
            threshold=0.0,
            graph_version=graph.gold_graph_version,
        )
        for m in truncated
    ]
    official = precedence_violation_rate(records, graph)
    events = ev.events_for_arm(
        suite=suite,
        task_id=graph.gold_task_key,
        seed=0,
        arm="lock",
        run=run,
        basis_k=n_asks,
        full=full,
        graph=graph,
        scores={},
    )
    counted = [e for e in events if e.counts_in_matched_metric]
    present = {m.node_id for m in truncated}
    total = sum(1 for p, c in ev._prereq_edges(graph) if present >= {p, c})
    rebuilt = len(counted) / total if total else float("nan")
    return rebuilt, official


# ------------------------------------------------------------------------------ population


@dataclass(frozen=True)
class Run:
    run_id: str
    task_id: str
    seed: int
    n_asks: int
    store: Path


def _ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _digest(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for rid in sorted(ids):
        h.update(rid.encode())
        h.update(b"\n")
    return h.hexdigest()


def load_suite(con, suite: str) -> dict[str, Any]:
    ids = {
        "base": (_ids(COHORT / "cohort" / f"run_ids.prompted.{suite}.txt"), COHORT),
        **{
            k: (_ids(SEEDREP / "run_ids" / f"run_ids.{v}.{suite}.txt"), SEEDREP)
            for k, v in S_NAMES.items()
        },
    }
    arms: dict[str, dict[str, Run]] = {}
    questions: dict[str, dict[int, str]] = {}
    stored: dict[str, float] = {}
    full: dict[str, list[ev.NodeMatch]] = {}
    templates: dict[str, str] = {}
    for arm, (rids, root) in ids.items():
        store = root / "scores_parquet"
        rows = con.execute(
            "SELECT run_id, task_id, seed, n_asks, template_id FROM read_parquet(?) "
            "WHERE run_id IN (SELECT unnest(?))",
            [str(store / "runs.parquet"), rids],
        ).fetchall()
        if len(rows) != len(rids):
            raise SystemExit(f"{suite} {arm}: {len(rows)} runs.parquet rows for {len(rids)} ids")
        arms[arm] = {r[0]: Run(r[0], r[1], int(r[2]), int(r[3] or 0), store) for r in rows}
        if arm == "base":
            templates = {r[1]: r[4] for r in rows if r[4]}
        for rid, turn, q in con.execute(
            "SELECT run_id, turn_idx, question FROM read_parquet(?) WHERE run_id IN "
            "(SELECT unnest(?))",
            [str(store / "turns.parquet"), rids],
        ).fetchall():
            questions.setdefault(rid, {})[int(turn)] = q or ""
        for rid, val in con.execute(
            "SELECT run_id, value FROM read_parquet(?) WHERE run_id IN (SELECT unnest(?)) "
            "AND metric_name = 'precedence_violation_rate'",
            [str(store / "scores.parquet"), rids],
        ).fetchall():
            stored[rid] = float(val)
        for r in arms[arm].values():
            full[r.run_id] = ev.full_records(con, str(store), r.run_id, suite, r.task_id)
    return {
        "arms": arms,
        "questions": questions,
        "stored": stored,
        "full": full,
        "templates": templates,
        "ids": {k: v[0] for k, v in ids.items()},
    }


# ------------------------------------------------------------------------------- pairing


def pooled_rates(
    suite: str,
    arms: Sequence[Mapping[str, Run]],
    base: Mapping[str, Run],
    data: Mapping[str, Any],
    graphs: Mapping[str, GoldGraph],
) -> dict[str, Any]:
    """table1_by_seed.pooled_symmetric's pairing, over the five rates at once."""

    def keyed(arm: Mapping[str, Run], what: str) -> dict[tuple[str, str, int], Run]:
        out = {}
        for r in arm.values():
            key = (suite, r.task_id, r.seed)
            if key in out:
                raise ValueError(f"{what}: two runs at {key}")
            out[key] = r
        return out

    b_by = keyed(base, "comparator")
    acc: dict[str, dict[str, tuple[list[float], list[float]]]] = {r: {} for r in RATES}
    tot = {"recipe": ArmRates(), "base": ArmRates()}
    n_pairs = 0
    for i, arm in enumerate(arms):
        a_by = keyed(arm, f"training seed #{i}")
        for key in sorted(set(a_by) & set(b_by)):
            graph = graphs.get(key[1])
            if graph is None:
                continue
            ra_, rb_ = a_by[key], b_by[key]
            k = min(ra_.n_asks, rb_.n_asks)
            got = {}
            for side, r in (("recipe", ra_), ("base", rb_)):
                got[side] = arm_rates(
                    suite=suite,
                    task_id=r.task_id,
                    seed=r.seed,
                    arm=side,
                    run=ev.RunArm(r.run_id, r.n_asks),
                    basis_k=k,
                    full=data["full"][r.run_id],
                    graph=graph,
                    questions=data["questions"].get(r.run_id, {}),
                )
            if got["recipe"].total == 0 or got["base"].total == 0:
                continue  # the metric is NaN on one side: pooled_symmetric drops the pair
            n_pairs += 1
            for rate in RATES:
                la, lb = acc[rate].setdefault(key[1], ([], []))
                la.append(got["recipe"].rate(rate))
                lb.append(got["base"].rate(rate))
            for side in ("recipe", "base"):
                t, g = tot[side], got[side]
                t.total += g.total
                t.n_viol += g.n_viol
                t.n_targeted += g.n_targeted
                t.n_incidental += g.n_incidental
                t.n_targeted_core += g.n_targeted_core
                t.n_targeted_cov50 += g.n_targeted_cov50
                t.n_incidental_names_parent += g.n_incidental_names_parent
                t.n_incidental_names_child_answer += g.n_incidental_names_child_answer
                t.child_kinds.update(g.child_kinds)
                t.events.extend(g.events)
    per = {
        rate: (
            {t: sum(a) / len(a) for t, (a, _) in acc[rate].items()},
            {t: sum(b) / len(b) for t, (_, b) in acc[rate].items()},
        )
        for rate in RATES
    }
    return {"per_task": per, "n_pairs": n_pairs, "totals": tot}


def _verdict(readings: Sequence[Mapping[str, Any]]) -> str:
    fifty = [x for x in readings if x["n_boot"] == 50000]
    if all(x["ci_lo"] > 0 for x in fifty) or all(x["ci_hi"] < 0 for x in fifty):
        return "DECIDED"
    return "SPANS_ZERO"


def _readings(per_a, per_b, resamples, clusters=None) -> list[dict[str, Any]]:
    out = []
    for nb, sd in resamples:
        est = paired_difference(per_a, per_b, clusters=clusters, n_boot=nb, seed=sd)
        out.append(
            {
                "n_boot": nb,
                "seed": sd,
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                "p_value": est.p_value,
            }
        )
    return out


def _totals_json(t: ArmRates) -> dict[str, Any]:
    return {
        "n_qualifying_edges": t.total,
        "n_violations": t.n_viol,
        "n_targeted": t.n_targeted,
        "n_incidental": t.n_incidental,
        "n_targeted_cov50": t.n_targeted_cov50,
        "n_targeted_core": t.n_targeted_core,
        "n_incidental_question_names_parent": t.n_incidental_names_parent,
        "n_incidental_question_names_child_answer": t.n_incidental_names_child_answer,
        "share_targeted": t.n_targeted / t.n_viol if t.n_viol else None,
        "child_match_kinds": dict(sorted(t.child_kinds.items())),
    }


# ------------------------------------------------------------------------------------ main


def run(suites: Sequence[str], out_json: Path, out_md: Path) -> dict[str, Any]:
    import duckdb

    from pi_eval.gold import load_graphs

    t0 = time.time()
    con = duckdb.connect()
    pub = json.loads(TABLE1.read_text())["cells"]["s1s2"]
    result: dict[str, Any] = {
        "question": "is the recipe's out-of-order excess TARGETED (the question at the child's "
        "turn names the child) or INCIDENTAL (co-retrieved by a question aimed elsewhere)?",
        "rule": "pair (suite, task, rollout seed); both arms at min(k_a, k_b); training seeds 1+2 "
        "averaged within the task; paired bootstrap over tasks (Table 3's rule)",
        "matcher": MATCHER_SPEC,
        "provenance": {},
        "lock_runs": {},
        "lock_table3": {},
        "suites": {},
    }
    scorer = sorted(
        {
            r[0]
            for st in (COHORT, SEEDREP)
            for r in con.execute(
                "SELECT DISTINCT scorer_hash FROM read_parquet(?)",
                [str(st / "scores_parquet" / "scores.parquet")],
            ).fetchall()
        }
    )
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    result["provenance"] = {
        "scorer_hash": scorer,
        "graph_version": "v1",
        "git_head": head,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "events_sha256": hashlib.sha256(Path(ev.__file__).read_bytes()).hexdigest(),
        "stores": {
            "comparator": "artifacts/completed_cohort_20260922/scores_parquet",
            "recipe": "artifacts/seedrep_gate_20260919/scores_parquet",
        },
        "run_id_sha256": {},
    }
    lock_ok = True
    for suite in suites:
        graphs = load_graphs(suite, "v1")
        data = load_suite(con, suite)
        result["provenance"]["run_id_sha256"][suite] = {
            k: {"n": len(v), "sha256": _digest(v)} for k, v in data["ids"].items()
        }
        # LOCK 1: every run, k = n_asks, against the scorer's stored value and the metric fn.
        lk: dict[str, Any] = {}
        for arm, runs in data["arms"].items():
            n = agree = 0
            bad = []
            for r in runs.values():
                graph = graphs.get(r.task_id)
                if graph is None:
                    bad.append({"run_id": r.run_id, "why": "no graph"})
                    continue
                n += 1
                rebuilt, official = run_lock_row(
                    full=data["full"][r.run_id], n_asks=r.n_asks, graph=graph, suite=suite
                )
                stored = data["stored"].get(r.run_id)
                if math.isnan(rebuilt):
                    same = stored is None and math.isnan(official)
                else:
                    same = (
                        stored is not None
                        and abs(rebuilt - stored) <= TOL
                        and abs(rebuilt - official) <= TOL
                    )
                agree += same
                if not same:
                    bad.append(
                        {
                            "run_id": r.run_id,
                            "rebuilt": rebuilt,
                            "stored": stored,
                            "official": official,
                        }
                    )
            lk[arm] = {"n_runs": n, "n_matched": agree, "mismatches": bad[:10]}
            lock_ok &= agree == n == len(runs)
        result["lock_runs"][suite] = lk

        pooled = pooled_rates(
            suite, [data["arms"]["s1"], data["arms"]["s2"]], data["arms"]["base"], data, graphs
        )
        # LOCK 2: the unsplit difference is Table 3's pooled out-of-order cell.
        per_a, per_b = pooled["per_task"]["viol"]
        rows = []
        for nb, sd in LOCK_RESAMPLES:
            est = paired_difference(per_a, per_b, n_boot=nb, seed=sd)
            want = next(
                x
                for x in pub[f"precedence_violation_rate::{suite}"]
                if x["n_boot"] == nb and x["seed"] == sd
            )
            got = {
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "n": est.n,
                "n_pairs": pooled["n_pairs"],
            }
            diff = max(abs(got[f] - want[f]) for f in ("delta", "ci_lo", "ci_hi"))
            ok = diff <= TOL and got["n"] == want["n"] and got["n_pairs"] == want["n_pairs"]
            lock_ok &= ok
            rows.append(
                {
                    "n_boot": nb,
                    "seed": sd,
                    "reproduced": got,
                    "published": {f: want[f] for f in ("delta", "ci_lo", "ci_hi", "n", "n_pairs")},
                    "max_abs_diff": diff,
                    "ok": ok,
                }
            )
        result["lock_table3"][suite] = rows
        if not lock_ok:
            result["suites"][suite] = {"withheld": "lock failed"}
            continue

        cells: dict[str, Any] = {}
        clusters = data["templates"] or None
        for rate in RATES:
            pa, pb = pooled["per_task"][rate]
            readings = _readings(pa, pb, RESAMPLES)
            cell = {
                "readings": readings,
                "verdict": _verdict(readings),
                "arm_levels": {
                    "recipe": statistics.fmean(pa[t] for t in sorted(pa)),
                    "base": statistics.fmean(pb[t] for t in sorted(pb)),
                },
                "near_zero_bound": any(
                    min(abs(x["ci_lo"]), abs(x["ci_hi"])) < 0.01 for x in readings
                ),
            }
            if clusters:
                cr = _readings(pa, pb, RESAMPLES[1:], clusters=clusters)
                cell["clustered_template"] = {"readings": cr, "verdict": _verdict(cr)}
            cells[rate] = cell
        per_seed = {}
        for s in ("s1", "s2"):
            one = pooled_rates(suite, [data["arms"][s]], data["arms"]["base"], data, graphs)
            per_seed[s] = {
                "n_pairs": one["n_pairs"],
                **{
                    rate: paired_difference(*one["per_task"][rate], n_boot=1000, seed=0).point
                    for rate in RATES
                },
            }
        ex: dict[str, list[dict[str, Any]]] = {"targeted": [], "incidental": []}
        for e in sorted(pooled["totals"]["recipe"].events, key=lambda e: (e["run_id"], e["child"])):
            lab = "targeted" if e["targeted"] else "incidental"
            if len(ex[lab]) < N_EXAMPLES:
                ex[lab].append(e)
        result["suites"][suite] = {
            "n_tasks": len(pooled["per_task"]["viol"][0]),
            "n_pairs": pooled["n_pairs"],
            "cells": cells,
            "per_seed_points": per_seed,
            "totals_over_pairs": {k: _totals_json(v) for k, v in pooled["totals"].items()},
            "examples_recipe": ex,
        }
        print(f"[{time.time() - t0:.0f}s] {suite} done", flush=True)
    result["lock_ok"] = bool(lock_ok)
    result["runtime_s"] = time.time() - t0
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    out_md.write_text(render_md(result))
    return result


def _cell_line(name: str, c: Mapping[str, Any]) -> str:
    r = {(x["n_boot"], x["seed"]): x for x in c["readings"]}
    r10, r1 = r[(10000, 0)], r[(1000, 0)]
    fifty = [r[(50000, s)] for s in (101, 202, 303)]
    lo50 = ", ".join(f"[{x['ci_lo']:+.6f}, {x['ci_hi']:+.6f}]" for x in fifty)
    lv = c["arm_levels"]
    return (
        f"| {name} | {lv['recipe']:.6f} | {lv['base']:.6f} | {r10['delta']:+.6f} "
        f"[{r10['ci_lo']:+.6f}, {r10['ci_hi']:+.6f}] | [{r1['ci_lo']:+.6f}, {r1['ci_hi']:+.6f}] "
        f"| {lo50} | {c['verdict']} |"
    )


def _side(x: Mapping[str, Any]) -> str:
    return "above zero" if x["ci_lo"] > 0 else "below zero" if x["ci_hi"] < 0 else "spans zero"


def _stability(c: Mapping[str, Any]) -> str:
    """Where the interval sits at 1k, 10k and the three 50k reads, and whether that moves."""
    sides = [_side(x) for x in c["readings"]]
    near = min(min(abs(x["ci_lo"]), abs(x["ci_hi"])) for x in c["readings"]) < 0.01
    moves = "STABLE" if len(set(sides)) == 1 else "MOVES WITH THE RESAMPLE COUNT"
    return f"{moves} ({'/'.join(sides)}){'; a bound within 0.01 of zero' if near else ''}"


def summary_lines(res: Mapping[str, Any]) -> list[str]:
    """The verdicts, read off the json. No sentence here is typed by hand about a number."""
    out = ["## Summary (generated from the json)", ""]
    pairs = (
        ("primary", "targeted", "incidental"),
        ("cov50", "targeted_cov50", "incidental_cov50"),
        ("core", "targeted_core", "incidental_core"),
    )
    for s, sres in res["suites"].items():
        if "cells" not in sres:
            continue
        c = sres["cells"]
        v10 = next(x for x in c["viol"]["readings"] if x["n_boot"] == 10000)
        out.append(
            f"- **{s}** unsplit recipe-minus-base {v10['delta']:+.4f} "
            f"[{v10['ci_lo']:+.4f}, {v10['ci_hi']:+.4f}], {c['viol']['verdict']} at 50k."
        )
        for inst, t, i in pairs:
            t10 = next(x for x in c[t]["readings"] if x["n_boot"] == 10000)
            i10 = next(x for x in c[i]["readings"] if x["n_boot"] == 10000)
            share = i10["delta"] / v10["delta"] if v10["delta"] else float("nan")
            out.append(
                f"  - {inst}: targeted {t10['delta']:+.4f} [{t10['ci_lo']:+.4f}, "
                f"{t10['ci_hi']:+.4f}] {c[t]['verdict']}, {_stability(c[t])}; incidental "
                f"{i10['delta']:+.4f} [{i10['ci_lo']:+.4f}, {i10['ci_hi']:+.4f}] "
                f"{c[i]['verdict']}, {_stability(c[i])}; incidental / unsplit point = {share:.2f}"
            )
    out.append("")
    return out


def render_md(res: Mapping[str, Any]) -> str:
    p = res["provenance"]
    lines = [
        "# Out-of-order resolution, split: TARGETED vs INCIDENTAL (lane L1b, T1e)",
        "",
        "Generated by `scripts/edge_validity/order_split.py` from `order_split.json` (every "
        "number below is read from that file). Command:",
        "",
        "    PI_GOLD_ROOT=$PWD/data/gold .venv/bin/python scripts/edge_validity/order_split.py",
        "",
        f"Provenance: scorer_hash {', '.join(p['scorer_hash'])}; graph_version {p['graph_version']}; "
        f"git HEAD {p['git_head']}; recipe store `{p['stores']['recipe']}`, comparator store "
        f"`{p['stores']['comparator']}`; run-id digests per suite and arm in the json "
        "(`provenance.run_id_sha256`).",
        "",
        "## Question and instrument",
        "",
        res["question"],
        "",
        f"Rule: {res['rule']}.",
        "",
        "TARGETED = the question at the child's matched turn names the child under the repo's "
        f"ASK instrument `{MATCHER_SPEC['function']}` (ASK_MIN_TERMS = "
        f"{MATCHER_SPEC['ASK_MIN_TERMS']}, ASK_MIN_COVERAGE = {MATCHER_SPEC['ASK_MIN_COVERAGE']}, "
        f"terms >= {MATCHER_SPEC['min_term_len']} chars, the node's own terms required); MuSiQue "
        "`#N` resolved by `resolve_placeholders` over `probe.node_answers` (node id sN, never "
        "`_resolve_refs`). INCIDENTAL = every other counted violation. targeted + incidental = "
        "the metric, run by run.",
        "",
        "A MEASURED property of the primary instrument: for a node whose text has no `#N`, "
        "`_matches` gets core == terms and requires EVERY term (100% coverage), although its "
        "docstring says v2's >= 50% rule is unchanged for such nodes. It therefore under-credits "
        "long natural-language needs (all StrategyQA and 2Wiki nodes, and MuSiQue's depth-0 "
        "question nodes). Two sensitivities, each with its incidental complement: "
        "`targeted_cov50` = `pi_eval.matcher.base.asked_about` (the documented rule: >= 2 hits "
        "and >= 50% of the terms) on the child's resolved text; `targeted_core` = every one of "
        "the child's OWN (unresolved) terms appears in the question.",
        "",
        "## Locks",
        "",
        "Per run (k = n_asks; events rebuild vs the stored `precedence_violation_rate`, value and "
        "presence, and vs the metric function):",
        "",
        "| suite | arm | n matched / n |",
        "|---|---|---|",
    ]
    for s, lk in res["lock_runs"].items():
        for arm, v in lk.items():
            lines.append(f"| {s} | {arm} | {v['n_matched']} / {v['n_runs']} |")
    lines += [
        "",
        "Unsplit recipe-minus-base against `artifacts/seed_identity_20260923/table1_by_seed.json` "
        "(cells.s1s2, precedence_violation_rate):",
        "",
        "| suite | n_boot@seed | published | reproduced | max abs diff | n / n_pairs | ok |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, rows in res["lock_table3"].items():
        for r in rows:
            pu, g = r["published"], r["reproduced"]
            lines.append(
                f"| {s} | {r['n_boot']}@{r['seed']} | {pu['delta']!r} [{pu['ci_lo']!r}, "
                f"{pu['ci_hi']!r}] | {g['delta']!r} [{g['ci_lo']!r}, {g['ci_hi']!r}] | "
                f"{r['max_abs_diff']:.3g} | {g['n']} / {g['n_pairs']} | {r['ok']} |"
            )
    lines += ["", f"All locks: {'PASS' if res['lock_ok'] else 'FAIL'}.", ""]
    lines += summary_lines(res)
    for s, sres in res["suites"].items():
        if "cells" not in sres:
            lines += [f"## {s}: withheld ({sres.get('withheld')})", ""]
            continue
        lines += [
            f"## {s} ({sres['n_tasks']} tasks, {sres['n_pairs']} pairs)",
            "",
            "Rates are per qualifying edge (the metric's own denominator). Levels are means over "
            "tasks; the difference is recipe minus base.",
            "",
            "| rate | recipe | base | diff [10k, seed 0] | 1k | 50k @101/202/303 | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for rate in RATES:
            lines.append(_cell_line(rate, sres["cells"][rate]))
        if any("clustered_template" in c for c in sres["cells"].values()):
            lines += ["", "Template-clustered (secondary; Table 3 does not cluster):", ""]
            for rate in RATES:
                cr = sres["cells"][rate]["clustered_template"]
                r10 = cr["readings"][0]
                lines.append(
                    f"- {rate}: {r10['delta']:+.6f} [{r10['ci_lo']:+.6f}, {r10['ci_hi']:+.6f}] "
                    f"(10k), {cr['verdict']} at 50k"
                )
        lines += ["", "Per training seed (points, same rule, one seed at a time):", ""]
        for sd, v in sres["per_seed_points"].items():
            vals = ", ".join(f"{rate} {v[rate]:+.6f}" for rate in RATES)
            lines.append(f"- {sd} ({v['n_pairs']} pairs): {vals}")
        lines += ["", "Event totals over the pairs (the base arm is counted once per pairing):", ""]
        for side, t in sres["totals_over_pairs"].items():
            lines.append(
                f"- {side}: {t['n_violations']} violations over {t['n_qualifying_edges']} "
                f"qualifying edges; targeted {t['n_targeted']}, incidental {t['n_incidental']} "
                "(of which the question names the PARENT: "
                f"{t['n_incidental_question_names_parent']}"
                "; the question names the child's gold ANSWER (a jump past it): "
                f"{t['n_incidental_question_names_child_answer']}); "
                f"targeted_cov50 {t['n_targeted_cov50']}; targeted_core {t['n_targeted_core']}; "
                f"child match kinds {t['child_match_kinds']}"
            )
        lines.append("")
    lines += [f"Runtime {res['runtime_s']:.0f} s.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "order_split.json")
    ap.add_argument("--md", type=Path, default=OUT_DIR / "ORDER_SPLIT.md")
    args = ap.parse_args(argv)
    if not os.environ.get("PI_GOLD_ROOT"):
        print("order_split: REFUSING: PI_GOLD_ROOT unset", file=sys.stderr)
        return 3
    res = run(args.suites, args.out, args.md)
    print(render_md(res))
    return 0 if res["lock_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
