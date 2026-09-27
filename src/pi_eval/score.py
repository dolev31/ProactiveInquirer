"""parquet + gold -> scores/parquet/{matches,scores}.parquet. The SCORING stage.

This is the one stage that reads gold, so it is the one stage that must run in a process
where `PI_GOLD_ROOT` is set. `gold_root()` raising anywhere else is the firewall working.

THE APPEND RULE. `scorer_hash = h(metric_defs_hash, graph_hash, matcher_hash, judge_pins)`.
Re-scoring under a new scorer_hash ADDS rows; it never mutates an existing one and never
re-rolls a rollout. That is not tidiness, it is the only way a reviewer can ask "what did
this table say before you changed the matcher?" and get an answer instead of a story. It also
means the expensive thing (rollouts, ~$14k) is decoupled from the cheap thing (scoring, a
groupby), so a metric bug costs minutes rather than a re-run.

WHAT Q IS, ON THE PARQUET-ONLY PATH. Evidence text is deliberately not stored (the corpus is
content-addressed and frozen), so the quality function used for phi_LOO, the frontier ladder
and the stopping test is

    Q(E) = |E n gold_ev_uids| / |gold_ev_uids|      -- `evidence_coverage`

a pure map from an evidence subset to a scalar, exactly the shape `pi_eval.metrics.qvalue`
wants, computable with zero LLM calls. Answer-text metrics (`answer_token_f1`,
`answer_exact_match`) and the USE rung of the ladder need `runs/<run_id>/outcome.json`; when
that directory is absent those rows are NOT EMITTED, rather than emitted as zero. A missing
measurement must look missing, because a fabricated zero averages into a table.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as _exc:  # pragma: no cover - names the extra instead of the module
    from pinq.extras import missing

    raise missing("pyarrow", why="pi score reads and writes parquet") from _exc

from pi_eval import schema as sch
from pi_eval.gold import GoldAccessError, GoldGraph, gold_root, load_graphs
from pi_eval.matcher.base import MatchRecord, MechanicalMatcher
from pi_eval.metrics import (
    discovery,
    environment,
    human,
    latent,
    quality,
    stopping,
    structure,
)
from pinq import tripwire
from pinq.budget import HARD_CURRENCY
from pinq.ids import canon, h

# Bump this whenever a metric's DEFINITION changes without its register entry changing
# (a different Q, a different null, a different denominator). It is what forces a new
# scorer_hash, and therefore new rows, instead of silently relabelling old ones.
METRIC_DEFS_VERSION = "v1"
DEFAULT_GRAPH_VERSION = "v1"
NAN = float("nan")

# Depth weights for DWR. PREREGISTERED: chosen before any arm ran, sealed here rather than
# in a notebook, because weights picked after seeing the data are a free parameter.
DWR_WEIGHTS: dict[int, float] = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0}


# Suites scored on their ANSWERS ONLY, because they ship no need-graph and never will.
#
# A LITERAL, NOT "has no graphs on disk". Those two conditions look identical from inside
# `score()` and mean opposite things: one is a benchmark whose design carries no need-graph,
# the other is a gold build that failed or was never run. The first must be scored on what it
# does have; the second must stay a hard skip that shows up in `n_runs_skipped_no_graph`.
# Inferring the branch from the data would silently convert every broken gold build into a
# quietly-degraded table.
#
# `pi_run.suites` keeps the declaration a human made (`SuiteSpec.gold_free`) and its `audit()`
# checks the two agree. Neither module imports the other; the set is passed in, exactly as the
# preregistration's is.
GOLD_FREE_SUITES: frozenset[str] = frozenset({"frames"})


class ScoringError(RuntimeError):
    """Fatal to the scoring stage. Never downgraded to a warning."""


# --------------------------------------------------------------------------- metric register


@dataclass(frozen=True, slots=True)
class MetricDef:
    """One declared metric.

    `judge_derived` is what gates the noise floor: only a judge-derived effect can be below
    `2*sigma_J/sqrt(n)`, and until sigma_J is measured and pinned every such effect is
    flagged rather than claimed. `indexed` metrics are emitted as `name#k` families (the
    frontier ladder, the per-turn phi values) so a curve survives into a row-oriented table.
    """

    name: str
    family: str  # quality | discovery | structure | qvalue | human | operational | frontier
    binary: bool = False
    judge_derived: bool = False
    higher_is_better: bool = True
    indexed: bool = False
    needs_answer: bool = False
    doc: str = ""


METRICS: tuple[MetricDef, ...] = (
    # ---- quality
    MetricDef("evidence_coverage", "quality", doc="|E n gold spans| / |gold spans|"),
    MetricDef("answer_token_f1", "quality", needs_answer=True, doc="token F1 vs gold answer"),
    MetricDef("answer_exact_match", "quality", binary=True, needs_answer=True, doc="normalized EM"),
    # The primary, decomposed against the SAME alias it scored. Recall is the length-robust
    # half: appending words cannot lower it. Measured on 164 musique runs, answer_token_f1
    # is rank-correlated with answer length at -0.677 pooled and -0.934 within
    # `inquirer_prompted` alone, so an F1 gap between arms that differ by 7 mean words is
    # not separable from a brevity gap. Reporting recall beside it is what makes the primary
    # interpretable rather than arguable.
    MetricDef(
        "answer_token_recall",
        "quality",
        needs_answer=True,
        doc="LENGTH-ROBUST half of the primary: gold tokens the answer contained",
    ),
    MetricDef(
        "answer_token_precision",
        "quality",
        needs_answer=True,
        doc="the half length destroys; f1 = harmonic mean of this and recall",
    ),
    # NAMED FOR WHAT IT MEASURES. `task_success` is `evidence_coverage >= 1.0` binarised --
    # every required need RESOLVED -- and is NOT answer correctness. The name has misled
    # readers of this file (and one plan) into treating it as an answer endpoint; the doc
    # string is the correction, and `answer_correct` below is the thing it was mistaken for.
    MetricDef("task_success", "quality", binary=True, doc="every required need resolved"),
    MetricDef(
        "answer_hedged",
        "quality",
        needs_answer=True,
        binary=True,
        doc="the Answerer declined to commit ('the evidence does not specify')",
    ),
    MetricDef(
        "answer_correct",
        "quality",
        needs_answer=True,
        binary=True,
        doc="gold answer (or an alias) appears in the answer; hedge- and length-robust",
    ),
    # ---- judge-derived (DeepResearchGym). Emitted by the judge harness inside score(), and
    # only when a judge client is configured; see `run_judges`. Every one of these is
    # judge_derived=True, which is what puts it behind the sigma_J noise floor in report.py.
    MetricDef(
        "kpr_incremental",
        "quality",
        judge_derived=True,
        needs_answer=True,
        doc="P3 PRIMARY: key points with >=1 Supported verdict / key points judged",
    ),
    MetricDef(
        "keypoint_recall",
        "quality",
        judge_derived=True,
        needs_answer=True,
        doc="RAW key-point recall: Supported judgments / all judgments. Comparable to upstream",
    ),
    MetricDef(
        "keypoint_contradiction_rate",
        "quality",
        judge_derived=True,
        needs_answer=True,
        higher_is_better=False,
        doc="judged key points the report contradicts; a contradiction is not an omission",
    ),
    MetricDef(
        "kpr_repetition_gap",
        "quality",
        judge_derived=True,
        needs_answer=True,
        higher_is_better=False,
        doc="raw - incremental; zero exactly when every key point got exactly one verdict",
    ),
    MetricDef(
        "citation_support",
        "quality",
        judge_derived=True,
        needs_answer=True,
        doc="mean of 1.0/0.5/0.0 over CHECKED claims; unresolvable sources are not scored 0",
    ),
    *(
        MetricDef(
            f"quality_{k}",
            "quality",
            judge_derived=True,
            needs_answer=True,
            doc=f"DRGym report rubric, {k}, raw 0-10",
        )
        for k in ("clarity", "depth", "balance", "breadth", "support", "insightfulness")
    ),
    # ---- discovery ladder
    MetricDef("rnr_ask", "discovery", doc="required-need recall, ASK rung"),
    MetricDef("rnr_resolve", "discovery", doc="required-need recall, RESOLVE rung"),
    MetricDef("rnr_use", "discovery", needs_answer=True, doc="RESOLVE rung + cited in answer"),
    MetricDef("n_needs_resolved", "discovery", doc="count of gold nodes at rank >= resolve"),
    # ---- structure
    MetricDef("cad", "structure", indexed=True, doc="C@d, indexed by depth"),
    MetricDef("cad_n", "structure", indexed=True, doc="|V_d|, printed in every C@d cell"),
    MetricDef("dwr", "structure", doc="sum_d w_d C@d, weights preregistered in DWR_WEIGHTS"),
    MetricDef("max_depth_reached", "structure", doc="deepest resolved node, -1 if none"),
    MetricDef(
        "precedence_violation_rate", "structure", higher_is_better=False, doc="gold DAG order"
    ),
    MetricDef("facet_breadth", "structure", doc="distinct gold facets touched"),
    # HORIZONTAL, where there are no facets. tau2 carries 0 on every graph (43/43 airline,
    # 112/112 retail), so facet_breadth reads as "no breadth" for the whole family while
    # vertical is well populated on the same graphs. Components of the prerequisite DAG carry
    # the same information without an annotation tau2 does not have.
    MetricDef(
        "breadth_components",
        "structure",
        doc="independent lines of inquiry resolved (weakly-connected prerequisite components)",
    ),
    MetricDef(
        "breadth_recall",
        "structure",
        doc="breadth_components / components available; absent when the graph has no chains",
    ),
    # A HEALTH CHECK ON THE ABOVE, not a proactivity claim. 90% of tau2 airline components are
    # singletons of unknown provenance; a breadth number over those measures the gold builder.
    MetricDef(
        "breadth_singleton_share",
        "structure",
        higher_is_better=False,
        doc="share of components that are lone nodes; high means breadth is measuring gaps",
    ),
    MetricDef("facet_total", "structure", doc="|facets| in the gold graph"),
    MetricDef("orphan_rate", "structure", higher_is_better=False, doc="nodes with no depth"),
    # ---- latent needs. THE VERTICAL CLAIM, measured directly rather than stratified: of the
    # needs that became askable only after their prerequisites were resolved, how many did the
    # policy take? Judge-free, so outside the sigma_J dead band. Exploratory -- see
    # report.EXPLORATORY_METRICS; nothing is added to the BH family.
    MetricDef(
        "latent_discovery_rate",
        "latent",
        doc="latent needs resolved / latent needs that ever became reachable",
    ),
    MetricDef(
        "latent_available", "latent", doc="latent needs that ever entered the frontier (the n)"
    ),
    MetricDef(
        "newly_reachable_share",
        "latent",
        doc="share of resolved latent needs taken the turn they appeared",
    ),
    # ---- question value
    MetricDef("phi_loo", "qvalue", indexed=True, doc="Q(E_T) - Q(E_T \\ ev(q_t)), per turn"),
    MetricDef("phi_null", "qvalue", indexed=True, doc="matched-volume question-blind null"),
    MetricDef("phi_loo_mean", "qvalue", doc="mean phi_LOO over turns"),
    MetricDef("phi_null_mean", "qvalue", doc="mean of the matched-volume null"),
    MetricDef("stop_overshoot", "qvalue", higher_is_better=False, doc="(k_hat - k*)^+"),
    MetricDef("stop_undershoot", "qvalue", higher_is_better=False, doc="(k* - k_hat)^+"),
    # ---- the STOP 2x2, per run but counted per DECISION POINT (see metrics.stopping).
    #
    # `stop_undershoot` above is 0 on every run ever scored -- k* is the argmax of a monotone
    # ladder and k_hat is its last index, so k* <= k_hat identically (1,395/1,395 on each gate
    # parquet). The dev gate replaced it with these cells, and THESE ARE WRITTEN: `score_run`
    # emits all six below (see the `stop2x2_*` emit block), so a paper table CAN ask how often a
    # policy stopped when it was already done, and one does. The previous wording here said
    # nothing wrote them into `scores.parquet`. That was true of an earlier revision and is false
    # of this one -- corrected 2026-09-20 after the stale comment nearly caused a lane to skip the
    # reading it describes as impossible; verified by 1,200 rows per model pin for every one of the
    # six on artifacts/testsplit_qa's sibling teacher store. A count is not a rate: divide
    # `stop2x2_n_stop_at_done` by `stop2x2_n_done` before comparing arms that take different
    # numbers of turns, or the comparison reads turn count. Summable over any grouping: the unit
    # is a state, not a run.
    MetricDef(
        "stop2x2_n_done", "qvalue", doc="decision points where required coverage was complete"
    ),
    MetricDef("stop2x2_n_stop_at_done", "qvalue", doc="of those, the ones where it STOPPED"),
    MetricDef("stop2x2_n_not_done", "qvalue", doc="decision points with required evidence missing"),
    MetricDef("stop2x2_n_ask_at_not_done", "qvalue", doc="of those, the ones where it ASKED"),
    MetricDef(
        "stop2x2_n_forced_stops",
        "qvalue",
        higher_is_better=False,
        doc="final states the HARNESS halted (budget/max_turns): excluded from the cells",
    ),
    MetricDef(
        "stop2x2_asks_after_done",
        "qvalue",
        higher_is_better=False,
        doc="questions bought at states that were already done -- the cost the cells hide",
    ),
    # ---- human-centred
    MetricDef(
        "adr",
        "human",
        doc="|V_found \\ V_human| / |V_annotated|, KB-discoverable only. EMITTED ONLY where "
        "gold_human_asked is actually annotated: with none annotated the metric is not "
        "computable and no row is written, because the arithmetic otherwise yields a hard 0.0",
    ),
    MetricDef("adr_n_universe", "human", doc="the KB-discoverable universe ADR is drawn from"),
    MetricDef(
        "adr_usefulness",
        "human",
        doc="mean gold_usefulness_rating over the beyond-human needs ADR counts. EMITTED "
        "ONLY where at least one such need is actually rated: a usefulness campaign can lag "
        "the gold_human_asked annotation it depends on, so with none rated the mean is not "
        "computable and no row is written, for the same reason adr itself withholds a row",
    ),
    MetricDef(
        "adr_usefulness_anticipated",
        "human",
        doc="the same mean over the needs the human DID anticipate, found-conditioned. The "
        "reference level adr_usefulness is read against: a mean of 4.5 on beyond-human needs "
        "is a claim only against what the same annotators rated their own anticipated needs, "
        "so neither number may be reported without the other",
    ),
    MetricDef("ceiling_private_share", "human", doc="share of needs no inquirer can reach"),
    MetricDef("ceiling_attainable", "human", doc="share of needs discoverable from the KB"),
    # THE ONLY PAIR THAT SEPARATES THE TWO CHANNELS. Every other discovery metric here is
    # indifferent to whether a question went to the retriever or to the customer, so none of
    # them can support "proactive without harassing". See metrics.human.user_private_ask for
    # the three denominators and for which of them are empty on which tau2 domain.
    MetricDef(
        "user_private_ask_precision",
        "human",
        doc="P(the need is user-private | the policy asked the USER). EMITTED ONLY where the "
        "run made at least one target='user' ask that names an annotated need: on a "
        "closed-channel arm the quantity does not exist, and an unattributable question is "
        "evidence of nothing rather than a miss",
    ),
    MetricDef(
        "user_private_ask_recall",
        "human",
        doc="P(the policy asked the USER | the need is user-private AND required). EMITTED "
        "ONLY where the domain HAS required user-private needs: on tau2 (banking) all 3,637 "
        "are optional and the denominator is empty, so a 0.0 there would report a miss on a "
        "question the domain cannot pose",
    ),
    MetricDef(
        "user_private_ask_n",
        "human",
        doc="target='user' asks in the run: 0 for every arm but inquirer_may_ask_user, and "
        "written even then so a withheld rate is never mistaken for a missing run",
    ),
    MetricDef(
        "user_private_ask_n_universe",
        "human",
        doc="denominator of user_private_ask_recall: REQUIRED user-private needs",
    ),
    # ---- environment
    MetricDef("unlock_and_invoke", "quality", binary=True, doc="discoverable tool unlocked+used"),
    # P1. The tau2 primary endpoint: agent-DB AND user-DB hash equality against a gold
    # environment, computed by tau2's own EnvironmentEvaluator and carried here on
    # `native.parquet`. Binary, judge-free, and NOT EMITTED for a task the environment
    # evaluator does not cover — see pi_eval.metrics.environment.tau_reward.
    MetricDef(
        "tau_reward",
        "quality",
        binary=True,
        doc="P1: tau2 DB+user-DB hash equality, from Outcome.native",
    ),
    MetricDef(
        "discoverable_tool_unlock_tau2",
        "quality",
        doc="share of the task's gold-required discoverable tools unlocked AND invoked",
    ),
    MetricDef(
        "discoverable_tool_unlock_n",
        "quality",
        doc="denominator of discoverable_tool_unlock_tau2: gold tool_unlock nodes",
    ),
    # ---- frontier ladder
    MetricDef("frontier_spend", "frontier", indexed=True, doc="cumulative spend at prefix k"),
    MetricDef("frontier_q", "frontier", indexed=True, doc="Q at prefix k"),
    # ---- operational (descriptive; wall_ms and usd are MACHINE ARTIFACTS)
    MetricDef("tok_prompt", "operational"),
    MetricDef("tok_completion", "operational"),
    MetricDef("tok_cached", "operational"),
    MetricDef("tok_total", "operational"),
    MetricDef("usd", "operational", doc="ARTIFACT: never a cross-arm claim"),
    MetricDef("wall_ms", "operational", doc="ARTIFACT: never a cross-arm claim"),
    MetricDef("retrieval_calls", "operational"),
    MetricDef("unique_docs", "operational"),
    MetricDef("n_turns", "operational"),
    MetricDef("n_asks", "operational"),
    MetricDef("n_evidence", "operational"),
    # ---- anticipation. tau2 only; NULL (not 0) on every suite with no user simulator.
    MetricDef(
        "n_user_turns",
        "quality",
        higher_is_better=False,
        doc="user-role messages in the tau2 transcript, incl. the opening task statement",
    ),
    MetricDef(
        "n_user_followups",
        "quality",
        higher_is_better=False,
        doc="THE ANTICIPATION ENDPOINT: user turns after the opening statement",
    ),
    # A covariate, not a claim: musique gold answers average 2.26 words against word_cap=30,
    # so answer_token_f1 is depressed by length alone and the effect had no instrument.
    MetricDef(
        "answer_n_words",
        "operational",
        needs_answer=True,
        doc="COVARIATE for the length confound on answer_token_f1; never a claim on its own",
    ),
    MetricDef(
        "answer_wellformed",
        "operational",
        binary=True,
        needs_answer=True,
        doc="0 when the answer is a serialized tool call; arm-dependent, must be reported",
    ),
    MetricDef("usd_per_need_discovered", "operational", higher_is_better=False),
    MetricDef("tokens_per_useful_need", "operational", higher_is_better=False),
)

BY_NAME: dict[str, MetricDef] = {m.name: m for m in METRICS}

# Machine artifacts. Recorded, never compared across arms as evidence. See CONTRIBUTING.md.
ARTIFACT_METRICS: frozenset[str] = frozenset({"usd", "wall_ms", "usd_per_need_discovered"})


def family_of(metric_name: str) -> str:
    """`phi_loo#3` -> `phi_loo`. Indexed families collapse to their base name."""
    return metric_name.split("#", 1)[0]


def define(metric_name: str) -> MetricDef | None:
    return BY_NAME.get(family_of(metric_name))


# --------------------------------------------------------------------------- hashes


def metric_defs_hash() -> str:
    return h(
        "metric_defs",
        METRIC_DEFS_VERSION,
        canon(
            [
                (m.name, m.family, m.binary, m.judge_derived, m.higher_is_better, m.indexed)
                for m in METRICS
            ]
        ),
        canon(sorted(DWR_WEIGHTS.items())),
        sch.schema_hash(),
    )


def graph_hash(graphs: Mapping[str, Mapping[str, GoldGraph]]) -> str:
    """A fingerprint of exactly the gold that was read. Changing a single edge changes the
    scorer_hash, so old rows keep meaning what they meant."""
    digest = []
    for suite in sorted(graphs):
        for key in sorted(graphs[suite]):
            g = graphs[suite][key]
            digest.append(
                (
                    suite,
                    key,
                    g.gold_graph_version,
                    sorted(
                        (n.gold_node_id, n.gold_depth, n.gold_partition, tuple(n.gold_ev_uids))
                        for n in g.gold_nodes
                    ),
                    sorted(
                        (e.gold_src_node_id, e.gold_dst_node_id, e.gold_edge_kind)
                        for e in g.gold_edges
                    ),
                )
            )
    return h("graphs", canon(digest))


def matcher_hash(matcher: Any) -> str:
    return h("matcher", str(matcher.matcher_id), str(matcher.matcher_family))


def scorer_hash(*, metric_defs: str, graph: str, matcher: str, judge_pins: Sequence[str]) -> str:
    return h("scorer", metric_defs, graph, matcher, canon(sorted(judge_pins)))


# --------------------------------------------------------------------------- parquet shims


@dataclass(frozen=True, slots=True)
class _Act:
    text: str


@dataclass(frozen=True, slots=True)
class _Turn:
    turn_idx: int
    retrieved_uids: tuple[str, ...]
    new_uids: tuple[str, ...]
    action: _Act


@dataclass(frozen=True, slots=True)
class _Answer:
    text: str
    cited_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Outcome:
    answer: _Answer | None


@dataclass(frozen=True, slots=True)
class _Traj:
    """The minimum a matcher needs, reconstructed from parquet.

    Deliberately NOT pinq.types.Trajectory: rebuilding a real Trajectory would require the
    evidence TEXT, which is not stored on purpose. Anything that genuinely needs text reads
    outcome.json and says so.
    """

    turns: tuple[_Turn, ...]
    outcome: _Outcome


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    text: str
    cited_uids: tuple[str, ...]


def load_answers(runs_root: str | Path) -> dict[str, AnswerRecord]:
    """Answer text and citations, keyed by run_id (== the run directory name).

    Absent directory -> empty mapping, and every answer-dependent metric is then simply not
    emitted for those runs.
    """
    root = Path(runs_root)
    out: dict[str, AnswerRecord] = {}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        oc = d / "outcome.json"
        if not d.is_dir() or not oc.exists():
            continue
        try:
            payload = json.loads(oc.read_text())
        except (OSError, ValueError):
            continue
        ans = payload.get("answer")
        if not ans:
            continue
        out[d.name] = AnswerRecord(
            text=str(ans.get("text", "")),
            cited_uids=tuple(ans.get("cited_unit_ids") or ()),
        )
    return out


# --------------------------------------------------------------------------- per-run metrics


def _coverage(uids: Iterable[str], gold_uids: frozenset[str]) -> float:
    return discovery.evidence_coverage(set(uids), set(gold_uids))


def _gold_uids(graph: GoldGraph) -> frozenset[str]:
    """Gold spans of the REQUIRED partition. Optional needs are scored separately; folding
    them into the denominator would make coverage depend on annotation generosity."""
    return frozenset(u for n in graph.required() for u in n.gold_ev_uids)


def _cum_spend(ledger_rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """turn_idx -> cumulative hard-currency spend after that turn."""
    out: dict[int, float] = {}
    for r in ledger_rows:
        if r.get("currency") == HARD_CURRENCY:
            out[int(r["turn_idx"])] = float(r["cumulative"])
    return out


def _emit(rows: list[dict[str, Any]], run_id: str, name: str, value: float, **kw: Any) -> None:
    rows.append(
        {
            "run_id": run_id,
            "metric_name": name,
            "scorer_hash": "",  # stamped once, at the end of score()
            "value": float(value),
            "ci_lo": kw.get("ci_lo", NAN),
            "ci_hi": kw.get("ci_hi", NAN),
            "n": int(kw.get("n", 1)),
            "graph_version": str(kw.get("graph_version", "")),
            "notes": str(kw.get("notes", "")),
        }
    )


def score_run(
    run: Mapping[str, Any],
    *,
    graph: GoldGraph,
    turns: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    env_calls: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    records: Sequence[MatchRecord],
    answer: AnswerRecord | None,
    native: Mapping[str, float] | None = None,
    null_draws: int = 8,
    null_pool: Sequence[str] = (),
    seed: int = 0,
    gold_free: bool = False,
) -> list[dict[str, Any]]:
    """Every metric for one run. Pure: same inputs, same rows, no clock and no network.

    `gold_free` suppresses every metric that is a function of the need-graph, for a suite that
    has none. It defaults to False and `score()` sets it from `GOLD_FREE_SUITES`, so no suite
    that ships a graph can reach the branch by accident.
    """
    run_id = str(run["run_id"])
    gv = str(graph.gold_graph_version or DEFAULT_GRAPH_VERSION)
    rows: list[dict[str, Any]] = []
    turns = sorted(turns, key=lambda t: int(t["turn_idx"]))
    gold_uids = _gold_uids(graph)
    required_ids = [n.gold_node_id for n in graph.required()]
    retrieved = [uid for t in turns for uid in (t.get("retrieved_uids") or ())]
    all_uids = frozenset(retrieved) | frozenset(str(e["uid"]) for e in evidence)

    def emit(name: str, value: float, **kw: Any) -> None:
        _emit(rows, run_id, name, value, graph_version=gv, **kw)

    # ---- quality
    cov = _coverage(all_uids, gold_uids)
    if not math.isnan(cov):
        emit("evidence_coverage", cov, n=len(gold_uids), notes="Q(E) over required gold spans")
        emit("task_success", 1.0 if cov >= 1.0 - 1e-12 else 0.0, n=len(gold_uids))
    if answer is not None and graph.gold_answer:
        # `graph.answer`, not `graph.gold_answer`: the raw field carries the canary nonce so
        # that a leak of the answer key carries it too, and the nonce must not enter a
        # comparison. See GoldGraph.answer.
        gold_answer = graph.answer
        f1, prec, rec = quality.decompose_over_aliases(answer.text, gold_answer, graph.gold_aliases)
        emit("answer_token_f1", f1)
        emit("answer_token_recall", rec, notes="length-robust: padding cannot lower this")
        emit("answer_token_precision", prec, notes="the half answer length destroys")
        emit(
            "answer_exact_match",
            quality.exact_match(answer.text, gold_answer, graph.gold_aliases),
        )
        # WHY BOTH. Measured over 448 musique runs with the shipped detector: 52.2% of
        # answers take `answerer_frozen`'s hedge clause, and on that half token_f1 is 0.003
        # against 0.315 -- while evidence_coverage is HIGHER (0.548 vs 0.500), so the refusal
        # is not a response to missing evidence. Hedge rate ranges 16.7%-83.3% ACROSS ARMS,
        # which makes any answer-quality contrast partly a prompt-compliance contrast.
        # `answer_hedged` is what makes that visible; it is NOT a quality judgement, because
        # refusing IS correct when the evidence genuinely lacks the answer. `answer_correct`
        # is length-robust but NOT a fix for the confound (corr -0.464 vs f1's -0.466): only
        # 1.7% of hedged answers name the gold span, so there is no diluted answer to
        # recover. Hold hedge rate fixed by design; do not normalise it away with a metric.
        emit("answer_hedged", 1.0 if quality.is_hedged(answer.text) else 0.0)
        emit(
            "answer_correct",
            quality.contains_answer(answer.text, gold_answer, graph.gold_aliases),
        )

    # ---- EVERYTHING BELOW NEEDS A NEED-GRAPH, AND `frames` HAS NONE.
    #
    # Not a filter: these rows are never COMPUTED. Traced against a node-free graph, the
    # guards already inside each block are individually right and collectively insufficient --
    # coverage, cad, dwr, orphan_rate, the latent block, adr, the ceiling and phi_loo all
    # correctly emit nothing on a NaN or an empty set, while `n_needs_resolved`,
    # `max_depth_reached`, `facet_breadth`, `facet_total`, `stop_overshoot`,
    # `stop_undershoot`, `frontier_spend#k` and `phi_null#j` all emit a value anyway. Six
    # families of publishable-looking zero, produced by a suite that has no structure to
    # measure. `stop_overshoot` is the clearest: with Q undefined at every prefix, k* is 0 by
    # the tie-break and the overshoot is just the turn count wearing a metric's name.
    #
    # The QUALITY block above and the COST block below are deliberately outside this guard:
    # the answer metrics are defined by the gold answer alone, and turns, tokens and dollars
    # are properties of the rollout. Those are exactly what an external answers-only benchmark
    # can carry.
    if not gold_free:
        # ---- discovery ladder
        if required_ids:
            ladder = discovery.rnr_ladder(records, required_ids)
            emit("rnr_ask", ladder["ask"], n=len(required_ids))
            emit("rnr_resolve", ladder["resolve"], n=len(required_ids))
            if answer is not None:
                emit("rnr_use", ladder["use"], n=len(required_ids))
        emit("n_needs_resolved", float(sum(1 for r in records if r.rank >= 2)), n=len(records))

        # ---- structure. C@d and |V_d| are emitted TOGETHER; never one without the other.
        cad = structure.coverage_at_depth(records, graph)
        for d, (recall, n_d) in sorted(cad.items()):
            emit(f"cad#{d}", recall, n=n_d, notes=f"|V_{d}|={n_d}")
            emit(f"cad_n#{d}", float(n_d), n=n_d, notes=f"|V_{d}|")
        if cad:
            emit("dwr", structure.depth_weighted_recall(cad, DWR_WEIGHTS), n=len(cad))
        emit("max_depth_reached", float(structure.max_depth_reached(records, graph)))
        pvr = structure.precedence_violation_rate(records, graph)
        if not math.isnan(pvr):
            emit("precedence_violation_rate", pvr)
        touched, total = structure.facet_breadth(records, graph)
        emit("facet_breadth", float(touched), n=total)
        # The facet-free horizontal axis, EMITTED ONLY WHERE THE GRAPH HAS PREREQUISITE
        # STRUCTURE.
        #
        # Without that guard it counts fragmentation. MEASURED on tau2 gold: 90% of airline
        # components and 85% of retail ones are SINGLETONS, and 144 of airline's 183 singletons
        # are `unknown` provenance -- nodes the builder could not trace to any source. Airline
        # task 18 has 28 components over 28 nodes because it has no prerequisite edges at all,
        # and only 49% of airline graphs (72% of retail) have even one. A median of 2
        # "independent lines of inquiry" on such a graph is a statement about the annotation,
        # not about the policy.
        #
        # `breadth_singleton_share` rides alongside so the fragmentation is visible in the table
        # rather than buried in the numerator: a breadth number over mostly-singleton components
        # is not a breadth number.
        b_touched, b_total, b_single = structure.breadth_components(records, graph)
        if b_total and any(
            getattr(e, "gold_edge_kind", "") == "prerequisite" for e in graph.gold_edges
        ):
            emit("breadth_components", float(b_touched), n=b_total)
            emit("breadth_recall", b_touched / b_total, n=b_total)
            emit("breadth_singleton_share", b_single / b_total, n=b_total)
        emit("facet_total", float(total), n=total)
        depths = {n.gold_node_id: n.gold_depth for n in graph.gold_nodes}
        if depths:
            emit("orphan_rate", sum(1 for v in depths.values() if v is None) / len(depths))

        # ---- latent needs. ABSENT, NOT ZERO, on a task with no latent needs: a flat graph cannot
        # exhibit vertical proactivity in either direction, and a 0.0 would read as "found none"
        # on a task that HAS none. That is why the rate is NaN there and no row is written.
        lat = latent.latent_labels(graph, records)
        if lat.n_latent_available > 0:
            emit(
                "latent_discovery_rate",
                lat.latent_discovery_rate,
                n=lat.n_latent_available,
                notes=f"{lat.n_latent_resolved} of {lat.n_latent_available} reachable latent needs",
            )
            emit("latent_available", float(lat.n_latent_available), n=lat.n_latent_available)
            took = [t for t in lat.per_turn.values() if t.is_latent]
            if took:
                emit(
                    "newly_reachable_share",
                    sum(1 for t in took if t.newly_reachable) / len(took),
                    n=len(took),
                )

        # ---- human-centred
        adr = human.anticipated_discovery_rate(records, graph)
        # n_annotated, NOT n_universe. `gold_human_asked` is a human annotation and is None on
        # every node this repository has built, so gating on the universe emitted a hard 0.0 for
        # every run -- a publishable-looking zero computed from a measurement that was never taken,
        # under the one metric whose purpose is to support "exceeds what the user thought to ask".
        # A missing measurement must be missing.
        if adr["n_annotated"]:
            emit("adr", adr["adr"], n=int(adr["n_annotated"]))
            emit("adr_n_universe", float(adr["n_universe"]))
            # mean_usefulness_beyond_human rests on n_rated, NOT n_beyond_human: a usefulness
            # campaign can lag the gold_human_asked annotation it depends on, so a task can have
            # beyond-human needs with zero of them rated yet. NaN there must stay a missing row.
            if not math.isnan(adr["mean_usefulness_beyond_human"]):
                emit("adr_usefulness", adr["mean_usefulness_beyond_human"], n=int(adr["n_rated"]))
            # The reference level, emitted on its own guard rather than beside the line above:
            # the two groups are rated by the same people but not necessarily at the same time,
            # so one side can be measured while the other is still absent. Neither is reportable
            # without the other, which is a rule for the TABLE, not a reason to fabricate a row.
            if not math.isnan(adr["mean_usefulness_anticipated"]):
                emit(
                    "adr_usefulness_anticipated",
                    adr["mean_usefulness_anticipated"],
                    n=int(adr["n_rated_anticipated"]),
                )
        ceiling = human.discoverability_ceiling(graph)
        if ceiling.n_total:
            emit("ceiling_private_share", ceiling.private_share, n=ceiling.n_total)
            emit("ceiling_attainable", ceiling.attainable, n=ceiling.n_total)

        # ---- the user channel. Gated on the PARTITION EXISTING, not on the run: musique,
        # strategyqa, wiki2, drgym and synth carry no user_private node at all (measured over
        # data/gold/graphs: 0 of 57,204 nodes), and a row there would be a measurement of a
        # partition that does not exist. Where it does exist the COUNT is always written --
        # including the 0 that a closed-channel arm earns -- so that a withheld rate reads as
        # "this arm could not ask" rather than as a missing run.
        upa = human.user_private_ask(turns, graph)
        if upa["n_private"] or upa["n_user_asks"]:
            emit(
                "user_private_ask_n",
                upa["n_user_asks"],
                n=len([t for t in turns if str(t.get("action_kind") or "") == "ask"]) or 1,
                notes="target='user' asks; 0 is the closed channel, not a failure",
            )
            emit("user_private_ask_n_universe", upa["n_universe"], n=int(upa["n_universe"]) or 1)
            # NaN on both is the closed channel and the empty denominator, and each must stay a
            # missing row rather than a plausible zero. See metrics.human.user_private_ask.
            if not math.isnan(upa["precision"]):
                emit(
                    "user_private_ask_precision",
                    upa["precision"],
                    n=int(upa["n_attributed"]),
                    notes=f"{int(upa['n_user_asks'])} user asks, "
                    f"{int(upa['n_user_asks'] - upa['n_attributed'])} unattributed",
                )
            if not math.isnan(upa["recall"]):
                emit(
                    "user_private_ask_recall",
                    upa["recall"],
                    n=int(upa["n_universe"]),
                    notes=f"{int(upa['n_reached'])} of {int(upa['n_universe'])} required "
                    "user-private needs named down the user channel",
                )

        # ---- frontier ladder, indexed by CUMULATIVE SPEND (never by turns: see metrics.frontier)
        spend_by_turn = _cum_spend(ledger)
        seen: set[str] = set()
        qs = [_coverage(seen, gold_uids)]
        spends = [0.0]
        for t in turns:
            seen |= set(t.get("retrieved_uids") or ())
            qs.append(_coverage(seen, gold_uids))
            spends.append(spend_by_turn.get(int(t["turn_idx"]), spends[-1]))
        for k, (s, q) in enumerate(zip(spends, qs)):
            emit(f"frontier_spend#{k}", s, notes="cumulative " + HARD_CURRENCY)
            if math.isnan(q):
                # NOT 0.0. Q is NaN when the task carries no required gold evidence, i.e. when
                # coverage is UNDEFINED rather than zero -- and the frontier AUC integrates over
                # these points, so a fabricated 0 drags the curve down for a task that never had a
                # curve. The point is omitted; `frontier_auc` reads the points that exist.
                continue
            emit(f"frontier_q#{k}", q, notes="Q at prefix k")

        # ---- stopping error, in QUESTION UNITS (see metrics.qvalue for why not quality units)
        finite = [0.0 if math.isnan(q) else q for q in qs]
        k_star = max(range(len(finite)), key=lambda i: (finite[i], -i))
        k_hat = len(turns)
        emit("stop_overshoot", float(max(0, k_hat - k_star)), n=k_hat)
        emit("stop_undershoot", float(max(0, k_star - k_hat)), n=k_hat)

        # ---- the STOP 2x2, counted over DECISION POINTS. The ladder handed to it is the same
        # {t: coverage} the `frontier_q#t` rows above carry, with the undefined points ABSENT
        # rather than zeroed -- `stopping.stop_2x2` skips and counts those, because "the task
        # has no required gold evidence" and "the policy retrieved none of it" are opposite
        # statements. `n_asks` and `stop_reason` come from the run record, not from len(turns),
        # so a truncated turn table reads as skipped states rather than a shorter episode.
        #
        # NO ROW when no decision point carried a coverage reading. Six zeros there would read
        # as a policy that made every decision correctly on a task that was never measured.
        cells = stopping.stop_2x2(
            {k: q for k, q in enumerate(qs) if not math.isnan(q)},
            n_asks=int(run.get("n_asks") or 0),
            stop_reason=str(run.get("stop_reason") or ""),
        )
        if cells.n_scored:
            note = f"{cells.n_skipped_no_coverage} state(s) skipped: coverage undefined"
            emit("stop2x2_n_done", float(cells.n_done), n=cells.n_scored, notes=note)
            emit("stop2x2_n_stop_at_done", float(cells.n_stop_at_done), n=cells.n_done)
            emit("stop2x2_n_not_done", float(cells.n_not_done), n=cells.n_scored, notes=note)
            emit("stop2x2_n_ask_at_not_done", float(cells.n_ask_at_not_done), n=cells.n_not_done)
            emit(
                "stop2x2_n_forced_stops",
                float(cells.n_forced_stops),
                notes="harness-halted final states, excluded from the cells",
            )
            emit("stop2x2_asks_after_done", float(cells.asks_after_done), n=cells.n_done)

        # ---- phi_LOO under Q = coverage, plus a matched-volume question-blind null
        base = _coverage(all_uids, gold_uids)
        phis: list[float] = []
        if not math.isnan(base):
            for t in turns:
                ev = frozenset(t.get("retrieved_uids") or ())
                if not ev:
                    continue
                phi = base - _coverage(all_uids - ev, gold_uids)
                phis.append(phi)
                emit(f"phi_loo#{int(t['turn_idx'])}", phi, n=len(ev), notes="Q=evidence_coverage")
            if phis:
                emit("phi_loo_mean", sum(phis) / len(phis), n=len(phis))
        nulls = _null_phis(
            base=base,
            all_uids=all_uids,
            gold_uids=gold_uids,
            volume=int(round(sum(len(t.get("retrieved_uids") or ()) for t in turns) / len(turns)))
            if turns
            else 0,
            pool=null_pool,
            draws=null_draws,
            seed=seed,
        )
        for j, v in enumerate(nulls):
            emit(f"phi_null#{j}", v, notes="matched-volume, question-blind")
        if nulls:
            emit("phi_null_mean", sum(nulls) / len(nulls), n=len(nulls))

    # ---- environment: the discoverable-tool prerequisite edge, binary and objective
    #
    # THE DENOMINATOR IS THE TASK, NOT THE POLICY. This used to emit only `if gated` -- only
    # when the policy had ATTEMPTED a gated call -- so a policy that never tried produced no
    # row at all and was dropped from the endpoint. That is the failing case: measured on the
    # first live tau2 sweep, `inquirer_prompted` executed 26-38 real tool calls and never once
    # reached `call_discoverable_agent_tool`, so on this metric it would simply have been
    # absent rather than scored 0. The arms that never attempt the mechanic are exactly the
    # ones the metric exists to catch, and dropping them flatters them.
    #
    # Gold decides applicability: a task whose graph names no discoverable tool has nothing to
    # unlock and is genuinely N/A, which stays absent. A task that DOES require one and got no
    # attempt scores 0.0.
    gated = [e for e in env_calls if bool(e.get("unlock_required"))]
    requires_unlock = any(n.gold_kind == "tool_unlock" for n in graph.gold_nodes)
    if gated or requires_unlock:
        hit = any(bool(e.get("unlock_satisfied")) and bool(e.get("ok")) for e in gated)
        emit(
            "unlock_and_invoke",
            1.0 if hit else 0.0,
            n=max(len(gated), 1),
            notes="0.0 with no attempt is a failure, not an absence",
        )

    # The RATE over the tools this task's gold actually names, next to the ANY-bit above. A
    # task whose gold names four discoverable tools and one that names a single tool are not
    # the same trial; see pi_eval.metrics.environment for why the denominator is gold.
    unlock = environment.discoverable_tool_unlock(env_calls, graph)
    if environment.is_measured(unlock["rate"]):
        emit(
            "discoverable_tool_unlock_tau2",
            unlock["rate"],
            n=int(unlock["n_required"]),
            notes=f"invoked {int(unlock['n_invoked'])}, unlocked {int(unlock['n_unlocked'])}",
        )
        emit("discoverable_tool_unlock_n", unlock["n_required"], n=int(unlock["n_required"]))

    # ---- the tau2 primary endpoint, carried on Outcome.native and graded by tau2 itself.
    # Absent for an ungraded run, and absent is NOT zero: emitting a fabricated 0 here would
    # put "the policy failed" in a table that means "this run was never graded".
    reward = environment.tau_reward(native or {})
    if environment.is_measured(reward):
        emit("tau_reward", reward, notes="tau2 EnvironmentEvaluator DB check (agent + user)")

    # ---- operational
    for name in (
        "tok_prompt",
        "tok_completion",
        "tok_cached",
        "tok_total",
        "usd",
        "wall_ms",
        "retrieval_calls",
        "unique_docs",
        "n_turns",
        "n_asks",
        "n_evidence",
    ):
        note = "MACHINE ARTIFACT: not a cross-arm claim" if name in ARTIFACT_METRICS else ""
        emit(name, float(run.get(name) or 0.0), notes=note)
    # ---- anticipation, on the same absent-is-not-zero footing as tau_reward. Kept OUT of
    # the loop above because that loop's `or 0.0` would score every user-less suite a
    # perfect 0 -- "the user never had to speak" -- for having no user at all.
    ut = environment.user_turns(run)
    if environment.is_measured(ut):
        emit("n_user_turns", ut, notes="incl. the opening task statement")
        emit(
            "n_user_followups",
            environment.user_followups(run),
            notes="ANTICIPATION: user turns the policy did not preempt",
        )
    if answer is not None:
        emit(
            "answer_wellformed",
            0.0 if quality.is_toolcall_answer(answer.text) else 1.0,
            notes="0 = the answer is a serialized tool call, not an answer",
        )
        emit(
            "answer_n_words",
            float(len(answer.text.split())),
            notes="COVARIATE for length; pairs with answer_token_f1",
        )
    n_res = sum(1 for r in records if r.rank >= 2)
    if n_res:
        emit(
            "usd_per_need_discovered",
            float(run.get("usd") or 0.0) / n_res,
            n=n_res,
            notes="MACHINE ARTIFACT: derived from usd",
        )
        emit("tokens_per_useful_need", float(run.get("tok_total") or 0.0) / n_res, n=n_res)
    return rows


def _null_phis(
    *,
    base: float,
    all_uids: frozenset[str],
    gold_uids: frozenset[str],
    volume: int,
    pool: Sequence[str],
    draws: int,
    seed: int,
) -> list[float]:
    """The null phi has no meaningful zero without one.

    Matched VOLUME and question-BLIND, the parquet-only analogue of
    `metrics.qvalue.random_question_null`: draw `volume` uids uniformly from the pool of
    units the SUITE's retrievers produced across all its tasks, keep the overlap with what
    this run actually holds, and measure the same leave-one-out drop.

    THE POOL IS PER SUITE, NOT PER TASK. A per-task pool is a draw from the units this run
    already retrieved, so on any suite where retrieval is precise the null reproduces phi
    exactly and reports that a policy is no better than itself. Drawing from the suite is
    what makes the null answer the real question: how much would a same-sized bundle of
    evidence chosen without regard to THIS task have been worth?
    """
    if math.isnan(base) or volume <= 0 or not pool:
        return []
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(draws):
        drawn = frozenset(rng.sample(list(pool), min(volume, len(pool)))) & all_uids
        out.append(base - _coverage(all_uids - drawn, gold_uids))
    return out


# --------------------------------------------------------------------------- the judge stage
#
# WHY THE JUDGE CLIENT ARRIVES THROUGH A HOOK RATHER THAN AN IMPORT. `pi_eval` depends on
# `pinq` and nothing else first-party: that is what keeps the gold-only package free of the
# runner, the providers and their transitive pins. But `pi score` IS the process that judges,
# and it has to reach a model somehow. So the client is a DEPLOYMENT choice named by
# `PI_JUDGE_CLIENT="module:factory"` and resolved at call time -- a live proxy client in
# production, `pi_run.cache.ReplayClient` in CI, a scripted fake in a test. Passing
# `judge=` explicitly bypasses the hook entirely and is what the tests do.
#
# WITH NO HOOK AND NO CLIENT, NOTHING IS JUDGED AND THE RESULT SAYS SO. It does not fall back
# to an unjudged zero: `judgments_rows_added: 0` with `judging: "skipped: ..."` is a visibly
# missing measurement, and a fabricated 0.0 for `kpr_incremental` would average into a table.

JUDGE_CLIENT_ENV = "PI_JUDGE_CLIENT"  # "package.module:factory", factory() -> JudgeLLM
JUDGE_MODEL_ENV = "PI_MODEL_JUDGE"  # the model pin, same variable the runner uses
JUDGE_FAMILY_ENV = "PI_JUDGE_FAMILY"  # e.g. "openai"; the unit Krippendorff alpha compares


def judge_client_from_env(env: Mapping[str, str] | None = None) -> tuple[Any | None, str, str, str]:
    """(client, model, family, reason). `client is None` means: do not judge, and here is why."""
    e = dict(os.environ if env is None else env)
    spec = e.get(JUDGE_CLIENT_ENV, "").strip()
    model = e.get(JUDGE_MODEL_ENV, "").strip()
    family = e.get(JUDGE_FAMILY_ENV, "openai").strip() or "openai"
    if not spec:
        return None, model, family, f"skipped: {JUDGE_CLIENT_ENV} unset"
    if not model:
        # Never defaulted. Two arms judged by different models and compared is the failure
        # this refusal exists to make impossible.
        return None, "", family, f"refused: {JUDGE_CLIENT_ENV} set but {JUDGE_MODEL_ENV} is not"
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        return None, model, family, f"refused: {JUDGE_CLIENT_ENV}={spec!r} is not 'module:factory'"
    factory = getattr(importlib.import_module(mod), attr)
    return factory(), model, family, "env"


def judge_derived_suites() -> frozenset[str]:
    """Suites whose PREREGISTERED endpoints include a judge-derived metric.

    Read from the seal rather than hardcoded: adding a judge-derived endpoint on a new suite
    must tighten this automatically, not leave it stale.
    """
    from pi_eval.prereg import POOLED, all_endpoints

    out: set[str] = set()
    for e in all_endpoints():
        d = define(e.metric)
        if not (d and d.judge_derived):
            continue
        if e.suite_id == POOLED:
            out.update(e.pool_suites or ())
        else:
            out.add(e.suite_id)
    return frozenset(out)


def question_gap_verdict(by_suite: Mapping[str, int], *, total: int) -> tuple[int, str | None]:
    """(n_skipped, refusal or None) for runs that have no question text.

    Every judge takes the question, so grading against an empty one measures a different
    instrument. The original guard refused whenever ANY run lacked a question -- and on this
    repository that meant 78 stale `synth` runs blocked judging drgym, the only suite with a
    judge-derived preregistered endpoint.

    So the refusal is kept exactly where it protects a number. A suite whose endpoint IS
    judge-derived must never be judged over a silently smaller denominator; anywhere else the
    run is SKIPPED and COUNTED, the same rule `pi_run.cmd_train.collect_rows` applies.
    """
    if not by_suite:
        return 0, None
    judged = judge_derived_suites()
    offenders = {s: n for s, n in sorted(by_suite.items()) if s in judged}
    n_skipped = sum(by_suite.values())
    if offenders:
        named = ", ".join(f"{s}: {n}" for s, n in offenders.items())
        return n_skipped, (
            f"{named} run(s) have no question text, and that suite carries a judge-derived "
            "preregistered endpoint. Judging it would compute the endpoint over a smaller "
            "denominator than the one declared. The usual cause is a run whose corpus "
            "directory is missing or was rebuilt under a new content hash."
        )
    return n_skipped, None


def question_cache_key(run: Mapping[str, Any]) -> tuple[str, str]:
    """(suite_id, corpus directory) -- where THIS run's questions live on disk.

    corpus_dir, NOT corpus_hash. They are different strings over different inputs: the
    directory is content-addressed and exists, while `corpus_hash` is an order-independent
    hash over (doc_id, title, sha256(text)) triples and names no path. Measured over the
    distinct triples on disk, `data/corpora/<suite>/<corpus_hash>/tasks.jsonl` exists for 0
    of 8 and the corpus_dir form for 5 of 8.

    SHARED ON PURPOSE. `scripts/measure_sigma_j.py` had its own copy keyed on corpus_hash
    alone, so it loaded ZERO questions for every run and would have graded every item with
    an empty question -- and sigma_J is the gate on whether any judge-derived metric may be
    reported at all. Two copies of one rule are free to disagree; this is the rule.

    Falls back to corpus_hash so runs written before `corpus_dir` existed stay readable.
    They miss, as they always did, and `n_questions_missing` says so out loud.
    """
    return (
        str(run.get("suite_id") or ""),
        str(run.get("corpus_dir") or run.get("corpus_hash") or ""),
    )


def load_questions(corpora_root: str | Path, suite_id: str, corpus_dir: str) -> dict[str, str]:
    """task_id -> question, from the PUBLIC corpus the run was rolled out against.

    Read by the run's own corpus DIRECTORY, not by "the newest directory": the corpora tree is
    content-addressed precisely so that a re-build cannot retro-fit different questions onto
    finished runs.

    THE ARGUMENT USED TO BE `corpus_hash`, AND THAT WAS A DIFFERENT STRING. `corpus_hash` is
    the adapter's order-independent hash over (doc_id, title, sha256(text)) triples;
    the directory is named for the sha256 of the tasks payload. Both are 'the corpus hash' in
    conversation and neither is the other, so this function opened
    `data/corpora/<suite>/<64-hex>/tasks.jsonl` -- a path that has never existed -- caught the
    absence, and returned {}. EVERY judge input therefore carried `question=""`, on the path
    that produces `kpr_incremental`, a PRIMARY endpoint. It was invisible because judging had
    never run: `PI_JUDGE_CLIENT` had no implementation, so the empty questions were never used.
    Two holes, and each one hid the other.
    """
    p = Path(corpora_root) / suite_id / corpus_dir / "tasks.jsonl"
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "id" in rec and "question" in rec:
            out[str(rec["id"])] = str(rec["question"])
    return out


def _judge_docs_from_cache(run_dir: Path, cache_root: str | Path | None) -> dict[str, str]:
    """Replay the run's retrieval. Imported lazily, and never fatal.

    The import is deferred for the same reason `tau2_build` defers its adapter imports: the
    drgym adapter is optional at scoring time, and `pytest -m "not integration"` must stay
    runnable without it. A reconstruction that fails for ANY reason returns {} -- the judge
    then skips this run and counts it, which is the same fail-closed behaviour an absent
    sidecar has always had. Scoring must not die because a cache is missing.
    """
    if cache_root is None:
        return {}
    try:
        from pinq_adapters.drgym.judge_docs import judge_docs_for

        docs, _missing = judge_docs_for(run_dir, cache_root)
        return docs
    except Exception:
        return {}


def load_judge_docs(
    runs_root: str | Path | None, run_id: str, *, cache_root: str | Path | None = None
) -> dict[str, str]:
    """{url: text} for the citation judge: the sidecar if present, else the search cache.

    Evidence TEXT is deliberately not stored in parquet (the corpus is content-addressed and,
    for drgym, hosted), so the citation judge cannot be reconstructed from the tables alone.
    An absent source means the citation judge IS NOT RUN for that run and the omission is
    counted -- it is not run against empty documents, which would score every claim
    `no_support` and report a citation failure that is really a fetch failure.

    `runs/<run_id>/judge_docs.json` has NO WRITER anywhere in this codebase and 0 of 729 run
    directories have one, so for as long as it was the only source `citation_support` -- the
    judge-derived metric T6_instrument reports -- was disqualified on every run ever scored. It stays supported
    and stays FIRST, because an explicit sidecar is a deliberate override.

    The fallback replays the run's own retrieval out of the drgym search cache, which stores
    every envelope verbatim. MEASURED over 60 real drgym runs: 0 missing queries, mean 17
    documents per retrieving run. It covers runs that already exist, including the P3 grid,
    which a rollout-time sidecar could only cover by re-rolling them.
    """
    if runs_root is None:
        return {}
    p = Path(runs_root) / run_id / "judge_docs.json"
    if not p.exists():
        return _judge_docs_from_cache(Path(runs_root) / run_id, cache_root)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v.strip()}


def judge_arm_pairs(suite_id: str, s1: Any | None = None) -> tuple[tuple[str, str], ...]:
    """The (treatment, comparator) arm pairs a judge-derived endpoint commits to on this suite.

    Read off the preregistration, so the judging invoice is spent on exactly the comparisons
    the paper is allowed to make. Judging every arm against every other would be a
    multiplicity problem bought with money.
    """
    from pi_eval import prereg

    s1 = s1 or prereg.default_stage1()
    out: list[tuple[str, str]] = []
    for e in prereg.all_endpoints(s1):
        d = BY_NAME.get(family_of(e.metric))
        if d is None or not d.judge_derived:
            continue
        if e.suite_id not in (prereg.POOLED, suite_id):
            continue
        pair = (e.treatment, e.comparator)
        if pair not in out:
            out.append(pair)
    return tuple(out)


def run_judges(
    *,
    runs: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, Mapping[str, GoldGraph]],
    answers: Mapping[str, AnswerRecord],
    judge: Any,
    judge_model: str,
    judge_family: str,
    corpora_root: str | Path,
    runs_root: str | Path | None,
    sigma_j: Mapping[str, float],
    seed: int,
    cap_usd: float | None = None,
    cache_root: str | Path | None = None,
) -> Any:
    """Build the judge inputs from parquet + gold and hand them to the judging harness."""
    from pi_eval.judges import harness

    questions: dict[tuple[str, str], dict[str, str]] = {}
    inputs: list[harness.RunInput] = []
    n_no_question = 0
    no_question_by_suite: dict[str, int] = {}
    for r in sorted(runs, key=lambda x: str(x["run_id"])):
        run_id = str(r["run_id"])
        suite_id, task_id = str(r["suite_id"]), str(r["task_id"])
        answer = answers.get(run_id)
        graph = graphs.get(suite_id, {}).get(task_id)
        if answer is None or graph is None or not answer.text.strip():
            continue
        key = question_cache_key(r)
        if key not in questions:
            questions[key] = load_questions(corpora_root, key[0], key[1])
        if not questions[key].get(task_id, "").strip():
            # A judge handed an empty question grades a report against nothing. Every DRGym
            # judge takes the question, so this silently changes what is being measured --
            # and it was the state of every run until `corpus_dir` existed.
            #
            # SKIPPED rather than judged, and counted per suite so the verdict below can tell
            # a stale calibration run apart from a hole in a judge-derived endpoint.
            n_no_question += 1
            no_question_by_suite[suite_id] = no_question_by_suite.get(suite_id, 0) + 1
            continue
        inputs.append(
            harness.RunInput(
                run_id=run_id,
                arm_id=str(r["arm_id"]),
                report=answer.text,
                task=harness.TaskInput(
                    suite_id=suite_id,
                    task_id=task_id,
                    question=questions[key].get(task_id, ""),
                    key_points=tuple(
                        (n.gold_node_id, n.gold_text) for n in graph.required() if n.gold_text
                    ),
                    docs=load_judge_docs(runs_root, run_id, cache_root=cache_root),
                ),
            )
        )
    if not inputs:
        return None
    n_skipped, refusal = question_gap_verdict(
        no_question_by_suite, total=len(inputs) + n_no_question
    )
    if refusal:
        # Not a warning to be scrolled past: a judge grading against an empty question is a
        # different instrument, and the number it produces is not the preregistered one.
        raise ScoringError(f"{refusal} (corpora_root={corpora_root})")
    if n_skipped:
        print(
            f"  judging SKIPPED {n_skipped} run(s) with no question text "
            f"{dict(sorted(no_question_by_suite.items()))}; no judge-derived endpoint reads "
            "those suites, so the skip cannot move a preregistered number.",
            file=sys.stderr,
        )
    pairs: list[tuple[str, str]] = []
    for suite_id in sorted({i.task.suite_id for i in inputs}):
        for p in judge_arm_pairs(suite_id):
            if p not in pairs:
                pairs.append(p)
    return harness.judge_runs(
        judge,
        inputs,
        cap_usd=cap_usd,
        judge_model=judge_model,
        judge_family=judge_family,
        arm_pairs=tuple(pairs),
        sigma_j=sigma_j,
        seed=seed,
    )


# --------------------------------------------------------------------------- append semantics


class ParquetSchemaDrift(ScoringError):
    """An existing parquet cannot be cast to the current schema. Raised BEFORE any judging."""


def assert_appendable(d: Path, names: Sequence[str]) -> None:
    """Every file `score()` will append to can still be cast to the schema it will be cast to.

    CHECKED BEFORE THE MONEY, NOT AFTER IT. `_append` does `existing.cast(schema)` at the very
    end of `score()`, long after `run_judges` has dispatched and been billed for every judge
    call. The on-disk judgments.parquet carried the pre-rename 15-column schema
    (`judge_id`, `presentation_order`, `winner`, `tie`, `prompt_hash`) against the current 22,
    so the first judged `pi score` would have raised

        ValueError: Target schema's field names are not matching the table's field names

    after paying for the entire judging pass and writing nothing. A preflight costs one read of
    an empty-to-small file and converts an expensive late crash into a cheap early refusal.
    """
    for name in names:
        path = d / f"{name}.parquet"
        if not path.exists():
            continue
        try:
            pq.read_table(path).cast(sch.schema_for(name))
        except Exception as exc:  # noqa: BLE001 - the type varies by pyarrow version
            on_disk = list(pq.read_table(path).schema.names)
            want = list(sch.schema_for(name).names)
            raise ParquetSchemaDrift(
                f"{path} cannot be cast to the current {name} schema and would fail at write "
                f"time, after any judging is already paid for.\n"
                f"  on disk ({len(on_disk)}): {on_disk}\n"
                f"  current ({len(want)}): {want}\n"
                f"  missing from disk: {sorted(set(want) - set(on_disk))}\n"
                f"  gone from code   : {sorted(set(on_disk) - set(want))}\n"
                f"  ({type(exc).__name__}) Delete the file to rebuild it, or add the rename to "
                "pi_run.compact._RENAMED if the column was renamed rather than replaced."
            ) from exc


def _append(path: Path, name: str, new_rows: Sequence[Mapping[str, Any]], key) -> tuple[int, int]:
    """Concatenate, never rewrite. Existing rows keep their position and their bytes.

    Deduplicated on `key` so re-running the identical scorer twice is a no-op rather than a
    doubling — which matters because 'run it again to be sure' is the most common thing a
    tired researcher does at 2am.
    """
    schema = sch.schema_for(name)
    existing = pq.read_table(path) if path.exists() else sch.empty_table(name)
    have = {key(r) for r in existing.to_pylist()}
    add = [r for r in new_rows if key(r) not in have]
    if not add:
        return existing.num_rows, 0
    combined = pa.concat_tables([existing.cast(schema), sch.to_table(name, list(add))])
    pq.write_table(combined, path)
    return combined.num_rows, len(add)


def _score_key(r: Mapping[str, Any]) -> tuple:
    return (r["run_id"], r["metric_name"], r["scorer_hash"])


def _judgment_key(r: Mapping[str, Any]) -> tuple:
    """judgment_id is the identity: it already hashes the judge, the prompt and the order."""
    return (r["judgment_id"],)


def _match_key(r: Mapping[str, Any]) -> tuple:
    return (r["run_id"], r["node_id"], r["matcher_id"], r["graph_version"])


# --------------------------------------------------------------------------- the stage


@dataclass(frozen=True, slots=True)
class ScoreResult:
    parquet_dir: str
    scorer_hash: str
    metric_defs_hash: str
    graph_hash: str
    matcher_hash: str
    judge_pins: tuple[str, ...]
    graph_version: str
    n_runs_scored: int
    n_runs_skipped_no_graph: int
    scores_rows_added: int
    scores_rows_total: int
    matches_rows_added: int
    matches_rows_total: int
    metrics_emitted: tuple[str, ...]
    answers_available: int
    judgments_rows_added: int = 0
    judgments_rows_total: int = 0
    judging: str = "skipped: no judge client"
    judge_report: Mapping[str, Any] = field(default_factory=dict)
    # What the JUDGE cost. `--spend-cap` is checked inside `run_sweep` and covers rollouts
    # only, so judging -- which runs over every eligible run with a report and gold, for every
    # preregistered arm pair -- was a second provider bill that nothing measured and nothing
    # bounded. `kpr_incremental` is a primary endpoint, so this is not a small path.
    judge_spend: Mapping[str, float] = field(default_factory=dict)

    # suite -> count, for runs whose corpus_hash disagrees with their gold graph's.
    # Non-empty means a rebuild has moved the gold out from under finished runs.
    runs_skipped_corpus_mismatch: Mapping[str, int] = field(default_factory=dict)

    # Which of the scored suites took the ANSWERS-ONLY branch. Recorded because a table has to
    # be able to say which scorer branch produced it: `evidence_coverage` absent for frames is
    # a property of the benchmark, and `evidence_coverage` absent for musique is a bug, and
    # nothing downstream could otherwise tell the two apart.
    gold_free_suites: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "parquet_dir": self.parquet_dir,
            "scorer_hash": self.scorer_hash,
            "metric_defs_hash": self.metric_defs_hash,
            "graph_hash": self.graph_hash,
            "matcher_hash": self.matcher_hash,
            "judge_pins": list(self.judge_pins),
            "graph_version": self.graph_version,
            "n_runs_scored": self.n_runs_scored,
            "n_runs_skipped_no_graph": self.n_runs_skipped_no_graph,
            "gold_free": bool(self.gold_free_suites),
            "gold_free_suites": list(self.gold_free_suites),
            "judge_spend": dict(sorted(self.judge_spend.items())),
            "runs_skipped_corpus_mismatch": dict(sorted(self.runs_skipped_corpus_mismatch.items())),
            "scores_rows_added": self.scores_rows_added,
            "scores_rows_total": self.scores_rows_total,
            "matches_rows_added": self.matches_rows_added,
            "matches_rows_total": self.matches_rows_total,
            "judgments_rows_added": self.judgments_rows_added,
            "judgments_rows_total": self.judgments_rows_total,
            "judging": self.judging,
            "judge_report": dict(self.judge_report),
            "metric_families": sorted({family_of(m) for m in self.metrics_emitted}),
            "answers_available": self.answers_available,
        }


def _read(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def _group(rows: Sequence[Mapping[str, Any]], key: str = "run_id") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r[key]), []).append(dict(r))
    return out


def score(
    parquet_dir: str | Path = "scores/parquet",
    *,
    runs_root: str | Path | None = "runs",
    graph_version: str = DEFAULT_GRAPH_VERSION,
    judge_pins: Sequence[str] = (),
    # Judging is a SECOND provider bill that --spend-cap (a run_sweep concept) never saw.
    judge_spend_cap: float | None = None,
    matcher: Any | None = None,
    null_draws: int = 8,
    seed: int = 0,
    judge: Any | None = None,
    judge_model: str = "",
    judge_family: str = "",
    corpora_root: str | Path | None = None,
    prereg_root: str | Path | None = None,
    # The drgym search cache, from which the citation judge's documents are replayed when a
    # run has no `judge_docs.json` sidecar -- which is every run, since nothing writes one.
    cache_root: str | Path | None = None,
) -> ScoreResult:
    try:
        gold_root()
    except GoldAccessError as exc:
        raise ScoringError(
            "pi score must run in the SCORING process, with PI_GOLD_ROOT set. "
            "If you are seeing this in a rollout worker, the firewall is working."
        ) from exc
    # FIREWALL LAYER 4, ARMED IN THE ONE PROCESS THAT HOLDS GOLD BY CONSTRUCTION.
    #
    # `assert_armed` had exactly one call site, in `pi_run.worker.assert_firewall` -- the rollout
    # path, which is the path that must NOT be able to read gold. This is the path that must, and
    # it never armed the layer at all. The scan itself was present (`MeteredClient.complete` calls
    # `assert_clean` on every judge request), but a scan against a registry it could not find
    # returns () exactly as a clean one does, and nothing recorded which had happened. Measured at
    # a4422c0 in a process with PI_GOLD_ROOT set and 12,576 wiki2 graphs in memory: `assert_clean`
    # over a judge payload carrying a real REGISTERED nonce returned clean, and `last_status()`
    # was None before and after -- so the layer was not merely unrecorded here, it was
    # undetectable.
    #
    # Raises CanaryLeak, deliberately un-wrapped: this is not a scoring problem to be reported
    # alongside a table, and a ScoringError is something a caller might reasonably catch.
    tripwire.assert_armed()

    d = Path(parquet_dir)
    runs = _read(d / "runs.parquet")
    if not runs:
        raise ScoringError(f"no runs in {d / 'runs.parquet'}: run `pi compact` first")
    turns = _group(_read(d / "turns.parquet"))
    evidence = _group(_read(d / "evidence.parquet"))
    env_calls = _group(_read(d / "env_calls.parquet"))
    ledger = _group(_read(d / "ledger.parquet"))
    # Outcome.native, long -> {run_id: {key: value}}. A suite with no environment contributes
    # no rows at all, so this stays empty rather than full of zeros.
    native: dict[str, dict[str, float]] = {}
    for row in _read(d / "native.parquet"):
        native.setdefault(str(row["run_id"]), {})[str(row["key"])] = float(row["value"])
    answers = load_answers(runs_root) if runs_root else {}

    graphs: dict[str, dict[str, GoldGraph]] = {}
    for suite in sorted({str(r["suite_id"]) for r in runs}):
        graphs[suite] = load_graphs(suite, graph_version)

    # BEFORE ANY JUDGING. See `assert_appendable`: the write happens at the end of this
    # function and a schema drift discovered there has already cost a full judging pass.
    assert_appendable(d, ("matches", "scores", "judgments"))

    mt = matcher or MechanicalMatcher()

    # ---- judging happens BEFORE the scorer_hash is computed, because the judge pins it
    # produces are part of that hash: a number is not reproducible unless the identity of the
    # instrument that produced it is inside its provenance.
    reason = "skipped: no judge client"
    if judge is None:
        judge, env_model, env_family, reason = judge_client_from_env()
        judge_model = judge_model or env_model
        judge_family = judge_family or env_family
    judged = None
    if judge is not None:
        if not judge_model:
            raise ScoringError(
                "a judge client was supplied without a judge model pin. It is never "
                "defaulted: two arms judged by different models and then compared is the "
                f"failure this refusal exists to make impossible. Pass judge_model= or set "
                f"{JUDGE_MODEL_ENV}."
            )
        judged = run_judges(
            runs=runs,
            graphs=graphs,
            answers=answers,
            judge=judge,
            judge_model=judge_model,
            judge_family=judge_family or "openai",
            corpora_root=Path(corpora_root) if corpora_root else _corpora_root(d),
            runs_root=runs_root,
            sigma_j=_prereg_sigma_j(prereg_root or d),
            seed=seed,
            cap_usd=judge_spend_cap,
            cache_root=cache_root,
        )
        reason = "judged" if judged is not None else "skipped: no run had both a report and gold"

    # THE DEAD BAND MUST COME FROM THE INSTRUMENT THAT TOOK THE MEASUREMENT.
    #
    # sigma_J is the judge's own noise floor and `pi_eval.report` uses it to suppress effects
    # smaller than it. A floor measured on one model applied to another is not conservative in
    # a knowable direction: a quieter judge's floor lets a noisier judge's effects through,
    # and a noisier judge's floor suppresses a quieter judge's real ones. Nothing compared the
    # two, because `load_sigma_j` returns only {metric: float} by design and the instrument's
    # identity lives in a sidecar nothing read.
    if judged is not None and judge_model:
        from pi_eval.prereg import sigma_j_judge

        measured_on = sigma_j_judge(prereg_root or d)
        if measured_on and measured_on != judge_model:
            raise ScoringError(
                f"sigma_J was measured on judge {measured_on!r} but these scores are being "
                f"judged by {judge_model!r}. The dead band belongs to the instrument that took "
                f"the measurement; re-run scripts/measure_sigma_j.py under this judge, or score "
                f"under the judge it was measured on."
            )

    judgment_rows = [j.to_row() for j in judged.judgments] if judged else []
    pins = list(judge_pins) + sorted({j.judge_pin for j in judged.judgments} if judged else ())
    sh = scorer_hash(
        metric_defs=metric_defs_hash(),
        graph=graph_hash(graphs),
        matcher=matcher_hash(mt),
        judge_pins=sorted(set(pins)),
    )

    # The null pool is per SUITE: every uid any arm's retriever surfaced for any task of it.
    # See _null_phis for why a per-task pool makes the null degenerate.
    pools: dict[str, list[str]] = {}
    for r in runs:
        k = str(r["suite_id"])
        pools.setdefault(k, [])
        for e in evidence.get(str(r["run_id"]), ()):
            pools[k].append(str(e["uid"]))
    pools = {k: sorted(set(v)) for k, v in pools.items()}

    score_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    n_scored = n_skipped = 0
    mismatched: dict[str, int] = {}
    for r in sorted(runs, key=lambda x: str(x["run_id"])):
        run_id = str(r["run_id"])
        graph = graphs.get(str(r["suite_id"]), {}).get(str(r["task_id"]))
        if graph is None:
            n_skipped += 1
            continue
        # THE GOLD MUST DESCRIBE THE CORPUS THE RUN WAS ROLLED AGAINST.
        #
        # The corpora tree is content-addressed; the gold tree is NOT -- one file per
        # (suite, graph_version), last build wins. So a rebuild silently repoints every gold
        # graph at a new corpus while finished runs still name the old one. The task ids
        # match, the graphs load, nothing errors -- and every gold_ev_uid refers to spans of
        # a corpus this run never saw, so `evidence_coverage` and the whole RNR ladder
        # collapse toward zero for a plumbing reason that is indistinguishable from a result.
        #
        # An empty hash on either side means "unknown" and is allowed: gold built before this
        # field existed, or a suite whose corpus is an upstream checkout. A DISAGREEMENT is
        # not allowed, and it is skipped and COUNTED rather than raised -- one stale suite
        # must not stop the other five from being scored, but it must be impossible to miss.
        # DIRECTORY against DIRECTORY. `gold_corpus_hash` is stamped by `write_corpus`'s return
        # value, which IS the directory name; comparing it against the manifest's `corpus_hash`
        # would compare two different hash functions over two different inputs and flag every
        # run as mismatched.
        want = str(getattr(graph, "gold_corpus_hash", "") or "")
        got = str(r.get("corpus_dir") or "")
        if want and got and want != got:
            key = f"{r['suite_id']}: gold={want[:12]} runs={got[:12]}"
            mismatched[key] = mismatched.get(key, 0) + 1
            n_skipped += 1
            continue
        n_scored += 1
        rec_answer = answers.get(run_id)
        traj = _Traj(
            turns=tuple(
                _Turn(
                    turn_idx=int(t["turn_idx"]),
                    retrieved_uids=tuple(t.get("retrieved_uids") or ()),
                    new_uids=tuple(t.get("new_uids") or ()),
                    action=_Act(text=str(t.get("question") or "")),
                )
                for t in sorted(turns.get(run_id, ()), key=lambda t: int(t["turn_idx"]))
            ),
            outcome=_Outcome(
                answer=None
                if rec_answer is None
                else _Answer(rec_answer.text, rec_answer.cited_uids)
            ),
        )
        records = list(mt.match(graph=graph, trajectory=traj, run_id=run_id))
        for m in records:
            match_rows.append(
                {
                    "run_id": m.run_id,
                    "suite_id": m.suite_id,
                    "task_id": m.task_id,
                    "node_id": m.node_id,
                    "match_kind": m.match_kind,
                    "matched_turn_idx": m.matched_turn_idx,
                    "matcher_id": m.matcher_id,
                    "matcher_family": m.matcher_family,
                    "matcher_score": float(m.matcher_score),
                    "threshold": float(m.threshold),
                    "graph_version": m.graph_version or graph_version,
                    "cited_uids": list(m.cited_uids),
                }
            )
        score_rows += score_run(
            r,
            graph=graph,
            turns=turns.get(run_id, ()),
            evidence=evidence.get(run_id, ()),
            env_calls=env_calls.get(run_id, ()),
            ledger=ledger.get(run_id, ()),
            records=records,
            answer=rec_answer,
            native=native.get(run_id, {}),
            null_draws=null_draws,
            null_pool=pools.get(str(r["suite_id"]), ()),
            seed=seed,
            gold_free=str(r["suite_id"]) in GOLD_FREE_SUITES,
        )

    # ---- judge-derived rows, one per (run, judge metric). Emitted only for runs the judge
    # actually produced a value for: a run whose verdict could not be parsed, or whose judge
    # was disqualified for position bias, contributes NO row rather than a zero.
    if judged is not None:
        graph_versions = {
            str(r["run_id"]): str(
                graphs.get(str(r["suite_id"]), {})
                .get(str(r["task_id"]), GoldGraph("", ""))
                .gold_graph_version
                or graph_version
            )
            for r in runs
        }
        for run_id, values in sorted(judged.metrics.items()):
            ns = judged.metric_n.get(run_id, {})
            for name, value in sorted(values.items()):
                # THE n THIS VALUE WAS COMPUTED OVER. Emitted with no `n=`, `_emit` defaults it
                # to 1 -- so kpr_incremental, the drgym PRIMARY endpoint, reached
                # scores.parquet claiming a sample size of one. 40 gold key points with
                # verdicts parsed for 2 gave keypoint_recall=1.0, n=1, and nothing downstream
                # could see that 38 were never judged. `report.floor_flag` divides by sqrt(n).
                n, why = ns.get(name, (1, ""))
                _emit(
                    score_rows,
                    run_id,
                    name,
                    value,
                    n=n,
                    graph_version=graph_versions.get(run_id, graph_version),
                    notes=f"judge-derived; {judge_model}" + (f"; {why}" if why else ""),
                )

    for row in score_rows:
        row["scorer_hash"] = sh

    m_total, m_added = _append(d / "matches.parquet", "matches", match_rows, _match_key)
    s_total, s_added = _append(d / "scores.parquet", "scores", score_rows, _score_key)
    j_total, j_added = _append(d / "judgments.parquet", "judgments", judgment_rows, _judgment_key)
    if judged is not None:
        # The instrument's own diagnostics: position bias, win/tie/loss, order inconsistency,
        # the length-adjusted effect and Krippendorff alpha. Beside the table, not inside it:
        # they are per (suite, criterion), and scores.parquet is keyed by run.
        (d / "judge_diagnostics.json").write_text(
            json.dumps(judged.as_dict(), indent=1, sort_keys=True, default=str) + "\n"
        )
    return ScoreResult(
        parquet_dir=str(d),
        scorer_hash=sh,
        metric_defs_hash=metric_defs_hash(),
        graph_hash=graph_hash(graphs),
        matcher_hash=matcher_hash(mt),
        judge_pins=tuple(sorted(set(pins))),
        graph_version=graph_version,
        n_runs_scored=n_scored,
        n_runs_skipped_no_graph=n_skipped,
        gold_free_suites=tuple(sorted({str(r["suite_id"]) for r in runs} & set(GOLD_FREE_SUITES))),
        runs_skipped_corpus_mismatch=dict(sorted(mismatched.items())),
        scores_rows_added=s_added,
        scores_rows_total=s_total,
        matches_rows_added=m_added,
        matches_rows_total=m_total,
        metrics_emitted=tuple(sorted({str(r["metric_name"]) for r in score_rows})),
        answers_available=len(answers),
        judgments_rows_added=j_added,
        judgments_rows_total=j_total,
        judging=reason,
        judge_report=judged.as_dict() if judged is not None else {},
        # Whatever the client can tell us. `spend()` is optional on the JudgeLLM protocol --
        # a scripted fake in a test has none -- so a client that cannot report is recorded as
        # UNKNOWN rather than as zero. A judge bill silently reported as $0.00 is exactly the
        # shape that lets an unbounded second provider bill go unnoticed.
        judge_spend=(
            dict(judge.spend())
            if judge is not None and callable(getattr(judge, "spend", None))
            else ({} if judge is None else {"unknown": 1.0})
        ),
    )


def _corpora_root(parquet_dir: Path) -> Path:
    """`scores/parquet` -> `data/corpora`: the public tree, two levels up from the tables.

    That inference is right when the parquet lives where `pi compact` puts it and WRONG the
    moment anyone passes `--parquet` somewhere else -- `/tmp/x/parquet` becomes
    `/tmp/data/corpora`, which does not exist, so every question came back empty. Caught the
    first time a live judge ran, by the guard that refuses to judge without question text;
    before that guard it would have judged four runs against nothing.

    So: fall back to the repo the code is running from when the inferred path has no corpora.
    An explicit `--corpora-root` beats both.
    """
    inferred = parquet_dir.resolve().parent.parent / "data" / "corpora"
    if inferred.is_dir():
        return inferred
    for cand in (Path.cwd(), *Path(__file__).resolve().parents):
        if (cand / "pyproject.toml").exists() and (cand / "data" / "corpora").is_dir():
            return cand / "data" / "corpora"
    return inferred


def _prereg_sigma_j(start: str | Path) -> dict[str, float]:
    from pi_eval.prereg import load_sigma_j

    return load_sigma_j(start)
