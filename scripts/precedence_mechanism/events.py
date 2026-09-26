"""Event-level reconstruction of `precedence_violation_rate` (Lane L1.3).

WHY THIS EXISTS. `artifacts/testsplit_plan_metrics_20260918/` measured the matched-cost
contrast (trained minus prompted) and found it adverse on MuSiQue: +0.0653 [+0.0118,
+0.1269] over 134 task pairs (`paper/results.tex:309-333`). That number is a RATE, one
scalar per run. To test a mechanism ("the trained policy skips a prerequisite when its
answer is already in parametric knowledge") the rate has to be broken back open into the
individual (parent, child) edges it was computed from -- which parent, which child, which
turn, and what became of the parent afterwards.

THE ARM-SPECIFIC BASIS, restated from `pinq_train.gate._matched_cost` and
`testsplit_plan_metrics_20260918/RESULT.md`'s "Reproducing" section. For a paired
(suite, task, seed): `k = trained_run.n_asks` (the trained arm is never truncated -- it is
already at its own natural stop); the baseline (prompted) arm is truncated to
`min(k, prompted_run.n_asks)`. A record is "in the arm's matched window" iff
`matched_turn_idx < basis_k`.

TWO DISTINCT THINGS ARE COMPUTED FOR EACH EDGE, on purpose, and must not be conflated:

  `counts_in_matched_metric`  True iff this edge is exactly what
                              `pi_eval.metrics.structure.precedence_violation_rate` counts as
                              a violation at matched cost: BOTH parent and child present in
                              the arm's MATCHED window, child's turn earlier. Summing this
                              column reproduces the published +0.0653 -- see
                              `lock_check` and `tests/test_precedence_mechanism.py`.

  `parent_resolved_later` /  Read off the run's FULL, untruncated trajectory (its own
  `parent_never_resolved`    natural stop), not the matched window. A matched-window
                              violation always has the parent resolved (that is what
                              qualifies it), so within that set alone these two columns are
                              constant. They earn their keep on the BROADER population this
                              module also emits: every edge where the child resolved before
                              the parent had resolved AT ALL, including children whose
                              matched-window partner resolved outside the window, or never.
                              That population is a superset of the counted metric and is
                              what step 1 of the brief asks for; `counts_in_matched_metric`
                              is how a reader tells which rows are inside the reported number
                              and which are the wider behavioural picture.

Nothing here reads or writes the shared store: every parquet path is passed in by the
caller, absolute, from the isolated `artifacts/testsplit_qa/scores_parquet` store the source
artifact names. Nothing here writes under `runs/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pi_eval.gold import GoldGraph
from pi_eval.matcher.base import MatchRecord
from pi_eval.metrics.structure import precedence_violation_rate

ARM_TRAINED = "inquirer_trained"
ARM_PROMPTED = "inquirer_prompted"


@dataclass(frozen=True, slots=True)
class RunArm:
    run_id: str
    n_asks: int


@dataclass(frozen=True, slots=True)
class NodeMatch:
    node_id: str
    match_kind: str
    turn: int


@dataclass(frozen=True, slots=True)
class ViolationEvent:
    suite: str
    task_id: str
    seed: int
    arm: str
    run_id: str
    basis_k: int
    parent_node_id: str
    child_node_id: str
    child_turn: int
    parent_match_kind: str
    child_match_kind: str
    parent_turn_full: int | None  # full, untruncated run; None if parent never resolved
    parent_resolved_later: bool
    parent_never_resolved: bool
    counts_in_matched_metric: bool  # True iff this is one of precedence_violation_rate's own
    answer_correct: float | None  # scores.parquet, run's own terminal (natural) value
    evidence_coverage: float | None  # scores.parquet, run's own terminal (natural) value


def full_records(con, store: str, run_id: str, suite: str, task_id: str) -> list[NodeMatch]:
    """Every matched node in `run_id`'s own full, natural trajectory. One row per node_id
    (measured on the isolated store: 9,781 of 9,781 (run_id, node_id) pairs are unique), so
    no ordering or dedup choice is being made here -- see the module docstring's "measured"
    note in `tests/test_precedence_mechanism.py`."""
    rows = con.execute(
        f"""
        select node_id, match_kind, matched_turn_idx
        from '{store}/matches.parquet'
        where run_id = ? and matched_turn_idx is not null
        """,
        [run_id],
    ).fetchall()
    return [NodeMatch(node_id=r[0], match_kind=r[1], turn=int(r[2])) for r in rows]


def run_arms(con, store: str, suite: str) -> dict[tuple[str, int], dict[str, RunArm]]:
    """{(task_id, seed): {arm_id: RunArm}} for the two inquiry arms on one suite."""
    rows = con.execute(
        f"""
        select task_id, seed, arm_id, run_id, n_asks
        from '{store}/runs.parquet'
        where suite_id = ? and arm_id in (?, ?)
        """,
        [suite, ARM_TRAINED, ARM_PROMPTED],
    ).fetchall()
    out: dict[tuple[str, int], dict[str, RunArm]] = {}
    for task_id, seed, arm_id, run_id, n_asks in rows:
        out.setdefault((task_id, int(seed)), {})[arm_id] = RunArm(run_id, int(n_asks))
    return out


def terminal_scores(con, store: str, run_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    """{run_id: {metric_name: value}}, the run's own stored terminal (natural, cap-8) value --
    never recomputed, because "at end" means the run's real end."""
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = con.execute(
        f"""
        select run_id, metric_name, value
        from '{store}/scores.parquet'
        where run_id in ({placeholders}) and metric_name in ('answer_correct', 'evidence_coverage')
        """,
        list(run_ids),
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for run_id, metric, value in rows:
        out.setdefault(run_id, {})[metric] = value
    return out


def _prereq_edges(graph: GoldGraph) -> list[tuple[str, str]]:
    return [
        (e.gold_src_node_id, e.gold_dst_node_id)
        for e in graph.gold_edges
        if e.gold_edge_kind == "prerequisite"
    ]


def events_for_arm(
    *,
    suite: str,
    task_id: str,
    seed: int,
    arm: str,
    run: RunArm,
    basis_k: int,
    full: list[NodeMatch],
    graph: GoldGraph,
    scores: Mapping[str, float],
) -> list[ViolationEvent]:
    """Every (parent, child) prerequisite edge where the child resolved in the arm's matched
    window while the parent had not resolved as of the child's turn, in this run's full
    trajectory. Superset of what `precedence_violation_rate` counts; see module docstring."""
    full_by_id = {m.node_id: m for m in full}
    matched_by_id = {m.node_id: m for m in full if m.turn < basis_k}
    out: list[ViolationEvent] = []
    for parent_id, child_id in _prereq_edges(graph):
        child = matched_by_id.get(child_id)
        if child is None:
            continue  # child not reached within this arm's matched window: out of scope
        parent_full = full_by_id.get(parent_id)
        if parent_full is not None and parent_full.turn <= child.turn:
            continue  # correctly ordered: not a skip
        parent_matched = matched_by_id.get(parent_id)
        counts_in_matched_metric = parent_matched is not None and parent_matched.turn > child.turn
        out.append(
            ViolationEvent(
                suite=suite,
                task_id=task_id,
                seed=seed,
                arm=arm,
                run_id=run.run_id,
                basis_k=basis_k,
                parent_node_id=parent_id,
                child_node_id=child_id,
                child_turn=child.turn,
                parent_match_kind=(parent_full.match_kind if parent_full else ""),
                child_match_kind=child.match_kind,
                parent_turn_full=(parent_full.turn if parent_full else None),
                parent_resolved_later=parent_full is not None,
                parent_never_resolved=parent_full is None,
                counts_in_matched_metric=counts_in_matched_metric,
                answer_correct=scores.get("answer_correct"),
                evidence_coverage=scores.get("evidence_coverage"),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class QualifyingEdge:
    """One prerequisite edge where BOTH parent and child are present in the arm's matched
    window -- exactly `precedence_violation_rate`'s own `total` population, one row per edge
    instead of one scalar per run. `is_violation` is that function's own `<` test. This is
    the population `scripts/precedence_mechanism/stratify.py` computes a rate over; the
    broader `ViolationEvent` table (parent resolved later or never, possibly outside the
    matched window) is the descriptive superset for the mechanism narrative, not this one."""

    suite: str
    task_id: str
    seed: int
    arm: str
    run_id: str
    parent_node_id: str
    child_node_id: str
    is_violation: bool


def qualifying_edges_for_arm(
    *,
    suite: str,
    task_id: str,
    seed: int,
    arm: str,
    run: RunArm,
    basis_k: int,
    full: list[NodeMatch],
    graph: GoldGraph,
) -> list[QualifyingEdge]:
    matched_by_id = {m.node_id: m for m in full if m.turn < basis_k}
    out: list[QualifyingEdge] = []
    for parent_id, child_id in _prereq_edges(graph):
        u = matched_by_id.get(parent_id)
        v = matched_by_id.get(child_id)
        if u is None or v is None:
            continue
        out.append(
            QualifyingEdge(
                suite=suite,
                task_id=task_id,
                seed=seed,
                arm=arm,
                run_id=run.run_id,
                parent_node_id=parent_id,
                child_node_id=child_id,
                is_violation=v.turn < u.turn,
            )
        )
    return out


def build_all_qualifying_edges(
    con, store: str, graphs: Mapping[str, GoldGraph], suite: str
) -> list[QualifyingEdge]:
    pairs = run_arms(con, store, suite)
    out: list[QualifyingEdge] = []
    for (task_id, seed), arms in sorted(pairs.items()):
        trained = arms.get(ARM_TRAINED)
        prompted = arms.get(ARM_PROMPTED)
        if trained is None or prompted is None:
            continue
        graph = graphs.get(task_id)
        if graph is None:
            continue
        k = trained.n_asks
        for arm, run, basis_k in (
            (ARM_TRAINED, trained, k),
            (ARM_PROMPTED, prompted, min(k, prompted.n_asks)),
        ):
            full = full_records(con, store, run.run_id, suite, task_id)
            out.extend(
                qualifying_edges_for_arm(
                    suite=suite,
                    task_id=task_id,
                    seed=seed,
                    arm=arm,
                    run=run,
                    basis_k=basis_k,
                    full=full,
                    graph=graph,
                )
            )
    return out


def build_all_events(
    con, store: str, graphs: Mapping[str, GoldGraph], suite: str
) -> list[ViolationEvent]:
    pairs = run_arms(con, store, suite)
    all_run_ids = [ra.run_id for arms in pairs.values() for ra in arms.values()]
    scores = terminal_scores(con, store, all_run_ids)
    events: list[ViolationEvent] = []
    for (task_id, seed), arms in sorted(pairs.items()):
        trained = arms.get(ARM_TRAINED)
        prompted = arms.get(ARM_PROMPTED)
        if trained is None or prompted is None:
            continue
        graph = graphs.get(task_id)
        if graph is None:
            continue
        k = trained.n_asks
        for arm, run, basis_k in (
            (ARM_TRAINED, trained, k),
            (ARM_PROMPTED, prompted, min(k, prompted.n_asks)),
        ):
            full = full_records(con, store, run.run_id, suite, task_id)
            events.extend(
                events_for_arm(
                    suite=suite,
                    task_id=task_id,
                    seed=seed,
                    arm=arm,
                    run=run,
                    basis_k=basis_k,
                    full=full,
                    graph=graph,
                    scores=scores.get(run.run_id, {}),
                )
            )
    return events


def lock_check(
    con, store: str, graphs: Mapping[str, GoldGraph], suite: str
) -> tuple[int, int, int]:
    """Recompute precedence_violation_rate PER RUN from the same truncated record set this
    module builds, via the real (unmodified) metric function, and compare against the
    formally-counted subset of `build_all_events`'s own output.

    Returns (n_runs_checked, n_mismatches, n_matched_events). Called from
    `tests/test_precedence_mechanism.py` against the real store; a nonzero mismatch count
    means the pairing/truncation logic above has drifted from the audited instrument and
    nothing downstream may be trusted.
    """
    pairs = run_arms(con, store, suite)
    n_checked = 0
    n_mismatch = 0
    n_matched_events = 0
    for (task_id, seed), arms in sorted(pairs.items()):
        trained = arms.get(ARM_TRAINED)
        prompted = arms.get(ARM_PROMPTED)
        if trained is None or prompted is None:
            continue
        graph = graphs.get(task_id)
        if graph is None:
            continue
        k = trained.n_asks
        for arm, run, basis_k in (
            (ARM_TRAINED, trained, k),
            (ARM_PROMPTED, prompted, min(k, prompted.n_asks)),
        ):
            full = full_records(con, store, run.run_id, suite, task_id)
            truncated = [m for m in full if m.turn < basis_k]
            records = [
                MatchRecord(
                    run_id=run.run_id,
                    suite_id=suite,
                    task_id=task_id,
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
            official_rate = precedence_violation_rate(records, graph)
            n_checked += 1

            events = events_for_arm(
                suite=suite,
                task_id=task_id,
                seed=seed,
                arm=arm,
                run=run,
                basis_k=basis_k,
                full=full,
                graph=graph,
                scores={},
            )
            counted = [e for e in events if e.counts_in_matched_metric]
            n_matched_events += len(counted)
            total_edges = sum(
                1
                for parent_id, child_id in _prereq_edges(graph)
                if {m.node_id for m in truncated} >= {parent_id, child_id}
            )
            rebuilt_rate = (len(counted) / total_edges) if total_edges else float("nan")
            both_nan = official_rate != official_rate and rebuilt_rate != rebuilt_rate
            if not both_nan and abs((official_rate or 0) - (rebuilt_rate or 0)) > 1e-9:
                n_mismatch += 1
    return n_checked, n_mismatch, n_matched_events
