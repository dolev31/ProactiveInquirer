"""DB and statistics glue for lane L1.11.

Reuses `scripts.stopping_answer_test.lib` (population loading, the BCa estimators it already
built and tested) rather than re-deriving a second copy -- the same reasoning that module's own
docstring gives for importing `pinq_train.gate` private names: a script is outside
import-linter's `root_packages`, so nothing here is a firewall concern, and a vetted function
beats a second hand-rolled one that could quietly diverge.

THE NEW PART THIS LANE NEEDS is node-identity-aware coverage: `matches.parquet` (per
`run_id, node_id`) crossed against the answer-node set `answer_node.py` computes from gold, and
a span-level (not node-level) retrieved-vs-gold-uid split for the non-answer-node coverage
decomposition. Neither exists in `scripts.stopping_answer_test.lib`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.answer_node_coverage.answer_node import AnswerNodeResult
from scripts.stopping_answer_test import lib as stopping_lib

from pi_eval.stats.inference import paired_difference
from pinq_train.gate import _in, _rows

EXPECTED_MATCHER_ID = "mechanical_v3"
EXPECTED_MATCHER_FAMILY = "rule"

__all__ = [
    "open_store",
    "covered_map",
    "retrieved_uids_by_run",
    "span_hit_counts",
    "cross_check_evidence_coverage",
    "p_covered_delta",
    "p_stop_given_uncovered_delta",
    "recall_by_coverage_group",
    "within_both_covered_recall_delta",
    "node_group_ladder",
    "matched_cost_gain_for_group",
    "coverage_gain_share",
]


def open_store(store: Path):
    """`stopping_lib.gate_con`'s connection (runs/turns/scores/ledger/calls), plus `matches`
    and `evidence` views that function does not attach because `pi train gate` never reads
    them. Asserts the matcher this lane depends on is the only one in the store -- a silent
    second `matcher_id` would mean `covered_map` is reading two different instruments as one.
    """
    con = stopping_lib.gate_con(store)
    for name in ("matches", "evidence"):
        p = Path(store) / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing from the isolated store {store}")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    seen = _rows(con, "SELECT DISTINCT matcher_id, matcher_family FROM matches")
    bad = [
        r
        for r in seen
        if r["matcher_id"] != EXPECTED_MATCHER_ID or r["matcher_family"] != EXPECTED_MATCHER_FAMILY
    ]
    if bad:
        raise ValueError(
            f"matches carries matcher(s) other than {EXPECTED_MATCHER_ID!r}: {bad}. "
            "covered_map's RESOLVE/USE threshold is calibrated to that matcher; a different "
            "one may not share the same match_kind semantics."
        )
    return con


# --------------------------------------------------------------------------- node-level: covered


def covered_map(
    con, runs: Sequence[Mapping[str, Any]], answer_nodes: Mapping[tuple[str, str], AnswerNodeResult]
) -> dict[str, bool | None]:
    """`run_id -> is the task's answer node covered by this run`.

    COVERED means `match_kind in ('resolve', 'use')` -- rank >= 2, the same bar
    `pi_eval.metrics.discovery.rnr` calls RESOLVE and the same bar `evidence_coverage`'s own
    definition operates at (a node's evidence uids fully retrieved), read from `matches.parquet`
    per the brief ("the same matcher the scorer uses"), not recomputed.

    `None` (not `False`) when the task's answer-node set is empty (`AnswerNodeResult.is_empty`,
    the `no_required_nodes` branch) -- there is no node to have covered, which is a different
    fact from "covered and it was not".
    """
    ids = [str(r["run_id"]) for r in runs]
    if not ids:
        return {}
    rows = _rows(
        con,
        f"SELECT run_id, node_id, match_kind FROM matches WHERE run_id IN {_in(ids)}",
    )
    resolved_by_run: dict[str, set[str]] = {}
    for r in rows:
        if r["match_kind"] in ("resolve", "use"):
            resolved_by_run.setdefault(str(r["run_id"]), set()).add(str(r["node_id"]))

    out: dict[str, bool | None] = {}
    for r in runs:
        rid = str(r["run_id"])
        key = (str(r["suite_id"]), str(r["task_id"]))
        res = answer_nodes.get(key)
        if res is None or res.is_empty:
            out[rid] = None
            continue
        out[rid] = bool(resolved_by_run.get(rid, set()) & set(res.node_ids))
    return out


# --------------------------------------------------------------------------- span-level: retrieved uids


def retrieved_uids_by_run(con, run_ids: Sequence[str]) -> dict[str, set[str]]:
    """`run_id -> the run's full retrieved evidence-uid set, exactly `pi_eval.score.score_run`'s
    `all_uids` (union of every turn's `retrieved_uids` and `evidence.uid`) -- so a coverage
    number computed here on this set reproduces the scorer's own numerator, checked below by
    `cross_check_evidence_coverage`.
    """
    ids = list({str(x) for x in run_ids})
    if not ids:
        return {}
    out: dict[str, set[str]] = {rid: set() for rid in ids}
    for r in _rows(con, f"SELECT run_id, retrieved_uids FROM turns WHERE run_id IN {_in(ids)}"):
        out[str(r["run_id"])].update(str(u) for u in (r["retrieved_uids"] or ()))
    for r in _rows(con, f"SELECT run_id, uid FROM evidence WHERE run_id IN {_in(ids)}"):
        out[str(r["run_id"])].add(str(r["uid"]))
    return out


def span_hit_counts(
    con,
    runs: Sequence[Mapping[str, Any]],
    gold_partition: Mapping[tuple[str, str], tuple[frozenset[str], frozenset[str]]],
) -> dict[str, dict[str, int]]:
    """Per run: `{"n_answer_hit", "n_answer_gold", "n_nonanswer_hit", "n_nonanswer_gold"}`,
    span-level (not node-level -- a multi-span node partially retrieved shows up as partial
    credit here, unlike `covered_map`'s all-or-nothing node view). Used only for the
    coverage-gain decomposition; `covered_map` remains the source of truth for "is the answer
    node covered".
    """
    retrieved = retrieved_uids_by_run(con, [r["run_id"] for r in runs])
    out: dict[str, dict[str, int]] = {}
    for r in runs:
        rid = str(r["run_id"])
        key = (str(r["suite_id"]), str(r["task_id"]))
        ans_uids, non_uids = gold_partition.get(key, (frozenset(), frozenset()))
        got = retrieved.get(rid, set())
        out[rid] = {
            "n_answer_hit": len(got & ans_uids),
            "n_answer_gold": len(ans_uids),
            "n_nonanswer_hit": len(got & non_uids),
            "n_nonanswer_gold": len(non_uids),
        }
    return out


def cross_check_evidence_coverage(
    con,
    runs: Sequence[Mapping[str, Any]],
    hit_counts: Mapping[str, dict[str, int]],
    *,
    scorer_hash: str,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """(n_checked, n_mismatched, examples): does `(n_answer_hit+n_nonanswer_hit)/(n_answer_gold
    +n_nonanswer_gold)` reproduce the run's own stored `evidence_coverage`? This is the
    provenance check CONTRIBUTING.md rule 1 asks for -- the span partition must reunite to the exact
    quantity already published, not a plausible-looking approximation of it.
    """
    stored = stopping_lib._metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    n_checked = 0
    mismatches: list[dict[str, Any]] = []
    for r in runs:
        rid = str(r["run_id"])
        want = stored.get(rid)
        h = hit_counts.get(rid)
        if want is None or h is None:
            continue
        gold_n = h["n_answer_gold"] + h["n_nonanswer_gold"]
        if gold_n == 0:
            continue
        got = (h["n_answer_hit"] + h["n_nonanswer_hit"]) / gold_n
        n_checked += 1
        if abs(got - want) > tol:
            mismatches.append({"run_id": rid, "recomputed": got, "stored": want})
    return {"n_checked": n_checked, "n_mismatched": len(mismatches), "examples": mismatches[:10]}


# --------------------------------------------------------------------------- task-level aggregation


def _task_ratio_rows(
    trained_runs: Sequence[Mapping[str, Any]],
    prompted_runs: Sequence[Mapping[str, Any]],
    trained_ind: Mapping[str, float | None],
    prompted_ind: Mapping[str, float | None],
) -> list[tuple[str, float, float, float, float]]:
    """`(task_id, trained_num, trained_den, prompted_num, prompted_den)` for
    `bca_paired_ratio_delta`, where `*_ind` maps run_id -> 0/1 (or None to exclude that run
    from both numerator and denominator -- `covered_map`'s `no_required_nodes` case)."""
    by_task_t: dict[str, list[str]] = {}
    for r in trained_runs:
        by_task_t.setdefault(str(r["task_id"]), []).append(str(r["run_id"]))
    by_task_p: dict[str, list[str]] = {}
    for r in prompted_runs:
        by_task_p.setdefault(str(r["task_id"]), []).append(str(r["run_id"]))

    rows: list[tuple[str, float, float, float, float]] = []
    for task in sorted(set(by_task_t) & set(by_task_p)):
        t_vals = [trained_ind[rid] for rid in by_task_t[task] if trained_ind.get(rid) is not None]
        p_vals = [prompted_ind[rid] for rid in by_task_p[task] if prompted_ind.get(rid) is not None]
        rows.append(
            (task, float(sum(t_vals)), float(len(t_vals)), float(sum(p_vals)), float(len(p_vals)))
        )
    return rows


def p_covered_delta(
    trained_runs: Sequence[Mapping[str, Any]],
    prompted_runs: Sequence[Mapping[str, Any]],
    covered: Mapping[str, bool | None],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """P(answer node covered), trained vs prompted, task-clustered BCa delta (ratio of sums,
    same estimator family as the campaign's other stop/coverage rates)."""
    ind = {rid: (1.0 if v else 0.0) if v is not None else None for rid, v in covered.items()}
    rows = _task_ratio_rows(trained_runs, prompted_runs, ind, ind)

    def compute(nb: int, sd: int) -> dict[str, Any]:
        return stopping_lib.bca_paired_ratio_delta(rows, n_boot=nb, seed=sd)

    out = stopping_lib.with_stability(compute, seed=seed)
    t_num, t_den = sum(r[1] for r in rows), sum(r[2] for r in rows)
    p_num, p_den = sum(r[3] for r in rows), sum(r[4] for r in rows)
    out["level_trained"] = (t_num / t_den) if t_den else float("nan")
    out["level_prompted"] = (p_num / p_den) if p_den else float("nan")
    out["n_tasks"] = len(rows)
    return out


def p_stop_given_uncovered_delta(
    trained_runs: Sequence[Mapping[str, Any]],
    prompted_runs: Sequence[Mapping[str, Any]],
    covered: Mapping[str, bool | None],
    stop_reason_by_run: Mapping[str, str],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """P(stop_reason == 'policy_stop' | answer node NOT covered), trained vs prompted.

    Denominator is runs whose answer node is uncovered (`covered is False`; `None` -- no
    answer node exists -- is excluded from both arms, same as `p_covered_delta`). Numerator is
    the subset of those that stopped voluntarily rather than being cut off by the harness
    (`stop_reason == 'policy_stop'`, the same STOP definition `pinq_train.gate._stop_2x2` uses).
    """

    def num_den(runs: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for r in runs:
            rid = str(r["run_id"])
            c = covered.get(rid)
            if c is None or c:
                continue  # covered, or undefined: not part of this conditional population
            stopped = stop_reason_by_run.get(rid) == "policy_stop"
            out[rid] = (1.0 if stopped else 0.0, 1.0)
        return out

    t_nd = num_den(trained_runs)
    p_nd = num_den(prompted_runs)

    by_task_t: dict[str, list[str]] = {}
    for r in trained_runs:
        by_task_t.setdefault(str(r["task_id"]), []).append(str(r["run_id"]))
    by_task_p: dict[str, list[str]] = {}
    for r in prompted_runs:
        by_task_p.setdefault(str(r["task_id"]), []).append(str(r["run_id"]))

    rows: list[tuple[str, float, float, float, float]] = []
    for task in sorted(set(by_task_t) & set(by_task_p)):
        t_rows = [t_nd[rid] for rid in by_task_t[task] if rid in t_nd]
        p_rows = [p_nd[rid] for rid in by_task_p[task] if rid in p_nd]
        rows.append(
            (
                task,
                sum(x[0] for x in t_rows),
                sum(x[1] for x in t_rows),
                sum(x[0] for x in p_rows),
                sum(x[1] for x in p_rows),
            )
        )

    def compute(nb: int, sd: int) -> dict[str, Any]:
        return stopping_lib.bca_paired_ratio_delta(rows, n_boot=nb, seed=sd)

    out = stopping_lib.with_stability(compute, seed=seed)
    t_num, t_den = sum(r[1] for r in rows), sum(r[2] for r in rows)
    p_num, p_den = sum(r[3] for r in rows), sum(r[4] for r in rows)
    out["level_trained"] = (t_num / t_den) if t_den else float("nan")
    out["level_prompted"] = (p_num / p_den) if p_den else float("nan")
    out["n_tasks"] = len(rows)
    out["n_uncovered_trained"] = int(t_den)
    out["n_uncovered_prompted"] = int(p_den)
    return out


# --------------------------------------------------------------------------- recall conditional on coverage


def _task_covered_groups(
    runs: Sequence[Mapping[str, Any]], covered: Mapping[str, bool | None]
) -> tuple[set[str], set[str]]:
    """A task is in the COVERED group for this arm if any of its runs (seeds) covered the
    answer node, and in NOT-COVERED if it has at least one run and none did. A task with only
    `None` (no answer node) runs is in neither."""
    by_task: dict[str, list[bool | None]] = {}
    for r in runs:
        by_task.setdefault(str(r["task_id"]), []).append(covered.get(str(r["run_id"])))
    covered_tasks, uncovered_tasks = set(), set()
    for task, vals in by_task.items():
        defined = [v for v in vals if v is not None]
        if not defined:
            continue
        (covered_tasks if any(defined) else uncovered_tasks).add(task)
    return covered_tasks, uncovered_tasks


def recall_by_coverage_group(
    con,
    runs: Sequence[Mapping[str, Any]],
    covered: Mapping[str, bool | None],
    *,
    scorer_hash: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Within ONE arm: task-level `answer_token_recall`, covered-tasks vs not-covered-tasks,
    two independent samples (`bca_two_sample_mean_delta`, the same tool
    `stopping_answer_test.step3_coverage_to_answer` uses for its high/low coverage-delta
    split -- here the split variable is answer-NODE identity, not total coverage sign)."""
    covered_tasks, uncovered_tasks = _task_covered_groups(runs, covered)
    recall = stopping_lib.task_level_metric(
        con, runs, "answer_token_recall", scorer_hash=scorer_hash
    )
    # task_level_metric keys on (suite_id, task_id); this lane's runs are one suite at a time.
    by_task_id = {k[1]: v for k, v in recall.items()}
    cov_vals = [by_task_id[t] for t in covered_tasks if t in by_task_id]
    unc_vals = [by_task_id[t] for t in uncovered_tasks if t in by_task_id]

    def compute(nb: int, sd: int) -> dict[str, Any]:
        return stopping_lib.bca_two_sample_mean_delta(cov_vals, unc_vals, n_boot=nb, seed=sd)

    out = stopping_lib.with_stability(compute, seed=seed)
    out["n_covered_tasks"] = len(cov_vals)
    out["n_uncovered_tasks"] = len(unc_vals)
    out["mean_covered"] = stopping_lib._mean(cov_vals)
    out["mean_uncovered"] = stopping_lib._mean(unc_vals)
    return out


def _paired_with_stability(
    a: Mapping[str, float], b: Mapping[str, float], *, n_boot: int, seed: int
) -> dict[str, Any]:
    """`paired_difference` (task-clustered BCa + sign-flip), routed through `with_stability`
    so a bound within 0.01 of zero gets the campaign's mandatory 50k/3-seed reread rather than
    being read off the 10k call. Matches `stopping_answer_test.step2_answer_quality`'s own
    convention of dropping `p_value` once wrapped: `with_stability`'s reread replaces `lo`/`hi`/
    `point` from the 50k, seed-0 replicate but has no slot for a p-value recomputed at that
    resample count, so carrying the STALE 10k p-value through would silently relabel it.
    """

    def compute(nb: int, sd: int) -> dict[str, Any]:
        est = paired_difference(a, b, clusters=None, n_boot=nb, seed=sd)
        return {"point": est.point, "lo": est.ci_lo, "hi": est.ci_hi, "n": est.n}

    return stopping_lib.with_stability(compute, seed=seed)


def within_both_covered_recall_delta(
    con,
    trained_runs: Sequence[Mapping[str, Any]],
    prompted_runs: Sequence[Mapping[str, Any]],
    covered: Mapping[str, bool | None],
    *,
    scorer_hash: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """trained-minus-prompted `answer_token_recall`, paired on task, RESTRICTED to tasks where
    BOTH arms covered the answer node on at least one seed. `paired_difference` (task-clustered
    BCa + sign-flip), the campaign's standard paired estimator -- restricted to a task subset,
    not a new statistic."""
    t_cov, _ = _task_covered_groups(trained_runs, covered)
    p_cov, _ = _task_covered_groups(prompted_runs, covered)
    both = t_cov & p_cov

    t_recall = stopping_lib.task_level_metric(
        con, trained_runs, "answer_token_recall", scorer_hash=scorer_hash
    )
    p_recall = stopping_lib.task_level_metric(
        con, prompted_runs, "answer_token_recall", scorer_hash=scorer_hash
    )
    a = {k[1]: v for k, v in t_recall.items() if k[1] in both}
    b = {k[1]: v for k, v in p_recall.items() if k[1] in both}
    out = _paired_with_stability(a, b, n_boot=n_boot, seed=seed)
    out["n_both_covered_tasks"] = len(both)
    return out


# --------------------------------------------------------------------------- coverage-gain decomposition


def node_group_ladder(
    con,
    runs: Sequence[Mapping[str, Any]],
    gold_group_by_task: Mapping[tuple[str, str], frozenset[str]],
) -> dict[str, dict[int, float]]:
    """`run_id -> {k: coverage of `gold_group_by_task[task]` after the first k turns}`.

    Reconstructed from `turns.retrieved_uids` (ordered by `turn_idx`, cumulative), exactly the
    merge `pi_eval.score` performs to build `scores.frontier_q#k` -- confirmed against it on
    the real store: `frontier_q#0 == 0.0` and `frontier_q#k` for `k=1..n_asks` matches turn
    count exactly (checked by hand before writing this). NOT a `pi score` re-run: `pi score`
    scores the POOLED required-evidence ladder only, into the shared instrument; this rebuilds
    the same walk from data already in the isolated store, restricted to an arbitrary uid
    subset, and writes nothing.

    Absent (not zero) for a run whose task has an empty `gold_group` -- coverage of zero gold
    spans is undefined, the same convention `_coverage_ladder` uses for a task with no
    required evidence at all.
    """
    ids = [str(r["run_id"]) for r in runs]
    if not ids:
        return {}
    turn_rows = _rows(
        con,
        f"SELECT run_id, turn_idx, retrieved_uids FROM turns WHERE run_id IN {_in(ids)} "
        "ORDER BY run_id, turn_idx",
    )
    turns_by_run: dict[str, list[tuple[int, list[str]]]] = {}
    for r in turn_rows:
        turns_by_run.setdefault(str(r["run_id"]), []).append(
            (int(r["turn_idx"]), list(r["retrieved_uids"] or ()))
        )

    out: dict[str, dict[int, float]] = {}
    for r in runs:
        rid = str(r["run_id"])
        key = (str(r["suite_id"]), str(r["task_id"]))
        group = gold_group_by_task.get(key)
        if not group:
            continue
        ordered = sorted(turns_by_run.get(rid, ()), key=lambda x: x[0])
        cum: set[str] = set()
        ladder = {0: 0.0}
        for k, (_, uids) in enumerate(ordered, start=1):
            cum.update(uids)
            ladder[k] = len(cum & group) / len(group)
        out[rid] = ladder
    return out


def matched_cost_gain_for_group(
    ck_runs: Sequence[Mapping[str, Any]],
    ba_runs: Sequence[Mapping[str, Any]],
    ladders: Mapping[str, Mapping[int, float]],
    n_asks_by_run: Mapping[str, int],
) -> dict[tuple[str, str], float]:
    """Per-task matched-cost delta for ONE node group's ladder: checkpoint's own terminal
    coverage of that group minus the mean, over same-`(suite, task, seed)` baseline runs, of
    their group-coverage at `min(checkpoint_k, baseline_n_asks)`.

    Structurally identical to `pinq_train.gate._matched_cost` / this package's
    `matched_cost_coverage_by_task` -- same `_by_key` grouping, same prefix-rung rule -- just
    parametrized over an arbitrary ladder instead of hardcoding `frontier_q#k`. Validated to
    reproduce that function's own pooled mean when handed the FULL required-gold-uid ladder
    (`tests/test_answer_node_coverage.py`, and cross-checked against the published
    `evidence_coverage` matched-cost delta on the real store in `RESULT.md`).
    """
    ck_keys = stopping_lib._by_key(ck_runs)
    ba_keys = stopping_lib._by_key(ba_runs)
    shared = sorted(set(ck_keys) & set(ba_keys))
    per_task: dict[tuple[str, str], list[float]] = {}
    for suite, task, _seed in shared:
        deltas: list[float] = []
        for c_rid in ck_keys[(suite, task, _seed)]:
            c_ladder = ladders.get(c_rid)
            k = n_asks_by_run.get(c_rid, 0)
            c_val = c_ladder.get(k) if c_ladder else None
            if c_val is None:
                continue
            rungs = []
            for b_rid in ba_keys[(suite, task, _seed)]:
                b_n = n_asks_by_run.get(b_rid, 0)
                rung = (ladders.get(b_rid) or {}).get(min(k, b_n))
                if rung is not None:
                    rungs.append(float(rung))
            if rungs:
                deltas.append(c_val - sum(rungs) / len(rungs))
        if deltas:
            per_task.setdefault((suite, task), []).append(sum(deltas) / len(deltas))
    return {k: sum(v) / len(v) for k, v in per_task.items()}


def coverage_gain_share(
    con,
    trained_runs: Sequence[Mapping[str, Any]],
    prompted_runs: Sequence[Mapping[str, Any]],
    hit_counts: Mapping[str, dict[str, int]],
    gold_partition_by_task: Mapping[tuple[str, str], tuple[frozenset[str], frozenset[str]]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Share of the trained arm's MATCHED-COST coverage gain landing on non-answer-node spans.

    Memory `unmatched-cost-inverts-the-sign` applies here verbatim: the trained arm asks far
    fewer questions, so its RAW terminal hit-counts can be (and on StrategyQA, ARE) lower than
    the prompted arm's on both node groups even though its MATCHED-COST coverage is higher on
    every suite. The raw hit-count deltas are still reported below (`sum_delta_*_hits_raw`),
    but the headline `share_nonanswer_of_gain` is built from `matched_cost_gain_for_group`, the
    same instrument the published `+0.1238/+0.0628/+0.0776` headline uses.

    THE DECOMPOSITION IS EXACT, not an approximation, by linearity: for disjoint groups
    answer/nonanswer partitioning the required gold uids, `coverage_full(run, k) =
    (|Ga|*cov_Ga(run,k) + |Gn|*cov_Gn(run,k)) / (|Ga|+|Gn|)` at every run and every k (the
    weights `|Ga|`, `|Gn|` are graph properties, constant across runs of one task), so the
    per-task matched-cost delta on the full ladder is exactly the `|G|`-weighted sum of the two
    groups' own per-task matched-cost deltas. Averaging that per-task identity over tasks
    preserves it, so `pooled_answer_weighted + pooled_nonanswer_weighted == pooled_total`
    holds exactly (asserted in the test, and cross-checked against
    `matched_cost_coverage_by_task`'s already-published-number-reproducing pooled mean here).
    """
    n_asks_by_run = {
        str(r["run_id"]): int(r["n_asks"] or 0) for r in [*trained_runs, *prompted_runs]
    }
    answer_group = {k: v[0] for k, v in gold_partition_by_task.items()}
    nonanswer_group = {k: v[1] for k, v in gold_partition_by_task.items()}
    full_group = {k: v[0] | v[1] for k, v in gold_partition_by_task.items()}

    ladders_answer = node_group_ladder(con, [*trained_runs, *prompted_runs], answer_group)
    ladders_nonanswer = node_group_ladder(con, [*trained_runs, *prompted_runs], nonanswer_group)
    ladders_full = node_group_ladder(con, [*trained_runs, *prompted_runs], full_group)

    mc_answer = matched_cost_gain_for_group(
        trained_runs, prompted_runs, ladders_answer, n_asks_by_run
    )
    mc_nonanswer = matched_cost_gain_for_group(
        trained_runs, prompted_runs, ladders_nonanswer, n_asks_by_run
    )
    mc_full_reconstructed = matched_cost_gain_for_group(
        trained_runs, prompted_runs, ladders_full, n_asks_by_run
    )

    # THE TASK POPULATION MUST MATCH `mc_full_reconstructed`'s, OR THE IDENTITY IS BETWEEN TWO
    # DIFFERENT SETS OF TASKS, NOT A CHECK OF ONE. On StrategyQA 86 of 200 tasks (43%) have BOTH
    # required nodes tied at the same max depth -- a flat two-fact comparison with no deeper
    # node -- so the depth-fallback answer-node set IS every required node and `nonanswer_group`
    # is empty BY CONSTRUCTION, not by a missing measurement. Its weight `|Gn|/|G|` is exactly
    # 0 there, so the group's own (otherwise-undefined) matched-cost delta must contribute 0,
    # not drop the task from the sum: `set(mc_answer) & set(mc_nonanswer)` would silently
    # exclude all 86, which is the wrong population for the pooled mean this cross-checks
    # against. `answer_group`/`nonanswer_group` genuinely undefined (present in neither dict)
    # only for a task whose FULL required-gold-uid set is itself empty -- ruled out already,
    # `evidence_coverage` is undefined there too and such a task is absent from the population
    # this function is ever called on.
    shared_tasks = sorted(set(full_group))
    weighted_answer: dict[str, float] = {}
    weighted_nonanswer: dict[str, float] = {}
    for suite, task in shared_tasks:
        na, nn = len(answer_group[(suite, task)]), len(nonanswer_group[(suite, task)])
        denom = na + nn
        if denom == 0:
            continue
        wa_val = 0.0
        if na:
            val = mc_answer.get((suite, task))
            if val is None:
                continue
            wa_val = (na / denom) * val
        wn_val = 0.0
        if nn:
            val = mc_nonanswer.get((suite, task))
            if val is None:
                continue
            wn_val = (nn / denom) * val
        weighted_answer[task] = wa_val
        weighted_nonanswer[task] = wn_val

    pooled_answer = stopping_lib._mean(list(weighted_answer.values()))
    pooled_nonanswer = stopping_lib._mean(list(weighted_nonanswer.values()))
    pooled_total = pooled_answer + pooled_nonanswer
    share_nonanswer = (pooled_nonanswer / pooled_total) if pooled_total else float("nan")

    # raw (unmatched, terminal) hit-count deltas, for context only -- see docstring.
    def task_means(runs: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
        by_task: dict[str, list[float]] = {}
        for r in runs:
            h = hit_counts.get(str(r["run_id"]))
            if h is not None:
                by_task.setdefault(str(r["task_id"]), []).append(float(h[key]))
        return {t: stopping_lib._mean(v) for t, v in by_task.items()}

    raw_shared = (
        set(task_means(trained_runs, "n_answer_hit"))
        & set(task_means(prompted_runs, "n_answer_hit"))
        & set(task_means(trained_runs, "n_nonanswer_hit"))
        & set(task_means(prompted_runs, "n_nonanswer_hit"))
    )
    t_ans, p_ans = (
        task_means(trained_runs, "n_answer_hit"),
        task_means(prompted_runs, "n_answer_hit"),
    )
    t_non, p_non = (
        task_means(trained_runs, "n_nonanswer_hit"),
        task_means(prompted_runs, "n_nonanswer_hit"),
    )
    sum_delta_answer_raw = sum(t_ans[t] - p_ans[t] for t in raw_shared)
    sum_delta_nonanswer_raw = sum(t_non[t] - p_non[t] for t in raw_shared)

    return {
        "n_tasks": len(weighted_answer),
        "n_tasks_zero_nonanswer_group": sum(1 for t in shared_tasks if not nonanswer_group[t]),
        "share_nonanswer_of_gain": share_nonanswer,
        "pooled_matched_cost_answer_weighted": pooled_answer,
        "pooled_matched_cost_nonanswer_weighted": pooled_nonanswer,
        "pooled_matched_cost_total_reconstructed": pooled_total,
        "pooled_matched_cost_total_direct_check": stopping_lib._mean(
            list(mc_full_reconstructed.values())
        ),
        "sum_delta_answer_hits_raw_unmatched": sum_delta_answer_raw,
        "sum_delta_nonanswer_hits_raw_unmatched": sum_delta_nonanswer_raw,
    }
