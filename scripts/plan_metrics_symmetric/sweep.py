"""The reproduction lock of `ladder.py`, extended from one plan metric to all twelve.

WHY THIS FILE EXISTS. `artifacts/symmetric_matched_cost_20260919/RESULT.md` establishes that
the matched-cost cells of `artifacts/testsplit_plan_metrics_20260918/contrasts.json` were
produced by code that is not in this repository ("case 3": no committed script rebuilds the
per-k ladder these metrics are read off). Lane L2.3's `ladder.py` closed that gap for exactly
one metric, `precedence_violation_rate`, and the design that makes it worth anything is
`reproduce_and_recompute`: it REFUSES to report a recomputed number until the reimplementation
first reproduces the PUBLISHED one. Without that lock, "our number differs" cannot be told
apart from "our code differs". This module runs that same lock across every remaining matched
cell and reports, per cell, whether the published value reproduces.

WHAT IT DOES NOT DO. It does not edit `ladder.py`. Lane L2.3 owns that module. The five
metrics `ladder.METRIC_FNS` does not define are added to it at RUNTIME, inside
`registered_metrics`, which refuses to shadow a name `ladder` already defines and restores the
table on exit. The pairing, the truncation and the bootstrap are therefore literally L2.3's
code on every cell -- which is the only way a lock on one metric says anything about another.

TWO METRICS ARE NOT RECORD-LEVEL, AND RIDE A SYNTHETIC ENCODING.
`evidence_coverage` is defined over retrieved evidence UIDs and `stop_overshoot` over the
TURN COUNT; neither reads `matches.parquet`. Rather than reimplement the truncation for them
(a second copy of the one thing that must not diverge), `coverage_ladder_records` encodes a
run's turn-level retrieval AS `MatchRecord`s: one `ev::<uid>` record per gold uid, stamped
with the turn it was first retrieved, plus one `turn::<i>` sentinel per turn row. Truncating
that set at `matched_turn_idx < k` -- `RunLadder.at`, unmodified -- yields exactly the gold
uids seen in the first k turns AND, by counting sentinels, exactly `k_hat = min(k, n_turns)`.
`pi_eval.score`'s own arithmetic is then restated over those two quantities and nothing else.

EVERY LOCK IS PAIRED WITH A NON-VACUITY PROBE. A published delta of exactly 0.0 reproduces
under any aggregation rule whatsoever, so a LOCKED verdict on such a cell certifies nothing.
`lock_cell` therefore also recomputes the cell with the checkpoint ladder shifted one rung
SHORT (`n_asks - 1`) and records whether that moves the delta away from the published value.
A cell whose delta does not move is reported `discriminating = False` and must be excluded
from any tally of locks -- the same control `artifacts/testsplit_plan_metrics_20260918/
RESULT.md` calls "NON-VACUITY" on its own interior lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import duckdb

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:  # `python scripts/plan_metrics_symmetric/sweep.py` puts the
    sys.path.insert(0, str(_SCRIPTS))  # package DIR on the path, not the package's parent.

from plan_metrics_symmetric import ladder  # noqa: E402

from pi_eval.matcher.base import MatchRecord  # noqa: E402
from pi_eval.metrics import discovery, latent, structure  # noqa: E402

# `_gold_uids` is imported rather than restated: it is the DENOMINATOR of every coverage
# number in the paper (required-partition spans only), and a second spelling of it here would
# be a silent divergence in the one quantity this module is locking against.
from pi_eval.score import _gold_uids  # noqa: E402
from pi_eval.stats.inference import paired_difference  # noqa: E402

EV_PREFIX = "ev::"
TURN_PREFIX = "turn::"

TOL = 1e-6
"""The lock tolerance, `ladder.reproduce_and_recompute`'s own default. Published cells are
carried at full float precision in `contrasts.json`, so anything above this is a real
difference in the computation, not a printing artefact."""


# ----------------------------------------------------------------- the synthetic encoding


def coverage_ladder_records(
    *,
    run_id: str,
    suite_id: str,
    task_id: str,
    turn_uids: Sequence[Sequence[str]],
    graph: Any,
) -> tuple[MatchRecord, ...]:
    """One run's turn-level retrieval, encoded so `ladder.RunLadder.at` can truncate it.

    `turn_uids[i]` is the uid list retrieved at turn i, in turn order. Only GOLD uids get an
    `ev::` record (a non-gold uid can never enter the numerator, and keeping it would make the
    record count read as evidence), each stamped with the FIRST turn it appeared on -- the
    scorer accumulates `seen |= retrieved`, so a later re-retrieval changes nothing.
    """
    gold = _gold_uids(graph)
    first: dict[str, int] = {}
    for i, uids in enumerate(turn_uids):
        for uid in uids or ():
            if uid in gold and uid not in first:
                first[uid] = i

    def rec(node_id: str, turn: int, kind: str) -> MatchRecord:
        return MatchRecord(
            run_id=run_id,
            suite_id=suite_id,
            task_id=task_id,
            node_id=node_id,
            match_kind=kind,  # type: ignore[arg-type]
            matched_turn_idx=turn,
            matcher_id="plan_metrics_symmetric.sweep",
            matcher_family="synthetic",
            matcher_score=1.0,
            threshold=1.0,
            graph_version=str(getattr(graph, "gold_graph_version", "") or ""),
        )

    out = [rec(EV_PREFIX + uid, t, "resolve") for uid, t in sorted(first.items())]
    # The sentinels are what make `k_hat` survive truncation. They are rank 0 ("none"), so no
    # rank-reading metric can mistake them for a resolved need.
    out += [rec(f"{TURN_PREFIX}{i}", i, "none") for i in range(len(turn_uids))]
    return tuple(out)


def _read_coverage_ladder(records: Iterable[MatchRecord]) -> tuple[dict[str, int], int]:
    """(gold uid -> first turn, number of turn rows surviving). Raises on a record set that is
    not a coverage ladder: handing `matches.parquet` records to a uid-level metric would
    return a plausible 0.0 instead of an error."""
    at: dict[str, int] = {}
    k_hat = 0
    for r in records:
        if r.node_id.startswith(EV_PREFIX):
            at[r.node_id[len(EV_PREFIX) :]] = int(r.matched_turn_idx or 0)
        elif r.node_id.startswith(TURN_PREFIX):
            k_hat += 1
        else:
            raise ValueError(
                f"{r.node_id!r} is not a coverage-ladder record; this metric reads a ladder "
                "built by coverage_ladder_records, never matches.parquet"
            )
    return at, k_hat


# ------------------------------------------------- the five metrics `ladder` does not define


def _evidence_coverage(records: Sequence[MatchRecord], graph: Any) -> float:
    at, _ = _read_coverage_ladder(records)
    return discovery.evidence_coverage(set(at), set(_gold_uids(graph)))


def _stop_overshoot(records: Sequence[MatchRecord], graph: Any) -> float:
    """`max(0, k_hat - k*)`, restated from `pi_eval.score` over the truncated prefix.

    `k_hat` is the number of turns the run HAS after truncation, not its untruncated length:
    the matched-cost rule removes the comparator's later turns, and charging it for them would
    measure the truncation. NaN rungs go in as 0.0 and the argmax breaks ties toward the
    EARLIEST index, both exactly as the scorer does it.
    """
    at, k_hat = _read_coverage_ladder(records)
    gold = set(_gold_uids(graph))
    qs = [
        discovery.evidence_coverage({u for u, t in at.items() if t < i}, gold)
        for i in range(k_hat + 1)
    ]
    finite = [0.0 if math.isnan(q) else q for q in qs]
    k_star = max(range(len(finite)), key=lambda i: (finite[i], -i))
    return float(max(0, k_hat - k_star))


def _facet_total(records: Sequence[MatchRecord], graph: Any) -> float:
    """`|facets|` in the gold graph. It does not read `records` AT ALL -- which is the whole
    content of its published cell being +0.0 on every suite at both bases."""
    return float(len(graph.gold_facets))


def _latent_discovery_rate(records: Sequence[MatchRecord], graph: Any) -> float:
    return latent.latent_labels(graph, records).latent_discovery_rate


def _newly_reachable_share(records: Sequence[MatchRecord], graph: Any) -> float:
    """Of the turns that took a LATENT need, the share where that need had just become
    askable. Absent (NaN), never 0.0, where no latent need was taken: `pi_eval.score` emits no
    row there, so a 0.0 would invent a population."""
    lab = latent.latent_labels(graph, records)
    if lab.n_latent_available <= 0:
        return float("nan")
    took = [t for t in lab.per_turn.values() if t.is_latent]
    if not took:
        return float("nan")
    return sum(1 for t in took if t.newly_reachable) / len(took)


EXTRA_METRIC_FNS: dict[str, Callable[[Sequence[MatchRecord], Any], float]] = {
    "evidence_coverage": _evidence_coverage,
    "stop_overshoot": _stop_overshoot,
    "facet_total": _facet_total,
    "latent_discovery_rate": _latent_discovery_rate,
    "newly_reachable_share": _newly_reachable_share,
}


def _facet_breadth_scorer(records: Sequence[MatchRecord], graph: Any) -> float:
    """`pi_eval.score`'s `facet_breadth`: `float(touched)`, a COUNT, emitted even when the
    graph carries no facets at all (`0/0` reading as "no breadth", which that artifact's own
    RESULT.md calls out as a dilution on wiki2).

    `ladder.METRIC_FNS["facet_breadth"]` is `touched/total`, NaN where `total == 0` -- a
    different function under the same name. This one is registered under a DIFFERENT name and
    only ever reported as a diagnostic beside the published cell, because deciding which of
    the two the paper should use is not this lane's call.
    """
    touched, _total = structure.facet_breadth(records, graph, level="resolve")
    return float(touched)


DIAGNOSTIC_METRIC_FNS: dict[str, Callable[[Sequence[MatchRecord], Any], float]] = {
    "facet_breadth_scorer": _facet_breadth_scorer,
}
"""Not published cells. Each entry answers "which function produced the published number?"
for a cell whose lock fails under the name `ladder` gives it."""

COVERAGE_LADDER_METRICS = frozenset({"evidence_coverage", "stop_overshoot"})
"""The metrics whose arms must be built by `load_coverage_ladders`, not `load_run_ladders`."""


@contextmanager
def registered_metrics(
    extra: Mapping[str, Callable[[Sequence[MatchRecord], Any], float]],
) -> Iterator[Mapping[str, Callable[[Sequence[MatchRecord], Any], float]]]:
    """Add metric functions to `ladder.METRIC_FNS` for the duration of a block.

    Runtime registration, not an edit: `ladder.py` is another lane's file and the point of
    this sweep is that every cell runs ITS pairing and truncation. Shadowing a name it already
    defines is refused -- that would silently lock a different function than the one under
    test -- and the table is restored on the way out.
    """
    clash = sorted(set(extra) & set(ladder.METRIC_FNS))
    if clash:
        raise ValueError(f"ladder.METRIC_FNS already defines {clash}; refusing to shadow it")
    before = dict(ladder.METRIC_FNS)
    ladder.METRIC_FNS.update(extra)
    try:
        yield ladder.METRIC_FNS
    finally:
        ladder.METRIC_FNS.clear()
        ladder.METRIC_FNS.update(before)


# ------------------------------------------------------------------- the published values


@dataclass(frozen=True)
class PublishedCell:
    metric: str
    suite: str
    published_delta: float
    published_lo: float
    published_hi: float
    published_n: int
    delta_distinct: int
    n_pairs: int
    pairs_dropped: int


def published_matched_cells(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], PublishedCell]:
    """The `matched`-basis, TASK-clustered cells of `contrasts.json`.

    Two ways to read the wrong number out of that file: the `unmatched` row carries a
    different delta under the same metric name, and every row also carries a `template`-
    clustered block. The published table this lock is against is the task-clustered matched
    column, and nothing else.
    """
    out: dict[tuple[str, str], PublishedCell] = {}
    for r in rows:
        if r.get("basis") != "matched":
            continue
        task = r["task"]
        out[(str(r["metric"]), str(r["suite"]))] = PublishedCell(
            metric=str(r["metric"]),
            suite=str(r["suite"]),
            published_delta=float(task["delta"]),
            published_lo=float(task["lo"]),
            published_hi=float(task["hi"]),
            published_n=int(task["n"]),
            delta_distinct=int(r.get("delta_distinct") or 0),
            n_pairs=int(r.get("n_pairs") or 0),
            pairs_dropped=int(r.get("pairs_dropped") or 0),
        )
    return out


# ------------------------------------------------- the DOCUMENTED pairing, as a diagnostic


@dataclass(frozen=True)
class SeedMatchedResult:
    metric: str
    suite: str
    delta: float
    ci_lo: float
    ci_hi: float
    n: int
    n_pairs: int
    """Pairs KEPT, which is what `contrasts.json`'s own `n_pairs` column counts: MEASURED, its
    musique `precedence_violation_rate` row carries n_pairs 245 and pairs_dropped 105 against
    350 shared (task, seed) keys."""
    n_pairs_shared: int
    pairs_dropped: int


def seed_matched_contrast(
    metric: str,
    ckpt: Mapping[str, ladder.RunLadder],
    base: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    seeds: Mapping[str, int],
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> SeedMatchedResult:
    """The pairing the published artifact says it used, run as an ALTERNATIVE HYPOTHESIS.

    `artifacts/testsplit_plan_metrics_20260918/RESULT.md:176`: "trained at k = its n_asks,
    baseline at min(k, its n_asks); average seeds into the task". Its step 4 is explicit that
    the pair is formed at `(suite, task, seed)`. `ladder.asymmetric_contrast` folds seeds into
    one `(suite, task)` key and averages every checkpoint run against EVERY baseline run of
    that task, so the same baseline run is read at two different truncations and both enter
    its mean. The two rules coincide only where the truncation does nothing.

    This function exists to tell those two apart on a cell that fails the lock. It is NOT a
    replacement for `ladder`'s rule and does not edit it: which of the two produced the
    published table is a question about the published table, and it is answered by measuring
    both against the same published float. `n_pairs` and `pairs_dropped` are returned because
    `contrasts.json` publishes them per cell, which makes them a second, independent check on
    a delta that happens to agree.
    """
    fn = ladder.METRIC_FNS[metric]

    def keyed(
        arm: Mapping[str, ladder.RunLadder], what: str
    ) -> dict[tuple[str, str, int], ladder.RunLadder]:
        out: dict[tuple[str, str, int], ladder.RunLadder] = {}
        for rid, lad in arm.items():
            key = (lad.suite_id, lad.task_id, int(seeds[rid]))
            if key in out:
                raise ValueError(
                    f"{what} has more than one run at {key}: {out[key].run_id} and {rid}. A "
                    "(task, seed) pairing cannot be formed from this cohort."
                )
            out[key] = lad
        return out

    ck_by = keyed(ckpt, "checkpoint arm")
    ba_by = keyed(base, "baseline arm")
    shared = sorted(set(ck_by) & set(ba_by))

    acc: dict[str, tuple[list[float], list[float]]] = {}
    dropped = 0
    for key in shared:
        _, task, _ = key
        graph = graphs.get(task)
        if graph is None:
            dropped += 1
            continue
        c_lad, b_lad = ck_by[key], ba_by[key]
        k = c_lad.n_asks
        a_val = fn(c_lad.at(k), graph)
        b_val = fn(b_lad.at(min(k, b_lad.n_asks)), graph)
        if math.isnan(a_val) or math.isnan(b_val):
            dropped += 1
            continue
        a_list, b_list = acc.setdefault(task, ([], []))
        a_list.append(a_val)
        b_list.append(b_val)

    per_task_a = {t: sum(a) / len(a) for t, (a, _) in acc.items()}
    per_task_b = {t: sum(b) / len(b) for t, (_, b) in acc.items()}
    est = paired_difference(per_task_a, per_task_b, n_boot=n_boot, seed=seed)
    return SeedMatchedResult(
        metric=metric,
        suite=next(iter(ckpt.values())).suite_id if ckpt else "",
        delta=est.point,
        ci_lo=est.ci_lo,
        ci_hi=est.ci_hi,
        n=est.n,
        n_pairs=len(shared) - dropped,
        n_pairs_shared=len(shared),
        pairs_dropped=dropped,
    )


# --------------------------------------------------------------------- the lock and probe


def probe_verdict(probe_delta: float, published_delta: float, tol: float) -> tuple[bool, str]:
    """Did the one-rung shift move this cell away from the published value?

    THREE OUTCOMES, NOT TWO. A probe that returns NaN moved the cell too -- the shift left it
    undefined, so the published value would not have reproduced -- but it is weaker evidence
    than a numeric move, because it shows the POPULATION collapsing rather than the value
    differing. The two are reported apart rather than pooled into one boolean.
    """
    if math.isnan(probe_delta):
        return True, (
            "a one-rung-short ladder leaves this cell undefined (no paired task keeps a "
            "defined value), so the shift does change the outcome"
        )
    if abs(probe_delta - published_delta) > tol:
        return True, (
            "a one-rung-short ladder gives a different delta, so reproducing the published "
            "value could have failed"
        )
    return False, (
        "a one-rung-short ladder gives the SAME delta, so this cell reproduces under a wrong "
        "ladder too and the shift cannot make it fail"
    )


@dataclass(frozen=True)
class LockRow:
    metric: str
    suite: str
    published_delta: float | None
    published_n: int | None
    repro_delta: float | None
    repro_n: int | None
    abs_diff: float | None
    verdict: str
    reason: str
    repro_ci_lo: float | None
    repro_ci_hi: float | None
    published_ci_lo: float | None
    published_ci_hi: float | None
    n_unsafe_pairs: int | None
    probe_delta: float | None
    discriminating: bool | None
    discriminating_reason: str
    # The DOCUMENTED (suite, task, seed) pairing, run on the same arms against the same
    # published float. Null when no seed map was supplied.
    seedmatched_delta: float | None = None
    seedmatched_n: int | None = None
    seedmatched_abs_diff: float | None = None
    seedmatched_verdict: str | None = None
    seedmatched_n_pairs: int | None = None
    seedmatched_pairs_dropped: int | None = None
    seedmatched_probe_delta: float | None = None
    seedmatched_discriminating: bool | None = None
    seedmatched_discriminating_reason: str | None = None
    published_n_pairs: int | None = None
    published_pairs_dropped: int | None = None
    pair_counts_match: bool | None = None


def _shift_one_rung_short(arm: Mapping[str, ladder.RunLadder]) -> dict[str, ladder.RunLadder]:
    """The same arm read one question short. The published-number lock's non-vacuity control:
    if this does NOT move the delta, the instrument cannot tell the published value from a
    wrong one and reproducing it is not evidence."""
    return {
        rid: ladder.RunLadder(
            run_id=lad.run_id,
            task_id=lad.task_id,
            suite_id=lad.suite_id,
            n_asks=max(0, lad.n_asks - 1),
            records=lad.records,
        )
        for rid, lad in arm.items()
    }


def lock_cell(
    metric: str,
    ckpt: Mapping[str, ladder.RunLadder],
    base: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    *,
    published_delta: float | None,
    published_n: int | None,
    published_lo: float | None = None,
    published_hi: float | None = None,
    tol: float = TOL,
    seed: int = 0,
    n_boot: int = 1000,
    probe_n_boot: int = 20,
    seeds: Mapping[str, int] | None = None,
    published_n_pairs: int | None = None,
    published_pairs_dropped: int | None = None,
) -> LockRow:
    """One (metric, suite) cell: reproduce, then probe.

    `probe_n_boot` is small on purpose -- the probe reads only the POINT estimate, which
    `paired_difference` computes from the observed units and not from the resamples.
    """
    blank = dict(
        metric=metric,
        suite=next(iter(ckpt.values())).suite_id if ckpt else "",
        published_delta=published_delta,
        published_n=published_n,
        published_ci_lo=published_lo,
        published_ci_hi=published_hi,
        repro_delta=None,
        repro_n=None,
        abs_diff=None,
        repro_ci_lo=None,
        repro_ci_hi=None,
        n_unsafe_pairs=None,
        probe_delta=None,
        discriminating=None,
        discriminating_reason="not probed",
        published_n_pairs=published_n_pairs,
        published_pairs_dropped=published_pairs_dropped,
    )
    if published_delta is None:
        return LockRow(
            **blank,
            verdict="UNAVAILABLE",
            reason="no matched-basis, task-clustered cell for this (metric, suite) in the "
            "published contrasts file",
        )
    if not ckpt or not base:
        return LockRow(
            **blank,
            verdict="UNAVAILABLE",
            reason="no runs loaded for one or both arms on this suite",
        )

    if seeds is not None:
        sm = seed_matched_contrast(metric, ckpt, base, graphs, seeds, seed=seed, n_boot=n_boot)
        sm_diff = None if math.isnan(sm.delta) else abs(sm.delta - published_delta)
        blank.update(
            seedmatched_delta=None if math.isnan(sm.delta) else sm.delta,
            seedmatched_n=sm.n,
            seedmatched_abs_diff=sm_diff,
            seedmatched_verdict=(
                "UNAVAILABLE" if sm_diff is None else ("LOCKED" if sm_diff <= tol else "FAILED")
            ),
            seedmatched_n_pairs=sm.n_pairs,
            seedmatched_pairs_dropped=sm.pairs_dropped,
            pair_counts_match=(
                None
                if published_n_pairs is None
                else (
                    sm.n_pairs == published_n_pairs
                    and sm.pairs_dropped == (published_pairs_dropped or 0)
                )
            ),
        )
        # A PROBE QUALIFIES A VERDICT, so it runs under the rule that produced that verdict.
        sm_probe = seed_matched_contrast(
            metric,
            _shift_one_rung_short(ckpt),
            base,
            graphs,
            seeds,
            seed=seed,
            n_boot=probe_n_boot,
        )
        sm_moved, sm_why = probe_verdict(sm_probe.delta, published_delta, tol)
        blank.update(
            seedmatched_probe_delta=(None if math.isnan(sm_probe.delta) else sm_probe.delta),
            seedmatched_discriminating=sm_moved,
            seedmatched_discriminating_reason=sm_why,
        )

    got = ladder.asymmetric_contrast(metric, ckpt, base, graphs, seed=seed, n_boot=n_boot)
    blank.update(
        repro_delta=got.delta,
        repro_n=got.n,
        repro_ci_lo=got.ci_lo,
        repro_ci_hi=got.ci_hi,
        n_unsafe_pairs=got.n_unsafe,
    )
    if math.isnan(got.delta):
        return LockRow(
            **blank,
            verdict="UNAVAILABLE",
            reason=f"reimplementation is undefined on this suite (n={got.n} paired tasks with "
            "a defined value on both arms)",
        )
    diff = abs(got.delta - published_delta)
    blank.update(abs_diff=diff)

    probe = ladder.asymmetric_contrast(
        metric, _shift_one_rung_short(ckpt), base, graphs, seed=seed, n_boot=probe_n_boot
    )
    moved, why = probe_verdict(probe.delta, published_delta, tol)
    blank.update(
        probe_delta=None if math.isnan(probe.delta) else probe.delta,
        discriminating=moved,
        discriminating_reason=why,
    )
    return LockRow(
        **blank,
        verdict="LOCKED" if diff <= tol else "FAILED",
        reason=(
            f"|reimplementation - published| = {diff:.3e} <= {tol:g}"
            if diff <= tol
            else f"|reimplementation - published| = {diff:.3e} > {tol:g}; the published number "
            "has no reproducible source in this repository"
        ),
    )


# ---------------------------------------------------------------------------- the loaders


def load_coverage_ladders(
    parquet_dir: Path,
    run_ids: Sequence[str],
    graphs: Mapping[str, Any],
) -> dict[str, ladder.RunLadder]:
    """`RunLadder`s carrying the SYNTHETIC coverage encoding, for the two metrics that are not
    read off `matches.parquet`."""
    con = duckdb.connect()
    ids = list(run_ids)
    ph = ",".join("?" for _ in ids)
    turns = con.execute(
        f"SELECT run_id, turn_idx, retrieved_uids "
        f"FROM read_parquet('{(Path(parquet_dir) / 'turns.parquet').as_posix()}') "
        f"WHERE run_id IN ({ph}) ORDER BY run_id, turn_idx",
        ids,
    ).fetchdf()
    runs = con.execute(
        f"SELECT run_id, suite_id, task_id, n_asks "
        f"FROM read_parquet('{(Path(parquet_dir) / 'runs.parquet').as_posix()}') "
        f"WHERE run_id IN ({ph})",
        ids,
    ).fetchdf()

    by_run: dict[str, list[list[str]]] = {rid: [] for rid in ids}
    for row in turns.itertuples(index=False):
        uids = list(row.retrieved_uids) if row.retrieved_uids is not None else []
        by_run.setdefault(row.run_id, []).append([str(u) for u in uids])

    out: dict[str, ladder.RunLadder] = {}
    for row in runs.itertuples(index=False):
        graph = graphs.get(row.task_id)
        if graph is None:
            continue
        out[row.run_id] = ladder.RunLadder(
            run_id=row.run_id,
            task_id=row.task_id,
            suite_id=row.suite_id,
            n_asks=int(row.n_asks or 0),
            records=coverage_ladder_records(
                run_id=row.run_id,
                suite_id=row.suite_id,
                task_id=row.task_id,
                turn_uids=by_run.get(row.run_id, []),
                graph=graph,
            ),
        )
    return out


def terminal_check(
    metric: str,
    arms: Mapping[str, ladder.RunLadder],
    graphs: Mapping[str, Any],
    stored: Mapping[tuple[str, str], float],
    *,
    tol: float = 1e-9,
) -> dict[str, int]:
    """At `k = n_asks` the ladder must reproduce the SCORER'S OWN stored value, run by run.

    This is the instrument check, separate from the published-number lock: it says whether the
    metric function in this module is the one `pi_eval.score` ran, independently of whether
    the published CONTRAST reproduces. Counted on PRESENCE as well as value -- a row this
    module emits where the scorer emitted none is a fabricated population.
    """
    fn = ladder.METRIC_FNS[metric]
    agree = differ = only_here = only_stored = 0
    for rid, lad in arms.items():
        graph = graphs.get(lad.task_id)
        if graph is None:
            continue
        mine = fn(lad.at(lad.n_asks), graph)
        theirs = stored.get((rid, metric))
        if math.isnan(mine) and theirs is None:
            continue
        if math.isnan(mine):
            only_stored += 1
        elif theirs is None:
            only_here += 1
        elif abs(mine - theirs) <= tol:
            agree += 1
        else:
            differ += 1
    return {
        "agree": agree,
        "differ": differ,
        "emitted_here_absent_in_store": only_here,
        "in_store_absent_here": only_stored,
    }


def _stored_values(parquet_dir: Path, run_ids: Sequence[str]) -> dict[tuple[str, str], float]:
    con = duckdb.connect()
    ids = list(run_ids)
    ph = ",".join("?" for _ in ids)
    df = con.execute(
        f"SELECT run_id, metric_name, value "
        f"FROM read_parquet('{(Path(parquet_dir) / 'scores.parquet').as_posix()}') "
        f"WHERE run_id IN ({ph})",
        ids,
    ).fetchdf()
    return {(r.run_id, r.metric_name): float(r.value) for r in df.itertuples(index=False)}


def _digest(ids: Sequence[str]) -> str:
    """Run-id provenance for a cell, at the granularity a cell HAS: which set of runs it was
    computed over. A list of 400 hex ids in a JSON row is unreadable; a digest of the sorted
    list is checkable."""
    h = hashlib.sha256()
    for rid in sorted(ids):
        h.update(rid.encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------------- main


def relative_to_root(path: Path, root: Path) -> str:
    """A path as a future clone of the repository would spell it.

    The store and the published file live in the SHARED checkout while this sweep runs from a
    worktree, so `Path.relative_to(<worktree>)` raises and the provenance block would record
    an absolute home path -- unreadable for anyone but its author, and the single most common
    way a research artifact stops being reproducible. `--repo-root` names the tree the paths
    belong to; anything genuinely outside it keeps its absolute form rather than being
    silently mangled into a plausible-looking wrong path.
    """
    path, root = Path(path).resolve(), Path(root).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _fmt(v: float | None, places: int = 6) -> str:
    if v is None:
        return "--"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.{places}f}"


def render_table(payload: Mapping[str, Any]) -> str:
    """The RESULT.md table, GENERATED from `locks.json` rather than retyped.

    A hand-typed table beside a machine-readable one is two sources of truth that disagree in
    the last digit, which this repository has already been bitten by once
    (`artifacts/symmetric_matched_cost_20260919/RESULT.md`, its own parenthetical).
    """
    head = (
        "| metric | suite | published | reimplementation | n pub | n reimpl | \\|diff\\| | "
        "verdict | discriminating | why |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for key in sorted(payload["cells"], key=lambda k: (k.split("::")[0], k.split("::")[1])):
        r = payload["cells"][key]
        disc = r["seedmatched_discriminating"]
        why = (r["seedmatched_discriminating_reason"] or "").split(",")[0]
        lines.append(
            f"| `{r['metric']}` | {r['suite']} | {r['published_delta']!r} | "
            f"{r['seedmatched_delta']!r} | {r['published_n']} | {r['seedmatched_n']} | "
            f"{r['seedmatched_abs_diff']:.2e} | **{r['seedmatched_verdict']}** | "
            f"{'yes' if disc else 'NO'} | {why} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", type=Path, default=repo / "artifacts" / "testsplit_qa")
    p.add_argument(
        "--published",
        type=Path,
        default=repo / "artifacts" / "testsplit_plan_metrics_20260918" / "contrasts.json",
    )
    p.add_argument(
        "--out-dir", type=Path, default=repo / "artifacts" / "plan_metric_locks_20260919"
    )
    p.add_argument("--suites", nargs="+", default=["musique", "strategyqa", "wiki2"])
    p.add_argument("--graph-version", default="v1")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-terminal-check", action="store_true")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=repo,
        help="the tree the store and published file belong to; provenance paths are recorded "
        "relative to it so the artifact carries no absolute home path",
    )
    args = p.parse_args(argv)

    from pi_eval.gold import load_graphs

    store = args.cohort_dir / "scores_parquet"
    published = published_matched_cells(json.loads(args.published.read_text()))
    metrics = sorted({m for m, _ in published})

    con = duckdb.connect()
    scorer_hashes = sorted(
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT scorer_hash FROM read_parquet('{(store / 'scores.parquet').as_posix()}')"
        ).fetchall()
    )
    graph_versions = sorted(
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT graph_version FROM read_parquet('{(store / 'matches.parquet').as_posix()}')"
        ).fetchall()
    )

    rows: list[LockRow] = []
    prov: dict[str, Any] = {}
    terminal: dict[str, dict[str, int]] = {}

    diagnostics: dict[str, Any] = {}

    with registered_metrics({**EXTRA_METRIC_FNS, **DIAGNOSTIC_METRIC_FNS}):
        for suite in args.suites:
            graphs = load_graphs(suite, args.graph_version)
            ck_ids = _read_ids(args.cohort_dir / f"run_ids.trained.{suite}.txt")
            ba_ids = _read_ids(args.cohort_dir / f"run_ids.prompted.{suite}.txt")
            seeds = {
                r[0]: int(r[1])
                for r in con.execute(
                    f"SELECT run_id, seed FROM read_parquet("
                    f"'{(store / 'runs.parquet').as_posix()}')"
                ).fetchall()
            }
            rec_ck = ladder.load_run_ladders(store, ck_ids)
            rec_ba = ladder.load_run_ladders(store, ba_ids)
            cov_ck = load_coverage_ladders(store, ck_ids, graphs)
            cov_ba = load_coverage_ladders(store, ba_ids, graphs)
            prov[suite] = {
                "n_trained_run_ids": len(ck_ids),
                "n_prompted_run_ids": len(ba_ids),
                "trained_run_id_digest_sha256": _digest(ck_ids),
                "prompted_run_id_digest_sha256": _digest(ba_ids),
                "n_gold_graphs": len(graphs),
            }
            if not args.skip_terminal_check:
                stored = _stored_values(store, list(ck_ids) + list(ba_ids))
                for metric in metrics:
                    ck, ba = (
                        (cov_ck, cov_ba) if metric in COVERAGE_LADDER_METRICS else (rec_ck, rec_ba)
                    )
                    terminal[f"{metric}::{suite}"] = terminal_check(
                        metric, {**ck, **ba}, graphs, stored
                    )
            for metric in metrics:
                cell = published.get((metric, suite))
                ck, ba = (cov_ck, cov_ba) if metric in COVERAGE_LADDER_METRICS else (rec_ck, rec_ba)
                row = lock_cell(
                    metric,
                    ck,
                    ba,
                    graphs,
                    published_delta=cell.published_delta if cell else None,
                    published_n=cell.published_n if cell else None,
                    published_lo=cell.published_lo if cell else None,
                    published_hi=cell.published_hi if cell else None,
                    seed=args.seed,
                    n_boot=args.n_boot,
                    seeds=seeds,
                    published_n_pairs=cell.n_pairs if cell else None,
                    published_pairs_dropped=cell.pairs_dropped if cell else None,
                )
                rows.append(row)
                print(
                    f"{metric:26s} {suite:11s} pub={_fmt(row.published_delta)} "
                    f"repro={_fmt(row.repro_delta)} n={row.published_n}/{row.repro_n} "
                    f"|d|={_fmt(row.abs_diff, 3) if row.abs_diff is None else f'{row.abs_diff:.3e}'} "
                    f"{row.verdict} | documented-pairing {row.seedmatched_verdict} "
                    f"({_fmt(row.seedmatched_delta)}, pairs "
                    f"{row.seedmatched_n_pairs}/{row.published_n_pairs}) "
                    f"discriminating={row.discriminating}",
                    flush=True,
                )

            # WHICH FUNCTION PRODUCED THE PUBLISHED `facet_breadth`? `ladder`'s is a ratio and
            # NaN on a facet-free graph; `pi_eval.score`'s is a count and 0.0 there. Both are
            # measured against the SAME published float rather than argued about.
            fb = published.get(("facet_breadth", suite))
            if fb is not None:
                d = seed_matched_contrast(
                    "facet_breadth_scorer",
                    rec_ck,
                    rec_ba,
                    graphs,
                    seeds,
                    seed=args.seed,
                    n_boot=args.n_boot,
                )
                diagnostics[f"facet_breadth_scorer::{suite}"] = {
                    "published_facet_breadth_delta": fb.published_delta,
                    "scorer_definition_delta": d.delta,
                    "abs_diff": abs(d.delta - fb.published_delta),
                    "n": d.n,
                    "published_n": fb.published_n,
                    "n_pairs": d.n_pairs,
                    "published_n_pairs": fb.n_pairs,
                    "pairs_dropped": d.pairs_dropped,
                    "published_pairs_dropped": fb.pairs_dropped,
                    "verdict": ("LOCKED" if abs(d.delta - fb.published_delta) <= TOL else "FAILED"),
                    "note": "pi_eval.score emits float(touched), a COUNT, and emits 0.0 on a "
                    "graph with no facets; ladder.METRIC_FNS['facet_breadth'] is "
                    "touched/total and NaN there.",
                }
                print(f"  diagnostic {diagnostics[f'facet_breadth_scorer::{suite}']}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "store": relative_to_root(store, args.repo_root),
            "published_source": relative_to_root(args.published, args.repo_root),
            "repo_root_note": "paths above are relative to --repo-root",
            "scorer_hash": scorer_hashes,
            "graph_version": graph_versions,
            "ladder_module": "scripts/plan_metrics_symmetric/ladder.py",
            "seed": args.seed,
            "n_boot": args.n_boot,
            "tolerance": TOL,
            "per_suite": prov,
        },
        "terminal_check": terminal,
        "diagnostics": diagnostics,
        "cells": {f"{r.metric}::{r.suite}::matched": asdict(r) for r in rows},
    }
    (args.out_dir / "locks.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "table.md").write_text(render_table(payload) + "\n")
    print(f"\nwrote {args.out_dir / 'locks.json'}")
    n_locked = sum(1 for r in rows if r.verdict == "LOCKED")
    n_disc = sum(1 for r in rows if r.verdict == "LOCKED" and r.discriminating)
    n_sm = sum(1 for r in rows if r.seedmatched_verdict == "LOCKED")
    n_sm_disc = sum(1 for r in rows if r.seedmatched_verdict == "LOCKED" and r.discriminating)
    print(
        f"ladder pairing:     {n_locked} LOCKED of {len(rows)} cells, "
        f"{n_disc} of those discriminating"
    )
    print(
        f"documented pairing: {n_sm} LOCKED of {len(rows)} cells, "
        f"{n_sm_disc} of those discriminating"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
