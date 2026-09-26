"""AGGREGATION: parquet -> paper objects. Every number that leaves here carries provenance.

Four rules are compiled into this module rather than left to discipline.

1. ELIGIBILITY IS A PREDICATE, NOT A HABIT. `ELIGIBLE` is one SQL string, it is inlined
   verbatim into the provenance record, and `assert_no_gold_exposed` re-checks the emitted
   run_id set against the runs table afterwards. Two independent mechanisms, because the
   failure being guarded against — an oracle arm or a dirty-tree run reaching a reported
   number — is silent, plausible and unrecoverable after publication.

2. THE MULTIPLICITY STORY IS STRUCTURAL. `PREREG_PRIMARY` is uncorrected (one preregistered
   contrast per suite). `SECONDARY` is ONE DECLARED FAMILY through `benjamini_hochberg`.
   Everything else is emitted as a point estimate with a CI, labelled `exploratory`, and
   carries NO p-value at all — not a corrected one, not an uncorrected one. A p-value that
   exists is a p-value that gets quoted.

3. `below_noise_floor` IS A COLUMN AND A BANNER, NEVER AN EXCEPTION. An effect under
   `2*sigma_J/sqrt(n)` is not reportable, but a linter that throws at 2am is a linter that
   gets disabled by 2:05am. It prints, loudly, in every rendering, forever.

4. A NUMBER WITHOUT PROVENANCE IS NOT A RESULT. `provenance()` names the run_id set, the
   scorer_hash, the graph_version, the code versions, the content hash of every input file
   and the exact query. `pi render` refuses to emit a table whose provenance no longer
   matches, which is what makes a hand-edited number impossible to keep.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pi_eval import schema as sch
from pi_eval.matcher.base import MechanicalMatcher
from pi_eval.metrics.frontier import Curve, auc_log2, frontier, log2_grid, step_at_spend
from pi_eval.prereg import (
    CI_RESAMPLES,
    KILL_SWITCH_MARGINS,
    MCNEMAR_EXACT,
    SIGN_FLIP_PERMUTATION,
    Endpoint,
    default_stage1,
)
from pi_eval.prereg import (
    KILL_SWITCHES as _KILL_SWITCHES,
)
from pi_eval.prereg import (
    POOLED as _POOLED,
)
from pi_eval.prereg import (
    SYNTH_COMPARATOR as _SYNTH_COMPARATOR,
)
from pi_eval.prereg import (
    SYNTH_TREATMENT as _SYNTH_TREATMENT,
)
from pi_eval.prereg import (
    load_sigma_j as _load_sigma_j,
)
from pi_eval.score import ARTIFACT_METRICS, define, family_of
from pi_eval.stats.inference import (
    Estimate,
    benjamini_hochberg,
    cluster_bootstrap,
    mcnemar_exact,
    noise_floor,
    paired_difference,
)
from pinq.budget import PARITY_CURRENCIES
from pinq.ids import canon, h

NAN = float("nan")
# Re-exported, never redefined: the pooling sentinel is declared beside the endpoints it
# applies to. SYNTH_TREATMENT / SYNTH_COMPARATOR come from prereg for the same reason and are
# re-exported here because callers (and tests) have always read them off this module.
POOLED = _POOLED
SYNTH_TREATMENT = _SYNTH_TREATMENT
SYNTH_COMPARATOR = _SYNTH_COMPARATOR
NOT_APPLICABLE = "--"  # how a NaN is TYPESET; the stored value stays NaN

PARITY_TOLERANCE = 0.02  # +-2% on token and retrieval currencies, at ARM level
PARITY_MIN_TASKS = 200  # below this the parity assertion is not licensed, only descriptive
FRONTIER_GRID_POINTS = 5

# The eligibility predicate. Inlined into every provenance record verbatim so a reviewer can
# paste it into duckdb and get the same rows.
MATCHER_ID = MechanicalMatcher.matcher_id

ELIGIBLE = (
    "r.status = 'ok' AND r.gold_exposed = FALSE AND r.is_dev_run = FALSE "
    "AND r.dirty = FALSE AND r.pilot_flag = FALSE AND r.canary_hit = FALSE "
    # A canary, a rehearsal or an oracle probe exists to be LOOKED AT before the analysis
    # is fixed. 60 of the tier0 canary's 66 runs passed this predicate before it was added.
    "AND r.exploratory = FALSE "
    "AND r.firewall_ok = TRUE AND r.counterfactual_kind = 'none' "
    # THE HELD-OUT SPLIT, as a predicate rather than a per-grid declaration (decided
    # 2026-09-15, plan v4 SS8.6). Nothing here filtered on split, so `pi suites eligibility`
    # exited 1 from the day it was written: harmless while every arm was prompted -- a policy
    # that never trained cannot have trained on these tasks -- and a silent comparison on the
    # trained arm's own training data the moment one is not. `tier1_trained` declares
    # `split: test` while the confirmatory grids declare none, so the two were not even
    # measuring the same population. The alternative was `split: test` in each grid: narrower,
    # and one forgotten line away from recurring in the next grid anyone writes. A grid can
    # forget this clause; the clause cannot be forgotten by a grid.
    #
    # DEV IS NOT HELD OUT EITHER. `pi train gate` reads the dev rollouts to decide which
    # checkpoint to promote, so a dev number is one the selection already optimised -- the
    # same reason `train_split_violations` demands `test` on both the stamped and the
    # recomputed side rather than merely refusing `train`.
    #
    # MEASURED THE DAY IT LANDED, over scores/parquet/runs.parquet: 34,155 train and 1,976 dev
    # rows left the eligible set and 1,537 test rows remain. Every table rendered before this
    # was computed over a different population and has to be re-rendered.
    "AND r.split = 'test' "
    # RECONCILIATION IS A GATE, NOT A COLUMN. `docs_ok` is the only detector for an unmetered
    # nested retrieval -- a Drafter that retrieves inside resolve() and folds the result into
    # its returned Evidence, which is invisible in tokens, invisible in retrieval_calls and
    # fatal to a budget-parity claim. `tokens_ok` is the same check for spend.
    #
    # Both were computed, written to status.json and compacted to these columns, and then read
    # by NOTHING: `worker.run_unit` raises on a token mismatch only for `llm_free` arms, i.e.
    # only where the ledger is empty and the check was already trivially true. So a run whose
    # accounting was known to be wrong reached every reported table. A field that records a
    # failure and gates nothing is a field that documents the failure rather than preventing
    # it. NULL is treated as ineligible on purpose: a run written before these columns existed
    # has no reconciliation evidence, and absent evidence is not evidence.
    "AND r.reconciled_tokens = TRUE AND r.reconciled_docs = TRUE"
)
# Debugging only. Anything produced under this must never reach a table; the assert exists
# precisely because someone will eventually pass it and forget.
CONTAMINATED = "r.status = 'ok'"


class AggregationError(RuntimeError):
    """Fatal to the aggregation. Never downgraded."""


class ScalarAucWithoutCurve(AggregationError):
    """A frontier AUC was requested without the curve that defines it.

    Not pedantry: the scalar hides crossings. Two arms with identical AUC can be ordered
    oppositely at every budget a reader cares about, and the only thing that shows it is the
    curve with its simultaneous band.
    """


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Contrast:
    """One preregistered comparison. `suite_id == POOLED` pools tasks across suites.

    NEVER CONSTRUCTED FROM A LITERAL ENDPOINT IN THIS MODULE. Every confirmatory and
    calibration Contrast below is `Contrast.of(endpoint)` over a `pi_eval.prereg.Endpoint`,
    because this module used to hold its own endpoint list and it disagreed with the sealed
    one on three metric names out of four (`tau_reward`/`task_success`,
    `token_f1`/`answer_token_f1`, `kpr_incremental`/`keypoint_recall`) with nothing on either
    side able to notice. EXPLORATORY contrasts, which are by definition not preregistered,
    are still minted here -- that is the one place a Contrast may be built locally, and it
    carries no p-value.
    """

    suite_id: str
    metric: str
    treatment: str
    comparator: str
    test: str = SIGN_FLIP_PERMUTATION
    claim: str = "confirmatory"  # confirmatory | calibration | exploratory
    note: str = ""
    # Carried from the sealed Endpoint. A POOLED contrast whose pool is not filtered here
    # silently includes every suite on disk -- measured: 69 of 252 rows (27.4%) of pooled
    # `evidence_coverage` came from `synth`, the calibration suite.
    pool_suites: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.suite_id}:{self.metric}:{self.treatment}-vs-{self.comparator}"

    @classmethod
    def of(cls, e: Endpoint) -> Contrast:
        return cls(
            e.suite_id,
            e.metric,
            e.treatment,
            e.comparator,
            e.test,
            e.claim,
            e.note,
            e.pool_suites,
        )


_STAGE1 = default_stage1()

# PRIMARY: the calibration endpoint, then one preregistered contrast per suite, alpha=0.05
# each, UNCORRECTED. strategyqa and wiki2 are deliberately absent: binary answers compress
# quality toward a coin flip, so strategyqa carries no primary endpoint (plan section 6) and
# wiki2 is a secondary-grid suite. READ FROM prereg -- see Contrast.of.
PREREG_PRIMARY: tuple[Contrast, ...] = tuple(
    Contrast.of(e) for e in (*_STAGE1.calibration, *_STAGE1.primary)
)

# SECONDARY: ONE declared family, BH-FDR q=0.05 across exactly these and nothing else.
SECONDARY: tuple[Contrast, ...] = tuple(Contrast.of(e) for e in _STAGE1.secondary)
SECONDARY_FAMILY_Q = _STAGE1.fdr_q

# EXPLORATORY: point estimate + task-clustered BCa CI, no p-value, ever.
EXPLORATORY_METRICS: tuple[str, ...] = (
    "rnr_ask",
    "rnr_use",
    "dwr",
    "max_depth_reached",
    "precedence_violation_rate",
    "facet_breadth",
    "adr",
    "phi_loo_mean",
    "phi_null_mean",
    "stop_overshoot",
    "stop_undershoot",
    # THE VERTICAL CLAIM, exploratory by choice. Adding it to SECONDARY would grow the BH
    # family from 17 to 18 and make every declared test strictly harder to reject, for a
    # measurement that has never been run. `classify()` returns "exploratory" for anything
    # outside primary/secondary/calibration, so naming it here is all that is required: it
    # renders in T1c with a CI and NO p-value, and confirmatory_count stays 20.
    "latent_discovery_rate",
    "newly_reachable_share",
    # BREADTH-OVER-COMPONENTS, declared 2026-09-19 (prereg-conformance audit, defect b:
    # `artifacts/prereg_breadth_conformance_20260919/RESULT.md`). `pi_eval.score` has always
    # emitted these three whenever a task's gold graph carries a prerequisite edge, but no
    # family named them, so they had no prereg status even though facet_breadth -- the metric
    # they were built to sit beside -- has been exploratory since T20. Declaring them here does
    # not change a single computed value; it only makes the declaration match what was already
    # measured and reported.
    "breadth_components",
    "breadth_recall",
    "breadth_singleton_share",
)

# DERIVED, never re-typed. This used to be a second hand-written list, and it disagreed with
# the one `default_stage1` seals: the report printed five rows and the preregistration
# committed to four. A report showing a decision the sealed document never made is precisely
# what preregistration exists to prevent, so there is now one list and this is a view of it.
KILLSWITCH_RULES: tuple[tuple[str, str], ...] = tuple(_KILL_SWITCHES.items())
KILLSWITCH_TREATMENT = "inquirer_prompted"
KILLSWITCH_METRIC = "evidence_coverage"


# --------------------------------------------------------------------------- table object


def _is_split(v: float | None, tol: float = 1e-9) -> bool:
    """A seed-average sitting exactly between success and failure."""
    return v is not None and abs(float(v) - 0.5) < tol


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            # NOT "nan", and NOT "0.00". This string is typeset into the paper, and the word
            # "nan" in a results cell is a reader's problem: it is either read as a bug or as
            # a value. An em-dash is the convention for "not applicable", and the distinction
            # it preserves is the one this codebase keeps fighting for -- `usd_per_need` is
            # NaN for an arm that resolved no needs, which is ABSENT rather than zero.
            #
            # The JSON keeps the NaN. Only the human-facing rendering changes, so nothing
            # downstream is told a different story from the reader.
            return NOT_APPLICABLE
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if abs(v) >= 1e6 or (v != 0 and abs(v) < 1e-4):
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


@dataclass(frozen=True, slots=True)
class Table:
    """A paper object: columns, rows, notes, warnings and its own provenance.

    `digest()` is over columns and rows only. It is what `pi render` compares against, so a
    table whose numbers moved cannot be silently re-emitted under the old provenance.
    """

    table_id: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return h(
            "table",
            self.table_id,
            canon(list(self.columns)),
            canon([[_fmt(r.get(c)) for c in self.columns] for r in self.rows]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [dict(r) for r in self.rows],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "digest": self.digest(),
        }

    def to_text(self) -> str:
        cells = [[_fmt(r.get(c)) for c in self.columns] for r in self.rows]
        widths = [
            max(len(c), *(len(row[i]) for row in cells)) if cells else len(c)
            for i, c in enumerate(self.columns)
        ]
        sep = "  "
        out = [f"{self.table_id}  {self.title}", "-" * (sum(widths) + 2 * len(widths))]
        out.append(sep.join(c.ljust(w) for c, w in zip(self.columns, widths)))
        out.append(sep.join("-" * w for w in widths))
        out += [sep.join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
        if not cells:
            out.append("(no eligible rows)")
        for w in self.warnings:
            out.append(f"!! {w}")
        for n in self.notes:
            out.append(f"   {n}")
        return "\n".join(out)

    def write(self, tables_dir: str | Path) -> Path:
        """`tables/<id>/{table.json,provenance.json}`. Provenance is BESIDE the table, always."""
        d = Path(tables_dir) / self.table_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "table.json").write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        (d / "provenance.json").write_text(
            json.dumps(dict(self.provenance), indent=2, sort_keys=True) + "\n"
        )
        return d


# --------------------------------------------------------------------------- provenance


def file_hashes(parquet_dir: str | Path) -> dict[str, str]:
    d = Path(parquet_dir)
    out: dict[str, str] = {}
    for name in sorted(sch.TABLES):
        p = d / f"{name}.parquet"
        out[f"{name}.parquet"] = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "ABSENT"
        )
    return out


def inputs_hash(parquet_dir: str | Path) -> str:
    return h("inputs", canon(sorted(file_hashes(parquet_dir).items())))


def provenance(
    *,
    table_id: str,
    parquet_dir: str | Path,
    run_ids: Sequence[str],
    scorer_hash: str,
    graph_version: str,
    code_versions: Sequence[str],
    query: str,
    tool: str,
    resampling: Mapping[str, int] | None = None,
    eligibility_predicate: str | None = None,
    prereg_ok: bool | None = None,
    prereg_note: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    files = file_hashes(parquet_dir)
    body: dict[str, Any] = {
        "table_id": table_id,
        "parquet_dir": str(parquet_dir),
        "run_ids": sorted(set(run_ids)),
        "n_run_ids": len(set(run_ids)),
        "scorer_hash": scorer_hash,
        "graph_version": graph_version,
        "code_version": sorted({c for c in code_versions if c}),
        "schema_hash": sch.schema_hash(),
        # THE PREDICATE THAT ACTUALLY RAN, not the module constant. This recorded `ELIGIBLE`
        # unconditionally, so a `--allow-contaminated` aggregation -- which runs CONTAMINATED --
        # produced a provenance record naming a filter it had not applied, and the exclusion
        # clause the sealed prereg adds would have been invisible too. Rule 1 is that a number
        # traces to what produced it; a provenance that names the wrong filter is worse than
        # one that names none, because it is checkable and passes.
        "eligibility_predicate": eligibility_predicate or ELIGIBLE,
        # Did every sealed prereg digest still match when this table was made? None means there
        # is no prereg directory. False must never be quietly published.
        "prereg_verified": prereg_ok,
        "prereg_exclusions": prereg_note,
        # THE RESAMPLING PARAMETERS. Every CI here is a bootstrap and every p a permutation,
        # so `seed`, `n_boot` and `n_perm` are part of how the number was produced -- and rule
        # 1 says a reported value traces to what produced it. Without them the same rows and
        # the same scorer_hash yield a different interval under `--seed 1`, and the artifact
        # cannot say which one it is.
        "resampling": dict(resampling or {}),
        "inputs": files,
        "inputs_hash": h("inputs", canon(sorted(files.items()))),
        "query": query,
        "tool": tool,
    }
    if extra:
        body["extra"] = dict(extra)
    # The digest deliberately EXCLUDES the timestamp: two identical aggregations an hour
    # apart must compare equal, or `pi render` would refuse on the clock alone.
    body["provenance_digest"] = h("prov", canon(body))
    body["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return body


# --------------------------------------------------------------------------- the connection


def _q(s: Any) -> str:
    return "'" + str(s).replace("'", "''") + "'"


@dataclass(frozen=True, slots=True)
class Agg:
    """One open aggregation context. Holds the duckdb connection and the pinned hashes."""

    parquet_dir: Path
    con: Any
    scorer_hash: str
    graph_version: str
    allow_contaminated: bool
    sigma_j: Mapping[str, float]
    # READ FROM THE PREREGISTRATION, not typed here. This was 10,000 while the drafted
    # claim_rule sealed 1,000 resamples, so the document and every rendered interval named
    # different numbers and nothing compared them. `n_perm` keeps its own default: the sealed
    # sentence is about the BCa interval, and a permutation count is a different quantity.
    n_boot: int = CI_RESAMPLES
    n_perm: int = 10_000
    seed: int = 0
    # (suite_id, task_id) the SEALED stage 1 excludes, plus a human note. Empty when nothing is
    # sealed yet, which is the current state of this repository.
    # The matcher whose rows may be pooled with these scores. matches.parquet is append-only
    # and carries several matcher_ids once the matcher has ever been bumped; matcher_id rides
    # into matcher_hash -> scorer_hash, so exactly one of them corresponds to the pinned
    # scorer_hash. Taken from the running code, which is the same code that produced the
    # scores -- and if it is not, the provenance drift gate is what catches that.
    matcher_id: str = MATCHER_ID
    excluded_pairs: frozenset[tuple[str, str]] = frozenset()
    # RUN ids the sealed stage 1 drops, keyed on run_id and nothing else. A duplicate run from
    # a sweep restart shares (suite, task, arm, seed) with the run of record, so putting it in
    # `excluded_pairs` would delete the measurement along with the duplicate.
    excluded_run_ids: frozenset[str] = frozenset()
    prereg_note: str = "no sealed stage 1; exclusions not applied"
    # Result of recomputing every sealed digest, or None when there is no prereg directory.
    prereg_ok: bool | None = None

    @property
    def predicate(self) -> str:
        """ELIGIBLE, plus whatever the SEALED preregistration excludes.

        The sealed document had no reader: `prereg.excluded()` had zero callers and nothing in
        src/ opened stage1.json, so the exclusion list was written, digested, frozen -- and had
        no effect on any analysis. `excluded_task_ids` was still enforced upstream (the builder
        drops a contaminated task from the corpus, verified 0 of 24 present), but
        `burned_pilot_ids` was not enforced anywhere: ELIGIBLE drops the pilot RUN via
        `pilot_flag = FALSE`, while the rule is about the TASK -- a tau chosen by looking at
        task T contaminates T's confirmatory run in every arm.

        Inlined into the predicate rather than filtered in Python because `predicate` is
        recorded verbatim in every provenance record, so the exclusion is visible to a reader
        of the table rather than buried in a helper.
        """
        base = CONTAMINATED if self.allow_contaminated else ELIGIBLE
        if self.excluded_pairs:
            pairs = ", ".join(f"({_q(su)}, {_q(t)})" for su, t in sorted(self.excluded_pairs))
            base = f"{base} AND (r.suite_id, r.task_id) NOT IN ({pairs})"
        if self.excluded_run_ids:
            ids = ", ".join(_q(r) for r in sorted(self.excluded_run_ids))
            base = f"{base} AND r.run_id NOT IN ({ids})"
        return base

    def sql(self, query: str) -> list[dict[str, Any]]:
        cur = self.con.execute(query)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def code_versions(self) -> list[str]:
        """The code versions that produced the numbers IN THIS AGGREGATION.

        This selected every distinct code_version in runs.parquet with no predicate, so
        provenance named commits whose runs were all excluded -- dev- runs, gold-exposed runs,
        pilot runs, unreconciled runs. Measured on this repository: 9 versions reported, 4 of
        them contributing an eligible run, 5 naming code that produced none of the numbers in
        the table they were attached to.

        Rule 1 is that a reported value traces to what produced it. A provenance listing code
        that produced nothing is a false trace that PASSES inspection, which is worse than an
        absent one.
        """
        return [
            str(r["code_version"])
            for r in self.sql(
                f"SELECT DISTINCT code_version FROM runs r WHERE ({self.predicate}) ORDER BY 1"
            )
        ]

    def prov(self, table_id: str, run_ids: Iterable[str], query: str, tool: str, **extra) -> dict:
        return provenance(
            table_id=table_id,
            parquet_dir=self.parquet_dir,
            run_ids=list(run_ids),
            scorer_hash=self.scorer_hash,
            graph_version=self.graph_version,
            code_versions=self.code_versions(),
            query=query,
            tool=tool,
            eligibility_predicate=self.predicate,
            prereg_ok=self.prereg_ok,
            prereg_note=self.prereg_note,
            resampling={"seed": self.seed, "n_boot": self.n_boot, "n_perm": self.n_perm},
            extra=extra or None,
        )


# sigma_J is a SEALED PREREGISTERED CONSTANT, so its reader lives with the preregistration and
# is re-exported here rather than reimplemented. A second copy of this loader is a second place
# for "which file did the floor come from?" to have two answers.
load_sigma_j = _load_sigma_j


def open_agg(
    parquet_dir: str | Path = "scores/parquet",
    *,
    scorer_hash: str | None = None,
    allow_contaminated: bool = False,
    n_boot: int = CI_RESAMPLES,
    n_perm: int = 10_000,
    seed: int = 0,
) -> Agg:
    from pinq.extras import require

    duckdb = require("duckdb", why="the kill-switch report queries parquet in place")

    d = Path(parquet_dir)
    missing = [n for n in sch.TABLES if not (d / f"{n}.parquet").exists()]
    if missing:
        raise AggregationError(f"missing parquet in {d}: {missing}. Run `pi compact` first.")
    con = duckdb.connect()
    for name in sch.TABLES:
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet({_q(d / f'{name}.parquet')})"
        )
    hashes = [
        str(r[0])
        for r in con.execute("SELECT DISTINCT scorer_hash FROM scores ORDER BY 1").fetchall()
    ]
    if scorer_hash is None:
        if not hashes:
            raise AggregationError(
                f"scores.parquet in {d} is empty: run `pi score` before aggregating."
            )
        if len(hashes) > 1:
            raise AggregationError(
                "several scorer_hashes present; a number must name exactly one. "
                f"Pass --scorer-hash. Present: {hashes}"
            )
        scorer_hash = hashes[0]
    elif scorer_hash not in hashes:
        raise AggregationError(f"scorer_hash {scorer_hash!r} not in scores.parquet: {hashes}")

    gv = [
        str(r[0])
        for r in con.execute(
            f"SELECT DISTINCT graph_version FROM scores WHERE scorer_hash = {_q(scorer_hash)}"
            " AND graph_version <> '' ORDER BY 1"
        ).fetchall()
    ]
    # THE SEAL IS CHECKED WHERE THE NUMBERS ARE MADE. No report path called `verify()`, so a
    # silently edited preregistration -- the one failure `verify`'s docstring names as
    # invalidating the whole multiplicity defence -- could not be noticed by anyone reading a
    # table. Recomputed here, recorded in provenance, and surfaced as a banner by the renderer.
    _root = _prereg_root(d)
    _prereg_ok: bool | None = None
    _ex_pairs: frozenset[tuple[str, str]] = frozenset()
    _ex_runs: frozenset[str] = frozenset()
    _ex_note = "no sealed stage 1; exclusions not applied"
    if _root is not None:
        from .prereg import excluded_pairs as _exp
        from .prereg import excluded_runs as _exr
        from .prereg import verify as _verify

        _res = _verify(_root)
        # UNSEALED READS AS "no preregistration", not as "a digest changed". `verify` returns
        # ok=False for both; see `seal_warnings_for`.
        _prereg_ok = None if (_res.checked == 0 and not _res.mismatches) else _res.ok
        _ex_pairs, _ex_note = _exp(_root)
        _ex_runs, _ex_run_note = _exr(_root)
        if _ex_runs:
            _ex_note = f"{_ex_note}; {_ex_run_note}"

    return Agg(
        parquet_dir=d,
        con=con,
        scorer_hash=scorer_hash,
        graph_version=_one_graph_version(gv, scorer_hash),
        allow_contaminated=allow_contaminated,
        sigma_j=load_sigma_j(d),
        excluded_pairs=_ex_pairs,
        excluded_run_ids=_ex_runs,
        prereg_note=_ex_note,
        prereg_ok=_prereg_ok,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )


def _one_graph_version(versions: Sequence[str], scorer_hash: str) -> str:
    """Exactly one, or refuse.

    `open_agg` raises on two scorer_hashes -- "a number must name exactly one" -- and then
    comma-joined two graph_versions into the string `"v1,v2"` and pooled their rows into one
    cell. A graph_version identifies WHICH GOLD a coverage number was measured against; two of
    them in one mean is two different denominators averaged together, and the provenance field
    that should have caught it instead recorded the pair as though it were an identity.

    The same refusal as for scorer_hash, for the same reason. `scorer_hash` folds in
    `graph_hash`, so this should be unreachable -- which is exactly why silently coping with it
    was wrong.
    """
    if len(versions) <= 1:
        return versions[0] if versions else ""
    raise AggregationError(
        f"several graph_versions present under scorer_hash {scorer_hash[:12]}: "
        f"{list(versions)}. A coverage number names the gold it was measured against, and "
        "pooling two of them averages two different denominators. Re-score, or select rows "
        "from one graph_version."
    )


def _prereg_root(start: Path) -> Path | None:
    """The `prereg/` directory at or above `start`. Same search as `load_sigma_j`, so the
    sigma_J the report uses and the seal it checks always come from one directory."""
    p = Path(start).resolve()
    for cand in (p, *p.parents):
        d = cand / "prereg"
        if d.is_dir():
            return d
    return None


# --------------------------------------------------------------------------- pulls


def metric_query(
    agg: Agg,
    metric: str,
    *,
    suite: str | None = None,
    pool_suites: Sequence[str] = (),
) -> str:
    # THE DECLARED DENOMINATOR, in the SQL. Without the pool filter a POOLED query means
    # "every suite that happens to be on disk", a claim that changes when a suite is added.
    where = [
        f"s.metric_name = {_q(metric)}",
        f"s.scorer_hash = {_q(agg.scorer_hash)}",
        *_suite_where(suite, pool_suites),
    ]
    return (
        "SELECT r.suite_id AS suite_id, r.arm_id AS arm_id, r.task_id AS task_id,\n"
        "       COALESCE(NULLIF(r.template_id, ''), r.task_id) AS cluster_id,\n"
        "       avg(s.value) AS value, count(*) AS n_rows,\n"
        "       list(DISTINCT r.run_id) AS run_ids\n"
        "FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        "WHERE " + " AND ".join(where) + f"\n  AND ({agg.predicate})\n"
        "GROUP BY 1, 2, 3, 4\nORDER BY 1, 2, 3"
    )


@dataclass(frozen=True, slots=True)
class Pull:
    """A metric, per (suite, arm, task). `by_arm` is what every paired test consumes."""

    metric: str
    query: str
    by_arm: Mapping[str, Mapping[str, float]]  # arm -> task_key -> value
    clusters: Mapping[str, str]  # task_key -> cluster_key
    run_ids: tuple[str, ...]
    suites: tuple[str, ...]

    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_arm))


def pull(
    agg: Agg,
    metric: str,
    *,
    suite: str | None = None,
    tasks: int | None = None,
    pool_suites: Sequence[str] = (),
) -> Pull:
    """Derived metrics are routed here so callers never special-case them."""
    if metric == "cad_ge2":
        return _pull_cad_ge2(agg, suite=suite, tasks=tasks, pool_suites=pool_suites)
    if metric == "frontier_auc":
        return _pull_frontier_auc(agg, suite=suite, tasks=tasks, pool_suites=pool_suites)
    q = metric_query(agg, metric, suite=suite, pool_suites=pool_suites)
    return _pull_rows(agg.sql(q), metric, q, tasks)


def _pull_rows(rows: Sequence[Mapping[str, Any]], metric: str, q: str, tasks: int | None) -> Pull:
    by_arm: dict[str, dict[str, float]] = {}
    clusters: dict[str, str] = {}
    run_ids: set[str] = set()
    suites: set[str] = set()
    keep = _task_allowlist(rows, tasks)
    for r in rows:
        key = f"{r['suite_id']}/{r['task_id']}"
        if keep is not None and key not in keep:
            continue
        by_arm.setdefault(str(r["arm_id"]), {})[key] = float(r["value"])
        clusters[key] = f"{r['suite_id']}/{r['cluster_id']}"
        run_ids |= {str(x) for x in (r.get("run_ids") or ())}
        suites.add(str(r["suite_id"]))
    return Pull(metric, q, by_arm, clusters, tuple(sorted(run_ids)), tuple(sorted(suites)))


def _task_allowlist(rows: Sequence[Mapping[str, Any]], tasks: int | None) -> set[str] | None:
    """`--n N` truncates to the first N task keys IN SORTED ORDER, never in arrival order:
    a kill-switch decision must not depend on how duckdb happened to scan the file."""
    if tasks is None:
        return None
    keys = sorted({f"{r['suite_id']}/{r['task_id']}" for r in rows})
    return set(keys[:tasks])


def _suite_where(suite: str | None, pool_suites: Sequence[str]) -> list[str]:
    """The suite predicate, shared by every pull. A POOLED pull without its declared pool
    silently means "every suite on disk" -- which is how `synth`, the calibration suite,
    reached 27.4% of pooled `evidence_coverage` rows."""
    if suite and suite != POOLED:
        return [f"r.suite_id = {_q(suite)}"]
    if pool_suites:
        return [f"r.suite_id IN ({', '.join(_q(x) for x in sorted(pool_suites))})"]
    return []


def _pull_cad_ge2(
    agg: Agg, *, suite: str | None, tasks: int | None, pool_suites: Sequence[str] = ()
) -> Pull:
    """C@d>=2, |V_d|-weighted. Weighting by cardinality is not a nicety: an unweighted mean
    over depths lets a depth holding two nodes outvote one holding forty."""
    where = [f"s.scorer_hash = {_q(agg.scorer_hash)}", *_suite_where(suite, pool_suites)]
    q = (
        "WITH d AS (\n"
        "  SELECT r.suite_id, r.arm_id, r.task_id,\n"
        "         COALESCE(NULLIF(r.template_id, ''), r.task_id) AS cluster_id, r.run_id,\n"
        "         split_part(s.metric_name, '#', 1) AS fam,\n"
        "         CAST(split_part(s.metric_name, '#', 2) AS INTEGER) AS depth, s.value\n"
        "  FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        "  WHERE " + " AND ".join(where) + f" AND ({agg.predicate})\n"
        "    AND split_part(s.metric_name, '#', 1) IN ('cad', 'cad_n')\n"
        ")\n"
        "SELECT suite_id, arm_id, task_id, cluster_id,\n"
        "       sum(c.value * n.value) / nullif(sum(n.value), 0) AS value,\n"
        "       count(*) AS n_rows, list(DISTINCT c.run_id) AS run_ids\n"
        "FROM d c JOIN d n USING (suite_id, arm_id, task_id, cluster_id, run_id, depth)\n"
        "WHERE c.fam = 'cad' AND n.fam = 'cad_n' AND c.depth >= 2\n"
        "GROUP BY 1, 2, 3, 4\nORDER BY 1, 2, 3"
    )
    return _pull_rows(agg.sql(q), "cad_ge2", q, tasks)


def _pull_frontier_auc(
    agg: Agg, *, suite: str | None, tasks: int | None, pool_suites: Sequence[str] = ()
) -> Pull:
    """AUC over the log2-spend axis, per (arm, task). Always accompanied by the curve: see
    `frontier_curves` and `ScalarAucWithoutCurve`."""
    ladders = frontier_ladders(agg, suite=suite, pool_suites=pool_suites)
    grid = _shared_grid(ladders)
    rows: list[dict[str, Any]] = []
    for (suite_id, arm, task_id, cluster_id, run_ids), (spends, qs) in sorted(ladders.items()):
        vals = [step_at_spend(spends, qs, x) for x in grid]
        rows.append(
            {
                "suite_id": suite_id,
                "arm_id": arm,
                "task_id": task_id,
                "cluster_id": cluster_id,
                "value": auc_log2(grid, vals),
                "n_rows": len(vals),
                "run_ids": list(run_ids),
            }
        )
    rows = [r for r in rows if not math.isnan(float(r["value"]))]
    return _pull_rows(rows, "frontier_auc", frontier_query(agg, suite), tasks)


def frontier_query(agg: Agg, suite: str | None, pool_suites: Sequence[str] = ()) -> str:
    where = [f"s.scorer_hash = {_q(agg.scorer_hash)}", *_suite_where(suite, pool_suites)]
    return (
        "SELECT r.suite_id, r.arm_id, r.task_id,\n"
        "       COALESCE(NULLIF(r.template_id, ''), r.task_id) AS cluster_id, r.run_id,\n"
        "       split_part(s.metric_name, '#', 1) AS fam,\n"
        "       CAST(split_part(s.metric_name, '#', 2) AS INTEGER) AS k, s.value\n"
        "FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        "WHERE " + " AND ".join(where) + f" AND ({agg.predicate})\n"
        "  AND split_part(s.metric_name, '#', 1) IN ('frontier_spend', 'frontier_q')\n"
        "ORDER BY 1, 2, 3, 7"
    )


def frontier_ladders(
    agg: Agg, *, suite: str | None = None, pool_suites: Sequence[str] = ()
) -> dict[tuple[str, str, str, str, tuple[str, ...]], tuple[list[float], list[float]]]:
    """(suite, arm, task, cluster, run_ids) -> (cumulative spend, Q at that spend).

    The x-axis is CUMULATIVE SPEND, never turns: `drafter_only` asks zero questions and has
    no turn index at all, so a paired Delta-AUC against it would be paired against nothing.
    """
    rows = agg.sql(frontier_query(agg, suite, pool_suites))
    acc: dict[tuple, dict[int, dict[str, float]]] = {}
    for r in rows:
        key = (
            str(r["suite_id"]),
            str(r["arm_id"]),
            str(r["task_id"]),
            str(r["cluster_id"]),
            str(r["run_id"]),
        )
        acc.setdefault(key, {}).setdefault(int(r["k"]), {})[str(r["fam"])] = float(r["value"])
    # average over seeds, keeping the run_id set for provenance
    merged: dict[tuple, dict[str, Any]] = {}
    for (suite_id, arm, task, cluster, run_id), ks in acc.items():
        m = merged.setdefault((suite_id, arm, task, cluster), {"runs": set(), "ks": {}})
        m["runs"].add(run_id)
        for k, fams in ks.items():
            slot = m["ks"].setdefault(k, {"frontier_spend": [], "frontier_q": []})
            for fam in ("frontier_spend", "frontier_q"):
                if fam in fams:
                    slot[fam].append(fams[fam])
    out: dict[tuple, tuple[list[float], list[float]]] = {}
    for (suite_id, arm, task, cluster), m in merged.items():
        spends, qs = [], []
        for k in sorted(m["ks"]):
            slot = m["ks"][k]
            if not slot["frontier_spend"] or not slot["frontier_q"]:
                continue
            spends.append(statistics.fmean(slot["frontier_spend"]))
            qs.append(statistics.fmean(slot["frontier_q"]))
        if spends:
            out[(suite_id, arm, task, cluster, tuple(sorted(m["runs"])))] = (spends, qs)
    return out


def _shared_grid(ladders: Mapping[tuple, tuple[Sequence[float], Sequence[float]]]) -> tuple:
    hi = max((max(s) for s, _ in ladders.values() if s), default=1.0)
    return log2_grid(1.0, max(hi, 2.0), FRONTIER_GRID_POINTS)


def frontier_curves(
    agg: Agg, *, suite: str | None = None, n_boot: int = 2000
) -> tuple[dict[str, Curve], tuple[float, ...]]:
    """arm -> Curve, on one shared grid. The ONLY supported way to obtain an AUC."""
    ladders = frontier_ladders(agg, suite=suite)
    grid = _shared_grid(ladders)
    per_arm: dict[str, dict[str, tuple[Sequence[float], Sequence[float]]]] = {}
    # THE CLUSTER MAP THE LADDER ALREADY CARRIES. `frontier_ladders` keys every entry by
    # cluster_id and this discarded it as `_cluster`, so the band in the F1 legend resampled
    # TASKS while the AUC point estimate reached through `frontier_auc` resampled CLUSTERS --
    # two intervals for one quantity, computed over different units. Measured on tau2's real
    # template sizes, nominal 95% AUC coverage: 52.0% at ICC=0.5 and 42.0% at ICC=0.8, against
    # 90.8% and 91.2% once clusters are the unit. Suffixed with the suite for the same reason
    # the task key is: two suites can mint the same template_id.
    clusters: dict[str, str] = {}
    for (suite_id, arm, task, cluster, _runs), lad in ladders.items():
        key = f"{suite_id}/{task}"
        per_arm.setdefault(arm, {})[key] = lad
        clusters[key] = f"{suite_id}/{cluster or task}"
    return (
        {
            arm: frontier(tasks, grid, n_boot=n_boot, seed=agg.seed, clusters=clusters)
            for arm, tasks in sorted(per_arm.items())
        },
        grid,
    )


def auc_row(curve: Curve) -> dict[str, float]:
    """The scalar, and ONLY together with the curve that produced it.

    `frontier` returns both; anything that extracts `curve.auc` from a curve that does not
    exist gets this exception instead of a number.
    """
    finite = [v for v in curve.mean if not math.isnan(v)]
    if len(finite) < 2 or len(curve.grid) < 2:
        raise ScalarAucWithoutCurve(
            "refusing to emit a frontier AUC without a curve: the scalar hides crossings, "
            f"and this curve has {len(finite)} finite point(s) on a grid of {len(curve.grid)}."
        )
    return {
        "auc": curve.auc,
        "auc_lo": curve.auc_lo,
        "auc_hi": curve.auc_hi,
        "n_tasks": float(curve.n_tasks),
        "n_grid": float(len(curve.grid)),
    }


# --------------------------------------------------------------------------- estimation


def _units(a: Mapping[str, float], b: Mapping[str, float], clusters) -> list[list[float]]:
    groups: dict[str, list[float]] = {}
    for k in sorted(set(a) & set(b)):
        if math.isnan(a[k]) or math.isnan(b[k]):
            continue
        groups.setdefault(clusters.get(k, k), []).append(a[k] - b[k])
    return list(groups.values())


def estimate(
    agg: Agg,
    p: Pull,
    treatment: str,
    comparator: str,
    *,
    binary: bool,
    with_p: bool,
) -> Estimate:
    a, b = p.by_arm.get(treatment, {}), p.by_arm.get(comparator, {})
    if not a or not b or not (set(a) & set(b)):
        return Estimate(NAN, NAN, NAN, None, 0, "unavailable", "no paired overlap")
    if binary and with_p:
        # A seed-averaged binary outcome is not binary. int(round(0.5)) == 0 in Python, so a
        # task an arm won on exactly one of two seeds was silently scored a LOSS -- on the
        # tau2 primary endpoint, whose reward IS binary. Such a task carries no consistent
        # signal about direction, so it is EXCLUDED and counted, never coerced.
        ties = sorted(k for k in (set(a) & set(b)) if _is_split(a.get(k)) or _is_split(b.get(k)))
        keep = (set(a) & set(b)) - set(ties)
        est = mcnemar_exact(
            {k: int(round(a[k])) for k in keep},
            {k: int(round(b[k])) for k in keep},
            # The SAME cluster map the paired CI uses. tau2's primary is preregistered
            # `cluster_by="template_id"`, and this call passed no map at all -- so the
            # preregistered clustering had no effect on the test it governs.
            clusters=p.clusters,
            n_perm=agg.n_perm,
            seed=agg.seed,
        )
        if ties:
            note = (
                f"{est.note}; excluded {len(ties)} task(s) split across seeds "
                f"(no consistent direction; coercing them would have scored each a loss)"
            )
            est = replace(est, note=note)
        return est
    if with_p:
        return paired_difference(
            a,
            b,
            clusters=p.clusters,
            n_boot=agg.n_boot,
            n_perm=agg.n_perm,
            seed=agg.seed,
        )
    units = _units(a, b, p.clusters)
    if not units:
        return Estimate(NAN, NAN, NAN, None, 0, "unavailable", "no paired overlap")
    point, lo, hi = cluster_bootstrap(units, n_boot=agg.n_boot, seed=agg.seed)
    n = sum(len(u) for u in units)
    return Estimate(point, lo, hi, None, n, "paired-BCa (exploratory)", f"{len(units)} clusters")


def level_estimate(agg: Agg, values: Mapping[str, float], clusters) -> Estimate:
    groups: dict[str, list[float]] = {}
    for k, v in values.items():
        if not math.isnan(v):
            groups.setdefault(clusters.get(k, k), []).append(v)
    if not groups:
        return Estimate(NAN, NAN, NAN, None, 0, "unavailable", "")
    point, lo, hi = cluster_bootstrap(list(groups.values()), n_boot=agg.n_boot, seed=agg.seed)
    n = sum(len(v) for v in groups.values())
    return Estimate(point, lo, hi, None, n, "level-BCa", f"{len(groups)} clusters")


def floor_flag(agg: Agg, metric: str, est: Estimate) -> tuple[float, bool, str]:
    """(noise_floor, below_noise_floor, note). NEVER raises. See rule 3 in the docstring."""
    d = define(metric)
    if d is None or not d.judge_derived:
        return 0.0, False, ""
    sigma = agg.sigma_j.get(family_of(metric))
    if sigma is None:
        return (
            float("inf"),
            True,
            "sigma_J not yet measured for this judge-derived metric; not reportable",
        )
    fl = noise_floor(sigma, est.n)
    below = (not math.isnan(est.point)) and abs(est.point) < fl
    return fl, below, f"sigma_J={sigma:g}" if below else ""


def _contrast_row(agg: Agg, c: Contrast, p: Pull, *, with_p: bool) -> dict[str, Any]:
    d = define(c.metric)
    binary = bool(d and d.binary) and c.test == MCNEMAR_EXACT
    est = estimate(agg, p, c.treatment, c.comparator, binary=binary, with_p=with_p)
    fl, below, note = floor_flag(agg, c.metric, est)
    return {
        "suite": "pooled" if c.suite_id == POOLED else c.suite_id,
        "metric": c.metric,
        "treatment": c.treatment,
        "comparator": c.comparator,
        "n": est.n,
        "delta": est.point,
        "ci_lo": est.ci_lo,
        "ci_hi": est.ci_hi,
        "p": est.p_value if with_p else None,
        "method": est.method,
        "noise_floor": fl,
        "below_noise_floor": below,
        "claim": c.claim,
        "notes": "; ".join(x for x in (c.note, est.note, note) if x),
        "_run_ids": p.run_ids,
    }


def _strip(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)


def _run_ids_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out: set[str] = set()
    for r in rows:
        out |= set(r.get("_run_ids", ()))
    return sorted(out)


def _floor_warnings(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    flagged = [f"{r['suite']}/{r['metric']}" for r in rows if r.get("below_noise_floor")]
    if not flagged:
        return ()
    return (
        "BELOW NOISE FLOOR (|effect| < 2*sigma_J/sqrt(n)) and therefore NOT REPORTABLE "
        "as an effect: " + ", ".join(flagged),
    )


# --------------------------------------------------------------------------- the tables


def primary_table(agg: Agg, *, tasks: int | None = None) -> Table:
    rows = []
    skipped = []
    for c in PREREG_PRIMARY:
        p = pull(agg, c.metric, suite=c.suite_id, tasks=tasks, pool_suites=c.pool_suites)
        row = _contrast_row(agg, c, p, with_p=True)
        if row["n"] == 0:
            skipped.append(f"{c.key} (no eligible paired rows)")
            continue
        rows.append(row)
    query = "\n\n".join(
        f"-- {c.key}\n{metric_query(agg, c.metric, suite=c.suite_id, pool_suites=c.pool_suites)}"
        for c in PREREG_PRIMARY
    )
    notes = [
        "PRIMARY: one preregistered contrast per suite, alpha=0.05 each, UNCORRECTED.",
        "Paired on shared tasks with common random numbers; CI is a task-clustered BCa "
        "bootstrap resampling CLUSTERS (template_id where a suite has one).",
        "p is an exact McNemar for binary suites and a sign-flip permutation otherwise.",
    ]
    if skipped:
        notes.append("Not emitted (no data): " + "; ".join(skipped))
    return Table(
        table_id="T1_primary",
        title="Primary endpoints (preregistered, uncorrected)",
        columns=(
            "suite",
            "metric",
            "treatment",
            "comparator",
            "n",
            "delta",
            "ci_lo",
            "ci_hi",
            "p",
            "method",
            "below_noise_floor",
            "claim",
        ),
        rows=_strip(rows),
        notes=tuple(notes),
        warnings=_floor_warnings(rows) + _seal_warnings(agg),
        provenance=agg.prov("T1_primary", _run_ids_of(rows), query, "pi agg --primary"),
    )


def secondary_table(agg: Agg, *, tasks: int | None = None) -> Table:
    """The DECLARED family, and only it, goes through BH-FDR."""
    rows = []
    for c in SECONDARY:
        p = pull(agg, c.metric, suite=c.suite_id, tasks=tasks, pool_suites=c.pool_suites)
        row = _contrast_row(agg, c, p, with_p=True)
        row["family_key"] = c.key
        rows.append(row)
    live = {r["family_key"]: r["p"] for r in rows if r["p"] is not None and r["n"] > 0}
    passed = benjamini_hochberg(live, SECONDARY_FAMILY_Q)
    for r in rows:
        r["bh_reject"] = bool(passed.get(r["family_key"], False))
        r["bh_family_size"] = len(live)
    query = "\n\n".join(
        f"-- {c.key}\n{metric_query(agg, c.metric, suite=c.suite_id, pool_suites=c.pool_suites)}"
        for c in SECONDARY
        if c.metric not in ("cad_ge2", "frontier_auc")
    )
    return Table(
        table_id="T1b_secondary",
        title=f"Secondary family (declared, BH-FDR q={SECONDARY_FAMILY_Q})",
        columns=(
            "suite",
            "metric",
            "treatment",
            "comparator",
            "n",
            "delta",
            "ci_lo",
            "ci_hi",
            "p",
            "bh_reject",
            "bh_family_size",
            "method",
            "below_noise_floor",
        ),
        rows=_strip(rows),
        notes=(
            f"ONE declared family of {len(SECONDARY)} tests. BH-FDR is applied across "
            "exactly these and nothing else.",
            "A contrast with n=0 has no p and is excluded from the family denominator "
            "rather than being counted as a test that was 'run'.",
        ),
        warnings=_floor_warnings(rows),
        provenance=agg.prov("T1b_secondary", _run_ids_of(rows), query, "pi agg --secondary"),
    )


def exploratory_table(agg: Agg, *, tasks: int | None = None) -> Table:
    """Point estimate + task-clustered BCa CI. NO p-value, corrected or otherwise."""
    rows = []
    treatment = KILLSWITCH_TREATMENT
    for metric in EXPLORATORY_METRICS:
        p = pull(agg, metric, tasks=tasks)
        for arm in p.arms():
            if arm == treatment:
                continue
            c = Contrast(POOLED, metric, treatment, arm, SIGN_FLIP_PERMUTATION, "exploratory")
            row = _contrast_row(agg, c, p, with_p=False)
            if row["n"] == 0:
                continue
            row["label"] = "exploratory"
            rows.append(row)
    query = "\n\n".join(f"-- {m}\n{metric_query(agg, m)}" for m in EXPLORATORY_METRICS)
    return Table(
        table_id="T1c_exploratory",
        title="Exploratory contrasts (no inferential claim, no p-values)",
        columns=(
            "suite",
            "metric",
            "treatment",
            "comparator",
            "n",
            "delta",
            "ci_lo",
            "ci_hi",
            "label",
            "method",
        ),
        rows=_strip(rows),
        notes=(
            "NO INFERENTIAL CLAIM. These are neither primary nor in the declared secondary "
            "family, so they carry a CI and no p-value at all.",
            "BH-FDR is NOT applied here: correcting an undeclared set would imply it was a "
            "family, which is exactly the claim being declined.",
        ),
        provenance=agg.prov("T1c_exploratory", _run_ids_of(rows), query, "pi agg --exploratory"),
    )


def arm_ladder_table(agg: Agg, *, tasks: int | None = None) -> Table:
    """T2: every arm's level on its suite's preregistered metric, plus the delta to the
    preregistered comparator. The ladder, not a winner."""
    rows = []
    queries = []
    for c in PREREG_PRIMARY:
        p = pull(agg, c.metric, suite=c.suite_id, tasks=tasks, pool_suites=c.pool_suites)
        if not p.by_arm:
            continue
        queries.append(f"-- {c.suite_id}\n{p.query}")
        for arm in p.arms():
            lvl = level_estimate(agg, p.by_arm[arm], p.clusters)
            delta = estimate(agg, p, arm, c.comparator, binary=False, with_p=False)
            rows.append(
                {
                    "suite": c.suite_id,
                    "metric": c.metric,
                    "arm": arm,
                    "n_tasks": lvl.n,
                    "mean": lvl.point,
                    "ci_lo": lvl.ci_lo,
                    "ci_hi": lvl.ci_hi,
                    "vs_comparator": c.comparator,
                    "delta": delta.point,
                    "delta_ci_lo": delta.ci_lo,
                    "delta_ci_hi": delta.ci_hi,
                    "_run_ids": p.run_ids,
                }
            )
    return Table(
        table_id="T2_arm_ladder",
        title="Arm ladder, per suite, on the preregistered metric",
        columns=(
            "suite",
            "metric",
            "arm",
            "n_tasks",
            "mean",
            "ci_lo",
            "ci_hi",
            "vs_comparator",
            "delta",
            "delta_ci_lo",
            "delta_ci_hi",
        ),
        rows=_strip(rows),
        notes=(
            "Levels carry a task-clustered BCa CI; deltas are paired against the "
            "preregistered comparator with NO p-value (the inferential claim lives in T1).",
        ),
        provenance=agg.prov(
            "T2_arm_ladder", _run_ids_of(rows), "\n\n".join(queries), "pi agg --arms"
        ),
    )


def cad_query(agg: Agg) -> str:
    return (
        "SELECT r.suite_id, r.arm_id,\n"
        "       CAST(split_part(s.metric_name, '#', 2) AS INTEGER) AS depth,\n"
        "       split_part(s.metric_name, '#', 1) AS fam,\n"
        "       avg(s.value) AS value, count(DISTINCT r.task_id) AS n_tasks,\n"
        "       list(DISTINCT r.run_id) AS run_ids\n"
        "FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        f"WHERE s.scorer_hash = {_q(agg.scorer_hash)} AND ({agg.predicate})\n"
        "  AND split_part(s.metric_name, '#', 1) IN ('cad', 'cad_n')\n"
        "GROUP BY 1, 2, 3, 4\nORDER BY 1, 2, 3, 4"
    )


def cad_table(agg: Agg) -> Table:
    """T3: depth-stratified coverage. |V_d| IS PRINTED IN EVERY CELL.

    Coverage at a depth holding two nodes is not comparable to coverage at a depth holding
    forty, so the cardinality travels with the number and is never a separate table someone
    forgets to look at.
    """
    q = cad_query(agg)
    rows_raw = agg.sql(q)
    acc: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    ntasks: dict[tuple[str, str], int] = {}
    run_ids: set[str] = set()
    depths: set[int] = set()
    for r in rows_raw:
        key = (str(r["suite_id"]), str(r["arm_id"]))
        acc.setdefault(key, {}).setdefault(int(r["depth"]), {})[str(r["fam"])] = float(r["value"])
        ntasks[key] = max(ntasks.get(key, 0), int(r["n_tasks"]))
        run_ids |= {str(x) for x in (r.get("run_ids") or ())}
        depths.add(int(r["depth"]))
    cols = ("suite", "arm", "n_tasks") + tuple(f"C@{d} (|V_{d}|)" for d in sorted(depths))
    rows = []
    for (suite_id, arm), per_depth in sorted(acc.items()):
        row: dict[str, Any] = {"suite": suite_id, "arm": arm, "n_tasks": ntasks[(suite_id, arm)]}
        for d in sorted(depths):
            cell = per_depth.get(d)
            if cell is None or "cad" not in cell or "cad_n" not in cell:
                row[f"C@{d} (|V_{d}|)"] = "-"
            else:
                row[f"C@{d} (|V_{d}|)"] = f"{cell['cad']:.3f} ({cell['cad_n']:.0f})"
        rows.append(row)
    return Table(
        table_id="T3_coverage_at_depth",
        title="C@d by arm, with |V_d| printed in every cell",
        columns=cols,
        rows=tuple(rows),
        notes=(
            "Every cell is 'recall (|V_d|)'. Anticipation Depth is deliberately absent: it "
            "is non-monotone (finding only v4 scores 2.0 and beats finding {v1..v4} at 1.25).",
            "|V_d| is the mean per-task cardinality of depth d over the eligible runs.",
            # THE DENOMINATOR'S QUALIFIER, which reached no table at all.
            #
            # `coverage_at_depth` buckets only nodes with a `gold_depth`, so a node the graph
            # cannot reach is in no C@d column, no DWR term and no |V_d|. `orphan_rate` is
            # emitted per run into scores.parquet -- 264 rows -- and `grep -n orphan report.py
            # render.py` matched NOTHING, so the share of gold this table is silent about was
            # computed, stored, and shown to nobody.
            #
            # It is not small. tau2's graph is 63.9% orphaned (3,637 of 5,790 nodes
            # unreachable), and tau2 carries the primary endpoint -- so every C@d number on it
            # describes about a third of its gold.
            *(
                (
                    "Depth-orphaned gold is EXCLUDED from every cell: a node with no reachable "
                    "depth is in no C@d column, no DWR term and no |V_d|. Mean orphan_rate over "
                    "the eligible runs, per suite: " + _orphan_note(agg) + ".",
                )
                if _orphan_note(agg)
                else ()
            ),
        ),
        provenance=agg.prov("T3_coverage_at_depth", sorted(run_ids), q, "pi agg --cad"),
    )


def seal_warnings_for(res: Any) -> tuple[str, ...]:
    """The warning for a `prereg.verify` result. `None` means there is no prereg root at all.

    UNSEALED IS NOT BROKEN. `verify` returns ok=False both when MANIFEST.json is missing and
    when a digest changed, and those are opposite findings: one says "seal stage 1", the
    other says "something was edited after sealing, stop". Collapsing them printed the
    alarming message for the harmless state from the moment a prereg/ directory existed at
    all -- which happened here when sigma_j.json was frozen into one. MEASURED then:
    checked=0, mismatches=[], missing=["MANIFEST.json"]. Nothing had ever been sealed, so no
    digest could have changed and there was nothing to explain.
    """
    if res is None or (getattr(res, "checked", 0) == 0 and not getattr(res, "mismatches", ())):
        return (
            "NOTHING IS SEALED. This table says 'preregistered' and no sealed preregistration "
            "exists on disk (`make prereg-verify` exits 1). The contrasts below are declared "
            "in pi_eval.prereg but were not frozen before the data was seen. Seal stage 1 "
            "before this table is read as confirmatory.",
        )
    if not getattr(res, "ok", False):
        return (
            "THE SEALED PREREGISTRATION NO LONGER VERIFIES: a stage file's digest does not "
            "match its manifest. Every claim of preregistration in this table is void until "
            "that is explained.",
        )
    return ()


def _seal_warnings(agg: Agg) -> tuple[str, ...]:
    """THE WORD "PREREGISTERED" IS A CLAIM, AND IT PRINTED EITHER WAY.

    T1 captions itself "Primary endpoints (preregistered, uncorrected)" and its first note says
    "one preregistered contrast per suite" -- while `make prereg-verify` exits 1 today, because
    there is no prereg/ directory at all. The provenance record says so honestly
    (`prereg_verified: null`) and nothing in the rendered artifact did.

    A WARNING and not an exception, for the reason rule 3 of this module's docstring gives about
    below_noise_floor: "a linter that throws at 2am is a linter that gets disabled by 2:05am. It
    prints, loudly, in every rendering, forever."
    """
    if agg.prereg_ok is None:
        return seal_warnings_for(None)
    if agg.prereg_ok is False:
        return seal_warnings_for(_Broken())
    return ()


class _Broken:
    """A verified-but-mismatched result, for the one branch `prereg_ok` still collapses."""

    ok = False
    checked = 1
    mismatches = ("<a stage file>",)


def _orphan_note(agg: Agg) -> str:
    """ "suite=0.639, suite=0.012" over the eligible runs, or "" when nothing recorded it."""
    rows = agg.sql(
        "SELECT r.suite_id AS suite_id, avg(s.value) AS v\n"
        "FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        f"WHERE s.metric_name = 'orphan_rate' AND s.scorer_hash = {_q(agg.scorer_hash)}\n"
        f"  AND ({agg.predicate})\n"
        "GROUP BY 1 ORDER BY 1"
    )
    return ", ".join(f"{r['suite_id']}={float(r['v']):.3f}" for r in rows if r["v"] is not None)


def ops_query(agg: Agg) -> str:
    return (
        "WITH m AS (\n"
        "  SELECT r.suite_id, r.arm_id, s.metric_name, r.run_id, s.value\n"
        "  FROM scores s JOIN runs r ON r.run_id = s.run_id\n"
        f"  WHERE s.scorer_hash = {_q(agg.scorer_hash)} AND ({agg.predicate})\n"
        "), c AS (\n"
        "  SELECT r.suite_id, r.arm_id, avg(CAST(c.cache_hit AS DOUBLE)) AS cache_hit_rate\n"
        "  FROM calls c JOIN runs r ON r.run_id = c.run_id\n"
        f"  WHERE ({agg.predicate})\n"
        "  GROUP BY 1, 2\n"
        # matches.parquet IS APPEND-ONLY, LIKE scores.parquet. This counted every row in it
        # with no matcher_id and no graph_version filter, while the `m` CTE directly above
        # pins s.scorer_hash -- so a re-score under a new matcher DOUBLED needs_discovered and
        # HALVED usd_per_need*, in a published table, with no error anywhere.
        #
        # Measured on this repository's live matches.parquet after the matcher was bumped from
        # mechanical_v1 to mechanical_v2: 249 rows under each id, and the CTE counted all 498.
        "), needs AS (\n"
        "  SELECT r.suite_id, r.arm_id, count(*) AS n_resolved\n"
        "  FROM matches mm JOIN runs r ON r.run_id = mm.run_id\n"
        f"  WHERE ({agg.predicate}) AND mm.match_kind IN ('resolve', 'use')\n"
        f"    AND mm.matcher_id = {_q(agg.matcher_id)}\n"
        f"    AND mm.graph_version = {_q(agg.graph_version)}\n"
        "  GROUP BY 1, 2\n"
        ")\n"
        "SELECT m.suite_id, m.arm_id, m.metric_name, avg(m.value) AS mean_value,\n"
        "       count(DISTINCT m.run_id) AS n_runs,\n"
        "       any_value(c.cache_hit_rate) AS cache_hit_rate,\n"
        "       any_value(needs.n_resolved) AS n_resolved,\n"
        "       list(DISTINCT m.run_id) AS run_ids\n"
        "FROM m LEFT JOIN c USING (suite_id, arm_id) LEFT JOIN needs USING (suite_id, arm_id)\n"
        "GROUP BY 1, 2, 3\nORDER BY 1, 2, 3"
    )


OPS_METRICS = (
    "tok_prompt",
    "tok_completion",
    "tok_cached",
    "tok_total",
    "usd",
    "retrieval_calls",
    "unique_docs",
    "wall_ms",
)


def ops_table(agg: Agg) -> Table:
    """T4: budget and operations. `wall_ms` and `usd` are MACHINE ARTIFACTS."""
    q = ops_query(agg)
    raw = agg.sql(q)
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    run_ids: set[str] = set()
    for r in raw:
        key = (str(r["suite_id"]), str(r["arm_id"]))
        slot = acc.setdefault(
            key, {"cache_hit_rate": r["cache_hit_rate"], "n_resolved": 0, "n_runs_by_metric": {}}
        )
        slot["n_resolved"] = int(r["n_resolved"] or 0)
        metric = str(r["metric_name"])
        slot[metric] = float(r["mean_value"])
        # PER METRIC. `n_runs` is `count(DISTINCT run_id)` within (suite, arm, metric), and
        # metrics are NOT emitted by the same runs -- tau_reward lands on 88 of tau2's 97
        # tasks, a judged metric only where there was an answer. Keeping one `n_runs` per cell
        # meant whichever metric the loop happened to see LAST supplied the multiplier for
        # `usd_total`, so the reported total spend was the usd mean times an unrelated count.
        slot["n_runs_by_metric"][metric] = int(r["n_runs"])
        slot["n_runs"] = max(slot.get("n_runs", 0), int(r["n_runs"]))
        run_ids |= {str(x) for x in (r.get("run_ids") or ())}
    rows = []
    for (suite_id, arm), slot in sorted(acc.items()):
        n_usd_runs = int(slot["n_runs_by_metric"].get("usd", 0))
        usd_total = float(slot.get("usd", 0.0)) * n_usd_runs
        n_res = int(slot.get("n_resolved", 0))
        row: dict[str, Any] = {"suite": suite_id, "arm": arm, "n_runs": slot.get("n_runs", 0)}
        for m in OPS_METRICS:
            label = f"{m}*" if m in ARTIFACT_METRICS else m
            row[label] = slot.get(m, NAN)
        chr_ = slot.get("cache_hit_rate")
        row["cache_hit_rate"] = NAN if chr_ is None else float(chr_)
        # PER RUN, like every other column in this table -- its title is "per-run means".
        # `n_resolved` is a COUNT over match rows across every run in the cell, so an arm that
        # happened to contribute more runs showed a bigger number for identical behaviour, and
        # a reader comparing the column across arms was comparing run counts. The ratio below
        # is unaffected: it divides a total by a total, and the normalisation cancels.
        n_runs = int(slot.get("n_runs", 0))
        row["needs_per_run"] = (n_res / n_runs) if n_runs else NAN
        row["usd_per_need*"] = (usd_total / n_res) if n_res else NAN
        rows.append(row)
    cols = (
        ("suite", "arm", "n_runs")
        + tuple(f"{m}*" if m in ARTIFACT_METRICS else m for m in OPS_METRICS)
        + ("cache_hit_rate", "needs_per_run", "usd_per_need*")
    )
    return Table(
        table_id="T4_budget_ops",
        title="Budget and operations (per-run means)",
        columns=cols,
        rows=tuple(rows),
        notes=(
            "* = MACHINE ARTIFACT. `wall_ms` and `usd` (and anything derived from usd) are "
            "recorded and NEVER compared across arms as evidence: an arm with more turns "
            "takes more wall time, so asserting parity there fails on day one and teaches "
            "you to disable the check.",
            "Token and retrieval currencies are the ones under the parity assertion (T5).",
        ),
        provenance=agg.prov("T4_budget_ops", sorted(run_ids), q, "pi agg --ops"),
    )


def _parity_ratio(mean: float, ref_mean: float) -> float:
    """Two arms that both spend nothing are AT parity, not undefined.

    Reporting nan there prints a red 'no' next to every zero-token arm on an LLM-free grid,
    and a check that cries wolf on day one is a check nobody reads on day thirty.
    """
    if math.isnan(mean) or math.isnan(ref_mean):
        return NAN
    if ref_mean == 0.0:
        return 1.0 if mean == 0.0 else float("inf")
    return mean / ref_mean


def parity_table(agg: Agg, *, reference: str = KILLSWITCH_TREATMENT) -> Table:
    """T5: cross-arm parity, +-2% on token and retrieval currencies, at ARM level.

    `wall_ms` and `usd` are excluded BY DESIGN, not by oversight.
    """
    rows = []
    queries = []
    run_ids: set[str] = set()
    fallback = ""
    for currency in PARITY_CURRENCIES:
        p = pull(agg, currency)
        if not p.by_arm:
            continue
        if reference not in p.by_arm:
            # A reference arm with no rollouts would make every ratio nan and every cell
            # read as a breach. Fall back deterministically and SAY SO in the column.
            fallback = reference
            reference = p.arms()[0]
        queries.append(f"-- {currency}\n{p.query}")
        run_ids |= set(p.run_ids)
        ref = p.by_arm.get(reference)
        ref_mean = statistics.fmean([v for v in ref.values() if not math.isnan(v)]) if ref else NAN
        for arm in p.arms():
            vals = [v for v in p.by_arm[arm].values() if not math.isnan(v)]
            mean = statistics.fmean(vals) if vals else NAN
            ratio = _parity_ratio(mean, ref_mean)
            within = (not math.isnan(ratio)) and abs(ratio - 1.0) <= PARITY_TOLERANCE
            rows.append(
                {
                    "currency": currency,
                    "arm": arm,
                    "n_tasks": len(vals),
                    "mean": mean,
                    "reference": reference,
                    "ratio_to_reference": ratio,
                    "within_2pct": within,
                    "licensed": len(vals) >= PARITY_MIN_TASKS,
                    "spends": bool(mean) or bool(ref_mean and not math.isnan(mean)),
                }
            )
    return Table(
        table_id="T5_parity",
        title=f"Cross-arm budget parity (+-{PARITY_TOLERANCE:.0%}, ARM level)",
        columns=(
            "currency",
            "arm",
            "n_tasks",
            "mean",
            "reference",
            "ratio_to_reference",
            "within_2pct",
            "licensed",
            "spends",
        ),
        rows=tuple(rows),
        notes=(
            "`wall_ms` and `usd` are EXCLUDED by design: they are machine artifacts.",
            f"`licensed` is n_tasks >= {PARITY_MIN_TASKS}; below that the parity assertion "
            "is descriptive only, because a +-2% claim on a handful of tasks is noise.",
            "A zero-question control (drafter_only) sits at ratio 0 BY CONSTRUCTION. Parity "
            "is a claim about arms that spend, so the breach warning is suppressed for a "
            "zero-spend arm rather than firing every run and being learned to ignore.",
        )
        + (
            (
                f"reference arm {fallback!r} has no eligible rollouts; fell back to "
                f"{reference!r}. The ratios below are NOT the preregistered parity claim.",
            )
            if fallback
            else ()
        ),
        warnings=tuple(
            f"PARITY BREACH: {r['currency']} arm {r['arm']} at "
            f"{r['ratio_to_reference']:.3f}x the reference"
            for r in rows
            if r["licensed"] and not r["within_2pct"] and r["spends"]
        ),
        provenance=agg.prov("T5_parity", sorted(run_ids), "\n\n".join(queries), "pi agg --parity"),
    )


def killswitch_table(agg: Agg, *, tasks: int | None = None) -> Table:
    """The week-1 gate. Five paired deltas with BCa CIs and the preregistered consequence.

    This is the command that tells the researcher whether to stop. A comparator that MATCHES
    (the CI covers zero) triggers the written rule; an arm that was never run says so
    instead of contributing a silent absence.
    """
    p = pull(agg, KILLSWITCH_METRIC, tasks=tasks)
    rows = []
    for arm, rule in KILLSWITCH_RULES:
        est = estimate(agg, p, KILLSWITCH_TREATMENT, arm, binary=False, with_p=True)
        # A THREE-WAY VERDICT, BECAUSE "THE CI COVERS ZERO" IS NOT EVIDENCE OF EQUIVALENCE.
        #
        # This read `else: MATCHES -> apply the rule` on any interval containing zero, which is
        # accepting the null: an interval covering zero is equally consistent with "no effect"
        # and with "an effect this pilot cannot resolve". Simulated at the pilot's real n=120
        # through paired_difference, a TRUE +2pt effect produced a zero-covering CI 77% of the
        # time at sd=0.20 and 87% at sd=0.30, and a true +4pt effect 38%/65% -- against
        # consequences up to "STOP; no reframe survives".
        #
        # EQUIVALENT requires the whole interval inside the preregistered margin, which is the
        # standard two-one-sided-tests reading and the only one that can support a stop.
        # INCONCLUSIVE says the pilot was too small to decide and is explicitly NOT a licence
        # to apply the rule.
        margin = float(KILL_SWITCH_MARGINS.get(arm, 0.0))
        # A COMPARATOR THAT CANNOT DIFFER IS NOT EVIDENCE OF EQUIVALENCE.
        #
        # `parallel_replay` replays inquirer_prompted's own recorded questions, so its query
        # strings ARE the treatment's query strings. Proven LLM-free on the synth suite: same
        # evidence uids and the same answer text on every task, because `draft()` and the
        # Answerer are pure in the evidence subset hash. Its delta on evidence_coverage is
        # identically zero before any data exists.
        #
        # Under the plain rule that read "MATCHES -> apply the rule". Under the equivalence
        # margin added a commit ago it reads the STRONGER "EQUIVALENT -> apply the rule": the
        # margin turned a hedged wrong answer into a confident one, which is worse. A verdict
        # of "drop the graph framing" must not be reachable by arithmetic.
        degenerate = est.n > 0 and est.point == 0.0 and est.ci_lo == 0.0 and est.ci_hi == 0.0
        if degenerate:
            verdict = "DEGENERATE -> comparator cannot differ; this switch tests nothing"
        elif est.n == 0:
            verdict = "NOT RUN"
        elif math.isnan(est.ci_lo) or math.isnan(est.ci_hi):
            verdict = "INDETERMINATE"
        elif est.ci_lo > 0:
            verdict = "SEPARATED"
        elif est.ci_hi < 0:
            verdict = "INVERTED -> comparator wins"
        elif margin > 0 and est.ci_lo >= -margin and est.ci_hi <= margin:
            verdict = f"EQUIVALENT (|CI| <= {margin:.2f}) -> apply the rule"
        elif margin > 0:
            verdict = f"INCONCLUSIVE (CI exceeds +/-{margin:.2f}) -> do NOT stop; n too small"
        else:
            verdict = "MATCHES -> apply the rule"
        rows.append(
            {
                "comparator": arm,
                "n": est.n,
                "delta": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "p": est.p_value,
                "verdict": verdict,
                "preregistered_rule": rule,
            }
        )
    return Table(
        table_id="K_killswitch",
        title=(
            # COUNTED, not typed. The literal said "five" while the list held six, and it
            # would have said five with seven -- the same two-declarations bug the rules
            # themselves were rewritten to remove, in the title of the table that reports them.
            f"Kill switch: {KILLSWITCH_TREATMENT} vs {len(rows)} comparators on "
            f"{KILLSWITCH_METRIC} (n={tasks if tasks else 'all'} tasks)"
        ),
        columns=("comparator", "n", "delta", "ci_lo", "ci_hi", "p", "verdict"),
        rows=tuple({k: v for k, v in r.items() if k != "preregistered_rule"} for r in rows),
        notes=tuple(f"{r['comparator']}: {r['preregistered_rule']}" for r in rows)
        + (
            "EQUIVALENT means the whole task-clustered BCa CI lies inside the preregistered "
            "margin for that comparator, which is the only reading that supports a stop. "
            "INCONCLUSIVE means the CI covers zero but extends beyond the margin: the pilot "
            "could not decide, and that is NOT a licence to apply the rule. The rule is "
            "preregistered: where it applies it is applied, not weighed.",
            "Margins (evidence_coverage points): "
            + ", ".join(f"{k}={v:.2f}" for k, v in sorted(KILL_SWITCH_MARGINS.items())),
        ),
        provenance=agg.prov(
            "K_killswitch", p.run_ids, p.query, "pi report killswitch", n_tasks=tasks
        ),
    )


_T6_COLUMNS: tuple[str, ...] = (
    "suite",
    "criterion",
    "n_pairs",
    "position_bias",
    "wins",
    "ties",
    "losses",
    "length_adjusted",
    "usable",
    "why",
)


def instrument_table(agg: Agg, tasks: int | None = None) -> Table:
    """T6: the JUDGE'S OWN diagnostics -- position bias, win/tie/loss, parse failures, alpha.

    `pi_eval.score` has always written `judge_diagnostics.json` and NOTHING READ IT: one
    grep hit in the repo, the write. A judge-derived number whose instrument diagnostics sit
    unrendered on disk is a number nobody can audit, and every judge-derived claim in the
    paper rests on "the judge was usable" -- which is an assumption until this table prints
    it.

    Reads the JSON rather than parquet because the diagnostics are per (suite, criterion)
    while scores.parquet is keyed by run; that is also why score.py writes them beside the
    tables instead of inside them.
    """
    path = Path(agg.parquet_dir) / "judge_diagnostics.json"
    warnings: list[str] = []
    notes: list[str] = []
    rows: list[dict[str, Any]] = []

    if not path.exists():
        # Judging is legitimately skippable (`--allow-no-judge`), so absence is a state to
        # report, not an error. An empty table that says why beats a crash and beats silence.
        warnings.append(
            "NO JUDGE DIAGNOSTICS: judging was not run for this scorer_hash, so every "
            "judge-derived endpoint is unmeasured rather than null."
        )
        return Table(
            table_id="T6_instrument",
            title="Judge instrument diagnostics",
            columns=_T6_COLUMNS,
            rows=(),
            notes=tuple(notes),
            warnings=tuple(warnings),
            provenance=agg.prov("T6_instrument", (), str(path), "pi agg --instrument"),
        )

    payload = json.loads(path.read_text())
    diags = dict(payload.get("diagnostics") or {})
    alpha = diags.pop("krippendorff_alpha_nominal", float("nan"))
    families = list(diags.pop("judge_families", ()) or ())

    for key, d in sorted(diags.items()):
        if not isinstance(d, Mapping):
            continue
        suite_id, _, criterion = str(key).partition("/")
        wins, ties, losses = (
            int(d.get("wins") or 0),
            int(d.get("ties") or 0),
            int(d.get("losses") or 0),
        )
        rows.append(
            {
                "suite": suite_id,
                "criterion": criterion,
                "n_pairs": int(d.get("n_pairs") or 0),
                "position_bias": d.get("position_bias"),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "length_adjusted": d.get("length_adjusted"),
                "usable": bool(d.get("usable")),
                "why": str(d.get("why") or ""),
            }
        )

    n_parse = int(payload.get("n_parse_failures") or 0)
    notes.append(
        f"parse failures: {n_parse}. Counted as a THIRD outcome, never dropped -- accuracy "
        "among parseable answers is not accuracy."
    )
    unjudged = list(payload.get("unjudged") or ())
    notes.append(f"unjudged runs: {len(unjudged)}. A run with no judgment is missing, not a loss.")
    notes.append(
        f"Krippendorff alpha (nominal): {_fmt(alpha)} over judge families {families or '[]'}. "
        "STRUCTURALLY NaN with a single family: alpha measures agreement BETWEEN raters, and "
        "one rater has nobody to agree with. This is a property of the single-model design, "
        "not a computation that failed."
    )

    disq = dict(payload.get("judges_disqualified") or {})
    if disq:
        warnings.append(
            "JUDGE DISQUALIFIED on "
            + ", ".join(f"{k} ({v})" for k, v in sorted(disq.items()))
            + ". Its metric rows are suppressed, so those endpoints are UNMEASURED."
        )
    if not agg.sigma_j:
        warnings.append(
            "sigma_j IS UNMEASURED, so the noise floor is +inf and every judge-derived "
            "effect is NOT REPORTABLE regardless of its interval. This is the conservative "
            "default working, not a missing number to be filled in with a guess: "
            "`scripts/measure_sigma_j.py` requires SIGMA_J_MIN_PAIRS pairs per metric."
        )

    return Table(
        table_id="T6_instrument",
        title="Judge instrument diagnostics",
        columns=_T6_COLUMNS,
        rows=tuple(rows),
        notes=tuple(notes),
        warnings=tuple(warnings),
        provenance=agg.prov("T6_instrument", (), str(path), "pi agg --instrument"),
    )


ALL_TABLES: dict[str, Any] = {
    "T1_primary": primary_table,
    "T1b_secondary": secondary_table,
    "T1c_exploratory": exploratory_table,
    "T2_arm_ladder": arm_ladder_table,
    "T3_coverage_at_depth": cad_table,
    "T4_budget_ops": ops_table,
    "T5_parity": parity_table,
    "T6_instrument": instrument_table,
}


# Public aliases. `pi_run.render` formats cells and quotes SQL literals identically, so a
# rendered table and a printed one can never disagree about what a number looks like.
fmt = _fmt
sql_literal = _q


# --------------------------------------------------------------------------- the assertion


class TrainSplitContamination(RuntimeError):
    """A trained arm was scored on a task it could have trained on. The table is void."""


class TrainIdSetUnresolved(AggregationError):
    """A run names an id set that cannot be found, so the question cannot be answered.

    NOT DOWNGRADED TO A WARNING, and not treated as clean. "I could not check" and "I checked
    and it was fine" are different answers, and a check that returns the second when it means
    the first is the failure this whole layer was built to remove -- a guarantee recorded,
    digested, and consulted by nothing.
    """


# Arms whose model pin is a checkpoint fitted on the TRAIN split. Named explicitly rather than
# inferred, because "does this arm's pin name an adapter" is not a question the manifest can
# answer, and a check whose subject set is derived wrongly finds nothing to find.
TRAINED_ARMS: frozenset[str] = frozenset({"inquirer_trained"})

# Where the exporter leaves the id sets: `data/rl/**/train_ids.<hash16>.txt`, written beside
# the `*.manifest.json` whose `train_id_set_hash` names them.
TRAIN_IDS_DIR_ENV = "PI_TRAIN_IDS_DIR"
_TRAIN_IDS_SUBDIR = ("data", "rl")


def _train_ids_roots(agg: Agg, ids_dir: str | Path | None) -> list[Path]:
    """Where to look for a sidecar, best first.

    AN EXPLICIT `ids_dir` IS EXCLUSIVE, not first-in-line. It is an override -- "the id sets
    are over here" -- and an override that silently falls through to the repo would make a
    caller who pointed at the wrong directory get an answer from a different one.
    """
    if ids_dir is not None:
        return [Path(ids_dir)]
    out: list[Path] = []
    env = os.environ.get(TRAIN_IDS_DIR_ENV)
    if env:
        out.append(Path(env))
    # Beside the parquet being aggregated: `<root>/scores/parquet` -> `<root>/data/rl`. The
    # same inference `_corpora_root` makes, for the same reason -- the run artifacts and the
    # export that produced their checkpoint normally live in one checkout.
    pq = getattr(agg, "parquet_dir", None)
    if pq is not None:
        out.append(Path(pq).resolve().parent.parent.joinpath(*_TRAIN_IDS_SUBDIR))
    # THE NEAREST repo root, and only that one. A worktree sits inside its parent checkout, so
    # walking every ancestor that has a pyproject.toml would let a worktree answer out of the
    # main checkout's exports -- a different repository's id sets, silently.
    for cand in (Path.cwd(), *Path(__file__).resolve().parents):
        if (cand / "pyproject.toml").exists():
            out.append(cand.joinpath(*_TRAIN_IDS_SUBDIR))
            break
    return list(dict.fromkeys(out))


def _train_id_set(stamp: str, agg: Agg, ids_dir: str | Path | None) -> frozenset[str]:
    """The `suite_id/task_id` set a stamped hash names, or a refusal.

    FOUND BY THE HASH, with no index in between: the exporter names the file after the first 16
    hex of the set-hash it holds the preimage of, so a file either IS the set a run names or is
    not there at all. An index would be a third place for the mapping to be wrong.
    """
    name = f"train_ids.{stamp[:16]}.txt"
    roots = _train_ids_roots(agg, ids_dir)
    for root in roots:
        if not root.is_dir():
            continue
        for hit in sorted(root.rglob(name)):
            return frozenset(ln.strip() for ln in hit.read_text().splitlines() if ln.strip())
    raise TrainIdSetUnresolved(
        f"run(s) stamped train_id_set_hash={stamp} but {name} is in none of "
        f"{[str(r) for r in roots]}. That hash names the id set the checkpoint was fitted on, "
        "and without the set a trained arm's rows cannot be checked against it. Point "
        f"--ids-dir (or ${TRAIN_IDS_DIR_ENV}) at the directory holding the export's sidecars, "
        "or re-run `pi train export` to write one. A missing proof is not a clean result."
    )


def train_split_violations(
    agg: Agg, run_ids: Sequence[str], *, ids_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Trained-arm runs sitting on anything but HELD-OUT tasks. Empty means clean.

    THE READER `train_id_set_hash` NEVER HAD. That hash went into every export manifest and
    was carried into runs.parquet and printed by `pi train status`; a repo-wide grep found it
    on the evaluation side in exactly one place -- the schema column definition -- and nothing
    ever put a value in that column, because nothing stamped it onto a run. `ELIGIBLE` never
    mentions `split`. `pinq_train.split` calls the manifest hash layer 3 of five, "so a later
    run can prove which ids a checkpoint saw", and nothing ever asked it to prove anything.
    Same failure the preregistration hit, in the words of `excluded_pairs`' own docstring:
    THE SEALED DOCUMENT HAD NO READER. This function is that reader, in the second half below.

    The protection that exists today is entirely grid-side -- `tier1_trained` sets
    `split: test` and a test asserts it -- which is one mechanism. `assert_no_gold_exposed`
    exists precisely so the analogous guarantee does not rest on one, "because the whole point
    is to survive someone widening that predicate".

    RECOMPUTED, NOT TRUSTED. The run's stamped `split` is compared against `pinq.splitting`
    as well, so a manifest written under a different bucket function is caught rather than
    believed -- the exact drift that once made the runner and the exporter disagree about
    which tasks were trainable.

    HELD OUT MEANS TEST, AND DEV IS NOT HELD OUT. This refused `train` only. That left dev
    open, and dev is precisely where the checkpoint was CHOSEN: `pi train gate` reads the dev
    rollouts to decide which checkpoint to promote, so a dev number for a trained arm is a
    number the selection already optimised. The tasks were not fitted, but the model was
    picked to do well on them, and nothing in a rendered table would say so. So the predicate
    now demands `test` on BOTH sides -- recomputed and stamped -- and anything else, including
    a disagreement between the two, is a violation.

    NOT WIDENED TO EVERY ARM. `inquirer_prompted` is the dev BASELINE and runs those tasks on
    purpose; firing on it would make the check fire on correct behaviour, which is how checks
    get disabled. The subject set stays `TRAINED_ARMS`.

    AND THE SECOND LINE OF DEFENCE: THE ID SET ITSELF (D7, 2026-09-15). Everything above is a
    statement about the split FUNCTION. It catches a task that is train or dev today, and it
    cannot catch a task that was EXPORTED into a training file and has since moved -- a
    re-bucketed suite, a `template_id` introduced after the export was taken, an export taken
    before `EVAL_ONLY_SUITES` grew. Afterwards `split_of` answers "test", and it is RIGHT: the
    wall is where it says it is, and the checkpoint still saw the task. So each row's stamped
    `train_id_set_hash` is resolved to the id set the exporter wrote beside its manifest, and a
    trained-arm row whose `suite_id/task_id` is IN that set is a violation whatever its split
    says. The split logic above is untouched: this is a second predicate beside it, not a
    widening of it, and `reason` says which one fired.

    A HASH WITH NO ID FILE IS A REFUSAL, not a pass. See `TrainIdSetUnresolved`. A NULL hash is
    not: every run made before the stamp existed carries one, as does every run whose Inquirer
    is not a registered checkpoint, and refusing those would refuse the entire corpus.
    """
    if not run_ids:
        return []
    from pinq.splitting import split_of

    ids = ", ".join(_q(r) for r in sorted(set(run_ids)))
    rows = agg.sql(
        "SELECT r.run_id, r.arm_id, r.suite_id, r.task_id, r.template_id, r.split, "
        "r.train_id_set_hash\n"
        f"FROM runs r WHERE r.run_id IN ({ids})\nORDER BY 1"
    )
    out: list[dict[str, Any]] = []
    # Resolved once per distinct hash, and ONLY for the rows the check is about: an untrained
    # arm's stamp says nothing about whether its row is legitimate, so refusing the whole
    # aggregation because one could not be resolved would fire on correct behaviour.
    seen: dict[str, frozenset[str]] = {}
    for r in rows:
        if str(r.get("arm_id")) not in TRAINED_ARMS:
            continue
        suite = str(r.get("suite_id") or "")
        # BOTH ids, in their own positions. Collapsing them to `template_id or task_id` and
        # passing that as the task id is equivalent for a HASHED suite -- `bucket()` hashes
        # `template_id or task_id` itself -- and wrong by construction for a FIXED-SPLIT one,
        # where `split_of` looks the id up in a table keyed by TASK id and a template id in
        # that position raises `UnknownFixedSplitTask` on a row this check was meant to
        # classify. Measured 2026-09-15: tau2_airline is the only suite in `FIXED_SPLITS` and
        # 0 of its 2,233 rows carry a template_id, so no row on disk takes that path today.
        recomputed = split_of(
            suite,
            str(r.get("task_id") or ""),
            template_id=str(r.get("template_id") or "") or None,
        )
        stamped = str(r.get("split") or "")
        reasons: list[str] = []
        if not (recomputed == "test" and stamped == "test"):
            reasons.append("split")
        stamp = str(r.get("train_id_set_hash") or "")
        if stamp:
            if stamp not in seen:
                seen[stamp] = _train_id_set(stamp, agg, ids_dir)
            if f"{suite}/{r.get('task_id')}" in seen[stamp]:
                reasons.append("train_id_set")
        if reasons:
            out.append({**r, "recomputed_split": recomputed, "reason": "+".join(reasons)})
    return out


def assert_no_gold_exposed(agg: Agg, run_ids: Sequence[str]) -> list[dict[str, Any]]:
    """The mechanical guarantee. Returns the offending rows; empty means clean.

    Re-derived from the runs table AFTER the fact rather than trusted from the eligibility
    predicate, because the whole point is to survive someone widening that predicate.
    """
    if not run_ids:
        return []
    ids = ", ".join(_q(r) for r in sorted(set(run_ids)))
    return agg.sql(
        "SELECT r.run_id, r.arm_id, r.suite_id, r.gold_exposed, r.is_dev_run, r.dirty, "
        "r.canary_hit, r.pilot_flag\n"
        f"FROM runs r WHERE r.run_id IN ({ids})\n"
        "  AND (r.gold_exposed = TRUE OR r.is_dev_run = TRUE OR r.run_id LIKE 'dev-%'"
        " OR r.canary_hit = TRUE)\nORDER BY 1"
    )
