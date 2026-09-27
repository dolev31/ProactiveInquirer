"""The online dev gate: does this checkpoint earn a test-split rollout?

WHAT THIS IS FOR (plan I.8/I.9, Tier B). The checkpoint runs as the Inquirer on the dev tasks
the prompted baselines already ran, and this module reads the compacted parquet back and
decides. It is CHECKPOINT SELECTION, not measurement: the test split is not touched until one
checkpoint per arm has been chosen here.

WHY duckdb AND NOT `pi_eval`. Import-linter contract 3 forbids `pinq_train -> pi_eval`, and it
is not bureaucracy: this module runs on the training side of the gold firewall. So the SQL is
written out here against the parquet files directly. Where a definition already exists in
`pi_eval` -- `cad_ge2`'s cardinality weighting is the one that matters -- it is REPLICATED with
the reasoning restated, not approximated, and `test_cad_ge2_equals_the_hand_computed_...` pins
the arithmetic against a value computed by hand rather than against the other implementation.

THE THREE THINGS A READER OF THE VERDICT NEEDS, which the JSON therefore carries:

  * WHICH ROWS. `selection` names the arms, the grids, the model id, the scorer hash and each
    arm's raw run count (`n_checkpoint_runs`, `n_baseline_runs`), and `pairing` says how many of
    those runs on each side found no twin and were dropped. An arm whose raw count is 0 never
    reaches this JSON at all -- see `EmptyArm` below.
  * WHICH COLUMNS. `criteria.stop_2x2.columns` spells out both axes and the UNIT, because
    `runs.parquet` has no coverage column -- the done axis comes from the `frontier_q#k`
    ladder in `scores` -- and because the unit is a decision point (run, t) rather than a run.
    It also records why the done axis is not `stop_undershoot`, which is identically 0.
  * WHETHER THE COMPARISON WAS FAIR. `k_matches_baseline` is reported per suite and is never
    gated: `k` is not a column of `runs.parquet` at all, so it is read off the grid files, and
    a mismatch makes a delta a retrieval-budget contrast as well as a training contrast. That
    is a fact a reader must be told, not one this command should decide for them.

AN ABSENT CRITERION FAILS. Not passes, not warns. A gate that passes for want of data emits a
PASS verdict with a provenance trail, which is precisely the artifact someone will cite.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "run_gate",
    "cad_ge2_by_run",
    "bootstrap_ci",
    "bca_ci",
    "GateInputsMissing",
    "LadderInconsistent",
    "EmptyArm",
    "IncompleteArm",
    "UnscoredArm",
    "COVERAGE_RULES",
    "follow_ups",
    "sign_test_p",
    "fork_paired_engagement",
]

# The endpoints I.9 Tier B names, and how each is read. `newly_reachable_share` is carried
# because it is informative and NOT gated, because Tier B does not name it: gating an endpoint
# nobody preregistered would let this command reject a checkpoint on a rule the plan never
# stated. The `gated` flag here is the CAP-8 rule's; under `--coverage-rule matched_cost` both
# `facet_breadth` and `cad_ge2` are downgraded to report-only in `run_gate`, each for the
# reason recorded there.
PAIRED_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("evidence_coverage", "gain", True),
    ("cad_ge2", "gain", True),
    ("facet_breadth", "no_loss", True),
    ("newly_reachable_share", "report", False),
)


COVERAGE_RULES: tuple[str, ...] = ("cap8", "matched_cost")
"""How the COVERAGE criterion picks the baseline it subtracts (D9, 2026-09-15).

`cap8` is the original: the baseline run's terminal `evidence_coverage`, i.e. the whole cap-8
episode. It charges a checkpoint that stops at three questions for the five it did not buy and
credits it nothing for the cost it saved, so it cannot decide "more required evidence at an equal
NUMBER OF QUESTIONS" in either direction. Questions, not retrieval calls: `matched_cost` matches
on `runs.n_asks` and this module reads `retrieval_calls` nowhere -- see `_matched_cost`.

`matched_cost` -- the default -- contrasts the checkpoint at its own natural stop k against the
BASELINE'S OWN PREFIX at the same k, read from `scores.frontier_q#k`. That prefix is a
legitimate cap-k run because `Inquirer.act(s: State)` takes no ledger, no context and no budget:
the base's k-th question is a function of the state alone, so the first k questions of a cap-8
run are the questions a cap-k run would have asked and its coverage after k asks is that run's
terminal coverage. A budget-aware policy would break this construction, which is one more reason
that parameter does not exist.

The cap-8 delta does not disappear: it is reported per suite in `matched_cost.by_suite` as the
cost line beside the matched one, and it is not gated.

`matched_cost` also moves three other criteria, each stamped at the top of the verdict and each
for the same reason -- the comparator, not the checkpoint, is what they were failing on:

  * `facet_breadth` -> REPORT-ONLY (`facet_rule`);
  * `cad_ge2` -> REPORT-ONLY (`cad_rule`), because C@d>=2 scales with the number of asks too
    and `scores` carries no `cad#d` ladder over prefixes to read it at matched k;
  * `length_equivalence` -> ONE-SIDED non-inferiority (`length_rule`), because the confound it
    exists to catch is a LONGER question, and a shorter one cannot produce it;
  * `stop_2x2` -> REPORT-ONLY (`stop_rule`), because the prompted base almost never stops, so
    its P(ASK | not done) is ~1 by construction and the rule is unreachable.

Each block in `run_gate` carries its measurements. Under `cap8` all four are unchanged."""


class GateInputsMissing(RuntimeError):
    """A table the gate needs is not in the parquet directory."""


class LadderInconsistent(RuntimeError):
    """The prefix ladder is not the column the coverage delta is denominated in.

    The gold-side instrument lock lives in `scripts/matched_cost.py`, which re-runs the real
    `MechanicalMatcher` against gold on every prefix and refuses unless every rung reproduces
    the verdicts' own scorer. Contract 3 forbids this module those imports, so the gate checks
    the one thing parquet alone can decide, and checks it on every baseline ladder the delta
    reads: `frontier_q#n_asks` IS `evidence_coverage`. Where the two disagree the prefix is
    measuring something other than the quantity being differenced, and no matched-cost number is
    reportable -- so this is raised, not warned.
    """


class GraphVersionAmbiguous(RuntimeError):
    """The runs a verdict selected carry more than one `graph_version`.

    `graph_version` is the third element of the provenance triple CONTRIBUTING.md rule 1 requires of
    every reported value, beside `run_id` and `scorer_hash`. It is an IDENTITY, not a
    measurement: there is no mean of two graph versions and no defensible rule for picking one,
    so a verdict spanning two of them cannot name the graph its numbers came from. Refused here
    for the same reason `EmptyArm` and `LadderInconsistent` are -- a verdict that is wrong about
    its own provenance is worse than no verdict, because the number in it looks reportable.

    Measured 2026-09-17 before this was written: 0 of 64 `artifacts/gate/*/parquet_*/
    scores.parquet` carry more than one `graph_version` (or more than one `scorer_hash`), so
    this guards a future mixed store rather than describing one on disk.
    """

    def __init__(self, *, versions: Sequence[str], parquet_dir: Path, n_runs: int) -> None:
        self.versions = list(versions)
        self.parquet_dir = Path(parquet_dir)
        self.n_runs = n_runs
        super().__init__(
            f"{len(self.versions)} distinct graph_version values over the {n_runs} runs "
            f"selected from {parquet_dir}: {self.versions}. A graph_version is run identity, "
            "not a quantity -- it is neither averaged nor chosen. Re-score the arm so one "
            "graph scored all of it, or gate the halves separately."
        )


class EmptyArm(RuntimeError):
    """One side of the checkpoint/baseline contrast selected zero OK runs.

    Incident, 2026-09-16: a staging step upstream matched nothing for the checkpoint arm, and
    `run_gate` went on to compute every criterion against the baseline alone -- writing two
    complete verdicts over an arm that was never in the parquet at all. That is not the case
    `criteria["..."]["reason"] == "no ok runs for the checkpoint arm"` already covers: that path
    is one criterion discovering, on its own, that ITS inputs are absent, and it fails exactly
    that criterion (see the module docstring, "AN ABSENT CRITERION FAILS"). An empty ARM is
    upstream of every criterion -- with zero rows there is no measurement to attempt, on any of
    them, and letting each one discover that separately produces a JSON that reads like twelve
    independent measurements rather than the one true fact (there was nothing to measure). So
    this is checked once, at selection, before a single criterion runs, and raised rather than
    computed: nothing is written to `--out` on this path (see `pi_run.cmd_train.cmd_train_gate`).
    """

    def __init__(
        self,
        *,
        side: str,
        arm: str,
        grid_names: Sequence[str],
        parquet_dir: Path,
        suites: Sequence[str],
    ) -> None:
        self.side = side
        self.arm = arm
        self.grid_names = list(grid_names)
        self.parquet_dir = str(parquet_dir)
        self.suites = list(suites)
        suite_text = (
            ", ".join(self.suites)
            if self.suites
            else "(none observed -- the other arm selected zero runs too)"
        )
        super().__init__(
            f"{side} arm {arm!r} selected 0 runs from grid(s) {self.grid_names!r} in "
            f"parquet dir {self.parquet_dir}. suite(s) this contrast is for (read off the "
            f"other, non-empty arm): {suite_text}. Refusing rather than computing a verdict "
            "whose checkpoint or baseline side is empty."
        )


class IncompleteArm(RuntimeError):
    """The checkpoint arm selected a run whose (suite, task, seed) key has no partner among the
    baseline arm's selected runs.

    Incident, 2026-09-17: six of eight gate arms were only partially run -- a staging step
    upstream had selected checkpoint runs for some but not all of the baseline's tasks -- and
    only a human noticing kept them out of a verdict. `EmptyArm` (above) catches zero rows on
    one side; it does not catch a PARTIAL arm, which is the more dangerous case because it looks
    complete: every criterion still computes, every cell still gets a value, and the printed
    verdict reads exactly like one over a full population. Underneath, every per-task criterion
    (the STOP 2x2 and the rest) is measured only over the PAIRED population -- `pairing` already
    records this as `n_paired_keys` next to each arm's own `n_checkpoint_runs`/`n_baseline_runs`
    -- so a partial arm silently drops the unpaired tasks from every criterion rather than
    counting them against either side.

    Correction, 2026-09-17: this originally raised on `ck_key_set != ba_key_set` -- either arm
    missing a key the other has. Re-measured at 2026-09-17T16:51:10+0300 against every real
    (non-symlink) `*.json` under `artifacts/gate`, recursive and at the top level -- two
    subtrees (`8b1-rescored-t19c/`, `8b2-t19c/`) are entirely symlink farms, `.json` entries
    included, and a walk that mis-globbed or double-counted them would misstate both the
    denominator and which side moved: an earlier pass reported 48 of 156, and both the 156 and
    the 48 were wrong (denominator inflated by symlinked duplicates and short of loose
    top-level files; numerator read a counter only later verdicts carry, not the `pairing`
    block every verdict carries). There are 150 real verdicts, every one with a `pairing`
    block. All 150 have a baseline arm with strictly more keys than the checkpoint
    (`n_unpaired_baseline > 0`); zero have a checkpoint key the baseline lacks
    (`n_unpaired_checkpoint` is 0 in all 150). That is not 150 near-misses: baselines are
    deliberately run at more seeds than the checkpoints compared against them (one instance,
    `qwen3-8b-sft.musique.json`: 132 paired, 132 unpaired-baseline, 0 unpaired-checkpoint, from
    a second baseline-only seed), pairing intersects on the shared keys and records what it set
    aside, and that is the design working, not a defect the corpus happens to tolerate. It
    drops nothing from the checkpoint and is not this failure. Only a checkpoint key the
    baseline lacks is: a checkpoint RUN that would be silently excluded from every criterion,
    with nothing in the verdict distinguishing that from a complete gate -- a hazard this check
    stays armed for even though it has not yet occurred once in 150 real verdicts. So this now
    raises when the checkpoint's key set is NOT A SUBSET of the baseline's; a baseline with
    extra keys never raises it and is paired and counted exactly as
    `test_unpaired_runs_are_dropped_and_counted` describes. This is checked once, here, before a
    single criterion runs, exactly where `EmptyArm` is checked: nothing is written to `--out` on
    this path (see `pi_run.cmd_train.cmd_train_gate`).

    Known, and now checked exactly rather than disclosed as a gap: a checkpoint that is short
    of its OWN grid's declared tasks but still a subset of the baseline's (e.g. staged on 40 of
    a 50-task grid whose baseline happens to cover all 50) passes THIS check -- a subset is a
    subset -- but is caught by `UnderfilledArm` immediately below `UnscoredArm`, given
    `--grids-root`, because the original incident was a shortfall in HOW MANY tasks ran, not a
    question of WHICH. What stays unreachable is the missing tasks' IDENTITY, not their count:
    naming them needs the grid's corpus-ordered id list (`Grid.select`, `data/corpora`), a
    dependency this module does not have and should not acquire, for the same three reasons the
    comment above this check gives. See `UnderfilledArm` for the count check and its one honest
    residual (a grid with no declared cap at all), and docs/TRAINING.md.

    NOT raised for a fork grid (T16b): a fork's pairing key is `(foreign_trace_sha,
    n_prefix_user_turns, seed)`, not `(suite_id, task_id, seed)`, and `_fork_pairs` already
    refuses its own version of this -- two runs at one fork key -- with its own message. This
    check is skipped whenever every selected checkpoint run carries a `foreign_trace_sha`, the
    same predicate `_fork_criteria` uses to decide a grid is a fork grid at all.
    """

    def __init__(
        self,
        *,
        checkpoint_arm: str,
        baseline_arm: str,
        n_checkpoint: int,
        n_baseline: int,
        n_paired: int,
        only_in_checkpoint: Sequence[str],
        only_in_baseline: Sequence[str],
    ) -> None:
        self.checkpoint_arm = checkpoint_arm
        self.baseline_arm = baseline_arm
        self.n_checkpoint = n_checkpoint
        self.n_baseline = n_baseline
        self.n_paired = n_paired
        self.only_in_checkpoint = list(only_in_checkpoint)
        self.only_in_baseline = list(only_in_baseline)

        def _fmt(missing: Sequence[str], *, side: str) -> str:
            if not missing:
                return f"0 tasks only in the {side} arm"
            examples = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            return f"{len(missing)} task(s) only in the {side} arm, e.g. {examples}{more}"

        super().__init__(
            f"checkpoint arm {checkpoint_arm!r} ({n_checkpoint} task/seed keys) selected a run "
            f"whose key has no partner in baseline arm {baseline_arm!r} ({n_baseline} "
            f"task/seed keys): {_fmt(self.only_in_checkpoint, side='checkpoint')}. Only "
            f"{n_paired} keys are paired. ({_fmt(self.only_in_baseline, side='baseline')}, "
            "which is not this failure -- a baseline with extra keys is paired and counted, "
            "not refused.) Every checkpoint run named above would be silently excluded from "
            "every criterion rather than counted against either side. Refusing rather than "
            "computing a verdict over a population smaller than the checkpoint's own selected "
            "runs while the printed verdict reads as if it covered every one of them."
        )


class UnscoredArm(RuntimeError):
    """One side of the checkpoint/baseline contrast has zero score rows for its selected runs.

    Near-miss, 2026-09-17, an hour before write-up: eight gate verdicts came back `passed:
    false` on every criterion, and the cause was not the checkpoints -- their `scores.parquet`
    held zero rows because the score step had never run, against 107,135 rows in the reference.
    Every criterion still computed a value anyway (a mean of nothing, or a baseline-only read,
    depending on the criterion), and the printed verdict was indistinguishable from "these
    checkpoints failed" -- it said so, on every criterion, eight times. Only a human checking
    row counts before reporting caught it before it entered the record. Neither `EmptyArm` nor
    `IncompleteArm` (above) catches this: the arm's runs are present in `runs.parquet` and
    cover the same tasks as the other arm, so both checks see a normal, complete, paired arm.
    The failure is one level deeper, in `scores.parquet` specifically. Checked once, here, for
    the same reason as the other two: an unscored arm is upstream of every criterion, not one
    of them, and a criterion given zero score rows still returns a value rather than an error.

    Deliberately not more precise than "zero rows total for this arm's runs": that is the shape
    that produced the near-miss (the score step never ran at all, not "ran but missed a
    metric"), and it is cheap to detect with the query `run_gate` already makes into `scores`
    at this point -- distinguishing "some rows missing" from "no rows at all" would need a
    second, metric-aware query this incident gives no reason to add.
    """

    def __init__(self, *, side: str, arm: str, n_runs: int, n_score_rows: int) -> None:
        self.side = side
        self.arm = arm
        self.n_runs = n_runs
        self.n_score_rows = n_score_rows
        super().__init__(
            f"{side} arm {arm!r} selected {n_runs} run(s) with {n_score_rows} score row(s) "
            "in `scores`. The score step has not been run on this parquet dir -- run "
            "`pi score` on it, then re-run `pi train gate`. Refusing rather than computing a "
            "verdict whose criteria would read as failures caused by the checkpoint rather "
            "than by missing scores."
        )


class UnderfilledArm(RuntimeError):
    """The checkpoint arm's PAIRED task count, for some suite in its own grid, is below that
    grid's own declared task count.

    Incident, 2026-09-17 (the same one `IncompleteArm` documents): a staging step selected
    checkpoint runs for 40 of the baseline's 50 tasks. The checkpoint's 40 keys are a SUBSET of
    the baseline's 50, so `IncompleteArm`'s directional rule -- refuse only when the checkpoint
    has a key the baseline lacks -- does not fire; a baseline run at more tasks than its
    checkpoint is the ordinary, tolerated shape that check exists to allow. The original
    incident was a shortfall in HOW MANY tasks ran, not a question of WHICH, so this checks the
    paired count against the one number that does not need the corpus to know: the grid's own
    declared `n_tasks` (or `len(task_ids)`), read off the grid FILE named by `--grids-root` and
    `grid_name` -- the same read `_k_by_suite` already makes for `k_matches_baseline`, with no
    new import: it parses conf/grids as data.

    Requires `--grids-root`; without it (or if the named grid is not found under it, or the
    grid declares neither `n_tasks` nor `task_ids`) this does not raise at all -- see
    `_k_by_suite`'s docstring for why a grid that cannot be resolved is stamped null with a
    reason rather than guessed at, which is `run_gate`'s `task_count` block either way.

    Does NOT name the missing tasks -- only their count is checked, because the identity of a
    missing task needs `Grid.select`'s corpus-ordered id list (`data/corpora`), a dependency
    this module does not have and should not assume; see the comment above `IncompleteArm`'s
    check for the fuller reasoning, which is unchanged by this class. Skipped for a fork grid,
    for the same reason `IncompleteArm` is: a fork's pairing key is not `(suite_id, task_id,
    seed)`, so a task count declared by `n_tasks` means something else there.
    """

    def __init__(
        self,
        *,
        checkpoint_arm: str,
        grid_name: str,
        shortfalls: Sequence[tuple[str, int, int]],  # (suite, n_paired, n_declared)
    ) -> None:
        self.checkpoint_arm = checkpoint_arm
        self.grid_name = grid_name
        self.shortfalls = list(shortfalls)
        detail = ", ".join(
            f"{suite!r}: {paired} paired of {declared} declared"
            for suite, paired, declared in self.shortfalls
        )
        super().__init__(
            f"checkpoint arm {checkpoint_arm!r} under grid {grid_name!r} is short of that "
            f"grid's own declared task count: {detail}. Refusing rather than computing a "
            "verdict over fewer tasks than the grid that defines this contrast calls for, "
            "while the printed verdict reads as if it covered every one of them."
        )


# --------------------------------------------------------------------------- plumbing


def _con(parquet_dir: Path):
    from pinq.extras import require

    duckdb = require("duckdb", why="pi train gate reads the compacted parquet")
    con = duckdb.connect()
    for name in ("runs", "turns", "scores", "ledger", "calls"):
        p = Path(parquet_dir) / f"{name}.parquet"
        if not p.exists():
            raise GateInputsMissing(
                f"{p} is missing. `pi train gate` reads the output of `pi compact`; point "
                "--parquet-dir at that directory."
            )
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con


def _q(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _in(vals: Iterable[str]) -> str:
    vals = list(vals)
    return "(" + ", ".join(_q(v) for v in vals) + ")" if vals else "('')"


def _rows(con, sql: str) -> list[dict[str, Any]]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else float("nan")


def _nan(x: float | None) -> float | None:
    """NaN is not JSON. `null` says 'not measured'; 0.0 would say 'measured, and zero'."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# --------------------------------------------------------------------------- selection


def _run_filter(*, arm: str, grids: Sequence[str]) -> str:
    return f"r.arm_id = {_q(arm)} AND r.grid_name IN {_in(grids)} AND r.status = 'ok'"


def _select_runs(con, *, arm: str, grids: Sequence[str], model_id: str | None) -> list[dict]:
    """The (run, suite, task, seed) keys for one arm.

    `model_id` is matched through `calls.parquet` (`actor='inquirer'`), because `runs.parquet`
    carries `model_pin_hash` and no model NAME. Under the two-pin protocol the same arm id runs
    against two different base models, so without this the two pins pool into one baseline and
    the delta is measured against their average -- a number that describes neither arm.
    """
    where = _run_filter(arm=arm, grids=grids)
    if model_id:
        where += (
            f" AND r.run_id IN (SELECT run_id FROM calls WHERE actor = 'inquirer' "
            f"AND model = {_q(model_id)})"
        )
    return _rows(
        con,
        f"SELECT r.run_id, r.suite_id, r.task_id, r.seed FROM runs r WHERE {where} "
        "ORDER BY r.suite_id, r.task_id, r.seed, r.run_id",
    )


def _by_key(runs: Sequence[Mapping[str, Any]]) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    for r in runs:
        out.setdefault((r["suite_id"], r["task_id"], r["seed"]), []).append(r["run_id"])
    return out


# --------------------------------------------------------------------------- cad_ge2


def cad_ge2_by_run(parquet_dir: Path, *, scorer_hash: str) -> dict[str, float]:
    """C@d>=2 per run, |V_d|-weighted. Derived: it is not stored anywhere.

    WEIGHTING BY CARDINALITY IS NOT A NICETY. An unweighted mean over depths lets a depth
    holding two nodes outvote one holding forty, so a policy that nails a shallow two-node
    frontier and misses a deep forty-node one scores as if it had done well.

    A run with nothing at depth >= 2 is ABSENT from the result, never 0.0: its graph has no
    C@d>=2 to report, and a fabricated zero would drag the arm's mean down for a task that
    never carried the quantity.
    """
    con = _con(Path(parquet_dir))
    rows = _rows(
        con,
        "WITH d AS (\n"
        "  SELECT s.run_id, split_part(s.metric_name, '#', 1) AS fam,\n"
        "         CAST(split_part(s.metric_name, '#', 2) AS INTEGER) AS depth, s.value\n"
        "  FROM scores s\n"
        f"  WHERE s.scorer_hash = {_q(scorer_hash)}\n"
        "    AND split_part(s.metric_name, '#', 1) IN ('cad', 'cad_n')\n"
        ")\n"
        "SELECT c.run_id, sum(c.value * n.value) / nullif(sum(n.value), 0) AS value\n"
        "FROM d c JOIN d n USING (run_id, depth)\n"
        "WHERE c.fam = 'cad' AND n.fam = 'cad_n' AND c.depth >= 2\n"
        "GROUP BY 1 ORDER BY 1",
    )
    return {r["run_id"]: float(r["value"]) for r in rows if r["value"] is not None}


def _metric_by_run(con, metric: str, *, scorer_hash: str) -> dict[str, float]:
    rows = _rows(
        con,
        "SELECT run_id, avg(value) AS value FROM scores "
        f"WHERE metric_name = {_q(metric)} AND scorer_hash = {_q(scorer_hash)} "
        "GROUP BY 1",
    )
    return {r["run_id"]: float(r["value"]) for r in rows if r["value"] is not None}


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    values: Sequence[float], *, seed: int, n_resamples: int, lo: float, hi: float
) -> tuple[float, float]:
    """Percentile bootstrap over the given units. Seeded, so a verdict is reproducible."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    def pct(p: float) -> float:
        idx = min(len(means) - 1, max(0, int(round(p * (len(means) - 1)))))
        return means[idx]

    return (pct(lo), pct(hi))


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics -- `pi_eval.stats.inference._percentile`.

    NOT `bootstrap_ci`'s nearest-index rule. The two differ by up to one order statistic, which
    is a visible amount at 1000 resamples, and the matched-cost interval has to be the interval
    `paired_difference` would have returned or the figure and the verdict are two estimators.
    """
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _phi_inv(p: float) -> float:
    """Acklam's rational approximation, accurate to ~1e-9 -- plenty for a CI endpoint, and the
    same approximation `pi_eval.stats.inference` uses, so the two agree to the last digit."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def bca_ci(
    values: Sequence[float], *, seed: int, n_resamples: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(point, lo, hi): a bias-corrected and accelerated bootstrap over TASKS.

    REPLICATED FROM `pi_eval.stats.inference.cluster_bootstrap`, not approximated, for the
    reason this module's header gives about `cad_ge2`: contract 3 forbids the import and D9
    names that interval, so the alternative to replicating it is reporting a different estimator
    under the same name. Each task here is a singleton cluster -- the per-task delta is already
    a mean over that task's seeds -- so the cluster estimand reduces to the mean over tasks, and
    the resampling, the bias correction and the jackknife acceleration are step for step the
    ones `paired_difference` runs. `statistics.fmean` rather than `sum(...)/n` for the same
    reason: pairwise summation rounds differently, and "reproduces P13 to 1e-12" is a claim
    about the last digit.

    BCa RATHER THAN THE PERCENTILE BOOTSTRAP `bootstrap_ci` USES. The statistic is a difference
    of correlated scores whose bootstrap distribution is routinely skewed, and percentile
    intervals under-cover in exactly that case. `bootstrap_ci` is left untouched because
    `--coverage-rule cap8` and every other paired criterion are the intervals already published.
    """
    vals = sorted(float(v) for v in values if not math.isnan(float(v)))
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    # SORTED, FOR THE SAME REASON `cluster_bootstrap` SORTS ITS UNITS, and replicated here for
    # the same reason the rest of this function is: resampling draws INDICES, so the caller's
    # list order was an input to the endpoints. `_fork_report` feeds this `by_point.values()`,
    # a dict filled in parquet row order rather than by a sorted key, so the exposure was live
    # and not hypothetical. Every value here is one singleton cluster, so there is no inner
    # level to canonicalise, and the NaNs are already gone. Measured on FRAMES: one 1/n step on
    # three of four published endpoints. The point estimate never moves -- a mean is a mean.
    theta_hat = statistics.fmean(vals)
    rng = random.Random(seed)
    n = len(vals)
    reps = [
        statistics.fmean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)
    ]
    if not reps:
        return (theta_hat, float("nan"), float("nan"))

    prop = sum(1 for r in reps if r < theta_hat) / len(reps)
    z0 = _phi_inv(min(max(prop, 1e-9), 1 - 1e-9))

    jack = [statistics.fmean(vals[:i] + vals[i + 1 :]) for i in range(n) if n > 1]
    if len(jack) > 1:
        jbar = statistics.fmean(jack)
        num = sum((jbar - x) ** 3 for x in jack)
        den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5)
        acc = num / den if den else 0.0
    else:
        acc = 0.0

    def adj(q: float) -> float:
        z = _phi_inv(q)
        denom = 1 - acc * (z0 + z)
        return _phi(z0 + (z0 + z) / denom) if denom else q

    return (theta_hat, _percentile(reps, adj(alpha / 2)), _percentile(reps, adj(1 - alpha / 2)))


def _paired_by_task_map(
    ckpt_keys: Mapping[tuple, list[str]],
    base_keys: Mapping[tuple, list[str]],
    by_run: Mapping[str, float],
) -> dict[tuple, float]:
    """Per-(suite, task) deltas, task-matched at the SAME SEED then averaged over seeds.

    Averaging over seeds before resampling is what keeps seeds from inflating the effective n:
    two seeds of one task are two measurements of one task, not two tasks.
    """
    per_task: dict[tuple, list[float]] = {}
    for key in sorted(set(ckpt_keys) & set(base_keys)):
        c = [by_run[r] for r in ckpt_keys[key] if r in by_run]
        b = [by_run[r] for r in base_keys[key] if r in by_run]
        if not c or not b:
            continue
        per_task.setdefault((key[0], key[1]), []).append(_mean(c) - _mean(b))
    return {k: _mean(per_task[k]) for k in sorted(per_task)}


def _paired_by_task(
    ckpt_keys: Mapping[tuple, list[str]],
    base_keys: Mapping[tuple, list[str]],
    by_run: Mapping[str, float],
) -> list[float]:
    """The same deltas as a list, in (suite, task) order -- the bootstrap's unit order."""
    return list(_paired_by_task_map(ckpt_keys, base_keys, by_run).values())


# --------------------------------------------------------------------------- criteria


def _criterion(
    *, value, baseline=None, threshold=None, gated: bool, passed, n: int, reason: str = "", **extra
) -> dict[str, Any]:
    out = {
        "value": _nan(value) if isinstance(value, float) else value,
        "baseline": _nan(baseline) if isinstance(baseline, float) else baseline,
        "threshold": threshold,
        "gated": gated,
        "passed": passed,
        "n": n,
        "reason": reason,
    }
    out.update(extra)
    return out


MALFORMED_TOLERANCE = 0.005
"""Measured on `artifacts/gate/8b2` (2026-09-16): six DPO checkpoints failed `malformed` on
strategyqa on exactly one malformed event in 333 runs (rate 0.003) against a baseline of 0.000
events -- a rounding-level event, not a behaviour. `malformed` now passes at
`rate <= max(baseline_rate, MALFORMED_TOLERANCE)`: a single stray event against an otherwise
clean baseline no longer fails the gate, while a baseline that is itself noisier still sets its
own (higher) bar rather than being clamped down to this constant.

`criteria["malformed"]` below keeps the EXACT shape it always had --
`test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` hashes `criteria` byte for
byte, and every field added here would move that hash even on a fixture where the tolerance
changes no outcome. So the tolerance, the raw event count and both rates are reported instead
in `malformed_tolerance`, a sibling of `criteria` in the verdict and therefore outside the
cap-8 hash."""


def _malformed(con, *, runs: Sequence[Mapping], label: str) -> tuple[float, int, int]:
    """Malformed events per OK run. `malformed` is a ledger CURRENCY, not a runs column.

    Returns `(rate, n_runs, n_events)`. The raw event count is returned alongside the rate so
    a verdict can report it exactly, rather than recovering it by rounding `rate * n_runs`.
    """
    ids = [r["run_id"] for r in runs]
    if not ids:
        return (float("nan"), 0, 0)
    rows = _rows(
        con,
        "SELECT coalesce(sum(charged), 0) AS n FROM ledger "
        f"WHERE currency = 'malformed' AND run_id IN {_in(ids)}",
    )
    n_events = int(rows[0]["n"])
    return (float(n_events) / len(ids), len(ids), n_events)


def _questions(con, runs: Sequence[Mapping]) -> list[tuple[str, str]]:
    ids = [r["run_id"] for r in runs]
    if not ids:
        return []
    rows = _rows(
        con,
        "SELECT r.task_id, t.question FROM turns t JOIN runs r ON r.run_id = t.run_id "
        f"WHERE t.run_id IN {_in(ids)} AND lower(t.action_kind) = 'ask' "
        "AND t.question IS NOT NULL AND t.question <> '' ORDER BY 1, 2",
    )
    return [(str(r["task_id"]), str(r["question"])) for r in rows]


def _questions_with_seed(con, runs: Sequence[Mapping]) -> list[tuple[str, Any, str]]:
    """Like `_questions`, but keeps `r.seed`.

    `_questions` drops seed by design -- it feeds `within_task_distinct_n`, which groups by
    task only. `criteria["distinct3"]["by_seed"]` needs the finer (task, seed) key instead
    (see `within_task_seed_distinct_n`), so this is a second query rather than a change to
    `_questions`: `_questions`' output also feeds `length_equivalence` and its arrival order is
    part of that criterion's own contract (see the comment above the `_length` call site), and
    `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` hashes `criteria` byte for
    byte -- nothing about `_questions` itself should move.
    """
    ids = [r["run_id"] for r in runs]
    if not ids:
        return []
    rows = _rows(
        con,
        "SELECT r.task_id, r.seed, t.question FROM turns t JOIN runs r ON r.run_id = t.run_id "
        f"WHERE t.run_id IN {_in(ids)} AND lower(t.action_kind) = 'ask' "
        "AND t.question IS NOT NULL AND t.question <> '' ORDER BY 1, 2, 3",
    )
    return [(str(r["task_id"]), r["seed"], str(r["question"])) for r in rows]


DONE_AT = 1.0 - 1e-12
"""`pi_run.cmd_train._done`'s threshold, restated here because contract 3 forbids the import.
The ONE threshold the export's `done_before` also uses, so "done" cannot mean two things."""


def _coverage_ladder(
    con, runs: Sequence[Mapping], *, scorer_hash: str
) -> dict[str, dict[int, float]]:
    """Required-evidence coverage after k asks, per run, from `scores.frontier_q#k`.

    `pi_eval.score` builds this ladder by merging each turn's `retrieved_uids` in order, so
    `frontier_q#k` is the coverage the policy HOLDS when it decides at t = k -- the same
    quantity `pi_run.cmd_train` writes as `coverage_before` on the export's decision rows. On
    all three gate parquets `frontier_q#0` is 0.0 on 1,395/1,395 runs and `frontier_q#n_asks`
    equals `evidence_coverage` exactly, so the two ends of the ladder agree with the two
    quantities already reported elsewhere.

    A point is ABSENT when the task carries no required gold evidence: the scorer omits it
    rather than emitting 0.0, because coverage is undefined there and not zero. The absence is
    carried through as an absence -- see `_stop_2x2`, which skips and counts such a state.
    """
    ids = [r["run_id"] for r in runs]
    if not ids:
        return {}
    rows = _rows(
        con,
        "SELECT run_id, CAST(split_part(metric_name, '#', 2) AS INTEGER) AS k, value FROM scores "
        "WHERE split_part(metric_name, '#', 1) = 'frontier_q' "
        f"AND scorer_hash = {_q(scorer_hash)} AND run_id IN {_in(ids)}",
    )
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        if r["value"] is None:
            continue
        v = float(r["value"])
        if math.isnan(v):
            continue
        out.setdefault(str(r["run_id"]), {})[int(r["k"])] = v
    return out


def _stop_2x2(
    con, runs: Sequence[Mapping], ladders: Mapping[str, Mapping[int, float]]
) -> dict[str, Any]:
    """P(STOP | done) and P(ASK | not done) over DECISION POINTS, not over runs.

    THE UNIT IS A STATE. One episode offers the policy n_asks+1 decisions, t = 0..n_asks, and
    a run-level 2x2 collapses all of them into the last one: a run that asked six questions
    after it was already done and then stopped scored 1.0, with the six wasted questions
    invisible. This mirrors the offline Tier A 2x2 (`pinq_train.eval_offline.stop_confusion`
    over the export's `done_before`), so the two tiers count the same events.

    DONE AXIS: `frontier_q#t >= 1 - 1e-12`, i.e. required-evidence coverage BEFORE the decision
    at t. NOT `stop_undershoot`: that metric is max(0, k* - k_hat) with k* the argmax of a
    ladder that is monotone in k and k_hat its last index, so k* <= k_hat identically and the
    metric is 0 on every run ever scored -- 1,395/1,395 on each gate parquet. Its "not done"
    cell was empty by construction and the criterion could only ever fail for want of data.
    The ladder comes from `scores` and not from `runs` because `runs.parquet` HAS NO COVERAGE
    COLUMN.

    STOP AXIS: ASK at every t < n_asks, because the episode has a turn there. STOP at t =
    n_asks ONLY on `runs.stop_reason = 'policy_stop'`. Under 'budget' or 'max_turns' the
    HARNESS halted the episode -- `pinq.loop` breaks on the refused retrieval charge or runs
    out of its for-loop -- and the policy was never consulted at that state, so it carries no
    decision: it is excluded and counted as `n_forced_stops`. Counting it as an ASK, which the
    run-level version did, credits the cap with the policy's judgement. The empty string occurs
    on exactly the `status='error'` rows, which `_run_filter` has already excluded.
    """
    n_done = n_not = stop_done = ask_not = 0
    n_forced = n_skipped = n_runs = 0
    asks_after_done: list[float] = []
    for r in runs:
        lad = ladders.get(str(r["run_id"]))
        if lad is None:
            n_skipped += int(r.get("n_asks") or 0) + 1
            continue
        n_runs += 1
        n_asks = int(r["n_asks"] or 0)
        policy_stop = str(r.get("stop_reason") or "") == "policy_stop"
        after_done = 0
        for t in range(n_asks + 1):
            action_is_stop = t == n_asks
            if action_is_stop and not policy_stop:
                n_forced += 1
                continue
            cov = lad.get(t)
            if cov is None:
                n_skipped += 1
                continue
            if cov >= DONE_AT:
                n_done += 1
                stop_done += int(action_is_stop)
                after_done += int(not action_is_stop)
            else:
                n_not += 1
                ask_not += int(not action_is_stop)
        asks_after_done.append(float(after_done))
    return {
        "p_stop_given_done": (stop_done / n_done) if n_done else float("nan"),
        "p_ask_given_not_done": (ask_not / n_not) if n_not else float("nan"),
        "n_done": n_done,
        "n_not_done": n_not,
        "n_done_stop": stop_done,
        "n_done_ask": n_done - stop_done,
        "n_not_done_ask": ask_not,
        "n_not_done_stop": n_not - ask_not,
        "n_states": n_done + n_not,
        "n_runs": n_runs,
        "n_forced_stops": n_forced,
        "n_skipped_no_coverage": n_skipped,
        # Questions bought at states that were ALREADY done, averaged over runs. Reported and
        # never gated: it is the same trade the two cells hold, in the unit an operator pays in.
        "mean_asks_after_done": _mean(asks_after_done),
    }


LADDER_TOL = 1e-12
"""`frontier_q#n_asks` and `evidence_coverage` are one quantity read twice, so the tolerance is
float noise and nothing else. `scripts/matched_cost.py` uses the same 1e-12 against gold."""


def _check_ladder_is_the_coverage_column(
    run_ids: Sequence[str],
    ladders: Mapping[str, Mapping[int, float]],
    coverage: Mapping[str, float],
    n_asks: Mapping[str, int],
) -> int:
    """Every baseline ladder the delta reads must END at that run's own `evidence_coverage`.

    A run with NO coverage row is skipped, not refused: the scorer omits the metric when the
    task carries no required gold evidence, and an absence is an absence. It is also excluded
    from the contrast by the same rule the cap-8 criterion uses, so nothing is scored off a
    ladder that was never checked.
    """
    bad: list[str] = []
    checked = 0
    for rid in run_ids:
        cov = coverage.get(rid)
        if cov is None:
            continue
        checked += 1
        top = n_asks.get(rid, 0)
        rung = (ladders.get(rid) or {}).get(top)
        if rung is None or abs(float(rung) - float(cov)) > LADDER_TOL:
            bad.append(f"{rid}: frontier_q#{top} is {rung!r} but evidence_coverage is {cov!r}")
    if bad:
        raise LadderInconsistent(
            f"{len(bad)} baseline run(s) whose prefix ladder does not end at the run's own "
            "evidence_coverage. The matched-cost rule subtracts a rung of that ladder from a "
            "checkpoint's evidence_coverage, so where the two ends disagree the delta is a "
            "difference of two different instruments and no matched-cost number is reportable. "
            "Re-score the parquet, or run with --coverage-rule cap8, which reads no ladder. "
            "First five:\n  " + "\n  ".join(sorted(bad)[:5])
        )
    return checked


def _matched_cost(
    con,
    *,
    ckpt: Sequence[Mapping],
    base: Sequence[Mapping],
    ck_keys: Mapping[tuple, list[str]],
    ba_keys: Mapping[tuple, list[str]],
    scorer_hash: str,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    """The coverage contrast at matched QUESTION COUNT, per suite and pooled.

    THE MATCHING VARIABLE IS `runs.n_asks`, NOT `runs.retrieval_calls`. This docstring said
    "matched retrieval cost" until 2026-09-18, while this module has never READ a
    `retrieval_calls` column at all (these two paragraphs are its only occurrences in the file,
    and both name it to say it is unused): every `k` and every charge below comes from `n_asks`.

    On every population this rule has been run against the two coincide exactly:
    `n_asks == retrieval_calls` on 3,369 of 3,369 runs of the test-split QA store, one retrieval
    per ask and no other retrieval channel. So no reported number was ever wrong, and nothing in
    the repo disagreed with the wrong name. That is why it survived: a label only gets checked
    when something contradicts it. The two names are NOT interchangeable in general: a policy
    that retrieved twice for one question, or asked without retrieving, would make them
    different currencies, and only this one is matched here. `columns` in the emitted verdict
    named `n_asks` correctly throughout.

    THE UNIT OF COST IS THE CHECKPOINT'S OWN k. For each paired (suite, task, seed) the
    checkpoint contributes its terminal `evidence_coverage` and the baseline contributes
    `frontier_q#min(k, its own n_asks)`.

    `min`, NOT k. A baseline run that stopped at two under cap 8 stops at two under cap 3 as
    well -- it is budget-blind, so a smaller cap cannot make it ask more -- and charging it k
    would invent spend it never made in the very comparison whose subject is spend. Those runs
    are counted as `n_baseline_shorter_than_k` because they are the cases where "matched" cost
    is matched only from above.

    Seeds are averaged within a task before the task enters the bootstrap, exactly as
    `_paired_by_task` does, so two seeds of one task stay one task.
    """
    cov = _metric_by_run(con, "evidence_coverage", scorer_hash=scorer_hash)
    n_asks = {str(r["run_id"]): int(r["n_asks"] or 0) for r in _with_stop(con, [*ckpt, *base])}
    shared = sorted(set(ck_keys) & set(ba_keys))
    base_used = sorted({rid for key in shared for rid in ba_keys[key]})
    ladders = _coverage_ladder(con, [{"run_id": rid} for rid in base_used], scorer_hash=scorer_hash)
    n_checked = _check_ladder_is_the_coverage_column(base_used, ladders, cov, n_asks)

    # SYMMETRIC READING, ADDED BESIDE THE ORIGINAL (2026-09-19 coordinator correction). The loop
    # below charges the CHECKPOINT its own full terminal `evidence_coverage` (never truncated)
    # and the BASELINE `frontier_q#min(k, its own n_asks)` -- a matched reading only when the
    # checkpoint spends no more than the baseline. Read the other way -- a higher-asking arm as
    # "checkpoint" against a lower-asking one, which every arm-vs-arm ordering not aimed at a
    # fixed budget-blind baseline can hit -- the baseline is never truncated and the checkpoint
    # keeps its own (larger) value: two different budgets reported as one delta. Measured
    # 2026-09-19: Persistence-only's own mean k meets or exceeds the untrained baseline's on two
    # of three suites, so `n_baseline_shorter_than_k` below is 100+ of 200 there, and one
    # arm-vs-arm cell (Persistence-only vs Mixed, MuSiQue) fully reverses sign between the two
    # readings. `ck_ladders` mirrors `ladders` (the baseline's) so BOTH sides can be read off
    # their own ladder at `min(k_ckpt, k_base)`; the ORIGINAL `deltas`/`by_suite` computation
    # below is untouched by this addition -- same variables, same order, same arithmetic -- so
    # every verdict already written from it reproduces bit for bit
    # (`tests/test_train_gate.py::test_a_safe_arms_matched_cost_verdict_is_byte_identical_before_and_after_the_symmetric_addition`).
    ck_used = sorted({rid for key in shared for rid in ck_keys[key]})
    ck_ladders = _coverage_ladder(
        con, [{"run_id": rid} for rid in ck_used], scorer_hash=scorer_hash
    )

    per_task: dict[tuple, list[float]] = {}
    per_task_sym: dict[tuple, list[float]] = {}
    sym_k_common: dict[str, list[float]] = {}
    stats: dict[str, dict[str, list[float] | int]] = {}
    for key in shared:
        suite = str(key[0])
        acc = stats.setdefault(
            suite, {"k": [], "charged": [], "trained_n_asks": [], "base_n_asks": [], "short": 0}
        )
        deltas: list[float] = []
        sym_deltas: list[float] = []
        for c_rid in ck_keys[key]:
            c_val = cov.get(c_rid)
            if c_val is None:
                continue
            k = n_asks.get(c_rid, 0)
            rungs: list[float] = []
            charged: list[float] = []
            short = 0
            for b_rid in ba_keys[key]:
                b_n = n_asks.get(b_rid, 0)
                at = min(k, b_n)
                rung = (ladders.get(b_rid) or {}).get(at)
                if rung is None:
                    continue  # the scorer omits the point where Q is undefined: absent, not 0
                rungs.append(float(rung))
                charged.append(float(at))
                short += int(b_n < k)
                acc["base_n_asks"].append(float(b_n))  # type: ignore[union-attr]
                # SYMMETRIC: the checkpoint is read off ITS OWN ladder at the same `at`, never
                # at its raw (possibly larger) terminal `c_val`. Purely additive: reads
                # `ck_ladders`, writes only `sym_deltas`/`sym_k_common`, touches nothing the
                # original computation above uses.
                ck_rung = (ck_ladders.get(c_rid) or {}).get(at)
                if ck_rung is not None:
                    sym_deltas.append(float(ck_rung) - float(rung))
                    sym_k_common.setdefault(suite, []).append(float(at))
            if not rungs:
                continue
            deltas.append(float(c_val) - _mean(rungs))
            acc["k"].append(float(k))  # type: ignore[union-attr]
            acc["charged"] += charged  # type: ignore[operator]
            acc["trained_n_asks"].append(float(k))  # type: ignore[union-attr]
            acc["short"] = int(acc["short"]) + short
        if deltas:
            per_task.setdefault((key[0], key[1]), []).append(_mean(deltas))
        if sym_deltas:
            per_task_sym.setdefault((key[0], key[1]), []).append(_mean(sym_deltas))

    by_task = {k: _mean(per_task[k]) for k in sorted(per_task)}
    by_task_sym = {k: _mean(per_task_sym[k]) for k in sorted(per_task_sym)}
    cap8 = _paired_by_task_map(ck_keys, ba_keys, cov)
    by_suite: dict[str, Any] = {}
    for suite in sorted({str(k[0]) for k in by_task} | {str(k[0]) for k in cap8}):
        vals = [v for k, v in by_task.items() if str(k[0]) == suite]
        c8 = [v for k, v in cap8.items() if str(k[0]) == suite]
        acc = stats.get(suite, {})
        point, lo, hi = bca_ci(vals, seed=seed, n_resamples=n_resamples)
        trained_mean_k = _mean(list(acc.get("k", ())))  # type: ignore[arg-type]
        baseline_mean_n_asks = _mean(list(acc.get("base_n_asks", ())))  # type: ignore[arg-type]
        by_suite[suite] = {
            "n_tasks": len(vals),
            "delta": _nan(point),
            "ci_lo": _nan(lo),
            "ci_hi": _nan(hi),
            "trained_mean_k": _nan(trained_mean_k),
            "baseline_mean_k_charged": _nan(_mean(list(acc.get("charged", ())))),  # type: ignore[arg-type]
            "n_baseline_shorter_than_k": int(acc.get("short", 0)),  # type: ignore[arg-type]
            "trained_mean_n_asks": _nan(_mean(list(acc.get("trained_n_asks", ())))),  # type: ignore[arg-type]
            "baseline_mean_n_asks": _nan(baseline_mean_n_asks),
            "cap8_coverage_delta": _nan(_mean(c8)),
            "cap8_n_tasks": len(c8),
        }
        # ADDED, 2026-09-19 (see the note above the symmetric block): neither key alters any
        # key already in this dict, so a reader of the ORIGINAL five fields above sees the
        # ORIGINAL five values. `outspent_comparator` is the checkpoint's OWN mean asking rate
        # exceeding the baseline's -- an aggregate of two fields already in this same dict,
        # not a new measurement -- which is what "derived from the count already recorded"
        # means here: `n_baseline_shorter_than_k` alone is nonzero even for safe arms (a few
        # outlier tasks), so a bare `> 0` would flag every cell including the ones this fix
        # must leave alone; the mean comparison is what actually separates them (measured:
        # Persistence-only's own mean k sits at or above the baseline's on two of three test
        # suites, every other arm's sits 2-4 points below it on every suite).
        sym_vals = [v for k, v in by_task_sym.items() if str(k[0]) == suite]
        sym_point, sym_lo, sym_hi = bca_ci(sym_vals, seed=seed, n_resamples=n_resamples)
        by_suite[suite]["symmetric"] = {
            "n_tasks": len(sym_vals),
            "delta": _nan(sym_point),
            "ci_lo": _nan(sym_lo),
            "ci_hi": _nan(sym_hi),
            "mean_k_common": _nan(_mean(sym_k_common.get(suite, []))),
            "columns": {
                "both": "scores.frontier_q#min(k_checkpoint, k_baseline) -- both arms off "
                "their own ladder at the shared, smaller budget; never either arm's raw "
                "untruncated evidence_coverage",
            },
        }
        by_suite[suite]["outspent_comparator"] = bool(
            not math.isnan(trained_mean_k)
            and not math.isnan(baseline_mean_n_asks)
            and trained_mean_k > baseline_mean_n_asks
        )

    pooled = list(by_task.values())
    point, lo, hi = bca_ci(pooled, seed=seed, n_resamples=n_resamples)
    return {
        "rule": "matched_cost",
        "gated": True,
        "metric": "evidence_coverage",
        "by_suite": by_suite,
        "pooled": {
            "n_tasks": len(pooled),
            "delta": _nan(point),
            "ci_lo": _nan(lo),
            "ci_hi": _nan(hi),
        },
        "deltas": pooled,
        "n_baseline_ladders_checked": n_checked,
        "columns": {
            "trained": "scores.evidence_coverage of the run, at its own n_asks",
            "baseline": "scores.frontier_q#min(k, baseline n_asks), k = the trained run's n_asks",
            "why_a_prefix_is_a_cap_k_run": "Inquirer.act(s: State) takes no budget, so the "
            "base's k-th question is a function of the state alone: the first k questions of a "
            "cap-8 run are what a cap-k run would have asked",
            "instrument_check": "every baseline ladder read here ends at that run's own "
            "evidence_coverage (LADDER_TOL 1e-12); the gold-side proof that the ladder is the "
            "verdicts' instrument is scripts/matched_cost.py's verify_instrument, which this "
            "module may not import (contract 3)",
            "ci": f"task-clustered BCa, {n_resamples} resamples, seed {seed}, seeds averaged "
            "within a task first",
            "cap8_coverage_delta": "the criterion this rule replaces, reported per suite and "
            "NOT gated: it is the cost line of an ordered pair, not a second verdict",
            "symmetric": "by_suite[*].symmetric: the same contrast with BOTH arms read off "
            "their own ladder at min(k_checkpoint, k_baseline) -- added 2026-09-19 beside the "
            "original computation, which this key does not alter",
            "outspent_comparator": "by_suite[*].outspent_comparator: True when the "
            "checkpoint's own trained_mean_k exceeds the baseline's own baseline_mean_n_asks "
            "for that suite, i.e. the direction in which the original (non-symmetric) delta "
            "above is not a matched-budget reading. run_gate refuses a pass/fail on any suite "
            "where this is True; read by_suite[*].symmetric there instead",
        },
    }


def _k_by_suite(
    grids_root: Path | None, names: Sequence[str]
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[int | None, str]]]:
    """`k` per (grid name, suite), and each grid's own declared task count, both read off the
    grid FILES in ONE pass -- the declared count costs nothing beyond `k` because it sits in the
    same parsed YAML dict this function already reads `k` from.

    `k` is not a column of `runs.parquet`, so there is no way to recover it from the run
    record. Reading the grids is the honest alternative; where a grid cannot be found the stamp
    is null with a reason, never a guess -- for `k` (that name absent from the first dict) and
    for the declared count (`(None, reason)` in the second).

    The declared count is `n_tasks` if the grid sets it, else `len(task_ids)` if the grid lists
    them explicitly, else `(None, reason)`: a grid selecting by `split` alone with no cap
    declares no count at all, and stamping a guess would be worse than stamping nothing. No
    grid under `conf/grids/` has this shape today (all 32 checked 2026-09-17 declare one or the
    other) -- a documented residual, not a live gap. See `UnderfilledArm`.

    Reading these files is not an import of `pi_run`: they are parsed as plain YAML data, the
    same way `pi_run.grids.Grid` itself is loaded from them elsewhere, so no import-linter
    contract is exercised by this function in either direction. (For the record: no contract
    forbids `pinq_train -> pi_run` either -- the four in pyproject.toml forbid the reverse
    direction and anything reaching `pi_eval`, and none names this one -- so that edge would be
    unused rather than disallowed if anything here ever did import it, which nothing does.)
    """
    if grids_root is None:
        return {}, {n: (None, "no --grids-root given") for n in names}
    from pinq.extras import require

    yaml = require("yaml", why="pi train gate reads conf/grids to stamp k and task counts")
    ks: dict[str, dict[str, int]] = {}
    declared: dict[str, tuple[int | None, str]] = {}
    found: set[str] = set()
    for p in sorted(Path(grids_root).rglob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except Exception:  # noqa: BLE001 - an unreadable grid is a missing stamp, not a crash
            continue
        name = str(raw.get("name", ""))
        if name not in names:
            continue
        found.add(name)
        default_k = int(raw.get("k", 5))
        by_suite = {str(a): int(b) for a, b in (raw.get("k_by_suite") or {}).items()}
        ks[name] = {s: by_suite.get(s, default_k) for s in map(str, raw.get("suites", ()))}
        n_tasks = raw.get("n_tasks")
        task_ids = raw.get("task_ids") or ()
        if n_tasks is not None:
            declared[name] = (int(n_tasks), "")
        elif task_ids:
            declared[name] = (len(task_ids), "")
        else:
            declared[name] = (
                None,
                f"grid {name!r} declares neither n_tasks nor task_ids -- no cap to check",
            )
    for n in names:
        if n not in found:
            declared[n] = (None, f"grid {n!r} not found under {grids_root}")
    return ks, declared


# --------------------------------------------------------------------------- fork grids
#
# A FORK GRID pits a checkpoint against a baseline on tau2 CONTINUATIONS -- each run resumes a
# reference trajectory at a cut point rather than starting the task fresh. `pi_eval.fork_report`
# is the gold-side protocol for this (pairing, `follow_ups`, the sign test); contract 3 forbids
# importing it here, so everything below is REPLICATED from it, not approximated, the same
# discipline `cad_ge2_by_run`'s docstring states. The one place this module's replica differs
# from the original is forced, not chosen: see `_fork_key`.


def _has_column(con, table: str, column: str) -> bool:
    """Whether the view `table` (already registered on `con` by `_con`) declares `column`.

    `foreign_trace_sha` reached `pi_eval.schema.RUNS` before this extension was written, but
    plenty of parquet dirs predate it too -- every hand-built fixture in `test_train_gate.py`
    among them, since `_build` infers its schema from dict keys and none of them ever set this
    field. `duckdb` refuses to SELECT a column a parquet file does not have, so fork detection
    checks first and treats an absent column as carrying no fork data, never as an error.
    """
    return bool(
        _rows(
            con,
            "SELECT 1 FROM information_schema.columns "
            f"WHERE table_name = {_q(table)} AND column_name = {_q(column)}",
        )
    )


def _select_fork_columns(con, run_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """`foreign_trace_sha`, `n_prefix_user_turns`, `n_user_turns` per run_id.

    All three are `pi_eval.schema.RUNS` columns (nullable; NULL/"" on every ordinary run that
    is not a fork). `{}` when the parquet dir predates the column -- see `_has_column` -- which
    `_fork_criteria` reads as "not a fork grid", never as a missing-table error: an old-shaped
    parquet dir can therefore never be misdetected as carrying forks.

    NEVER `foreign_prefix_k`. That field is on the fork's manifest.json (it is in
    `pinq.ids.SEMANTIC_FIELDS`, which is exactly why it is NOT also duplicated onto
    `runs.parquet` -- see `_fork_key` for what stands in for it here.
    """
    if not run_ids or not _has_column(con, "runs", "foreign_trace_sha"):
        return {}
    rows = _rows(
        con,
        "SELECT run_id, foreign_trace_sha, n_prefix_user_turns, n_user_turns FROM runs "
        f"WHERE run_id IN {_in(run_ids)}",
    )
    return {str(r["run_id"]): r for r in rows}


def _native_reward_by_run(
    con, parquet_dir: Path, run_ids: Sequence[str], *, key: str = "tau_reward"
) -> dict[str, float]:
    """Mean `native.value` per run for `native.key = key`.

    `native.parquet` is POPULATED by `pi compact` (`pi_eval.schema.POPULATED`) but `_con` does
    not wire it up: every OTHER criterion in this module reads runs/turns/scores/ledger/calls
    only, so a parquet dir -- or a fixture -- built before fork_reward existed correctly has no
    native.parquet at all. The view is therefore created HERE, lazily, and its absence is `{}`,
    not `GateInputsMissing`: a fork grid with no native table still gets a `fork_followups`
    verdict, and `fork_reward` reports zero paired rewards rather than refusing the whole gate.
    """
    if not run_ids:
        return {}
    p = Path(parquet_dir) / "native.parquet"
    if not p.is_file():
        return {}
    con.execute(f"CREATE VIEW IF NOT EXISTS native AS SELECT * FROM read_parquet('{p.as_posix()}')")
    rows = _rows(
        con,
        "SELECT run_id, avg(value) AS value FROM native "
        f"WHERE key = {_q(key)} AND run_id IN {_in(run_ids)} GROUP BY 1",
    )
    return {str(r["run_id"]): float(r["value"]) for r in rows if r["value"] is not None}


def follow_ups(run: Mapping[str, Any]) -> int:
    """REPLICATED from `pi_eval.fork_report.follow_ups`, byte for byte: the user turns a
    continuation needed beyond the prefix it was handed."""
    return int(run.get("n_user_turns") or 0) - int(run.get("n_prefix_user_turns") or 0)


def sign_test_p(fewer: int, more: int) -> float:
    """REPLICATED from `pi_eval.fork_report.sign_test_p`, byte for byte: the two-sided exact
    sign test over the informative pairs (ties dropped). NaN when there are none."""
    n = fewer + more
    if n == 0:
        return float("nan")
    k = min(fewer, more)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def _fork_key(row: Mapping[str, Any]) -> tuple:
    """The pairing key. `fork_report.paired_engagement` pairs on `(foreign_trace_sha,
    foreign_prefix_k, seed)`; `foreign_prefix_k` is not a `runs.parquet` column (see
    `_select_fork_columns`), so this substitutes `n_prefix_user_turns` -- the count of user
    turns IN that prefix, which `tau2_runner.run` computes as `count_user_turns(prefix)` and
    which therefore varies with the prefix cut exactly as `foreign_prefix_k` does, for the same
    trace.

    VERIFIED, NOT ASSUMED. On the 204 real runs behind
    `docs/reports/forks_tau2_retail_test.json` (read off `runs/*/manifest.json` and
    `status.json` on 2026-09-16, the same files `scripts/report_forks.py` reads), 2 of the 32
    forked traces are cut at two different `foreign_prefix_k` each (6 and 26; 4 and 18), and
    `(foreign_trace_sha, n_prefix_user_turns)` still separates every one of them: zero
    collisions. Pairing on it reproduces that report's 102 pairs and its pooled diff_mean
    (-4.313725490196078) and fewer/more (71/24) bit for bit --
    `test_fork_paired_engagement_replicates_the_retail_headline` pins this. Where the proxy
    ever DOES collide, `_fork_pairs` refuses rather than picking a side, the same way
    `paired_engagement` refuses two runs of one arm at one key.
    """
    return (row.get("foreign_trace_sha"), row.get("n_prefix_user_turns"), row.get("seed"))


def _fork_pairs(
    rows: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> tuple[dict[tuple, dict[str, Mapping[str, Any]]], int, int]:
    """Group `rows` by `_fork_key`, one row per side per key. REPLICATES
    `pi_eval.fork_report.paired_engagement`'s grouping loop -- including its refusal: two rows
    of the SAME side at the same key raise rather than let one be silently dropped, because
    which one "counts" would be a choice a reader of the verdict cannot see. A row with no
    `foreign_trace_sha` is not a fork; it is counted in `n_not_forks`, never paired.

    Returns `(pairs, n_unpaired, n_not_forks)`; `pairs` holds only keys where both sides landed.
    """
    by_key: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    n_not_forks = 0
    for r in rows:
        if r.get("arm_id") not in (treatment, control):
            continue
        if not r.get("foreign_trace_sha"):
            n_not_forks += 1
            continue
        key = _fork_key(r)
        slot = by_key.setdefault(key, {})
        if r["arm_id"] in slot:
            raise ValueError(
                f"two {r['arm_id']} runs at fork key {key}: "
                f"{slot[r['arm_id']]['run_id']} and {r['run_id']}"
            )
        slot[r["arm_id"]] = r
    pairs = {k: v for k, v in by_key.items() if len(v) == 2}
    n_unpaired = sum(len(v) for v in by_key.values() if len(v) != 2)
    return pairs, n_unpaired, n_not_forks


def fork_paired_engagement(
    rows: Sequence[Mapping[str, Any]], *, treatment: str, control: str
) -> dict[str, Any]:
    """REPLICATED from `pi_eval.fork_report.paired_engagement` (contract 3 forbids importing
    it): the same pooling, the same per-seed breakdown, the same provenance shape. The one
    deliberate difference is the pairing key -- `_fork_key` substitutes `n_prefix_user_turns`
    for `foreign_prefix_k`, which is not a `runs.parquet` column.

    `run_gate`'s own `fork_followups`/`fork_reward` criteria do NOT read this function's
    `pooled` block: they cluster by FORK POINT, averaging seeds within it first (see
    `_fork_criteria`), which is a different aggregation from the flat pool below -- the two
    happen to agree on the retail headline only because every fork point there carries exactly
    the same number of seeds (see `_fork_criteria`'s docstring). This function exists so the
    REPLICATION is itself a named, tested, importable thing, the role `cad_ge2_by_run` plays
    for `cad_ge2`.
    """
    pairs, n_unpaired, n_not_forks = _fork_pairs(rows, treatment=treatment, control=control)
    per_seed: dict[Any, dict[str, Any]] = {}
    pooled_diff: list[float] = []
    codes: dict[str, int] = {}
    ids: list[str] = []
    for seed in sorted({k[2] for k in pairs}, key=lambda x: (x is None, x)):
        ps = [v for k, v in pairs.items() if k[2] == seed]
        t = [follow_ups(v[treatment]) for v in ps]
        c = [follow_ups(v[control]) for v in ps]
        d = [a - b for a, b in zip(t, c)]
        pooled_diff += d
        for v in ps:
            for arm in (treatment, control):
                cv = str(v[arm].get("code_version") or "")[:7]
                codes[cv] = codes.get(cv, 0) + 1
                ids.append(str(v[arm]["run_id"]))
        fewer, more = sum(x < 0 for x in d), sum(x > 0 for x in d)
        per_seed[seed] = {
            "n_pairs": len(ps),
            "follow_ups": {"treatment": statistics.mean(t), "control": statistics.mean(c)},
            "diff_mean": statistics.mean(d),
            "fewer": fewer,
            "more": more,
            "sign_test_p": sign_test_p(fewer, more),
            "reward": {
                "treatment": statistics.mean(
                    float(v[treatment].get("tau_reward") or 0.0) for v in ps
                ),
                "control": statistics.mean(float(v[control].get("tau_reward") or 0.0) for v in ps),
            },
        }
    fewer, more = sum(x < 0 for x in pooled_diff), sum(x > 0 for x in pooled_diff)
    allp = list(pairs.values())
    pooled = {
        "n_pairs": len(allp),
        "diff_mean": statistics.mean(pooled_diff) if pooled_diff else float("nan"),
        "fewer": fewer,
        "more": more,
        "sign_test_p": sign_test_p(fewer, more),
        "reward": {
            "treatment": statistics.mean(float(v[treatment].get("tau_reward") or 0.0) for v in allp)
            if allp
            else float("nan"),
            "control": statistics.mean(float(v[control].get("tau_reward") or 0.0) for v in allp)
            if allp
            else float("nan"),
        },
    }
    return {
        "treatment": treatment,
        "control": control,
        "n_pairs": len(allp),
        "n_unpaired": n_unpaired,
        "n_not_forks": n_not_forks,
        "per_seed": per_seed,
        "pooled": pooled,
        "provenance": {
            "code_versions": codes,
            "seeds": sorted(per_seed, key=lambda x: (x is None, x)),
            "run_ids_sha": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16],
            "n_runs": len(ids),
        },
    }


def _merged_fork_rows(
    rows: Sequence[Mapping[str, Any]], fork_cols: Mapping[str, Mapping[str, Any]], arm_id: str
) -> list[dict[str, Any]]:
    """`_select_runs` rows joined with `_select_fork_columns` by run_id, `arm_id` stamped back
    on: the SELECT that produced `rows` already filtered by arm but does not carry the column,
    and `_fork_pairs` needs it to tell the two sides apart in one combined list."""
    out = []
    for r in rows:
        fc = fork_cols.get(r["run_id"])
        if fc is None:
            continue
        out.append({**r, **fc, "arm_id": arm_id})
    return out


def _fork_criteria(
    con,
    parquet_dir: Path,
    *,
    ckpt: Sequence[Mapping[str, Any]],
    base: Sequence[Mapping[str, Any]],
    checkpoint_arm: str,
    baseline_arm: str,
    bootstrap_seed: int,
    n_resamples: int,
) -> dict[str, Any] | None:
    """`fork_followups` and `fork_reward`, or `None` on a grid that is not a fork grid.

    A FORK GRID is detected when every SELECTED CHECKPOINT run carries a non-empty
    `foreign_trace_sha` (the baseline is not checked: nothing here requires the baseline side
    to be forked too, only that it can be paired against one that is). On a grid that is not a
    fork grid this returns `None` and `run_gate` adds neither criterion, so `criteria` for an
    ordinary grid is untouched byte for byte -- what
    `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` checks.

    THE CI'S UNIT IS A FORK POINT, `(foreign_trace_sha, n_prefix_user_turns)`, NOT A PAIR.
    Seeds are averaged within a fork point before it enters `bca_ci`, exactly the reason
    `_paired_by_task_map` averages seeds within a task first: two seeds of one fork point are
    two measurements of that fork point, not two fork points, and letting them inflate n would
    overstate the CI's precision. `fewer`/`more`/`sign_test_p`, by contrast, are counted over
    PAIRS, matching `pi_eval.fork_report.paired_engagement`'s own units for those numbers.
    """
    ck_fork = _select_fork_columns(con, [r["run_id"] for r in ckpt])
    if not ckpt or not all(ck_fork.get(r["run_id"], {}).get("foreign_trace_sha") for r in ckpt):
        return None
    ba_fork = _select_fork_columns(con, [r["run_id"] for r in base])

    ck_rows = _merged_fork_rows(ckpt, ck_fork, checkpoint_arm)
    ba_rows = _merged_fork_rows(base, ba_fork, baseline_arm)
    reward_by_run = _native_reward_by_run(
        con, parquet_dir, [r["run_id"] for r in ck_rows + ba_rows]
    )
    for r in ck_rows + ba_rows:
        r["tau_reward"] = reward_by_run.get(r["run_id"])

    pairs, n_unpaired, _n_not_forks = _fork_pairs(
        ck_rows + ba_rows, treatment=checkpoint_arm, control=baseline_arm
    )

    def clustered(metric_fn) -> tuple[list[float], list[float]]:
        by_point: dict[tuple, list[float]] = {}
        pair_deltas: list[float] = []
        for key, v in pairs.items():
            d = metric_fn(v[checkpoint_arm]) - metric_fn(v[baseline_arm])
            pair_deltas.append(d)
            by_point.setdefault((key[0], key[1]), []).append(d)
        return pair_deltas, [_mean(vs) for vs in by_point.values()]

    fu_pairs, fu_cluster = clustered(follow_ups)
    fewer = sum(1 for d in fu_pairs if d < 0)
    more = sum(1 for d in fu_pairs if d > 0)
    p = sign_test_p(fewer, more)
    point, lo, hi = bca_ci(fu_cluster, seed=bootstrap_seed, n_resamples=n_resamples)
    columns_common = {
        "pairing_key": "(foreign_trace_sha, n_prefix_user_turns, seed); n_prefix_user_turns "
        "substitutes for foreign_prefix_k, which is not a runs.parquet column -- see _fork_key",
        "unit_of_the_ci": "one fork point (foreign_trace_sha, n_prefix_user_turns), seeds "
        "averaged within it first -- like _paired_by_task_map",
        "unit_of_fewer_more": "one paired (checkpoint run, baseline run) at one fork point and "
        "seed -- like pi_eval.fork_report.paired_engagement",
        "ci": f"fork-point-clustered BCa, {n_resamples} resamples, seed {bootstrap_seed}",
    }
    followups_crit = _criterion(
        value=point,
        threshold="lower CI bound <= 0 (no confidently-measured rise in follow-ups)",
        gated=True,
        passed=bool(fu_cluster) and not math.isnan(lo) and not (lo > 0),
        n=len(fu_cluster),
        reason="" if fu_cluster else "no paired fork runs",
        ci_lo=_nan(lo),
        ci_hi=_nan(hi),
        n_pairs=len(fu_pairs),
        n_unpaired=n_unpaired,
        fewer=fewer,
        more=more,
        sign_test_p=_nan(p),
        columns={
            **columns_common,
            "follow_ups": "n_user_turns - n_prefix_user_turns, replicated from "
            "pi_eval.fork_report.follow_ups (contract 3 forbids importing it)",
        },
    )

    def _reward(row: Mapping[str, Any]) -> float:
        return float(row.get("tau_reward") or 0.0)

    tr_pairs, tr_cluster = clustered(_reward)
    r_point, r_lo, r_hi = bca_ci(tr_cluster, seed=bootstrap_seed, n_resamples=n_resamples)
    reward_crit = _criterion(
        value=r_point,
        threshold="reported, not gated",
        gated=False,
        passed=True,
        n=len(tr_cluster),
        reason="report-only: tau_reward is recorded beside follow-ups and never folded into "
        "the gate, the same way pi_eval.fork_report keeps it beside the sign test rather than "
        "pooling it in"
        if tr_cluster
        else "no paired fork runs carry a native tau_reward row",
        ci_lo=_nan(r_lo),
        ci_hi=_nan(r_hi),
        n_pairs=len(tr_pairs),
        # The SAME fewer/more as fork_followups, not a reward-sign count: a reader of the
        # reward delta should not have to cross-reference fork_followups to see whether it was
        # bought with more turns or fewer. fork_report.py never defines a reward-specific
        # fewer/more, so there is nothing else these could replicate.
        fewer=fewer,
        more=more,
        sign_test_p=_nan(p),
        columns={
            **columns_common,
            "metric": "native.parquet, key='tau_reward', per run; missing -> 0.0, replicated "
            "from paired_engagement's 'or 0.0'",
            "fewer_more": "borrowed from fork_followups: how many of the same paired fork "
            "points needed fewer/more follow-ups, not a reward-sign count",
        },
    )
    return {"fork_followups": followups_crit, "fork_reward": reward_crit}


# --------------------------------------------------------------------------- the gate


def run_gate(
    *,
    parquet_dir: Path,
    grid_name: str,
    baseline_grid_names: Sequence[str],
    checkpoint_arm: str = "inquirer_trained",
    baseline_arm: str = "inquirer_prompted",
    baseline_model_id: str | None = None,
    checkpoint_model_id: str | None = None,
    length_margin: float = 0.10,
    distinct_floor: float = 0.65,
    scorer_hash: str | None = None,
    bootstrap_seed: int = 0,
    n_resamples: int = 1000,
    grids_root: Path | None = None,
    coverage_rule: str = "matched_cost",
) -> dict[str, Any]:
    """Every criterion, its verdict, and the rows it was computed over."""
    from pinq_train.rung1_sft.collapse import within_task_distinct_n, within_task_seed_distinct_n

    if coverage_rule not in COVERAGE_RULES:
        raise ValueError(
            f"coverage_rule {coverage_rule!r} is not one of {COVERAGE_RULES}. Refused rather "
            "than defaulted: a verdict whose `coverage_rule` field and whose number disagree "
            "is worse than no verdict."
        )

    parquet_dir = Path(parquet_dir)
    con = _con(parquet_dir)

    # `checkpoint_model_id` is `--baseline-model-id`'s mirror image: a shared isolated store
    # that compacts several checkpoints under one arm_id + grid_name (exactly what serving them
    # together on one vLLM job and running one `pi compact` produces) needs the same
    # calls.parquet disambiguation on THIS side too, or every checkpoint sharing that
    # arm/grid is pooled into one average -- the per-seed blending the seed columns exist to
    # catch. See `test_the_checkpoint_model_id_is_resolved_through_calls_parquet`.
    ckpt = _select_runs(con, arm=checkpoint_arm, grids=[grid_name], model_id=checkpoint_model_id)
    base = _select_runs(
        con, arm=baseline_arm, grids=list(baseline_grid_names), model_id=baseline_model_id
    )

    # ---- EmptyArm (T19d). Checked ONCE, here, before any criterion runs: a criterion whose
    # inputs are absent fails on its own terms (the module docstring's "AN ABSENT CRITERION
    # FAILS"), but a whole ARM with zero rows is not one criterion's problem, it is every
    # criterion's, and letting each discover it separately writes a JSON that reads like twelve
    # independent measurements instead of the one true fact -- there was nothing to measure. See
    # `EmptyArm`'s docstring for the incident this refuses.
    if not ckpt or not base:
        empty_side, empty_arm, empty_grids, other = (
            ("checkpoint", checkpoint_arm, [grid_name], base)
            if not ckpt
            else ("baseline", baseline_arm, list(baseline_grid_names), ckpt)
        )
        raise EmptyArm(
            side=empty_side,
            arm=empty_arm,
            grid_names=empty_grids,
            parquet_dir=parquet_dir,
            suites=sorted({r["suite_id"] for r in other}),
        )

    ck_keys, ba_keys = _by_key(ckpt), _by_key(base)
    shared = set(ck_keys) & set(ba_keys)

    # ---- IncompleteArm. Checked once, here, right after EmptyArm and before any criterion
    # runs, for the same reason: a partial arm is every criterion's problem, not one's, and a
    # partial arm still produces a value for each of them, so nothing downstream can tell a
    # complete gate from one that quietly measured only its paired subset. Skipped for a fork
    # grid -- `_fork_pairs` (below) already refuses ITS shape of this, keyed on
    # `(foreign_trace_sha, n_prefix_user_turns, seed)` rather than `(suite_id, task_id, seed)`,
    # and a task-id-keyed check here would never agree with a fork run's identity anyway (every
    # fork shares the trace's original task_id). Detected with the same predicate
    # `_fork_criteria` uses below: every selected checkpoint run carries a `foreign_trace_sha`.
    #
    # Correction, 2026-09-17: this used to raise on `ck_key_set != ba_key_set` -- SET EQUALITY,
    # either side missing a key the other has. Re-measured at 2026-09-17T16:51:10+0300 against
    # every real (non-symlink) *.json under artifacts/gate, recursive and top-level -- two
    # subtrees (8b1-rescored-t19c/, 8b2-t19c/) are entirely symlink farms, .json entries
    # included, so a walk that mis-globbed or double-counted them would misstate both the count
    # and which side moved: an earlier pass reported 48 of 156, and both numbers were wrong
    # (denominator inflated by symlinked duplicates and short of loose top-level files;
    # numerator read a counter only later verdicts carry, not the pairing block every verdict
    # carries). There are 150 real verdicts, every one with a pairing block: all 150 have a
    # baseline arm with strictly MORE keys than the checkpoint (one instance, 132 paired, 132
    # unpaired-baseline, 0 unpaired-checkpoint -- the scale ladder's smallest-model rung runs
    # every checkpoint task at one extra seed on the baseline side only), and zero have a
    # checkpoint key the baseline lacks. That is not a partial arm -- every checkpoint run has
    # its baseline partner, and the intersection-pairing plus the pairing counts below already
    # handle it correctly; nothing is dropped from the checkpoint's side. Baselines are
    # deliberately run at more seeds than the checkpoints compared against them; pairing
    # intersects on the shared keys and records what it set aside -- that is the design
    # working, not a defect this corpus happens to tolerate. The two directions are not the
    # same hazard: a checkpoint key the baseline lacks means a checkpoint RUN that would be
    # silently dropped from every criterion -- always an error, refused unconditionally below,
    # a hazard this check stays armed for even though it has not yet occurred once in 150 real
    # verdicts. A baseline key the checkpoint lacks means only that the baseline was run at
    # more seeds or tasks than the checkpoint needed -- normal, and left alone. So the check is
    # now directional: refuse when the checkpoint's key set is NOT a subset of the baseline's;
    # the reverse containment never raises.
    #
    # This subset rule alone does not catch every partial arm: a checkpoint staged on 40 of a
    # baseline's 50 tasks is still a subset of the 50 and would pass. Catching THAT shape needs
    # the grid's OWN declared task set (`selection.grid_name`), not the other arm's -- and for
    # exactly one property of that set, the COUNT, it is now reachable and checked: see
    # `UnderfilledArm`, right after `UnscoredArm` below. `--grids-root` is wired (this argument,
    # and `pi_run.cmd_train.cmd_train_gate`'s real call site now passes it), and the existing
    # `_k_by_suite` grid-reading helper is no longer dead code in production -- it reads
    # `n_tasks` off the same parsed YAML it already reads `k` from, at no new cost. The one
    # thing still unreachable from here is the missing tasks' IDENTITY, not their count: turning
    # a count into literal ids needs `Grid.select(all_ids, split_of)` (src/pi_run/grids.py),
    # which needs the suite's corpus-ordered id list (`data/corpora`), a dependency this module
    # does not have and should not assume, since the compact step exists specifically to
    # decouple gate-time analysis from the raw corpus -- reading grid FILES as data is not the
    # same as importing `pi_run` to resolve ids through it, and no `import-linter` contract
    # forbids the former (none of the four in pyproject.toml names the `pinq_train -> pi_run`
    # direction at all; they forbid the reverse and anything reaching `pi_eval`), so that edge is
    # unused rather than disallowed, and moot regardless because this path never creates it. The
    # original incident (below, and in `UnderfilledArm`'s docstring) was a shortfall in HOW MANY
    # tasks ran, not a question of WHICH, so the count alone covers it. Investigated and accepted
    # (coordinator, 2026-09-17), decided further the same day: the identity gap is real and
    # stays -- see `UnderfilledArm` -- but the count this paragraph used to disclaim is now
    # checked exactly, and a grid declaring no cap at all (neither `n_tasks` nor `task_ids`) is
    # the one honest residual left, documented, not closed, where `UnderfilledArm` is.
    ck_fork_cols = _select_fork_columns(con, [r["run_id"] for r in ckpt])
    is_fork_grid = bool(ckpt) and all(
        ck_fork_cols.get(r["run_id"], {}).get("foreign_trace_sha") for r in ckpt
    )
    ck_key_set, ba_key_set = set(ck_keys), set(ba_keys)
    if not is_fork_grid and not ck_key_set.issubset(ba_key_set):
        only_ck = sorted({k[1] for k in ck_key_set - ba_key_set})
        only_ba = sorted({k[1] for k in ba_key_set - ck_key_set})
        raise IncompleteArm(
            checkpoint_arm=checkpoint_arm,
            baseline_arm=baseline_arm,
            n_checkpoint=len(ck_key_set),
            n_baseline=len(ba_key_set),
            n_paired=len(shared),
            only_in_checkpoint=only_ck,
            only_in_baseline=only_ba,
        )

    # ---- UnscoredArm. Checked once, here, right after IncompleteArm and before any criterion
    # runs, for the same reason: an arm with zero score rows is upstream of every criterion, not
    # one of them, and a criterion given zero score rows still returns a value -- a mean of
    # nothing, or a baseline-only read -- rather than an error, so the printed verdict reads as
    # the checkpoint failing rather than as the checkpoint never having been scored. See
    # `UnscoredArm`'s docstring for the near-miss this refuses.
    ck_ids, ba_ids = [r["run_id"] for r in ckpt], [r["run_id"] for r in base]
    ck_score_n = _rows(con, f"SELECT count(*) AS n FROM scores WHERE run_id IN {_in(ck_ids)}")[0][
        "n"
    ]
    ba_score_n = _rows(con, f"SELECT count(*) AS n FROM scores WHERE run_id IN {_in(ba_ids)}")[0][
        "n"
    ]
    if ck_score_n == 0 or ba_score_n == 0:
        empty_side, empty_arm, n_runs, n_score = (
            ("checkpoint", checkpoint_arm, len(ck_ids), ck_score_n)
            if ck_score_n == 0
            else ("baseline", baseline_arm, len(ba_ids), ba_score_n)
        )
        raise UnscoredArm(side=empty_side, arm=empty_arm, n_runs=n_runs, n_score_rows=n_score)

    # ---- UnderfilledArm. Checked once, here, right after UnscoredArm and before any
    # criterion runs, for the same reason as the other three: a checkpoint short of its own
    # grid's declared tasks is every criterion's problem, not one's, and (unlike EmptyArm,
    # IncompleteArm and UnscoredArm) the subset rule above cannot see it at all -- 40 of a
    # 50-task grid, all 40 present in a baseline that covers 50, is a perfectly good subset.
    #
    # This closes the disclosed gap the comment above `IncompleteArm`'s check and
    # `test_the_subset_rule_alone_does_not_catch_a_checkpoint_still_short_of_its_own_grids_tasks`
    # used to describe, but only ITS shape: the original 2026-09-17 incident was a shortfall in
    # HOW MANY tasks ran, not a question of WHICH, and the paired count is exactly what "how
    # many" means once IncompleteArm has already guaranteed every checkpoint key has a baseline
    # partner (so `shared`, restricted to this grid's suites, equals the checkpoint's own key
    # set here -- not a coincidence, a consequence of the check above). What is counted within
    # that restriction is DISTINCT task ids, not raw key volume: a grid can declare more than
    # one seed (`Grid.seeds`), so the same task contributes one `shared` key per seed, and
    # `n_tasks` is a task count, never a task-times-seed count -- counting raw keys would
    # inflate the paired count by a factor of `len(seeds)` and silence this guard exactly where
    # it should fire hardest. Comparing the per-suite distinct-task count to the grid's OWN
    # declared `n_tasks`/`task_ids` needs nothing this module does not already read:
    # `_k_by_suite` below parses conf/grids as plain data with no import of `pi_run` -- that
    # direction is unused rather than disallowed (none of the four `import-linter` contracts in
    # pyproject.toml names it; they forbid the reverse direction and anything reaching
    # pi_eval), and this path never creates the edge regardless, since it is a YAML read, not a
    # Python import.
    #
    # Requires `--grids-root` (dead until now: `pi train gate` had no flag for it and
    # `cmd_train_gate` never passed one). Skipped, like `IncompleteArm`, for a fork grid --
    # fork runs are keyed on `(foreign_trace_sha, n_prefix_user_turns, seed)`, not
    # `(suite_id, task_id, seed)`, so a task COUNT means something different there and this
    # module does not currently gate it.
    #
    # WHAT THIS STILL CANNOT DO: name the missing tasks, or catch a checkpoint short of tasks
    # the baseline is ALSO short of (both arms only see the count they were each given; nothing
    # here or in IncompleteArm ever asks the corpus how many tasks a suite has, only how many a
    # grid FILE declares). Both need `Grid.select`'s corpus-ordered id list (`data/corpora`), a
    # dependency this module does not have and should not assume -- unchanged from the reasons
    # above `IncompleteArm`'s check. The identity of a missing task stays unreachable; whether
    # there were enough of them no longer does. That is exactly the incident this refuses: it
    # was a shortfall in count, and the count is now checked exactly.
    #
    # ONE HONEST RESIDUAL: a grid that selects by `split` alone with no `n_tasks` or `task_ids`
    # cap declares no count to check against, and this guard stamps that `None` with a reason
    # rather than guessing one -- it does not raise, and does not claim to have checked
    # anything. No grid under conf/grids/ has this shape today (all 32 checked 2026-09-17
    # declare one or the other); if one ever does, the null-stamp-with-a-reason discipline
    # already in force here is the remedy, not a new one to design.
    ks, declared_n_tasks = _k_by_suite(grids_root, [grid_name, *baseline_grid_names])
    task_count: dict[str, Any] = {
        "grid_name": grid_name,
        "declared": None,
        "reason": "",
        "by_suite": {},
    }
    if is_fork_grid:
        task_count["reason"] = "skipped for a fork grid -- task count means something else there"
    else:
        declared_n, declared_reason = declared_n_tasks.get(grid_name, (None, "not requested"))
        task_count["declared"] = declared_n
        task_count["reason"] = declared_reason
        if declared_n is not None:
            by_suite = {s: len({k[1] for k in shared if k[0] == s}) for s in ks.get(grid_name, {})}
            task_count["by_suite"] = by_suite
            shortfalls = [(s, n, declared_n) for s, n in by_suite.items() if n < declared_n]
            if shortfalls:
                raise UnderfilledArm(
                    checkpoint_arm=checkpoint_arm, grid_name=grid_name, shortfalls=shortfalls
                )

    selected_ids = [r["run_id"] for r in ckpt] + [r["run_id"] for r in base]

    if scorer_hash is None:
        rows = _rows(
            con,
            "SELECT scorer_hash, count(*) n FROM scores "
            f"WHERE run_id IN {_in(selected_ids)} "
            "GROUP BY 1 ORDER BY n DESC, scorer_hash",
        )
        scorer_hash = str(rows[0]["scorer_hash"]) if rows else ""

    # ---- graph_version: the third element of the provenance triple (CONTRIBUTING.md rule 1). Read
    # from the SAME rows `scorer_hash` is read from and filtered to that hash, because what the
    # verdict must name is the graph THIS scoring ran against -- a parquet holding two
    # generations would otherwise report the newer hash beside the older graph. Where the
    # selected rows disagree the verdict cannot name one graph at all, so this refuses rather
    # than picking (`GraphVersionAmbiguous`); `_metric_by_run` and every criterion below
    # already filter on `scorer_hash`, so this adds no row to what the numbers are computed
    # over. Empty ("") on a parquet with no scores for these runs, matching `scorer_hash`.
    graph_rows = _rows(
        con,
        "SELECT DISTINCT graph_version FROM scores "
        f"WHERE run_id IN {_in(selected_ids)} AND scorer_hash = {_q(scorer_hash)} "
        "ORDER BY 1",
    )
    graph_versions = [str(r["graph_version"]) for r in graph_rows]
    if len(graph_versions) > 1:
        raise GraphVersionAmbiguous(
            versions=graph_versions, parquet_dir=parquet_dir, n_runs=len(selected_ids)
        )
    graph_version = graph_versions[0] if graph_versions else ""

    criteria: dict[str, Any] = {}

    # ---- malformed: a ledger currency, per OK run. `criteria["malformed"]` stays byte for
    # byte what it always was (see MALFORMED_TOLERANCE's docstring); only the `passed` rule
    # moves, and the detail lives beside `criteria` in `malformed_tolerance`.
    ck_mal, n_ck, ck_events = _malformed(con, runs=ckpt, label="checkpoint")
    ba_mal, _, _ = _malformed(con, runs=base, label="baseline")
    mal_bound = max(ba_mal, MALFORMED_TOLERANCE)
    ok = (not math.isnan(ck_mal)) and (not math.isnan(ba_mal)) and ck_mal <= mal_bound
    criteria["malformed"] = _criterion(
        value=ck_mal,
        baseline=ba_mal,
        threshold=f"<= max(baseline, {MALFORMED_TOLERANCE})",
        gated=True,
        passed=bool(ok),
        n=n_ck,
        reason="" if n_ck else "no ok runs for the checkpoint arm",
    )
    malformed_tolerance = {
        "tolerance": MALFORMED_TOLERANCE,
        "n_events": ck_events,
        "n_runs": n_ck,
        "baseline_rate": _nan(ba_mal),
        "rate": _nan(ck_mal),
    }

    # ---- within-task distinct-3 on the checkpoint's questions
    pairs = _questions(con, ckpt)
    d3 = within_task_distinct_n(pairs) if pairs else float("nan")
    # `by_seed`/`n_seeds`: an evaluation run at more than one seed pools every seed of a task
    # into `value` above, so a checkpoint that repeats its own (fully state-dependent)
    # questions across seeds is scored as if it had collapsed -- see
    # `within_task_seed_distinct_n`'s docstring and
    # `test_distinct3_by_seed_is_not_halved_by_a_second_seed_that_repeats_the_first`.
    seed_triples = _questions_with_seed(con, ckpt)
    d3_by_seed = within_task_seed_distinct_n(seed_triples) if seed_triples else float("nan")
    n_seeds = len({seed for _, seed, _ in seed_triples})
    # `passed` GATES ON `d3_by_seed`, not on `d3`. Measured on `artifacts/seedrep_gate_20260919/
    # distinct3.json`: all 9 trained cells (s0/s1/s2 x musique/strategyqa/wiki2) failed this
    # floor on the seed-pooled `value` while every one of them passed on `by_seed`
    # (0.804-0.929) -- a two-eval-seed run pools two near-identical (state-dependent, at
    # temperature 0) rollouts of a task into one group, which measures how many seeds were
    # evaluated rather than whether the policy collapsed (see
    # `test_distinct3_passes_on_by_seed_when_only_the_seed_pooled_value_fails_the_floor`, and
    # its non-vacuity twin `test_distinct3_fails_when_the_policy_asks_one_question_forever`,
    # which stays False under this same rule because a real collapse fails both groupings).
    # `value` keeps the exact bytes it always had -- unchanged by this fix, see
    # `test_distinct3_by_seed_is_not_halved_by_a_second_seed_that_repeats_the_first` -- so a
    # verdict's reported LEVEL never moves, only which bit decides pass/fail. Falls back to
    # `d3` when `by_seed` is unmeasurable (no seed recorded), which cannot happen while `d3`
    # itself is measurable: both come from the same `pairs`/`seed_triples` presence check.
    gating_value = d3_by_seed if not math.isnan(d3_by_seed) else d3
    criteria["distinct3"] = _criterion(
        value=d3,
        baseline=_nan(within_task_distinct_n(_questions(con, base))) if base else None,
        threshold=distinct_floor,
        gated=True,
        passed=bool((not math.isnan(gating_value)) and gating_value >= distinct_floor),
        n=len({t for t, _ in pairs}),
        reason="" if pairs else "no ASK turns recorded for the checkpoint arm",
        measure="within_task_distinct_n, GATED ON by_seed (NOT the pooled distinct-3: the "
        "pooled floor of 0.55 is unreachable on musique, whose own task questions score "
        "0.4685)",
        by_seed=_nan(d3_by_seed),
        n_seeds=n_seeds,
    )

    # ---- question length. Two-sided TOST at cap 8; one-sided non-inferiority at matched cost.
    length_rule = "not_longer" if coverage_rule == "matched_cost" else "tost"
    criteria["length_equivalence"] = _length(
        con,
        ckpt,
        base,
        margin=length_margin,
        seed=bootstrap_seed,
        n_resamples=n_resamples,
        rule=length_rule,
    )

    # ---- the STOP 2x2
    ck_runs, ba_runs = _with_stop(con, ckpt), _with_stop(con, base)
    ck_stop = _stop_2x2(con, ck_runs, _coverage_ladder(con, ck_runs, scorer_hash=scorer_hash))
    ba_stop = _stop_2x2(con, ba_runs, _coverage_ladder(con, ba_runs, scorer_hash=scorer_hash))
    cells_ok = all(
        (not math.isnan(ck_stop[c])) and (not math.isnan(ba_stop[c])) and ck_stop[c] >= ba_stop[c]
        for c in ("p_stop_given_done", "p_ask_given_not_done")
    )
    criteria["stop_2x2"] = _criterion(
        value=ck_stop,
        baseline=ba_stop,
        threshold="both cells >= baseline",
        gated=True,
        passed=bool(cells_ok),
        n=ck_stop["n_done"] + ck_stop["n_not_done"],
        reason=""
        if ck_stop["n_done"] and ck_stop["n_not_done"]
        else "one cell has no runs, so it cannot be compared",
        columns={
            "unit": "one decision point (run, t), t = 0..n_asks",
            "stop_axis": "ASK at every t < runs.n_asks; runs.stop_reason at t = runs.n_asks",
            "stop_means": "policy_stop = a STOP decision; budget/max_turns = the harness "
            "halted the episode and the policy was never consulted at that state, so it is "
            "excluded and counted as n_forced_stops",
            "done_axis": "frontier_q#t >= 1-1e-12 before the decision at t",
            "done_means": "scores.metric_name='frontier_q#t' is required-evidence coverage "
            "after t asks, i.e. the coverage held when the decision at t was made; the "
            "threshold is pi_run.cmd_train._done's, so the online and offline 2x2 agree",
            "not_stop_undershoot": "stop_undershoot is max(0, k* - k_hat) over a MONOTONE "
            "ladder whose k_hat is its last index, so it is identically 0 and its not-done "
            "cell was empty by construction",
            "filter": "runs.status='ok'",
            "why_two_tables": "runs.parquet carries no coverage column",
        },
    )

    # ---- paired deltas
    cad = cad_ge2_by_run(parquet_dir, scorer_hash=scorer_hash)
    for metric, rule, gated in PAIRED_METRICS:
        by_run = (
            cad if metric == "cad_ge2" else _metric_by_run(con, metric, scorer_hash=scorer_hash)
        )
        deltas = _paired_by_task(ck_keys, ba_keys, by_run)
        lo, hi = bootstrap_ci(
            deltas, seed=bootstrap_seed, n_resamples=n_resamples, lo=0.025, hi=0.975
        )
        delta = _mean(deltas)
        if not deltas:
            passed, reason = False, f"no paired runs carry {metric}"
        elif rule == "gain":
            passed, reason = bool(delta > 0 and lo > 0), ""
        elif rule == "no_loss":
            passed, reason = bool(hi >= 0), ""
        else:
            passed, reason = True, "reported, not gated: I.9 Tier B does not name this endpoint"
        criteria[metric] = _criterion(
            value=delta,
            threshold={"gain": "delta > 0, CI excludes 0", "no_loss": "CI upper bound >= 0"}.get(
                rule, "reported"
            ),
            gated=gated,
            passed=passed,
            n=len(deltas),
            reason=reason,
            ci_lo=_nan(lo),
            ci_hi=_nan(hi),
        )

    # ---- the coverage rule (D9). `matched_cost` REPLACES the cap-8 coverage criterion computed
    # just above with the contrast against the baseline's own prefix at the same k, and keeps
    # the cap-8 delta in `matched_cost.by_suite[*].cap8_coverage_delta` as the cost line. Every
    # other criterion, `cad_ge2` included, is the cap-8 contrast under both rules: `scores` has
    # no cad ladder -- `cad#d` is terminal only -- so a matched-k C@d>=2 cannot be read from
    # parquet at all, and `scripts/matched_cost.py` re-runs the matcher to get one.
    mc: dict[str, Any] | None = None
    if coverage_rule == "matched_cost":
        mc = _matched_cost(
            con,
            ckpt=ckpt,
            base=base,
            ck_keys=ck_keys,
            ba_keys=ba_keys,
            scorer_hash=scorer_hash,
            seed=bootstrap_seed,
            n_resamples=n_resamples,
        )
        deltas = mc.pop("deltas")
        lo, hi = mc["pooled"]["ci_lo"], mc["pooled"]["ci_hi"]
        delta = mc["pooled"]["delta"]
        cap8_value = criteria["evidence_coverage"]["value"]
        # REFUSAL, 2026-09-19 (coordinator's convention: keep the existing arithmetic exactly
        # as it is -- so it stays byte-identical for the suites it was always safe for -- and
        # make a MISREADING impossible rather than merely documented, by refusing the pooled
        # pass/fail outright when it would be pooling in an unmatched suite). A suite is unsafe
        # when `by_suite[suite].outspent_comparator` is True; see `_matched_cost`'s columns
        # entry for what that means and `symmetric` for the reading to use instead.
        outspent_suites = sorted(
            s for s, cell in mc["by_suite"].items() if cell.get("outspent_comparator")
        )
        if outspent_suites:
            criteria["evidence_coverage"] = _criterion(
                value=delta,
                threshold="delta > 0, CI excludes 0",
                gated=True,
                passed=None,
                n=len(deltas),
                reason=(
                    f"REFUSED: not matched under the default rule on {outspent_suites} -- the "
                    "checkpoint's own mean asking rate meets or exceeds the baseline's there, "
                    "so the pooled delta above would include at least one suite where the "
                    "checkpoint's raw terminal value was compared against the baseline's own "
                    "(shorter) terminal rather than a shared matched budget. Read "
                    "matched_cost.by_suite[<suite>].symmetric for a budget-matched number on "
                    "the affected suite(s); unaffected suites are unchanged."
                ),
                ci_lo=lo,
                ci_hi=hi,
                comparator="matched_k",
                ci_method=f"task-clustered BCa, {n_resamples} resamples, seed {bootstrap_seed}",
                cap8_value=cap8_value,
                refused=True,
                outspent_suites=outspent_suites,
            )
        else:
            criteria["evidence_coverage"] = _criterion(
                value=delta,
                threshold="delta > 0, CI excludes 0",
                gated=True,
                passed=bool(
                    deltas and delta is not None and delta > 0 and lo is not None and lo > 0
                ),
                n=len(deltas),
                reason="" if deltas else "no paired runs carry evidence_coverage",
                ci_lo=lo,
                ci_hi=hi,
                comparator="matched_k",
                ci_method=f"task-clustered BCa, {n_resamples} resamples, seed {bootstrap_seed}",
                cap8_value=cap8_value,
            )

        # ---- facet_breadth becomes REPORT-ONLY under this rule (D9 follow-up, 2026-09-15).
        #
        # It is the same confound the coverage rule removes, one metric over: facet breadth
        # scales with the NUMBER OF ASKS, so an arm that asks three questions where the base
        # asks eight loses it for the reason it was trained to. Measured on the 4B trio at cap
        # 8: -0.075, -0.078, -0.091, -0.091, -0.096, -0.121 -- six of six failing no-loss while
        # every one of them is at or ahead of the base per question asked.
        #
        # AND IT CANNOT BE MOVED TO MATCHED k. `scores.parquet` carries `facet_breadth` as ONE
        # terminal metric per run (1,395 rows, 1 distinct metric_name) and no `facet_breadth#k`
        # ladder; only `frontier_q` is indexed by prefix. Recomputing it at k means re-running
        # the facet assignment against gold, which contract 3 forbids here.
        #
        # SO IT IS REPORTED, NOT DROPPED: value, interval, n and the no-loss outcome it would
        # have had all stay in the JSON, `gated` is false, and `facet_rule` says which rule was
        # in force. This is TIER B -- checkpoint SELECTION -- and gating it here would reject
        # every trained checkpoint for asking less. The Tier-C preregistered endpoint is
        # untouched: it remains a reported secondary at cap 8, and nothing sealed moves.
        fb = dict(criteria["facet_breadth"])
        criteria["facet_breadth"] = {
            **fb,
            "gated": False,
            "passed": True,
            "no_loss": fb["passed"],
            "reason": "report_only under --coverage-rule matched_cost: facet_breadth scales "
            "with the number of asks, like the cap-8 coverage delta this rule replaces, and "
            "scores.parquet carries no facet_breadth#k ladder to read it at matched k. Value, "
            "CI and n are reported; `no_loss` is the verdict the cap-8 rule would have given.",
        }

        # ---- cad_ge2 becomes REPORT-ONLY too (D9 follow-up, 2026-09-15). Same confound, same
        # obstruction, and now with the matched-k measurement to say which of the two it is.
        #
        # THE CONFOUND. C@d>=2 is a recall over depth>=2 nodes, so it rises with the number of
        # asks exactly as cap-8 coverage and facet breadth do. Measured on all twelve verdicts
        # at cap 8: -0.035, -0.059, -0.065, -0.082, -0.088, -0.091, -0.097, -0.113, -0.113,
        # -0.124, -0.177 (and one at -0.113) -- every checkpoint losing, while asking a quarter
        # to a half as many questions as the base.
        #
        # AND THE MATCHED-k MEASUREMENT SAYS THE LOSS IS THE CONFOUND. `scripts/matched_cost.py`
        # re-runs `MechanicalMatcher` on every trajectory PREFIX against gold and contrasts the
        # trained arm at its own stop with the base's prefix at the same k. All six 8B intervals
        # SPAN 0 -- musique +0.0294 [-0.0765, +0.1294], -0.0118 [-0.1118, +0.0882], +0.0529
        # [-0.0474, +0.1471]; strategyqa +0.0376 [-0.0376, +0.1075], +0.0591 [-0.0235, +0.1342],
        # +0.0054 [-0.0806, +0.0806] (figures/P13_matched_cost/provenance.json, reproduced to
        # full float equality from the committed inputs). There is no depth loss at equal cost.
        #
        # AND IT CANNOT BE MOVED TO MATCHED k HERE. `scores.parquet` carries `cad#d`/`cad_n#d`
        # as TERMINAL per-depth rows -- 4 depths, one value each per run -- and no ladder over
        # prefixes; only `frontier_q` is indexed by k. Rebuilding C@d>=2 at a prefix needs the
        # node -> gold_depth map, which is gold and which contract 3 forbids this module. It is
        # not recoverable from the compacted tables either: `matches.parquet` gives the turn a
        # node resolved, but the per-depth rows are AGGREGATES, and on 114 of the 465 dev tasks
        # more than one node -> depth map reproduces every stored `cad#d` row. The resulting
        # spread reaches 0.48 of the metric's range at k=1, so a gate that picked one map would
        # be reporting its own arbitrary choice.
        #
        # SO IT IS REPORTED, NOT DROPPED, in the shape `facet_breadth` already uses: the value,
        # its interval, n, and `no_loss` -- the verdict D9's follow-up named -- all stay, with
        # `cap8_cad_ge2_delta` saying what the number is denominated in and `cad_rule` saying
        # which rule was in force. The Tier-C preregistered endpoint is untouched.
        cg = dict(criteria["cad_ge2"])
        criteria["cad_ge2"] = {
            **cg,
            "gated": False,
            "passed": True,
            "no_loss": bool(cg["ci_hi"] is not None and cg["ci_hi"] >= 0),
            "cap8_cad_ge2_delta": cg["value"],
            "comparator": "cap8",
            "reason": "report_only under --coverage-rule matched_cost: cad_ge2 scales with the "
            "number of asks, like the cap-8 coverage delta this rule replaces, and scores "
            "carries cad#d as a terminal per-depth row with no ladder over prefixes, so it "
            "cannot be read at matched k without the gold node->depth map contract 3 forbids "
            "here. The matched-k contrast is scripts/matched_cost.py / P13, where all six 8B "
            "intervals span 0. `no_loss` is the verdict the cap-8 no-loss rule would give.",
            "matched_k_lives_in": "scripts/matched_cost.py (gold side); "
            "figures/P13_matched_cost/provenance.json",
        }

        # ---- the STOP 2x2 becomes REPORT-ONLY (D9 follow-up, 2026-09-15).
        #
        # THE PROMPTED BASE IS A DEGENERATE COMPARATOR ON THIS AXIS, and the rule "both cells >=
        # the prompted base" inherits the degeneracy. Measured on the six gate parquets:
        # P(STOP | done) is 0.1525 (8B musique), 0.1460 (8B strategyqa), 0.0066 and 0.0081 at
        # 4B; P(ASK | not done) is 0.9642, 0.9633, 0.9985, 0.9994. The second cell is ~1
        # BECAUSE the first is ~0: a policy that essentially never stops asks at every state,
        # done or not. So the rule asks a real stopper to keep asking as often as a policy that
        # cannot stop, which no stopper can do -- all twelve verdicts failed it, every one of
        # them on the second cell, while every checkpoint beat the base on the first (0.88-1.00
        # against 0.007-0.15).
        #
        # STOPPING IS STILL SELECTED, OUTSIDE THE GATE, by the N1 rule: P(ASK | not done) must
        # rise by >= 0.03 against the REFERENCE CHECKPOINT (a comparator that can stop) AND
        # P(STOP | done) must be >= 0.85. That contrast is between two trained policies, so it
        # is not degenerate, and it is not this command's to make.
        #
        # SO EVERY CELL IS KEPT: both rates, all four counts, n_states, n_runs, n_forced_stops,
        # n_skipped_no_coverage, mean_asks_after_done and the whole `columns` block are
        # untouched; only `gated` moves, with `both_cells_ge_baseline` holding the verdict the
        # cap-8 rule would have given and `stop_rule` saying which rule was in force.
        st = dict(criteria["stop_2x2"])
        criteria["stop_2x2"] = {
            **st,
            "gated": False,
            "passed": True,
            "both_cells_ge_baseline": st["passed"],
            "reason": "report_only under --coverage-rule matched_cost: the prompted base almost "
            "never stops, so its P(ASK | not done) is ~1 by construction and 'both cells >= the "
            "base' is unreachable by any policy that stops at all. Every cell is reported; "
            "`both_cells_ge_baseline` is the verdict the cap-8 rule would have given. Stopping "
            "is selected by the N1 rule against the reference checkpoint, outside this gate.",
            "n1_rule": "P(ASK | not done) rise >= 0.03 vs the reference checkpoint AND "
            "P(STOP | done) >= 0.85",
        }

    # ---- fork grids (tau2 continuations). Absent entirely -- not merely empty -- on a grid
    # that is not a fork grid, so `criteria` there is untouched byte for byte. See
    # `_fork_criteria`.
    fork = _fork_criteria(
        con,
        parquet_dir,
        ckpt=ckpt,
        base=base,
        checkpoint_arm=checkpoint_arm,
        baseline_arm=baseline_arm,
        bootstrap_seed=bootstrap_seed,
        n_resamples=n_resamples,
    )
    if fork is not None:
        criteria["fork_followups"] = fork["fork_followups"]
        criteria["fork_reward"] = fork["fork_reward"]

    matched = coverage_rule == "matched_cost"
    stop_rule = "report_only" if matched else "both_cells_ge_baseline"

    suites = sorted({k[0] for k in shared})
    k_stamp: dict[str, Any] = {}
    for s in suites:
        ck_k = ks.get(grid_name, {}).get(s)
        ba_k = next((ks[g][s] for g in baseline_grid_names if s in ks.get(g, {})), None)
        k_stamp[s] = None if ck_k is None or ba_k is None else (ck_k == ba_k)

    out: dict[str, Any] = {
        "passed": all(c["passed"] for c in criteria.values() if c["gated"]),
        "coverage_rule": coverage_rule,
        # The four rules `matched_cost` moves, each named where a reader of the verdict looks
        # first. Under `cap8` these read back the rules that have always been in force.
        "facet_rule": "report_only" if matched else "no_loss",
        "cad_rule": "report_only" if matched else "gain",
        "length_rule": length_rule,
        "stop_rule": stop_rule,
        "criteria": criteria,
        "malformed_tolerance": malformed_tolerance,
        # NOT nested inside `criteria`: `test_the_cap8_rule_reproduces_the_previous_verdict_
        # bit_for_bit` hashes `criteria` byte for byte, so this sits beside it, the same way
        # `malformed_tolerance` above does. `criteria["distinct3"]["measure"]` and
        # `rung1.manifest.json`'s dataset report both answer to the prose "within-task
        # distinct-3", and printing them side by side under that one phrase is exactly the
        # mixup this field exists to prevent.
        "distinct3_scope": (
            "criteria['distinct3'] is within_task_distinct_n measured on the CHECKPOINT's own "
            "generated dev questions at inference, fresh per checkpoint -- two checkpoints "
            "trained on the same file can score differently here. It is NOT "
            "rung1.manifest.json's dataset['within_task_distinct_3_of_dataset'], which is a "
            "property of the training file and is identical for every arm trained on the same "
            "export, regardless of base model or seed."
        ),
        "selection": {
            "checkpoint_arm": checkpoint_arm,
            "baseline_arm": baseline_arm,
            "grid_name": grid_name,
            "baseline_grid_names": list(baseline_grid_names),
            "baseline_model_id": baseline_model_id,
            "baseline_model_id_source": "calls.parquet actor='inquirer'",
            "checkpoint_model_id": checkpoint_model_id,
            "checkpoint_model_id_source": "calls.parquet actor='inquirer'",
            "scorer_hash": scorer_hash,
            # The provenance triple's third element, from `scores.graph_version` over exactly
            # these runs. Here and not in `criteria` on purpose: `criteria` is hashed byte for
            # byte by `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit`, so a
            # field added there would move the identity of every verdict ever written.
            "graph_version": graph_version,
            "parquet_dir": str(parquet_dir),
            # T19d: the raw per-arm counts `run_gate` itself checks for EmptyArm before any
            # criterion runs. NOT inside `criteria` -- that dict is hashed bit-for-bit by
            # `test_the_cap8_rule_reproduces_the_previous_verdict_bit_for_bit` -- and duplicated
            # (unchanged) in `pairing` below, which counts the same two numbers alongside what
            # survived task-pairing. This copy is here so a reader of `selection` alone, the
            # block the module docstring says names "which rows", sees both arms' populations
            # without cross-referencing a second block.
            "n_checkpoint_runs": len(ckpt),
            "n_baseline_runs": len(base),
        },
        "pairing": {
            "n_checkpoint_runs": len(ckpt),
            "n_baseline_runs": len(base),
            "n_paired_keys": len(shared),
            "n_unpaired_checkpoint": len(set(ck_keys) - shared),
            "n_unpaired_baseline": len(set(ba_keys) - shared),
            "key": "(suite_id, task_id, seed)",
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "n_resamples": n_resamples,
            "unit": "(suite_id, task_id)",
            # WHAT "unit" ABOVE DOES AND DOES NOT MEAN. Stated because "task-clustered" reads as
            # "clustered" to anyone who does not happen to know that one suite templates several
            # of its tasks -- and a number described as clustered when it is not is the same
            # mislabelling that put pooled means under a clustered heading elsewhere.
            "clustering": (
                "TASK-LEVEL, AND EXPLICITLY NOT TEMPLATE-CLUSTERED. Every interval in this "
                "verdict resamples (suite_id, task_id) pairs, seeds averaged within a task "
                "first (`_paired_by_task`). It is the right unit for THIS command, which is a "
                "selection gate choosing between checkpoints at task level. It is NOT the "
                "estimand `pi_eval.stats.inference.cluster_bootstrap` computes: that one "
                "resamples TEMPLATE clusters, COALESCE(NULLIF(template_id,''), task_id), and "
                "the two differ on every suite that emits a template_id -- measured 2026-09-17, "
                "musique 589 tasks in 560 clusters, tau2 97 in 18, tau2_golden 12 in 5; the "
                "other seven suites emit none and so cluster one task each by construction. "
                "HOW FAR APART: on the frontier musique population (200 tasks in 186 clusters, "
                "172 templates holding one task and 14 holding two) pooled against cluster "
                "mean is +0.000816 on evidence_coverage but +0.011803 on "
                "newly_reachable_share -- about fourteen times as large at the SAME 1.08 tasks "
                "per cluster, so the gap is a property of the metric and cannot be bounded from "
                "the cluster ratio alone. Consequence for a reader: do not publish a "
                "template-clustered figure out of this file. `pi_eval.report` passes the "
                "template cluster map unconditionally to `paired_difference`, `mcnemar_exact` "
                "and `cluster_bootstrap`, so a figure taken through the reporting path is "
                "template-clustered and correct with nobody having to remember to ask for it."
            ),
        },
        "k_matches_baseline": k_stamp,
        "task_count": task_count,
        "thresholds": {
            "length_margin": length_margin,
            "distinct_floor": distinct_floor,
            "malformed_tolerance": MALFORMED_TOLERANCE,
        },
    }
    if mc is not None:
        out["matched_cost"] = mc
    return out


def _with_stop(con, runs: Sequence[Mapping]) -> list[dict]:
    """`stop_reason` AND `n_asks`: the 2x2's unit is a decision point, and n_asks is how many
    of them an episode offered. Taken from `runs` rather than counted off `turns` so a
    truncated turn table shows up as a skipped state instead of a shortened episode."""
    ids = [r["run_id"] for r in runs]
    if not ids:
        return []
    return _rows(
        con, f"SELECT run_id, stop_reason, n_asks FROM runs WHERE run_id IN {_in(ids)} ORDER BY 1"
    )


LENGTH_RULES: tuple[str, ...] = ("tost", "not_longer")
"""What `length_equivalence` gates on. `tost` is the cap-8 rule and is untouched; `not_longer`
is the matched-cost rule -- see `_length` and the D9 follow-up block in `run_gate`."""


def _length(
    con, ckpt, base, *, margin: float, seed: int, n_resamples: int, rule: str = "tost"
) -> dict[str, Any]:
    """Mean question length in words, with a bootstrap test against the baseline.

    EQUIVALENCE, NOT A DIFFERENCE TEST. "we found no significant difference" is not the claim;
    the claim is that any difference is smaller than `margin`. So under `tost` the 90% CI of
    the relative difference (two one-sided tests at 0.05) must lie WHOLLY inside +/- margin.
    scipy is not used: the same percentile bootstrap that produces the paired CIs produces this
    one, so the verdict has one resampling story rather than two.

    `not_longer` DROPS THE LOWER TEST, and only the lower test (D9 follow-up, 2026-09-15). The
    criterion exists for ONE confound: a longer question retrieves more by accident, so an
    unchecked length gain reads as a question-quality gain. A SHORTER question cannot produce
    that error in either direction -- it can only make the checkpoint's own coverage and depth
    numbers harder to earn -- so failing a checkpoint for it rejects on an artefact. The upper
    test is unchanged, the same 0.05 one-sided bound on the same relative scale, so verbosity is
    caught exactly as before. The two-sided verdict is kept beside it as `tost_equivalent`,
    because "shorter, and by how much" is a fact about the arm a reader wants.
    """
    if rule not in LENGTH_RULES:
        raise ValueError(f"length rule {rule!r} is not one of {LENGTH_RULES}")
    one_sided = rule == "not_longer"
    extra: dict[str, Any] = {}
    ck = [len(q.split()) for _, q in _questions(con, ckpt)]
    ba = [len(q.split()) for _, q in _questions(con, base)]
    if not ck or not ba:
        if one_sided:
            extra["tost_equivalent"] = False
        return _criterion(
            value=float("nan"),
            baseline=float("nan"),
            threshold=margin,
            gated=True,
            passed=False,
            n=0,
            reason="one arm recorded no questions, so equivalence is undefined",
            **extra,
        )
    m_ck, m_ba = _mean(ck), _mean(ba)
    if m_ba == 0:
        if one_sided:
            extra["tost_equivalent"] = False
        return _criterion(
            value=m_ck,
            baseline=m_ba,
            threshold=margin,
            gated=True,
            passed=False,
            n=len(ck),
            reason="the baseline's mean question length is 0, so a relative margin is undefined",
            **extra,
        )
    # SORTED, FOR THE SAME REASON `bca_ci` AND `pi_eval.stats.inference.cluster_bootstrap` SORT
    # THEIR UNITS, and placed here for the same reason they place it: immediately before the
    # resampling and after every filter, so the RNG's first draw already reads a canonical list.
    # The resampling draws INDICES, so `ck[rng.randrange(len(ck))]` reads a different question for
    # the same draw when the caller's list was built in a different order -- and this caller's
    # order came from `_questions`' `ORDER BY task_id, question`, which is canonical over KEYS and
    # says nothing about VALUES. That made the endpoints a property of what one query happened to
    # return: add a column to that ORDER BY, change the join, or run a suite whose task ids sort
    # differently, and the bounds move with no code change and no signal.
    #
    # THE UNIT HERE IS ONE QUESTION'S WORD COUNT -- a scalar -- so unlike `cluster_bootstrap`
    # there is no inner level to canonicalise: sorting reorders units and cannot reorder values
    # within one. `m_ck`, `m_ba` and `n` are properties of the multiset and are computed above;
    # MEASURED over all 182 recorded `criteria.length_equivalence` intervals under `artifacts/`
    # (2026-09-18): every one reproduces bit for bit in arrival order, every one's endpoints move
    # under this sort, and the two means and `n` move on none of them.
    #
    # NOT LATENT. This moved published bounds, unlike `bca_ci`'s sort: all five printed intervals
    # in `appendix_stopping.tex` Table L move at three decimals and the prose margin there goes
    # from 0.008 to 0.007. Every pass and fail verdict is unchanged. The audit that opened this
    # reported the interval as reproducing identically under both rules and it does not.
    ck, ba = sorted(ck), sorted(ba)
    rng = random.Random(seed)
    rels = []
    for _ in range(n_resamples):
        a = _mean([ck[rng.randrange(len(ck))] for _ in range(len(ck))])
        b = _mean([ba[rng.randrange(len(ba))] for _ in range(len(ba))])
        rels.append((a - b) / b if b else float("nan"))
    rels.sort()
    lo = rels[max(0, int(round(0.05 * (len(rels) - 1))))]
    hi = rels[min(len(rels) - 1, int(round(0.95 * (len(rels) - 1))))]
    tost = bool(lo >= -margin and hi <= margin)
    extra = (
        {
            "tost_equivalent": tost,
            "test": "bootstrap NON-INFERIORITY on the relative difference: the 95% upper "
            "bound is at most +margin. Shorter always passes; the upper test is the TOST's, "
            "unchanged. `tost_equivalent` is the two-sided verdict, reported not gated.",
        }
        if one_sided
        else {"test": "bootstrap TOST on the relative difference, 90% CI inside +/- margin"}
    )
    return _criterion(
        value=m_ck,
        baseline=m_ba,
        threshold=margin,
        gated=True,
        passed=bool(hi <= margin) if one_sided else tost,
        n=len(ck),
        ci_lo=_nan(lo),
        ci_hi=_nan(hi),
        **extra,
    )
