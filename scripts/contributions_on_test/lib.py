"""Shared tooling for Lane L1.2: contributions A and B recomputed on the held-out test split.

Two small, pure pieces of logic used by both `length_control_test_split.py` (contribution A,
the token-charged length control) and `dominance_test_split.py` (contribution B, the
trained-vs-teacher frontier): parsing a run-id list file, and the CLAUDE.md/ANALYSIS-RULES
population assertion ("before computing, assert every run_id in your population is in
runs.parquet and has scores at the scorer_hash your source artifact names").

Nothing here reads gold, calls an LLM, or writes to `scores/parquet/`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import duckdb

# `pinq.ids.h()` mints run_id (and semantic_hash, model_pin_hash, ...) as 32 lowercase hex
# chars. Every other column that shares a run-id list file with it -- task_id, seed,
# template_id -- is either not hex (musique task ids contain letters like "hop" and
# underscores), a different length (strategyqa task ids are 20 hex chars, not 32), or a bare
# digit (seed). Matching on shape, not position, is what makes one reader correct against
# THREE incompatible column orders actually on disk in this repo (see the three formats below).
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def main_checkout(start: Path | None = None) -> Path:
    """The main checkout's absolute path, derived from `start` (default: this file) rather
    than a literal -- a hardcoded `/Users/<name>/...` string would both be wrong on any other
    machine and fail `scripts/check_no_home_paths.sh`, which greps committed `.py` files for
    exactly that pattern. ANALYSIS RULES for this lane require `scores/parquet` and every
    artifact under `artifacts/` to be read by absolute path FROM THE MAIN CHECKOUT specifically
    (a worktree has neither -- both are untracked), so a relative path is not a substitute.

    Walks up from `start` for a `.claude/worktrees/<hash>` segment (the shape every worktree
    agent runs in) and returns its parent, i.e. the main checkout; a `start` that is not inside
    a worktree (the main checkout itself) returns `start`'s own repo root.
    """
    here = (start or Path(__file__)).resolve()
    parts = here.parts
    if ".claude" in parts and "worktrees" in parts:
        i = parts.index(".claude")
        return Path(*parts[:i])
    # Not inside a worktree: climb to the repo root (this file lives at <root>/scripts/
    # contributions_on_test/lib.py, three levels down).
    return here.parents[2]


def read_run_ids(path: Path) -> list[str]:
    """The `run_id` column out of a run-id list file, tolerant of every format on disk here.

    Comment lines (leading `#`) and blank lines are skipped. FOUR shapes of data line are in
    use across the artifacts this lane reads, and picking a fixed column INDEX silently gets
    three of the four wrong (a column that "parses" but is a task_id or a seed, not a run_id,
    with no error either way):

    * one field: a bare run_id -- `artifacts/testsplit_qa`, `artifacts/length_control_tokens`
      and `artifacts/length_tokens`'s `run_ids.*.txt`.
    * `run_id \\t task_id \\t seed` -- `artifacts/frontier`'s `run_ids.*.cap*.txt`. run_id FIRST.
    * `task_id \\t seed \\t run_id` -- `artifacts/frontier_strategyqa`'s `run_ids.*.cap*.tsv`.
      run_id LAST.
    * `run_id \\t suite \\t seed \\t split \\t status \\t usd \\t task_id` --
      `artifacts/n6/run_ids.*.txt`. run_id first again, but here CONTENT ALONE is ambiguous:
      a wiki2 task_id can itself be 32 lowercase hex characters (measured:
      `00c727580bde11eba7f7acde48001122`), so two fields match the run_id shape and
      content-only resolution must refuse rather than guess.

    Resolution order, each one a stronger signal than the last: (1) if a `#`-prefixed header
    line naming the columns (tab-separated, containing the literal token `run_id`) precedes
    the data, its column INDEX is used and every value it names is still shape-checked (a
    header that names the wrong column would fail the check rather than being trusted blindly);
    (2) otherwise, CONTENT alone must find exactly one 32-hex field or the line is refused.
    """
    out: list[str] = []
    run_id_col: int | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            header_fields = [f.strip() for f in line.lstrip("#").strip().split("\t")]
            if "run_id" in header_fields:
                run_id_col = header_fields.index("run_id")
            continue
        fields = [f.strip() for f in line.split("\t")]
        if len(fields) == 1:
            out.append(fields[0])
            continue
        if run_id_col is not None and run_id_col < len(fields):
            candidate = fields[run_id_col]
            if not _RUN_ID_RE.match(candidate):
                raise ValueError(
                    f"{path}: header named column {run_id_col} as run_id, but {fields!r}'s "
                    f"value there ({candidate!r}) is not 32 lowercase hex characters"
                )
            out.append(candidate)
            continue
        matches = [f for f in fields if _RUN_ID_RE.match(f)]
        if len(matches) != 1:
            raise ValueError(
                f"{path}: cannot identify a unique run_id column in {fields!r} -- "
                f"{len(matches)} of its fields look like a 32-hex run_id, expected exactly 1, "
                "and no '# ...run_id...' header column preceded this line"
            )
        out.append(matches[0])
    return out


class PopulationIncomplete(RuntimeError):
    """The population assertion failed: some run_id is missing, unscored, or scored at the
    wrong scorer_hash/graph_version. Never silently dropped -- see ANALYSIS RULES: poll for
    artifacts/compaction_20260918/RESULT.md and re-read before computing anything further."""

    def __init__(
        self,
        parquet_dir: Path,
        missing_runs: Sequence[str],
        missing_scores: Sequence[str],
        wrong_hash: Sequence[str],
    ) -> None:
        self.parquet_dir = parquet_dir
        self.missing_runs = list(missing_runs)
        self.missing_scores = list(missing_scores)
        self.wrong_hash = list(wrong_hash)
        msg = (
            f"{parquet_dir}: population assertion failed -- "
            f"{len(self.missing_runs)} run_id(s) absent from runs.parquet, "
            f"{len(self.missing_scores)} absent from scores.parquet, "
            f"{len(self.wrong_hash)} scored at an unexpected scorer_hash/graph_version. "
            "Poll for artifacts/compaction_20260918/RESULT.md and re-read before retrying."
        )
        super().__init__(msg)


@dataclass(frozen=True, slots=True)
class PopulationReport:
    parquet_dir: str
    n_checked: int
    scorer_hash: str
    graph_version: str
    metric_name: str = "evidence_coverage"
    extra: dict = field(default_factory=dict)


def assert_population_scored(
    parquet_dir: Path,
    run_ids: Sequence[str],
    *,
    expected_scorer_hash: str,
    expected_graph_version: str = "v1",
    metric_name: str = "evidence_coverage",
) -> PopulationReport:
    """Assert every id in `run_ids` is in `parquet_dir/runs.parquet` AND carries a `metric_name`
    score in `parquet_dir/scores.parquet` at exactly `(expected_scorer_hash, expected_graph_version)`.

    This is CLAUDE.md rule 1 ("a number without provenance is not a result") applied before a
    single number is computed, not after: a run_id silently absent from either table would make
    every downstream mean an unweighted average over whoever happened to be scored, not the
    population the source artifact named.
    """
    con = duckdb.connect()
    have_runs = {
        r[0]
        for r in con.execute(
            f"SELECT run_id FROM read_parquet('{parquet_dir}/runs.parquet')"
        ).fetchall()
    }
    scored_rows = con.execute(
        f"SELECT run_id, scorer_hash, graph_version FROM read_parquet('{parquet_dir}/scores.parquet') "
        f"WHERE metric_name = '{metric_name}'"
    ).fetchall()
    scored = {r[0]: (r[1], r[2]) for r in scored_rows}

    want = set(run_ids)
    missing_runs = sorted(want - have_runs)
    missing_scores = sorted(r for r in want if r in have_runs and r not in scored)
    wrong_hash = sorted(
        r
        for r in want
        if r in scored and scored[r] != (expected_scorer_hash, expected_graph_version)
    )
    if missing_runs or missing_scores or wrong_hash:
        raise PopulationIncomplete(parquet_dir, missing_runs, missing_scores, wrong_hash)
    return PopulationReport(
        parquet_dir=str(parquet_dir),
        n_checked=len(want),
        scorer_hash=expected_scorer_hash,
        graph_version=expected_graph_version,
        metric_name=metric_name,
    )


class TurnsDropped(RuntimeError):
    """The compaction bug this check exists for: a run whose `turns.jsonl` is intact and
    non-empty on disk compacted to `n_turns=0` (and/or has no `calls.parquet` Inquirer rows),
    which zeroes any token or coverage charge silently rather than refusing."""

    def __init__(self, report: "TurnsNotDroppedReport") -> None:
        self.report = report
        super().__init__(
            f"{report.parquet_dir}: {report.n_zeroed_n_turns} run(s) with a non-empty "
            f"turns.jsonl on disk compacted to n_turns=0, {report.n_missing_inquirer_calls} "
            "have no Inquirer row in calls.parquet. First few: "
            f"n_turns=0: {report.zeroed_run_ids[:5]}, no calls: {report.missing_call_run_ids[:5]}"
        )


@dataclass(frozen=True, slots=True)
class TurnsNotDroppedReport:
    parquet_dir: str
    n_checked_total: int
    n_nonempty_on_disk: int
    n_zeroed_n_turns: int
    n_missing_inquirer_calls: int
    zeroed_run_ids: tuple[str, ...] = ()
    missing_call_run_ids: tuple[str, ...] = ()


def assert_turns_not_dropped(
    parquet_dir: Path,
    run_ids: Sequence[str],
    *,
    runs_root: Path,
) -> TurnsNotDroppedReport:
    """For every run_id whose `<runs_root>/<run_id>/turns.jsonl` is non-empty on disk, assert
    `runs.parquet.n_turns > 0` for it AND that `calls.parquet` carries at least one
    `actor='inquirer'` row for it. Raises `TurnsDropped` naming exactly which run_ids failed
    either check, rather than silently reading a dropped row as a zero-turn run.

    Names a real defect (2026-09-18): a compaction pass dropped the turn, evidence and call
    rows of 21,001 runs whose `turns.jsonl` was intact on disk -- they read as `n_turns=0`,
    coverage 0, with their `calls.parquet` rows missing entirely, which zeroes any token charge
    with no symptom other than a suspiciously round number.
    """
    ids = list(run_ids)
    con = duckdb.connect()
    lst = ",".join(f"'{i}'" for i in ids)
    n_turns_by_id = dict(
        con.execute(
            f"SELECT run_id, n_turns FROM read_parquet('{parquet_dir}/runs.parquet') "
            f"WHERE run_id IN ({lst})"
        ).fetchall()
    )
    has_inquirer_call = {
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT run_id FROM read_parquet('{parquet_dir}/calls.parquet') "
            f"WHERE actor = 'inquirer' AND run_id IN ({lst})"
        ).fetchall()
    }

    nonempty_on_disk = 0
    zeroed: list[str] = []
    missing_calls: list[str] = []
    for rid in ids:
        p = Path(runs_root) / rid / "turns.jsonl"
        try:
            non_empty = p.is_file() and p.stat().st_size > 0
        except OSError:
            non_empty = False
        if not non_empty:
            continue
        nonempty_on_disk += 1
        if not n_turns_by_id.get(rid):
            zeroed.append(rid)
        if rid not in has_inquirer_call:
            missing_calls.append(rid)

    report = TurnsNotDroppedReport(
        parquet_dir=str(parquet_dir),
        n_checked_total=len(ids),
        n_nonempty_on_disk=nonempty_on_disk,
        n_zeroed_n_turns=len(zeroed),
        n_missing_inquirer_calls=len(missing_calls),
        zeroed_run_ids=tuple(zeroed),
        missing_call_run_ids=tuple(missing_calls),
    )
    if zeroed or missing_calls:
        raise TurnsDropped(report)
    return report


class PooledCodeVersions(RuntimeError):
    """A hazard measured on the shared store (2026-09-18): `_select_runs` carries no
    `code_version` filter, so a store holding one arm's runs from more than one rollout code
    version pools them into one mean silently, and the same nominal (suite, task, seed) key can
    carry different `evidence_coverage` values across them. Raised naming exactly which arm and
    which keys, rather than averaging over whichever rows happened to be present."""

    def __init__(
        self,
        arm: str,
        code_versions: dict[str, int],
        duplicate_keys: dict[tuple[str, str, int], int],
    ) -> None:
        self.arm = arm
        self.code_versions = code_versions
        self.duplicate_keys = duplicate_keys
        super().__init__(
            f"{arm}: {len(code_versions)} distinct code_version(s) {code_versions} and "
            f"{len(duplicate_keys)} (suite,task,seed) key(s) with more than one row -- "
            "rebuild from a single-code_version run-id list or filter explicitly before "
            "pairing, a pool here would average over two different populations silently"
        )


def assert_one_code_version_and_no_duplicate_keys(
    parquet_dir: Path, run_ids: Sequence[str], *, arm: str
) -> dict[str, Any]:
    """For one arm's run_ids in `parquet_dir/runs.parquet`: exactly one distinct `code_version`,
    and no (suite_id, task_id, seed) key with more than one row. Returns
    `{"code_versions": {...}, "n_duplicate_keys": int}` when clean; raises
    `PooledCodeVersions` naming the arm and the offending keys otherwise.
    """
    con = duckdb.connect()
    lst = ",".join(f"'{i}'" for i in run_ids)
    rows = con.execute(
        f"SELECT suite_id, task_id, seed, code_version FROM read_parquet('{parquet_dir}/runs.parquet') "
        f"WHERE run_id IN ({lst})"
    ).fetchall()
    code_versions: dict[str, int] = {}
    key_counts: dict[tuple[str, str, int], int] = {}
    for suite_id, task_id, seed, code_version in rows:
        code_versions[code_version] = code_versions.get(code_version, 0) + 1
        key = (suite_id, task_id, seed)
        key_counts[key] = key_counts.get(key, 0) + 1
    duplicate_keys = {k: c for k, c in key_counts.items() if c > 1}
    if len(code_versions) > 1 or duplicate_keys:
        raise PooledCodeVersions(arm, code_versions, duplicate_keys)
    return {"code_versions": code_versions, "n_duplicate_keys": 0}


NEAR_ZERO_THRESHOLD = 0.01


def near_zero_bounds(
    ci_lo: float, ci_hi: float, threshold: float = NEAR_ZERO_THRESHOLD
) -> list[str]:
    """Which named bound(s) (`'lo'`, `'hi'`) of a CI sit within `threshold` of zero.

    Coordinator rule (added mid-task, citing a peer's measurement: at 1k resamples 3 of 7 gated
    cells whose bound sits within 0.01 of zero flip that bound's sign across bootstrap seeds,
    and one printed pass dies at 10k): a bound this close to zero is read again at 50k
    resamples under three seeds before its sign is trusted.
    """
    out: list[str] = []
    if abs(ci_lo) < threshold:
        out.append("lo")
    if abs(ci_hi) < threshold:
        out.append("hi")
    return out


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def stability_verdict(
    bounds_by_seed: dict[int, tuple[float, float]], flagged: Sequence[str]
) -> str:
    """`'stable'` if every flagged bound (a subset of `{'lo', 'hi'}`) has the same sign across
    every seed in `bounds_by_seed` (seed -> (ci_lo, ci_hi)); otherwise `'undecided'`.

    A verdict whose sign is not stable is reported as undecided, never rounded to a pass or a
    fail -- that is the whole point of the check the coordinator rule adds.
    """
    idx = {"lo": 0, "hi": 1}
    for b in flagged:
        signs = {_sign(vals[idx[b]]) for vals in bounds_by_seed.values()}
        if len(signs) > 1:
            return "undecided"
    return "stable"


def elects_to_stop(n_total: int, ceiling_hit: int) -> int:
    """Units that emitted STOP before the cap, i.e. did NOT hit the ceiling.

    `ceiling_hit` is `stop_reason in {'budget', 'max_turns'}`; everything else in a population
    of `n_total` scored units stopped on its own. Guards rather than silently returning a
    negative count, which would read as a measurement instead of a contradiction in the inputs.
    """
    if ceiling_hit > n_total or ceiling_hit < 0 or n_total < 0:
        raise ValueError(f"nonsensical inputs: n_total={n_total} ceiling_hit={ceiling_hit}")
    return n_total - ceiling_hit
