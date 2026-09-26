"""`pi` — argparse, no framework.

A CLI framework would buy plugin discovery and lose the property that matters here: every
command's full surface is readable in one file, so "what exactly did that invocation do" is
answered by reading, not by tracing decorators. One dispatch table, one file.

`pi run` UNSETS PI_GOLD_ROOT IN THE PARENT before planning anything. Workers inherit the
parent environment, so scrubbing here is what makes pinq's process-level firewall true rather
than aspirational — and pi_run.worker.assert_firewall re-checks it in the child, because a
guarantee with one enforcement point is a guarantee with one bug away from nothing.

pi_eval is imported ONLY inside the scoring-side handlers (`compact`, `score`, `agg`,
`render`, `report`). Those run in the scoring process; nothing on the rollout path may pull
gold-side code into its address space, which is why every one of those imports is local to
its function and none of them is at module scope.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from pi_run.cache import DiskCache, cache_root
from pi_run.cmd_data import register as _register_data
from pi_run.cmd_gold import register as _register_gold
from pi_run.cmd_suites import register as _register_suites
from pi_run.cmd_train import register as _register_train
from pi_run.manifest import GitInfo, git_info, repo_root
from pi_run.suites import REGISTRY
from pi_run.sweep import (
    SpendCapInvalid,
    max_concurrency,
    plan,
    progress_printer,
    run_sweep,
    spend_cap,
    summarize,
)
from pi_run.worker import DEFAULT_UNIT_TIMEOUT_S, firewall_probe, unit_timeout_s
from pinq_expt import arms as arm_table

# Naive per-rollout token draw, from the plan's cost model. Used only by `pi cost estimate`;
# every real number comes from the ledger, never from this.
GRIDS: dict[str, dict[str, Any]] = {
    "A": {"rollouts": 11849, "tok_in": 253_000, "tok_out": 8_800, "note": "main, 17 arms, M1"},
    "A2": {"rollouts": 8800, "tok_in": 120_000, "tok_out": 5_000, "note": "secondary suites"},
    "B": {"rollouts": 6246, "tok_in": 180_000, "tok_out": 7_000, "note": "generality, M2/M3/M4"},
    "C": {"rollouts": 2776, "tok_in": 180_000, "tok_out": 7_000, "note": "training, M4"},
    "D": {"rollouts": 2088, "tok_in": 180_000, "tok_out": 7_000, "note": "seed variance"},
    "E": {"rollouts": 572, "tok_in": 90_000, "tok_out": 4_000, "note": "PARE transfer"},
}
CACHE_DISCOUNT = 0.27  # prompt caching on the append-only prefix

# Duplicated from pinq_adapters.llm.litellm_client so `pi env doctor` reports key presence
# even when the `run` extras are not installed. Presence only: a value is never printed.
_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "litellm_proxy": "LITELLM_API_KEY",
}
_ROLE_ENV = {
    "inquirer": "PI_MODEL_INQUIRER",
    "drafter": "PI_MODEL_DRAFTER",
    "answerer": "PI_MODEL_ANSWERER",
    "user_sim": "PI_MODEL_USERSIM",
    "judge": "PI_MODEL_JUDGE",
}


# --------------------------------------------------------------------------- helpers


def runs_root(root: str | os.PathLike[str], explicit: str | os.PathLike[str] | None = None) -> Path:
    """The runs directory: `--runs-root` wins, then `PI_RUNS_ROOT`, then `<root>/runs`.

    Mirrors `cache_root()`'s own precedence exactly (explicit flag, then its env var, then a
    default), because a sweep recipe that says "set ABSOLUTE PI_CACHE_ROOT and PI_RUNS_ROOT"
    should get the same behaviour from both. It used to not: `cmd_run`, `cmd_compact` and
    `cmd_score` each inlined `Path(a.runs_root) if a.runs_root else root / "runs"`, so
    `PI_RUNS_ROOT` alone did nothing for any of the three, while `pi env doctor`'s diagnostic
    line already read it (for display only) and three `scripts/launch_controls/*.py` plus
    `scripts/label_ordering_test/01_select_and_link.py` already read it as their own runs-root
    override. A variable honoured by some entry points and ignored by others is worse than one
    ignored everywhere -- see `test_pi_run_honours_PI_RUNS_ROOT_with_no_explicit_flag`.
    """
    p = Path(explicit or os.environ.get("PI_RUNS_ROOT") or (Path(root) / "runs"))
    return p if p.is_absolute() else Path.cwd() / p


RUN_DEFAULTS: dict[str, object] = {"n": 5, "seeds": ["0"], "max_turns": 16, "k": 5}


def resolved_run_defaults(a: argparse.Namespace) -> dict[str, object]:
    """The standalone `pi run` values, applying defaults the parser no longer supplies.

    The parser's defaults became None so that an EXPLICIT flag is distinguishable from an
    unpassed one -- without that distinction `grid_flag_conflicts` cannot tell "the operator
    typed --seeds 7" from "argparse filled in 0". The real defaults live here instead.
    """
    out: dict[str, object] = {}
    for name, default in RUN_DEFAULTS.items():
        v = getattr(a, name, None)
        out[name] = default if v is None else v
    return out


def grid_flag_conflicts(
    a: argparse.Namespace,
    *,
    seeds: Sequence[int] | None = None,
    n_tasks: int | None = None,
    max_turns: int | None = None,
    k: int | None = None,
    concurrency: int | None = None,
    task_ids: Sequence[str] = (),
    grid: Any = None,
    suite_id: str = "",
) -> tuple[str, ...]:
    """Flags typed beside `--sweep` that CONTRADICT the grid. Empty when there is no clash.

    THE BUG THIS CLOSES. The grid loop assigned `sub.seeds = [str(x) for x in grid.seeds]`
    and the same for n, max_turns, k and budget_cap, unconditionally -- so
    `pi run --sweep <grid> --seeds 7 --max-turns 1 --k 2` ran the GRID's values while its
    operator believed otherwise. That is the same class as the bug that billed $0.7158 by
    letting `--sweep` override `--arm`; that one was fixed in 431bada and never generalised.

    REFUSED rather than resolved either way, matching what `--arm` already does. Silently
    honouring the flag and silently honouring the grid are both wrong answers to a
    contradiction the operator did not intend -- and on a money-spending command the wrong
    answer is billed before anyone notices.

    `--arm` is deliberately absent: it FILTERS a grid rather than contradicting it.
    A flag whose value MATCHES the grid is redundant, not a conflict.

    A SIXTH FLAG, 2026-09-17 (found by another lane, verified here): `sub.task =
    list(grid.task_ids) or []` is the same unconditional assignment, and this function never
    looked at `--task`. It is worse than the other five: there is no shape in which an
    explicit `--task` survives the grid loop. A grid with its own `task_ids` replaces it with
    those; a grid with none (the common case -- most grids select by `n_tasks`/`split`
    instead) replaces it with `[]`, which then falls through to `cmd_run`'s `n_tasks`-based
    selection. So `pi run --sweep <200-task grid> --task one_task` ran all 200 while its
    operator believed they had pinned one, with nothing in the banner an operator would think
    to check -- and that is exactly the validation step several launches use to prove a tree
    is clean before committing to a large campaign. Checked the same way as the other five:
    an explicit, non-default `--task` that would be silently discarded is refused, never
    resolved either way. `--task` uses `action="append", default=[]`, so its explicit-vs-
    default sentinel is "non-empty list" rather than the `None` the other five use -- same
    discipline, different sentinel, because that is the sentinel this flag already has.

    2026-09-18: THE LIST WAS THE DEFECT. Six flags were checked by hand while the grid loop
    assigned seventeen fields, and the docstring above named `budget_cap` among the fields the
    assignment broke while the code below never inspected it -- so
    `--budget-cap 4 --sweep <cap-8 grid>` was accepted, discarded, and run at 8. Two more,
    `--split` and `--task-offset`, had never been named at all, `--k` was compared against
    `grid.k` while the loop assigned `grid.k_for(suite_id)`, and adding a seventh entry by
    hand would have left the eighth for whoever's campaign found it. So this function no
    longer holds a list: it delegates to `grid_discarded_flags`, which DIFFS the operator's
    namespace against the one `grid_child_namespace` will actually dispatch. Assigned and
    checked are now the same set by construction. This signature is kept because it is the
    unit-test surface for the six flags above, and it now answers by the same diff the
    `--sweep` path uses; pass `grid=` to diff against a real grid.
    """
    from pi_run import grids

    if grid is None:
        # A CALL THAT SUPPLIES NEITHER CHECKS NOTHING, so it is refused rather than answered
        # with `()`. A guard whose caller cannot tell "clean" from "never looked" is the
        # failure mode this whole function exists to remove.
        missing = [
            n for n, v in (("seeds", seeds), ("max_turns", max_turns), ("k", k)) if v is None
        ]
        if missing:
            raise TypeError(
                f"grid_flag_conflicts: pass grid=, or all of {missing} -- a call with neither "
                "would report no conflicts without having compared anything"
            )
        # A Grid built from THIS FUNCTION'S OWN KEYWORDS, so the unit-test surface and the
        # `--sweep` surface run the same diff. Constructing a real `grids.Grid` (rather than
        # accepting a keyword per grid field) is deliberate: a NEW grid field arrives here at
        # its dataclass default automatically, so this adapter cannot become the second list
        # that drifts out of sync.
        grid = grids.Grid(
            name="(from keywords)",
            description="",
            suites=(suite_id,),
            arms=(),
            seeds=tuple(int(x) for x in seeds or ()),
            n_tasks=n_tasks,
            task_ids=tuple(str(t) for t in task_ids),
            max_turns=int(max_turns),
            k=int(k),
            concurrency=concurrency,
        )
    return grid_discarded_flags(a, grid_child_namespace(a, grid, suite_id))


# Fields `grid_child_namespace` sets for a reason OTHER than "the grid overrides the command
# line", so a difference between the operator's namespace and the child's is not a discarded
# flag. This dict is the only way a field escapes `grid_discarded_flags`, so every entry
# carries the reason it is not a discard -- an entry added without one is how the eighth
# blind spot gets in.
GRID_DIFF_EXEMPT: dict[str, str] = {
    "arm": (
        "--arm FILTERS a grid rather than contradicting it (431bada); cmd_run narrows the "
        "grid's arm list to it and REFUSES an --arm the grid does not contain."
    ),
    "spend_cap": (
        "the operator's cap is HONOURED, as a cap on the whole grid: cmd_run threads the "
        "remainder through the suites, so the child's value is lower by what earlier suites "
        "billed. That is the flag working, not the flag being discarded."
    ),
    "sweep": (
        "cleared so the child `pi run` dispatches one suite instead of recursing into the "
        "grid. Structural; there is no operator intent to contradict."
    ),
}


# Extra prose for a field whose bare before/after does not say what the operator needs to know.
# An ABSENT note degrades to the generic message; it can never turn the check off, which is
# why this table being incomplete is survivable and `GRID_DIFF_EXEMPT` being wrong is not.
def _discard_note(field: str, sub: argparse.Namespace) -> str:
    if field == "task":
        n = getattr(sub, "n", None)
        return (
            " -- the grid replaces --task with the "
            f"{n if n is not None else 'unbounded number of'} task(s) it selects by "
            "n_tasks/split, unconditionally, not only when the two disagree"
        )
    if field == "concurrency":
        return ". That ceiling is what the suite's HOST asked for, not a local preference."
    return ""


@functools.lru_cache(maxsize=1)
def _run_flag_defaults() -> tuple[tuple[str, Any], ...]:
    """Every `pi run` dest paired with its parser default, in parser order.

    Obtained by PARSING a bare `run` rather than by restating the defaults, so it is the
    parser's own answer and cannot go stale the way a hand-copied list does. `fn`/`cmd` are
    dispatch bookkeeping, not flags.
    """
    a = build_parser().parse_args(["run"])
    return tuple((k, v) for k, v in vars(a).items() if k not in ("fn", "cmd"))


def run_flag_defaults() -> dict[str, Any]:
    """The `pi run` parser defaults, as a fresh dict (the cached tuple stays immutable)."""
    return dict(_run_flag_defaults())


def _same_flag_value(x: Any, y: Any) -> bool:
    """Equality as the CLI means it: `--seeds 7` and a grid's `(7,)` are the same value.

    The grid loop writes `[str(x) for x in grid.seeds]` while argparse hands back strings, so a
    plain `==` on a list would call a matching seed a conflict and refuse every legitimate
    sweep. Compared elementwise as strings for sequences, plainly otherwise.
    """
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [str(v) for v in x] == [str(v) for v in y]
    return bool(x == y)


def grid_discarded_flags(a: argparse.Namespace, sub: argparse.Namespace) -> tuple[str, ...]:
    """Flags the operator TYPED that `grid_child_namespace` would silently overwrite.

    STRUCTURAL, NOT A SECOND LIST. The source of truth is the child namespace itself, so a
    field becomes checked at the moment it becomes assigned; there is no list of inspected
    fields to fall behind the list of assigned ones. That is the defect this replaces: the
    loop assigned seventeen fields and the checker inspected six of them, and the seventh --
    `budget_cap`, named in the checker's own docstring -- was found only after
    `--budget-cap 4 --sweep <cap-8 grid>` had been accepted and silently run at 8.

    THE TWO WAYS A REFUSAL HERE COULD BE WRONG, and what stops each:

      * firing when the flag AGREES with the grid would break every legitimate sweep. A field
        whose typed value equals what the child will carry is redundant, not a conflict, and
        is skipped -- compared with `_same_flag_value`, because `--seeds 7` and `(7,)` are the
        same value written two ways.
      * firing on a value ARGPARSE supplied would refuse every sweep that types no flag at
        all. Skipped by comparing against the parser's own default for that dest. This is why
        `--seeds/--n/--max-turns/--k` default to None rather than to their real values
        (`resolved_run_defaults` applies those on the non-sweep path instead) -- without that,
        the default and an explicit flag are the same bytes and nothing here can tell them
        apart. TWO FLAGS STILL CANNOT BE TOLD APART, and neither can be fixed from inside this
        function: `--task-offset` defaults to 0, so an explicit `--task-offset 0` is invisible
        (a non-zero one is caught), and `--pilot` is `store_true`, so only an explicit `--pilot`
        is visible (`--pilot` cannot be typed as False). Both are silent in exactly the
        direction where the typed value already equals the parser default, so neither can cause
        a WRONG refusal; each can only miss one, and both are named in the parser rather than
        assumed away.

    REFUSED rather than resolved either way, matching `--arm` and the six flags before it.
    """
    out: list[str] = []
    for field, default in _run_flag_defaults():
        if field in GRID_DIFF_EXEMPT:
            continue
        typed = getattr(a, field, default)
        if _same_flag_value(typed, default):
            continue  # argparse supplied it; there is no operator intent to contradict
        if not hasattr(sub, field):
            continue  # the grid does not touch this field
        assigned = getattr(sub, field)
        if _same_flag_value(typed, assigned):
            continue  # redundant, not a conflict
        flag = "--" + field.replace("_", "-")
        out.append(
            f"{flag} {typed!r} but the grid makes it {assigned!r}{_discard_note(field, sub)}"
        )
    return tuple(out)


def grid_child_namespace(a: argparse.Namespace, grid: Any, suite_id: str) -> argparse.Namespace:
    """The child `pi run` namespace a `--sweep` dispatches for one suite.

    THE ONE PLACE THE GRID OVERRIDES A COMMAND LINE. `grid_discarded_flags` diffs this
    function's output against the operator's own namespace, so the set of fields the grid
    assigns and the set it checks are the same set by construction. Adding a field here makes
    it refused-if-contradicted immediately; there is nothing to add anywhere else. Anything
    assigned to the child OUTSIDE this function is invisible to that diff, which is why
    `tests/test_sweep_flag_conflict.py` walks `cmd_run`'s syntax tree and fails if the sweep
    loop grows another `sub.<field> =` of its own.

    `--arm` and `--spend-cap` are deliberately left to `cmd_run`: they are not overrides (see
    GRID_DIFF_EXEMPT) and each has its own refusal.
    """
    sub = argparse.Namespace(**vars(a))
    sub.sweep = None
    sub.suite = suite_id
    # PROVENANCE onto every unit. The hash was computed and printed by cmd_run and entered no
    # record, so a run could not be traced to the grid text that made it.
    sub.grid_sha256 = grid.source_sha256
    sub.grid_name = grid.name
    sub.seeds = [str(x) for x in grid.seeds]
    sub.n = grid.n_tasks if grid.n_tasks is not None else a.n
    sub.task = list(grid.task_ids) or []
    sub.split = grid.split
    sub.task_offset = grid.task_offset
    sub.max_turns = grid.max_turns
    # PER SUITE, not grid-wide. `k` is not a free parameter (see grids.Grid.k): NINE grids
    # tracked at main declare `k_by_suite` (ten in a tree carrying the untracked
    # frontier_trained_musique_cap24), and on FOUR of them a per-suite value differs from the
    # grid-wide `k` -- tier1_confirmatory's is {strategyqa: 2, wiki2: 3} against `k: 5`. The
    # rest declare it redundantly (every entry equals `k`), so they take this code path without
    # the value changing; the four is the count that could record a wrong `k`, and it does not
    # move between the two trees. The check this replaces compared `--k` with `grid.k`, so
    # `--k 5` was
    # accepted and strategyqa still ran at 2 -- a field that was CHECKED and still silently
    # discarded, which is why the diff is taken against the child namespace of every suite
    # rather than against the grid's own fields.
    sub.k = grid.k_for(suite_id)
    sub.budget_cap = grid.budget_cap
    sub.pilot = grid.pilot
    sub.exploratory = grid.exploratory
    if grid.concurrency is not None:
        sub.concurrency = grid.concurrency
    return sub


def dirty_sweep_refusal(gi: GitInfo, *, allow_dirty: bool, root: Path) -> str | None:
    """The refusal message for launching `--sweep` on a dirty tree, or None to proceed.

    Incident, 2026-09-17: a 400-unit evaluation sweep was launched while three unrelated
    tracked files sat uncommitted in this checkout. Every one of the 400 runs came out dev-
    stamped (`RunManifest.dirty` follows `GitInfo.dirty` straight into the run_id) --
    training-only, inadmissible as an eval arm -- and the loss was invisible until someone read
    the manifests after the fact. `cmd_run`'s non-sweep tail already prints a DIRTY banner in
    this case, and the operator missed it: a banner printed once per suite, after real
    per-unit planning, deep in a long log, is not a control for a CAMPAIGN the way it might be
    for one smoke-test invocation. So a `--sweep` on a dirty tree now refuses outright, before
    a single suite is dispatched, rather than banner-and-proceed.

    `--allow-dirty` restores exactly today's prior behaviour, for the one legitimate case: a
    dirty tree is not a mistake when the intent is deliberately collecting dev- training data.

    A plain (non-`--sweep`) `pi run` is UNCHANGED either way -- this is a campaign-launch
    refusal, not a rule about the tree.
    """
    if allow_dirty or not gi.dirty:
        return None
    if gi.dirty_files:
        listing = "\n    ".join(gi.dirty_files)
        file_block = (
            f"  {len(gi.dirty_files)} dirty file(s) (git status --porcelain -uno):\n    {listing}"
        )
    else:
        # git itself failed (absent, not a repo, timed out) -- GitInfo.dirty_files is then
        # empty by construction (see GitInfo's docstring), so there is no list to print and no
        # honest count either; say so rather than print "0 files" beside a dirty=True tree.
        file_block = (
            "  dirty, but git could not enumerate the files (git is absent, broken, or this "
            f"is not a repo at {root}) -- refusing on dirty=True alone"
        )
    return (
        f"REFUSING --sweep on a dirty tree at {root}:\n"
        f"{file_block}\n"
        "  Every unit this campaign launches would be stamped dev- and therefore "
        "training-only, never an eval arm. Two ways forward:\n"
        "    (a) commit or set aside the changes above, then re-run; or\n"
        "    (b) launch from a pinned clean worktree:\n"
        "          git worktree add --detach <path> <sha>\n"
        "          ln -s <this root>/data/corpora <path>/data/corpora   # and data/raw\n"
        "          pi run --sweep ... --root <path> --runs-root <this root>/runs\n"
        "  Pass --allow-dirty to proceed anyway and stamp every unit dev- (training-only), "
        "the way this command did before today."
    )


def _resolve_corpus(root: Path, suite_id: str, explicit: str | None) -> Path:
    """Locate the built corpus for a suite.

    Self-sourced suites (tau2, pare, drgym) read an upstream checkout or a hosted index rather
    than a corpus this repo built, so demanding a data/corpora/<suite> directory for them would
    block a runnable sweep on a directory that is never supposed to exist. They get the base
    path unchecked; their adapters raise with an actionable remedy (TAU2_DATA_DIR, a DRGym key)
    if what they actually need is missing.
    """
    from pi_run.worker import SELF_SOURCED

    if suite_id in SELF_SOURCED and not explicit:
        return root / "data" / "corpora" / suite_id
    if explicit:
        p = Path(explicit)
        if not (p / "tasks.jsonl").exists() and (p.parent / "tasks.jsonl").exists():
            p = p.parent
        return p
    base = root / "data" / "corpora" / suite_id
    cands = sorted(
        (d for d in base.glob("*") if (d / "tasks.jsonl").exists()), key=lambda d: d.name
    )
    if not cands:
        raise SystemExit(
            f"no corpus for suite {suite_id!r} under {base}. "
            "Build it first (pi_eval.build.synth_build.build for synth) or pass --corpus."
        )
    if len(cands) > 1:
        # Ambiguity is a configuration error, not something to guess at: two corpus hashes
        # are two different frozen corpora and picking the newer one silently changes the
        # corpus_hash inside every run identity.
        raise SystemExit(
            f"{len(cands)} corpora under {base}: {[c.name for c in cands]}. Pass --corpus."
        )
    return cands[0]


def _scrub_gold_root() -> bool:
    """Returns True if it had to remove one, so the caller can say so out loud."""
    if os.environ.pop("PI_GOLD_ROOT", None):
        return True
    return False


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------- pi run


def _fmt_cap(v: float | None) -> str:
    return "none" if v is None else f"${v:.2f}"


def _load_questions_files(specs: Sequence[str], arm_ids: Sequence[str]) -> dict[str, dict]:
    """Parse `--questions-from-file`, which may be repeated and may be scoped to one arm.

        --questions-from-file data/gold/q.json                 # every replay arm
        --questions-from-file gold_evidence=data/gold/qg.json  # this arm only
        --questions-from-file oracle_vreq=data/gold/qv.json

    THE SCOPED FORM IS WHY THIS EXISTS. `UnitSpec.questions` is keyed by task_id alone and
    `plan()` hands the SAME mapping to every arm with `requires_questions=True`, so a single
    unscoped file gives `gold_evidence` and `oracle_vreq` byte-identical inputs -- which is
    exactly the collapse `pi gold questions` was written to remove, reintroduced one layer up.
    The two ceiling arms bound different things and must be seeded from different files.

    Returns `{arm_id_or_"": {task_id: [q, ...]}}`. An unscoped entry is the default for any arm
    with no entry of its own.
    """
    out: dict[str, dict] = {}
    for spec in specs:
        arm, sep, path = spec.partition("=")
        if not sep:
            arm, path = "", spec
        elif arm not in arm_ids:
            raise SystemExit(
                f"--questions-from-file {spec!r}: {arm!r} is not among the arms being run "
                f"({sorted(arm_ids)}). A file scoped to an arm nobody runs is silently ignored."
            )
        if arm in out:
            raise SystemExit(f"--questions-from-file: {arm or '<default>'} given twice")
        out[arm] = _load_questions_file(Path(path))
    return out


def _load_questions_file(path: Path) -> dict[str, list[str]]:
    """`{task_id: [question, ...]}` from a JSON file, validated on the way in.

    Validated rather than trusted because this file is the INPUT to the ceiling arms, and a
    ceiling arm fed a silently-empty pool does not fail: it stops on turn 0, scores like
    `drafter_only`, and reports "no headroom" -- the exact conclusion that would cancel the
    musique track. A wrong shape has to be an error here, not a null result three hours later.
    """
    obj = json.loads(path.read_text())
    if isinstance(obj, Mapping) and "questions" in obj:
        obj = obj["questions"]  # allow a wrapper carrying provenance alongside the payload
    if not isinstance(obj, Mapping):
        raise SystemExit(f"{path}: expected an object mapping task_id -> [question, ...]")
    out: dict[str, list[str]] = {}
    for tid, qs in obj.items():
        if isinstance(qs, str) or not isinstance(qs, Sequence):
            raise SystemExit(f"{path}: task {tid!r} maps to {type(qs).__name__}, not a list")
        vals = [str(q) for q in qs if str(q).strip()]
        if vals:
            out[str(tid)] = vals
    if not out:
        raise SystemExit(f"{path}: no task carries a non-empty question list")
    return out


# Suites whose reward cannot be produced by the flat retrieve/draft/answer loop, and the door
# each one has instead. `pi run` refuses these BEFORE it prints a unit count.
#
# The refusal already existed one layer down -- `UserBenchSuite.retriever()` and `.actuator()`
# both raise, deliberately, so that a flat run can never emit a complete all-zero results file
# for a benchmark whose reward lives inside a gym. What did not exist was a place for that
# raise to land. `run_unit` calls `suite.retriever(...)` OUTSIDE its try, so the exception went
# up through the process pool and out of `main` as a `concurrent.futures` traceback -- after
# the header line had already announced `units=2`. An operator saw a crash and a half-made runs
# directory, which is what a bug looks like, rather than a refusal, which is what it was.
#
# Read off `SuiteSpec.driven_by` rather than a second hard-coded list: the registry already
# records that userbench is "gym" and tau2 is "orchestrator", and a duplicate list is a list
# that goes stale.
#
# ONLY "gym", AND NOT "orchestrator". tau2 is orchestrator-driven and IS runnable through
# `pi run`: `worker.run_unit` special-cases `DIALOGUE_SUITES` on its first line and hands the
# unit to `stages.tau2_runner.run_tau2_unit`, and `conf/grids/tier1_trained.yaml` and
# `tier2_confirmatory.yaml` both drive it exactly that way. The first draft of this table
# refused both roles on the symmetry of the two docstrings, which would have taken down every
# live tau2 campaign in the name of tidiness. A suite belongs here only when `run_unit` has
# NO path for it -- which is the difference between "needs a different driver, and has one"
# and "has no driver at all".
_FLAT_RUN_DOORS: Mapping[str, str] = {
    "gym": (
        "its reward is produced inside a gym environment, whose retrieval ([search]) and "
        "actuation ([answer]) are scored actions rather than free local operations. Drive "
        "UserBenchSuite.session(task_id) instead -- see docs/DATA.md."
    ),
}


def flat_run_refusal(suite_id: str) -> str:
    """Why `pi run --suite <id>` cannot run this suite flat, or "" if it can.

    Returns a MESSAGE rather than raising, so the caller decides the exit code and nothing
    here has to know about argparse. Empty string is the common case and means proceed.
    """
    spec = REGISTRY.get(suite_id)
    reason = _FLAT_RUN_DOORS.get(getattr(spec, "driven_by", ""), "") if spec else ""
    if not reason:
        return ""
    return (
        f"pi run --suite {suite_id}: refusing a flat run. {reason}\n"
        f"  ({suite_id} is registered driven_by={spec.driven_by!r}; "
        "`pi suites map` prints the roles.)"
    )


def cmd_run(a: argparse.Namespace) -> int:
    had_gold = _scrub_gold_root()
    if had_gold:
        print("PI_GOLD_ROOT was set; unset for this process tree (rollouts must not read gold)")

    # A malformed cap is an operator typo, and it must read as a refusal rather than as a
    # traceback out of pi_run.sweep. Resolved here, before any grid is loaded or any suite
    # is dispatched.
    try:
        spend_cap(getattr(a, "spend_cap", None))
    except SpendCapInvalid as exc:
        print(f"  refusing: {exc}", file=sys.stderr)
        return 2

    root = Path(a.root).resolve() if a.root else repo_root()

    grid = None
    if getattr(a, "sweep", None):
        from pi_run import grids

        grid = grids.load(a.sweep)
        # A grid is the unit the preregistration freezes, so its content hash rides into the
        # manifest: a silently widened grid then changes run identity instead of appearing as
        # extra rows nobody notices.
        print(
            f"grid={grid.name} sha={grid.source_sha256[:12]} suites={list(grid.suites)} "
            f"arms={len(grid.arms)} units={grid.n_units}"
            + ("  [PILOT: ids are burned]" if grid.pilot else "")
            + ("  [EXPLORATORY]" if grid.exploratory else "")
        )

        # REFUSE a --sweep campaign on a dirty tree before a single suite is dispatched --
        # see `dirty_sweep_refusal`'s docstring for the incident. A plain `pi run` is untouched.
        refusal = dirty_sweep_refusal(
            git_info(str(root)), allow_dirty=bool(getattr(a, "allow_dirty", False)), root=root
        )
        if refusal:
            print(refusal, file=sys.stderr)
            return 2

        # THE SPEND CAP IS FOR THE GRID, NOT FOR EACH SUITE IN IT.
        #
        # This loop calls `cmd_run` once per suite, and each call passed `a.spend_cap`
        # unchanged -- so a 3-suite grid like tier1_confirmatory (musique, strategyqa, wiki2)
        # would honour `--spend-cap 50` three separate times and could bill $150. The flag says
        # "cap", the operator reads "cap", and the invoice is a multiple of it.
        #
        # The remaining budget is threaded through instead, so the LAST suite gets whatever the
        # first two left. A suite reached with nothing left is refused rather than run at zero
        # cap, because a cap of 0.0 is indistinguishable from "no cap" in the flag's own
        # encoding (None means unlimited).
        # REFUSE a flag that contradicts the grid, BEFORE any suite runs and therefore
        # before any money is spent. Silently discarding these is how the canary would have
        # run at seed 0 -- a warm seed -- while its operator had typed `--seeds 7`.
        #
        # ONCE PER SUITE, over the child namespace each suite would actually receive, because
        # `k` is per suite (`grid.k_for`) and `suite` is the loop variable: a grid-wide diff
        # accepted `--k 5` on tier1_confirmatory and still ran strategyqa at 2. Every suite is
        # checked before the FIRST one is dispatched, so a conflict that only the third suite
        # would have hit does not cost the first two.
        clashes: tuple[str, ...] = ()
        for _suite_id in grid.suites:
            for _msg in grid_flag_conflicts(a, grid=grid, suite_id=_suite_id):
                if _msg not in clashes:
                    clashes = (*clashes, _msg)
        if clashes:
            print(
                f"  refusing: these flags contradict grid {grid.name}:\n    "
                + "\n    ".join(clashes)
                + "\n  Edit the grid, or drop the flag. `--arm` is the one flag that FILTERS "
                "a grid rather than contradicting it.",
                file=sys.stderr,
            )
            return 2

        rc = 0
        remaining = spend_cap(getattr(a, "spend_cap", None))
        for suite_id in grid.suites:
            if remaining is not None and remaining <= 0:
                print(
                    f"  SKIPPED suite {suite_id}: the grid's spend cap is exhausted. "
                    "Raise --spend-cap or run the remaining suites separately.",
                    file=sys.stderr,
                )
                rc |= 1
                continue
            # EVERY grid override lives in `grid_child_namespace`, and `grid_discarded_flags`
            # diffs its output against `a` -- so a field added there is refused-if-contradicted
            # without a second list to update. Do not assign a grid value to `sub` here: this
            # loop is outside that diff, and a `sub.<field> = grid.<field>` added below is
            # exactly the silent discard the diff exists to catch (asserted by AST in
            # tests/test_sweep_flag_conflict.py).
            sub = grid_child_namespace(a, grid, suite_id)
            sub.spend_cap = remaining
            # `--arm` FILTERS THE GRID, it is not silently discarded.
            #
            # This was `sub.arm = list(grid.arms)`, which threw away an explicit --arm. An
            # operator narrowing a sweep to one cheap arm -- exactly what you do to rehearse a
            # grid safely -- got all 13 arms instead, including every billed one. I did this to
            # myself while testing the spend cap: `--arm fake_chain` on tier1_confirmatory ran
            # the full arm list and billed $0.72 before the cap stopped it.
            #
            # An --arm outside the grid is REFUSED rather than silently empty, because "run
            # nothing" and "run everything" are both wrong answers to a typo.
            wanted = [x for x in (getattr(a, "arm", None) or []) if x]
            if wanted:
                unknown = sorted(set(wanted) - set(grid.arms))
                if unknown:
                    print(
                        f"  --arm {unknown} is not in grid {grid.name} (has: {sorted(grid.arms)})",
                        file=sys.stderr,
                    )
                    return 2
                sub.arm = [x for x in grid.arms if x in set(wanted)]
            else:
                sub.arm = list(grid.arms)
            rc |= cmd_run(sub)
            if remaining is not None:
                remaining = max(0.0, remaining - float(getattr(sub, "_billed_usd", 0.0)))
                print(f"  grid budget: ${remaining:.2f} left after {suite_id}", file=sys.stderr)
        return rc

    # The parser's defaults are None so an explicit flag is detectable; apply them here.
    for _name, _value in resolved_run_defaults(a).items():
        setattr(a, _name, _value)

    refusal = flat_run_refusal(a.suite)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    corpus = _resolve_corpus(root, a.suite, a.corpus)
    run_dir = runs_root(root, a.runs_root)
    cache = Path(a.cache_root) if a.cache_root else cache_root(os.environ.get("PI_CACHE_ROOT"))
    run_dir.mkdir(parents=True, exist_ok=True)

    from pi_run.worker import load_suite

    suite = load_suite(a.suite, str(corpus))
    # SPLIT-AWARE SELECTION. This was `list(suite.task_ids())[: a.n]` -- the first N in corpus
    # order, with `split` consulted nowhere in this file. `tier1_trained` therefore evaluated
    # the trained arm on 55 of 97 musique tasks that `pinq.splitting.split_of` calls `train`,
    # and nothing eval-side reads the `split` column, so it could not be repaired afterwards.
    task_ids = (
        list(a.task)
        if a.task
        else _select_task_ids(
            suite,
            a.suite,
            n=a.n,
            split=getattr(a, "split", None),
            offset=int(getattr(a, "task_offset", 0) or 0),
        )
    )
    arm_ids = list(a.arm) or list(arm_table.arm_ids())
    seeds = [int(s) for s in a.seeds]

    questions: dict[str, list[str]] = {}
    by_arm: dict[str, dict] = {}
    from_file = list(getattr(a, "questions_from_file", None) or [])
    if from_file:
        # An EXPLICIT question list, from `pi gold questions` or any other producer. It wins
        # over mining a prior sweep, and that is the whole point of the flag: the two ceiling
        # arms were both seeded from `recorded_questions(arm_id="inquirer_prompted")`, i.e.
        # from the treatment's own questions, which made `gold_evidence` input-identical to
        # `parallel_replay` and the headroom gate a tautology.
        by_arm = _load_questions_files(from_file, arm_ids)
        questions = by_arm.get("", {})
        for scope, qs in sorted(by_arm.items()):
            print(f"questions for {scope or '<all replay arms>'}: {len(qs)} tasks")
    seed_from = (
        None
        if from_file
        else (
            getattr(a, "seed_questions_from", None)
            or (str(run_dir) if any(arm_table.get(x).requires_questions for x in arm_ids) else None)
        )
    )
    questions_by_seed: dict[int, dict[str, list[str]]] = {}
    if seed_from and Path(seed_from).exists():
        from pinq_expt.policies.controls import recorded_questions

        mine_arm = getattr(a, "seed_questions_arm", None) or "inquirer_prompted"
        # PER SEED, because a replay arm is paired with the treatment run at ITS OWN seed and
        # the treatment asks different questions at each. Mining once, keyed by task, handed
        # seed 1 the seed-0 list and made parallel_replay's "identically zero by construction"
        # delta non-zero -- an indictment of the harness manufactured by the seeding.
        questions_by_seed = {
            sd: recorded_questions(seed_from, arm_id=mine_arm, suite_id=a.suite, seed=sd)
            for sd in seeds
        }
        questions = recorded_questions(seed_from, arm_id=mine_arm, suite_id=a.suite)
        got = {sd: len(q) for sd, q in questions_by_seed.items() if q}
        if got:
            print(f"seeded replay arms from {seed_from}: tasks per seed {got}")
        elif questions:
            print(f"seeded replay arms from {seed_from}: {len(questions)} tasks (no seed match)")

    replay = [x for x in arm_ids if arm_table.get(x).requires_questions]
    if replay and not questions:
        # Say it once, here, rather than letting four arms fail per task with the same message.
        src = f"--questions-from-file {from_file}" if from_file else str(seed_from)
        print(
            f"NOTE: {replay} replay a recorded question list and none was found in {src}. "
            "Run an inquirer arm first, or pass --seed-questions-from / --questions-from-file."
        )

    gi = git_info(str(root))
    concurrency = max_concurrency(a.concurrency)
    specs = plan(
        suite_id=a.suite,
        corpus_dir=str(corpus),
        task_ids=task_ids,
        arm_ids=arm_ids,
        seeds=seeds,
        runs_root=str(run_dir),
        cache_root=str(cache),
        max_turns=a.max_turns,
        k=a.k,
        budget_cap=a.budget_cap,
        pilot=a.pilot,
        resume=not a.no_resume,
        code_version=gi.sha,
        dirty=gi.dirty,
        dirty_files=gi.dirty_files,
        concurrency=concurrency,
        questions=questions or None,
        questions_by_seed=questions_by_seed or None,
        # Per-arm overrides. Without these the two ceiling arms receive one mapping and
        # are input-identical, which is the tautology `pi gold questions` exists to break.
        questions_by_arm={k: v for k, v in by_arm.items() if k},
        timeout_s=getattr(a, "unit_timeout", None),
        exploratory=bool(getattr(a, "exploratory", False)),
        grid_sha256=getattr(a, "grid_sha256", ""),
        grid_name=getattr(a, "grid_name", ""),
    )
    print(
        f"suite={a.suite} tasks={len(task_ids)} arms={len(arm_ids)} seeds={len(seeds)} "
        f"units={len(specs)} concurrency={concurrency} "
        f"cap={_fmt_cap(spend_cap(getattr(a, 'spend_cap', None)))} code={gi.sha[:12]}"
        + (" DIRTY -> dev- run ids" if gi.dirty else "")
    )
    # The callback is not decoration. Without one a three-hour sweep printed its header,
    # went silent, and printed its summary; the only way to distinguish a wedged tunnel
    # from slow progress was to sit watching `ls runs/ | wc -l`.
    results = run_sweep(
        specs,
        concurrency=concurrency,
        cap_usd=getattr(a, "spend_cap", None),
        on_result=progress_printer(len(specs)),
    )
    summary = summarize(results)
    # What this invocation actually BILLED, handed back to the grid loop so a multi-suite grid
    # can enforce ONE cap across its suites rather than one cap each. See the loop above.
    a._billed_usd = float(summary.get("usd_billed") or 0.0)
    for r in results:
        if r.get("status") == "error":
            print(f"  error {r.get('run_id')}: {r.get('error')}", file=sys.stderr)
    _emit({"summary": summary, "runs_root": str(run_dir), "corpus": str(corpus)})
    # summary["ok"] resolves RESUMED units to the status they were resumed from and counts
    # timeouts, so a sweep of previously-failed units cannot exit 0 by having re-read them.
    return 0 if summary["ok"] else 1


# --------------------------------------------------------------------------- pi compact


def _select_task_ids(suite, suite_id: str, *, n: int, split: str | None, offset: int) -> list[str]:
    """The suite's task ids, filtered by split and offset before the head-N truncation."""
    from pinq.splitting import split_of

    ids = list(suite.task_ids())
    if split:
        ids = [t for t in ids if split_of(suite_id, t, _template_of(suite, t)) == split]
    return ids[offset:][:n]


def _template_of(suite, task_id: str) -> str | None:
    # DELEGATES rather than reimplementing. This was `str(fn(task_id)) or None`, and
    # `str(None)` is the truthy string "None", so the `or None` never fired: every task of a
    # suite whose `template_id()` returns None collapsed onto the one split key
    # "<suite>|None" and so into ONE bucket. Selection then disagreed with the manifest,
    # which stamps via `template_id_of` and buckets per task -- and
    # `pi run --suite tau2_retail --split test` selected 0 of 114 tasks and exited ok=true.
    from pi_run.manifest import template_id_of

    return template_id_of(suite, task_id)


# The lock directory's name, when PI_STORE_LOCK does not say otherwise. BESIDE THE STORE, not
# in a session scratchpad: a lock nobody else can name is not a lock. Measured 2026-09-18 --
# two lanes each "held the lock" that evening, in two different per-session scratchpad
# directories, and compacted the same scores/parquet seventeen seconds apart.
STORE_LOCK_DIRNAME = ".store.lock"


def _store_lock_state(out: Path) -> tuple[bool, str, Path]:
    """(held, owner, path). Held means the directory exists AND names an owner.

    `mkdir` is the atomic step and writing `owner` is a second one, so a directory without an
    owner file is a lock still being taken or one a crash left behind. Treating it as held
    would let a dead lane block every lane after it; treating it as free is the same race the
    guard exists to stop -- so it is reported as NOT held and the operator is told the path,
    which makes a stale lock visible rather than silent.
    """
    env = os.environ.get("PI_STORE_LOCK")
    lock = Path(env) if env else Path(out) / STORE_LOCK_DIRNAME
    owner_file = lock / "owner"
    if not lock.is_dir() or not owner_file.is_file():
        return False, "", lock
    owner = owner_file.read_text().strip()
    return bool(owner), owner, lock


def _require_store_lock(out: Path, root: Path, *, what: str) -> int:
    """0 to proceed, non-zero to refuse. Only the SHARED store is guarded.

    An isolated `--out` is not the shared store: `artifacts/frames/FRAMES.md` builds one
    outside the repository on purpose, and a guard that fired there would be the guard people
    route around.
    """
    shared = (root / "scores" / "parquet").resolve()
    try:
        target = Path(out).resolve()
    except OSError:  # pragma: no cover - resolve() on a path we are about to create
        target = Path(out).absolute()
    if target != shared:
        return 0
    held, owner, lock = _store_lock_state(out)
    if held:
        print(f"shared store is locked by {owner}", file=sys.stderr)
        return 0
    print(
        f"no store lock held; take it with mkdir {lock}\n"
        f"  {what} refused: it would write {target}, which other lanes compact and score.\n"
        f"  Two lanes raced this directory on 2026-09-18 and the later batch of renames\n"
        f"  silently discarded the earlier one's whole result. Take the lock, write an owner\n"
        f"  file inside it, and release both when you are done.",
        file=sys.stderr,
    )
    return 3


def cmd_compact(a: argparse.Namespace) -> int:
    # The ONLY place this CLI touches pi_eval. Compaction is the scoring process.
    from pi_run.compact import compact

    root = Path(a.root).resolve() if a.root else repo_root()
    run_dir = runs_root(root, a.runs_root)
    out = Path(a.out) if a.out else root / "scores" / "parquet"
    # BEFORE the scan, not after: a refusal that has already replaced six tables is not one.
    rc = _require_store_lock(out, root, what="pi compact")
    if rc:
        return rc
    res = compact(run_dir, out, include_dev=not a.exclude_dev, replace=a.replace)
    if res.orphaned_score_rows:
        print(
            f"WARNING: {res.orphaned_score_rows} score rows now reference a run that is not in "
            f"runs.parquet. Every reporting query inner-joins the two, so those rows appear in "
            f"NO table. Re-compact the runs-root that produced them, or re-score.",
            file=sys.stderr,
        )
    _emit(res.as_dict())

    # A COMPACTION THAT LOST TURN ROWS MUST NOT EXIT 0. It did once, on 2026-09-18: 21,001 runs
    # whose turns.jsonl holds 134,370 lines contributed rows keyed to a NULL run_id, every
    # inner join dropped them, and `pi score` then read those runs as n_turns=0 and
    # evidence_coverage=0.0 -- a wrong number wearing the shape of a right one. Exit code,
    # not a warning: the sweep scripts check it and a warning on stderr is read by nobody.
    lost = res.runs_with_turns_on_disk_but_none_compacted
    if lost:
        print(
            f"ERROR: {len(lost)} run(s) hold a non-empty turns.jsonl and have NO row in "
            f"turns.parquet. Their turn, ask and evidence counts read as 0 in every table. "
            f"First: {', '.join(lost[:5])}",
            file=sys.stderr,
        )
        return 1

    return 0


# --------------------------------------------------------------------------- pi env doctor


# Environment variables that are not settings but PREREQUISITES: without them a whole suite or
# tier cannot run at all, and the failure surfaces deep inside an adapter constructor several
# minutes into a sweep. Doctor's job is to move that discovery to second zero.
_SUITE_ENV = {
    "TAU2_DATA_DIR": "tau2 (the wheel ships no data/; point at a repo checkout)",
    "PARE_BENCHMARK_SPLITS_DIR": "pare",
    "DRGYM_API_KEY": "drgym / Tier 3 (hosted search index)",
}


def _canary_carriers(root: str | Path = ".") -> dict[str, str]:
    """suite -> which leakable string carries its nonce, or why none does.

    A single `canaries_configured: 16752` reads as uniform coverage and is not. `write_graphs`
    mints one nonce per graph and threads it into `gold_answer` -- chosen because it is the one
    gold string that NEVER legitimately reaches a model, so a hit has no benign explanation --
    but only when `gold_answer` was already non-empty for that row.

    tau2, drgym and the tau2_* transfer suites HAVE NO `gold_answer`: their gold is entirely
    needs, and their only leakable string is `gold_text`, which legitimately reaches a model
    twice over -- the judges are handed gold_text key points, and `pi gold questions` emits
    gold_text to seed the two ceiling arms. A nonce there would abort judging and both ceiling
    arms on their first call. Measured 2026-09-17: these are the tool-use / transfer suites, not
    the primary QA endpoint -- frames, musique, strategyqa, synth and wiki2 all DO carry the
    nonce in `gold_answer`. So layer 4 cannot fire on a text leak on the blind suites, and no
    amount of registry size changes that; that is a fact about the experiment, reported here
    rather than left to be inferred from a count.

    `gold_answer` emptiness is therefore the structural, code-derived signal this function uses
    to tell that legitimate blindness apart from an actual fault: a suite whose sampled rows have
    a NON-empty `gold_answer` that still carries no nonce did not go blind by design -- minting
    or threading failed for it -- and is reported as "FAULT", never folded into the same "NONE"
    bucket a caller might otherwise treat as always-benign.

    Returns {} only when no suite directories exist under <root>/data/gold/graphs at all; a
    caller must not treat that the same as "every suite was checked and found legitimately
    blind" -- see `_firewall_armed`, which is what draws that line.
    """
    import json as _json
    import re as _re

    tok = _re.compile(r"PINQCANARY_[0-9A-F]{16}")
    out: dict[str, str] = {}
    gdir = Path(root) / "data" / "gold" / "graphs"
    for suite_dir in sorted(p for p in gdir.glob("*") if p.is_dir()):
        fields: set[str] = set()
        saw_nonempty_answer = False
        for f in sorted(suite_dir.glob("*.jsonl")):
            for line in f.open():
                if not line.strip():
                    continue
                row = _json.loads(line)
                if str(row.get("gold_answer") or ""):
                    saw_nonempty_answer = True
                for k, v in row.items():
                    if isinstance(v, str) and k != "gold_canary" and tok.search(v):
                        fields.add(k)
                break  # one row per file is enough: the carrier is structural, not per-task
        if fields:
            out[suite_dir.name] = ", ".join(sorted(fields))
        elif saw_nonempty_answer:
            out[suite_dir.name] = (
                "FAULT - gold_answer is non-empty on sampled rows but carries no nonce; "
                "minting or threading failed for this suite specifically -- rebuild its gold"
            )
        else:
            out[suite_dir.name] = (
                "NONE - no gold_answer; gold_text reaches judges and the ceiling arms legitimately"
            )
    return out


def _firewall_armed(root: Path, canaries: Sequence[str]) -> tuple[bool, str]:
    """(is this verify-firewall scan trustworthy, why not).

    Mirrors `pinq.tripwire.assert_armed`'s own rule, for the same reason: a fresh checkout with
    nothing under <root>/data/gold has nothing to leak and must stay runnable -- a guard that
    fires on a clean clone is a guard someone deletes. So only a root where gold WAS built but
    this scan cannot see what it needs to see is refused; a suite that legitimately carries no
    nonce (see `_canary_carriers`) is never treated as a reason to refuse.
    """
    gold_dir = Path(root) / "data" / "gold"
    if not gold_dir.is_dir():
        return True, f"no data/gold under {root}; nothing built, nothing to leak"

    graphs_dir = gold_dir / "graphs"
    carriers = _canary_carriers(root)
    if not carriers:
        return False, (
            f"gold is built under {gold_dir} but no suite directories were found under "
            f"{graphs_dir}. Layer 4 would have nothing to scan for and report clean on every "
            f"request regardless of content. Rebuild gold (`pi data build`) under {root}, or "
            "point --root at the tree that actually holds the built graphs."
        )

    faults = sorted(s for s, v in carriers.items() if v.startswith("FAULT"))
    if faults:
        return False, (
            f"under {graphs_dir}, {len(faults)} suite(s) should carry a nonce and do not: "
            f"{', '.join(faults)}. Minting or threading likely failed for them specifically -- "
            "this is not the same as a suite that is blind by design. Rebuild gold for those "
            "suites before trusting this scan."
        )

    if not canaries:
        cdir = Path(root) / "data" / "canaries"
        return False, (
            f"gold is built under {gold_dir} but no canary strings were loaded from {cdir} "
            "and no PI_CANARY_SALT or --canary was given. The scan below would compare every "
            f"cached request against an empty needle set and report clean regardless of "
            f"content. Rebuild gold under {root} to (re)populate the registry, or set "
            "PI_CANARY_ROOT / PI_CANARY_SALT."
        )

    return True, f"{len(carriers)} suite(s) classified, {len(canaries)} canaries loaded from {root}"


def _probe_roles(pins: Mapping[str, str], timeout: float = 30.0) -> dict[str, Any]:
    """One tiny completion per DISTINCT pinned model. Reachability, measured.

    Doctor is documented as reporting "provider reachability" and made zero network calls: it
    checked that an env var was non-empty, which is true of a typo, an expired key, a proxy
    that is down and a VPN that is not connected. The whole point of running doctor is that
    those four are the reasons a sweep fails at 02:00, and a green report that cannot
    distinguish them is worse than no report -- it moves the diagnosis after the spend.

    Deduplicated by model id: five roles on one model is one call, not five. Errors are
    reported as the exception type and message, never re-raised; doctor reports, it does not
    crash.
    """
    from pinq.budget import BudgetLedger
    from pinq_adapters.llm.litellm_client import MeteredClient

    out: dict[str, Any] = {}
    by_model: dict[str, list[str]] = {}
    for role, model in sorted(pins.items()):
        if model:
            by_model.setdefault(model, []).append(role)
    for model, roles in sorted(by_model.items()):
        # A throwaway ledger with an unreachable cap: the probe must not be able to trip a
        # budget, and nothing reads this ledger back.
        client = MeteredClient(BudgetLedger(cap=10**9), timeout=timeout, max_attempts=1)
        t0 = time.perf_counter()
        try:
            text, tel = client.complete(
                role=roles[0],
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                seed=0,
                # Not 5. A reasoning model bills its reasoning channel inside max_tokens, so a
                # 5-token budget came back HTTP 200 with an empty string -- indistinguishable
                # from a model that has stopped emitting content, which is one of the failures
                # this probe exists to catch. 32 content tokens leaves room for the channel and
                # still costs about a hundredth of a cent.
                max_tokens=32,
            )
            out[model] = {
                "roles": roles,
                "ok": True,
                "http_status": tel.http_status,
                "ms": int((time.perf_counter() - t0) * 1000),
                "tok_total": tel.tok_prompt + tel.tok_completion + tel.tok_reasoning,
                "reply": text.strip()[:40],
            }
        except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
            out[model] = {
                "roles": roles,
                "ok": False,
                "ms": int((time.perf_counter() - t0) * 1000),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
    return out


def cmd_env_doctor(a: argparse.Namespace) -> int:
    # Pricing is dependency-light; the client needs the `run` extras. Doctor must still work
    # in a broken environment, because a broken environment is when you run it.
    from pinq_adapters.llm.pricing import PriceTable

    try:
        from pinq_adapters.llm.litellm_client import provider_key_status, role_pins
    except ImportError:  # `run` extras absent

        def provider_key_status():
            return {p: bool(os.environ.get(v)) for p, v in sorted(_KEY_ENV.items())}

        def role_pins():
            return {r: os.environ.get(v, "") for r, v in sorted(_ROLE_ENV.items())}

    root = repo_root()
    try:
        pt = PriceTable.load()
        price = {"version": pt.version, "path": pt.path, "models": len(pt.models)}
    except Exception as exc:  # noqa: BLE001 - doctor reports, never crashes
        price = {"error": f"{type(exc).__name__}: {exc}"}

    optional: dict[str, bool] = {}
    for mod in ("litellm", "tenacity", "pyarrow", "duckdb", "pandas", "yaml"):
        try:
            __import__(mod)
            optional[mod] = True
        except Exception:  # noqa: BLE001
            optional[mod] = False

    reach: dict[str, Any] = {}
    if not getattr(a, "no_probe", False):
        try:
            reach = _probe_roles(role_pins(), timeout=float(getattr(a, "probe_timeout", 30.0)))
        except ImportError as exc:  # `run` extras absent
            reach = {"<client>": {"ok": False, "error": f"ImportError: {exc}"}}

    gi = git_info(str(root))
    _emit(
        {
            "python": sys.version.split()[0],
            "repo_root": str(root),
            "code_version": gi.sha,
            "dirty": gi.dirty,
            # Presence only. A value is never printed, hashed into an id, or written to a run.
            "provider_keys_present": provider_key_status(),
            "role_pins": role_pins(),
            "price_table": price,
            "cache_root": str(cache_root()),
            "runs_root": str(runs_root(root)),
            "PI_GOLD_ROOT_set": bool(os.environ.get("PI_GOLD_ROOT")),
            "PI_MAX_CONCURRENCY": max_concurrency(),
            "PI_UNIT_TIMEOUT_S": unit_timeout_s(),
            "PI_SPEND_CAP_USD": spend_cap(),
            # Presence only, same rule as the provider keys: a path is not a secret but a
            # value here would still be machine-specific noise in a shared report.
            "suite_env_present": {
                var: {"set": bool(os.environ.get(var)), "needed_by": why}
                for var, why in sorted(_SUITE_ENV.items())
            },
            "modules": optional,
            "arms": list(arm_table.arm_ids()),
            "reachability": reach,
        }
    )
    # Non-zero when a probe was requested and something is unreachable: doctor is meant to be
    # the first line of a runbook, and a runbook step that always exits 0 is a comment.
    return 0 if all(v.get("ok") for v in reach.values()) else 2


# --------------------------------------------------------------------------- pi cache stats


def cmd_verify_tau2(a: argparse.Namespace) -> int:
    """Replay each tau2 task's OWN answer key and check the gold DB hash comes back.

    THE ONLY CHECK THAT SEPARATES A BROKEN HARNESS FROM A BAD POLICY, and it costs zero
    tokens. `tau_reward` is the paper's tau2 primary endpoint and had never been validated
    against upstream even once; if replaying the evaluation criteria does not reproduce the
    gold hash, then `env_kwargs`, the `read_log_allowlist`, `initial_state` or the synthesized
    trajectory is wrong, and every tau2 number ever produced here measures our plumbing.

    Exit 1 on any task that fails. A task whose criteria carry no actions is reported as
    SKIPPED and does not fail the check: absent and failing must not look alike.

    `--suite tau2_retail` replays the OTHER domain. It costs nothing and it was unreachable:
    both suites route through one dialogue driver, so a harness defect on the retail side is
    exactly as invisible as the banking ones this command was written to expose.
    """
    from pi_run.stages.tau2_runner import NO_ACTIONS, replay_gold
    from pi_run.worker import load_suite

    # EITHER tau2 SUITE, through the same loader the sweep uses. This built `Tau2Suite()`
    # unconditionally, so the only instrument that can tell a broken harness from a bad policy
    # was pointed at banking and banking alone -- and retail, whose grading path was wrong in
    # a way this check is precisely designed to catch, had never been run through it once.
    try:
        suite = load_suite(a.suite, "")
    except Exception as exc:  # noqa: BLE001 - doctor-style reporting, never a traceback
        _emit({"ok": False, "suite": a.suite, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    ids = list(suite.task_ids())
    if a.n:
        ids = ids[: a.n]
    rows = replay_gold(suite, ids)
    failed = [r for r in rows if r.db_reward is not None and not r.ok]
    # THREE OUTCOMES, NOT TWO. `db_reward is None` covers two unlike events: a task whose
    # criteria carry NO ACTIONS (nothing to grade) and a task whose grade RAISED (something to
    # grade, and this harness could not). Both were reported as "skipped" and neither failed
    # the check -- so on retail, 15 tasks whose trajectory replay died inside upstream's
    # `set_state` printed as a clean `reproduced: 97, ok: true`. This command exists to tell a
    # broken harness from a bad policy; laundering the broken half into a skip list defeats it.
    skipped = [r for r in rows if r.db_reward is None and r.error == NO_ACTIONS]
    errors = [r for r in rows if r.db_reward is None and r.error != NO_ACTIONS]
    graded = len(rows) - len(skipped) - len(errors)
    _emit(
        {
            # REPORTED, because the two suites have different denominators and a bare
            # "97 of 97" with no suite beside it is a number without provenance.
            "suite": suite.suite_id,
            "domain": suite.domain,
            "tasks": len(rows),
            "graded": graded,
            "reproduced": graded - len(failed),
            # NOTHING TO GRADE. Distinct from `errors` below: these tasks carry no evaluation
            # actions at all, so there is no answer key to replay and no claim being made.
            "skipped": [r.task_id for r in skipped],
            # THE HARNESS COULD NOT GRADE THESE, which is precisely the condition this command
            # exists to surface, and it fails the check.
            "errors": [{"task_id": r.task_id, "error": r.error[:200]} for r in errors[:10]],
            "n_errors": len(errors),
            "failures": [
                {
                    "task_id": r.task_id,
                    "db_reward": r.db_reward,
                    "reward_basis": list(r.reward_basis),
                    "failed_calls": r.n_failed_calls,
                    "error": r.error[:200],
                }
                for r in failed[:10]
            ],
            # BANKING: 87 DB + 1 DB/NL_ASSERTION carry a DB basis, so tau_reward is EMITTED
            # on 88 of 97; the 9 ACTION-basis tasks are graded here but withheld from the
            # endpoint, because an ENV evaluator hands them a free 1.0 that is not a
            # measurement. Retail's own split is whatever this prints for it.
            "tau_reward_emitted_on": sum(
                1 for r in rows if any("DB" in b.upper() for b in r.reward_basis)
            ),
            "ok": not failed and not errors,
        }
    )
    return 0 if not failed and not errors else 1


def _asks_failure(
    *, arm_id: str, home: tuple[str, ...], suite: str, mean_asks: float
) -> str | None:
    """The asks verdict for one (suite, arm) cell, or None when it passes.

    OFF-SUITE IS NOT A DEFECT. An arm that declares `home_suites` is only meaningful there:
    fake_chain's ChainInquirer walks the SYNTHETIC suite's dependency chain, so on musique
    it has no chain to walk and asks nothing. That is a suite mismatch, and reporting it as
    "expected to ask but did not" reads as a dead policy.

    The exemption is narrow on purpose: an on-suite zero-ask cell still fails, which is the
    finding this was derived from rather than a way of hiding it.
    """
    if home and suite not in home:
        return None
    if mean_asks > 0:
        return None
    # ABSENT IS NOT ZERO. `mean_asks` is averaged over scores.parquet rows, so a suite that
    # has been RUN but not yet SCORED has no `n_asks` rows and arrives here as NaN. Printing
    # that as "mean n_asks == 0.0" states a measurement nobody took -- and it was wrong in
    # the only case it ever fired on: tau2/inquirer_prompted reported 0.0 while the runs
    # themselves carried 8, 8, 1, 8 (true mean 6.25). The cell in the same report printed
    # NaN, so the two paths disagreed about whether the number existed at all.
    if mean_asks != mean_asks:  # NaN
        return "no n_asks rows for this cell; run `pi score` before verifying asks"
    return f"mean n_asks == {mean_asks} but this arm is expected to ask"


def _live_call_failure(*, arm_id: str, exempt: bool, n_live: int, n_all: int) -> str | None:
    """The live-call verdict for one (suite, arm) cell, or None when it passes.

    THE EXEMPTION IS NOT A MUTE. An arm declared `input_identical_by_design` duplicates
    another arm's requests on purpose -- parallel_replay replays a recorded question list,
    pare_plus_inquirer_prompted is the same PromptedInquirer with the same template -- so
    identical requests give identical cache keys and 100% cache hits are CORRECT. The
    canary reported both as dead code paths and advised "use an unused --seeds value",
    which cannot help: the collision is with another arm in the SAME sweep, not with a warm
    cache from an earlier one.

    ZERO TOTAL CALLS IS STILL REPORTED, exempt or not: 0 of 0 means the calls table is
    missing or the compaction did not run, which is a different failure wearing the same
    number.
    """
    if n_all <= 0:
        return "no calls recorded at all (0 of 0): calls.parquet is missing or `pi compact` did not run"
    if n_live > 0:
        return None
    if exempt:
        return None
    return (
        f"0 of {n_all} calls reached the provider (all cache hits): the code path ran, "
        "the model did not. Use an unused --seeds value."
    )


def cmd_verify_judge(a: argparse.Namespace) -> int:
    """Grade items whose correct answer the judge's OWN RUBRIC dictates, and report accuracy.

    The judge's self-consistency diagnostics -- position bias, order agreement, alpha -- are
    all passed perfectly by a judge that is confidently and CONSISTENTLY WRONG. Nothing
    measured whether it applies the rule it was handed. This does.

    Calls `kpr.grade` / `quality.grade` DIRECTLY rather than going through
    `harness.judge_runs`, which grades both orders and needs gold: this is a property of the
    instrument, not of a run, and it must be answerable without a rollout.
    """
    from pi_eval.judges import known_answer
    from pi_eval.judges._llm import JudgeParseError

    items = known_answer.build_items()
    if a.items:
        items = items[: int(a.items)]

    # The SAME loader `pi score` uses, so this verifies the judge that actually grades --
    # a second resolution path could verify a different client than the one in service.
    client_spec = os.environ.get("PI_JUDGE_CLIENT")
    model = os.environ.get("PI_MODEL_JUDGE", "")
    if not client_spec or not model:
        print(
            "PI_JUDGE_CLIENT and PI_MODEL_JUDGE must both be set; there is otherwise no "
            "judge to verify. See src/pi_run/judge_client.py.",
            file=sys.stderr,
        )
        return 2
    import importlib

    mod, _, attr = client_spec.partition(":")
    if not mod or not attr:
        print(f"PI_JUDGE_CLIENT={client_spec!r} is not 'module:factory'", file=sys.stderr)
        return 2
    judge = getattr(importlib.import_module(mod), attr)()
    llm = getattr(judge, "llm", judge)

    pairs: list[tuple[Any, Any]] = []
    n_err = 0
    for it in items:
        try:
            if it.kind == "kpr":
                from pi_eval.judges import kpr

                js = kpr.grade(
                    llm,
                    key_points=((it.item_id, it.key_point),),
                    report=it.report,
                    suite_id="verify",
                    task_id=it.item_id,
                    run_id=it.item_id,
                    judge_model=model,
                    seed=0,
                )
                pairs.append((it.expected, js[0].label if js else None))
            else:
                from pi_eval.judges import quality

                js = quality.grade(
                    llm,
                    question=it.question,
                    report=it.report,
                    suite_id="verify",
                    task_id=it.item_id,
                    run_id=it.item_id,
                    judge_model=model,
                    seed=0,
                )
                # `Judgment.criterion` is the KEY STRING (types.Criterion is a Literal),
                # not the quality.Criterion dataclass -- quality.grade passes `c.key`.
                got = next((j.score for j in js if j.criterion == "support"), None)
                pairs.append((it.expected, got))
        except JudgeParseError:
            # THE THIRD OUTCOME. Recorded as a miss, never dropped: accuracy among
            # parseable replies is the number a broken judge scores well on.
            pairs.append((it.expected, None))
            n_err += 1
        except Exception as exc:  # a transport failure is not a judge verdict
            print(f"  {it.item_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            pairs.append((it.expected, None))
            n_err += 1

    summary = known_answer.summarize(pairs)
    ok, why = known_answer.verdict(summary, min_accuracy=a.min_accuracy)
    summary["judge_model"] = model
    # JUDGE SPEND IS A SEPARATE BILL from `pi run --spend-cap`, and a campaign budget stated
    # as one number is wrong. Reported here so this command's cost is attributable.
    if hasattr(judge, "spend"):
        summary["spend"] = judge.spend()
    summary["ok"] = ok
    summary["verdict"] = why
    _emit(summary)
    return 0 if ok else 1


def cmd_verify_arms(a: argparse.Namespace) -> int:
    """Did every arm actually DO what it claims, against a real model?

    `status == ok` is not enough. An arm whose policy failed to parse its own output stops on
    turn 0, retrieves nothing, answers from the question alone, and writes a well-formed row --
    it is `drafter_only` in disguise, costing money and reporting the baseline's behaviour under
    the treatment's name. Measured on a real musique sweep: `random_q` and `parallel_replay`
    both did exactly that (n_asks 0.0 and 0.17, coverage 0.0 and 0.167), and nothing complained.

    Three assertions per (suite, arm):
      status      every unit ok
      billed      usd_billed > 0 for an arm that is not llm_free -- a cache-warm probe proves
                  the code path and NOTHING about the provider, because CachingClient keys on
                  request bytes and `_telemetry_from` reports a hit's recorded usd as-if
      asks        mean n_asks > 0 for an arm whose `expects_asks` is True
    """
    import duckdb

    from pinq_expt import arms as arm_table

    pq = Path(a.parquet_dir or "scores/parquet")
    runs, scores = pq / "runs.parquet", pq / "scores.parquet"
    if not runs.exists():
        _emit({"ok": False, "why": f"no runs.parquet at {runs}; run `pi compact` first"})
        return 2

    con = duckdb.connect()
    where = ["1=1"]
    if a.seed is not None:
        where.append(f"r.seed = {int(a.seed)}")
    if a.suite:
        where.append("r.suite_id = " + "'" + str(a.suite).replace("'", "''") + "'")
    rows = con.execute(
        f"SELECT r.suite_id, r.arm_id, r.status, count(*) AS n, "
        f"       sum(COALESCE(r.usd, 0.0)) AS usd "
        f"FROM read_parquet('{runs}') r WHERE {' AND '.join(where)} "
        "GROUP BY 1,2,3 ORDER BY 1,2,3"
    ).fetchall()
    if not rows:
        _emit({"ok": False, "why": "no runs matched; nothing to verify"})
        return 2

    # DID A REAL CALL HAPPEN, or did the cache answer everything? `runs.usd` charges a cache
    # hit its recorded cost AS-IF (see `cache._telemetry_from`), so a non-zero `usd` is NOT
    # evidence that the provider was reached. `calls.cache_hit` is. A probe on a warm seed
    # exercises the code path and proves nothing about the model.
    live: dict[tuple[str, str], tuple[int, int]] = {}
    calls = pq / "calls.parquet"
    if calls.exists():
        for su, ar, n_live, n_all in con.execute(
            f"SELECT r.suite_id, r.arm_id, "
            f"       sum(CASE WHEN c.cache_hit THEN 0 ELSE 1 END), count(*) "
            f"FROM read_parquet('{calls}') c JOIN read_parquet('{runs}') r ON r.run_id = c.run_id "
            f"WHERE {' AND '.join(where)} GROUP BY 1,2"
        ).fetchall():
            live[(str(su), str(ar))] = (int(n_live or 0), int(n_all or 0))

    asks: dict[tuple[str, str], float] = {}
    if scores.exists():
        for su, ar, v in con.execute(
            f"SELECT r.suite_id, r.arm_id, avg(s.value) FROM read_parquet('{scores}') s "
            f"JOIN read_parquet('{runs}') r ON r.run_id = s.run_id "
            f"WHERE s.metric_name = 'n_asks' AND {' AND '.join(where)} GROUP BY 1,2"
        ).fetchall():
            asks[(str(su), str(ar))] = float(v)

    report: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for su, ar, status, n, usd in rows:
        cell = seen.setdefault(
            (str(su), str(ar)), {"suite": su, "arm": ar, "n": 0, "usd": 0.0, "status": {}}
        )
        cell["n"] += int(n)
        cell["usd"] += float(usd or 0.0)
        cell["status"][str(status)] = int(n)

    for (su, ar), cell in sorted(seen.items()):
        arm = arm_table.ARMS.get(ar)
        cell["mean_asks"] = round(asks.get((su, ar), float("nan")), 3)
        cell["expects_asks"] = bool(getattr(arm, "expects_asks", True)) if arm else None
        cell["llm_free"] = bool(getattr(arm, "llm_free", False)) if arm else None
        bad = []
        if set(cell["status"]) != {"ok"}:
            bad.append(f"status {cell['status']}")
        n_live, n_all = live.get((su, ar), (0, 0))
        cell["live_calls"] = n_live
        cell["total_calls"] = n_all
        cell["input_identical_by_design"] = (
            bool(getattr(arm, "input_identical_by_design", False)) if arm else None
        )
        if arm is not None and not arm.llm_free:
            msg = _live_call_failure(
                arm_id=ar,
                exempt=bool(getattr(arm, "input_identical_by_design", False)),
                n_live=n_live,
                n_all=n_all,
            )
            if msg:
                bad.append(msg)
        if arm is not None and arm.expects_asks:
            cell["home_suites"] = list(getattr(arm, "home_suites", ()) or ())
            msg = _asks_failure(
                arm_id=ar,
                home=tuple(getattr(arm, "home_suites", ()) or ()),
                suite=su,
                # NaN, matching the cell above: a missing key means unscored, not silent.
                mean_asks=float(asks.get((su, ar), float("nan"))),
            )
            if msg:
                bad.append(msg)
        cell["ok"] = not bad
        if bad:
            failures.append(f"{su}/{ar}: " + "; ".join(bad))
        report.append(cell)

    _emit(
        {
            "cells": report,
            "n_cells": len(report),
            "n_failed": len(failures),
            "failures": failures,
            "ok": not failures,
        }
    )
    return 1 if failures else 0


def cmd_verify_judge_fidelity(a: argparse.Namespace) -> int:
    """THE OBLIGATION docs/DATA.md MAKES IN WRITING, WHICH NO CODE PATH COULD DISCHARGE.

    The DRGym eval harness has NO LICENCE FILE, so its judge prompts are reimplemented here
    from visible text rather than vendored. DATA.md justifies that with "a published fidelity
    delta (`pi_eval.judges.fidelity`)", and `judges/{kpr,citation,quality}.py` each repeat the
    promise in their module docstrings.

    `pi_eval.judges.fidelity` is fully built -- ReferenceKPR, load_kpr_reference, kpr_delta,
    KPRDelta -- with tests. It had NO production caller: `grep -rn fidelity src/ tests/
    scripts/` matched only docstrings. So the repository promised a published number that
    nothing could compute, which is the same defect class as a sealed constant nothing reads.

    This is the caller. It cannot invent the reference data -- that is upstream's own judge
    output on the same reports -- so when the file is absent it says exactly what is needed and
    exits non-zero, rather than reporting a delta of zero or nothing at all.
    """
    from pi_eval.judges.fidelity import NoOverlap, kpr_delta, load_kpr_reference

    ref_path = Path(a.reference) if a.reference else None
    if ref_path is None or not ref_path.exists():
        _emit(
            {
                "ok": False,
                "reference": str(ref_path) if ref_path else None,
                "why": (
                    "no upstream reference labels on disk. docs/DATA.md commits to publishing a "
                    "fidelity delta against the DRGym harness because its prompts are "
                    "reimplemented here, not vendored. Produce upstream's "
                    "evaluation_results_kpr_*.json over the SAME reports and pass --reference."
                ),
                "needs": [
                    "--reference PATH",
                    "--system",
                    "--judge-model",
                    "--source-url",
                    "--commit",
                ],
            }
        )
        return 1

    import pyarrow.parquet as pq

    jf = Path(a.parquet_dir or "scores/parquet") / "judgments.parquet"
    if not jf.exists():
        _emit({"ok": False, "why": f"no judgments.parquet at {jf}; run `pi score` first"})
        return 1

    from pi_eval.judges.types import Judgment

    ours = []
    for r in pq.read_table(jf).to_pylist():
        if str(r.get("criterion")) != "keypoint":
            continue
        ours.append(Judgment(**{k: r[k] for k in Judgment.__dataclass_fields__ if k in r}))

    ref = load_kpr_reference(
        ref_path,
        system=a.system,
        judge_model=a.judge_model,
        source_url=a.source_url,
        commit=a.commit,
    )
    try:
        delta = kpr_delta(ours, ref, ours_model=a.ours_model or "")
    except NoOverlap as exc:
        _emit({"ok": False, "why": str(exc)})
        return 1
    import dataclasses as _dc

    out = delta.as_dict() if hasattr(delta, "as_dict") else _dc.asdict(delta)
    out["ok"] = True
    _emit(out)
    return 0


def partition_divergences(
    diverged: list[dict[str, str]], baseline: Mapping[str, Any] | None
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split diverged calls into (new, known, unused_baseline_entries).

    A divergence is KNOWN only if the baseline names its exact (run_id, request_sha). Keyed
    on both because one bad call in a run must not excuse the rest of that run.

    WHY A BASELINE EXISTS AT ALL. The 41 divergences measured here predate 95d6524 and are
    unrepairable by construction: the recorded response_sha is what the run actually
    received (rewriting it falsifies provenance), and a re-roll cannot overwrite the run
    because `code_version` is in SEMANTIC_FIELDS, so it lands in a NEW directory -- which
    would put two admissible runs in each P3 grid cell and corrupt the aggregate. The
    choice is between deleting real experimental records and recording the closed set.

    `unused` is returned so a baseline that has stopped matching shrinks visibly. A
    baseline that quietly covers a live bug is the failure mode this must not become.
    """
    known_keys = {
        (str(e.get("run_id")), str(e.get("request_sha")))
        for e in ((baseline or {}).get("known") or [])
    }
    new: list[dict[str, str]] = []
    known: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for d in diverged:
        key = (str(d.get("run_id")), str(d.get("request_sha")))
        if key in known_keys:
            known.append(d)
            seen.add(key)
        else:
            new.append(d)
    unused = [{"run_id": r, "request_sha": q} for r, q in sorted(known_keys - seen)]
    return new, known, unused


def cmd_cache_verify(a: argparse.Namespace) -> int:
    """The join `pi_eval.schema` documents and nothing performed.

    schema.py:185 says of CALLS.request_sha: "== the cache key; joins runs to
    cache/<xx>/<sha>.json". No code anywhere did that join, so nothing could tell whether the
    cache still holds the response a recorded run actually received -- which is exactly what
    the artifact's "given the cache, every number regenerates" depends on.

    It could not hold before, either: the cache was last-writer-wins on the argument that two
    racers "write byte-identical files", which docs/REPRODUCE.md's own opening paragraph
    denies. Now that writes are first-writer-wins, this check is what proves the property
    rather than assuming it.

    Reports three counts and exits nonzero on the third:
      matched      the run's response_sha equals the cached entry's
      missing      the request is not in the cache (a replay would raise CacheMiss)
      DIVERGED     the cache holds a different response than the run recorded

    Every reading also names `cache_root` (the resolved ABSOLUTE path, not the argument as
    given -- `pi_run.cache.cache_root()` resolves a relative root against the READING
    process's cwd, so a probe run from a different cwd than the sweep silently reads a
    different directory) and `cache_entries` (what that root held). Without this a reading
    could not be audited after the fact: a request-not-in-cache count looks identical whether
    the cache is genuinely thin or the probe is simply pointed at the wrong directory.

    An absent or empty cache root REFUSES (exit 2, same code as the missing-calls.parquet
    precondition above -- distinct from the exit-1 divergence path) rather than reporting
    counts: with zero entries, `missing_from_cache` would equal `calls_checked` no matter
    what the run actually cached, which is indistinguishable from a wrong-directory read.
    """
    import pyarrow.parquet as pq

    cache = DiskCache(a.cache_root)
    calls = Path(a.parquet_dir or "scores/parquet") / "calls.parquet"
    if not calls.exists():
        _emit({"error": f"no calls.parquet at {calls}; run `pi compact` first"})
        return 2
    cache_stats = cache.stats()
    if cache_stats.entries == 0:
        _emit(
            {
                "error": "cache_root_empty_or_missing",
                "cache_root": cache_stats.root,
                "cache_entries": 0,
                "reason": (
                    f"resolved cache root {cache_stats.root} does not exist, or exists and "
                    "holds zero entries; a reading from here would be indistinguishable from "
                    "reading the wrong directory, so no calls_checked/matched/"
                    "missing_from_cache counts are reported"
                ),
            }
        )
        return 2
    matched = missing = 0
    diverged: list[dict[str, str]] = []
    for row in pq.read_table(calls).to_pylist():
        req, resp = str(row.get("request_sha") or ""), str(row.get("response_sha") or "")
        if not req:
            continue
        rec = cache.get(req)
        if rec is None:
            missing += 1
            continue
        cached = str(rec.get("response_sha") or "")
        if resp and cached and resp != cached:
            diverged.append(
                {
                    "run_id": str(row.get("run_id")),
                    # FULL, not truncated: schema.py calls this "the cache key; joins runs to
                    # cache/<xx>/<sha>.json", and it is what the baseline matches on. The two
                    # sha columns below stay short because they are only ever eyeballed.
                    "request_sha": req,
                    "run_recorded": resp[:12],
                    "cache_holds": cached[:12],
                }
            )
        else:
            matched += 1
    baseline = None
    bpath = Path(getattr(a, "baseline", None) or "docs/known_cache_divergences.json")
    if bpath.exists():
        try:
            baseline = json.loads(bpath.read_text())
        except (OSError, ValueError):
            baseline = None
    new_d, known_d, unused = partition_divergences(diverged, baseline)
    _emit(
        {
            # What was checked, named so a reading can be audited after the fact: which cache,
            # holding how many entries. See the docstring above for why this matters.
            "cache_root": cache_stats.root,
            "cache_entries": cache_stats.entries,
            "calls_checked": matched + missing + len(diverged),
            "matched": matched,
            "missing_from_cache": missing,
            # `diverged`/`n_diverged` keep their meaning: EVERY divergence, baselined or not.
            "diverged": new_d[:20] if new_d else diverged[:20],
            "n_diverged": len(diverged),
            # The split that decides the exit code. A baselined divergence is closed history
            # (see docs/known_cache_divergences.json); a new one means 95d6524 regressed.
            "n_new": len(new_d),
            "n_known": len(known_d),
            "new": new_d[:20],
            # A baseline entry that no longer matches: the list should only ever shrink, and
            # a stale one is how a baseline quietly turns into a blanket.
            "baseline_unused": unused[:20],
            "baseline": str(bpath) if baseline is not None else None,
            "ok": not new_d,
        }
    )
    return 1 if new_d else 0


def cmd_cache_stats(a: argparse.Namespace) -> int:
    cache = DiskCache(a.cache_root)
    st = cache.stats()
    out = st.as_dict()
    out["mean_bytes"] = round(st.bytes / st.entries, 1) if st.entries else 0.0
    if a.by_model:
        by: dict[str, int] = {}
        for _, rec in cache.iter_records():
            m = str(rec.get("model", "?"))
            by[m] = by.get(m, 0) + 1
        out["by_model"] = dict(sorted(by.items()))
    _emit(out)
    return 0


# --------------------------------------------------------------------------- pi verify firewall


def _canary_strings(root: Path, extra: Sequence[str]) -> list[str]:
    """Canaries come from data/canaries/ plus PI_CANARY_SALT plus anything named on the CLI.

    A run must never be able to echo one back, so the scan is over the CACHED REQUESTS: the
    cache holds the exact serialized bytes that were sent, which is the narrowest place a
    leak could hide and the only one that proves what actually crossed the wire.
    """
    out: list[str] = [s for s in extra if s]
    salt = os.environ.get("PI_CANARY_SALT", "")
    if salt:
        out.append(salt)
    cdir = root / "data" / "canaries"
    if cdir.exists():
        for f in sorted(cdir.glob("*.txt")):
            out += [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    return sorted(set(out))


def cmd_verify_firewall(a: argparse.Namespace) -> int:
    root = Path(a.root).resolve() if a.root else repo_root()
    canaries = _canary_strings(root, a.canary or [])
    carriers = _canary_carriers(root)

    # An unarmed scan must REFUSE, not report a trivial pass. `_canary_carriers` used to be
    # called without `root` (always inspecting cwd) and fed nothing into `ok` either way, so a
    # root whose data/gold/graphs never got built -- or a suite whose mint failed -- read
    # exactly like a clean scan: zero nonces to look for is indistinguishable from a genuinely
    # clean one. See `_firewall_armed` for the three states this distinguishes.
    armed_ok, armed_why = _firewall_armed(root, canaries)
    if not armed_ok:
        print(f"REFUSED: {armed_why}", file=sys.stderr)
        _emit(
            {
                "ok": False,
                "refused": True,
                "reason": armed_why,
                "root": str(root),
                "canaries_configured": len(canaries),
                "canary_carriers": carriers,
            }
        )
        return 2

    had_gold = _scrub_gold_root()

    # 1. The real worker path: probe from inside a pool child, not from this process.
    with ProcessPoolExecutor(max_workers=1) as ex:
        probe = ex.submit(firewall_probe).result()
    worker_ok = (not probe["gold_root_set"]) and (not probe["pi_eval_imported"])

    # 2. Canary scan over every cached request.
    cache = DiskCache(a.cache_root)
    scanned = 0
    hits: list[dict[str, str]] = []
    for sha, rec in cache.iter_records():
        scanned += 1
        blob = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        for c in canaries:
            if c in blob:
                hits.append({"request_sha": sha, "canary": c[:8] + "..."})

    ok = worker_ok and not hits
    _emit(
        {
            "worker": probe,
            "worker_ok": worker_ok,
            "parent_had_gold_root": had_gold,
            "canaries_configured": len(canaries),
            # WHICH SUITES LAYER 4 CAN ACTUALLY FIRE ON, per suite, rather than a single count
            # that implies uniform coverage. See `_canary_carriers`.
            "canary_carriers": carriers,
            "cache_entries_scanned": scanned,
            "canary_hits": hits,
            "ok": ok,
            "refused": False,
            "note": ""
            if canaries
            else "no gold built at this root and no PI_CANARY_SALT/--canary given: "
            "nothing to scan for",
        }
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------- pi cost estimate


def observed_arm_usage(parquet_dir: str | Path) -> dict[str, dict[str, float]]:
    """Per-arm MEDIAN token usage from real runs. Feeds `pi cost estimate`.

    Median, not mean: one runaway rollout should not reprice a 30k-unit grid. Deliberately
    NOT routed through pi_eval.report -- a budget-planning command has no business pulling
    gold-side code into its address space just to read four integer columns.
    """
    import statistics

    from pinq.extras import require

    require("pyarrow", why="reading scores/parquet")
    import pyarrow.parquet as pq  # noqa: E402

    p = Path(parquet_dir) / "runs.parquet"
    if not p.exists():
        return {}
    rows = [
        r
        for r in pq.read_table(p).to_pylist()
        if r.get("status") == "ok" and not r.get("is_dev_run")
    ]
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(str(r["arm_id"]), []).append(r)
    return {
        arm: {
            "n_runs": float(len(rs)),
            "tok_prompt_median": float(statistics.median(int(r["tok_prompt"] or 0) for r in rs)),
            "tok_completion_median": float(
                statistics.median(int(r["tok_completion"] or 0) for r in rs)
            ),
            "tok_total_median": float(statistics.median(int(r["tok_total"] or 0) for r in rs)),
            "retrieval_calls_median": float(
                statistics.median(float(r["retrieval_calls"] or 0.0) for r in rs)
            ),
        }
        for arm, rs in sorted(by_arm.items())
    }


def _observed_per_rollout(observed: dict[str, dict[str, float]]) -> tuple[float, float] | None:
    """Median-of-arm-medians for (prompt, completion) tokens per rollout, or None.

    The outer statistic is a median too: a single pathological arm (an oracle arm, a broken
    one) must not set the price of the whole grid.

    LLM-FREE ARMS ARE EXCLUDED, and leaving them in produced a $0.00 budget. `fake_chain`,
    `fake_depth1` and `fake_drafter_only` spend zero tokens BY DECLARATION -- that is what
    `llm_free` means and the worker raises if one ever spends any. A repository that has run
    the smoke test and one real arm therefore has three zero-token arms and one real one, and
    the median of {0, 0, 0, 9107} is 0.

    Measured before the fix: `pi cost estimate` priced Grid A's 11,849 rollouts at $0.00 with
    `within_gate: true`, from a parquet that did contain a real arm at 9,107 prompt tokens per
    rollout. An operator budgeting a campaign against that reads "within gate" and proceeds.
    The old `max(...) <= 0` guard could not catch it, because the max was not zero -- only the
    median was.

    With nothing informative left, this returns None and the caller falls back to the declared
    PRIOR and says `token_source: prior`. A guess labelled a guess is worth more than a
    measurement of the wrong population.
    """
    import statistics

    from pinq_expt import arms as arm_table

    def _spends(arm: str) -> bool:
        try:
            return not arm_table.get(arm).llm_free
        except Exception:  # noqa: BLE001 - an arm this table does not know is not evidence
            return False

    usable = {a: v for a, v in observed.items() if v.get("n_runs") and _spends(a)}
    prompt = [v["tok_prompt_median"] for v in usable.values()]
    completion = [v["tok_completion_median"] for v in usable.values()]
    if not prompt or not completion or max(prompt) + max(completion) <= 0:
        return None
    return statistics.median(prompt), statistics.median(completion)


DEFAULT_COST_MODEL = "anthropic/claude-sonnet-4-5"


def _cost_model(explicit: str | None) -> tuple[str, str]:
    """(model to price at, why). Defaults to THE MODEL THE SWEEP WILL ACTUALLY BILL.

    `--model` defaulted to a hardcoded `anthropic/claude-sonnet-4-5` while every role in this
    repository is pinned to gpt-oss-120b (PI_MODEL_INQUIRER and friends), and every recorded
    call in calls.parquet is `openai/aws/gpt-oss-120b`. So the headline number was priced for a
    model the experiment does not use:

        default (sonnet-4-5)   $2,009.40 naive / $1,429.07 cached
        pinned (gpt-oss-120b)  $  100.47 naive / $   68.23 cached

    A factor of 21, on the number a spend decision is made from, printed by a command whose
    entire job is to produce that number. It reports `model` in its output, but the total is
    what gets read -- and both directions are dangerous: cancelling a $68 campaign that reads
    as $1,429, or under-budgeting when the default is the cheaper one.

    The Inquirer's pin is the right default because it is the role that dominates the bill: it
    runs every turn, while the Answerer runs once.
    """
    if explicit:
        return explicit, "--model"
    pinned = os.environ.get("PI_MODEL_INQUIRER", "").strip()
    if pinned:
        return pinned, "PI_MODEL_INQUIRER"
    return DEFAULT_COST_MODEL, "fallback default (no PI_MODEL_INQUIRER set)"


def cmd_cost_estimate(a: argparse.Namespace) -> int:
    model, model_source = _cost_model(a.model)
    """Priors until there are runs; per-arm MEDIANS from the ledger the moment there are.

    A prior is a guess about a token count. Once a single arm has actually executed, the
    parquet knows the answer, and a budget built on the guess when the measurement exists is
    the reason campaigns overrun. Medians rather than means: one runaway rollout must not
    reprice a 30,000-unit grid.
    """
    from pinq_adapters.llm.pricing import PriceTable

    pt = PriceTable.load(strict=not a.lenient)
    observed: dict[str, dict[str, float]] = {}
    parquet_dir = None
    if not a.priors_only:
        root = Path(a.root).resolve() if a.root else repo_root()
        parquet_dir = Path(a.parquet) if a.parquet else root / "scores" / "parquet"
        try:
            observed = observed_arm_usage(parquet_dir)
        except Exception as exc:  # noqa: BLE001 - a planning command never crashes on data
            observed = {}
            print(f"(no observed usage from {parquet_dir}: {exc})", file=sys.stderr)
    if a.arm and observed:
        observed = {a.arm: observed[a.arm]} if a.arm in observed else {}
    per_rollout = _observed_per_rollout(observed)

    grids = [g.upper() for g in (a.grid or list(GRIDS))]
    rows = []
    total_naive = 0.0
    total_cached = 0.0
    for g in grids:
        if g not in GRIDS:
            raise SystemExit(f"unknown grid {g!r}; known: {sorted(GRIDS)}")
        spec = GRIDS[g]
        n = a.rollouts or spec["rollouts"]
        source = "prior"
        if per_rollout is not None:
            tok_in_each, tok_out_each = per_rollout
            source = "observed-median"
        else:
            tok_in_each, tok_out_each = spec["tok_in"], spec["tok_out"]
        tok_in = int(n * tok_in_each)
        tok_out = int(n * tok_out_each)
        naive = pt.usd(model, tok_prompt=tok_in, tok_completion=tok_out)
        # Prompt caching applies to the append-only prefix only; completions never cache.
        cached = pt.usd(
            model,
            tok_prompt=tok_in,
            tok_completion=tok_out,
            tok_cached=int(tok_in * (1.0 - CACHE_DISCOUNT)),
        )
        total_naive += naive
        total_cached += cached
        rows.append(
            {
                "grid": g,
                "rollouts": n,
                "tok_in": tok_in,
                "tok_out": tok_out,
                "tok_in_per_rollout": int(tok_in_each),
                "tok_out_per_rollout": int(tok_out_each),
                "source": source,
                "usd_naive": round(naive, 2),
                "usd_cached": round(cached, 2),
                "note": spec["note"],
            }
        )
    _emit(
        {
            "model": model,
            "model_source": model_source,
            "price_table": pt.version,
            "token_source": "observed-median" if per_rollout else "prior",
            "parquet_dir": str(parquet_dir) if parquet_dir else None,
            "observed_by_arm": observed,
            "grids": rows,
            "usd_naive_total": round(total_naive, 2),
            "usd_cached_total": round(total_cached, 2),
            "usd_with_reroll_margin": round(total_cached * 1.4, 2),
            "hard_gate_usd": 15000,
            "within_gate": total_cached * 1.4 < 15000,
        }
    )
    return 0


# --------------------------------------------------------------------------- pi score


def cmd_score(a: argparse.Namespace) -> int:
    """The scoring process, and the only one permitted to read gold."""
    # FIRST, before gold is read and before a judge is priced. `pi score` APPENDS to
    # scores.parquet in the same directory `pi compact` rewrites, so it races the same way,
    # and a refusal that arrives after $36 of judging is not a refusal.
    _score_root = Path(a.root).resolve() if a.root else repo_root()
    rc = _require_store_lock(_parquet_dir(a, _score_root), _score_root, what="pi score")
    if rc:
        return rc
    if a.gold_root:
        os.environ["PI_GOLD_ROOT"] = str(Path(a.gold_root).resolve())
    if not os.environ.get("PI_GOLD_ROOT"):
        print(
            "PI_GOLD_ROOT is unset. `pi score` IS the scoring process and must be able to "
            "read gold; pass --gold-root or export it.",
            file=sys.stderr,
        )
        return 2
    from pi_eval.score import score

    root = Path(a.root).resolve() if a.root else repo_root()
    res = score(
        _parquet_dir(a, root),
        runs_root=None if a.no_answers else runs_root(root, a.runs_root),
        graph_version=a.graph_version,
        judge_pins=list(a.judge_pin),
        # Explicit beats inferred. `_corpora_root` guesses from --parquet's grandparent, which
        # is right only when the parquet is where `pi compact` put it.
        corpora_root=Path(a.corpora_root) if a.corpora_root else None,
        judge_spend_cap=a.judge_spend_cap,
        null_draws=a.null_draws,
        seed=a.seed,
        # The citation judge's documents are replayed from here when a run has no
        # `judge_docs.json` sidecar. Every other cache-reading subcommand already wires
        # PI_CACHE_ROOT; `score` was the one that did not, so the judge saw no documents.
        cache_root=cache_root(getattr(a, "cache_root", None)),
    )
    _emit(res.as_dict())
    # A SKIPPED JUDGE DELETES A PRIMARY ENDPOINT, AND SAID SO ONLY IN A JSON FIELD.
    #
    # `pi score` reports `judging: "skipped: no judge client"` and exits 0. docs/REPRODUCE.md's
    # own command block runs `pi score --gold-root ...` with no PI_JUDGE_CLIENT set, so the
    # documented reproduction path silently emits ZERO rows for every judge-derived metric --
    # including `kpr_incremental`, the drgym PRIMARY endpoint (P3) -- and nothing downstream
    # distinguishes "measured nothing" from "measured zero".
    #
    # This is the repo's own rule ("a missing measurement must look missing") applied to the
    # command that produces the measurements. `--allow-no-judge` is the explicit opt-out, so a
    # judge-free scoring run is a thing you SAY rather than a thing that happens.
    if str(res.judging).startswith("skipped") and not getattr(a, "allow_no_judge", False):
        from pi_eval.prereg import PRIMARY, SECONDARY
        from pi_eval.score import define

        judged_endpoints = sorted(
            {
                e.metric
                for e in (*PRIMARY, *SECONDARY)
                if (define(e.metric) or None) and getattr(define(e.metric), "judge_derived", False)
            }
        )
        if judged_endpoints:
            print(
                f"JUDGING WAS SKIPPED ({res.judging}) but these PREREGISTERED endpoints are "
                f"judge-derived and now have no rows: {judged_endpoints}. Set PI_JUDGE_CLIENT "
                "(see src/pi_run/judge_client.py) or pass --allow-no-judge to say you meant it.",
                file=sys.stderr,
            )
            return 1
    return 0


# --------------------------------------------------------------------------- pi agg


def _parquet_dir(a: argparse.Namespace, root: Path) -> Path:
    return Path(a.parquet) if a.parquet else root / "scores" / "parquet"


def _open_agg(a: argparse.Namespace, root: Path):
    from pi_eval.prereg import CI_RESAMPLES
    from pi_eval.report import open_agg

    return open_agg(
        _parquet_dir(a, root),
        scorer_hash=a.scorer_hash,
        allow_contaminated=getattr(a, "allow_contaminated", False),
        # `--n-boot` defaults to None so that no literal lives beside the sealed constant.
        n_boot=CI_RESAMPLES if a.n_boot is None else a.n_boot,
        n_perm=a.n_perm,
        seed=a.seed,
    )


def contamination_report(rows: Sequence[Mapping[str, Any]]) -> str:
    """What `pi agg` prints when a trained arm reached a table it must not be in.

    A FUNCTION SO THE MESSAGE IS TESTABLE. The check has two predicates now -- the split, and
    membership of the exported id set -- and a second reason that fires but does not reach the
    printed line is a refusal nobody can act on: "re-run with `split: test`" is the wrong
    instruction for a row whose split already IS test and whose task the checkpoint saw anyway.
    """
    head = [
        "ASSERTION FAILED: a TRAINED arm was scored on tasks it may have been fitted on.",
        "The checkpoint may have seen these very tasks, so every number in the table is void "
        "and nothing downstream can tell.",
        "  reason=split         the task is not HELD OUT: recomputed or stamped split is not "
        "test (dev included -- the checkpoint was CHOSEN on dev). Re-run the grid with "
        "`split: test`.",
        "  reason=train_id_set  the task is IN the id set the checkpoint's export recorded, "
        "whatever the split says today. Re-run on tasks that export did not contain.",
    ]
    body = [
        f"  {v.get('table')}: {v.get('run_id')} arm={v.get('arm_id')} "
        f"{v.get('suite_id')}/{v.get('task_id')} reason={v.get('reason')} "
        f"stamped={v.get('split')!r} recomputed={v.get('recomputed_split')!r} "
        f"train_id_set_hash={str(v.get('train_id_set_hash') or '')[:16]}"
        for v in rows
    ]
    return "\n".join(head + body)


def cmd_agg(a: argparse.Namespace) -> int:
    from pi_eval import report as rp
    from pi_eval.report import AggregationError

    root = Path(a.root).resolve() if a.root else repo_root()
    try:
        agg = _open_agg(a, root)
    except AggregationError as exc:
        print(f"CANNOT AGGREGATE: {exc}", file=sys.stderr)
        return 2
    tables_dir = Path(a.tables) if a.tables else root / "tables"

    wanted = [k for k, flag in _AGG_FLAGS.items() if getattr(a, flag)]
    if a.all or not wanted:
        wanted = list(rp.ALL_TABLES)

    rc = 0
    offenders_total: list[dict] = []

    contaminated: list[dict] = []
    for tid in wanted:
        builder = rp.ALL_TABLES[tid]
        table = builder(agg, tasks=a.n) if _accepts_tasks(builder) else builder(agg)
        d = table.write(tables_dir)
        (d / f"{tid}.txt").write_text(table.to_text() + "\n")
        print(table.to_text())
        print(f"   -> {d}/table.json  +  {d}/provenance.json")
        print()
        if a.assert_no_gold_exposed:
            offenders = rp.assert_no_gold_exposed(agg, table.provenance.get("run_ids", []))
            offenders_total += [{**o, "table": tid} for o in offenders]
        # ALWAYS, not behind the flag. A trained arm scored on a task it could have trained on
        # voids the table it appears in, and nothing in the output says so -- there is no
        # opt-in case for tolerating it. `train_id_set_hash` was recorded by the runner and
        # read by nobody; this is its reader.
        try:
            contaminated += [
                {**v, "table": tid}
                for v in rp.train_split_violations(
                    agg, table.provenance.get("run_ids", []), ids_dir=a.ids_dir
                )
            ]
        except rp.TrainIdSetUnresolved as exc:
            # NOT a warning and not a skipped table. "I could not check" is a different answer
            # from "I checked and it was clean", and a rendered table cannot say which it got.
            print(f"CANNOT CHECK CONTAMINATION: {exc}", file=sys.stderr)
            return 2
    if contaminated:
        print(contamination_report(contaminated), file=sys.stderr)
        rc = 2
    if a.assert_no_gold_exposed:
        if offenders_total:
            print(
                "ASSERTION FAILED: contaminated runs reached a reported table.\n"
                "gold_exposed / dev- / canary_hit runs are oracle, ceiling or dirty-tree "
                "rollouts and must never contribute a published number.",
                file=sys.stderr,
            )
            for o in offenders_total:
                print(f"  {o['table']}: {o['run_id']} arm={o['arm_id']} ", file=sys.stderr)
            rc = 2
        else:
            print("assert-no-gold-exposed: clean (no gold_exposed, dev- or canary runs)")
    return rc


_AGG_FLAGS = {
    "T1_primary": "primary",
    "T1b_secondary": "secondary",
    "T1c_exploratory": "exploratory",
    "T2_arm_ladder": "arms",
    "T3_coverage_at_depth": "cad",
    "T4_budget_ops": "ops",
    "T5_parity": "parity",
    "T6_instrument": "instrument",
}


def _accepts_tasks(fn) -> bool:
    import inspect

    return "tasks" in inspect.signature(fn).parameters


# --------------------------------------------------------------------------- pi render


def cmd_render(a: argparse.Namespace) -> int:
    """THE SAME CONTAMINATION GATE `pi agg` APPLIES, on the command that writes the .tex.

    `cmd_agg` was the only caller of `train_split_violations` in the repository, and `pi render`
    builds the same tables from the same predicate. So the check could be skipped by rendering
    without aggregating first -- and rendering is the step whose output reaches the paper. The
    refusal text is `contamination_report`, called here rather than copied, so the operator is
    told the same thing by both commands.
    """
    from pi_eval import report as rp
    from pi_eval.report import AggregationError
    from pi_run.render import RenderRefused, render_all

    root = Path(a.root).resolve() if a.root else repo_root()
    try:
        agg = _open_agg(a, root)
    except AggregationError as exc:
        print(f"CANNOT RENDER: {exc}", file=sys.stderr)
        return 2
    try:
        res = render_all(
            agg,
            tables_dir=Path(a.tables) if a.tables else root / "tables",
            figures_dir=Path(a.figures) if a.figures else root / "figures",
            table_ids=list(a.table) or None,
            figure_ids=list(a.figure) or None,
            tasks=a.n,
            force=a.force,
            n_boot=a.n_boot_frontier,
            ids_dir=a.ids_dir,
        )
    except RenderRefused as exc:
        print(f"RENDER REFUSED: {exc}", file=sys.stderr)
        return 2
    except rp.TrainIdSetUnresolved as exc:
        # NOT a warning and not a rendered-anyway table, exactly as in `cmd_agg`: "I could not
        # check" is a different answer from "I checked and it was clean", and a .tex on disk
        # cannot say which of the two it got.
        print(f"CANNOT CHECK CONTAMINATION: {exc}", file=sys.stderr)
        return 2
    for w in res.warnings:
        print(f"!! {w}", file=sys.stderr)
    _emit(res.as_dict())
    for msg in res.refused:
        print(f"RENDER REFUSED: {msg}", file=sys.stderr)
    if res.contaminated:
        print(contamination_report(list(res.contaminated)), file=sys.stderr)
    return 0 if not (res.refused or res.contaminated) else 2


# --------------------------------------------------------------------------- pi report


def cmd_report_killswitch(a: argparse.Namespace) -> int:
    """THE week-1 gate. Read the verdict column, then read the rule next to it.

    Exits 1 when any comparator MATCHES, so the gate can be wired into a script and the
    consequence is a failed step rather than a paragraph somebody skims at 2am.
    """
    from pi_eval import report as rp
    from pi_eval.report import AggregationError

    root = Path(a.root).resolve() if a.root else repo_root()
    try:
        agg = _open_agg(a, root)
    except AggregationError as exc:
        print(f"CANNOT REPORT: {exc}", file=sys.stderr)
        return 2
    table = rp.killswitch_table(agg, tasks=a.n)
    d = table.write(Path(a.tables) if a.tables else root / "tables")
    (d / "K_killswitch.txt").write_text(table.to_text() + "\n")
    print(table.to_text())
    print()
    print("PREREGISTERED DECISION RULES (applied, not weighed):")
    for row in table.rows:
        rule = dict(rp.KILLSWITCH_RULES)[str(row["comparator"])]
        print(f"  [{row['verdict']:<28}] {row['comparator']}: {rule}")
    print()
    print(f"provenance -> {d}/provenance.json")
    matched = [r for r in table.rows if str(r["verdict"]).startswith("MATCHES")]
    if matched:
        print(f"EXIT 1: {len(matched)} comparator(s) MATCH. Apply the rule above.")
    return 1 if matched else 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pi", description="ProactiveInquirer runtime")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a (task x arm x seed) sweep")
    r.add_argument("--suite")
    r.add_argument("--arm", action="append", default=[], help="repeatable; default = all arms")
    r.add_argument("--task", action="append", default=[], help="repeatable; overrides --n")
    r.add_argument(
        "--n",
        type=int,
        default=None,
        help="first N task ids of the suite (default 5)",
    )
    r.add_argument("--seeds", nargs="+", default=None)  # None = unpassed; see grid_flag_conflicts
    r.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default=None,
        help="restrict task selection to this split before the head-N truncation",
    )
    r.add_argument(
        "--task-offset",
        type=int,
        default=0,
        help="skip this many of the FILTERED task list; makes grid slices disjoint",
    )
    r.add_argument("--root", default=None)
    r.add_argument("--corpus", default=None)
    r.add_argument("--runs-root", default=None)
    r.add_argument("--cache-root", default=None)
    r.add_argument(
        "--max-turns", type=int, default=None
    )  # default 16, applied in resolved_run_defaults
    r.add_argument("--k", type=int, default=None)  # default 5, applied in resolved_run_defaults
    r.add_argument("--budget-cap", type=int, default=None)
    r.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="worker processes. OVERRIDES PI_MAX_CONCURRENCY (default 8)",
    )
    r.add_argument(
        "--spend-cap",
        type=float,
        default=None,
        help="campaign USD cap; overrides PI_SPEND_CAP_USD. Remaining units are cancelled, "
        "not run, and re-issuing the command resumes (a resumed unit bills nothing)",
    )
    r.add_argument("--sweep", help="a conf/grids/*.yaml grid; overrides suite/arm/n/seeds")
    r.add_argument(
        "--allow-dirty",
        action="store_true",
        help="proceed with --sweep on a dirty tree, stamping every unit dev- (training-only, "
        "never an eval arm) -- for the deliberate case of collecting training data from an "
        "uncommitted tree. Without it, --sweep refuses on a dirty tree; a plain `pi run` "
        "(no --sweep) is unaffected either way and keeps printing the DIRTY banner.",
    )
    r.add_argument("--seed-questions-from", help="runs_root of a prior sweep, for replay arms")
    r.add_argument(
        "--questions-from-file",
        action="append",
        default=[],
        metavar="[ARM=]PATH",
        help="JSON {task_id: [question, ...]} for the replay/ceiling arms; wins over mining "
        "a prior sweep. Produced by `pi gold questions`. REPEATABLE, and may be scoped to "
        "one arm (gold_evidence=path) -- the two ceiling arms bound different things and "
        "must not share a file.",
    )
    r.add_argument(
        "--unit-timeout",
        type=int,
        default=None,
        # Read from the constant rather than restated as a literal -- the literal said 1800,
        # DEFAULT_UNIT_TIMEOUT_S is 3600, and nothing caught the two drifting apart (found
        # independently by another lane, 2026-09-17). An f-string cannot go stale this way.
        help="seconds one unit may run before it is abandoned as status=timeout "
        f"(default PI_UNIT_TIMEOUT_S, then {DEFAULT_UNIT_TIMEOUT_S}; 0 disables)",
    )
    r.add_argument(
        "--seed-questions-arm", help="arm whose questions to replay (default inquirer_prompted)"
    )
    r.add_argument("--pilot", action="store_true", help="mark runs pilot_flag=true")
    r.add_argument("--no-resume", action="store_true", help="re-run units that already have status")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser(
        "compact",
        help="runs/ -> scores/parquet/",
        description="Writing the SHARED scores/parquet requires the store lock: a directory "
        f"named by $PI_STORE_LOCK, or <out>/{STORE_LOCK_DIRNAME} by default, holding an "
        "`owner` file. Take it with `mkdir` (atomic), write who you are into owner, and "
        "remove both when done. An isolated --out elsewhere needs no lock. See docs/RUNBOOK.md.",
    )
    c.add_argument("--root", default=None)
    c.add_argument("--runs-root", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--exclude-dev", action="store_true", help="drop dev- runs from the tables")
    c.add_argument(
        "--replace",
        action="store_true",
        help="REPLACE the run-side tables instead of merging by run_id. Deliberately drops "
        "runs this pass did not see, which orphans their score rows; the count is reported.",
    )
    c.set_defaults(fn=cmd_compact)

    e = sub.add_parser("env", help="environment checks")
    esub = e.add_subparsers(dest="sub", required=True)
    ed = esub.add_parser("doctor", help="provider/module/pin report; never prints a key value")
    ed.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the live reachability call per pinned model (offline/CI)",
    )
    ed.add_argument("--probe-timeout", type=float, default=30.0)
    ed.set_defaults(fn=cmd_env_doctor)

    ca = sub.add_parser("cache", help="response cache")
    casub = ca.add_subparsers(dest="sub", required=True)
    cs = casub.add_parser("stats")
    cs.add_argument("--cache-root", default=None)
    cs.add_argument("--by-model", action="store_true")
    cs.set_defaults(fn=cmd_cache_stats)
    cv = casub.add_parser(
        "verify", help="join calls.request_sha to the cache and report divergence"
    )
    cv.add_argument("--cache-root", default=None)
    cv.add_argument("--parquet-dir", default=None)
    cv.add_argument(
        "--baseline",
        default=None,
        help="JSON of KNOWN divergences that predate the fix (default "
        "docs/known_cache_divergences.json). A divergence outside it fails the check.",
    )
    cv.set_defaults(fn=cmd_cache_verify)

    v = sub.add_parser("verify", help="assertions that must hold before a number is believed")
    vsub = v.add_subparsers(dest="sub", required=True)
    vjf = vsub.add_parser(
        "judge-fidelity",
        help="agreement with upstream's own judge labels; docs/DATA.md commits to publishing it",
    )
    vjf.add_argument("--reference", default=None, help="upstream evaluation_results_kpr_*.json")
    vjf.add_argument("--parquet-dir", default=None)
    vjf.add_argument("--system", default="drgym-upstream")
    vjf.add_argument("--judge-model", default="")
    vjf.add_argument("--source-url", default="")
    vjf.add_argument("--commit", default="")
    vjf.add_argument("--ours-model", default="")
    vjf.set_defaults(fn=cmd_verify_judge_fidelity)

    va = vsub.add_parser(
        "arms", help="every arm ran, billed, and actually asked -- not drafter_only in disguise"
    )
    va.add_argument("--parquet-dir", default=None)
    va.add_argument("--seed", type=int, default=None, help="restrict to one seed")
    va.add_argument("--suite", default=None)
    va.set_defaults(fn=cmd_verify_arms)

    vj = vsub.add_parser(
        "judge",
        help="grade items whose answer the judge's own rubric dictates; reports accuracy",
    )
    vj.add_argument("--items", type=int, default=None, help="first N items (default: all)")
    vj.add_argument("--min-accuracy", type=float, default=0.90)
    vj.set_defaults(fn=cmd_verify_judge)

    vf = vsub.add_parser("firewall", help="PI_GOLD_ROOT unset in a worker + canary scan")
    vf.add_argument("--root", default=None)
    vf.add_argument("--cache-root", default=None)
    vf.add_argument("--canary", action="append", default=[])
    vf.set_defaults(fn=cmd_verify_firewall)
    vt = vsub.add_parser(
        "tau2", help="replay each task's own answer key; the gold DB hash must come back"
    )
    vt.add_argument(
        "--replay-gold",
        action="store_true",
        help="the only mode; kept explicit so the command reads as what it does",
    )
    vt.add_argument("--n", type=int, default=None, help="first N task ids, sorted")
    vt.add_argument(
        "--suite",
        default="tau2",
        choices=("tau2", "tau2_retail"),
        help="which tau2 domain to replay (default: tau2, i.e. banking_knowledge)",
    )
    vt.set_defaults(fn=cmd_verify_tau2)

    def _analysis_flags(sp, *, tables: bool = True) -> None:
        """Flags shared by every parquet-reading command, so `--n 120` means one thing."""
        sp.add_argument("--root", default=None)
        sp.add_argument("--parquet", default=None, help="scores/parquet dir")
        if tables:
            sp.add_argument("--tables", default=None, help="output dir for table artifacts")
        sp.add_argument("--scorer-hash", default=None, help="required when several exist")
        sp.add_argument("--n", type=int, default=None, help="first N task ids, sorted")
        # NO LITERAL HERE. The resample count is a SEALED constant (`pi_eval.prereg`
        # CI_RESAMPLES); this file carried its own 10,000 while the drafted claim_rule said
        # 1,000, so the document and every rendered interval named different numbers. The
        # default cannot be spelled at parser-build time either: `build_parser` runs on the
        # ROLLOUT path and this module's contract is that no gold-side import reaches that
        # address space. So it is None here and resolved in `_open_agg`, which is already a
        # scoring-side function.
        sp.add_argument(
            "--n-boot",
            type=int,
            default=None,
            help="BCa resamples; default: the sealed pi_eval.prereg.CI_RESAMPLES",
        )
        sp.add_argument("--n-perm", type=int, default=10_000)
        sp.add_argument("--seed", type=int, default=0)

    sc = sub.add_parser("score", help="parquet + gold -> matches/scores (NEEDS PI_GOLD_ROOT)")
    sc.add_argument("--root", default=None)
    sc.add_argument("--parquet", default=None)
    sc.add_argument("--runs-root", default=None, help="read answer text + citations from here")
    sc.add_argument(
        "--cache-root",
        default=None,
        help="search cache to replay the citation judge's documents from "
        "(default PI_CACHE_ROOT). A run with no judge_docs.json sidecar -- which is every "
        "run -- gets its documents from here or not at all.",
    )
    sc.add_argument("--no-answers", action="store_true", help="parquet-only: skip answer metrics")
    sc.add_argument("--gold-root", default=None, help="sets PI_GOLD_ROOT for this process")
    sc.add_argument(
        "--allow-no-judge",
        action="store_true",
        help="score without a judge, knowing every judge-derived endpoint will have no rows",
    )
    sc.add_argument(
        "--judge-spend-cap",
        type=float,
        default=None,
        help="USD cap on JUDGING, a second provider bill that --spend-cap (a sweep concept) "
        "never covered. Overrides PI_JUDGE_SPEND_CAP_USD. Exceeding it writes NOTHING: a "
        "half-judged parquet gives some arms judge-derived rows and not others.",
    )
    sc.add_argument(
        "--corpora-root",
        default=None,
        help="the PUBLIC corpora tree the runs read. Inferred from --parquet's grandparent, "
        "which is right only when the parquet is where `pi compact` put it.",
    )
    sc.add_argument("--graph-version", default="v1")
    sc.add_argument("--judge-pin", action="append", default=[], help="repeatable; enters the hash")
    sc.add_argument("--null-draws", type=int, default=8)
    sc.add_argument("--seed", type=int, default=0)
    sc.set_defaults(fn=cmd_score)

    ag = sub.add_parser("agg", help="parquet -> paper tables, each with a provenance.json")
    _analysis_flags(ag)
    ag.add_argument("--primary", action="store_true", help="T1: preregistered endpoints")
    ag.add_argument("--secondary", action="store_true", help="T1b: the declared BH-FDR family")
    ag.add_argument("--exploratory", action="store_true", help="T1c: CIs, never a p-value")
    ag.add_argument("--arms", action="store_true", help="T2: the arm ladder")
    ag.add_argument("--cad", action="store_true", help="T3: C@d with |V_d|")
    ag.add_argument("--ops", action="store_true", help="T4: budget and operations")
    ag.add_argument("--parity", action="store_true", help="T5: cross-arm parity")
    ag.add_argument(
        "--instrument",
        action="store_true",
        help="T6: judge diagnostics -- position bias, parse failures, alpha, usability",
    )
    ag.add_argument("--all", action="store_true")
    ag.add_argument(
        "--assert-no-gold-exposed",
        action="store_true",
        help="exit non-zero if a gold_exposed or dev- run reached a table",
    )
    ag.add_argument(
        "--allow-contaminated",
        action="store_true",
        help="DEBUG ONLY: keep dev-/gold_exposed runs. Never for a reported number.",
    )
    # WHERE THE EXPORTED ID SETS ARE. An OVERRIDE, not the mechanism: `train_ids.<hash16>.txt`
    # is normally found under `data/rl/` from the hash a run was stamped with. This is for a
    # checkout whose exports live elsewhere -- and it never silently falls back, because an
    # override that answers from a different directory than the one it was pointed at is worse
    # than no answer.
    ag.add_argument(
        "--ids-dir",
        default=None,
        help="directory holding train_ids.<hash16>.txt sidecars (default: data/rl)",
    )
    ag.set_defaults(fn=cmd_agg)

    rn = sub.add_parser("render", help="parquet -> tables/*.tex and figures/*.pdf")
    _analysis_flags(rn)
    rn.add_argument("--figures", default=None)
    rn.add_argument("--table", action="append", default=[], help="repeatable; default = all")
    rn.add_argument("--figure", action="append", default=[], help="repeatable; default = all")
    rn.add_argument("--all", action="store_true", help="default; kept for symmetry")
    rn.add_argument("--n-boot-frontier", type=int, default=2000)
    rn.add_argument("--force", action="store_true", help="overwrite a hand-edited or drifted table")
    # The same override `pi agg` takes, because `pi render` runs the same contamination check.
    # A flag on one command and not the other would mean the check can be answered in one and
    # only refused in the other.
    rn.add_argument(
        "--ids-dir",
        default=None,
        help="directory holding train_ids.<hash16>.txt sidecars (default: data/rl)",
    )
    rn.add_argument("--allow-contaminated", action="store_true", help=argparse.SUPPRESS)
    rn.set_defaults(fn=cmd_render)

    rep = sub.add_parser("report", help="decision reports")
    repsub = rep.add_subparsers(dest="sub", required=True)
    ks = repsub.add_parser(
        "killswitch", help="the week-1 gate: the paired deltas + the preregistered rules"
    )
    _analysis_flags(ks)
    ks.add_argument("--allow-contaminated", action="store_true", help=argparse.SUPPRESS)
    ks.set_defaults(fn=cmd_report_killswitch)

    # `pi data` and `pi gold` live in their own modules: they are the only subcommands whose
    # handlers are gold-side by nature, and keeping them out of this file is what lets the
    # data track and the runtime track edit at the same time without a merge conflict.
    _register_data(sub)
    _register_gold(sub)
    _register_train(sub)
    # LOCAL ON PURPOSE, AND THIS ONE ACTUALLY MATTERS. `cmd_annotate` imports
    # `pi_eval.annotate` at module scope -- the only `cmd_*` module that does. This file's
    # own docstring says pi_eval is imported only inside scoring-side handlers and never at
    # module scope, and importing the registration here at module scope broke that: the
    # multiprocessing start method is `spawn`, so every pool child re-imports `__main__`,
    # which is the `pi` console script, which does `from pi_run.cli import main` -- and that
    # pulled pi_eval into the address space of every rollout worker.
    #
    # `pi verify firewall` was reporting exactly this: worker_ok false, pi_eval_imported
    # true, ok false, exit 1. Layer 3's load-bearing half still held (PI_GOLD_ROOT stays
    # unset, so gold_root() raises), which is why nothing leaked -- but the stricter
    # assertion the project wrote down for itself was false, and had been since cmd_annotate
    # landed.
    #
    # Making the import local fixed that under `spawn` only, which is macOS. Linux defaults to
    # `fork`, and a forked child inherits whatever this parser loaded before the pool was
    # built, so on the cluster every rollout worker still carried pi_eval and the probe said
    # so. The registration therefore comes from `cmd_annotate_args`, which never imports
    # pi_eval, and `cmd_annotate` itself loads only when an annotate subcommand runs.
    from pi_run.cmd_annotate_args import register as _register_annotate

    _register_annotate(sub)
    _register_suites(sub)

    co = sub.add_parser("cost", help="budget planning")
    cosub = co.add_subparsers(dest="sub", required=True)
    ce = cosub.add_parser("estimate")
    ce.add_argument("--grid", action="append", default=[], help="A A2 B C D E; default = all")
    ce.add_argument(
        "--model",
        default=None,
        help="default: the model pinned for the Inquirer role (PI_MODEL_INQUIRER), because "
        "that is what the sweep will actually bill",
    )
    ce.add_argument("--rollouts", type=int, default=None, help="override the grid's rollout count")
    ce.add_argument("--lenient", action="store_true", help="price unknown models at zero")
    ce.add_argument("--root", default=None)
    ce.add_argument(
        "--parquet",
        default=None,
        help="use per-arm token MEDIANS from these runs instead of the plan's priors",
    )
    ce.add_argument("--priors-only", action="store_true", help="ignore observed runs")
    ce.add_argument("--arm", default=None, help="price against one arm's observed median")
    ce.set_defaults(fn=cmd_cost_estimate)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
