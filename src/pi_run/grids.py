"""Declarative sweep grids.

A grid is the unit a researcher reasons about ("Grid A, main, 17 arms, M1, three suites"),
and it is exactly what the preregistration freezes. Keeping it in a file rather than in a
shell history is what makes "the sweep config is frozen at prereg" a checkable statement:
the file's sha256 goes into the run manifest, so a silently widened grid shows up as a
changed run identity rather than as extra rows nobody notices.

Deliberately thin: it resolves to the same `plan()` arguments the CLI already takes, so a
grid can never do something a flag could not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class Grid:
    name: str
    description: str
    suites: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    n_tasks: int | None = None
    task_ids: tuple[str, ...] = ()
    # WHICH TASKS ARE ELIGIBLE, and WHERE IN THEM THE SLICE STARTS.
    #
    # Selection was `list(suite.task_ids())[: n]` -- the first N in corpus order, with `split`
    # consulted nowhere in cli.py. Two consequences, both measured:
    #
    #   * `tier1_trained` took the first 97 musique ids, of which 55 are `split == "train"`.
    #     It evaluated the trained arm on 57% training tasks, and no eval-side consumer reads
    #     the `split` column, so it cannot be fixed after the fact.
    #   * every grid started at 0, so pilot(120) was a strict PREFIX of confirmatory(200) and
    #     trained(97) a prefix of pilot. A pilot that chooses a threshold burns its tasks; a
    #     prefix guarantees every burned task is also a confirmatory task.
    #
    # `split` filters before the head-N; `task_offset` skips that many of the FILTERED list.
    # 600 unused musique tasks make disjoint slices free.
    #
    # NOTE tier1_oracle deliberately shares confirmatory's tasks: a ceiling is only meaningful
    # measured on the tasks it is a ceiling FOR. That overlap is intended and asserted as such.
    split: str | None = None
    task_offset: int = 0
    max_turns: int = 16
    k: int = 5
    # k is NOT a free parameter: it decides whether the phenomenon is measurable at all.
    # As k grows the full question's top-k absorbs more gold evidence and leaves less for a
    # sub-question to add. Measured (% of tasks with sub-question-only gold evidence, n=100):
    #     musique     k2 63%  k3 67%  k5 67%  k8 62%   robust everywhere
    #     strategyqa  k2 42%  k3 29%  k5 21%  k8 17%   needs k=2
    #     wiki2       k2 71%  k3 62%  k5 36%  k8 21%   needs k<=5
    # A single k=5 across a multi-suite grid would make strategyqa unmeasurable, so per-suite
    # k is a first-class field rather than something to remember.
    k_by_suite: dict[str, int] = field(default_factory=dict)
    budget_cap: int | None = None
    # A CEILING THE SUITE'S HOST ASKED FOR, not a property of this machine. DeepResearchGym
    # granted its key with "do not issue many parallel calls (e.g., max 16)"; that belongs
    # with the suite, because PI_MAX_CONCURRENCY is global and read from the environment,
    # so a laptop's setting would otherwise ride into a request to someone else's server.
    # None = leave it to the runner.
    concurrency: int | None = None
    pilot: bool = False
    exploratory: bool = False
    notes: str = ""
    source_sha256: str = ""

    def select(self, all_ids: Sequence[str], split_of) -> list[str]:
        """The task ids this grid runs, from the suite's full ordered list.

        `split_of(task_id) -> "train"|"dev"|"test"`. Explicit `task_ids` win outright.
        """
        if self.task_ids:
            return list(self.task_ids)
        pool = [t for t in all_ids if self.split is None or split_of(t) == self.split]
        pool = pool[self.task_offset :]
        return pool[: self.n_tasks] if self.n_tasks is not None else pool

    def k_for(self, suite_id: str) -> int:
        """The retrieval budget for one suite, defaulting to the grid-wide k."""
        return self.k_by_suite.get(suite_id, self.k)

    @property
    def n_units(self) -> int:
        """Rollout count, so a grid's cost is visible before it is launched."""
        per = len(self.task_ids) or (self.n_tasks or 0)
        return per * len(self.arms) * len(self.seeds) * len(self.suites)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: str | Path) -> Grid:
    from pinq.extras import require

    # grids are the only place YAML is used, and a grid is what a sweep IS.
    yaml = require("yaml", why="conf/grids/*.yaml")

    p = Path(path)
    raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    missing = {"name", "suites", "arms"} - set(raw)
    if missing:
        raise ValueError(f"{p}: grid is missing {sorted(missing)}")
    # AN UNKNOWN KEY IS A TYPO OR A FIELD THIS LOADER FORGOT, AND BOTH ARE SILENT.
    # This built the Grid from an explicit allowlist and ignored everything else, so a grid
    # author writing `split: test` got no error and no effect -- the run would quietly use the
    # default. That is exactly how `tier1_trained` came to evaluate on training tasks.
    known = {f.name for f in fields(Grid)} - {"source_sha256"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{p}: unknown grid key(s) {unknown}. A key this loader does not read is silently "
            f"ignored, so it is refused instead. Known keys: {sorted(known)}"
        )
    return Grid(
        name=raw["name"],
        description=raw.get("description", ""),
        suites=tuple(raw["suites"]),
        arms=tuple(raw["arms"]),
        seeds=tuple(raw.get("seeds", (0,))),
        n_tasks=raw.get("n_tasks"),
        task_ids=tuple(raw.get("task_ids", ())),
        split=raw.get("split"),
        task_offset=int(raw.get("task_offset", 0)),
        max_turns=int(raw.get("max_turns", 16)),
        k=int(raw.get("k", 5)),
        k_by_suite={str(a): int(b) for a, b in (raw.get("k_by_suite") or {}).items()},
        budget_cap=raw.get("budget_cap"),
        concurrency=raw.get("concurrency"),
        pilot=bool(raw.get("pilot", False)),
        exploratory=bool(raw.get("exploratory", False)),
        notes=raw.get("notes", ""),
        source_sha256=_sha(p),
    )


def load_all(root: str | Path = "conf/grids") -> list[Grid]:
    return [load(p) for p in sorted(Path(root).glob("*.yaml"))]


def trained_grid_violation(grid: Grid, trained_arms: Sequence[str] | frozenset[str]) -> str | None:
    """Why this grid may not run a trained arm the way it is written. None means it may.

    THE RULE. A grid carrying an arm whose model pin is a fitted checkpoint must declare
    `split: test`, unless it is `exploratory: true`, in which case it must declare
    `split: dev`.

    WHY THE EXEMPTION IS NOT A LOOPHOLE. Checkpoint selection has to run the trained arm on
    dev -- dev is the only split it may look at before the test split is opened even once -- so
    a rule that flatly forbade a trained arm off `test` would forbid the dev gate itself and
    would be deleted the first time someone needed it. `exploratory` is the flag that keeps
    those rows out of every reported table, so the exemption is tied to it rather than to a
    grid name, and a dev grid WITHOUT it is the dangerous case: rows that look like any other
    rows, drawn from the split the checkpoint was chosen on.

    WHY THE SUBJECT SET IS PASSED IN. `TRAINED_ARMS` lives in `pi_eval.report`, and importing
    it here would put `pi_run.grids -> pi_eval` in the graph for a predicate that does not need
    gold. The caller supplies it, which also lets a test drive the rule directly.

    This generalises `test_the_trained_grid_evaluates_only_on_held_out_tasks`, which checks one
    grid by name because that is the grid whose defect prompted it -- `tier1_trained` took the
    first 97 musique ids, 55 of them train-split, and nothing eval-side reads the `split`
    column, so it could not be repaired afterwards.
    """
    if not (set(grid.arms) & set(trained_arms)):
        return None
    if grid.exploratory:
        if grid.split != "dev":
            return (
                f"exploratory grid carries {sorted(set(grid.arms) & set(trained_arms))} but "
                f"declares split={grid.split!r}; an exploratory trained grid is checkpoint "
                "selection and may only sit on dev"
            )
        return None
    if grid.split != "test":
        return (
            f"carries {sorted(set(grid.arms) & set(trained_arms))} but declares "
            f"split={grid.split!r}. A trained arm off the test split is either training data "
            "or the split the checkpoint was selected on; both are void in a reported table. "
            "Set `split: test`, or `split: dev` with `exploratory: true` for a dev gate."
        )
    return None


def total_units(grids: Sequence[Grid]) -> int:
    return sum(g.n_units for g in grids)
