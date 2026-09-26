"""S0 — assemble the trace pool the graph is mined from.

A single successful trace exhibits ONE SUFFICIENT PATH. It certifies neither necessity nor
completeness, and it is structurally blind to the optional-but-valuable needs that the
Inquirer is supposed to find. So the pool is never one trace: it is a factorial over
generators, and the union across sufficient paths approximates a cover while the
intersection approximates a necessary core. Both are reported; neither is collapsed.

The factorial is also the contamination control. Requiring >= 2 distinct model families and
>= 2 distinct policy forms before a candidate is promoted is what stops one generator's
habits (a planner that always dumps the schema first) from manufacturing a phantom need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

PolicyForm = Literal["chain", "breadth", "react", "plan_then_act"]
Success = Callable[[float], bool]


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell of the generator factorial: (model family x policy form x retrieval variant)."""

    model_family: str
    policy_form: PolicyForm
    retrieval_variant: str

    @property
    def cell_id(self) -> str:
        return f"{self.model_family}/{self.policy_form}/{self.retrieval_variant}"


@dataclass(frozen=True, slots=True)
class Trace:
    trace_id: str
    suite_id: str
    task_id: str
    cell: Cell
    seed: int
    success: bool
    native_score: float
    retrieved_uids: tuple[str, ...]
    turn_uids: tuple[tuple[int, tuple[str, ...]], ...]  # (turn_idx, uids) in issue order
    unlocked_tools: tuple[str, ...] = ()
    mutating_calls: tuple[str, ...] = ()


@dataclass
class TracePool:
    """Successes AND failures. The failure pool is not waste — it is the negative control:
    a need appearing at the same rate in failures as in successes is non-diagnostic."""

    suite_id: str
    task_id: str
    successes: list[Trace] = field(default_factory=list)
    failures: list[Trace] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return len(self.successes)

    @property
    def n_failure(self) -> int:
        return len(self.failures)

    @property
    def cells(self) -> set[str]:
        return {t.cell.cell_id for t in self.successes}

    @property
    def families(self) -> set[str]:
        return {t.cell.model_family for t in self.successes}

    @property
    def policy_forms(self) -> set[str]:
        return {t.cell.policy_form for t in self.successes}


# Suite-specific success predicates. Thresholds are preregistered, not tuned after the fact.
SUCCESS_THRESHOLDS: dict[str, float] = {
    "tau2": 1.0,  # binary DB-hash reward: nothing short of exact counts
    "musique": 0.9,  # token-F1
    "synth": 0.9,
    "strategyqa": 1.0,
    "wiki2": 0.9,
    "drgym": 0.0,  # set at runtime to the p75 of drafter_only KPR
}


def is_success(suite_id: str, score: float, *, override: float | None = None) -> bool:
    thresh = override if override is not None else SUCCESS_THRESHOLDS.get(suite_id, 1.0)
    return score >= thresh


def build_pool(traces: Iterable[Trace], *, suite_id: str, task_id: str) -> TracePool:
    pool = TracePool(suite_id=suite_id, task_id=task_id)
    for t in traces:
        (pool.successes if t.success else pool.failures).append(t)
    return pool


def pool_is_admissible(
    pool: TracePool, *, min_cells: int = 2, min_families: int = 2
) -> tuple[bool, str]:
    """A pool too thin to support the contamination controls must not yield gold at all.

    Returning a reason (rather than a bare False) is what lets the miner report WHY a task
    was skipped, so a systematically-skipped stratum is visible instead of silently absent.
    """
    if pool.n_success == 0:
        return False, "no successful traces"
    if len(pool.cells) < min_cells:
        return False, f"only {len(pool.cells)} distinct generator cells (< {min_cells})"
    if len(pool.families) < min_families:
        return False, f"only {len(pool.families)} model families (< {min_families})"
    return True, "ok"


def order_index(trace: Trace) -> dict[str, int]:
    """First turn at which each uid was retrieved. The substrate of the order-consistency
    screen in edges.py — and it is ORDER, which is why that screen can only ever veto an
    edge, never establish one."""
    out: dict[str, int] = {}
    for turn_idx, uids in trace.turn_uids:
        for u in uids:
            out.setdefault(u, turn_idx)
    return out


def cell_grid(
    families: Sequence[str], forms: Sequence[PolicyForm], variants: Sequence[str]
) -> list[Cell]:
    return [Cell(f, p, v) for f in families for p in forms for v in variants]
