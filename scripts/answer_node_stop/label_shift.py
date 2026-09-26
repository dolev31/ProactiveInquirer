"""How far apart are two SFT files that differ only in which gold fact decided STOP?

TWO STAGES, REPORTED SEPARATELY, because they answer different questions and a single number
would hide the second one entirely:

  * THE STATE-LEVEL SHIFT. Every state that reached a label decision in both files, joined on
    `(suite_id, task_id, run_id, turn_idx)` -- the state key, the thing `export_sft` groups by,
    one target per state. This is "how many decisions did the rule change", and it is the number
    a reader of the paper wants.
  * THE FILE-LEVEL SHIFT. Rows actually on disk, after `export_sft`'s exact dedupe. The dedupe
    key is `(suite, task, state_text, action_json)`, so relabelling a state can SPLIT a pair of
    states that previously collapsed into one row, or MERGE two that did not. The file counts
    therefore need not move by the same amount as the decisions, and the difference is not an
    error -- it is the dedupe doing its job on a changed action.

A state present in one file and absent from the other is counted in its own bucket
(`only_in_v1` / `only_in_variant`) and never silently dropped: under the variant a state whose
answer node is covered emits a STOP where v1 emitted nothing (its best ASK missed the floor), so
these buckets are a real part of the shift, not a join failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: The state key `export_sft` groups by. One target per state per file, so this is a 1:1 join.
StateKey = tuple[str, str, str, int]


def state_key(e) -> StateKey:  # `e` is a pinq_train.export.dataset.Example
    return (str(e.suite_id), str(e.task_id), str(e.run_id), int(e.turn_idx))


@dataclass(frozen=True, slots=True)
class Shift:
    """The 2x2 of `is_stop` across the two labels, plus the states only one file holds."""

    stop_to_stop: int
    stop_to_ask: int
    ask_to_stop: int
    ask_to_ask: int
    only_in_v1: int
    only_in_variant: int
    n_v1: int
    n_variant: int
    n_stop_v1: int
    n_stop_variant: int

    @property
    def stop_share_v1(self) -> float:
        return self.n_stop_v1 / self.n_v1 if self.n_v1 else float("nan")

    @property
    def stop_share_variant(self) -> float:
        return self.n_stop_variant / self.n_variant if self.n_variant else float("nan")

    def render(self) -> str:
        return (
            f"STOP->STOP {self.stop_to_stop}  STOP->ASK {self.stop_to_ask}  "
            f"ASK->STOP {self.ask_to_stop}  ASK->ASK {self.ask_to_ask}  "
            f"only_v1 {self.only_in_v1}  only_variant {self.only_in_variant}\n"
            f"rows {self.n_v1} -> {self.n_variant}   "
            f"stop_share {self.stop_share_v1:.4f} -> {self.stop_share_variant:.4f}"
        )


def shift(v1: Sequence, variant: Sequence) -> Shift:
    """The 2x2 above, over two `Example` lists produced from ONE `collect_rows` walk.

    ONE WALK IS NOT A CONVENIENCE HERE, it is what makes the number mean anything. `runs/` is
    shared and grows under the export; two walks minutes apart can see different run directories,
    and a row that exists in one file and not the other would then be counted as a relabelling.
    Callers that cannot guarantee one walk must say so beside the table.
    """
    a = {state_key(e): bool(e.is_stop) for e in v1}
    b = {state_key(e): bool(e.is_stop) for e in variant}
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for k in a.keys() & b.keys():
        cells[(a[k], b[k])] += 1
    return Shift(
        stop_to_stop=cells[(True, True)],
        stop_to_ask=cells[(True, False)],
        ask_to_stop=cells[(False, True)],
        ask_to_ask=cells[(False, False)],
        only_in_v1=len(a.keys() - b.keys()),
        only_in_variant=len(b.keys() - a.keys()),
        n_v1=len(a),
        n_variant=len(b),
        n_stop_v1=sum(1 for v in a.values() if v),
        n_stop_variant=sum(1 for v in b.values() if v),
    )


def stop_share(examples: Iterable) -> tuple[int, int, float]:
    """(n_stop, n_rows, share) -- the quantity `STOP_SHARE_CEILING` is compared against.

    Recomputed from the rows rather than read off a manifest counter:
    `n_stop_done_before_dedupe` is incremented BEFORE the exact dedupe and is 6,505 higher than
    the row count on the shipped export, so quoting it as a stop share overstates by that much.
    """
    rows = list(examples)
    n_stop = sum(1 for e in rows if e.is_stop)
    return n_stop, len(rows), (n_stop / len(rows) if rows else float("nan"))


def within_task_distinct3(examples: Iterable) -> float:
    """The gate's own within-task distinct-3 over the ASK rows' questions.

    THE GATE'S FUNCTION, NOT A SECOND IMPLEMENTATION (`pinq_train.rung1_sft.collapse`), because
    the floor being reported against is that function's floor. Relabelling ASK -> STOP deletes
    questions, and the diversity gate gets weakest exactly as the STOP share climbs -- so this
    has to be read beside the stop share, never instead of it.
    """
    from pinq_train.export.distinctness import question_of
    from pinq_train.rung1_sft.collapse import within_task_distinct_n

    pairs = [
        (f"{e.suite_id}/{e.task_id}", question_of(e.action_json)) for e in examples if not e.is_stop
    ]
    return within_task_distinct_n(pairs)


def manifest_deltas(man_v1, man_variant) -> dict[str, tuple[int, int]]:
    """Every integer manifest counter that moved, as `{field: (v1, variant)}`.

    A COUNTER THAT DID NOT MOVE IS THE CLAIM. "Everything else identical" is checkable exactly
    here: the cohort refusals, the leak drops, the split refusals and the duplicate drops must
    all be equal, and a lane that prints only the counters it expected to change cannot tell a
    clean swap from a second, unnoticed one.
    """
    out: dict[str, tuple[int, int]] = {}
    a, b = vars(man_v1), vars(man_variant)
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, int) and isinstance(vb, int) and not isinstance(va, bool) and va != vb:
            out[k] = (va, vb)
    return out


def unchanged_counters(man_v1, man_variant, fields: Sequence[str]) -> Mapping[str, int]:
    """The counters a caller ASSERTS are label-invariant, with their shared value.

    Raises on any that differ: a population difference between the two files would make the
    label shift above meaningless, and finding that out from a printed table nobody re-read is
    how it would otherwise be missed.
    """
    a, b = vars(man_v1), vars(man_variant)
    bad = {f: (a.get(f), b.get(f)) for f in fields if a.get(f) != b.get(f)}
    if bad:
        raise AssertionError(
            f"these counters must be label-invariant and are not: {bad}. The two files do not "
            "hold the same population, so no count of rows that changed label is a label shift."
        )
    return {f: a.get(f) for f in fields}
