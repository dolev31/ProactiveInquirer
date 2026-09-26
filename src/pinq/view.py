"""The single choke point through which a suite's raw record becomes agent-visible.

One gold key in one suite's task JSON would defeat the import contract, the process split
and the type wall simultaneously. This is the only place that can stop it, so it is
deliberately strict: unknown keys raise, missing keys raise, nothing is coerced silently.
"""

from __future__ import annotations

from typing import Any

from .types import TaskView

ALLOWED: frozenset[str] = frozenset(TaskView.__dataclass_fields__)


class LeakageError(ValueError):
    """Raised when a field outside the TaskView allowlist reaches the agent boundary."""


def make_view(**kw: Any) -> TaskView:
    extra = set(kw) - ALLOWED
    if extra:
        raise LeakageError(
            f"fields not in the TaskView allowlist: {sorted(extra)}. "
            "If this is gold, it belongs in data/gold/ and is read only by pi_eval."
        )
    missing = ALLOWED - set(kw)
    if missing:
        raise LeakageError(f"incomplete view, missing: {sorted(missing)}")
    if not isinstance(kw["word_cap"], int) or kw["word_cap"] <= 0:
        raise LeakageError("word_cap must be a positive int")
    for key in ("task_id", "suite_id", "question", "corpus_id", "corpus_hash"):
        if not isinstance(kw[key], str) or not kw[key]:
            raise LeakageError(f"{key} must be a non-empty str")
    # `instructions` IS CHECKED TOO, AND WAS THE ONE FIELD THAT WAS NOT.
    #
    # Five of the six string fields were type-checked and this one accepted any object at all --
    # a GoldNode, a dict of gold spans, a whole GoldGraph -- which `promptlib.render` then
    # stringifies straight into the prompt. That is exactly the leak this function exists to
    # refuse: the allowlist above rejects an unknown FIELD NAME, and a caller who put gold in a
    # known field walked past it.
    #
    # Empty is allowed here and not above: a suite with no extra instructions is ordinary,
    # while a task with no question or no corpus_id is malformed.
    if not isinstance(kw["instructions"], str):
        raise LeakageError(
            f"instructions must be a str, got {type(kw['instructions']).__name__}. It renders "
            "verbatim into the prompt, so anything else is a gold object one str() away from "
            "the model."
        )
    return TaskView(**kw)
