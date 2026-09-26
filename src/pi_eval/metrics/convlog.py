"""What may be measured from a logged conversation, and what may not.

THE RULE THIS MODULE ENCODES. `metrics.human` records why FutureConversationCoverage was
deleted: in a logged conversation the user's third turn exists BECAUSE of the assistant's
second, so a need the agent PREEMPTS never appears as a later utterance and the metric scores
a miss on exactly the success case. That failure belongs to any metric scored over a
TRAJECTORY. Every rate here is instead a property of one state and one observed object -- a
question that was asked, an item the user actually wrote -- so nothing depends on a turn that
would have existed under a different policy.

Follow-up COUNTS are the boundary case. The count is real and worth printing; the comparison
is not, because the user turns a different policy would have produced are in no log. So the
count carries its own refusal (`comparable_across_arms=False`) and `compare_followups` raises
rather than returning a difference nobody could interpret.

ABSENT IS NOT ZERO, twice over. With no labels of a kind the rate is NaN and n is 0, so a
table prints NOT RUN rather than a plausible zero. And `cant_tell` leaves BOTH the numerator
and the denominator: an undecided unit is not a decided negative, and counting it as one would
let an annotator lower any rate here by giving up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# CLOSED VOCABULARIES. An instrument whose labels can drift is an instrument whose numbers
# cannot be compared across batches, so an unrecognised label raises instead of being skipped:
# silently dropping a row shrinks a denominator and moves a rate with nothing saying so.
B1_LABELS = ("necessary", "answer_in_state", "answer_inferable", "default_existed", "cant_tell")
B2_LABELS = ("stated_already", "inferable_from_state", "user_private", "new_task", "cant_tell")
B3_LABELS = (
    "stop_was_right",
    "should_have_asked",
    "should_have_continued",
    "should_have_stopped_earlier",
    "cant_tell",
)

_VOCAB: dict[str, tuple[str, ...]] = {"B1": B1_LABELS, "B2": B2_LABELS, "B3": B3_LABELS}
_UNDECIDED = "cant_tell"

# The wasted verdicts: three different ways the answer was already available. Kept as one set
# because the rate is about the person's spent attention, and all three spend it.
_WASTED = frozenset({"answer_in_state", "answer_inferable", "default_existed"})
# Items the state could have yielded. `user_private` and `new_task` are things no policy could
# have anticipated, and charging the agent for them would be charging it for not reading a mind.
_MISSABLE = frozenset({"stated_already", "inferable_from_state"})


class IllegalComparison(RuntimeError):
    """A comparison the log cannot support. See the module docstring."""


@dataclass(frozen=True, slots=True)
class Rate:
    value: float
    n: int


@dataclass(frozen=True, slots=True)
class Descriptive:
    """A number that is real but is not evidence for a contrast."""

    value: float
    n: int
    comparable_across_arms: bool
    why_not: str


def _decided(records: Iterable[Mapping[str, object]], kind: str) -> list[str]:
    out: list[str] = []
    for r in records:
        if r.get("kind") != kind:
            continue
        label = str(r.get("label"))
        if label not in _VOCAB[kind]:
            raise ValueError(f"{kind}: {label!r} is not in the closed vocabulary {_VOCAB[kind]}")
        if label != _UNDECIDED:
            out.append(label)
    return out


def _rate(labels: Sequence[str], hits: frozenset[str]) -> Rate:
    if not labels:
        return Rate(math.nan, 0)
    return Rate(sum(1 for x in labels if x in hits) / len(labels), len(labels))


def wasted_ask_rate(records: Iterable[Mapping[str, object]]) -> Rate:
    """Of the questions actually put to the person, how many spent a turn on nothing."""
    return _rate(_decided(records, "B1"), _WASTED)


def necessary_ask_precision(records: Iterable[Mapping[str, object]]) -> Rate:
    """The complement over the same denominator, reported separately because a reader asking
    "was it right to ask?" should not have to subtract."""
    return _rate(_decided(records, "B1"), frozenset({"necessary"}))


def anticipation_miss_rate(records: Iterable[Mapping[str, object]]) -> Rate:
    """How often the person had to state something the state already implied."""
    return _rate(_decided(records, "B2"), _MISSABLE)


def over_action_rate(records: Iterable[Mapping[str, object]]) -> Rate:
    """How often the agent acted where it should have asked."""
    return _rate(_decided(records, "B3"), frozenset({"should_have_asked"}))


def followups_per_task(counts: Sequence[int]) -> Descriptive:
    """The mean number of user turns after the opening statement.

    Descriptive by construction. It describes the conversations that happened, not a property
    of the policy that can be compared to another policy's.
    """
    n = len(counts)
    return Descriptive(
        value=sum(counts) / n if n else math.nan,
        n=n,
        comparable_across_arms=False,
        why_not=(
            "the user turns a different policy would have produced exist in no log, so a "
            "difference here is a counterfactual and not a measurement"
        ),
    )


def compare_followups(a: Descriptive, b: Descriptive) -> float:
    """Exists only to refuse. A caller reaching for this has the wrong instrument: compare
    policies by replaying held-out states, where both actions are observed under one state."""
    raise IllegalComparison(a.why_not if not a.comparable_across_arms else b.why_not)
