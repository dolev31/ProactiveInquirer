"""Known-answer items for `pi verify judge`: does the judge return what its rubric mandates?

WHY THIS EXISTS. Every judge-derived number in the paper rests on the judge being usable,
and the only evidence for that was the judge's own self-consistency diagnostics -- position
bias, order agreement, Krippendorff alpha. A judge that is confidently and CONSISTENTLY
WRONG passes all of them. Self-consistency is not accuracy, and nothing here measured
accuracy.

WHY THE ANSWERS ARE NOT OPINIONS. Each item's correct answer is dictated by the prompt the
judge is handed, quoted from the rubric itself:

    kpr, verbatim   -> Supported     "the report affirms, explains or reinforces the key point"
    kpr, absent     -> Omitted       "the report never addresses the key point"
    kpr, negated    -> Contradicted  "the report asserts something incompatible with it"
    quality, no URL -> support == 0  "if no part of the report gives source URLs, the rating is 0"

So a miss is not a disagreement about a hard case. It is the instrument failing to apply a
rule it was just given, which is the weakest thing that can be asked of it.

The text is CONSTRUCTED, never drawn from gold: these strings reach a live model, and gold
carries canary nonces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Plain, checkable facts. Each yields three kpr items (verbatim / absent / negated), so the
# label distribution is balanced by construction -- an unbalanced suite lets a judge that
# always answers one label look competent.
FACTS: tuple[tuple[str, str], ...] = (
    (
        "water boils at 100 degrees Celsius at sea level",
        "water boils at 40 degrees Celsius at sea level",
    ),
    ("the Pacific is the largest ocean on Earth", "the Arctic is the largest ocean on Earth"),
    ("insulin is produced in the pancreas", "insulin is produced in the liver"),
    (
        "the Amazon river flows into the Atlantic Ocean",
        "the Amazon river flows into the Pacific Ocean",
    ),
    (
        "copper conducts electricity better than rubber",
        "rubber conducts electricity better than copper",
    ),
    (
        "Mount Everest lies on the border of Nepal and China",
        "Mount Everest lies on the border of Peru and Chile",
    ),
    (
        "photosynthesis converts light energy into chemical energy",
        "photosynthesis converts chemical energy into light energy",
    ),
    ("the human heart has four chambers", "the human heart has seven chambers"),
    ("sound travels faster in water than in air", "sound travels faster in air than in water"),
    ("Saturn has a visible ring system", "Saturn has no ring system of any kind"),
    ("penicillin is an antibiotic", "penicillin is an antiviral vaccine"),
    ("the Sahara is the largest hot desert", "the Sahara is a temperate rainforest"),
    ("iron rusts when exposed to oxygen and water", "iron rusts only in a vacuum"),
    ("bees pollinate flowering plants", "bees do not interact with flowering plants"),
    ("the speed of light is finite", "the speed of light is infinite"),
    ("DNA carries hereditary information", "DNA carries no hereditary information"),
    ("Antarctica is covered by an ice sheet", "Antarctica is covered by tropical forest"),
    ("aluminium is lighter than lead by volume", "aluminium is heavier than lead by volume"),
    ("the Moon orbits the Earth", "the Earth orbits the Moon"),
    ("vitamin C is water soluble", "vitamin C is insoluble in water"),
)

# Filler for the OMITTED items: on-topic prose that addresses something else entirely, so
# the judge is not merely detecting an empty string.
UNRELATED: str = (
    "Municipal recycling programmes vary widely between regions. Collection schedules, "
    "accepted materials and contamination thresholds are set locally, and the resulting "
    "differences make national comparisons difficult to interpret."
)


@dataclass(frozen=True, slots=True)
class KnownAnswerItem:
    """One item whose correct answer the rubric dictates."""

    item_id: str
    kind: str  # "kpr" | "quality"
    expected: Any  # a kpr label, or the mandated quality score
    report: str
    key_point: str = ""
    question: str = ""


def build_items() -> tuple[KnownAnswerItem, ...]:
    """The suite. Deterministic: same items, same order, every call."""
    out: list[KnownAnswerItem] = []
    for i, (fact, negation) in enumerate(FACTS):
        out.append(
            KnownAnswerItem(
                item_id=f"kpr-sup-{i:02d}",
                kind="kpr",
                expected="Supported",
                key_point=fact,
                # The key point VERBATIM, so "affirms" cannot be argued.
                report=f"A brief note on the topic. It is established that {fact}. "
                "This follows from standard references on the subject.",
            )
        )
        out.append(
            KnownAnswerItem(
                item_id=f"kpr-omi-{i:02d}",
                kind="kpr",
                expected="Omitted",
                key_point=fact,
                report=UNRELATED,
            )
        )
        out.append(
            KnownAnswerItem(
                item_id=f"kpr-con-{i:02d}",
                kind="kpr",
                expected="Contradicted",
                key_point=fact,
                report=f"A brief note on the topic. It is established that {negation}. "
                "This is stated plainly and without qualification.",
            )
        )
    for i, (fact, _) in enumerate(FACTS):
        out.append(
            KnownAnswerItem(
                item_id=f"qual-nourl-{i:02d}",
                kind="quality",
                # "if no part of the report gives source URLs, the rating is 0"
                expected=0.0,
                question=f"What is known about the claim that {fact}?",
                report=(
                    f"It is widely accepted that {fact}. Experts say this has been understood "
                    "for a long time, and studies show it holds under ordinary conditions. "
                    "Further work continues in this area."
                ),
            )
        )
    return tuple(out)


def summarize(pairs: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    """(expected, got) -> accuracy, per-label accuracy, confusion, parse failures.

    `got is None` means the judge's reply did not parse. It is counted as a THIRD OUTCOME
    and as a MISS, never dropped: accuracy over the parseable subset is the one number a
    judge that fails on every hard item scores well on. Both are reported, so the gap
    between them is visible rather than a choice made silently.

    An empty run reports `accuracy: None` rather than 1.0 -- 0 of 0 correct is not perfect.
    """
    n = len(pairs)
    n_parse = sum(1 for _, got in pairs if got is None)
    n_right = sum(1 for exp, got in pairs if got is not None and got == exp)
    n_parseable = n - n_parse

    by_label: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for exp, got in pairs:
        key = str(exp)
        slot = by_label.setdefault(key, {"n": 0, "n_correct": 0, "n_parse_failures": 0})
        slot["n"] += 1
        if got is None:
            slot["n_parse_failures"] += 1
        elif got == exp:
            slot["n_correct"] += 1
        confusion.setdefault(key, {})
        got_key = "<unparseable>" if got is None else str(got)
        confusion[key][got_key] = confusion[key].get(got_key, 0) + 1
    for slot in by_label.values():
        slot["accuracy"] = (slot["n_correct"] / slot["n"]) if slot["n"] else None

    return {
        "n_items": n,
        "n_correct": n_right,
        "n_parse_failures": n_parse,
        "accuracy": (n_right / n) if n else None,
        # Reported BESIDE the real number so the difference cannot be mistaken for it.
        "accuracy_among_parseable": (n_right / n_parseable) if n_parseable else None,
        "by_label": by_label,
        "confusion": confusion,
    }


def verdict(summary: Mapping[str, Any], *, min_accuracy: float = 0.90) -> tuple[bool, str]:
    """Usable or not, with the reason. A judge that cannot apply its own rubric is not one.

    0.90 rather than 1.00 because an LLM judge is stochastic and a single hard-to-parse
    reply should not condemn the instrument; well below that and the judge is not reading
    the rule it was given.
    """
    acc = summary.get("accuracy")
    if acc is None:
        return False, "no items were graded, so nothing was measured"
    if acc < min_accuracy:
        return False, (
            f"accuracy {acc:.3f} < {min_accuracy:.2f} on items whose answer its own rubric "
            f"dictates ({summary['n_parse_failures']} of {summary['n_items']} did not parse)"
        )
    return True, f"accuracy {acc:.3f} over {summary['n_items']} rubric-mandated items"
