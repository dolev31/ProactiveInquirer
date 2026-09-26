"""Judge records. The judge is an INSTRUMENT with a characterized error profile, not an oracle."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

Criterion = Literal[
    "paired_pref",
    "keypoint",
    "citation_support",
    "presupposition",
    "clarity",
    "depth",
    "balance",
    "breadth",
    "support",
    "insightfulness",
]

KeyPointLabel = Literal["Supported", "Omitted", "Contradicted"]
CitationLabel = Literal["full_support", "partial_support", "no_support"]

CITATION_SCORE: dict[str, float] = {
    "full_support": 1.0,
    "partial_support": 0.5,
    "no_support": 0.0,
}


@dataclass(frozen=True, slots=True)
class Judgment:
    judgment_id: str
    run_id_a: str
    run_id_b: str | None  # None => absolute grading (key points, quality rubric)
    suite_id: str
    task_id: str
    criterion: Criterion
    order: Literal["ab", "ba"]
    judge_family: str
    judge_model: str
    judge_prompt_sha: str
    temperature: float = 0.0
    pref_sign: int | None = None  # -1 / 0 / +1, from A's point of view
    magnitude: float | None = None
    label: str | None = None
    score: float | None = None
    key_point_id: str | None = None
    # Length is recorded on EVERY judgment: an LLM judge pays roughly +0.3-0.8 Likert per
    # doubling of answer length, and the Inquirer arm's answers are systematically longer.
    len_a_words: int = 0
    len_b_words: int = 0
    retest_group_id: str | None = None
    paraphrase_id: str | None = None
    judge_pin: str = ""
    # sha256 of the judge's RAW reply. The verdict is a MEASUREMENT, so the byte string it was
    # read from is its provenance: without this, a re-parse under a fixed parser cannot be told
    # apart from a re-judge under a changed model.
    response_sha: str = ""

    def to_row(self) -> dict[str, Any]:
        """One `pi_eval.schema.JUDGMENTS` row. Field-for-field, by construction.

        Built by iterating the dataclass fields rather than by listing column names, so a
        field added here without a matching column raises SchemaViolation on the next write
        instead of being dropped on the floor.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


def judgment_from_row(row: Mapping[str, Any]) -> Judgment:
    """The inverse of `to_row`. An unknown column RAISES rather than being ignored.

    An ignored column is how a table and its reader drift: the writer starts recording
    something, the reader silently drops it, and the loss is invisible until someone asks the
    table a question it can no longer answer.
    """
    names = {f.name for f in fields(Judgment)}
    unknown = sorted(set(row) - names)
    if unknown:
        raise ValueError(
            f"judgment row carries unknown column(s) {unknown}; known: {sorted(names)}"
        )
    return Judgment(**{k: v for k, v in row.items() if k in names})  # type: ignore[arg-type]
