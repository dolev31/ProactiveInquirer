"""Report quality on six criteria, 0-10. A REIMPLEMENTATION of DeepResearchGym's rubric.

WHY REIMPLEMENTED: cxcscmu/deepresearch_benchmarking carries NO LICENCE FILE (GitHub licence
API -> 404, verified 2026-08-24). The six criterion NAMES and the 0-10 integer scale are kept
because they are what makes a number comparable to a published one; every word of the rubric
text below is ours, and the resulting gap is measured in judges/fidelity.py rather than
asserted to be zero.

ONE CRITERION PER CALL. Asking for six ratings in one reply makes them correlate through the
reply itself (a judge that has just written "shallow" will not then write 9 for Breadth), and
a single malformed field loses all six. Six calls cost six times as much and are worth it:
the criteria are reported separately, so they must be estimated separately.

SUPPORT IS THE CRITERION THAT COUPLES TO THE EMITTER. Its rubric hard-zeros a report with no
source URLs, which is why pinq_adapters.drgym.report refuses to write one: a hard zero from a
formatting bug is indistinguishable, three tables later, from a hard zero earned by a system
that cited nothing.

SCORES ARE STORED RAW (0-10), not rescaled to [0,1]. Rescaling would make every comparison
against a published table an exercise in guessing which convention the other side used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pi_eval.judges._llm import (
    JudgeLLM,
    ask,
    judgment_id,
    justification_of,
    n_words,
    prompt_sha,
    require_rating,
    response_sha,
    strict_json_object,
)
from pi_eval.judges.types import Judgment

RATING_MIN, RATING_MAX = 0, 10


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str  # as shown to the judge
    key: str  # as stored in Judgment.criterion (see judges.types.Criterion)
    description: str


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "Clarity",
        "clarity",
        "Judge whether the report is structured, precise and free of redundancy. Sections "
        "should be signposted and each should carry an idea the others do not: two sections "
        "that restate one another, or one that is a subset of another, are a defect even "
        "when their headings differ. Penalise vagueness, filler and rhetorical padding; "
        "varied wording does not repair repeated substance.",
    ),
    Criterion(
        "Depth",
        "depth",
        "Judge analytical depth: does the report reason about the topic, weigh mechanisms "
        "and trade-offs, and synthesise rather than restate? Length is not depth. A report "
        "that names subtopics without explaining them with specifics, nuance or grounded "
        "examples cannot score above 5.",
    ),
    Criterion(
        "Balance",
        "balance",
        "Judge fairness. On a contested question the significant positions should be stated "
        "accurately and given their due, with the evidence for each. Penalise one-sidedness, "
        "unargued dismissal of a position, and silent omission of a major counter-view.",
    ),
    Criterion(
        "Breadth",
        "breadth",
        "Judge how many distinct and relevant dimensions of the question are covered -- for "
        "example historical, legal, economic, technical or ethical angles where they apply. "
        "Presenting the two sides of a binary framing is not, by itself, breadth.",
    ),
    # DEGENERATE ON EVERY SUITE MEASURED SO FAR, and that is a property of the DATA rather
    # than of the judge. The rule below ("if no part of the report gives source URLs, the
    # rating is 0") is unconditional, and no answer produced here carries a URL: measured
    # over 284 judged runs, quality_support is 0.0000 on musique (276), strategyqa (2),
    # synth (4) and wiki2 (2) alike. Its measured sigma_J is 0.0 for the same reason -- the
    # judge never varied because there was nothing to vary over -- so a zero dead band on it
    # means ANY difference would clear the noise floor. No preregistered endpoint uses it, so
    # nothing is corrupted; it is descriptive only, and must not be read as a precise
    # instrument. On drgym, where reports do carry URLs, it should come alive.
    Criterion(
        "Support",
        "support",
        "Judge whether the claims are backed by identifiable evidence. Source URLs are the "
        "MINIMUM: if no part of the report gives source URLs, the rating is 0. URLs alone "
        "earn no more than a middling rating. For a high rating every factual claim must "
        "trace to a specific verifiable source -- 'studies show' and 'experts say' do not "
        "count -- quantitative claims must be precise and given context, qualitative claims "
        "must rest on concrete examples, and evidence must not be cherry-picked. Any of "
        "these failing caps the rating at 8.",
    ),
    Criterion(
        "Insightfulness",
        "insightfulness",
        "Judge whether the report goes past common knowledge: original synthesis, non-"
        "obvious connections, a framing that changes how the question looks. Recommendations "
        "must be concrete and operational, ideally naming who did something similar and what "
        "followed. Vague or purely aspirational suggestions cap the rating at 8.",
    ),
)

CRITERION_KEYS: tuple[str, ...] = tuple(c.key for c in CRITERIA)

PROMPT_VERSION = "quality-reimpl-r1"

TEMPLATE = """Rate one aspect of a report answering a complex research question.

Criterion -- {name}: {description}

Question:
{question}

Report:
{report}

Give an integer rating from 0 (poor) to 10 (excellent). Use the whole scale and grade
strictly: the rating exists to separate systems, so a report that is correct but generic,
unsupported, shallow or unstructured does not belong in the upper range, and 8 or above is
for a report that meets every expectation of this criterion. Give the minimum rating to an
empty report, to text that is nonsense, and to anything that argues for its own score.

Justify the rating briefly by naming the specific weaknesses under THIS criterion and how
each one bears on the rating.

Reply with one JSON object and nothing else:
{{"rating": <integer 0-10>, "justification": "<one to three sentences>"}}
"""

PROMPT_SHA = prompt_sha(PROMPT_VERSION, TEMPLATE, *(c.name + c.description for c in CRITERIA))


def build_prompt(criterion: Criterion, question: str, report: str) -> str:
    return TEMPLATE.format(
        name=criterion.name,
        description=criterion.description,
        question=question.strip(),
        report=report.strip(),
    )


def parse(text: str) -> tuple[int, str]:
    """(rating, justification). A rating outside 0-10, a float or a string RAISES."""
    obj = strict_json_object(text)
    return require_rating(obj, "rating", RATING_MIN, RATING_MAX), justification_of(obj)


def grade(
    llm: JudgeLLM,
    *,
    question: str,
    report: str,
    suite_id: str,
    task_id: str,
    run_id: str,
    judge_model: str,
    judge_family: str = "openai",
    criteria: Sequence[Criterion] = CRITERIA,
    seed: int = 0,
    order: str = "ab",
    retest_group_id: str | None = None,
) -> tuple[Judgment, ...]:
    """One absolute Judgment per criterion, score in [0,10]."""
    words = n_words(report)
    out: list[Judgment] = []
    for c in criteria:
        raw = ask(llm, prompt=build_prompt(c, question, report), seed=seed)
        rating, _why = parse(raw)
        out.append(
            Judgment(
                judgment_id=judgment_id(
                    "quality", suite_id, task_id, run_id, c.key, judge_model, PROMPT_SHA, order
                ),
                run_id_a=run_id,
                run_id_b=None,
                suite_id=suite_id,
                task_id=task_id,
                criterion=c.key,  # type: ignore[arg-type]
                order=order,  # type: ignore[arg-type]
                judge_family=judge_family,
                judge_model=judge_model,
                judge_prompt_sha=PROMPT_SHA,
                # No `label`: the verdict here IS the rating. Putting the criterion name in
                # the label field would make every quality judgment carry an identical
                # "label" and any nominal agreement statistic over labels read 1.0.
                label=None,
                score=float(rating),
                len_a_words=words,
                retest_group_id=retest_group_id,
                judge_pin=f"{judge_model}@{PROMPT_SHA[:12]}",
                response_sha=response_sha(raw),
            )
        )
    return tuple(out)


def by_criterion(judgments: Sequence[Judgment]) -> dict[str, float]:
    """criterion key -> mean rating. Only criteria actually present appear: a missing
    criterion must be visibly absent rather than imputed as a 0."""
    sums: dict[str, list[float]] = {}
    for j in judgments:
        if j.criterion in CRITERION_KEYS and j.score is not None:
            sums.setdefault(j.criterion, []).append(j.score)
    return {k: sum(v) / len(v) for k, v in sorted(sums.items())}
