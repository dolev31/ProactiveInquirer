"""Key-point recall. A REIMPLEMENTATION of DeepResearchGym's KPR judge, not a port of it.

WHY REIMPLEMENTED. The upstream harness (github.com/cxcscmu/deepresearch_benchmarking) is
public but carries NO LICENCE FILE -- verified by direct call on 2026-08-24: GitHub's licence
API returns 404 for that repository. Public is not licensed, and a paper that vendors
unlicensed prompt text into an Apache-2.0 repository has a distribution problem, not a
citation problem. So the prompt below is written from scratch against the same RUBRIC, the
label vocabulary and the JSON contract are preserved because they are the interface that
makes numbers comparable, and the resulting disagreement with upstream is MEASURED and
published rather than assumed away -- see judges/fidelity.py.

WHAT THE RUBRIC IS. One gold key point and one report per call. The report either supports
the point (affirms, explains or reinforces it), omits it (never addresses it), or contradicts
it (asserts something incompatible). Recall is the Supported share, so a contradiction and an
omission score the same 0.0 -- but the LABEL is kept on every Judgment, because
"this system contradicted 8% of the key points" and "this system missed them" are different
findings about a system and collapsing them at judgement time makes the second unrecoverable.

ONE KEY POINT PER CALL, deliberately. Grading the whole list in one call makes the verdicts
non-exchangeable (an early "Omitted" conditions the rest) and makes a single malformed reply
lose the whole task instead of one point.
"""

from __future__ import annotations

from typing import Sequence

from pi_eval.gold import GoldGraph
from pi_eval.judges._llm import (
    JudgeLLM,
    ask,
    judgment_id,
    justification_of,
    n_words,
    prompt_sha,
    require_label,
    response_sha,
    strict_json_object,
)
from pi_eval.judges.types import Judgment

LABELS: tuple[str, ...] = ("Supported", "Omitted", "Contradicted")

# Recall is the Supported share: a contradiction is not partial credit. The label survives on
# the Judgment so contradiction rate stays reportable separately.
KEYPOINT_SCORE: dict[str, float] = {"Supported": 1.0, "Omitted": 0.0, "Contradicted": 0.0}

PROMPT_VERSION = "kpr-reimpl-r1"

TEMPLATE = """Decide how a research report treats one key point.

Key point:
{key_point}

Report:
{report}

Choose exactly one label:
- "Supported": the report affirms, explains or reinforces the key point.
- "Omitted": the report never addresses the key point.
- "Contradicted": the report asserts something incompatible with the key point.

Judge only what the report actually says. Do not credit a point the report merely gestures
at, and do not penalise wording that differs from the key point while making the same claim.

Reply with one JSON object and nothing else:
{{"label": "<Supported|Omitted|Contradicted>", "justification": "<one or two sentences>"}}
"""

PROMPT_SHA = prompt_sha(PROMPT_VERSION, TEMPLATE)


def build_prompt(key_point: str, report: str) -> str:
    return TEMPLATE.format(key_point=key_point.strip(), report=report.strip())


def parse(text: str) -> tuple[str, str]:
    """(label, justification). Raises JudgeParseError on anything else -- never defaults."""
    obj = strict_json_object(text)
    return require_label(obj, "label", LABELS), justification_of(obj)


def key_points_of(graph: GoldGraph) -> tuple[tuple[str, str], ...]:
    """(node_id, text) for every REQUIRED node. The DRGym builder emits no other partition."""
    return tuple((n.gold_node_id, n.gold_text) for n in graph.required())


def grade(
    llm: JudgeLLM,
    *,
    key_points: Sequence[tuple[str, str]],
    report: str,
    suite_id: str,
    task_id: str,
    run_id: str,
    judge_model: str,
    judge_family: str = "openai",
    seed: int = 0,
    order: str = "ab",
    retest_group_id: str | None = None,
) -> tuple[Judgment, ...]:
    """One absolute Judgment per key point. run_id_b is None: this is grading, not a contest.

    A malformed verdict propagates: the caller decides whether to retry or to record the item
    as unjudged, and neither decision belongs to a parser.
    """
    out: list[Judgment] = []
    words = n_words(report)
    for node_id, text in key_points:
        raw = ask(llm, prompt=build_prompt(text, report), seed=seed)
        label, _why = parse(raw)
        out.append(
            Judgment(
                judgment_id=judgment_id(
                    "kpr", suite_id, task_id, run_id, node_id, judge_model, PROMPT_SHA, order
                ),
                run_id_a=run_id,
                run_id_b=None,
                suite_id=suite_id,
                task_id=task_id,
                criterion="keypoint",
                order=order,  # type: ignore[arg-type]
                judge_family=judge_family,
                judge_model=judge_model,
                judge_prompt_sha=PROMPT_SHA,
                label=label,
                score=KEYPOINT_SCORE[label],
                key_point_id=node_id,
                len_a_words=words,
                retest_group_id=retest_group_id,
                judge_pin=f"{judge_model}@{PROMPT_SHA[:12]}",
                response_sha=response_sha(raw),
            )
        )
    return tuple(out)


def recall(judgments: Sequence[Judgment]) -> float:
    """The endpoint: Supported / all judged key points. NaN on an empty set, never 0.0.

    Zero would read as "the system supported nothing"; NaN reads as "nothing was judged",
    and those are different facts about a run.
    """
    js = [j for j in judgments if j.criterion == "keypoint" and j.label is not None]
    if not js:
        return float("nan")
    return sum(1 for j in js if j.label == "Supported") / len(js)


def label_counts(judgments: Sequence[Judgment]) -> dict[str, int]:
    out = {lab: 0 for lab in LABELS}
    for j in judgments:
        if j.criterion == "keypoint" and j.label in out:
            out[j.label] += 1
    return out
