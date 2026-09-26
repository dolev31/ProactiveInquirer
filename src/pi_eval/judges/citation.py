"""Citation support. A REIMPLEMENTATION of DeepResearchGym's citation judge.

WHY REIMPLEMENTED: the upstream harness has NO LICENCE FILE (GitHub's licence API returns
404 for cxcscmu/deepresearch_benchmarking, verified 2026-08-24), so its prompt text is not
ours to redistribute. The prompts here are written from scratch; the three-label vocabulary
and the 1.0 / 0.5 / 0.0 scoring are preserved because they are the interface that makes the
number comparable, and the residual disagreement is measured in judges/fidelity.py.

TWO STAGES, AND THE SECOND ONE NEEDS DOCUMENTS.
  1. EXTRACT (claim, source URLs) pairs from the report. Only claims the report itself ties
     to a URL are eligible; an uncited sentence is not a failed citation, it is not a
     citation. This is why drgym.report refuses to emit a URL-free report: with no URLs this
     stage extracts nothing and "no claims" is indistinguishable from "no supported claims".
  2. CHECK each claim against the text of the documents it cites: full / partial / no
     support.

WHERE THE DOCUMENT TEXT COMES FROM, AND A DELIBERATE DIVERGENCE FROM UPSTREAM. Upstream
crawls each cited URL at judging time and, when the crawl fails, feeds the checker an error
string -- which scores that claim as no_support. We pass in the text the RETRIEVER returned
(`docs_from_units`), because that is the text the system actually read and because a judge
whose verdict depends on whether a site was reachable this afternoon is not reproducible.
A claim whose sources have no text available here is therefore recorded as UNRESOLVED and
excluded from the denominator rather than scored 0.0; `grade` returns that count and it is
published next to the mean. The difference from upstream's behaviour is a known, named
fidelity delta, not an accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pi_eval.judges._llm import (
    JudgeLLM,
    JudgeParseError,
    ask,
    judgment_id,
    justification_of,
    n_words,
    prompt_sha,
    require_label,
    response_sha,
    strict_json_object,
)
from pi_eval.judges.types import CITATION_SCORE, Judgment
from pinq.types import EvidenceUnit

LABELS: tuple[str, ...] = ("full_support", "partial_support", "no_support")

PROMPT_VERSION = "citation-reimpl-r1"

EXTRACT_TEMPLATE = """Extract every factual claim in the report that is explicitly tied to a
source URL in the report itself.

Report:
{report}

Rules:
- Include a claim only when the report attaches one or more URLs to it. Skip framing,
  summaries and opinions that carry no source.
- Restate each claim as one self-contained sentence.
- Copy the URLs verbatim from the report. Never invent, complete or normalise a URL.

Reply with one JSON object and nothing else:
{{"claims": [{{"claim_id": 1, "claim": "<sentence>", "sources": ["<url>", ...]}}, ...]}}
"""

CHECK_TEMPLATE = """Decide whether the cited sources support the statement.

Statement:
{claim}

Cited sources:
{sources}

A fluent statement can still overstate its sources, so check every part of it against the
source text. Ask yourself whether "according to these sources, <statement>" would be
accurate.

Choose exactly one label:
- "full_support": every part of the statement is supported by the sources.
- "partial_support": some parts are supported and others are absent from the sources.
- "no_support": the sources support no part of the statement.

Reply with one JSON object and nothing else:
{{"support": "<full_support|partial_support|no_support>", "justification": "<one sentence>"}}
"""

EXTRACT_SHA = prompt_sha(PROMPT_VERSION, "extract", EXTRACT_TEMPLATE)
CHECK_SHA = prompt_sha(PROMPT_VERSION, "check", CHECK_TEMPLATE)
PROMPT_SHA = prompt_sha(PROMPT_VERSION, EXTRACT_SHA, CHECK_SHA)


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: int
    claim: str
    sources: tuple[str, ...]


def docs_from_units(units: Sequence[EvidenceUnit]) -> dict[str, str]:
    """url -> text, from what the retriever returned.

    `EvidenceUnit.title` carries the URL for the drgym suite (FineWeb records have no title),
    which is exactly the string a report is expected to cite literally.
    """
    return {u.title: u.text for u in units if u.title.startswith(("http://", "https://"))}


def build_extract_prompt(report: str) -> str:
    return EXTRACT_TEMPLATE.format(report=report.strip())


def build_check_prompt(claim: str, sources: Sequence[str], docs: Mapping[str, str]) -> str:
    body = "\n\n".join(f"[{i + 1}] {url}\n{docs.get(url, '')}" for i, url in enumerate(sources))
    return CHECK_TEMPLATE.format(claim=claim.strip(), sources=body)


def parse_claims(text: str) -> tuple[Claim, ...]:
    """Strict: the schema we asked for, or JudgeParseError. A claim with no source is not a
    citation and must not silently become one with an empty source list."""
    obj = strict_json_object(text)
    if "claims" not in obj or not isinstance(obj["claims"], list):
        raise JudgeParseError(f"extractor returned no 'claims' list; keys: {sorted(obj)}")
    out: list[Claim] = []
    for i, item in enumerate(obj["claims"]):
        if not isinstance(item, dict):
            raise JudgeParseError(f"claims[{i}] is {type(item).__name__}, expected an object")
        cid = item.get("claim_id")
        claim = item.get("claim")
        sources = item.get("sources")
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise JudgeParseError(f"claims[{i}].claim_id={cid!r} is not an int")
        if not isinstance(claim, str) or not claim.strip():
            raise JudgeParseError(f"claims[{i}].claim is empty or not a string")
        if not isinstance(sources, list) or not sources:
            raise JudgeParseError(f"claims[{i}].sources is empty; an uncited claim is not a claim")
        if not all(isinstance(s, str) and s.strip() for s in sources):
            raise JudgeParseError(f"claims[{i}].sources contains a non-URL entry: {sources!r}")
        out.append(Claim(cid, claim.strip(), tuple(s.strip() for s in sources)))

    # claim_id IS THE IDENTITY OF A CITATION JUDGMENT, so it has to be unique.
    # `grade` hashes it into judgment_id ("citation", suite, task, run, str(claim_id), model,
    # prompt_sha, order), and harness.py dedups on judgment_id -- so two claims the judge
    # emitted with the same claim_id collide and the SECOND IS SILENTLY DROPPED. Verified:
    # two different claims both carrying claim_id=1 hash to 4b3e925c746bf504..., identically.
    #
    # The effect is not a crash but a shorter denominator: citation_support is computed over
    # fewer claims than the judge actually produced, and the loss is invisible because nothing
    # downstream ever sees the count the judge emitted. Refused here, alongside every other
    # malformed-payload check, because this is the last point at which the duplicate is visible.
    ids = [c.claim_id for c in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise JudgeParseError(
            f"claim_id is not unique: {dupes} repeated. claim_id is hashed into judgment_id, "
            "so duplicates collide and every claim after the first is silently discarded."
        )
    return tuple(out)


def parse_support(text: str) -> tuple[str, str]:
    obj = strict_json_object(text)
    return require_label(obj, "support", LABELS), justification_of(obj)


def extract_claims(llm: JudgeLLM, report: str, *, seed: int = 0) -> tuple[Claim, ...]:
    return parse_claims(ask(llm, prompt=build_extract_prompt(report), seed=seed))


def grade(
    llm: JudgeLLM,
    *,
    report: str,
    docs: Mapping[str, str],
    suite_id: str,
    task_id: str,
    run_id: str,
    judge_model: str,
    judge_family: str = "openai",
    seed: int = 0,
    order: str = "ab",
    retest_group_id: str | None = None,
) -> tuple[tuple[Judgment, ...], dict[str, Any]]:
    """(judgments, diagnostics). One Judgment per CHECKED claim.

    diagnostics carries n_claims, n_checked and n_unresolved -- the claims whose cited URLs
    have no text here. Reporting the mean without that count would let a report full of
    unfetchable citations look like a report with no citation problems.
    """
    # SKIP BEFORE SPENDING. Extraction is a live LLM call, and every claim it returns is
    # dropped below unless `docs` carries text for one of its cited URLs. With no documents
    # that loop discards everything, so the call buys nothing -- measured, it bought nothing
    # 860 times in one scoring pass (540 drgym + 320 musique), and would again on every pass.
    #
    # `pi_eval.score.load_judge_docs` already documents this as the intent ("an absent
    # sidecar means the citation judge IS NOT RUN for that run and the omission is counted");
    # it was simply never implemented at the call site. Nothing writes judge_docs.json today,
    # so this is the state of every run.
    #
    # The reason is reported as what it IS. "no judgment parsed" sends the next reader to the
    # parser, and the parser is fine: on a real drgym report it extracts 7 claims correctly.
    if not report.strip() or not any(str(v).strip() for v in docs.values()):
        return (), {
            "n_claims": 0,
            "n_checked": 0,
            "n_unresolved": 0,
            "skipped": True,
            "why": (
                "no source documents for this run (runs/<run_id>/judge_docs.json is absent "
                "or empty), so no cited claim could be checked against anything"
            )
            if report.strip()
            else "the report is empty, so it carries no cited claims",
        }
    claims = extract_claims(llm, report, seed=seed)
    words = n_words(report)
    out: list[Judgment] = []
    unresolved: list[int] = []
    for c in claims:
        if not any(docs.get(u) for u in c.sources):
            unresolved.append(c.claim_id)
            continue
        raw = ask(llm, prompt=build_check_prompt(c.claim, c.sources, docs), seed=seed)
        label, _why = parse_support(raw)
        out.append(
            Judgment(
                judgment_id=judgment_id(
                    "citation",
                    suite_id,
                    task_id,
                    run_id,
                    str(c.claim_id),
                    judge_model,
                    PROMPT_SHA,
                    order,
                ),
                run_id_a=run_id,
                run_id_b=None,
                suite_id=suite_id,
                task_id=task_id,
                criterion="citation_support",
                order=order,  # type: ignore[arg-type]
                judge_family=judge_family,
                judge_model=judge_model,
                judge_prompt_sha=PROMPT_SHA,
                label=label,
                score=CITATION_SCORE[label],
                key_point_id=f"claim{c.claim_id}",
                len_a_words=words,
                retest_group_id=retest_group_id,
                judge_pin=f"{judge_model}@{PROMPT_SHA[:12]}",
                response_sha=response_sha(raw),
            )
        )
    diag = {
        "n_claims": len(claims),
        "n_checked": len(out),
        "n_unresolved": len(unresolved),
        "unresolved_claim_ids": unresolved,
    }
    return tuple(out), diag


def mean_support(judgments: Sequence[Judgment]) -> float:
    """Mean of 1.0 / 0.5 / 0.0 over checked claims. NaN when nothing was checked."""
    scores = [
        j.score for j in judgments if j.criterion == "citation_support" and j.score is not None
    ]
    return sum(scores) / len(scores) if scores else float("nan")
