"""An automatic LLM annotation pass, asking a model the SAME questions the human annotation
tool asks. This module produces `annotator_kind: "llm"` records and nothing else -- what
those records may and may not become is decided entirely by `pi_eval.annotate` (`consensus`
counts human raters only, `merge_into_graphs` refuses a non-human consensus). Nothing here
overrides that; see that module's docstring for the argument.

FOLLOWS THE JUDGE PATTERN IN `pi_eval.judges._llm`, DELIBERATELY. Same `JudgeLLM` Protocol
(structural: `pi_eval` still does not import `pinq_adapters`, and the client is injected by
the caller), the same `ask`/`prompt_sha`/`response_sha`, and above all the same refusal to
default a reply it cannot read.

WHY THAT REFUSAL IS STRONGER HERE THAN IT IS FOR A JUDGE. A judge that defaults to a label
biases a recall estimate. An annotation pass that defaults an A1 answer FABRICATES a claim
about what a person would have thought to ask -- the exact quantity `pi_eval.annotate`'s
whole rater-kind wall exists to keep a model from ever supplying. So a reply this module
cannot parse raises `AnnotationParseError` (a `JudgeParseError`, so an existing `except
JudgeParseError` still catches it) and the caller counts the item as unparsed. It is never
silently turned into a well-formed record.

WHY A1 RATES EVERY NODE, TICKED OR NOT. `gold_usefulness_rating` and the ADR reference level
are read off `usefulness`, and the anticipated side (what a person would ask) is the
denominator ADR is measured against. Asking the model to rate only the nodes it ticked would
make the anticipated branch the expensive one to annotate, and it is the one this whole
campaign exists to measure.

WHY A2 IS SHOWN NEITHER `chosen_run_id` NOR THE MARGIN. `sample_a2_items` already blinds the
bundle for a human; the LLM pass reads the same bundle a human would read, so the model's
preference is a genuine, unhinted judgment rather than an echo of the pipeline's own pick.

WHY `role="annotator"` RATHER THAN THE JUDGE ROLE `ask` DEFAULTS TO. Judging and annotating
are different measurements with different model pins (`PI_MODEL_JUDGE` vs `PI_MODEL_ANNOTATOR`
-- see `pi_run.cmd_annotate`), and `MeteredClient.model_for` resolves a role from an explicit
`models={...}` override before it ever consults the environment, so passing `role="annotator"`
here is what keeps this pass from silently resolving to whatever the judge happens to be
pinned to. (`ask`'s `actor="judge"` on the wire is a fixed artifact of the shared helper and
affects only this process's own call-telemetry bookkeeping, never which model answers.)

RATIONALE, `reasoning`, `basis` AND (for A1) `likelihood` ARE TOP-LEVEL RECORD FIELDS, SIBLINGS
OF `response`, NEVER INSIDE IT. `response` is the object `pi_eval.annotate.consensus`/
`iaa_report` compute agreement over; free text entering it would sit inside an equality
comparison those depend on, and `likelihood` is a per-node number those functions were never
written to average -- it would corrupt the tick-vs-tick comparison they already do. A
rationale is REQUIRED here (see `AnnotationParseError` in `parse_reply`) because a model can
always say why in the same call for near-zero marginal cost, and a model verdict with no stated
reason is the one hardest to audit later -- the same reason a human annotator is NOT required
to (see `pi_eval.annotate.validate_records`, which lets a human record omit it). `reasoning` is
different again: it is the provider's own reasoning STRING for the call, when one exists, never
synthesised and never evidence for a verdict -- see `_reasoning_trace`.

WHY A1 ALSO ASKS FOR A PER-NODE `likelihood` (0..100), NOT JUST `ticked`. A live pass over the
gpt-oss-120b model measured an A1 tick rate over 80% -- a response distribution pinned near
saturation carries almost no information about whether the model is closer to a person than to
chance, whichever way the true rate turns out to lie. A checkbox is trivial to saturate; a
number the model must commit to per candidate is not, and it survives being compared to a real
human tick rate once one exists, where a binary would only ever say "ticked" or not. This is
deliberately NOT a second attempt to recover `is_latent`/`gold_depth` under a different name --
`likelihood` is never checked against depth anywhere in this module, and the prompt asks only
about what a person would have thought to ask, exactly as `ticked` already does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from pi_eval.annotate import (
    _LABELS,
    A2_BASIS_VALUES,
    A7_BASIS_VALUES,
    LLM_ID_PREFIX,
    RATIONALE_MAX_CHARS,
)
from pi_eval.judges._llm import (
    JudgeLLM,
    JudgeParseError,
    ask_with_telemetry,
    prompt_sha,
    require_label,
    response_sha,
    strict_json_object,
)

TOOL_VERSION = "pi_annotate_llm/1"

# Pinned at the call site, like the judge's -- but at 1.0, and NOT by preference.
#
# The frontier models refuse anything else. Probed against the live proxy, every GPT-5.5 and
# GPT-5.6 variant answers "Unsupported value: 'temperature' does not support 0.0 with this
# model. Only the default (1) value is supported." So the old justification here -- that t=0
# keeps two passes over one bundle from disagreeing with themselves -- describes a setting
# that is no longer available, and leaving it would have been a comment asserting a property
# the code cannot have.
#
# What carries reproducibility instead is what already carried it for the judges: the request
# cache is keyed on the exact request bytes, so re-running a scored pass REPLAYS rather than
# re-samples, and a re-scored number is recomputed from the cache rather than re-measured.
# The temperature rides on `model_pin` (`<model>@t1.0`), so a pass run at a different one is a
# different rater and cannot be pooled with this one by accident.
#
# There is also a measurement argument for sampling rather than taking the mode. A single t=0
# answer is a point estimate of the model's most likely response; what this campaign wants to
# know is how a model's DISTRIBUTION compares with a human tick rate, and that is what
# repeated draws at t=1 -- and the per-need `likelihood` the A1 prompt asks for -- actually
# estimate.
ANNOTATOR_TEMPERATURE = 1.0
_EMPHASIS_B4 = " \t\n*_`\"'"

SYSTEM = "You are a careful annotator. You reply with one JSON object and nothing else."


class AnnotationParseError(JudgeParseError):
    """A model's annotation reply this module cannot read. Never downgraded to a label -- see
    the module docstring for why that is a stronger requirement here than it is for a judge."""


def _node_ids_of(item: Mapping[str, Any]) -> tuple[dict, ...]:
    return tuple((item.get("payload") or {}).get("nodes") or ())


# --------------------------------------------------------------------------- prompts


def _lines_evidence(evidence: Any) -> list[str]:
    """No evidence means no section. A header with nothing under it asserts that the evidence
    set is EMPTY, which is a different claim from "not shown to you" and the one more likely
    to change a judgment."""
    if not evidence:
        return []
    out = ["", "Evidence gathered so far:"]
    for e in evidence:
        out.append(f"  - ({e.get('title', '')}) {e.get('text', '')}")
    return out


def _build_a1(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    question = str(ctx.get("question") or "")
    nodes = _node_ids_of(item)
    lines = [
        "Before reading anything, which of these would a typical person asking this question "
        "have thought to ask about?",
        "",
        f"Question: {question}",
        "",
        "This is a prediction of what a PERSON would anticipate wanting to know, not what is "
        "logically required to answer the question -- those two differ, and the gap between "
        "them is exactly what is being measured.",
        "",
        "Candidate follow-up needs:",
    ]
    for n in nodes:
        lines.append(f"- {n['node_id']}: {n['text']}")
    lines += [
        "",
        "Most people bring only a small, specific set of things they already want to know when "
        "they ask a question like this -- not a checklist covering everything the topic "
        "touches. If you notice yourself ticking most of the candidates above, stop and "
        "re-read your selection: that pattern usually means you judged whether a need helps "
        "answer the question rather than whether an ordinary asker would actually have named "
        "it up front. Tick only what a typical person would have thought to ask, before "
        "reading anything.",
        "",
        "Rate EVERY candidate above on a 1..5 usefulness scale, whether or not you tick it --"
        " do not omit any of them.",
        "",
        "Also give, for every candidate above, ticked or not, a likelihood: out of 100 people "
        "who asked this exact question, how many would have listed this need before starting? "
        "Answer with a number from 0 to 100 for each one.",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " selection.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"ticked": [<node_id>, ...], "usefulness": {"<node_id>": <1..5>, ...}, '
        '"likelihood": {"<node_id>": <0..100>, ...}, '
        '"rationale": "<why you ticked what you ticked>"}',
    ]
    return "\n".join(lines)


def _build_a2(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    payload = item.get("payload") or {}
    question = str(ctx.get("question") or "")
    a_text = str((payload.get("option_a") or {}).get("question") or "")
    b_text = str((payload.get("option_b") or {}).get("question") or "")
    lines = [
        "Which is the better next question to ask, to move toward a complete answer?",
        "",
        f"Original question: {question}",
    ]
    history = ctx.get("history") or ()
    if history:
        lines.append("")
        lines.append("Conversation so far:")
        for h in history:
            lines.append(f"  Q: {h.get('q', '')}")
            lines.append(f"  A: {h.get('a', '')}")
    lines += _lines_evidence(ctx.get("evidence"))
    draft = str(ctx.get("draft") or "")
    if draft:
        lines.append("")
        lines.append(f"Current draft answer: {draft}")
    # THE FALLBACK IS THE NORMAL CASE, NOT AN EDGE ONE. `sample_a2_items` replays the parent
    # run to produce structured evidence/history/draft and falls back to the verbatim rendered
    # prompt when that replay is unavailable -- which, measured on the live bundle, was every
    # single A2 item, because the only preference pairs on disk come from `dev-` runs whose
    # state cannot be replayed. Without this the model chose between two questions having read
    # nothing, while the human read the whole state in a panel: not a weaker judgment, a
    # judgment about a different question, and an agreement number over the two would compare
    # nothing. Shown only when the structured fields are absent, so a good replay is never
    # displaced by a raw template.
    if not history and not (ctx.get("evidence") or ()) and not draft:
        state = str(ctx.get("state_text") or "")
        if state:
            lines += ["", "What the system had in front of it:", state]
    lines += [
        "",
        f"Option A: {a_text}",
        f"Option B: {b_text}",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " choice. If you picked a or b (not a tie or both_bad), also name the single biggest"
        " reason it won, from exactly one of: targets_an_unresolved_need, names_its_entities,"
        " less_redundant, better_scoped, other -- use other if none of those genuinely fits.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"choice": "a"|"b"|"tie"|"both_bad", "rationale": "<why>", "basis": '
        '"targets_an_unresolved_need"|"names_its_entities"|"less_redundant"|"better_scoped"'
        '|"other"}',
        '(omit "basis" if choice is "tie" or "both_bad")',
    ]
    return "\n".join(lines)


def _build_a3_node(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    lines = [
        "Judge this candidate follow-up need against the question and the evidence for it.",
        "",
        f"Question: {ctx.get('question', '')}",
        f"Candidate need: {ctx.get('node_text', '')}",
    ]
    lines += _lines_evidence(ctx.get("evidence"))
    lines += [
        "",
        "verdict: is this need required to reach a complete answer, merely optional / "
        "nice-to-have, or not actually a need at all?",
        "discoverability: could a typical person have found this by looking it up (kb), does "
        "it require knowledge only the asker has (user_private), or can you not tell?",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " verdict.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"verdict": "required"|"optional"|"not_a_need", '
        '"discoverability": "kb"|"user_private"|"cant_tell", "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_a3_edge(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    lines = [
        # DIRECTION, stated the way `pi_eval.gold.GoldEdge` defines it: "prerequisite means
        # v_dst is UNANSWERABLE until v_src is resolved". SRC is the prerequisite, DST the
        # dependent. These labels were inverted, and a live pass returned 167 of 200 wiki2
        # edges as False -- on a suite where an edge exists exactly when
        # `objects[i] == subjects[j]`, so it is true by construction -- with rationales that
        # read the dependency backwards. The human tool asks "must the first be resolved
        # before the second is even askable?"; asking a model the mirror image of that makes
        # the two answers non-comparable, which is the whole point of running both.
        "Must the first need below be resolved before the second can even be ASKED, "
        "given the question?",
        "",
        f"Question: {ctx.get('question', '')}",
        f"Need A (the possible prerequisite): {ctx.get('src_text', '')}",
        f"Need B (the one that may depend on A): {ctx.get('dst_text', '')}",
        "",
        "Answer true only if B is unaskable until A is known. Two facts that could be looked "
        "up in either order are not a prerequisite, however naturally one follows the other.",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " answer.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"holds": true|false|"unsure", "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_a3_match(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    lines = [
        "Does the question that was actually asked address this candidate need?",
        "",
        f"Question asked: {ctx.get('asked_question', '')}",
        f"Candidate need: {ctx.get('node_text', '')}",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " answer.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"addresses": true|false, "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_a3_missing(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    lines = [
        "Below is the full list of follow-up needs already identified for this question. Are "
        "there any genuine needs a typical person would have that are NOT already on this "
        "list? An empty list is a legitimate answer.",
        "",
        f"Question: {ctx.get('question', '')}",
        "",
        "Already identified:",
    ]
    for n in ctx.get("node_list") or ():
        lines.append(f"- {n}")
    lines += [
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale -- why"
        " those are (or are not) genuine gaps.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"missing_needs": [<str>, ...], "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_a4(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    lines = [
        "Ignore whether the ANSWER appears anywhere. Judge only this: reading the original"
        " task and nothing else, could someone have known to ASK this question?",
        "",
        "Say `stated_in_task` if the task already names or refers to the thing being asked"
        " about -- even when the task does not say what that thing IS. Say `evidence_only`"
        " only if the question could not have been formulated without first reading evidence"
        " gathered along the way.",
        "",
        "A question whose answer is absent from the task is still `stated_in_task` when the"
        " task told you the thing existed.",
        "",
        f"Original question: {ctx.get('question', '')}",
        f"Question asked: {ctx.get('asked_question', '')}",
    ]
    parents = ctx.get("parent_units") or ()
    reply_shape = '{"latency": "stated_in_task"|"evidence_only"|"unsure"'
    if parents:
        lines.append("")
        lines.append("The asker says this question was prompted by:")
        for u in parents:
            lines.append(f"  - ({u.get('title', '')}) {u.get('text', '')}")
        lines.append("")
        lines.append("Also judge: are these plausible things that would have prompted it?")
        reply_shape += ', "parent_uids_ok": true|false'
    reply_shape += ', "rationale": "<why>"}'
    lines += [
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " answer.",
        "",
        "Reply with one JSON object and nothing else:",
        reply_shape,
    ]
    return "\n".join(lines)


def _build_a5(item: Mapping[str, Any]) -> str:
    """The A5 stop judgment, in two variants.

    THE DEFAULT VARIANT SHOWS THE RATER THE GOLD FRONTIER, and that turns out to decide the
    verdict. It prints "Candidate follow-up needs the run did NOT resolve before stopping"
    followed by the unresolved required nodes -- or "(none -- every mined need was resolved)".
    So the instrument names what is missing and then asks whether anything was missing.
    MEASURED 2026-09-07 over the 545 items with a clean-panel majority:

        needs named in the prompt   n     said "should have asked more"
        none                       264     0.0%
        1                          190    26.3%
        2                           62    79.0%
        3                           21   100.0%
        4                            8   100.0%

    Stratifying on that list and pooling, the separation against blinded `answer_correct`
    falls from 24.5 points to 9.0 (n=275), and the "no false positives on the complete-
    coverage states" result contributes nothing at all -- no rater ever said otherwise there
    because the prompt told it nothing remained.

    THE BLIND VARIANT (`payload["blind"]`) omits BOTH the final answer and the candidate
    section. Omits, not empties: printing "(none -- every mined need was resolved)" with no
    candidates would assert the opposite anchor just as strongly, and an empty
    "Final answer given:" line invites the rater to treat a missing answer as a failure.
    What is left -- the task question, the dialogue, the evidence actually retrieved -- is
    what a reader would have to judge a stop from.
    """
    ctx = item.get("context") or {}
    payload = item.get("payload") or {}
    blind = bool(payload.get("blind"))
    lines = [
        (
            "A run stopped after gathering the evidence below. Was stopping the right call, "
            "or was there more it should have found out first?"
            if blind
            else "A run stopped and gave the answer below. Was stopping the right call, or "
            "was there more it should have found out first?"
        ),
        "",
        f"Question: {ctx.get('question', '')}",
    ]
    history = ctx.get("history") or ()
    if history:
        lines.append("")
        lines.append("Conversation so far:")
        for h in history:
            lines.append(f"  Q: {h.get('q', '')}")
            lines.append(f"  A: {h.get('a', '')}")
    lines += _lines_evidence(ctx.get("evidence"))
    if not blind:
        lines += [
            "",
            f"Final answer given: {ctx.get('answer', '')}",
            "",
            "Candidate follow-up needs the run did NOT resolve before stopping:",
        ]
        candidates = payload.get("candidates") or ()
        if candidates:
            for c in candidates:
                lines.append(f"- {c.get('node_id')}: {c.get('text')}")
        else:
            lines.append("(none -- every mined need was resolved)")
    lines += [
        "",
        "verdict: was stopping here the right call (stopping_was_right), should it have asked "
        "more first (should_have_asked_more), should it have stopped earlier -- before some of "
        "what it already gathered (should_have_stopped_earlier) -- or can you not tell "
        "(cant_tell)?",
        (
            "If, and only if, your verdict is should_have_asked_more, say in a few words what "
            "it still needed to find out."
            if blind
            else "If, and only if, your verdict is should_have_asked_more, name every "
            "candidate above (by its node_id) that was still genuinely needed."
        ),
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " verdict.",
        "",
        "Reply with one JSON object and nothing else:",
        (
            '{"verdict": "stopping_was_right"|"should_have_asked_more"|'
            '"should_have_stopped_earlier"|"cant_tell", '
            '"missing": ["<what it still needed to find out>", ...], "rationale": "<why>"}'
            if blind
            else '{"verdict": "stopping_was_right"|"should_have_asked_more"|'
            '"should_have_stopped_earlier"|"cant_tell", "missing": [<node_id>, ...], '
            '"rationale": "<why>"}'
        ),
        '(omit "missing", or leave it empty, unless verdict is "should_have_asked_more")',
    ]
    return "\n".join(lines)


def _build_a6(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    payload = item.get("payload") or {}
    lines = [
        "Rank these candidate next questions, best first, by how well each moves toward a "
        "complete answer.",
        "",
        f"Original question: {ctx.get('question', '')}",
    ]
    history = ctx.get("history") or ()
    if history:
        lines.append("")
        lines.append("Conversation so far:")
        for h in history:
            lines.append(f"  Q: {h.get('q', '')}")
            lines.append(f"  A: {h.get('a', '')}")
    lines += _lines_evidence(ctx.get("evidence"))
    draft = str(ctx.get("draft") or "")
    if draft:
        lines.append("")
        lines.append(f"Current draft answer: {draft}")
    # Same fallback, for the same measured reason, as `_build_a2`: A6 items are built by A2's
    # replay path, so when that replay is unavailable the model must read the state the human
    # reads in the panel, or the two are answering about different states.
    if not history and not (ctx.get("evidence") or ()) and not draft:
        state = str(ctx.get("state_text") or "")
        if state:
            lines += ["", "What the system had in front of it:", state]
    lines += ["", "Candidates:"]
    for c in payload.get("candidates") or ():
        lines.append(f"- {c.get('candidate_id')}: {c.get('question')}")
    lines += [
        "",
        "Answer with a TIER for every candidate above: 1 is best, higher numbers are worse.",
        "TIERS ARE NOT A STRICT ORDER. Give two candidates the SAME tier when you genuinely "
        "consider them equivalent -- that is a tie, and it is a real answer. Do not break a "
        "tie you do not believe in; a preference you invented is worse than a tie you meant. "
        'Gaps are allowed: {"c0": 1, "c1": 1, "c2": 3} says the first two are tied at the '
        "top and the third is clearly worse.",
        "Every candidate must appear exactly once, and every tier must be a positive whole number.",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " ranking.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"tiers": {"<candidate_id>": <positive integer>, ...}, "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_a7(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    payload = item.get("payload") or {}
    question = str(ctx.get("question") or "")
    a_text = str((payload.get("option_a") or {}).get("question") or "")
    b_text = str((payload.get("option_b") or {}).get("question") or "")
    lines = [
        "Two candidate next questions were proposed from the same point in the same "
        "conversation. Judge each one, then say which is the better ANTICIPATORY move.",
        "",
        f"Original question: {question}",
    ]
    history = ctx.get("history") or ()
    if history:
        lines.append("")
        lines.append("Conversation so far:")
        for h in history:
            lines.append(f"  Q: {h.get('q', '')}")
            lines.append(f"  A: {h.get('a', '')}")
    lines += _lines_evidence(ctx.get("evidence"))
    draft = str(ctx.get("draft") or "")
    if draft:
        lines.append("")
        lines.append(f"Current draft answer: {draft}")
    # Same fallback, for the same measured reason, as `_build_a2`: A7 items are built by A2's
    # replay path, so when that replay is unavailable the model must read the state the human
    # reads in the panel, or the two are answering about different states.
    if not history and not (ctx.get("evidence") or ()) and not draft:
        state = str(ctx.get("state_text") or "")
        if state:
            lines += ["", "What the system had in front of it:", state]
    lines += [
        "",
        f"Option A: {a_text}",
        f"Option B: {b_text}",
        "",
        "For EACH option, judge its reach: does this question ask about something the task "
        "statement already names (stays_stated), or something only nameable after reading -- "
        "a need the original question never states (reaches_unstated)? Answer cant_tell only "
        "when you genuinely cannot decide.",
        "For each side you judge reaches_unstated, you MUST also name the unstated need it "
        "reaches for, in unstated_need_a / unstated_need_b -- the named need is what makes "
        "the judgment auditable. Omit that field for any other side.",
        "",
        "preference: which option is the better ANTICIPATORY move? A polished question about "
        "something the task already names loses to a clumsy one reaching a genuine unstated "
        "need. Use tie or both_bad when neither wins.",
        "If you picked a or b (not a tie or both_bad), also name what separated them, from "
        "exactly one of: anticipation (the winner reaches an unstated need the loser does "
        "not), hygiene (both reach equally far; the winner is simply the better-formed "
        "question), no_difference -- use no_difference if nothing genuinely separated them.",
        "",
        "Also give a short (one or two sentence, at most 400 characters) rationale for your"
        " judgment.",
        "",
        "Reply with one JSON object and nothing else:",
        '{"a_reaches": "reaches_unstated"|"stays_stated"|"cant_tell", '
        '"b_reaches": "reaches_unstated"|"stays_stated"|"cant_tell", '
        '"unstated_need_a": "<the unstated need option A reaches for>", '
        '"unstated_need_b": "<the unstated need option B reaches for>", '
        '"preference": "a"|"b"|"tie"|"both_bad", '
        '"basis": "anticipation"|"hygiene"|"no_difference", "rationale": "<why>"}',
        '(include "unstated_need_a"/"unstated_need_b" ONLY for a side judged '
        '"reaches_unstated"; omit "basis" if preference is "tie" or "both_bad")',
    ]
    return "\n".join(lines)


def _lines_conv_state(ctx) -> list[str]:
    """The three sections every B item shows. Shared so the three prompts cannot drift into
    describing the same state in three different shapes, which would make their verdicts
    incomparable for no reason anybody chose."""
    return [
        f"The person's task: {ctx.get('task', '')}",
        "",
        "What has been said since:",
        str(ctx.get("prior_turns", "")) or "(none)",
        "",
        "What the agent has already done:",
        str(ctx.get("tool_trace", "")) or "(none)",
    ]


_B_RATIONALE = (
    "Also give a short (one or two sentences, at most 400 characters) rationale.",
    "",
    "Reply with one JSON object and nothing else:",
)


def _build_b1(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    return "\n".join(
        [
            "An agent working with a person stopped and asked them the question below. Judge "
            "whether it needed to ask, using ONLY what is shown here.",
            "",
            *_lines_conv_state(ctx),
            "",
            f"The question it asked: {ctx.get('asked_question', '')}",
            "",
            "verdict: was asking necessary (necessary), was the answer already stated above "
            "(answer_in_state), could the answer have been worked out from what is above "
            "without asking (answer_inferable), was there an obvious default it could have "
            "taken and mentioned instead (default_existed), or can you not tell (cant_tell)?",
            "If, and only if, your verdict is default_existed, say in default_action what that "
            "obvious default was.",
            "",
            *_B_RATIONALE,
            '{"verdict": "necessary"|"answer_in_state"|"answer_inferable"|"default_existed"'
            '|"cant_tell", "default_action": "<the default>", "rationale": "<why>"}',
            '(omit "default_action" unless verdict is "default_existed")',
        ]
    )


def _build_b1b(item: Mapping[str, Any]) -> str:
    """B1 without the default option. Everything else is word-for-word B1's, because a prompt
    that also reworded the question would confound the one change under test."""
    ctx = item.get("context") or {}
    return "\n".join(
        [
            "An agent working with a person stopped and asked them the question below. Judge "
            "whether it needed to ask, using ONLY what is shown here.",
            "",
            *_lines_conv_state(ctx),
            "",
            f"The question it asked: {ctx.get('asked_question', '')}",
            "",
            "verdict: was asking necessary (necessary), was the answer already stated above "
            "(answer_in_state), could the answer have been worked out from what is above "
            "without asking (answer_inferable), or can you not tell (cant_tell)?",
            "",
            *_B_RATIONALE,
            '{"verdict": "necessary"|"answer_in_state"|"answer_inferable"|"cant_tell", '
            '"rationale": "<why>"}',
        ]
    )


def _build_b4(item: Mapping[str, Any]) -> str:
    """Propose the question, or decline. Shown the same state B3 sees and NOT the reply --
    showing what the person said next would turn proposing into copying, and every proposal
    would then be trivially necessary."""
    ctx = item.get("context") or {}
    return "\n".join(
        [
            "An agent working with a person reached the point below and carried on without "
            "asking them anything. Judge whether there was a question it should have asked "
            "first, using ONLY what is shown here.",
            "",
            *_lines_conv_state(ctx),
            "",
            f"The last thing the agent said: {ctx.get('final_assistant_text', '')}",
            "",
            "verdict: was there a question it should have asked before going on "
            "(question_needed), was none needed because the work could proceed on what it had "
            "(none_needed), or can you not tell (cant_tell)?",
            "Answer none_needed whenever the agent could reasonably have picked a default and "
            "said so. Most of the time that is the right answer; a question is worth a "
            "person's attention only when the answer would change what happens next and "
            "nothing above settles it.",
            "If, and only if, your verdict is question_needed, give in question the exact "
            "question it should have asked, phrased as you would put it to the person.",
            "",
            *_B_RATIONALE,
            '{"verdict": "question_needed"|"none_needed"|"cant_tell", '
            '"question": "<the question>", "rationale": "<why>"}',
            '(omit "question" unless verdict is "question_needed")',
        ]
    )


def _build_b2(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    payload = item.get("payload") or {}
    lines = [
        "An agent reported back to a person, and the person replied with the things below. "
        "For each one, judge whether the agent could have known it was wanted from what is "
        "shown, without being told.",
        "",
        *_lines_conv_state(ctx),
        "",
        "What the person then said:",
        str(ctx.get("next_turn", "")),
        "",
        "The separate things they asked for:",
    ]
    for entry in payload.get("items") or ():
        lines.append(f"  {entry.get('item_key')}: {entry.get('text')}")
    lines += [
        "",
        "For each key above give one label: it was already stated earlier "
        "(stated_already), it could have been worked out from what is above "
        "(inferable_from_state), it depends on something only this person knows "
        "(user_private), it is a new task unrelated to what came before (new_task), or you "
        "cannot tell (cant_tell).",
        "Label only the keys you are sure about. Leaving a key out is fine and is not the "
        "same as calling it new_task.",
        "",
        *_B_RATIONALE,
        '{"verdicts": {"<key>": "stated_already"|"inferable_from_state"|"user_private"'
        '|"new_task"|"cant_tell", ...}, "rationale": "<why>"}',
    ]
    return "\n".join(lines)


def _build_b3(item: Mapping[str, Any]) -> str:
    ctx = item.get("context") or {}
    return "\n".join(
        [
            "An agent working with a person reached the point below. Judge what it should have "
            "done there, using ONLY what is shown here.",
            "",
            *_lines_conv_state(ctx),
            "",
            f"The last thing the agent said: {ctx.get('final_assistant_text', '')}",
            "",
            "verdict: was stopping to report the right call (stop_was_right), should it have "
            "asked the person something first (should_have_asked), should it have carried on "
            "working without stopping (should_have_continued), should it have stopped earlier "
            "than it did (should_have_stopped_earlier), or can you not tell (cant_tell)?",
            "If, and only if, your verdict is should_have_asked, give in question the exact "
            "question it should have asked.",
            "",
            *_B_RATIONALE,
            '{"verdict": "stop_was_right"|"should_have_asked"|"should_have_continued"'
            '|"should_have_stopped_earlier"|"cant_tell", "question": "<the question>", '
            '"rationale": "<why>"}',
            '(omit "question" unless verdict is "should_have_asked")',
        ]
    )


_PROMPT_BUILDERS = {
    "A1": _build_a1,
    "A2": _build_a2,
    "A3_node": _build_a3_node,
    "A3_edge": _build_a3_edge,
    "A3_match": _build_a3_match,
    "A3_missing": _build_a3_missing,
    "A4": _build_a4,
    "A5": _build_a5,
    "A6": _build_a6,
    "A7": _build_a7,
    "B1": _build_b1,
    "B1b": _build_b1b,
    "B4": _build_b4,
    "B2": _build_b2,
    "B3": _build_b3,
}


def build_prompt(item: Mapping[str, Any]) -> str:
    tt = str(item.get("task_type"))
    builder = _PROMPT_BUILDERS.get(tt)
    if builder is None:
        raise ValueError(f"no LLM-pass prompt for task_type {tt!r}")
    return builder(item)


# --------------------------------------------------------------------------- parsing


def _parse_a1(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    node_ids = {str(n["node_id"]) for n in _node_ids_of(item)}
    ticked = obj.get("ticked")
    if not isinstance(ticked, list) or not all(isinstance(x, str) for x in ticked):
        raise AnnotationParseError(f"ticked={ticked!r} must be a list of node ids")
    ticked_set = {str(x) for x in ticked}
    if not ticked_set <= node_ids:
        raise AnnotationParseError(f"ticked {ticked_set - node_ids} are not on this item")
    usefulness = obj.get("usefulness")
    if not isinstance(usefulness, Mapping):
        raise AnnotationParseError(f"usefulness={usefulness!r} must be an object")
    # Every node, ticked or not -- see the module docstring on why the anticipated branch
    # must not be the cheap one to skip rating.
    missing = node_ids - {str(k) for k in usefulness}
    if missing:
        raise AnnotationParseError(f"usefulness has no rating for {sorted(missing)}")
    ratings: dict[str, float] = {}
    for nid, v in usefulness.items():
        if str(nid) not in node_ids:
            raise AnnotationParseError(f"usefulness rating for {nid} is not on this item")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnnotationParseError(f"usefulness[{nid}]={v!r} must be a number")
        if not (1.0 <= float(v) <= 5.0):
            raise AnnotationParseError(f"usefulness[{nid}]={v} is outside 1..5")
        ratings[str(nid)] = float(v)
    return {"ticked": sorted(ticked_set), "usefulness": ratings}


def _parse_a1_likelihood(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, float]:
    """The per-node 0..100 estimate parsed OUT of `response` (see the module docstring on why
    it must land as a sibling of it, not a member). Validated exactly as strictly as
    `usefulness` above -- every node id present, no id that is not on this item, no non-numeric
    or out-of-range value -- because a defaulted likelihood would be the same fabrication this
    module already refuses for `ticked`/`usefulness`, just in a shape a downstream average could
    hide."""
    node_ids = {str(n["node_id"]) for n in _node_ids_of(item)}
    likelihood = obj.get("likelihood")
    if not isinstance(likelihood, Mapping):
        raise AnnotationParseError(f"likelihood={likelihood!r} must be an object")
    missing = node_ids - {str(k) for k in likelihood}
    if missing:
        raise AnnotationParseError(f"likelihood has no value for {sorted(missing)}")
    out: dict[str, float] = {}
    for nid, v in likelihood.items():
        if str(nid) not in node_ids:
            raise AnnotationParseError(f"likelihood for {nid} is not on this item")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnnotationParseError(f"likelihood[{nid}]={v!r} must be a number")
        if not (0.0 <= float(v) <= 100.0):
            raise AnnotationParseError(f"likelihood[{nid}]={v} is outside 0..100")
        out[str(nid)] = float(v)
    return out


def _parse_a2(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    return {"choice": require_label(obj, "choice", ("a", "b", "tie", "both_bad"))}


def _parse_a3_node(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verdict": require_label(obj, "verdict", ("required", "optional", "not_a_need")),
        "discoverability": require_label(
            obj, "discoverability", ("kb", "user_private", "cant_tell")
        ),
    }


def _parse_a3_edge(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    if "holds" not in obj:
        raise AnnotationParseError(f"verdict has no 'holds' field; keys: {sorted(obj)}")
    holds = obj["holds"]
    if holds not in (True, False, "unsure"):
        raise AnnotationParseError(f"holds={holds!r} is not one of (True, False, 'unsure')")
    return {"holds": holds}


def _parse_a3_match(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    if "addresses" not in obj:
        raise AnnotationParseError(f"verdict has no 'addresses' field; keys: {sorted(obj)}")
    addresses = obj["addresses"]
    if not isinstance(addresses, bool):
        raise AnnotationParseError(f"addresses={addresses!r} must be a bool")
    return {"addresses": addresses}


def _parse_a3_missing(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    needs = obj.get("missing_needs")
    if not isinstance(needs, list) or not all(isinstance(x, str) for x in needs):
        raise AnnotationParseError(f"missing_needs={needs!r} must be a list of strings")
    return {"missing_needs": list(needs)}  # an empty list is a real answer, not a parse failure


def _parse_a4(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "latency": require_label(obj, "latency", ("stated_in_task", "evidence_only", "unsure"))
    }
    if (item.get("context") or {}).get("parent_units"):
        ok = obj.get("parent_uids_ok")
        if not isinstance(ok, bool):
            raise AnnotationParseError(
                f"parent_uids_ok={ok!r} must be a bool: this item carries parent_units, so the "
                "sub-judgment is answerable and its absence is not a valid reply"
            )
        out["parent_uids_ok"] = ok
    return out


_A5_VERDICTS = (
    "stopping_was_right",
    "should_have_asked_more",
    "should_have_stopped_earlier",
    "cant_tell",
)


def _parse_a5(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    verdict = require_label(obj, "verdict", _A5_VERDICTS)
    payload = item.get("payload") or {}
    blind = bool(payload.get("blind"))
    node_ids = {str(c.get("node_id")) for c in payload.get("candidates") or ()}
    missing = obj.get("missing")
    if missing is None:
        missing = []
    if not isinstance(missing, list) or not all(isinstance(x, str) for x in missing):
        raise AnnotationParseError(f"missing={missing!r} must be a list of strings")
    missing_set = {str(x) for x in missing}
    # A BLIND ITEM SHIPS NO CANDIDATES, so `missing` is free prose and there is nothing to
    # check it against. Without this branch every blind `should_have_asked_more` response
    # fails to parse -- the membership test below rejects it against an EMPTY node id set, and
    # then the rule two blocks down requires it to be non-empty. The verdict most of the
    # instrument exists to detect would have been discarded, silently, on every item; the
    # rationale cap did exactly this and cost 24% of a campaign's coverage.
    if not blind and not missing_set <= node_ids:
        raise AnnotationParseError(
            f"missing {missing_set - node_ids} are not candidates on this item"
        )
    # REQUIRED iff should_have_asked_more, else it must be empty -- same rule
    # `pi_eval.annotate._response_errors` enforces for a human record, and for the identical
    # reason: `missing` is what turns a verdict into training data (a supervised example of
    # the SPECIFIC question the policy should have asked from this state), and a verdict that
    # claims something was missed without naming it is unusable for that purpose, not merely
    # incomplete.
    if verdict == "should_have_asked_more" and not missing_set:
        raise AnnotationParseError(
            "verdict=should_have_asked_more requires a non-empty 'missing' naming which "
            "candidate(s) were still needed"
        )
    if verdict != "should_have_asked_more" and missing_set:
        raise AnnotationParseError(
            f"missing={sorted(missing_set)} but verdict={verdict!r} is not "
            "should_have_asked_more: missing must be empty for every other verdict"
        )
    return {"verdict": verdict, "missing": sorted(missing_set)}


def _parse_a6(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    """Every candidate, exactly once, as a positive integer tier.

    A MISSING CANDIDATE IS A PARSE ERROR, not a partial ranking. That candidate has no pairwise
    reading against any other, so `pi_eval.annotate.a6_pair_labels` emits NOTHING for the whole
    item -- a silently dropped ranking, which is the shape of failure this module exists to
    refuse. Defaulting the absent tier would be worse still: it would fabricate C(k,2)-many
    preferences the model never expressed, straight into DPO training data.

    Equal tiers are left exactly as given: a tie is a real answer and is never broken here.
    """
    candidate_ids = {
        str(c.get("candidate_id")) for c in (item.get("payload") or {}).get("candidates") or ()
    }
    tiers = obj.get("tiers")
    if not isinstance(tiers, Mapping):
        raise AnnotationParseError(f"tiers={tiers!r} must be an object of candidate_id -> tier")
    missing = candidate_ids - {str(k) for k in tiers}
    if missing:
        raise AnnotationParseError(f"tiers has no tier for {sorted(missing)}")
    out: dict[str, int] = {}
    for cid, v in tiers.items():
        if str(cid) not in candidate_ids:
            raise AnnotationParseError(f"tier for {cid} is not a candidate on this item")
        if isinstance(v, bool) or not isinstance(v, int):
            raise AnnotationParseError(f"tiers[{cid}]={v!r} must be a positive integer")
        if v < 1:
            raise AnnotationParseError(f"tiers[{cid}]={v} must be a positive integer")
        out[str(cid)] = int(v)
    return {"tiers": out}


_A7_REACHES = ("reaches_unstated", "stays_stated", "cant_tell")


def _parse_a7(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    """Per-side reach, the preference, and the named need iff that side reaches.

    `unstated_need_<side>` is REQUIRED iff that side is `reaches_unstated`, in both
    directions -- same rule `pi_eval.annotate._response_errors` enforces for a human record,
    and for the identical reason A5's `missing` has it: the named need is what makes a
    reaches label auditable rather than a vibe, so a side that claims to reach for something
    unstated without naming it is not a smaller version of a valid answer, it is an unusable
    one -- and a need named for a stays_stated side is an invented one.
    """
    out: dict[str, Any] = {
        "a_reaches": require_label(obj, "a_reaches", _A7_REACHES),
        "b_reaches": require_label(obj, "b_reaches", _A7_REACHES),
        "preference": require_label(obj, "preference", ("a", "b", "tie", "both_bad")),
    }
    for side in ("a", "b"):
        need = obj.get(f"unstated_need_{side}")
        if out[f"{side}_reaches"] == "reaches_unstated":
            if not isinstance(need, str) or not need.strip():
                raise AnnotationParseError(
                    f"{side}_reaches=reaches_unstated requires a non-empty "
                    f"unstated_need_{side} naming what the question reaches for"
                )
            out[f"unstated_need_{side}"] = need.strip()
        elif isinstance(need, str) and need.strip():
            raise AnnotationParseError(
                f"unstated_need_{side}={need!r} but {side}_reaches is "
                f"{out[f'{side}_reaches']!r}: the field must be absent unless that side is "
                "reaches_unstated"
            )
    return out


def _parse_b4(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    verdict = require_label(obj, "verdict", _LABELS["B4"])
    q = str(obj.get("question") or "").strip(_EMPHASIS_B4)
    if verdict == "question_needed" and not q:
        raise AnnotationParseError("question_needed must carry the question it proposes")
    if verdict != "question_needed" and q:
        raise AnnotationParseError(f"a question was proposed but the verdict is {verdict!r}")
    out: dict[str, Any] = {"verdict": verdict}
    if q:
        out["question"] = q
    return out


def _parse_b1b(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    """`default_existed` is REFUSED rather than mapped onto `answer_inferable`. Silently
    folding it in would hide the very behaviour this variant exists to measure."""
    return {"verdict": require_label(obj, "verdict", _LABELS["B1b"])}


def _parse_b1(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    verdict = require_label(obj, "verdict", _LABELS["B1"])
    default = str(obj.get("default_action") or "").strip()
    # THE SAME RULE THE HUMAN PATH ENFORCES, and for the same reason: this verdict lands in
    # the wasted-ask numerator, and "you did not need to ask" with no statement of what the
    # agent should have done instead is unfalsifiable.
    if verdict == "default_existed" and not default:
        raise AnnotationParseError(
            "verdict=default_existed requires default_action naming the default that existed"
        )
    if verdict != "default_existed" and default:
        raise AnnotationParseError(
            f"default_action given but verdict={verdict!r} is not default_existed"
        )
    out: dict[str, Any] = {"verdict": verdict}
    if default:
        out["default_action"] = default
    return out


def _parse_b2(item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    keys = {str(e.get("item_key")) for e in (item.get("payload") or {}).get("items") or ()}
    verdicts = obj.get("verdicts")
    if not isinstance(verdicts, Mapping):
        raise AnnotationParseError(f"verdicts={verdicts!r} must be an object keyed by item_key")
    out: dict[str, str] = {}
    for k, v in verdicts.items():
        if str(k) not in keys:
            raise AnnotationParseError(f"verdict for {k!r}, which is not an item on this task")
        if not isinstance(v, str) or v not in _LABELS["B2"]:
            raise AnnotationParseError(f"{k}={v!r} is not one of {_LABELS['B2']}")
        out[str(k)] = v
    # A PARTIAL ANSWER IS ACCEPTED. An unmentioned item is one the model said nothing about,
    # and `annotate._units` drops it; forcing a label here would manufacture the judgment the
    # miss rate is computed from.
    return {"verdicts": out}


def _parse_b3(_item: Mapping[str, Any], obj: Mapping[str, Any]) -> dict[str, Any]:
    verdict = require_label(obj, "verdict", _LABELS["B3"])
    question = str(obj.get("question") or "").strip()
    # `should_have_asked` IS the supervised target: it becomes an ASK row whose text is this
    # question. Without it there is no training example, only a complaint.
    if verdict == "should_have_asked" and not question:
        raise AnnotationParseError(
            "verdict=should_have_asked requires question naming what should have been asked"
        )
    if verdict != "should_have_asked" and question:
        raise AnnotationParseError(
            f"question given but verdict={verdict!r} is not should_have_asked"
        )
    out: dict[str, Any] = {"verdict": verdict}
    if question:
        out["question"] = question
    return out


_PARSERS = {
    "A1": _parse_a1,
    "A2": _parse_a2,
    "A3_node": _parse_a3_node,
    "A3_edge": _parse_a3_edge,
    "A3_match": _parse_a3_match,
    "A3_missing": _parse_a3_missing,
    "A4": _parse_a4,
    "A5": _parse_a5,
    "A6": _parse_a6,
    "A7": _parse_a7,
    "B1": _parse_b1,
    "B1b": _parse_b1b,
    "B4": _parse_b4,
    "B2": _parse_b2,
    "B3": _parse_b3,
}


@dataclass(frozen=True, slots=True)
class ParsedReply:
    """The full read of a model's JSON reply.

    `response` is the SAME shape a human record carries for this item's task_type -- the object
    `consensus`/`iaa_report` compute agreement over. `rationale`, `basis` and `likelihood` are
    kept OUT of it and returned as separate fields on purpose: see the module docstring on why
    they must land as siblings of `response` on the record, never inside it.
    """

    response: dict[str, Any]
    rationale: str
    basis: str | None = None
    # A1 only -- see the module docstring on why a per-node number, not just `ticked`.
    likelihood: dict[str, float] | None = None


def _require_rationale(obj: Mapping[str, Any]) -> str:
    """Every task type's prompt now asks for this; a model verdict with no stated reason is the
    one hardest to audit later (see the module docstring). Enforced HERE, at parse time, so an
    unreadable or over-length rationale is a counted parse failure -- never a record written
    with a truncated or fabricated one, matching this module's refusal to default anything else
    it cannot read.
    """
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise AnnotationParseError(f"rationale={rationale!r} must be a non-empty string")
    return clamp_rationale(rationale.strip())[0]


def clamp_rationale(rationale: object) -> tuple[object, bool]:
    """Bound the stored rationale. Returns (text, was_truncated).

    THE BELIEF THAT CHANGED, and why. The docstring above used to end: "an unreadable or
    over-length rationale is a counted parse failure -- never a record written with a
    truncated or fabricated one, matching this module's refusal to default anything else it
    cannot read." The refusal to default is right and is unchanged for every LABEL. It was
    over-applied here, because a truncated rationale is not a default: the text WAS read, in
    full, and only the stored copy is shortened. Nothing is invented.

    What it cost. `RATIONALE_MAX_CHARS` states its own purpose -- "explanatory metadata, never
    a label ... every consensus/gate computation must be blind to whether it is present ...
    long free text costs tokens on every future read". So a record was being discarded, tier
    map and all, because a field the gates are required to IGNORE ran a few characters long.
    MEASURED on the A6 pass: 14 items lost for gemini, 35 for sonnet, 60 for gpt-oss, and
    three-rater coverage fell from 400 items to 303 -- a 24% hole in the campaign. `--resume`
    could not heal it: it keys on (item_id, prompt_sha), so the retry re-issued an identical
    prompt, hit the response cache and was rejected again. The loss was deterministic.

    Truncation satisfies the cap's stated purpose exactly -- the stored string is bounded, so
    future reads stay cheap -- and keeps the measurement. The ellipsis is load-bearing: an
    auditor must be able to tell a reason that was CUT from one written short.
    """
    if not isinstance(rationale, str) or len(rationale) <= RATIONALE_MAX_CHARS:
        return rationale, False
    return rationale[: RATIONALE_MAX_CHARS - 3].rstrip() + "...", True


def parse_reply(item: Mapping[str, Any] | str, text: str) -> ParsedReply:
    """Raises `AnnotationParseError` -- never a default -- on anything the model wrote that this
    module cannot read as `ParsedReply`.

    `item` may be a bare task_type string for the task types whose parser reads nothing off
    the item (A7 validates no id against a payload the way A1/A5/A6 must).

    `strict_json_object` and `require_label` (reused from `pi_eval.judges._llm`) raise the
    PARENT `JudgeParseError`, not our subclass, so a bare `not JSON` or `not one of {...}`
    reply is caught here and re-raised as `AnnotationParseError` -- a caller that only ever
    expects to catch this module's own error would otherwise let the commonest failure shape
    (the model did not even emit an object) slip past it uncaught.
    """
    if isinstance(item, str):
        tt, item = item, {"task_type": item}
    else:
        tt = str(item.get("task_type"))
    parser = _PARSERS.get(tt)
    if parser is None:
        raise AnnotationParseError(f"no LLM-pass parser for task_type {tt!r}")
    try:
        obj = strict_json_object(text)
        response = parser(item, obj)
        rationale = _require_rationale(obj)
        basis = None
        if tt == "A2":
            # Required when there was a winner to explain; optional for a tie/both_bad, which
            # names nothing to attribute a win to -- but if the model supplies one anyway, it
            # still has to be a real vocabulary member, not a free pass.
            if response["choice"] in ("a", "b") or "basis" in obj:
                basis = require_label(obj, "basis", A2_BASIS_VALUES)
        elif tt == "A7":
            # Same rule as A2's, against A7's OWN vocabulary -- the one in which
            # `anticipation` exists; see A7_BASIS_VALUES in `pi_eval.annotate`.
            if response["preference"] in ("a", "b") or "basis" in obj:
                basis = require_label(obj, "basis", A7_BASIS_VALUES)
        likelihood = None
        if tt == "A1":
            likelihood = _parse_a1_likelihood(item, obj)
        return ParsedReply(
            response=response, rationale=rationale, basis=basis, likelihood=likelihood
        )
    except AnnotationParseError:
        raise
    except JudgeParseError as exc:
        raise AnnotationParseError(str(exc)) from exc


# --------------------------------------------------------------------------- the record


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reasoning_trace(telemetry: Any) -> str | None:
    """The provider's reasoning STRING for this call, if it attached one -- never its token
    COUNT. `tok_reasoning` on `pinq.types.CallTelemetry` is tracked purely for BILLING (a
    reasoning model bills its reasoning channel inside `max_tokens`; see `REASONING_HEADROOM`
    in `pinq_train.rung0_gepa.search` and `MeteredClient._budget`'s multiplicative headroom for
    the identical reason), and tokens billed is not the same claim as a string returned.
    `pinq_adapters.llm.litellm_client._read_response` reads only `message.content` today, never
    `message.reasoning_content` -- so a call through the real `MeteredClient` always yields None
    here, and that is not a bug in this function, it is what the client currently discards. This
    is checked defensively (`getattr`, never assumed present) purely so a future client whose
    telemetry DOES carry the string is picked up without another wall-shaped rule; it is not
    evidence that one currently does.

    Never synthesised -- an absent trace is an absent key on the record, never `""` -- and never
    treated as evidence for a verdict: it is a MODEL ARTIFACT, what the model reports about its
    own process, read the same way `justification_of` in `pi_eval.judges._llm` is: informative,
    never scored.
    """
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(telemetry, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def annotate_item(
    llm: JudgeLLM,
    item: Mapping[str, Any],
    *,
    bundle_id: str,
    seed: int = 0,
    model_pin: str,
    annotator_id: str,
) -> dict[str, Any]:
    """One full `annotator_kind: "llm"` record for `item`. Raises `AnnotationParseError` (and
    writes nothing) when the reply cannot be read; the caller counts that, it does not paper
    over it -- see the module docstring.
    """
    if not annotator_id.startswith(LLM_ID_PREFIX):
        raise ValueError(
            f"annotator_id {annotator_id!r} must start with {LLM_ID_PREFIX!r}: an id that "
            "does not would be indistinguishable from a person's, and `validate_records` "
            "exists to refuse exactly that."
        )
    if not str(model_pin).strip():
        raise ValueError("model_pin must be non-empty: an unpinned model is unattributable")

    prompt = build_prompt(item)
    t0 = time.perf_counter()
    raw, tel = ask_with_telemetry(
        llm,
        prompt=prompt,
        seed=seed,
        role="annotator",
        system=SYSTEM,
        temperature=ANNOTATOR_TEMPERATURE,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    parsed = parse_reply(item, raw)

    item_id = str(item["item_id"])
    record: dict[str, Any] = {
        "record_id": f"{bundle_id}/{item_id}/{annotator_id}",
        "bundle_id": bundle_id,
        "item_id": item_id,
        "task_type": str(item["task_type"]),
        "annotator_id": annotator_id,
        "annotator_kind": "llm",
        "model_pin": str(model_pin),
        "elapsed_ms": elapsed_ms,
        "ts": _now_iso(),
        "tool_version": TOOL_VERSION,
        "response": parsed.response,
        # Top-level siblings of `response`, never inside it -- see the module docstring.
        "rationale": parsed.rationale,
        # So a prompt revision is a DIFFERENT rater rather than more samples of the same one --
        # see the module docstring and `pi_run.cmd_annotate`'s --resume.
        "prompt_sha": prompt_sha(prompt),
        "response_sha": response_sha(raw),
        # What this one call cost, from `CallTelemetry.usd` (tokens x the pinned price table,
        # so deterministic). Recorded, never compared across arms -- that distinction is the
        # repo's rule for `usd`, and recording is the permitted half. It is here because
        # `--max-usd` otherwise reads `ledger.spent`, and `pinq.budget` states the ledger is
        # per-process and "needs no locking at all": under `--concurrency > 1` that is a
        # lock-free counter feeding the one guard that stops an overspend. Summing per-record
        # costs in the single bookkeeping thread does not have that problem.
        "usd": float(getattr(tel, "usd", 0.0) or 0.0),
    }
    if parsed.basis is not None:
        record["basis"] = parsed.basis
    if parsed.likelihood is not None:
        # Top-level, sibling of `response` -- see the module docstring and `ParsedReply`.
        record["likelihood"] = parsed.likelihood
    reasoning = _reasoning_trace(tel)
    if reasoning is not None:
        record["reasoning"] = reasoning
    return record
