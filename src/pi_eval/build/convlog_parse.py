"""Claude Code session JSONL -> decision points. THE RAW-INPUT SIDE of the convlog adapter.

A decision point is a place in a real conversation where the agent had already done some
work and had to choose between asking the user, reporting back, or carrying on acting, and
where the log records what the user did next. `next_user_turn` is the ANSWER KEY: everything
downstream (phi_LOO, the stop test, the reaction label) is defined in terms of it, which is
exactly why it must never leak into `ConvState` — a state that already contains the answer
would make every label trivially satisfiable and the dataset would measure nothing. That
invariant is checked by scrub-then-emit here, not trusted to callers: `ConvState` and
`DecisionPoint.next_user_turn` are built from independent slices of the transcript, and the
slice that becomes `next_user_turn` is never copied into the slice that becomes state.

WHY THIS LIVES IN pi_eval.build AND NOT pinq_adapters. `pi_eval.build.*` is where every other
suite's RAW upstream data is turned into a public corpus; convlog's "raw upstream data" is an
engineer's own session transcript, which is at least as sensitive as anything else this
package scrubs before it is allowed to reach `data/corpora/`. Importing
`pinq_adapters.convlog.types` from here is the SAME direction `strategyqa_build.py` imports
`pinq_adapters.paragraphs`'s constants — the corpus-hash convention lives on the adapter side,
the thing that mints records lives on the gold-build side, and pinq_adapters may never import
back.

SEGMENTS, MECHANICALLY. A `isCompactSummary` record ends one segment and starts the next: the
new segment's `task_statement` is the summary text and its `prior_user_turns` /
`prior_assistant_text` start EMPTY, because the policy standing at that point in the real
conversation could not see anything before the compaction either. `tool_trace` resets at the
same boundary for the same reason — a tool call made before compaction is not evidence the
post-compaction policy could have observed.

WHY `tool_trace` IS CUMULATIVE ACROSS A SEGMENT (not reset at every decision point) while
`prior_user_turns`/`prior_assistant_text` also accumulate, but the LAST assistant text is
excluded from its own yield decision's `prior_assistant_text`: a decision point's own text is
what MADE it a yield (it is already exposed as `asked_question`/`observed_action`), so
folding it into "prior" context as well would just be restating the same field under a
different name. An `ask_tool` or `interrupt` decision has no such self-text — the thing that
made it a decision was a tool call or the NEXT turn, not the last thing the assistant said —
so nothing is excluded there.

MECHANICAL RECIPE, ON PURPOSE. `reaction` is regex-only, precisely so that phi_LOO's outcome
side never depends on the same kind of model judgement its DRAFTER side uses — a reaction
label that itself came from an LLM would make "did asking help" partly a test of judge
agreement rather than of the policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from pi_eval.build.scrub import scrub_strict
from pinq_adapters.convlog.types import ConvState, ToolStep

MIN_HUMAN_PROMPTS = 3  # below this a session is a probe of the system prompt, not a conversation
PASTE_MAX = 4000
_PASTE_HEAD = 200

_KEPT_TYPES = frozenset({"user", "assistant"})

# Order matters, same discipline as pi_eval.build.scrub._RULES: the first pattern that matches
# wins, so a reply that is BOTH an interruption and a correction (say) is filed under the more
# specific, more mechanically certain signal first.
_CORRECTION_RE = re.compile(r"\b(no[,.]|not what|wrong|instead|i meant|revert|undo)\b", re.I)
_EXTRA_ITEM_RE = re.compile(r"\b(also|and then|in addition|plus)\b", re.I)
_SHORT_YES_PREFIXES = (
    "yes",
    "ok",
    "okay",
    "sure",
    "go ahead",
    "do it",
    "proceed",
    "both",
    "commit",
    "approved",
)
_INTERRUPT_PREFIX = "[Request interrupted by user]"
_TRAILING_MARKDOWN = " \t\n*_`"


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """One place the agent chose, plus what the user did next.

    `next_user_turn` is the label; nothing in `state` is derived from it. See the module
    docstring for why that separation is load-bearing rather than incidental.
    """

    state: ConvState
    kind: str  # "ask_tool" | "interrupt" | "yield"
    observed_action: str  # "ASK_USER" | "ACTING" | "REPORT"
    asked_question: str
    # WHAT THE AGENT LAST SAID, whether or not it was a question. `asked_question` is empty on a
    # REPORT yield and on an interrupt, and the yield's own final text is deliberately kept out
    # of `prior_assistant_text` -- so without this field the thing B3 is asking about was in the
    # record nowhere at all, for 1,308 of the 2,146 real decision points.
    final_assistant_text: str
    next_user_turn: str
    reaction: str
    # THE USER HAS SENT THIS EXACT TEXT BEFORE IN THIS SESSION. Not a leak -- the state holds
    # an earlier identical turn because the user really did send one -- but not an independent
    # draw either. Measured over 41 real sessions: 305 decision points overlap their own state
    # this way and 220 of them come from ONE monitoring session whose prompt fires on a loop,
    # which alone is 20% of every decision point mined. Carried as a field so an exporter has
    # to actively ignore it rather than remember to recompute it.
    repeats_earlier_turn: bool = False


# --------------------------------------------------------------------------- parse_session


def parse_session(text: str, *, session_id: str) -> list[dict]:
    """JSONL -> the kept records, in log order. Everything else is dropped here so that
    `decision_points` never has to re-derive "is this the conversation" from scratch.

    Tolerates malformed lines (a truncated write, a partial flush) by skipping them: one bad
    line must not sink an entire session's worth of decision points. Drops `isSidechain`
    traffic (a subagent talking to the main agent — treating its question as a question to the
    USER would invent proactivity that never happened) and `isMeta` records (harness-injected
    reminders, not anything a human typed).
    """
    kept: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("type") not in _KEPT_TYPES:
            continue
        if rec.get("isSidechain") or rec.get("isMeta"):
            continue
        kept.append(rec)
    return kept


# --------------------------------------------------------------------------- text handling


def _normalize(text: str, session_id: str = "") -> str:
    """Scrub, redact the session's own id, then elide any paste.

    FAIL-CLOSED scrubbing happens first, over the whole string, so a secret spanning what will
    become the elided remainder is still caught before truncation could hide it; elision then
    runs on the already-clean text.

    THE SESSION ID IS REDACTED FROM ITS OWN STATE. Claude Code writes a scratchpad path
    containing the session uuid and people paste those paths, so the id arrives inside the
    CONTENT -- 573 of 2,146 real states carried it. `template_id` IS the session id, so leaving
    it in the prompt hands the policy the split key. `scrub` cannot do this: it is a generic
    table and this value is different for every session.
    """
    text = _elide_paste(scrub_strict(text))
    return text.replace(session_id, "<session>") if session_id else text


def _elide_paste(text: str) -> str:
    """A "single text" is a line: pasted content lands on its own line, and eliding per-line
    (rather than the whole message) keeps the surrounding prose — which is often the entire
    point of the turn — intact instead of discarding it along with the paste."""
    return "\n".join(_elide_line(line) for line in text.split("\n"))


def _elide_line(line: str) -> str:
    if len(line) <= PASTE_MAX:
        return line
    digest = hashlib.sha256(line.encode()).hexdigest()[:8]
    return f"{line[:_PASTE_HEAD]}[paste {len(line)} chars sha:{digest}]"


def _is_human_prompt(content: Any) -> bool:
    """A human prompt is free text, not a tool answer. A `tool_result`-bearing user record is
    the harness handing a tool's output back, never something a human typed."""
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _strip_trailing_markdown(text: str) -> str:
    return text.rstrip(_TRAILING_MARKDOWN)


def _classify_reaction(text: str) -> str:
    if text.startswith(_INTERRUPT_PREFIX):
        return "interrupt"
    if _CORRECTION_RE.search(text):
        return "correction"
    if _EXTRA_ITEM_RE.search(text):
        return "extra_item"
    if len(text) <= 40 and text.lower().startswith(_SHORT_YES_PREFIXES):
        return "short_yes"
    if "?" in text:
        return "own_question"
    return "other"


def _tool_arg_summary(name: str, tool_input: dict, session_id: str = "") -> str:
    """A short one-line rendering of a tool call. The handful of keys checked cover the tools
    that actually generate decision-relevant trace entries (Bash, Edit/Write/Read); anything
    else falls back to a compact rendering rather than growing this list without end."""
    for key in ("command", "file_path", "path", "pattern", "query", "url"):
        if key in tool_input:
            return _normalize(str(tool_input[key]), session_id)[:_PASTE_HEAD]
    if tool_input:
        return _normalize(json.dumps(tool_input, sort_keys=True), session_id)[:_PASTE_HEAD]
    return name


def _tool_result_is_error(block: dict) -> bool:
    return bool(block.get("is_error") or block.get("isError"))


def _count_human_prompts(records: list[dict]) -> int:
    n = 0
    for rec in records:
        if rec.get("type") != "user" or rec.get("isCompactSummary"):
            continue
        if _is_human_prompt(rec.get("message", {}).get("content")):
            n += 1
    return n


# --------------------------------------------------------------------------- decision_points


def decision_points(records: list[dict], *, session_id: str) -> list[DecisionPoint]:
    """The kept records of one session -> its decision points, in log order.

    Ten of the 41 measured sessions are one-prompt probes of the system prompt rather than
    conversations; below `MIN_HUMAN_PROMPTS` this returns `[]` so they cannot dominate a
    per-session statistic.
    """
    if _count_human_prompts(records) < MIN_HUMAN_PROMPTS:
        return []

    dps: list[DecisionPoint] = []
    dp_index = 0

    # Per-segment accumulators. Reset together at every compaction boundary.
    task_statement = ""
    after_compaction = False
    seg_user_turns: list[str] = []
    seg_assistant_texts: list[str] = []
    tool_trace: list[ToolStep] = []
    pending_tool_uses: dict[str, tuple[str, str]] = {}  # tool_use id -> (tool, arg_summary)
    pending_ask: dict[str, Any] | None = None  # an ask_tool decision awaiting its answer

    cwd_raw = ""
    git_branch_raw = ""
    seen_first_prompt = False

    def current_state() -> ConvState:
        return ConvState(
            session_id=session_id,
            dp_index=dp_index,
            cwd_norm=_normalize(cwd_raw, session_id) if cwd_raw else "",
            git_branch_norm=_normalize(git_branch_raw, session_id) if git_branch_raw else "",
            after_compaction=after_compaction,
            task_statement=task_statement,
            prior_user_turns=tuple(seg_user_turns),
            prior_assistant_text=tuple(seg_assistant_texts),
            tool_trace=tuple(tool_trace),
        )

    for rec in records:
        if rec.get("cwd"):
            cwd_raw = rec["cwd"]
        if rec.get("gitBranch"):
            git_branch_raw = rec["gitBranch"]

        if rec["type"] == "user":
            content = rec.get("message", {}).get("content")

            if rec.get("isCompactSummary"):
                task_statement = _normalize(_text_of(content), session_id)
                after_compaction = True
                seg_user_turns = []
                seg_assistant_texts = []
                tool_trace = []
                pending_tool_uses = {}
                # A pending AskUserQuestion cannot survive a compaction: nothing in the log
                # here supplies an answer for it, and the segment it belongs to just ended.
                pending_ask = None
                continue

            if not _is_human_prompt(content):
                # A tool_result. Finalize whichever pending thing it answers.
                # `_is_human_prompt` also returns False for content that is neither str nor
                # list -- an aborted record carries `content: null` -- and iterating that
                # raises. A record with nothing in it answers nothing; skip it.
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                        continue
                    tuid = block.get("tool_use_id", "")
                    result_text = _normalize(str(block.get("content") or ""), session_id)
                    ok = not _tool_result_is_error(block)

                    if pending_ask is not None and pending_ask["tool_use_id"] == tuid:
                        dps.append(
                            DecisionPoint(
                                state=pending_ask["state"],
                                kind="ask_tool",
                                observed_action="ASK_USER",
                                asked_question=pending_ask["asked_question"],
                                final_assistant_text=pending_ask["final_assistant_text"],
                                next_user_turn=result_text,
                                reaction=_classify_reaction(result_text),
                                repeats_earlier_turn=result_text in seg_user_turns,
                            )
                        )
                        pending_ask = None
                    elif tuid in pending_tool_uses:
                        tool, arg_summary = pending_tool_uses.pop(tuid)
                        tool_trace.append(
                            ToolStep(
                                tool=tool,
                                arg_summary=arg_summary,
                                ok=ok,
                                n_chars=len(result_text),
                                head=result_text[:_PASTE_HEAD],
                            )
                        )
                continue

            # A genuine human prompt.
            text = _normalize(_text_of(content), session_id)

            if not seen_first_prompt:
                task_statement = text
                seen_first_prompt = True
                continue

            if pending_ask is not None:
                # No tool_result answered it yet; this real human turn does.
                dps.append(
                    DecisionPoint(
                        state=pending_ask["state"],
                        kind="ask_tool",
                        observed_action="ASK_USER",
                        asked_question=pending_ask["asked_question"],
                        final_assistant_text=pending_ask["final_assistant_text"],
                        next_user_turn=text,
                        reaction=_classify_reaction(text),
                        repeats_earlier_turn=text in seg_user_turns,
                    )
                )
                pending_ask = None
            else:
                last = seg_assistant_texts[-1] if seg_assistant_texts else ""
                if text.startswith(_INTERRUPT_PREFIX):
                    kind, observed_action, asked_question = "interrupt", "ACTING", ""
                else:
                    if _strip_trailing_markdown(last).endswith("?"):
                        kind, observed_action, asked_question = "yield", "ASK_USER", last
                    else:
                        kind, observed_action, asked_question = "yield", "REPORT", ""

                # The yield decision's own last text is excluded from ITS OWN prior context —
                # it is already exposed as asked_question/observed_action, not "prior" to
                # itself. ask_tool/interrupt decisions have no such self-text to exclude.
                prior_texts = seg_assistant_texts[:-1] if kind == "yield" else seg_assistant_texts
                state = ConvState(
                    session_id=session_id,
                    dp_index=dp_index,
                    cwd_norm=_normalize(cwd_raw, session_id) if cwd_raw else "",
                    git_branch_norm=_normalize(git_branch_raw, session_id)
                    if git_branch_raw
                    else "",
                    after_compaction=after_compaction,
                    task_statement=task_statement,
                    prior_user_turns=tuple(seg_user_turns),
                    prior_assistant_text=tuple(prior_texts),
                    tool_trace=tuple(tool_trace),
                )
                dps.append(
                    DecisionPoint(
                        state=state,
                        kind=kind,
                        observed_action=observed_action,
                        asked_question=asked_question,
                        final_assistant_text=last,
                        next_user_turn=text,
                        reaction=_classify_reaction(text),
                        repeats_earlier_turn=text in seg_user_turns,
                    )
                )
                dp_index += 1

            seg_user_turns.append(text)

        elif rec["type"] == "assistant":
            content = rec.get("message", {}).get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    continue  # the user never saw this; it may never enter state
                if btype == "text":
                    # `.get("text", "")` returns None when the key exists and is null,
                    # which the default never covers. Same for every other block field
                    # read below.
                    seg_assistant_texts.append(_normalize(block.get("text") or "", session_id))
                elif btype == "tool_use":
                    name = block.get("name", "")
                    tool_input = block.get("input", {}) or {}
                    if name == "AskUserQuestion":
                        # An interrupted or malformed call can carry no questions at all.
                        # Losing the rest of the conversation over it would be a far
                        # larger error than recording the ask with an empty question.
                        questions = tool_input.get("questions") or [{}]
                        first = questions[0] if isinstance(questions[0], dict) else {}
                        question = first.get("question") or ""
                        pending_ask = {
                            "state": current_state(),
                            "asked_question": _normalize(question, session_id),
                            "tool_use_id": block.get("id", ""),
                            "final_assistant_text": (
                                seg_assistant_texts[-1] if seg_assistant_texts else ""
                            ),
                        }
                        dp_index += 1
                    else:
                        pending_tool_uses[block.get("id", "")] = (
                            name,
                            _tool_arg_summary(name, tool_input, session_id),
                        )

    return dps
