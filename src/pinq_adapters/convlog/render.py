"""Render a `ConvState` into the `inquirer_convlog` prompt the policy is shown.

This is the only place a session's recorded state becomes text handed to a model, so it is
the only place a piece of the answer key -- or a piece of bookkeeping the policy is not
supposed to see -- could leak into the question. Every choice below follows from that.

WHY `head` IS NEVER RENDERED
`ToolStep.head` is un-vetted tool output: the start of a file dump, a paste, a credential, or
(in the replay setting this adapter feeds) the very thing the label is graded against. The
policy is shown that a tool ran, what it was asked to do, whether it succeeded and how big the
result was -- `n_chars` -- which is enough to condition a decision on, without smuggling the
result itself into the prompt. `ConvState.tool_trace` is a *trace*, not a transcript, and the
renderer preserves that distinction on the way out.

WHY AN EMPTY SECTION SAYS "(none)" INSTEAD OF RENDERING BLANK
A blank section is indistinguishable from a section that was silently dropped -- and a dropped
section is an undeclared ablation, the exact failure mode `promptlib.render`'s unresolved-
placeholder check exists to catch one level up. The three sections that can be empty depending
on the state -- the person's prior turns, the assistant's prior turns, and the tool trace --
each report their own emptiness explicitly.

WHY prior_turns GROUPS BY SPEAKER INSTEAD OF ZIPPING TWO TUPLES TOGETHER
`ConvState` gives `prior_user_turns` and `prior_assistant_text` as two independently-lengthed
tuples with no shared index tying one to the other -- there is no field that says "this
assistant turn answered that user turn". Zipping them positionally would invent a pairing the
data does not contain. Instead each list is rendered oldest-first under its own speaker label,
which is the one ordering the state actually guarantees, and keeps the two lists' emptiness
independently visible (an arm that dropped only the assistant's prior replies, say, is not
disguised as an arm that also dropped the person's).

WHY `session_id` AND `dp_index` NEVER APPEAR
Both identify *which row this is*, not what happened in it. Showing them would hand the policy
a position in a conversation it is not meant to be counting -- the same reasoning that keeps
`ConvState` itself budget-blind (see `pinq_adapters/convlog/types.py`).

DETERMINISM
No dict iteration, no timestamp, no hash of anything unordered: every section is built from
the state's own tuples in their given order, so the same `ConvState` always renders to the
same bytes -- required for the replay evaluation, which grades two policies against the same
rendered prompt.
"""

from __future__ import annotations

from pinq import promptlib
from pinq_adapters.convlog.types import ConvState, ToolStep

_NONE = "(none)"

# THE RENDER IS CAPPED; THE RECORD IS NOT. Measured on the first real pilot bundle: median
# prompt 45,585 characters, max 276,143, because a session with 437 decision points folds its
# entire history into every later one. At that size the model's context window does the
# truncating -- from the FRONT, silently -- and an ablation nobody declared is exactly what the
# rest of this package exists to prevent. So the renderer drops the OLDEST material itself and
# says how much it dropped, which is also what the real session did when it compacted.
#
# THIS IS NOT A BUDGET SIGNAL. It says how much history was elided, never how much of anything
# remains to spend; `ConvState` still carries every turn, and `Inquirer.act` still sees no cap.
# A cap on the WHOLE rendered state, split between the sections below rather than handed to
# each of them: a constant named for the total that each section then spends in full is a
# constant whose name is wrong by a factor of the number of sections.
MAX_RENDERED_CHARS = 9000
_TURNS_SHARE, _TRACE_SHARE = 2 / 3, 1 / 3
MAX_TURN_CHARS = 1200
# The task statement gets its own cap and is clipped from the HEAD, not the tail. A first
# prompt is often a long paste and a compaction summary is long by construction -- measured on
# the real bundle it was the dominant field at a 18,782-character median. The asymmetry with
# the conversation cap is deliberate: recency is what matters in a dialogue, precedence is what
# matters in a brief, and the end of a 40,000-character paste is log output.
MAX_TASK_CHARS = 3000


def render_task(text: str) -> str:
    if len(text) <= MAX_TASK_CHARS:
        return text
    return text[:MAX_TASK_CHARS] + f"\n[{len(text) - MAX_TASK_CHARS} further characters elided]"


def _clip(text: str) -> str:
    if len(text) <= MAX_TURN_CHARS:
        return text
    return text[:MAX_TURN_CHARS] + f" [+{len(text) - MAX_TURN_CHARS} chars]"


def _tail_within(lines: list[str], budget: int, unit: str) -> str:
    """The most recent lines that fit, oldest-dropped, with a line naming what went.

    Recency rather than a sample: the decision being judged was made at the END of this
    conversation, and the turns next to it are the ones that bear on it.
    """
    if not lines:
        return _NONE
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) > budget and kept:
            break
        kept.append(line)
        used += len(line)
    kept.reverse()
    dropped = len(lines) - len(kept)
    if dropped:
        kept.insert(0, f"[{dropped} earlier {unit} elided]")
    return "\n".join(kept)


def render_prior_turns(state: ConvState) -> str:
    """Both speakers, oldest first, grouped rather than zipped -- see module docstring."""
    half = int(MAX_RENDERED_CHARS * _TURNS_SHARE) // 2
    person = _tail_within([f"PERSON: {_clip(t)}" for t in state.prior_user_turns], half, "turns")
    assistant = _tail_within(
        [f"ASSISTANT: {_clip(t)}" for t in state.prior_assistant_text], half, "turns"
    )
    return f"PERSON'S PRIOR TURNS:\n{person}\n\nASSISTANT'S PRIOR TURNS:\n{assistant}"


def _render_tool_step(step: ToolStep) -> str:
    """The call and its shape -- never `step.head`, which is un-vetted tool output."""
    status = "ok" if step.ok else "FAILED"
    return f"{step.tool}({step.arg_summary}) -> {status}, {step.n_chars} chars"


def render_tool_trace(state: ConvState) -> str:
    return _tail_within(
        [_render_tool_step(step) for step in state.tool_trace],
        int(MAX_RENDERED_CHARS * _TRACE_SHARE),
        "tool calls",
    )


def _render_workspace(state: ConvState) -> str:
    """`cwd_norm`, plus the branch when there is one (`git_branch_norm` is `""` when there
    isn't -- `ConvState` has no `Optional` field for it)."""
    if state.git_branch_norm:
        return f"{state.cwd_norm} (branch: {state.git_branch_norm})"
    return state.cwd_norm


def render_conv_state(state: ConvState, *, user_channel_open: bool = True) -> str:
    """The `inquirer_convlog` prompt for one decision point of one Claude Code session.

    `user_channel_open` selects the real user-channel paragraph or its token-matched placebo
    (see `pinq/promptlib.py`'s `PARITY_FAMILIES["user_channel"]`) -- it does not read anything
    off `state`, because whether the channel is open is a property of the arm being run, not
    of the session being replayed.
    """
    fragment = "fragment_user_channel" if user_channel_open else "fragment_user_channel_placebo"
    return promptlib.render(
        "inquirer_convlog",
        task=render_task(state.task_statement),
        prior_turns=render_prior_turns(state),
        tool_trace=render_tool_trace(state),
        workspace=_render_workspace(state),
        user_channel=promptlib.load(fragment),
    )
