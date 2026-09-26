"""Rendering a ConvState into the prompt the policy is shown.

The renderer is the only place a conversation state becomes text, so it is the only place a
piece of the answer key could leak into the question. Everything below is either that, or
determinism -- without which the replay evaluation compares two different prompts.
"""

from pinq import promptlib
from pinq_adapters.convlog.render import render_conv_state
from pinq_adapters.convlog.types import ConvState, ToolStep


def state(**over):
    base = dict(
        session_id="s1",
        dp_index=2,
        cwd_norm="~/PycharmProjects/demo",
        git_branch_norm="main",
        after_compaction=False,
        task_statement="Add a retry to the uploader.",
        prior_user_turns=("yes, also add a timeout",),
        prior_assistant_text=("I can add a fixed 3-attempt retry.",),
        tool_trace=(ToolStep("Bash", "cat upload.py", True, 42, "def upload(): ..."),),
    )
    base.update(over)
    return ConvState(**base)


def test_every_state_field_that_carries_content_reaches_the_prompt():
    out = render_conv_state(state())
    assert "Add a retry to the uploader." in out
    assert "yes, also add a timeout" in out
    assert "I can add a fixed 3-attempt retry." in out
    assert "cat upload.py" in out
    assert "~/PycharmProjects/demo" in out
    assert "{{" not in out


def test_the_render_is_deterministic():
    """The replay evaluation grades one policy's action against another's label at the SAME
    state. If the rendering wobbled, they would not be at the same state."""
    assert render_conv_state(state()) == render_conv_state(state())


def test_an_empty_section_says_so_rather_than_rendering_blank():
    """A blank EVIDENCE-shaped hole is an ablation nobody declared: the policy cannot tell
    'nothing has happened yet' from 'this section was dropped'."""
    out = render_conv_state(state(prior_user_turns=(), prior_assistant_text=(), tool_trace=()))
    assert out.count("(none)") >= 3


def test_the_tool_trace_shows_the_call_and_not_the_output():
    """Tool output is where a paste, a credential or the answer itself would ride in. The
    head is bounded and the full result is never rendered."""
    long_head = "Z" * 500
    out = render_conv_state(state(tool_trace=(ToolStep("Bash", "ls", True, 99999, long_head),)))
    assert long_head not in out
    assert "99999" in out or "99,999" in out


def test_the_user_channel_paragraph_is_real_here_and_the_placebo_elsewhere():
    """In a Claude Code session the channel WAS open, so a state rendered without it
    misdescribes the decision the agent faced (D15). The placebo keeps prompt length constant
    for any arm that closes it."""
    assert "THE USER CHANNEL IS OPEN" in render_conv_state(state())
    closed = render_conv_state(state(), user_channel_open=False)
    assert "THE USER CHANNEL IS OPEN" not in closed
    a = promptlib.count_tokens(promptlib.load("fragment_user_channel"))
    b = promptlib.count_tokens(promptlib.load("fragment_user_channel_placebo"))
    assert abs(a - b) <= 8, "the two fragments must stay token-matched"


def test_the_prompt_never_names_the_session_or_the_decision_index():
    """`session_id` and `dp_index` are bookkeeping that identifies the row. Showing them
    would give the policy a position in a conversation it is not supposed to be counting."""
    out = render_conv_state(state())
    assert "s1" not in out
    assert "dp_index" not in out


def test_the_rendered_state_is_bounded_and_says_what_it_dropped():
    """Measured on the real pilot bundle before this existed: median prompt 45,585 characters,
    max 276,143. A session with 437 decision points accumulates its whole history into every
    later one. At that size the model's own context window does the truncating, from the front,
    silently -- which is an ablation nobody declared and nobody can see.

    So the render is capped, keeps the MOST RECENT material, and states how much it dropped.
    The cap is on rendered text, not on the record: `ConvState` still holds everything, and
    nothing here tells the policy how many turns remain.
    """
    from pinq_adapters.convlog.render import MAX_RENDERED_CHARS

    big = state(
        prior_user_turns=tuple(f"user turn {i} " + "x" * 900 for i in range(60)),
        prior_assistant_text=tuple(f"assistant turn {i} " + "y" * 900 for i in range(60)),
        tool_trace=tuple(ToolStep("Bash", f"cmd {i}", True, 10, "h") for i in range(200)),
    )
    out = render_conv_state(big)
    # The cap is on the state; the template itself is a fixed ~2k on top of it.
    assert len(out) < MAX_RENDERED_CHARS + 4000
    assert "elided" in out
    # The most recent material is what survives; the oldest is what goes.
    assert "user turn 59" in out
    assert "user turn 0 " not in out
    assert "cmd 199" in out


def test_a_state_under_the_cap_is_rendered_whole():
    out = render_conv_state(state())
    assert "elided" not in out


def test_the_task_statement_is_capped_from_the_HEAD_not_the_tail():
    """Measured on the real bundle after the history cap landed: `task` was still the dominant
    field at a median of 18,782 characters and a maximum of 39,613, because a first prompt is
    often a long paste and a compaction summary is long by construction.

    It is clipped from the FRONT, unlike the conversation, and the asymmetry is the point: the
    opening of a task statement states the task, while the end of a 40,000-character paste is
    log output. Recency is what matters in a conversation; precedence is what matters in a
    brief.
    """
    from pinq_adapters.convlog.render import MAX_TASK_CHARS, render_task

    long_task = "Rewrite the uploader. " + "TAIL " * 20000
    out = render_conv_state(state(task_statement=long_task))
    assert "Rewrite the uploader." in out
    assert len(render_task(long_task)) < MAX_TASK_CHARS + 200
    assert "elided" in render_task(long_task)
    assert render_task("short brief") == "short brief"
