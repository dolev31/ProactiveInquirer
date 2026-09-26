"""Claude Code session JSONL -> decision points.

A decision point is a place where the agent chose between asking the user, reporting back and
carrying on acting, and where the log records what the user did next. The next user turn is
the ANSWER KEY: it must never appear inside the state the policy is shown.
"""

import pathlib

import pytest

from pi_eval.build.convlog_parse import (
    MIN_HUMAN_PROMPTS,
    PASTE_MAX,
    decision_points,
    parse_session,
)
from pinq_adapters.convlog.render import render_conv_state

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "convlog_session.jsonl"


@pytest.fixture(scope="module")
def dps():
    recs = parse_session(FIXTURE.read_text(), session_id="S1")
    return decision_points(recs, session_id="S1")


def test_noise_records_never_reach_a_decision_point(dps):
    """queue-operation, file-history-snapshot, isMeta and SIDECHAIN traffic are not the
    conversation. A sidechain is a subagent talking to the main agent, so treating its
    question as a question to the USER would invent proactivity that never happened."""
    blob = repr(dps)
    assert "sidechain" not in blob.lower()
    assert "ignore me" not in blob


def test_the_three_decision_point_kinds_are_found(dps):
    assert [d.kind for d in dps] == ["yield", "ask_tool", "interrupt", "yield"]
    assert [d.observed_action for d in dps] == ["ASK_USER", "ASK_USER", "ACTING", "ASK_USER"]


def test_a_trailing_question_is_the_asked_question(dps):
    assert dps[0].asked_question.endswith("Want exponential backoff too?")


def test_the_ask_tool_question_is_parsed_from_its_input(dps):
    assert dps[1].asked_question == "Which timeout should the uploader use?"


def test_the_next_user_turn_is_the_key_and_is_never_inside_the_state(dps):
    """If the state carried the reply, every label would be trivially satisfiable and the
    dataset would measure nothing."""
    assert dps[0].next_user_turn.startswith("yes, also add a timeout")
    for d in dps:
        assert d.next_user_turn not in repr(d.state)


def test_mechanical_reaction_classes(dps):
    assert dps[0].reaction == "extra_item"  # "yes, ALSO add a timeout"
    assert dps[2].reaction == "interrupt"
    assert dps[3].reaction == "short_yes"


def test_long_pastes_are_elided_with_their_length_and_digest(dps):
    task = dps[0].state.task_statement
    assert len(task) < PASTE_MAX
    assert "[paste 5000 chars sha:" in task


def test_a_compaction_summary_opens_a_new_segment(dps):
    assert [d.state.after_compaction for d in dps] == [False, False, False, True]
    # The summary becomes the new segment's task statement; the pre-compaction turns do not
    # come with it, because the policy at that point could not see them either.
    assert "uploader now retries" in dps[3].state.task_statement
    assert dps[3].state.prior_user_turns == ()


def test_tool_trace_records_the_call_not_its_output(dps):
    trace = dps[1].state.tool_trace
    assert [s.tool for s in trace] == ["Bash"]
    assert trace[0].n_chars > 0
    assert len(trace[0].head) <= 200


def test_state_carries_scrubbed_text_only(dps):
    assert "ghp_" not in repr(dps)
    assert "/Users/lc" not in repr(dps)
    assert dps[0].state.cwd_norm.startswith("~/")


def test_probe_sessions_are_refused(tmp_path):
    """Ten of the 41 measured sessions are one-prompt probes of the system prompt. They are
    not conversations and they would dominate any per-session statistic."""
    import json

    thin = "".join(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}, "sessionId": "P"})
        + "\n"
        for _ in range(MIN_HUMAN_PROMPTS - 1)
    )
    assert decision_points(parse_session(thin, session_id="P"), session_id="P") == []


def _session(*records):
    """Three human prompts so the probe guard does not swallow the session, plus whatever
    degenerate records the caller wants to throw at the parser."""
    import json

    filler = [
        {"type": "user", "message": {"role": "user", "content": f"prompt {i}"}, "sessionId": "D"}
        for i in range(3)
    ]
    return "".join(json.dumps(r) + "\n" for r in [*filler, *records])


def test_a_record_whose_content_is_null_does_not_crash_the_session():
    """Real logs carry these: an assistant record aborted by an API error has no content at
    all. A parser that raises here loses every decision point in the session, and the loss is
    silent because the session simply disappears from the corpus."""
    text = _session(
        {"type": "assistant", "message": {"role": "assistant", "content": None}, "sessionId": "D"},
        {"type": "user", "message": {"role": "user", "content": None}, "sessionId": "D"},
        {
            "type": "user",
            "message": {"role": "user", "content": "and now a real turn"},
            "sessionId": "D",
        },
    )
    decision_points(parse_session(text, session_id="D"), session_id="D")


def test_an_ask_tool_call_with_no_questions_does_not_crash():
    """`questions[0]` on an empty list is an IndexError. An interrupted or malformed tool call
    is not a reason to lose the rest of the conversation."""
    text = _session(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t9",
                        "name": "AskUserQuestion",
                        "input": {"questions": []},
                    }
                ],
            },
            "sessionId": "D",
        },
        {"type": "user", "message": {"role": "user", "content": "whatever"}, "sessionId": "D"},
    )
    dps = decision_points(parse_session(text, session_id="D"), session_id="D")
    assert [d.kind for d in dps][-1] == "ask_tool"
    assert dps[-1].asked_question == ""


def test_assistant_content_given_as_a_bare_string_is_read_as_text():
    text = _session(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": "Shall I continue?"},
            "sessionId": "D",
        },
        {"type": "user", "message": {"role": "user", "content": "yes"}, "sessionId": "D"},
    )
    dps = decision_points(parse_session(text, session_id="D"), session_id="D")
    assert dps[-1].observed_action == "ASK_USER"
    assert dps[-1].asked_question == "Shall I continue?"


def test_no_state_contains_its_own_next_turn_when_every_turn_is_distinct():
    """The structural invariant, stated so it cannot be argued about.

    Measured on 41 real sessions, 305 decision points had a `next_user_turn` that also appeared
    somewhere in their own state, which looks exactly like the answer key leaking. It was not:
    220 of them came from ONE monitoring session in which the same long prompt fires on a loop,
    so the state legitimately held an identical EARLIER turn. This test removes the ambiguity by
    making every turn distinct -- then any overlap at all is a real leak.
    """
    import json

    rows = []
    for i in range(6):
        rows.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Assistant statement number {i}. Shall I go on?"}
                    ],
                },
                "sessionId": "U",
            }
        )
        rows.append(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": f"Distinct user turn number {i} with enough words to be unmistakable.",
                },
                "sessionId": "U",
            }
        )
    text = "".join(json.dumps(r) + "\n" for r in rows)
    dps = decision_points(parse_session(text, session_id="U"), session_id="U")
    assert dps
    for d in dps:
        assert d.next_user_turn not in repr(d.state)


def test_a_turn_the_user_already_sent_verbatim_is_flagged():
    """A repeated prompt is not a leak, but it is not an independent draw either: one looping
    session contributed 220 such decision points and 20% of the whole corpus. Whoever exports
    rows has to be able to cap them, and a flag they must actively ignore is safer than a
    property they must remember to recompute."""
    import json

    repeated = "Check every job on the cluster and report anything stalled, with its node."
    rows = []
    for _ in range(3):
        rows.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Nothing stalled. Check again?"}],
                },
                "sessionId": "R",
            }
        )
        rows.append(
            {"type": "user", "message": {"role": "user", "content": repeated}, "sessionId": "R"}
        )
    text = "".join(json.dumps(r) + "\n" for r in rows)
    dps = decision_points(parse_session(text, session_id="R"), session_id="R")
    assert [d.repeats_earlier_turn for d in dps] == [False, True]


def test_the_session_id_is_redacted_out_of_its_own_state():
    """Claude Code writes a scratchpad path containing the session's own uuid, and people paste
    those paths. Measured over the 41 real sessions: 573 of 2,146 rendered states contained the
    session id, in the task statement (501), a tool argument (190) or a prior turn (120).

    The renderer never emits it -- the id arrives inside the CONTENT. But `template_id` IS the
    session id, so leaving it in the prompt puts the split key itself in front of the policy,
    and the test asserting 'the prompt never names the session' would be passing only because
    the synthetic fixture never mentions one.
    """
    import json

    sid = "0a120f12-d3a0-4d93-8c00-a0d739fbb67e"
    rows = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": f"read /private/tmp/claude-501/{sid}/scratchpad/notes.md",
            },
            "sessionId": sid,
        },
        {"type": "user", "message": {"role": "user", "content": "second turn"}, "sessionId": sid},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done. Shall I continue?"}],
            },
            "sessionId": sid,
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "yes please carry on"},
            "sessionId": sid,
        },
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    dps = decision_points(parse_session(text, session_id=sid), session_id=sid)
    assert dps
    for d in dps:
        # `state.session_id` is the row's KEY and must keep holding the id -- it is never
        # rendered. What must not carry it is any field that becomes prompt text.
        assert d.state.session_id == sid
        content = (
            d.state.task_statement,
            d.state.cwd_norm,
            *d.state.prior_user_turns,
            *d.state.prior_assistant_text,
            *[t.arg_summary for t in d.state.tool_trace],
            *[t.head for t in d.state.tool_trace],
            d.next_user_turn,
            d.asked_question,
        )
        assert not any(sid in v for v in content)
        assert sid not in render_conv_state(d.state)
    assert "<session>" in dps[0].state.task_statement


def test_a_report_yield_keeps_the_text_it_reported():
    """B3 asks "should the agent have stopped and reported here?", and the thing it reported is
    the answer's whole subject. The record dropped it: a yield's own final text is excluded from
    `prior_assistant_text` (correctly -- it is the action, not context for the action), and
    `asked_question` is empty unless that text ended in a question mark. So for the 1,308 of
    2,146 real decision points that are REPORT, the reported text was in the record nowhere at
    all, and a consumer reaching for it silently got the PREVIOUS statement instead.
    """
    import json

    rows = [
        {"type": "user", "message": {"role": "user", "content": f"turn {i}"}, "sessionId": "Z"}
        for i in range(3)
    ]
    rows += [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "FIRST statement."}],
            },
            "sessionId": "Z",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "SECOND: I rewrote the config loader and it builds."}
                ],
            },
            "sessionId": "Z",
        },
        {"type": "user", "message": {"role": "user", "content": "hm ok"}, "sessionId": "Z"},
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    d = decision_points(parse_session(text, session_id="Z"), session_id="Z")[-1]
    assert d.observed_action == "REPORT"
    assert d.final_assistant_text.startswith("SECOND:")
    # And it stays out of the prior context, which is what made it the decision rather than
    # background to it.
    assert not any("SECOND" in v for v in d.state.prior_assistant_text)


def test_an_asking_yield_reports_the_same_text_as_both_question_and_final():
    import json

    rows = [
        {"type": "user", "message": {"role": "user", "content": f"turn {i}"}, "sessionId": "Y"}
        for i in range(3)
    ]
    rows += [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Retry is in. Shall I add a test?"}],
            },
            "sessionId": "Y",
        },
        {"type": "user", "message": {"role": "user", "content": "go ahead"}, "sessionId": "Y"},
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    d = decision_points(parse_session(text, session_id="Y"), session_id="Y")[-1]
    assert d.observed_action == "ASK_USER"
    assert d.asked_question == d.final_assistant_text
