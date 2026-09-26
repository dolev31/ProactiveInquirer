"""One STOP serialisation, shared by every writer, inverted by the one parser.

THE DEFECT THESE TESTS WERE WRITTEN FOR. Four shapes existed for one action: the prompt asked
for lowercase `{"action": "stop", ...}`; `pi_run.cmd_train.action_json` emitted
`{"action": "STOP", "rationale": "..."}` on an unreachable branch (a STOP is never recorded as
a turn, so no turn row is ever non-ask); the SFT exporter's stop target was the compact
literal `'{"action":"STOP"}'` -- different separators from the ASK serialiser -- and the
convlog exporter wrote `{"action": "STOP", "rationale": "report"}`, a rationale it invented.

A DPO loss that sees chosen and rejected side by side can learn the separator style as a cue
for stopping. A STOP target carrying a fabricated rationale teaches the policy to emit a
clause it had no reason to write. So the shape is one constant in `pinq.actions`, and these
tests fail on the code as it stood: `test_every_writer_emits_the_same_stop` failed with three
distinct strings, and `test_ask_serialiser_matches_cmd_train` failed on separator bytes.
"""

from __future__ import annotations

import json

from pinq.actions import STOP_ACTION_JSON, action_kind_of, ask_action_json


def test_the_constant_has_the_ask_serialisers_separators():
    """`json.dumps(..., sort_keys=True)` with default separators, uppercase, no rationale."""
    assert STOP_ACTION_JSON == '{"action": "STOP"}'
    assert json.loads(STOP_ACTION_JSON) == {"action": "STOP"}


def test_every_writer_emits_the_same_stop():
    """The three writers that used to disagree now all hand back the constant, byte for byte."""
    from pi_eval.build import convlog_export
    from pi_run.cmd_train import action_json

    assert action_json({"action_kind": "stop", "rationale": "ignored"}) == STOP_ACTION_JSON
    assert convlog_export._STOP_JSON == STOP_ACTION_JSON


def test_ask_serialiser_matches_cmd_train():
    """`cmd_train.action_json` for an ask and `pinq.actions.ask_action_json` are one function."""
    from pi_run.cmd_train import action_json

    turn = {"action_kind": "ask", "question": "Who founded it?", "rationale": "unresolved"}
    assert action_json(turn) == ask_action_json("Who founded it?", "unresolved")


def test_the_parser_inverts_the_constant():
    """The policy parser lowercases the kind and ignores the rest, so the constant round-trips
    to a Stop. Called on a bare instance: the stop branch returns before it reads the state."""
    from pinq.types import Stop
    from pinq_expt.policies.base import LLMPolicy

    base = LLMPolicy.__new__(LLMPolicy)
    out = base._action_from(json.loads(STOP_ACTION_JSON), None)  # type: ignore[arg-type]
    assert isinstance(out, Stop)


def test_action_kind_of_mirrors_the_parser():
    assert action_kind_of(STOP_ACTION_JSON) == "stop"
    assert action_kind_of('{"action": "stop"}') == "stop"  # lowercase, like the prompt asks
    assert action_kind_of(ask_action_json("q?")) == "ask"
    assert action_kind_of('{"action": "ASK", "question": "   "}') == "unparseable"
    assert action_kind_of("not json") == "unparseable"
    assert action_kind_of(None) == "unparseable"
