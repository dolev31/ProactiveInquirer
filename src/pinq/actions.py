"""The JSON contract between the Inquirer and the loop: one serialiser, one inverse.

WHY THIS FILE EXISTS. There were FOUR shapes for one action. The prompt asks for
`{"action": "stop", "rationale": ...}` (lowercase); `pi_run.cmd_train.action_json` emitted
`{"action": "STOP", "rationale": "..."}` on a branch nothing could reach (a STOP is never
recorded as a turn, so no turn row is ever non-ask); the SFT exporter's stop target was the
compact literal `'{"action":"STOP"}'` with no spaces after the separators; and the convlog
exporter wrote `{"action": "STOP", "rationale": "report"}` with a rationale it invented.

None of these were coordinated with the ASK serialiser, which uses `json.dumps(...,
sort_keys=True)` and therefore the default separators. A DPO loss that sees the chosen and
rejected completions side by side can learn a *surface* cue -- compact separators versus
spaced ones -- that has nothing to do with stopping. And a STOP target carrying a fabricated
rationale teaches the policy to emit a clause it never had a reason to write.

THE SHAPE. `json.dumps({"action": "STOP"}, sort_keys=True)`: the ASK serialiser's separators,
uppercase because the ASK target is already uppercase, and no rationale because the model's
STOP rationale is never recorded anywhere (`_action_from` returns `Stop(reason="policy_stop")`
and drops the text). The parser lowercases the kind and ignores everything else, so this
round-trips to a `Stop`.

STDLIB ONLY. `pinq` is the detachable core, and this is the one place a constant shared by
`pi_run`, `pi_eval` and `pinq_train` may live -- the same argument `pinq.wire` makes for the
HTTP seam. Every consumer imports the constant; none retypes it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

ActionKind = Literal["ask", "stop", "unparseable"]

# The one STOP. Byte-for-byte what a training row, a preference pair side and the convlog
# shard write, so that "did this row stop" is a string equality and not a parse.
STOP_ACTION_JSON: str = json.dumps({"action": "STOP"}, sort_keys=True)


def ask_action_json(question: str, rationale: str = "") -> str:
    """The ASK serialiser. Key order is canonical (`sort_keys`) so two policies emitting the
    same question produce the same bytes, which the exporter's identical-action guard needs."""
    return json.dumps(
        {"action": "ASK", "question": str(question), "rationale": str(rationale)},
        sort_keys=True,
    )


def action_kind_of(action_json: Any) -> ActionKind:
    """The inverse: what kind of action a serialised string is.

    Normalises exactly as the policy parser does (`pinq_expt.policies.base._action_from`:
    `str(obj.get("action", "")).strip().lower()`), so the two can never disagree about a
    row. An ASK with an empty question is `unparseable`, again mirroring the parser, which
    returns None for it rather than an Ask.
    """
    try:
        obj = json.loads(action_json) if isinstance(action_json, str) else action_json
    except (TypeError, ValueError):
        return "unparseable"
    if not isinstance(obj, dict):
        return "unparseable"
    kind = str(obj.get("action", "")).strip().lower()
    if kind == "stop":
        return "stop"
    if kind == "ask" and str(obj.get("question", "")).strip():
        return "ask"
    return "unparseable"
