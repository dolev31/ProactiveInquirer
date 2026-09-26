"""Regression test for the exit-code-1 bug in score_checkpoint_paired.py: `scored()` called
`logprob(render(state, action))` -- passing the whole Rendered object as ONE argument -- but
torch_logprob_fn's real _logprob(prompt_ids, completion_ids) takes TWO positional arguments,
exactly as eo._score calls it. This test uses a fake render/logprob pair that mimics that real
two-argument calling convention and would raise TypeError under the old (buggy) scored().
"""

from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_checkpoint_paired import score_rows_paired  # noqa: E402

Rendered = namedtuple("Rendered", ["prompt_ids", "completion_ids"])


def fake_render(state: str, action: str) -> Rendered:
    return Rendered(prompt_ids=(1, 2, 3), completion_ids=tuple(ord(c) for c in action[:2]))


def fake_logprob(prompt_ids, completion_ids) -> float:
    # two-argument signature, exactly like the real torch_logprob_fn's _logprob
    return -float(len(prompt_ids) + sum(completion_ids))


def test_score_rows_paired_calls_logprob_with_two_positional_args():
    rows = [
        {"pair_id": "p1", "state_text": "s1", "chosen_json": "AA", "rejected_json": "ZZZZZ"},
        {"pair_id": "p2", "state_text": "s2", "chosen_json": "ZZZZZ", "rejected_json": "AA"},
    ]
    cache: dict = {}
    result = score_rows_paired(rows, render=fake_render, logprob=fake_logprob, cache=cache)
    assert result["n"] == 2
    assert set(result["per_pair"]) == {"p1", "p2"}
    # "AA" -> completion_ids (65, 65) sum=130+prompt(3)= -133 lp; "ZZZZZ"[:2]->"ZZ" (90,90) -> -183
    # chosen shorter margin should be higher (less negative) for p1 (chosen="AA")
    assert result["per_pair"]["p1"]["ok"] == 1
    assert result["per_pair"]["p2"]["ok"] == 0
