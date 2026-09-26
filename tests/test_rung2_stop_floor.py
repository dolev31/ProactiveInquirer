"""A RUN THAT ASKED FOR STOPPING DATA AND GOT NONE MUST NOT TRAIN SILENTLY.

MEASURED on `data/rl/pairs.jsonl`: among `ask_stop` pairs, ASK is the chosen side 1,356 times
and STOP 236. A run that includes `ask_stop` in order to teach stopping, and whose file turns
out to be almost entirely ASK-wins, trains "keep asking" while its config says it trained
stopping -- and `n_pairs` cannot tell the two apart. `n_stop_chosen` was already computed and
reported; like the STOP-share ceiling in rung 1, a number that is printed and acted on by
nothing documents the failure instead of preventing it.

The floor is 0.0 by default, so it refuses nothing until a run declares what it expects. That
declaration lands in `cfg.sha`.
"""

from __future__ import annotations

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.rung2_dpo import DPOConfig, TooFewStopPairs, pairs_report, preflight


def _p(chosen=None, rejected=None):
    return {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": chosen if chosen is not None else ask_action_json("who?"),
        "rejected_json": rejected if rejected is not None else ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask" if chosen is None and rejected is None else "ask_stop",
    }


def _cfg(**over):
    # The DEFAULT reference (rung 1's adapter, frozen). This file is about the STOP-share floor,
    # which is a property of the pairs and not of the reference policy -- but a config still has
    # to name one, and naming the default keeps this file from re-testing the merge path.
    return DPOConfig(base_model="m", adapter="a", **over)


def test_the_report_carries_the_share_not_only_the_count():
    rows = [_p(chosen=STOP_ACTION_JSON), _p(rejected=STOP_ACTION_JSON), _p(), _p()]
    rep = pairs_report(rows)
    assert rep["n_stop_chosen"] == 1
    assert rep["stop_chosen_share"] == pytest.approx(0.25)


def test_a_floor_the_data_cannot_meet_stops_the_run(tmp_path):
    """1 STOP-chosen pair in 4 is 0.25; a run that declared 0.30 is not the run it configured."""
    rows = [_p(chosen=STOP_ACTION_JSON), _p(rejected=STOP_ACTION_JSON), _p(), _p()]
    with pytest.raises(TooFewStopPairs, match="0.25"):
        preflight(
            _cfg(min_stop_chosen_share=0.30, include_pair_kinds=("ask_ask", "ask_stop")), rows
        )


def test_a_floor_the_data_meets_passes():
    rows = [_p(chosen=STOP_ACTION_JSON), _p(rejected=STOP_ACTION_JSON), _p(), _p()]
    rep = preflight(
        _cfg(min_stop_chosen_share=0.25, include_pair_kinds=("ask_ask", "ask_stop")), rows
    )
    assert rep["stop_chosen_share"] == pytest.approx(0.25)


def test_the_default_floor_refuses_nothing():
    """An ask_ask-only run has no STOP-chosen pairs by construction and is the PRIMARY arm."""
    rep = preflight(_cfg(), [_p(), _p()])
    assert rep["stop_chosen_share"] == 0.0


def test_the_floor_is_in_the_config_sha():
    assert _cfg().sha != _cfg(min_stop_chosen_share=0.1).sha
