"""A run that trained on rater-ordered pairs must not claim the identity of one that did not.

WHAT `label_source` SEPARATES. `pairs.rater.jsonl` carries two populations: pairs the
exporter's own rule ordered ("rule") and pairs it refused that only a rater majority could
order ("rater"). They are different evidence and the paper compares them, so a consumer must
be able to take one, the other or both -- and the choice must land in `cfg.sha`, because a
filter applied by hand before training produces two runs that claim one identity, which is
exactly what provenance exists to prevent (the same argument as `include_pair_kinds`).

LEGACY ROWS ARE "rule", NOT UNKNOWN. Every pair written before this field existed was ordered
by the rule; reading a missing field as anything else would silently drop the entire existing
corpus the first time someone filtered on it.
"""

from __future__ import annotations

import json

import pytest

from pinq.actions import ask_action_json
from pinq_train.rung2_dpo import DPOConfig, load_pairs, pairs_report


def _p(label_source="rule", **over):
    r = {
        "suite_id": "musique",
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": ask_action_json("who?"),
        "rejected_json": ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": "ask_ask",
    }
    if label_source is not None:
        r["label_source"] = label_source
    r.update(over)
    return r


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_a_label_source_can_be_excluded_and_the_drop_is_counted(tmp_path):
    p = _write(tmp_path, [_p("rule"), _p("rater"), _p("rater")])
    rows, drops = load_pairs(p, include_label_sources=("rule",))
    assert len(rows) == 1 and rows[0]["label_source"] == "rule"
    assert drops["label_source"] == 2


def test_the_default_keeps_both(tmp_path):
    p = _write(tmp_path, [_p("rule"), _p("rater")])
    rows, drops = load_pairs(p)
    assert len(rows) == 2 and "label_source" not in drops


def test_a_row_without_the_field_is_a_rule_pair(tmp_path):
    """The whole existing corpus predates the field. Reading it as unknown would delete it."""
    p = _write(tmp_path, [_p(None), _p("rater")])
    rows, _ = load_pairs(p, include_label_sources=("rule",))
    assert len(rows) == 1


def test_the_filter_is_in_the_config_sha():
    base = DPOConfig(base_model="m", adapter="a")
    assert base.sha != DPOConfig(base_model="m", adapter="a", include_label_sources=("rule",)).sha


def test_the_config_refuses_a_filter_that_would_keep_nothing():
    with pytest.raises(ValueError, match="include_label_sources"):
        DPOConfig(base_model="m", adapter="a", include_label_sources=()).validate()
    with pytest.raises(ValueError, match="include_label_sources"):
        DPOConfig(base_model="m", adapter="a", include_label_sources=("human",)).validate()


def test_the_report_carries_the_split(tmp_path):
    rep = pairs_report([_p("rule"), _p("rater"), _p("rater")])
    assert rep["n_by_label_source"] == {"rule": 1, "rater": 2}
