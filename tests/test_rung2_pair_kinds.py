"""The loader must let an ask_stop pair in, and must say what it kept out.

THE BLOCKER THESE TESTS WERE WRITTEN FOR. `assert_length_guard` RAISES on any pair whose two
questions differ by more than `len_delta_max` characters. An ask_stop pair is a ~40-120 char
question against an 18-byte STOP constant, so every one of them violates it, and rung 2 would
refuse to start the moment the exporter began emitting the category it exists to train on.
The guard is right for ask_ask and meaningless for ask_stop: the delta there is what the two
ACTIONS are, not a property of the questions.

WHY THE KIND IS RE-DERIVED AND NOT TRUSTED. `pair_kind` is a string on a jsonl file that can
be filtered, concatenated or hand-edited between the export and the run. If a mislabelled
ask_ask pair claimed to be ask_stop it would skip the length guard -- the loader's re-check
would be disabled by the very field it is supposed to be checking. So the kind is recomputed
from the two action payloads with `pinq.actions.action_kind_of` (the policy parser's own
normalisation) and a label that disagrees is fatal.

WHY THE FILTERS LIVE ON DPOConfig. `cfg.sha` covers every field, so two runs that trained on
different subsets of one pairs file have different config shas. A filter applied by hand
before training -- `grep -v paraphrase pairs.jsonl` -- produces two runs that claim the same
identity, which is the thing provenance exists to prevent.

AND WHY THE DROPS ARE RETURNED. A filter that removes rows without saying how many is a
filter nobody can audit, so `load_pairs` hands back the counts and they ride into
`rung2.manifest.json` through `pairs_report`.
"""

from __future__ import annotations

import json

import pytest

from pinq.actions import STOP_ACTION_JSON, ask_action_json
from pinq_train.rung2_dpo import DPOConfig, load_pairs, pair_kind_of

LONG = "who founded the company that first manufactured the device described in the passage?"


def _p(kind="ask_ask", *, suite="musique", chosen=None, rejected=None, distinct=None, **over):
    r = {
        "suite_id": suite,
        "task_id": "t1",
        "run_id": "p",
        "turn_idx": 1,
        "state_text": "S",
        "chosen_json": chosen if chosen is not None else ask_action_json("who?"),
        "rejected_json": rejected if rejected is not None else ask_action_json("when?"),
        "margin": 0.5,
        "pair_kind": kind,
    }
    if distinct is not None:
        r["q_distinctness"] = distinct
    r.update(over)
    return r


def _write(tmp_path, rows):
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_the_kind_is_recomputed_from_the_action_payloads():
    assert pair_kind_of(_p()) == "ask_ask"
    assert pair_kind_of(_p("ask_stop", chosen=STOP_ACTION_JSON)) == "ask_stop"
    assert pair_kind_of(_p("ask_stop", rejected=STOP_ACTION_JSON)) == "ask_stop"


def test_a_label_that_disagrees_with_the_payloads_is_fatal():
    """An ask_ask pair wearing an ask_stop label would skip the length guard -- the re-check
    disabled by the field it is checking."""
    with pytest.raises(ValueError, match="pair_kind"):
        pair_kind_of(_p("ask_stop"))  # two ASKs claiming to be ask_stop
    with pytest.raises(ValueError, match="pair_kind"):
        pair_kind_of(_p("ask_ask", chosen=STOP_ACTION_JSON))


def test_an_ask_stop_pair_loads_despite_the_length_delta(tmp_path):
    p = _write(tmp_path, [_p("ask_stop", chosen=STOP_ACTION_JSON, rejected=ask_action_json(LONG))])
    rows, drops = load_pairs(p, len_delta_max=40)
    assert len(rows) == 1 and drops == {}


def test_an_ask_ask_pair_over_the_cap_still_raises(tmp_path):
    """The regression the exemption above could cause. This is the confound the guard exists
    for and it is still fatal."""
    p = _write(tmp_path, [_p(chosen=ask_action_json(LONG), rejected=ask_action_json("who?"))])
    with pytest.raises(ValueError, match="longer question wins"):
        load_pairs(p, len_delta_max=40)


def test_a_kind_can_be_excluded_and_the_drop_is_counted(tmp_path):
    p = _write(
        tmp_path,
        [_p(), _p("ask_stop", chosen=STOP_ACTION_JSON), _p("ask_stop", rejected=STOP_ACTION_JSON)],
    )
    rows, drops = load_pairs(p, include_pair_kinds=("ask_ask",))
    assert len(rows) == 1 and rows[0]["pair_kind"] == "ask_ask"
    assert drops["pair_kind"] == 2


def test_a_suite_can_be_excluded_and_the_drop_is_counted(tmp_path):
    p = _write(tmp_path, [_p(), _p(suite="strategyqa"), _p(suite="strategyqa")])
    rows, drops = load_pairs(p, exclude_suites=("strategyqa",))
    assert len(rows) == 1 and drops["suite"] == 2


def test_paraphrase_pairs_can_be_filtered_out(tmp_path):
    """The sidecar's ordering is paraphrase < related < different. `min_q_distinctness` names
    the FLOOR, so "related" keeps related and different."""
    p = _write(
        tmp_path,
        [
            _p(distinct="paraphrase"),
            _p(distinct="related"),
            _p(distinct="different"),
        ],
    )
    rows, drops = load_pairs(p, min_q_distinctness="related")
    assert [r["q_distinctness"] for r in rows] == ["related", "different"]
    assert drops["q_distinctness"] == 1


def test_asking_for_distinctness_the_rows_do_not_carry_is_fatal(tmp_path):
    """A filter that silently matches nothing is worse than an error: the run would claim to
    have excluded paraphrases and have excluded none, under a config sha that says it did."""
    p = _write(tmp_path, [_p(), _p()])
    with pytest.raises(ValueError, match="q_distinctness"):
        load_pairs(p, min_q_distinctness="related")


def test_the_filters_are_in_the_config_sha():
    """Two runs over different subsets of one file must not claim one identity."""
    base = DPOConfig(base_model="m", adapter="a")
    assert base.sha != DPOConfig(base_model="m", adapter="a", exclude_suites=("wiki2",)).sha
    assert base.sha != DPOConfig(base_model="m", adapter="a", include_pair_kinds=("ask_ask",)).sha
    assert base.sha != DPOConfig(base_model="m", adapter="a", min_q_distinctness="related").sha


def test_the_config_refuses_a_filter_that_would_keep_nothing():
    with pytest.raises(ValueError, match="include_pair_kinds"):
        DPOConfig(base_model="m", adapter="a", include_pair_kinds=()).validate()
    with pytest.raises(ValueError, match="include_pair_kinds"):
        DPOConfig(base_model="m", adapter="a", include_pair_kinds=("ask_maybe",)).validate()
    with pytest.raises(ValueError, match="min_q_distinctness"):
        DPOConfig(base_model="m", adapter="a", min_q_distinctness="somewhat").validate()


def test_the_report_carries_the_kind_split_and_the_stop_direction(tmp_path):
    from pinq_train.rung2_dpo import pairs_report

    rows = [
        _p(),
        _p("ask_stop", chosen=STOP_ACTION_JSON),
        _p("ask_stop", rejected=STOP_ACTION_JSON),
    ]
    rep = pairs_report(rows)
    assert rep["n_by_pair_kind"] == {"ask_ask": 1, "ask_stop": 2}
    assert rep["n_stop_chosen"] == 1
