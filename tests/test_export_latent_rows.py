"""What an exported decision point must carry for a latent-need dataset.

Three additions, and one deliberate NON-addition.

THE LABELS. `is_latent` / `latent_depth` / `newly_reachable` / `frontier_size` come from
`pi_eval.metrics.latent` and must be attached GOLD-SIDE, in `rows_from_run`, because
`pinq_train` may never import `pi_eval` (import-linter contract 1). They cross to the trainer
as plain numbers on the row dict.

THE SELF-REPORT, AS A FEATURE AND NOT A TARGET. `Ask.parent_uids` is the policy's own claim
about which evidence made it aware of the need -- the prompt asks for exactly that ("cite in
parent_uids the units whose content made you aware of this need, or leave it empty if the
need is stated in the task itself"), and 45% of recorded ask-turns carry one. It is the only
signal in the repository that is ABOUT latency and is produced by the policy rather than
derived from gold, which makes it the natural validation target for the automatic label.

It must NOT go into `action_json`. That string is the SFT target and `rung1_sft.mask`
supervises exactly its tokens, so a 64-hex uid would take most of the supervised positions
and train the policy to memorise content hashes. It would also silently move `len_delta`,
which is the length guard's entire job. So it rides as a row FIELD, audited and never scored
-- the treatment `pinq.types.Ask` already prescribes.

THE WEIGHTS FLAG. `headroom_normalised` had no way in: `_weights` built RewardWeights from
`--tau` alone. Measured ON, the depth signal in `value` roughly doubles (+0.208 -> +0.420)
while the "ask early" gradient largely dissolves (-0.644 -> -0.148).
"""

from __future__ import annotations

import argparse
import json

from pi_eval.gold import GoldEdge, GoldGraph, GoldNode
from pi_eval.matcher.base import MatchRecord
from pi_run.cmd_train import _weights, action_json, latent_fields


def _turn(**over):
    t = {
        "action_kind": "ask",
        "question": "who owned The Collegian",
        "rationale": "need the owner",
        "parent_uids": [],
        "turn_idx": 0,
    }
    t.update(over)
    return t


# ------------------------------------------------------- the target stays clean


def test_the_action_target_carries_no_uids() -> None:
    """The loss mask supervises these tokens. A content hash in there is memorisation."""
    js = action_json(_turn(parent_uids=["692d48a2e33a2c66", "b1582a72bd56d544"]))
    assert "692d48a2" not in js
    assert set(json.loads(js)) == {"action", "question", "rationale"}


def test_the_target_is_unchanged_by_the_addition() -> None:
    """len_delta and the tokenized action length must not move for an existing row."""
    assert action_json(_turn()) == json.dumps(
        {"action": "ASK", "question": "who owned The Collegian", "rationale": "need the owner"},
        sort_keys=True,
    )


# --------------------------------------------------------------- the row fields


def _graph():
    def n(nid, depth):
        return GoldNode(
            gold_suite="musique",
            gold_task_key="t0",
            gold_node_id=nid,
            gold_text=f"need {nid}",
            gold_partition="required",
            gold_depth=depth,
        )

    return GoldGraph(
        gold_suite="musique",
        gold_task_key="t0",
        gold_nodes=(n("a", 0), n("b", 1)),
        gold_edges=(
            GoldEdge(
                gold_suite="musique",
                gold_task_key="t0",
                gold_src_node_id="a",
                gold_dst_node_id="b",
                gold_edge_kind="prerequisite",
            ),
        ),
    )


def _rec(nid, turn):
    return MatchRecord(
        run_id="r0",
        suite_id="musique",
        task_id="t0",
        node_id=nid,
        match_kind="resolve",
        matched_turn_idx=turn,
        matcher_id="mechanical_v3",
        matcher_family="rule",
        matcher_score=1.0,
        threshold=1.0,
        graph_version="v1",
    )


def test_a_latent_turn_is_labelled() -> None:
    got = latent_fields(_graph(), [_rec("a", 0), _rec("b", 1)], turn_idx=1)
    assert got["is_latent"] is True
    assert got["latent_depth"] == 1
    assert got["newly_reachable"] is True
    assert got["frontier_size"] == 1


def test_a_depth_zero_turn_is_labelled_too() -> None:
    """Every row carries the fields; absence of a label is not the same as a missing key."""
    got = latent_fields(_graph(), [_rec("a", 0), _rec("b", 1)], turn_idx=0)
    assert got["is_latent"] is False and got["latent_depth"] == 0


def test_a_turn_that_resolved_nothing_is_minus_one_not_zero() -> None:
    got = latent_fields(_graph(), [_rec("a", 0)], turn_idx=3)
    assert got["latent_depth"] == -1 and got["is_latent"] is False


def test_the_self_report_rides_as_a_field() -> None:
    """`parent_uids` is the policy's own latency claim: kept, audited, never scored."""
    from pi_run.cmd_train import self_reported_parents

    assert self_reported_parents(_turn(parent_uids=["u1", "u2"])) == ("u1", "u2")
    assert self_reported_parents(_turn()) == ()


# ------------------------------------------------------------------ the flag


def test_the_headroom_flag_reaches_the_weights() -> None:
    a = argparse.Namespace(tau=0.05, headroom_normalised=True)
    assert _weights(a).headroom_normalised is True


def test_it_is_off_unless_asked_for() -> None:
    """No existing export may change scale without someone saying so."""
    assert _weights(argparse.Namespace(tau=0.05)).headroom_normalised is False


def test_the_two_new_latent_fields_survive_to_the_written_row():
    """REGRESSION. `has_gold_node` and `nameable_from_task` were computed by
    `latent_fields`, carried on the row dict, and dropped at write time because `Example`
    did not declare them -- `write_jsonl` serialises `asdict(it)`, so an undeclared key is
    silently lost. A full export produced 43,609 rows with both fields None before this.

    The file's own comment warns about exactly this, which is why the test is here and not
    a docstring.
    """
    from dataclasses import fields

    from pinq_train.export.dataset import Example

    names = {f.name for f in fields(Example)}
    assert "has_gold_node" in names, "computed gold-side, must be declared to survive asdict()"
    assert "nameable_from_task" in names

    base = dict(
        suite_id="musique",
        task_id="t0",
        run_id="r0",
        turn_idx=0,
        state_text="s",
        action_json="{}",
        value=0.0,
        is_stop=False,
    )
    ex = Example(**base, has_gold_node=True, nameable_from_task=True)
    from dataclasses import asdict

    row = asdict(ex)
    assert row["has_gold_node"] is True
    assert row["nameable_from_task"] is True
    # and the default must be None, not False: "not measured" is not "not nameable"
    assert Example(**base).nameable_from_task is None
