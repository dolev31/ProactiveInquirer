"""ASKING THE USER FOR WHAT ONLY THE USER KNOWS, as a measurement rather than a claim.

`user_private_ask_precision` and `user_private_ask_recall` are named in the evaluation plan
(datasets_description/18_qa_training_and_evaluation_start.md:361,412) and existed in no code.
They are the only pair that separates "the policy asked the USER" from "the policy asked the
RETRIEVER", which is the operational content of proactive-without-harassing: every other
discovery metric in this repository is indifferent to which channel a question went down.

THE THREE THINGS THAT MAKE IT COMPUTABLE, each pinned by a test below.

  1. THE PARTITION. `GoldNode.gold_discoverability` is `kb | user_private | unknown` and is
     populated on the tau2 family only. `discoverability_ceiling` already reads it.
  2. THE DENOMINATOR. On tau2 (banking) every one of the 3,637 `user_private` nodes is
     `optional` -- there are ZERO required user-private needs -- so the recall denominator is
     EMPTY there. NaN, and no row: a 0/0 rendered as 0.0 would report that the policy asked
     the user for nothing it needed, on the one domain where the question is unanswerable.
     tau2_airline (58 required user-private nodes at v1) and tau2_retail (219) are well formed.
  3. THE CHANNEL. `Ask.target` is `{"kb","user"}`, is carried on `turns.parquet` and is
     therefore an objective record. A closed-channel arm emits zero user-target asks, which
     makes its recall STRUCTURALLY 0 -- entailed by the arm, not measured -- so the rates are
     withheld there and only the count is written.

WHY THE ASK INSTRUMENT IS THE MATCHER'S AND NOT A NEW ONE. `matcher.base.asked_about` already
decides "did this question mention this need" under ASK_MIN_COVERAGE/ASK_MIN_TERMS, and those
constants ride into matcher_hash -> scorer_hash. A second term-overlap rule here would be a
second instrument answering the same question, which is the shape this repository keeps finding
bugs in.
"""

from __future__ import annotations

import math

import pytest

from pi_eval.gold import GoldGraph, GoldNode


def _node(nid, disc, partition="required", text=None):
    return GoldNode(
        gold_suite="s",
        gold_task_key="t",
        gold_node_id=nid,
        gold_text=text if text is not None else nid,
        gold_partition=partition,
        gold_discoverability=disc,
    )


def _graph(nodes):
    return GoldGraph(gold_suite="s", gold_task_key="t", gold_nodes=tuple(nodes))


def _ask(idx, question, target="user"):
    return {"turn_idx": idx, "action_kind": "ask", "question": question, "target": target}


# The two needs used throughout. Two distinctive terms each, so `asked_about` fires on the
# share rule rather than on the degenerate single-term branch.
PRIVATE = _node("p1", "user_private", text="preferred seating cabin economy")
KB = _node("k1", "kb", text="baggage allowance policy weight")


# ------------------------------------------------------------------ registration


def test_the_two_metrics_are_registered_in_the_scorer():
    """They must flow through scoring and reporting like every other metric, which means
    being in METRICS -- not living in a side script that no table can cite."""
    from pi_eval.score import BY_NAME

    for name in ("user_private_ask_precision", "user_private_ask_recall"):
        assert name in BY_NAME, name
        assert BY_NAME[name].judge_derived is False, f"{name} must not sit behind sigma_J"


# ------------------------------------------------------------------ the channel


def test_a_question_sent_to_the_retriever_is_not_an_ask_to_the_user():
    """THE WHOLE POINT. A kb-target question that names a user-private need is a retrieval
    attempt, not harassment, and must not enter either numerator."""
    from pi_eval.metrics.human import user_private_ask

    g = _graph([PRIVATE, KB])
    out = user_private_ask([_ask(0, "PREFERRED SEATING CABIN ECONOMY", target="kb")], g)
    assert out["n_user_asks"] == 0.0
    assert math.isnan(out["precision"])
    assert out["n_reached"] == 0.0


def test_a_user_ask_that_names_a_private_need_is_precise():
    from pi_eval.metrics.human import user_private_ask

    out = user_private_ask([_ask(0, "PREFERRED SEATING CABIN ECONOMY")], _graph([PRIVATE, KB]))
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert (out["n_user_asks"], out["n_attributed"], out["n_universe"]) == (1.0, 1.0, 1.0)


def test_a_user_ask_for_something_the_kb_holds_is_imprecise():
    """Asking the customer for the baggage policy IS the harassment the metric exists to
    catch: the answer was in the knowledge base."""
    from pi_eval.metrics.human import user_private_ask

    out = user_private_ask([_ask(0, "BAGGAGE ALLOWANCE POLICY WEIGHT")], _graph([PRIVATE, KB]))
    assert out["precision"] == pytest.approx(0.0)
    assert out["recall"] == pytest.approx(0.0)


def test_precision_is_a_rate_over_attributed_user_asks():
    from pi_eval.metrics.human import user_private_ask

    out = user_private_ask(
        [
            _ask(0, "PREFERRED SEATING CABIN ECONOMY"),
            _ask(1, "BAGGAGE ALLOWANCE POLICY WEIGHT"),
        ],
        _graph([PRIVATE, KB]),
    )
    assert out["precision"] == pytest.approx(0.5)


# ------------------------------------------------------------------ the denominators


def test_an_unattributable_user_ask_is_excluded_rather_than_scored_zero():
    """A question that names no annotated need is not evidence that the policy asked for
    something the KB held; it is evidence of nothing, and the count says how much."""
    from pi_eval.metrics.human import user_private_ask

    out = user_private_ask(
        [_ask(0, "PREFERRED SEATING CABIN ECONOMY"), _ask(1, "HELLO HOW ARE YOU TODAY")],
        _graph([PRIVATE, KB]),
    )
    assert out["n_user_asks"] == 2.0
    assert out["n_attributed"] == 1.0
    assert out["precision"] == pytest.approx(1.0)


def test_recall_is_nan_when_no_required_need_is_user_private():
    """THE tau2-BANKING DEGENERACY. Every user_private node there is `optional`, so the
    denominator is empty. 0.0 would read "asked the user for none of what it needed"."""
    from pi_eval.metrics.human import user_private_ask

    g = _graph([_node("p1", "user_private", partition="optional", text="preferred cabin economy")])
    out = user_private_ask([_ask(0, "PREFERRED CABIN ECONOMY")], g)
    assert out["n_universe"] == 0.0
    assert math.isnan(out["recall"])
    assert out["precision"] == pytest.approx(1.0), "an optional need is still user-private"


def test_recall_counts_needs_not_asks():
    """Two asks naming the same need are one need reached, never two."""
    from pi_eval.metrics.human import user_private_ask

    other = _node("p2", "user_private", text="travel insurance purchased yes")
    out = user_private_ask(
        [_ask(0, "PREFERRED SEATING CABIN ECONOMY"), _ask(1, "SEATING CABIN ECONOMY PREFERRED")],
        _graph([PRIVATE, other]),
    )
    assert out["n_reached"] == 1.0
    assert out["recall"] == pytest.approx(0.5)


def test_no_user_ask_leaves_both_rates_undefined():
    """A closed-channel arm cannot exhibit either quantity. Recall 0.0 here is entailed by the
    arm table, not measured, and a quantity that cannot take the other value is not evidence."""
    from pi_eval.metrics.human import user_private_ask

    out = user_private_ask([_ask(0, "ANYTHING", target="kb")], _graph([PRIVATE, KB]))
    assert math.isnan(out["precision"])
    assert math.isnan(out["recall"])
    assert out["n_user_asks"] == 0.0


# ------------------------------------------------------------------ through the scorer


def _score(turns, nodes):
    from pi_eval.score import score_run

    run = {"run_id": "r", "suite_id": "s", "task_id": "t", "arm_id": "a"}
    rows = score_run(
        run,
        graph=_graph(nodes),
        turns=turns,
        evidence=[],
        env_calls=[],
        ledger=[],
        records=[],
        answer=None,
    )
    return {r["metric_name"]: r["value"] for r in rows}


def test_the_scorer_emits_both_rates_on_a_well_formed_partition():
    by = _score([_ask(0, "PREFERRED SEATING CABIN ECONOMY")], [PRIVATE, KB])
    assert by["user_private_ask_precision"] == pytest.approx(1.0)
    assert by["user_private_ask_recall"] == pytest.approx(1.0)
    assert by["user_private_ask_n"] == 1.0
    assert by["user_private_ask_n_universe"] == 1.0


def test_the_scorer_withholds_the_recall_row_on_a_degenerate_partition():
    """No row, not a zero: the metric must be ABSENT where its denominator is empty."""
    by = _score(
        [_ask(0, "PREFERRED CABIN ECONOMY")],
        [_node("p1", "user_private", partition="optional", text="preferred cabin economy")],
    )
    assert "user_private_ask_recall" not in by
    assert by["user_private_ask_n_universe"] == 0.0
    assert by["user_private_ask_precision"] == pytest.approx(1.0)


def test_the_scorer_writes_the_count_but_no_rate_for_a_closed_channel_run():
    by = _score([_ask(0, "PREFERRED SEATING CABIN ECONOMY", target="kb")], [PRIVATE, KB])
    assert by["user_private_ask_n"] == 0.0
    assert "user_private_ask_precision" not in by
    assert "user_private_ask_recall" not in by


def test_the_metric_is_absent_on_a_suite_with_no_user_private_partition():
    """musique, strategyqa, wiki2 and drgym carry no user_private node at all. A row there
    would be a measurement of a partition that does not exist."""
    by = _score([_ask(0, "BAGGAGE ALLOWANCE POLICY WEIGHT", target="kb")], [KB])
    assert "user_private_ask_n" not in by
    assert "user_private_ask_precision" not in by


def test_the_duplicated_target_string_matches_the_action_it_reads():
    """`human.USER_TARGET` is a copy of `pinq.types.Ask.target`'s "user" literal, because the
    firewall forbids gold-side code importing the rollout package. The copy is held equal
    HERE, in a test that may import both sides -- the same arrangement as the tau2 tool names
    in `metrics.environment`."""
    import typing

    from pi_eval.metrics.human import USER_TARGET
    from pinq.types import Ask

    allowed = typing.get_args(typing.get_type_hints(Ask)["target"])
    assert USER_TARGET in allowed, allowed
    assert Ask(text="q", target=USER_TARGET).target == USER_TARGET
