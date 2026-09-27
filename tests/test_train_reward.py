"""The training reward, and the four ways it refuses to be gamed.

Every test here corresponds to a strategy that would score well under an obvious alternative
formulation: paraphrasing questions at a matcher, splitting one question into five, asking a
question that returns what is already held, and stopping immediately to bank the stop bonus.
A reward is only as good as the strategies it makes unprofitable, so those are what is tested
rather than the arithmetic.
"""

import pytest

from pinq.wire import ScoreResponse, TurnScore
from pinq_train.reward import (
    POLICY_STOP,
    RewardWeights,
    redundancy,
    reward_of,
    shaped_phi,
    stop_indicator,
    tau_from_pilot,
    turn_values,
)

W = RewardWeights()


def _resp(
    potential,
    *,
    turns=None,
    stop_reason=POLICY_STOP,
    n_ret=None,
    tok_total=0,
    n_malformed=0,
    wall_ms=0,
    episode_id="e1",
):
    """A ScoreResponse shaped exactly as pi_run.serve.score builds one."""
    k = len(potential) - 1
    ts = (
        turns
        if turns is not None
        else [TurnScore(turn_idx=i, n_retrieved=1, n_new=1) for i in range(k)]
    )
    return ScoreResponse(
        episode_id=episode_id,
        split="train",
        q_terminal=potential[-1],
        q_ladder=tuple(potential),
        potential=tuple(potential),
        turns=tuple(ts),
        stopped=stop_reason == POLICY_STOP,
        stop_reason=stop_reason,
        n_ret=float(k if n_ret is None else n_ret),
        tok_total=tok_total,
        wall_ms=wall_ms,
        n_malformed=n_malformed,
    )


# ------------------------------------------------------------------ 1. anti-matcher-gaming


def test_the_reward_cannot_be_moved_by_rewriting_a_question():
    """THE anti-gaming property. Phi is over RETRIEVED EvidenceUnits, so two episodes that
    retrieved the same units score identically no matter what the questions said. Had phi
    been a similarity between question text and a gold need, the optimal policy would have
    been a paraphrase generator that never retrieves anything new."""
    turns_a = [TurnScore(turn_idx=0, n_retrieved=2, n_new=2, qid="q-plain")]
    turns_b = [TurnScore(turn_idx=0, n_retrieved=2, n_new=2, qid="q-stuffed-with-gold-words")]
    a = reward_of(_resp([0.0, 0.6], turns=turns_a))
    b = reward_of(_resp([0.0, 0.6], turns=turns_b))
    assert a.total == b.total
    assert a.shaping_term == b.shaping_term


# ------------------------------------------------------------------ 2/3. fragmentation


def test_the_shaping_sum_never_exceeds_the_quality_gain():
    assert sum(shaped_phi([0.0, 0.2, 0.5, 0.9])) == pytest.approx(0.9)
    assert sum(shaped_phi([0.1, 0.1, 0.1])) == pytest.approx(0.0)


def test_splitting_one_question_into_five_pays_no_more_and_costs_more():
    """Fragmentation is the cheapest exploit of any per-turn shaping term."""
    whole = _resp([0.0, 0.8])  # one question, all of the gain
    frag = _resp([0.0, 0.16, 0.32, 0.48, 0.64, 0.8])  # the same gain, five ways
    rw, rf = reward_of(whole), reward_of(frag)
    assert rw.shaping_term == pytest.approx(rf.shaping_term), "shaping must be split-invariant"
    assert rf.retrieval_cost > rw.retrieval_cost
    assert rf.total < rw.total, "five retrievals for one question's worth must be strictly worse"


def test_the_cap_binds_when_phi_and_q_are_on_different_scales():
    """With Phi = Q the differences telescope and the cap is slack. The cap exists for the
    configuration where they are not the same scale, which is where fragmentation pays."""
    assert sum(shaped_phi([0.0, 0.5, 0.4, 0.9], cap=0.3)) == pytest.approx(0.3)


def test_shaping_is_path_independent():
    """Potential-based: the total depends on the endpoints only, so the shaping term cannot
    express a preference over ORDER that the task reward does not already have."""
    assert sum(shaped_phi([0.0, 0.7, 0.9])) == pytest.approx(sum(shaped_phi([0.0, 0.2, 0.9])))


# ------------------------------------------------------------------ 4. redundancy


def test_a_turn_that_returns_only_known_units_earns_nothing_and_pays_rho():
    turns = [
        TurnScore(turn_idx=0, n_retrieved=2, n_new=2),
        TurnScore(turn_idx=1, n_retrieved=2, n_new=0),  # everything already held
    ]
    r = reward_of(_resp([0.0, 0.5, 0.5], turns=turns))
    assert r.turns[1].phi_tilde == 0.0
    assert r.turns[1].rho == 1.0
    assert turn_values(_resp([0.0, 0.5, 0.5], turns=turns))[1].value < 0.0


def test_redundancy_is_zero_when_nothing_was_retrieved():
    assert redundancy(0, 0) == 0.0
    assert redundancy(4, 1) == pytest.approx(0.75)


# ------------------------------------------------------------------ 5. the stop term


def test_a_budget_stop_earns_no_stop_credit():
    """STOP was the harness's decision, not the policy's. Paying for it would confound the
    policy's stopping rule with the cap the experiment announced."""
    r = reward_of(_resp([0.0, 0.9], stop_reason="budget"))
    assert r.sigma == 0.0 and r.stop_term == 0.0


def test_stopping_immediately_cannot_collect_the_stop_bonus():
    r = reward_of(_resp([0.0], stop_reason=POLICY_STOP))
    assert r.sigma == 0.0, "a zero-question episode is evidence about nothing"


def test_one_junk_question_then_stop_cannot_buy_the_stop_bonus():
    """The exploit a naive '+1 for stopping' pays: ask something worthless, stop, collect."""
    r = reward_of(_resp([0.0, 0.0]))
    assert r.sigma == 0.0
    assert r.total < 0.0, "it still pays for the retrieval it wasted"


def test_stopping_while_the_last_question_was_still_gaining_is_penalised():
    r = reward_of(_resp([0.0, 0.2, 0.9]))  # last step 0.7 >> tau
    assert r.sigma == -1.0


def test_stopping_after_the_returns_flatten_is_credited():
    w = RewardWeights()
    r = reward_of(_resp([0.0, 0.9, 0.9 + w.tau / 2]))
    assert r.sigma == 1.0


def test_stop_indicator_reads_the_threshold_it_is_given():
    assert stop_indicator(stop_reason=POLICY_STOP, phi=[0.5], tau=0.05) == -1.0
    assert stop_indicator(stop_reason=POLICY_STOP, phi=[0.5], tau=0.9) == 0.0
    assert stop_indicator(stop_reason="max_turns", phi=[0.0], tau=0.05) == 0.0


# ------------------------------------------------------------------ weights and provenance


def test_c_ret_must_equal_tau():
    """The retrieval price IS the formalism's stopping threshold. Decoupling them lets the
    policy's cost model and its stop rule disagree, and neither is then the formalism's."""
    with pytest.raises(ValueError, match="c_ret"):
        reward_of(_resp([0.0, 0.5]), RewardWeights(c_ret=0.2, tau=0.05))


def test_a_negative_coefficient_is_refused():
    with pytest.raises(ValueError, match="negative"):
        RewardWeights(w_phi=-1.0).validate()


def test_every_weight_changes_the_sha():
    """A stored reward value must name the weights that produced it (CONTRIBUTING.md rule 1)."""
    base = RewardWeights().sha
    assert RewardWeights(w_phi=0.6).sha != base
    assert RewardWeights(lambda_fmt=0.3).sha != base
    assert RewardWeights().sha == base


def test_tau_is_the_35th_percentile_of_pilot_phi():
    vals = [i / 100 for i in range(101)]
    assert tau_from_pilot(vals) == pytest.approx(0.35, abs=0.01)
    with pytest.raises(ValueError, match="measured"):
        tau_from_pilot([])


def test_from_pilot_ties_c_ret_to_the_measured_threshold():
    w = RewardWeights.from_pilot([0.0, 0.1, 0.2, 0.3])
    w.validate()
    assert w.c_ret == w.tau


# ------------------------------------------------------------------ costs and refusals


def test_a_malformed_generation_is_charged():
    clean = reward_of(_resp([0.0, 0.5]))
    bad = reward_of(_resp([0.0, 0.5], n_malformed=2))
    assert bad.total == pytest.approx(clean.total - 2 * W.lambda_fmt)


def test_tokens_are_charged_per_thousand_and_latency_is_free_by_default():
    """wall_ms is a machine artifact (CONTRIBUTING.md); a reward that depends on it produces a
    checkpoint that cannot be reproduced on another host."""
    r = reward_of(_resp([0.0, 0.5], tok_total=12_000, wall_ms=600_000))
    assert r.token_cost == pytest.approx(12 * W.c_tok)
    assert r.latency_cost == 0.0


def test_an_unsupported_measurement_is_refused_rather_than_rewarded():
    resp = _resp([0.0, 0.5])
    from dataclasses import replace

    with pytest.raises(ValueError, match="unsupported|usable"):
        reward_of(replace(resp, supported=False, note="stop_probe needs a rollout"))


def test_a_ladder_that_disagrees_with_the_turns_is_refused():
    """Zip-and-truncate would silently drop the reward of the last turns."""
    bad = _resp([0.0, 0.3, 0.6], turns=[TurnScore(turn_idx=0, n_retrieved=1, n_new=1)])
    with pytest.raises(ValueError, match="disagree"):
        reward_of(bad)


def test_the_breakdown_sums_to_the_total():
    """`total` must be a sum of the stored fields and nothing else, or a re-weighting is a
    re-run instead of a spreadsheet."""
    r = reward_of(_resp([0.0, 0.4, 0.7], tok_total=3000, n_malformed=1))
    assert r.total == pytest.approx(
        r.task_term
        + r.shaping_term
        - r.redundancy_term
        + r.stop_term
        - r.retrieval_cost
        - r.token_cost
        - r.latency_cost
        - r.malformed_cost
    )
