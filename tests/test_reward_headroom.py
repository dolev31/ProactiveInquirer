"""Headroom-normalised shaping: pay for the fraction of what is LEFT, not of the whole.

WHY. MEASURED over 361 non-STOP decision points on 13 tasks (the current SFT export, joined
to `matches.parquet` at `mechanical_v2` and to gold `gold_depth`):

    Spearman(value, gold depth)                     = +0.208   95% CI [+0.065, +0.483]
    Spearman(value, turn_idx)                       = -0.643
    Spearman(depth,  turn_idx)                      = +0.257
    Spearman(value, depth | turn held fixed)        = +0.507   95% CI [+0.241, +0.751]

    mean value: turn 0 -> +0.2016 ... turn 5 -> +0.0045     (45x)
    mean value at turn 0: depth 0 +0.115 -> depth 2 +0.228   (2x)

So the reward ALREADY prefers deep needs -- at rho ~ +0.5 once turn is controlled -- and the
"ask early" gradient is ~20x larger and points the other way. A latent need is reachable only
AFTER the evidence that reveals it, so the two forces oppose by construction, and the raw
signal a policy actually sees is the diluted +0.208.

The cause is not a bug: `phi_tilde` is a gain over a FIXED gold set, so a later question has
mechanically less left to find. The fix is the denominator, not a depth bonus.

WHY -log(1 - Q) AND NOT phi/(1 - Q). Dividing each increment by the remaining headroom is the
obvious move and it BREAKS the two load-bearing properties in reward.py's docstring:
potential-based shaping (property 2, Ng-Harada-Russell invariance) requires the term to be
Phi(s_{t+1}) - Phi(s_t) for SOME Phi, and the ratio form is not; the telescoping cap
(property 3, "splitting one useful question into five fragments must not pay five times")
then stops binding.

Psi = -log(1 - Q) is a potential whose increments ARE the headroom-normalised gain:

    Psi(s_{t+1}) - Psi(s_t) = -log(1 - (Q_{t+1} - Q_t)/(1 - Q_t))

i.e. a strictly increasing function of the fraction of REMAINING coverage the turn captured.
Both properties survive because it is still a potential, and `shaped_phi`'s default cap
`p[-1] - p[0]` still binds with equality.
"""

from __future__ import annotations

import math

import pytest

from pinq.wire import ScoreResponse, TurnScore
from pinq_train.reward import (
    POLICY_STOP,
    RewardWeights,
    headroom_potential,
    reward_of,
    shaped_phi,
)


def _resp(potential, *, stop_reason=POLICY_STOP):
    k = len(potential) - 1
    return ScoreResponse(
        episode_id="e",
        split="train",
        q_terminal=potential[-1],
        q_ladder=tuple(potential),
        potential=tuple(potential),
        turns=tuple(TurnScore(turn_idx=i, n_retrieved=1, n_new=1) for i in range(k)),
        stopped=stop_reason == POLICY_STOP,
        stop_reason=stop_reason,
        n_ret=float(k),
        tok_total=0,
        wall_ms=0,
        n_malformed=0,
        ok=True,
        supported=True,
    )


# --------------------------------------------------------------- the transform


def test_zero_coverage_maps_to_zero_potential() -> None:
    """Psi(0) = 0, so an episode that found nothing is still worth nothing."""
    assert headroom_potential([0.0])[0] == pytest.approx(0.0)


def test_it_is_strictly_increasing() -> None:
    """Order must be preserved, or the transform would reverse a preference."""
    out = headroom_potential([0.0, 0.25, 0.5, 0.75])
    assert all(b > a for a, b in zip(out, out[1:]))


def test_a_late_gain_is_worth_more_than_the_same_early_gain() -> None:
    """THE POINT. +0.2 of coverage from a base of 0.7 beats +0.2 from a base of 0.0.

    Under the raw potential these two are identical, which is exactly why the reward could
    not distinguish a hard late question from an easy first one.
    """
    early = headroom_potential([0.0, 0.2])
    late = headroom_potential([0.7, 0.9])
    assert (late[1] - late[0]) > (early[1] - early[0])


def test_full_coverage_is_finite() -> None:
    """-log(0) is +inf; a finite reward may never depend on a clip that is not there."""
    out = headroom_potential([0.0, 1.0])
    assert all(math.isfinite(x) for x in out)
    assert out[1] > out[0]


def test_out_of_range_coverage_is_clamped_not_trusted() -> None:
    """Q is a fraction. A scorer that emits 1.4 or -0.1 is broken, and log() would raise."""
    out = headroom_potential([-0.1, 1.4])
    assert all(math.isfinite(x) for x in out)


# ------------------------------------------- the two properties that must survive


def test_it_is_still_a_potential_so_the_shaping_telescopes() -> None:
    """Property 2. The increments must sum to Psi(end) - Psi(start), path-independently."""
    psi = headroom_potential([0.0, 0.3, 0.5, 0.9])
    assert sum(shaped_phi(psi)) == pytest.approx(psi[-1] - psi[0])


def test_splitting_one_question_into_five_still_cannot_pay_twice() -> None:
    """Property 3, on the transformed ladder: fragmentation is what the cap exists for."""
    whole = headroom_potential([0.0, 0.8])
    frags = headroom_potential([0.0, 0.16, 0.32, 0.48, 0.64, 0.8])
    assert sum(shaped_phi(frags)) == pytest.approx(sum(shaped_phi(whole)))


# ------------------------------------------------------- wiring into the reward


def test_it_is_off_by_default_and_changes_nothing() -> None:
    """No existing number may move. This is opt-in, and the default must be byte-identical."""
    assert RewardWeights().headroom_normalised is False
    r = _resp([0.0, 0.4, 0.6])
    assert reward_of(r).total == pytest.approx(reward_of(r, RewardWeights()).total)


def test_enabling_it_reweights_the_late_turn_upward() -> None:
    """The behavioural claim, end to end through `reward_of`.

    Two equal raw gains of +0.3. Off: identical shaping. On: the second is worth more,
    because it took a larger share of what was left.
    """
    on = RewardWeights(headroom_normalised=True)
    br_off = reward_of(_resp([0.0, 0.3, 0.6]))
    br_on = reward_of(_resp([0.0, 0.3, 0.6]), on)
    off = [t.phi_tilde for t in br_off.turns]
    got = [t.phi_tilde for t in br_on.turns]
    assert off[0] == pytest.approx(off[1]), "raw gains are equal by construction"
    assert got[1] > got[0], "headroom-normalised, the later equal gain must pay more"


def test_the_flag_changes_the_provenance_sha() -> None:
    """Two rewards on different scales must never share a weights_sha."""
    assert RewardWeights().sha != RewardWeights(headroom_normalised=True).sha


def test_the_c_ret_equals_tau_identity_is_untouched() -> None:
    """The stopping threshold IS the retrieval price, on whichever scale is in force."""
    with pytest.raises(ValueError):
        RewardWeights(headroom_normalised=True, c_ret=0.05, tau=0.09).validate()
    RewardWeights(headroom_normalised=True).validate()
