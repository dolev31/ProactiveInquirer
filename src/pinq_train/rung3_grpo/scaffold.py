"""Rung 3 -- GRPO. OPTIONAL, UNVALIDATED, AND ON A HARD KILL DATE OF 2026-09-12.

READ THIS BEFORE READING THE CODE
=================================
Nothing in this file has ever been run. It is a scaffold: a config, the wiring that connects
`pinq_train.reward` to a rollout server, and a written description of the contract such a
server must satisfy. There is no trainer here, there is no verl integration here, and there is
no evidence that any of it works. Calling `train()` raises. That is deliberate, and it is the
honest state of this rung rather than a gap somebody forgot to fill.

WHY IT IS OPTIONAL AND WHY THE KILL DATE IS IN THE CODE
-------------------------------------------------------
Rung 3 is the only rung that cannot be run on rented hardware for a few hundred dollars: the
plan prices it at $2,400 and four days. It is also the only rung whose success is genuinely
uncertain in advance, for the reason rung 2's docstring gives -- terminal-reward policy
gradient has to estimate `V(s_t)` from returns collected across tasks whose difficulty
variance dwarfs the treatment effect. GRPO's group-relative baseline mitigates that (the group
shares a prompt), but it does not remove it, and G=8 groups on ~6-turn episodes is a thin
estimator for a wide variance.

So the rung starts only on an EXPLICIT WRITTEN GO, and it dies on 2026-09-12 whether or not it
has converged. `assert_go()` enforces both. The date is a constant in code rather than a note
in a plan because a kill date that lives in a document is a kill date that gets renegotiated
at 2 a.m. on the day it fires.

THE WRITTEN FALLBACK, COMMITTED IN ADVANCE
------------------------------------------
If the reward curve is not monotone over 300 steps by the kill date: kill it, and SHIP THE
LEARNING CURVE AS A REPORTED NEGATIVE RESULT WITH A DIAGNOSIS. Not a footnote, not a
"future work" line -- a figure, the reward decomposition that explains which term failed to
move, and the variance argument above. The strongest version of this paper containing no
trained model at all is already a complete paper; rung 3 is upside, and treating upside as a
requirement is how a deadline gets missed for a result nobody needed.

THE ROLLOUT-SERVER CONTRACT
---------------------------
Whatever trainer eventually drives this (verl, a custom loop) must speak the seam that already
exists, and must not be allowed to bypass it:

  POST /rollout   one episode per group member. `RolloutRequest.policy_base_url` addresses the
                  trainer's own vLLM engine, so the policy being optimised is the policy being
                  rolled out. The server must run with PI_GOLD_ROOT UNSET.
  POST /score     components for one recorded episode, from a process with PI_GOLD_ROOT SET.
                  Refuses any task outside `train` with 403. The trainer applies the weights;
                  the server never returns a reward.
  POST /retrieve  Search-R1 shaped, so the retrieval a baseline sees is byte-identical.
  GET  /healthz   `gold_root_set` decides which of the two roles a process may play.

`group_rewards()` below is the only computation this module actually performs, and it is the
one piece that is unit-tested: given G scored episodes for one prompt, it produces the
group-relative advantages Dr-GRPO uses. Everything else is a description.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

from pinq.ids import canon, h
from pinq.wire import ScoreResponse
from pinq_train.reward import DEFAULT_WEIGHTS, RewardWeights, reward_of

# Hard kill. Not a suggestion, not a target, and not in a document where it can be renegotiated.
KILL_DATE = date(2026, 9, 12)

# What "it is working" means, decided in advance so it cannot be decided by whatever the curve
# happens to look like on the day.
MONOTONE_WINDOW_STEPS = 300

# The endpoints any trainer driving this rung must speak. Pinned against
# `pinq.wire.Health.endpoints` by tests/test_train_rungs.py so the two cannot drift.
ROLLOUT_SERVER_CONTRACT: tuple[str, ...] = ("/rollout", "/score", "/retrieve", "/healthz")


class Rung3NotStarted(RuntimeError):
    """Rung 3 was invoked without a written go, or after its kill date."""


class Rung3NotValidated(NotImplementedError):
    """There is no trainer here. Said out loud rather than implied by an empty function."""


@dataclass(frozen=True, slots=True)
class Rung3Config:
    """Dr-GRPO with clip-higher and a KL to rung 2. Values from the plan; none is validated."""

    base_model: str = ""
    adapter: str = ""  # rung 2's checkpoint: both the init and the KL reference
    out_dir: str = "artifacts/rung3"
    rollout_url: str = "http://127.0.0.1:8078"  # PI_GOLD_ROOT unset
    score_url: str = "http://127.0.0.1:8077"  # PI_GOLD_ROOT set
    group_size: int = 8  # G
    steps: int = 600
    learning_rate: float = 1e-6
    kl_coef: float = 0.02
    clip_low: float = 0.2
    clip_high: float = 0.28  # clip-higher: asymmetric, to keep exploration alive
    normalize_advantage_by_std: bool = False  # Dr-GRPO drops the std term
    seed: int = 0
    usd_budget: float = 2_400.0

    @property
    def sha(self) -> str:
        return h("rung3", canon(asdict(self)))

    def validate(self) -> None:
        if self.group_size < 2:
            raise ValueError(
                f"group_size={self.group_size}: a group of one has no relative baseline, which "
                "is the only thing GRPO has instead of a critic."
            )
        if not self.adapter:
            raise ValueError(
                "adapter is empty. Rung 3 initialises from rung 2 AND takes its KL reference "
                "from it; starting anywhere else makes the arm incomparable to the rung it is "
                "supposed to improve on."
            )
        if self.clip_high < self.clip_low:
            raise ValueError("clip_high < clip_low: clip-higher means the upper bound is wider")


def assert_go(*, written_go: bool, today: date | None = None) -> None:
    """Both gates, in one place. Raises `Rung3NotStarted` with the reason."""
    now = today or date.today()
    if not written_go:
        raise Rung3NotStarted(
            "rung 3 starts only on an explicit written go. It is the one rung whose success is "
            "genuinely uncertain in advance and the only one that cannot be rented cheaply; "
            "starting it by default is how a deadline is missed for upside nobody needed."
        )
    if now > KILL_DATE:
        raise Rung3NotStarted(
            f"today is {now.isoformat()}, past the hard kill date {KILL_DATE.isoformat()}. The "
            "written fallback applies: ship the learning curve as a reported negative result "
            "with a diagnosis. Do not restart it."
        )


# --------------------------------------------------------------------------- the one computation


@dataclass(frozen=True, slots=True)
class GroupAdvantages:
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean: float
    std: float


def group_rewards(
    responses: Sequence[ScoreResponse],
    weights: RewardWeights | None = None,
    *,
    normalize_by_std: bool = False,
) -> GroupAdvantages:
    """G scored episodes for ONE prompt -> group-relative advantages.

    `normalize_by_std=False` is Dr-GRPO's correction and the default. Dividing by the group's
    standard deviation up-weights groups that happened to be homogeneous, which on this task
    means up-weighting the EASY prompts -- exactly the prompts where the treatment has the
    least room to show anything. The bias is small per step and systematic across every step.
    """
    if len(responses) < 2:
        raise ValueError(
            f"{len(responses)} episode(s) in a group: the group IS the baseline, so a group of "
            "one has no advantage signal at all."
        )
    w = weights or DEFAULT_WEIGHTS
    rewards = [reward_of(r, w).total for r in responses]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if normalize_by_std and std > 0:
        adv = [(r - mean) / std for r in rewards]
    else:
        adv = [r - mean for r in rewards]
    return GroupAdvantages(rewards=tuple(rewards), advantages=tuple(adv), mean=mean, std=std)


def is_monotone(curve: Sequence[float], *, window: int = MONOTONE_WINDOW_STEPS) -> bool:
    """The go/no-go on the reward curve, decided in advance.

    "Monotone" is deliberately weak -- the mean of the last third exceeds the mean of the first
    third over the window. A strict step-by-step monotonicity test would fail on any real RL
    curve and would therefore never be applied honestly.
    """
    c = list(curve)[-window:]
    if len(c) < 3:
        return False
    third = max(1, len(c) // 3)
    return statistics.fmean(c[-third:]) > statistics.fmean(c[:third])


def train(cfg: Rung3Config, *, written_go: bool = False, today: date | None = None) -> None:
    """There is no trainer here. This raises, and says why.

    `today` exists so a caller (a test) can say which day it is asking about; the gate is
    date-dependent and a test that relies on the wall clock stops meaning anything after the
    kill date.
    """
    assert_go(written_go=written_go, today=today)
    cfg.validate()
    raise Rung3NotValidated(
        "rung 3 is a scaffold: config, reward wiring and the rollout-server contract, and "
        "nothing else. No GRPO step has ever been executed in this repository and no claim "
        "about its behaviour is supported by anything here. Wire verl (or another trainer) to "
        f"{ROLLOUT_SERVER_CONTRACT} using group_rewards() for the advantage, and treat the "
        "first run as a debugging session. If the curve is not monotone over "
        f"{MONOTONE_WINDOW_STEPS} steps by {KILL_DATE.isoformat()}, the written fallback "
        "applies: ship the curve as a negative result with a diagnosis."
    )
