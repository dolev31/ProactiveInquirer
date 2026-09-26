"""Rung 3 -- GRPO. OPTIONAL and UNVALIDATED, with a hard kill date of 2026-09-12.

Nothing here has ever been run. It is a config, the reward wiring, and a written description
of the rollout-server contract; `train()` raises. See `scaffold.py`, which says so at length
and states the fallback that was committed in advance.
"""

from .scaffold import (
    KILL_DATE,
    MONOTONE_WINDOW_STEPS,
    ROLLOUT_SERVER_CONTRACT,
    GroupAdvantages,
    Rung3Config,
    Rung3NotStarted,
    Rung3NotValidated,
    assert_go,
    group_rewards,
    is_monotone,
    train,
)

__all__ = [
    "KILL_DATE",
    "MONOTONE_WINDOW_STEPS",
    "ROLLOUT_SERVER_CONTRACT",
    "GroupAdvantages",
    "Rung3Config",
    "Rung3NotStarted",
    "Rung3NotValidated",
    "assert_go",
    "group_rewards",
    "is_monotone",
    "train",
]
