"""The HTTP seam. The only way `pinq_train` reaches a score.

Four endpoints, and the reason each exists:

  POST /rollout   one episode, with the policy addressed as an OpenAI-compatible base_url so
                  a trained checkpoint, a prompted open model and a frontier API run through
                  one code path.
  POST /score     what an episode was worth, and the REFUSAL of anything outside `train`.
  POST /retrieve  Search-R1's contract verbatim, so external baselines can be pointed at our
                  frozen corpus rather than at a differently-built index.
  GET  /healthz   whether PI_GOLD_ROOT is set, which decides which of the two roles this
                  process is allowed to play.

Nothing is imported here that a caller might not want: `handle_*` are plain functions over
dataclasses, and `create_app` (which needs `fastapi`, from the `serve` extra) is imported
lazily from `.app` by whoever actually starts a server.
"""

from .retrieve import RetrieverPool, SuiteNotConfigured, handle_retrieve
from .rollout import (
    CorpusNotFound,
    FrozenRolesWouldMove,
    check_base_url,
    handle_rollout,
    policy_pin,
    resolve_corpus,
)
from .score import (
    EVAL_ONLY_SUITES,
    EpisodeNotFound,
    NoGoldForTask,
    SplitRefused,
    assert_scorable,
    handle_score,
    scorer_hash,
)

__all__ = [
    "EVAL_ONLY_SUITES",
    "CorpusNotFound",
    "EpisodeNotFound",
    "FrozenRolesWouldMove",
    "NoGoldForTask",
    "RetrieverPool",
    "SplitRefused",
    "SuiteNotConfigured",
    "assert_scorable",
    "check_base_url",
    "handle_retrieve",
    "handle_rollout",
    "handle_score",
    "policy_pin",
    "resolve_corpus",
    "scorer_hash",
]
