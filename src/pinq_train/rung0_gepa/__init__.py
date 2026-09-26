"""Rung 0 -- GEPA-style prompt optimisation. No GPU, and the one rung that must ship.

It produces the Inquirer prompt the UNTRAINED and FRONTIER families share, which is what
turns "trained vs prompted" into a comparison of model pins rather than a prompt contest.
See `search.py` for the argument in full.
"""

from .config import (
    CACHE_DISCOUNT,
    DEFAULT_PRICE_TABLE,
    K_BY_SUITE,
    CostProjection,
    Rung0Config,
    Rung0ConfigError,
)
from .search import (
    Candidate,
    Evaluator,
    OpenAIReflector,
    Reflector,
    SeamEvaluator,
    SearchRefused,
    SearchResult,
    TaskKey,
    TaskResult,
    accept_mutation,
    call_with_retry,
    is_transient_http,
    mean_reward,
    openai_base,
    ownership,
    pareto_front,
    proxy_model_name,
    rollout_succeeded,
    run_search,
    sample_parent,
    task_keys,
    write_winner,
)

__all__ = [
    "CACHE_DISCOUNT",
    "DEFAULT_PRICE_TABLE",
    "K_BY_SUITE",
    "Candidate",
    "CostProjection",
    "Evaluator",
    "OpenAIReflector",
    "Reflector",
    "Rung0Config",
    "Rung0ConfigError",
    "SearchRefused",
    "SearchResult",
    "SeamEvaluator",
    "TaskKey",
    "TaskResult",
    "accept_mutation",
    "mean_reward",
    "call_with_retry",
    "is_transient_http",
    "openai_base",
    "rollout_succeeded",
    "proxy_model_name",
    "ownership",
    "pareto_front",
    "run_search",
    "sample_parent",
    "task_keys",
    "write_winner",
]
