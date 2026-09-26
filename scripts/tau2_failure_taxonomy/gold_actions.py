"""tau2's own gold action set and registered tool names, read directly from the `tau2` package.

NOT THROUGH `pi_eval`. `evaluation_criteria` here is tau2-bench's OWN reference solution for
its OWN 97/50/114-task benchmark sets, loaded through `tau2.registry` exactly as
`pinq_adapters.tau2.actuator._compute_reward` loads it at rollout time (that module's own
docstring: "evaluation_criteria is read only inside _compute_reward, after the last plan step
has run, and never reaches a prompt"). It is not this repository's `data/gold` (MuSiQue /
StrategyQA / wiki2 information-need graphs), so none of this needs `PI_GOLD_ROOT` and none of
it is behind the firewall `pi_eval.gold.gold_root()` guards -- it is upstream benchmark data,
public in the tau2-bench release, read the same way the rollout code itself reads it.

Every tau2 import is inside a function body, matching `pinq_adapters/tau2/_probe.py`'s own
convention, so importing this module costs nothing on a machine without the `tau2` extra
installed; only calling one of its functions does.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

# suite_id (this repo's) -> tau2 domain name (upstream's), for the three tau2 suites this lane
# reads. `tau2_golden` is deliberately absent: it is reached only through a different retrieval
# variant that is out of scope for a failure taxonomy over recorded runs (see
# `pinq_adapters/tau2/_probe.py`'s GOLDEN_RETRIEVAL_VARIANT docstring).
_DOMAIN_OF_SUITE: Mapping[str, str] = {
    "tau2_retail": "retail",
    "tau2_airline": "airline",
    "tau2": "banking_knowledge",
}


def domain_for(suite_id: str) -> str | None:
    return _DOMAIN_OF_SUITE.get(suite_id)


@lru_cache(maxsize=None)
def _env_kwargs_for(domain: str) -> tuple[tuple[str, str], ...]:
    """Extra `env_kwargs` a domain's constructor needs, mirroring what this repository's own
    suite builder passes (`pinq_adapters.tau2.suite.py`'s `env_kwargs()`), NOT tau2's default.

    Banking's registry default is a dense/embedding retrieval variant that raises without an
    OpenAI key (measured directly: `OpenAIError: Missing credentials`); this repository never
    runs that variant (`_probe.RETRIEVAL_VARIANT = "bm25"`, chosen specifically because it
    "is offline and dependency-free"). Reusing that constant rather than re-hardcoding "bm25"
    here is the point: a future change to which variant this repo runs would otherwise silently
    desync this module from the one that actually ran the campaigns it is auditing.
    """
    if domain == "banking_knowledge":
        from pinq_adapters.tau2._probe import RETRIEVAL_VARIANT

        return (("retrieval_variant", RETRIEVAL_VARIANT),)
    return ()


@lru_cache(maxsize=None)
def _env_for(domain: str) -> object:
    from tau2.registry import registry

    kwargs = dict(_env_kwargs_for(domain))
    return registry.get_env_constructor(domain)(**kwargs)


@lru_cache(maxsize=None)
def registered_tool_names(suite_id: str) -> frozenset[str]:
    """Every tool name callable by EITHER role (agent, user) for this suite's domain.

    Checking only `env.get_tools()` (the agent's toolkit) undercounts: banking's gold actions
    route several steps through `env.get_user_tools()` instead (`apply_for_credit_card`,
    `call_discoverable_user_tool`, ...), and a name that is real on the user's toolkit is not
    "a tool the environment lacks" just because the agent's toolkit does not also carry it.
    Measured directly (`artifacts/tau2_failure_taxonomy_20260919/RESULT.md` ss4): 15 agent +
    6 user = 21 distinct names for banking_knowledge/bm25.
    """
    domain = _DOMAIN_OF_SUITE.get(suite_id)
    if domain is None:
        raise ValueError(f"no tau2 domain mapped for suite_id={suite_id!r}")
    env = _env_for(domain)
    names = {t.name for t in env.get_tools()}
    get_user_tools = getattr(env, "get_user_tools", None)
    if callable(get_user_tools):
        names |= {t.name for t in get_user_tools()}
    return frozenset(names)


@lru_cache(maxsize=None)
def _tasks_for(domain: str) -> tuple:
    from tau2.registry import registry

    loader = registry.get_tasks_loader(domain)
    return tuple(loader())


@lru_cache(maxsize=None)
def _gold_by_task(suite_id: str) -> Mapping[str, tuple[str, ...]]:
    domain = _DOMAIN_OF_SUITE.get(suite_id)
    if domain is None:
        raise ValueError(f"no tau2 domain mapped for suite_id={suite_id!r}")
    out: dict[str, tuple[str, ...]] = {}
    for task in _tasks_for(domain):
        criteria = getattr(task, "evaluation_criteria", None)
        actions = getattr(criteria, "actions", None) or [] if criteria is not None else []
        out[str(task.id)] = tuple(a.name for a in actions)
    return out


def gold_action_names(suite_id: str, task_id: str) -> tuple[str, ...] | None:
    """The gold `evaluation_criteria.actions[*].name` sequence for one task, or None when the
    suite/task is not one tau2's own registry knows (a fork's `task_id` is always a real tau2
    task id, so None here means a wiring bug in the caller, not a legitimate miss)."""
    by_task = _gold_by_task(suite_id)
    return by_task.get(task_id)


def unresolvable_gold_actions(suite_id: str, task_id: str) -> tuple[str, ...]:
    """Gold action names for this task that name a tool NEITHER toolkit registers. Empty when
    every gold action resolves, empty also when the task has no gold or no actions (an empty
    gold action list, e.g. retail's "the correct outcome is a database that does not change",
    is not a missing tool)."""
    names = gold_action_names(suite_id, task_id) or ()
    if not names:
        return ()
    registered = registered_tool_names(suite_id)
    return tuple(n for n in names if n not in registered)


def reward_basis(suite_id: str, task_id: str) -> tuple[str, ...]:
    domain = _DOMAIN_OF_SUITE.get(suite_id)
    if domain is None:
        return ()
    for task in _tasks_for(domain):
        if str(task.id) == task_id:
            criteria = getattr(task, "evaluation_criteria", None)
            basis = getattr(criteria, "reward_basis", None) or () if criteria is not None else ()
            return tuple(str(getattr(b, "value", b)).upper() for b in basis)
    return ()


def all_task_ids(suite_id: str) -> Sequence[str]:
    return tuple(_gold_by_task(suite_id))
