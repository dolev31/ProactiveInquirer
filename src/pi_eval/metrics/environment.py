"""Environment-side metrics: what the policy DID, not what it said.

Everything here reads `env_calls.parquet` rows and a `GoldGraph`, and nothing here reads
answer text, question text or a judge. That is the point of the tau2 suite: the discoverable
tool has an unguessable four-digit suffix, so "did this policy cross the prerequisite edge?"
is a binary fact about an executed call rather than an opinion about a paragraph.

THE TWO NUMBERS, AND WHY BOTH EXIST

  `unlock_and_invoke` (already emitted by score.py) is ANY: did the run get at least one
  gated call through? It is binary, so McNemar applies, which is why the preregistration
  pairs it with an exact test.

  `discoverable_tool_unlock_tau2` is the RATE over the tools this task's required documents
  actually unlock. A task whose gold names four discoverable tools and a task whose gold
  names one are not the same trial, and collapsing both to a single bit throws away the
  distinction between crossing one edge and crossing all four.

WHY THE DENOMINATOR IS GOLD AND NOT THE RUN'S OWN CALLS
  Counting successes over attempts would reward a policy for attempting less: a run that
  unlocked exactly one tool and stopped would score 1.0. The denominator is the set of
  tool_unlock nodes in the task's gold graph — the tools named by documents the benchmark
  author declared REQUIRED — so the metric asks "of what was reachable, how much did you
  reach?" and a policy cannot improve it by trying fewer things.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from pi_eval.gold import GoldGraph

NAN = float("nan")

# The two tau2 tool names the prerequisite mechanic runs through. Duplicated from
# `pinq_adapters.tau2.actuator` ON PURPOSE: the firewall forbids pi_eval importing an
# adapter, so the strings are held equal by tests/test_adapters_tau2.py rather than by an
# import that would make gold-side code depend on the rollout package being installed.
UNLOCK_TOOL = "unlock_discoverable_agent_tool"
CALL_TOOL = "call_discoverable_agent_tool"


def _target_of(call: Mapping[str, Any]) -> str:
    """The discoverable tool a call names, from its stored kwargs.

    `kwargs_json` is `pinq.ids.canon(kwargs)`, i.e. plain JSON, so this is a parse and not a
    heuristic. A row whose kwargs do not parse yields "" and is counted as naming nothing —
    never as naming some default tool.
    """
    try:
        kwargs = json.loads(str(call.get("kwargs_json") or "{}"))
    except (TypeError, ValueError):
        return ""
    if not isinstance(kwargs, Mapping):
        return ""
    return str(kwargs.get("agent_tool_name") or kwargs.get("tool_name") or "")


def tools_unlocked(env_calls: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Tools this run successfully unlocked. An `ok` unlock call and nothing weaker.

    A failed unlock leaves the tool locked, so crediting it would fabricate a crossed
    prerequisite edge — the one thing this metric family exists to make impossible.
    """
    return frozenset(
        t
        for c in env_calls
        if str(c.get("tool_name")) == UNLOCK_TOOL and bool(c.get("ok")) and (t := _target_of(c))
    )


def tools_invoked(env_calls: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Tools this run actually CALLED through the gate: `unlock_required` and satisfied and ok.

    Reads `unlock_required` / `unlock_satisfied` rather than re-deriving them from the tool
    name, because those two columns are what the actuator observed at execution time. A call
    that was gated and whose gate was not satisfied is a call the environment refused, and it
    must not count toward a rate.
    """
    return frozenset(
        t
        for c in env_calls
        if bool(c.get("unlock_required"))
        and bool(c.get("unlock_satisfied"))
        and bool(c.get("ok"))
        and (t := _target_of(c))
    )


def required_tools(graph: GoldGraph) -> tuple[str, ...]:
    """The discoverable tools this task's gold graph says are unlockable, sorted.

    Read from `gold_aliases`, which `pi_eval.build.tau2_build` fills with the tool name
    verbatim. `gold_text` is an English sentence ("the tool X exists and can be unlocked")
    and splitting its third word would be a join key that a reworded docstring breaks.
    """
    return tuple(
        sorted(
            {
                alias
                for n in graph.gold_nodes
                if n.gold_kind == "tool_unlock"
                for alias in n.gold_aliases
                if alias
            }
        )
    )


def discoverable_tool_unlock(
    env_calls: Sequence[Mapping[str, Any]], graph: GoldGraph
) -> dict[str, float]:
    """The tau2 discoverable-tool endpoint, as a rate plus the two counts behind it.

    Returns NaN for the rate when the task's gold names no discoverable tool. NaN and not
    0.0: 48 of the 97 tasks require no unlock at all, and scoring them 0 would report that
    every policy fails half the suite at something the suite never asked for.
    """
    want = frozenset(required_tools(graph))
    if not want:
        return {"rate": NAN, "n_required": 0.0, "n_invoked": 0.0, "n_unlocked": 0.0}
    invoked = tools_invoked(env_calls) & want
    return {
        "rate": len(invoked) / len(want),
        "n_required": float(len(want)),
        "n_invoked": float(len(invoked)),
        "n_unlocked": float(len(tools_unlocked(env_calls) & want)),
    }


def tau_reward(native: Mapping[str, float]) -> float:
    """The tau2 primary endpoint, read from the stored native reward. NaN when absent.

    ABSENT IS NOT ZERO, and this is the whole reason the function exists rather than a
    `.get(k, 0.0)` at the call site. `tau_reward` is missing for a task whose reward_basis
    the environment evaluator does not cover (9 of 97 are ACTION-basis) and for a run whose
    grading replay failed. A zero there would say "the policy did the wrong thing"; a NaN
    says "this run was not graded", and only one of those is true.
    """
    if "tau_reward" not in native:
        return NAN
    try:
        return float(native["tau_reward"])
    except (TypeError, ValueError):
        return NAN


def is_measured(value: float) -> bool:
    return not math.isnan(value)


def user_turns(run: Mapping[str, Any]) -> float:
    """How many times the USER spoke, from the stored tau2 transcript count. NaN when absent.

    ABSENT IS NOT ZERO, for the same reason as `tau_reward` and a sharper one. Only the tau2
    suite has a user simulator at all; on musique, strategyqa, wiki2 and drgym the field is
    null. A zero there would read "the user never had to say anything" -- the best possible
    score on the anticipation endpoint -- for every run on every suite that has no user.

    THIS IS NOT A COUNT OF OUR OWN ASKS. `tau2_runner` drives the inner policy with
    `allow_user_target=False`, so an ASK is charged as a wasted turn and never reaches the
    transcript (`pinq.loop.run_loop`). Every user message is the simulator speaking of its
    own accord, which is what makes the endpoint impossible to inflate by asking more.
    """
    value = run.get("n_user_turns")
    if value is None:
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def user_followups(run: Mapping[str, Any]) -> float:
    """User turns AFTER the opening task statement. NaN when unmeasured.

    WHAT IT MEASURES, AT SCALE, AND WHY THE PILOT NUMBER MUST NOT BE QUOTED.

    A 2-task pilot showed drafter_only 17.00 against inquirer_prompted 2.25 -- an 86.8%
    reduction, and it was on its way into a headline. Run over 84 tasks x 3 seeds (482
    reportable runs):

        drafter_only        16.00
        inquirer_prompted   14.09        = 11.9% reduction
        paired difference   +1.91  sd 7.59  95% CI [+0.28, +3.53]
        paired t p=0.024 | Wilcoxon p=0.004 | sign test 53 better / 30 worse, p=0.015

    The effect is REAL -- three tests agree at p < 0.05 -- and it is SEVEN TIMES SMALLER than
    the pilot. Two tasks happened to be ones the inquirer handled unusually well. The paired sd
    of 7.59 against a mean difference of 1.91 is why: per-task variance dwarfs the effect, so
    any handful of tasks can show almost anything.

    Quote 11.9% with its CI. The pilot figure is an artifact of n=2.

    The raw count has no meaningful zero: the user always speaks once to state the task, and
    charging a policy for that turn scores every arm at least 1 on a metric whose floor is
    supposed to mean "the user never had to come back". The follow-ups are the anticipation
    quantity -- each one is a need the policy could have resolved and did not.

    Clamped at 0. A transcript with no user message is malformed rather than a run that
    somehow anticipated the task statement itself.
    """
    turns = user_turns(run)
    if math.isnan(turns):
        return NAN
    return max(0.0, turns - 1.0)
