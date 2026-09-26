"""The observation wall, and the only object allowed to hold a live `TravelEnv`.

THE SECOND FOOTGUN, STATED PLAINLY
    In tau2 the answer key lives in a task record the adapter can simply decline to read. In
    UserBench it lives on the env object the agent loop is holding, and -- worse -- inside
    the observation that same loop is handed on every step:

      env.state_list["remaining_best_options"]     the ids that pay choice_best_reward (1.0)
      env.state_list["remaining_correct_options"]  the ids that pay choice_correct_reward (0.8)
      env.remaining_preferences                    the preferences not yet elicited
      observation["task_description"]              the full `scenario` prose, every
                                                   preference written out (see
                                                   `_probe.scenario_public` for the
                                                   measurement)
      info["preferences_summary"]                  a formatted list of exactly the
                                                   preferences the agent is supposed to be
                                                   discovering

    Nothing in upstream separates these from the parts an agent may see. `[answer]` scoring
    is pure set membership against `remaining_best_options` / `remaining_correct_options`,
    with no judge in the loop, so a policy that could read one field of the env it was handed
    would score 1.0 on every task and the result would be a well-formed lie.

WHAT THIS FILE DOES ABOUT IT
    Two things, in the spirit of the `TaskView` type wall.

    1. `ObservationView` carries the agent-visible fields under names that SHARE NO FIELD
       NAME with the raw observation or info dicts -- `agent_feedback`, not `feedback`. So
       `ObservationView(**observation)` is a TypeError, `dataclasses.asdict()` cannot round-
       trip a raw key into it, and a future field added upstream cannot arrive by accident.
       This is the same trick that keeps `GoldNode` (`gold_*`) from typechecking as a
       `TaskView`, and it is enforced by
       tests/test_adapters_userbench.py::test_the_observation_view_shares_no_field_name_with_the_raw_observation.

    2. `UserBenchSession` owns the env and never returns it. `step()` returns a
       `StepResult`, which holds an `ObservationView` and a `TurnMetrics` and no reference to
       the env, the task record or the state list.

WHY THE PREFERENCE COUNTERS ARE ON THE RECORD SIDE AND NOT THE AGENT SIDE
    `remaining_preferences`, `total_preferences` and `elicitation_ratio` are in the raw
    observation and they are all derived from the answer key. Telling a policy "you have three
    preferences left to find" is telling it something the simulated user never said, and it
    turns elicitation into a countdown. Upstream's own baseline never sees them either --
    eval.py threads `observation["feedback"]` into the model's context and records the
    counters separately. They belong in `TurnMetrics`, which is scored and never shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .simulator import (
    SimulatorFailureCounter,
    SimulatorPin,
    counted_simulator,
    openai_endpoint,
)

# The raw keys TravelEnv puts in front of a caller. Written out so the disjointness of
# `ObservationView`'s field names is a checked property rather than a claim.
RAW_OBSERVATION_KEYS: frozenset[str] = frozenset(
    {
        "task_description",
        "goal",
        "feedback",
        "step_count",
        "episode_complete",
        "total_preferences",
        "remaining_preferences",
        "elicitation_ratio",
        "active_elicited_preferences",
        "passive_elicited_preferences",
        "last_reward",
    }
)
RAW_INFO_KEYS: frozenset[str] = frozenset(
    {
        "task_id",
        "preferences_summary",
        "action_history",
        "conversation_history",
        "elicited_preferences",
        "newly_elicited_preferences",
        "final_elicitation_ratio",
        "total_reward",
    }
)

# The subset of the above that is the answer key or a direct function of it. Named so that a
# leak is a named failure rather than a judgement call, and so a reviewer can check the list
# against upstream instead of against a paraphrase.
GOLD_SURFACE_KEYS: frozenset[str] = frozenset(
    {
        # the full scenario prose: every preference, written out
        "task_description",
        # the preferences the agent has not yet found, formatted for display
        "preferences_summary",
        # counts of, and progress through, the same hidden set
        "total_preferences",
        "remaining_preferences",
        "elicitation_ratio",
        "final_elicitation_ratio",
        "elicited_preferences",
        "newly_elicited_preferences",
    }
)

# Prefixes of the four legal actions. `[finish]` takes no payload; the other three do.
ACTION_PREFIXES: tuple[str, ...] = ("[search]", "[action]", "[answer]", "[finish]")


class ObservationShapeChanged(RuntimeError):
    """A key this wall depends on is missing from the raw observation.

    Loud on purpose. If upstream renames `feedback`, the alternative to raising is an agent
    that is shown an empty string every turn and a table of zeros that nobody can explain.
    """


class EpisodeAlreadyOver(RuntimeError):
    """A step was attempted after the env declared the episode complete."""


@dataclass(frozen=True, slots=True)
class ObservationView:
    """Everything the policy may see, and nothing else.

    `agent_feedback` is the env's `observation["feedback"]`: the simulated user's reply, the
    database's search results, or the verdict on an `[answer]`. It is precisely the channel
    upstream's own baseline agent reads, which is what keeps our numbers comparable to the
    published ones.

    The `agent_` prefix is not decoration -- see this module's docstring.
    """

    agent_feedback: str
    agent_step: int
    agent_episode_over: bool


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    """The scored side of one turn. Recorded in the run log, never shown to a policy.

    `active_elicited` / `passive_elicited` are UserBench's own counters and the reason this
    suite is worth running. If the agent goes `elicitation_interval` turns (default 3)
    without asking a preference question that the judge recognises, the simulated user
    VOLUNTEERS one unprompted and `passive_elicited` goes up. So a policy that never asks
    still accumulates preferences, and a headline "preferences elicited" number that pooled
    the two would credit the agent for the environment giving up on it. `active_elicited` is
    the numerator that means something.
    """

    reward: float
    terminated: bool
    truncated: bool
    active_elicited: int
    passive_elicited: int
    total_preferences: int
    remaining_preferences: int

    @property
    def active_elicitation_share(self) -> float:
        """`active / total` AS OF THIS TURN, and the denominator moves. Read the next paragraph.

        `total_preferences` is upstream's `len(remaining) + len(elicited)`, and an `[answer]`
        under `one_choice_per_aspect` DELETES the still-unelicited preferences of the aspect
        it answered (`evaluate_action` pops them out of `available_preferences`, which is the
        env's own `remaining_preferences` list). So the denominator SHRINKS as a policy
        commits, and the share taken from the last turn of an episode is not taken over the
        same set as the share taken from the first. Measured on the first live episode:
        travel22/`apartment:2-86|rental_car:2-60` opened with `total_preferences=4` and ended
        with 3, after one `[answer]`.

        That is upstream's accounting and this adapter does not change it -- changing it would
        be adapting the benchmark to suit the metric. It does mean a cross-task average of
        this property is over unlike denominators; the honest per-episode figure is the pair
        (`active_elicited`, `total_preferences` AT RESET), both of which are recorded on
        every turn.

        0.0 when the task carries no preferences at all, which is a degenerate scenario
        rather than a perfect score.
        """
        return (self.active_elicited / self.total_preferences) if self.total_preferences else 0.0


@dataclass(frozen=True, slots=True)
class StepResult:
    """What a driver gets back. Holds no env, no task record and no state list."""

    observation: ObservationView
    metrics: TurnMetrics


def make_observation(raw: Mapping[str, Any]) -> ObservationView:
    """Project a raw TravelEnv observation onto the agent-visible fields.

    A PROJECTION, NOT AN ALLOWLIST CHECK, and the difference matters. `pinq.view.make_view`
    raises on an unknown key because its caller chooses the keys; here the dict is upstream's
    and always contains `task_description`, so raising on unknown keys would raise on every
    real step. Instead this reads three named keys and copies nothing else -- so a field
    added upstream is ignored by construction rather than by a list someone has to maintain.

    Missing keys DO raise: an absent `feedback` means the shape changed, and silently
    substituting "" would show the agent nothing and blame the policy for it.
    """
    missing = [k for k in ("feedback", "step_count", "episode_complete") if k not in raw]
    if missing:
        raise ObservationShapeChanged(
            f"TravelEnv observation is missing {missing}; the agent-visible surface can no "
            "longer be derived and this adapter must be re-checked against the checkout"
        )
    return ObservationView(
        agent_feedback=str(raw["feedback"]),
        agent_step=int(raw["step_count"]),
        agent_episode_over=bool(raw["episode_complete"]),
    )


def make_metrics(
    raw: Mapping[str, Any],
    *,
    reward: float,
    terminated: bool,
    truncated: bool,
    counters: Mapping[str, Any] | None = None,
) -> TurnMetrics:
    """The record side. `counters` is `env.state_list`, and it is not an optimisation.

    WHY THE OBSERVATION IS NOT ENOUGH. `travel_env.py` builds THREE observation dicts and
    only one of them -- the `step()` success path -- writes `active_elicited_preferences` /
    `passive_elicited_preferences`. The `[finish]` branch returns before either is set, and
    so does `reset()`. Reading them out of the observation alone therefore reports 0 active
    and 0 passive for every episode a policy ENDED, which is the case in which it is most
    likely to have earned a non-zero one: a silent floor of zero on the headline number this
    whole suite exists to report. `test_the_elicitation_counters_survive_the_turn_that_ends
    _the_episode` is that bug, pinned.

    THE OBSERVATION STILL WINS WHERE IT SPEAKS. `counters` is consulted only for a key the
    observation omits, because what upstream reported is what upstream reported; if a later
    version stopped copying the state list into the observation, that must show up as a
    change in behaviour rather than be papered over here.

    `counters` IS THE ANSWER KEY'S OWN DICT -- `remaining_best_options` is in it. Two ints are
    read out by name and nothing else is copied, so the returned `TurnMetrics` holds no
    reference to it; `test_the_counters_are_read_from_the_state_list_without_carrying_it`
    checks that rather than trusting this paragraph.
    """

    def _count(key: str) -> int:
        if key in raw:
            return int(raw[key])
        return int((counters or {}).get(key, 0))

    return TurnMetrics(
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        active_elicited=_count("active_elicited_preferences"),
        passive_elicited=_count("passive_elicited_preferences"),
        total_preferences=int(raw.get("total_preferences", 0)),
        remaining_preferences=int(raw.get("remaining_preferences", 0)),
    )


@dataclass
class UserBenchSession:
    """One episode. Points the endpoint, arms the failure counter, owns the env, hands back
    only views.

    Used as a context manager, and that is not a style choice: `__enter__` sets
    `OPENAI_BASE_URL` from the pin and arms `counted_simulator` BEFORE calling `env_factory`,
    so there is no window in which a `TravelEnv` exists whose simulator calls could go
    unwatched or reach an endpoint nothing recorded. A session cannot be handed an
    already-built env for the same reason -- both would then be installed after the fact and
    would have nothing to say about whatever had already run.

    THE ENDPOINT IS SET THROUGH THE ENVIRONMENT because there is no other lever:
    `travelgym/env/prompts.py` builds `OpenAI(api_key=model_config["api_key"])` with no
    `base_url`, so the OpenAI SDK falls back to `OPENAI_BASE_URL`. It is restored on exit,
    including on the failure path -- see `_unwind`.

    `env_factory` is a zero-argument callable rather than a config, so this is drivable
    offline against a stub. The tests do that; it is the only way the refusal can be checked
    without an API key. A stub session passes `pin=None` and touches the environment not at
    all.
    """

    env_factory: Callable[[], Any]
    # (module, attr) pairs; None -> travelgym's real entry points, resolved on entry.
    targets: Sequence[tuple[Any, str]] | None = None
    counter: SimulatorFailureCounter = field(default_factory=SimulatorFailureCounter)
    # WHICH MODEL ANSWERS AS THE USER. None is the offline shape -- a session driving a stub
    # has no endpoint to name -- and `UserBenchSuite.session()` never passes it: it resolves
    # a pin from the environment and refuses if there is none. See `simulator.SimulatorPin`.
    pin: SimulatorPin | None = None
    label: str = "userbench"

    _env: Any = field(default=None, init=False, repr=False)
    _guard: Any = field(default=None, init=False, repr=False)
    _endpoint: Any = field(default=None, init=False, repr=False)
    _turn: int = field(default=0, init=False)

    def __enter__(self) -> "UserBenchSession":
        # ENDPOINT FIRST, THEN THE COUNTER, THEN THE ENV. The endpoint has to be in place
        # before any client can be built, and the counter before any call can be made; the
        # env is last because building one is what makes both reachable.
        self._endpoint = openai_endpoint(self.pin)
        self._endpoint.__enter__()
        self._guard = counted_simulator(self.targets, counter=self.counter)
        try:
            self._guard.__enter__()
            self._env = self.env_factory()
        except BaseException:
            self._unwind()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        env, self._env = self._env, None
        close = getattr(env, "close", None)
        if callable(close):
            close()
        self._unwind(*exc)

    def _unwind(self, *exc: Any) -> None:
        """Drop the guard and the endpoint, in the reverse of the order they were taken.

        Shared with the failure path in `__enter__` so that a factory that raises cannot
        leave either one installed -- a leaked `model_call` wrapper would make the NEXT
        session observe a counter it never armed, and a leaked `OPENAI_BASE_URL` would
        redirect whatever ran next in the process.
        """
        exc = exc or (None, None, None)
        guard, self._guard = self._guard, None
        if guard is not None:
            guard.__exit__(*exc)
        endpoint, self._endpoint = self._endpoint, None
        if endpoint is not None:
            endpoint.__exit__(*exc)

    # ------------------------------------------------------------------------ the one door

    def reset(self) -> StepResult:
        """The opening observation.

        `TravelEnv.__init__` already calls `reset()`, so this is the second one; upstream's
        own eval.py does exactly the same, and reset is idempotent for `data_mode="single"`
        (it re-selects `tasks[0]` and rebuilds the preference and state lists from it).
        Calling it here rather than trusting construction means the opening line the agent
        sees comes from the same call that produced the state it will act against.
        """
        env = self._require_env()
        raw, _info = env.reset()
        self.counter.raise_if_any(where=f"{self.label} reset")
        return StepResult(
            observation=make_observation(raw),
            metrics=make_metrics(
                raw,
                reward=0.0,
                terminated=False,
                truncated=False,
                counters=getattr(env, "state_list", None),
            ),
        )

    def step(self, action: str) -> StepResult:
        """One `[search]` / `[action]` / `[answer]` / `[finish]`, refused if the simulator failed.

        THE ORDER OF THE THREE LINES BELOW IS THE POINT. The env runs first and produces
        upstream's plausible zero; the counter is consulted second and raises; the result is
        built third and is therefore never built for a turn that had no measurement. Checking
        the counter before the step, or building the result before the check, would each
        re-open the hole this package exists to close.
        """
        env = self._require_env()
        if getattr(env, "episode_complete", False):
            raise EpisodeAlreadyOver(
                f"{self.label}: the episode is complete; start a new session rather than "
                "resetting this one, so that a run id maps to exactly one episode"
            )
        if not str(action).lstrip().startswith(ACTION_PREFIXES):
            # Upstream's own answer is "Your response format is wrong and cannot be parsed
            # properly." with reward 0.0 -- another well-formed zero. It is a real part of the
            # benchmark for a POLICY's malformed output, but a driver that sends an unprefixed
            # string is a harness bug, and the two must not look alike in a results file.
            raise ValueError(
                f"{self.label}: action must start with one of {ACTION_PREFIXES}, got "
                f"{str(action)[:40]!r}. UserBench scores an unprefixed action as 0.0 with a "
                "canned reply, so this is refused rather than silently measured."
            )
        self._turn += 1
        raw, reward, terminated, truncated, _info = env.step(action)
        self.counter.raise_if_any(where=f"{self.label} turn {self._turn} ({action.split(']')[0]}])")
        return StepResult(
            observation=make_observation(raw),
            metrics=make_metrics(
                raw,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                # `env.state_list` is the authoritative copy of the two elicitation counters,
                # and the ONLY copy on a `[finish]` turn -- see `make_metrics`.
                counters=getattr(env, "state_list", None),
            ),
        )

    def usage(self) -> dict[str, int]:
        """The counter, for the run record. Always recorded -- see `as_dict`."""
        return self.counter.as_dict()

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError(
                f"{self.label}: no env. UserBenchSession must be used as a context manager, "
                "so that the simulator failure counter is armed before the env exists."
            )
        return self._env
