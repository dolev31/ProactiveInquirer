"""UserBench (Salesforce, `travelgym`), VIEW SIDE. Apache-2.0, EVALUATION ONLY.

WHY THIS SUITE IS IN THE PAPER
    It is the one benchmark here that this project did not build. Every other suite's need
    graph, or its judge, or its corpus split, was constructed by the people whose claim it
    is being used to support, and "you measured yourself with your own ruler" is the first
    thing a reviewer says. UserBench's environment, its user simulator, its rewards and its
    85/15 split are all somebody else's, published before this work existed. That is its
    entire value, and it is also the reason nothing in this file may improve it: an
    adaptation that made the benchmark easier, cleaner or better matched to our metrics would
    hand the ruler back to us.

    Concretely: the option pools, the `[search]` judge, the `elicitation_interval` at which
    the simulated user gives up and volunteers a preference, and the 1.0/0.8/0.2/0.2 reward
    ladder are all taken as they are. What this file adds is a wall (`session.py`), a refusal
    (`simulator.py`), and a choice of which upstream field an agent may see.

EVAL-ONLY, FOR TAU2'S REASON
    `pinq.splitting.EVAL_ONLY_SUITES` contains "userbench", so `split_of` returns "test" for
    every task and no row from here can reach `data/rl/`. Training on a benchmark whose
    independence is the reason it is being cited would destroy the only claim it carries,
    and it would do so invisibly -- the resulting number would still look like a transfer
    result. Upstream ships 2,651 train scenarios; this adapter cannot see them, because
    `SPLIT` is a constant in `_probe` rather than a parameter.

WHAT AN AGENT IS SHOWN
    `initial_description` -- the user's incomplete opening -- and never `scenario`, which
    spells out the preferences. The measurement behind that line is in
    `_probe.scenario_public`; the observation-side half of the same wall is `session.py`.

WHAT IS NOT DRIVEN FROM HERE
    `retriever()` and `actuator()` both refuse. UserBench's retrieval is the `[search]`
    action, judged by an LLM inside the env, and its actuation is `[answer]` against the
    env's own state list. Neither is a free local operation, and a local stand-in for either
    would be a different benchmark wearing this one's name. `session()` is the only door.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import TaskId, TaskView
from pinq.view import make_view

from ._probe import (
    ENVS,
    N_TEST_BY_ENV,
    N_TEST_TASKS,
    PINNED_COMMIT,
    SPLIT,
    UserBenchUnavailable,
    available,
    data_file,
    env_tag,
    load_scenarios,
    scenario_public,
    split_index,
    userbench_root,
)
from .session import UserBenchSession
from .simulator import SimulatorFailureCounter, SimulatorPin

CORPUS_ID = "userbench_travelgym_v1"

# Public by construction: this is a restatement of the action space upstream itself puts in
# the system prompt of every row of the split index. It is NOT copied from that prompt --
# ours is hashed into RunManifest.prompt_hashes and must be ours to change.
INSTRUCTIONS = (
    "You are a travel-planning assistant talking to a user whose stated request is "
    "deliberately incomplete. Every turn is exactly one of:\n"
    "  [search] <query>   -- query the travel database for one travel aspect at a time\n"
    "  [action] <message> -- say something to the user; ask about one concrete, specific "
    "preference rather than a general question\n"
    "  [answer] <ID>      -- recommend exactly one option id, e.g. H12 or F3\n"
    "  [finish]           -- end the session\n"
    "The user will not volunteer their preferences if you ask well; you have to elicit them."
)

# TravelGymConfig's own default. Upstream's README publishes `--max_turns 20` while eval.py's
# argparse default is 8, so upstream disagrees with itself; the config default is the one
# that is a declaration rather than a CLI convenience, and pinning it here means the turn
# budget is a recorded decision instead of whichever file was read last.
MAX_STEPS = 20

# TravelGymConfig's own default, and NOT eval.py's. `eval.py --one_choice` is declared
# `type=bool`, so argparse runs `bool("False")` and any value at all -- "False", "0", "no" --
# yields True, while omitting the flag yields False and silently overrides the config's True.
# The flag is therefore incapable of expressing one of its two states. This adapter never
# goes through eval.py (see this module's `session`) and pins the config default explicitly.
ONE_CHOICE_PER_ASPECT = True

# Pinned so the episode is replayable. `TravelEnv.__init__` seeds `random` and `numpy.random`
# from this BEFORE loading tasks, which fixes two things that are otherwise not reproducible:
# the shuffle of each aspect's option list, and `random.choice(available_preferences)` in the
# passive-elicitation path. Left unset (upstream's default), `travelgym.env.task_data` seeds
# numpy once at import and every subsequent env construction continues that stream, so the
# second env built in a process sees a different option order than the first.
ENV_SEED = 0

# Upstream's config default is 15.0 seconds and it is a TRANSPORT parameter, not part of what
# the benchmark measures: no reward, judgement or option pool depends on it. It is raised here
# because of what a timeout now COSTS. Upstream swallows one into `model_call` -> None -> a
# canned reply and reward 0.0, so a slow endpoint quietly depressed a score; with the failure
# counter armed the same timeout REFUSES the episode instead, so a too-tight value turns
# ordinary latency into an aborted run. 15.0 was chosen for gpt-4o's latency, and a reasoning
# model behind a proxy is not that. Recorded here rather than left at the default so the value
# a number was produced under is a decision someone made.
SIMULATOR_TIMEOUT_S = 90.0

# Upstream's config default, kept. It is the CONTENT budget for a short JSON judgement, and on
# a reasoning model the hidden channel is billed inside it -- so if the reasoning ever eats the
# whole budget the content comes back empty, `parse_output_as_json("")` returns None, and the
# failure counter refuses the run. That is the loud outcome, not a silent zero, which is why
# this stays at upstream's number instead of being pre-emptively inflated.
SIMULATOR_MAX_TOKENS = 2048


class UserBenchNeedsGymLoop(RuntimeError):
    """Raised when UserBench is asked for a flat retrieval or actuation it cannot honestly do.

    The analogue of `Tau2NeedsOrchestrator`: a loud refusal, because the alternative is a
    plausible number produced by a mechanism the benchmark does not contain.
    """


@dataclass
class UserBenchSuite:
    """Satisfies pinq.protocols.TaskSuite over UserBench's 255-task test split.

    The split index is read eagerly (it is a few kilobytes of parquet); the ~200 MB of
    scenarios is read lazily, one data file at a time, because `task_ids()` does not need it
    and a validator that only wants a task count should not pay for the corpus.
    """

    suite_id: ClassVar[str] = "userbench"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = CORPUS_ID
    instructions: ClassVar[str] = INSTRUCTIONS
    pinned_commit: ClassVar[str] = PINNED_COMMIT

    root: Path | None = None
    envs: tuple[str, ...] = ENVS
    strict_counts: bool = True
    max_steps: int = MAX_STEPS
    one_choice_per_aspect: bool = ONE_CHOICE_PER_ASPECT
    seed: int = ENV_SEED
    simulator_timeout_s: float = SIMULATOR_TIMEOUT_S
    simulator_max_tokens: int = SIMULATOR_MAX_TOKENS

    corpus_hash: str = field(default="", init=False)
    _env_of: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _order: tuple[str, ...] = field(default=(), init=False, repr=False)
    _scenarios: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = userbench_root(self.root)
        rows = split_index(self.root, self.envs, SPLIT)
        self._order = tuple(key for key, _ in rows)
        self._env_of = {key: env for key, env in rows}

        if self.strict_counts:
            # Assert, never assume. These are the denominators of every UserBench row in the
            # paper, and an upstream re-split would move all of them without touching code.
            if len(self._order) != len(self._env_of):
                raise UserBenchUnavailable(
                    f"{len(self._order)} split rows but {len(self._env_of)} distinct scenario "
                    "keys -- the split index has duplicates and every per-task number would "
                    "be weighted by how often a task happened to be listed"
                )
            for env in self.envs:
                want = N_TEST_BY_ENV.get(env)
                got = sum(1 for e in self._env_of.values() if e == env)
                if want is not None and got != want:
                    raise UserBenchUnavailable(
                        f"{env}: expected {want} {SPLIT} tasks at {PINNED_COMMIT}, found {got}"
                    )
            if self.envs == ENVS and len(self._order) != N_TEST_TASKS:
                raise UserBenchUnavailable(
                    f"expected {N_TEST_TASKS} {SPLIT} tasks across {ENVS}, found {len(self._order)}"
                )

        # WHAT THIS HASH COVERS, AND WHAT IT DOES NOT. It covers the identity of the TASK
        # LIST -- which scenario keys, in which environment -- plus the byte size of each
        # data file the tasks are drawn from. It is a tripwire against a swapped or
        # re-generated corpus, not a cryptographic pin of 200 MB of JSON: digesting the
        # scenarios would cost seconds on every `task_ids()`, and the cryptographic pin
        # already exists and is `PINNED_COMMIT`, checked out by hand and recorded in
        # docs/DATA.md. Saying which of the two this is, here, is the point.
        sizes = {
            t: str(data_file(self.root, t).stat().st_size) for t in {env_tag(e) for e in self.envs}
        }
        self.corpus_hash = _corpus_hash((key, env, sizes[env_tag(env)]) for key, env in rows)

    # ------------------------------------------------------------------ TaskSuite protocol

    def task_ids(self) -> tuple[TaskId, ...]:
        """Upstream's own file order, NOT sorted.

        Deliberately different from tau2, which sorts. Sorting a UserBench split would
        interleave the three environments, so a `--limit N` smoke run would silently sample
        across travel22/33/44 instead of taking the head of one; keeping file order makes a
        truncated run a truncated run of a named environment.
        """
        return tuple(self._order)

    def view(self, tid: TaskId) -> TaskView:
        """The task as the AGENT may see it: the user's opening line, and nothing else.

        `initial_description` is a genuinely incomplete request -- "I am planning a family
        trip to Austin from November 10th to November 15th, staying in an apartment and
        renting a car for the entire duration." -- and it is exactly what the env returns as
        `observation["feedback"]` at reset, which is the only channel upstream's own baseline
        agent reads. So this view is faithful to the published setup as well as non-leaky.

        The sibling field `scenario` is what makes this a decision rather than a lookup: it
        reads like a task description and it contains the preferences. See
        `_probe.scenario_public` for the numbers.
        """
        record = self._scenario_record(str(tid))
        return make_view(
            task_id=str(tid),
            suite_id=self.suite_id,
            question=scenario_public(record)["initial_description"],
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            # One frozen cap across every arm, so an arm can never win on utterance length.
            # 60 words is long enough for a specific preference question and far too short
            # for a policy to recite a plan at the user; the `[answer]` payload it also
            # bounds is a single option id.
            word_cap=60,
        )

    def retriever(self, tid: TaskId) -> Any:
        """REFUSES. UserBench retrieval is the `[search]` action, judged by an LLM in the env.

        A local retriever here would have to rank the scenario's `all_options`, which the env
        hands over ONLY after a search whose arguments the judge accepted -- and that
        acceptance is one of the four things the benchmark pays for
        (`search_correct_reward=0.2`). Serving the options locally would delete a scored step
        and quietly inflate every downstream number. Drive `session()` instead.
        """
        raise UserBenchNeedsGymLoop(
            f"userbench/{tid}: retrieval is the env's `[search]` action, which is scored by an "
            "LLM judge inside TravelEnv; there is no free local corpus to search. Use "
            "UserBenchSuite.session(task_id) and send `[search] <query>`."
        )

    def actuator(self, tid: TaskId) -> Any:
        """REFUSES, and does not return None.

        None is what the TaskSuite protocol permits for a suite with no stateful world, and
        UserBench has one -- `[answer]` mutates `env.state_list` and is the only thing that
        pays the 1.0. A None here would let a flat `pi run --suite userbench` proceed and
        emit a complete, plausible, all-zero results file, which is precisely the failure
        this whole adapter is built around. Raising means the refusal does not depend on
        which of `retriever()` and `actuator()` the caller happened to reach first.
        """
        raise UserBenchNeedsGymLoop(
            f"userbench/{tid}: actuation is the env's `[answer] <ID>` action against "
            "TravelEnv's own state list. Use UserBenchSuite.session(task_id)."
        )

    # ------------------------------------------------------------------ userbench specifics

    def env_of(self, tid: TaskId) -> str:
        """Which of travel22/33/44 a task came from. Reported, and used to pick its data file."""
        return self._env_of[str(tid)]

    def template_id(self, tid: TaskId) -> str:
        """`<env>:<sorted aspect set>` -- e.g. `travel22:hotel+restaurant`.

        WHY THIS FIELD. UserBench's 255 test tasks are not 255 independent draws: a scenario
        key is `<aspect>:<n>-<serial>|<aspect>:<n>-<serial>`, and two tasks over the same
        aspects share the option schemas, the legal search arguments and the failure modes.
        Resampling tasks rather than aspect-sets understates the SE for exactly the reason it
        does on tau2.

        DERIVED FROM THE KEY, WHICH IS PUBLIC -- the aspect names are in the search space the
        agent is told about, so unlike tau2's `required_documents`-derived template this one
        touches no gold at all. Measured over the 255 test tasks at PINNED_COMMIT: 27
        clusters, largest 18 (`travel22:hotel+restaurant`), median 9, zero singletons.

        The env is part of the key on purpose. `travel22:hotel+restaurant` and
        `travel33:hotel+restaurant` differ in how many preferences per aspect are hidden,
        which is the difficulty axis of the whole benchmark; pooling them would merge the
        easy and hard versions of the same scenario into one cluster. Env-free, the same 255
        tasks fall into 9 clusters instead of 27.

        Note this never reaches the split: `split_of` short-circuits on EVAL_ONLY_SUITES
        before it hashes anything. It is here for the confidence intervals.
        """
        key = str(tid)
        aspects = sorted({part.split(":")[0] for part in key.split("|") if part})
        return f"{self.env_of(key)}:" + "+".join(aspects)

    def session(
        self,
        tid: TaskId,
        *,
        counter: SimulatorFailureCounter | None = None,
        pin: SimulatorPin | None = None,
    ) -> UserBenchSession:
        """A live episode for one task, with the simulator failure counter armed.

        Returns an UNENTERED context manager. The env is built inside `__enter__`, after the
        endpoint is pointed and the guard is installed, so no `TravelEnv` ever exists whose
        simulator calls are unwatched -- see `session.UserBenchSession`.

        THE PIN IS RESOLVED HERE, AND EAGERLY. `pin` defaults to
        `SimulatorPin.from_env()`, which raises `SimulatorNotPinned` when
        `PI_MODEL_USERSIM` (or the key that goes with the endpoint) is unset. Resolving it
        before the env is built means the refusal costs nothing and names the missing
        variable; leaving it to upstream means running the user simulator on
        `TravelGymConfig`'s own default -- `gpt-4o`, on whatever `OPENAI_API_KEY` happens to
        be -- which is a different simulator from every other suite in the paper and is
        recorded nowhere. See `simulator.SimulatorPin`.

        `config.data_mode = "single"` re-reads all eight data files through
        `travelgym.env.task_data.get_task_by_id`, once per session (~1.4 s measured at
        PINNED_COMMIT with the files in page cache). It also means
        `config.wrong_choice_number` / `noise_choice_number` are inert, because
        `get_task_by_id` calls `load_tasks()` with no config: measured at PINNED_COMMIT the
        shipped pools are already exactly 3 correct / 10 wrong / 5 noise per aspect, i.e.
        the defaults those two fields would have applied, so the inertness changes nothing
        here -- but a reader who set them to 2 and 1 and believed it would be wrong.
        """
        ok, why = available(self.root)
        if not ok:
            raise UserBenchUnavailable(why)
        key = str(tid)
        if key not in self._env_of:
            raise KeyError(f"userbench: no {SPLIT} task {key!r}")
        pin = pin if pin is not None else SimulatorPin.from_env()

        def factory() -> Any:
            import travelgym

            config = travelgym.get_default_config()
            config.data_mode = "single"
            config.data_source = key
            config.max_steps = self.max_steps
            config.one_choice_per_aspect = self.one_choice_per_aspect
            config.seed = self.seed
            config.timeout = self.simulator_timeout_s
            config.max_tokens = self.simulator_max_tokens
            # model_name and api_key, which upstream would otherwise default to gpt-4o and
            # $OPENAI_API_KEY. The endpoint half of the pin cannot be set here -- upstream
            # builds `OpenAI(api_key=...)` with no base_url -- and is applied by the session.
            pin.apply(config)
            return travelgym.TravelEnv(config)

        return UserBenchSession(
            env_factory=factory,
            counter=counter if counter is not None else SimulatorFailureCounter(),
            pin=pin,
            label=f"userbench/{key}",
        )

    # ------------------------------------------------------------------------------ private

    def _scenario_record(self, key: str) -> dict[str, Any]:
        """The raw scenario dict. GOLD-BEARING: read only by `view()` through
        `scenario_public`, and by nothing else in this package."""
        if key not in self._env_of:
            raise KeyError(f"userbench: no {SPLIT} task {key!r}")
        tag = env_tag(self._env_of[key])
        if tag not in self._scenarios:
            assert self.root is not None  # set in __post_init__
            self._scenarios[tag] = load_scenarios(self.root, tag)
        record = self._scenarios[tag].get(key)
        if record is None:
            raise UserBenchUnavailable(
                f"scenario {key!r} is in the {SPLIT} split index but not in "
                f"{data_file(self.root, tag) if self.root else tag}; the checkout is "
                f"inconsistent with itself and should be re-checked out at {PINNED_COMMIT}"
            )
        return record
