"""The failure counter. Without it, a UserBench run cannot be trusted to be a run at all.

THE DEFECT, EXACTLY
    UserBench's user simulator is an LLM call. `travelgym/env/prompts.py:model_call` makes
    it, and on failure it does not raise -- it prints, retries twice more, and RETURNS
    `None`. It also returns `None` without retrying at all when the model's text does not
    parse as JSON, because upstream's `parse_output_as_json` returns None rather than
    raising.

    Every caller of `model_call` sits inside a `try` that swallows the consequence:

      * `evaluate_action`'s `[action]` branch asserts on the `None`, catches its own
        AssertionError, sets `judgment_type = "1"` and `reward = 0.0`, and returns the canned
        string "I'm sorry, I'm not sure how to respond to your latest utterance right now."
      * `evaluate_action`'s `[search]` branch catches and returns "Currently the searching
        backend is experiencing some issues. Please try again later." with reward 0.0.
      * `eval.py:rollout` wraps the whole episode in `except Exception` and returns
        `rewards if len(rewards) > 0 else [0]`.

    So a run with a wrong API key, an exhausted quota, a rate limit, or a model that stopped
    emitting JSON produces a COMPLETE, WELL-FORMED, ENTIRELY-ZERO results file. Every task
    present, every reward 0.0, no traceback, no non-zero exit code. Under CONTRIBUTING.md rule 1
    that file is indistinguishable from a real measurement of a policy that never scored, and
    there is nothing in it that a reader could use to tell the difference.

WHY THE FEEDBACK STRING CANNOT BE USED TO DETECT THIS
    It is the same string either way. `evaluate_action` raises `Exception("Simulate a system
    error")` DELIBERATELY on every `search_failure_interval`-th search (default: every 5th),
    and that deliberate error lands in the same `except` and produces the same "the searching
    backend is experiencing some issues" text as a real API failure. A detector that pattern-
    matched the feedback would either miss real failures or flag the env's own designed
    behaviour as broken. The only place the two are distinguishable is the return value of
    `model_call` itself, which is why the counter goes there and nowhere else.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO
    It wraps `model_call` (and `async_model_call`, so the guard is not silently inert on the
    `step_async` path) and OBSERVES the return value. It returns that value UNCHANGED. It
    must: an interposition that altered a response, retried, or substituted a default would
    make our rollouts something other than upstream's benchmark, and the whole reason this
    suite is in the paper is that its authors, not we, defined what it measures.

    It does not raise from inside the wrapper either, because it cannot: the caller's
    `except Exception` would swallow that raise exactly as it swallows every other. The
    count is taken where the truth is, and the refusal is made one layer out, by
    `UserBenchSession.step`, after the env has returned -- see `raise_if_any`.

THE RULE THIS ENCODES
    A turn whose simulator call failed has no reward. Not a zero reward: no reward. The run
    is refused, loudly, at the turn it happened, rather than completed and averaged.
"""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, MutableSequence, Sequence

# Every place a UserBench user-simulator call is actually made, as (module path, attribute).
# BOTH entries are load-bearing. `TravelEnv.step` uses the sync one and `TravelEnv.step_async`
# the async one; a guard that patched only the first would report a clean run for every
# async episode, which is the exact shape of failure this module exists to prevent.
SIMULATOR_ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("travelgym.env.prompts", "model_call"),
    ("travelgym.env.prompt_async", "async_model_call"),
)


# --------------------------------------------------------------------------------- the pin
#
# WHOSE MODEL ANSWERS AS THE USER, AND WHY IT CANNOT BE LEFT TO UPSTREAM
#     `TravelGymConfig.model_name` defaults to "gpt-4o" and `api_key` defaults to
#     `os.getenv("OPENAI_API_KEY")`. An adapter that sets neither runs the user simulator on
#     whatever those two happen to name, which is a different simulator from the one every
#     other suite in this project uses and is not recorded anywhere. Against this project's
#     own proxy it is a 403 that `model_call` swallows into `None`; against an endpoint that
#     does serve gpt-4o it is a silently different measurement. The first is loud and dead,
#     the second is quiet and wrong, and only the second is dangerous.
#
#     So the pin is REQUIRED and never defaulted -- the same rule, for the same reason, as
#     `MeteredClient.model_for`: "Defaulting here would let two arms silently run on
#     different models."

# TravelGymConfig's default at PINNED_COMMIT. Transcribed so that "we are not using
# upstream's default" is a checkable statement rather than a claim, and so an upstream change
# to it is a visible diff -- `test_upstream_still_defaults_to_the_model_we_refuse_to_use`
# checks this string against the real config.
UPSTREAM_DEFAULT_MODEL = "gpt-4o"

# The user-simulator ROLE PIN, shared with every other suite on purpose. UserBench's simulated
# user and ours are the same role, and giving this suite a private variable would let the two
# drift apart without anything saying so.
SIM_MODEL_ENV = "PI_MODEL_USERSIM"

# The proxy pair, and the direct pair. They are resolved TOGETHER: an OpenAI key against a
# proxy URL (or the reverse) is a 401, which upstream turns into `None`, which this module
# reports as a simulator failure -- a true statement that points at the wrong fix.
PROXY_BASE_URL_ENV, PROXY_KEY_ENV = "LITELLM_BASE_URL", "LITELLM_API_KEY"
DIRECT_BASE_URL_ENV, DIRECT_KEY_ENV = "OPENAI_BASE_URL", "OPENAI_API_KEY"

# litellm's routing prefix for "an OpenAI-compatible endpoint". It selects a HANDLER; it is
# not part of the model id and never reaches the wire. See `endpoint_model`.
_LITELLM_OPENAI_PREFIX = "openai/"


class SimulatorNotPinned(RuntimeError):
    """No usable (model, key, endpoint) for the user simulator.

    Separate from `SimulatorCallFailed`: that one means a call was made and came back
    unusable, this one means no call should be made at all. The fixes are different and so
    are the messages.
    """


def endpoint_model(spec: str) -> str:
    """A litellm model spec, reduced to the id the endpoint itself will accept.

    `openai/aws/gpt-oss-120b` means "the OpenAI-compatible handler, model aws/gpt-oss-120b";
    litellm strips the prefix before the request goes out. travelgym does not go through
    litellm -- `prompts.model_call` builds a bare `OpenAI(...)` and passes `model_name`
    through verbatim -- so the prefix has to come off here.

    EXACTLY ONE PREFIX, AND ONLY `openai/`. `azure/gpt-oss-120b` is a real model id on this
    project's proxy, so a rule that dropped any leading segment would ask for
    `gpt-oss-120b`, which is not served, and the 404 would be swallowed into a simulator
    failure with no hint of the cause.
    """
    spec = str(spec).strip()
    return spec[len(_LITELLM_OPENAI_PREFIX) :] if spec.startswith(_LITELLM_OPENAI_PREFIX) else spec


@dataclass(frozen=True, slots=True)
class SimulatorPin:
    """Which model answers as the user, on which endpoint, with which key.

    Constructed from the environment by `from_env` and applied to a `TravelGymConfig` by
    `apply`. `base_url is None` means the OpenAI SDK's own default endpoint -- and it means
    it POSITIVELY: `openai_endpoint` then REMOVES any inherited `OPENAI_BASE_URL`, because a
    pin that does not fully determine the endpoint is not a pin.
    """

    model: str
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SimulatorPin":
        env = os.environ if env is None else env
        spec = str(env.get(SIM_MODEL_ENV, "")).strip()
        if not spec:
            raise SimulatorNotPinned(
                f"the UserBench user simulator has no model pin: set {SIM_MODEL_ENV} (the "
                "same role pin every other suite uses, e.g. "
                f"{SIM_MODEL_ENV}=openai/aws/gpt-oss-120b). Upstream's default is "
                f"{UPSTREAM_DEFAULT_MODEL!r}, and defaulting to it would run this suite's "
                "simulated user on a different model from the rest of the paper without "
                "recording that anywhere."
            )

        proxy = str(env.get(PROXY_BASE_URL_ENV, "")).strip()
        if proxy:
            base_url, key_var = _api_root(proxy), PROXY_KEY_ENV
        else:
            direct = str(env.get(DIRECT_BASE_URL_ENV, "")).strip()
            base_url, key_var = (_api_root(direct) if direct else None), DIRECT_KEY_ENV
        api_key = str(env.get(key_var, "")).strip()
        if not api_key:
            raise SimulatorNotPinned(
                f"{SIM_MODEL_ENV} is {spec!r} and the endpoint is "
                f"{base_url or 'the OpenAI default'}, but {key_var} is unset. The key and the "
                "endpoint are resolved together because a key from one paired with a URL "
                "from the other is a 401, and upstream turns a 401 into a simulator failure "
                "-- a true report that points at the wrong fix."
            )
        return cls(model=endpoint_model(spec), api_key=api_key, base_url=base_url)

    def apply(self, config: Any) -> Any:
        """Write this pin onto a `TravelGymConfig`. Returns it, so a factory can chain.

        BOTH fields, always. Leaving either at upstream's default is the bug this class
        exists to close, and a config that arrives holding `gpt-4o` must not leave holding
        it.
        """
        config.model_name = self.model
        config.api_key = self.api_key
        return config

    def as_dict(self) -> dict[str, str]:
        """For the run record. Rule 1: a UserBench number has to name the user simulator it
        was produced against. The key is not in it, and no field name contains "key", because
        this is written to disk."""
        return {"simulator_model": self.model, "simulator_base_url": self.base_url or ""}


def _api_root(url: str) -> str:
    """A base URL, ending at the API root the OpenAI SDK expects.

    `LITELLM_BASE_URL` is stored without a path because `litellm.completion(api_base=...)`
    appends one itself. The OpenAI SDK treats `base_url` as the root and appends only
    `/chat/completions`, so the bare host would POST one path segment short. Idempotent, so a
    URL that already names `/v1` is left alone.
    """
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


@contextmanager
def openai_endpoint(pin: SimulatorPin | None) -> Iterator[None]:
    """Point upstream's bare `OpenAI(api_key=...)` at `pin`'s endpoint for this block.

    `prompts.model_call` and `prompt_async.async_model_call` both construct their client with
    an api_key and NOTHING ELSE, so the endpoint can only be chosen through
    `OPENAI_BASE_URL`. This sets it for the duration of a session and restores whatever was
    there before -- including restoring "absent", which is why the removal is done by key
    rather than by assigning "".

    `pin is None` is the offline shape and touches nothing: a session driving a stub has no
    endpoint to name and must not delete a variable it does not manage.
    """
    if pin is None:
        yield
        return
    had = DIRECT_BASE_URL_ENV in os.environ
    before = os.environ.get(DIRECT_BASE_URL_ENV)
    if pin.base_url:
        os.environ[DIRECT_BASE_URL_ENV] = pin.base_url
    else:
        os.environ.pop(DIRECT_BASE_URL_ENV, None)
    try:
        yield
    finally:
        if had:
            os.environ[DIRECT_BASE_URL_ENV] = before or ""
        else:
            os.environ.pop(DIRECT_BASE_URL_ENV, None)


class SimulatorCallFailed(RuntimeError):
    """A user-simulator call did not return a usable response, so this turn has no reward.

    Raised INSTEAD OF returning the zero that upstream would have recorded. The distinction
    is the whole point: a zero is a measurement and this is the absence of one.
    """


@dataclass
class SimulatorFailureCounter:
    """Counts user-simulator calls and the ones that came back unusable.

    Three counters rather than one because they have three different meanings and only two
    of them are faults:

      calls      -- every interception. Reported so that "0 failures" can be distinguished
                    from "the guard was never armed", which would otherwise look identical
                    in a summary and is the classic way a monitor rots into decoration.
      failures   -- `model_call` returned None: three API attempts exhausted, or the model's
                    text did not parse as JSON on the first attempt. Definitively unusable.
      malformed  -- it returned something that is not a Mapping (upstream's JSON extractor
                    happily returns a list or a scalar for `[1,2]` or `3`). Every downstream
                    consumer does `"type" in judgment` or `judgment["response"]`, so this
                    lands in the same swallowing `except` and produces the same silent zero.
    """

    calls: int = 0
    failures: int = 0
    malformed: int = 0
    # What the failures looked like, capped: an error report that dumps 20 identical lines
    # is an error report people stop reading.
    samples: MutableSequence[str] = field(default_factory=list)
    sample_cap: int = 5

    @property
    def bad(self) -> int:
        return self.failures + self.malformed

    @property
    def clean(self) -> bool:
        return self.bad == 0

    def observe(self, result: Any, *, label: str = "model_call") -> Any:
        """Record what one simulator call returned, and hand it back untouched."""
        self.calls += 1
        if result is None:
            self.failures += 1
            self._sample(f"{label} returned None (retries exhausted, or output was not JSON)")
        elif not isinstance(result, Mapping):
            self.malformed += 1
            self._sample(f"{label} returned {type(result).__name__}, not a JSON object")
        return result

    def _sample(self, text: str) -> None:
        if len(self.samples) < self.sample_cap:
            self.samples.append(text)

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """`fn` with an observer around its return value. Async-preserving.

        `iscoroutinefunction` rather than a try/await dance: wrapping an async function with
        a sync wrapper returns a coroutine object, which is not None and IS not a Mapping, so
        the counter would score every single async call as malformed and refuse every run.
        A guard that always fires is as useless as one that never does.
        """
        label = getattr(fn, "__name__", "model_call")
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapped(*a: Any, **kw: Any) -> Any:
                return self.observe(await fn(*a, **kw), label=label)

            return awrapped

        @functools.wraps(fn)
        def wrapped(*a: Any, **kw: Any) -> Any:
            return self.observe(fn(*a, **kw), label=label)

        return wrapped

    def raise_if_any(self, *, where: str) -> None:
        """Refuse to let a turn with a failed simulator call be scored.

        Called by `UserBenchSession.step` after `env.step` has returned -- i.e. after
        upstream has already produced its plausible zero, and before anything records it.

        The counter is CUMULATIVE over the episode, so once a turn has failed every later
        turn of the same session raises too. That is deliberate and not a leaked flag: the
        simulated user's state is the conversation history, and a turn it never answered has
        already corrupted every turn after it. There is no partial credit to salvage.
        """
        if self.clean:
            return
        raise SimulatorCallFailed(
            f"{where}: {self.bad} of {self.calls} user-simulator call(s) came back unusable "
            f"({self.failures} returned None, {self.malformed} were not JSON objects). "
            f"UserBench scores such a turn as reward 0.0 with a plausible canned response, so "
            f"a completed run would be indistinguishable from a policy that simply failed. "
            f"Refusing. Observed: {'; '.join(self.samples) or 'no detail captured'}. "
            f"Check the simulator model pin and its credentials, then re-run: a UserBench "
            f"result is only admissible if this counter is zero."
        )

    def as_dict(self) -> dict[str, int]:
        """For the run record. Recorded ALWAYS, including on a clean run, because
        `simulator_calls: 0` is how a reader learns the guard never ran."""
        return {
            "simulator_calls": self.calls,
            "simulator_failures": self.failures,
            "simulator_malformed": self.malformed,
        }


def travelgym_entry_points() -> tuple[tuple[Any, str], ...]:
    """Resolve `SIMULATOR_ENTRY_POINTS` to live module objects.

    Imported here rather than at module scope so that everything above is testable with
    `travelgym` absent -- which is the state of every machine that has not cloned a 200 MB
    benchmark, CI included.
    """
    import importlib

    resolved: list[tuple[Any, str]] = []
    for module_path, attr in SIMULATOR_ENTRY_POINTS:
        module = importlib.import_module(module_path)
        if not hasattr(module, attr):
            raise SimulatorCallFailed(
                f"{module_path}.{attr} does not exist, so the user-simulator call cannot be "
                "counted. Upstream has been refactored; re-derive SIMULATOR_ENTRY_POINTS "
                "against the checkout before running anything, because an unarmed guard "
                "produces an all-zero result file that looks exactly like a real one."
            )
        resolved.append((module, attr))
    return tuple(resolved)


@contextmanager
def counted_simulator(
    targets: Sequence[tuple[Any, str]] | None = None,
    *,
    counter: SimulatorFailureCounter | None = None,
) -> Iterator[SimulatorFailureCounter]:
    """Arm the counter over `targets` for the duration of the block.

    `targets` is a sequence of (module, attribute) pairs and defaults to travelgym's two
    real entry points. It is a parameter rather than a hard-coded import so this can be
    exercised offline against a stub module -- the tests in tests/test_adapters_userbench.py
    do exactly that, which is what lets the guard be checked on a machine with no checkout,
    no key and no network.

    Patching the module ATTRIBUTE is correct and not a shortcut: upstream's
    `generate_judge_search`, `generate_judge_response` and `evaluate_action` all reach
    `model_call` through a module-global lookup at call time, so rebinding the global
    intercepts all three. A wrapper installed on an imported alias would miss them.

    Restores the originals in `finally`, including when the body raises. A guard that leaked
    its patch would make the next test in the same process observe a counter it never armed.
    """
    counter = counter if counter is not None else SimulatorFailureCounter()
    resolved = tuple(targets) if targets is not None else travelgym_entry_points()
    originals = [(module, attr, getattr(module, attr)) for module, attr in resolved]
    try:
        for module, attr, original in originals:
            setattr(module, attr, counter.wrap(original))
        yield counter
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)
