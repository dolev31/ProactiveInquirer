"""UserBench adapter: the simulator failure counter, the observation wall, the refusals.

EVERYTHING HERE RUNS OFFLINE. No `travelgym`, no UserBench checkout, no API key, no network.
That is not a convenience -- it is the only way the guard in `pinq_adapters.userbench.
simulator` can be checked at all, because the failure it prevents happens exactly when the
provider is unreachable, and a test that needed a reachable provider could never reproduce
it.

THE STUB BELOW IS THE WHOLE TEST FIXTURE, and it is a faithful transcription of upstream's
control flow at commit 80506d2, not a mock of what we wish it did:

  * `travelgym/env/prompts.py:model_call` returns `None` after three failed attempts, and
    also returns `None` -- without retrying -- when the model's text does not parse as JSON.
  * `evaluate_action`'s `[action]` branch asserts on that None, catches its own
    AssertionError, and returns a canned apology with reward 0.0.
  * `evaluate_action`'s `[answer]` branch never calls the model at all: scoring is pure set
    membership against `state_list["remaining_best_options"]` / `remaining_correct_options`,
    so a genuinely wrong answer ALSO scores 0.0.

The second and third bullets are the same observable outcome -- (canned-ish string, 0.0) --
from two completely different causes, and
`test_a_failed_simulator_call_is_indistinguishable_from_a_real_zero` pins that. Everything
else in this file exists because of it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import time
import types

import pytest

from pinq.view import LeakageError, make_view
from pinq_adapters.userbench import (
    ACTION_PREFIXES,
    ENV_SEED,
    ENVS,
    GOLD_SURFACE_KEYS,
    MAX_STEPS,
    N_TEST_BY_ENV,
    N_TEST_TASKS,
    ONE_CHOICE_PER_ASPECT,
    PINNED_COMMIT,
    RAW_INFO_KEYS,
    RAW_OBSERVATION_KEYS,
    EpisodeAlreadyOver,
    ObservationShapeChanged,
    ObservationView,
    SimulatorCallFailed,
    SimulatorFailureCounter,
    UserBenchNeedsGymLoop,
    UserBenchSession,
    UserBenchSuite,
    UserBenchUnavailable,
    available,
    counted_simulator,
    env_tag,
    make_metrics,
    make_observation,
    scenario_public,
    userbench_root,
)
from pinq_adapters.userbench.simulator import (
    SIMULATOR_ENTRY_POINTS,
    UPSTREAM_DEFAULT_MODEL,
    SimulatorNotPinned,
    SimulatorPin,
    endpoint_model,
)

# The exact strings upstream returns when it swallows a failure, transcribed from
# travelgym/env/prompts.py at PINNED_COMMIT. Written out so that a change to upstream's
# fallback text is a visible diff here rather than a silent divergence of the stub.
CANNED_ACTION_FAILURE = (
    "I'm sorry, I'm not sure how to respond to your latest utterance right now. Please try again."
)
CANNED_SEARCH_FAILURE = (
    "Currently the searching backend is experiencing some issues. Please try again later."
)


# --------------------------------------------------------------------------------- the stub


def stub_prompts_module(responses):
    """A stand-in for `travelgym.env.prompts` whose `model_call` yields `responses` in order.

    A MODULE OBJECT, and the env below reaches `model_call` THROUGH IT rather than through a
    captured reference. That is how upstream does it -- `evaluate_action`,
    `generate_judge_search` and `generate_judge_response` all resolve the module global at
    call time -- and it is the only arrangement under which `counted_simulator`'s
    monkeypatching of a module attribute is a real test of the real mechanism.
    """
    queue = list(responses)

    def model_call(system_prompt, user_prompt, model_config):
        return queue.pop(0) if queue else None

    return types.SimpleNamespace(model_call=model_call)


class StubTravelEnv:
    """TravelEnv's swallow, transcribed. Holds gold in the same places upstream does.

    `step` returns the same five-tuple, with `observation` carrying `task_description` (the
    full scenario prose) and the preference counters, exactly as upstream's does.
    """

    def __init__(self, prompts, *, best_ids=("H1",), scenario="SCENARIO: user wants a pool."):
        self._prompts = prompts
        self.episode_complete = False
        self.step_count = 0
        self.closed = False
        # The gold that lives on the env object the driver is holding. Footgun 2.
        self.state_list = {
            "remaining_best_options": list(best_ids),
            "remaining_correct_options": list(best_ids),
            "active_elicited_preferences": 0,
            "passive_elicited_preferences": 0,
        }
        self._scenario = scenario

    def _obs(self, feedback, reward, *, with_counters=True):
        """Upstream's observation.

        `with_counters=False` is the `[finish]` and `reset()` shape, and it is not a
        simplification: `travel_env.py` builds THREE observation dicts and only the one on
        the `step()` success path carries `active_elicited_preferences` /
        `passive_elicited_preferences`. The other two omit both keys entirely, which is why
        `make_metrics` cannot read them out of the observation alone.
        """
        obs = {
            "task_description": self._scenario,  # GOLD: every preference, in prose
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_preferences": 4,
            "remaining_preferences": 3,  # GOLD: how many are left to find
            "elicitation_ratio": 0.25,
            "last_reward": reward,
        }
        if with_counters:
            obs["active_elicited_preferences"] = self.state_list["active_elicited_preferences"]
            obs["passive_elicited_preferences"] = self.state_list["passive_elicited_preferences"]
        return obs

    def _info(self):
        return {
            "task_id": "t",
            # GOLD: a formatted list of exactly what the agent is meant to discover
            "preferences_summary": ["hotel-amenities: prefers a rooftop pool..."],
            "action_history": [],
            "conversation_history": [],
            "elicited_preferences": [],
            "newly_elicited_preferences": [],
            "total_reward": 0.0,
        }

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.episode_complete = False
        return (
            self._obs("Let's start the conversation!", 0.0, with_counters=False),
            self._info(),
        )

    def step(self, action):
        self.step_count += 1
        if action.startswith("[action]"):
            # Upstream: judge, assert, swallow, canned apology, reward 0.0.
            judgment = self._prompts.model_call("sys", "user", {})
            try:
                assert judgment is not None and "type" in judgment
                self.state_list["active_elicited_preferences"] += 1
                return (
                    self._obs("Sure -- I'd like a rooftop pool.", 0.2),
                    0.2,
                    False,
                    False,
                    self._info(),
                )
            except Exception:
                return self._obs(CANNED_ACTION_FAILURE, 0.0), 0.0, False, False, self._info()
        if action.startswith("[answer]"):
            # NO MODEL CALL AT ALL. Pure set membership -- this is the honest zero.
            oid = action[len("[answer]") :].strip()
            if oid in self.state_list["remaining_best_options"]:
                self.state_list["remaining_best_options"].remove(oid)
                self.episode_complete = True
                return (
                    self._obs("Your chosen options contain the best option!", 1.0),
                    1.0,
                    True,
                    False,
                    self._info(),
                )
            return (
                self._obs(
                    "Your chosen options do not contain any of the best or correct options.", 0.0
                ),
                0.0,
                False,
                False,
                self._info(),
            )
        if action.startswith("[finish]"):
            self.episode_complete = True
            # NO COUNTERS in this observation -- see `_obs`. Upstream's `[finish]` branch
            # returns before the counters are ever read.
            return (
                self._obs("Session ended.", 0.0, with_counters=False),
                0.0,
                True,
                False,
                self._info(),
            )
        # [search]: judged by the model, and the swallow is shared with the env's own
        # DELIBERATE every-Nth-search failure, which is the point of
        # test_a_search_failure_string_cannot_tell_the_two_causes_apart.
        judgment = self._prompts.model_call("sys", "user", {})
        if judgment is None or "alignment_judgement" not in judgment:
            return self._obs(CANNED_SEARCH_FAILURE, 0.0), 0.0, False, False, self._info()
        return (
            self._obs("Here are all the options for <hotel>: ...", 0.2),
            0.2,
            False,
            False,
            self._info(),
        )

    def close(self):
        self.closed = True


def session_over(responses, *, pin=None, **env_kw):
    """A session whose guard is armed over the stub module, ready to enter."""
    prompts = stub_prompts_module(responses)
    return UserBenchSession(
        env_factory=lambda: StubTravelEnv(prompts, **env_kw),
        targets=((prompts, "model_call"),),
        counter=SimulatorFailureCounter(),
        pin=pin,
        label="userbench/stub",
    )


# ----------------------------------------------------- (a0) the simulator model is pinned
#
# UPSTREAM'S DEFAULT IS `gpt-4o` AND NOTHING IN THE FIRST DRAFT OF THIS ADAPTER CHANGED IT.
# `UserBenchSuite.session()` set `data_mode`, `data_source`, `max_steps`,
# `one_choice_per_aspect` and `seed`, and left `model_name` and `api_key` at
# TravelGymConfig's defaults -- so every episode asked whatever endpoint `OPENAI_BASE_URL`
# happened to name for a model this project does not run, on a key it does not own. Against
# the project's own proxy that is a 403 on every call, which `model_call` swallows into
# `None`; the failure counter then refuses the run. The refusal is correct and the run is
# still dead, and on an endpoint that DOES serve gpt-4o it would not have refused at all --
# it would have quietly produced numbers for a different user simulator than every other
# suite in the paper.


def test_the_simulator_model_must_be_pinned_and_is_never_defaulted():
    """The same rule as `MeteredClient.model_for`: defaulting a pin lets two arms silently
    run on different models. Here it would silently run on somebody else's."""
    with pytest.raises(SimulatorNotPinned, match="PI_MODEL_USERSIM"):
        SimulatorPin.from_env({})


def test_upstreams_default_model_is_overwritten_rather_than_inherited():
    """`apply` must WRITE both fields. A config that arrives holding upstream's defaults and
    leaves holding them is the whole bug."""
    config = types.SimpleNamespace(model_name=UPSTREAM_DEFAULT_MODEL, api_key=None)
    pin = SimulatorPin(model="aws/gpt-oss-120b", api_key="sk-x", base_url="https://p/v1")
    pin.apply(config)
    assert config.model_name == "aws/gpt-oss-120b" != UPSTREAM_DEFAULT_MODEL
    assert config.api_key == "sk-x"


def test_the_litellm_provider_prefix_is_stripped_because_the_endpoint_never_sees_it():
    """`openai/aws/gpt-oss-120b` is a LITELLM routing spec: `openai/` selects the
    OpenAI-compatible handler and only the remainder is put on the wire. travelgym builds a
    bare `OpenAI(...)` and puts the string on the wire verbatim, so the prefix has to come
    off here or the proxy is asked for a model whose id contains a slash it does not have.

    MEASURED against the proxy's own `/v1/models` at the time of writing: the list contains
    `aws/gpt-oss-120b` and `azure/gpt-oss-120b`, and no id beginning `openai/`.

    Exactly one prefix is stripped, and only `openai/`. `azure/gpt-oss-120b` is a real id on
    that same proxy, so a rule that stripped any leading segment would silently ask for
    `gpt-oss-120b`, which is not served.
    """
    assert endpoint_model("openai/aws/gpt-oss-120b") == "aws/gpt-oss-120b"
    assert endpoint_model("azure/gpt-oss-120b") == "azure/gpt-oss-120b"
    assert endpoint_model("gpt-4o") == "gpt-4o"
    # ... and only the FIRST one, so a provider-shaped model id keeps its own path.
    assert endpoint_model("openai/openai/x") == "openai/x"


def test_the_proxy_url_gets_the_v1_that_litellm_adds_and_the_openai_sdk_does_not():
    """`LITELLM_BASE_URL` is stored without a path because `litellm.completion(api_base=...)`
    appends one. The OpenAI SDK does not: it treats base_url as the API root. Passing the
    bare host through would POST to `/chat/completions` instead of `/v1/chat/completions`."""
    pin = SimulatorPin.from_env(
        {
            "PI_MODEL_USERSIM": "openai/aws/gpt-oss-120b",
            "LITELLM_BASE_URL": "https://proxy.example.com",
            "LITELLM_API_KEY": "sk-proxy",
        }
    )
    assert pin.base_url == "https://proxy.example.com/v1"
    assert pin.api_key == "sk-proxy"
    # Idempotent: a URL that already names the API root is not given a second one.
    assert (
        SimulatorPin.from_env(
            {
                "PI_MODEL_USERSIM": "openai/m",
                "LITELLM_BASE_URL": "https://proxy.example.com/v1/",
                "LITELLM_API_KEY": "k",
            }
        ).base_url
        == "https://proxy.example.com/v1"
    )


def test_the_key_and_the_url_always_come_from_the_same_source():
    """An OpenAI key against a proxy URL, or the reverse, is a 401 -- which `model_call`
    turns into `None`, which reads as a simulator failure rather than as a mismatched pair.
    So the two are resolved together and a missing partner is refused by name."""
    with pytest.raises(SimulatorNotPinned, match="LITELLM_API_KEY"):
        SimulatorPin.from_env(
            {"PI_MODEL_USERSIM": "openai/m", "LITELLM_BASE_URL": "https://proxy.example.com"}
        )
    with pytest.raises(SimulatorNotPinned, match="OPENAI_API_KEY"):
        SimulatorPin.from_env({"PI_MODEL_USERSIM": "gpt-4o-mini"})
    # No proxy: the OpenAI SDK's own default endpoint, and its own key.
    direct = SimulatorPin.from_env({"PI_MODEL_USERSIM": "gpt-4o-mini", "OPENAI_API_KEY": "sk-d"})
    assert direct.base_url is None and direct.api_key == "sk-d"


def test_the_pin_reports_itself_for_the_run_record_and_never_its_key():
    """A UserBench number has to name the user simulator it was produced against -- that is
    rule 1 -- and the record is written to disk, so the key may not be in it."""
    pin = SimulatorPin(model="aws/gpt-oss-120b", api_key="sk-secret", base_url="https://p/v1")
    row = pin.as_dict()
    assert row["simulator_model"] == "aws/gpt-oss-120b"
    assert row["simulator_base_url"] == "https://p/v1"
    assert "sk-secret" not in repr(row)
    assert not any("key" in k for k in row)


def test_the_session_points_upstreams_bare_client_at_the_pinned_endpoint():
    """`prompts.model_call` builds `OpenAI(api_key=...)` with NO base_url, so the endpoint
    can only be set through the environment. The session owns that for its own duration,
    next to the counter it arms, and puts it back afterwards."""
    monkey = dict(os.environ)
    os.environ.pop("OPENAI_BASE_URL", None)
    try:
        pin = SimulatorPin(model="m", api_key="k", base_url="https://proxy.example.com/v1")
        with session_over([], pin=pin):
            assert os.environ["OPENAI_BASE_URL"] == "https://proxy.example.com/v1"
        assert "OPENAI_BASE_URL" not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(monkey)


def test_a_stale_base_url_cannot_redirect_a_pin_that_names_no_proxy():
    """The pin determines the endpoint COMPLETELY. A leftover `OPENAI_BASE_URL` from another
    tool would otherwise send the user simulator somewhere the run record does not name."""
    monkey = dict(os.environ)
    os.environ["OPENAI_BASE_URL"] = "https://somewhere.else/v1"
    try:
        with session_over([], pin=SimulatorPin(model="m", api_key="k", base_url=None)):
            assert "OPENAI_BASE_URL" not in os.environ
        assert os.environ["OPENAI_BASE_URL"] == "https://somewhere.else/v1"
    finally:
        os.environ.clear()
        os.environ.update(monkey)


def test_a_session_with_no_pin_touches_the_environment_at_all():
    """`pin=None` is the offline/stub shape used by every test above, and it must leave the
    environment exactly as it found it rather than deleting a variable it does not manage."""
    monkey = dict(os.environ)
    os.environ["OPENAI_BASE_URL"] = "https://untouched/v1"
    try:
        with session_over([]):
            assert os.environ["OPENAI_BASE_URL"] == "https://untouched/v1"
    finally:
        os.environ.clear()
        os.environ.update(monkey)


# ------------------------------------------------------------- (a) the suite refuses to guess


def test_the_suite_refuses_to_construct_without_userbench_dir(monkeypatch):
    """No checkout, no corpus, no run. A fabricated or empty corpus reads downstream as a
    policy that found nothing, which is a wrong number rather than an error."""
    monkeypatch.delenv("USERBENCH_DIR", raising=False)
    with pytest.raises(UserBenchUnavailable, match="USERBENCH_DIR is unset"):
        UserBenchSuite()


def test_the_refusal_names_the_remedy_and_the_pinned_commit(monkeypatch):
    """A message that says only "unavailable" sends the reader to read this package's source.
    The remedy is three shell lines and they belong in the exception."""
    monkeypatch.delenv("USERBENCH_DIR", raising=False)
    with pytest.raises(UserBenchUnavailable) as exc:
        userbench_root()
    text = str(exc.value)
    assert "git clone" in text
    assert PINNED_COMMIT in text
    assert "docs/DATA.md" in text


def test_an_empty_userbench_dir_is_the_same_as_unset(monkeypatch):
    monkeypatch.setenv("USERBENCH_DIR", "   ")
    with pytest.raises(UserBenchUnavailable, match="USERBENCH_DIR is unset"):
        userbench_root()


def test_pointing_at_the_travelgym_package_instead_of_the_repo_root_is_named(tmp_path):
    """The likeliest wrong answer, given the package and the repo have different names."""
    (tmp_path / "data").mkdir()
    with pytest.raises(UserBenchUnavailable, match="repository ROOT"):
        userbench_root(tmp_path)


def test_available_reports_which_of_the_three_things_is_missing(tmp_path):
    """The checkout, `travelgym` on sys.path and pyarrow are three problems with three fixes;
    one flat "unavailable" would send a reader to fix the wrong one."""
    (tmp_path / "travelgym" / "data").mkdir(parents=True)
    ok, why = available(tmp_path)
    assert not ok
    assert "travelgym_data_" in why  # the data files, named


# ------------------------------------------------- (b) the counter turns a zero into a raise


def test_a_failed_simulator_call_is_indistinguishable_from_a_real_zero():
    """THE DEFECT, PINNED. This test asserts the bug exists, and it must keep passing.

    Both branches produce a plausible feedback string and reward 0.0. One is a policy that
    recommended the wrong hotel; the other is a user simulator that never answered. Nothing
    in the five-tuple separates them, which is why counting has to happen inside `model_call`
    and cannot be done by inspecting the step result.
    """
    prompts = stub_prompts_module([None])
    env = StubTravelEnv(prompts)
    _obs_failed, reward_failed, *_ = env.step("[action] what sort of hotel do you like?")

    env2 = StubTravelEnv(stub_prompts_module([]))
    _obs_wrong, reward_wrong, *_ = env2.step("[answer] H99")

    assert reward_failed == reward_wrong == 0.0
    assert isinstance(_obs_failed["feedback"], str) and _obs_failed["feedback"]
    assert isinstance(_obs_wrong["feedback"], str) and _obs_wrong["feedback"]


def test_a_search_failure_string_cannot_tell_the_two_causes_apart():
    """Upstream raises `Exception("Simulate a system error")` on every Nth search ON PURPOSE
    and routes it into the same `except` as a real API failure, producing the same text. So a
    detector built on the feedback string would either miss real failures or condemn the
    benchmark's own designed behaviour."""
    real_failure = StubTravelEnv(stub_prompts_module([None])).step("[search] hotels in Austin")
    # The env's deliberate error arrives with no model call at all; its text is identical.
    assert real_failure[0]["feedback"] == CANNED_SEARCH_FAILURE
    assert real_failure[1] == 0.0


def test_the_session_raises_instead_of_recording_the_zero():
    """(b) and the core of the package. Same step as above, driven through the session."""
    with session_over([None]) as sess:
        with pytest.raises(SimulatorCallFailed) as exc:
            sess.step("[action] what sort of hotel do you like?")
    text = str(exc.value)
    assert "Refusing" in text
    assert "1 of 1" in text
    assert "reward 0.0" in text
    # Attributed to the right branch. `None` is also not a Mapping, so a counter that had
    # lost its `is None` arm would still refuse -- correctly, but for the wrong stated
    # reason, and the message a reader acts on would name the wrong upstream failure mode.
    assert sess.counter.failures == 1
    assert sess.counter.malformed == 0


def test_a_genuine_zero_is_not_a_simulator_failure():
    """The other half, without which the guard would be a guard against running at all.

    A wrong `[answer]` makes NO simulator call, scores 0.0, and must pass straight through.
    A refusal that fired here would make every imperfect policy look like a broken machine.
    """
    with session_over([]) as sess:
        result = sess.step("[answer] H99")
    assert result.metrics.reward == 0.0
    assert sess.counter.clean
    assert sess.usage() == {
        "simulator_calls": 0,
        "simulator_failures": 0,
        "simulator_malformed": 0,
    }


def test_a_successful_turn_is_counted_and_not_refused():
    with session_over([{"type": "2", "preference_id": "P1"}]) as sess:
        result = sess.step("[action] do you want a pool?")
    assert result.metrics.reward == pytest.approx(0.2)
    assert sess.counter.calls == 1 and sess.counter.clean


def test_an_unparseable_simulator_response_is_counted_too():
    """`parse_output_as_json` returns a list for "[1,2]" without raising, and every consumer
    then does `"type" in judgment` -- False for a list -- and lands in the same swallow."""
    with session_over([[1, 2]]) as sess:
        with pytest.raises(SimulatorCallFailed, match="not JSON objects"):
            sess.step("[action] anything?")


# --------------------------------------------- (c) tests that FAIL if the counter is removed


def test_removing_the_wrapper_makes_the_zero_come_back():
    """RULE 2, DIRECTLY. Runs the SAME session with the guard armed over NOTHING.

    This is what the code looks like the moment someone deletes `counted_simulator` from
    `UserBenchSession.__enter__`, or narrows `SIMULATOR_ENTRY_POINTS` past the entry point
    actually in use: the counter observes no calls, `raise_if_any` has nothing to say, and
    the step returns a complete, plausible, zero-reward result. Asserting that outcome here
    is what makes the three tests above non-tautological -- they cannot be satisfied by an
    `raise_if_any` that raises unconditionally, and they cannot be satisfied by a counter
    that is never wired to anything.
    """
    prompts = stub_prompts_module([None])
    unguarded = UserBenchSession(
        env_factory=lambda: StubTravelEnv(prompts),
        targets=(),  # <- the removal, expressed as an empty target list
        label="userbench/unguarded",
    )
    with unguarded as sess:
        result = sess.step("[action] what sort of hotel do you like?")

    assert result.metrics.reward == 0.0
    assert result.observation.agent_feedback == CANNED_ACTION_FAILURE
    assert sess.counter.calls == 0, "an unarmed counter sees nothing -- and reports clean"
    assert sess.counter.clean

    # And the same session WITH the guard refuses, on identical inputs. If the wrapper were
    # removed from the real code path, this half is what would start failing.
    with session_over([None]) as guarded:
        with pytest.raises(SimulatorCallFailed):
            guarded.step("[action] what sort of hotel do you like?")


def test_a_counter_that_never_records_cannot_satisfy_the_refusal_tests():
    """The other way the guard could rot: `observe` stops recognising a failure.

    Subclassing to blind ONLY the None-detection, leaving the wiring intact, must make the
    refusal disappear -- which is what proves the refusal is driven by the detection rather
    than by the mere presence of a wrapper.
    """

    class BlindCounter(SimulatorFailureCounter):
        def observe(self, result, *, label="model_call"):
            self.calls += 1
            return result

    prompts = stub_prompts_module([None])
    blinded = UserBenchSession(
        env_factory=lambda: StubTravelEnv(prompts),
        targets=((prompts, "model_call"),),
        counter=BlindCounter(),
        label="userbench/blind",
    )
    with blinded as sess:
        result = sess.step("[action] anything?")
    assert sess.counter.calls == 1  # armed, wired, and still silent
    assert result.metrics.reward == 0.0


# -------------------------------------------------------------------------- the guard itself


def test_the_guard_restores_the_entry_point_even_when_the_body_raises():
    """A leaked patch would make the next session in the same process observe a counter it
    never armed -- i.e. exactly the false 'clean' this package exists to prevent."""
    prompts = stub_prompts_module([{"type": "1"}])
    original = prompts.model_call
    with pytest.raises(RuntimeError, match="boom"):
        with counted_simulator(((prompts, "model_call"),)):
            assert prompts.model_call is not original
            raise RuntimeError("boom")
    assert prompts.model_call is original


def test_the_wrapper_returns_the_response_unchanged():
    """OBSERVE, NEVER INTERPOSE. A wrapper that substituted a default, retried, or repaired a
    response would make our rollouts something other than the published benchmark, and the
    only reason this suite is in the paper is that its rules are not ours."""
    payload = {"type": "2", "preference_id": "P3"}
    prompts = stub_prompts_module([payload])
    with counted_simulator(((prompts, "model_call"),)) as counter:
        got = prompts.model_call("s", "u", {})
    assert got is payload
    assert counter.calls == 1 and counter.clean


def test_the_async_entry_point_is_counted_too():
    """`TravelEnv.step_async` goes through `async_model_call`, which fails the same way.

    A sync wrapper around an async function returns a coroutine -- not None, not a Mapping --
    so a guard that did not branch on `iscoroutinefunction` would score EVERY async call as
    malformed and refuse every run. This pins both halves: the failure is seen, and a
    successful async call is not miscounted.
    """

    async def async_model_call(system_prompt, user_prompt, model_config):
        return None

    async def async_ok(system_prompt, user_prompt, model_config):
        return {"type": "1"}

    module = types.SimpleNamespace(async_model_call=async_model_call, ok=async_ok)
    with counted_simulator(((module, "async_model_call"), (module, "ok"))) as counter:
        assert asyncio.run(module.async_model_call("s", "u", {})) is None
        assert asyncio.run(module.ok("s", "u", {})) == {"type": "1"}
    assert counter.calls == 2
    assert counter.failures == 1
    assert counter.malformed == 0


def test_both_real_entry_points_are_declared():
    """If upstream's sync path were the only one declared, every `step_async` episode would
    report a clean run. The names are checked against the constant, not against travelgym,
    so this runs with no checkout."""
    assert ("travelgym.env.prompts", "model_call") in SIMULATOR_ENTRY_POINTS
    assert ("travelgym.env.prompt_async", "async_model_call") in SIMULATOR_ENTRY_POINTS


def test_a_clean_run_still_records_the_counter():
    """`simulator_calls: 0` is how a reader learns the guard never ran. Omitting the key on a
    clean run would make 'no failures' and 'no guard' identical in the results file."""
    counter = SimulatorFailureCounter()
    assert counter.as_dict() == {
        "simulator_calls": 0,
        "simulator_failures": 0,
        "simulator_malformed": 0,
    }


# ------------------------------------------------------------------- the observation wall


def test_the_observation_view_shares_no_field_name_with_the_raw_observation():
    """The type wall, checked. `ObservationView(**observation)` must be a TypeError, so no
    raw key -- `task_description` least of all -- can arrive by splat or by asdict."""
    fields = set(ObservationView.__dataclass_fields__)
    assert fields & RAW_OBSERVATION_KEYS == set()
    assert fields & RAW_INFO_KEYS == set()
    assert fields & GOLD_SURFACE_KEYS == set()


def test_the_scenario_and_the_preference_summary_never_reach_the_view():
    """The concrete leak: `observation["task_description"]` is the full scenario prose and it
    is present on every single step."""
    prompts = stub_prompts_module([])
    env = StubTravelEnv(prompts, scenario="SECRET: the user insists on a rooftop pool.")
    raw, *_ = env.reset()
    assert "SECRET" in raw["task_description"]  # it really is there

    view = make_observation(raw)
    dumped = repr(dataclasses.asdict(view))
    assert "SECRET" not in dumped
    assert "rooftop pool" not in dumped


def test_the_step_result_carries_no_reference_to_the_env_or_its_state_list():
    """Footgun 2: `state_list["remaining_best_options"]` is the answer key and it lives on the
    object the driver holds. Nothing handed back may reach it."""
    with session_over([]) as sess:
        result = sess.step("[answer] H99")
    dumped = repr(dataclasses.asdict(result))
    assert "remaining_best_options" not in dumped
    assert "H1" not in dumped  # the best id the stub env is holding
    assert "preferences_summary" not in dumped


def test_the_preference_counters_are_recorded_and_not_shown():
    """They are derived from the answer key. Recorded, because they are the metric; not
    shown, because "three preferences left" is a countdown the simulated user never gave."""
    with session_over([{"type": "2"}]) as sess:
        result = sess.step("[action] pool?")
    assert result.metrics.remaining_preferences == 3
    assert result.metrics.total_preferences == 4
    assert not hasattr(result.observation, "remaining_preferences")


def test_active_and_passive_elicitation_are_kept_apart():
    """If the agent goes `elicitation_interval` turns without a recognised preference
    question, the simulated user VOLUNTEERS one. Pooling the two would credit the agent for
    the environment giving up on it."""
    with session_over([{"type": "2"}]) as sess:
        result = sess.step("[action] pool?")
    assert result.metrics.active_elicited == 1
    assert result.metrics.passive_elicited == 0
    assert result.metrics.active_elicitation_share == pytest.approx(0.25)


def test_active_elicitation_share_is_zero_for_a_task_with_no_preferences():
    m = make_metrics({"total_preferences": 0}, reward=0.0, terminated=False, truncated=False)
    assert m.active_elicitation_share == 0.0


def test_the_elicitation_counters_survive_the_turn_that_ends_the_episode():
    """THE NUMBER THIS SUITE EXISTS TO REPORT MUST NOT BE ZEROED BY THE LAST TURN.

    `travel_env.py` builds three observation dicts and only ONE of them -- the `step()`
    success path -- carries `active_elicited_preferences` / `passive_elicited_preferences`.
    The `[finish]` branch returns before those keys are ever written, and so does `reset()`.
    Reading them out of the observation with `.get(..., 0)` therefore reports 0 active and 0
    passive for any episode a policy ended with `[finish]` -- which is exactly what a policy
    that has finished its work does. The result is a silent floor of zero on the headline
    elicitation number, on the tasks most likely to have earned a non-zero one.

    The authoritative copy is `env.state_list`, which the session already holds and which
    upstream only ever COPIES into the observation. So the session reads it there.
    """
    with session_over([{"type": "2"}]) as sess:
        turn = sess.step("[action] pool?")
        assert turn.metrics.active_elicited == 1  # the observation carried it
        end = sess.step("[finish]")
    # ... and the observation for THIS turn does not.
    assert end.metrics.active_elicited == 1, "a [finish] must not zero the episode's counters"
    assert end.metrics.passive_elicited == 0
    assert end.metrics.terminated is True


def test_the_counters_are_read_from_the_state_list_without_carrying_it():
    """`make_metrics` may take the state list, but a `StepResult` may still hold no gold.

    `state_list` is the answer key's home -- `remaining_best_options` sits in the same dict.
    Passing it in to recover two integers must not put a reference to it anywhere a driver
    can reach.
    """
    counters = {
        "active_elicited_preferences": 2,
        "passive_elicited_preferences": 1,
        "remaining_best_options": ["H1"],  # the answer key, in the same dict
    }
    m = make_metrics(
        {"total_preferences": 4},
        reward=0.0,
        terminated=False,
        truncated=False,
        counters=counters,
    )
    assert (m.active_elicited, m.passive_elicited) == (2, 1)
    assert "H1" not in repr(dataclasses.asdict(m))


def test_the_observation_still_wins_over_the_state_list():
    """On the step() success path upstream writes both, and what the observation says is what
    upstream reported. The state list is the FALLBACK, not an override -- otherwise a future
    upstream that stopped copying would change the number rather than surface the change."""
    m = make_metrics(
        {"active_elicited_preferences": 5, "passive_elicited_preferences": 0},
        reward=0.0,
        terminated=False,
        truncated=False,
        counters={"active_elicited_preferences": 99, "passive_elicited_preferences": 99},
    )
    assert (m.active_elicited, m.passive_elicited) == (5, 0)


def test_make_observation_raises_when_the_shape_changes():
    """Substituting "" for a missing `feedback` would show the agent nothing every turn and
    blame the policy for the resulting zeros."""
    with pytest.raises(ObservationShapeChanged, match="feedback"):
        make_observation({"step_count": 1, "episode_complete": False})


def test_make_observation_ignores_a_field_added_upstream():
    """A projection, not an allowlist check: the dict is upstream's, so a new key must be
    ignored rather than raise -- otherwise the adapter breaks on an upstream release that
    added a field it does not even read."""
    view = make_observation(
        {"feedback": "hi", "step_count": 2, "episode_complete": False, "brand_new_key": "x"}
    )
    assert view.agent_feedback == "hi" and view.agent_step == 2


# ------------------------------------------------------------------------- the session shell


def test_a_session_used_without_entering_refuses():
    """The guard is armed in `__enter__`, so an env that exists outside one is an env whose
    simulator calls were never counted."""
    sess = session_over([])
    with pytest.raises(RuntimeError, match="context manager"):
        sess.step("[answer] H1")


def test_an_unprefixed_action_is_refused_rather_than_scored():
    """Upstream answers an unprefixed action with "Your response format is wrong" and reward
    0.0. That is a real part of the benchmark for a POLICY, and a harness bug for a DRIVER;
    the two must not look alike in a results file."""
    with session_over([]) as sess:
        with pytest.raises(ValueError, match="must start with one of"):
            sess.step("what sort of hotel do you like?")


def test_every_declared_action_prefix_is_accepted():
    assert set(ACTION_PREFIXES) == {"[search]", "[action]", "[answer]", "[finish]"}
    with session_over([{"alignment_judgement": "True"}]) as sess:
        assert sess.step("[search] hotels").metrics.reward == pytest.approx(0.2)


def test_stepping_a_finished_episode_refuses_by_name():
    with session_over([]) as sess:
        sess.step("[answer] H1")  # the best id: terminates
        with pytest.raises(EpisodeAlreadyOver):
            sess.step("[action] anything else?")


def test_exiting_closes_the_env_and_drops_the_reference():
    prompts = stub_prompts_module([])
    envs = []

    def factory():
        env = StubTravelEnv(prompts)
        envs.append(env)
        return env

    sess = UserBenchSession(env_factory=factory, targets=((prompts, "model_call"),))
    with sess:
        sess.reset()
    assert envs[0].closed is True
    with pytest.raises(RuntimeError, match="context manager"):
        sess.step("[answer] H1")


def test_a_failing_env_factory_still_restores_the_entry_point_and_the_endpoint():
    """Otherwise a checkout with a broken data file would leave the patch installed and the
    next session would report a clean run it never watched.

    BOTH are unwound, and both matter for the same reason. `__enter__` installs the endpoint
    and then the counter, so a factory that raises between them (a missing data file, an
    unknown task id) can leave either behind: a leaked `model_call` wrapper makes the next
    session observe a counter it never armed, and a leaked `OPENAI_BASE_URL` redirects
    whatever runs next in the process to an endpoint nothing recorded.
    """
    prompts = stub_prompts_module([])
    original = prompts.model_call
    saved = dict(os.environ)
    os.environ.pop("OPENAI_BASE_URL", None)

    def factory():
        raise UserBenchUnavailable("no data file")

    try:
        with pytest.raises(UserBenchUnavailable):
            with UserBenchSession(
                env_factory=factory,
                targets=((prompts, "model_call"),),
                pin=SimulatorPin(model="m", api_key="k", base_url="https://proxy.example/v1"),
            ):
                pass
        assert prompts.model_call is original
        assert "OPENAI_BASE_URL" not in os.environ
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ------------------------------------------------------------------------- pure helpers


def test_scenario_public_returns_the_opening_line_and_not_the_scenario():
    """The view-side half of the wall, on a fabricated record so it runs with no checkout."""
    record = {
        "initial_description": "I need a hotel in Austin from the 10th.",
        "scenario": "I need a hotel in Austin and I insist on a rooftop pool and late check-out.",
        "dimensions": ["hotel"],
        "hotel": {
            "best_id": "H4",
            "correct_ids": ["H4", "H9"],
            "preferences": [["hotel", "a", "b", "c"]],
        },
    }
    public = scenario_public(record)
    assert public == {"initial_description": "I need a hotel in Austin from the 10th."}
    assert "rooftop pool" not in repr(public)
    assert "H4" not in repr(public)


def test_scenario_public_raises_when_the_opening_line_is_missing():
    """An empty question would make `make_view` raise anyway; raising HERE says which field
    of which upstream record changed."""
    with pytest.raises(UserBenchUnavailable, match="initial_description"):
        scenario_public({"scenario": "..."})


def test_a_view_built_from_the_public_projection_survives_make_view():
    """The projection's output is exactly what `view()` feeds `make_view`, so the allowlist
    must accept it. A field that leaked in would raise LeakageError, which is the firewall
    working."""
    view = make_view(
        task_id="hotel:2-1|flight:2-2",
        suite_id="userbench",
        question="I need a hotel in Austin from the 10th.",
        instructions="...",
        corpus_id="userbench_travelgym_v1",
        corpus_hash="deadbeef",
        word_cap=60,
    )
    assert view.question.startswith("I need a hotel")
    with pytest.raises(LeakageError):
        make_view(
            task_id="t",
            suite_id="userbench",
            question="q",
            instructions="i",
            corpus_id="c",
            corpus_hash="h",
            word_cap=60,
            scenario="the full preference prose",
        )


def test_env_tag_maps_an_env_to_its_data_file():
    assert env_tag("travel22") == "22"
    assert env_tag("travel444") == "444"
    assert env_tag("22") == "22"


def test_the_declared_counts_are_self_consistent():
    """Two numbers written down twice have to agree, or one of them is decoration."""
    assert sum(N_TEST_BY_ENV.values()) == N_TEST_TASKS == 255
    assert set(N_TEST_BY_ENV) == set(ENVS)


def test_template_id_groups_by_env_and_aspect_set():
    """Derived from the scenario KEY, which is public. Checked without a checkout by calling
    the unbound method against a stand-in that knows only which env a task came from."""
    suite = UserBenchSuite.__new__(UserBenchSuite)
    suite._env_of = {
        "hotel:2-1|restaurant:2-2": "travel22",
        "restaurant:2-9|hotel:2-7": "travel22",
        "hotel:3-1|restaurant:3-2": "travel33",
    }
    a = UserBenchSuite.template_id(suite, "hotel:2-1|restaurant:2-2")
    b = UserBenchSuite.template_id(suite, "restaurant:2-9|hotel:2-7")
    c = UserBenchSuite.template_id(suite, "hotel:3-1|restaurant:3-2")
    assert a == b == "travel22:hotel+restaurant"
    # The env is part of the key ON PURPOSE: travel33 hides three preferences per aspect
    # rather than two, which is the difficulty axis of the whole benchmark.
    assert c == "travel33:hotel+restaurant"
    assert c != a


def test_userbench_is_eval_only_and_registered_live():
    """The registration, checked from the adapter's side as well as the registry's."""
    from pi_run.suites import REGISTRY
    from pi_run.worker import SELF_SOURCED
    from pinq.splitting import EVAL_ONLY_SUITES, split_of

    assert "userbench" in EVAL_ONLY_SUITES
    assert split_of("userbench", "hotel:2-1|flight:2-2") == "test"
    assert "userbench" in SELF_SOURCED
    spec = REGISTRY["userbench"]
    assert spec.status == "live"
    assert spec.required_env == ("USERBENCH_DIR",)
    assert not spec.mines_training_data


def test_a_flat_pi_run_is_refused_before_it_starts_rather_than_by_traceback():
    """`pi run --suite userbench` must not begin a sweep it cannot finish.

    Both `retriever()` and `actuator()` refuse, so a flat run cannot silently emit an
    all-zero results file -- that part already worked. What it DID do was print
    `suite=userbench tasks=2 arms=1 seeds=1 units=2`, fork workers, and then die on an
    unhandled `UserBenchNeedsGymLoop` out of `run_unit`, with a `concurrent.futures`
    traceback wrapped around it and a half-created runs directory left behind. A refusal
    that arrives after the header has promised two units is a crash, not a refusal.

    The decision is read from `SuiteSpec.driven_by` -- "gym" -- rather than from a second
    hard-coded suite list that would go stale next to the registry.
    """
    from pi_run.cli import flat_run_refusal

    message = flat_run_refusal("userbench")
    assert message, "a gym-driven suite may not be run flat"
    # It has to say what to do INSTEAD, or the reader's next move is to delete the check.
    assert "UserBenchSuite.session" in message
    assert "userbench" in message
    # AND tau2 IS STILL RUNNABLE. It is orchestrator-driven and reads as the same shape as
    # userbench in every docstring, but `worker.run_unit` dispatches DIALOGUE_SUITES to
    # `stages.tau2_runner.run_tau2_unit` on its first line, and two live grids
    # (tier1_trained, tier2_confirmatory) drive it through `pi run`. Refusing it here on the
    # symmetry of the prose would have taken those down; this line is why the table keys on
    # "has no driver" rather than on "needs a special driver".
    assert flat_run_refusal("tau2") == ""
    assert flat_run_refusal("tau2_retail") == ""
    # ... and an ordinary flat suite is not refused either.
    assert flat_run_refusal("musique") == ""
    assert flat_run_refusal("synth") == ""
    assert flat_run_refusal("no-such-suite") == ""


def test_retriever_and_actuator_refuse_and_name_the_door(monkeypatch):
    """Both refuse. `actuator()` deliberately does not return None: a None would let a flat
    `pi run --suite userbench` proceed and emit a complete, all-zero results file."""
    suite = UserBenchSuite.__new__(UserBenchSuite)
    with pytest.raises(UserBenchNeedsGymLoop, match="session"):
        UserBenchSuite.retriever(suite, "t")
    with pytest.raises(UserBenchNeedsGymLoop, match="session"):
        UserBenchSuite.actuator(suite, "t")


# ------------------------------------------------------------------------------ integration
#
# CREDENTIALS ARE SNAPSHOTTED AT IMPORT, ON PURPOSE. conftest's autouse `_hermetic_env`
# scrubs every `PI_MODEL_*`, `LITELLM_*`, `OPENAI_API_KEY` and `OPENAI_BASE_URL` out of
# `os.environ` before each test, so that a developer with a populated `.env` and CI without
# one run the same suite. That is right, and it means a live test cannot read the pin out of
# the ambient environment at call time. Collection happens before any fixture runs, so the
# snapshot below is taken while the variables are still there -- and the pin is then passed
# EXPLICITLY into `session(pin=...)` rather than smuggled back into `os.environ`, which also
# makes it visible in this file which four variables a live UserBench run actually needs.
_LIVE_ENV = {
    k: os.environ.get(k, "")
    for k in (
        "PI_MODEL_USERSIM",
        "LITELLM_BASE_URL",
        "LITELLM_API_KEY",
        "OPENAI_API_KEY",
        "PI_USERBENCH_LIVE",
    )
}

# AN EXPLICIT OPT-IN, BECAUSE THESE TWO TESTS SPEND MONEY.
#
# `integration` alone is not enough. CONTRIBUTING.md's own pre-flight is a bare `pytest -q`, which
# runs integration tests; the marker's declared meaning is "requires an external service or a
# large download", and every other test carrying it costs nothing. A billed test that hides
# behind a marker meaning "slow" is how a developer with a populated `.env` discovers a
# provider bill they did not ask for -- the same reason this repository puts `--spend-cap` on
# the command line instead of in `.env`. Three episodes are a few tenths of a cent, which is
# exactly the amount that makes people stop noticing.
#
# The free integration tests below -- upstream's `evaluate_action` against a dead stub, the
# real config's defaults, the real checkout's task list -- are deliberately NOT gated on this.
_SPEND_OPT_IN = "PI_USERBENCH_LIVE"

# The three headline environments, one task each. travel22 hides two preferences per aspect,
# travel33 three, travel44 four, and that is the difficulty axis of the benchmark -- running
# three tasks from one env would exercise the machinery and say nothing about the range.
_LIVE_ENVS = ENVS


def _live_pin():
    """The pin, or a skip naming the variable that is missing. Never a default. SPENDS."""
    if str(_LIVE_ENV.get(_SPEND_OPT_IN, "")).strip().lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(
            f"{_SPEND_OPT_IN} is not set. This test talks to a paid user simulator; set "
            f"{_SPEND_OPT_IN}=1 to run it."
        )
    try:
        return SimulatorPin.from_env(_LIVE_ENV)
    except SimulatorNotPinned as exc:
        pytest.skip(str(exc))


def _live_suite():
    pytest.importorskip("travelgym", reason="needs the UserBench checkout installed")
    try:
        return UserBenchSuite()
    except UserBenchUnavailable as exc:
        pytest.skip(str(exc))


# Every public field of an option, per aspect, exactly as upstream prints it in the schema
# block at the top of a successful `[search]`. PUBLIC: the agent is shown this text. Used
# below to build preference questions that are concrete rather than vague, because
# upstream's judge scores a vague one as type "4" and a concrete one as "2" or "3", and a
# policy that only ever produced type "4" would exercise one branch of the simulator.
_ASPECT_FIELDS = {
    "flight": ("layovers", "in-flight amenities"),
    "hotel": ("room type", "hotel amenities"),
    "restaurant": ("cuisine", "price level"),
    "apartment": ("bedroom and bathroom count", "apartment amenities"),
    "rental_car": ("vehicle category", "insurance cover"),
}


def _option_ids(feedback):
    """The option ids upstream printed in a search result. The ONLY place a driver may learn
    them: `env.state_list` holds the answer key and the session never hands it over."""
    return re.findall(r'"id":\s*"([ACFHR]\d+)"', feedback)


def _trivial_episode(sess, question, aspects, echo):
    """Search each aspect, ask two concrete questions per aspect, recommend one option each.

    DELIBERATELY NOT A GOOD POLICY, and it is not trying to be. It exists to drive every
    branch of the env that costs a model call -- the `[search]` judge, the `[action]` judge,
    the preference responder and the proactive-elicitation responder -- so that the machinery
    is exercised end to end. Its score is not a claim about anything.

    It reads only public things: the task key's aspect names, the user's opening line, and
    the option ids upstream printed back at it.
    """
    seen: list[str] = []
    turns: list[tuple[str, float]] = []

    def act(action):
        result = sess.step(action)
        turns.append((action, result.metrics.reward))
        seen.extend(_option_ids(result.observation.agent_feedback))
        echo(
            f"  turn {result.observation.agent_step:>2}  reward={result.metrics.reward:>4.2f}  "
            f"active={result.metrics.active_elicited} passive={result.metrics.passive_elicited}  "
            f"| {action[:70]}"
        )
        echo(f"      -> {result.observation.agent_feedback[:150].replace(chr(10), ' | ')}")
        return result

    for aspect in aspects:
        result = act(f"[search] {aspect.replace('_', ' ')} options. {question}")
        if result.observation.agent_episode_over:
            return turns, seen, result
        for field in _ASPECT_FIELDS.get(aspect, ("price",)):
            result = act(
                f"[action] For the {aspect.replace('_', ' ')}, do you have a particular "
                f"requirement about the {field}?"
            )
            if result.observation.agent_episode_over:
                return turns, seen, result

    for aspect in aspects:
        initial = {"flight": "F", "apartment": "A", "rental_car": "C", "hotel": "H"}.get(
            aspect, "R"
        )
        for oid in seen:
            if oid.startswith(initial):
                result = act(f"[answer] {oid}")
                break
        else:
            continue
        if result.observation.agent_episode_over:
            return turns, seen, result
    return turns, seen, result


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("env_name", _LIVE_ENVS)
def test_a_live_episode_runs_end_to_end(env_name, capsys):
    """ONE REAL EPISODE, against the real env and a real user simulator. COSTS MONEY.

    This is the test the rest of the package was written blind against. Everything above
    proves the guard fires on a stub; only this proves that a `TravelEnv` can be built at
    all, that the pin reaches upstream's client, that a `[search]` is judged, that a
    preference question is answered by a model, and that the counters come back.

    It asserts the MACHINERY, never the score. A trivial policy's reward is not a finding and
    must not become one by being asserted: what is checked is that turns happened, that every
    simulator call came back usable, and that `active`/`passive` elicitation are counted
    separately. Run with `-s` to see the transcript.
    """
    suite = _live_suite()
    pin = _live_pin()
    tid = next(t for t in suite.task_ids() if suite.env_of(t) == env_name)
    aspects = sorted({p.split(":")[0] for p in str(tid).split("|") if p})
    question = suite.view(tid).question

    with capsys.disabled():
        echo = print
        echo(f"\n=== {env_name} / {tid}  ({suite.template_id(tid)})")
        echo(f"    simulator: {pin.as_dict()}")
        echo(f"    opening:   {question}")
        started = time.time()
        with suite.session(tid, pin=pin) as sess:
            opening = sess.reset()
            assert opening.observation.agent_feedback == question, (
                "the env's own opening line must be the view's question, or the agent is "
                "being shown something other than what this suite claims it shows"
            )
            turns, seen, last = _trivial_episode(sess, question, aspects, echo)
            usage = sess.usage()
        echo(
            f"    -> turns={len(turns)} rewards={[r for _, r in turns]} "
            f"micro_max={max((r for _, r in turns), default=0.0)} "
            f"micro_avg={sum(r for _, r in turns) / max(len(turns), 1):.3f}"
        )
        echo(
            f"    -> active={last.metrics.active_elicited} "
            f"passive={last.metrics.passive_elicited} "
            f"of {last.metrics.total_preferences} preferences; "
            f"{usage}; {time.time() - started:.1f}s"
        )

    # THE MACHINERY, not the score.
    assert len(turns) >= len(aspects), "every aspect must have reached the env"
    # AN INVARIANT, NOT A SCORE. The first draft of this line was `assert seen` -- "a search
    # must have returned option ids" -- and travel44 failed it on the first live run because
    # both of its searches were judged wrong: one on arguments, one on an assertion inside
    # upstream's own search judge. Neither is a defect; a trivial policy writing a bad query
    # is precisely what `search_correct_reward` exists to price, and asserting that it did
    # not happen was asserting a score while claiming to assert the machinery.
    #
    # What upstream really guarantees is the biconditional: `search_correct_reward` (0.2) is
    # paid on exactly the branch that prints the schema and the option list, and no other
    # branch prints ids. That holds however badly the policy searches.
    accepted = [a for a, r in turns if a.startswith("[search]") and r == pytest.approx(0.2)]
    assert bool(seen) == bool(accepted), (
        f"{len(accepted)} search(es) were paid search_correct_reward but "
        f"{len(seen)} option ids came back; upstream prints the option list on exactly "
        "that branch"
    )
    assert usage["simulator_calls"] > 0, "an unarmed guard reports zero and looks identical"
    assert usage["simulator_failures"] == 0 and usage["simulator_malformed"] == 0
    assert last.metrics.total_preferences > 0
    # Counted apart. Which of the two is non-zero depends on the policy and the judge, and
    # asserting either would be asserting a score.
    assert last.metrics.active_elicited >= 0 and last.metrics.passive_elicited >= 0
    assert (
        last.metrics.active_elicited + last.metrics.passive_elicited
        <= last.metrics.total_preferences
    )


@pytest.mark.integration
def test_a_wrong_model_pin_refuses_the_turn_instead_of_scoring_it_zero():
    """THE CENTREPIECE, ON THE LIVE PATH. Costs one rejected request.

    Everything about the failure counter can be, and is, checked offline against a stub. What
    a stub cannot check is that the REAL failure -- a real endpoint rejecting a real request
    -- travels the path this package assumes it does: provider error -> upstream's three
    retries -> `model_call` returns None -> `evaluate_action` swallows it into reward 0.0 and
    a plausible sentence -> `UserBenchSession.step` refuses.

    The model id below is deliberately not served. Upstream's own behaviour, with the guard
    removed, is `("Currently the searching backend is experiencing some issues. Please try
    again later.", [], 0.0)` -- a complete, scoreable, entirely plausible zero, and the SAME
    string the env emits for its own deliberate every-fifth-search error. This asserts that
    what comes back instead is a refusal.
    """
    suite = _live_suite()
    live = _live_pin()
    wrong = SimulatorPin(
        model="a-model-this-proxy-does-not-serve",
        api_key=live.api_key,
        base_url=live.base_url,
    )
    tid = suite.task_ids()[0]
    with suite.session(tid, pin=wrong) as sess:
        sess.reset()  # no model call: the opening line is read from the task record
        with pytest.raises(SimulatorCallFailed) as excinfo:
            sess.step("[search] hotels in Austin")
        usage = sess.usage()
    assert usage["simulator_calls"] == 1, "the call must have been made and observed"
    assert usage["simulator_failures"] == 1
    # The message has to name what to check, because the alternative outcome is a table of
    # zeros with no explanation in it.
    assert "user-simulator call(s) came back unusable" in str(excinfo.value)
    assert "simulator model pin" in str(excinfo.value)


@pytest.mark.integration
def test_the_second_reset_and_the_pinned_seed_hold_on_a_real_env():
    """Two claims the adapter rests on, checked against a real `TravelEnv`. COSTS NOTHING.

    Building an env makes NO model call -- `__init__` loads the task, seeds the RNGs and
    calls `reset()`, and the first call happens on the first `step()`. So both of these are
    free to check, and neither was ever checked.

      1. `UserBenchSession.reset()` is the SECOND reset (`TravelEnv.__init__` already called
         one) and its docstring claims that is harmless. If it were not -- if a second reset
         reshuffled the option pools or rebuilt the preference list differently -- the agent's
         opening line would come from one state and its rewards from another.
      2. `ENV_SEED` is pinned so that the second env built in a process sees the same option
         order as the first. Without it `travelgym.env.task_data` seeds numpy once at import
         and each construction continues that stream, so a sweep's task 2 would be scored
         against a differently-ordered pool than a rerun of task 2 alone.
    """
    pytest.importorskip("travelgym", reason="needs the UserBench checkout installed")
    import copy

    import travelgym

    config = travelgym.get_default_config()
    config.data_mode, config.data_source = "single", "apartment:2-86|rental_car:2-60"
    config.seed, config.max_steps = ENV_SEED, MAX_STEPS
    config.one_choice_per_aspect = ONE_CHOICE_PER_ASPECT
    # A pin whose key is never used: nothing below makes a call.
    SimulatorPin(model="unused", api_key="unused", base_url=None).apply(config)

    env = travelgym.TravelEnv(config)
    state_after_init = copy.deepcopy(env.state_list)
    prefs_after_init = copy.deepcopy(env.remaining_preferences)
    order_after_init = {k: [o["id"] for o in v] for k, v in env.current_task["all_options"].items()}

    obs, _info = env.reset()
    assert copy.deepcopy(env.state_list) == state_after_init
    assert copy.deepcopy(env.remaining_preferences) == prefs_after_init
    assert {
        k: [o["id"] for o in v] for k, v in env.current_task["all_options"].items()
    } == order_after_init
    # ... and the opening line the agent is shown is the task's own, not the scenario prose.
    assert obs["feedback"] == env.current_task["initial_desc"]
    assert obs["feedback"] != env.current_task["scenario"]

    second = travelgym.TravelEnv(config)
    assert {
        k: [o["id"] for o in v] for k, v in second.current_task["all_options"].items()
    } == order_after_init, "config.seed must make the second env in a process reproducible"


@pytest.mark.integration
def test_upstream_still_defaults_to_the_model_we_refuse_to_use():
    """`UPSTREAM_DEFAULT_MODEL` is transcribed, so it can rot. Checked against the real config.

    If upstream changes its default, the constant is wrong and the comment that explains why
    the pin exists stops describing reality -- but nothing else would fail, because the
    adapter overwrites the field either way. This is the only thing that would say so.
    """
    pytest.importorskip("travelgym", reason="needs the UserBench checkout installed")
    import travelgym

    config = travelgym.get_default_config()
    assert config.model_name == UPSTREAM_DEFAULT_MODEL
    assert config.timeout == 15.0, "the transport default SIMULATOR_TIMEOUT_S replaces"
    assert config.one_choice_per_aspect is ONE_CHOICE_PER_ASPECT
    assert config.max_steps == MAX_STEPS


@pytest.mark.integration
def test_the_pin_is_what_reaches_upstreams_config():
    """The factory `session()` builds must write both fields onto a real TravelGymConfig.

    Constructing the config is free; constructing the ENV is not, so this checks the config
    without ever calling `TravelEnv`.
    """
    pytest.importorskip("travelgym", reason="needs the UserBench checkout installed")
    import travelgym

    config = travelgym.get_default_config()
    SimulatorPin(model="aws/gpt-oss-120b", api_key="sk-x", base_url="https://p/v1").apply(config)
    config.data_mode, config.data_source = "single", "hotel:2-1"
    config.validate()  # upstream's own validator must still accept what we wrote
    assert config.model_name == "aws/gpt-oss-120b"
    assert config.api_key == "sk-x"


@pytest.mark.integration
def test_the_guard_intercepts_upstreams_real_evaluate_action():
    """The stub above, replaced by upstream's actual `evaluate_action`. COSTS NOTHING.

    `model_call` is replaced with one that returns `None` -- exactly what upstream's own
    returns after three failed attempts -- so no request is ever made. What this proves that
    the stub cannot is that `evaluate_action`, `generate_judge_search` and
    `generate_judge_response` really do resolve `model_call` through the MODULE GLOBAL at
    call time, which is the assumption `counted_simulator` rests on. If upstream ever
    rewrites that into a captured alias or an injected client, the guard goes silently inert
    and every run reports clean; this is the test that would say so.

    Verified by hand against commit 80506d2 before it was written down:
      * one `[action]` turn with a dead simulator makes TWO calls (judge, then responder),
        both counted, and upstream returns reward 0.0 with a canned apology;
      * one `[search]` turn makes one call, counted, reward 0.0;
      * the env's own DELIBERATE every-5th-search error produces the BYTE-IDENTICAL feedback
        string with ZERO model calls, and is correctly not refused;
      * `[answer] <best id>` makes no call at all and scores 1.0.
    """
    pytest.importorskip("travelgym", reason="needs the UserBench checkout installed")
    from travelgym.env import prompts as upstream

    task = {
        "id": "t",
        "scenario": "...",
        "dimensions": ["hotel"],
        "all_options": {},
        "arguments": {},
    }
    state_config = {
        "search_correct_reward": 0.2,
        "preference_correct_reward": 0.2,
        "choice_correct_reward": 0.8,
        "choice_best_reward": 1.0,
        "search_failure_interval": 5,
        "elicitation_interval": 3,
        "wrong_choice_penalty": 0.0,
        "one_choice_per_aspect": True,
    }

    def state():
        return {
            "search_times": 0,
            "nonpreference_times": 0,
            "search_arguments": ["hotel"],
            "remaining_best_options": ["H1"],
            "remaining_correct_options": ["H1"],
            "choice_initials": [],
            "active_elicited_preferences": 0,
            "passive_elicited_preferences": 0,
        }

    dead = lambda system_prompt, user_prompt, model_config: None  # noqa: E731
    original, upstream.model_call = upstream.model_call, dead
    try:
        # A dead simulator: upstream scores it 0.0, and the counter sees every call.
        counter = SimulatorFailureCounter()
        with counted_simulator(((upstream, "model_call"),), counter=counter):
            _resp, _elicited, reward = upstream.evaluate_action(
                "[action] what sort of hotel?", task, state_config, {}, [], [], state()
            )
        assert reward == 0.0, "upstream must be unchanged by the guard"
        assert counter.failures == counter.calls > 0
        with pytest.raises(SimulatorCallFailed):
            counter.raise_if_any(where="userbench/integration")

        # The env's OWN deliberate every-Nth-search error: same feedback text, no model call,
        # and therefore no refusal. This is the discrimination the whole design turns on.
        deliberate = SimulatorFailureCounter()
        st = state()
        st["search_times"] = state_config["search_failure_interval"] - 1
        with counted_simulator(((upstream, "model_call"),), counter=deliberate):
            _resp2, _e, reward2 = upstream.evaluate_action(
                "[search] hotels", task, state_config, {}, [], [], st
            )
        assert reward2 == 0.0
        assert deliberate.calls == 0 and deliberate.clean

        # `[answer]` is pure set membership: no model call, so nothing to fail.
        answered = SimulatorFailureCounter()
        with counted_simulator(((upstream, "model_call"),), counter=answered):
            _resp3, _e, reward3 = upstream.evaluate_action(
                "[answer] H1", task, state_config, {}, [], [], state()
            )
        assert reward3 == pytest.approx(1.0)
        assert answered.calls == 0 and answered.clean
    finally:
        upstream.model_call = original


@pytest.mark.integration
def test_against_a_real_checkout():
    """Everything above with the real thing. Constructs the suite and takes a view.

    It does NOT construct a TravelEnv and does not require `travelgym` to be importable --
    only USERBENCH_DIR and pyarrow. Two separate reasons, both load-bearing: building an env
    starts a dialogue with a paid user simulator, and the whole task list, the split, the
    view and the corpus hash are derived from the parquet index and the scenario JSON without
    upstream's code being involved at all. Gating this on `available()` would skip the one
    check that reads real bytes on every machine that has the data but not the package.
    """
    try:
        suite = UserBenchSuite()
    except UserBenchUnavailable as exc:
        pytest.skip(str(exc))
    tids = suite.task_ids()
    assert len(tids) == N_TEST_TASKS
    for env, want in N_TEST_BY_ENV.items():
        assert sum(1 for t in tids if suite.env_of(t) == env) == want
    view = suite.view(tids[0])
    assert view.question
    record = suite._scenario_record(str(tids[0]))
    # The whole point, on real data: the opening line is not the scenario.
    assert view.question != record["scenario"]
    assert len(view.question) < len(record["scenario"])
