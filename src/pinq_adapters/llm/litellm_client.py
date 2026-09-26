"""The ONE multi-provider binding. Every token the experiment spends passes through here.

Three properties this file exists to guarantee, each of which is load-bearing elsewhere:

  1. THE POLICY STAYS BUDGET-BLIND. MeteredClient debits the ledger internally and exposes
     no read access to it. An Inquirer can therefore call a model without being able to
     observe remaining budget, which is what makes prefix-k of a B=16 rollout exchangeable
     with a true B=k run (see pinq.types.State: "DELIBERATELY ABSENT: remaining budget").
     There is deliberately no `.ledger` property and no `.spent()`; adding one would silently
     void the prefix-exchangeability argument that the main table rests on.

  2. USD IS DETERMINISTIC. Cost is tokens x a pinned price table, never the provider's own
     dollar field. A provider's number depends on account tier, credits and billing day, so
     it cannot be recomputed from a stored artifact; ours can, which is what makes re-pricing
     a finished sweep a parquet re-read instead of a re-run.

  3. ONE ROLE, ONE PIN. Models are chosen per ROLE from the environment, never per call site,
     so "the Answerer is frozen across all arms" is enforced by construction rather than by
     everyone remembering to pass the same string.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from tenacity import (
    Retrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from pinq import tripwire
from pinq.budget import BudgetLedger
from pinq.ids import canon, h, request_sha
from pinq.types import CallTelemetry, ModelPin

# Pricing lives next door so `pi cost estimate` and parquet re-pricing need neither
# tenacity nor a provider SDK. Re-exported here because this is the obvious place to look.
from .pricing import DEFAULT_PRICE_TABLE, LLMConfigError, PriceTable  # noqa: F401

# One env var per ROLE. A role is a scientific object (the frozen Answerer, the policy under
# test); a model id is an implementation detail that must be swappable without touching code.
# BEFORE SWAPPING `PI_MODEL_INQUIRER` FOR A STRONGER MODEL, READ THIS. Measured twice, because
# the first measurement was wrong in a way that looked decisive.
#
# THE FIRST RUN was at ACT_MAX_TOKENS 400 x reasoning_headroom 3.0 = 1200, and gpt-5's reasoning
# channel consumed the entire budget: tok_completion == tok_reasoning == 1200 on 196 calls, with
# EMPTY content. Empty response -> parse failure -> STOP. It read as "gpt-5 stops at turn 0 49%
# of the time and scores answer_correct 0.098 against gpt-oss's 0.525", i.e. a verdict on the
# model. `_budget`'s docstring predicts exactly this confusion.
#
# THE SECOND RUN, at PI_LLM_REASONING_HEADROOM=10 (4,000 tokens), on turn-0 branch runs:
#
#     generator        n   answer_correct   stop@0   $/run    distinct cand/state   dup%
#     gpt-oss-120b  1362            0.539     1.2%   0.017                   6.48   19.2%
#     gpt-5          111            0.658     0.0%   0.227                   5.64   28.8%
#
# gpt-5 is BETTER per candidate -- 22% higher answer_correct, and it never once failed to ask --
# and 13.7x more expensive, because the reasoning channel it needs is billed.
#
# BUT IT IS WORSE FOR THIS JOB. Preference pairs are built from DISTINCT candidates at one
# state, and gpt-5 produces 5.64 distinct of 8 against gpt-oss's 6.48 (28.8% duplicates against
# 19.2%). Fewer distinct candidates means fewer usable pairs per state, and the quality gap it
# does win is in the ANSWER rather than in the question being ranked. So gpt-oss stays the
# candidate generator: more contrast, 13.7x cheaper, and on-policy for the model being trained.
#
# A stronger model is worth revisiting for a job where per-candidate quality is what matters --
# elicitation, or the reflector -- rather than one that needs variety.
ROLE_ENV: Mapping[str, str] = {
    "inquirer": "PI_MODEL_INQUIRER",
    "drafter": "PI_MODEL_DRAFTER",
    "answerer": "PI_MODEL_ANSWERER",
    "user_sim": "PI_MODEL_USERSIM",
    "judge": "PI_MODEL_JUDGE",
}

# Direct provider keys. Presence is reported by `pi env doctor`; values are NEVER printed,
# hashed into an id, or written to a run artifact.
PROVIDER_KEY_ENV: Mapping[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "litellm_proxy": "LITELLM_API_KEY",
}

# --------------------------------------------------------------------------- routing


def _provider_of(model: str, base_url: str | None) -> str:
    """Which provider actually served this. A proxy is its own provider: routing through
    LiteLLM changes rate limits, error codes and retry behaviour, so collapsing it into the
    underlying vendor would make the operational telemetry uninterpretable."""
    if base_url:
        return "litellm_proxy"
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _retryable() -> tuple[type[BaseException], ...]:
    """Retry only transport-shaped failures. A malformed-request error is a bug, and burning
    four attempts on it hides the bug behind a 30-second delay.

    The stdlib socket errors are always included, not only as a fallback: a proxy in front of
    the provider can surface a raw ConnectionError that never becomes a litellm type, and
    that is exactly the case worth retrying.
    """
    base: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    try:  # litellm is a `run` extra; the module must stay importable without it
        import litellm

        return base + (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            # APIError is the class a VPN-gated proxy raises when the tunnel drops: litellm
            # wraps the underlying connect failure and it never becomes APIConnectionError.
            # Without this the most likely interruption of a multi-hour sweep -- a laptop
            # sleeping -- got ZERO retries and permanently errored every in-flight unit.
            litellm.APIError,
        )
    except Exception:  # pragma: no cover - exercised only in installs without litellm
        return base


# AN UPSTREAM INFRASTRUCTURE FAILURE ARRIVES AS AuthenticationError, AND ONLY THE BODY SAYS SO.
# `_retryable()` excludes AuthenticationError on purpose -- a wrong key does not become right in 148
# seconds. But the gateway in front of this project returns 401 AuthenticationError for failures of its
# OWN infrastructure. Measured verbatim 2026-09-22 across three outages of one campaign:
#
#   401 - Authentication Error, Error in connector: Error querying the database: FATAL: ...
#   Authentication Error, All connection attempts failed. Received Model Group=aws/claude-sonnet-5
#
# Both are transport-shaped, and neither says anything about our credentials: probed direct in the same
# minute, both keys returned 200 on both frozen roles. Those outages produced 164 AuthenticationError
# units with ZERO retries, because the exception CLASS cannot tell "your key is wrong" from "our database
# is down" and only the MESSAGE can. Same hazard already recorded for the 403 permission wall, which
# also shares a class with transients.
#
# The class stays OUT of the retryable tuple; this predicate decides per exception. A credential error
# still fails fast, which is what makes the ladder informative rather than a 148-second delay on a bug.
_INFRA_AUTH_MARKERS = (
    "error in connector",
    "querying the database",
    "all connection attempts failed",
)


def is_infrastructure_auth_error(exc: BaseException) -> bool:
    """Is this AuthenticationError actually the PROVIDER's infrastructure failing, not our key?

    Speaks only about AuthenticationError: anything else is False, so the predicate cannot widen
    silently into classes that already have their own handling.
    """
    try:
        import litellm
    except Exception:  # pragma: no cover - installs without litellm
        return False
    if not isinstance(exc, litellm.AuthenticationError):
        return False
    body = str(exc).lower()
    return any(m in body for m in _INFRA_AUTH_MARKERS)


@dataclass(frozen=True, slots=True)
class _Response:
    text: str
    tok_prompt: int
    tok_completion: int
    tok_reasoning: int
    tok_cached: int
    http_status: int


# --------------------------------------------------------------------------- the client


# The retry ladder's defaults, NAMED. They used to live only as signature defaults, which
# made them unreadable once the constructor started consulting the environment -- and a
# test that asserted the ladder spans three rate-limit windows was reading
# `signature(...).parameters[...].default`, i.e. the sentinel. 8 attempts from 4.0s with a
# 30s cap is 4+8+16+30+30+30+30 ~= 148s, which is three 60s windows.
DEFAULT_MAX_ATTEMPTS = 8

# THE REASONING HEADROOM, and why it needs an env knob rather than a constructor default.
#
# `_budget` grows max_tokens so a reasoning model's hidden channel does not eat the caller's
# CONTENT budget. 3.0 was measured on gpt-oss-120b and is right for it. It is NOT right for
# every model, and the parameter's own docstring already says the headroom "is a property of
# the model, not of the caller" -- but there was no way to set it per model without editing
# code.
#
# MEASURED on 196 gpt-5 inquirer calls at ACT_MAX_TOKENS=400 x 3.0 = 1200:
#     tok_completion  mean 1063.4  median 1200  max 1200
#     tok_reasoning   mean 1039.0  median 1200  max 1200
# completion == reasoning, both pinned at the cap: the whole budget went to reasoning and the
# content was EMPTY. Empty response -> parse failure -> STOP, which was then reported as
# "gpt-5 stops at turn 0 49.1% of the time" and "answer_correct 0.098 vs 0.525" -- a verdict on
# the model that was really a verdict on the budget. Exactly the confusion `_budget`'s
# docstring warns about, made easy to reach by the missing knob.
DEFAULT_REASONING_HEADROOM = 3.0
DEFAULT_RETRY_INITIAL_WAIT = 4.0


def _headroom_from(env: Mapping[str, str]) -> float:
    """`PI_LLM_REASONING_HEADROOM`, floored at 1.0.

    Below 1.0 the client would ask for FEWER tokens than the caller's content budget, which is
    this parameter's own bug inverted; garbage falls back to the default rather than taking a
    sweep down over a typo in an env var.
    """
    raw = str(env.get("PI_LLM_REASONING_HEADROOM", "")).strip()
    if not raw:
        return DEFAULT_REASONING_HEADROOM
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_REASONING_HEADROOM


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return default


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, "")).strip())
    except (TypeError, ValueError):
        return default


class MeteredClient:
    """Implements pinq.protocols.LLM.

    Note what is NOT here: any way to read the ledger. `self._ledger` is private and no
    method returns spend, remaining budget, or a call count. That single omission is what
    lets a policy hold this object and still satisfy the budget-agnostic-by-type contract.
    """

    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        price_table: PriceTable | None = None,
        models: Mapping[str, str] | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        # A per-minute rate limit needs a ladder that SPANS minutes. At 4 attempts starting
        # at 0.5s the total backoff is 3.6-6.5s, so all four attempts land inside the same
        # window and the unit is guaranteed to fail. 8 attempts from 4.0s spans roughly
        # 4+8+16+30+30+30+30 = 148s, i.e. three windows.
        # gpt-oss reasons for 40-700 tokens before emitting content, inside max_tokens.
        # 3.0 leaves room for the reasoning channel without capping content.
        reasoning_headroom: float | None = None,
        max_attempts: int | None = None,
        retry_initial_wait: float | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        env = env if env is not None else os.environ
        self._ledger = ledger
        self._prices = price_table or PriceTable.load()
        self._models = dict(models or {})
        self._temperature = temperature
        self._timeout = timeout
        self._reasoning_headroom = (
            max(1.0, float(reasoning_headroom))
            if reasoning_headroom is not None
            else _headroom_from(env)
        )
        # The ladder is a property of the PROVIDER's rate limiter, so it belongs in the
        # environment rather than only in a constructor default. 8 attempts from 4.0s spans
        # ~148s and is right against a real per-minute limit; it is exactly wrong when there is
        # no provider to reach, where it turns a unit that fails on its first call into a
        # 148-second wait -- which a test of the offline path then pays, per unit.
        #
        # AN EXPLICIT ARGUMENT WINS, and the env is only the default. Same rule and same reason
        # as `sweep.max_concurrency`: a caller that states a number is giving an instruction for
        # this construction, and a variable exported once for an afternoon must not outrank it.
        # Getting this backwards is how a test that exists to exercise the retry ladder came to
        # be silently configured never to retry.
        self._max_attempts = max(
            1,
            max_attempts
            if max_attempts is not None
            else _int_env(env, "PI_LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
        )
        self._retry_initial_wait = (
            retry_initial_wait
            if retry_initial_wait is not None
            else _float_env(env, "PI_LLM_RETRY_WAIT_S", DEFAULT_RETRY_INITIAL_WAIT)
        )
        self._base_url = base_url or env.get("LITELLM_BASE_URL") or None
        self._api_key = api_key or env.get("LITELLM_API_KEY") or None
        self._env = dict(env)
        self._extra_body = dict(extra_body or {})
        self._seq = 0

    # ------------------------------------------------------------------ configuration

    def model_for(self, role: str) -> str:
        if role in self._models:
            return self._models[role]
        var = ROLE_ENV.get(role)
        model = self._env.get(var, "") if var else ""
        if not model:
            raise LLMConfigError(
                f"role {role!r} has no model pin: set {var or 'PI_MODEL_' + role.upper()}. "
                "Defaulting here would let two arms silently run on different models."
            )
        return model

    def pin(self, role: str) -> ModelPin:
        """The manifest-facing view. base_url is HASHED, never stored: it can carry a key in
        a query string, and a run artifact is the one place a secret must never reach.

        `adapter_sha` comes from `conf/checkpoints.json` and is None for every model id that
        registry does not list -- which is every id in use today. That is deliberate and is
        asserted in `tests/test_checkpoint_registry.py`: the field is inside `run_id`, so
        filling it in for an id that has already run would rename finished runs. Under I.14
        one vLLM process serves several adapters under one base url, so without this a
        re-trained adapter pushed under the same deployment name would be invisible to the
        manifest and `--resume` would skip it as work already done.
        """
        from pinq_adapters.llm.checkpoints import adapter_sha_of

        model = self.model_for(role)
        return ModelPin(
            role=role,  # type: ignore[arg-type]
            model_id=model,
            provider=_provider_of(model, self._base_url),
            base_url_sha=h("base_url", self._base_url or ""),
            adapter_sha=adapter_sha_of(model),
            sampling_sha=h("sampling", f"t={self._temperature}"),
            price_table_version=self._prices.version,
        )

    def _budget(self, content_tokens: int | None) -> int | None:
        """Grow max_tokens to cover a REASONING CHANNEL, where the provider has one.

        On a reasoning model the reasoning tokens are billed inside `max_tokens`, so a caller
        asking for 700 content tokens can receive 700 reasoning tokens and EMPTY CONTENT.
        Measured on gpt-oss-120b over tau2: an answerer call returned
        `completion=700, reasoning=701` and an empty answer -- which then scores 0 on every
        metric and reads as a model failure rather than a budget bug.

        The headroom is multiplicative and configurable because it is a property of the model,
        not of the caller: the caller's number stays the CONTENT budget it meant, which is
        also what keeps the frozen Answerer's word cap comparable across arms and providers.
        """
        if content_tokens is None:
            return None
        return int(content_tokens * self._reasoning_headroom)

    def request_payload(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> dict[str, Any]:
        """The EXACT bytes that identify this request. Its sha is the cache key.

        Sampling parameters are inside it on purpose: two calls that differ only in
        temperature are different requests and must not share a cache entry.
        """
        payload: dict[str, Any] = {
            "model": self.model_for(role),
            "messages": [dict(m) for m in messages],
            "seed": seed,
            "temperature": self._temperature,
            "max_tokens": self._budget(max_tokens),
        }
        payload.update({k: v for k, v in sorted(kw.items()) if k != "actor"})
        if self._extra_body:
            payload["extra_body"] = dict(sorted(self._extra_body.items()))
        return payload

    # ------------------------------------------------------------------ the call

    def complete(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, CallTelemetry]:
        actor = kw.pop("actor", role)
        payload = self.request_payload(
            role=role, messages=messages, seed=seed, max_tokens=max_tokens, **kw
        )
        req_sha = request_sha(payload)
        # FIREWALL LAYER 4, AT THE ONLY PLACE IT CAN FIRE. `pi verify firewall` scans the
        # response cache, whose record carries the response text and the request's SHA -- never
        # the request. So a gold answer interpolated into a PROMPT was written nowhere the scan
        # could see it, and the layer could only ever have caught a model echoing gold back.
        # Here the request still exists, has not been sent, and has not been billed.
        tripwire.assert_clean(canon(payload), where=f"a {role} request")
        model = payload["model"]
        provider = _provider_of(model, self._base_url)

        retryer = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(initial=self._retry_initial_wait, max=30.0),
            retry=retry_if_exception_type(_retryable())
            | retry_if_exception(is_infrastructure_auth_error),
            reraise=True,
        )

        t_start = time.perf_counter()
        busy_ms = 0
        attempts = 0
        last_error: BaseException | None = None
        resp: _Response | None = None
        for attempt in retryer:
            with attempt:
                attempts += 1
                t0 = time.perf_counter()
                try:
                    resp = self._dispatch(payload)
                except BaseException as exc:  # noqa: BLE001 - re-raised by tenacity
                    last_error = exc
                    raise
                finally:
                    busy_ms += int((time.perf_counter() - t0) * 1000)
        assert resp is not None
        wall_ms = int((time.perf_counter() - t_start) * 1000)
        # Everything not spent inside a dispatch was spent waiting out a backoff. That is the
        # only honest definition of a rate-limit stall available without provider cooperation.
        stall_ms = max(0, wall_ms - busy_ms)

        tel = CallTelemetry(
            call_id=self._next_call_id(req_sha),
            actor=actor,  # type: ignore[arg-type]
            model=model,
            provider=provider,
            request_sha=req_sha,
            response_sha=h("resp", resp.text),
            tok_prompt=resp.tok_prompt,
            tok_completion=resp.tok_completion,
            tok_reasoning=resp.tok_reasoning,
            tok_cached=resp.tok_cached,
            usd=self._prices.usd(
                model,
                tok_prompt=resp.tok_prompt,
                tok_completion=resp.tok_completion,
                tok_reasoning=resp.tok_reasoning,
                tok_cached=resp.tok_cached,
            ),
            wall_ms=wall_ms,
            ttft_ms=0,  # populated only when streaming; an artifact either way, never a claim
            cache_hit=False,
            retries=attempts - 1,
            rate_limit_stall_ms=stall_ms,
            http_status=resp.http_status,
            provider_error_code=_error_code(last_error) if attempts > 1 else None,
            attempt=attempts - 1,
        )
        self._ledger.record_call(tel)  # debited here, unreadable from here
        return resp.text, tel

    def _next_call_id(self, req_sha: str) -> str:
        """Deterministic and unique: hash of the request plus a per-client sequence. A uuid4
        would make two identical replays diff, which is exactly what the replay test checks."""
        self._seq += 1
        return h("call", req_sha, str(self._seq))[:32]

    def _dispatch(self, payload: Mapping[str, Any]) -> _Response:
        import litellm  # lazy: importing litellm costs seconds and every worker pays it

        kwargs = dict(payload)
        extra = kwargs.pop("extra_body", None)
        resp = litellm.completion(
            **kwargs,
            api_base=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            num_retries=0,  # tenacity owns retries; two retry layers make `retries` a lie
            drop_params=True,  # providers differ on `seed`; dropping beats a hard failure
            **self._routing(str(kwargs.get("model", ""))),
            **({"extra_body": extra} if extra else {}),
        )
        return _read_response(resp)

    def _routing(self, model: str) -> dict[str, str]:
        """Name the provider for a LOCALLY SERVED model id, and for nothing else.

        litellm infers the provider from the model prefix. A vLLM deployment has no prefix --
        it is `qwen3-8b-sft`, the `--lora-modules` name -- and the SDK raises
        `BadRequestError: LLM Provider NOT provided` before it opens a socket.

        THE NAME CANNOT SIMPLY BE PREFIXED INSTEAD. `PriceTable.rates` and
        `conf/checkpoints.json` are both keyed on the bare served name
        (docs/GPU_RUNBOOK.md 7: "reuse those served names, or `PriceTable.rates` raises
        `LLMConfigError`"), and `model_id` is inside `ModelPin.key` -> `run_id`. Pinning
        `litellm_proxy/qwen3-8b-sft` would miss both lookups and rename the arm. So the id
        stays bare on the wire and the provider is named beside it.

        `litellm_proxy` is what `_provider_of` already reports for any call with a base url,
        so this tells the SDK what the telemetry has always said. A model that HAS a prefix is
        left alone: every frozen role is `openai/aws/gpt-oss-120b`, already routable, and
        overriding its provider would move it.
        """
        if self._base_url and "/" not in model:
            return {"custom_llm_provider": "litellm_proxy"}
        return {}


def _read_response(resp: Any) -> _Response:
    """Read tokens only. The provider's dollar field is deliberately not read: see module
    docstring, property 2."""
    text = ""
    choices = getattr(resp, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        text = (getattr(message, "content", None) or "") if message else ""
    usage = getattr(resp, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cdet = getattr(usage, "completion_tokens_details", None)
    pdet = getattr(usage, "prompt_tokens_details", None)
    reasoning = int(getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0
    cached = int(getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0
    return _Response(
        text=text,
        tok_prompt=prompt,
        tok_completion=completion,
        tok_reasoning=reasoning,
        tok_cached=min(cached, prompt),  # normalise to the subset convention the table assumes
        http_status=200,
    )


def _error_code(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    return type(exc).__name__


def provider_key_status(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Presence only. `pi env doctor` prints these booleans and never a value."""
    env = env if env is not None else os.environ
    return {p: bool(env.get(var)) for p, var in sorted(PROVIDER_KEY_ENV.items())}


def role_pins(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env if env is not None else os.environ
    return {role: env.get(var, "") for role, var in sorted(ROLE_ENV.items())}
