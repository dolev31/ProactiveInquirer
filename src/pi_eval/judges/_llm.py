"""Shared plumbing for the three DeepResearchGym judges: the seam, the parser, the pins.

WHY A STRICT PARSER IS A SCIENTIFIC REQUIREMENT, NOT A STYLE PREFERENCE. Upstream uses
OpenAI structured outputs, so a malformed verdict is impossible there and no parser exists to
get wrong. We reimplement against a provider-agnostic client, which means we DO have to
parse, and the tempting shape --

    label = result.get("label", "Supported")     # or: except Exception: return "Supported"

-- turns every provider hiccup into a passing grade and biases the recall estimate upward by
however often the judge stutters. Nothing here defaults. A verdict this module cannot read
raises JudgeParseError, the item is recorded as unjudged, and the unjudged rate is reported
next to the metric.

TEMPERATURE IS PINNED AT THE CALL SITE, not left to the client's constructor default, so a
run cannot silently sample its judge because someone changed a default two packages away.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol

from pinq.ids import h

JUDGE_TEMPERATURE = 0.0
JUDGE_SYSTEM = "You are a careful evaluator. You reply with one JSON object and nothing else."

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class JudgeParseError(ValueError):
    """The judge's reply was not a verdict this module can read. Never downgraded to a label."""


class JudgeLLM(Protocol):
    """Structural: pinq_adapters.llm.MeteredClient and pi_run.cache.ReplayClient both fit."""

    def complete(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, Any]: ...


def prompt_sha(*parts: str) -> str:
    """The pin that goes on every Judgment. Change the prompt, change the hash, change the
    judge -- which is what stops two prompt revisions from being averaged into one number."""
    return h("judge_prompt", *parts)


def judgment_id(*parts: str) -> str:
    return h("judgment", *parts)[:32]


def response_sha(text: str) -> str:
    """Digest of the judge's RAW reply, domain-separated.

    Recorded on every Judgment because the verdict is a MEASUREMENT and this is the byte
    string it was read from. Without it, a re-parse under a fixed parser and a re-judge under
    a changed model produce indistinguishable rows.
    """
    return h("judge_response", text)


def ask_with_telemetry(
    llm: JudgeLLM,
    *,
    prompt: str,
    seed: int = 0,
    role: str = "judge",
    max_tokens: int | None = None,
    system: str = JUDGE_SYSTEM,
    temperature: float | None = None,
) -> tuple[str, Any]:
    """Same request `ask` sends, but also returns the call's telemetry. `ask` discards it
    because no judge in this package reads it; `pi_eval.annotate_llm` needs it to check whether
    THIS call's provider attached a reasoning trace (see that module's `_reasoning_trace`).

    `temperature` defaults to the judge's, so every judge is unchanged, but a CALLER THAT IS
    NOT A JUDGE must be able to state its own. Hardcoding `JUDGE_TEMPERATURE` here made the
    annotation pass sample at the judge's setting whatever its own pin said -- one shared
    helper quietly collapsing two roles into one rater, which is the thing
    `litellm_client`'s ONE ROLE, ONE PIN rule exists to prevent. It surfaced in production:
    every annotator call went out at 0.0 and the frontier models rejected it outright.
    """
    return llm.complete(
        role=role,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        seed=seed,
        max_tokens=max_tokens,
        temperature=JUDGE_TEMPERATURE if temperature is None else temperature,
        actor="judge",
    )


def ask(
    llm: JudgeLLM,
    *,
    prompt: str,
    seed: int = 0,
    role: str = "judge",
    max_tokens: int | None = None,
    system: str = JUDGE_SYSTEM,
) -> str:
    text, _tel = ask_with_telemetry(
        llm, prompt=prompt, seed=seed, role=role, max_tokens=max_tokens, system=system
    )
    return text


def strict_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object. A fenced ```json block is unwrapped; nothing else is repaired.

    Unwrapping a code fence is the single concession, because it is a rendering artifact of
    the transport rather than a difference in the verdict. Everything beyond that -- trailing
    prose, a bare label, a list -- is a judge that did not answer the question asked, and
    guessing what it meant is how a parser starts inventing verdicts.
    """
    if not isinstance(text, str) or not text.strip():
        raise JudgeParseError("judge returned an empty response")
    body = text.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge response is not JSON ({exc}): {body[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise JudgeParseError(f"judge response is a {type(obj).__name__}, expected an object")
    return obj


def require_label(obj: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> str:
    """Exact, case-sensitive membership. `allowed` is the label set the prompt asked for."""
    if key not in obj:
        raise JudgeParseError(f"verdict has no {key!r} field; keys: {sorted(obj)}")
    v = obj[key]
    if not isinstance(v, str) or v not in allowed:
        raise JudgeParseError(f"{key}={v!r} is not one of {allowed}")
    return v


def require_rating(obj: Mapping[str, Any], key: str, lo: int, hi: int) -> int:
    """An integer in [lo, hi]. bool is rejected explicitly: True would otherwise read as 1."""
    if key not in obj:
        raise JudgeParseError(f"verdict has no {key!r} field; keys: {sorted(obj)}")
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise JudgeParseError(
            f"{key}={v!r} is a {type(v).__name__}, expected an int in [{lo},{hi}]"
        )
    if not lo <= v <= hi:
        raise JudgeParseError(f"{key}={v} is outside [{lo},{hi}]")
    return v


def justification_of(obj: Mapping[str, Any], key: str = "justification") -> str:
    """Optional and never scored: it is read by humans during adjudication, nothing else."""
    v = obj.get(key, "")
    return v if isinstance(v, str) else ""


def n_words(text: str) -> int:
    return len(text.split())
