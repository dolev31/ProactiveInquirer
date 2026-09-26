"""Launch guard: is the thinking channel actually OFF on every local route this campaign will
use, through the SAME proxy URL the rollout itself talks to?

WHY THIS EXISTS. `conf/serving/litellm.tau2.yaml` never attached
`extra_body.chat_template_kwargs.enable_thinking: false` to its `hosted_vllm/*` routes (fixed
alongside this file -- see the config's own comment), and the in-job config
`scripts/hpc/tau2_cluster_campaign.sh` generates is a straight `sed` of that same file. A
CONFIG fix is not a PROOF that a given launch is actually serving with the channel off: the
config could drift again, a peer's edit could reorder a mapping and silently keep the wrong
key (`router_add.py`'s own `duplicate_keys` guard exists for exactly that failure mode), or
the vLLM process behind the proxy could simply be older than the flag. Only a real completion,
through the real proxy URL, answers the question this script asks.

WHAT COUNTS AS A REFUSAL, and why each one is checked BEHAVIOURALLY rather than by reading the
config back:

  1. the completion's content contains a `<think>` / `</think>` marker -- the channel emitted
     literally into content, the shape a template can take when a provider ignores the kwarg;
  2. the message carries a non-empty `reasoning_content` or `reasoning` field -- the channel
     emitted into its own field instead, which (1) would not catch;
  3. `completion_tokens >= 0.9 * max_tokens` -- the call hit its cap, the direct, measured
     symptom of a reasoning channel eating the content budget (see
     `pinq_adapters.llm.litellm_client._budget`'s own docstring for the mechanism: MEASURED
     43.5% of prompted calls capped this way over a real tau2 campaign);
  4. the response is not a 200 with a completion at all -- an unreachable proxy or backend must
     refuse the launch, not silently probe nothing;
  5. ZERO routes were probed -- a guard that checks nothing must not read as a guard that
     passed. An empty model list is a MISCONFIGURATION of the guard's own inputs, not a clean
     bill of health.

Each refusal is checked independently and reported per route; ANY refusal anywhere REFUSES the
whole launch (exit 3). PASS (exit 0) prints one evidence line per route: model, completion
tokens, and the first 60 characters of what it said.

THE TRANSPORT IS INJECTABLE, and this matters for both testing and reuse. `probe_route` takes a
`post` callable (`(url, payload, timeout) -> (status, json_body, raw_text)`); no test in
`tests/test_thinking_probe.py` makes a real HTTP call, and this script itself does not run
against a live server as part of `pytest`. The coordinator's own use case is why `--proxy-url`
and `--config` are both REQUIRED arguments rather than defaults baked in here: a rerun may go
from this Mac, through a local litellm proxy, to a cluster vLLM on a tunnelled port -- not only
through the in-job cluster config -- so neither the proxy address nor the config path may be
assumed.

`--out PATH` records the full evidence as JSON, written on BOTH verdicts (PASS and REFUSE): a
downstream reader refuses to trust a rerun without this record, because `chat_template_kwargs`
lives in the proxy's own config and sits OUTSIDE run identity (`RunManifest` hashes the model
id and the base url, never the proxy's routing table) -- this file is the only artifact that
ties a specific run to what the proxy was actually configured to send.

    python scripts/tau2_concordance/thinking_probe.py \\
        --proxy-url http://127.0.0.1:4010 \\
        --config conf/serving/litellm.tau2.yaml \\
        --out artifacts/.../thinking_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

Transport = Callable[[str, dict[str, Any], float], tuple[int, dict[str, Any] | None, str]]

THINK_MARKERS = ("<think>", "</think>")
DEFAULT_MAX_TOKENS = 32
DEFAULT_TIMEOUT_S = 60.0
CAP_FRACTION = 0.9


def default_post(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any] | None, str]:
    """The real transport: one POST, stdlib only (this repo's own convention for a standalone
    probe -- see `scripts/tau2_campaign/route_map.py`, which reaches the same proxy with
    `urllib.request` rather than pulling in `requests` or `httpx` for a single call)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback proxy
            raw = r.read().decode("utf-8", errors="replace")
            try:
                body: dict[str, Any] | None = json.loads(raw)
            except json.JSONDecodeError:
                body = None
            return int(r.status), body, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = None
        return int(e.code), body, raw
    except Exception as e:  # noqa: BLE001 - a transport failure is a refusal, not a crash
        return 0, None, f"{type(e).__name__}: {e}"


def local_models_from_config(config_text: str) -> dict[str, dict[str, Any]]:
    """Every `hosted_vllm/*` route's model_name -> its configured `chat_template_kwargs`
    (empty dict if the route has none). Uses PyYAML rather than a hand regex: this file reads
    the config, it does not edit it, so there is no "two thirds of the file is prose" reason
    to avoid the loader the way `router_add.py` does for writes.
    """
    import yaml

    doc = yaml.safe_load(config_text) or {}
    out: dict[str, dict[str, Any]] = {}
    for m in doc.get("model_list") or []:
        params = m.get("litellm_params") or {}
        model = str(params.get("model") or "")
        if not model.startswith("hosted_vllm/"):
            continue
        name = str(m.get("model_name") or "")
        if not name:
            continue
        kwargs = (params.get("extra_body") or {}).get("chat_template_kwargs") or {}
        out[name] = dict(kwargs)
    return out


def probe_route(
    *,
    proxy_url: str,
    model: str,
    chat_template_kwargs: dict[str, Any],
    max_tokens: int,
    timeout: float,
    post: Transport,
) -> dict[str, Any]:
    """One short chat completion through `proxy_url`, and the verdict on it. Returns a route
    record shaped for the `--out` JSON: never raises -- a transport failure is a refusal, not
    an exception the caller must also handle.
    """
    url = proxy_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: READY"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    record: dict[str, Any] = {
        "model": model,
        "chat_template_kwargs": chat_template_kwargs,
        "completion_tokens": None,
        "max_tokens": max_tokens,
        "thinking_detected": False,
        "refuse_reason": None,
    }
    status, body, raw = post(url, payload, timeout)
    if status != 200 or body is None:
        record["refuse_reason"] = (
            f"the response is not a 200 with a completion (http={status}, body={raw[:300]!r})"
        )
        return record

    choices = body.get("choices") or []
    message = (choices[0] or {}).get("message") if choices else None
    if not isinstance(message, dict):
        record["refuse_reason"] = "the response is not a 200 with a completion (no message)"
        return record

    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    record["completion_tokens"] = completion_tokens
    record["content_preview"] = content[:60]

    has_think_marker = any(marker in content for marker in THINK_MARKERS)
    has_reasoning_field = bool(reasoning.strip())
    record["thinking_detected"] = bool(has_think_marker or has_reasoning_field)

    if has_think_marker:
        record["refuse_reason"] = "content contains a <think>/</think> marker"
        return record
    if has_reasoning_field:
        record["refuse_reason"] = "message carries a non-empty reasoning_content/reasoning field"
        return record
    if completion_tokens is None:
        record["refuse_reason"] = "usage.completion_tokens is missing; the cap cannot be checked"
        return record
    if completion_tokens >= CAP_FRACTION * max_tokens:
        record["refuse_reason"] = (
            f"completion_tokens ({completion_tokens}) >= {CAP_FRACTION} * max_tokens "
            f"({max_tokens}) -- the call hit its cap"
        )
        return record
    return record


def run(
    *,
    proxy_url: str,
    config_path: Path,
    models: Sequence[str] | None,
    max_tokens: int,
    timeout: float,
    post: Transport,
) -> dict[str, Any]:
    """The full guard, independent of argv/stdout: returns the exact `--out` document. `main`
    is the only thing that prints or exits, so this is what every test in
    `tests/test_thinking_probe.py` calls directly.
    """
    config_text = config_path.read_text()
    config_sha256 = hashlib.sha256(config_text.encode()).hexdigest()
    by_model = local_models_from_config(config_text)

    if models:
        targets = {name: by_model.get(name, {}) for name in models}
    else:
        targets = by_model

    result: dict[str, Any] = {
        "verdict": "PASS",
        "proxy_url": proxy_url,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "routes": [],
    }

    if not targets:
        result["verdict"] = "REFUSE"
        result["reason"] = (
            "zero routes were probed: no hosted_vllm/* route in the config (or none matched "
            "the requested --model names). A guard that probes nothing must not pass."
        )
        return result

    routes = []
    any_refused = False
    for model, kwargs in sorted(targets.items()):
        rec = probe_route(
            proxy_url=proxy_url,
            model=model,
            chat_template_kwargs=kwargs,
            max_tokens=max_tokens,
            timeout=timeout,
            post=post,
        )
        routes.append(rec)
        if rec.get("refuse_reason"):
            any_refused = True
    result["routes"] = routes
    result["verdict"] = "REFUSE" if any_refused else "PASS"
    return result


def main(argv: list[str] | None = None, *, post: Transport = default_post) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proxy-url", required=True, help="the SAME proxy URL the rollout uses")
    ap.add_argument("--config", required=True, type=Path, help="the litellm config in effect")
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="probe only this model (repeatable); default is every hosted_vllm/* route",
    )
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--out", type=Path, default=None, help="write the full evidence JSON here")
    a = ap.parse_args(argv)

    if not a.config.is_file():
        print(f"thinking_probe: REFUSING: no config at {a.config}", file=sys.stderr)
        return 3

    result = run(
        proxy_url=a.proxy_url,
        config_path=a.config,
        models=a.models,
        max_tokens=a.max_tokens,
        timeout=a.timeout,
        post=post,
    )

    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))

    if result["verdict"] == "REFUSE" and not result["routes"]:
        print(f"thinking_probe: REFUSE: {result.get('reason')}", file=sys.stderr)
        return 3

    for rec in result["routes"]:
        status = "REFUSE" if rec.get("refuse_reason") else "ok"
        print(
            f"thinking_probe: {status} model={rec['model']} "
            f"completion_tokens={rec.get('completion_tokens')} "
            f"content[:60]={rec.get('content_preview', '')!r}"
            + (f" -- {rec['refuse_reason']}" if rec.get("refuse_reason") else "")
        )

    if result["verdict"] == "REFUSE":
        print("thinking_probe: REFUSE -- see per-route reasons above", file=sys.stderr)
        return 3
    print(f"thinking_probe: PASS -- {len(result['routes'])} route(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
