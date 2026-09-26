"""Every refusal branch of the tau2 thinking-channel launch guard, plus the vacuity branch and
one clean pass. No test here makes a real network call -- `run()`/`probe_route()` take an
injected `post` transport, and the guard itself is never run against a live server (see the
task's own instruction and the module docstring of `scripts/tau2_concordance/thinking_probe.py`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tau2_concordance" / "thinking_probe.py"


def _mod():
    spec = importlib.util.spec_from_file_location("thinking_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["thinking_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _mod()

CONFIG_TEXT = """\
model_list:
  - model_name: qwen3-8b-base
    litellm_params:
      model: hosted_vllm/qwen3-8b-base
      api_base: http://127.0.0.1:8402/v1
      api_key: unused-by-vllm
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
  - model_name: qwen3-8b-dpo-stacked-notdone-both
    litellm_params:
      model: hosted_vllm/qwen3-8b-dpo-stacked-notdone-both
      api_base: http://127.0.0.1:8402/v1
      api_key: unused-by-vllm
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
  - model_name: "*"
    litellm_params:
      model: openai/*
      api_base: os.environ/PINQ_GATEWAY_BASE_URL
      api_key: os.environ/LITELLM_API_KEY
"""

NO_LOCAL_ROUTES_TEXT = """\
model_list:
  - model_name: "*"
    litellm_params:
      model: openai/*
      api_base: os.environ/PINQ_GATEWAY_BASE_URL
      api_key: os.environ/LITELLM_API_KEY
"""


@pytest.fixture
def config(tmp_path) -> Path:
    p = tmp_path / "litellm.tau2.yaml"
    p.write_text(CONFIG_TEXT)
    return p


def _clean_response(completion_tokens: int = 3) -> tuple[int, dict, str]:
    body = {
        "choices": [{"message": {"content": "READY"}}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": 10},
    }
    return 200, body, json.dumps(body)


def _post_returning(fixed):
    """A transport that returns the same (status, body, raw) for every call, or looks the
    call up by model name when given a dict keyed on model."""

    def post(url, payload, timeout):
        if isinstance(fixed, dict):
            return fixed[payload["model"]]
        return fixed

    return post


# --------------------------------------------------------------------- local_models_from_config


def test_local_models_from_config_reads_only_hosted_vllm_routes():
    models = tp.local_models_from_config(CONFIG_TEXT)
    assert set(models) == {"qwen3-8b-base", "qwen3-8b-dpo-stacked-notdone-both"}
    for kwargs in models.values():
        assert kwargs == {"enable_thinking": False}


def test_local_models_from_config_is_empty_when_no_local_route_exists():
    assert tp.local_models_from_config(NO_LOCAL_ROUTES_TEXT) == {}


# --------------------------------------------------------------------- refusal branches


def test_refuses_on_a_literal_think_marker_in_content(config):
    body = {
        "choices": [{"message": {"content": "<think>reasoning here</think>READY"}}],
        "usage": {"completion_tokens": 5},
    }
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert route["thinking_detected"] is True
    assert "think" in route["refuse_reason"]


def test_refuses_on_a_nonempty_reasoning_content_field(config):
    body = {
        "choices": [{"message": {"content": "READY", "reasoning_content": "let me think..."}}],
        "usage": {"completion_tokens": 5},
    }
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert route["thinking_detected"] is True
    assert "reasoning" in route["refuse_reason"]


def test_refuses_on_a_nonempty_reasoning_field_alias(config):
    """Some providers use `reasoning` rather than `reasoning_content`; both must be caught."""
    body = {
        "choices": [{"message": {"content": "READY", "reasoning": "internal monologue"}}],
        "usage": {"completion_tokens": 5},
    }
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"


def test_refuses_when_completion_hits_the_cap(config):
    """>= 0.9 * max_tokens: the measured symptom of the reasoning channel eating the content
    budget (43.5% of prompted calls, per the memory this guard exists to prevent recurring)."""
    body = {
        "choices": [{"message": {"content": "partial output cut off mid-sen"}}],
        "usage": {"completion_tokens": 29},  # 29 / 32 = 0.906 >= 0.9
    }
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert "cap" in route["refuse_reason"]
    assert route["thinking_detected"] is False  # the cap is a distinct signal from (1)/(2)


def test_does_not_refuse_just_under_the_cap_threshold(config):
    body = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"completion_tokens": 28},  # 28 / 32 = 0.875 < 0.9
    }
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "PASS"


def test_refuses_on_a_non_200_response(config):
    post = _post_returning((500, None, "internal server error"))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert "not a 200" in route["refuse_reason"]


def test_refuses_on_a_200_with_no_completion_body(config):
    """A 200 whose body has no choices/message is not a completion either."""
    body = {"choices": []}
    post = _post_returning((200, body, json.dumps(body)))
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert "not a 200" in route["refuse_reason"]


def test_refuses_on_a_transport_failure(config):
    """A connection failure (no server behind the proxy) must refuse, not raise."""

    def post(url, payload, timeout):
        return 0, None, "ConnectionRefusedError: [Errno 61] Connection refused"

    result = tp.run(
        proxy_url="http://127.0.0.1:1",
        config_path=config,
        models=["qwen3-8b-base"],
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    (route,) = result["routes"]
    assert "not a 200" in route["refuse_reason"]


# --------------------------------------------------------------------- vacuity branch


def test_refuses_when_zero_routes_are_probed(tmp_path):
    """A guard that probes nothing must not pass. Config has no local route at all."""
    p = tmp_path / "litellm.tau2.yaml"
    p.write_text(NO_LOCAL_ROUTES_TEXT)

    calls = []

    def post(url, payload, timeout):
        calls.append(payload)
        return _clean_response()

    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=p,
        models=None,
        max_tokens=32,
        timeout=5,
        post=post,
    )
    assert result["verdict"] == "REFUSE"
    assert result["routes"] == []
    assert "zero routes" in result["reason"]
    assert calls == [], "the vacuity guard must not have probed anything to reach this state"


def test_refuses_when_requested_model_names_match_nothing(config):
    """Vacuity by way of a bad --model filter, not only an empty config."""
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=["not-a-real-model"],
        max_tokens=32,
        timeout=5,
        post=_post_returning(_clean_response()),
    )
    # a requested-but-absent model still gets a route entry (kwargs unknown, {}), then is
    # probed and must be judged on its own response -- it is not vacuous, since one route was
    # named. Assert this explicitly so the vacuity test above stays about TRUE emptiness.
    assert len(result["routes"]) == 1


def test_main_exit_code_3_and_out_file_on_vacuity(tmp_path):
    p = tmp_path / "litellm.tau2.yaml"
    p.write_text(NO_LOCAL_ROUTES_TEXT)
    out = tmp_path / "out.json"

    rc = tp.main(
        ["--proxy-url", "http://127.0.0.1:9", "--config", str(p), "--out", str(out)],
        post=_post_returning(_clean_response()),
    )
    assert rc == 3
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "REFUSE"
    assert doc["routes"] == []
    assert doc["proxy_url"] == "http://127.0.0.1:9"
    assert doc["config_path"] == str(p)
    assert doc["config_sha256"]


# --------------------------------------------------------------------- one pass


def test_a_clean_response_on_every_route_passes(config):
    result = tp.run(
        proxy_url="http://127.0.0.1:9",
        config_path=config,
        models=None,
        max_tokens=32,
        timeout=5,
        post=_post_returning(_clean_response()),
    )
    assert result["verdict"] == "PASS"
    assert len(result["routes"]) == 2
    for route in result["routes"]:
        assert route["refuse_reason"] is None
        assert route["thinking_detected"] is False
        assert route["chat_template_kwargs"] == {"enable_thinking": False}


def test_main_exit_code_0_and_out_file_on_pass(config, tmp_path):
    out = tmp_path / "out.json"
    rc = tp.main(
        ["--proxy-url", "http://127.0.0.1:9", "--config", str(config), "--out", str(out)],
        post=_post_returning(_clean_response()),
    )
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "PASS"
    assert len(doc["routes"]) == 2
    assert (
        doc["config_sha256"]
        == __import__("hashlib").sha256(config.read_text().encode()).hexdigest()
    )


def test_main_exit_code_3_and_out_file_when_one_route_refuses(config, tmp_path):
    out = tmp_path / "out.json"
    responses = {
        "qwen3-8b-base": _clean_response(),
        "qwen3-8b-dpo-stacked-notdone-both": (
            200,
            {
                "choices": [{"message": {"content": "<think>oops</think>READY"}}],
                "usage": {"completion_tokens": 5},
            },
            "",
        ),
    }
    rc = tp.main(
        ["--proxy-url", "http://127.0.0.1:9", "--config", str(config), "--out", str(out)],
        post=_post_returning(responses),
    )
    assert rc == 3
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "REFUSE"
    reasons = {r["model"]: r["refuse_reason"] for r in doc["routes"]}
    assert reasons["qwen3-8b-base"] is None
    assert "think" in reasons["qwen3-8b-dpo-stacked-notdone-both"]


def test_main_refuses_when_config_is_missing(tmp_path):
    rc = tp.main(
        ["--proxy-url", "http://127.0.0.1:9", "--config", str(tmp_path / "nope.yaml")],
        post=_post_returning(_clean_response()),
    )
    assert rc == 3
