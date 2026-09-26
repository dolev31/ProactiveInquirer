"""The reflector's defaults produced HTTP 403, and the search died three seconds in.

`OpenAIReflector` posts to `{base_url}/chat/completions`. The default base_url is
`LITELLM_BASE_URL`, which for a litellm proxy is the ROOT ("https://host") -- the OpenAI-
compatible routes live under `/v1`. So the default posted to `https://host/chat/completions`
and got 403 Forbidden.

The default model is `PI_MODEL_INQUIRER`, which carries litellm's PROVIDER prefix
("openai/aws/gpt-oss-120b"). That prefix tells the litellm CLIENT which provider to use; a
proxy addressed directly over HTTP wants the bare deployment name ("aws/gpt-oss-120b").

Both are silent until the first reflection, which is after the seed generation has already been
rolled out and paid for -- `gen 0 seed ... mean=0.9300` was printed, then the search died. On
the full config that is real money spent to reach an exception.
"""

from __future__ import annotations

import pytest

from pinq_train.rung0_gepa.search import openai_base, proxy_model_name


@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://host", "https://host/v1"),
        ("https://host/", "https://host/v1"),
        ("https://host/v1", "https://host/v1"),
        ("https://host/v1/", "https://host/v1"),
    ],
)
def test_the_openai_route_prefix_is_added_once(given: str, expected: str) -> None:
    """Idempotent: a base_url that already names /v1 must not become /v1/v1."""
    assert openai_base(given) == expected


def test_an_empty_base_url_stays_empty() -> None:
    """So the caller's own 'reflector is unreachable' error fires, rather than a confusing
    request to '/v1'."""
    assert openai_base("") == ""


def test_a_non_v1_path_is_respected() -> None:
    """A proxy mounted under a custom path is not ours to rewrite."""
    assert openai_base("https://host/openai/deployments") == "https://host/openai/deployments"


@pytest.mark.parametrize(
    "given,expected",
    [
        ("openai/aws/gpt-oss-120b", "aws/gpt-oss-120b"),
        ("openai/Azure/gpt-5-2025-08-07", "Azure/gpt-5-2025-08-07"),
        ("aws/gpt-oss-120b", "aws/gpt-oss-120b"),
        ("", ""),
    ],
)
def test_the_litellm_provider_prefix_is_stripped(given: str, expected: str) -> None:
    """`openai/` tells the litellm CLIENT which provider to use. A proxy addressed directly
    over HTTP wants the deployment name, and rejects the prefixed form."""
    assert proxy_model_name(given) == expected


def test_only_the_leading_provider_segment_is_stripped() -> None:
    """A model whose own name contains 'openai' must survive."""
    assert proxy_model_name("azure/openai-gpt-4") == "azure/openai-gpt-4"
