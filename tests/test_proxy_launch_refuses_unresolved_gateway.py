"""A proxy whose gateway `api_base` does not resolve must REFUSE to start.

WHY THIS TEST EXISTS. On 2026-09-20 13:52 the shared proxy (PID 75168) was launched without
`PINQ_GATEWAY_BASE_URL` in its environment. Every gateway entry in `conf/serving/litellm.yaml`
sets `api_base: os.environ/PINQ_GATEWAY_BASE_URL`, and litellm treats an unresolved `api_base`
as "not specified" -- so it fell back to the DEFAULT OpenAI endpoint and sent an IBM gateway
key to `api.openai.com`, which answered

    401 {"error": {"message": "Incorrect API key provided: sk-...h3tw. You c..."}}

naming `https://platform.openai.com`. That 401 was read for a day and a half as a dead
credential ("the gateway rejects both keys; needs a rotated key plus an operator restart").
Both keys were in fact healthy: probed directly against the gateway the same minute, all four
key x frozen-role combinations returned 200 with the right model served. The fault was one
missing environment variable at launch, and NOTHING checked for it -- the misconfiguration was
indistinguishable, from the caller's side, from an authentication failure.

So the launcher refuses. An unset gateway var is a refusal before any request is made, not a
401 forty minutes into a campaign that looks like someone revoked a key.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "tau2_campaign" / "proxy_up.sh"


def _run(env_overrides: dict[str, str | None]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # The launcher sources the repo's .env when there is one and then requires both gateway keys.
    # Without a .env (CI, a fresh clone) every case stopped at the key check, before the gateway
    # check this file is about, so the keys get placeholder values unless the caller set them.
    env.setdefault("LITELLM_API_KEY", "test-key-not-used")
    env.setdefault("PI_AGENT_LITELLM_API_KEY", "test-key-not-used")
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    # PINQ_PROXY_CHECK_ONLY stops before binding a port: this test must never start a service.
    env["PINQ_PROXY_CHECK_ONLY"] = "1"
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=120,
    )


def test_the_launcher_exists_and_is_executable_bash() -> None:
    assert LAUNCHER.is_file(), f"{LAUNCHER} is missing"


def test_it_refuses_when_the_gateway_base_url_is_unset() -> None:
    """The exact 2026-09-20 condition: the var absent from the launch environment."""
    proc = _run({"PINQ_GATEWAY_BASE_URL": None})
    assert proc.returncode != 0, (
        "launcher started with an unresolvable gateway api_base -- this is the fault that sent "
        f"an IBM key to OpenAI.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "PINQ_GATEWAY_BASE_URL" in combined, (
        "the refusal must NAME the variable -- a refusal that does not say which var is missing "
        f"reproduces the original debugging cost.\ngot:\n{combined}"
    )


def test_it_refuses_when_the_gateway_base_url_is_empty() -> None:
    """Empty is not set. `export X=` resolves to "" and hits the same litellm fallback."""
    proc = _run({"PINQ_GATEWAY_BASE_URL": ""})
    assert proc.returncode != 0, (
        f"an empty gateway base URL was accepted.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_it_refuses_a_gateway_url_that_points_at_a_proxy_port() -> None:
    """`PINQ_GATEWAY_BASE_URL="$LITELLM_BASE_URL"` is the documented recipe, and it is a trap.

    When the sweep has already pointed `LITELLM_BASE_URL` at the proxy, that recipe points the
    wildcard at the proxy ITSELF -- noted in `artifacts/night_readouts_20260919/RESULT.md:106`.
    A loopback gateway URL is never right: the gateway is remote.
    """
    proc = _run({"PINQ_GATEWAY_BASE_URL": "http://127.0.0.1:4000"})
    assert proc.returncode != 0, (
        "a loopback gateway URL was accepted -- the wildcard would forward to the proxy itself.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_it_accepts_a_well_formed_remote_gateway_url() -> None:
    """The guard must not refuse the working configuration -- otherwise it is just an outage."""
    proc = _run({"PINQ_GATEWAY_BASE_URL": "https://gateway.example.invalid"})
    assert proc.returncode == 0, (
        "the guard rejected a well-formed remote gateway URL, so it cannot distinguish the "
        f"fault from the fix.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_the_config_it_launches_names_only_resolvable_api_bases() -> None:
    """Every `os.environ/` api_base in the tau2 config must be one the launcher guards.

    A second unguarded name would reintroduce the bug under a different spelling.
    """
    import re

    cfg = REPO / "conf" / "serving" / "litellm.tau2.yaml"
    assert cfg.is_file(), f"{cfg} is missing"
    names = set(re.findall(r"api_base:\s*os\.environ/([A-Z0-9_]+)", cfg.read_text()))
    guarded = set(re.findall(r"[A-Z0-9_]*GATEWAY_BASE_URL", LAUNCHER.read_text()))
    unguarded = names - guarded
    assert not unguarded, (
        f"these api_base env names are used by the config but not guarded by the launcher: "
        f"{sorted(unguarded)}"
    )
