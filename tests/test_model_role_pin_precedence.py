"""Sourcing `.env` must not overwrite a model-role pin a caller deliberately set.

THE DEFECT, MEASURED 2026-09-19. `.env` sets every `PI_MODEL_*` role to `openai/aws/gpt-oss-120b`
(inquirer, drafter, answerer, user_sim, judge alike). `scripts/tau2_campaign/phase5_shard.sh`
wanted the drafter and answerer FROZEN at `openai/aws/claude-sonnet-5` regardless, and wrote:

    export PI_MODEL_DRAFTER=${PI_MODEL_DRAFTER:-openai/aws/claude-sonnet-5}

placed AFTER `. .env` (which the script sources to reach `TAU2_DATA_DIR` and the gateway key --
sourcing it is the standing instruction, because a missing key hangs in retry backoff instead of
erroring). Since `.env` already set `PI_MODEL_DRAFTER`, the `:-` default never applied: the
variable was non-empty, so bash left it alone. The Inquirer escaped only because its own branch
used a plain `export PI_MODEL_INQUIRER=...` with no `:-`.

MEASURED ON THE 408 RUN DIRECTORIES THIS PRODUCED. Every sampled manifest's `pins.drafter` and
`pins.answerer` record `model_id: "openai/aws/gpt-oss-120b"`, not `claude-sonnet-5` -- e.g.
`runs/005e0b9debcf458f5bc98c8aed12a553/manifest.json` under the rescued campaign root. The
comparison against the published `claude-sonnet-5` airline column (RESULT.md's
`d7cd7be043bbc488`, 536 runs) was therefore void for every one of the 408 units: the drafter and
answerer never ran as the model the column names.

THE FIX FOLLOWS `scripts/tau2_campaign/load_env.sh`'s OWN PATTERN for `PI_RUNS_ROOT` /
`PI_CACHE_ROOT`: capture a caller-set value BEFORE sourcing the file, source it, then restore
the captured value AFTER -- so a value the caller set explicitly always wins, with no `:-`
idiom (which only wins against an EMPTY variable, and `.env` never leaves these empty) anywhere
in the chain. This test sources the ACTUAL `load_env.sh`, exactly as `test_env_root_precedence.py`
does for the roots, and requires every `PI_MODEL_*` role to be protected the same way.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent.parent / "scripts" / "tau2_campaign" / "load_env.sh"

MODEL_VARS = (
    "PI_MODEL_INQUIRER",
    "PI_MODEL_DRAFTER",
    "PI_MODEL_ANSWERER",
    "PI_MODEL_USERSIM",
    "PI_MODEL_JUDGE",
)


def _run(env_file: Path, caller: dict[str, str]) -> subprocess.CompletedProcess:
    """Assemble the environment the way a shard does: absolute roots plus frozen role pins,
    THEN source `.env`, via the shared helper -- and print what survived."""
    prints = "; ".join(f'echo "{v}=${{{v}:-}}"' for v in MODEL_VARS)
    script = (
        f'set -u; PINQ_ENV_FILE="{env_file}"; . "{HELPER}" || exit $?; '
        f'{prints}; echo "KEY=${{LITELLM_API_KEY:-}}"'
    )
    env = {**os.environ, **caller}
    for v in (*MODEL_VARS, "PI_RUNS_ROOT", "PI_CACHE_ROOT"):
        env.pop(v, None)
    env.update(caller)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """A `.env` shaped exactly like this repository's: every role pinned to gpt-oss-120b,
    relative roots, and a key worth having -- the file every lane sources."""
    p = tmp_path / ".env"
    p.write_text(
        "PI_CACHE_ROOT=./cache\n"
        "PI_RUNS_ROOT=./runs\n"
        "LITELLM_API_KEY=sk-test\n"
        "PI_MODEL_INQUIRER=openai/aws/gpt-oss-120b\n"
        "PI_MODEL_DRAFTER=openai/aws/gpt-oss-120b\n"
        "PI_MODEL_ANSWERER=openai/aws/gpt-oss-120b\n"
        "PI_MODEL_USERSIM=openai/aws/gpt-oss-120b\n"
        "PI_MODEL_JUDGE=openai/aws/gpt-oss-120b\n"
    )
    return p


def test_the_helper_exists_and_is_what_is_under_test():
    """Without this, every assertion below is vacuous the same way two of the root tests once
    were: `. missing_file` is not fatal, and a caller's own value survives if nothing was
    sourced at all -- the property would hold for the wrong reason."""
    assert HELPER.is_file(), f"{HELPER} does not exist; every assertion below would be vacuous"


def test_a_caller_frozen_drafter_and_answerer_survive_sourcing_env(env_file, tmp_path):
    """THE REGRESSION, REPRODUCED. This is the exact shape of the phase-5 bug: the caller wants
    drafter/answerer frozen at claude-sonnet-5 no matter what `.env` says, and needs the frozen
    value to survive `. .env`. Before this lane's fix, `load_env.sh` only protected
    PI_RUNS_ROOT/PI_CACHE_ROOT, so this currently FAILS: sourcing the helper lets the file's
    gpt-oss-120b win, identically to what phase5_shard.sh's `${VAR:-default}` did.
    """
    caller = {
        "PI_RUNS_ROOT": str(tmp_path / "runs"),
        "PI_CACHE_ROOT": str(tmp_path / "cache"),
        "PI_MODEL_DRAFTER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_ANSWERER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_USERSIM": "openai/aws/gpt-oss-120b",
    }
    r = _run(env_file, caller)
    assert r.returncode == 0, r.stderr
    assert "KEY=sk-test" in r.stdout, (
        f"the env file was not sourced, so this proves nothing about precedence:\n{r.stdout}"
    )
    assert "PI_MODEL_DRAFTER=openai/aws/claude-sonnet-5" in r.stdout, (
        "the caller's frozen drafter pin did not survive sourcing .env -- this is the exact "
        f"mechanism that mis-pinned 408 units:\n{r.stdout}\n{r.stderr}"
    )
    assert "PI_MODEL_ANSWERER=openai/aws/claude-sonnet-5" in r.stdout, r.stdout


def test_a_role_the_caller_leaves_unset_still_comes_from_the_file(env_file, tmp_path):
    """NON-VACUITY. A fix that stopped sourcing `.env` altogether would pass the test above and
    break every arm's inquirer pin, which legitimately wants the file's default when the caller
    sets nothing for a given role (e.g. judge, which tau2 shards never freeze)."""
    caller = {"PI_RUNS_ROOT": str(tmp_path / "runs"), "PI_CACHE_ROOT": str(tmp_path / "cache")}
    r = _run(env_file, caller)
    assert r.returncode == 0, r.stderr
    assert "PI_MODEL_JUDGE=openai/aws/gpt-oss-120b" in r.stdout, r.stdout


def test_all_five_roles_are_protected_not_just_drafter(env_file, tmp_path):
    """The task was to extend the SAME treatment to every PI_MODEL_* role, not just the two the
    first campaign happened to get wrong -- inquirer's own per-arm pin must survive too."""
    caller = {
        "PI_RUNS_ROOT": str(tmp_path / "runs"),
        "PI_CACHE_ROOT": str(tmp_path / "cache"),
        "PI_MODEL_INQUIRER": "qwen3-8b-dpo-stacked-notdone-both",
        "PI_MODEL_DRAFTER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_ANSWERER": "openai/aws/claude-sonnet-5",
        "PI_MODEL_USERSIM": "openai/aws/gpt-oss-120b",
        "PI_MODEL_JUDGE": "openai/aws/claude-sonnet-5",
    }
    r = _run(env_file, caller)
    assert r.returncode == 0, r.stderr
    for v in MODEL_VARS:
        assert f"{v}={caller[v]}" in r.stdout, f"{v} lost precedence to .env:\n{r.stdout}"
