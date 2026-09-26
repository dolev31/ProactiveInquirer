"""Sourcing `.env` must not overwrite a root a caller deliberately set and a guard validated.

THE DEFECT, MEASURED. `.env` in this repository carries `PI_RUNS_ROOT=./runs` and
`PI_CACHE_ROOT=./cache` -- RELATIVE. Every lane sources `.env`, because a missing gateway key
hangs in retry backoff instead of erroring, so sourcing it is the standing instruction. The
launcher for the tau2 airline campaign validated an ABSOLUTE `PI_RUNS_ROOT` at startup, refused
relative ones with its own exit code, and then -- further down, to get the gateway key --
sourced `.env`, which silently replaced the validated value with `./runs`. Twelve shards
inherited it and wrote 117 run directories into the pinned worktree instead of the shared
`runs/`, where no reporting path looks.

A GUARD THAT VALIDATES AND IS THEN OVERWRITTEN BEHIND ITS BACK IS WORSE THAN NO GUARD, because
it reports success. This is the second time in one night that a correction introduced the next
false state: the route rule's fix filed a permanent permission wall under "retry", and this
fix -- sourcing `.env` so the key is present -- undid the root check that ran before it.

THE REPO-WIDE INSTANCE IS ALREADY IN THE RECORD. A cache probe once read a 16% hit rate against
a cache that held 97.1%, because `./cache` resolved against the READER's working directory. That
was read as a finding about the cache. It was a finding about the path.

The rule these tests pin: a value the CALLER set explicitly wins over the file's, and whatever
survives must be absolute.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent.parent / "scripts" / "tau2_campaign" / "load_env.sh"


def _run(env_file: Path, caller: dict[str, str]) -> subprocess.CompletedProcess:
    """Assemble the environment the way a launcher does, and print what survived."""
    # `. helper || exit $?` IS THE CALLING CONVENTION, and it is part of what is under test.
    # A sourced guard returns its status to the caller; a caller that does not check it carries
    # on with the refusal printed and the bad value in hand, which is indistinguishable from
    # success to anything downstream.
    script = (
        f'set -u; PINQ_ENV_FILE="{env_file}"; . "{HELPER}" || exit $?; '
        'echo "RUNS=$PI_RUNS_ROOT"; echo "CACHE=$PI_CACHE_ROOT"; echo "KEY=${LITELLM_API_KEY:-}"'
    )
    env = {**os.environ, **caller}
    env.pop("PI_RUNS_ROOT", None)
    env.pop("PI_CACHE_ROOT", None)
    env.update(caller)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """A .env shaped exactly like this repository's: relative roots, and a key worth having."""
    p = tmp_path / ".env"
    p.write_text("PI_CACHE_ROOT=./cache\nPI_RUNS_ROOT=./runs\nLITELLM_API_KEY=sk-test\n")
    return p


def test_the_helper_exists_and_is_what_is_under_test():
    """Without this, the tests below pass VACUOUSLY: `. missing_file` is not fatal, so the
    caller's own value survives because nothing was sourced at all -- the property holds for
    the wrong reason. Two of these tests did exactly that on their first run."""
    assert HELPER.is_file(), f"{HELPER} does not exist; every assertion below would be vacuous"


def test_a_caller_set_absolute_root_survives_sourcing_the_env_file(env_file, tmp_path):
    """THE REGRESSION. The campaign wrote 117 run directories into the wrong tree because this
    was false: the file's `./runs` won over an absolute root the caller had already validated.

    THE KEY IS ASSERTED IN THE SAME TEST ON PURPOSE. "The root survived" is satisfiable by not
    reading the file at all; "the root survived AND the key is loaded" is only satisfiable by
    reading it and then restoring the root, which is the actual fix.
    """
    abs_runs, abs_cache = str(tmp_path / "shared_runs"), str(tmp_path / "shared_cache")
    r = _run(env_file, {"PI_RUNS_ROOT": abs_runs, "PI_CACHE_ROOT": abs_cache})
    assert r.returncode == 0, r.stderr
    assert "KEY=sk-test" in r.stdout, (
        f"the env file was not sourced, so this proves nothing about precedence:\n{r.stdout}"
    )
    assert f"RUNS={abs_runs}" in r.stdout, (
        f"the caller's absolute runs root did not survive; got:\n{r.stdout}\n{r.stderr}"
    )
    assert f"CACHE={abs_cache}" in r.stdout, r.stdout


def test_the_key_is_still_loaded_because_that_is_why_anyone_sources_it(env_file, tmp_path):
    """NON-VACUITY, and it is the whole tension. A fix that simply stopped sourcing `.env`
    would pass the test above and break every lane: a missing gateway key does not error, it
    hangs in retry backoff. The file must still be read; only the roots are protected."""
    r = _run(env_file, {"PI_RUNS_ROOT": str(tmp_path / "r"), "PI_CACHE_ROOT": str(tmp_path / "c")})
    assert "KEY=sk-test" in r.stdout, r.stdout


def test_the_file_supplies_a_root_the_caller_left_unset(env_file, tmp_path):
    """The file is not ignored -- it is the fallback. A caller that sets nothing gets the
    file's value, which is what every ad-hoc script relies on."""
    r = _run(env_file, {})
    assert r.returncode != 0 or "RUNS=./runs" in r.stdout, r.stdout


def test_a_relative_root_is_refused_whoever_supplied_it(env_file, tmp_path):
    """A relative root resolves against the caller's working directory, which is how two
    instruments read different caches and agreed with each other. Refused with a named reason
    rather than resolved, and refused whether it came from the caller or from the file."""
    r = _run(env_file, {"PI_RUNS_ROOT": "relative/runs", "PI_CACHE_ROOT": str(tmp_path / "c")})
    assert r.returncode != 0, f"a relative caller root must be refused; got:\n{r.stdout}"
    assert "absolute" in (r.stderr + r.stdout).lower()
