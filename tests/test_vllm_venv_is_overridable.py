"""`scripts/hpc/common.sh` must let the caller say where the vLLM venv is.

WHY. `VLLM_VENV` was hardcoded to `$REPO/.venv-vllm`, and a venv is not in the repository -- it is
generated and ignored. A PINNED WORKTREE carries neither untracked nor ignored files, so for any
worktree that path does not exist and `serve.sh` dies at its own guard:

    FATAL: /proj/pinq/user/wt-tau2/.venv-vllm missing -- run scripts/hpc/setup_env.sh first

which is what killed the first cluster tau2 preflight (LSF 895633) four seconds in. The guard was
right; the path was unreachable by construction. Running a campaign from a pinned worktree is not an
exotic case -- it is the only way to get a clean tree, and a dirty tree stamps every run `dev-`.

`ART` on the line above already takes `PINQ_ARTIFACTS`, so this is the established pattern here, not
a new one. The default is unchanged on purpose: every existing caller must keep resolving exactly
where it did before, so this test pins BOTH directions.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "scripts" / "hpc" / "common.sh"


def _resolve(env_extra: dict[str, str], fake_repo: str) -> str:
    """Print what common.sh resolves VLLM_VENV to, without running a job."""
    script = (
        f'REPO="{fake_repo}"\n'
        + "\n".join(
            ln
            for ln in COMMON.read_text().splitlines()
            if "VLLM_VENV" in ln and not ln.strip().startswith("#")
        )
        + '\necho "$VLLM_VENV"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env_extra},
        timeout=60,
    )
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def test_the_default_is_unchanged_for_every_existing_caller() -> None:
    """Without the override, it must still be exactly $REPO/.venv-vllm."""
    assert _resolve({}, "/tmp/fake-repo") == "/tmp/fake-repo/.venv-vllm"


def test_an_explicit_vllm_venv_is_honoured() -> None:
    """The case a pinned worktree needs: the venv lives outside the tree being run."""
    got = _resolve({"PINQ_VLLM_VENV": "/u/user/ProactiveInquirer/.venv-vllm"}, "/tmp/fake-repo")
    assert got == "/u/user/ProactiveInquirer/.venv-vllm", (
        "common.sh ignored PINQ_VLLM_VENV, so a pinned worktree cannot serve -- which is exactly "
        f"how LSF 895633 died. got {got!r}"
    )


def test_the_assignment_uses_the_same_default_idiom_as_ART() -> None:
    """Guard against a re-hardcode: the line must be a `${VAR:-default}` expansion."""
    line = next(ln for ln in COMMON.read_text().splitlines() if re.match(r"\s*VLLM_VENV=", ln))
    assert "PINQ_VLLM_VENV" in line and ":-" in line, (
        f"VLLM_VENV is assigned without an overridable default: {line!r}"
    )
