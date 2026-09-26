"""Turn a missing optional dependency into an instruction.

An optional extra fails at import time, several frames below whatever the researcher typed,
with `ModuleNotFoundError: No module named 'pyarrow'`. That message is accurate and useless:
it does not say which of this project's nine extras carries pyarrow, and the answer is not
guessable from the module name. `make venv` used to install only `[dev]`, so a stranger's
first `make gate` died exactly this way -- on a dependency the Makefile never installed, with
an error naming neither the Makefile nor the extra.

Stdlib only, like the rest of `pinq`.
"""

from __future__ import annotations

from types import ModuleType

# module -> the extra in pyproject's [project.optional-dependencies] that provides it.
EXTRA_OF: dict[str, str] = {
    "duckdb": "analysis",
    "matplotlib": "analysis",
    "pandas": "analysis",
    "pyarrow": "analysis",
    "scipy": "analysis",
    "statsmodels": "analysis",
    "litellm": "run",
    "tenacity": "run",
    "yaml": "run",
    "fastapi": "serve",
    "uvicorn": "serve",
    "datasets": "train",  # also in [suites]; rung 2 is the caller that fails loudest
    "httpx": "suites",
    "rank_bm25": "tau2",
    "tau2": "tau2",
    "peft": "train",
    "torch": "train",
    "transformers": "train",
    "trl": "train",
    "vllm": "train-grpo",
}


class MissingExtra(ImportError):
    """A required optional dependency is not installed, and the message says which extra."""


def missing(module: str, *, why: str = "") -> MissingExtra:
    """The exception to raise for an absent optional dependency. Names the extra.

    Returned rather than raised so it can be used from an `except ImportError` handler
    wrapping a top-of-file import, which is the only placement that does not push every
    later import past a statement and trip E402.
    """
    extra = EXTRA_OF.get(module.split(".")[0])
    hint = (
        f'install it with: uv pip install -e ".[{extra}]"'
        if extra
        else "it is not one of this project's declared extras; see pyproject.toml"
    )
    tail = f" ({why})" if why else ""
    return MissingExtra(f"{module} is required{tail}. {hint}")


def require(module: str, *, why: str = "") -> ModuleType:
    """Import `module`, or raise naming the extra and the exact command that installs it."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise missing(module, why=why) from exc
