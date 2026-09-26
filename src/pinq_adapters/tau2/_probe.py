"""Is tau2 actually importable here, and where is its data?

WHY A PROBE RATHER THAN A MODULE-SCOPE IMPORT
    tau2 pulls litellm, pydantic and a domain registry at import time. If any module the
    default test run touches imported it at module scope, `pytest -m "not integration"`
    would stop being runnable on a machine without the extra — which is precisely the
    machine most of this repo is developed on. Every tau2 import in this package therefore
    lives inside a function body, and this probe is the only thing that reports the outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

DOMAIN = "banking_knowledge"

# Both counts are asserted, not assumed: a silent upstream data change would otherwise
# alter every denominator in the paper without altering a single line of code.
N_DOCUMENTS = 698
N_TASKS = 97

# `bm25` is offline and dependency-free. NEVER `alltools`: it raises at construction
# without OPENAI_API_KEY plus a sandbox runtime plus ripgrep, so it cannot be reproduced.
RETRIEVAL_VARIANT = "bm25"

# Banking with the required documents already in the prompt. Constructs offline -- it uses
# KnowledgeToolsPlain and needs no embedder, no sandbox and no ripgrep -- so D13's objection
# to `alltools` does not apply. It is reached only through the `tau2_golden` suite id, never
# by passing a different argument to the `tau2` suite: `retrieval_variant` is in neither the
# manifest nor SEMANTIC_FIELDS, so the two conditions would otherwise share a semantic_hash.
GOLDEN_RETRIEVAL_VARIANT = "golden_retrieval"


def available() -> tuple[bool, str]:
    """(importable, reason). The reason is shown in skip messages, so it must be specific."""
    try:
        import tau2  # noqa: F401
    except ImportError as exc:
        return False, f"tau2 not installed ({exc}); install with: uv pip install -e '.[tau2]'"
    try:
        from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR
    except ImportError as exc:
        return False, f"tau2 installed but banking_knowledge is missing ({exc})"
    if not Path(KNOWLEDGE_DOCUMENTS_DIR).is_dir():
        # Installing the extra is NOT sufficient. The wheel ships code only; `data/` lives
        # at the repo root, outside the package, so a pip install leaves DATA_DIR pointing
        # at a path that does not exist. The 698 documents must be fetched from a checkout
        # and TAU2_DATA_DIR pointed at it, BEFORE tau2 is first imported (DATA_DIR is
        # resolved at import time and never re-read).
        return False, (
            f"tau2 is installed but its data is not: {KNOWLEDGE_DOCUMENTS_DIR} does not exist. "
            "The wheel ships no data/ directory. Clone sierra-research/tau2-bench at v1.0.1 "
            "and export TAU2_DATA_DIR=<checkout>/data before importing tau2."
        )
    return True, "tau2 available"


def documents_dir() -> Path:
    from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR

    return Path(KNOWLEDGE_DOCUMENTS_DIR)


def tasks_dir() -> Path:
    from tau2.domains.banking_knowledge.utils import KNOWLEDGE_TASK_SET_PATH

    return Path(KNOWLEDGE_TASK_SET_PATH)


def domain_data_dir(domain: str) -> Path:
    """`<tau2 data>/tau2/domains/<domain>/` — the directory holding db.json and tasks.json.

    DERIVED FROM TAU2_DATA_DIR rather than from a per-domain upstream constant, because only
    the knowledge domains export one (`KNOWLEDGE_DOCUMENTS_DIR`). retail, airline and telecom
    ship a `db.json` and a `tasks.json` and nothing that names them, so the layout is the only
    handle there is. Raises with the same instruction `available()` gives rather than returning
    a path that does not exist, since a missing corpus reads downstream as "the policy
    retrieved nothing" rather than as a broken install.
    """
    import os

    root = os.environ.get("TAU2_DATA_DIR", "")
    if not root:
        raise RuntimeError(
            "TAU2_DATA_DIR is unset. Clone tau2-bench-data and export "
            "TAU2_DATA_DIR=<checkout>/data before building a tau2 domain."
        )
    d = Path(root) / "tau2" / "domains" / domain
    if not d.is_dir():
        raise RuntimeError(f"no tau2 domain at {d}; check TAU2_DATA_DIR and the domain name")
    return d


def load_documents(root: Path | None = None) -> tuple[dict[str, str], ...]:
    """The 698 knowledge documents as plain {id, title, content} dicts.

    Plain dicts on purpose: this is what the pure extractor in discoverable.py consumes,
    so the same function serves the live corpus and the committed fixture.
    """
    d = Path(root) if root is not None else documents_dir()
    out = []
    for path in sorted(d.glob("*.json")):
        rec = json.loads(path.read_text())
        out.append(
            {
                "id": rec.get("id", path.stem),
                "title": rec.get("title", ""),
                "content": rec.get("content", ""),
            }
        )
    return tuple(out)


def load_task_records(root: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Raw task JSON, globbed from `tasks/task_*.json`.

    NEVER read the aggregate `tasks.json`: upstream ships it stale for 13 of the 97 tasks,
    so an aggregate read silently scores a handful of tasks against the wrong criteria.
    """
    d = Path(root) if root is not None else tasks_dir()
    return tuple(json.loads(p.read_text()) for p in sorted(d.glob("task_*.json")))


# WHICH DOMAINS HAND THE AGENT THEIR POLICY. Default: all of them, because upstream `policy.md`
# IS the agent's system prompt and withholding it is not playing the benchmark.
#
# MEASURED on matched fork points, adding the policy: airline tau_reward 0.464 -> 0.661
# (p=0.0074), retail 0.533 -> 0.417 (p=0.065). One change, opposite signs -- the policy makes the
# agent ACT LESS, which is right where its failures over-act (airline) and wrong where they
# under-act (retail). Turning it off buys retail reward and costs faithfulness, so it is an
# explicit override a reader of the manifest can see, never a silent constant.
NO_POLICY_ENV = "PI_TAU2_NO_POLICY"


def policy_enabled(domain: str, env: Mapping[str, str] | None = None) -> bool:
    """Does this domain give the agent its own `policy.md`? True unless explicitly switched off."""
    import os

    raw = (env if env is not None else os.environ).get(NO_POLICY_ENV, "") or ""
    off = {x.strip().lower() for x in raw.split(",") if x.strip()}
    return str(domain).strip().lower() not in off
