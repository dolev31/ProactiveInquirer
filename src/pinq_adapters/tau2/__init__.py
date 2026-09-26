"""tau2 / tau-Knowledge adapter (banking_knowledge domain of sierra-research/tau2-bench).

Every symbol here is importable with tau2 ABSENT: the package is offline-first, and the
only things that touch upstream are function bodies guarded by `available()`.
"""

from ._probe import (
    DOMAIN,
    GOLDEN_RETRIEVAL_VARIANT,
    N_DOCUMENTS,
    N_TASKS,
    RETRIEVAL_VARIANT,
    available,
    load_documents,
    load_task_records,
)
from .actuator import Tau2Actuator
from .discoverable import (
    TOOL_TOKEN_RE,
    DocToolEdge,
    docs_naming,
    documented_but_not_a_tool,
    extract_tool_edges,
    scan_tool_tokens,
    tools_never_documented,
    unlockable_tools,
)
from .retriever import Tau2Retriever, parse_kb_results, pick_kb_tool
from .suite import CORPUS_ID, KNOWLEDGE_FIXED_DATE, Tau2Suite, Tau2SuiteUnavailable

__all__ = [
    "CORPUS_ID",
    "DOMAIN",
    "KNOWLEDGE_FIXED_DATE",
    "N_DOCUMENTS",
    "N_TASKS",
    "GOLDEN_RETRIEVAL_VARIANT",
    "RETRIEVAL_VARIANT",
    "TOOL_TOKEN_RE",
    "DocToolEdge",
    "Tau2Actuator",
    "Tau2Retriever",
    "Tau2Suite",
    "Tau2SuiteUnavailable",
    "available",
    "docs_naming",
    "documented_but_not_a_tool",
    "extract_tool_edges",
    "load_documents",
    "load_task_records",
    "parse_kb_results",
    "pick_kb_tool",
    "scan_tool_tokens",
    "tools_never_documented",
    "unlockable_tools",
]
