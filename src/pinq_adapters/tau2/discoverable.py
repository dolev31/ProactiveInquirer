"""tau2's document -> tool prerequisite edges, extracted mechanically.

WHY THIS MODULE IS PURE AND TAKES PLAIN DICTS
    This is the paper's strongest single piece of evidence, so it must be verifiable
    without a 1.4k-file checkout on the machine. Everything here is stdlib over
    {id, title, content} mappings and a list of tool names, which is why the whole edge
    set can be unit-tested against a committed fixture with tau2 absent.

WHY THE EDGE IS OBJECTIVE
    Banking tools carry an unguessable four-digit suffix (`open_bank_account_4821`).
    `unlock_discoverable_agent_tool(name)` only succeeds once the naming document has been
    read, and the suffix cannot be brute-forced from the stem. So "document D names tool T"
    is a *binary-verifiable prerequisite*: no annotator, no LLM, no judgement call. A policy
    either read the document or it cannot invoke the tool.

WHY BOTH FILTERS ARE LOAD-BEARING — measured on the real v1.0.1 corpus (698 docs):
    * 46 distinct suffixed tokens appear in documents; 46 suffixed tools exist; they
      overlap in only 45. `downgrade_credit_card_3847` is written in a document but is NOT
      a real tool — a decoy. Emitting it would be a false prerequisite.
    * `example_agent_tool_0000` is a real tool named in NO document, so it has no
      unlocking prerequisite and must yield no edge.
    Dropping either filter corrupts the edge set, which is why both directions are tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

# Stem is lowercase/underscore, suffix is exactly four digits. Anchored on word boundaries
# so `see_open_bank_account_4821.` and a bare mention both match, while `v2_1234x` does not.
TOOL_TOKEN_RE = re.compile(r"\b[a-z][a-z_]*_\d{4}\b")


@dataclass(frozen=True, slots=True)
class DocToolEdge:
    """`doc_id` must be read before `tool_name` can be unlocked, hence invoked."""

    doc_id: str
    doc_title: str
    tool_name: str


def scan_tool_tokens(text: str) -> frozenset[str]:
    """Every `name_1234`-shaped token in a blob of document text.

    Deliberately NOT filtered against the tool list here: callers need the raw set to
    measure how many documented tokens are decoys.
    """
    return frozenset(TOOL_TOKEN_RE.findall(text or ""))


def extract_tool_edges(
    documents: Iterable[Mapping[str, str]],
    tool_names: Iterable[str],
) -> tuple[DocToolEdge, ...]:
    """Cross-reference documented tokens against tools that actually exist.

    A token yields an edge only if BOTH hold: it appears in a document, and it is a live
    tool name. See the module docstring for why each direction removes real cases.

    Returned sorted by (doc_id, tool_name) so the edge set is a deterministic artifact that
    can be hashed into a graph_version.
    """
    known = frozenset(tool_names)
    edges: set[DocToolEdge] = set()
    for doc in documents:
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        for token in scan_tool_tokens(doc.get("content", "")):
            if token in known:
                edges.add(DocToolEdge(doc_id=doc_id, doc_title=title, tool_name=token))
    return tuple(sorted(edges, key=lambda e: (e.doc_id, e.tool_name)))


def documented_but_not_a_tool(
    documents: Iterable[Mapping[str, str]],
    tool_names: Iterable[str],
) -> frozenset[str]:
    """Decoy tokens: written in the corpus, not backed by a tool. Reported, never an edge."""
    known = frozenset(tool_names)
    seen: set[str] = set()
    for doc in documents:
        seen |= scan_tool_tokens(doc.get("content", ""))
    return frozenset(seen - known)


def tools_never_documented(
    documents: Iterable[Mapping[str, str]],
    tool_names: Iterable[str],
) -> frozenset[str]:
    """Suffixed tools no document names — unreachable by reading, so they have no edge."""
    seen: set[str] = set()
    for doc in documents:
        seen |= scan_tool_tokens(doc.get("content", ""))
    return frozenset(t for t in tool_names if TOOL_TOKEN_RE.fullmatch(t) and t not in seen)


def unlockable_tools(edges: Iterable[DocToolEdge]) -> tuple[str, ...]:
    """The tools reachable at all, sorted. The denominator of unlock-and-invoke."""
    return tuple(sorted({e.tool_name for e in edges}))


def docs_naming(edges: Iterable[DocToolEdge], tool_name: str) -> tuple[str, ...]:
    """Every document that unlocks `tool_name`. More than one is common and legitimate:
    the prerequisite is a disjunction (read ANY of them), never a conjunction."""
    return tuple(sorted({e.doc_id for e in edges if e.tool_name == tool_name}))
