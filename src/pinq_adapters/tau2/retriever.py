"""Knowledge-base retrieval against a live tau2 Environment.

WHY THIS IS NOT AN Actuator
    A KB search is a READ. tau2 marks these tools `ToolType.READ` and they provably cannot
    move `get_db_hash()`. Every EnvCall this module emits therefore carries
    `mutating=False`, which is what lets tripwire #6 ("mutating EnvCalls outside
    evaluation_criteria.actions must be 0") stay meaningful: if retrieval were logged as
    mutating, that count would never be 0 and the tripwire would be disabled on day one.

WHY THE TOOL NAME IS DISCOVERED AT RUNTIME
    Under the pinned `bm25` variant the tool is `KB_search(query)` and takes NO `k`.
    `KB_search_bm25(query, k)` exists only under `alltools`, which is banned for
    reproducibility. Hard-coding either name breaks against the other, so we ask the
    environment which one it actually exposes and pass `k` only where it is accepted.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from pinq.types import EnvCall, EvidenceUnit

from .discoverable import DocToolEdge, extract_tool_edges

# Preference order. `KB_search_bm25` first so an `alltools` environment still works, but
# `KB_search` is what the reproducible `bm25` arm will actually bind to.
KB_TOOL_PREFERENCE = ("KB_search_bm25", "KB_search")

# tau2 formats every KB hit as:  "<n>. <title>\n   ID: <doc_id>\n   Score: <f>\n   Content: <text>"
# Parsed rather than re-derived because the tool returns a formatted string, not structured
# results (upstream's own TODO). Pure and fixture-tested so a format drift is a red test.
_HIT_RE = re.compile(
    r"^\s*\d+\.\s*(?P<title>.*?)\n"
    r"\s*ID:\s*(?P<doc_id>\S+)\n"
    r"\s*Score:\s*(?P<score>[-\d.eE+]+)\n"
    r"\s*Content:\s*(?P<content>.*?)"
    r"(?=\n\s*\d+\.\s|\n\s*\[Timing:|\Z)",
    re.S | re.M,
)


def parse_kb_results(blob: str) -> tuple[dict[str, Any], ...]:
    """Turn tau2's formatted KB_search string into {doc_id, title, score, content} records."""
    if not blob or "No relevant documents found" in blob:
        return ()
    out = []
    for m in _HIT_RE.finditer(blob):
        try:
            score = float(m.group("score"))
        except ValueError:
            score = 0.0
        out.append(
            {
                "doc_id": m.group("doc_id").strip(),
                "title": m.group("title").strip(),
                "score": score,
                "content": m.group("content").strip(),
            }
        )
    return tuple(out)


def pick_kb_tool(tool_names: Sequence[str]) -> str:
    """Which KB search tool this environment exposes. Raises rather than guessing."""
    names = set(tool_names)
    for cand in KB_TOOL_PREFERENCE:
        if cand in names:
            return cand
    raise RuntimeError(
        f"no KB search tool among {KB_TOOL_PREFERENCE} in this environment; "
        f"saw {sorted(names)[:12]}... — is the retrieval_variant set to 'no_knowledge'?"
    )


class Tau2Retriever:
    """Satisfies pinq.protocols.Retriever over a tau2 Environment.

    Holds its own EnvCall log: run_loop drives retrieval but only the Actuator's calls
    reach the Outcome, so a retriever that did not record would leave the read half of the
    interaction unscoreable.
    """

    def __init__(
        self,
        env: Any,
        *,
        corpus_id: str,
        corpus_hash: str,
        tool_name: str | None = None,
        tool_edges: Sequence[DocToolEdge] = (),
    ) -> None:
        self._env = env
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash
        self._tool = tool_name or pick_kb_tool([t.name for t in env.get_tools()])
        self._accepts_k = self._tool != "KB_search"
        self._calls: list[EnvCall] = []
        self._seq = 0
        # doc_id -> tools it unlocks, so a read can be credited with what it made reachable
        self._unlocks: dict[str, set[str]] = {}
        for e in tool_edges:
            self._unlocks.setdefault(e.doc_id, set()).add(e.tool_name)
        self.turn_idx = 0

    @property
    def env_calls(self) -> tuple[EnvCall, ...]:
        return tuple(self._calls)

    @property
    def unlocked_by_reading(self) -> frozenset[str]:
        """Tools whose naming document this run has actually retrieved.

        This is the objective precondition for `unlock_discoverable_agent_tool`, and it is
        the numerator the discoverable-tool endpoint is built on.
        """
        out: set[str] = set()
        for c in self._calls:
            for doc_id in c.result_digest.split(",") if c.result_digest else ():
                out |= self._unlocks.get(doc_id, set())
        return frozenset(out)

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        from pinq.ids import canon

        kwargs: dict[str, Any] = {"query": query}
        if self._accepts_k:
            kwargs["k"] = k
        ok = True
        try:
            raw = self._env.make_tool_call(self._tool, requestor="assistant", **kwargs)
        except Exception as exc:  # a failed read is data, not a crash: log it and return ()
            ok, raw = False, f"ERROR: {exc}"
        hits = parse_kb_results(raw if isinstance(raw, str) else str(raw))[:k]

        self._seq += 1
        self._calls.append(
            EnvCall(
                seq=self._seq,
                turn_idx=self.turn_idx,
                requestor="assistant",
                tool_name=self._tool,
                kwargs_json=canon(kwargs),
                ok=ok,
                # the doc_ids ARE the digest: it is what makes the read auditable offline
                result_digest=",".join(h["doc_id"] for h in hits),
                mutating=False,
            )
        )
        return tuple(
            EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=h["doc_id"],
                # whole-document span, same convention as every other adapter, so the uid
                # is a pure function of content location and stays stable across runs
                span=f"0:{len(h['content'])}",
                title=h["title"],
                text=h["content"],
                score=h["score"],
            )
            for h in hits
        )


def env_tool_names(env: Any) -> frozenset[str]:
    """Every tool that EXISTS in this environment, visible or not.

    `get_tools()` alone is the wrong set and returns an empty edge list. That is the
    discoverability mechanic working as designed: the 44 suffixed tools are deliberately
    HIDDEN from the advertised tool list (only 15 tools are visible, among them
    `unlock_discoverable_agent_tool`) precisely so they cannot be found without reading.
    The toolkit's own `get_discoverable_tools()` is the authoritative registry, and it
    already excludes the `downgrade_credit_card_3847` decoy.
    """
    names: set[str] = {t.name for t in env.get_tools()}
    toolkit = getattr(env, "tools", None)
    getter = getattr(toolkit, "get_discoverable_tools", None)
    if getter is not None:
        try:
            names |= set(getter())
        except Exception:
            pass
    return frozenset(names)


def edges_for_env(documents: Sequence[dict], env: Any) -> tuple[DocToolEdge, ...]:
    """Extract doc->tool edges against the tools THIS environment really has.

    Cross-referencing against the live registry rather than a static allowlist is what
    drops the decoy without hand-maintaining an exception list.
    """
    return extract_tool_edges(documents, env_tool_names(env))
