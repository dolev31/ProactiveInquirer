"""POST /retrieve — Search-R1's retrieval contract, over OUR frozen corpus.

WHY BYTE-COMPATIBILITY IS WORTH THE FEW LINES IT COSTS. Search-R1, ReCall and Search-o1 all
speak one small protocol: post `{"queries": [...], "topk": k, "return_scores": bool}`, get
back `{"result": [[{"document": {"id", "contents"}, "score"}, ...], ...]}`. Matching it
exactly means those baselines can be pointed at this repo's corpus with a URL change and no
code change — so "we compared against Search-R1" means the same documents, the same
tokenizer and the same k, rather than the same citation. A baseline run against a different
index is not a baseline; it is a second experiment reported as a control.

WHAT IS ADDITIVE, AND WHY. Two request fields (`suite_id`, `task_id`) select which frozen
corpus answers; omit them and the server's configured default answers, so an unmodified
upstream client still works. One response field (`uid`) carries
`EvidenceUnit.uid = h(corpus_id, doc_id, span)`, which is what lets a retrieval made through
this endpoint join back to `evidence.parquet`. A doc id cannot do that job: identity in this
codebase is `(corpus_id, doc_id, span)` and nothing else, precisely so that hashing a
model-produced string into it is impossible.

NO GOLD IS REACHABLE FROM HERE. This endpoint reads a corpus, which is a public, frozen,
content-addressed artifact. It never imports `pi_eval`, and the retriever it uses is the same
object a rollout worker uses, so an external baseline and our own arms are served identical
bytes for identical queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pinq.wire import RetrievedDoc, RetrieveRequest, RetrieveResponse


class TaskNotSpecified(RuntimeError):
    """No task named and no default. Refused rather than answered from an arbitrary task."""


class SuiteNotConfigured(RuntimeError):
    """The request named no suite and the server has no default. Refused rather than guessed:
    silently answering from the wrong corpus produces a baseline number that looks fine."""


class RetrieverPool:
    """Suite adapters and per-task retrievers, built once and reused.

    A retriever holds a parsed corpus; rebuilding it per request would turn a BM25 lookup
    into a corpus load and make the endpoint's latency a property of the caller's request
    rate. Keyed by `(suite_id, task_id)` because retrieval in this codebase is scoped to a
    task's pool, which is what makes discovery metrics well defined.
    """

    def __init__(
        self,
        *,
        root: str | Path = ".",
        default_suite: str | None = None,
        default_task: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.default_suite = default_suite
        self.default_task = default_task
        self._suites: dict[str, Any] = {}
        self._retrievers: dict[tuple[str, str], Any] = {}

    def suite(self, suite_id: str) -> Any:
        if suite_id not in self._suites:
            from pi_run.serve.rollout import resolve_corpus
            from pi_run.worker import load_suite

            self._suites[suite_id] = load_suite(suite_id, str(resolve_corpus(self.root, suite_id)))
        return self._suites[suite_id]

    def retriever(self, suite_id: str, task_id: str | None) -> Any:
        """The retriever for ONE task. Refuses when no task is named.

        This used to fall back to `str(s.task_ids()[0])` -- an arbitrary task, whichever the
        corpus happened to list first. On the paragraph suites that is not a mild default: every
        MuSiQue and 2Wiki task carries its OWN closed pool of ~20 paragraphs, so a query for
        task B answered from task A's pool retrieves paragraphs about entirely unrelated
        entities, at full BM25 confidence, and returns them as evidence. Nothing downstream can
        tell those from a genuine retrieval.

        `resolve` already refuses a missing SUITE with exactly this reasoning -- "answering from
        an arbitrary corpus would produce a baseline number that looks correct and is not" --
        and then answered from an arbitrary task inside it.
        """
        s = self.suite(suite_id)
        tid = task_id or (self.default_task or "")
        if not tid:
            raise TaskNotSpecified(
                f"no task_id in the request and no default configured for suite {suite_id!r}. "
                "Each task in a paragraph suite has its own closed paragraph pool, so answering "
                "from another task's retriever returns confident, well-formed, unrelated "
                "evidence. Send task_id, or start the server with --task."
            )
        key = (suite_id, tid)
        if key not in self._retrievers:
            self._retrievers[key] = s.retriever(tid)
        return self._retrievers[key]

    def resolve(self, req: RetrieveRequest) -> Any:
        suite_id = req.suite_id or self.default_suite
        if not suite_id:
            raise SuiteNotConfigured(
                "no suite_id in the request and no default configured. Start the server with "
                "--suite, or send suite_id; answering from an arbitrary corpus would produce a "
                "baseline number that looks correct and is not."
            )
        return self.retriever(suite_id, req.task_id)


def handle_retrieve(req: RetrieveRequest, *, pool: RetrieverPool) -> RetrieveResponse:
    retriever = pool.resolve(req)
    k = max(1, int(req.topk))
    batches: list[tuple[RetrievedDoc, ...]] = []
    for query in req.queries:
        units = list(retriever.search(str(query), k))[:k]
        batches.append(
            tuple(
                RetrievedDoc(
                    # Upstream's shape exactly: `contents` is title-then-text, which is what
                    # Search-R1's prompt templates expect to slice.
                    document={
                        "id": u.doc_id,
                        "contents": f'"{u.title}"\n{u.text}' if u.title else u.text,
                    },
                    score=float(u.score) if req.return_scores else 0.0,
                    uid=u.uid,
                )
                for u in units
            )
        )
    return RetrieveResponse(result=tuple(batches))
