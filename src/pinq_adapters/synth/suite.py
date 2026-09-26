"""Synthetic suite, VIEW SIDE.

Reads only data/corpora/synth/<hash>/tasks.jsonl. It has no idea a gold graph exists — the
generator that knows G lives in pi_eval and writes to a different directory. That split is
the same one every real adapter follows, and it is enforced at build time rather than by
convention.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Sequence

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import EvidenceUnit, TaskId, TaskView
from pinq.view import make_view

INSTRUCTIONS = (
    "You may issue retrieval queries against a closed pool of records before answering. "
    "A record can only be found by quoting its identifier."
)


class TokenRetriever:
    """Exact-token retrieval: a record surfaces only when its identifier appears in the query.

    This is the synthetic analogue of tau2's discoverable tools — depth d+1 is unreachable
    until depth d has been read — which is what makes the suite able to distinguish a policy
    that follows a dependency chain from one that merely issues many queries.
    """

    def __init__(
        self,
        units: Sequence[EvidenceUnit],
        tokens: dict[str, str],
        corpus_id: str,
        corpus_hash: str,
    ) -> None:
        self._units = list(units)
        self._tokens = tokens  # uid -> key_token
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        q = query.upper()
        hits = [u for u in self._units if self._tokens.get(u.uid, "\0") in q]
        return hits[:k]


@dataclass
class SynthSuite:
    root: Path
    suite_id: ClassVar[str] = "synth"
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = "synth_v1"
    _idx: dict = field(default_factory=dict, init=False, repr=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        rows = [json.loads(x) for x in (self.root / "tasks.jsonl").read_text().splitlines() if x]
        self._idx = {r["id"]: r for r in rows}
        self.corpus_hash = _corpus_hash(
            (f"{r['id']}:{d['doc_id']}", d["title"], hashlib.sha256(d["text"].encode()).hexdigest())
            for r in rows
            for d in r["docs"]
        )

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._idx)

    def _units(self, tid: TaskId) -> tuple[list[EvidenceUnit], dict[str, str]]:
        units, tokens = [], {}
        for d in self._idx[tid]["docs"]:
            u = EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=d["doc_id"],
                span=f"0:{len(d['text'])}",
                title=d["title"],
                text=d["text"],
            )
            units.append(u)
            tokens[u.uid] = d["key_token"]
        return units, tokens

    def view(self, tid: TaskId) -> TaskView:
        return make_view(
            task_id=tid,
            suite_id=self.suite_id,
            question=self._idx[tid]["question"],
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=60,
        )

    def retriever(self, tid: TaskId) -> TokenRetriever:
        units, tokens = self._units(tid)
        return TokenRetriever(units, tokens, self.corpus_id, self.corpus_hash)

    def actuator(self, tid: TaskId):
        return None
