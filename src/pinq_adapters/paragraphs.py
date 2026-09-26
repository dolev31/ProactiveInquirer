"""The closed-paragraph-pool suite, VIEW SIDE — shared by musique, strategyqa and wiki2.

All three upstream datasets ship the same physical thing: a question plus a fixed pool of
paragraphs, some supporting and some distractors. Their gold differs enormously (a
human-authored decomposition, a strategy, a Wikidata triple chain), but gold lives in
pi_eval and never enters here, so on THIS side the three suites collapse to one class with
different constants.

That collapse is a correctness property, not a tidiness one. The uid of an evidence unit is
h(corpus_id, doc_id, span), and the gold builder mints the same uid independently via
pi_eval.build.common.unit_uid. One shared implementation of "how a paragraph becomes a unit"
means there is exactly one convention to keep in sync instead of three, and each suite's
tests assert the two sides still agree.

Like SynthSuite, this reads only data/corpora/<suite>/<hash>/tasks.jsonl and has no idea a
gold graph exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import EvidenceUnit, TaskId, TaskView
from pinq.view import make_view

from .retrieval.bm25 import BM25Retriever

# The exact key set of a public record. Asserted by every suite's tests, because the whole
# firewall reduces to "no gold key was ever written into data/corpora/".
RECORD_KEYS = frozenset({"id", "question", "paragraphs"})
PARAGRAPH_KEYS = frozenset({"idx", "title", "text"})


@dataclass
class ParagraphSuite:
    """Subclasses supply constants only; behaviour is fixed here on purpose."""

    root: Path
    suite_id: ClassVar[str] = ""
    suite_version: ClassVar[str] = "v1"
    corpus_id: ClassVar[str] = ""
    instructions: ClassVar[str] = ""
    word_cap: ClassVar[int] = 30
    _idx: dict = field(default_factory=dict, init=False, repr=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        rows = [json.loads(x) for x in (self.root / "tasks.jsonl").read_text().splitlines() if x]
        self._idx = {r["id"]: r for r in rows}
        # Order-independent over (doc_id, title, sha256(text)): re-emitting the same corpus
        # with the tasks shuffled must not invalidate a sweep's run identity.
        self.corpus_hash = _corpus_hash(
            (
                f"{r['id']}:{p['idx']}",
                p["title"],
                hashlib.sha256(p["text"].encode()).hexdigest(),
            )
            for r in rows
            for p in r["paragraphs"]
        )

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._idx)

    def units(self, tid: TaskId) -> tuple[EvidenceUnit, ...]:
        """One unit per paragraph. span is the whole paragraph, mirroring common.unit_uid.

        Paragraph-level rather than sentence-level even where upstream annotates sentences
        (2Wiki's supporting_facts carry a sent_id): the retriever returns paragraphs, and a
        gold uid the retriever can never emit is a node that can never be resolved.
        """
        return tuple(
            EvidenceUnit.make(
                corpus_id=self.corpus_id,
                doc_id=f"{tid}:{p['idx']}",
                span=f"0:{len(p['text'])}",
                title=p["title"],
                text=p["text"],
            )
            for p in self._idx[tid]["paragraphs"]
        )

    def view(self, tid: TaskId) -> TaskView:
        return make_view(
            task_id=tid,
            suite_id=self.suite_id,
            question=self._idx[tid]["question"],
            instructions=self.instructions,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=self.word_cap,
        )

    def retriever(self, tid: TaskId) -> BM25Retriever:
        return BM25Retriever(
            self.units(tid), corpus_id=self.corpus_id, corpus_hash=self.corpus_hash
        )

    def actuator(self, tid: TaskId) -> None:
        """No stateful world: these suites are read-only corpora. tau2 and PARE are where an
        Actuator appears, and scoring there reads the EnvCall log rather than the answer."""
        return None
