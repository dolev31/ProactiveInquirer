"""DeepResearchGym suite, VIEW SIDE: Researchy queries against a hosted open-web index.

Reads only data/corpora/drgym/<hash>/tasks.jsonl -- `{"id", "question"}` per line, produced
by pi_eval.build.drgym_build from the benchmark's own query list. The gold key points live
in data/gold/ and nothing here knows they exist.

THIS IS THE ONLY OPEN-CORPUS SUITE, and three of its properties follow from that:

  1. `retriever()` is a NETWORK CALL. Every other suite indexes a pool that shipped with the
     task; this one queries a hosted index that needs an API key. So the suite has an
     OFFLINE MODE (`offline=True`, the default) which replays a content-addressed cache and
     raises on a miss. Every test in this repo runs in that mode, and so does CI.

  2. `corpus_hash` PINS THE QUERY SET AND THE ENDPOINT, NOT THE DOCUMENTS. It is
     h(query set, corpus name, endpoint path). It is honest about what it can certify: the
     document set behind a hosted index can change under us and no client can detect it.
     The reproducible artifact for a DRGym rollout is therefore the search cache, and
     `evidence.subset_hash` over the replayed units -- not the corpus hash -- is what says
     two runs saw the same evidence.

  3. There is NO closed-pool ceiling. In musique a policy can in principle retrieve all 20
     paragraphs; here recall is unbounded and every metric that divides by "the pool" is
     undefined. That is why this suite's endpoint is key-point recall (judged, in pi_eval)
     rather than a pool-relative retrieval number.

WORD CAP. Reports are the unit of output, so the cap is 1000 words rather than the 30 of a
span-answer suite. It is frozen across arms for the same reason as everywhere else: an LLM
judge pays roughly +0.3-0.8 Likert per doubling of length, so a cap that varies by arm makes
length the thing being measured.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Sequence

from pinq.ids import corpus_hash as _corpus_hash
from pinq.types import EvidenceUnit, TaskId, TaskView
from pinq.view import make_view

from .cache import CachingSearch, ReplaySearch, SearchCache, ShardedJsonCache
from .client import CORPUS_IDS, DrGymClient, corpus_of

# The exact key set of a public record. Asserted by the tests: the firewall reduces to "no
# gold key was ever written into data/corpora/".
RECORD_KEYS = frozenset({"id", "question"})

WORD_CAP = 1000
DEFAULT_K = 10

INSTRUCTIONS = (
    "Research this question with a web search tool and write a report. Cite your sources by "
    "including the literal URL of each source next to the claim it supports; a report with "
    "no URLs cannot be scored for support at all."
)


def default_cache_root(env: dict[str, str] | None = None) -> Path:
    """Same root the runner uses. Keys are domain-separated, so sharing it is safe."""
    e = env if env is not None else os.environ
    p = Path(e.get("PI_CACHE_ROOT") or "cache")
    return p if p.is_absolute() else Path.cwd() / p


class HostedRetriever:
    """Satisfies pinq.protocols.Retriever structurally, over either search backend."""

    def __init__(self, backend: Any, *, corpus_id: str, corpus_hash: str) -> None:
        self._backend = backend
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        return self._backend.search(query, k)


@dataclass
class DrGymSuite:
    """`offline=True` is the default because the network is the exception here, not the rule.

    `cache` accepts anything with DiskCache's get/put shape -- the runner passes
    pi_run.cache.DiskCache directly.
    """

    root: Path
    corpus: str = "fineweb"
    # The runner sets this from `pi_run.worker.drgym_offline`, i.e. from PI_DRGYM_ONLINE.
    # It stays True unless that is set explicitly: holding a DRGYM_API_KEY does NOT flip a
    # confirmatory sweep from replaying its recorded cache to re-querying a live index.
    offline: bool = True
    cache: SearchCache | None = None
    cache_root: Path | None = None
    client: DrGymClient | None = None
    suite_id: ClassVar[str] = "drgym"
    suite_version: ClassVar[str] = "v1"
    _idx: dict = field(default_factory=dict, init=False, repr=False)
    corpus_id: str = field(default="", init=False)
    corpus_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        spec = corpus_of(self.corpus)
        self.corpus_id = CORPUS_IDS[spec.name]
        rows = [json.loads(x) for x in (self.root / "tasks.jsonl").read_text().splitlines() if x]
        self._idx = {r["id"]: r for r in rows}
        # (task id, corpus name, sha256(query)) triples, order-independent. What this pins is
        # the QUERY SET plus which hosted index was asked -- see the module docstring for
        # what it deliberately cannot pin.
        self.corpus_hash = _corpus_hash(
            (
                r["id"],
                f"{spec.name}{spec.path}",
                hashlib.sha256(r["question"].encode()).hexdigest(),
            )
            for r in rows
        )
        if self.cache is None:
            self.cache = ShardedJsonCache(self.cache_root or default_cache_root())

    # ------------------------------------------------------------------ TaskSuite

    def task_ids(self) -> tuple[TaskId, ...]:
        return tuple(self._idx)

    def view(self, tid: TaskId) -> TaskView:
        return make_view(
            task_id=tid,
            suite_id=self.suite_id,
            question=self._idx[tid]["question"],
            instructions=INSTRUCTIONS,
            corpus_id=self.corpus_id,
            corpus_hash=self.corpus_hash,
            word_cap=WORD_CAP,
        )

    def retriever(self, tid: TaskId) -> HostedRetriever:
        """One backend for every task: the index is global, not per-task.

        Offline replays; online reads through the same cache, so the first live run IS the
        recording and no separate capture step exists to forget to run.
        """
        spec = corpus_of(self.corpus)
        assert self.cache is not None  # set in __post_init__
        if self.offline:
            backend: Any = ReplaySearch(self.cache, corpus=spec, corpus_id=self.corpus_id)
        else:
            backend = CachingSearch(self._client(), self.cache)
        return HostedRetriever(backend, corpus_id=self.corpus_id, corpus_hash=self.corpus_hash)

    def actuator(self, tid: TaskId) -> None:
        """No stateful world: search is a read-only corpus behind HTTP."""
        return None

    # ------------------------------------------------------------------ probe

    def _client(self) -> DrGymClient:
        if self.client is None:
            self.client = DrGymClient(corpus=self.corpus)
        return self.client

    def available(self) -> tuple[bool, str]:
        """(usable, reason). Offline mode is always usable; online needs the key."""
        if self.offline:
            return True, f"offline replay from {type(self.cache).__name__}; no key required"
        return self._client().available()
