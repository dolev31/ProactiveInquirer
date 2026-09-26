"""Content-addressed search cache: what makes an OPEN-corpus rollout replayable.

WHY THIS EXISTS AT ALL. Every other suite here retrieves from a file whose sha256 is pinned,
so "re-run the scorer two years later" is a local operation. DRGym retrieves from a hosted
index over a live web corpus: the documents behind a query can change, the service can go
away, and no client can prove otherwise. The only artifact that can be frozen is THE
RESPONSE, so every search is written to a content-addressed store and the offline mode
replays it. A DRGym number in the paper is reproducible from this cache or it is not
reproducible at all, and saying that plainly is better than implying a pin we do not have.

WHY IT IS BYTE-COMPATIBLE WITH pi_run.cache.DiskCache AND STILL DOES NOT IMPORT IT.
Same layout (`<root>/<sha[:2]>/<sha>.json`), same canonical JSON encoding, same
tempfile + os.replace write, so `pi_run.cache.DiskCache` is a drop-in for the `cache=`
argument and the runner passes exactly that object -- which is what
tests/test_adapters_drgym.py asserts by writing with DiskCache and reading back through the
suite. What it does NOT do is import pi_run: `pinq_adapters` is a leaf that reads
`data/corpora/` and hits its own benchmark, and `pi_run` imports `pi_eval.schema` for the
parquet contract. Importing upward would put the gold package one hop from every adapter
and make the firewall contract depend on which module a run-layer refactor happens to touch.

WHY THE KEY OMITS THE HOST. The key is h("drgym_search", corpus, endpoint, query, k) --
the REQUEST, domain-separated so it can share a cache root with LLM request_sha entries
without colliding. `DRGYM_BASE_URL` is deployment (a mirror, a local replica, a proxy), not
content: pointing at a mirror of the same index must hit the same entry, or a replay is
tied to whichever hostname happened to serve it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from pinq.ids import h
from pinq.types import EvidenceUnit

from .client import Corpus, DrGymClient, docs_from_envelope, units_of


class SearchCacheMiss(KeyError):
    """Raised by ReplaySearch when a query was never recorded.

    Loud on purpose, exactly like pi_run.cache.CacheMiss: in offline mode a miss means this
    code path would have made a network call, so it must be a stack trace rather than an
    empty evidence set that quietly scores as "the policy found nothing".
    """


def search_key(*, corpus: str, endpoint: str, query: str, k: int) -> str:
    return h("drgym_search", corpus, endpoint, query, str(int(k)))


class SearchCache(Protocol):
    """Structural: pi_run.cache.DiskCache satisfies it without knowing this file exists."""

    def get(self, sha: str) -> dict[str, Any] | None: ...

    def put(self, sha: str, record: Mapping[str, Any]) -> Any: ...


class ShardedJsonCache:
    """The default store, when the runner does not pass its own.

    Deliberately a copy of pi_run.cache.DiskCache's on-disk behaviour rather than a subclass:
    the compatibility that matters is the BYTES on disk, and that is asserted by a test that
    writes with one class and reads with the other.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path(self, sha: str) -> Path:
        return self.root / sha[:2] / f"{sha}.json"

    def get(self, sha: str) -> dict[str, Any] | None:
        try:
            return json.loads(self.path(sha).read_bytes())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None  # external damage; the entry is regenerable by definition

    def put(self, sha: str, record: Mapping[str, Any]) -> Path:
        p = self.path(sha)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{sha[:8]}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p


def record_of(*, corpus: Corpus, query: str, k: int, envelope: Mapping[str, Any]) -> dict[str, Any]:
    """What is stored: the API's OWN envelope, verbatim, plus the request that produced it.

    Storing the raw base64 envelope rather than normalised documents means a replay runs the
    decoder and the per-corpus field mapping too, so a change to either is caught by the
    replay instead of being frozen into the cache.
    """
    return {
        "corpus": corpus.name,
        "endpoint": corpus.path,
        "query": query,
        "k": int(k),
        "envelope": dict(envelope),
    }


class ReplaySearch:
    """Cache-only retrieval: a miss RAISES instead of dispatching. The offline mode."""

    def __init__(self, cache: SearchCache, *, corpus: Corpus, corpus_id: str) -> None:
        self._cache = cache
        self.corpus = corpus
        self.corpus_id = corpus_id
        self.hits = 0

    def search(self, query: str, k: int) -> tuple[EvidenceUnit, ...]:
        sha = search_key(corpus=self.corpus.name, endpoint=self.corpus.path, query=query, k=k)
        rec = self._cache.get(sha)
        if rec is None:
            raise SearchCacheMiss(
                f"offline replay: no cached search for corpus={self.corpus.name} k={k} "
                f"query={query!r} (key {sha}). Set PI_DRGYM_ONLINE=1 (with DRGYM_API_KEY) "
                "to fetch and record it, or point PI_CACHE_ROOT at the released cache. "
                "Holding a key alone does NOT go online: a confirmatory sweep replays."
            )
        self.hits += 1
        return units_of(docs_from_envelope(rec["envelope"], self.corpus), self.corpus_id)


class CachingSearch:
    """Read-through cache around a live DrGymClient. The record-once, replay-forever path."""

    def __init__(self, client: DrGymClient, cache: SearchCache) -> None:
        self._client = client
        self._cache = cache
        self.corpus = client.corpus
        self.corpus_id = client.corpus_id
        self.hits = 0
        self.misses = 0

    def search(self, query: str, k: int) -> tuple[EvidenceUnit, ...]:
        sha = search_key(corpus=self.corpus.name, endpoint=self.corpus.path, query=query, k=k)
        rec = self._cache.get(sha)
        if rec is None:
            self.misses += 1
            envelope = self._client.fetch(query, k)
            rec = record_of(corpus=self.corpus, query=query, k=k, envelope=envelope)
            self._cache.put(sha, rec)
        else:
            self.hits += 1
        return units_of(docs_from_envelope(rec["envelope"], self.corpus), self.corpus_id)
