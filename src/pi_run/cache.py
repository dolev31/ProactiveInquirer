"""Content-addressed LLM response cache. Sharded, atomic, and completely lock-free.

WHY THE KEY IS NOT THE RUN IDENTITY. The cache key is `pinq.ids.request_sha(payload)` — a
hash of the exact serialized request. It is deliberately a different hash from
`RunManifest.semantic_hash`, which is run identity. Fusing them looks tidier and is a trap:
semantic_hash includes code_version, budget_cap and every prompt hash, so fixing a typo in
one prompt would invalidate the cache for a 30k-rollout sweep. Keeping them separate means a
prompt typo evicts exactly the entries whose bytes changed.

WHY THERE IS NO LOCK, AND WHY IT IS FIRST-WRITER-WINS. Writes go through a tempfile in the same
directory plus `os.link`, which is atomic on APFS and on every POSIX filesystem we run on.

This used to say two racers "are computing the same content-addressed value, so they write
byte-identical files and last-writer-wins is not merely tolerable, it is correct". THAT
ARGUMENT IS FALSE, and docs/REPRODUCE.md says so in its own opening paragraph: "Temperature 0
is not bitwise deterministic under batched inference -- two identical requests to the same
provider can differ." The stored record holds `text`, `response_sha`, `tok_completion`,
`tok_reasoning` and `usd`, every one of which differs between two racers that got different
text. Measured:

    racer A: "Paris is the capital."   response_sha sha_a, tok_completion 7, usd 0.00021
    racer B: "The capital is Paris."   response_sha sha_b, tok_completion 8, usd 0.00024
    byte-identical? False
    after put(A) then put(B), a replay of RUN A returns run B's text and B's response_sha.

The key is the request payload and the Answerer is FROZEN across every arm by design, so two
arms reaching the same (view, evidence, seed) issue byte-identical payloads and collide on one
key by construction. Under last-writer-wins the artifact's central claim -- "given the cache,
every number regenerates" -- is false whenever a race occurred.

So the FIRST response observed for a request is the canonical one and is never overwritten:
`os.link` fails with FileExistsError if the entry already exists, which is race-free rather
than a check-then-write. A replay is therefore stable, and re-running a sweep cannot silently
change what a completed run replays to.

The record still excludes wall_ms, ttft_ms, retries and rate_limit_stall_ms: those are machine
artifacts, they are meaningless to replay, and they are re-synthesised on read.

No SQLite: a single writer lock across 64 processes is the one contention point this design
does not need, and a corrupted db file loses an entire sweep's cache rather than one entry.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from pinq import tripwire
from pinq.ids import canon, h, request_sha
from pinq.types import CallTelemetry

DEFAULT_CACHE_ROOT = "cache"

# Stored verbatim; everything else in CallTelemetry is a machine artifact. See module docstring.
_DETERMINISTIC_FIELDS = (
    "model",
    "provider",
    "request_sha",
    "response_sha",
    "tok_prompt",
    "tok_completion",
    "tok_reasoning",
    "tok_cached",
    "usd",
    "http_status",
)


class CacheMiss(KeyError):
    """Raised by ReplayClient when a request is not already cached.

    This is the whole point of ReplayClient: in CI a miss means the test would have made a
    network call, so it must be a loud failure rather than a $40 invoice.
    """


def cache_root(root: str | os.PathLike[str] | None = None) -> Path:
    p = Path(root or os.environ.get("PI_CACHE_ROOT") or DEFAULT_CACHE_ROOT)
    return p if p.is_absolute() else Path.cwd() / p


@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    shards: int
    bytes: int
    root: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "shards": self.shards,
            "bytes": self.bytes,
            "root": self.root,
        }


def _serialize(record: Mapping[str, Any]) -> bytes:
    """One canonical encoding, so two independent writers of the same key agree bit for bit."""
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


class DiskCache:
    """Layout: <root>/<sha[:2]>/<sha>.json — 256 shards, which keeps any one directory small
    enough that a 30k-rollout sweep does not turn `ls` into a minute-long operation."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = cache_root(root)

    def path(self, sha: str) -> Path:
        return self.root / sha[:2] / f"{sha}.json"

    def get(self, sha: str) -> dict[str, Any] | None:
        p = self.path(sha)
        try:
            return json.loads(p.read_bytes())
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            # A torn file cannot happen through put(), so this is external damage. Treat it
            # as a miss rather than crashing a sweep: the entry is regenerable by definition.
            return None

    def put_or_get(self, sha: str, record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Write, and report what the cache KEPT: None if we won, the winner's record if not.

        `put` returned a Path whether it won or lost, so a caller could not tell -- which is
        why a losing racer went on returning its own text, and why 41 recorded calls held a
        `response_sha` the cache does not have. The distinction is the whole fix; the write
        itself is unchanged and still race-free via `os.link`.
        """
        self.put(sha, record)
        kept = self.get(sha)
        if kept is None:
            return None  # unreadable entry; `get` already treats that as a miss
        if str(kept.get("response_sha", "")) == str(record.get("response_sha", "")):
            return None
        return kept

    def put(self, sha: str, record: Mapping[str, Any]) -> Path:
        p = self.path(sha)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize(record)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{sha[:8]}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            # FIRST WRITER WINS, race-free. `os.link` refuses when the target exists, so two
            # racers cannot overwrite each other and a replay of a completed run keeps
            # returning the text that run actually received. See the module docstring for why
            # last-writer-wins was unsound: the stored record is NOT identical between racers.
            try:
                os.link(tmp, p)
            except FileExistsError:
                pass  # another worker got there first; its answer is the canonical one
            os.unlink(tmp)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return p

    def __contains__(self, sha: str) -> bool:
        return self.path(sha).exists()

    def iter_records(self) -> Iterator[tuple[str, dict[str, Any]]]:
        if not self.root.exists():
            return
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir():
                continue
            for f in sorted(shard.glob("*.json")):
                try:
                    yield f.stem, json.loads(f.read_bytes())
                except json.JSONDecodeError:
                    continue

    def stats(self) -> CacheStats:
        entries = 0
        total = 0
        shards = 0
        if self.root.exists():
            for shard in self.root.iterdir():
                if not shard.is_dir():
                    continue
                shards += 1
                for f in shard.glob("*.json"):
                    entries += 1
                    total += f.stat().st_size
        return CacheStats(entries=entries, shards=shards, bytes=total, root=str(self.root))


# --------------------------------------------------------------------------- clients


def _payload_of(inner: Any, role: str, messages: list[Mapping[str, Any]], seed: int, mt, kw):
    """Use the wrapped client's own serializer when it has one.

    Reconstructing the payload here instead would let the cache key drift from the bytes that
    are actually sent, and a cache whose key is not the request is worse than no cache.
    """
    fn = getattr(inner, "request_payload", None)
    if fn is not None:
        return fn(role=role, messages=messages, seed=seed, max_tokens=mt, **kw)
    return {
        "model": getattr(inner, "model", role),
        "messages": [dict(m) for m in messages],
        "seed": seed,
        "max_tokens": mt,
        **{k: v for k, v in sorted(kw.items())},
    }


def _record_from(text: str, tel: CallTelemetry) -> dict[str, Any]:
    rec: dict[str, Any] = {f: getattr(tel, f) for f in _DETERMINISTIC_FIELDS}
    rec["text"] = text
    rec["actor"] = tel.actor
    return rec


def _telemetry_from(rec: Mapping[str, Any], *, actor: str, wall_ms: int, seq: int) -> CallTelemetry:
    """A hit is charged its recorded tokens and USD, with cache_hit=True.

    Charging as-if keeps arms comparable no matter how warm the cache happened to be; the
    real invoice is recoverable at any time as `sum(usd) where not cache_hit`.
    """
    return CallTelemetry(
        call_id=h("call", str(rec.get("request_sha", "")), "hit", str(seq))[:32],
        actor=actor,  # type: ignore[arg-type]
        model=str(rec.get("model", "")),
        provider=str(rec.get("provider", "")),
        request_sha=str(rec.get("request_sha", "")),
        response_sha=str(rec.get("response_sha", "")),
        tok_prompt=int(rec.get("tok_prompt", 0)),
        tok_completion=int(rec.get("tok_completion", 0)),
        tok_reasoning=int(rec.get("tok_reasoning", 0)),
        tok_cached=int(rec.get("tok_cached", 0)),
        usd=float(rec.get("usd", 0.0)),
        wall_ms=wall_ms,
        ttft_ms=0,
        cache_hit=True,
        retries=0,
        rate_limit_stall_ms=0,
        http_status=int(rec.get("http_status", 200)),
    )


class CachingClient:
    """Read-through cache around any pinq.protocols.LLM.

    Like MeteredClient it exposes no ledger access; the hit/miss counters it does expose are
    about the cache, not about spend, and are what `pi cache stats` reports.
    """

    def __init__(self, inner: Any, cache: DiskCache | None = None, *, ledger: Any = None) -> None:
        self._inner = inner
        self._cache = cache or DiskCache()
        self._ledger = ledger
        self.hits = 0
        self.misses = 0
        # Lost write races. A correctness event nobody could audit before: it is the count
        # of calls we paid for whose answer was discarded in favour of the canonical one.
        self.races = 0

    def pin(self, role: str) -> Any:
        """Delegate to the wrapped client.

        A pure passthrough, and it has to exist: the worker asks the client it HOLDS for the
        model pins that go into the manifest, and the client it holds is this wrapper. Without
        the delegation the worker saw no `pin` attribute, wrote `"pins": {}`, and produced runs
        whose identity was blind to which checkpoint had answered.
        """
        return self._inner.pin(role)

    def request_payload(self, **kw: Any) -> dict[str, Any]:
        return _payload_of(
            self._inner,
            kw["role"],
            kw["messages"],
            kw["seed"],
            kw.get("max_tokens"),
            {k: v for k, v in kw.items() if k not in ("role", "messages", "seed", "max_tokens")},
        )

    def key_for(self, **kw: Any) -> str:
        return request_sha(self.request_payload(**kw))

    def complete(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, CallTelemetry]:
        actor = kw.get("actor", role)
        payload = _payload_of(self._inner, role, messages, seed, max_tokens, kw)
        # FIREWALL LAYER 4, ON THE PATH THAT ACTUALLY RUNS.
        #
        # `tripwire.assert_clean` had exactly one call site, inside `MeteredClient.complete` --
        # and the block below returns from the cache BEFORE reaching it. `ReplayClient` never
        # calls through at all. So on the path docs/REPRODUCE.md ships as the artifact ("given
        # the cache, every number regenerates with no network"), the only layer that catches a
        # gold string reaching a model was 100% dark: a nonce in a prompt that hit the cache was
        # scanned by nothing.
        #
        # Scanned here, on the payload that is about to become a key, so a hit, a miss and a
        # replay are all covered by one check. Cheap: `scan` short-circuits on a single
        # substring test for the shared prefix.
        tripwire.assert_clean(canon(payload), where=f"a cached {role} request")
        sha = request_sha(payload)
        t0 = time.perf_counter()
        rec = self._cache.get(sha)
        if rec is not None:
            self.hits += 1
            wall = int((time.perf_counter() - t0) * 1000)
            tel = _telemetry_from(rec, actor=actor, wall_ms=wall, seq=self.hits)
            if self._ledger is not None:
                self._ledger.record_call(tel)
            return str(rec.get("text", "")), tel
        self.misses += 1
        text, tel = self._inner.complete(
            role=role, messages=messages, seed=seed, max_tokens=max_tokens, **kw
        )
        kept = self._cache.put_or_get(sha, _record_from(text, tel))
        if kept is not None:
            # WE LOST THE RACE, so we return what the cache KEPT rather than what we got.
            #
            # Temperature 0 is not bitwise deterministic and the Answerer is frozen across
            # arms by design, so two workers issuing byte-identical payloads concurrently
            # routinely receive DIFFERENT text. First-writer-wins made the cache stable but
            # left the loser returning its own answer anyway -- so the losing run acted on,
            # and recorded, a response the cache does not hold, and replaying it produced the
            # winner's text instead. `pi cache verify` measured 41 such calls in 11,681, 14 of
            # them in the P3 grid. That is "given the cache, every number regenerates" being
            # false, which is the artifact's central claim.
            #
            # Only `text` and `response_sha` are adopted. The call really was dispatched and
            # really was billed, so tokens and usd stay ours: the ledger is debited by the
            # inner client (litellm_client.py:361) and reconciliation compares token SUMS,
            # so refunding here would make the ledger and the trajectory disagree.
            self.races += 1
            text = str(kept.get("text", text))
            tel = replace(tel, response_sha=str(kept.get("response_sha", tel.response_sha)))
        return text, tel


class ReplayClient:
    """Cache-only client: a miss RAISES instead of dispatching.

    It wraps an LLM so that the key it looks up is byte-identical to the key the real client
    would have written — but it never calls through. That is what makes the whole loop, every
    arm and the sweep runner testable offline and deterministically in CI, and turns an
    accidental $40 test run into a stack trace on line one.
    """

    def __init__(self, inner: Any, cache: DiskCache | None = None, *, ledger: Any = None) -> None:
        self._inner = inner
        self._cache = cache or DiskCache()
        self._ledger = ledger
        self.hits = 0

    def pin(self, role: str) -> Any:
        """Delegate to the wrapped client. A replayed judgment still has to say which model
        produced the bytes it replayed, or `scorer_hash` would not carry the instrument."""
        return self._inner.pin(role)

    def request_payload(self, **kw: Any) -> dict[str, Any]:
        return _payload_of(
            self._inner,
            kw["role"],
            kw["messages"],
            kw["seed"],
            kw.get("max_tokens"),
            {k: v for k, v in kw.items() if k not in ("role", "messages", "seed", "max_tokens")},
        )

    def complete(
        self,
        *,
        role: str,
        messages: list[Mapping[str, Any]],
        seed: int,
        max_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, CallTelemetry]:
        actor = kw.get("actor", role)
        payload = _payload_of(self._inner, role, messages, seed, max_tokens, kw)
        # Layer 4. A replay never reaches MeteredClient, so without this the offline
        # reproduction path -- the one the artifact ships -- scans nothing. See CachingClient.
        tripwire.assert_clean(canon(payload), where=f"a replayed {role} request")
        sha = request_sha(payload)
        rec = self._cache.get(sha)
        if rec is None:
            raise CacheMiss(
                f"replay-only client: {sha} is not cached (role={role}). "
                "A miss here means this code path would have made a network call."
            )
        self.hits += 1
        tel = _telemetry_from(rec, actor=actor, wall_ms=0, seq=self.hits)
        if self._ledger is not None:
            self._ledger.record_call(tel)
        return str(rec.get("text", "")), tel
