"""A racer that loses the write must return what the cache KEPT, not what it received.

`pi cache verify` joins calls.parquet to the cache and found 41 of 11,681 recorded calls
where the run's `response_sha` is not the one the cache holds for that request
(`missing_from_cache: 0` -- disagreement, not absence). 14 of those runs are from the
2026-08-28 P3 grid, i.e. AFTER first-writer-wins landed on 2026-08-25 (5dc35d8), so this is
live behaviour and not historical dirt.

The mechanism is the one the cache's own docstring documents and then only half-fixes.
Temperature 0 is not bitwise deterministic (docs/REPRODUCE.md, opening paragraph), and the
Answerer is frozen across arms by design, so two workers routinely issue byte-identical
payloads concurrently and get DIFFERENT text. First-writer-wins keeps the CACHE stable --
but the loser then does this:

    text, tel = self._inner.complete(...)
    self._cache.put(sha, _record_from(text, tel))   # silently loses
    return text, tel                                # ...and returns its own anyway

so the losing run acts on, and records, a response the cache does not hold. Replaying that
run returns the winner's text instead. "Given the cache, every number regenerates" is false
for exactly those calls, which is the artifact's central claim.

The money is NOT refunded and must not be: the losing call really was dispatched and really
was billed. Only `text` and `response_sha` are adopted from the canonical record, so the
ledger (debited by the inner client at litellm_client.py:361) keeps the real spend and
reconciliation -- which compares token SUMS, not shas -- is untouched.
"""

from __future__ import annotations

from typing import Any

from pi_run.cache import CachingClient, DiskCache, _record_from
from pinq.types import CallTelemetry


def _tel(sha: str, *, tok: int, usd: float) -> CallTelemetry:
    return CallTelemetry(
        call_id="c" + sha[:4],
        actor="answerer",
        model="m",
        provider="p",
        request_sha="req",
        response_sha=sha,
        tok_prompt=10,
        tok_completion=tok,
        usd=usd,
    )


class _Inner:
    """An LLM that returns a different response each call -- the real non-determinism.

    `during_call` models the ONLY interleaving that produces a divergence: the rival worker
    finishes and writes while OUR request is in flight. Writing the rival's entry before the
    call instead would just be a cache hit, and the loser would never dispatch -- the first
    version of this test made exactly that mistake and passed against the unfixed code.
    """

    def __init__(self, *responses: tuple[str, CallTelemetry], during_call=None) -> None:
        self._responses = list(responses)
        self._during_call = during_call
        self.calls = 0

    def pin(self, role: str) -> Any:  # pragma: no cover - passthrough
        return None

    def complete(self, **kw: Any) -> tuple[str, CallTelemetry]:
        out = self._responses[self.calls]
        self.calls += 1
        if self._during_call is not None:
            self._during_call()
        return out


def _client(tmp_path, inner) -> CachingClient:
    return CachingClient(inner, DiskCache(tmp_path / "cache"))


ARGS = {"role": "answerer", "messages": [{"role": "user", "content": "hi"}], "seed": 0}


def _rival(cache: DiskCache, text: str, tel: CallTelemetry):
    """The other worker, finishing mid-flight and winning the write."""

    def _win() -> None:
        c = CachingClient(_Inner((text, tel)), cache)
        c.complete(**ARGS)

    return _win


def test_the_loser_returns_the_winners_text_and_sha(tmp_path) -> None:
    """The whole point: after a lost race, run and cache agree by construction."""
    cache = DiskCache(tmp_path / "cache")
    winner_text, winner_tel = "Paris is the capital.", _tel("sha_a", tok=7, usd=0.00021)
    loser_text, loser_tel = "The capital is Paris.", _tel("sha_b", tok=8, usd=0.00024)

    cl = CachingClient(
        _Inner((loser_text, loser_tel), during_call=_rival(cache, winner_text, winner_tel)),
        cache,
    )
    text, tel = cl.complete(**ARGS)

    assert text == winner_text, "returned its own text; the run would be unreplayable"
    assert tel.response_sha == "sha_a", "recorded a response_sha the cache does not hold"


def test_the_losers_real_spend_is_still_billed(tmp_path) -> None:
    """We dispatched the call and we paid for it. Adopting the text does not refund it.

    The ledger is debited by the INNER client on a miss, so these fields must survive the
    swap or the ledger and the trajectory would disagree and reconciliation would fail.
    """
    cache = DiskCache(tmp_path / "cache")
    loser_tel = _tel("sha_b", tok=8, usd=0.00024)
    _, tel = CachingClient(
        _Inner(
            ("lost", loser_tel), during_call=_rival(cache, "won", _tel("sha_a", tok=7, usd=0.00021))
        ),
        cache,
    ).complete(**ARGS)

    assert tel.tok_completion == 8, "billed the winner's tokens; we paid for our own call"
    assert tel.usd == 0.00024
    assert tel.call_id == loser_tel.call_id


def test_the_winner_is_unaffected(tmp_path) -> None:
    """The overwhelmingly common path must not change at all."""
    inner = _Inner(("mine", _tel("sha_a", tok=7, usd=0.00021)))
    text, tel = _client(tmp_path, inner).complete(**ARGS)
    assert (text, tel.response_sha, tel.tok_completion) == ("mine", "sha_a", 7)


def test_a_plain_hit_is_still_a_hit(tmp_path) -> None:
    """No extra dispatch: the second call must not reach the inner client at all."""
    cache = DiskCache(tmp_path / "cache")
    inner = _Inner(("mine", _tel("sha_a", tok=7, usd=0.00021)))
    c = CachingClient(inner, cache)
    c.complete(**ARGS)
    c.complete(**ARGS)
    assert inner.calls == 1, "dispatched twice for one cached request"
    assert (c.hits, c.misses) == (1, 1)


def test_put_reports_whether_it_won(tmp_path) -> None:
    """The primitive the fix rests on: `put` must distinguish won from lost.

    It returned a Path either way, so the caller could not tell -- which is why the loser
    silently returned its own text for as long as it did.
    """
    cache = DiskCache(tmp_path / "cache")
    a = _record_from("A", _tel("sha_a", tok=7, usd=0.00021))
    b = _record_from("B", _tel("sha_b", tok=8, usd=0.00024))
    assert cache.put_or_get("k" * 32, a) is None, "first writer should report a win"
    kept = cache.put_or_get("k" * 32, b)
    assert kept is not None, "second writer should report a loss"
    assert kept["text"] == "A" and kept["response_sha"] == "sha_a"


def test_the_race_is_counted(tmp_path) -> None:
    """A silent correctness event is one nobody can audit.

    Surfaced per run as `cache_races` in status.json, so a sweep that raced a lot is visible
    without re-running `pi cache verify` over the whole corpus.
    """
    cache = DiskCache(tmp_path / "cache")
    c = CachingClient(
        _Inner(
            ("lost", _tel("sha_b", tok=8, usd=0.00024)),
            during_call=_rival(cache, "won", _tel("sha_a", tok=7, usd=0.00021)),
        ),
        cache,
    )
    assert c.races == 0
    c.complete(**ARGS)
    assert c.races == 1
