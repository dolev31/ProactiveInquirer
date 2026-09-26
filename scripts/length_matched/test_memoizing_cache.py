"""Proves the memoizing cache in `score_checkpoint.py` returns the SAME numbers as no cache at
all, and that it actually gets hit across nested ladder rungs -- the two properties the whole
point of caching depends on and that a silent bug here would corrupt without a trace.

WHY THIS MATTERS MORE THAN A TYPICAL CACHE TEST. `score_checkpoint.py`'s cache sits directly in
front of the log-prob evaluation `pair_accuracy` orders its verdict on. A cache that returns a
STALE or a WRONG value for some `(state_text, action_json)` key would silently change an ordering
decision (`chosen_lp > rejected_lp`) on exactly the pairs this task exists to score, and nothing
downstream -- `pair_accuracy` itself, or a human reading its `acc` field -- could tell a cached
wrong answer from an honestly-computed one. Per CLAUDE.md rule 2, this is written and run BEFORE
any GPU job trusts the cache with real weights.

Runs with NO torch and NO transformers: a scripted `render`/`logprob` pair plays the role of
`torch_logprob_fn`'s output, deterministic and free, so this is a $0, seconds-long check.

Three properties asserted, each printed:

  1. EQUIVALENCE. `pair_accuracy` run through the memoizing wrapper (score_checkpoint.py's
     `eo._score` monkeypatch, exercised exactly as `main()` uses it) returns a byte-identical
     dict to `pair_accuracy` run with the ORIGINAL, uncached `_score` -- on the same rows, same
     render, same logprob. If the cache changed even one ordering, `acc`, `acc_by_len_sign` or
     `len_sign_gap` would differ and this assertion is what would catch it.
  2. HIT COUNT IS EXACT, NOT JUST NONZERO. A pair repeated across two calls (simulating a pair
     that appears in both `ask_ask_all.jsonl` and a ladder rung file, e.g. every `tol0` pair is
     also in `ask_ask_all`) must be a cache HIT on the second call, and the count of hits must
     equal the exact number of repeated (state, action) keys computed by hand -- not "some hits
     happened", which a cache that hit on the wrong key could also produce.
  3. A CHANGED VALUE FOR A REPEATED KEY IS NOT RE-READ (the cache is not accidentally a no-op
     passthrough): the scripted logprob function is swapped to return a DIFFERENT number for the
     same completion after the first call, and the second call's result must still reflect the
     FIRST (cached) value, proving the wrapper is really short-circuiting the second computation
     rather than happening to agree with it.

Usage:
    python scripts/length_matched/test_memoizing_cache.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pinq_train.eval_offline as eo

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORE_SCRIPT = REPO_ROOT / "scripts" / "length_matched" / "score_checkpoint.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_pairs() -> list[dict[str, Any]]:
    # Two distinct ask_ask pairs, `state_text`/`chosen_json`/`rejected_json` are all
    # `pair_accuracy` reads (plus `pair_kind`/`decided_by`, optional -- it reads them with
    # `.get`). Actions are real ASK-shaped JSON so `action_kind_of` classifies them "ask", not
    # "stop", exactly like a real ask_ask row.
    return [
        {
            "pair_id": "p1",
            "state_text": "state-A",
            "chosen_json": '{"action": "ASK", "question": "short one?", "rationale": ""}',
            "rejected_json": '{"action": "ASK", "question": "a somewhat longer one?", "rationale": ""}',
            "pair_kind": "ask_ask",
            "decided_by": "test",
        },
        {
            "pair_id": "p2",
            "state_text": "state-B",
            "chosen_json": '{"action": "ASK", "question": "another longer question here?", "rationale": ""}',
            "rejected_json": '{"action": "ASK", "question": "brief?", "rationale": ""}',
            "pair_kind": "ask_ask",
            "decided_by": "test",
        },
    ]


def main() -> int:
    mod = _load_module(SCORE_SCRIPT, "length_matched_score_checkpoint")

    calls: list[tuple[str, str]] = []

    def render(state: str, action: str) -> eo.Rendered:
        # Deterministic, nonzero-length completion ids so `_score`'s zero-length guard never
        # fires; prompt/completion ids are just the string's code points.
        return eo.Rendered(
            prompt_ids=tuple(ord(c) for c in state) or (0,),
            completion_ids=tuple(ord(c) for c in action),
        )

    # `logprob` records every REAL invocation (a cache hit must not add an entry here) and
    # returns a value that changes on a second call for the SAME completion, so property 3
    # can tell a genuine short-circuit from a value that merely happened to match.
    value_for: dict[tuple[int, ...], list[float]] = {}

    def logprob(prompt_ids: Any, completion_ids: Any) -> float:
        key = tuple(completion_ids)
        calls.append(("logprob", str(key)))
        seen = value_for.setdefault(key, [])
        seen.append(0.0)
        # First call for a key returns -len(key); every call after returns a DIFFERENT number
        # (+1000 offset), so a cache bug that re-reads on the second call is visible as a
        # changed `acc`/`mean_margin` rather than an accidental match.
        return -float(len(key)) if len(seen) == 1 else -float(len(key)) + 1000.0

    rows = _make_pairs()

    # --- baseline: pair_accuracy through the ORIGINAL, uncached _score -------------------
    baseline = eo.pair_accuracy(rows, render=render, logprob=logprob)
    calls_after_baseline = len(calls)
    assert calls_after_baseline == 4, (
        f"expected 4 real logprob calls scoring 2 pairs x 2 sides uncached, got "
        f"{calls_after_baseline}: {calls}"
    )

    # --- through the memoizing wrapper, called TWICE (simulating two ladder rungs that ------
    # --- share every one of these pairs, e.g. ask_ask_all and a tol rung that is its subset) --
    calls.clear()
    value_for.clear()
    cached_score, stats = mod._memoizing_logprob(logprob, render)
    orig_score = eo._score
    eo._score = cached_score
    try:
        first = eo.pair_accuracy(rows, render=render, logprob=logprob)
        calls_after_first = len(calls)
        second = eo.pair_accuracy(rows, render=render, logprob=logprob)
        calls_after_second = len(calls)
    finally:
        eo._score = orig_score

    # Property 1: EQUIVALENCE with the uncached baseline (first call, cold cache).
    assert first == baseline, (
        f"cached first call disagrees with uncached baseline:\n{first}\n{baseline}"
    )

    # Property 2: exact hit accounting. 4 distinct (state, action) keys total (2 pairs x
    # 2 sides); the first call is 4 misses / 0 hits; the second call, over the SAME 4 keys,
    # is 0 misses / 4 hits, and issues NO new real logprob calls.
    assert calls_after_first == 4, f"first call: expected 4 real calls, got {calls_after_first}"
    assert calls_after_second == 4, (
        f"second call added real logprob calls ({calls_after_second - calls_after_first} more) "
        "-- the cache is not being hit"
    )
    assert stats["total"] == 8, f"expected 8 total _score calls (2 calls x 4 keys), got {stats}"
    assert stats["hits"] == 4, (
        f"expected exactly 4 cache hits (the second call, in full), got {stats}"
    )

    # Property 3: the second call reflects the CACHED (first-seen) value, not a fresh one --
    # `logprob` would have returned +1000 on a second real call, which would flip every
    # `chosen_lp > rejected_lp` comparison (both sides shift by the same +1000, but a
    # cache-miss on only one side of a pair would not) and change `acc`/`mean_margin`.
    assert second == first, (
        f"second call diverged from the first -- the cache is not short-circuiting real "
        f"computation, it just happens to agree once:\n{second}\n{first}"
    )

    print("property 1 (cached == uncached baseline): OK")
    print(f"property 2 (exact hit count): OK  stats={stats}")
    print("property 3 (cache is a real short-circuit, not a coincidence): OK")
    print(
        f"baseline/first/second pair_accuracy.acc = {baseline['acc']} / {first['acc']} / {second['acc']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
