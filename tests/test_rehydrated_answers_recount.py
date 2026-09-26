"""Lane L0.9: does re-applying the answerer's own word-cap to the cached raw text reproduce
the `n_words` a rehydrated run's `outcome.json` already carries?

Two synthetic cases, both plausible on the real population (`artifacts/rehydrated_answers_
20260918/RESULT.md`): a run whose raw cached text merely ran past its own word_cap (the
common case: reasoning preamble, or the model simply not stopping), where re-capping recovers
the original count exactly; and a run where the cache holds FEWER words than were originally
scored -- the shape a first-writer-wins cache race under non-determinism produces (measured
example in the real population: stored 57, cached text only 20 words) -- which no amount of
capping can recover, and the script must say so rather than claim it fixed it.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rehydrated_answers.recount import check_run, recount_answer, scan


def _rehydrated_run(root: Path, run_id: str, *, word_cap: int, text: str, n_words: int) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "suite_id": "tau2",
                "task_id": "t1",
                "arm_id": "drafter_only",
                "word_cap": word_cap,
                "rehydrated_from": "scores/parquet + cache",
            }
        )
    )
    (d / "outcome.json").write_text(
        json.dumps({"answer": {"text": text, "n_words": n_words}, "stop_reason": "policy_stop"})
    )
    return d


def _normal_run(root: Path, run_id: str) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {"run_id": run_id, "suite_id": "tau2", "task_id": "t1", "arm_id": "drafter_only"}
        )
    )
    (d / "outcome.json").write_text(
        json.dumps(
            {"answer": {"text": "a short answer", "n_words": 3}, "stop_reason": "policy_stop"}
        )
    )
    return d


# --------------------------------------------------------------------------- the pure function


def test_recount_caps_at_word_cap_exactly_like_the_answerer():
    text = " ".join(f"w{i}" for i in range(50))  # 50 words
    assert recount_answer(text, word_cap=30) == 30
    assert recount_answer(text, word_cap=100) == 50


def test_recount_of_empty_word_cap_is_zero():
    assert recount_answer("some words here", word_cap=0) == 0


# ----------------------------------------------------------------- the truncation-only case


def test_a_raw_cache_text_that_only_overran_its_word_cap_is_reproduced_exactly(tmp_path):
    """The common, fixable shape: capped = raw[:word_cap] is what the run originally scored,
    and today's cache still holds that same raw text (no race)."""
    long_text = "We need to search the knowledge base first. " + " ".join(
        f"word{i}" for i in range(40)
    )
    stored = recount_answer(long_text, word_cap=10)  # what scoring would have produced
    d = _rehydrated_run(tmp_path, "a" * 32, word_cap=10, text=long_text, n_words=stored)

    r = check_run(d)

    assert r["contradicts"] is True  # stored (10) != raw_recount (len(long_text.split()))
    assert r["reproduces"] is True
    assert r["recomputed_n_words"] == stored


# ------------------------------------------------------- the un-fixable cache-race case


def test_a_cache_race_that_shortened_the_text_cannot_be_reproduced_by_capping(tmp_path):
    """Stored n_words (57) EXCEEDS today's cached text's own length (20 words): no value of
    word_cap can cap UP to a word count the text does not have. The script must report this
    as non-reproducing, not silently invent a match."""
    short_text = " ".join(f"word{i}" for i in range(20))  # only 20 words today
    d = _rehydrated_run(tmp_path, "b" * 32, word_cap=180, text=short_text, n_words=57)

    r = check_run(d)

    assert r["contradicts"] is True
    assert r["reproduces"] is False
    assert r["recomputed_n_words"] == 20


# --------------------------------------------------------------------------- scan filters


def test_a_normal_run_is_not_in_the_scan_at_all(tmp_path):
    _normal_run(tmp_path, "c" * 32)

    results = scan(tmp_path, only_contradicting=False)

    assert results == []


def test_only_contradicting_excludes_an_agreeing_rehydrated_run(tmp_path):
    _rehydrated_run(tmp_path, "d" * 32, word_cap=30, text="one two three", n_words=3)

    assert scan(tmp_path, only_contradicting=True) == []
    assert len(scan(tmp_path, only_contradicting=False)) == 1
