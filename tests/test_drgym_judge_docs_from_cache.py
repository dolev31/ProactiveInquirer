"""The citation judge's documents, replayed from the search cache.

`citation_support` is preregistered and has never produced one judgment: the judge needs
`{url: text}`, `load_judge_docs` reads it from `runs/<run_id>/judge_docs.json`, and that
sidecar has NO WRITER anywhere in the codebase -- 0 of 729 run directories have one.

The inputs were always on disk. `turns.jsonl.question` is the retriever query verbatim, `k`
is in the manifest, `corpus_id` is in evidence.jsonl, and the drgym cache stores each API
envelope verbatim with the full document text. MEASURED over 60 real drgym runs: 0 missing
queries, mean 17 documents reconstructed per retrieving run.

The 0-document runs are `drafter_only`, which on drgym retrieves nothing at all (0 evidence
lines, 0 ask-turns across 40 runs). Zero is the true answer there, not a gap.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from pinq_adapters.drgym.cache import search_key
from pinq_adapters.drgym.client import CORPORA
from pinq_adapters.drgym.judge_docs import judge_docs_from_cache

CORPUS = CORPORA["fineweb"]
K = 5


class _Cache:
    def __init__(self, entries: dict[str, Any] | None = None) -> None:
        self._e = dict(entries or {})

    def get(self, sha: str) -> Any:
        return self._e.get(sha)

    def put(self, sha: str, rec: Any) -> None:
        self._e[sha] = rec


def _envelope(*docs: tuple[str, str]) -> dict[str, Any]:
    """The provider's own shape: {"results": [base64(json), ...]}."""
    return {
        "results": [
            base64.b64encode(
                json.dumps({"id": f"d{i}", "url": u, "text": t, "language": "en"}).encode()
            ).decode()
            for i, (u, t) in enumerate(docs)
        ]
    }


def _run(
    tmp_path: Path, *, questions: list[str], suite: str = "drgym", corpus_ids=("drgym_fineweb_v1",)
) -> Path:
    d = tmp_path / "run1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"suite_id": suite, "k": K}))
    (d / "evidence.jsonl").write_text(
        "\n".join(json.dumps({"corpus_id": c, "uid": f"u{i}"}) for i, c in enumerate(corpus_ids))
    )
    (d / "turns.jsonl").write_text("\n".join(json.dumps({"question": q}) for q in questions))
    return d


def _cache_with(*queries_docs) -> _Cache:
    c = _Cache()
    for q, docs in queries_docs:
        c.put(
            search_key(corpus=CORPUS.name, endpoint=CORPUS.path, query=q, k=K),
            {"envelope": _envelope(*docs)},
        )
    return c


def test_reconstructs_url_to_text(tmp_path) -> None:
    d = _run(tmp_path, questions=["who wrote it"])
    cache = _cache_with(("who wrote it", [("https://a.example/x", "the text of A")]))
    docs, missing = judge_docs_from_cache(d, cache)
    assert docs == {"https://a.example/x": "the text of A"}
    assert missing == 0


def test_a_query_absent_from_the_cache_is_counted_not_ignored(tmp_path) -> None:
    """A judge handed no sources scores every claim `no_support`.

    That reports a citation failure which is really a fetch failure, so the omission has to
    be visible rather than folded into an empty dict.
    """
    d = _run(tmp_path, questions=["cached", "not cached"])
    cache = _cache_with(("cached", [("https://a.example/x", "A")]))
    docs, missing = judge_docs_from_cache(d, cache)
    assert docs == {"https://a.example/x": "A"}
    assert missing == 1


def test_an_undecodable_entry_counts_as_missing(tmp_path) -> None:
    """Not as an empty result, for the same reason."""
    d = _run(tmp_path, questions=["q"])
    c = _Cache(
        {
            search_key(corpus=CORPUS.name, endpoint=CORPUS.path, query="q", k=K): {
                "envelope": {"results": ["!!not base64!!"]}
            }
        }
    )
    docs, missing = judge_docs_from_cache(d, c)
    assert docs == {}
    assert missing == 1


def test_a_repeated_question_is_one_lookup(tmp_path) -> None:
    """Two identical queries are one cache entry; counting it twice would inflate `missing`."""
    d = _run(tmp_path, questions=["q", "q", "q"])
    docs, missing = judge_docs_from_cache(d, _Cache())
    assert missing == 1, "the same query counted more than once"


def test_a_non_drgym_run_is_not_a_failure(tmp_path) -> None:
    """musique reports contain no URLs at all; there is nothing to reconstruct."""
    d = _run(tmp_path, questions=["q"], suite="musique")
    assert judge_docs_from_cache(d, _cache_with(("q", [("https://a/x", "A")]))) == ({}, 0)


def test_a_run_that_mixed_corpora_refuses_to_guess(tmp_path) -> None:
    """Which corpus a URL came from decides which text sits under a citation."""
    d = _run(tmp_path, questions=["q"], corpus_ids=("drgym_fineweb_v1", "drgym_clueweb22_v1"))
    assert judge_docs_from_cache(d, _cache_with(("q", [("https://a/x", "A")]))) == ({}, 0)


def test_a_run_that_retrieved_nothing_yields_nothing(tmp_path) -> None:
    """drafter_only on drgym: 0 evidence, 0 ask-turns across 40 real runs. Zero is true."""
    d = _run(tmp_path, questions=[], corpus_ids=())
    assert judge_docs_from_cache(d, _Cache()) == ({}, 0)


def test_non_url_titles_are_excluded(tmp_path) -> None:
    """`docs_from_units` keys on the URL because that is what a report cites literally."""
    d = _run(tmp_path, questions=["q"])
    cache = _cache_with(("q", [("not-a-url", "junk"), ("https://a/x", "A")]))
    docs, _ = judge_docs_from_cache(d, cache)
    assert docs == {"https://a/x": "A"}
