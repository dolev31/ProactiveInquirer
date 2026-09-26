"""Reconstruct the citation judge's {url: text} from the recorded search cache.

WHY THIS EXISTS. `citation_support` is a judge-derived metric -- reported in T6_instrument,
NOT among PRIMARY/SECONDARY/EXPLORATORY and not in prereg/sigma_j.json -- and it had never
produced a single judgment. The judge needs the DOCUMENT TEXT a report was supposed to cite, and
`pi_eval.score.load_judge_docs` reads it from `runs/<run_id>/judge_docs.json` -- a sidecar
with no writer anywhere in the codebase. 0 of 729 run directories have one, so the judge
was disqualified on every run ever scored.

WHY NOT WRITE THE SIDECAR AT ROLLOUT TIME. It would cover no existing run: the $10.81 P3
grid would have to be re-rolled to gain a metric it already has the inputs for. Evidence
TEXT is also deliberately never copied into a run directory (worker.py:299-301), and
duplicating ~34 MB of upstream corpus text into `runs/` would both break that policy and
put redistributable text somewhere `docs/DATA.md` does not govern.

WHY THIS WORKS INSTEAD. Everything needed is already on disk:

  - `turns.jsonl.question` IS the retriever query, verbatim (loop.py:87 <-> worker.py:236)
  - `k` is in the manifest, `corpus_id` is in `evidence.jsonl`
  - the drgym search cache stores each API envelope VERBATIM, keyed by
    `search_key(corpus, endpoint, query, k)` -- and it holds the full document text

so the exact retrieval a run performed can be replayed offline. MEASURED over 25 drgym
runs: 388 of 388 queries present in the cache, 100%. The text stays in `cache/` (gitignored,
and already governed by docs/DATA.md) and never enters a run directory.

A query the cache does not hold is SKIPPED AND COUNTED, never silently treated as an empty
document set: a judge handed no sources scores every claim `no_support` and reports a
citation failure that is really a fetch failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _corpus_name(corpus_id: str) -> str | None:
    from .client import CORPUS_IDS

    for name, cid in CORPUS_IDS.items():
        if cid == corpus_id:
            return name
    return None


def judge_docs_from_cache(run_dir: str | Path, cache: Any) -> tuple[dict[str, str], int]:
    """({url: text}, n_queries_missing) for one run, replayed from the search cache.

    `cache` is anything with `.get(sha) -> dict | None` (`pi_run.cache.DiskCache` satisfies
    it structurally, which is how this module stays free of a pi_run import).

    Returns empty for any run this cannot apply to -- a non-drgym suite, no evidence, no
    ask-turns -- because those are not failures: they are runs with nothing to cite.
    """
    from .cache import search_key
    from .client import CORPORA, docs_from_envelope, units_of

    d = Path(run_dir)
    try:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, 0
    if str(manifest.get("suite_id") or "") != "drgym":
        return {}, 0

    corpus_ids = set()
    try:
        for line in (d / "evidence.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                cid = json.loads(line).get("corpus_id")
                if cid:
                    corpus_ids.add(str(cid))
    except (OSError, ValueError):
        return {}, 0
    if len(corpus_ids) != 1:
        # Zero means nothing was retrieved. More than one means the run mixed corpora, and
        # guessing which one a URL came from would put the wrong text under a citation.
        return {}, 0
    name = _corpus_name(next(iter(corpus_ids)))
    if name is None or name not in CORPORA:
        return {}, 0
    corpus = CORPORA[name]
    k = int(manifest.get("k") or 0)

    questions: list[str] = []
    try:
        for line in (d / "turns.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = str(json.loads(line).get("question") or "").strip()
            if q and q not in questions:  # the same query twice is one cache entry
                questions.append(q)
    except (OSError, ValueError):
        return {}, 0

    docs: dict[str, str] = {}
    missing = 0
    for q in questions:
        rec = cache.get(search_key(corpus=corpus.name, endpoint=corpus.path, query=q, k=k))
        if rec is None:
            missing += 1
            continue
        try:
            units = units_of(docs_from_envelope(rec.get("envelope"), corpus), name)
        except Exception:
            # A cache entry that will not decode is a missing entry, not an empty result.
            missing += 1
            continue
        for u in units:
            if u.title.startswith(("http://", "https://")) and u.text.strip():
                docs.setdefault(u.title, u.text)
    return docs, missing


def judge_docs_for(run_dir: str | Path, cache_root: str | Path) -> tuple[dict[str, str], int]:
    """`judge_docs_from_cache` with the store constructed here, from a path.

    The construction lives on THIS side of the wall on purpose. `pi_eval` may not import
    `pi_run` -- that is one of the four contracts, and the judge client is resolved by string
    for the same reason -- so score.py cannot build a `pi_run.cache.DiskCache` itself.
    `ShardedJsonCache` reads byte-identical files (there is a test that writes with one class
    and reads with the other), and it lives in `pinq_adapters`, which `pi_eval` may import.
    """
    from .cache import ShardedJsonCache

    return judge_docs_from_cache(run_dir, ShardedJsonCache(cache_root))
