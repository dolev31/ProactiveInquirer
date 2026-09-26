"""Okapi BM25 over a task's OWN closed paragraph pool. Stdlib only, forty lines of arithmetic.

WHY HAND-ROLLED. Retrieval is part of the measuring instrument, not a convenience: every
number in the paper is a function of which paragraphs a question surfaced. A scored
trajectory therefore has to be recomputable from the corpus alone, years later, on a machine
that never installed our lockfile — so the scorer cannot be a third-party package whose
tokenizer or idf variant may change under a caret range. rank_bm25 would put a version pin
between the paper and its numbers in exchange for code that fits on one screen.

WHY PER TASK. The index is built over that task's own released pool (20 paragraphs for
MuSiQue, ~10 for 2Wiki), never over a global corpus. idf is consequently computed inside the
closed pool, which is what keeps retrieval difficulty comparable across tasks: every task's
retriever sees the pool size and the gold-to-distractor ratio the dataset shipped with,
rather than a ratio that drifts with however many suites happen to be loaded.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Sequence

from pinq.types import EvidenceUnit

K1 = 1.2
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs. No stemming and no stopword list.

    Both would be tunable knobs sitting between a question and its evidence, and a knob that
    is never reported is a degree of freedom the reader cannot audit. The one thing this
    does do is fold digits in with letters, because dates and ordinals carry most of the
    discriminative signal in multi-hop comparison questions.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 with k1=1.2, b=0.75, written out so the formula is readable at the call site.

    idf uses the Lucene variant ln(1 + (N - df + 0.5) / (df + 0.5)) rather than textbook
    Okapi's ln((N - df + 0.5) / (df + 0.5)). In a 20-document pool a term occurring in more
    than half the paragraphs gets a NEGATIVE textbook idf, so matching it would score a
    document DOWN — an artefact that only bites on tiny closed pools, which is precisely the
    regime we run in. The +1 form is strictly positive and monotone in rarity.
    """

    def __init__(self, docs: Sequence[Sequence[str]]) -> None:
        self.n = len(docs)
        self.lens = [len(d) for d in docs]
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        self.tf: list[Counter[str]] = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.idf = {term: math.log(1.0 + (self.n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def scores(self, query: str) -> list[float]:
        """Query term frequency is linear, as in Okapi: a term repeated in the query counts
        twice. Deliberate — a policy that repeats an entity name is emphasising it."""
        out = [0.0] * self.n
        if not self.avgdl:
            return out
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + K1 * (1.0 - B + B * self.lens[i] / self.avgdl)
                out[i] += idf * f * (K1 + 1.0) / denom
        return out


class BM25Retriever:
    """Satisfies pinq.protocols.Retriever structurally: corpus_id, corpus_hash, search().

    Title and body are indexed as one bag. The title carries the entity name that a
    decomposition step usually asks about ("#1 >> spouse" resolves against a paragraph whose
    title IS the entity), so dropping it would make the second hop of most MuSiQue chains
    unretrievable for reasons that have nothing to do with the policy.
    """

    def __init__(self, units: Sequence[EvidenceUnit], *, corpus_id: str, corpus_hash: str) -> None:
        self._units = tuple(units)
        self._index = BM25Index([tokenize(f"{u.title} {u.text}") for u in self._units])
        self.corpus_id = corpus_id
        self.corpus_hash = corpus_hash

    def search(self, query: str, k: int) -> tuple[EvidenceUnit, ...]:
        """Top-k by score, ties broken by uid so the ranking is reproducible bit-for-bit.

        A query matching nothing returns NOTHING, rather than the k least-bad paragraphs.
        Padding would hand a policy free evidence for a nonsense question and turn every
        discovery metric into a function of pool size instead of of the question asked.
        """
        scores = self._index.scores(query)
        ranked = sorted(
            (-s, u.uid, i) for i, (u, s) in enumerate(zip(self._units, scores)) if s > 0.0
        )
        # rounded so the score serialized into telemetry is stable across platforms
        return tuple(replace(self._units[i], score=round(-neg, 6)) for neg, _, i in ranked[:k])
