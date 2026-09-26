"""The retriever retail does not have, made explicit.

`run_loop` takes a Retriever positionally and calls `.search` on every ASK, so one has to
exist. Retail has no free-text search surface: 15 typed, key-addressed tools over a relational
DB, and nothing that takes a query. Returning an empty result is the honest implementation.

WHY NOT SYNTHESISE ONE. Indexing the 1,550 DB records for BM25 would make the loop work and
would measure a benchmark that does not exist -- the real retail agent cannot search, only look
up by key -- so any transfer claim built on it would not be a claim about tau2 retail. The
build scope called that Option A and it was rejected for exactly this reason.

WHERE EVIDENCE COMES FROM INSTEAD. Tool calls: `retail_units.uids_from_env_calls` reconstructs
which records a turn read from what its calls ASKED FOR, never from what they returned. And a
question the policy addresses to the customer is answered by the user simulator -- which is the
channel Option B opens and the one the 219 user_private gold nodes live behind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pinq.types import EvidenceUnit


@dataclass(frozen=True, slots=True)
class RetailNullRetriever:
    """Satisfies `pinq.protocols.Retriever`; returns nothing, by design and not by failure."""

    corpus_id: str
    corpus_hash: str

    def search(self, query: str, k: int) -> Sequence[EvidenceUnit]:
        """Always empty. NOT an error and NOT a degraded mode.

        An ASK on retail is either addressed to the customer -- answered by the user simulator,
        outside this channel -- or it is a query against a corpus that has no query interface.
        Charging the turn and returning nothing is what makes the second case visible in
        `n_retrieved` rather than silently plausible.
        """
        return ()
