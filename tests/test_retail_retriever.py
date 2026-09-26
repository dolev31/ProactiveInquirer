"""Retail has NO retrieval surface, and the honest implementation says so rather than faking one.

Banking wraps `search_documents` over 698 knowledge documents. Retail has 15 typed,
key-addressed tools over a relational DB and no free-text search at all. Two wrong answers were
available here and both are worse than an empty one:

  * synthesising a BM25 index over the 1,550 DB records. That measures a benchmark that does
    not exist -- the real retail agent cannot search, it can only look up by key -- and any
    transfer claim made on it would not be a claim about tau2 retail.
  * omitting the retriever entirely, which `run_loop` cannot accept: it takes one positionally
    and calls `.search` on every ASK.

So the retriever exists, satisfies the protocol, and returns nothing. Evidence on retail comes
from what TOOL CALLS asked for (`retail_units.uids_from_env_calls`), and a question the policy
addresses to the customer is answered by the user simulator, not by a corpus.
"""

from __future__ import annotations

from pinq_adapters.tau2.retail_retriever import RetailNullRetriever


def test_it_satisfies_the_retriever_protocol() -> None:
    from pinq.protocols import Retriever

    r = RetailNullRetriever(corpus_id="tau2_retail", corpus_hash="abc")
    assert isinstance(r, Retriever)


def test_search_returns_nothing_for_any_query() -> None:
    r = RetailNullRetriever(corpus_id="tau2_retail", corpus_hash="abc")
    assert r.search("where is my order", 5) == ()
    assert r.search("", 0) == ()


def test_it_carries_the_corpus_identity_so_the_manifest_is_still_honest() -> None:
    """`corpus_hash` still has to be the real one: it is what binds a run to the DB its gold
    was built against, and `pi score` refuses a mismatched pairing."""
    r = RetailNullRetriever(corpus_id="tau2_retail", corpus_hash="deadbeef")
    assert r.corpus_id == "tau2_retail"
    assert r.corpus_hash == "deadbeef"


def test_the_suite_hands_out_a_working_channel_carrying_its_hash() -> None:
    """WAS `test_the_suite_hands_out_one`, asserting a `RetailNullRetriever`. That belief was
    deliberately changed; the class itself is unchanged and still tested above.

    A null retriever was an honest description of the domain -- retail has 15 typed
    key-addressed tools and no free-text search -- but its consequence was that an ASK could
    not gain evidence at all. Measured, on three runs: 33 exported rows, every one with
    `value = -0.05`, `coverage_before = 0.0`, `frontier_size = 6`, and
    `n_no_target_dropped = 33`. Gold nodes existed, the policy asked sixteen questions, and not
    one could reach a record. Retail was a declared training source that could not produce a
    training row for a structural reason.

    `ToolBackedRetriever` postdates `RetailNullRetriever`; giving retail the same channel
    airline and telecom use makes its arms comparable rather than a special case. The property
    this test was really defending -- the retriever carries the SUITE's corpus_hash, because
    that hash binds a run to the DB its gold was built against and `pi score` refuses a
    mismatched pairing -- is unchanged and asserted below.
    """
    from pinq_adapters.tau2._probe import available

    if not available()[0]:
        import pytest

        pytest.skip(available()[1])
    from pinq_adapters.tau2.retail_suite import Tau2RetailSuite
    from pinq_adapters.tau2.tool_retriever import ToolBackedRetriever

    s = Tau2RetailSuite()
    r = s.retriever(s.task_ids()[0])
    assert isinstance(r, ToolBackedRetriever)
    assert r.corpus_hash == s.corpus_hash, "a retriever whose hash differs from the suite's"
