"""The shared content-term tokeniser, and the two instruments that must agree on it.

`pi_eval.matcher.base` decides whether a question MENTIONS a need; `pi_eval.mining.partition`
decides whether a document COVERS one. Both ask "which words carry the need and which are
scaffolding?" -- and two stoplists for one question is the shape this repository keeps finding
bugs in.
"""

from __future__ import annotations

import pytest

from pi_eval.text import STOPWORDS, content_terms, raw_terms

DOC = (
    "Houston Baptist University is a private university in Houston, Texas, "
    "founded in 1960 by the Baptist General Convention."
)


@pytest.mark.parametrize(
    "question",
    [
        "Houston Baptist University founding year",
        "When was Houston Baptist University founded?",
        "Could you please tell me in what year Houston Baptist University was originally founded?",
    ],
)
def test_the_discoverability_verdict_does_not_move_with_the_phrasing(question):
    """THE DEFECT THIS EXISTS FOR. `discoverability_of` scored the RAW interrogative ASK, token
    for token, with no filtering -- so the polite phrasing scored 0.357 and was classified
    `user_private` while the terse one scored 0.600 and was `kb`. Same need, three ways of
    asking, two verdicts.

    `private_share` is the hard ceiling on what any autonomous inquirer could reach, which
    CLAUDE.md calls the single most important number for the framing. It was partly a property
    of how politely the mining policy happened to write, and a verbose policy manufactures a
    higher ceiling.
    """
    from pi_eval.mining.partition import discoverability_of

    kind, cov, _doc = discoverability_of(question, [{"id": "d1", "content": DOC}])
    assert kind == "kb", f"cov={cov:.3f} for {question!r}"


def test_the_unfiltered_tokeniser_really_did_flip_the_verdict():
    """Not a hypothetical: the old behaviour is still reachable through `raw_terms`, and this
    shows the number it produced."""
    need = raw_terms(
        "Could you please tell me in what year Houston Baptist University was originally founded?"
    )
    raw_cov = len(need & raw_terms(DOC)) / len(need)
    assert raw_cov < 0.60, f"{raw_cov:.3f}"

    terms = content_terms(
        "Could you please tell me in what year Houston Baptist University was originally founded?"
    )
    assert len(terms & content_terms(DOC)) / len(terms) >= 0.60


def test_scaffolding_is_removed_and_content_is_kept():
    assert content_terms("When was the University founded?") == {"university", "founded"}
    assert content_terms("Could you please tell me about it") == set()
    assert "houston" in content_terms("Houston, Texas.")
    # Numbers are content: a year or an account number is exactly the thing being asked about.
    assert "1960" in content_terms("founded in 1960")


def test_the_stoplist_is_small_enough_to_read():
    """A large stoplist is a tuning surface, and an instrument that decides a published ceiling
    should be describable in one sentence."""
    assert len(STOPWORDS) < 130
    for essential in ("what", "when", "please", "the", "was", "could"):
        assert essential in STOPWORDS


def test_both_instruments_use_the_one_tokeniser():
    """A second copy of this list is a second place for "which words are scaffolding?" to have
    two answers."""
    import inspect

    from pi_eval.matcher import base as matcher
    from pi_eval.mining import partition

    assert "content_terms" in inspect.getsource(matcher._terms)
    assert "content_terms" in inspect.getsource(partition.discoverability_of)
    for mod in (matcher, partition):
        src = inspect.getsource(mod)
        assert "_STOP = frozenset" not in src, f"{mod.__name__} grew its own stoplist"


def test_the_tokeniser_is_pinned_into_the_mined_graphs_identity():
    """It decides kb-vs-user_private and therefore the ceiling, so two graphs partitioned under
    different tokenisers are two measurements and must not share a hash."""
    import inspect

    from pi_eval.mining import pipeline
    from pi_eval.mining.partition import DISCOVERABILITY_PIN

    assert DISCOVERABILITY_PIN
    assert "DISCOVERABILITY_PIN" in inspect.getsource(pipeline.MinedGraph.graph_hash.fget)
