"""Retail is SELF-SOURCED: it reads the upstream checkout, not a corpus this repo built.

The distinction `SELF_SOURCED` draws is "did this repository BUILD the task list?". For retail
the answer is no -- `retail_build` deliberately writes NO corpus tree, because a second copy of
upstream's db.json would carry a second corpus_hash and the adapter's uids would disagree with
gold's. Demanding `data/corpora/tau2_retail/` would therefore block every run on a directory
that is never supposed to exist.
"""

from __future__ import annotations

from pi_run.worker import CORPUS_BACKED, SELF_SOURCED


def test_retail_is_self_sourced() -> None:
    assert "tau2_retail" in SELF_SOURCED


def test_retail_is_not_also_corpus_backed() -> None:
    """The two sets decide different code paths; membership in both is a contradiction."""
    assert not (set(SELF_SOURCED) & set(CORPUS_BACKED))
    assert "tau2_retail" not in CORPUS_BACKED


def test_resolve_corpus_does_not_demand_a_built_tree(tmp_path) -> None:
    from pi_run.cli import _resolve_corpus

    p = _resolve_corpus(tmp_path, "tau2_retail", None)
    assert p == tmp_path / "data" / "corpora" / "tau2_retail"  # unchecked, as tau2 gets
