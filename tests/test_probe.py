"""The retrieval-sensitivity probe: the cheapest check that can end a suite.

It must FAIL loudly on a degenerate retriever, because the failure it guards against looks
exactly like a null result and would otherwise be debugged as a policy problem for two weeks.
"""

import pytest

from pi_eval.probe import probe, probe_task


class _Unit:
    def __init__(self, uid):
        self.uid = uid


class DegenerateRetriever:
    """Returns the same top-k whatever it is asked. The exact failure mode: every arm then
    converges on one evidence.subset_hash and every paired delta is 0.00 +/- 0.00."""

    def search(self, query, k):
        return [_Unit(f"u{i}") for i in range(k)]


class KeywordRetriever:
    """Returns units whose id appears in the query. Sub-questions therefore reach evidence the
    full question does not."""

    def search(self, query, k):
        return [_Unit(t) for t in query.split() if t.startswith("u")][:k]


def test_a_degenerate_retriever_fails_the_probe():
    tasks = [
        probe_task(
            task_id=f"t{i}",
            question="the full question",
            subquestions=["sub one", "sub two", "sub three"],
            gold_uids=frozenset({"u0", "u1"}),
            retriever=DegenerateRetriever(),
            k=5,
        )
        for i in range(10)
    ]
    res = probe(tasks, suite_id="degenerate")
    assert res.median_distinct_sets == 1.0, "every query returned the same set"
    assert res.frac_tasks_with_subq_only_evidence == 0.0
    assert not res.passed
    assert "do not discriminate" in res.verdict
    assert res.consequence, "a failing gate must state its consequence"


def test_a_discriminating_retriever_passes():
    tasks = [
        probe_task(
            task_id=f"t{i}",
            question="u0 topic",
            subquestions=["u1 detail", "u2 detail"],
            gold_uids=frozenset({"u0", "u1", "u2"}),
            retriever=KeywordRetriever(),
            k=5,
        )
        for i in range(10)
    ]
    res = probe(tasks, suite_id="ok")
    assert res.median_distinct_sets >= 2.0
    assert res.frac_tasks_with_subq_only_evidence == 1.0
    assert res.passed and res.verdict == "PASS"


def test_the_full_question_set_counts_toward_distinctness():
    """If every sub-question merely reproduces the full question's set, that is ONE set, not
    zero -- otherwise a degenerate suite would look like it had no data rather than no signal."""
    t = probe_task(
        task_id="t",
        question="q",
        subquestions=["a", "b"],
        gold_uids=frozenset({"u0"}),
        retriever=DegenerateRetriever(),
        k=3,
    )
    assert t.n_distinct_sets == 1


def test_subq_only_hits_exclude_what_the_full_question_already_found():
    """The quantity of interest is evidence inquiry ADDS, not evidence it duplicates."""
    t = probe_task(
        task_id="t",
        question="u0",
        subquestions=["u0", "u1"],
        gold_uids=frozenset({"u0", "u1"}),
        retriever=KeywordRetriever(),
        k=5,
    )
    assert t.full_question_hits == 1
    assert t.subq_only_hits == 1, "u0 was already found; only u1 is added"


def test_an_empty_probe_is_a_failure_not_a_pass():
    """No data must never read as a green light."""
    res = probe([], suite_id="empty")
    assert res.n_tasks == 0 and not res.passed


def test_thresholds_are_explicit_and_reported():
    res = probe([], suite_id="x", min_median_distinct=3.0, min_frac_subq_only=0.5)
    assert res.min_median_distinct == 3.0 and res.min_frac_subq_only == 0.5


@pytest.mark.integration
def test_musique_retrieval_discriminates_on_the_real_corpus():
    """Measured on the built corpus: median 4.50 distinct top-5 sets, 67% of tasks have
    sub-question-only gold evidence. That 67% is the headroom the Inquirer exploits."""
    import json
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if not any((root / "data" / "corpora" / "musique").glob("*/tasks.jsonl")):
        pytest.skip("musique corpus not built")
    out = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "probe_retrieval.py"),
            "--suite",
            "musique",
            "--n",
            "40",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    d = json.loads(re.search(r"\{.*\}", out.stdout, re.S).group(0))
    assert d["passed"] is True
    assert d["median_distinct_sets"] >= 2.0


class TokenGatedRetriever:
    """Returns a unit only when the query quotes its exact token -- like the synth suite, where
    the key lives in the DOCUMENT and never in the need text."""

    def search(self, query, k):
        return [_Unit("u0")] if "K1A2B3C4" in query else []


def test_an_inapplicable_probe_is_not_a_failure():
    """On synth and tau2 a gold node is a FACT or a document title, and the thing that keys
    retrieval lives in the document. Reporting FAIL there would be the instrument mistaking its
    own inapplicability for a property of the suite -- the exact error this module exists to
    prevent elsewhere. Measured: synth reports INAPPLICABLE, musique PASSes."""
    tasks = [
        probe_task(
            task_id=f"t{i}",
            question="report the final value",
            subquestions=["the value for facet 0 at step 1 is V001"],
            gold_uids=frozenset({"u0"}),
            retriever=TokenGatedRetriever(),
            k=5,
        )
        for i in range(8)
    ]
    res = probe(tasks, suite_id="synth-like")
    assert not res.applicable
    assert not res.passed
    assert "INAPPLICABLE" in res.verdict
    assert "FAIL" not in res.verdict


def test_applicability_needs_only_one_task_to_retrieve_something():
    good = probe_task(
        task_id="ok",
        question="q",
        subquestions=["K1A2B3C4"],
        gold_uids=frozenset({"u0"}),
        retriever=TokenGatedRetriever(),
        k=5,
    )
    bad = probe_task(
        task_id="no",
        question="q",
        subquestions=["nothing"],
        gold_uids=frozenset({"u0"}),
        retriever=TokenGatedRetriever(),
        k=5,
    )
    assert probe([good, bad], suite_id="mixed").applicable
    assert not probe([bad], suite_id="none").applicable
