"""The citation judge must not pay to extract claims it can never check.

MEASURED. `pi render` disqualified citation_support on both suites that have reports:

    drgym/citation_support   no judgment parsed (540 attempted)
    musique/citation_support no judgment parsed (320 attempted)

and `scripts/measure_sigma_j.py` reported citation_support with n_pairs=0, sigma=None.

The cause is not a parser. `citation.grade` calls `extract_claims` -- a LIVE LLM call -- on
every run, then drops every claim whose cited URLs have no text in `docs`. `docs` comes from
`runs/<run_id>/judge_docs.json`, and NOTHING WRITES THAT FILE: 0 of 729 run directories have
one. So the judge extracts claims correctly, discards all of them as unresolved, and returns
no judgments -- having already billed one extraction per run, on every scoring pass, forever.

`load_judge_docs`'s own docstring states the intended behaviour: "An absent sidecar means the
citation judge IS NOT RUN for that run and the omission is counted -- it is not run against
empty documents". That intent was never implemented at the call site.

Two things follow. The judge must SKIP before spending, and the reason reported must be the
true one: "no source documents", not "no judgment parsed", which sends the next reader to
the parser.
"""

from __future__ import annotations


class _Boom:
    """Any LLM call at all is a failure of the contract under test."""

    def complete(self, **kw):
        raise AssertionError("citation.grade called the model with no documents to check against")


def test_no_documents_means_no_llm_call_at_all() -> None:
    from pi_eval.judges import citation

    judgments, diags = citation.grade(
        _Boom(),
        report="Some report with a claim. https://example.com/a",
        docs={},
        suite_id="drgym",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert judgments == ()
    assert diags.get("skipped") is True


def test_the_diagnostic_names_the_real_reason() -> None:
    """ "no judgment parsed" sends the reader to the parser; the parser is fine."""
    from pi_eval.judges import citation

    _, diags = citation.grade(
        _Boom(), report="r", docs={}, suite_id="s", task_id="t", run_id="r", judge_model="m"
    )
    why = str(diags.get("why", "")).lower()
    assert "document" in why or "judge_docs" in why, diags


def test_an_empty_report_also_skips() -> None:
    from pi_eval.judges import citation

    judgments, diags = citation.grade(
        _Boom(),
        report="",
        docs={"u": "text"},
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert judgments == () and diags.get("skipped") is True


def test_with_documents_it_still_runs() -> None:
    """The skip must be narrow: given docs, the judge must behave exactly as before."""
    from pi_eval.judges import citation

    class _LLM:
        def __init__(self):
            self.calls = 0

        def complete(self, **kw):
            self.calls += 1
            if self.calls == 1:
                return (
                    '{"claims": [{"claim_id": 1, "claim": "c", "sources": ["https://a"]}]}',
                    None,
                )
            return ('{"support": "full_support", "justification": "ok"}', None)

    llm = _LLM()
    judgments, diags = citation.grade(
        llm,
        report="A claim. https://a",
        docs={"https://a": "supporting text"},
        suite_id="s",
        task_id="t",
        run_id="r",
        judge_model="m",
    )
    assert llm.calls == 2, "extract + one check"
    assert len(judgments) == 1
    assert not diags.get("skipped")
