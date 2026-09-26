"""The A5 stop instrument shows the rater the gold frontier, and that decides the verdict.

WHAT WAS MEASURED (2026-09-07, 545 items with a clean-panel majority). `_build_a5` prints a
section headed "Candidate follow-up needs the run did NOT resolve before stopping", listing
the unresolved required gold nodes -- or "(none -- every mined need was resolved)". The
verdict tracks it almost perfectly:

    needs named   n     said "should have asked more"
    none         264      0.0%
    1            190     26.3%
    2             62     79.0%
    3             21    100.0%
    4              8    100.0%

Holding that list fixed and pooling over strata, the separation against blinded
`answer_correct` falls from 24.5 points to 9.0 (n=275). The paper's "no false positives on
204 complete-coverage states" contributes nothing to the within-stratum estimate: no rater
ever said otherwise there because the prompt told it nothing remained.

The blinding comment in `sample_a5_items` withholds `answer_correct`, `evidence_coverage`,
`stop_reason` and `n_unresolved` from the item -- and then ships the candidate LIST, which
reveals the count it was hiding. That is the defect these tests pin.

The blind variant omits both the final answer and the candidate section. OMITS, not empties:
"(none -- every mined need was resolved)" printed with no candidates asserts the opposite
anchor just as strongly, and a bare "Final answer given:" invites the rater to read a missing
answer as a failure.
"""

from __future__ import annotations

from pi_eval.annotate_llm import build_prompt

ITEM = {
    "item_id": "i1",
    "task_type": "A5",
    "context": {
        "question": "Who founded the company that made the device?",
        "evidence": [{"uid": "u1", "title": "T", "text": "Some retrieved passage."}],
        "history": [{"q": "who made the device?", "a": "Acme."}],
        "answer": "Jane Roe.",
    },
    "payload": {"candidates": [{"node_id": "n7", "text": "the founder of Acme"}]},
}


def _blind(item):
    return {**item, "payload": {"blind": True}}


def test_the_default_variant_still_shows_the_answer_and_the_frontier():
    """Not a change to the shipped instrument: the existing records were collected with it and
    must stay reproducible from the bundle."""
    p = build_prompt(ITEM)
    assert "Final answer given: Jane Roe." in p
    assert "n7: the founder of Acme" in p


def test_the_blind_variant_withholds_the_answer():
    p = build_prompt(_blind(ITEM))
    assert "Jane Roe" not in p
    assert "Final answer given" not in p


def test_the_blind_variant_withholds_the_gold_frontier():
    p = build_prompt(_blind(ITEM))
    assert "n7" not in p
    assert "did NOT resolve" not in p


def test_the_blind_variant_does_not_assert_the_opposite_anchor():
    """An empty candidate section is not neutral. Printing the '(none)' line would tell the
    rater every need was resolved, which is the same defect pointing the other way."""
    p = build_prompt(_blind(ITEM))
    assert "every mined need was resolved" not in p


def test_the_blind_variant_still_shows_what_a_reader_would_judge_from():
    p = build_prompt(_blind(ITEM))
    assert "Who founded the company that made the device?" in p
    assert "Some retrieved passage." in p
    assert "who made the device?" in p


def test_the_blind_variant_asks_for_prose_not_node_ids():
    """The default asks the rater to name candidates by `node_id`. A blind rater was never
    shown one, so asking for them would force either a refusal or an invention."""
    p = build_prompt(_blind(ITEM))
    assert "node_id" not in p
    assert "what it still needed to find out" in p


def test_the_verdict_vocabulary_is_unchanged():
    """Same four verdicts, so the two variants are comparable and `consensus` needs no change."""
    for item in (ITEM, _blind(ITEM)):
        p = build_prompt(item)
        for v in (
            "stopping_was_right",
            "should_have_asked_more",
            "should_have_stopped_earlier",
            "cant_tell",
        ):
            assert v in p


# ----------------------------------------------------------------- parsing


def test_a_blind_should_have_asked_more_response_parses():
    """THE SILENT-LOSS DEFECT. `_parse_a5` checks `missing` against the candidate node ids and
    then REQUIRES it to be non-empty when the verdict is should_have_asked_more. A blind item
    ships no candidates, so that pair of rules rejects every blind response carrying the very
    verdict the instrument exists to detect -- on every item, with no error a resume could
    heal. The rationale cap failed this way and cost 24% of a campaign's coverage."""
    from pi_eval.annotate_llm import parse_reply

    got = parse_reply(
        _blind(ITEM),
        '{"verdict": "should_have_asked_more", '
        '"missing": ["who actually founded Acme"], "rationale": "the founder is never named"}',
    )
    assert got.response["verdict"] == "should_have_asked_more"
    assert got.response["missing"] == ["who actually founded Acme"]


def test_the_sighted_parser_still_refuses_an_invented_node_id():
    """The regression the branch above could cause: on a normal item `missing` is still
    checked against the candidates, because there it IS a node id."""
    import pytest

    from pi_eval.annotate_llm import AnnotationParseError, parse_reply

    with pytest.raises(AnnotationParseError, match="not candidates"):
        parse_reply(
            ITEM,
            '{"verdict": "should_have_asked_more", "missing": ["n99"], "rationale": "r"}',
        )


def test_a_blind_verdict_still_needs_to_say_what_was_missing():
    """Prose instead of a node id, but a verdict that claims something was missed without
    naming it is still unusable as supervision."""
    import pytest

    from pi_eval.annotate_llm import AnnotationParseError, parse_reply

    with pytest.raises(AnnotationParseError, match="requires a non-empty"):
        parse_reply(
            _blind(ITEM),
            '{"verdict": "should_have_asked_more", "missing": [], "rationale": "r"}',
        )
