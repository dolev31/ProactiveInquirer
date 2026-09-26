"""`pi verify judge`: does the instrument return the answer its own rubric mandates?

Plan item 2.1, the last item inside the approved spend. Every judge-derived number in the
paper rests on the judge being usable, and until now the only evidence for that was the
judge's own self-consistency diagnostics (position bias, order agreement) -- which a judge
that is confidently and consistently WRONG passes perfectly.

The items are constructed so the correct answer is not a matter of opinion. Each one is
mandated by the prompt the judge is given:

  kpr, verbatim  -> Supported     "the report affirms, explains or reinforces the key point"
  kpr, absent    -> Omitted       "the report never addresses the key point"
  kpr, negated   -> Contradicted  "the report asserts something incompatible"
  quality, no URL-> support == 0  "if no part of the report gives source URLs, the rating is 0"

A PARSE FAILURE IS A THIRD OUTCOME, never a silent drop. Counting only parseable replies
turns accuracy into accuracy-among-parseable, which is the one number a broken judge scores
well on.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- the items


def test_the_suite_has_enough_items_to_say_anything() -> None:
    from pi_eval.judges.known_answer import build_items

    items = build_items()
    assert len(items) >= 60, f"only {len(items)} items"


def test_every_kpr_label_is_exercised() -> None:
    from pi_eval.judges.known_answer import build_items
    from pi_eval.judges.kpr import LABELS

    got = {i.expected for i in build_items() if i.kind == "kpr"}
    assert got == set(LABELS), f"missing {set(LABELS) - got}"


def test_the_labels_are_balanced() -> None:
    """An unbalanced suite lets a judge that always says one label look competent."""
    import collections

    from pi_eval.judges.known_answer import build_items

    counts = collections.Counter(i.expected for i in build_items() if i.kind == "kpr")
    assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_a_supported_item_really_contains_its_key_point() -> None:
    """If the construction were wrong the 'known' answer would not be known."""
    from pi_eval.judges.known_answer import build_items

    for i in build_items():
        if i.kind == "kpr" and i.expected == "Supported":
            assert i.key_point.lower() in i.report.lower(), i.key_point


def test_an_omitted_item_never_mentions_its_key_point() -> None:
    """CONTENT words, not the first word: many key points open with "the", which any
    English filler contains. Checking the first word tested the filler's grammar."""
    from pi_eval.judges.known_answer import build_items

    stop = {"the", "of", "in", "on", "at", "is", "are", "a", "an", "and", "than", "into"}
    for i in build_items():
        if i.kind != "kpr" or i.expected != "Omitted":
            continue
        content = [w.strip(".,") for w in i.key_point.lower().split() if w not in stop]
        content = [w for w in content if len(w) > 4]
        assert content, f"no content word in {i.key_point!r}"
        overlap = [w for w in content if w in i.report.lower()]
        assert not overlap, f"{i.key_point!r} leaks {overlap} into its OMITTED report"


def test_every_quality_item_is_url_free() -> None:
    """The rubric's zero is conditional on there being no URL anywhere in the report."""
    from pi_eval.judges.known_answer import build_items

    for i in build_items():
        if i.kind == "quality":
            assert "http" not in i.report.lower() and "www." not in i.report.lower()


def test_items_are_deterministic() -> None:
    from pi_eval.judges.known_answer import build_items

    assert [i.item_id for i in build_items()] == [i.item_id for i in build_items()]


def test_items_carry_no_canary_nonce() -> None:
    """Constructed text, not gold text -- but asserted, because it reaches a live model."""
    from pi_eval.judges.known_answer import build_items

    for i in build_items():
        assert "PINQCANARY" not in (i.report + i.key_point)


# --------------------------------------------------------------------------- the scoring


def test_a_parse_failure_is_its_own_outcome() -> None:
    from pi_eval.judges.known_answer import summarize

    out = summarize([("Supported", "Supported"), ("Omitted", None), ("Omitted", "Omitted")])
    assert out["n_parse_failures"] == 1
    assert out["n_items"] == 3
    assert out["accuracy"] == pytest.approx(2 / 3), "a parse failure is not a free pass"


def test_accuracy_is_over_ALL_items_not_the_parseable_ones() -> None:
    """The distinction this whole file exists for."""
    from pi_eval.judges.known_answer import summarize

    out = summarize([("Supported", "Supported")] + [("Omitted", None)] * 9)
    assert out["accuracy"] == pytest.approx(0.1)
    assert out["accuracy_among_parseable"] == pytest.approx(1.0)


def test_per_label_accuracy_is_reported() -> None:
    from pi_eval.judges.known_answer import summarize

    out = summarize([("Supported", "Supported"), ("Supported", "Omitted"), ("Omitted", "Omitted")])
    assert out["by_label"]["Supported"]["accuracy"] == pytest.approx(0.5)
    assert out["by_label"]["Omitted"]["accuracy"] == pytest.approx(1.0)


def test_the_confusion_matrix_shows_the_direction_of_the_error() -> None:
    """ "60% accurate" hides whether the judge over- or under-credits; the matrix does not."""
    from pi_eval.judges.known_answer import summarize

    out = summarize([("Omitted", "Supported"), ("Omitted", "Supported")])
    assert out["confusion"]["Omitted"]["Supported"] == 2


def test_a_perfect_run_reports_perfectly() -> None:
    from pi_eval.judges.known_answer import summarize

    out = summarize([("Supported", "Supported"), ("Contradicted", "Contradicted")])
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["n_parse_failures"] == 0


def test_an_empty_run_is_not_a_perfect_score() -> None:
    from pi_eval.judges.known_answer import summarize

    out = summarize([])
    assert out["n_items"] == 0
    assert out["accuracy"] is None, "0/0 must not read as 1.0"
