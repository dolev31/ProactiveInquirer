"""An over-long RATIONALE must not discard the LABEL it sits beside.

THE DEFECT. `pi_eval.annotate.RATIONALE_MAX_CHARS` says exactly what a rationale is:

    A RATIONALE is explanatory metadata, never a label: it sits beside `response`, not inside
    it, and every consensus/gate computation below must be blind to whether it is present.
    Capped short on purpose -- a REASON, not an essay: long free text costs tokens on every
    future read.

The cap therefore exists to bound READ COST. But `annotate_llm` raised `AnnotationParseError`
on a rationale one character over, and the caller counts that as UNPARSED and writes no
record -- so a valid, well-formed tier map was thrown away because a field the gates are
required to ignore ran long.

MEASURED, on the A6 pass: 14 of 430 items lost for gemini, 35 for sonnet, 60 for gpt-oss ---
and three-rater coverage fell from 400 to 303 items, a 24% hole. `--resume` cannot heal it:
it keys on (item_id, prompt_sha), so a retry re-issues the identical prompt, hits the response
cache, and is rejected again. The loss is deterministic.

TRUNCATE, DO NOT REJECT. Truncation satisfies the cap's stated purpose exactly -- the stored
rationale is bounded, so future reads stay cheap -- while keeping the measurement. The record
is marked so an auditor can tell a truncated reason from a short one, and nothing downstream
reads the rationale for a label anyway.
"""

from __future__ import annotations

from pi_eval.annotate import RATIONALE_MAX_CHARS


def _rationale(n: int) -> str:
    return "x" * n


def test_a_rationale_at_the_cap_is_untouched():
    from pi_eval.annotate_llm import clamp_rationale

    r = _rationale(RATIONALE_MAX_CHARS)
    out, was = clamp_rationale(r)
    assert out == r and was is False


def test_an_over_long_rationale_is_truncated_not_rejected():
    """The label survives. This is the case that cost 97 items of three-rater coverage."""
    from pi_eval.annotate_llm import clamp_rationale

    out, was = clamp_rationale(_rationale(RATIONALE_MAX_CHARS + 73))
    assert was is True
    assert len(out) <= RATIONALE_MAX_CHARS, "the cap's purpose is a bounded stored string"
    assert out.endswith("..."), "an auditor must see that it was cut, not written short"


def test_the_truncated_text_is_the_beginning_of_the_reason():
    """A rationale states its conclusion first; keeping the head keeps the useful part."""
    from pi_eval.annotate_llm import clamp_rationale

    body = "Candidate 3 is best because it disambiguates the director. " + "y" * 500
    out, _ = clamp_rationale(body)
    assert out.startswith("Candidate 3 is best because it disambiguates the director.")


def test_an_empty_rationale_is_still_refused_elsewhere():
    """Truncation must not become a way to smuggle in an empty reason: an llm record with no
    stated reason is the one hardest to audit later, and `validate_records` still refuses it."""
    from pi_eval.annotate_llm import clamp_rationale

    out, was = clamp_rationale("")
    assert out == "" and was is False


def test_a_non_string_is_left_for_the_validator():
    from pi_eval.annotate_llm import clamp_rationale

    out, was = clamp_rationale(None)
    assert out is None and was is False
