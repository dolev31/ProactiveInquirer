"""`pinq` core invariants that three ids and one prompt depend on.

Each of these was a well-formed value that was wrong: an id that is not a function of its
inputs, a prompt field that could be filled by data, and an allowlist with a gap in it.
"""

import pytest

from pinq.ids import NonCanonical, canon, request_sha

# ------------------------------------------------------------------ canon is reproducible


def test_a_set_canonicalizes_the_same_way_in_every_process():
    """`default=str` sent a set through repr, whose order depends on PYTHONHASHSEED. Measured
    across three seeds on one set:

        seed=1 -> {"x":"{'beta', 'delta', 'gamma', 'alpha'}"}
        seed=2 -> {"x":"{'delta', 'gamma', 'alpha', 'beta'}"}
        seed=3 -> {"x":"{'gamma', 'beta', 'delta', 'alpha'}"}

    canon feeds request_sha (the LLM cache key) and semantic_hash (run identity), so that is a
    worker missing a cache entry it paid for, and two identical experiments getting different
    run_ids that --resume cannot match."""
    assert (
        canon({"x": {"alpha", "beta", "gamma", "delta"}})
        == '{"x":["alpha","beta","delta","gamma"]}'
    )
    assert canon({"x": {3, 1, 2}}) == canon({"x": frozenset({2, 3, 1})})


def test_an_object_that_cannot_be_canonicalized_raises_instead_of_hashing_its_address():
    """`default=str` on an object without __repr__ put its MEMORY ADDRESS in the id, so the
    same value hashed differently on every call."""

    class Thing:
        pass

    with pytest.raises(NonCanonical, match="cannot be canonicalized reproducibly"):
        canon({"x": Thing()})
    with pytest.raises(NonCanonical):
        request_sha({"messages": [{"role": "user", "content": Thing()}]})


def test_a_path_still_canonicalizes():
    """The one other value with an obviously right answer, and it appears in real payloads."""
    from pathlib import Path

    assert canon({"p": Path("/a/b")}) == '{"p":"/a/b"}'


# ------------------------------------------------- a prompt field cannot inject a placeholder


def _tmpl(text, monkeypatch):
    from pinq import promptlib

    monkeypatch.setattr(promptlib, "load", lambda name: text)
    return promptlib


def test_a_field_value_cannot_be_filled_in_by_a_later_field(monkeypatch):
    """`render` looped text.replace() field by field, so whatever a value CONTAINED stayed live
    for every later field. The values are corpus text and tool output -- 2Wiki paragraphs,
    drgym documents, tau2 tool results -- so a paragraph carrying the literal {{question}} had
    the question substituted into it. Which field could inject which depended on `fields`
    insertion order, so the same template and inputs could render differently with the keyword
    arguments in another order."""
    pl = _tmpl("EVIDENCE:\n{{evidence}}\n\nQUESTION: {{question}}\n", monkeypatch)
    out = pl.render("t", evidence="Doc 1: ... see also {{question}} ...", question="capital?")
    assert "{{question}}" in out, "the value's placeholder must stay inert"
    assert out.count("capital?") == 1, "substituted exactly once, into the template's own hole"


def test_a_forgotten_field_still_raises(monkeypatch):
    pl = _tmpl("Q: {{question}}\nMISSING: {{nope}}", monkeypatch)
    with pytest.raises(pl.UnresolvedPlaceholder, match="nope"):
        pl.render("t", question="q1")


def test_data_that_merely_mentions_a_placeholder_does_not_abort_the_rollout(monkeypatch):
    """The unresolved check reads the TEMPLATE, not the rendered output. Scanning the output
    would abort a run because a retrieved article about templating said "{{x}}"."""
    pl = _tmpl("E: {{evidence}}", monkeypatch)
    assert pl.render("t", evidence="the syntax is {{x}}") == "E: the syntax is {{x}}"


# ------------------------------------------------------- the view allowlist has no gap


def test_instructions_must_be_a_string():
    """Five of the six string fields were type-checked; `instructions` accepted any object --
    a GoldNode, a dict of gold spans, a whole GoldGraph -- which promptlib.render then
    stringifies straight into the prompt. The allowlist rejects an unknown FIELD NAME, and a
    caller who put gold in a KNOWN field walked past it."""
    from pinq.view import LeakageError, make_view

    base = dict(
        task_id="t", suite_id="s", question="q", corpus_id="c", corpus_hash="ch", word_cap=30
    )

    class _Gold:
        def __str__(self):
            return "ANSWER: Tomas Varga"

    with pytest.raises(LeakageError, match="instructions must be a str"):
        make_view(**base, instructions=_Gold())
    # empty is fine here and not for the others: a suite with no extra instructions is ordinary
    assert make_view(**base, instructions="").instructions == ""
