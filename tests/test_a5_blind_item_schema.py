"""A blinded A5 item is a valid item, and a blinded item that still carries the frontier is not.

The bundle validator refused the blinded instrument outright: `candidates` is in
`_PAYLOAD_REQUIRED["A5"]` and any key not required is "unexpected", so the first blind bundle
died with three errors per item. The point of `--a5-blind` is that the candidate list -- the
unresolved gold nodes -- is NOT in the shipped item, so the schema has to say that instead of
assuming it.

WHY `blind` AND `candidates` ARE MUTUALLY EXCLUSIVE rather than merely optional: the whole
claim of the variant is that the item does not carry the frontier. An item marked blind that
still shipped the list would produce a bundle whose name says one thing and whose bytes say
another, and the measurement it exists to make would be silently the old one.
"""

from __future__ import annotations

from pi_eval.annotate import bundle_shape_errors

BASE = {
    "item_id": "a" * 16,
    "task_type": "A5",
    "context": {
        "question": "Who founded the company?",
        "evidence": [{"uid": "u1", "title": "T", "text": "passage"}],
        "history": [{"q": "q", "a": "a"}],
        "answer": "Jane Roe.",
    },
    "payload": {"candidates": [{"node_id": "n1", "text": "the founder"}]},
    "provenance": {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"},
}


def _errs(item):
    """The validator works on a bundle; wrap one item in the smallest valid envelope."""
    errs = bundle_shape_errors({"bundle_id": "b", "items": [item]})
    # Errors are labelled `<item_id>/<task_type>`, not by index. Filtering on an index matched
    # nothing and made two of these tests pass while asserting on an empty list.
    return [e for e in errs if str(item["item_id"]) in e]


def _blind():
    it = {**BASE, "payload": {"blind": True}}
    it["context"] = {k: v for k, v in BASE["context"].items() if k != "answer"}
    return it


def test_the_sighted_item_is_still_valid():
    assert _errs(BASE) == []


def test_a_blind_item_is_valid_without_candidates_or_an_answer():
    assert _errs(_blind()) == []


def test_a_blind_item_that_still_ships_the_frontier_is_refused():
    it = _blind()
    it["payload"] = {"blind": True, "candidates": [{"node_id": "n1", "text": "the founder"}]}
    errs = _errs(it)
    assert errs and any("blind" in e for e in errs)


def test_a_sighted_item_without_candidates_is_still_refused():
    """The regression the change could cause: `candidates` stays required when not blind."""
    it = {**BASE, "payload": {}}
    assert any("candidates" in e for e in _errs(it))


def test_the_sampler_builds_a_blind_item_the_schema_accepts():
    """THE BUG THIS CAUGHT. `sample_a5_items(blind=True)` dropped `candidates` from the
    payload and left `answer` in the context, so every item it produced was refused by the
    rule directly above -- the second blinded bundle build died on all 320 items.

    The two must move together: the variant's whole claim is that the shipped item holds
    neither the gold frontier nor the answer, and a sampler that drops one but not the other
    produces a bundle whose name says blind and whose bytes are half sighted.
    """
    from pi_run.cmd_annotate import _make_item

    prov = {"suite": "musique", "task_id": "t1", "task_key": "t1", "graph_version": "v1"}
    ctx = {
        "question": "Who founded the company?",
        "evidence": [{"uid": "u1", "title": "T", "text": "passage"}],
        "history": [{"q": "q", "a": "a"}],
    }
    it = _make_item("A5", prov, {"blind": True}, ctx)
    assert _errs(it) == []
    assert "answer" not in it["context"] and "candidates" not in it["payload"]
