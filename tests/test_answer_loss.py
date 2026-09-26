"""Lane L3 tooling (`scripts/answer_loss/`): where the evidence gain is lost before the answer.

Hermetic: every fixture is synthetic. No gold root, no isolated store, no network. The
DB-facing and model-facing halves of the two scripts are exercised by their locks on the real
data (decompose.json / reanswer_dryrun.json), not here; this file pins the pure functions those
locks are built from, so a lock that passes cannot be passing because a helper is wrong in the
same direction on both sides.
"""

from __future__ import annotations

import math
import random

import pytest
from scripts.answer_loss import decompose as d
from scripts.answer_loss import reanswer as ra

from pi_eval.metrics.quality import contains_answer, normalize
from pinq.types import Evidence, EvidenceUnit
from pinq_expt.components import UNIT_CHARS, render_evidence

# --------------------------------------------------------------------------- answer location


def test_normalized_tokens_reunite_to_normalize():
    """The offset map must see exactly the token stream `contains_answer` sees, or an offset
    could be reported for a match the scorer would not make (or miss one it would)."""
    texts = [
        "The capital, Paris, is in France.",
        "U.S. state of New-York -- a place; the end.",
        "x a y an z the",
        "Thor\u202f:\u202fRagnarok was released on\u202f24\u202fOctober\u202f2017.",
        "",
        "   leading and trailing   ",
    ]
    for t in texts:
        assert [tok for tok, _s, _e in d.normalized_tokens(t)] == normalize(t).split()


def test_answer_span_is_the_first_token_sequence_match():
    text = "Founded in 1985 by John Smith. Later, Paris became its home; Paris again."
    span = d.answer_span(text, "Paris", ())
    assert span is not None
    start, end = span
    assert text[start:end].startswith("Paris")
    assert start == text.index("Paris")


def test_answer_span_is_token_sequence_not_substring():
    # contains_answer's own docstring: a substring test credits "18" in "1985".
    assert d.answer_span("Founded in 1985", "18", ()) is None
    assert contains_answer("Founded in 1985", "18") == 0.0


def test_answer_span_takes_the_earliest_alias():
    text = "The Big Apple, also called New York City, is large."
    span = d.answer_span(text, "New York City", ("Big Apple",))
    assert span is not None and span[0] == text.index("Big Apple")


def test_answer_span_agrees_with_contains_answer_on_random_texts():
    rng = random.Random(7)
    vocab = ["paris", "the", "Paris,", "new", "york", "city.", "1985", "a", "U.S.", "state"]
    golds = ["Paris", "New York City", "1985", "U.S. state", "the city"]
    for _ in range(400):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 12)))
        g = rng.choice(golds)
        assert (d.answer_span(text, g, ()) is not None) == bool(contains_answer(text, g))


def test_window_reading_distinguishes_start_past_from_not_visible():
    pad = "word " * 300  # 1500 chars
    far = pad[:1300] + " Zanzibar " + pad[:100]
    r = d.window_reading(far, "Zanzibar", (), window=UNIT_CHARS)
    assert r == {"contains": True, "starts_past": True, "visible": False, "longer": True}

    near = "Zanzibar " + pad
    r = d.window_reading(near, "Zanzibar", (), window=UNIT_CHARS)
    assert r == {"contains": True, "starts_past": False, "visible": True, "longer": True}

    # an answer that STRADDLES the cut starts before 1,200 and is still not visible
    straddle = "x" * 1190 + " New York City " + "y" * 50
    start = straddle.index("New")
    assert start < UNIT_CHARS < start + len("New York City")
    r = d.window_reading(straddle, "New York City", (), window=UNIT_CHARS)
    assert r["starts_past"] is False and r["visible"] is False and r["contains"] is True

    short = "Zanzibar is an island."
    assert d.window_reading(short, "Zanzibar", (), window=UNIT_CHARS)["longer"] is False
    assert d.window_reading(short, "Pemba", (), window=UNIT_CHARS)["contains"] is False


# --------------------------------------------------------------------------- the stage chain


def test_cumulative_stages_are_nested():
    assert d.cumulative_stages(False, 1.0, 1.0) == (0.0, 0.0, 0.0)
    assert d.cumulative_stages(True, 0.0, 1.0) == (1.0, 0.0, 0.0)
    assert d.cumulative_stages(True, 1.0, 0.0) == (1.0, 1.0, 0.0)
    assert d.cumulative_stages(True, 1.0, 1.0) == (1.0, 1.0, 1.0)
    # a boolean-answer task has no draft stage: excluded, never scored as a miss
    assert d.cumulative_stages(True, None, 1.0) == (1.0, None, None)


def test_is_boolean_answer():
    assert d.is_boolean_answer("Yes")
    assert d.is_boolean_answer(" no. ")
    assert d.is_boolean_answer("True")
    assert not d.is_boolean_answer("Yes Minister")
    assert not d.is_boolean_answer("1985")


# --------------------------------------------------------------------------- task aggregation


def test_task_means_average_within_task_and_refuse_unbalanced():
    runs = [
        ("t1", 1.0),
        ("t1", 0.0),
        ("t1", 1.0),
        ("t1", 1.0),
        ("t2", 0.0),
        ("t2", 0.0),
        ("t2", 1.0),
        ("t2", 0.0),
    ]
    got = d.task_means(runs, expect_per_task=4)
    assert got == {"t1": 0.75, "t2": 0.25}
    assert list(got) == sorted(got)  # sorted keys: the bootstrap's input order
    with pytest.raises(ValueError):
        d.task_means(runs[:-1], expect_per_task=4)


def test_task_means_skip_none_values():
    got = d.task_means([("t1", None), ("t1", 1.0)], expect_per_task=2)
    assert got == {"t1": 1.0}
    assert d.task_means([("t1", None), ("t1", None)], expect_per_task=2) == {}


def test_pair_task_means_is_pooled_symmetric():
    # (task, recipe value, comparator value) per pair; two training seeds x two rollout seeds
    pairs = [("t1", 1.0, 0.0), ("t1", 0.0, 0.0), ("t1", 1.0, 1.0), ("t1", 1.0, 0.0)]
    a, b = d.pair_task_means(pairs)
    assert a == {"t1": 0.75} and b == {"t1": 0.25}


# --------------------------------------------------------------------------- primary pieces


def test_implied_gain():
    assert d.implied_gain(0.1, 0.5, 0.2) == pytest.approx(0.03)
    assert math.isnan(d.implied_gain(0.1, float("nan"), 0.2))


def test_mde_formula_and_monotone_in_n():
    diffs = [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.05, -0.05]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    want = (1.959963984540054 + 0.8416212335729143) * sd / math.sqrt(n)
    assert d.mde(diffs) == pytest.approx(want, rel=1e-12)
    assert d.mde(diffs * 4) < d.mde(diffs)


def test_stratum_labels():
    assert d.stratum(True, True) == "both"
    assert d.stratum(True, False) == "only_recipe"
    assert d.stratum(False, True) == "only_comparator"
    assert d.stratum(False, False) == "neither"


def test_decided_needs_every_50k_interval_on_one_side():
    pos = [{"n_boot": 50000, "ci_lo": 0.01, "ci_hi": 0.2}] * 3
    assert d.decided(pos) == "DECIDED (+)"
    neg = [{"n_boot": 50000, "ci_lo": -0.3, "ci_hi": -0.001}] * 3
    assert d.decided(neg) == "DECIDED (-)"
    mixed = [
        {"n_boot": 50000, "ci_lo": 0.01, "ci_hi": 0.2},
        {"n_boot": 50000, "ci_lo": -0.001, "ci_hi": 0.2},
        {"n_boot": 50000, "ci_lo": 0.02, "ci_hi": 0.2},
    ]
    assert d.decided(mixed) == "not decided"
    with pytest.raises(ValueError):
        d.decided(pos[:2])


# --------------------------------------------------------------------------- re-answer pieces


def _unit(uid: str, text: str) -> EvidenceUnit:
    return EvidenceUnit(uid=uid, corpus_id="c", doc_id="d", span=uid, title=f"T{uid}", text=text)


def test_render_full_keeps_a_long_unit_whole_where_the_harness_cuts_it():
    long_text = "".join(f"{i:05d} " for i in range(400))  # 2,400 chars
    assert len(long_text) > UNIT_CHARS
    ev = Evidence.of([_unit("u1", long_text), _unit("u2", "short")])
    full = ra.render_full(ev)
    assert long_text in full
    assert long_text not in render_evidence(ev)  # the drafter/answerer window
    assert long_text[:UNIT_CHARS] in render_evidence(ev)
    # same canonical unit order and header format as the harness rendering
    assert full == render_evidence(ev, unit_chars=10**9)


def test_prefix_uids_follow_turn_order_not_list_order():
    turns = [
        {"turn_idx": 2, "retrieved_uids": ["c"]},
        {"turn_idx": 0, "retrieved_uids": ["a", "b"]},
        {"turn_idx": 1, "retrieved_uids": ["b", "d"]},
    ]
    assert ra.prefix_uids(turns, 1) == {"a", "b"}
    assert ra.prefix_uids(turns, 2) == {"a", "b", "d"}
    assert ra.prefix_uids(turns, 3) == {"a", "b", "c", "d"}
    with pytest.raises(ValueError):
        ra.prefix_uids(turns, 4)
    with pytest.raises(ValueError):
        ra.prefix_uids(turns, 0)


def test_word_cap_matches_the_frozen_answerer():
    assert ra.word_cap("a  b\nc d e", 3) == "a b c"
    assert ra.word_cap("", 5) == ""


def test_canary_refusal_fires_on_the_prefix_even_when_unregistered():
    ra.assert_no_canary("an ordinary prompt", frozenset({"PINQCANARY_ABC"}), where="t")
    with pytest.raises(ra.CanaryRefused):
        ra.assert_no_canary("x PINQCANARY_ABC y", frozenset({"PINQCANARY_ABC"}), where="t")
    with pytest.raises(ra.CanaryRefused):
        ra.assert_no_canary("x PINQCANARY_FFFF y", frozenset({"PINQCANARY_ABC"}), where="t")


def test_prompts_carry_no_draft_in_evidence_only_and_nothing_but_the_question_closed_book():
    ev = Evidence.of([_unit("u1", "Paris is the capital of France.")])
    q = "What is the capital of France?"
    p3 = ra.evidence_only_prompt(q, ev, word_cap=30)
    assert "Paris is the capital of France." in p3 and q in p3 and "unknown" in p3
    assert "DRAFT" not in p3
    p0 = ra.closed_book_prompt(q, word_cap=30)
    assert q in p0 and "Paris" not in p0 and "EVIDENCE" not in p0


def test_request_key_is_sha256_of_canonical_request_bytes():
    req = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0}
    k1 = ra.request_key(req)
    k2 = ra.request_key(dict(reversed(list(req.items()))))
    assert k1 == k2 and len(k1) == 64
    assert ra.request_key({**req, "temperature": 1}) != k1


def test_did_per_task():
    r3, c3, r1, c1 = {"t": 0.6}, {"t": 0.4}, {"t": 0.5}, {"t": 0.45}
    a, b = ra.did_inputs(r3, c3, r1, c1)
    assert a["t"] - b["t"] == pytest.approx((0.6 - 0.4) - (0.5 - 0.45))


def test_an_empty_reply_cut_by_the_token_cap_is_an_error_not_a_wrong_answer():
    """Claude Opus 5 spends hidden reasoning tokens inside max_tokens. MEASURED in this lane's
    first full pass at max_tokens=300: 416 of 5,425 Opus replies finished 'length', 395 of them with
    EMPTY content (98 of 200 MuSiQue closed-book calls), and were scored as wrong answers.
    An empty, cap-truncated reply says nothing about the evidence; it must never be scored."""
    assert ra.usable_content("Paris", "stop") == "Paris"
    assert ra.usable_content("unknown", "stop") == "unknown"
    # a non-empty reply that ran into the cap is kept: the answer is stated first
    assert ra.usable_content("Paris, because", "length") == "Paris, because"
    with pytest.raises(ra.ReplyTruncated):
        ra.usable_content("", "length")
    with pytest.raises(ra.ReplyTruncated):
        ra.usable_content("   ", "length")


class _FakeOpus(ra.OpusClient):
    """`OpusClient` with the network and disk replaced: the reply depends only on the cap."""

    def __init__(self, tmp_path, replies):
        super().__init__(tmp_path, frozenset(), api_key="unused")
        self.replies = replies  # max_tokens -> (text, finish_reason)
        self.sent: list[int] = []

    def _complete_locked(self, req, key):
        self.sent.append(req["max_tokens"])
        text, fr = self.replies[req["max_tokens"]]
        ra.usable_content(text, fr)
        return {"request_key": key, "request": req, "text": text, "finish_reason": fr}


def test_an_empty_cap_truncated_reply_is_resent_once_at_the_next_cap(tmp_path):
    """MEASURED in the 12:26 full pass at max_tokens=8000: 5 of 2,976 distinct Opus requests (all
    MuSiQue closed-book) spent all 8,000 completion tokens and returned no content, leaving 5
    planned calls unanswered. Such a request is re-sent at the next cap of the ladder; a reply
    usable at the first cap is never re-sent, so every finished request keeps its key and cache."""
    lo, hi = ra.OPUS_MAX_TOKENS_LADDER
    fake = _FakeOpus(tmp_path, {lo: ("", "length"), hi: ("Paris", "stop")})
    res = fake.complete("q", where="t")
    assert res["text"] == "Paris" and res["max_tokens"] == hi
    assert fake.sent == [lo, hi]

    fine = _FakeOpus(tmp_path, {lo: ("Paris", "stop"), hi: ("never", "stop")})
    res = fine.complete("q", where="t")
    assert res["text"] == "Paris" and res["max_tokens"] == lo
    assert fine.sent == [lo]

    stuck = _FakeOpus(tmp_path, {lo: ("", "length"), hi: ("", "length")})
    with pytest.raises(ra.ReplyTruncated):
        stuck.complete("q", where="t")
    assert stuck.sent == [lo, hi]
