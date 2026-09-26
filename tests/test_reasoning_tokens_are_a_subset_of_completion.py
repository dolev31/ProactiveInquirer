"""`tok_reasoning` is a SUBSET of `tok_completion`, so pricing must not add it a second time.

Measured 2026-09-19 against the live gateway, model aws/gpt-oss-120b, one call per spend read,
45s apart so nothing could coalesce:

    in=1592 out= 110 reasoning=101    in=1594 out= 106 reasoning= 96
    in=1593 out= 101 reasoning= 91    in=1596 out= 131 reasoning=122
    in=1596 out=  82 reasoning= 71

`completion_tokens` is 110 while `reasoning_tokens` is 101 -- reasoning is INSIDE completion, as
the OpenAI usage schema defines it, and `_read_response` copies both fields straight across:

    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning  = int(getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0

`PriceTable.usd` already knows this convention for the other nested counter -- its own docstring
says "tok_cached is a SUBSET of tok_prompt; only the remainder is billed at input rate", and it
subtracts: `uncached = max(0, tok_prompt - tok_cached)`. It does not do the same for reasoning; it
adds `tok_reasoning * rate_reasoning` on top of the full `tok_completion * rate_out`. So every
reasoning-model call is billed for its reasoning twice.

Scale, on the 408 rescued tau2 units (gpt-oss-120b, non-cache-hit): tok_prompt 19,198,503,
tok_completion 2,633,082, tok_reasoning 1,910,572. The ledger recorded $6.2875; charging
completion once gives $4.8546. The recorded figure is 1.2952x the correct one.

Over the whole runs store -- the compacted parquet's 2,159,621 calls PLUS the 508,493 in the 74,349
run directories the parquet had not yet seen, each population reproducing its own recorded column at
1.000000 -- $1917.5448 was recorded where $1377.1588 is correct: overstated by $540.3861, a factor of
1.3924. Do not quote that as a closed account: the store moves, and compaction had not been run.

Two independent measurements say the single-charge figure is the right one:
  - on the four calls above that the gateway posted, our table applied to completion ONCE gives
    ratios 1.054 / 1.051 / 1.062 / 1.043 against what the gateway charged -- median 1.052.
    Double-charging would put that ratio near 1.36.
  - the gateway charged 0.000000000 for one call in five, so its counter is a lower bound; the
    remaining spread is posting failure, not price.

These tests fail on the pre-fix arithmetic. The first one is the mechanism; the rest pin the
convention so it cannot be reintroduced for a third nested counter.
"""

from __future__ import annotations

import json

import pytest

from pinq_adapters.llm.pricing import PriceTable

# A call where reasoning is most of the completion, as the measurement above found it to be.
IN, OUT, REASON = 1592, 110, 101
RATE_IN, RATE_OUT = 0.15, 0.75


@pytest.fixture
def table(tmp_path):
    p = tmp_path / "price.json"
    p.write_text(
        json.dumps(
            {
                "version": "test-2026-09",
                "models": {
                    "aws/gpt-oss-120b": {
                        "input": RATE_IN,
                        "output": RATE_OUT,
                        "cached_input": None,
                        "reasoning": None,
                    }
                },
            }
        )
    )
    return PriceTable.load(p)


def _charge(table, **kw):
    return table.usd("aws/gpt-oss-120b", **kw)


def test_reasoning_is_not_charged_on_top_of_completion(table):
    """The mechanism: completion already contains the reasoning tokens."""
    expected = (IN * RATE_IN + OUT * RATE_OUT) / 1e6
    got = _charge(table, tok_prompt=IN, tok_completion=OUT, tok_reasoning=REASON)
    assert got == pytest.approx(expected), (
        f"reasoning charged twice: {got:.9f} vs {expected:.9f} "
        f"(excess {got - expected:.9f}, ratio {got / expected:.4f})"
    )


def test_declaring_reasoning_tokens_never_raises_the_bill(table):
    """Reporting the breakdown of a completion cannot cost more than not reporting it."""
    silent = _charge(table, tok_prompt=IN, tok_completion=OUT, tok_reasoning=0)
    declared = _charge(table, tok_prompt=IN, tok_completion=OUT, tok_reasoning=REASON)
    assert declared == pytest.approx(silent), (
        "a provider that exposes reasoning_tokens is billed more than one that hides it, "
        "for the identical completion"
    )


def test_all_reasoning_completion_is_charged_once(table):
    """The degenerate case the client's own module docstring records: completion == reasoning."""
    got = _charge(table, tok_prompt=IN, tok_completion=1200, tok_reasoning=1200)
    expected = (IN * RATE_IN + 1200 * RATE_OUT) / 1e6
    assert got == pytest.approx(expected), (
        f"charged 2x output on an all-reasoning completion: {got:.9f}"
    )


def test_a_separate_reasoning_rate_replaces_the_output_rate_for_that_subset(table, tmp_path):
    """When the table prices reasoning separately, the subset moves to that rate -- it is not extra.

    Priced explicitly at half the output rate, a completion that is entirely reasoning must cost
    half of one priced entirely at the output rate, never one and a half times it.
    """
    p = tmp_path / "price_reasoning.json"
    p.write_text(
        json.dumps(
            {
                "version": "test-2026-09",
                "models": {
                    "aws/gpt-oss-120b": {
                        "input": RATE_IN,
                        "output": RATE_OUT,
                        "cached_input": None,
                        "reasoning": RATE_OUT / 2,
                    }
                },
            }
        )
    )
    t2 = PriceTable.load(p)
    got = t2.usd("aws/gpt-oss-120b", tok_prompt=0, tok_completion=1000, tok_reasoning=1000)
    expected = 1000 * (RATE_OUT / 2) / 1e6
    assert got == pytest.approx(expected), (
        f"a priced reasoning subset was added to the output charge instead of replacing it: "
        f"{got:.9f} vs {expected:.9f}"
    )


def test_the_cached_subset_convention_still_holds(table):
    """Non-vacuity guard: the sibling subset must keep working, or this file proves nothing.

    If a fix to reasoning broke tok_cached, the two would be inconsistent again in the other
    direction and this file would pass while the arithmetic was still wrong.
    """
    got = _charge(table, tok_prompt=1000, tok_completion=0, tok_cached=400)
    expected = 600 * RATE_IN / 1e6  # cached_input null -> the cached subset is free
    assert got == pytest.approx(expected), (
        f"tok_cached is no longer a subset of tok_prompt: {got:.9f}"
    )


def test_reasoning_exceeding_completion_cannot_be_charged_twice(table):
    """A malformed usage block must not become a bigger bill than a well-formed one.

    Guards the clamp: whatever the provider reports, the reasoning subset cannot exceed the
    completion it is part of.
    """
    got = _charge(table, tok_prompt=0, tok_completion=100, tok_reasoning=500)
    expected = 100 * RATE_OUT / 1e6
    assert got == pytest.approx(expected), (
        f"an over-reported reasoning count inflated the charge to {got:.9f}"
    )
