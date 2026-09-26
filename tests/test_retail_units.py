"""Which DB records a retail tool call RETRIEVED, derived from the call's arguments.

WHY FROM THE ARGUMENTS AND NEVER FROM THE RESULT
    A tau2 tool result is a customer record. `tau2_runner` records a `result_digest` and never
    the payload, so the result is not available to name what came back — and must not be made
    available. Retail's tools are key-addressed: MEASURED, 458 of 550 required calls (83.3%)
    name a record directly in their arguments, so the identity is recoverable from the request
    alone. Nothing here reads a tool result.

WHY THIS DUPLICATES THE GOLD BUILDER'S CONVENTION
    `pinq_adapters` may never import `pi_eval` (contract 1), so `record_doc_id`/`record_text`
    exist on both sides and `test_gold_and_adapter_agree_on_every_uid` is what keeps them from
    drifting. This is the same arrangement `common.unit_uid` and `EvidenceUnit.make()` already
    have for the paragraph suites, and for the same reason: a uid mismatch is invisible in
    every downstream number, because it reads as "the policy retrieved nothing relevant"
    rather than as an error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinq_adapters.tau2.retail_units import (
    RetailIndex,
    record_doc_id,
    record_text,
    uids_for_call,
)

DB = {
    "users": {"user_77": {"user_id": "user_77", "email": "a@b.c"}},
    "orders": {
        "#W100": {
            "order_id": "#W100",
            "user_id": "user_77",
            "items": [{"item_id": "item_5150", "product_id": "prod_9"}],
        }
    },
    "products": {
        "prod_9": {
            "product_id": "prod_9",
            "name": "Lamp",
            "variants": {"item_5150": {"item_id": "item_5150", "price": 1.0}},
        }
    },
}


@pytest.fixture
def idx():
    return RetailIndex.from_db(DB)


def test_a_keyed_lookup_names_exactly_its_own_record(idx) -> None:
    got = uids_for_call(idx, "get_order_details", {"order_id": "#W100"})
    assert got == (idx.uid("orders", "#W100"),)


def test_each_table_is_reached_by_its_own_argument(idx) -> None:
    assert uids_for_call(idx, "get_user_details", {"user_id": "user_77"}) == (
        idx.uid("users", "user_77"),
    )
    assert uids_for_call(idx, "get_product_details", {"product_id": "prod_9"}) == (
        idx.uid("products", "prod_9"),
    )


def test_an_item_id_resolves_to_the_product_that_contains_it(idx) -> None:
    """`item_id` names a VARIANT, which is not a top-level record. The record a call for it
    actually reads is the product holding the variant."""
    assert uids_for_call(idx, "get_item_details", {"item_id": "item_5150"}) == (
        idx.uid("products", "prod_9"),
    )


def test_a_list_argument_yields_every_record_it_names(idx) -> None:
    got = uids_for_call(
        idx, "return_delivered_order_items", {"order_id": "#W100", "item_ids": ["item_5150"]}
    )
    assert set(got) == {idx.uid("orders", "#W100"), idx.uid("products", "prod_9")}


def test_a_tool_that_reads_nothing_yields_nothing(idx) -> None:
    """`calculate` and `transfer_to_human_agents` touch no record. Returning a uid for them
    would credit the policy with evidence it never obtained."""
    assert uids_for_call(idx, "calculate", {"expression": "1+1"}) == ()
    assert uids_for_call(idx, "transfer_to_human_agents", {"summary": "x"}) == ()


def test_an_unknown_id_yields_nothing_rather_than_a_fabricated_uid(idx) -> None:
    """A uid for a record that does not exist can never be matched by gold, so it would be a
    silent zero rather than a loud miss."""
    assert uids_for_call(idx, "get_order_details", {"order_id": "#NOPE"}) == ()


def test_find_by_email_names_no_record_because_the_id_is_in_the_RESULT(idx) -> None:
    """The 75 `find_*` calls return a user_id rather than taking one. Reading it would mean
    reading a tool result, which this module must not do. They contribute no evidence here,
    and the subsequent keyed call is what records the user record."""
    assert uids_for_call(idx, "find_user_id_by_email", {"email": "a@b.c"}) == ()


def test_results_are_deduplicated_and_ordered(idx) -> None:
    """`retrieved_uids` is joined and set-hashed downstream; a duplicate would inflate
    `n_retrieved` and therefore the redundancy penalty."""
    got = uids_for_call(
        idx,
        "return_delivered_order_items",
        {"order_id": "#W100", "item_ids": ["item_5150", "item_5150"]},
    )
    assert len(got) == len(set(got)) == 2
    assert list(got) == sorted(got)


# ------------------------------------------------------------------ the agreement test


def test_gold_and_adapter_agree_on_every_uid() -> None:
    """The whole reason both sides may define this. Run over the REAL db, not a fixture."""
    from pi_eval.build.retail_build import corpus_records
    from pi_eval.build.retail_build import record_uid as gold_uid

    p = Path("data/upstream/tau2-bench-data/tau2/domains/retail/db.json")
    if not p.exists():
        pytest.skip("upstream tau2-bench-data checkout not present")
    db = json.loads(p.read_text())
    idx = RetailIndex.from_db(db)
    gold = {r["uid"] for r in corpus_records(db)}
    mine = {idx.uid(t, r) for t in db for r in db[t]}
    assert mine == gold, "adapter and gold mint different uids for the same records"
    # and spot-check one against the gold function directly
    t = "orders"
    rid = next(iter(db[t]))
    assert idx.uid(t, rid) == gold_uid(t, rid, db[t][rid])


def test_gold_and_adapter_agree_on_the_corpus_hash() -> None:
    from pi_eval.build.retail_build import corpus_hash_of

    p = Path("data/upstream/tau2-bench-data/tau2/domains/retail/db.json")
    if not p.exists():
        pytest.skip("upstream tau2-bench-data checkout not present")
    db = json.loads(p.read_text())
    assert RetailIndex.from_db(db).corpus_hash() == corpus_hash_of(db)


def test_record_text_is_canonical_on_both_sides() -> None:
    from pi_eval.build.retail_build import record_text as gold_text

    rec = {"b": 2, "a": 1}
    assert record_text(rec) == gold_text(rec)
    assert record_doc_id("orders", "#W1") == "orders:#W1"
