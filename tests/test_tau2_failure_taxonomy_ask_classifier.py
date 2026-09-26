"""scripts/tau2_failure_taxonomy/ask_classifier.py, pinned against the 40-ask hand-validation
worksheet (Lane L4.3b, `artifacts/tau2_failure_taxonomy_20260919/RESULT.md` ss1).

Every string below is a REAL `question` field copied from a real run's `turns.jsonl` (retail,
airline; published and degraded populations), not an invented example -- so a future edit that
"simplifies" the regex and breaks one of these is breaking agreement with a hand-checked case,
not an arbitrary unit test. Two cases (`test_imperative_confirmation_of_a_record_fact_is_still_
customer`, the "reservation number for the flight you took" case) are pinned with a comment
explaining why the answer is the documented rule's, not the only defensible one -- see the
module docstring's rule 1 and this file's own notes.
"""

from __future__ import annotations

from scripts.tau2_failure_taxonomy.ask_classifier import classify_ask


def test_empty_or_none_is_unclear():
    assert classify_ask(None) == "unclear"
    assert classify_ask("") == "unclear"
    assert classify_ask("   ") == "unclear"


# ------------------------------------------------------------------------- tool-directed asks


def test_order_lookup_by_id_is_tool():
    q = "What are the full details (status, items, total, payment history/payment method used) of order #W4923227?"
    assert classify_ask(q) == "tool"


def test_order_lookup_with_named_user_is_tool():
    q = "What are the details (status, address, items) of order #W5733668 for user ethan_garcia_1261?"
    assert classify_ask(q) == "tool"


def test_user_id_lookup_by_name_and_zip_is_tool():
    # the find_user_id_by_name_zip shape: identity WORDS (name, zip) used as an ALREADY-GIVEN
    # lookup key, not solicited from anyone -- rule 2 fires before rule 3 can call it customer.
    q = "What is the user id associated with name 'Chen Johnson' and zip code 77004?"
    assert classify_ask(q) == "tool"


def test_user_id_lookup_for_a_customer_with_named_fields_is_tool():
    q = "What is the user id for the customer with first name Yusuf, last name Li, and zip code 91148?"
    assert classify_ask(q) == "tool"


def test_flight_status_lookup_plural_flights_is_tool():
    # regression pin: `flights` (plural) must match as readily as `flight` (singular).
    q = "What is the current status (e.g., available, delayed, on time, cancelled) of flights HAT176 (CLT-DTW) and HAT097 (DTW-PHX) on 2024-05-20?"
    assert classify_ask(q) == "tool"


def test_reservation_details_lookup_is_tool():
    q = "get_reservation_details for reservation JG7FMM: what cabin class and flight details does it contain?"
    assert classify_ask(q) == "tool"


def test_membership_level_lookup_is_tool():
    q = "What is the membership level (regular, silver, or gold) of user sophia_taylor_9065?"
    assert classify_ask(q) == "tool"


# --------------------------------------------------------------------- customer-directed asks


def test_direct_your_order_number_is_customer():
    q = "Could you please provide your order number so I can look up the specific order details for you?"
    assert classify_ask(q) == "customer"


def test_bare_what_is_your_order_id_is_customer():
    assert classify_ask("What is your order ID?") == "customer"


def test_imperative_solicit_without_your_is_still_customer():
    # the fix this worksheet forced: "provide the order ID for the bookshelf" has no "your"
    # at all, but "could you please provide" addresses "you" by its own grammar regardless of
    # what noun follows -- see the module docstring's note on this exact pair of examples.
    q = "Could you please provide the order ID for the bookshelf so I can look up its delivery status and proceed with the return?"
    assert classify_ask(q) == "customer"


def test_third_person_customers_email_is_customer():
    q = "What is the customer's email, or their full name and zip code, for account authentication?"
    assert classify_ask(q) == "customer"


def test_third_person_customers_user_id_is_customer():
    # regression pin: the possessive pattern must cover "user id"/"username", not only "name".
    assert classify_ask("What is the customer's user id?") == "customer"


def test_confirmation_of_intent_is_customer():
    q = "Could you please confirm if you would like us to proceed with the name and zip code provided (Mei Ahmed, 78705) to authenticate your account?"
    assert classify_ask(q) == "customer"


def test_imperative_confirmation_of_a_record_fact_is_still_customer():
    # A DEBATABLE CASE, pinned deliberately. The fact being confirmed (an order's status) is a
    # record fact, but the task's own taxonomy names "confirmation" as a customer-class example
    # regardless of subject matter, and rule 1 (imperative solicitation) is checked before rule
    # 2 (record noun) for exactly this reason: confirmation-shaped language always counts as
    # addressed to the customer here, even when what is being confirmed lives in a record.
    q = "Could you please confirm if the order W5061109 is still in a pending status?"
    assert classify_ask(q) == "customer"


def test_passenger_date_of_birth_request_is_customer():
    q = "Could you please provide the number of passengers traveling and their first name, last name, and date of birth?"
    assert classify_ask(q) == "customer"


# ------------------------------------------------------------------------------------ unclear


def test_meta_question_about_tool_availability_is_not_forced():
    # Neither a solicitation of the person nor a record lookup in the sense this rule checks;
    # documented as a known gap rather than silently forced into one bucket (module docstring
    # rule 4). Accepts either tool or unclear -- what it must NOT do is claim "customer".
    q = "Does answering this request require access to the user's actual order/account system (e.g., order lookup, inventory modification, or cancellation processing)?"
    assert classify_ask(q) in ("tool", "unclear")
