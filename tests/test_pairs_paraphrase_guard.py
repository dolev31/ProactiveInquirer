"""Two candidates that ask the same thing in different words are not a preference pair.

CALIBRATED on the first six ask_ask pairs ever exported. Three were real -- an ask to the record
against an ask to the customer, token-set Jaccard 0.00. Three were the same order lookup reworded:

    "For order #W4219264: what is its status, items, and shipping address?"
    "What are the items, status, and shipping address on order #W4219264?"

token-set Jaccard 1.00, 0.62 and 0.50 -- and the exporter ranked them apart by a 0.10 margin that
came from retrieval luck, not from the question. That is the hazard the byte-identical guard
already names ("they cleared margin because their VALUES differ despite the text being the same"),
one step removed. Trigram overlap missed all three (0.06, 0.06, 0.05): reordering a clause changes
every trigram and no meaning.

Token-set overlap with stopwords removed, at 0.5. The three paraphrases are at or above it; the
three real pairs are at zero; a threshold in between is not a knife-edge.
"""

from pinq_train.export.dataset import is_paraphrase, question_tokens


def test_the_reworded_order_lookup_is_a_paraphrase():
    a = '{"question": "For order #W4219264: what is its status, items, and shipping address?"}'
    b = '{"question": "What are the items, status, and shipping address on order #W4219264?"}'
    assert is_paraphrase(a, b)


def test_the_partial_reword_at_the_threshold_is_a_paraphrase():
    a = '{"question": "What are the status, shipping address, and item details of order #W4219264 for user noah_ito_3850?"}'
    b = '{"question": "What are the items, status, and shipping address on order #W4219264?"}'
    assert is_paraphrase(a, b)


def test_record_versus_customer_is_not_a_paraphrase():
    a = '{"question": "What are the current status, order date, items, and payment method for order #W2702727?"}'
    b = '{"question": "What is the customer\'s email, or their full name and zip code, for identity verification?"}'
    assert not is_paraphrase(a, b)


def test_stopwords_do_not_count_as_shared_meaning():
    assert "the" not in question_tokens('{"question": "What is the status of the order?"}')
    assert "status" in question_tokens('{"question": "What is the status of the order?"}')


def test_a_short_question_whose_one_content_word_flips_the_need_is_not_a_paraphrase():
    """FOUND BY THE EXISTING SUITE. `test_answer_leak_guard` pairs "where did he die?" against
    "when did he die?": token-set Jaccard 0.60, and the first version of this guard dropped it.
    Those are DIFFERENT needs -- a place and a date -- sharing three function-like words. The
    three real paraphrases in the calibration set share five or six content tokens ("order",
    the order id, "status", "items", "shipping", "address"). A ratio alone cannot tell a reworded
    six-token lookup from a four-token question with one word changed; a minimum shared count
    can, and it leaves the calibration set exactly where it was."""
    assert not is_paraphrase(
        '{"question": "where did he die?"}', '{"question": "when did he die?"}'
    )
    assert not is_paraphrase(
        '{"question": "who owned The Collegian"}', '{"question": "who owned it"}'
    ), "the length guard's own fixture must reach the length guard"


def test_a_politely_reworded_ask_is_a_paraphrase():
    """FOUND BY THE ONE-CONTRAST TEST. "Could you provide the user's email address or full name and
    zip code for verification?" scored 0.46 against "What is the user's email, or full name and zip
    code, for identity verification?" because "could", "you", "provide" counted as content. Request
    and politeness words carry no need; they are stopwords here."""
    a = '{"question": "What is the user\'s email, or full name and zip code, for identity verification?"}'
    b = '{"question": "Could you provide the user\'s email address or full name and zip code for verification?"}'
    assert is_paraphrase(a, b)


def test_a_broader_ask_that_subsumes_a_narrower_one_is_a_contrast():
    """The proactive ask often asks for MORE than the adequate one. Containment alone would call
    that a paraphrase; the Jaccard floor keeps it a pair when the extra tokens are real needs."""
    a = '{"question": "What are the status, items and payment method on order #W1?"}'
    b = '{"question": "What are the status, items and payment method on order #W1, and can the shipping address still be changed before dispatch?"}'
    assert not is_paraphrase(a, b)


def test_an_underscored_field_name_is_the_same_word_as_its_spaced_form():
    """MEASURED on pilot D: 'details (product id, price, availability, options) of item id
    1569765161' and 'details (price, availability, and product_id) of item_id 1569765161'
    passed the guard at Jaccard 0.36 because `product_id` and `product id` were different
    tokens. Same need, same record, one wording; the pair was then ordered by speed."""
    a = "What are the details (price, availability, and product_id) of item_id 1569765161?"
    b = "What are the details (product id, price, availability, options) of item id 1569765161?"
    assert is_paraphrase(a, b)
