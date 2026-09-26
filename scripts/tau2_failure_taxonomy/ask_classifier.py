"""Classify one Inquirer ask's TEXT as addressed to the customer or to the tools.

WHY THIS IS A CONTENT RULE AND NOT THE `target` FIELD. Every ask in every population this lane
reads carries `target='kb'` (the user channel is closed for `inquirer_prompted`/`self_ask`, and
the degraded population's own `inquirer_may_ask_user` still resolves almost every ask through
the same retrieval channel -- see `build_taxonomy.py`). `target` says which CHANNEL the ask was
routed through, not what the ask is ABOUT. The bug this lane exists to characterise is exactly
the gap between the two: a question can be routed to the records channel while being WORDED as
if the customer were being asked -- "What is your order number?" -- and the records channel has
no customer to ask, so it comes back empty. Classifying `question` text is what makes that gap
visible; classifying `target` would erase it (it is constant).

THE RULE, stated so it can be checked by a reader rather than trusted:

  1. A 2nd-person, direct solicitation of a fact that lives only with the person addressed --
     their identity (name, user id, username, date of birth), a way to reach them (email,
     phone, mailing address), or their confirmation/consent/preference ("would you like",
     "can you confirm", "your budget") -- is CUSTOMER, even when the fact requested is itself
     the KEY to a record ("what is your order number" is soliciting a fact from the customer,
     not looking anything up).
  2. Otherwise, a question that names a business record by type and keys it to an ALREADY-KNOWN
     identifier ("orders associated with user id X", "details of reservation Y", "status of
     flight Z", "the return policy for category W") is TOOL: the fact lives in the system, not
     in anyone's memory, and the phrasing is a lookup rather than a solicitation.
  3. Otherwise, a question still built from identity/contact/confirmation vocabulary (including
     3rd-person: "the customer's email address") is CUSTOMER.
  4. Otherwise UNCLEAR -- reported as its own share rather than forced into one of the two
     classes, per the same rule this repository applies to a near-zero bootstrap bound: an
     honest "don't know" beats a silent coin flip.

VALIDATED BY HAND on 40 asks drawn from both the published and the degraded populations --
see `artifacts/tau2_failure_taxonomy_20260919/RESULT.md` ss1 for the worksheet and the
resulting agreement rate. This module is deliberately small and dependency-free (no tau2, no
pinq import) so that worksheet can re-run the identical function offline.
"""

from __future__ import annotations

import re
from typing import Literal

AskClass = Literal["customer", "tool", "unclear"]

# 2nd-person direct solicitation. Two shapes, because they carry different evidence of who is
# being addressed:
#   IMPERATIVE fires unconditionally -- "could you please provide ..." addresses "you" by its
#   own grammar no matter what noun follows, so it does not need a nearby "your" to confirm it
#   (measured gap: without this, "could you please provide the order ID for the bookshelf" and
#   "could you please provide your order number" were scored TOOL and CUSTOMER respectively,
#   for the same imperative shape aimed at the same kind of fact).
#   WH_YOUR / YOUR_FACT require an explicit "your", because a bare "what is the order ID for
#   the bookshelf" (no "you" anywhere) is not obviously addressed at a person at all.
_IMPERATIVE_SOLICIT = re.compile(
    r"\b(please (provide|confirm|give|share|verify)|"
    r"could you (please )?(provide|confirm|give|tell|share|verify)|"
    r"can you (please )?(provide|confirm|give|tell|share|verify)|"
    r"would you (like|mind|prefer)|"
    r"do you (want|wish|need|have|confirm)|"
    r"are you (sure|able)|"
    r"may i (have|get))\b",
    re.IGNORECASE,
)
_WH_YOUR = re.compile(r"\b(what'?s?|what is|how (much|many))\b.{0,40}\byour\b", re.IGNORECASE)
_YOUR_FACT = re.compile(
    r"\byour (own )?(name|user ?id|username|email|e-?mail|phone|number|address|zip|"
    r"date of birth|dob|identity|account|budget|preferen|spending|income)\b",
    re.IGNORECASE,
)

# A record-type noun (plural-aware: "orders", "flights", ... are the common case in practice,
# not the singular), the vocabulary tau2's own retail/airline/banking tools are named for
# (`get_order_details`, `get_reservation_details`, `search_direct_flight`, the banking
# credit-card/account tools -- see `gold_actions.py`) -- OR a lookup-framing phrase that keys a
# fact to an identifier already on the table ("associated with", "on file for", "details of",
# "for the customer with first name X, last name Y" -- the `find_user_id_by_name_zip` shape).
_RECORD_NOUN = re.compile(
    r"\b(orders?|reservations?|bookings?|itinerar(y|ies)|flights?|baggage|gift ?cards?|"
    r"polic(y|ies)|products?|items?|warrant(y|ies)|refunds?|"
    r"transactions?|memberships?|credit cards?|debit cards?|account balance|"
    r"interest rate|annual fee|credit limit|statements?|disputes?|"
    r"payment methods?|user records?)\b|"
    r"\b(associated with|on file for|details? of|status of)\b|"
    r"\bfor (a |the )?(customer|user)\b.{0,20}\b(with|named|whose)\b",
    re.IGNORECASE,
)

# Identity/contact/confirmation vocabulary WITHOUT requiring 2nd person (rule 3): "the
# customer's email address", "confirm the customer's identity".
_IDENTITY_OR_CONTACT_OR_CONFIRM = re.compile(
    r"\b(identity|identify (the|this) (customer|user|caller)|"
    r"(customer|user|caller)'?s? (name|user ?id|username|email|e-?mail|phone|address|zip|"
    r"date of birth|identity)|"
    r"confirm(ation|ing)?|verify|verification|"
    r"e-?mail address|phone number|date of birth)\b",
    re.IGNORECASE,
)


def classify_ask(question: str | None) -> AskClass:
    """`question` is `turns.jsonl`'s own `question` field. Never raises: an absent or empty
    ask text (measured: 0 occurrences in every population this lane reads, but a taxonomy
    that crashes on the one row it has not seen is worse than one that reports it unclear)."""
    if not question or not question.strip():
        return "unclear"
    text = question.strip()

    if _IMPERATIVE_SOLICIT.search(text) or _WH_YOUR.search(text) or _YOUR_FACT.search(text):
        return "customer"
    if _RECORD_NOUN.search(text):
        return "tool"
    if _IDENTITY_OR_CONTACT_OR_CONFIRM.search(text):
        return "customer"
    return "unclear"
