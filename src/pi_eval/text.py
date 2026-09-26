"""One stoplist, one content-term tokeniser, for every instrument that compares need text.

TWO INSTRUMENTS NEEDED THE SAME THING AND WOULD HAVE GROWN TWO COPIES OF IT.
`pi_eval.matcher.base` decides whether a question MENTIONS a need; `pi_eval.mining.partition`
decides whether a document COVERS one. Both ask "which words here carry the need, and which are
scaffolding?", and two stoplists for one question is the shape this repository keeps finding
bugs in.

WHY SCAFFOLDING HAS TO GO. `discoverability_of` scored the raw interrogative ASK against a
document, token for token, with no filtering -- so the verdict moved with the PHRASING rather
than the need. Measured on one document about Houston Baptist University:

    "Houston Baptist University founding year"                     cov 0.600  -> kb
    "When was Houston Baptist University founded?"                 cov 0.667  -> kb
    "Could you please tell me in what year ... was founded?"        cov 0.357  -> user_private

The same need, three ways of asking, two different verdicts. `private_share` -- which CLAUDE.md
calls the hard ceiling on what any autonomous inquirer could reach, and "the single most
important number for the framing" -- was therefore partly a property of how politely the
mining policy happened to write, and a verbose policy manufactures a higher ceiling.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

# Deliberately small. A large stoplist is a tuning surface, and an instrument that decides a
# published ceiling should be describable in one sentence: interrogative scaffolding, auxiliary
# verbs, and the politeness a model wraps a question in.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those there here
    what when where which who whom whose why how
    is are was were be been being am
    do does did done doing
    have has had having
    can could may might must shall should will would
    of in on at to for from by with about into over under
    and or but if then than as so such
    you your me my we our it its they them their he she his her
    please tell give show find get let know say
    i s t re ve ll d m
    """.split()
)


def content_terms(text: str, *, min_len: int = 1) -> frozenset[str]:
    """The words that carry the need, lowercased, with scaffolding removed."""
    return frozenset(
        w for w in _WORD.findall((text or "").lower()) if len(w) >= min_len and w not in STOPWORDS
    )


def raw_terms(text: str) -> frozenset[str]:
    """Every word, unfiltered. Kept for callers that genuinely want the surface form."""
    return frozenset(_WORD.findall((text or "").lower()))
