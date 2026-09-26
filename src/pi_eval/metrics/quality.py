"""Task-quality primitives. Deliberately judge-free where a gold answer exists."""

from __future__ import annotations

import json
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _overlap(pred: str, gold: str) -> tuple[int, int, int]:
    """(overlap, |pred|, |gold|) over normalized token multisets."""
    p, g = normalize(pred).split(), normalize(gold).split()
    return sum((Counter(p) & Counter(g)).values()), len(p), len(g)


def token_recall(pred: str, gold: str) -> float:
    """Of the gold tokens, how many did the answer contain?

    THE LENGTH-ROBUST HALF OF THE PRIMARY. Appending words can never lower this, which is
    exactly why it exists: on 164 musique runs, `answer_token_f1` is rank-correlated with
    answer length at -0.677 pooled and -0.934 within `inquirer_prompted` alone, and the arms
    differ in mean length by 7 words. An F1 gap of that size cannot be told apart from a
    brevity gap. Recall can.

    An empty prediction scores 0, never 1 -- `token_f1` returns `float(p == g)` for the
    empty case, which is right for a symmetric measure and wrong for a one-sided one.
    """
    overlap, _, n_gold = _overlap(pred, gold)
    return overlap / n_gold if n_gold else 0.0


def token_precision(pred: str, gold: str) -> float:
    """Of what the answer said, how much was gold. This is the half length destroys."""
    overlap, n_pred, _ = _overlap(pred, gold)
    return overlap / n_pred if n_pred else 0.0


def decompose_over_aliases(
    pred: str, gold: str, aliases: tuple[str, ...] = ()
) -> tuple[float, float, float]:
    """(f1, precision, recall) against the alias that MAXIMIZES F1 -- the one the primary scored.

    Not three independent maxima. `best_over_aliases` reports the F1 of one particular gold
    string; taking the best recall over aliases separately could report the recall of a
    DIFFERENT string, and then the two numbers decompose nothing. The claim being made is
    "recall is the length-robust half of this primary", and that sentence is only true if
    both halves are measured against the same gold.
    """
    # Empty gold strings are dropped BEFORE the argmax. `token_f1("", "")` is 1.0, so an
    # empty alias would hand a perfect primary to an answer that said nothing. Measured: 0
    # of 16,768 current gold records carry one -- this guards a silent failure, cheaply.
    candidates = tuple(g for g in (gold, *aliases) if str(g or "").strip())
    if not candidates:
        return 0.0, 0.0, 0.0
    best = max(candidates, key=lambda g: token_f1(pred, g))
    return token_f1(pred, best), token_precision(pred, best), token_recall(pred, best)


def exact_match(pred: str, gold: str, aliases: tuple[str, ...] = ()) -> float:
    n = normalize(pred)
    return float(any(n == normalize(g) for g in (gold, *aliases)))


def best_over_aliases(pred: str, gold: str, aliases: tuple[str, ...] = ()) -> float:
    return max(token_f1(pred, g) for g in (gold, *aliases)) if gold or aliases else 0.0


# THE REFUSAL IDIOM, not negation. `answerer_frozen` ends with "Where the evidence does not
# support a part of the task, say so in a clause rather than guessing", and an answer that
# takes that instruction scores ~0 on every answer-quality metric in this module. The pattern
# below is deliberately keyed on the IDIOM ("the evidence does not ...", "cannot be
# determined") rather than on "not"/"no", because a gold answer may legitimately contain a
# negation -- "Not Guilty", "The No. 1 Ladies' Detective Agency" -- and a detector that fired
# on those would silently reclassify correct answers as refusals.
_HEDGE = re.compile(
    r"\b(?:"
    r"does\s+not\s+(?:specify|state|say|indicate|mention|provide|confirm|support|name|give)"
    r"|do\s+not\s+(?:specify|state|say|indicate|mention|provide|confirm|support|name|give)"
    r"|no\s+(?:evidence|information|mention|indication|details?)\b"
    r"|provides?\s+no\s+information"
    r"|cannot\s+be\s+determined|cannot\s+determine|can\s*not\s+be\s+determined"
    r"|cannot\s+(?:answer|conclude)"
    r"|is\s+unclear|it\s+is\s+unclear|unclear\s+from"
    r"|not\s+(?:specified|stated|given|available|mentioned|provided)"
    r"|insufficient\s+(?:evidence|information)|is\s+insufficient"
    r"|not\s+possible\s+to\s+(?:determine|answer)"
    r")\b",
    re.IGNORECASE,
)


def is_hedged(text: str) -> bool:
    """Did the Answerer decline to commit rather than name an answer?

    NOT a quality judgement on its own: a refusal is CORRECT when the evidence genuinely does
    not carry the answer. It becomes a confound only when it is compared across arms without
    being held fixed, which is why this is emitted as its own metric instead of being folded
    into `answer_token_f1`. See tests/test_answer_hedging.py for the measurement that
    motivated it.

    An empty answer returns False: that is a different failure (`answer_wellformed`) and
    merging the two would make each one unreadable.
    """
    if not text or not text.strip():
        return False
    return bool(_HEDGE.search(text))


def contains_answer(pred: str, gold: str, aliases: tuple[str, ...] = ()) -> float:
    """1.0 if the answer NAMES the gold span (or an alias), else 0.0. Judge-free.

    LENGTH-ROBUST, NOT HEDGE-ROBUST. It exists because `token_f1` divides by the length of the
    prediction, so a correct answer wrapped in a refusal clause -- "the evidence does not
    specify ...; the Dominican Republic, a top Caribbean source, is 18,705" -- scores 0.10
    while containing two gold answers. This asks the weaker question: was it named at all?

    IT DOES NOT REMOVE THE HEDGE CONFOUND, and the first draft of this docstring claimed it
    did. MEASURED over 448 musique runs: corr(hedge, answer_correct) = -0.464 against
    corr(hedge, token_f1) = -0.466 -- indistinguishable. The rescue case above is real but
    RARE: only 1.7% of hedged answers name the gold span at all (0.017 mean here, against
    0.383 on the committed half). A hedging Answerer is not diluting an answer it found; it
    is declining to produce one. So hedge rate must be held fixed by DESIGN (see
    `is_hedged`), and cannot be normalised away by choosing a different quality metric.

    WHY THIS IS NOT A REPLACEMENT FOR F1. It cannot see a wrong answer offered alongside a
    right one, so it is gameable by enumeration. What bounds that here is `TaskView.word_cap`
    (30 on musique), frozen across arms -- an answer has no room to list candidates. Report it
    BESIDE f1/precision, never instead of them.

    Matching is over TOKEN SEQUENCES after the shared `normalize`, never raw substrings: a
    substring test credits gold "18" against "Founded in 1985", and short numeric golds are
    common. Empty gold scores 0.0, unlike `token_f1('', '')`, which is 1.0 by its own
    contract -- inheriting that would hand every task with no gold answer a free success.
    """
    p_toks = normalize(pred).split()
    if not p_toks:
        return 0.0
    for g in (gold, *aliases):
        g_toks = normalize(g).split()
        if not g_toks:
            continue
        n = len(g_toks)
        if any(p_toks[i : i + n] == g_toks for i in range(len(p_toks) - n + 1)):
            return 1.0
    return 0.0


# MEASURED ON THE NORMALISED FORM, not the raw one. "U.S." is four raw characters and two
# normalised; measuring the raw form let it past the floor, whereupon it flagged every question
# containing "which U.S. state ..." -- context the question needs, not a leaked answer.
#
# 5 rather than 4 because a four-digit year clears 4. Gold "2003" against "What date did Bush
# declare the war in Iraq (the 2003 invasion)?" is WORLD KNOWLEDGE used to disambiguate -- the
# model knows the Iraq War was 2003 the way anyone does -- and is categorically unlike
# gpt-5.6-terra naming "Hassan Gouled Aptidon", which can only come from holding the answer
# chain. Both known true positives ("Moscow", "Marcia") are six characters and survive.
#
# Measured over 6,788 recorded questions: raw>=4 flagged 113 (1.66%), normalised>=4 flagged 85,
# normalised>=5 flagged 75 (1.10%), with no known true positive lost at any step.
_MIN_LEAK_LEN = 5

# `gold_answer` carries a canary nonce appended by `write_graphs`. Comparing against the raw
# field never matches -- which is exactly how the first pass of this measurement reported 0%
# leakage for every model, including one that had plainly written the answer into its query.
_CANARY = re.compile(r"\s*PINQCANARY_[0-9A-Fa-f]+\s*")


def leaks_answer(question: str, gold: str, aliases: tuple[str, ...] = ()) -> bool:
    """Does this QUESTION already contain the answer it is supposed to be seeking?

    A stronger model has read the benchmark. Measured over 25 musique tasks at turn 0 with no
    evidence retrieved: gpt-oss-120b 0/25, gpt-5.6-terra 3/25 (12%), claude-opus-5 1/21 --
    "Boris Yeltsin death where did he die Moscow Central Clinical Hospital", gold answer
    "Moscow". It is not inferring the chain; it is recalling it.

    THE REWARD CANNOT DETECT THIS ON ITS OWN. Naming the answer retrieves the gold passage
    perfectly, so a leaked candidate earns MAXIMUM evidence gain and wins its preference pair.
    Contamination concentrates in precisely the candidates the pipeline selects as best.

    VERBATIM RECALL ONLY. "Where did Russia's first president die?" is invisible here, so this
    bounds the damage rather than removing it -- a reason to keep the frontier generator under
    measurement, not a licence to trust it.

    Shares `normalize` and the token-sequence match with `contains_answer`: one definition of
    "this text names that span", so the two cannot disagree about a candidate.
    """
    clean = _CANARY.sub(" ", gold or "")
    if len(normalize(clean).replace(" ", "")) < _MIN_LEAK_LEN:
        return False
    return bool(contains_answer(question, clean, tuple(_CANARY.sub(" ", a) for a in aliases)))


def is_toolcall_answer(text: str) -> bool:
    """True when the "answer" is a serialized tool call rather than an answer.

    MEASURED, and arm-dependent: 40.0% of random_q answers, 28.6% of drafter_only and 14.3%
    of verbosity are a retrieval request like
    `{"query": "...", "top_n": 5, "source": "corpus"}`, against 0.0% for every inquirer
    variant, ircot, checklist, query_expansion and compute_matched. An arm comparison that
    does not report this rate cannot tell "asked better questions" apart from "emitted
    prose at all".

    A JSON OBJECT OR ARRAY, not merely parseable JSON. `json.loads("1960")` succeeds and a
    year is an answer; requiring a container is what keeps a correct short answer from being
    counted as a malformation. Prose that merely contains braces does not parse and is
    likewise safe.
    """
    t = (text or "").strip()
    if not t or t[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(t), (dict, list))
    except (ValueError, TypeError):
        return False
