"""How different are the two questions in a pair? A LABEL, never a filter.

THE DEFECT THIS MEASURES. `export_pairs` drops a pair whose two actions are byte-identical
(`n_identical_dropped` = 7,025 on the live export). It cannot drop a pair whose two questions
are PARAPHRASES, and paraphrases are what a temperature-1.0 sampler mostly produces at a state
where only one need is really available. Measured over 49 freshly forked states, taking the
mean pairwise content-word Jaccard between the eight candidates at each:

    >= 0.70   near-paraphrase       18% of states
    0.40-0.70 related               33%
    <  0.40   genuinely different   49%

An example from the near-paraphrase group, three of eight candidates at one state:

    "Which country has the world's oldest navy and what is its primary language?"
    "Which country possesses the world's oldest navy and what is the primary language spoken..."
    "Which country has the oldest navy in the world and what language is spoken there?"

Those are distinct strings, so every guard in the exporter passes them. They are one question.
A preference over two of them is a preference over WORDING, and the outcome label that orders
them -- which rollout happened to answer correctly -- is downstream luck on that pair, because
the two questions retrieve the same thing. Training on it teaches the policy a phrasing. That
is the same failure the A2 campaign found from the other direction, where the automatic margin
turned out to track retrieval novelty rather than question quality.

WHY A LABEL AND NOT A GUARD. A threshold on a similarity score is a judgement call, and this
repository's pattern for a judgement call is to attach the number and let the consumer decide
(`DPOConfig.min_q_distinctness` puts that choice in `cfg.sha`). It is also why the field is
attached by `pi train promote` rather than by the exporter: the exporter's output feeds a
`scorer_hash`, and adding a computed column there would re-identify every existing row for a
quantity that is diagnostic rather than definitional.

WHY CONTENT-WORD JACCARD AND NOT AN EMBEDDING. It needs no model, no key and no network, so it
is reproducible from the artifact alone by anyone, forever -- the same argument
`scripts/validate_pairs.py` makes for itself. It is blunt: it calls two questions different
when they differ only in a named entity, and that is the direction that costs recall rather
than the direction that admits noise. An embedding version can be added later and compared
against it; this one cannot rot.
"""

from __future__ import annotations

import json
import re

# Function words carry no information about WHICH need a question pursues, and leaving them in
# floors the similarity of any two English questions at a misleadingly high number.
STOPWORDS = frozenset(
    """the a an of in to and or is are was were what which who whom whose that this these
    those for on at by with from as it its their his her be been being do does did have has
    had can could would should will may might about into over under between""".split()
)

# Ordered least distinct first, so `min_q_distinctness` can name a FLOOR. Mirrors
# `pinq_train.rung2_dpo.Q_DISTINCTNESS`, which is the consumer of this label.
PARAPHRASE_AT = 0.70
RELATED_AT = 0.40

_WORD = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """1.0 when the two token sets are identical, 0.0 when disjoint.

    Two empty questions score 1.0 rather than 0.0: they ARE the same (empty) question, and
    scoring them maximally different would quietly promote the most degenerate pairs in the
    set to the top of any diversity-sorted selection.
    """
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def question_of(action_json: str) -> str:
    try:
        d = json.loads(action_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    return str(d.get("question") or d.get("text") or "")


def distinctness(chosen_json: str, rejected_json: str) -> tuple[float, str]:
    """(jaccard, bucket) for one pair's two actions.

    A STOP parses to an empty question, so an ask_stop pair scores against "" -- which is
    correct in the only sense that matters here: the two ACTIONS are maximally different, and
    the bucket is never the reason such a pair is kept or dropped (`include_pair_kinds` is).
    """
    j = jaccard(
        content_tokens(question_of(chosen_json)), content_tokens(question_of(rejected_json))
    )
    bucket = "paraphrase" if j >= PARAPHRASE_AT else ("related" if j >= RELATED_AT else "different")
    return round(j, 4), bucket
