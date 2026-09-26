"""Attach a semantic-distance field to `pairs.jsonl` as METADATA. Never a filter.

SUCCEEDED BY `src/pinq_train/export/distinctness.py`, checked 2026-09-19: same thresholds
(paraphrase >= 0.70, related >= 0.40), same measured statistics restated for a later export,
now attached by `pi train promote` as a proper module rather than a standalone script. Kept
here rather than deleted because `src/pinq_train/rung2_dpo/train.py`'s own `min_q_distinctness`
error message still names this file by path as the thing to run first; deleting it would leave
that live, user-facing instruction pointing at nothing. Not a live divergence hazard -- the two
implementations agree -- and not the version to extend; extend `distinctness.py` instead.

THE DEFECT THIS MEASURES. The exporter drops a pair whose two actions are byte-identical
(`n_identical_dropped` = 3,272 on the last export). It cannot drop a pair whose two questions
are PARAPHRASES, and paraphrases are what a temperature-1.0 sampler mostly produces at a state
where only one need is really available. Measured over 49 freshly forked states, taking the
mean pairwise content-word Jaccard between the eight candidates at each:

    >= 0.70  near-paraphrase        18% of states
    0.40-0.70 related               33%
    <  0.40  genuinely different    49%

An example from the near-paraphrase group, all eight candidates at one state:

    "Which country has the world's oldest navy and what is its primary language?"
    "Which country possesses the world's oldest navy and what is the primary language spoken..."
    "Which country has the oldest navy in the world and what language is spoken there?"

Those are eight distinct strings, so every guard in the exporter passes them. They are one
question. A preference over two of them is a preference over wording, and the outcome label
that orders them -- which rollout happened to answer correctly -- is noise on that pair,
because the two questions retrieve the same thing and the difference is downstream luck.

Training on such a pair teaches the policy to prefer a phrasing. That is the same failure the
A2 campaign found from the other direction, where the automatic margin turned out to track
retrieval novelty rather than question quality.

WHY A SIDECAR RATHER THAN A FILTER, AND WHY NOT IN THE EXPORTER. Two reasons, and the second
is the one that matters. A threshold on a similarity score is a judgement call, and this
repository's pattern for a judgement call is to attach the number and let the consumer decide
(see `curate_pairs_with_a2.py`, which attaches rater verdicts and filters nothing). And the
exporter's output is an input to a `scorer_hash`; adding a computed column there would
re-identify every existing row for a quantity that is diagnostic rather than definitional.

WHY CONTENT-WORD JACCARD AND NOT AN EMBEDDING. It needs no model, no key and no network, so it
is reproducible from the artifact alone by anyone, forever -- the same argument
`validate_pairs.py` makes for itself. It is a blunt instrument: it will call two questions
different when they differ only in a named entity, and that is the direction that costs
recall rather than the direction that admits noise. An embedding version can be added later
and compared against it; this one cannot rot.

Usage:
    .venv/bin/python scripts/curate_pairs_with_diversity.py <in.jsonl> <out.jsonl>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Function words carry no information about WHICH need a question pursues, and leaving them in
# floors the similarity of any two English questions at a misleadingly high number.
STOPWORDS = frozenset(
    """the a an of in to and or is are was were what which who whom whose that this these
    those for on at by with from as it its their his her be been being do does did have has
    had can could would should will may might about into over under between""".split()
)

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


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])

    n = 0
    buckets = {"paraphrase": 0, "related": 0, "different": 0}
    with dst.open("w") as out:
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qc = question_of(row.get("chosen_json", ""))
            qr = question_of(row.get("rejected_json", ""))
            j = jaccard(content_tokens(qc), content_tokens(qr))
            row["q_jaccard"] = round(j, 4)
            # A label, not a filter. Named so a reader of the row knows what it means without
            # having to remember which direction the number runs.
            row["q_distinctness"] = (
                "paraphrase" if j >= 0.70 else ("related" if j >= 0.40 else "different")
            )
            buckets[row["q_distinctness"]] += 1
            out.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1

    print(f"rows {n} -> {dst}")
    for k in ("different", "related", "paraphrase"):
        v = buckets[k]
        print(f"  {k:11s} {v:6d}  ({v / n:.1%})" if n else f"  {k}: 0")
    print()
    print("`different` is the subset that can teach WHICH need to pursue. `paraphrase` pairs")
    print("differ in wording only; their outcome label is downstream luck, not question choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
