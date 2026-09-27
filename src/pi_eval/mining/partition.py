"""S7 — required / optional / dropped, and the discoverability partition.

TWO DIFFERENT CUTS, DELIBERATELY KEPT APART.

`partition` answers "does the answer need this?" and is decided by the ABLATION, not by
frequency: NECESSARY -> required, CONTRIBUTORY -> optional, INERT with a tight interval ->
dropped, UNTESTABLE -> kept as optional but excluded from required-recall denominators.
Frequency alone never promotes anything; that distinction is the whole annotation argument.

`discoverability` answers "could an autonomous inquirer ever get this?" and is orthogonal.
A need only the user knows ("I plan to move in three years") is unreachable by any amount of
inquiry, so counting it in a recall denominator measures nothing about the policy. The
USER-PRIVATE SHARE IS THE CEILING on any such system and is reported as a headline number,
not buried.

The instrument for that cut is a knob, and the knob matters: on real tau2 data an n-gram
containment rule put 98.2% of needs in `user_private` and swung 38 points across n=2..5.
So the default is per-document TOKEN OVERLAP with an explicit threshold, `partition_elasticity`
ships the whole curve, and the operating point is recorded in the graph pin. A ceiling number
reported without its elasticity is not a result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from pi_eval.text import content_terms, raw_terms

Partition = Literal["required", "optional", "dropped"]
Discoverability = Literal["kb", "user_private", "unknown"]

_WORD = re.compile(r"[a-z0-9]+")


# The tokeniser IS part of the instrument, so it rides into the mined graph's identity next to
# theta and the NLI pin. Two graphs partitioned under different tokenisers are two different
# measurements and must not share a hash.
DISCOVERABILITY_PIN = "content_terms@v1"


def tokens(text: str) -> frozenset[str]:
    """Every word, unfiltered. Kept for callers that want the surface form."""
    return raw_terms(text)


@dataclass(frozen=True, slots=True)
class PartitionVerdict:
    node_id: str
    partition: Partition
    discoverability: Discoverability
    best_overlap: float
    best_doc_id: str | None
    counts_in_required_denominator: bool


def partition_from_ablation(verdict: str) -> tuple[Partition, bool]:
    """Ablation verdict -> partition, plus whether it counts in the required denominator."""
    if verdict == "NECESSARY":
        return "required", True
    if verdict == "CONTRIBUTORY":
        return "optional", True
    if verdict == "INERT":
        return "dropped", False
    return "optional", False  # UNTESTABLE: kept, but never inflates a required denominator


def discoverability_of(
    node_text: str,
    documents: Iterable[Mapping[str, str]],
    *,
    threshold: float = 0.60,
) -> tuple[Discoverability, float, str | None]:
    """KB-discoverable iff some single document covers `threshold` of the need's CONTENT terms.

    Per-DOCUMENT rather than corpus-wide on purpose: a need whose tokens are scattered across
    forty documents is not something a retrieval step can hand you, and treating it as
    discoverable would overstate what any inquirer could reach.

    CONTENT TERMS, NOT RAW TOKENS, and the difference is the whole verdict. The only real
    producer of candidates hands this the raw interrogative ASK, and `tokens()` did no
    filtering -- so interrogative scaffolding and politeness counted against coverage.
    Measured on one document about Houston Baptist University:

        "Houston Baptist University founding year"              0.600  -> kb
        "When was Houston Baptist University founded?"          0.667  -> kb
        "Could you please tell me in what year ... founded?"     0.357  -> USER_PRIVATE

    The same need, three ways of asking, two different verdicts. `private_share` is the hard
    ceiling on what any autonomous inquirer could reach and CONTRIBUTING.md calls it the single most
    important number for the framing -- so it was partly a property of how politely the mining
    policy happened to write, and a verbose policy manufactures a higher ceiling. With content
    terms all three phrasings land on `kb`.
    """
    need = content_terms(node_text)
    if not need:
        return "unknown", 0.0, None
    best, best_id = 0.0, None
    for doc in documents:
        cov = len(need & content_terms(doc.get("content", ""))) / len(need)
        if cov > best:
            best, best_id = cov, doc.get("id")
    return ("kb" if best >= threshold else "user_private"), best, best_id


def partition(
    nodes: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, str]],
    *,
    threshold: float = 0.60,
) -> list[PartitionVerdict]:
    out: list[PartitionVerdict] = []
    for n in nodes:
        part, counts = partition_from_ablation(str(n.get("ablation_verdict", "UNTESTABLE")))
        disc, cov, doc_id = discoverability_of(
            str(n.get("text", "")), documents, threshold=threshold
        )
        out.append(PartitionVerdict(str(n["node_id"]), part, disc, cov, doc_id, counts))
    return out


def partition_elasticity(
    nodes: Sequence[Mapping[str, object]],
    documents: Sequence[Mapping[str, str]],
    thresholds: Sequence[float] = (0.40, 0.50, 0.60, 0.70, 0.80),
) -> dict[float, float]:
    """user_private share as a function of the threshold.

    Shipped with every ceiling number. If the share moves violently across this curve, the
    ceiling is a property of the instrument rather than of the task, and the paper has to say
    so instead of quoting a scalar.
    """
    out: dict[float, float] = {}
    for t in thresholds:
        v = partition(nodes, documents, threshold=t)
        priv = sum(1 for x in v if x.discoverability == "user_private")
        out[t] = priv / len(v) if v else float("nan")
    return out
