"""Need-discovery metrics. Every one of these reads matcher output, never question text."""

from __future__ import annotations

from typing import Sequence

from pi_eval.matcher.base import MatchRecord

LADDER = ("ask", "resolve", "use")
_RANK = {"none": 0, "ask": 1, "resolve": 2, "use": 3}


def rnr(records: Sequence[MatchRecord], node_ids: Sequence[str], level: str) -> float:
    """Required-Need Recall at one rung of the ladder.

    Reported as the full ladder, never as a single number: a policy that asks about a need,
    retrieves nothing and moves on scores 1.0 on ASK and 0.0 on USE, and only the second
    tells you anything about the answer.
    """
    if not node_ids:
        return float("nan")
    want = _RANK[level]
    by_id = {r.node_id: r for r in records}
    hit = sum(1 for n in node_ids if n in by_id and by_id[n].rank >= want)
    return hit / len(node_ids)


def rnr_ladder(records: Sequence[MatchRecord], node_ids: Sequence[str]) -> dict[str, float]:
    out = {lvl: rnr(records, node_ids, lvl) for lvl in LADDER}
    # The invariant, compiled in. If this ever fires, the matcher is inconsistent and every
    # coverage number in the paper is suspect.
    assert out["ask"] >= out["resolve"] >= out["use"] - 1e-12, out
    return out


def evidence_coverage(retrieved_uids: set[str], gold_uids: set[str]) -> float:
    """EC = |gold spans retrieved| / |gold spans|.

    Granularity-invariant: unlike node-count recall it does not move when an annotator splits
    one need into three, which is why it is the primary QUANTITATIVE coverage claim.
    """
    if not gold_uids:
        return float("nan")
    return len(retrieved_uids & gold_uids) / len(gold_uids)
