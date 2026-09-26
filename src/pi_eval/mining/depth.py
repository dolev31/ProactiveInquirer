"""S6a — the depth-0 frontier.

Depth is measured FROM information explicitly stated in x, so the seed set is not a detail:
move one node in or out and every depth downstream of it shifts by one. Two signals decide
membership, and both are recorded so the choice is auditable:

  * ENTITY COVERAGE (mechanical, preferred). The need's text is entity-covered by the
    question — every capitalised span / quoted token / identifier in the need already appears
    in x. No model involved, so no annotator prior leaks in.
  * EARLY-RESOLVE (fallback, flagged). Across the trace pool the need is resolved at turn 0
    in >= `early_rate` of traces AND its retrieval query closely matches x. This is weaker
    evidence — it can mark a node as a seed merely because it is easy — so every node
    admitted this way carries `basis="early_resolve"` and the share is reported.

Anything not admitted by either signal is NOT a seed. It may still be unreachable, in which
case it has no depth at all and counts toward orphan_rate rather than being given a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

_TOKEN = re.compile(r"\b[A-Z][\w'-]*\b|\b\d[\d,.]*\b|\"[^\"]+\"")


def salient_tokens(text: str) -> frozenset[str]:
    return frozenset(t.strip('"').lower() for t in _TOKEN.findall(text or ""))


@dataclass(frozen=True, slots=True)
class SeedVerdict:
    node_id: str
    is_seed: bool
    basis: str
    coverage: float
    early_rate: float


def seed_set(
    question: str,
    node_texts: Mapping[str, str],
    *,
    early_resolved: Mapping[str, float] | None = None,
    coverage_min: float = 0.99,
    early_rate: float = 0.8,
) -> list[SeedVerdict]:
    q = salient_tokens(question)
    early = early_resolved or {}
    out: list[SeedVerdict] = []
    for node_id, text in node_texts.items():
        toks = salient_tokens(text)
        cov = (len(toks & q) / len(toks)) if toks else 0.0
        rate = early.get(node_id, 0.0)
        if toks and cov >= coverage_min:
            out.append(SeedVerdict(node_id, True, "entity_coverage", cov, rate))
        elif rate >= early_rate:
            out.append(SeedVerdict(node_id, True, "early_resolve", cov, rate))
        else:
            out.append(SeedVerdict(node_id, False, "none", cov, rate))
    return out


def seed_basis_shares(verdicts: Sequence[SeedVerdict]) -> dict[str, float]:
    """Reported alongside every depth table: a seed set dominated by the weaker signal makes
    every depth claim weaker, and hiding that would be the easiest way to overstate depth."""
    seeds = [v for v in verdicts if v.is_seed]
    if not seeds:
        return {"entity_coverage": 0.0, "early_resolve": 0.0, "n_seeds": 0}
    n = len(seeds)
    return {
        "entity_coverage": sum(1 for v in seeds if v.basis == "entity_coverage") / n,
        "early_resolve": sum(1 for v in seeds if v.basis == "early_resolve") / n,
        "n_seeds": n,
    }
