"""S2 — canonicalization.

Two propositions are THE SAME NEED iff they mutually entail one another under a pinned NLI
model at threshold theta. Everything about this is a judgement call, so all of it is
parameterised and reported: theta is swept over {0.60..0.95} and the sign of the primary
contrast must be stable across [0.75, 0.95] or no claim is made.

The entailment function is INJECTED rather than imported, so the clustering logic is unit
testable with an exact oracle and no torch in the test environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from pinq.ids import h

# (premise, hypothesis) -> P(entailment)
EntailFn = Callable[[str, str], float]

_NUM = re.compile(r"\d+(?:\.\d+)?")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def numerals(text: str) -> set[str]:
    return set(_NUM.findall(text))


@dataclass(frozen=True, slots=True)
class Candidate:
    text: str
    trace_id: str
    cell_id: str
    model_family: str
    policy_form: str
    ev_uids: tuple[str, ...]
    turn_idx: int
    provenance: str  # "mechanical" | "llm_elicited"


@dataclass
class NeedCluster:
    medoid: str
    members: list[Candidate] = field(default_factory=list)

    @property
    def ev_uids(self) -> tuple[str, ...]:
        return tuple(sorted({u for m in self.members for u in m.ev_uids}))

    @property
    def trace_ids(self) -> set[str]:
        return {m.trace_id for m in self.members}

    @property
    def cell_ids(self) -> set[str]:
        return {m.cell_id for m in self.members}

    @property
    def families(self) -> set[str]:
        return {m.model_family for m in self.members}

    @property
    def policy_forms(self) -> set[str]:
        return {m.policy_form for m in self.members}

    def node_id(self, suite: str, task: str, theta: float, nli_pin: str) -> str:
        return h("node", suite, task, h("txt", normalize(self.medoid)), f"{theta:.2f}", nli_pin)


def mutually_entails(a: str, b: str, entail: EntailFn, theta: float) -> bool:
    """Symmetric containment, with a hard numeral guard.

    Without the guard, "the fee is 3%" and "the fee is 30%" cluster together under most
    entailment models, which silently merges two different needs and inflates every recall
    number that reads the cluster.
    """
    if numerals(a) != numerals(b):
        return False
    return entail(a, b) >= theta and entail(b, a) >= theta


def cluster(
    candidates: Sequence[Candidate], entail: EntailFn, theta: float = 0.85
) -> list[NeedCluster]:
    """Greedy agglomeration against cluster medoids.

    Greedy rather than transitive-closure on purpose: transitivity would let a chain of
    near-misses merge two genuinely different needs (a "diameter" blow-up). Every member
    must mutually entail the MEDOID, which bounds the cluster diameter at 2.
    """
    clusters: list[NeedCluster] = []
    # Longest first: a specific proposition makes a better medoid than a vague one.
    for cand in sorted(candidates, key=lambda c: (-len(c.text), c.text)):
        placed = False
        for cl in clusters:
            if mutually_entails(cand.text, cl.medoid, entail, theta):
                cl.members.append(cand)
                placed = True
                break
        if not placed:
            clusters.append(NeedCluster(medoid=cand.text, members=[cand]))
    return clusters


def exact_entail(a: str, b: str) -> float:
    """A deterministic oracle for tests: entailment iff normalized strings match.

    Keeping a trivial EntailFn in the library (rather than only in tests) means the whole
    mining pipeline can be exercised end to end with zero model downloads.
    """
    return 1.0 if normalize(a) == normalize(b) else 0.0


def theta_sweep(
    candidates: Sequence[Candidate], entail: EntailFn, thetas: Sequence[float]
) -> dict[float, int]:
    """Granularity elasticity, made visible: node count as a function of theta.

    Node count drives RNR, PP and every coverage number, so a metric that moves sharply
    across this sweep is a metric that is really measuring the annotator's granularity.
    """
    return {t: len(cluster(candidates, entail, t)) for t in thetas}
