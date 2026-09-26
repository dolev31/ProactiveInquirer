"""Per-question value.

The originally proposed Question Utility was a PREFIX MARGINAL, Q(A_{t+1}) - Q(A_t): the
value of q_t given the one prefix this policy happened to produce, i.e. a single permutation
in Shapley terms. With complements it reports ~0 for both halves of a pair that are jointly
decisive; with substitutes it hands all credit to whichever came first. So the prefix
marginal is legal as a TRAINING reward (within-group, where the bias is shared) and illegal
as a REPORTED metric.

What is reported is leave-one-out:

    phi_LOO(q) = Q(A(E_T)) - Q(A(E_T \\ ev(q)))

with ev(q) taken from Turn.retrieved_uids -- an objective record of what the retriever
returned -- and never from Ask.parent_uids, which the policy under test reports about itself.

phi requires Drafter.draft() to be pure in (view, evidence.subset_hash, seed). That is an
architectural constraint, not a metric detail.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from pinq.budget import BudgetLedger
from pinq.types import Evidence, TaskView, Trajectory

# Q is any pure map from an evidence subset to a scalar score.
QFn = Callable[[TaskView, Evidence], float]


@dataclass(frozen=True, slots=True)
class Phi:
    turn_idx: int
    phi: float
    n_units: int
    n_new_units: int
    qid: str


def make_q(drafter, answerer, gold: str, *, seed: int = 0, score=None) -> QFn:
    """Build a memoized Q over evidence subsets.

    Memoization is keyed by EVIDENCE SUBSET HASH, never by prefix index k. Keying by k would
    make every leave-one-out re-draft a guaranteed cache miss, which is the difference
    between phi costing T lookups and T fresh generations.
    """
    from pi_eval.metrics.quality import best_over_aliases

    score = score or (lambda text: best_over_aliases(text, gold))
    cache: dict[str, float] = {}

    def q(view: TaskView, ev: Evidence) -> float:
        key = ev.subset_hash
        if key not in cache:
            led = BudgetLedger(cap=10**9)
            draft = drafter.draft(view, ev, seed=seed, ledger=led)
            ans = answerer.answer(view, ev, draft, seed=seed, ledger=led)
            cache[key] = score(ans.text)
        return cache[key]

    return q


def phi_loo(traj: Trajectory, q: QFn) -> tuple[float, list[Phi]]:
    base = q(traj.view, traj.evidence)
    out: list[Phi] = []
    for t in traj.turns:
        uids = frozenset(t.retrieved_uids)
        if not uids:
            continue
        lo = q(traj.view, traj.evidence.without(uids))
        out.append(
            Phi(
                turn_idx=t.turn_idx,
                phi=base - lo,
                n_units=len(uids),
                n_new_units=len(t.new_uids),
                qid=getattr(t.action, "qid", ""),
            )
        )
    return base, out


def phi_prefix_marginal(traj: Trajectory, q: QFn) -> list[Phi]:
    """The biased sequential estimator, computed only so the paper can PUBLISH
    rho(phi_hat, phi_LOO) rather than assert that the two agree."""
    out: list[Phi] = []
    prev = q(traj.view, traj.prefix(0).evidence)
    for k, t in enumerate(traj.turns, start=1):
        cur = q(traj.view, traj.prefix(k).evidence)
        out.append(
            Phi(
                t.turn_idx,
                cur - prev,
                len(t.retrieved_uids),
                len(t.new_uids),
                getattr(t.action, "qid", ""),
            )
        )
        prev = cur
    return out


def random_question_null(
    traj: Trajectory,
    q: QFn,
    retriever,
    pool: Sequence[str],
    *,
    seed: int = 0,
    n: int = 8,
    k: int = 5,
) -> list[float]:
    """The null phi has no meaningful zero without.

    Matched VOLUME, not matched content: draw questions from other tasks, retrieve the same
    number of units through the same retriever, and measure the same leave-one-out drop.
    This controls for 'any k units help a bit'.
    """
    rng = random.Random(seed)
    n_units = max(1, len(traj.evidence) // max(1, traj.n_asks or 1))
    base = q(traj.view, traj.evidence)
    out: list[float] = []
    for _ in range(n):
        text = rng.choice(list(pool)) if pool else ""
        units = list(retriever.search(text, k))[:n_units]
        uids = frozenset(u.uid for u in units) & traj.evidence.uids
        out.append(base - q(traj.view, traj.evidence.without(uids)))
    return out


def stopping_error(traj: Trajectory, q: QFn) -> dict[str, float]:
    """Signed stopping error in QUESTION UNITS, over/under reported separately.

    Deliberately not 'stop regret in quality units': argmax_k over K+1 noisy prefixes is
    upward-biased by E[max] of the noise (~sigma*sqrt(2 ln K)), so a PERFECT stopper would
    exhibit regret. Counting questions is far more noise-robust.
    """
    scores = [q(traj.view, traj.prefix(k).evidence) for k in range(len(traj.turns) + 1)]
    k_star = max(range(len(scores)), key=lambda i: (scores[i], -i))
    k_hat = len(traj.turns)
    return {
        "k_hat": float(k_hat),
        "k_star": float(k_star),
        "overshoot": float(max(0, k_hat - k_star)),
        "undershoot": float(max(0, k_star - k_hat)),
        "q_at_k_hat": scores[k_hat],
        "q_at_k_star": scores[k_star],
    }
