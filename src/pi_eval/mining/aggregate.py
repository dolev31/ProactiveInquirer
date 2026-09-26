"""S3 — cross-trace aggregation, and the frequency/necessity distinction.

This is the stage where the annotation methodology either holds or quietly fails. Frequency
across traces produces a CANDIDATE and nothing more. Only the ablation in ablate.py — drop
the evidence, re-run, measure — produces a causal claim, and only that promotes a candidate
to `required`. Conflating the two is the single most common way a mined dataset becomes a
record of what some policy happened to do.

Two contamination controls live here:
  * promotion requires >= 2 model families AND >= 2 policy forms, so one generator's habit
    cannot mint a need;
  * lift against a matched FAILED-trace pool, because a need that appears just as often in
    failures is not diagnostic of anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .canon import NeedCluster
from .pool import TracePool


@dataclass(frozen=True, slots=True)
class MinedNode:
    node_id: str
    text: str
    ev_uids: tuple[str, ...]
    support: int
    cell_support: int
    model_families: tuple[str, ...]
    policy_forms: tuple[str, ...]
    generator_entropy: float
    fail_support: int
    lift: float
    lift_p: float
    promoted: bool
    reason: str


def generator_entropy(cluster: NeedCluster, all_cells: Sequence[str]) -> float:
    """Shannon entropy of the cluster's support across factorial cells, normalized to [0,1].

    A need supported by every cell equally scores 1.0; one supported by a single cell scores
    0.0 and is a prime suspect for policy contamination. The bottom decile is dropped.
    """
    if not all_cells:
        return 0.0
    counts: dict[str, int] = {}
    for m in cluster.members:
        counts[m.cell_id] = counts.get(m.cell_id, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log(c / total) for c in counts.values() if c)
    max_ent = math.log(len(all_cells)) if len(all_cells) > 1 else 1.0
    return ent / max_ent if max_ent else 0.0


def haldane_lift(a: int, n_a: int, b: int, n_b: int) -> tuple[float, float]:
    """Lift of success-rate over failure-rate with a Haldane-Anscombe 0.5 correction.

    The correction matters because a need appearing in 6/6 successes and 0/6 failures gives
    an infinite raw lift, and infinities do not survive contact with a bootstrap.
    """
    pa = (a + 0.5) / (n_a + 1.0)
    pb = (b + 0.5) / (n_b + 1.0)
    lift = pa / pb if pb > 0 else float("inf")
    # Two-proportion z-test on the log-odds; a screening statistic, never a reported p-value.
    se = math.sqrt(1 / (a + 0.5) + 1 / (n_a - a + 0.5) + 1 / (b + 0.5) + 1 / (n_b - b + 0.5))
    z = math.log((pa / (1 - pa)) / (pb / (1 - pb))) / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return lift, p


def aggregate(
    clusters: Sequence[NeedCluster],
    pool: TracePool,
    *,
    suite: str,
    task: str,
    theta: float,
    nli_pin: str,
    fail_clusters: Sequence[NeedCluster] = (),
    min_support: int = 2,
    min_cell_support: int = 2,
    min_families: int = 2,
    entropy_floor_quantile: float = 0.10,
) -> list[MinedNode]:
    all_cells = sorted(pool.cells)
    fail_by_text = {c.medoid.lower(): len(c.trace_ids) for c in fail_clusters}

    ents = sorted(generator_entropy(c, all_cells) for c in clusters)
    floor = ents[int(len(ents) * entropy_floor_quantile)] if ents else 0.0

    out: list[MinedNode] = []
    for cl in clusters:
        support = len(cl.trace_ids)
        cell_support = len(cl.cell_ids)
        fams = tuple(sorted(cl.families))
        forms = tuple(sorted(cl.policy_forms))
        ent = generator_entropy(cl, all_cells)
        fail_support = fail_by_text.get(cl.medoid.lower(), 0)
        lift, lift_p = haldane_lift(support, pool.n_success, fail_support, max(pool.n_failure, 1))

        reasons = []
        if support < min_support:
            reasons.append(f"support {support} < {min_support}")
        if cell_support < min_cell_support:
            reasons.append(f"cell_support {cell_support} < {min_cell_support}")
        if len(fams) < min_families:
            reasons.append(f"families {len(fams)} < {min_families}")
        if ent < floor and len(clusters) > 10:
            reasons.append(
                f"generator_entropy {ent:.3f} below the {entropy_floor_quantile:.0%} floor"
            )

        out.append(
            MinedNode(
                node_id=cl.node_id(suite, task, theta, nli_pin),
                text=cl.medoid,
                ev_uids=cl.ev_uids,
                support=support,
                cell_support=cell_support,
                model_families=fams,
                policy_forms=forms,
                generator_entropy=ent,
                fail_support=fail_support,
                lift=lift,
                lift_p=lift_p,
                promoted=not reasons,
                reason="; ".join(reasons) or "promoted to candidate",
            )
        )
    return out
