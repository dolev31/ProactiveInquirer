"""Lane L1.1: is the diversity-gate flip on the test split an instrument artifact?

`criteria["distinct3"]` (`pinq_train.gate`, reading `within_task_distinct_n` from
`pinq_train.rung1_sft.collapse`) groups ASK questions by `task_id` ONLY, pooling every seed of
a task into one group before scoring trigram diversity. A checkpoint evaluated at N seeds that
asks the SAME state-dependent questions under every seed is therefore scored as if it had
collapsed by a factor close to N: duplicating a fixed set of k distinct questions once (a
second seed repeating the first verbatim) doubles the trigram total while adding no new
trigram, which halves distinct-n by construction. This module recomputes the statistic four
ways so the pooling artifact can be told apart from a real collapse:

    (a) pooled by task_id, every seed together -- the gate's own computation, reused verbatim
        via `pinq_train.rung1_sft.collapse.within_task_distinct_n` so variant (a) is bit-
        identical to `criteria["distinct3"]["value"]` by construction, not by re-derivation.
    (b) grouped by (task_id, seed) -- a second seed repeating the first can no longer move it.
    (c) pooled by task_id, restricted to seed 0 only -- what the gate would have read had only
        one seed ever run, which is what it reads on development today.
    (d) variant (a)'s pooled-by-task computation done separately within each seed present,
        then the resulting per-seed scalars averaged with equal weight per seed.

It also classifies cross-seed determinism: for a fixed (task_id, turn_idx), whether the
question text is byte-identical across the seeds that reached it, or "near-identical" under
this repo's calibrated paraphrase guard (content-word Jaccard >= 0.5 with >= 4 shared content
tokens -- see `pinq_train.export.distinctness.content_tokens`/`jaccard`, which own the
tokenizer and stopword list this reuses; the memory note that names
`pinq_train.export.dataset.is_paraphrase` as the calibration's home is stale, that function does
not exist in this tree -- verified by a repository-wide grep on 2026-09-18).
"""

from __future__ import annotations

import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pinq_train.export.distinctness import content_tokens, jaccard  # noqa: E402
from pinq_train.rung1_sft.collapse import (  # noqa: E402
    DEFAULT_N,
    MIN_PER_TASK,
    distinct_n,
    within_task_distinct_n,
)

# The repo's calibrated near-identical rule (see module docstring): a ratio alone over-fires on
# short questions, so a minimum shared-token count is required alongside the Jaccard floor.
NEAR_IDENTICAL_JACCARD = 0.5
NEAR_IDENTICAL_MIN_SHARED = 4

AskRow = Mapping[str, Any]  # {"task_id": str, "seed": int, "turn_idx": int, "question": str}


# --------------------------------------------------------------------------- the four variants


def variant_a_pooled_by_task(rows: Sequence[AskRow]) -> tuple[float, int]:
    """(a) The gate's own computation: pooled by task_id, every seed together.

    Returns `(value, n_groups_scored)` -- `n_groups_scored` is the number of tasks that met
    `MIN_PER_TASK` and therefore contributed to the mean, which can be less than the number of
    distinct task ids in `rows` (a task with a single question contributes nothing).
    """
    pairs = [(r["task_id"], r["question"]) for r in rows]
    by_task: dict[str, list[str]] = defaultdict(list)
    for task_id, text in pairs:
        if text:
            by_task[str(task_id)].append(text)
    n_scored = sum(1 for v in by_task.values() if len(v) >= MIN_PER_TASK)
    return within_task_distinct_n(pairs), n_scored


def variant_b_by_task_seed(rows: Sequence[AskRow]) -> tuple[float, int]:
    """(b) Grouped by (task_id, seed): a second seed repeating the first can no longer halve it."""
    by_group: dict[tuple[str, Any], list[str]] = defaultdict(list)
    for r in rows:
        if r["question"]:
            by_group[(str(r["task_id"]), r["seed"])].append(r["question"])
    scored = [distinct_n(v, DEFAULT_N) for v in by_group.values() if len(v) >= MIN_PER_TASK]
    value = sum(scored) / len(scored) if scored else float("nan")
    return value, len(scored)


def variant_c_seed0_only(rows: Sequence[AskRow]) -> tuple[float, int]:
    """(c) Pooled by task_id exactly as the gate does, restricted to seed-0 runs only."""
    seed0 = [r for r in rows if r["seed"] == 0]
    return variant_a_pooled_by_task(seed0)


def variant_d_per_seed_averaged(rows: Sequence[AskRow]) -> tuple[float, dict[Any, float]]:
    """(d) Variant (a) computed separately within each seed present, then the per-seed scalars
    averaged with equal weight per seed. Returns `(value, {seed: per_seed_value})`."""
    seeds = sorted({r["seed"] for r in rows}, key=lambda s: (s is None, s))
    per_seed: dict[Any, float] = {}
    for s in seeds:
        sub = [r for r in rows if r["seed"] == s]
        per_seed[s], _ = variant_a_pooled_by_task(sub)
    vals = [v for v in per_seed.values() if v == v]  # drop NaN
    value = sum(vals) / len(vals) if vals else float("nan")
    return value, per_seed


def all_variants(rows: Sequence[AskRow]) -> dict[str, Any]:
    """All four variants plus the population sizes the brief asks every cell to carry."""
    a_value, a_n = variant_a_pooled_by_task(rows)
    b_value, b_n = variant_b_by_task_seed(rows)
    c_value, c_n = variant_c_seed0_only(rows)
    d_value, d_per_seed = variant_d_per_seed_averaged(rows)
    n_tasks = len({r["task_id"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    return {
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
        "seeds": sorted({r["seed"] for r in rows}, key=lambda s: (s is None, s)),
        "a_pooled_by_task": {"value": a_value, "n_task_groups_scored": a_n},
        "b_by_task_seed": {"value": b_value, "n_task_seed_groups_scored": b_n},
        "c_seed0_only": {"value": c_value, "n_task_groups_scored": c_n},
        "d_per_seed_averaged": {"value": d_value, "per_seed": d_per_seed},
    }


# --------------------------------------------------------------------------- cross-seed determinism


def cross_seed_groups(rows: Sequence[AskRow]) -> dict[tuple[str, int], dict[Any, str]]:
    """`{(task_id, turn_idx): {seed: question}}`, restricted to groups where >= 2 distinct
    seeds reached that turn index -- the only groups a cross-seed comparison can be made on."""
    by_group: dict[tuple[str, int], dict[Any, str]] = defaultdict(dict)
    for r in rows:
        by_group[(str(r["task_id"]), int(r["turn_idx"]))][r["seed"]] = r["question"]
    return {k: v for k, v in by_group.items() if len(v) >= 2}


def _near_identical(a: str, b: str) -> bool:
    ta, tb = content_tokens(a), content_tokens(b)
    shared = len(ta & tb)
    return jaccard(ta, tb) >= NEAR_IDENTICAL_JACCARD and shared >= NEAR_IDENTICAL_MIN_SHARED


def classify_group(seed_to_text: Mapping[Any, str]) -> str:
    """ "identical" if every pairwise comparison among the group's seeds is a byte-exact match,
    "near_identical" if not identical but every pair clears the calibrated paraphrase guard,
    else "different". A byte-identical pair always also clears the guard (Jaccard 1.0, shared
    = all content tokens), so `identical` is a subset of what `near_identical`-or-better means;
    the two shares reported by `cross_seed_determinism` are therefore nested, not disjoint."""
    texts = list(seed_to_text.values())
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return "different"
    if all(a == b for a, b in pairs):
        return "identical"
    if all(_near_identical(a, b) for a, b in pairs):
        return "near_identical"
    return "different"


def cross_seed_determinism(rows: Sequence[AskRow]) -> dict[str, Any]:
    """Share of (task, turn_idx) positions that are byte-identical, and share that are at least
    near-identical (>= 0.5 Jaccard, >= 4 shared content tokens), across the seeds that reached
    that position. `near_identical_or_better` INCLUDES the identical share (see `classify_group`).
    """
    groups = cross_seed_groups(rows)
    n = len(groups)
    if n == 0:
        return {
            "n_task_turn_positions_with_ge2_seeds": 0,
            "share_identical": float("nan"),
            "share_near_identical_or_better": float("nan"),
        }
    labels = [classify_group(g) for g in groups.values()]
    n_identical = sum(1 for lbl in labels if lbl == "identical")
    n_near = sum(1 for lbl in labels if lbl == "near_identical")
    return {
        "n_task_turn_positions_with_ge2_seeds": n,
        "n_identical": n_identical,
        "n_near_identical_only": n_near,
        "n_different": n - n_identical - n_near,
        "share_identical": n_identical / n,
        "share_near_identical_or_better": (n_identical + n_near) / n,
    }


# --------------------------------------------------------------------------- sampling temperature


def resolve_temperature(
    sampling_sha: str, *, candidates: Sequence[float] = tuple()
) -> float | None:
    """The temperature that reproduces `pins.<role>.sampling_sha` in a run manifest.

    `sampling_sha = pinq.ids.h("sampling", f"t={temperature}")`
    (`pinq_adapters.llm.litellm_client.LiteLLMClient._telemetry`) hashes nothing else, so the
    hash is invertible by brute force over a small candidate set -- there is no registry that
    maps a sampling_sha back to its params. Returns None if no candidate matches.
    """
    from pinq.ids import h

    cands = candidates or (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.5,
        2.0,
    )
    for t in cands:
        if h("sampling", f"t={t}") == sampling_sha:
            return t
    return None
