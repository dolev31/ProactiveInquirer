"""Join two checkpoints' per-pair `score_checkpoint_paired.py` output by `pair_id` and run a
genuine PAIRED contrast -- McNemar's exact test on discordant pairs, plus a paired bootstrap CI
on the accuracy difference.

WHY NOT THE PAPER'S OWN METHOD. `appendix_instruments.tex` compares two checkpoints' accuracies
with a Newcombe interval for the difference of two INDEPENDENT proportions. Both checkpoints here
are scored on the IDENTICAL 970 pairs (or the identical matched-rung subset) -- a paired design,
not two independent samples. Treating correlated draws as independent is the same mistake this
project corrected on seed replicates: the standard error of an independent-samples test is too
large when the two arms share almost all of their variance, so it can only make a real
difference look LESS significant than it is, never more -- conservative in one direction, and
worth naming as a limitation of the number it replaces rather than silently matching it.

McNemar keys on DISCORDANT pairs only (one checkpoint right, the other wrong) -- pairs where both
are right or both are wrong carry no information about which checkpoint orders better, exactly
as a matched pair with identical outcomes contributes nothing to a matched-cost contrast
elsewhere in this project (see the paired-cost family of checks). `n_a_only` + `n_b_only` is the
effective sample size of this test, and it is reported alongside the raw n so a small discordant
count is visible, not hidden inside a p-value.
"""

from __future__ import annotations

import random
from typing import Any

from scipy.stats import binomtest


def join_by_pair_id(
    per_pair_a: dict[str, dict[str, Any]], per_pair_b: dict[str, dict[str, Any]]
) -> dict[str, tuple[int, int]]:
    """`{pair_id: (ok_a, ok_b)}`. Refuses (raises) on any pair_id present in one side only --
    the two checkpoints must have been scored on the same rung file, so a mismatch here means a
    stale or mismatched pair of `*.paired.json` files, not a result to average over.
    """
    ids_a, ids_b = set(per_pair_a), set(per_pair_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)[:5]
        only_b = sorted(ids_b - ids_a)[:5]
        raise ValueError(
            f"pair_id sets differ: {len(ids_a - ids_b)} only in A (e.g. {only_a}), "
            f"{len(ids_b - ids_a)} only in B (e.g. {only_b}) -- refusing to join mismatched rungs"
        )
    return {pid: (per_pair_a[pid]["ok"], per_pair_b[pid]["ok"]) for pid in per_pair_a}


def mcnemar_contrast(joined: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Exact McNemar test: is checkpoint B favoured over checkpoint A on the discordant pairs?

    `diff = acc_b - acc_a`, matching the paper's own sign convention (preference minus
    imitation, positive favours the preference arm).
    """
    n = len(joined)
    if n == 0:
        raise ValueError("no joined pairs -- empty rung")
    both_correct = sum(1 for a, b in joined.values() if a == 1 and b == 1)
    both_wrong = sum(1 for a, b in joined.values() if a == 0 and b == 0)
    a_only = sum(1 for a, b in joined.values() if a == 1 and b == 0)
    b_only = sum(1 for a, b in joined.values() if a == 0 and b == 1)
    assert both_correct + both_wrong + a_only + b_only == n
    n_discordant = a_only + b_only
    acc_a = sum(a for a, _ in joined.values()) / n
    acc_b = sum(b for _, b in joined.values()) / n
    pvalue = (
        binomtest(b_only, n_discordant, 0.5, alternative="two-sided").pvalue
        if n_discordant > 0
        else None
    )
    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff_b_minus_a": acc_b - acc_a,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "n_discordant": n_discordant,
        "mcnemar_exact_p_two_sided": pvalue,
        "note": (
            "no discordant pairs: A and B agree on every pair, diff is exactly 0 by "
            "construction, not a null result of a test with no power"
            if n_discordant == 0
            else None
        ),
    }


def paired_bootstrap_ci(
    joined: dict[str, tuple[int, int]],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI on `acc_b - acc_a`, resampling PAIRS (not the two arms
    separately) -- this is what keeps the resample paired: each bootstrap draw picks a pair_id
    and takes BOTH checkpoints' outcomes on it together, preserving their correlation.
    """
    pairs = list(joined.values())
    n = len(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        acc_a = sum(a for a, _ in sample) / n
        acc_b = sum(b for _, b in sample) / n
        diffs.append(acc_b - acc_a)
    diffs.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci_lo": diffs[lo_idx],
        "ci_hi": diffs[hi_idx],
        "point_diff_b_minus_a": (sum(b for _, b in pairs) - sum(a for a, _ in pairs)) / n,
    }
