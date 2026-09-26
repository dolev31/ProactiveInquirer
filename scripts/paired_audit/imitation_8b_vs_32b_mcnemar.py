"""Recompute imitation-8B vs imitation-32B on the 970 ask/ask pairs with the paired test.

`paper/appendix_instruments.tex` and `paper/results.tex` report this contrast with a Newcombe
interval for the difference of two INDEPENDENT proportions (delta +0.0052, CI
[-0.0393, +0.0495], p=0.820, Holm-corrected 0.820). Both checkpoints are scored on the SAME 970
pairs from `artifacts/length_matched_stratum_20260919/pairs/ask_ask_all.jsonl` -- the data are
paired, not independent, so Newcombe's method is the wrong instrument (see
paired-vs-independent audit, 2026-09-20). This script is read-only against that artifact and
against `src/pi_eval/stats/inference.py`; it writes nothing outside
`artifacts/paired_vs_independent_audit_20260920/`.

Import path follows the precedent in `scripts/power_analysis.py`: `scripts/` is not a
root_package in the import-linter config, so importing `pi_eval.stats.inference` from here does
not touch the gold firewall contracts (nothing under `pinq`/`pinq_adapters`/`pinq_expt`/
`pinq_train` does this import; only this audit script does).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pi_eval.stats.inference import mcnemar_exact  # noqa: E402

STRATUM = Path(__file__).resolve().parents[2] / "artifacts" / "length_matched_stratum_20260919"
PAIRS_FILE = STRATUM / "pairs" / "ask_ask_all.jsonl"
IMIT_8B = STRATUM / "eval_paired" / "qwen3-8b-sft-headline.paired.json"
IMIT_32B = STRATUM / "eval_paired" / "qwen3-32b-sft-headline.paired.json"


def load_per_pair(path: Path) -> dict[str, int]:
    doc = json.loads(path.read_text())
    per_pair = doc["results"]["unmatched_ask_ask_all"]["paired"]["per_pair"]
    return {pair_id: int(rec["ok"]) for pair_id, rec in per_pair.items()}


def load_clusters(path: Path) -> dict[str, str]:
    clusters: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            pair_id = row["pair_id"]
            template_id = row.get("template_id")
            task_id = row["task_id"]
            clusters[pair_id] = template_id if template_id else task_id
    return clusters


def sign_test_on_discordant(a: dict[str, int], b: dict[str, int], keys: list[str]) -> dict:
    """Exact two-sided sign test over the discordant PAIRS -- ignores clustering.

    Reported for completeness (the thin-data refusal rule asks for it explicitly), but see
    `sign_test_on_discordant_clusters` for the estimand-consistent version: with 57 clusters
    covering 970 pairs unevenly, treating each discordant pair as an independent coin flip is
    the same independence assumption `mcnemar_exact` itself refuses once clustering is
    non-trivial.
    """
    import math

    b10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    b01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n_disc = b10 + b01
    if n_disc == 0:
        return {"n_discordant": 0, "b10": 0, "b01": 0, "sign_p": 1.0}
    k = min(b10, b01)
    p = min(1.0, 2 * sum(math.comb(n_disc, i) for i in range(k + 1)) / (2**n_disc))
    return {"n_discordant": n_disc, "b10": b10, "b01": b01, "sign_p": p}


def sign_test_on_discordant_clusters(
    a: dict[str, int], b: dict[str, int], keys: list[str], clusters: dict[str, str]
) -> dict:
    """Exact two-sided sign test over CLUSTERS whose net discordant direction is non-zero.

    Per-cluster net = sum(a[k]-b[k]) over the cluster's pairs. A cluster nets to zero either
    because it has no discordant pairs or because its discordant pairs cancel -- either way it
    is a TIED unit and carries no directional information, matching the worked-example
    convention (48/56 tied units, 8 informative).
    """
    import math

    groups: dict[str, float] = {}
    for k in keys:
        c = clusters.get(k, k)
        groups[c] = groups.get(c, 0.0) + (a[k] - b[k])
    n_clusters = len(groups)
    favor_a = sum(1 for v in groups.values() if v > 0)
    favor_b = sum(1 for v in groups.values() if v < 0)
    tied = n_clusters - favor_a - favor_b
    n_informative = favor_a + favor_b
    if n_informative == 0:
        return {
            "n_clusters": n_clusters,
            "n_tied_clusters": tied,
            "n_informative_clusters": 0,
            "favor_a": 0,
            "favor_b": 0,
            "sign_p": 1.0,
        }
    k = min(favor_a, favor_b)
    p = min(1.0, 2 * sum(math.comb(n_informative, i) for i in range(k + 1)) / (2**n_informative))
    return {
        "n_clusters": n_clusters,
        "n_tied_clusters": tied,
        "n_informative_clusters": n_informative,
        "favor_a": favor_a,
        "favor_b": favor_b,
        "sign_p": p,
    }


def main() -> None:
    label_8b = "qwen3-8b-sft-headline"
    label_32b = "qwen3-32b-sft-headline"

    a8b = load_per_pair(IMIT_8B)
    a32b = load_per_pair(IMIT_32B)
    clusters = load_clusters(PAIRS_FILE)

    keys = sorted(set(a8b) & set(a32b))
    n_ties = sum(1 for k in keys if a8b[k] == a32b[k])

    # Self-check fingerprint 1: does the CI move on a re-seed while the point stays identical?
    # Self-check fingerprint 2: does the CI fail to move AT ALL across re-seeds (fake stability)?
    reseed_results = []
    for seed in (0, 1, 2, 7, 12345):
        est = mcnemar_exact(a8b, a32b, clusters=clusters, n_perm=10_000, seed=seed)
        reseed_results.append(
            {
                "seed": seed,
                "point": est.point,
                "ci_lo": est.ci_lo,
                "ci_hi": est.ci_hi,
                "p_value": est.p_value,
                "method": est.method,
                "note": est.note,
            }
        )

    est0 = mcnemar_exact(a8b, a32b, clusters=clusters, n_perm=10_000, seed=0)

    # Self-check: same seed, SHUFFLED input dict order -- catches an unsorted-resample bug in
    # `mcnemar_exact`/`cluster_bootstrap` (point AND bounds should be bit-identical to est0,
    # since both functions sort their keys/units internally before resampling).
    import random as _random

    shuffled_keys = list(a8b.keys())
    _random.Random(99).shuffle(shuffled_keys)
    a8b_shuffled = {k: a8b[k] for k in shuffled_keys}
    a32b_shuffled = {k: a32b[k] for k in reversed(list(a32b.keys()))}
    est0_shuffled = mcnemar_exact(
        a8b_shuffled, a32b_shuffled, clusters=clusters, n_perm=10_000, seed=0
    )
    order_invariance_check = {
        "point_matches": est0.point == est0_shuffled.point,
        "ci_lo_matches": est0.ci_lo == est0_shuffled.ci_lo,
        "ci_hi_matches": est0.ci_hi == est0_shuffled.ci_hi,
        "p_value_matches": est0.p_value == est0_shuffled.p_value,
    }

    sign_pairs = sign_test_on_discordant(a8b, a32b, keys)
    sign_clusters = sign_test_on_discordant_clusters(a8b, a32b, keys, clusters)

    # Pair-weighted delta, for comparison against the published (unclustered) figure only --
    # NOT the estimand the corrected test uses; the cluster-level estimate is.
    pair_weighted_delta = (sum(a8b[k] for k in keys) - sum(a32b[k] for k in keys)) / len(keys)

    out = {
        "contrast": f"{label_8b} minus {label_32b}",
        "n_pairs": len(keys),
        "n_ties": n_ties,
        "n_clusters": len(set(clusters.get(k, k) for k in keys)),
        "pair_weighted_delta_unclustered": pair_weighted_delta,
        "mcnemar_exact_seed0": {
            "point": est0.point,
            "ci_lo": est0.ci_lo,
            "ci_hi": est0.ci_hi,
            "p_value": est0.p_value,
            "n": est0.n,
            "method": est0.method,
            "note": est0.note,
        },
        "reseed_check": reseed_results,
        "order_invariance_check": order_invariance_check,
        "sign_test_on_discordant_pairs_unclustered": sign_pairs,
        "sign_test_on_discordant_clusters": sign_clusters,
        "as_published": {
            "delta": 0.0052,
            "ci_lo": -0.0393,
            "ci_hi": 0.0495,
            "p_value": 0.820,
            "holm_p": 0.820,
            "method": "Newcombe (independent-samples difference of proportions)",
        },
    }
    out_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "paired_vs_independent_audit_20260920"
        / "imitation_8b_vs_32b_mcnemar.json"
    )
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
