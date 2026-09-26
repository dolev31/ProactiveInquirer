"""Turn 7 `score_checkpoint.py` output files into the ladder table, the chance test, and the
non-vacuity control that `artifacts/length_matched_stratum_20260919/RESULT.md` reports.

WHAT THIS DOES NOT DO. It does not re-score anything: every number here is read straight out of
the `pair_accuracy` block each `*.lenmatch.json` already wrote on the cluster (n, acc,
acc_by_len_sign, len_sign_gap). This script only aggregates and tests what is already measured.

THE UNIT-OF-ANALYSIS DECISION (task requirement 5). Two units are reported, never conflated:

  1. PER-CHECKPOINT, PER-RUNG item-level test: for one checkpoint's n ordering decisions on one
     rung, a two-sided exact binomial test against p=0.5 treats each pair as an independent trial
     -- the standard (if imperfect) assumption for "is this checkpoint's ordering better than a
     coin flip", same assumption the required reading's own n=25 power argument used.
  2. ACROSS CHECKPOINTS: the 7 scored checkpoints are NOT 7 independent replicates. The three
     E39-8b seeds share training data and initialization; they are correlated draws of ONE
     underlying training run, not three draws of the population of "trained checkpoints". The
     cross-checkpoint view here therefore reports two things side by side: all 7 points (labelled
     by family), and a second summary where the 3 E39-8b seeds are collapsed to ONE family unit
     (their mean accuracy) before any sign test across families -- giving n=5 independent family
     units, not 7 pairs.

REFUSAL RULE (task requirement 3). A rung is not refused by hiding its numbers -- every rung's n,
acc and p-value are printed. A rung is refused a CONCLUSION when its n lacks power: this script
reports, per rung, the exact two-sided binomial power (alpha=0.05) to detect a 0.60 vs 0.50
accuracy (a modest, previously-plausible effect size in this project), and RESULT.md must not
assert "above chance" or "at chance" for any (checkpoint, rung) cell whose power to detect 0.60 is
below 0.80 -- it may only report the point estimate and say the n cannot support a claim either
way, mirroring the required reading's own treatment of n=25 (P(X>=17|n=25,p=0.5)=0.0539).

Usage:
    python scripts/length_matched/analyze_ladder.py --eval-dir <dir with *.lenmatch.json> \
        [--out artifacts/length_matched_stratum_20260919/ladder_summary.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

RUNG_ORDER = [
    "unmatched_ask_ask_all",
    "tol0",
    "tol2",
    "tol5",
    "tol10",
    "tol20",
    "tol30",
]

# Established in artifacts/tierA_length_decomposition_20260919/RESULT.md (required reading):
# len_sign_gap is negative in 124/124 stored Tier-A verdicts, median -0.2187, over the UNMATCHED
# ask_ask population. The non-vacuity control checks the SAME sign and SAME rough magnitude
# recur here, on the SAME checkpoints' unmatched_ask_ask_all cell, under THIS script's code path.
REFERENCE_LEN_SIGN_GAP_MEDIAN = -0.2187


def load_verdicts(eval_dir: Path) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    files = sorted(eval_dir.glob("*.lenmatch.json"))
    if not files:
        raise SystemExit(f"no *.lenmatch.json files found under {eval_dir}")
    for f in files:
        data = json.loads(f.read_text())
        label = data["label"]
        verdicts[label] = data
    return verdicts


def family_of(label: str) -> str:
    # The 3 E39-8b seeds are one family (correlated draws of one training run); every other
    # checkpoint is its own family. Matched by prefix, not by a guessed field.
    if label.startswith("E39-8b."):
        return "E39-8b"
    return label


def exact_binom_power(n: int, p0: float, p1: float, alpha: float = 0.05) -> float:
    """Exact two-sided binomial power to detect p1 when the true rate is p1, testing against p0.

    Finds the rejection region of the exact two-sided binomtest at `p0` (the set of k for which
    the two-sided p-value < alpha), then sums P(X=k | n, p1) over that region. No normal
    approximation -- these n are small enough (25-970) that scipy's exact test is cheap.
    """
    reject_ks = [k for k in range(n + 1) if binomtest(k, n, p0).pvalue < alpha]
    if not reject_ks:
        return 0.0
    from scipy.stats import binom

    return float(sum(binom.pmf(k, n, p1) for k in reject_ks))


def analyze(verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"rungs": {}, "non_vacuity": {}, "checkpoints": sorted(verdicts)}

    for rung in RUNG_ORDER:
        rung_out: dict[str, Any] = {"per_checkpoint": {}}
        ns_seen = set()
        for label, data in sorted(verdicts.items()):
            res = data["results"].get(rung)
            if res is None:
                continue
            pa = res["pair_accuracy"]
            n = pa["n"]
            ns_seen.add(n)
            acc = pa["acc"]
            k_correct = round(acc * n) if n else 0
            bt = binomtest(k_correct, n, 0.5, alternative="two-sided") if n else None
            rung_out["per_checkpoint"][label] = {
                "family": family_of(label),
                "n": n,
                "acc": acc,
                "k_correct_rounded": k_correct,
                "binom_p_two_sided_vs_0.5": bt.pvalue if bt else None,
                "len_sign_gap": pa.get("len_sign_gap"),
                "mean_margin": pa.get("mean_margin"),
            }
        if len(ns_seen) == 1:
            n_common = ns_seen.pop()
            rung_out["n"] = n_common
            rung_out["power_to_detect_0.60_vs_0.50_alpha0.05"] = exact_binom_power(
                n_common, 0.5, 0.60
            )
            rung_out["power_to_detect_0.65_vs_0.50_alpha0.05"] = exact_binom_power(
                n_common, 0.5, 0.65
            )
            rung_out["refuse_conclusion"] = (
                rung_out["power_to_detect_0.60_vs_0.50_alpha0.05"] < 0.80
            )
        elif ns_seen:
            # Different checkpoints saw a different n on the "same" rung -- e.g. a stratum file
            # was rebuilt between scoring runs. Report the raw disagreement rather than average
            # over it silently.
            rung_out["n"] = sorted(ns_seen)
            rung_out["WARNING"] = "checkpoints disagree on n for this rung -- see per_checkpoint"

        # Family-collapsed cross-checkpoint sign test: one point per family (mean acc of its
        # checkpoints), n_families independent units, sign test against p=0.5 of "family acc>0.5".
        family_accs: dict[str, list[float]] = {}
        for label, cell in rung_out["per_checkpoint"].items():
            family_accs.setdefault(cell["family"], []).append(cell["acc"])
        family_means = {fam: sum(v) / len(v) for fam, v in family_accs.items()}
        n_fam = len(family_means)
        n_above = sum(1 for v in family_means.values() if v > 0.5)
        n_at = sum(1 for v in family_means.values() if v == 0.5)
        sign_test_p = (
            binomtest(n_above, n_fam - n_at, 0.5, alternative="two-sided").pvalue
            if (n_fam - n_at) > 0
            else None
        )
        rung_out["family_means"] = family_means
        rung_out["cross_family_sign_test"] = {
            "n_families": n_fam,
            "n_families_above_0.5": n_above,
            "n_families_at_0.5": n_at,
            "two_sided_p": sign_test_p,
            "caveat": (
                f"n={n_fam} independent family units (E39-8b's 3 seeds collapsed to 1); "
                "a sign test at this n has very limited power on its own and is reported "
                "alongside the per-checkpoint item-level tests, not as a replacement for them"
            ),
        }
        out["rungs"][rung] = rung_out

    # Non-vacuity: unmatched_ask_ask_all's len_sign_gap must reproduce the known negative gap.
    unmatched = out["rungs"].get("unmatched_ask_ask_all", {}).get("per_checkpoint", {})
    gaps = {label: cell["len_sign_gap"] for label, cell in unmatched.items()}
    negative = {k: v for k, v in gaps.items() if v is not None and v < 0}
    out["non_vacuity"] = {
        "reference_median_len_sign_gap_from_tierA_124_verdicts": REFERENCE_LEN_SIGN_GAP_MEDIAN,
        "len_sign_gap_by_checkpoint": gaps,
        "n_checkpoints_negative": len(negative),
        "n_checkpoints_total": len(gaps),
        "all_negative": len(negative) == len(gaps) and len(gaps) > 0,
        "median_observed": (sorted(gaps.values())[len(gaps) // 2] if gaps else None),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    verdicts = load_verdicts(args.eval_dir)
    summary = analyze(verdicts)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
