"""Sanity check on the A6 top-tier statistic: is the naive null 1/k an artifact of ties?

If a rater commonly puts MORE than one candidate in the top tier (a real, allowed A6 judgment
-- equal tiers are a genuine tie, not a forced order), then P(a UNIFORMLY RANDOM position is in
the top tier) is (top-tier size)/k, not 1/k, even for a rater who reads content perfectly and
is fully blind to position (positions are shuffled at build time). 1/k understates the true
null whenever ties are common, which would flag every rater regardless of any real position
effect -- exactly the "instrument error reads as a finding" trap this repo's own memory names
repeatedly. This script measures the actual top-tier size distribution and recomputes the
statistic with a tie-aware null (mean of m_i/k_i, m_i = that item's own top-tier size) instead
of the naive mean(1/k_i), for comparison only -- it does not change the shipped
implementation, `pi_run.cmd_annotate._a6_slot_bias`, which matches the literal task spec
(null 1/k). See artifacts/slot_bias_scope_20260918/RESULT.md for the reading of the result.

Lives under scripts/, not artifacts/, for the same reason as a6_real_stats.py in this
directory: it is tooling, not a result, and artifacts/ is excluded from ruff on the recorded
ground that no python lives there.

Run with `PYTHONPATH=$PWD/src` set (see CLAUDE.md), from the repository root:
    .venv/bin/python scripts/slot_bias_scope/a6_tie_check.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from pi_run.cmd_annotate import _binomial_two_sided_p
from pi_run.promote import rater_of

ROOT = os.path.expanduser("~/pi-corpus-backup/annotations-20260914")
RECORD_ROOTS = ["a6", "a6b", "fixed1", "fixed2", "growth1"]


def _load_items_by_bundle() -> dict[str, dict[str, dict]]:
    items_by_bundle: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(f"{ROOT}/bundles/*.json")):
        if f.endswith(".key.json"):
            continue
        b = json.load(open(f))
        bid = b["manifest"]["bundle_id"]
        items_by_bundle[bid] = {str(it["item_id"]): it for it in b["items"]}
    return items_by_bundle


def _load_a6_records() -> list[dict]:
    seen: set[str] = set()
    records: list[dict] = []
    for root in RECORD_ROOTS:
        for f in sorted(glob.glob(f"{ROOT}/{root}/r_*.jsonl")):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("task_type") != "A6":
                    continue
                rid = r.get("record_id")
                if rid in seen:
                    continue
                seen.add(rid)
                records.append(r)
    return records


def main() -> None:
    items_by_bundle = _load_items_by_bundle()
    records = _load_a6_records()

    sizes = []
    for r in records:
        item = items_by_bundle.get(r["bundle_id"], {}).get(str(r["item_id"]))
        if item is None:
            continue
        candidates = item["payload"]["candidates"]
        shown = {c["candidate_id"] for c in candidates}
        tiers = (r.get("response") or {}).get("tiers")
        if not isinstance(tiers, dict) or not tiers:
            continue
        rated = {
            cid: t
            for cid, t in tiers.items()
            if cid in shown and isinstance(t, (int, float)) and not isinstance(t, bool)
        }
        if not rated:
            continue
        top = min(rated.values())
        sizes.append(sum(1 for t in rated.values() if t == top))

    print(f"n responses with a readable tier map: {len(sizes)}")
    print(f"mean top-tier size m: {sum(sizes) / len(sizes):.3f}")
    print(
        f"share with m == 1 (a genuine unique winner, no tie at the top): "
        f"{sum(1 for m in sizes if m == 1) / len(sizes):.3f}"
    )
    print(
        f"share with m >= 2 (a tie at the top): {sum(1 for m in sizes if m >= 2) / len(sizes):.3f}"
    )
    print()

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[rater_of(r["annotator_id"])].append(r)

    print(
        f"{'rater':<20} {'n':>6} {'P(first in top)':>16} {'naive 1/k':>10} "
        f"{'tie-aware m/k':>14} {'p (naive)':>12} {'p (tie-aware)':>14}"
    )
    for rater, recs in sorted(groups.items()):
        n = n_first = 0
        p0_naive_sum = 0.0
        p0_tie_sum = 0.0
        for r in recs:
            item = items_by_bundle.get(r["bundle_id"], {}).get(str(r["item_id"]))
            if item is None or item["task_type"] != "A6":
                continue
            candidates = item["payload"]["candidates"]
            if len(candidates) < 3:
                continue
            first_id = candidates[0]["candidate_id"]
            shown = {c["candidate_id"] for c in candidates}
            tiers = (r.get("response") or {}).get("tiers")
            if not isinstance(tiers, dict) or first_id not in tiers:
                continue
            rated = {
                cid: t
                for cid, t in tiers.items()
                if cid in shown and isinstance(t, (int, float)) and not isinstance(t, bool)
            }
            if first_id not in rated:
                continue
            top = min(rated.values())
            m = sum(1 for t in rated.values() if t == top)
            k = len(candidates)
            n += 1
            n_first += rated[first_id] == top
            p0_naive_sum += 1.0 / k
            p0_tie_sum += m / k
        p_first = n_first / n
        p0_naive = p0_naive_sum / n
        p0_tie = p0_tie_sum / n
        p_naive = _binomial_two_sided_p(n_first, n, p0_naive)
        p_tie = _binomial_two_sided_p(n_first, n, p0_tie)
        print(
            f"{rater:<20} {n:>6} {p_first:>16.4f} {p0_naive:>10.4f} {p0_tie:>14.4f} "
            f"{p_naive:>12.3g} {p_tie:>14.3g}"
        )


if __name__ == "__main__":
    main()
