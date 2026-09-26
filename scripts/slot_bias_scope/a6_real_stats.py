"""Run the A6 top-tier slot-bias statistic (`pi_run.cmd_annotate._slot_bias`, task_type "A6")
on the real, immutable A6 annotation records under `~/pi-corpus-backup/annotations-20260914`,
per rater and per bundle. Read-only against the backup root.

The DEFAULT null there is now TIE-AWARE (mean of each response's own top-tier-size/k, not the
naive mean of 1/k): a tie at the top tier is the majority case on this instrument (58.8% of
real responses), and the naive null flags every rater as a result -- see
`_a6_slot_bias`'s docstring and artifacts/slot_bias_scope_20260918/RESULT.md. `expected`/`p`
below are the tie-aware (default) reading; `p_naive` is printed alongside as the misspecified
comparison only, never as a second valid reading.

Corpus definition matches the one already validated in
`paper/sections/appendix_validity.tex`'s own measurement comment (`a6/`, `a6b/`, `fixed1/`,
`fixed2/`, `growth1/` r_*.jsonl, deduplicated on record_id, restricted to task_type == "A6"):
that comment reports 4,363 A6 records after dedup (gpt-oss-120b 1,458, gemini-3.1-pro-preview
1,458, claude-sonnet-5 1,447) over 5 bundles totalling 1,458 items -- this script prints its
own raw counts so that figure is independently reproduced, not assumed.

Output pasted at artifacts/slot_bias_scope_20260918/RESULT.md. Lives under scripts/, not
artifacts/, because ruff's `extend-exclude` for `artifacts/` is justified on that tree holding
no python (see tests/test_artifacts_hold_no_python.py) -- this is tooling, not a result.

Run with `PYTHONPATH=$PWD/src` set (see CLAUDE.md), from the repository root:
    .venv/bin/python scripts/slot_bias_scope/a6_real_stats.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from pi_run.cmd_annotate import _slot_bias
from pi_run.promote import rater_of

ROOT = os.path.expanduser("~/pi-corpus-backup/annotations-20260914")
RECORD_ROOTS = ["a6", "a6b", "fixed1", "fixed2", "growth1"]
ALPHA = 0.001  # same abort threshold promote.SLOT_BIAS_ALPHA / a7_artifacts.ALPHA use for A7


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
    raw_count = 0
    for root in RECORD_ROOTS:
        for f in sorted(glob.glob(f"{ROOT}/{root}/r_*.jsonl")):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                raw_count += 1
                if r.get("task_type") != "A6":
                    continue
                rid = r.get("record_id")
                if rid in seen:
                    continue
                seen.add(rid)
                records.append(r)
    print(f"# raw JSONL lines scanned: {raw_count}")
    print(f"# deduplicated A6 records: {len(records)}")
    by_rater_raw: dict[str, int] = defaultdict(int)
    for r in records:
        by_rater_raw[rater_of(r["annotator_id"])] += 1
    for rater, n in sorted(by_rater_raw.items()):
        print(f"#   {rater}: {n} records")
    print()
    return records


def main() -> None:
    items_by_bundle = _load_items_by_bundle()
    records = _load_a6_records()

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(rater_of(r["annotator_id"]), r["bundle_id"])].append(r)

    print(
        f"{'rater':<20} {'bundle':<18} {'n_items':>8} {'P(first in top)':>16} "
        f"{'expected_tie':>12} {'p':>12} {'p_naive':>12}"
    )
    for (rater, bid), recs in sorted(groups.items()):
        items = items_by_bundle.get(bid, {})
        stat = _slot_bias({"items": list(items.values())}, recs, "A6")["top_tier"]
        bid_short = bid.split("-")[-1]
        print(
            f"{rater:<20} {bid_short:<18} {stat['n']:>8} {stat['p_first']:>16.4f} "
            f"{stat['expected']:>12.4f} {stat['binomial_p']:>12.3g} {stat['p_naive']:>12.3g}"
        )

    print()
    print("pooled per rater (all bundles combined)")
    pooled: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pooled[rater_of(r["annotator_id"])].append(r)
    all_items = [it for m in items_by_bundle.values() for it in m.values()]
    print(
        f"{'rater':<20} {'n_items':>8} {'P(first in top)':>16} {'expected_tie':>12} "
        f"{'p':>12} {'p_naive':>12}  flag"
    )
    for rater, recs in sorted(pooled.items()):
        stat = _slot_bias({"items": all_items}, recs, "A6")["top_tier"]
        flag = "FLAGGED" if stat["n"] and stat["binomial_p"] < ALPHA else "ok"
        print(
            f"{rater:<20} {stat['n']:>8} {stat['p_first']:>16.4f} {stat['expected']:>12.4f} "
            f"{stat['binomial_p']:>12.3g} {stat['p_naive']:>12.3g}  {flag}"
        )


if __name__ == "__main__":
    main()
