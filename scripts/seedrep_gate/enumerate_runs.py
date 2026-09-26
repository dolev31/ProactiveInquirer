"""Enumerate the seed-replicate run directories by reading every manifest.

A shell glob over `runs/` returns a false EMPTY once the tree has ~174k subdirectories
(see memory `shell-glob-over-runs-returns-empty`), and a relative `find -newermt` reads as
a clean zero. So the selector of record here is a full Python scan of `runs/*/manifest.json`
with an explicit predicate, and the script PRINTS the total number of manifests it opened so
that a zero result can be told apart from a scan that never ran.

Writes one run-id list per (model_id, suite) into the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--grid", default="tier1_trained_qa_base")
    args = ap.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(args.model)

    opened = 0
    scanned = 0
    hits: dict[tuple[str, str], list[str]] = defaultdict(list)
    fields: dict[str, Counter] = defaultdict(Counter)

    with os.scandir(runs_root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            scanned += 1
            mpath = os.path.join(entry.path, "manifest.json")
            try:
                with open(mpath, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            opened += 1
            # cheap prefilter before json.loads on 178k files
            if not any(m.encode() in raw for m in wanted):
                continue
            m = json.loads(raw)
            mid = (m.get("pins") or {}).get("inquirer", {}).get("model_id")
            if mid not in wanted:
                continue
            if m.get("grid_name") != args.grid:
                continue
            hits[(mid, m.get("suite_id"))].append(entry.name)
            for f in (
                "code_version",
                "split",
                "dirty",
                "arm_id",
                "budget_cap",
                "max_turns",
                "grid_sha256",
                "model_pin_hash",
                "train_id_set_hash",
                "seed",
                "pilot_flag",
                "exploratory",
                "gold_exposed",
                "canary_hit",
                "firewall_ok",
                "counterfactual_kind",
            ):
                fields[f"{mid}:{f}"][repr(m.get(f))] += 1
            for role in ("inquirer", "answerer", "drafter"):
                p = (m.get("pins") or {}).get(role, {})
                fields[f"{mid}:{role}.base_url_sha"][repr(p.get("base_url_sha"))] += 1
                fields[f"{mid}:{role}.model_id"][repr(p.get("model_id"))] += 1

    print(f"run dirs scanned: {scanned}")
    print(f"manifests opened: {opened}")
    if opened == 0:
        print("REFUSING: opened zero manifests; the scan did not run", file=sys.stderr)
        return 2
    total = 0
    for (mid, suite), ids in sorted(hits.items()):
        ids.sort()
        path = os.path.join(out_dir, f"run_ids.{mid}.{suite}.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"{path}: {len(ids)}")
        total += len(ids)
    print(f"total matched: {total}")
    print("--- field census over matched runs ---")
    for k in sorted(fields):
        print(f"{k}: {dict(fields[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
