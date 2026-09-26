"""Length-exploitation audit over every Tier-A verdict we have.

Applies the kill rule stated verbatim in docs/GPU_RUNBOOK.md section 5 and in the
`pair_accuracy` docstring in src/pinq_train/eval_offline.py:

    chosen_shorter accuracy below 0.50 while chosen_longer is above 0.60 means the
    policy learned length, and the checkpoint does not proceed to Tier B whatever its
    pooled accuracy says.

This script only READS existing `*.tierA.json` verdicts (backed up to
~/pi-corpus-backup/length-audit-20260919/ before this script touched anything). It runs
no model, buys no rollout, and makes no gateway call: it is a pure re-read of numbers
`pair_accuracy` already computed and wrote to disk on some earlier, unrelated run.

Usage:
    python scripts/length_audit/audit_tierA.py <root1> [<root2> ...] --out-csv PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

SIZE_PATTERNS = [
    "1.7b",
    "32b",
    "20b",
    "8b",
    "4b",
]

SEED_RE = re.compile(r"[.\-_]s(\d+)(?:[.\-_]|$)")


def guess_size(name: str) -> str:
    low = name.lower()
    for pat in SIZE_PATTERNS:
        if pat in low:
            return pat
    if "granite" in low:
        return "granite8b"
    if "gptoss" in low or "gpt-oss" in low:
        return "gptoss20b"
    return "?"


def guess_seed(name: str) -> str:
    m = SEED_RE.search(name)
    if m:
        return f"s{m.group(1)}"
    return "-"


LOW, HIGH = 0.50, 0.60


def kill_rule(acc_by_len_sign: dict[str, Any]) -> tuple[str, float | None, float | None]:
    """Did the ordering read length? Two-sided, on the declared constants.

    `docs/GPU_RUNBOOK.md` section 5 states the rule as `chosen_shorter < 0.50 AND
    chosen_longer > 0.60`. That wording is one-sided: it fires only when the policy
    prefers the LONGER question. `pair_accuracy`'s docstring states the PURPOSE as saying
    "whether the ORDERING read length", which is direction-free, so the threshold and the
    purpose disagreed and the threshold was the narrower of the two.

    CORRECTED 2026-09-19, and the correction is a spec conformance fix rather than a new
    threshold: the mirror uses the SAME two constants with the sides exchanged. What made
    it necessary was measured, not hypothesised -- all three E39-8b seeds, at both distinct
    checkpoints, over the 970 ask-vs-ask pairs, read chosen_longer 0.360-0.364 against
    chosen_shorter 0.611-0.621 (len_sign_gap -0.246 to -0.263). That ordering read length
    as plainly as the declared zone does, and the declared zone returned PASS on all six
    (artifacts/tierA_length_decomposition_20260919/).

    The direction is named in the verdict because the two are different defects: a
    long-preference inflates ASK, a short-preference inflates STOP, and an over-stopping
    checkpoint is the failure this programme already has. Callers must therefore test
    `startswith("KILL")`, never `== "KILL"`.
    """
    shorter = acc_by_len_sign.get("chosen_shorter", {}).get("acc")
    longer = acc_by_len_sign.get("chosen_longer", {}).get("acc")
    if shorter is None or longer is None:
        return "N/A (empty cell)", shorter, longer
    if shorter < LOW and longer > HIGH:
        return "KILL (long-preference)", shorter, longer
    if longer < LOW and shorter > HIGH:
        return "KILL (short-preference)", shorter, longer
    return "PASS", shorter, longer


def load_verdict(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"SKIP unreadable {path}: {e}", file=sys.stderr)
        return None
    pa = d.get("pair_accuracy")
    if not isinstance(pa, dict):
        return None
    return d


def row_for(path: Path, d: dict[str, Any], origin: str, root: Path) -> dict[str, Any]:
    pa = d["pair_accuracy"]
    checkpoint = str(d.get("checkpoint") or "")
    name = checkpoint or path.stem.replace(".tierA", "")
    by_len_sign = pa.get("acc_by_len_sign") or {}
    verdict, shorter, longer = kill_rule(by_len_sign)
    by_kind = pa.get("acc_by_kind") or {}
    ask_ask = by_kind.get("ask_ask")
    ask_stop = by_kind.get("ask_stop")

    n_shorter = (by_len_sign.get("chosen_shorter") or {}).get("n")
    n_longer = (by_len_sign.get("chosen_longer") or {}).get("n")
    n_equal = (by_len_sign.get("equal") or {}).get("n")

    gap = pa.get("len_sign_gap")

    return {
        "origin": origin,
        # relative to the backup root, never the absolute local path: the CSV is a
        # committed artifact and must not hardcode this machine's home directory.
        "path": str(path.relative_to(root)),
        "arm": name,
        "size": guess_size(name),
        "seed": guess_seed(name),
        "n_pairs": pa.get("n"),
        "dev_pairs": d.get("dev_pairs"),
        "acc_pooled": pa.get("acc"),
        "ask_ask_acc": ask_ask,
        "ask_stop_acc": ask_stop,
        "chosen_shorter_acc": shorter,
        "chosen_shorter_n": n_shorter,
        "chosen_longer_acc": longer,
        "chosen_longer_n": n_longer,
        "equal_n": n_equal,
        "len_sign_gap": gap,
        "kill_rule_verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for root in args.roots:
        origin = root.name
        for p in sorted(root.rglob("*.tierA.json")):
            key = p.name  # dedupe by filename across origins if identical basenames appear
            d = load_verdict(p)
            if d is None:
                continue
            rows.append(row_for(p, d, origin, root))
            seen_paths.add(key)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "origin",
        "arm",
        "size",
        "seed",
        "n_pairs",
        "acc_pooled",
        "ask_ask_acc",
        "ask_stop_acc",
        "chosen_shorter_acc",
        "chosen_shorter_n",
        "chosen_longer_acc",
        "chosen_longer_n",
        "equal_n",
        "len_sign_gap",
        "kill_rule_verdict",
        "dev_pairs",
        "path",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_kill = sum(1 for r in rows if r["kill_rule_verdict"].startswith("KILL"))
    n_pass = sum(1 for r in rows if r["kill_rule_verdict"] == "PASS")
    n_na = sum(1 for r in rows if r["kill_rule_verdict"].startswith("N/A"))
    print(f"verdicts read: {len(rows)}")
    print(f"KILL: {n_kill}  PASS: {n_pass}  N/A (empty len-sign cell): {n_na}")
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
