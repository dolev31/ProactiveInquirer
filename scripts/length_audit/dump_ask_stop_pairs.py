"""Per-pair breakdown of the dev-set ask_stop pairs, and a mechanical-bias prediction.

This is the $0 data-only half of the length-exploitation audit's item 4 (the 11/115
constant). It does NOT run any checkpoint: no model, no torch, no GPU. What it dumps for
every `ask_stop` pair in a `pairs.dev.jsonl` file is `chosen_len`, `rejected_len`,
`len_sign` and `predicted_ok` -- the last one is a PREDICTION under the hypothesis stated
in the `pinq_train.eval_offline` module docstring ("SUM of completion log-probs ... prefers
STOP on nearly every state, for a reason that has nothing to do with stopping"), not an
observed value. `predicted_ok` = True iff the chosen side is the shorter one (i.e. chosen
== STOP), which is what a policy that mechanically always scores STOP above any ASK would
get right.

It CANNOT reproduce the real per-pair `ok` a checkpoint actually produced -- that needs a
forward pass through the trained weights, which this script does not do and which was out
of scope for a $0 offline audit. Report the distinction: this script gives the
model-independent, deterministic half of item 4 (how many of the 115 pairs even could be
explained by pure mechanical length bias), not a re-measurement of any checkpoint.

Filters on the row's own `pair_kind` field, not on `action_kind_of`: `pair_accuracy`'s
`acc_by_kind` groups by that same field, and `pair_kind` splits STOP pairs into two
populations that must not be pooled -- `ask_stop` (115 rows in pairs.dev.jsonl) and
`ask_stop_synth` (3,877 rows), which score 1.0 identically everywhere on file and is a
separate, uninteresting population (the "synth" STOP side is not competing against a real
ASK the policy considered).

Usage:
    python scripts/length_audit/dump_ask_stop_pairs.py data/rl/dev/pairs.dev.jsonl --out-csv PATH \
        [--pair-kind ask_stop]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pinq.actions import action_kind_of
from pinq_train.export.dataset import _action_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_path", type=Path)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument(
        "--pair-kind",
        default="ask_stop",
        help="Filter to rows whose own pair_kind field equals this (default: ask_stop).",
    )
    args = ap.parse_args()

    rows = []
    n_total = 0
    n_ask_stop = 0
    with args.pairs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            p = json.loads(line)
            if str(p.get("pair_kind") or "") != args.pair_kind:
                continue
            chosen_json = str(p["chosen_json"])
            rejected_json = str(p["rejected_json"])
            chosen_kind = action_kind_of(chosen_json)
            rejected_kind = action_kind_of(rejected_json)
            n_ask_stop += 1
            chosen_len = len(_action_text(chosen_json))
            rejected_len = len(_action_text(rejected_json))
            delta = chosen_len - rejected_len
            len_sign = (
                "chosen_longer" if delta > 0 else ("chosen_shorter" if delta < 0 else "equal")
            )
            chosen_is_stop = chosen_kind == "stop"
            predicted_ok = chosen_is_stop  # mechanical "always prefer the shorter (STOP)" bias
            rows.append(
                {
                    "pair_id": p.get("pair_id"),
                    "pair_kind": p.get("pair_kind"),
                    "chosen_kind": chosen_kind,
                    "rejected_kind": rejected_kind,
                    "chosen_len": chosen_len,
                    "rejected_len": rejected_len,
                    "len_sign": len_sign,
                    "chosen_is_stop": chosen_is_stop,
                    "predicted_ok_mechanical_bias": predicted_ok,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "pair_kind",
        "chosen_kind",
        "rejected_kind",
        "chosen_len",
        "rejected_len",
        "len_sign",
        "chosen_is_stop",
        "predicted_ok_mechanical_bias",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_chosen_stop = sum(1 for r in rows if r["chosen_is_stop"])
    n_chosen_ask = n_ask_stop - n_chosen_stop
    print(f"total pairs in file: {n_total}")
    print(f"pair_kind == {args.pair_kind!r} rows: {n_ask_stop}")
    print(f"  chosen == STOP (chosen_shorter by construction): {n_chosen_stop}")
    print(f"  chosen == ASK  (chosen_longer by construction):  {n_chosen_ask}")
    print(
        "predicted accuracy under 'always prefer the shorter (STOP)' mechanical bias: "
        f"{n_chosen_stop}/{n_ask_stop} = {n_chosen_stop / n_ask_stop:.6f}"
    )
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
