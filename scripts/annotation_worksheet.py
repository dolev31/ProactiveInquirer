#!/usr/bin/env python
"""Dump a slice of bundle items as readable, BLINDED text for an annotator to work from.

Reads the bundle only — never the `.key.json` — so nothing printed can carry the shown-order
coin flip, a margin, a mechanical label, or a foil's expected answer. An annotator who has
seen any of those is no longer answering the question the instrument asks.

Two rendering classes, and the distinction is the whole tool:
  full()  — a string the annotator DECIDES ON. Shown whole, always. The first campaign cut
            decision-critical text four separate times (160-char options, 4-unit evidence
            caps, 320-char evidence, 200-char answers) and every cut was found by a worker
            mid-task, not by the tool's author; one of them voided an entire instrument.
  wrap()  — genuine background, may be truncated.

This file lives in the repo, not in a scratch directory: its predecessor was lost to a tmp
cleaner mid-campaign, and a fidelity gate that can vanish is not a gate.

    annotation_worksheet.py --bundle B.json --type A7 --ids-file slice.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def wrap(s: str | None, n: int = 150) -> str:
    """CONTEXT only. Never call this on a string the annotator is judging."""
    return (s or "").replace("\n", " ").strip()[:n]


def full(s: str | None) -> str:
    """A string the annotator decides on. Shown whole, always."""
    return (s or "").replace("\n", " ").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument(
        "--ids-file", default=None, help="item_ids, one per line; overrides --start/--n"
    )
    a = ap.parse_args()

    b = json.loads(Path(a.bundle).read_text())
    pool = [i for i in b["items"] if i["task_type"] == a.type]
    if a.ids_file:
        # Bundle order is not label-neutral (the first campaign shipped a bundle sorted by
        # its hidden label); explicit shuffled id lists keep each slice a fair sample.
        want = [x.strip() for x in Path(a.ids_file).read_text().splitlines() if x.strip()]
        by_id = {i["item_id"]: i for i in pool}
        missing = [x for x in want if x not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} id(s) not in {a.type}: {missing[:3]}")
        items = [by_id[x] for x in want]
    else:
        items = pool[a.start : a.start + a.n]

    print(f"# {a.type}  {len(items)} items  bundle {b['manifest']['bundle_id']}")
    for k, it in enumerate(items):
        c = it.get("context") or {}
        p = it.get("payload") or {}
        print(f"\n[{k}] {it['item_id']}")
        if c.get("question"):
            print(f"  TASK: {full(c['question'])}")
        # Everything read so far, WHOLE: for a judgment about what the task does or does not
        # state, a fact hidden past a cut reads as absent and biases the exact label pair
        # the instrument exists to separate.
        for e in c.get("evidence") or []:
            print(f"  ev: ({wrap(e.get('title'), 60)}) {full(e.get('text'))}")
        for h in c.get("history") or []:
            print(f"  askedQ: {full(h.get('q'))}")
            print(f"  gotA  : {full(h.get('a'))}")
        if c.get("draft"):
            print(f"  DRAFT: {full(c['draft'])}")
        if c.get("state_text") and not (c.get("evidence") or c.get("history")):
            print(f"  STATE: {full(c['state_text'])}")
        if p.get("option_a"):
            print(f"  A: {full(p['option_a'].get('question'))}")
            print(f"  B: {full(p['option_b'].get('question'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
