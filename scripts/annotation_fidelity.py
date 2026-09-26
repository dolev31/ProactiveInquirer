#!/usr/bin/env python
"""Does a worksheet dump actually show the annotator everything they are judging?

Compares a bundle against a RENDERED DUMP FILE, string by string, and fails loudly on any
decision-critical text an annotator would judge but never see. Run it on the exact file a
worker will read, BEFORE dispatching work.

Verify the artifact, not the tool. The first campaign's checker re-ran the (since-fixed)
rendering tool and reported an instrument clean while the dump its annotators had actually
read contained zero of 1,111 evidence units — the whole instrument was voided and redone.
Hence `--dump` is the only mode here.

A field is decision-critical when the judgment is ABOUT it, or when the instrument's
question is whether the shown material is sufficient (A7's "does the task statement name
this?" makes the task question, the full evidence/history, and both options load-bearing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# per task type: (context string fields, whether evidence/history/draft must be whole)
CRITICAL: dict[str, tuple[list[str], bool]] = {
    "A1": (["question"], False),
    "A2": (["question"], True),
    "A3_node": (["question", "node_text"], False),
    "A3_edge": (["question", "src_text", "dst_text"], False),
    "A3_match": (["asked_question", "node_text"], False),
    "A3_missing": (["question"], False),
    "A4": (["question", "asked_question"], False),
    "A5": (["question", "answer"], True),
    "A6": (["question"], True),
    "A7": (["question"], True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dump", required=True, help="the rendered worksheet file a worker will read")
    ap.add_argument("--type", required=True)
    a = ap.parse_args()

    b = json.loads(Path(a.bundle).read_text())
    out = Path(a.dump).read_text()
    ctx_fields, whole_ev = CRITICAL[a.type]
    items = [i for i in b["items"] if i["task_type"] == a.type and i["item_id"] in out]
    if not items:
        print(f"NO {a.type} items from this bundle appear in {a.dump}", file=sys.stderr)
        return 1

    miss: dict[str, int] = {}

    def check(label: str, s: str | None) -> None:
        s = (s or "").replace("\n", " ").strip()
        if s and s not in out:
            miss[label] = miss.get(label, 0) + 1

    for i in items:
        c = i.get("context") or {}
        p = i.get("payload") or {}
        for f in ctx_fields:
            check(f"context.{f}", c.get(f))
        for f in ("option_a", "option_b"):
            if isinstance(p.get(f), dict):
                check(f"payload.{f}.question", p[f].get("question"))
        for lst, key in (("nodes", "text"), ("candidates", "text")):
            for e in p.get(lst) or []:
                check(f"payload.{lst}[]", e.get(key) or e.get("question"))
        if whole_ev:
            for e in c.get("evidence") or []:
                check("context.evidence[].text", e.get("text"))
            for h in c.get("history") or []:
                check("context.history[].q", h.get("q"))
                check("context.history[].a", h.get("a"))
            check("context.draft", c.get("draft"))

    bad = sum(miss.values())
    print(
        f"{a.type}: {len(items)} items in the dump — "
        + ("ALL DECISION-CRITICAL TEXT SHOWN" if not bad else "TRUNCATED")
    )
    for k, v in sorted(miss.items()):
        print(f"  {k:<28} {v} not shown in full")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
