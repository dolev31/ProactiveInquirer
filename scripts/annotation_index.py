#!/usr/bin/env python
"""Collect every annotation record into one place, with provenance for each rater.

WHY THIS EXISTS. A campaign accumulates record files from several raters at different times:
an automatic pass, an ablation of that pass at a different prompt or model, a human session,
a hand-adjudication. Left as loose JSONL they are indistinguishable at a glance, and the one
question you must be able to answer about any annotation -- WHO said this, about WHICH bundle,
under WHICH prompt -- becomes archaeology. This walks the record files, verifies each against
the bundle it claims, and writes:

  annotations/all-records.jsonl   every record, each stamped with the `source` file it came from
  annotations/INDEX.json          one entry per source: rater, kind, model pin, prompt shas,
                                  counts by task type, and what that source may back

RATER PRECEDENCE, AND THE ONE PLACE IT DOES NOT APPLY. `PRECEDENCE` orders raters for
adjudication and for review: when sources disagree, the earlier rater is shown first and is
the one a reviewer reads. That is a preference about ATTENTION and it is freely chosen.

It is not a route into gold. `pi_eval.annotate.consensus` counts human raters only and
`merge_into_graphs` refuses a consensus that counted a model, so no ordering written here can
promote an `annotator_kind: "llm"` record into `gold_human_asked`. That field is a claim about
what a PERSON would have thought to ask; a model's prediction of it is a different quantity
with the same shape, and ADR computed over one would be a published number from a measurement
nobody took. Precedence decides who is read first, never who counts as a person.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Read first when sources disagree. See the module docstring for what this does NOT do.
PRECEDENCE: tuple[str, ...] = (
    "llm:claude-fable-5",  # hand-read in-session, item by item; preferred as the reference read
    "llm:azure/gpt-5.6-terra",
    "llm:openai/aws/gpt-oss-120b",
)


def _load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rank(annotator_id: str) -> int:
    return PRECEDENCE.index(annotator_id) if annotator_id in PRECEDENCE else len(PRECEDENCE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--suite", default="musique")
    ap.add_argument("--records", action="append", default=[], help="a records JSONL to index")
    ap.add_argument("--bundles", default=None, help="dir holding <bundle_id>.json (default gold)")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    human = root / "data" / "gold" / "human" / a.suite
    bundles_dir = Path(a.bundles) if a.bundles else human / "bundles"
    out_dir = human / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles: dict[str, dict] = {}
    for p in sorted(bundles_dir.glob("*.json")):
        if p.name.endswith(".key.json"):
            continue
        try:
            b = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        bundles[b["manifest"]["bundle_id"]] = b

    index: list[dict] = []
    merged: list[dict] = []
    for spec in a.records:
        p = Path(spec)
        if not p.exists():
            index.append({"source": str(p), "status": "MISSING"})
            continue
        recs = _load(p)
        if not recs:
            index.append({"source": p.name, "status": "EMPTY"})
            continue
        bids = Counter(str(r.get("bundle_id") or "") for r in recs)
        raters = Counter(str(r.get("annotator_id") or "") for r in recs)
        kinds = Counter(str(r.get("annotator_kind") or "human") for r in recs)
        bid = bids.most_common(1)[0][0]
        bundle = bundles.get(bid)

        errs: list[str] = []
        if bundle is None:
            errs.append(f"no bundle {bid} on disk: these records cannot be verified")
        else:
            from pi_eval.annotate import validate_records

            errs = validate_records(bundle, recs)

        entry = {
            "source": p.name,
            "path": str(p),
            "bundle_id": bid,
            "n_records": len(recs),
            "raters": dict(raters),
            "annotator_kind": dict(kinds),
            "by_task_type": dict(Counter(str(r.get("task_type")) for r in recs)),
            "model_pins": sorted({str(r["model_pin"]) for r in recs if r.get("model_pin")}),
            "distinct_prompt_shas": len({r["prompt_sha"] for r in recs if r.get("prompt_sha")}),
            "with_rationale": sum(1 for r in recs if (r.get("rationale") or "").strip()),
            "usd": round(sum(float(r.get("usd") or 0.0) for r in recs), 4),
            "validation_errors": len(errs),
            "first_error": errs[0] if errs else None,
            # What this source is allowed to back. A model source can never back gold; that is
            # enforced in `pi_eval.annotate`, and stated here so a reader of the index does not
            # have to go and check.
            "may_back_gold": all(k == "human" for k in kinds),
            "precedence": min(_rank(r) for r in raters),
        }
        index.append(entry)
        for r in recs:
            merged.append({**r, "source": p.name})

    index.sort(key=lambda e: (e.get("precedence", 99), e.get("source", "")))
    merged.sort(key=lambda r: (_rank(str(r.get("annotator_id"))), str(r.get("item_id"))))

    (out_dir / "all-records.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in merged) + "\n"
    )
    by_rater: dict[str, Counter] = defaultdict(Counter)
    for r in merged:
        by_rater[str(r.get("annotator_id"))][str(r.get("task_type"))] += 1
    (out_dir / "INDEX.json").write_text(
        json.dumps(
            {
                "suite": a.suite,
                "precedence": list(PRECEDENCE),
                "precedence_note": (
                    "Order for adjudication and review only. `consensus` counts human raters "
                    "only and `merge_into_graphs` refuses a non-human consensus, so no "
                    "ordering here can promote a model record into gold_human_asked."
                ),
                "n_records": len(merged),
                "by_rater": {k: dict(v) for k, v in sorted(by_rater.items())},
                "sources": index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"{len(merged)} records from {len(index)} source(s) -> {out_dir}")
    for e in index:
        if e.get("status"):
            print(f"  {e['source']:<28} {e['status']}")
            continue
        gold = "gold-eligible" if e["may_back_gold"] else "reference only"
        bad = f"  {e['validation_errors']} INVALID" if e["validation_errors"] else ""
        print(
            f"  {e['source']:<28} {e['n_records']:>5} rec  {list(e['raters'])[0]:<28}"
            f" {gold:<15}{bad}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
