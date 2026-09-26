"""Attach A2 rater verdicts to `pairs.jsonl` as METADATA. Never a filter.

A rater disagreeing with a mechanical label is a fact about the pair, not a verdict on it, so
nothing is dropped here. Training code can weight or exclude on these fields with the reason
visible; a silent filter would hide which pairs a model and the pipeline disagreed about,
which is exactly the population worth looking at.

    python scripts/curate_pairs_with_a2.py <bundle-stem> <pairs.jsonl> <out.jsonl> \
        sol=<records.jsonl> opus5=<records.jsonl>

IF YOU RE-RUN THE BUNDLE WITH FOILS, CHECK FOR ID COLLISIONS FIRST. `annotate.item_id` hashes
task_type, provenance and payload but NOT context, and a foil rewrites the question inside
context -- so a planted item and its unplanted twin share an id, `--resume` reports "already
done", and one record is reused against a prompt that has changed underneath it. Measured on
convlog by `proactiveinquirer-a4`: 114-157 items per rater held two records with two different
prompts.

The bundle behind `artifacts/validation/` is clean of this (441 items, 441 distinct ids, 0
context divergence) ONLY because it was built with `--n-a7-foils 0`, so nothing rewrote a
question. That is a property of that invocation, not of the code. Add foils and the collision
returns; fold a digest of the shown context into provenance first, as `convlog_sample` now
does.

Fields added to every row (raters that never saw a pair get "absent"):

    a2_verdict_<tag>   chosen | rejected | tie | both_bad | absent
    a2_agree           every rater that expressed a preference favoured OUR chosen
    a2_contested       some rater favoured the REJECTED candidate
    a2_both_bad        some rater judged both candidates poor
    a2_n_raters        how many raters saw this pair
    a2_bundle_id       provenance of the sample
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, "src")
from pi_eval.annotate import _units  # the repo's own decoder; order semantics live there


def identity(row: "Mapping[str, Any]") -> tuple:
    """What a pair IS: the two candidate runs compared at one state.

    MEASURED 2026-09-04: a re-export changed EVERY pair_id -- 0 of 27,266 matched the rated
    set, while the same comparisons were still there. A pair_id is a digest of whatever the
    exporter hashed that day. Joining on identity recovered 354 of 435 rated pairs where
    pair_id recovered none. Never key annotation to a pair_id.
    """
    return (
        row.get("suite_id"),
        row.get("task_id"),
        row.get("branch_of_run_id"),
        row.get("branch_turn_idx"),
        row.get("turn_idx"),
        row.get("chosen_run_id"),
        row.get("rejected_run_id"),
    )


def _rows(path: Path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def load_records(items: dict, keyi: dict, path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task_type") != "A2":
            continue
        resp = r["response"]
        resp = ast.literal_eval(resp) if isinstance(resp, str) else resp
        u = _units(items[r["item_id"]], resp, keyi.get(r["item_id"]))
        if u:
            out[r["item_id"]] = u[0][2]
    return out


def main() -> int:
    stem, pairs_p, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    args = sys.argv[4:]
    # The export the rating was BUILT on: needed to resolve pair_id -> identity once.
    rated_p = Path(next(a.split("=", 1)[1] for a in args if a.startswith("rated=")))
    raters = dict(a.split("=", 1) for a in args if not a.startswith("rated="))
    bundle = json.loads((stem.with_suffix(".json")).read_text())
    key = json.loads(Path(str(stem) + ".key.json").read_text())
    items = {i["item_id"]: i for i in bundle["items"]}
    ki = key["items"]
    keyi = dict(ki) if isinstance(ki, dict) else {x["item_id"]: x for x in ki}

    # pair_id resolves to an IDENTITY once, via the export the rating was built on; the
    # verdicts are then keyed to that identity and survive any later re-export.
    rated_rows = {str(r["pair_id"]): r for r in _rows(rated_p) if r.get("pair_id")}
    by_ident: dict[tuple, dict[str, str]] = {}
    for tag, p in raters.items():
        for iid, verdict in load_records(items, keyi, Path(p)).items():
            pid = (keyi.get(iid) or {}).get("pair_id")
            row = rated_rows.get(str(pid)) if pid else None
            if row is not None:
                by_ident.setdefault(identity(row), {})[tag] = verdict

    bundle_id = bundle.get("manifest", {}).get("bundle_id", stem.name)
    n_tagged = 0
    with out_p.open("w") as w:
        for line in pairs_p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            v = by_ident.get(identity(row), {})
            for tag in raters:
                row[f"a2_verdict_{tag}"] = v.get(tag, "absent")
            decided = [x for x in v.values() if x in ("chosen", "rejected")]
            row["a2_n_raters"] = len(v)
            row["a2_agree"] = bool(decided) and all(x == "chosen" for x in decided)
            row["a2_contested"] = any(x == "rejected" for x in v.values())
            row["a2_both_bad"] = any(x == "both_bad" for x in v.values())
            row["a2_bundle_id"] = bundle_id if v else ""
            n_tagged += bool(v)
            w.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        f"wrote {out_p}  ({n_tagged} of {sum(1 for _ in pairs_p.open())} pairs carry A2 verdicts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
