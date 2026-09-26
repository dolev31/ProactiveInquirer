#!/usr/bin/env python
"""Turn the A7 pass into things a trainer can consume, and refuse to build on a bad pass.

THREE ARTIFACTS, one purpose each:

  pairs.anticipation.jsonl   the DPO subset whose ONLY consistent contrast is anticipation:
                             the chosen side reaches for something the task never states,
                             the rejected side does not, and the rater's preference agrees.
                             Training on this optimises the definition of proactivity rather
                             than question hygiene -- measured, the full pair set is decided
                             by hygiene about 70% of the time.
  latency_relabel.jsonl      per (run_id, turn_idx, candidate_run_id): what a careful read
                             of the DECISION POINT says, beside the mechanical `is_latent`
                             flag the trainset is selected by. A4 put that flag at 63.5%
                             concordance with a careful read; this is the correction, as a
                             sidecar -- gold is never touched.
  sft.anticipation.keys      (run_id, turn_idx) of SFT rows whose action was judged to reach.
                             A key filter, not a copy: a second copy of 48MB of state_text
                             would drift from the export it came from.

THE GATE COMES FIRST. `_slot_bias` runs before anything is written and a significant result
ABORTS the build. A rater that labels by shown slot rather than content produced 1,886
usable-looking records in this campaign -- it caught 99 of 99 planted foils and still had to
be quarantined -- and every canonical aggregate over it looked healthy, because the
shown-order coin flip averages a slot effect away. So the check cannot be advisory.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pi_run.cmd_annotate import _slot_bias  # noqa: E402

ALPHA = 0.001  # the slot-bias abort threshold


def wilson(k: int, n: int) -> dict:
    if not n:
        return {"p": None, "n": 0}
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4), "n": n}


def foil_caught(response: dict, expected: object) -> bool:
    """Did the rater give the planted answer? Handles BOTH shapes `pi annotate export` plants.

    An A7 foil pins named response FIELDS -- `{"b_reaches": "stays_stated"}` -- while an
    A3_node attention check pins the VERDICT alone -- `"not_a_need"`. This called `.items()`
    unconditionally, so a bundle built with `--n-attention` died with `AttributeError: 'str'
    object has no attribute 'items'` AFTER the slot-bias gate had run: the expensive pass
    finished, the rater was declared clean, and no artifact was written.

    A missing field is a MISS, never an exception. A rater that answered in a different schema
    failed the check; it did not break the build.
    """
    if isinstance(expected, dict):
        return all(response.get(f) == v for f, v in expected.items())
    return response.get("verdict") == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="the a7-campaign directory")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--pairs", required=True, help="data/rl/pairs.jsonl the items were drawn from")
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True, help="data/rl")
    ap.add_argument("--rater", default="llm:claude-opus-5")
    a = ap.parse_args()

    camp = Path(a.campaign)
    bundle = json.loads(Path(a.bundle).read_text())
    key = json.loads(Path(a.key).read_text())["items"]
    recs = [
        json.loads(line)
        for f in sorted(glob.glob(str(camp / "batches" / "*.jsonl")))
        for line in open(f)
        if line.strip()
    ]
    if not recs:
        print("no records", file=sys.stderr)
        return 2

    # ---- gate: a slot-reading rater aborts the build ------------------------------------
    gate = _slot_bias(bundle, recs, "A7")
    for name, st in sorted(gate.items()):
        flag = "FLAGGED" if st.get("n") and st["binomial_p"] < ALPHA else "ok"
        print(
            f"  slot bias [{name}] P(first)={st.get('p_first'):.4f} n={st.get('n')} "
            f"p={st.get('binomial_p'):.3g}  {flag}"
        )
        if st.get("n") and st["binomial_p"] < ALPHA:
            print(
                "ABORT: this rater labels by shown slot, not content. Its aggregates would "
                "look healthy anyway -- the order coin flip averages the effect away -- so "
                "nothing is written.",
                file=sys.stderr,
            )
            return 2

    foils = [r for r in recs if (key.get(r["item_id"]) or {}).get("attention_check")]
    caught = sum(
        1
        for r in foils
        if foil_caught(r["response"], key[r["item_id"]]["attention_check"]["expected"])
    )
    real = [r for r in recs if not (key.get(r["item_id"]) or {}).get("attention_check")]

    # one record per item; a duplicate (planted retest) keeps the first and is counted
    seen: dict[str, dict] = {}
    n_dup = 0
    for r in real:
        if r["item_id"] in seen:
            n_dup += 1
            continue
        seen[r["item_id"]] = r

    prompt_shas = sorted({r["prompt_sha"] for r in recs if r.get("prompt_sha")})
    raters = sorted({r["annotator_id"] for r in recs})

    # ---- artifact 1: the anticipation DPO subset ----------------------------------------
    by_pair: dict[str, dict] = {}
    for iid, r in seen.items():
        k = key.get(iid) or {}
        order = k.get("order")
        if not order or not k.get("pair_id"):
            continue
        resp = r["response"]
        chosen_side = "a" if order == "ab" else "b"
        ch = resp[f"{chosen_side}_reaches"]
        rj = resp["b_reaches" if chosen_side == "a" else "a_reaches"]
        picked_chosen = resp["preference"] == chosen_side
        by_pair[k["pair_id"]] = {
            "chosen_reaches": ch,
            "rejected_reaches": rj,
            "preference": "chosen"
            if picked_chosen
            else ("rejected" if resp["preference"] in ("a", "b") else resp["preference"]),
            "basis": r.get("basis"),
            "rater": r["annotator_id"],
            "item_id": iid,
            "unstated_need": resp.get(f"unstated_need_{chosen_side}"),
        }

    out = Path(a.out)
    kept = 0
    counts = Counter()
    with open(out / "pairs.anticipation.jsonl", "w") as fh:
        for line in open(a.pairs):
            row = json.loads(line)
            a7 = by_pair.get(row.get("pair_id"))
            if not a7:
                counts["not_judged"] += 1
                continue
            # The subset: the ONLY consistent contrast between the two sides is the reach.
            if not (
                a7["chosen_reaches"] == "reaches_unstated"
                and a7["rejected_reaches"] == "stays_stated"
                and a7["preference"] == "chosen"
            ):
                counts["judged_but_not_an_anticipation_contrast"] += 1
                continue
            fh.write(json.dumps({**row, "a7": a7}, sort_keys=True) + "\n")
            kept += 1

    # ---- artifact 2: the decision-point latency relabel ---------------------------------
    relabel = 0
    conf = Counter()
    with open(out / "latency_relabel.jsonl", "w") as fh:
        for iid, r in seen.items():
            k = key.get(iid) or {}
            if not k.get("order"):
                continue
            prov = next((i["provenance"] for i in bundle["items"] if i["item_id"] == iid), {})
            order = k["order"]
            for side, run in (
                ("a", "chosen" if order == "ab" else "rejected"),
                ("b", "rejected" if order == "ab" else "chosen"),
            ):
                rid = k.get(f"{run}_run_id")
                if not rid:
                    continue
                read = r["response"][f"{side}_reaches"]
                fh.write(
                    json.dumps(
                        {
                            "run_id": prov.get("run_id"),
                            "turn_idx": prov.get("turn_idx"),
                            "candidate_run_id": rid,
                            "reaches": read,
                            "mechanical_is_latent": k.get("is_latent"),
                            "rater": r["annotator_id"],
                            "item_id": iid,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                relabel += 1
                if run == "chosen":
                    conf[(bool(k.get("is_latent")), read)] += 1

    # ---- artifact 3: the anticipatory SFT key filter ------------------------------------
    # JOINED ON THE QUESTION TEXT, not the run id. `export_sft` stamps the PARENT run id --
    # the state key -- rather than the candidate's, which is why "zero pairs could be joined
    # back to their candidates' properties" is a documented property of this export
    # (tests/test_pairs_are_auditable.py). So a candidate_run_id join returns nothing, silently.
    # The state (run_id, turn_idx) plus the normalised question text identifies the row.
    def norm_q(x: str | None) -> str:
        return " ".join(str(x or "").split()).casefold()

    reach_q: set[tuple[str, int, str]] = set()
    items_by_id = {i["item_id"]: i for i in bundle["items"]}
    for iid, r in seen.items():
        k = key.get(iid) or {}
        if not k.get("order"):
            continue
        prov = items_by_id[iid].get("provenance") or {}
        payload = items_by_id[iid].get("payload") or {}
        for side in ("a", "b"):
            if r["response"][f"{side}_reaches"] != "reaches_unstated":
                continue
            q = norm_q((payload.get(f"option_{side}") or {}).get("question"))
            if q and prov.get("run_id") is not None and prov.get("turn_idx") is not None:
                reach_q.add((str(prov["run_id"]), int(prov["turn_idx"]), q))

    sft_keys = 0
    for line in open(a.sft):
        row = json.loads(line)
        if row.get("is_stop"):
            continue
        try:
            q = norm_q(json.loads(row.get("action_json") or "{}").get("question"))
        except (TypeError, ValueError):
            continue
        if (str(row.get("run_id")), int(row.get("turn_idx", -1)), q) in reach_q:
            sft_keys += 1
    with open(out / "sft.anticipation.keys", "w") as fh:
        for line in open(a.sft):
            row = json.loads(line)
            if row.get("is_stop"):
                continue
            try:
                q = norm_q(json.loads(row.get("action_json") or "{}").get("question"))
            except (TypeError, ValueError):
                continue
            if (str(row.get("run_id")), int(row.get("turn_idx", -1)), q) in reach_q:
                fh.write(f"{row['run_id']}\t{row['turn_idx']}\n")

    agree = conf[(True, "reaches_unstated")] + conf[(False, "stays_stated")]
    man = {
        "rater_ids": raters,
        "n_prompt_shas": len(prompt_shas),
        "prompt_shas": prompt_shas[:8],
        "bundle_id": bundle["manifest"]["bundle_id"],
        "item_set_hash": bundle["manifest"].get("item_set_hash"),
        "n_records": len(recs),
        "n_items_judged": len(seen),
        "n_retest_duplicates": n_dup,
        "foil_catch": wilson(caught, len(foils)),
        "slot_bias": gate,
        "pairs_anticipation": {"kept": kept, **{k: v for k, v in counts.items()}},
        "latency_relabel_rows": relabel,
        "relabel_vs_mechanical": {
            "concordance": wilson(agree, sum(conf.values())),
            "cells": {f"is_latent={k[0]}|{k[1]}": v for k, v in sorted(conf.items())},
        },
        "sft_anticipation_keys": sft_keys,
        "sft_join": "state (run_id,turn_idx) + normalised question text -- candidate run ids do not appear in sft.jsonl",
        "source_records_sha256": hashlib.sha256(
            "".join(sorted(json.dumps(r, sort_keys=True) for r in recs)).encode()
        ).hexdigest(),
    }
    (out / "a7.manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f"\n  pairs.anticipation.jsonl : {kept} pairs")
    print(f"  latency_relabel.jsonl    : {relabel} rows")
    print(f"  sft.anticipation.keys    : {sft_keys} rows")
    print(f"  a7.manifest.json written (raters={raters})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
