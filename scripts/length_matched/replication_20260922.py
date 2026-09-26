"""Replicate the preference-minus-imitation ask_ask contrast across seeds, objectives and size.

    .venv/bin/python scripts/length_matched/replication_20260922.py \
        > artifacts/length_matched_replication_20260922/report.json

WHY THIS EXISTS. The paper's ranking claim rests on ONE preference checkpoint
(`qwen3-8b-dpo-stacked-notdone-both`, +0.0536 over 970 pairs, `build_report_20260919.py`). This
asks whether that is a property of the preference stage or of that one run, by scoring every
other trained preference checkpoint on the SAME pair file with the SAME instrument
(`score_checkpoint_paired.py`, staged tree verified 79 files / 0 bad) and joining each to ITS OWN
starting adapter. It reuses `join_by_pair_id`, `mcnemar_contrast` and `paired_bootstrap_ci`
verbatim at the published settings (10,000 resamples, seed 20260919), and the first contrast it
computes is the published one, so a disagreement there is a bug here and not a result.

THE REFERENCE IS THE CHECKPOINT'S OWN START, never a neighbour's. `E10/s0` is not a separate
imitation run: its adapter_sha equals `qwen3-8b-sft-headline`'s (a255a40b...) and its per-pair
decisions agree on 970 of 970, so the objective arms and the headline arms share one reference.
The 32B DPO started from `rung1-32b-headline-s1/checkpoint-1600`, which is therefore its causal
reference; the 32B imitation checkpoint already in the 0919 record is a DIFFERENT SFT seed, and a
contrast against it is reported but labelled as crossing seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_20260919 import RUNG_ORDER  # noqa: E402
from paired_contrast import join_by_pair_id, mcnemar_contrast, paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts" / "length_matched_stratum_20260919" / "eval_paired"
NEW = ROOT / "artifacts" / "length_matched_replication_20260922" / "eval_paired"

# (contrast name, imitation file, preference file, is the imitation the preference's own start)
CONTRASTS = [
    (
        "published_stacked_8b",
        OLD / "qwen3-8b-sft-headline",
        OLD / "qwen3-8b-dpo-stacked-notdone-both",
        False,
    ),
    (
        "notdone_both_8b_s0",
        OLD / "qwen3-8b-sft-headline",
        NEW / "qwen3-8b-dpo-notdone-both-s0",
        True,
    ),
    (
        "notdone_both_8b_s1",
        OLD / "qwen3-8b-sft-headline",
        NEW / "qwen3-8b-dpo-notdone-both-s1",
        True,
    ),
    ("cdpo_8b", NEW / "qwen3-8b-sft-E10-s0", NEW / "qwen3-8b-E50-cdpo-s0", True),
    ("ipo_8b", NEW / "qwen3-8b-sft-E10-s0", NEW / "qwen3-8b-E50-ipo-s0", True),
    ("ld_8b", NEW / "qwen3-8b-sft-E10-s0", NEW / "qwen3-8b-E50-ld-s0", True),
    ("ancestor_pairs_8b", NEW / "qwen3-8b-sft-E10-s0", NEW / "qwen3-8b-E62-s0", True),
    (
        "notdone_both_32b_step200",
        NEW / "qwen3-32b-sft-headline-s1-step1600",
        NEW / "qwen3-32b-dpo-notdone-both-s200",
        True,
    ),
    (
        "notdone_both_32b_step200_vs_other_sft_seed",
        OLD / "qwen3-32b-sft-headline",
        NEW / "qwen3-32b-dpo-notdone-both-s200",
        False,
    ),
    (
        "notdone_both_32b_step400",
        NEW / "qwen3-32b-sft-headline-s1-step1600",
        NEW / "qwen3-32b-dpo-notdone-both-s400",
        True,
    ),
    # THE ARM THE PAPER REPORTS. `published_stacked_8b` above is the stacked recipe's seed 0, which a
    # resume reloaded to its start (resumed_from checkpoint-1800; 0.0026 from its init). Seeds 1 and 2
    # are the recipe's fresh runs (resumed_from None). Against imitation they give the total preference
    # effect; against their own start, the control DPO, what the stacking stage itself adds.
    (
        "stacked_8b_fresh_s1",
        OLD / "qwen3-8b-sft-headline",
        NEW / "qwen3-8b-dpo-stacked-notdone-both-s1",
        False,
    ),
    (
        "stacked_8b_fresh_s2",
        OLD / "qwen3-8b-sft-headline",
        NEW / "qwen3-8b-dpo-stacked-notdone-both-s2",
        False,
    ),
    (
        "control_8b",
        OLD / "qwen3-8b-sft-headline",
        NEW / "qwen3-8b-dpo-headline-control",
        True,
    ),
    (
        "stacking_over_control_s1",
        NEW / "qwen3-8b-dpo-headline-control",
        NEW / "qwen3-8b-dpo-stacked-notdone-both-s1",
        True,
    ),
    (
        "stacking_over_control_s2",
        NEW / "qwen3-8b-dpo-headline-control",
        NEW / "qwen3-8b-dpo-stacked-notdone-both-s2",
        True,
    ),
]


def _load(stem: Path) -> dict:
    return json.loads(Path(f"{stem}.paired.json").read_text())


def main() -> int:
    report: dict = {
        "pairs_sha256": None,
        "n_boot": 10000,
        "seed": 20260919,
        "contrasts": {},
    }
    for name, imit_stem, pref_stem, own_start in CONTRASTS:
        imit, pref = _load(imit_stem), _load(pref_stem)
        for doc in (imit, pref):
            sha = doc["stratum_manifest_source_sha256"]
            if report["pairs_sha256"] is None:
                report["pairs_sha256"] = sha
            elif sha != report["pairs_sha256"]:
                raise SystemExit(
                    f"{doc['label']} was scored on pair file {sha}, not {report['pairs_sha256']}"
                )
        rungs = {}
        for rung in RUNG_ORDER:
            ra, rb = imit["results"].get(rung), pref["results"].get(rung)
            if not ra or not rb:
                continue
            joined = join_by_pair_id(ra["paired"]["per_pair"], rb["paired"]["per_pair"])
            rungs[rung] = {
                "mcnemar": mcnemar_contrast(joined),
                "paired_bootstrap_95ci": paired_bootstrap_ci(joined, n_boot=10000, seed=20260919),
                "len_sign_gap": {
                    "imitation": ra["pair_accuracy_crosscheck"].get("len_sign_gap"),
                    "preference": rb["pair_accuracy_crosscheck"].get("len_sign_gap"),
                },
            }
        report["contrasts"][name] = {
            "imitation": {
                "label": imit["label"],
                "adapter_sha": imit["checkpoint_provenance"].get("adapter_sha"),
            },
            "preference": {
                "label": pref["label"],
                "adapter_sha": pref["checkpoint_provenance"].get("adapter_sha"),
            },
            "imitation_is_own_start": own_start,
            "rungs": rungs,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
