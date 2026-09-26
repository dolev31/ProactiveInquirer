"""Does the preference dataset actually teach proactivity? Re-runnable, from the artifact alone.

THREE CHECKS, AND ONLY ONE OF THEM IS EVIDENCE.

  task success  -- TAUTOLOGICAL. `_outcome_prefers` orders pairs on exactly this, so a result
                   below 100% would mean the rule is broken. It is a rule-confirmation, not a
                   finding, and reporting it as a finding would be circular.
  speed         -- MOSTLY TAUTOLOGICAL, for the same reason: `_quicker` ranks on
                   turns_to_complete whenever the outcomes agree.
  latent pursuit-- GENUINE **ONLY ON THE CONTROL EXPORT**, and this script now reads which
                   export it was handed rather than asserting it. `ExportManifest.rank_rule`
                   is "outcome" for `pairs.jsonl`, where nothing in the ranking references a
                   latent-need label, so a systematic skew is an EMERGENT property of ranking
                   on outcome, speed and evidence gain. It is "anticipation" for
                   `pairs.anticipation.jsonl`, where `_anticipates` puts `newly_reachable`
                   into the precedence -- and there the check is TRUE BY CONSTRUCTION on every
                   pair it decided. Quoting the second as evidence would be circular.

                   The sentence that used to stand here -- "verified by inspection of
                   `_prefers`/`_quicker`/`_outcome_prefers`, which contain zero such
                   references" -- was a claim about code this script cannot see, and it went
                   stale the moment `--rank anticipation` was added. The manifest beside the
                   file is checked instead.

Discordant pairs only, for the latent check: pairs where both sides agree carry no information
about which signal did the choosing. ask_stop pairs are EXCLUDED from it: a STOP retrieves
nothing, so it is `is_latent=False` by construction and every ask_stop pair would count as a
discordant win for whichever side happened to be the question.

Usage: .venv/bin/python scripts/validate_pairs.py <pairs.jsonl>
"""

from __future__ import annotations

import collections
import json
import math
import statistics as ss
import sys
from pathlib import Path


def one_sided_p(k: int, n: int) -> float:
    """P(X >= k | n, p=0.5). Exact, so a reader does not have to trust a normal approximation
    at counts where it is poor."""
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def rank_rule_of(path: Path) -> str:
    """The ordering the export was written under, from the manifest beside the file.

    A manifest that EXISTS but carries no `rank_rule` was written before the flag existed,
    and there was no other ordering then: that is the control, and saying so is a fact about
    the writer, not an assumption. NO manifest at all is genuinely unknown -- the file could
    have been filtered, concatenated or re-ranked by anything -- and the difference decides
    whether the headline below is evidence or a rule restated, so it is not guessed.
    """
    m = path.parent / (path.name.replace(".jsonl", "") + ".manifest.json")
    if not m.exists():
        return "unknown"
    try:
        return str(json.loads(m.read_text()).get("rank_rule") or "outcome")
    except Exception:
        return "unknown"


def main(path: str) -> int:
    pairs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not pairs:
        print("no pairs")
        return 2
    rank = rank_rule_of(Path(path))
    states = {(p["task_id"], p["run_id"], p["turn_idx"]) for p in pairs}
    per = collections.Counter((p["task_id"], p["run_id"], p["turn_idx"]) for p in pairs)
    v = sorted(per.values())
    print(f"pairs {len(pairs)}   states {len(states)}")
    print(
        f"pairs/state: min {min(v)} median {ss.median(v):.0f} max {max(v)}   "
        f"top-10 share {100 * sum(v[-10:]) / len(pairs):.1f}%"
    )

    label = {
        "outcome": "genuine: the ranking never sees these fields",
        "anticipation": (
            "*** TAUTOLOGICAL: this export ranks on newly_reachable. NOT EVIDENCE. "
            "Re-run on the control export (pairs.jsonl, rank_rule=outcome)"
        ),
        "unknown": (
            "*** rank_rule UNKNOWN (no manifest beside the file): cannot tell whether this "
            "is a measurement or a rule-confirmation"
        ),
    }[rank]
    print(f"\nLATENT PURSUIT  ({label})")
    # ask_stop pairs excluded: a STOP retrieves nothing and is `is_latent=False` by
    # construction, so every one of them would read as a discordant win for the question side.
    ask_ask = [p for p in pairs if str(p.get("pair_kind") or "ask_ask") == "ask_ask"]
    if len(ask_ask) != len(pairs):
        print(f"  {len(pairs) - len(ask_ask)} ask_stop pairs excluded from this check")
    d = [
        (p.get("is_latent"), p.get("rejected_is_latent"))
        for p in ask_ask
        if p.get("is_latent") is not None and p.get("rejected_is_latent") is not None
    ]
    c = sum(1 for a, b in d if a and not b)
    r = sum(1 for a, b in d if b and not a)
    if c + r:
        print(f"  discordant {c + r}:  chosen-only {c}  rejected-only {r}")
        print(f"  chosen wins {100 * c / (c + r):.1f}%   p = {one_sided_p(max(c, r), c + r):.2e}")
    nr = [
        (p.get("newly_reachable"), p.get("rejected_newly_reachable"))
        for p in ask_ask
        if p.get("newly_reachable") is not None and p.get("rejected_newly_reachable") is not None
    ]
    if nr:
        # THE KEY'S OWN INPUT, reported separately. On the control export this is the sharper
        # version of the claim above (a trajectory property rather than a retrieval property);
        # on the anticipation export it is the rule restated and means nothing.
        cn = sum(1 for a, b in nr if a and not b)
        rn = sum(1 for a, b in nr if b and not a)
        if cn + rn:
            print(
                f"  newly_reachable discordant {cn + rn}:  chosen-only {cn}  rejected-only "
                f"{rn}  -> {100 * cn / (cn + rn):.1f}%"
                + ("   (by construction under this rank)" if rank == "anticipation" else "")
            )
    cd = [p.get("latent_depth", -1) for p in ask_ask]
    rd = [p.get("rejected_latent_depth", -1) for p in ask_ask]
    print(f"  mean depth: chosen {ss.mean(cd):+.3f}  rejected {ss.mean(rd):+.3f}")

    print("\nSPEED  (mostly tautological: _quicker ranks on this)")
    sp = [
        (p["turns_to_complete"], p["rejected_turns_to_complete"])
        for p in pairs
        if p.get("turns_to_complete", -1) >= 0 and p.get("rejected_turns_to_complete", -1) >= 0
    ]
    f = sum(1 for a, b in sp if a < b)
    s = sum(1 for a, b in sp if a > b)
    if f + s:
        print(f"  comparable {len(sp)}:  faster {f}  slower {s}  tied {len(sp) - f - s}")
        print(f"  chosen faster {100 * f / (f + s):.1f}%")

    print("\nTASK SUCCESS  (tautological: _outcome_prefers ranks on this; <100% means broken)")
    oc = [
        (p["answer_correct"], p["rejected_answer_correct"])
        for p in pairs
        if p.get("answer_correct") == p.get("answer_correct")
        and p.get("rejected_answer_correct") == p.get("rejected_answer_correct")
    ]
    w = sum(1 for a, b in oc if a > b)
    ls = sum(1 for a, b in oc if a < b)
    if w + ls:
        print(
            f"  chosen-correct-only {w}  rejected-correct-only {ls}  -> {100 * w / (w + ls):.1f}%"
        )
        if ls:
            print("  *** RULE VIOLATION: a rejected candidate answered where the chosen did not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/rl/pairs.jsonl"))
