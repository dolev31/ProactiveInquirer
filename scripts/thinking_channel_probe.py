"""Was the thinking channel ON for the base arm on the unverifiable listener?

WHY THIS EXISTS. `artifacts/frontier_strategyqa` ran its 8B base arm through a listener at
:4007 for which NO configuration survives in the repository, while its trained arm ran through
:4000, whose config is committed. That config attaches
`extra_body.chat_template_kwargs.enable_thinking: false` to `qwen3-8b-base` specifically, and its
own comment says a served prompt with the thinking channel on "is a DIFFERENT PROMPT than the one
the adapter was fitted to". So if the :4007 routing table omitted that flag, the base arm emitted
on the thinking channel and the comparator was systematically handicapped.

That provenance question is unresolvable: the launch record is gone. This converts it into a
BEHAVIOURAL one that the run records can answer.

THE SAME MODEL ON TWO LISTENERS. `qwen3-8b-base` as inquirer, at :4007 in the implicit-reasoning
frontier and at :4000 in the external one. The artifact itself records the two pins:
`base8b :4007 -> 5d6fce016f3d` and `base8b :4000 -> e9af49a4b5ff`.

THREE SIGNALS WERE INTENDED AND TWO ARE MEASURABLE.
  1. `tok_reasoning` on the INQUIRER calls, filtered by actor. Non-zero on one listener and zero on
     the other is the flag.
  2. The literal channel in the recorded turn, which catches a provider that emits it without
     counting it. `turns.jsonl` carries `response_text`, so this reads the text and not a counter.
  3. INTENDED AND NOT MEASURABLE: the malformed-action rate. `action_kind` takes exactly one value,
     `ask`, across every population here, so no malformed rate can be computed from these records.
     The probe reports the observed value set rather than printing a zero that would read as an
     absence of malformed actions. A field that exists is not the same as a field that varies.

FILTER BY ACTOR OR THE ANSWER INVERTS. The turn-level `usage_tok_reasoning` is NON-ZERO in every
population, including both base arms, and that is the drafter and answerer rather than the
questioner: they are pinned to a reasoning model and emit on that channel by design. A reading of
the turn-level aggregate would conclude the thinking channel was on everywhere. The per-actor
breakdown below is what separates them, and it is printed rather than summarised.

THE POSITIVE CONTROL IS PART OF THE PROBE. A zero from a counter that is never populated is
indistinguishable from a zero from a counter that is populated and reads zero, so the teacher arm
is measured alongside: its questioner IS a reasoning model, so it must show non-zero, and if it
does not then this instrument is dead and its zeros mean nothing.

THE SUITE IS A CONFOUND ON RATES AND NOT ON SIGNAL 1. The two base populations are different
suites, so any per-call rate could be task difficulty. Whether a model emits on the thinking
channel at all does not depend on the question.

DIRECTION, stated before the numbers so it cannot be chosen afterwards. A handicapped comparator
makes the trained arm's deficit look SMALLER than it is and makes the larger questioner appear to
pull away MORE than it does. Both run in this paper's favour, which is the direction to check
hardest rather than accept most easily.

    python -m scripts.thinking_channel_probe
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

#: The root must be NAMED, not derived from `__file__`. A first pass derived it and reported zero
#: ids in every population, because this file lives in a worktree that does not hold `runs/` or
#: either frontier artifact. That is a clean zero produced by looking in the wrong place, which is
#: the exact reading this probe exists to avoid producing about the data. `--root` is required and
#: the script refuses rather than defaulting, and it prints how many ids it listed so that an empty
#: population can never be mistaken for an empty measurement.
ROOT = Path(".")
RUNS = Path(".")

POPULATIONS = {
    ":4007 base8b (implicit-reasoning, listener UNVERIFIABLE)": (
        "artifacts/frontier_strategyqa",
        "base8b",
    ),
    ":4000 base8b (external, listener config COMMITTED)": ("artifacts/frames_frontier", "base8b"),
    ":4000 trained (implicit-reasoning, same listener as above)": (
        "artifacts/frontier_strategyqa",
        "trained",
    ),
    "POSITIVE CONTROL teacher (gateway, questioner IS a reasoning model)": (
        "artifacts/frontier_strategyqa",
        "teacher",
    ),
}
CAPS = ("cap4", "cap8", "cap12", "cap16", "cap24")
THINK_MARKERS = ("<think>", "</think>", "<|think|>", "<thinking>")


def run_ids(artifact: str, arm: str, limit: int) -> list[str]:
    out: list[str] = []
    for cap in CAPS:
        p = ROOT / artifact / f"run_ids.{arm}.{cap}.tsv"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                out.append(parts[2].strip())
            if len(out) >= limit:
                return out
    return out


def probe(ids: list[str]) -> dict[str, object]:
    calls = reasoning_nonzero = 0
    runs_seen = runs_missing = 0
    think_in_text = actions = 0
    by_actor: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[object] = collections.Counter()
    turn_level_total = 0
    for rid in ids:
        d = RUNS / rid
        if not d.is_dir():
            runs_missing += 1
            continue
        runs_seen += 1
        cp = d / "calls.jsonl"
        if cp.exists():
            for line in cp.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                tr = int(r.get("tok_reasoning") or 0)
                by_actor[str(r.get("actor"))] += tr
                if r.get("actor") != "inquirer":
                    continue
                calls += 1
                reasoning_nonzero += int(tr > 0)
        tp = d / "turns.jsonl"
        if tp.exists():
            for line in tp.read_text().splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                actions += 1
                kinds[t.get("action_kind")] += 1
                turn_level_total += int(t.get("usage_tok_reasoning") or 0)
                if any(m in str(t.get("response_text") or "") for m in THINK_MARKERS):
                    think_in_text += 1
    return {
        "runs_read": runs_seen,
        "runs_missing": runs_missing,
        "INQUIRER calls": calls,
        "INQUIRER calls with reasoning tokens": reasoning_nonzero,
        "reasoning tokens by actor": dict(by_actor),
        "turn-level usage_tok_reasoning (CONFOUNDED, all actors)": turn_level_total,
        "turns read": actions,
        "response_text containing a think marker": think_in_text,
        "action_kind values observed": dict(kinds),
    }


def main(argv: list[str]) -> int:
    global ROOT, RUNS
    if len(argv) < 2:
        print("usage: python -m scripts.thinking_channel_probe <repo root holding runs/>")
        return 2
    ROOT = Path(argv[1]).resolve()
    RUNS = ROOT / "runs"
    if not RUNS.is_dir():
        print(f"REFUSING: {RUNS} is not a directory. A missing runs/ would report zeros.")
        return 2
    print(f"root: {ROOT}")
    limit = 120
    empty = 0
    for label, (artifact, arm) in POPULATIONS.items():
        ids = run_ids(artifact, arm, limit)
        print(f"=== {label}")
        print(f"    ids listed: {len(ids)} (capped at {limit} per population)")
        if not ids:
            empty += 1
            print("    NO IDS: this population was not measured. Not a reading about the data.")
            continue
        for k, v in probe(ids).items():
            print(f"    {k:42s} {v}")
    if empty:
        print(f"\n{empty} population(s) had no ids. Any conclusion drawn over them is unfounded.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
