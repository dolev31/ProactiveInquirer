#!/usr/bin/env python3
"""Lane L0.9: can a rehydrated run's `answer_n_words` be re-derived from its cached text?

`scripts/restore_corpus.py` writes `outcome.json["answer"]["text"]` from the answerer call's
RAW cached response (`cache[request_sha]["text"]`) but `["n_words"]` from this same run's
ORIGINAL `answer_n_words` in `runs.parquet` -- the count AFTER `FrozenLLMAnswerer.answer`
(`pinq_expt/components.py`) applies `capped = " ".join(text.split()[:word_cap])`. The two
fields can disagree (measured: 786 of 22,605 rehydrated runs,
`artifacts/rehydrated_answers_20260918/RESULT.md`).

`recount_answer` reproduces that ONE transformation -- nothing else touches `answer.text`
between the answerer call and `n_words` on the ORIGINAL (non-rehydrated) path, so it is the
only candidate for a cheap, correct re-derivation. It is not guaranteed to reproduce every
row: the response cache is first-writer-wins under a race (see
`docs/...`/`src/pi_run/cache.py`'s own docstring, "the ledger records the discarded sample"),
so the text a rehydrated run's cache entry holds today is not always the exact sample that run
actually received, and capping cannot recover a sample that was never cached. This script
measures the reproduction rate rather than assuming it; it never rewrites a run directory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def recount_answer(text: str, word_cap: int) -> int:
    """Exactly `FrozenLLMAnswerer.answer`'s own capping (`pinq_expt/components.py`):
    `capped = " ".join(text.split()[:word_cap])`, then `len(capped.split())`. Written the same
    way rather than as `min(len(text.split()), word_cap)` so a future change to the answerer's
    capping (e.g. capping on characters, or trimming punctuation first) has exactly one
    definition to change, here and there both."""
    if word_cap <= 0:
        return 0
    capped = " ".join(text.split()[:word_cap])
    return len(capped.split())


def check_run(run_dir: Path) -> dict | None:
    """None if `run_dir` is not a rehydrated run with a recorded answer -- there is nothing to
    re-derive for a normal run, and a cache miss at rehydration time already recorded no
    answer at all (`restore_corpus.py`'s `no_answer_text`)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if not manifest.get("rehydrated_from"):
        return None
    outcome_path = run_dir / "outcome.json"
    if not outcome_path.exists():
        return None
    outcome = json.loads(outcome_path.read_text())
    answer = outcome.get("answer")
    if answer is None:
        return None
    word_cap = int(manifest.get("word_cap") or 0)
    text = str(answer.get("text") or "")
    stored = int(answer.get("n_words") or 0)
    raw_recount = len(text.split())
    recomputed = recount_answer(text, word_cap)
    return {
        "run_id": str(manifest.get("run_id", run_dir.name)),
        "word_cap": word_cap,
        "stored_n_words": stored,
        "raw_recount": raw_recount,
        "recomputed_n_words": recomputed,
        "contradicts": stored != raw_recount,
        "reproduces": recomputed == stored,
    }


def scan(runs_root: Path, only_contradicting: bool) -> list[dict]:
    out = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            r = check_run(d)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        if r is None:
            continue
        if only_contradicting and not r["contradicts"]:
            continue
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument(
        "--only-contradicting",
        action="store_true",
        help="restrict to runs where stored n_words != len(text.split()) -- the population "
        "this script exists to re-derive; omit to also see agreeing rows (reproduces trivially)",
    )
    p.add_argument("--sample", type=int, default=0, help="0 = every matching run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    results = scan(args.runs_root, args.only_contradicting)
    if args.sample and len(results) > args.sample:
        results = random.Random(args.seed).sample(results, args.sample)

    n = len(results)
    n_ok = sum(1 for r in results if r["reproduces"])
    summary = {
        "runs_root": str(args.runs_root),
        "only_contradicting": args.only_contradicting,
        "n_sampled": n,
        "n_reproduced": n_ok,
        "reproduction_rate": (n_ok / n) if n else None,
    }
    print(json.dumps(summary, indent=1))
    for r in results:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
