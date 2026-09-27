"""Score one checkpoint's `pair_accuracy` on the length-matched ladder AND its unmatched parent.

WHAT THIS DOES NOT DO. It does not reimplement `pair_accuracy`, `torch_logprob_fn`,
`checkpoint_provenance`, `dataset_provenance`, `tokenizer_source`, or `eval_offline_load_kwargs`.
Every one of those is imported from `pinq_train.eval_offline` / `pi_run.cmd_train` and called
verbatim -- the same functions `pi train eval-offline` calls in `cmd_train_eval_offline`. The
only new code here is (a) a loop over several `--dev-pairs`-shaped files instead of one, and (b)
a memoizing cache in front of `logprob`, because the ladder's rungs are NESTED subsets of
`ask_ask_all.jsonl` and without a cache every pair in `tol30` would be re-scored by the model
once per rung it also belongs to (a pair at delta=0 appears in all seven files).

WHY ONE PROCESS SCORES ALL SEVEN FILES INSTEAD OF SEVEN CALLS TO THE CLI. Loading an 8B-or-larger
adapter onto a GPU is the expensive step (minutes), not evaluating ~1,000 pairs once it is
resident (seconds). Seven `pi train eval-offline` invocations would reload the model seven
times for no reason `pair_accuracy` cares about; this script loads it once.

PROVENANCE (CONTRIBUTING.md rule 1). Every output file's `checkpoint` block is `checkpoint_provenance`
verbatim (identical to what `cmd_train_eval_offline` writes), so `adapter_sha` and, where a
`--registry-name` was passed, `adapter_sha_verified` are present exactly as they would be from
the tracked CLI. Every stratum file's block is `dataset_provenance` verbatim, so `scorer_hash`
and `graph_version` are read off the ROWS the same way `cmd_train_eval_offline` reads them --
never off this script's own say-so.

`--reference-ask` IS DELIBERATELY NOT A FLAG HERE. `stop_confusion` and `sft_nll` (the two Tier-A
metrics that need it) are not computed by this script at all: every `--dev-pairs`-shaped file it
scores is the ask_ask-only stratum built by `build_stratum.py`, which excludes any pair with a
STOP on either side by `action_kind_of` (see that script's docstring) -- there is no STOP row in
any file this script reads, so a reference ASK to score STOP rows against would have nothing to
apply to. This is the opposite failure mode from the trap named in the task ("a missing
--reference-ask reads as a model finding"): here the flag is inapplicable by construction, not
silently omitted from a call that needed it.

Usage (run where transformers/torch resolve -- inside an LSF job, after the torch
device/dtype defaults are set, exactly as scripts/hpc/eval_offline.sh does it):

    python scripts/length_matched/score_checkpoint.py \
        --checkpoint /proj/pinq/user/artifacts/E39-8b/s0/checkpoint-707 \
        --pairs-dir artifacts/length_matched_stratum_20260919/pairs \
        --out /proj/pinq/user/artifacts/length_matched_stratum_20260919/eval/E39-8b.s0.ckpt-707.json \
        --label E39-8b.s0.ckpt-707 \
        [--registry-name qwen3-8b-dpo-headline-control] \
        [--registry conf/checkpoints.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pinq_train.eval_offline as eo
from pi_run.cmd_train import eval_offline_load_kwargs, tokenizer_source
from pi_run.manifest import repo_root


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _memoizing_logprob(logprob_fn: Any, render_fn: Any) -> Any:
    """Wrap `render`+`logprob` in a cache keyed on `(state_text, action_json)`.

    `pair_accuracy` calls `_score(render, logprob, state, action)` twice per pair (once for
    `chosen_json`, once for `rejected_json`) and never caches across pairs itself -- it is a
    pure per-call function by design (module docstring: "the tokenizer and the weights enter
    only through those two [callables]"), so caching belongs in the caller, not in a change to
    it. The key is the exact pair `_score` uses; the cache never sees a model weight or a
    tokenizer id, only the two strings already in the row.
    """
    cache: dict[tuple[str, str], float] = {}
    calls = {"total": 0, "hits": 0}
    # Captured NOW, before the caller installs `cached_score` as `eo._score`. A lookup of
    # `eo._score` done INSIDE `cached_score` at call time would resolve to `cached_score`
    # itself once that monkeypatch is live -- infinite recursion on every cache miss, caught
    # by `test_memoizing_cache.py` property 1 (RecursionError) before this ever ran against a
    # real checkpoint.
    real_score = eo._score

    def cached_score(render: Any, logprob: Any, state: str, action: str) -> tuple[float, int]:
        key = (state, action)
        calls["total"] += 1
        if key in cache:
            calls["hits"] += 1
            return cache[key], 0
        lp, n_tok = real_score(render, logprob, state, action)
        cache[key] = lp
        return lp, n_tok

    return cached_score, calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--pairs-dir",
        required=True,
        type=Path,
        help="artifacts/length_matched_stratum_20260919/pairs (build_stratum.py's --out-dir)",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", required=True, help="a name for this checkpoint in the output")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--registry-name", default=None)
    ap.add_argument("--chat-template", action="store_true", default=True)
    ap.add_argument("--no-chat-template", dest="chat_template", action="store_false")
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    args = ap.parse_args()

    manifest_path = args.pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ladder = manifest["ladder"]

    files: dict[str, Path] = {"unmatched_ask_ask_all": args.pairs_dir / "ask_ask_all.jsonl"}
    for rung_name, rung in sorted(ladder.items(), key=lambda kv: kv[1]["tolerance_chars"]):
        files[rung_name] = args.pairs_dir / f"{rung_name}.jsonl"

    ckpt = args.checkpoint
    if not ckpt.exists():
        print(f"--checkpoint {ckpt} does not exist", file=sys.stderr)
        return 2

    root = repo_root()
    registry_path = args.registry or (root / "conf" / "checkpoints.json")
    try:
        ckpt_prov = eo.checkpoint_provenance(
            ckpt, registry_path=registry_path, registry_name=args.registry_name
        )
    except eo.AdapterIdentityMismatch as exc:
        print(f"REFUSED before loading weights: {exc}", file=sys.stderr)
        return 2

    import transformers as tr

    tok_src = tokenizer_source(ckpt)
    tok = tr.AutoTokenizer.from_pretrained(tok_src)
    model = tr.AutoModelForCausalLM.from_pretrained(str(ckpt), **eval_offline_load_kwargs()).eval()
    render, raw_logprob = eo.torch_logprob_fn(
        model, tok, chat_template=args.chat_template, enable_thinking=args.enable_thinking
    )

    # Monkeypatch-free memoization: `pair_accuracy` calls the module-level `_score`, so the cache
    # has to sit in front of `_score` itself for this process's calls to hit it. We do that by
    # temporarily replacing `eo._score` with the memoizing wrapper -- restored in `finally`, and
    # the wrapper's fallback path calls the ORIGINAL `eo._score`, so `pair_accuracy`'s own logic
    # (margins, by_kind, by_decided, by_len_sign, the STOP exclusion) is untouched; only which
    # function computes a given (state, action)'s log-prob the second time it is asked is new.
    cached_score, calls = _memoizing_logprob(raw_logprob, render)
    orig_score = eo._score
    eo._score = cached_score  # type: ignore[attr-defined]

    results: dict[str, Any] = {}
    try:
        for name, path in files.items():
            rows = _read_jsonl(path)
            pa = eo.pair_accuracy(rows, render=render, logprob=raw_logprob)
            results[name] = {
                "pair_accuracy": pa,
                "dataset_provenance": eo.dataset_provenance(path, rows),
                "n_rows_in_file": len(rows),
            }
    finally:
        eo._score = orig_score  # type: ignore[attr-defined]

    out = {
        "label": args.label,
        "checkpoint": str(ckpt),
        "tokenizer": tok_src,
        "chat_template": bool(args.chat_template),
        "enable_thinking": bool(args.enable_thinking),
        "checkpoint_provenance": ckpt_prov,
        "stratum_manifest_path": str(manifest_path),
        "stratum_manifest_source_sha256": manifest["source_sha256"],
        "cache_stats": calls,
        "results": results,
        "note_reference_ask": (
            "not applicable: every file scored here is ask_ask-only (STOP excluded by "
            "action_kind_of in build_stratum.py), so no STOP row exists to score against a "
            "reference ASK; stop_confusion and sft_nll are not computed by this script"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2, sort_keys=True)
    args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
