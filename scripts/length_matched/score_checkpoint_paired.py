"""Score one checkpoint's per-PAIR ordering decision on the length-matched ladder, keyed by
`pair_id`, so two checkpoints' outputs can be joined into a genuine PAIRED contrast.

WHY THIS EXISTS ALONGSIDE `score_checkpoint.py`. That script calls `pinq_train.eval_offline.
pair_accuracy`, which returns only aggregates (`acc`, `acc_by_kind`, `acc_by_len_sign`,
`len_sign_gap`, `mean_margin`, `n`) -- see its `return` statement. Two checkpoints' aggregate
accuracies on the SAME 970-pair rung are two independent-looking numbers; the paper's own
Newcombe interval already treats them that way, and the coordinator flagged this as the wrong
test for a design where both checkpoints are scored on the IDENTICAL pairs (a paired/correlated
structure, not two independent samples). A paired test (McNemar on discordant pairs, or a paired
bootstrap over the per-pair difference) needs to know, PER PAIR, whether each checkpoint got it
right -- data `pair_accuracy` computes internally (`ok = int(chosen_lp > rejected_lp)`, per pair,
in its loop) and then discards. This script keeps it, keyed by the row's own `pair_id` (never by
file position -- `build_stratum.py`'s ladder rungs need not enumerate pairs in the same relative
order across files, and joining by an explicit id is the only join that cannot silently misalign
two checkpoints' vectors).

WHAT THIS DOES NOT DO. It does not reimplement rendering, log-prob, or model loading: `render`,
`raw_logprob`, and `checkpoint_provenance` are the exact same calls `score_checkpoint.py` makes,
imported from `pinq_train.eval_offline` / `pi_run.cmd_train`. The only new logic is the loop
that records `ok`/`margin` per `pair_id` instead of folding them into a running mean, plus the
same memoizing cache `score_checkpoint.py` uses (a pair at delta=0 appears in every ladder rung
nested inside `ask_ask_all.jsonl`, so without a cache it would be re-scored once per rung).

It ALSO recomputes the aggregate `pair_accuracy` dict per rung (calling `eo.pair_accuracy`
verbatim, sharing the same cache), so this file's aggregate numbers are checkable against
`score_checkpoint.py`'s independently -- if the two disagree on the SAME checkpoint and rung,
that is a bug in one of the two loops, not a result.

Usage (same environment/torch-defaults contract as `hpc_score.sh`; run under an LSF job after
`torch.set_default_device('cuda')` / `set_default_dtype(torch.bfloat16)`):

    python scripts/length_matched/score_checkpoint_paired.py \
        --checkpoint /proj/pinq/user/artifacts/rung1-8b-headline \
        --pairs-dir artifacts/length_matched_stratum_20260919/pairs \
        --out /proj/pinq/user/artifacts/length_matched_stratum_20260919/eval_paired/qwen3-8b-sft-headline.paired.json \
        --label qwen3-8b-sft-headline \
        --registry-name qwen3-8b-sft-headline --registry conf/checkpoints.json
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


def score_rows_paired(
    rows: list[dict[str, Any]], *, render: Any, logprob: Any, cache: dict[tuple[str, str], float]
) -> dict[str, Any]:
    """Per-pair ok/margin keyed by `pair_id`, plus the same aggregate `eo.pair_accuracy` computes.

    `cache` is shared across calls (across rungs) by the caller, keyed on `(state_text,
    action_json)` exactly as `score_checkpoint.py`'s memoizing wrapper does -- the identical key
    `pair_accuracy`'s own `_score` calls use, so a cache hit here is provably the same (state,
    action) pair `pair_accuracy` would have scored, not a coincidentally-matching string.
    """

    def scored(state: str, action: str) -> float:
        key = (state, action)
        if key not in cache:
            r = render(state, action)
            lp = float(logprob(r.prompt_ids, r.completion_ids))
            cache[key] = lp
        return cache[key]

    per_pair: dict[str, dict[str, Any]] = {}
    n = 0
    n_correct = 0
    for p in rows:
        pid = str(p["pair_id"])
        state = str(p["state_text"])
        chosen_json = str(p["chosen_json"])
        rejected_json = str(p["rejected_json"])
        chosen_lp = scored(state, chosen_json)
        rejected_lp = scored(state, rejected_json)
        ok = int(chosen_lp > rejected_lp)
        margin = chosen_lp - rejected_lp
        if pid in per_pair:
            raise SystemExit(f"duplicate pair_id within one rung file: {pid!r}")
        per_pair[pid] = {"ok": ok, "margin": margin}
        n += 1
        n_correct += ok

    return {
        "per_pair": per_pair,
        "n": n,
        "acc": (n_correct / n) if n else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--pairs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", required=True)
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

    cache: dict[tuple[str, str], float] = {}
    results: dict[str, Any] = {}
    for name, path in files.items():
        rows = _read_jsonl(path)
        paired = score_rows_paired(rows, render=render, logprob=raw_logprob, cache=cache)
        # Cross-check against eo.pair_accuracy's own aggregate on the same rows, sharing the
        # cache so this costs no extra forward passes. A disagreement here is a bug, not noise.
        eo_orig_score = eo._score
        try:

            def _cached_score(
                render_fn: Any, logprob_fn: Any, state: str, action: str
            ) -> tuple[float, int]:
                key = (state, action)
                if key in cache:
                    return cache[key], 0
                lp, n_tok = eo_orig_score(render_fn, logprob_fn, state, action)
                cache[key] = lp
                return lp, n_tok

            eo._score = _cached_score  # type: ignore[attr-defined]
            pa = eo.pair_accuracy(rows, render=render, logprob=raw_logprob)
        finally:
            eo._score = eo_orig_score  # type: ignore[attr-defined]

        if pa["n"] == paired["n"] and abs(pa["acc"] - paired["acc"]) > 1e-9:
            raise SystemExit(
                f"aggregate mismatch on {name}: eo.pair_accuracy acc={pa['acc']!r} vs "
                f"this script's paired acc={paired['acc']!r} at equal n={pa['n']} -- refusing "
                "to write a result that disagrees with itself"
            )

        results[name] = {
            "pair_accuracy_crosscheck": pa,
            "paired": paired,
            "n_rows_in_file": len(rows),
        }

    out = {
        "label": args.label,
        "checkpoint": str(ckpt),
        "tokenizer": tok_src,
        "chat_template": bool(args.chat_template),
        "enable_thinking": bool(args.enable_thinking),
        "checkpoint_provenance": ckpt_prov,
        "stratum_manifest_path": str(manifest_path),
        "stratum_manifest_source_sha256": manifest["source_sha256"],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2, sort_keys=True)
    args.out.write_text(text + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
