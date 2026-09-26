"""Build the length-matched ask_ask stratum, as a tolerance LADDER, from the dev pair pool.

WHY THIS SCRIPT EXISTS. `artifacts/tierA_length_decomposition_20260919/RESULT.md` section 4 found
a real content signal in the ask_ask pairs whose two questions are EXACTLY the same length -- but
n=25, and it is the SAME 25 pairs in all 124 stored verdicts (one sample, not 124 replicates).
Exact equality is a needlessly strict control; this script relaxes it into a ladder of tolerances
so a rung can be chosen (or refused) on the n it actually has.

THE LENGTH BASIS IS THE EVALUATOR'S OWN, RECOMPUTED FROM THE ROW, NEVER READ OFF
`PreferencePair.len_delta`. That field on disk is `abs(len(_action_text(chosen)) -
len(_action_text(rejected)))` -- unsigned -- and `pinq_train.eval_offline._len_sign`'s own
docstring exists to warn against reading it as a signed quantity (a mistake made once already,
2026-09-19, in the predecessor analysis). This script re-derives the *signed* delta the same way
`_len_sign` does, importing the exact two functions `pinq_train.eval_offline` imports for the
purpose (`pinq.actions.action_kind_of`, `pinq_train.export.dataset._action_text`), so this and
the evaluator cannot disagree about a row.

THE STOP EXCLUSION IS BY `action_kind_of`, NOT BY THE `pair_kind` STRING, for the identical reason
`pair_accuracy` itself does it that way (see its docstring): `pair_kind` is a string on a jsonl
file that a concatenation, a filter, or a hand-edit could get out of sync with the payload, and a
mislabelled row would put a STOP back into the split that exists to detect length.

WHAT IT WRITES, under `--out-dir` (an artifacts directory, so no `.py` and nothing ruff would
lint lives there):
  - `ask_ask_all.jsonl`        every ask_ask pair (STOP-free), unfiltered by length -- the
                                population the non-vacuity control and the chance test are read
                                against.
  - `tol<N>.jsonl` for each rung in the ladder -- the subset of `ask_ask_all.jsonl` whose
    |signed length delta| <= N.
  - `manifest.json`            source file identity (sha256, row counts), the exclusion counts,
    and the ladder's n at every rung, so `n` in `RESULT.md` is read off a file this script wrote
    and not retyped by hand.

Usage:
    python scripts/length_matched/build_stratum.py \
        --pairs data/rl/dev/pairs.dev.jsonl \
        --out-dir artifacts/length_matched_stratum_20260919/pairs \
        --tolerances 0 2 5 10 20 40 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pinq.actions import action_kind_of
from pinq_train.export.dataset import _action_text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_len_delta(chosen_json: str, rejected_json: str) -> int:
    """`len(_action_text(chosen)) - len(_action_text(rejected))`, signed.

    Identical formula to `pinq_train.eval_offline._len_sign`, kept as an int here (that function
    returns only the three-way bucket name) because the ladder needs the magnitude to threshold
    on, not just the sign.
    """
    return len(_action_text(chosen_json)) - len(_action_text(rejected_json))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True))
            fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="data/rl/dev/pairs.dev.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--tolerances",
        type=int,
        nargs="+",
        default=[0, 2, 5, 10, 20],
        help="|signed length delta| <= N (chars) for each rung",
    )
    args = ap.parse_args()

    rows = load_rows(args.pairs)
    n_total = len(rows)

    ask_ask: list[dict[str, Any]] = []
    n_excluded_stop = 0
    kind_counts: dict[str, int] = {}
    for p in rows:
        chosen_json = str(p["chosen_json"])
        rejected_json = str(p["rejected_json"])
        ck, rk = action_kind_of(chosen_json), action_kind_of(rejected_json)
        kind_counts[f"{ck}/{rk}"] = kind_counts.get(f"{ck}/{rk}", 0) + 1
        if "stop" in (ck, rk):
            n_excluded_stop += 1
            continue
        delta = signed_len_delta(chosen_json, rejected_json)
        q = dict(p)
        q["_signed_len_delta"] = delta  # carried through so the ladder step is pure filtering
        ask_ask.append(q)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.out_dir / "ask_ask_all.jsonl"
    write_jsonl(all_path, ask_ask)

    ladder: dict[str, dict[str, Any]] = {}
    for tol in sorted(set(args.tolerances)):
        subset = [q for q in ask_ask if abs(q["_signed_len_delta"]) <= tol]
        out_path = args.out_dir / f"tol{tol}.jsonl"
        write_jsonl(out_path, subset)
        n_shorter = sum(1 for q in subset if q["_signed_len_delta"] < 0)
        n_longer = sum(1 for q in subset if q["_signed_len_delta"] > 0)
        n_equal = sum(1 for q in subset if q["_signed_len_delta"] == 0)
        ladder[f"tol{tol}"] = {
            "tolerance_chars": tol,
            "n": len(subset),
            "n_chosen_shorter": n_shorter,
            "n_chosen_longer": n_longer,
            "n_equal": n_equal,
            "path": str(out_path),
        }

    manifest = {
        "source_pairs_file": str(args.pairs),
        "source_sha256": sha256_file(args.pairs),
        "n_rows_total": n_total,
        "n_excluded_stop": n_excluded_stop,
        "n_ask_ask": len(ask_ask),
        "kind_pair_counts": kind_counts,
        "ask_ask_all_path": str(all_path),
        "ladder": ladder,
        "length_basis": (
            "len(_action_text(chosen_json)) - len(_action_text(rejected_json)), signed; "
            "identical formula to pinq_train.eval_offline._len_sign, NOT PreferencePair.len_delta "
            "(that field is abs() and carries no sign)"
        ),
        "stop_exclusion_basis": (
            "pinq.actions.action_kind_of(chosen_json/rejected_json) != 'stop' on both sides, "
            "the same test pinq_train.eval_offline.pair_accuracy uses for its ask_ask split -- "
            "not the row's own pair_kind string"
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"source: {args.pairs}  sha256={manifest['source_sha256']}")
    print(f"rows total: {n_total}   excluded (STOP on either side): {n_excluded_stop}")
    print(f"ask_ask (STOP-free): {len(ask_ask)}")
    print("pair_kind(chosen)/pair_kind(rejected) census:")
    for k, v in sorted(kind_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("\nladder:")
    for tol in sorted(set(args.tolerances)):
        e = ladder[f"tol{tol}"]
        print(
            f"  tol<= {tol:>3} chars: n={e['n']:>4}  "
            f"(shorter={e['n_chosen_shorter']}, longer={e['n_chosen_longer']}, equal={e['n_equal']})"
        )
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
