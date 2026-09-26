#!/usr/bin/env python3
"""The dev slices the per-checkpoint Tier-A evaluator scores.

    .venv/bin/python scripts/dev_samples.py            # -> data/rl/dev/*.sample.jsonl

WHY A SAMPLE. `pi train eval-offline` has no `--limit`. Over the full dev files it is
6,975 x 3 + 4,962 x 2 = 30,849 forward passes of an 8B model for ONE checkpoint, and rung 1
writes a checkpoint every 200 optimizer steps (~45 min at the measured 13 s/step) while
`save_total_limit=2` deletes the older one. An evaluation that cannot finish inside that
window is not slow, it is impossible: the checkpoint it was scoring is gone. These slices are
1,000 SFT rows and 1,485 pairs -- 5,970 forwards -- which fits.

WHAT IS SAMPLED AND WHAT IS NOT, because the two halves of the pair file are not the same
measurement:

  * ask_ask (970) and ask_stop (115) are kept WHOLE. `acc_by_kind['ask_ask']` is the only cell
    of the pair accuracy not decided by the ~60-character ask-vs-stop length asymmetry, so it
    is the cell a training curve is read off; sampling it would add a sampling error to the y
    axis on top of the checkpoint-to-checkpoint movement the figure exists to show. 115
    ask_stop pairs is already single-digits-per-bin territory.
  * ask_stop_synth (3,877) IS sampled, to 400. It is synthetic by construction and its cell is
    the one the length asymmetry decides; 400 pins that cell to +-2.5 points, which is finer
    than any movement it will be asked to support.
  * The SFT rows are sampled STRATIFIED BY `label_rule`, which in this corpus is the same
    partition as `is_stop` and as `done_before` (2,444 ask_clears_floor / 4,531 stop_done, with
    no row crossing). `nll_per_token` pools both, and a STOP action is ~7 tokens against tens
    for an ASK, so a sample that drifted toward STOP would lower the pooled NLL without the
    checkpoint having changed.

DETERMINISM IS THE POINT, not a nicety. Every point on the dev curve must be computed over the
SAME rows or the curve is not a curve, and the file that defines those rows is rebuilt by
whoever next runs this. Selection is by `random.Random(0)` over the source file's own order,
the output keeps that order, and the source LINE is copied verbatim rather than re-serialised,
so a re-run is byte-identical and a sampled row is provably a source row.

REFUSES A NON-DEV ROW, AND WRITES NOTHING WHEN IT DOES. `pi_run.cmd_train._dev_rows_or_refuse`
already refuses one, but it refuses on the cluster, hours later, when a GPU is already held.
A train row here would select the checkpoint on its own training set; a test row would burn the
held-out split before it is read once.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]

DEV = REPO / "data/rl/dev"
SFT_IN = DEV / "sft.dev.jsonl"
PAIRS_IN = DEV / "pairs.dev.jsonl"
SFT_OUT = "sft.dev.sample.jsonl"
PAIRS_OUT = "pairs.dev.sample.jsonl"

SEED = 0
N_SFT = 1000
N_SYNTH = 400
KEEP_WHOLE = ("ask_ask", "ask_stop")
SAMPLED_KIND = "ask_stop_synth"


class Refused(RuntimeError):
    """A refusal, never a filter. Dropping the offending row is how a train row gets scored."""


Row = tuple[str, dict]  # the source line, verbatim, and its parse


def read_dev(path: Path, label: str) -> list[Row]:
    """Every line of `path`, refusing the whole file on the first row that is not `split=dev`."""
    if not path.exists():
        raise Refused(f"{label}: {path} does not exist")
    out: list[Row] = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split", ""))
        if split != "dev":
            raise Refused(
                f"{label}: {path} row {i} has split={split!r}, not 'dev'. Refusing the whole "
                "file: these rows select a checkpoint, so a train row would select on the "
                "training set and a test row would burn the held-out split before it is read "
                "once. Dropping the row instead would make that invisible."
            )
        out.append((line, row))
    if not out:
        raise Refused(f"{label}: {path} holds no rows")
    return out


def allocate(sizes: dict[str, int], k: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of `k` over strata, summing to exactly `k`.

    Ties on the fractional part break by stratum name, so the allocation is a function of the
    counts alone -- no dict ordering, no insertion order, no seed.
    """
    total = sum(sizes.values())
    if total <= k:
        return dict(sizes)
    exact = {s: n * k / total for s, n in sizes.items()}
    floors = {s: int(v) for s, v in exact.items()}
    short = k - sum(floors.values())
    order = sorted(sizes, key=lambda s: (-(exact[s] - floors[s]), s))
    for s in order[:short]:
        floors[s] += 1
    return {s: min(floors[s], sizes[s]) for s in sizes}


def stratified(rows: Sequence[Row], key: str, k: int, seed: int) -> list[Row]:
    """`k` rows, proportional by `row[key]`, selected by index and returned in source order."""
    strata: dict[str, list[int]] = {}
    for i, (_, row) in enumerate(rows):
        strata.setdefault(str(row.get(key, "")), []).append(i)
    quota = allocate({s: len(v) for s, v in strata.items()}, k)
    rng = random.Random(seed)
    picked: set[int] = set()
    for s in sorted(strata):  # sorted: the draw must not depend on which stratum was seen first
        picked.update(rng.sample(strata[s], quota[s]))
    return [rows[i] for i in sorted(picked)]


def take(rows: Sequence[Row], kind: str, k: int, seed: int) -> list[Row]:
    """`k` rows of one `pair_kind`, in source order."""
    idx = [i for i, (_, r) in enumerate(rows) if str(r.get("pair_kind", "")) == kind]
    if len(idx) <= k:
        return [rows[i] for i in idx]
    return [rows[i] for i in sorted(random.Random(seed).sample(idx, k))]


def write(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line, _ in rows))


def counts(rows: Sequence[Row], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, r in rows:
        out[str(r.get(key, ""))] = out.get(str(r.get(key, "")), 0) + 1
    return dict(sorted(out.items()))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sft", default=str(SFT_IN))
    ap.add_argument("--pairs", default=str(PAIRS_IN))
    ap.add_argument("--out-dir", default=str(DEV))
    ap.add_argument("--n-sft", type=int, default=N_SFT)
    ap.add_argument("--n-synth", type=int, default=N_SYNTH)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    # BOTH FILES ARE READ AND VALIDATED BEFORE EITHER IS WRITTEN. A refusal that had already
    # written the SFT sample would leave a half-built pair of files whose two halves came from
    # different runs, and nothing downstream reads them as a pair.
    try:
        sft = read_dev(Path(a.sft), "--sft")
        pairs = read_dev(Path(a.pairs), "--pairs")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    sft_sample = stratified(sft, "label_rule", a.n_sft, a.seed)
    whole = [r for r in pairs if str(r[1].get("pair_kind", "")) in KEEP_WHOLE]
    synth = take(pairs, SAMPLED_KIND, a.n_synth, a.seed)
    keep = {id(r) for r in whole} | {id(r) for r in synth}
    pair_sample = [r for r in pairs if id(r) in keep]  # source order, both populations merged

    out = Path(a.out_dir)
    write(out / SFT_OUT, sft_sample)
    write(out / PAIRS_OUT, pair_sample)

    src_sft, src_pairs = counts(sft, "label_rule"), counts(pairs, "pair_kind")
    got_sft, got_pairs = counts(sft_sample, "label_rule"), counts(pair_sample, "pair_kind")
    print(f"seed {a.seed}")
    print(f"{out / SFT_OUT}: {len(sft_sample)} of {len(sft)} rows, stratified by label_rule")
    for s in sorted(src_sft):
        share = got_sft.get(s, 0) / len(sft_sample)
        print(f"  {s:<20} {got_sft.get(s, 0):>5} of {src_sft[s]:>5}   {share:.4f} of the sample")
    print(f"{out / PAIRS_OUT}: {len(pair_sample)} of {len(pairs)} pairs")
    for s in sorted(src_pairs):
        whole_here = "whole" if s in KEEP_WHOLE else "sampled"
        print(f"  {s:<20} {got_pairs.get(s, 0):>5} of {src_pairs[s]:>5}   ({whole_here})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
