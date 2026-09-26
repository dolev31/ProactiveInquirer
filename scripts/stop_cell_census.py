"""Census of the gate stop 2x2 cells on disk, and the identity that makes one caption useless.

WHY THIS EXISTS. A proposal was made to print `n_done_ask` beside any stop cell reading 1.0, on
the belief that such a reading might be structurally pinned. `pinq_train.gate._stop_2x2` computes

    p_stop_given_done = stop_done / n_done        n_done_ask = n_done - stop_done

so the rate is 1.0 if and only if `n_done_ask` is 0. The two are one statement written twice, and
the caption would print `0` on every cell that could ever carry it. This script measures that
against the verdicts rather than asserting it, and measures the two things that are NOT entailed:
how far the statistic actually ranges, and whether the cells at the ceiling are the small ones.

IT PRINTS MEASUREMENTS AND NOT A VERDICT. Every line below is a count or a range read off the
files. A script that decided "artefact" or "clean" from one branch would be asserting the thing
it was built to test.

IT ALSO PRINTS WHAT IT MATCHED, not only what it found, and that is not decoration. Two separate
checks against this same artifact tree returned a clean zero for reasons that had nothing to do
with the data: one filtered on the wrong key path and read a per-row analysis table as though it
were the offline block, and its replacement filtered `isinstance(n_done, int)` when Tier A writes
those counts as floats, so a correct key path still matched nothing. Both answers looked like
findings about the corpus. A scan that reports "examined N, matched M" distinguishes a true
absence from an instrument that never looked, and neither of those failures would have survived
one line of that kind.

ROOTS. `artifacts/` is per-worktree and the shared checkout has its own, so both are walked and
the union is deduplicated by resolved path. A run from either tree sees the same population.

WHY THE DEDUPE IS LOAD-BEARING AND NOT HYGIENE. `artifacts/gate/8b2-t19c` holds 22 symlinks and
`artifacts/gate/8b1-rescored-t19c` holds 2, each mirroring a file in a sibling directory. A glob
therefore reaches 209 paths for 185 physical verdicts in the shared checkout, and the inflation is
exactly the 24 links. Counting by path is not careless, it answers a different question, but only
the resolved count is a count of verdicts. One cell reading 1.0 sits inside a duplicated file, so
a path-wise count of the ceiling cells reads 10 where a resolved count reads 9. The worktree tree
adds 12 further physical files that the shared tree does not hold, which is where 197 comes from.

    .venv/bin/python -m scripts.stop_cell_census [extra_artifacts_root ...]
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator


def artifacts_roots(extra: list[str]) -> list[Path]:
    """This worktree's `artifacts/`, the shared checkout's, and anything named on the command line."""
    roots: list[Path] = []
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        roots.append(Path(top) / "artifacts")
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        roots.append(Path(common).resolve().parent / "artifacts")
    except (subprocess.CalledProcessError, OSError):
        pass
    roots.extend(Path(e) for e in extra)
    out: list[Path] = []
    for r in roots:
        if r.is_dir() and r.resolve() not in {o.resolve() for o in out}:
            out.append(r)
    return out


def stop_cells(
    roots: list[Path], examined: list[int] | None = None
) -> Iterator[tuple[Path, str, dict[str, Any]]]:
    """Every `criteria.stop_2x2.{value,baseline}` block under the roots, each file read once.

    `examined` is an out-parameter for the denominator of this scan: how many distinct JSON files
    were opened at all. A caller that prints only the matches cannot tell a corpus with no such
    blocks from a filter that matched none of them, and this scan has been wrong that way twice.
    """
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            if examined is not None:
                examined[0] += 1
            try:
                doc = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not isinstance(doc, dict):
                continue
            criteria = doc.get("criteria")
            block = criteria.get("stop_2x2") if isinstance(criteria, dict) else None
            if not isinstance(block, dict):
                continue
            for side in ("value", "baseline"):
                cell = block.get(side)
                if isinstance(cell, dict) and cell.get("p_stop_given_done") is not None:
                    yield path, side, cell


def main(argv: list[str]) -> int:
    roots = artifacts_roots(argv[1:])
    if not roots:
        print("no artifacts root found: pass one on the command line", file=sys.stderr)
        return 2
    for r in roots:
        print(f"root: {r}")
    examined = [0]
    cells = list(stop_cells(roots, examined))
    n_seen = examined[0]
    if not cells:
        print("no stop_2x2 cells found under those roots", file=sys.stderr)
        return 2

    files = {p.resolve() for p, _, _ in cells}
    print(f"json files examined             {n_seen}")
    print(f"verdict files with a stop_2x2   {len(files)}  (distinct physical, symlinks resolved)")
    print(f"cells                           {len(cells)}")

    # NaN IS A READING HERE, NOT A GAP, and it must never reach a sort. `_stop_2x2` returns NaN
    # for a cell whose denominator is zero, so a NaN `p_ask_given_not_done` says the arm met no
    # not-done state at all. Python orders NaN by comparisons that are all False, so `sorted()`
    # on a list holding one returns an arrangement that is not sorted and whose first element is
    # not the minimum. An earlier reading of this same population reported a low of 0.5829 that
    # way. Both cells are therefore split into finite values and a counted NaN population.
    stop_all = [float(c["p_stop_given_done"]) for _, _, c in cells]
    ask_all = [
        float(c["p_ask_given_not_done"])
        for _, _, c in cells
        if c.get("p_ask_given_not_done") is not None
    ]
    stop = [v for v in stop_all if not math.isnan(v)]
    ask = [v for v in ask_all if not math.isnan(v)]
    done = [int(c["n_done"]) for _, _, c in cells if c.get("n_done") is not None]
    print(f"p_stop_given_done NaN            {len(stop_all) - len(stop)}  (n_done == 0)")
    print(f"p_ask_given_not_done NaN         {len(ask_all) - len(ask)}  (n_not_done == 0)")
    print(f"distinct finite p_stop_given_done {len(set(stop))}")
    print(f"p_stop_given_done range          {min(stop):.4f} - {max(stop):.4f}  (n={len(stop)})")
    print(f"p_ask_given_not_done range       {min(ask):.4f} - {max(ask):.4f}  (n={len(ask)})")
    print(
        f"n_done range                     {min(done)} - {max(done)}, median {int(statistics.median(done))}"
    )

    # POOLED IS THE WRONG UNIT FOR THIS ONE, and a pooled range hid the fact once. The two sides
    # of a verdict are a trained checkpoint and the prompted policy it is measured against, and
    # they occupy disjoint parts of the range. Reporting `0.0000 - 1.0000` over both together
    # describes the contrast rather than either arm, and it is what let a real baseline-side span
    # be mistaken for a value that did not exist.
    for key, label in (("value", "checkpoint"), ("baseline", "baseline")):
        side = [float(c["p_stop_given_done"]) for _, s, c in cells if s == key]
        side = [v for v in side if not math.isnan(v)]
        if side:
            print(
                f"  p_stop, {label:10s} n={len(side):4d}  {min(side):.4f} - {max(side):.4f}"
                f"  exactly 1.0: {side.count(1.0)}"
            )

    # THE IDENTITY, tested both ways rather than stated.
    ones = [c for _, _, c in cells if float(c["p_stop_given_done"]) == 1.0]
    zero_ask = [c for _, _, c in cells if c.get("n_done_ask") == 0]
    print(f"cells reading exactly 1.0        {len(ones)}")
    print(
        f"  of those, n_done_ask != 0      {sum(1 for c in ones if c.get('n_done_ask') not in (0, None))}"
    )
    print(f"cells with n_done_ask == 0       {len(zero_ask)}")
    print(
        f"  of those, rate != 1.0          {sum(1 for c in zero_ask if float(c['p_stop_given_done']) != 1.0)}"
    )
    # CHECKED ONLY WHERE THE RATE IS FINITE. A NaN fails `> 1e-9` and would be counted as
    # agreement, which is the same trap in a second place.
    checkable = [
        c
        for _, _, c in cells
        if c.get("n_done")
        and c.get("n_done_stop") is not None
        and not math.isnan(float(c["p_stop_given_done"]))
    ]
    off = sum(
        1
        for c in checkable
        if abs(float(c["p_stop_given_done"]) - c["n_done_stop"] / c["n_done"]) > 1e-9
    )
    print(f"cells where rate != stop/done    {off}  (of {len(checkable)} checkable)")

    # WHETHER THE CEILING CELLS ARE THE SMALL ONES. Not entailed, so it is measured.
    if ones:
        band_lo = min(int(c["n_done"]) for c in ones)
        band_hi = max(int(c["n_done"]) for c in ones)
        inside = [
            c
            for _, _, c in cells
            if float(c["p_stop_given_done"]) != 1.0
            and c.get("n_done")
            and band_lo <= int(c["n_done"]) <= band_hi
        ]
        q25 = statistics.quantiles(done, n=4)[0]
        print(f"n_done on the 1.0 cells          {band_lo} - {band_hi}")
        print(f"  the distribution's q25         {q25:.0f}")
        print(f"  all of them at or below q25    {all(int(c['n_done']) <= q25 for c in ones)}")
        print(f"non-1.0 cells inside that band   {len(inside)}")
        if inside:
            print(
                f"  lowest rate among them         {min(float(c['p_stop_given_done']) for c in inside):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
