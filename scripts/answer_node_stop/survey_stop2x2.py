"""Is the gate's `P(STOP | done)` capable of anything other than 1.0? Read every verdict on disk.

THE OBSERVATIONAL HALF of the degeneracy question. The forcing half is
`tests/test_stop_2x2_is_not_degenerate.py`, which constructs a done state the policy did not
stop at and asserts the cell moves off 1.0. This one asks what the cell has ACTUALLY done across
every verdict the repository holds, because "it could move" and "it ever did" are two claims and
a statistic that is free to move and never has is still not carrying information.

`n_done_ask` IS THE COLUMN THAT DECIDES IT. It counts done states at which the policy asked, and
it is the only thing that can pull the ratio below 1.0. A cell reading 1.0 WITH `n_done_ask > 0`
would be a real measurement; a cell reading 1.0 with `n_done_ask == 0` had no room to move on
that population, which is a fact about those runs rather than about the statistic.

THE SIDE KEYS ARE READ FROM THE BLOCK, NOT GUESSED. A first pass looked for a key named
"checkpoint", found none, and printed "0 checkpoint cells" -- a guessed field name returning
absence in the shape of a finding about the data. The checkpoint side lives under `value`, the
gate's generic criterion key, and the constant below is where that is written down.

USAGE:  python -m scripts.answer_node_stop.survey_stop2x2 [--root <repo>]
Read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

#: verdict key -> the side it holds. `value` is the gate's generic key for the criterion's own
#: measurement, which is the CHECKPOINT; `baseline` is spelled out.
SIDES = {"value": "checkpoint", "baseline": "baseline"}


def repo_root() -> Path:
    """This file is `<repo>/scripts/answer_node_stop/survey_stop2x2.py`.

    Derived from `__file__` rather than written as a literal: `scripts/check_no_home_paths.sh`
    refuses a committed absolute home path, and it refused an earlier draft of this file for
    exactly that.
    """
    return Path(__file__).resolve().parents[2]


def collect(root: Path) -> list[tuple]:
    rows = []
    for p in sorted(glob.glob(str(root / "artifacts" / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        s = (d.get("criteria") or {}).get("stop_2x2") or d.get("stop_2x2")
        if not isinstance(s, dict):
            continue
        for key, side in SIDES.items():
            cell = s.get(key)
            if not isinstance(cell, dict) or "p_stop_given_done" not in cell:
                continue
            rows.append(
                (
                    Path(p).name,
                    side,
                    cell["p_stop_given_done"],
                    cell.get("n_done"),
                    cell.get("n_done_ask"),
                    cell.get("n_done_stop"),
                    cell.get("n_forced_stops"),
                    cell.get("mean_asks_after_done"),
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="repo root; default: derived from this file")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else repo_root()

    rows = collect(root)
    print(f"verdict files: {len({r[0] for r in rows})}   (file, side) cells: {len(rows)}\n")
    for side in ("checkpoint", "baseline"):
        sub = [r for r in rows if r[1] == side]
        if not sub:
            print(f"=== {side}: NO CELLS FOUND -- check the key name before believing this ===\n")
            continue
        print(f"=== {side}: {len(sub)} cells ===")
        dist = collections.Counter(round(r[2], 6) if isinstance(r[2], float) else r[2] for r in sub)
        for v, n in sorted(dist.items(), key=lambda x: -x[1])[:8]:
            print(f"   p_stop_given_done = {v!r:<12} x{n}")
        ones = [r for r in sub if isinstance(r[2], float) and abs(r[2] - 1.0) < 1e-12]
        movable = [r for r in sub if isinstance(r[4], (int, float)) and r[4] > 0]
        vs = [r[2] for r in sub if isinstance(r[2], float)]
        print(f"   exactly 1.0                                : {len(ones)} / {len(sub)}")
        print(f"   n_done_ask > 0 (the cell had room to move) : {len(movable)} / {len(sub)}")
        if vs:
            print(f"   min / max observed                         : {min(vs):.6f} / {max(vs):.6f}")
        ones_with_room = [r for r in ones if isinstance(r[4], (int, float)) and r[4] > 0]
        print(f"   1.0 DESPITE having room to move            : {len(ones_with_room)}")
        print()

    forced0 = [r for r in rows if r[1] == "checkpoint" and r[6] == 0]
    ones0 = [r for r in forced0 if isinstance(r[2], float) and abs(r[2] - 1.0) < 1e-12]
    print(f"=== checkpoint cells with n_forced_stops == 0: {len(forced0)} ===")
    print(f"   of those, reading exactly 1.0: {len(ones0)}")
    for v, n in sorted(collections.Counter(round(r[2], 4) for r in forced0).items())[:10]:
        print(f"     p={v}  x{n}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a driver
    raise SystemExit(main())
