"""The lock line for the paper lane: s1 and s2 in the four-seed record against (a) the committed
table1_by_seed.json record, cell for cell, and (b) the seed-1/seed-2 columns Table 1 prints
(paper/iclr2027 2/figures/table1_recipe.tex), at the table's three decimals.

(b) is checked under both roundings. The paper's stated convention is half-up on an exact-decimal
tie. Python's format() rounds the binary value. Where they differ, the table's printed digit is
named, so a tie cannot be read as a mismatch or a match by accident.
"""

from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "paper" / "iclr2027 2" / "figures" / "table1_recipe.tex"
REC = REPO / "artifacts" / "seed_identity_20260923" / "table1_by_seed.json"
ROWS = (
    "evidence_coverage",
    "facet_breadth_scorer",
    "dwr",
    "max_depth_reached",
    "precedence_violation_rate",
)
SUITES = ("musique", "strategyqa", "wiki2")


def hu(x: float, d: int = 3) -> str:
    q = Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:+.{d}f}"


def main(path: str) -> int:
    four = json.load(open(path))["table1"]["cells"]
    rec = json.load(open(REC))["cells"]
    seed_rows = [ln for ln in TEX.read_text().splitlines() if "seed 1/seed 2" in ln]
    assert len(seed_rows) == len(ROWS), len(seed_rows)
    bad = 0
    for metric, ln in zip(ROWS, seed_rows):
        pairs = re.findall(r"\$([+-]\d\.\d{3})\$/\$([+-]\d\.\d{3})\$", ln)
        assert len(pairs) == 3, ln
        for suite, (p1, p2) in zip(SUITES, pairs):
            key = f"{metric}::{suite}"
            for sd, printed in (("s1", p1), ("s2", p2)):
                got = four[key][sd]
                same_record = {k: got[k] for k in rec[sd][key]} == rec[sd][key]
                x = got["delta"]
                h, py = hu(x), f"{x:+.3f}"
                ok_print = printed in (h, py)
                note = "" if h == py else f" TIE: half-up {h}, binary {py}, table prints {printed}"
                print(
                    f"{key:40s} {sd} delta={x!r} record_equal={same_record} printed={printed}"
                    f" match={ok_print}{note}"
                )
                bad += (not same_record) + (not ok_print)
    print(f"LOCK s1/s2: {'PASS' if bad == 0 else 'FAIL'} ({bad} mismatches over 30 cells)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
