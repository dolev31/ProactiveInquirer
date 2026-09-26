"""Replay the declared one-sided rule and the corrected two-sided rule over real history.

A fixture proves a guard CAN fire. Only real history shows whether it refuses work the
programme depends on, so this prints every stored Tier-A verdict under both rules and the
disagreement between them. It measures; it decides nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOW, HIGH = 0.50, 0.60


def declared(shorter: float, longer: float) -> str:
    """The rule as written in docs/GPU_RUNBOOK.md section 5: long-preference only."""
    return "KILL" if (shorter < LOW and longer > HIGH) else "PASS"


def corrected(shorter: float, longer: float) -> str:
    if shorter < LOW and longer > HIGH:
        return "KILL(long)"
    if longer < LOW and shorter > HIGH:
        return "KILL(short)"
    return "PASS"


def main(roots: list[str]) -> int:
    files: list[Path] = []
    for r in roots:
        files.extend(sorted(p for p in Path(r).rglob("*tierA*.json") if p.is_file()))
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cells = (d.get("pair_accuracy") or {}).get("acc_by_len_sign") or {}
        s = (cells.get("chosen_shorter") or {}).get("acc")
        lo = (cells.get("chosen_longer") or {}).get("acc")
        if s is None or lo is None:
            rows.append((f.name, None, None, "N/A", "N/A"))
            continue
        rows.append((f.name, s, lo, declared(s, lo), corrected(s, lo)))

    print(f"{'verdict file':<52} {'short':>6} {'long':>6} {'gap':>7}  declared -> corrected")
    n_flip = 0
    for name, s, lo, dv, cv in rows:
        if s is None:
            print(f"{name[:52]:<52} {'-':>6} {'-':>6} {'-':>7}  N/A")
            continue
        gap = lo - s
        flip = dv != cv.split("(")[0] + ("" if cv == "PASS" else "")
        flip = (dv == "PASS") and cv.startswith("KILL")
        n_flip += flip
        print(
            f"{name[:52]:<52} {s:>6.3f} {lo:>6.3f} {gap:>7.3f}  "
            f"{dv:<5} -> {cv}{'   <== NEWLY FLAGGED' if flip else ''}"
        )
    scored = [r for r in rows if r[1] is not None]
    n_kill_declared = sum(1 for r in scored if r[3] == "KILL")
    n_kill_corrected = sum(1 for r in scored if r[4].startswith("KILL"))
    print()
    print(f"scored verdicts:            {len(scored)}  (of {len(rows)} files)")
    print(f"KILL under declared rule:   {n_kill_declared}")
    print(f"KILL under corrected rule:  {n_kill_corrected}")
    print(f"newly flagged by the fix:   {n_flip}")
    print(f"still PASS under corrected: {len(scored) - n_kill_corrected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["artifacts"]))
