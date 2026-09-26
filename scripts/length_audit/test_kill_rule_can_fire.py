"""Proves the kill rule CAN fire -- the check the "0 KILL among 112 evaluable verdicts" headline is
missing on its own.

This project's repeated failure mode (recorded four times in one day elsewhere: a slot-bias
gate that returned {} for every call site because the one condition it checked never
occurred; a scanner regex that could not match any real occurrence of what it looked for; a
contamination guard that went inert on a prefixed pin; a docstring claiming a nonzero default
a zero-filter cannot produce) is a rule that never fires being silently reported the same way
as a rule that CANNOT fire. "0 KILL" by itself is a null or an unknown; nothing about that
number distinguishes the two. This script is the distinguishing evidence.

Four checks, each asserted and each printed, run through the SAME `kill_rule()` function and
the SAME `audit_tierA.py` CLI the real audit used -- not a reimplementation:

  1. PLANTED verdict, obviously inside the kill zone (chosen_shorter=0.30,
     chosen_longer=0.70) -> must read KILL.
  2. BOUNDARY pair: 0.499/0.601 (just inside the rule) -> KILL; 0.501/0.599 (just outside)
     -> PASS. Shows the thresholds are read as specified (< 0.50 and > 0.60, strict), not
     inverted and not off-by-one.
  3. PERTURBATION of one real, currently-PASS verdict already committed to this repo
     (artifacts/a14_complete_20260919/rung2-8b-headline-control.tierA.json,
     chosen_shorter=0.6319/n=470, chosen_longer=0.4611/n=475): its own two buckets are
     swapped and the verdict must flip PASS -> KILL. Proves the rule reads the fields real
     Tier-A JSON actually populates, not just a synthetic dict shaped to match.
  4. END-TO-END through the file-based script, not just the bare function: all of the above
     are written out as full *.tierA.json files to a temp directory and scored by
     `audit_tierA.py`'s real CLI via subprocess; the resulting CSV's `kill_rule_verdict`
     column is checked against expectations.

Exit code 0 iff every assertion holds. A failure here is not a bug report about this test --
it retracts "0 KILL" reported in RESULT.md and kill_rule_table.csv, and this script says so
when it fails, per the project rule that a check must not hardcode its own verdict.

Usage:
    python scripts/length_audit/test_kill_rule_can_fire.py
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "length_audit" / "audit_tierA.py"
# A REAL verdict that passes, so check 3 exercises the PASS direction on real data rather
# than only on fixtures. RE-POINTED 2026-09-19: this was
# `rung2-8b-headline-control.tierA.json`, which passed only because the rule was one-sided.
# Its cells are chosen_shorter 0.632 / chosen_longer 0.461 -- a short-preference, and the
# corrected two-sided rule flags it. Leaving it here would have made check 3a assert that a
# length-reading verdict passes, which is the belief the fix exists to remove.
# `qwen3-4b-dpo-headline-control` passes under BOTH rules (0.598 / 0.455: its chosen_shorter
# cell is below the 0.60 arm, so the mirror does not fire) and is the honest PASS control.
REAL_PASS_VERDICT = (
    REPO_ROOT / "artifacts" / "a14_complete_20260919" / "qwen3-4b-dpo-headline-control.tierA.json"
)
# Kept as a real SHORT-PREFERENCE verdict, so check 3c asserts the new arm fires on real data.
REAL_SHORT_PREF_VERDICT = (
    REPO_ROOT / "artifacts" / "a14_complete_20260919" / "rung2-8b-headline-control.tierA.json"
)


def _load_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_tierA", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUDIT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _by_len_sign(shorter_acc: float, longer_acc: float, n: int = 50) -> dict[str, Any]:
    return {
        "chosen_shorter": {"acc": shorter_acc, "n": n},
        "chosen_longer": {"acc": longer_acc, "n": n},
        "equal": {"acc": 0.5, "n": 0},
    }


def _verdict_json(name: str, shorter_acc: float, longer_acc: float) -> dict[str, Any]:
    """A full, schema-shaped *.tierA.json body (docs/GPU_RUNBOOK.md section 5), not a
    trimmed-down stand-in -- so the end-to-end CLI check exercises the exact same parsing
    `load_verdict`/`row_for` apply to the real 216 verdicts."""
    return {
        "checkpoint": name,
        "tokenizer": "synthetic",
        "dev_sft": "synthetic.jsonl",
        "dev_pairs": "synthetic.pairs.jsonl",
        "n_sft_rows": 1,
        "n_pairs": 100,
        "chat_template": "synthetic",
        "enable_thinking": False,
        "sft_nll": {"mean": 0.0, "n": 1},
        "stop_confusion": {"n": 1},
        "pair_accuracy": {
            "n": 100,
            "acc": 0.5,
            "acc_by_kind": {"ask_ask": 0.5, "ask_stop": 0.5},
            "acc_by_decided_by": {},
            "mean_margin": 0.0,
            "acc_by_len_sign": _by_len_sign(shorter_acc, longer_acc),
            "len_sign_gap": longer_acc - shorter_acc,
        },
    }


def main() -> int:
    audit = _load_audit_module()
    failures: list[str] = []

    def check(label: str, got: str, expected: str) -> None:
        status = "ok" if got == expected else "FAIL"
        print(f"[{status}] {label}: got {got!r}, expected {expected!r}")
        if got != expected:
            failures.append(label)

    # 1. Planted verdict, obviously in the kill zone.
    verdict, _, _ = audit.kill_rule(_by_len_sign(0.30, 0.70))
    check("1. planted (0.30/0.70) via kill_rule()", verdict, "KILL (long-preference)")

    # 2. Boundary pair: just inside vs just outside.
    verdict, _, _ = audit.kill_rule(_by_len_sign(0.499, 0.601))
    check(
        "2a. boundary just-inside (0.499/0.601) via kill_rule()", verdict, "KILL (long-preference)"
    )
    verdict, _, _ = audit.kill_rule(_by_len_sign(0.501, 0.599))
    check("2b. boundary just-outside (0.501/0.599) via kill_rule()", verdict, "PASS")

    # 3. Perturbation of one real, currently-PASS verdict already committed to this repo.
    if not REAL_PASS_VERDICT.exists():
        failures.append(f"3. real verdict file missing: {REAL_PASS_VERDICT}")
        print(f"[FAIL] 3. real verdict file missing: {REAL_PASS_VERDICT}")
    else:
        real = json.loads(REAL_PASS_VERDICT.read_text())
        real_by_len_sign = real["pair_accuracy"]["acc_by_len_sign"]
        verdict, shorter, longer = audit.kill_rule(real_by_len_sign)
        check(
            f"3a. real verdict as-is (chosen_shorter={shorter}, chosen_longer={longer})",
            verdict,
            "PASS",
        )
    # 3b/3c. A real SHORT-PREFERENCE verdict: it must fire as-is on the corrected arm, and fire
    # on the declared arm once its buckets are swapped. Both directions, on real data.
    if not REAL_SHORT_PREF_VERDICT.exists():
        failures.append(f"3b. real verdict file missing: {REAL_SHORT_PREF_VERDICT}")
        print(f"[FAIL] 3b. real verdict file missing: {REAL_SHORT_PREF_VERDICT}")
    else:
        sp = json.loads(REAL_SHORT_PREF_VERDICT.read_text())
        sp_cells = sp["pair_accuracy"]["acc_by_len_sign"]
        verdict, shorter, longer = audit.kill_rule(sp_cells)
        check(
            f"3b. real short-preference verdict as-is "
            f"(chosen_shorter={shorter}, chosen_longer={longer})",
            verdict,
            "KILL (short-preference)",
        )
        swapped = copy.deepcopy(sp_cells)
        swapped["chosen_shorter"], swapped["chosen_longer"] = (
            sp_cells["chosen_longer"],
            sp_cells["chosen_shorter"],
        )
        verdict, shorter, longer = audit.kill_rule(swapped)
        check(
            f"3c. same verdict, buckets swapped (chosen_shorter={shorter}, chosen_longer={longer})",
            verdict,
            "KILL (long-preference)",
        )

    # 4. End-to-end through the real file-based CLI, not just the bare function.
    cases = {
        "planted-kill": (_verdict_json("planted-kill", 0.30, 0.70), "KILL (long-preference)"),
        "boundary-kill": (_verdict_json("boundary-kill", 0.499, 0.601), "KILL (long-preference)"),
        "boundary-pass": (_verdict_json("boundary-pass", 0.501, 0.599), "PASS"),
    }
    if REAL_SHORT_PREF_VERDICT.exists():
        real = json.loads(REAL_SHORT_PREF_VERDICT.read_text())
        (
            real["pair_accuracy"]["acc_by_len_sign"]["chosen_shorter"],
            real["pair_accuracy"]["acc_by_len_sign"]["chosen_longer"],
        ) = (
            real["pair_accuracy"]["acc_by_len_sign"]["chosen_longer"],
            real["pair_accuracy"]["acc_by_len_sign"]["chosen_shorter"],
        )
        real["checkpoint"] = "perturbed-real-rung2-8b-headline-control"
        cases["perturbed-real"] = (real, "KILL (long-preference)")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, (body, _expected) in cases.items():
            (tmp / f"{name}.tierA.json").write_text(json.dumps(body))
        out_csv = tmp / "out.csv"
        proc = subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(tmp), "--out-csv", str(out_csv)],
            capture_output=True,
            text=True,
        )
        print(proc.stdout.strip())
        if proc.returncode != 0:
            failures.append("4. audit_tierA.py CLI exited nonzero")
            print(f"[FAIL] 4. audit_tierA.py CLI exited nonzero:\n{proc.stderr}")
        else:
            rows_by_name = {}
            with out_csv.open() as f:
                for row in csv.DictReader(f):
                    rows_by_name[Path(row["path"]).stem.replace(".tierA", "")] = row
            for name, (_body, expected) in cases.items():
                got = rows_by_name.get(name, {}).get("kill_rule_verdict", "<missing row>")
                check(f"4. end-to-end CLI, {name}", got, expected)

    print()
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED -- the kill rule CANNOT be trusted:")
        for f in failures:
            print(f"  - {f}")
        print(
            "This retracts any '0 KILL' figure reported elsewhere until the rule is fixed "
            "and this script passes."
        )
        return 1
    print("RESULT: all checks passed -- the kill rule fires on a planted verdict, on both")
    print("boundary sides, in BOTH directions on real verdicts, and through the file-based CLI.")
    print()
    print("RETRACTED 2026-09-19. This script used to close by saying the 0-KILL figure was")
    print("'a property of the data, not of the rule being unable to fire'. That sentence was")
    print("wrong, and passing these checks never licensed it: they show the rule CAN fire, not")
    print("that it can fire in the direction the data actually runs. Measured over 124 real")
    print("scored verdicts, len_sign_gap is negative in 124 of 124 -- every checkpoint, every")
    print("size, every objective prefers the SHORTER question -- and the one-sided rule tested")
    print("only the long-preference. So 0 KILL was entailed by the rule's direction. The rule")
    print("is now two-sided on the same constants and flags 49 of those 124.")
    print("See artifacts/tierA_length_decomposition_20260919/RESULT.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
