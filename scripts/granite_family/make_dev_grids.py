#!/usr/bin/env python3
"""Derive per-seed and per-baseline dev-gate grids for the Granite family reading (lane L2.5).

WHY THESE FILES HAVE TO EXIST. `pinq_train.gate.run_gate` selects checkpoint runs with
`_select_runs(con, arm=checkpoint_arm, grids=[grid_name], model_id=None)` -- the `model_id=None`
is not a missing feature, it is because `runs.parquet` carries only `model_pin_hash`, not a
model name, on the checkpoint side (see `pi train gate --help`'s note on `--baseline-model-id`,
which exists ONLY for the baseline). Three granite seeds run the identical grid
(`conf/grids/dev_select_<suite>.yaml`) with different `PI_MODEL_INQUIRER`, so without a
distinct `grid_name` per seed, `pi train gate` cannot tell seed 0's rows from seed 1's or
seed 2's -- it would silently pool all three into one contrast. `conf/grids/tier1_trained_qa_base.yaml`'s
own docstring already establishes the fix for exactly this shape: "the grid is duplicated
under a distinct grid_name ... a $0.4182 spend basis is". This script applies the same fix
mechanically, by TEXT substitution rather than a YAML round-trip, because every one of these
grid files carries load-bearing prose comments that `yaml.safe_load` + `yaml.dump` would
silently discard.

Two families of output, from each `conf/grids/dev_select_<suite>.yaml`:

  * `dev_select_<suite>_granite_s{0,1,2}.yaml` -- byte-identical except `name:`, for the three
    trained checkpoints (`arms: [inquirer_trained]` unchanged).
  * `dev_select_<suite>_granite_base.yaml` -- byte-identical except `name:` AND
    `arms:\n  - inquirer_trained` -> `arms:\n  - inquirer_prompted`, so the granite PROMPTED
    base can be swept on the identical dev task population under a selectable grid_name.

Usage:
    python scripts/granite_family/make_dev_grids.py [--check]

`--check` verifies the files are byte-identical to what this script would write, and exits
non-zero otherwise, without writing -- the CI-safe form.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRIDS = REPO_ROOT / "conf" / "grids"
SUITES = ("musique", "strategyqa", "wiki2")
SEEDS = (0, 1, 2)


def derive(source_text: str, source_name: str, new_name: str, *, prompted_base: bool) -> str:
    if f"name: {source_name}\n" not in source_text:
        raise ValueError(f"expected 'name: {source_name}' in the source grid, found none")
    out = source_text.replace(f"name: {source_name}\n", f"name: {new_name}\n", 1)
    if prompted_base:
        old_arms = "arms:\n  - inquirer_trained\n"
        new_arms = "arms:\n  - inquirer_prompted\n"
        if old_arms not in out:
            raise ValueError(f"expected {old_arms!r} (exact) in the source grid, found none")
        out = out.replace(old_arms, new_arms, 1)
    return out


def planned_outputs() -> dict[Path, str]:
    plan: dict[Path, str] = {}
    for suite in SUITES:
        src_path = GRIDS / f"dev_select_{suite}.yaml"
        src_name = f"dev_select_{suite}"
        src_text = src_path.read_text()
        for seed in SEEDS:
            new_name = f"dev_select_{suite}_granite_s{seed}"
            plan[GRIDS / f"{new_name}.yaml"] = derive(
                src_text, src_name, new_name, prompted_base=False
            )
        base_name = f"dev_select_{suite}_granite_base"
        plan[GRIDS / f"{base_name}.yaml"] = derive(
            src_text, src_name, base_name, prompted_base=True
        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    plan = planned_outputs()
    if a.check:
        bad = []
        for path, text in plan.items():
            if not path.exists() or path.read_text() != text:
                bad.append(path)
        if bad:
            for p in bad:
                existing = p.read_text().splitlines() if p.exists() else []
                diff = "\n".join(
                    difflib.unified_diff(existing, plan[p].splitlines(), str(p), "planned")
                )
                print(diff, file=sys.stderr)
            print(f"{len(bad)} of {len(plan)} derived grids are stale or missing", file=sys.stderr)
            return 1
        print(f"{len(plan)} derived grids up to date")
        return 0

    for path, text in plan.items():
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
